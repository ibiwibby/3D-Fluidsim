# GPU Fluid Simulation: First Progress Checkpoint

## Objective and scope

Build a CUDA implementation of a **2 × 2, two-dimensional lattice Boltzmann grid using D2Q9**, and run it on a GPU at TACC Lonestar6 or Vista. Demonstrate initialization, collision, streaming, and repeated timesteps with checks that can be evaluated without a CPU solver.

This document is an implementation plan; it contains no simulation code. The CPU will handle configuration, GPU allocation, kernel launches, and reporting. All fluid updates will run on the GPU. Reading back results and checking simple invariants on the host does not require a CPU fluid solver.

The checkpoint establishes the update machinery for the lattice Boltzmann project described in `../ECE379K_Project_Proposal.md`. Four cells are sufficient for debugging, but cannot resolve realistic flow or demonstrate GPU speedup. Walls, obstacles, inlet/outlet conditions, 3D, CPU reference implementations, and performance optimization are later milestones.

Use periodic boundaries initially. A distribution leaving one side enters the opposite side; this is not a solid-wall bounce.

## 1. Initialize the grid

### Establish the TACC execution environment

- Choose one system for the first working run. Request a short, single-node GPU allocation and use one GPU and one host process. MPI and multi-GPU communication are unnecessary.
- On Lonestar6, select an NVIDIA GPU partition, such as `gpu-a100-dev` or `gpu-a100-small`, subject to your allocation and current availability. Its ordinary CPU nodes do not provide the GPU required here. See the [Lonestar6 user guide](https://docs.tacc.utexas.edu/hpc/lonestar6/).
- On Vista, choose the Grace-Hopper GPU subsystem, such as `gh-dev`, rather than the Grace-only subsystem. See the [Vista user guide](https://docs.tacc.utexas.edu/hpc/vista/).
- Check the current system guide for interactive allocation or Slurm submission details. Load a compatible compiler and CUDA module; record their versions. Compile on the target system, with a GPU architecture target appropriate to the allocated hardware. Rebuild when moving between systems: their host architectures differ.
- Execute GPU tests inside the allocation. Record the GPU model, CUDA toolkit version, compiler, and job ID so the demonstration is reproducible.

### Fix the lattice conventions

Set `Nx = 2`, `Ny = 2`, and `N = Nx × Ny = 4`, but keep dimensions configurable. Let x increase rightward and y upward. Flatten coordinates using `cell = x + Nx × y`:

| Cell | Coordinates |
| --- | --- |
| 0 | (0, 0) |
| 1 | (1, 0) |
| 2 | (0, 1) |
| 3 | (1, 1) |

Each cell stores nine populations, denoted `f_i`. These represent directional fluid distributions, not nine separately tracked particles. Adopt one direction ordering throughout initialization, kernels, tests, and output:

| i | Direction | Velocity vector c_i | Weight w_i |
| --- | --- | --- | --- |
| 0 | Rest | (0, 0) | 4/9 |
| 1 | East | (1, 0) | 1/9 |
| 2 | North | (0, 1) | 1/9 |
| 3 | West | (-1, 0) | 1/9 |
| 4 | South | (0, -1) | 1/9 |
| 5 | Northeast | (1, 1) | 1/36 |
| 6 | Northwest | (-1, 1) | 1/36 |
| 7 | Southwest | (-1, -1) | 1/36 |
| 8 | Southeast | (1, -1) | 1/36 |

Use lattice units: grid spacing and timestep are both 1. Start with double precision for straightforward debugging. Use BGK relaxation time `tau = 1`, hence relaxation rate `omega = 1/tau = 1`. Keep this parameter separate from the timestep.

Initialize every cell to density `rho = 1` and velocity `u = (0, 0)`. At rest, equilibrium populations are simply `f_i = w_i × rho`. Every cell therefore sums to density 1, has zero momentum, and the grid has total mass 4.

Initialization may be a GPU kernel or a host-created initial condition copied once to the GPU. Either is compatible with a GPU-only solver.

**Completion check:** read back initialization and confirm all 36 populations match their intended values, each cell has density 1, and total mass is 4 within floating-point tolerance.

## 2. Allocate two GPU buffers

Allocate two device arrays, A and B, each containing `9 × N` values. For four cells, that is 36 doubles or 288 bytes per buffer, 576 bytes total for the populations.

Use a structure-of-arrays layout: population `i` at cell `cell` is stored at `i × N + cell`. Values for neighboring cells in the same direction are contiguous, which is useful when the grid grows. Density and velocity can be derived from the populations; they need not be persistent simulation arrays yet.

For this checkpoint, use the following two-kernel timestep:

1. A holds the current populations.
2. The collision kernel updates each cell's populations **in place in A**. Each thread must read all nine local populations before writing its results. No thread accesses another cell during collision.
3. The streaming kernel reads the post-collision populations from A and writes every population in B.
4. Swap the roles of A and B. The newly streamed buffer is the current state for the next step.

This arrangement uses exactly two buffers while keeping the operations independently testable. In-place collision is safe because it is local. In-place streaming is not safe because threads could overwrite values that neighbors still need to read.

Launch both kernels in the same CUDA stream so streaming starts after collision completes. A host-side pointer swap does not copy device data. Keep both buffers allocated and resident throughout the timestep loop. Copy back only requested checkpoints or final results, using the current buffer after all swaps.

Check allocation, copy, and launch errors. During debugging, synchronize after each kernel to expose execution errors near their source. Release both allocations at shutdown. B does not need an initial physical state if streaming fully overwrites all its entries before they are read.

**Completion check:** both buffers exist, initialization survives a readback, and the current-buffer bookkeeping is correct after both odd and even numbers of steps.

## 3. Implement D2Q9 collision and streaming

### Map cells to GPU threads

Use one active thread per cell. A conventional block can contain more than four threads; threads whose cell index is outside `N` must return without accessing arrays. Compute launch dimensions from `N` so larger grids do not require a new kernel design.

### Collision: compute the local fluid state and relax populations

For each cell, first load its nine populations and compute:

\[
\rho = \sum_{i=0}^{8} f_i,\qquad
\mathbf{u} = \frac{\sum_{i=0}^{8} f_i\mathbf{c}_i}{\rho}.
\]

For this unforced model, use the equilibrium distribution:

\[
f_i^{eq} = w_i\rho\left[1 + 3(\mathbf{c}_i\cdot\mathbf{u})
+ \frac{9}{2}(\mathbf{c}_i\cdot\mathbf{u})^2
- \frac{3}{2}(\mathbf{u}\cdot\mathbf{u})\right].
\]

Then apply the BGK update:

\[
f_i^{post} = f_i - \omega(f_i-f_i^{eq}).
\]

The equilibrium and relaxation formulation follows the standard lattice Boltzmann construction described in [M-Star's theory documentation](https://docs.mstarcfd.com/17_Theory/theory-and-implementation.html); use the D2Q9 directions and weights listed above for this checkpoint.

With `omega = 1`, post-collision values equal the local equilibrium values. Collision conserves local mass and momentum, up to roundoff. Flag nonpositive density or nonfinite values as failures rather than silently replacing them.

### Streaming: move each population to its neighbor

Use **pull streaming**: each thread owns one destination cell and fills all nine destination populations. For destination `(x, y)` and direction `i`, read from:

\[
x_{src} = (x-c_{ix}+N_x)\bmod N_x,\qquad
y_{src} = (y-c_{iy}+N_y)\bmod N_y.
\]

The destination receives:

\[
f_i^{next}(x,y)=f_i^{post}(x_{src},y_{src}).
\]

The added dimensions prevent negative indices for these unit lattice offsets. The direction index stays the same during streaming. The rest population remains in its own cell. Each destination entry has one writer, so no atomics or inter-thread barriers are needed.

**Completion check:** one complete timestep finishes without CUDA errors, writes all 36 output entries, and yields a readable current state. Run a device memory checker, such as CUDA Compute Sanitizer, on the small cases when available in the installed toolkit.

## 4. Check equilibrium and collision correctness

### Rest-state test

Run the uniform initialization from step 1 for 1, 2, 10, and 1,000 timesteps. The odd/even runs help expose incorrect buffer selection.

At each selected checkpoint, reconstruct density and velocity from the returned populations and calculate:

- Maximum population error relative to the initial equilibrium values.
- Maximum density error relative to 1.
- Maximum absolute x and y velocity components relative to zero.
- Relative mass error: `abs(M(t) - M(0)) / M(0)`, where `M = sum of cell densities`.
- Whether any value is NaN or infinite.

For this small double-precision equilibrium test, use an initial acceptance threshold of `1e-12` for each maximum absolute error and relative mass error. These are proposed checkpoint tolerances, not accuracy claims for a future flowing simulation. Investigate failures before changing thresholds.

### Collision-only test

Equilibrium alone is insufficient: a kernel that does nothing would pass it. Test collision with a controlled nonequilibrium input at one cell:

- Begin with the rest equilibrium populations.
- Add `delta = 0.01` to both the east and west populations.
- Subtract `2 × delta` from the rest population.
- Leave the other populations unchanged.

This preserves density 1 and zero momentum, and all populations remain positive. At `omega = 1`, one collision-only operation must restore the original equilibrium populations. Verify all nine values and local mass and momentum. Do not stream during this test.

**Completion check:** the rest state stays unchanged over repeated full steps, and the perturbed collision-only case relaxes to the hand-calculated equilibrium.

## 5. Check streaming separately

Disable collision for this test. Fill every population with a unique, exactly representable diagnostic label:

\[
f_i(cell)=100i+cell+1.
\]

These labels are a memory-routing test, not a physical fluid state. Run streaming once and compare all 36 outputs against a hand-prepared destination table using the source-coordinate rule in step 3. Exact equality is appropriate because streaming only copies these values.

For example, at destination cell 0, coordinates `(0, 0)`, the expected values are:

| Direction i | Source cell on a periodic 2 × 2 grid | Expected label |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 1 | 102 |
| 2 | 2 | 203 |
| 3 | 1 | 302 |
| 4 | 2 | 403 |
| 5 | 3 | 504 |
| 6 | 3 | 604 |
| 7 | 3 | 704 |
| 8 | 3 | 804 |

Also check that every original label appears exactly once after the step. Streaming is a permutation and must neither lose nor duplicate entries. Two streaming-only steps on this particular periodic 2 × 2 grid should restore the entire original array.

The small domain imposes a real testing limitation: moving east or west reaches the same cell, and all four diagonals reach the same diagonal neighbor. A wrong sign in the streaming offset can therefore pass this test. Retaining distinct direction labels detects direction-index mix-ups, but does not remove that spatial ambiguity.

Keep 2 × 2 as the requested checkpoint. As the first follow-up, repeat the same label test on a configurable 3 × 3 grid to distinguish opposite directions before trusting larger simulations.

**Completion check:** all 36 labels arrive at their expected locations, remain unique, and return to the starting array after two streaming-only steps.

## 6. Display results and prepare the checkpoint demonstration

Use a text report; an animation would add little information for four cells. After the relevant GPU operations finish, copy back the current populations and print:

- Test name, grid dimensions, precision, relaxation time, and timestep count.
- GPU model, compiler/CUDA versions, and TACC system/job ID.
- A row for each cell containing its coordinates, density, x velocity, and y velocity.
- Initial mass, final mass, relative mass error, and maximum state errors.
- Pass/fail results for initialization, equilibrium, collision-only, and streaming-only tests.
- Full population tables for a failed test, identifying the cell, direction, expected value, and actual value.

Computing these reporting quantities on the host is sufficient for four cells; the timestep physics remains on the GPU. Later, reductions and field extraction can move to the GPU if output costs become significant.

Preserve a short run log and document the actual module setup, build command, allocation/submission procedure, and execution command once implementation exists. A clean run on either Lonestar6 or Vista satisfies this checkpoint; running on both is an optional portability check.

Timing is optional. If reported, separate setup, transfers, kernel execution, and output. Use CUDA events for device execution timing and ensure completion before reporting it. Four cells mostly measure launch overhead, so do not interpret the results as a speedup or scaling benchmark.

### Definition of done

- [ ] The CUDA program runs successfully on an allocated TACC GPU.
- [ ] All four cells initialize with the expected nine populations.
- [ ] Collision and streaming execute on the GPU with two persistent buffers.
- [ ] Repeated full steps preserve the uniform rest state and total mass within the stated tolerance.
- [ ] The collision-only perturbation returns to equilibrium as predicted.
- [ ] Streaming-only labels match all hand-calculated destinations.
- [ ] CUDA error checks and the available device memory check report no failures.
- [ ] A saved report and reproducible run instructions document the result.

After this checkpoint, test 3 × 3 routing, then use a larger 2D grid for a small density disturbance and visualization. Subsequent milestones can add boundary conditions, a CPU reference, and the 3D lattice model. The GPU buffer lifecycle and validation structure can carry forward.
