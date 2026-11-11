# GPU Fluid Simulation — Checkpoint Work Completed

**Status as of November 11, 2026:** The 2 × 2 CUDA checkpoint has been implemented, compiled, and run successfully on TACC Vista. Initialization, collision, streaming, and repeated rest-state checks passed. The density-disturbance run generated snapshots, and the local visualization was generated and reviewed.

## 1. Goal and relationship to the final project

The final project in [ECE379K_Project_Proposal.md](ECE379K_Project_Proposal.md) is a three-dimensional lattice Boltzmann fluid simulator, eventually including CPU/GPU comparison, channel flow, obstacles, correctness validation, and performance measurements.

This checkpoint establishes a small, understandable implementation of the core GPU update cycle before scaling the domain or adding more complex physics. We initially considered tracking one particle in 2D. However, lattice Boltzmann represents fluid through distributions on grid cells, so we chose a small lattice instead. This makes the checkpoint directly relevant to the proposed numerical method.

The checkpoint demonstrates:

- Initializing fluid distributions on a GPU.
- Computing local collision updates.
- Moving distributions between neighboring cells through streaming.
- Repeating updates with persistent GPU buffers.
- Checking simple, independently predictable results and mass conservation.
- Exporting numerical snapshots and visualizing them locally without X-forwarding.

It does not yet demonstrate realistic resolved flow, GPU speedup, or the final proposal's CPU-versus-GPU agreement target.

## 2. Scope and numerical choices

| Choice | Implemented setting and reason |
| --- | --- |
| Domain | 2 × 2 cells: small enough to inspect every stored value |
| Lattice | D2Q9: two dimensions, nine directional populations per cell |
| Boundaries | Periodic: leaving one side reenters on the opposite side |
| Collision | Single-relaxation-time BGK, default `tau = 1` |
| Precision | Double precision |
| Units | Lattice units, with spacing and timestep equal to 1 |
| Execution | One host process and one CUDA GPU; one active thread per cell |
| CPU role | Configuration, launch control, test fixtures, readback, checks, and file output |
| Rest-state run | 1,000 timesteps with selected checkpoints |
| Demonstration run | 100 timesteps with a small density increase in one cell |
| Tolerance | `1e-12` for the implemented numerical checks |

There is no CPU fluid solver. The host can prepare initial conditions and derive quantities for reporting, but collision and streaming execute on the GPU.

## 3. Where the implementation lives

All links below are relative to this document at the project root.

| File | Purpose |
| --- | --- |
| [src/main.cu](src/main.cu) | Command-line modes, device inspection, initialization/collision reports, streaming label test, timestep loop, and rest-state checkpoint checks |
| [src/kernels.cu](src/kernels.cu) | GPU kernels and host launch wrappers for initialization, BGK collision, and periodic pull streaming |
| [src/include/lbm/kernels.cuh](src/include/lbm/kernels.cuh) | Public launch interfaces and their buffer/stream contracts |
| [src/simulation.cu](src/simulation.cu) | Device allocation, cleanup, uploads/downloads, isolated update calls, synchronization, and buffer swaps |
| [src/include/lbm/simulation.cuh](src/include/lbm/simulation.cuh) | `Simulation` class interface and ownership of its two GPU buffers |
| [src/include/lbm/config.hpp](src/include/lbm/config.hpp) | Dimensions, precision, timestep counts, relaxation time, tolerance, and density-increase defaults |
| [src/include/lbm/d2q9.cuh](src/include/lbm/d2q9.cuh) | Shared direction vectors, weights, and population indexing for host and GPU code |
| [src/include/lbm/cuda_utils.cuh](src/include/lbm/cuda_utils.cuh) | `CUDA_CHECK` helper, reporting the CUDA error, expression, file, and line |
| [src/disturbance.cu](src/disturbance.cu) | Disturbance initialization, evolution, field extraction, mass checks, and CSV writing |
| [src/include/lbm/disturbance.hpp](src/include/lbm/disturbance.hpp) | Entry point for the disturbance demonstration |
| [src/Makefile](src/Makefile) | CUDA compilation/linking, architecture-specific builds, run targets, and Compute Sanitizer target |
| [scripts/animate.py](scripts/animate.py) | CSV validation and headless generation of the density/velocity GIF and initial/final PNGs |
| [scripts/requirements.txt](scripts/requirements.txt) | Python plotting dependencies: Matplotlib, NumPy, and Pillow |
| [src/checkpoint_steps.md](src/checkpoint_steps.md) | Original detailed checkpoint plan and hand-checkable expectations |
| [src/README.md](src/README.md), [scripts/README.md](scripts/README.md) | Build, execution, output, and visualization instructions |

## 4. Fluid state and GPU update details

### State representation

Each cell holds nine values `f_i`. These are directional fluid distributions, not nine individually tracked particles. Directions are ordered as rest, east, north, west, south, northeast, northwest, southwest, southeast. Their weights are `4/9` for rest, `1/9` for axial directions, and `1/36` for diagonals.

Coordinates flatten to `cell = x + Nx*y`. Populations use a structure-of-arrays layout:

```text
index = direction * number_of_cells + cell
```

The 2 × 2 domain therefore has 36 populations. Each double-precision buffer occupies 288 bytes; the two population buffers occupy 576 bytes in total, excluding other runtime memory. Directions and dimensions are not hard-coded into the kernel's cell count, so the kernels can operate on larger configured 2D grids.

### GPU initialization

`initialize_rest_kernel` assigns one thread to each cell and writes `f_i = w_i`. This represents density 1 and zero velocity everywhere. Launches use 128 threads per block, with out-of-range threads returning immediately. Only four threads are active for the default grid.

### BGK collision

`collide_kernel` loads all nine local populations before any global writes. It computes:

```text
rho = sum(f_i)
u = sum(f_i * c_i) / rho
f_eq_i = w_i * rho * [1 + 3(c_i·u) + 4.5(c_i·u)^2 - 1.5(u·u)]
f_post_i = f_i - (1/tau) * (f_i - f_eq_i)
```

Collision writes back into the current buffer. Each thread owns one cell, so it does not read another thread's partially updated state. Invalid density or nonfinite state triggers a device trap; the synchronization error terminates the run rather than continuing with corrupted values.

### Periodic pull streaming

`stream_periodic_kernel` assigns one thread to each destination cell. For each direction `i`, it reads the corresponding population from the upstream cell `(x-c_ix, y-c_iy)`, wrapping at the domain edges. It preserves the direction index and writes into the separate next buffer. The rest distribution remains in its own cell.

Each destination population has exactly one writer; atomics are unnecessary. Streaming cannot safely overwrite the current buffer, since neighbors may still need its old values.

### Buffer lifecycle and timestep control

`Simulation` allocates two persistent buffers and prevents copying of their ownership. It provides:

- `initialize()`: GPU equilibrium initialization and synchronization.
- `upload()`: copy a complete host fixture or initial condition into the current buffer.
- `download()`: copy the current populations back to the host.
- `collide_only()`: collision without streaming or a swap.
- `stream_only()`: stream to the next buffer, synchronize, then swap pointers.
- `step()`: collision followed by streaming and the swap.

The destructor frees both allocations. Launch errors are checked immediately; synchronization exposes device execution errors. Synchronization after each operation is deliberate for this checkpoint and is not a performance optimization.

## 5. Executable modes and checks

| Mode / Make target | What it does |
| --- | --- |
| `--device-info` / `device-info` | Prints the selected GPU and compute capability |
| `--run` / `run` | Initialization only; verifies all populations, density 1, zero velocity, and expected mass |
| `--collision-check` / `collision-check` | Perturbs one cell, runs exactly one collision at `tau=1`, and checks restoration of equilibrium |
| `--streaming-check` / `streaming-check` | Uses unique labels on a fixed 2 × 2 grid to verify one and two streaming operations without collision |
| `--simulate` / `simulate` | Evolves uniform rest for the configured number of full timesteps and checks selected states |
| `--disturbance [output.csv]` / `disturbance` | Evolves a density bump and exports every state to CSV |

The collision fixture adds `0.01` to east and west and subtracts `0.02` from rest at cell 0. It preserves density and momentum but is not at equilibrium. At `tau=1`, one collision must restore the equilibrium populations. This catches a no-op collision that a rest-state test alone would miss.

The streaming fixture uses labels `100*direction + cell + 1`. All 36 outputs after one stream are compared with a hand-calculated source-cell table. After two streams on this particular periodic grid, the labels must return to their original positions. Exact comparisons also establish that no labels were lost or duplicated and exercise both buffer swaps.

`--simulate` checks steps 0, 1, 2, 10, and 1,000 when within the run length, plus the final step without duplicating checks. Reports include maximum population and density errors, maximum absolute velocity components, and relative mass error. Invalid state or excessive error stops the run with nonzero status.

## 6. Density disturbance and visualization

The disturbance mode starts with density 1.01 at `(0,0)` and density 1 at the other cells, all at zero velocity. The host sets the bumped cell's equilibrium populations to `w_i * 1.01` and uploads that initial condition. Subsequent evolution uses the same GPU collision/streaming code as the rest-state run.

For this case, expected total mass is **4.01**, not 4. The code checks finite populations and fields, positive density, and mass conservation before writing each snapshot. It does not require the evolving disturbance to stay at the original rest equilibrium.

The default `disturbance.csv` contains:

```text
step,x,y,density,ux,uy
```

There is one row per cell per saved timestep. Steps 0 through 100 give **101 snapshots and 404 data rows**, plus the header. Files use sufficient precision to round-trip doubles. File errors fail the run; previously completed snapshots are flushed to disk. An optional output filename is accepted, and an existing file at that path is overwritten.

After copying the CSV from TACC, `scripts/animate.py`:

- Rejects malformed values, duplicate cells, and incomplete or inconsistent grids.
- Sorts snapshots by timestep and renders a density heatmap beside velocity arrows.
- Keeps color and arrow scales fixed across frames.
- Labels small-grid densities and displays timestep, total mass, and maximum speed.
- Hides velocity arrows below `1e-12` to avoid magnifying roundoff.
- Uses a headless backend and Pillow, so no X-forwarding, GUI session, or FFmpeg is needed.

The available artifacts are [results/disturbance.csv](results/disturbance.csv), [results/disturbance.gif](results/disturbance.gif), [results/disturbance_initial.png](results/disturbance_initial.png), and [results/disturbance_final.png](results/disturbance_final.png). The CSV was inspected locally: it contains the expected 101 snapshots, and all four final densities are approximately **1.0025**, with zero recorded velocity. The animation was generated and visually accepted by Harish.

This behavior fits the special 2 × 2 periodic geometry: opposite directions reach the same neighbor, so the symmetric initialization can redistribute density while retaining zero velocity. It is a demonstration of the update mechanism, not a resolved fluid-flow benchmark.

## 7. Validation completed on Vista

The following evidence comes from Harish's Vista terminal output shared during implementation. The reported GPU was **NVIDIA GH200 120GB**, compute capability **9.0**. The build used `CUDA_ARCH=sm_90` and completed without reported compilation errors.

| Check | Observed result |
| --- | --- |
| Initialization | PASS; four unit-density cells, zero velocity, mass 4 |
| Collision-only fixture | PASS; all populations returned to equilibrium |
| Streaming-only fixture | PASS; all one-step and two-step labels matched |
| Rest evolution through 1,000 steps | All scheduled checkpoints passed; maximum reported population error about `1.11e-16`, density/mass error about `2.22e-16` |
| Density disturbance through 100 steps | 101 snapshots saved; mass 4.01 with reported relative error at most about `2.21e-16` in the printed checkpoints |
| Compute Sanitizer memcheck | Zero reported errors for initialization, collision-only, streaming-only, and rest simulation |

The `memcheck` target does **not** include the disturbance mode. Its ordinary numerical checks passed, but no disturbance-specific sanitizer run is claimed. These results validate the exercised small cases; they do not establish general flow accuracy or performance.

## 8. Reproducing the checkpoint

Inside an allocated Vista GPU session, with the appropriate compiler/CUDA environment loaded:

```sh
cd src
make CUDA_ARCH=sm_90
make device-info CUDA_ARCH=sm_90
make run CUDA_ARCH=sm_90
make collision-check CUDA_ARCH=sm_90
make streaming-check CUDA_ARCH=sm_90
make simulate CUDA_ARCH=sm_90
make disturbance CUDA_ARCH=sm_90
make memcheck CUDA_ARCH=sm_90
```

The Makefile defaults to `sm_80`, intended for an A100 target. Pass `sm_90` consistently on Vista. Build outputs are separated by architecture and debug setting, such as `src/build/sm_90/debug-0/checkpoint`. `DEBUG=1` enables device debugging; `make clean` removes generated build files. The Makefile does not request a TACC allocation or submit a scheduler job.

On the Mac, after copying the CSV into `results/`, run from the project root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements.txt
python scripts/animate.py results/disturbance.csv
```

The script defaults to the project's `results/` for generated graphics regardless of the current working directory. `--fps` adjusts playback speed. Generated files in `results/` are ignored by its `.gitignore`; share or deliberately track the desired checkpoint artifacts separately when setting up version control.

## 9. Boundaries of this checkpoint and next work

The checkpoint implementation and its basic Vista execution/visualization demonstration are complete. The following remain outside the completed scope:

1. A 3 × 3 streaming test to distinguish positive and negative directions that alias on 2 × 2.
2. Larger 2D grids with meaningful velocity fields and stronger physical validation.
3. Solid walls, inlets/outlets, obstacles, and channel-flow cases.
4. A CPU reference solver and CPU/GPU agreement measurements.
5. Extension to a 3D lattice model and 3D geometry.
6. Profiling, performance benchmarks, and optimization of synchronization, memory traffic, and transfers.

Dimensions and settings can be changed in `Config` and rebuilt; there is no general runtime configuration parser yet. The streaming-only test deliberately remains fixed at 2 × 2. The Python renderer loads all snapshots into memory and targets small demonstrations. Four active GPU threads are useful for correctness debugging, not for measuring acceleration.
