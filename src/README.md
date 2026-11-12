# CUDA checkpoint skeleton

This repository's `src/` is the active source directory. Make future code changes
here, not in the separate `FinalProject/src/` copy outside the repository.
The current disturbance configuration is 64 × 64, 50 steps, and a sharp central 4 × 4
square at density 5.00; all other cells start at 1.00.

Framework for the 2 × 2 GPU D2Q9 checkpoint in [checkpoint_steps.md](checkpoint_steps.md).
GPU initialization, BGK collision, and periodic pull streaming are implemented,
with separate check modes and repeated rest-state checkpoint checks.
The density-disturbance mode exports CSV snapshots for local visualization.
GPU execution has not yet been validated. The local animation script is available
in [scripts/animate.py](../scripts/animate.py), with [setup instructions](../scripts/README.md).

## Files

| File | Responsibility |
| --- | --- |
| `main.cu` | Command selection, CUDA device inspection, future run/report entry point |
| `disturbance.cu`, `include/lbm/disturbance.hpp` | Density disturbance, field extraction, mass checks, CSV snapshots |
| `include/lbm/config.hpp` | Grid dimensions, precision, steps, relaxation time, tolerance |
| `include/lbm/d2q9.cuh` | Shared direction convention, weights, population indexing |
| `include/lbm/cuda_utils.cuh` | CUDA error reporting |
| `include/lbm/kernels.cuh`, `kernels.cu` | Initialization, collision, and streaming kernels and launch interfaces |
| `include/lbm/simulation.cuh`, `simulation.cu` | Two device buffers, cleanup, stepping, swaps, readback |
| `Makefile` | CUDA compilation, header dependencies, debug builds, execution targets |

## Build on TACC

Load the site's compatible CUDA/compiler modules, then build from this directory
using `nvcc` with C++17 support. Use the architecture of your allocated GPU:

```sh
# Lonestar6 A100 target
make CUDA_ARCH=sm_80

# Vista Grace-Hopper target; build natively on Vista
make CUDA_ARCH=sm_90
```

The default is `sm_80`. Check the actual allocated hardware rather than assuming
every GPU node on a system is identical. See the [Lonestar6 guide](https://docs.tacc.utexas.edu/hpc/lonestar6/)
and [Vista guide](https://docs.tacc.utexas.edu/hpc/vista/) for current GPU allocation
and module instructions. Rebuild on each system; do not copy host executables
between the systems.

Run GPU commands **inside an allocated GPU job**. These Make targets do not
request an allocation or submit a Slurm job. Pass the same architecture on every
invocation; examples below assume an A100:

```sh
make help CUDA_ARCH=sm_80
make device-info CUDA_ARCH=sm_80
make run CUDA_ARCH=sm_80
make collision-check CUDA_ARCH=sm_80
make streaming-check CUDA_ARCH=sm_80
make simulate CUDA_ARCH=sm_80
make disturbance CUDA_ARCH=sm_80
```

`help` requires no GPU access. `device-info` checks device visibility.
`run` allocates both buffers, initializes the rest state on the GPU, and downloads
all populations. It prints cell coordinates, density, velocity, and total mass,
then checks all populations against their equilibrium weights, each cell against
density 1 and zero velocity, and relative mass error against the configured
tolerance. It returns a nonzero status on failure. It does not run timesteps.

`simulate` (`--simulate`) starts from uniform rest and calls `Simulation::step()`
exactly `Config::steps` times (default 1,000). Change `steps` in
`include/lbm/config.hpp` and rebuild to change the duration. Zero steps reports
the initialized state; negative steps are rejected. Each timestep performs GPU
collision, streaming, and a buffer swap. Readbacks occur at steps 0, 1, 2, 10,
and 1,000 when within the run length, plus the final step. Each step is checked
only once, including zero-length runs. Each report gives maximum population and
density errors, maximum absolute velocity components, total mass, and relative
mass error against the expected rest-state mass (`Nx * Ny`). Nonfinite values or
nonpositive density fail the check. The run stops immediately with status 1 on
failure; it returns 0 only after all scheduled checks pass. These checks validate
rest-state preservation, not general flow accuracy. Use the separate disturbance
mode for CSV output.

## Density disturbance and CSV output

`make disturbance` or `--disturbance` initializes all 16 cells with
**x = 30..33 and y = 30..33** at density **5.00** on a **64 × 64** grid. Every other cell
starts at density **1.00**. This forms a sharp central 4 × 4 square, with no
initial gradient or smoothing. The square remains centered at (31.5,31.5). Density 5 is a strong disturbance,
not a validated low-amplitude flow case; finite-state and mass checks alone do
not establish physical accuracy. Initial velocity is zero. Each cell's populations
are initialized to equilibrium weights times its density; subsequent collision
and streaming evolve them on the GPU.

Defaults in `include/lbm/config.hpp` are `disturbance_steps = 50`, sixteen
`disturbance_centers` (selected cell coordinates), and `density_increase = 4.0`.
`include/lbm/initial_density.hpp` implements the exact coordinate selection.
Coordinates must lie inside the grid; repeated coordinates do not stack increases.
The mode uses the configured relaxation time;
`--simulate` still uses its own `steps = 1000` and uniform rest initialization.

The default output is `disturbance.csv` in the current working directory. An
optional path can be supplied directly to the executable:

```sh
./build/sm_80/debug-0/checkpoint --disturbance my_run.csv
```

An existing file at that path is overwritten. Its parent directory must exist.
File open, write, and close failures cause a nonzero exit status. No GUI or
X-forwarding is used. Copy the finished CSV from TACC to your Mac for plotting.

The CSV header is `step,x,y,density,ux,uy`. Coordinates use x rightward and y
upward; values are in lattice units. Rows are ordered by timestep, then by cell
(`x + Nx*y`). Snapshots include **step 0 and every completed timestep** through
the final step. With the defaults, there are 51 snapshots and 208,896 data rows,
plus the header. Zero disturbance steps saves only the initial snapshot.
Floating-point values are written with round-trip double precision.

Each snapshot is read from the current GPU buffer after any swap. Before writing,
the program checks all populations and derived fields for finite values, positive
density, and total mass against the sum of the prescribed initial density field
(**4160.00** for these defaults),
using the configured relative-error tolerance. These checks allow density and
velocity to evolve; they do not impose the rest-state equilibrium test. A failed
check stops the run, leaving earlier completed snapshots for inspection. Output
I/O failures may leave a partial file; only a successful run establishes that all
requested snapshots were saved.

On a periodic **2 × 2** grid, opposite directions reach the same neighbor. With
this symmetric, zero-velocity initial condition, density redistributes while
velocity can remain zero up to roundoff. A density heatmap shows the change;
stationary velocity arrows were expected in the original checkpoint. The current
64 × 64 default removes that directional ambiguity. Run `python scripts/animate_density.py src/disturbance.csv` from
the project root after installing `scripts/requirements.txt` to save a GIF and
initial/final density-only PNGs in `results/64x64/`. The original `animate.py`
remains available for density plus velocity. CSV output still includes `ux,uy`;
the new density-only script does not parse or display them.

For the current uniform-rest mode, expect 4096 density-1 cells with zero velocity
and total mass 4096 (up to roundoff). The original 2 × 2 configuration passed on
Vista; the new 4 × 4 density-5 settings still require GPU execution verification.

`collision-check` (executable option `--collision-check`) initializes the grid,
downloads its populations, and perturbs cell 0: east and west each gain 0.01,
and rest loses 0.02. It uploads this fixture, executes exactly one GPU collision,
and downloads the result without streaming or swapping buffers. This mode fixes
`tau=1` regardless of the configuration default, so the expected result is rest
equilibrium. It prints all nine populations of cell 0 before and after collision,
then checks every cell's populations, density, velocity, and total mass using the
same tolerance as initialization. A no-op collision fails the population check
even though density and velocity are unchanged. Success returns 0; failure returns 1.

`streaming-check` (`--streaming-check`) uses a fixed 2 × 2 grid and uploads unique
labels `100 * direction + cell + 1`. It skips collision and compares every output
after one stream against a hand-calculated source-cell table. A second stream
must reproduce the original labels exactly. This also checks both buffer swaps
and that no labels were lost or duplicated. All expected/actual values are printed;
failure returns 1. A periodic 2 × 2 grid cannot distinguish opposite spatial
directions, so a 3 × 3 routing check remains a useful next validation case.

```sh
make DEBUG=1 CUDA_ARCH=sm_80
make memcheck DEBUG=1 CUDA_ARCH=sm_80
make clean
```

Debug mode enables device debugging. Build outputs live in architecture/debug
subdirectories under `build/`. `clean` removes those generated outputs. If you
change compiler versions or other build flags, clean before rebuilding.

## Implementation order

1. Implemented: initialization launcher and GPU kernel in `kernels.cu`, with
   initial-state readback and checks in `main.cu`. Verify execution on TACC later.
2. Implemented: BGK collision with one thread per cell, all nine values loaded
   before in-place writes, and relaxation rate `1/tau`. Invalid density or
   nonfinite state triggers a device trap reported at synchronization; terminate
   the run on that error. Periodic pull streaming writes a separate buffer,
   followed by synchronization and a pointer swap. All implemented
   launchers use the supplied stream and check launch errors. The simulation
   controller already synchronizes after each operation for checkpoint debugging.
3. Implemented: initialization, collision-only, and streaming-only checks, plus
   complete-state uploads for test fixtures, and repeated rest-state checks
   using the inputs and expected outputs in `checkpoint_steps.md`. No CPU fluid
   solver is required.
4. Implemented: `--simulate` checks intermediate and final rest states;
   `--disturbance` evolves a density increase and saves every state to CSV.
   Local visualization is available in `scripts/`; TACC execution checks are next.

Grid dimensions are configured in `Config`; kernel launches must derive their
size from the configuration and guard inactive threads. Keep collision local and
in place, stream into the separate buffer, then swap. The executable supports
initialization, collision-only, streaming-only, repeated-timestep, and disturbance modes.
CUDA compilation and GPU checks await TACC.
