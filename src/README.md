# CUDA checkpoint skeleton

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

`make disturbance` or `--disturbance` initializes cell `(8,8)` at density **1.10**
and every other cell at density **1**, all with zero velocity. It sets the bumped
cell's nine populations to their local equilibrium weights times density, then
runs collision and streaming on the GPU. This initial condition is distinct from
the nonequilibrium fixture used by `--collision-check`.

Defaults in `include/lbm/config.hpp` are a 16 × 16 grid, `disturbance_steps = 100`,
`disturbance_x = 8`, `disturbance_y = 8`, and `density_increase = 0.10`.
Coordinates outside the configured grid are rejected. The mode uses the configured relaxation time;
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
the final step. With the defaults, there are 101 snapshots and 25,856 data rows,
plus the header. Zero disturbance steps saves only the initial snapshot.
Floating-point values are written with round-trip double precision.

Each snapshot is read from the current GPU buffer after any swap. Before writing,
the program checks all populations and derived fields for finite values, positive
density, and total mass against `Nx*Ny + density_increase` (default **256.10**),
using the configured relative-error tolerance. These checks allow density and
velocity to evolve; they do not impose the rest-state equilibrium test. A failed
check stops the run, leaving earlier completed snapshots for inspection. Output
I/O failures may leave a partial file; only a successful run establishes that all
requested snapshots were saved.

On a periodic **2 × 2** grid, opposite directions reach the same neighbor. With
this symmetric, zero-velocity initial condition, density redistributes while
velocity can remain zero up to roundoff. A density heatmap shows the change;
stationary velocity arrows were expected in the original checkpoint. The current
16 × 16 default removes that directional ambiguity. Run `python scripts/animate.py src/disturbance.csv` from
the project root after installing `scripts/requirements.txt` to save a GIF and
initial/final PNGs in `results/`.

For the current uniform-rest mode, expect 256 density-1 cells with zero velocity
and total mass 256 (up to roundoff). The original 2 × 2 configuration passed on
Vista; the new 16 × 16 disturbance settings still require execution verification.

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
