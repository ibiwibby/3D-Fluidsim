# D3Q19 CUDA fluid simulation

The 64-cubed preset and optional closed x walls are documented in
[tacc/README.md](tacc/README.md). Use `make -C src_3D channel64`, or select
`--channel --size 64 --closed-ends` through the runtime options.
The TACC package and run scripts are in `tacc/`.
Open saved VTI/PVD data with the [browser renderer](../renderer/README.md).

This is a separate 3D implementation. The existing `src/` 2D solver and its
visualization scripts are unchanged.

## What runs

- **32 × 32 × 32 lattice sites**, double precision, BGK collision, `tau = 1`.
- D3Q19: one rest direction, six axial directions, twelve face diagonals; no body diagonals.
- Weights: `1/3` at rest, `1/18` on axes, `1/36` on diagonals.
- Original rest/disturbance modes remain periodic on x, y, and z.
- New `--channel` mode uses stationary walls, a central sphere, and sustained forcing (details below).
- Default disturbance: a sharp central **4 × 4 × 4 cube**, indices **14..17** on
  each axis, at density **1.10**; background density **1.00**, velocity initially zero.
- **50 timesteps**, snapshots from step 0 through step 50. Expected mass **32774.4**.

The earlier 2D density-5 experiment is deliberately not the default here: begin
3D verification with a small disturbance. Basic finite-state and mass checks do
not establish physical accuracy or guarantee low-Mach validity for arbitrary inputs.

## Build and run on an A100

Within an allocated GPU session, load the available compatible CUDA/compiler
modules (`module load cuda`, then `nvcc --version`). From the repository root:

```sh
make -C src_3D CUDA_ARCH=sm_80
make -C src_3D device-info CUDA_ARCH=sm_80
make -C src_3D check CUDA_ARCH=sm_80
make -C src_3D memcheck CUDA_ARCH=sm_80
make -C src_3D rest CUDA_ARCH=sm_80 OUTPUT=../results/3D_rest
make -C src_3D run CUDA_ARCH=sm_80
```

`sm_80` is the Makefile default for A100. Use `sm_90` on Vista GH200 or `sm_86`
on the applicable laptop GPU. Rebuild natively on each host architecture; do not
copy Vista executables to Lonestar6. Make targets do not allocate GPU resources.
See [Lonestar6](https://docs.tacc.utexas.edu/hpc/lonestar6/) or
[Vista](https://docs.tacc.utexas.edu/hpc/vista/) for allocation/module details.

The executable is `src_3D/build/sm_80/debug-0/fluid3d`. `DEBUG=1` enables device
debugging in a separate build directory. `make -C src_3D clean` removes build
outputs, not simulation results.

The default output directory for `make run` is **`results/3D/`** in the repo.
Choose a new directory for each run; the program rejects nonempty directories
rather than mixing or overwriting snapshots.

```sh
# OUTPUT paths are relative to src_3D when using make -C src_3D.
make -C src_3D run STEPS=100 OUTPUT_EVERY=5 OUTPUT=../results/3D_run2

# Direct invocation: paths are relative to the current working directory.
./src_3D/build/sm_80/debug-0/fluid3d --disturbance \
    --steps 50 --output-every 2 --output results/3D_run3
```

Run modes are `--device-info`, `--check`, `--rest`, `--disturbance`, and `--channel`.
`--steps`, `--output-every`, and `--output` apply to rest/disturbance runs. Other
parameters, including dimensions, bump width, and relaxation time, are in
`include/lbm/config.hpp`; change them and rebuild. Negative steps and invalid
parameters are rejected. A zero-step run saves just the initialized state. The
final state is saved even when its step is not a multiple of the output interval.

## Code map and update sequence

| File | Purpose |
| --- | --- |
| `include/lbm/config.hpp` | Defaults, allocation sizing, input validation |
| `include/lbm/d3q19.hpp` | Direction convention, weights, flattening, equilibrium |
| `simulation.cu`, `include/lbm/simulation.cuh` | GPU kernels and RAII ownership of device state |
| `include/lbm/cuda_utils.cuh` | CUDA error checks, reused from the 2D framework |
| `main.cu` | Modes, CLI, GPU checks, timestep control, diagnostics |
| `output.cpp`, `include/lbm/output.hpp` | Dependency-free binary VTI and PVD writing |
| `tests/host_checks.cpp` | Host checks of lattice moments, equilibrium, sizing, output fixture |
| `tests/check_vti.py` | Independent binary file framing/order checks |
| `Makefile` | A100 build, run, test and sanitizer targets |

Flatten sites as `x + nx*(y + ny*z)` with x varying fastest. Populations remain
in structure-of-arrays form, `direction * site_count + site`. Kernels use one
active CUDA thread per site with a bounds guard, 128 threads per block.

Each timestep computes local density and all three velocity components, applies
BGK collision in place, pulls the 19 populations from periodic upstream neighbors
into the second buffer, then swaps pointers. The collision kernel loads all local
populations before writing. Two population buffers persist for the run. A separate
GPU buffer holds density and xyz velocity fields for output.

A GPU extraction kernel computes `rho,ux,uy,uz`. Only these four values per site
are downloaded for snapshots and diagnostics, not all 19 distributions. Full
population download is available for correctness tests. Invalid nonfinite state
or nonpositive density triggers an error; a device trap requires terminating the
run, not attempting to recover that CUDA context.

For debugging, kernels synchronize after each operation and field checks run
every step, even if full snapshots are less frequent. This is a correctness-first
implementation, not a tuned performance benchmark. Future optimizations can move
reductions to the device and reduce synchronization and diagnostic transfers.

## Output: binary fields, small CSV summaries

At 32³, CSV is still possible, but binary structured output is a better foundation
for later 128³/256³ runs. The implemented output is:

| File | Contents |
| --- | --- |
| `flow_000000.vti`, etc. | Uncompressed binary Float64 density, 3-component velocity, and a solid mask |
| `flow.pvd` | Snapshot filenames and lattice timestamps for a ParaView time series |
| `diagnostics.csv` | One row per step: mass, relative mass error, min/max density, max speed, mean x velocity, max Mach number |
| `run.txt` | Dimensions, steps, output interval, tau, and initialization parameters |

The VTI representation uses **point data at the LBM lattice-site coordinates**
(0 through N−1), unit spacing, origin zero. This avoids ambiguity between the
number of lattice sites and the number of geometric VTK cells. Velocity is stored
as interleaved xyz tuples. The writer uses little-endian raw appended binary with
UInt64 byte-count headers; a general-purpose XML parser cannot parse the raw
payload as text, so use a VTK-aware reader. No VTK C++ library is required to build.
The layout follows the [VTK XML format specification](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html).

Each 32³ field snapshot has **1.25 MiB** of raw density/velocity data, plus a small
header. Fifty-one snapshots use approximately **63.75 MiB**. The two GPU population
buffers total **9.5 MiB**, plus **1 MiB** for extracted fields, excluding runtime
allocations. For comparison, with the solid mask, double-precision field snapshots are 80 MiB at 128³
and 640 MiB at 256³; save less frequently at those sizes.

Output alternatives:

- **Binary VTI + PVD (implemented):** simple uniform-grid geometry and direct
  scientific-viewer support; separate files are easy to transfer or inspect.
- **HDF5 + XDMF:** useful for larger time series, compression, chunked access, and
  potentially parallel I/O, but adds a library dependency and metadata work.
- **NumPy `.npy`/`.npz`:** convenient for a Python-only analysis workflow, but
  requires an agreed shape/order and additional viewer conversion.
- **CSV:** retain for small diagnostics and debugging; avoid millions of repeated
  coordinate/value text rows for full 3D fields.

These visualization snapshots are **not restart checkpoints**: restarting LBM
requires all directional populations and the corresponding configuration, not
just density and velocity.

## Viewing on your laptop

Copy the entire output directory, including `flow.pvd` and all referenced `.vti`
files. Open **`flow.pvd` in ParaView**, apply it, and use a Slice through the center
colored by density or velocity magnitude. Fix the color range over all timesteps
and use the time controls to inspect the wave. Streamlines and volume rendering
can be added later. No X-forwarding is needed when the viewer runs locally.
The current 2D CSV animation scripts do not read these 3D files.

A failed run can leave a valid partial series: the PVD lists only completed
snapshots, but that does not mean the requested final step completed. Check the
process exit status and success message. Preserve the terminal output separately
if you need GPU/compiler provenance; `run.txt` records simulation parameters.

## Validation and current limits

`make check` runs on the GPU:

1. Rest initialization and a nonequilibrium collision fixture with zero net momentum.
2. Exact labeled streaming against an independent forward-push oracle on a
   **5 × 6 × 7** domain, checking all directions, periodic faces/edges, and two swaps.
3. Rest and nonzero uniform xyz flow for 1,000 timesteps, checking all populations
   and extracted fields. The non-cubic test also exercises partial CUDA blocks.

`make rest` additionally checks the configured 32³ domain at every step.
Disturbance runs check finite fields, positive density, and relative mass error
against `site_count + bump_width^3*density_increase` using the configured tolerance.

Local checks requiring no CUDA:

```sh
make -C src_3D host-check
```

These passed during implementation: lattice symmetry, second/fourth moments,
equilibrium moments, config sizing, binary VTI offsets/lengths, x/y/z order, vector
component order, and PVD time metadata. Build-command dry runs also passed.
**CUDA compilation/execution and opening the generated files in ParaView still
need verification on the target environment.** No speedup, 3D physical benchmark,
or CPU/GPU reference agreement is claimed yet. The method follows standard BGK
LBM with the D3Q19 weights described in [this theory reference](https://docs.mstarcfd.com/17_Theory/theory-and-implementation.html).


## Flow around a stationary obstacle

The new `--channel` mode models a completely fluid-filled square duct, with flow
in **+x** around a central radius-4 sphere on the 32³ grid. The sphere is centered
at (15.5,15.5,15.5) and represented by solid lattice sites (a stair-step surface).
The y=0/31 and z=0/31 planes are stationary walls. x is periodic, so this is a
repeating channel segment driven by body force, not an inlet/outlet experiment
or an isolated sphere in unbounded flow.

The initial fluid density is 1 with zero physical velocity. Guo forcing supplies
a constant force density `force_x=1e-5` only at fluid sites. Physical velocity
includes the half-force correction `u=(sum(f_i*c_i)+F/2)/rho` in both collision
and output. Initialization compensates for that correction so step 0 is at rest.
See [Guo, Zheng and Shi (2002)](https://doi.org/10.1103/PhysRevE.65.046308)
for the force discretization. The force source is
`w_i*(1-omega/2)*[3*(c_ix-u_x)+9*(c_i·u)*c_ix]*F_x`.

At fluid-solid links, halfway bounce-back returns the outgoing population to
its opposite direction at the same fluid site. Solid sites are skipped by
collision and do not supply populations to the fluid. Only fluid mass is counted.
The wall lies halfway along each fluid-solid link; geometric accuracy of the
stair-step sphere still needs refinement studies.

From the repository root on A100:

```sh
make -C src_3D check CUDA_ARCH=sm_80
make -C src_3D memcheck CUDA_ARCH=sm_80
make -C src_3D channel CUDA_ARCH=sm_80
```

Use `sm_90` for Vista. The channel defaults to **3,000 steps**, writing every
**50 steps** plus step 0 and the final state into **results/3D_channel/** (61
snapshots). This is a starting transient run length, not a claim of convergence.
A steady solution should be established by comparing later velocity fields or
flow-rate history. The default tau=1 implies lattice viscosity 1/6.

```sh
make -C src_3D channel CHANNEL_STEPS=5000 CHANNEL_OUTPUT=../results/channel_long
# Baseline without the obstacle, using the same walls and force:
./src_3D/build/sm_80/debug-0/fluid3d --channel --no-obstacle \
    --steps 3000 --output-every 50 --output results/channel_empty
```

`diagnostics.csv` reports fluid mass, density range, maximum speed, mean fluid
x velocity, and maximum Mach number. The channel run rejects maximum Mach >=0.1
as a chosen low-Mach guard; reduce `force_x` in config if necessary. Finite-state
and mass checks remain active. These guards do not replace a physical reference
benchmark. Neither vortex shedding nor high-Reynolds-number flow is promised.

### ParaView view

Open the output's `flow.pvd` with its VTI files alongside it. Each file contains:

- `density` and `velocity`: fluid fields; set to **zero at solid sites** as output
  placeholders, not physical fluid values. Exclude solids from density plots.
- `solid`: **0=fluid, 1=wall, 2=sphere**. This is not liquid fraction.

For a first view, create a central z slice (origin 15.5,15.5,15.5; normal 0,0,1)
colored by **velocity magnitude**, with a fixed color range over time. On a second
branch from the original dataset, apply Threshold to `solid`, retaining 2..2
(and all-point matching if offered), then show it as a solid gray surface to
identify the obstacle. A Threshold retaining 0..0 can isolate fluid for
streamlines; place streamline seeds upstream near x=4 across the interior duct.
The point-data mask can make thresholded surfaces offset by part of a lattice
cell; the numerical geometry is defined by the site mask and halfway links.
A volume view of density alone is not the clearest way to show this nearly
constant-density flow. Use velocity slices/streamlines, with an Outline for context.

### Added validation

`--check` now also tests exact bounce-back routing against a forward-routing
oracle, zero-force rest with walls and sphere, force mass/momentum increments,
and a 200-step driven case for fluid mass conservation, positive mean x flow,
finite fields, and zero output velocity in solids. Host tests verify opposite
indices, force moments, geometry, and VTI solid-mask framing. Host tests passed
locally; the **new CUDA channel tests and run still need execution on TACC**.
A quantitative no-obstacle duct-flow benchmark and grid refinement are next;
free-surface physics and liquid fraction are intentionally deferred.
