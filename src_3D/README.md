# D3Q19 CUDA fluid simulation

This is a separate 3D implementation. The existing `src/` 2D solver and its
visualization scripts are unchanged.

## What runs

- **32 × 32 × 32 lattice sites**, double precision, BGK collision, `tau = 1`.
- D3Q19: one rest direction, six axial directions, twelve face diagonals; no body diagonals.
- Weights: `1/3` at rest, `1/18` on axes, `1/36` on diagonals.
- Periodic boundaries on x, y, and z. No solid walls, obstacles, forcing, or inlet/outlet yet.
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

Run modes are `--device-info`, `--check`, `--rest`, and `--disturbance`.
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
| `flow_000000.vti`, etc. | Uncompressed binary Float64 density and 3-component velocity |
| `flow.pvd` | Snapshot filenames and lattice timestamps for a ParaView time series |
| `diagnostics.csv` | One row per step: mass, relative mass error, min/max density, max speed |
| `run.txt` | Dimensions, steps, output interval, tau, and initialization parameters |

The VTI representation uses **point data at the LBM lattice-site coordinates**
(0 through N−1), unit spacing, origin zero. This avoids ambiguity between the
number of lattice sites and the number of geometric VTK cells. Velocity is stored
as interleaved xyz tuples. The writer uses little-endian raw appended binary with
UInt64 byte-count headers; a general-purpose XML parser cannot parse the raw
payload as text, so use a VTK-aware reader. No VTK C++ library is required to build.
The layout follows the [VTK XML format specification](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html).

Each 32³ field snapshot has **1 MiB** of raw density/velocity data, plus a small
header. Fifty-one snapshots use approximately **51 MiB**. The two GPU population
buffers total **9.5 MiB**, plus **1 MiB** for extracted fields, excluding runtime
allocations. For comparison, double-precision field snapshots are 64 MiB at 128³
and 512 MiB at 256³; save less frequently at those sizes.

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
