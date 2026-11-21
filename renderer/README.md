# VTK browser renderer

**Flow:** TACC GPU runs CUDA D3Q19 ? VTI snapshots + PVD timeline ? local Python
packs the data into one HTML file ? browser plays the saved simulation.
Python runs on the CPU. Browser JavaScript handles playback, interpolation,
tracers and camera controls on the CPU; **Three.js/WebGL2 renders on your local
GPU**. The browser does not rerun LBM. Moving the camera needs no new TACC run.

## Run it

From the repository root, with Python 3.10+:

```sh
python -m pip install -r renderer/requirements.txt
python renderer/src/import_vtk.py results/channel64_closed_3600379_20261120T235757Z/flow.pvd renderer/output/closed.html
```

Double-click `renderer/output/closed.html` to open in current Chrome or Edge.
The generated HTML is self-contained and can be shared without Python or VTK.
Use Play/Pause and the time slider; drag to orbit and scroll to zoom. Choose
volume or slice, density or speed, and optionally arrows or flow tracers.
Tracer travel defaults to **10x exaggeration**; choose 1x for actual lattice
velocities. Playback loops, and adaptive colors can emphasize tiny residuals.

## New GPU recordings

The [64-cubed launcher](../src_3D/tacc/README.md) now defaults to **600 steps,
saved every 5 steps** (121 snapshots, about 1.2 GiB per case). On an allocated
Vista GPU, from `src_3D`:

```sh
CUDA_ARCH=sm_90 bash tacc/run64.sh closed
```

Use `periodic` or `both` for the other cases. Copy the entire output directory
back, including `flow.pvd`, every referenced `.vti`, and `run.txt`. Run the
Python command above with that new PVD path. The example already in the repo
is the older sparse recording, not the new 600-step run.

**Input:** uncompressed single-piece VTI ImageData, PointData `density`
(scalar), `velocity` (3 components), optional `solid` (0 fluid, 1 wall,
2 obstacle). ASCII or raw appended arrays are supported. PVD timestamps must
increase and grids must match. `run.txt` supplies boundary labels and detects
partial recordings. A single VTI also works, without animation.

This is a field visualization of a filled channel/box, not a liquid surface.
If a future solver exports a physical liquid fraction in [0,1], add
`--fill-field liquid_fraction` to extract its surface. Density alone does not
identify an air/water interface.

`src/` contains the importer and viewer; `vendor/` contains licensed Three.js.
Generated HTML goes in ignored `output/`. Large recordings load fully into
browser memory; use a shorter series if memory is limited.
