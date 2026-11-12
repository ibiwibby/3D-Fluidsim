# Local checkpoint visualization

## Density-only 64 × 64 demonstration

The new `animate_density.py` draws only a density heatmap. It ignores velocity
columns while the CUDA output continues to contain `step,x,y,density,ux,uy`.
After installing the requirements below, run from the repository root:

```sh
make -C src disturbance CUDA_ARCH=sm_86
python scripts/animate_density.py src/disturbance.csv
```

Use `sm_90` on Vista. The simulation initializes all 16 cells with x and y in 30..33
at density 5.00, with all other cells at 1.00, forming a sharp 4 × 4
square in the center of a 64 × 64 grid. It evolves for 50 steps.
Including step 0, that produces 51 frames. The separate uniform-rest validation
mode still runs 1,000 steps.

Outputs default to `results/64x64/disturbance_density.gif`,
`disturbance_density_initial.png`, and `disturbance_density_final.png`. A fixed
color scale is shared across frames. There are no arrows or velocity statistics.
The script accepts `--output-dir`, `--fps`, and `--dpi`, and requires only the
four CSV columns `step,x,y,density`; extra columns are ignored. Existing matching
output files are overwritten. Earlier results elsewhere are preserved.

## Original density-and-velocity visualization

Run from the project root on your Mac:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements.txt
python scripts/animate.py path/to/downloaded/disturbance.csv
```

Omit the input path to read `src/disturbance.csv`. By default, outputs go to the
project's `results/` directory regardless of the working directory:

- `disturbance.gif`: density heatmap and velocity arrows for all saved timesteps.
- `disturbance_initial.png`: first snapshot.
- `disturbance_final.png`: final snapshot.

Output names use the input CSV's stem; rerunning overwrites matching outputs.
The script does not modify the input. Copy the CSV from TACC before running it.
No display, X-forwarding, or FFmpeg is required. Rendering uses Matplotlib's
headless Agg backend and [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).

```sh
python scripts/animate.py results/disturbance.csv --fps 3
python scripts/animate.py my_run.csv --output-dir results/my_run
```

The CSV must have `step,x,y,density,ux,uy` columns, finite values, positive density,
and one row per cell per snapshot. Incomplete grids and duplicate cells are
rejected. Unordered rows are supported; frames are sorted by step. Every saved
snapshot is displayed for an equal duration, so playback speed is illustrative,
not physical time. A complete CSV prefix from an interrupted run can be rendered;
this script cannot certify whether the simulation finished successfully.

Density colors and [velocity arrow scale](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html)
stay fixed across the animation. Cell labels appear for grids up to 64 cells.
Arrows below `1e-12` speed are hidden to avoid magnifying roundoff; the displayed
maximum speed still uses the original data. Change this with `--velocity-floor`.
For the periodic 2x2 disturbance case, zero velocity while density redistributes
is expected. The heatmap shows that redistribution.

All frames are loaded into memory; this script targets the small checkpoint,
not large 3D production datasets. The GIF, initial PNG, and final PNG are generated
only when you provide a CSV; no synthetic output is saved in `results/` by setup.
