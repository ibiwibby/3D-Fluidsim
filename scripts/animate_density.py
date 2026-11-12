#!/usr/bin/env python3
"""Render only density from simulation CSVs; velocity columns are ignored."""
import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_snapshots(path):
    """Validate rectangular, complete snapshots without assuming row order."""
    frames = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"step", "x", "y", "density"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires columns: step,x,y,density")
        for line, row in enumerate(reader, 2):
            try:
                step, x, y = (int(row[key]) for key in ("step", "x", "y"))
                density = float(row["density"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line}") from exc
            if min(step, x, y) < 0 or not math.isfinite(density):
                raise ValueError(f"Negative index or nonfinite value on CSV line {line}")
            if density <= 0:
                raise ValueError(f"Nonpositive density on CSV line {line}")
            frame = frames.setdefault(step, {})
            if (x, y) in frame:
                raise ValueError(f"Duplicate cell ({x},{y}) at step {step}")
            frame[x, y] = density
    if not frames:
        raise ValueError("CSV has no snapshots")
    steps = sorted(frames)
    first = frames[steps[0]]
    nx = max(x for x, _ in first) + 1
    ny = max(y for _, y in first) + 1
    if len(first) != nx * ny:
        raise ValueError("First snapshot is not a complete rectangle starting at (0,0)")
    for step in steps:
        if frames[step].keys() != first.keys():
            raise ValueError(f"Incomplete or inconsistent grid at step {step}")
    return steps, frames, nx, ny


def render(args, steps, frames, nx, ny):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np

    density = np.array([[[frames[s][x, y] for x in range(nx)]
                         for y in range(ny)] for s in steps])
    low, high = float(density.min()), float(density.max())
    if low == high:
        padding = max(abs(low) * 1e-6, 1e-12)
        low, high = low - padding, high + padding
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    heat = ax.imshow(density[0], origin="lower", interpolation="nearest",
                     cmap="viridis", vmin=low, vmax=high)
    fig.colorbar(heat, ax=ax, label="Density (lattice units)")
    ax.set(xlabel="x", ylabel="y", aspect="equal")
    title = ax.set_title("")
    note = fig.supxlabel("", fontsize=9)

    def update(index):
        heat.set_data(density[index])
        title.set_text(f"Density | {nx} × {ny} | Step {steps[index]}")
        note.set_text(f"Mass: {density[index].sum():.12g} | Fixed color scale")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.csv.stem + "_density"
    gif = args.output_dir / f"{stem}.gif"
    animation = FuncAnimation(fig, update, frames=len(steps),
                              interval=1000 / args.fps, blit=False, repeat=True)
    try:
        animation.save(gif, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
        for index, suffix in ((0, "initial"), (len(steps) - 1, "final")):
            update(index)
            fig.savefig(args.output_dir / f"{stem}_{suffix}.png", dpi=args.dpi)
    finally:
        plt.close(fig)
    print(f"Rendered {len(steps)} density snapshots ({nx}x{ny}) to {gif}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=ROOT / "src/disturbance.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/64x64")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.fps <= 50 or args.dpi <= 0:
        parser.error("fps must be 1–50 and dpi must be positive")
    try:
        render(args, *load_snapshots(args.csv))
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install scripts/requirements.txt.", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Density animation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
