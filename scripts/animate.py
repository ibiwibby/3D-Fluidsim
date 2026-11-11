#!/usr/bin/env python3
"""Render CUDA snapshot CSVs locally without a display server."""

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
        required = {"step", "x", "y", "density", "ux", "uy"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires columns: step,x,y,density,ux,uy")
        for line, row in enumerate(reader, 2):
            try:
                step, x, y = (int(row[key]) for key in ("step", "x", "y"))
                fields = tuple(float(row[key]) for key in ("density", "ux", "uy"))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line}") from exc
            if min(step, x, y) < 0 or not all(math.isfinite(v) for v in fields):
                raise ValueError(f"Negative index or nonfinite value on CSV line {line}")
            if fields[0] <= 0:
                raise ValueError(f"Nonpositive density on CSV line {line}")
            frame = frames.setdefault(step, {})
            if (x, y) in frame:
                raise ValueError(f"Duplicate cell ({x},{y}) at step {step}")
            frame[x, y] = fields
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
    # Delay imports so --help and CSV validation work without plotting packages.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np

    data = np.array([
        [[frames[step][x, y] for x in range(nx)] for y in range(ny)]
        for step in steps
    ])
    rho, ux, uy = data[..., 0], data[..., 1], data[..., 2]
    low, high = float(rho.min()), float(rho.max())
    if low == high:
        padding = max(abs(low) * 1e-6, 1e-12)
        low, high = low - padding, high + padding
    max_speed = float(np.hypot(ux, uy).max())
    # Do not visually magnify floating-point noise into apparent motion.
    arrow_scale = max(max_speed, args.velocity_floor) / 0.65
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), layout="constrained")
    fig.set_facecolor("#f7f8fa")
    heat = axes[0].imshow(rho[0], origin="lower", interpolation="nearest",
                          vmin=low, vmax=high, cmap="viridis")
    axes[0].set_title("Density — fixed color scale")
    fig.colorbar(heat, ax=axes[0], label="Density (lattice units)", shrink=0.75)
    axes[1].set_title("Velocity — fixed arrow scale")
    axes[1].scatter(x, y, color="#adb5bd", s=12)
    arrows = axes[1].quiver(x, y, np.zeros_like(x, dtype=float),
                           np.zeros_like(y, dtype=float), angles="xy",
                           scale_units="xy", scale=arrow_scale, pivot="mid",
                           color="#176b87", width=0.012)
    if max_speed >= args.velocity_floor:
        axes[1].quiverkey(arrows, 0.5, 1.02, max_speed,
                          f"{max_speed:.3g} lattice units", labelpos="E")
    annotations = []
    if nx * ny <= 64:
        for yy in range(ny):
            for xx in range(nx):
                annotations.append((xx, yy, axes[0].text(
                    xx, yy, "", ha="center", va="center", fontsize=10,
                    color="white", bbox=dict(facecolor="black", alpha=0.45, edgecolor="none"))))
    for ax in axes:
        ax.set(xlim=(-0.5, nx - 0.5), ylim=(-0.5, ny - 0.5), xlabel="x", ylabel="y")
        ax.set_aspect("equal")
        if nx <= 16 and ny <= 16:
            ax.set_xticks(range(nx))
            ax.set_yticks(range(ny))
            ax.set_xticks(np.arange(nx + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(ny + 1) - 0.5, minor=True)
            ax.grid(which="minor", alpha=0.3)
    title = fig.suptitle("")
    note = fig.supxlabel("", fontsize=9)

    def update(index):
        heat.set_data(rho[index])
        speed = np.hypot(ux[index], uy[index])
        moving = speed >= args.velocity_floor
        arrows.set_UVC(np.ma.masked_where(~moving, ux[index]),
                       np.ma.masked_where(~moving, uy[index]))
        for xx, yy, text in annotations:
            text.set_text(f"{rho[index, yy, xx]:.6f}")
        title.set_text(f"D2Q9 fluid checkpoint | {nx} × {ny} | Step {steps[index]}")
        note.set_text(
            f"Mass: {rho[index].sum():.12g}    Max speed: {speed.max():.3g}\n"
            f"Arrows below {args.velocity_floor:g} are hidden; gray dots mark cell centers."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.csv.stem
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
    print(f"Rendered {len(steps)} snapshots ({nx}x{ny}) to {gif}")
    print(f"Initial/final PNGs saved in {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=ROOT / "src/disturbance.csv",
                        help="CUDA snapshot CSV (default: project src/disturbance.csv)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results",
                        help="Output directory (default: project results/)")
    parser.add_argument("--fps", type=int, default=5, help="Playback frames/sec (1–50; default 5)")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--velocity-floor", type=float, default=1e-12,
                        help="Hide arrows below this speed; values in CSV are unchanged")
    args = parser.parse_args()
    if not 1 <= args.fps <= 50 or args.dpi <= 0:
        parser.error("fps must be 1–50 and dpi must be positive")
    if not math.isfinite(args.velocity_floor) or args.velocity_floor <= 0:
        parser.error("velocity-floor must be finite and positive")
    try:
        render(args, *load_snapshots(args.csv))
    except ImportError as exc:
        print(f"Missing plotting dependency: {exc}. Install scripts/requirements.txt.", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Animation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
