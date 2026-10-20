"""Plot supplied spatial fold/role assignments without recomputing the split.

Input CSV columns: sample_id,x,y,fold,role_4km,role_8km. Coordinates are
projected metres; each role is T (train), C (calibration), E (evaluation), or
X (excluded), for the same held-out fold in both buffer columns.
"""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects, font_manager

from .style import configure, plt
from .pdf_assembly import FOLD_SIZE, FOLD_SPINES, assemble_fold

FOLD_COLORS = ["#8297AF", "#E5BE32", "#FC8D62", "#66C2A5", "#B5A4C2"]
ROLE_COLORS = {"T": FOLD_COLORS[0], "C": FOLD_COLORS[1], "E": FOLD_COLORS[2], "X": "#BBBBBB"}


def validate(data, heldout=1):
    required = {"sample_id", "x", "y", "fold", "role_4km", "role_8km"}
    if not required <= set(data.columns) or data.sample_id.duplicated().any() or data[list(required)].isnull().any().any():
        raise ValueError("A complete, unique sample/coordinate/fold/role table is required")
    if set(data.fold) != {1, 2, 3, 4, 5} or heldout not in {1, 2, 3, 4, 5}:
        raise ValueError("fold must contain five fixed folds numbered 1 to 5")
    xy = data[["x", "y"]].to_numpy(float) / 1000
    if not np.isfinite(xy).all() or (np.ptp(xy, axis=0) <= 0).any():
        raise ValueError("Coordinates must be finite projected metres with nonzero extent")
    for buffer in (4, 8):
        roles = data[f"role_{buffer}km"].to_numpy(str)
        if not set(roles) <= set(ROLE_COLORS):
            raise ValueError("Role columns allow only T, C, E, X")
        if not np.array_equal(roles == "E", data.fold.to_numpy() == heldout):
            raise ValueError("Evaluation-role rows must match the supplied held-out fold")
    return xy


def plot(data, output, heldout=1, png=False):
    xy = validate(data, heldout)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    width, height = FOLD_SIZE
    # Publication canvas and physical axes position. The original exporter used
    # five-percent autoscale margins and 2.4 pt^2 vector sample markers.
    margin = np.ptp(xy, axis=0) * .05
    lower, upper = xy.min(axis=0) - margin, xy.max(axis=0) + margin
    if not (400 <= lower[0] < upper[0] <= 700 and 2300 <= lower[1] < upper[1] <= 2600):
        raise ValueError("This publication layout requires Maoming-Yangjiang UTM 49N coordinates; "
                         "use a different cartographic layout for another region")
    # Pad the shorter extent to retain an equal physical map scale *and* the
    # full fixed axes width. This prevents aspect='equal' moving the spines
    # away from their aligned legend when a supplied subset has another extent.
    target_ratio = (FOLD_SPINES[1] - FOLD_SPINES[0]) / 81.653162810
    centre = (lower + upper) / 2
    span = upper - lower
    if span[0] / span[1] < target_ratio:
        span[0] = span[1] * target_ratio
    else:
        span[1] = span[0] / target_ratio
    lower, upper = centre - span / 2, centre + span / 2
    font_path = font_manager.findfont(plt.rcParams["font.family"][0], fallback_to_default=False)
    for i in range(3):
        fig = plt.figure(figsize=(width / 72, height / 72))
        ax = fig.add_axes([FOLD_SPINES[0] / width, 47.205299668 / height,
                          (FOLD_SPINES[1] - FOLD_SPINES[0]) / width, 81.653162810 / height])
        if i == 0:
            counts = []
            for fold, color in enumerate(FOLD_COLORS, 1):
                selected = data.fold.to_numpy() == fold
                counts.append(int(selected.sum()))
                ax.scatter(xy[selected, 0], xy[selected, 1], s=2.4, color=color, linewidths=0)
                ax.text(*np.median(xy[selected], axis=0), str(fold), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="black",
                        path_effects=[patheffects.withStroke(linewidth=1.8, foreground="white")])
            title = "Fixed spatial folds"
            ax.text(.5, -.31, "Folds 1-5:\n" + ", ".join(map(str, counts)), transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.9, linespacing=1.35)
        else:
            buffer = (4, 8)[i - 1]
            roles = data[f"role_{buffer}km"].to_numpy(str)
            counts = {role: int((roles == role).sum()) for role in "TCEX"}
            for role in ("X", "T", "C", "E"):
                selected = roles == role
                ax.scatter(xy[selected, 0], xy[selected, 1], s=2.4, color=ROLE_COLORS[role], linewidths=0)
            title = f"Fold {heldout}: {buffer} km buffer"
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (km)", fontsize=8.3, labelpad=4)
        ax.set_ylabel("Northing (km)", fontsize=8.3, labelpad=4)
        ax.set_title(title, fontsize=8.9, loc="left", pad=5)
        # Fixed coordinate ticks belong to the Maoming-Yangjiang figure layout,
        # not to the algorithm or spatial split.
        ax.set_xticks([440, 520, 600])
        ax.set_yticks([2380, 2440, 2500])
        # Explicit tick positions must not expand a smaller supplied display
        # extent and thereby shrink the equal-aspect axes after alignment.
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        ax.tick_params(labelsize=7.6, length=2, pad=2, width=.65, color="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(.65)
        fig.canvas.draw()
        position = ax.get_position()
        if not np.allclose([position.x0 * width, position.x1 * width], FOLD_SPINES, atol=1e-7, rtol=0):
            raise RuntimeError("Spatial panel frame moved away from its aligned legend")
        destination = output / f"Fig03{'abc'[i]}.pdf"
        if i == 0:
            fig.savefig(destination)
        else:
            stream = BytesIO()
            fig.savefig(stream, format="pdf")
            stream.seek(0)
            assemble_fold(stream, counts, font_path, destination, remove_old_footer=False)
        plt.close(fig)
        if png:
            # Render the completed PDF so that the PNG includes its footer.
            import shutil
            import subprocess
            renderer = shutil.which("pdftoppm")
            if renderer is None:
                raise RuntimeError("--png requires Poppler pdftoppm on PATH; PDF has already been saved")
            subprocess.run([renderer, "-singlefile", "-r", "180", "-png", str(destination),
                            str(destination.with_suffix(""))], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--heldout-fold", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--png", action="store_true", help="Also render the assembled PDF using Poppler")
    args = parser.parse_args()
    configure(args.font_dir, args.font_family)
    plot(pd.read_csv(args.input), args.output, args.heldout_fold, args.png)


if __name__ == "__main__":
    main()
