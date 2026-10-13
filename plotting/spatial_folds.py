"""Plot supplied spatial fold/role assignments without recomputing the split.

Input CSV columns: sample_id,x,y,fold,role_4km,role_8km. Coordinates are
projected metres; each role is T (train), C (calibration), E (evaluation), or
X (excluded), for the same held-out fold in both buffer columns.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from .style import configure, save, plt

FOLD_COLORS = ["#8297AF", "#E5BE32", "#FC8D62", "#66C2A5", "#B5A4C2"]
ROLE_COLORS = {"T": FOLD_COLORS[0], "C": FOLD_COLORS[1], "E": FOLD_COLORS[2], "X": "#BBBBBB"}


def plot(data, output, heldout=1, png=False):
    required = {"sample_id", "x", "y", "fold", "role_4km", "role_8km"}
    if not required <= set(data.columns) or data.sample_id.duplicated().any() or data[list(required)].isnull().any().any():
        raise ValueError("A complete, unique sample/coordinate/fold/role table is required")
    if set(data.fold) != {1, 2, 3, 4, 5}:
        raise ValueError("fold must contain five fixed folds numbered 1 to 5")
    xy = data[["x", "y"]].to_numpy(float) / 1000
    if not np.isfinite(xy).all():
        raise ValueError("Coordinates must be finite projected metres")
    xlim = (xy[:, 0].min() - 8, xy[:, 0].max() + 8)
    ylim = (xy[:, 1].min() - 8, xy[:, 1].max() + 8)
    for i in range(3):
        fig = plt.figure(figsize=(3.2, 3.1))
        ax = fig.add_axes([.19, .28, .77, .60])
        if i == 0:
            counts = []
            for fold, color in enumerate(FOLD_COLORS, 1):
                selected = data.fold.to_numpy() == fold
                counts.append(int(selected.sum()))
                ax.scatter(xy[selected, 0], xy[selected, 1], s=5.5, color=color, linewidths=0, rasterized=True)
                ax.text(*np.median(xy[selected], axis=0), str(fold), ha="center", va="center", fontsize=12,
                    fontweight="bold", path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
            title = "Fixed spatial folds"
            summary = "Folds 1–5:\n" + ", ".join(map(str, counts))
        else:
            buffer = (4, 8)[i - 1]
            roles = data[f"role_{buffer}km"].to_numpy(str)
            if not set(roles) <= set(ROLE_COLORS):
                raise ValueError("Role columns allow only T, C, E, X")
            if not np.array_equal(roles == "E", data.fold.to_numpy() == heldout):
                raise ValueError("Evaluation-role rows must match the supplied held-out fold")
            counts = {}
            for role in ("X", "T", "C", "E"):
                selected = roles == role
                counts[role] = int(selected.sum())
                ax.scatter(xy[selected, 0], xy[selected, 1], s=5.5, color=ROLE_COLORS[role], linewidths=0, rasterized=True)
            title = f"Fold {heldout}: {buffer} km buffer"
            summary = f"T {counts['T']:,}   C {counts['C']:,}   E {counts['E']:,}\nExcluded {counts['X']:,}"
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting (km)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Northing (km)", fontsize=11)
        ax.set_title(title, fontsize=12, loc="left", pad=6)
        ax.tick_params(labelsize=10, length=3, width=.7, color="black")
        for spine in ax.spines.values():
            spine.set_color("black")
        fig.text(.57, .09, summary, ha="center", va="center", fontsize=10)
        save(fig, output, "fig03_" + "abc"[i], png)
    fig = plt.figure(figsize=(6, .4))
    handles = [Line2D([], [], marker="s", linestyle="", markersize=5, color=ROLE_COLORS[role],
               markeredgecolor="black", markeredgewidth=.3, label=label)
               for role, label in [("T", "Train (T)"), ("C", "Calibration (C)"), ("E", "Test (E)"), ("X", "Excluded")]]
    fig.legend(handles=handles, ncol=4, loc="center", frameon=False, fontsize=11, handletextpad=.3, columnspacing=.8)
    save(fig, output, "fig03_legend", png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--heldout-fold", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    configure(args.font_dir, args.font_family)
    plot(pd.read_csv(args.input), args.output, args.heldout_fold, args.png)


if __name__ == "__main__":
    main()

