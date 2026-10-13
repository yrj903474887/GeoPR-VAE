"""Shared publication typography, colours and external legends."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch
import numpy as np

METHODS = ["AE", "VAE", "VAE_patch", "GANomaly", "Skip_GANomaly", "GeoPR-VAE"]
LABELS = {"AE": "AE", "VAE": "VAE", "VAE_patch": "Patch VAE", "GANomaly": "GANomaly",
          "Skip_GANomaly": "Skip-GANomaly", "GeoPR-VAE": "GeoPR-VAE"}
BASE_COLORS = dict(AE="#809BB2", VAE="#DDB526", VAE_patch="#8B8E93", GANomaly="#398E88",
                   Skip_GANomaly="#9975AB", **{"GeoPR-VAE": "#EE7955"})
COLORS = {m: to_hex(.75 * np.asarray(to_rgb(c)) + .25) for m, c in BASE_COLORS.items()}
LINE_STYLES = dict(AE="-", VAE="--", VAE_patch=":", GANomaly="-.",
                   Skip_GANomaly=(0, (5, 2, 1, 2)), **{"GeoPR-VAE": "-"})


def configure(font_dir=None, family="Times New Roman"):
    """Use a locally licensed Times installation; no font files are distributed."""
    if font_dir:
        for path in Path(font_dir).glob("*.ttf"):
            font_manager.fontManager.addfont(str(path))
    try:
        font_manager.findfont(family, fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError("Install Times New Roman, pass --font-dir, or explicitly choose --font-family") from error
    plt.rcParams.update({"font.family": family, "font.size": 18, "axes.labelsize": 19,
        "xtick.labelsize": 17, "ytick.labelsize": 17, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False, "savefig.facecolor": "white", "path.simplify": False})


def frame(ax, scale=1, grid_x=False):
    ax.grid(axis="both" if grid_x else "y", color="#d9d9d9", linewidth=.6 * scale)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0 * scale)
    ax.tick_params(length=4 * scale, width=.85 * scale, pad=6 * scale, color="#000000")


def legend_below(fig, ax, handles=None, bottom=.035, fontsize=17):
    if handles is None:
        handles = [Patch(facecolor=COLORS[m], edgecolor="black", linewidth=.8, label=LABELS[m]) for m in METHODS]
    position = ax.get_position()
    ordered = [handles[i] for i in (0, 3, 1, 4, 2, 5)]
    legend = fig.legend(handles=ordered, loc="lower left", bbox_to_anchor=(position.x0, bottom, position.width, .15),
        bbox_transform=fig.transFigure, mode="expand", borderaxespad=0, ncol=3,
        frameon=True, fancybox=True, facecolor="white", edgecolor="black", framealpha=1,
        fontsize=fontsize, handlelength=1.4, handletextpad=.35, labelspacing=.25,
        columnspacing=.55, borderpad=.35)
    legend.get_frame().set_linewidth(.8)
    return legend


def new_axes(ylabel, title=None):
    fig = plt.figure(figsize=(8, 5.4))
    ax = fig.add_axes([.155, .33, .825, .61])
    ax.set_ylabel(ylabel, labelpad=5)
    frame(ax)
    if title:
        ax.set_title(title, fontsize=19, pad=7)
    legend_below(fig, ax)
    return fig, ax


def save(fig, output, stem, png=False):
    """Save a standalone unlettered panel for LaTeX subfigure placement."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (stem + ".pdf"), dpi=400)
    if png:
        fig.savefig(output / (stem + ".png"), dpi=180)
    plt.close(fig)

