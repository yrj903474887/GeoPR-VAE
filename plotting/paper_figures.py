"""Draw the quantitative paper panels from supplied CSV/NPZ results.

Every output is a separate PDF without a baked subfigure label. No experiment
values are bundled with the source distribution. Run with --help for inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D

from evaluation.controlled import bootstrap_mean_ci
from evaluation.regional import load_npz, read_csv
from .style import METHODS, LABELS, COLORS, LINE_STYLES, configure, frame, legend_below, new_axes, save, plt

NOISES = ["original", "independent_gaussian", "heavy_tails_t5", "hidden_E_scale_jump"]
NOISE_LABELS = ["Original", "Gaussian", "Heavy tails", "Scale jump"]
GROUPS = [("Cu10", "Cu–Pb–Zn–Ag"), ("W12", "W–Sn–Mo–Bi")]


def selected_values(table, method, buffer, metric, **filters):
    rows = table[(table.method == method) & (table.buffer_m == buffer)]
    for name, value in filters.items():
        rows = rows[rows[name] == value]
    rows = rows.sort_values("base_seed")
    if len(rows) != 10 or rows.base_seed.nunique() != 10:
        raise ValueError(f"Expected ten distinct realization means: {method}, {buffer}, {filters}")
    values = rows[metric].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite plotted observations")
    return values


def box(ax, values, position, color, width=.65, jitter=.16, size=23):
    ax.boxplot([values], positions=[position], widths=width, patch_artist=True, whis=1.5,
        showfliers=False, manage_ticks=False,
        boxprops=dict(facecolor=color, edgecolor="black", linewidth=1),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1), capprops=dict(color="black", linewidth=1), zorder=2)
    ax.scatter(position + np.linspace(-jitter, jitter, len(values)), values, s=size, marker="o",
        facecolor="white", edgecolor="black", linewidth=.75, zorder=3)


def recovery(base, output, png=False):
    for metric, name, label in [("denoising_rmse", "fig04_a", "Recovery RMSE"),
                               ("amplitude_gain_abs_error", "fig04_c", "Mean casewise\namplitude-gain error")]:
        fig, ax = new_axes(label)
        combined = []
        for bi, buffer in enumerate((4000, 8000)):
            for mi, method in enumerate(METHODS):
                values = selected_values(base, method, buffer, metric)
                box(ax, values, mi + bi * 8, COLORS[method])
                combined.extend(values)
        ax.set(xlim=(-.75, 13.75), xticks=[2.5, 10.5], xticklabels=["4 km", "8 km"])
        ax.set_xlabel("Spatial buffer", labelpad=9)
        span = max(max(combined) - min(combined), .01)
        ax.set_ylim((min(combined) - .12 * span, max(combined) + .13 * span) if metric == "denoising_rmse"
                    else (0, max(combined) * 1.13))
        save(fig, output, name, png)


def strata(table, output, scope, png=False):
    isnoise = scope == "noise"
    metric = "denoising_rmse" if isnoise else "amplitude_gain_abs_error"
    categories = NOISES if isnoise else [.25, .5]
    for buffer in (4000, 8000):
        fig, ax = new_axes("Recovery RMSE" if isnoise else "Mean casewise\namplitude-gain error", f"{buffer // 1000} km")
        tops = []
        for j, category in enumerate(categories):
            for mi, method in enumerate(METHODS):
                values = selected_values(table, method, buffer, metric, **{scope: category})
                mean = float(values.mean())
                lo, hi = bootstrap_mean_ci(values)
                ax.bar([j + (mi - 2.5) * .13], [mean], width=.118, bottom=0,
                    color=COLORS[method], edgecolor="black", linewidth=.65, zorder=3,
                    yerr=[[mean - lo], [hi - mean]],
                    error_kw=dict(ecolor="black", elinewidth=.9, capsize=2, capthick=.8))
                tops.append(hi)
        ax.set_xlim(-.55, len(categories) - .45)
        ax.set_xticks(range(len(categories)), NOISE_LABELS if isnoise else ["0.25", "0.50"])
        ax.set_xlabel("Noise condition" if isnoise else "Enrichment amplitude", labelpad=9)
        # The original pairwise axes are shared; expand rather than clip new results.
        upper = max(1.30 if isnoise else .18, max(tops) * 1.05)
        ax.set_ylim(0, upper)
        save(fig, output, f"fig05_{'noise' if isnoise else 'gain'}_{buffer // 1000}km", png)


def weak(table, output, png=False):
    for metric, name, label in [("delta_auc", "fig_weak_delta_auc", "AUROC"),
             ("delta_core_recall_top10", "fig_weak_delta_recall", "Core recall\n(top 10% of query points)")]:
        fig = plt.figure(figsize=(8, 5.4))
        ax = fig.add_axes([.155, .30, .825, .64])
        for bi, buffer in enumerate((4, 8)):
            for mi, method in enumerate(METHODS):
                rows = table[(table.method == method) & (table.buffer_km == buffer)].sort_values("fold")
                if rows.fold.tolist() != [1, 2, 3, 4, 5]:
                    raise ValueError("Expected exactly five weak-injection fold means per method/buffer")
                values = rows[metric].to_numpy(float)
                if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                    raise ValueError("AUROC and core recall must be finite and in [0,1]")
                box(ax, values, mi + bi * 8, COLORS[method], .62, .085, 22)
        ax.set(xlim=(-.75, 13.75), ylim=(-.035, 1.045), xticks=[2.5, 10.5], xticklabels=["4 km", "8 km"])
        ax.set_ylabel(label, labelpad=9)
        frame(ax)
        if metric == "delta_auc":
            ax.axhline(.5, color="#8b8b8b", linestyle=":", linewidth=.8, zorder=1)
        legend_below(fig, ax)
        save(fig, output, name, png)


def capture(curves, metrics, output, png=False):
    lookup = {(row["model"], row["site_group"]): row for row in metrics}
    for group, title in GROUPS:
        fig = plt.figure(figsize=(6, 5.5))  # established 12:11 panel ratio
        ax = fig.add_axes([.165, .245, .81, .66])
        area = np.asarray(curves["area_fraction"]) * 100
        ax.plot([0, 100], [0, 100], color="#b7b7b7", linestyle=":", linewidth=.9, zorder=1)
        for method in METHODS:
            curve = np.asarray(curves[method + "__" + group])
            if len(curve) != len(area) or curve[0] != 0 or curve[-1] != 1 or (np.diff(curve) < 0).any():
                raise ValueError("Invalid complete-area capture curve")
            row = lookup[method, group]
            n = int(row["n"])
            budget10 = int(.1 * (len(area) - 1))
            if int(round(curve[budget10] * n)) != int(row["N10"]):
                raise ValueError("Capture-curve and N10 table values disagree")
            events = np.unique(np.r_[0, np.flatnonzero(np.diff(curve)) + 1, len(curve) - 1])
            ax.step(area[events], curve[events] * 100, where="post", color=COLORS[method],
                linestyle=LINE_STYLES[method], linewidth=2.2 if method == "GeoPR-VAE" else 1.85,
                marker="s", markersize=4, markeredgewidth=.35, markeredgecolor="black")
        ax.set(xlim=(0, 100), ylim=(-1, 103), xticks=np.arange(0, 101, 20), yticks=np.arange(0, 101, 20))
        ax.set_xlabel("Selected area (%)", fontsize=14.25, labelpad=5)
        ax.set_ylabel("Cumulative occurrence\ncapture (%)", fontsize=14.25, labelpad=7)
        ax.set_title(title, fontsize=14.25, pad=9)
        frame(ax, .75, True)
        ax.tick_params(labelsize=12.75)
        handles = [Line2D([], [], color=COLORS[m], linestyle=LINE_STYLES[m], marker="s", markersize=5.5,
            markeredgewidth=.4, markeredgecolor="black", linewidth=2, label=LABELS[m]) for m in METHODS]
        legend_below(fig, ax, handles, bottom=.034, fontsize=12.75)
        save(fig, output, "fig7_" + group, png)


def distance(sites, metrics, output, png=False):
    lookup = {(row["model"], row["site_group"]): row for row in metrics}
    maximum = max(float(row["distance10_km"]) for row in sites if row["model"] in METHODS)
    for group, title in GROUPS:
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_axes([.15, .245, .82, .63])
        for i, method in enumerate(METHODS):
            rows = sorted([row for row in sites if row["model"] == method and row["site_group"] == group], key=lambda r: r["site_id"])
            values = np.array([float(row["distance10_km"]) for row in rows])
            mean = float(lookup[method, group]["mean_distance10_km"])
            if not len(values) or not np.isfinite(values).all() or not np.isclose(values.mean(), mean, atol=1e-12, rtol=0):
                raise ValueError("Per-occurrence distances and summary mean disagree")
            box(ax, values, i, COLORS[method], .46, .135, 32)
            ax.scatter([i], [mean], marker="D", s=46, facecolors="#303030", edgecolors="black", linewidths=.65, zorder=6)
            ax.annotate(f"{mean:.2f}", xy=(i, mean), xytext=(21, 0), textcoords="offset points", ha="left", va="center",
                fontsize=16, color="#303030", zorder=7, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        ax.set_xlim(-.52, 5.75)
        ax.set_ylim(-3, max(105, maximum * 1.05))
        ax.set_xticks(range(6), ["AE", "VAE", "Patch\nVAE", "GANomaly", "Skip-\nGANomaly", "GeoPR-\nVAE"], fontsize=16)
        ax.set_ylabel("Distance to selected\nanomaly area (km)", fontsize=18, labelpad=10)
        ax.tick_params(axis="y", labelsize=16)
        frame(ax)
        fig.text(.56, .975, title, ha="center", va="top", fontsize=19)
        save(fig, output, "fig8_" + group, png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("recovery", "noise", "amplitude", "weak", "capture", "distance"))
    parser.add_argument("--input", type=Path, required=True, help="CSV for recovery/noise/amplitude/weak/distance; NPZ for capture")
    parser.add_argument("--metrics", type=Path, help="Regional association summary CSV; required for capture/distance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    configure(args.font_dir, args.font_family)
    if args.kind in ("capture", "distance"):
        if args.metrics is None:
            parser.error("--metrics is required")
        metrics = read_csv(args.metrics)
        if args.kind == "capture":
            capture(load_npz(args.input), metrics, args.output, args.png)
        else:
            distance(read_csv(args.input), metrics, args.output, args.png)
    else:
        table = pd.read_csv(args.input)
        if args.kind == "recovery":
            recovery(table, args.output, args.png)
        elif args.kind == "weak":
            weak(table, args.output, args.png)
        else:
            strata(table, args.output, args.kind, args.png)


if __name__ == "__main__":
    main()
