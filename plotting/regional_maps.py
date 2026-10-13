"""Plot fixed-association fields, with display-only nearest-neighbour edge fill.

The supplied NPZ holds the effective evaluation grid. Missing display cells
inside the study boundary receive their nearest effective-cell value; this
never changes the scores, occurrence matching, or area used for evaluation.
Boundary input is a WGS84 GeoJSON Polygon/MultiPolygon or FeatureCollection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, FuncFormatter
from pyproj import Transformer, Geod
import shapely
from shapely.geometry import shape, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, transform

from evaluation.regional import ASSOCIATIONS, MATCH_TARGETS, load_npz, read_csv, normalize_sites, validate_geometry
from .style import METHODS, LABELS, configure, save, plt

DISPLAY_GROUPS = {"Bi-Mo-W": "Bi-Mo-W", "Mo": "Bi-Mo-W", "Cu": "Cu-Pb-Zn", "Cu-Pb-Zn": "Cu-Pb-Zn",
                  "Pb-Sn-Zn": "Pb-Sn-Zn", "Pb-Zn": "Pb-Sn-Zn", "Zn": "Pb-Sn-Zn",
                  "Sn": "Sn-W", "Sn-W": "Sn-W", "W": "Sn-W"}
SYMBOLS = {"Bi-Mo-W": ("^", "#27AFC2"), "Cu-Pb-Zn": ("s", "#D86A9A"),
           "Pb-Sn-Zn": ("h", "#55A868"), "Sn-W": ("d", "#D94B4B")}


def load_boundary(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if obj["type"] == "FeatureCollection":
        geometry = unary_union([shape(f["geometry"]) for f in obj["features"]])
    elif obj["type"] == "Feature":
        geometry = shape(obj["geometry"])
    else:
        geometry = shape(obj)
    if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty or not geometry.is_valid:
        raise ValueError("A valid polygonal study boundary is required")
    xmin, ymin, xmax, ymax = geometry.bounds
    if not (-180 <= xmin <= xmax <= 180 and -90 <= ymin <= ymax <= 90):
        raise ValueError("Boundary GeoJSON must be in longitude/latitude WGS84")
    return geometry


def boundary_path(geometry):
    vertices, codes = [], []
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        polygon = orient(polygon, sign=1.0)
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = np.asarray(ring.coords)
            vertices.extend(coords)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(vertices, codes)


def display_geometry(grid, boundary, crs, cell_size):
    """Build display geometry independently of the evaluation field."""
    _, xy = validate_geometry(grid)
    to_projected = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_geographic = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    projected_boundary = transform(to_projected.transform, boundary)
    xmin, ymin, xmax, ymax = projected_boundary.bounds
    origin = xy[0]
    lower = np.floor((np.array([xmin, ymin]) - origin) / cell_size).astype(int)
    upper = np.ceil((np.array([xmax, ymax]) - origin) / cell_size).astype(int)
    xs = origin[0] + np.arange(lower[0], upper[0] + 1) * cell_size
    ys = origin[1] + np.arange(lower[1], upper[1] + 1) * cell_size
    gx, gy = np.meshgrid(xs, ys)
    display_xy = np.column_stack((gx.ravel(), gy.ravel()))
    boxes = shapely.box(display_xy[:, 0] - cell_size / 2, display_xy[:, 1] - cell_size / 2,
                        display_xy[:, 0] + cell_size / 2, display_xy[:, 1] + cell_size / 2)
    candidate = np.flatnonzero(shapely.intersects(projected_boundary, boxes))
    inside = np.zeros(len(display_xy), dtype=bool)
    inside[candidate] = shapely.area(shapely.intersection(boxes[candidate], projected_boundary)) > 1e-6
    nearest = np.zeros(len(display_xy), dtype=int)
    cols = np.rint((xy[:, 0] - xs[0]) / cell_size).astype(int)
    rows = np.rint((xy[:, 1] - ys[0]) / cell_size).astype(int)
    if not np.allclose(xy, np.column_stack((xs[cols], ys[rows])), atol=1e-6, rtol=0):
        raise ValueError("Effective grid is not on the declared square lattice")
    original_indices = rows * len(xs) + cols
    original_mask = np.zeros(len(display_xy), dtype=bool)
    original_mask[original_indices] = True
    fill = np.flatnonzero(inside & ~original_mask)
    if len(fill):
        tree = shapely.STRtree(shapely.points(xy))
        pairs = tree.query_nearest(shapely.points(display_xy[fill]), all_matches=True)
        order = np.lexsort((np.asarray(grid["grid_id"])[pairs[1]], pairs[0]))
        first = np.r_[True, np.diff(pairs[0, order]) != 0]
        chosen = order[first]
        np.testing.assert_array_equal(pairs[0, chosen], np.arange(len(fill)))
        nearest[fill] = pairs[1, chosen]
    nearest[original_indices] = np.arange(len(xy))
    # Boundary clipping handles partial cells. Keep original cells even when
    # their centre falls just outside the geographic polygon due to precision.
    inside[original_indices] = True
    ex, ey = np.meshgrid(np.r_[xs - cell_size / 2, xs[-1] + cell_size / 2],
                         np.r_[ys - cell_size / 2, ys[-1] + cell_size / 2])
    lon, lat = to_geographic.transform(ex, ey)
    return dict(shape=(len(ys), len(xs)), nearest=nearest, inside=inside,
                original_indices=original_indices, lon=lon, lat=lat, boundary=boundary)


def coordinate_label(value, position=None, hemisphere="E"):
    absolute = abs(value)
    degree = int(absolute)
    minute = int(round((absolute - degree) * 60))
    if minute == 60:
        degree, minute = degree + 1, 0
    return f"{degree}°{minute}′{hemisphere}"


def site_display_group(row):
    key = row.get("symbol_group", "")
    if key in DISPLAY_GROUPS:
        return DISPLAY_GROUPS[key]
    targets = set(row["primary_target_elements"].split(";"))
    # Display grouping only; evaluate_methods uses exact recorded target tokens.
    for members, name in [({"Bi", "Mo"}, "Bi-Mo-W"), ({"Cu"}, "Cu-Pb-Zn"),
                          ({"Pb", "Zn"}, "Pb-Sn-Zn"), ({"Sn", "W"}, "Sn-W")]:
        if targets & members:
            return name
    raise ValueError("No publication symbol for matching occurrence: " + row["site_id"])


def draw(ax, values, context, records, crs, limits=None):
    field = np.asarray(values, float)
    raster = field[context["nearest"]].copy()
    raster[~context["inside"]] = np.nan
    np.testing.assert_array_equal(raster[context["original_indices"]], field)
    mesh = ax.pcolormesh(context["lon"], context["lat"], np.ma.masked_invalid(raster.reshape(context["shape"])),
                         cmap="RdYlBu_r", vmin=0, vmax=1, shading="flat", rasterized=True, zorder=1)
    mesh.set_clip_path(PathPatch(boundary_path(context["boundary"]), transform=ax.transData))
    polygons = list(context["boundary"].geoms) if context["boundary"].geom_type == "MultiPolygon" else [context["boundary"]]
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            coords = np.asarray(ring.coords)
            ax.plot(coords[:, 0], coords[:, 1], color="black", linewidth=.4, zorder=2)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    present = set()
    for group, (marker, color) in SYMBOLS.items():
        selected = [r for r in records if site_display_group(r) == group]
        if selected:
            longitude, latitude = transformer.transform([r["x"] for r in selected], [r["y"] for r in selected])
            ax.scatter(longitude, latitude, s=29, marker=marker, facecolors=color,
                       edgecolors="black", linewidths=.6, zorder=4)
            present.add(group)
    if limits is None:
        xl, yb, xr, yt = context["boundary"].bounds
        dx, dy = xr - xl, yt - yb
        limits = xl - .04 * dx, xr + .04 * dx, yb - .04 * dy, yt + .04 * dy
    xl, xr, yb, yt = limits
    ax.set(xlim=(xl, xr), ylim=(yb, yt))
    ax.set_aspect(1 / np.cos(np.deg2rad((yb + yt) / 2)))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(coordinate_label))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: coordinate_label(v, p, "N")))
    ax.tick_params(direction="in", length=3, width=.6, pad=2.3, labelsize=10.2,
                   top=True, right=True, labeltop=True, labelright=True)
    for tick in ax.yaxis.get_major_ticks():
        for label in (tick.label1, tick.label2):
            label.set_rotation(90)
            label.set_va("center")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(.6)
    ax.grid(color="#89949a", linewidth=.28, alpha=.3, linestyle=":")
    ax.text(.94, .933, "N", ha="center", va="center", transform=ax.transAxes, fontsize=10.5, zorder=7)
    ax.annotate("", xy=(.94, .90), xytext=(.94, .78), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", mutation_scale=11, linewidth=.8, color="black"), zorder=7)
    dx, dy = xr - xl, yt - yb
    latitude, end = yb + .07 * dy, xr - .10 * dx
    geod = Geod(ellps="WGS84")
    start, _, _ = geod.fwd(end, latitude, 270, 20000)
    middle, _, _ = geod.fwd(end, latitude, 270, 10000)
    if start < xl:
        raise ValueError("Map extent is too narrow for its 20 km scale bar")
    height = .009 * dy
    ax.add_patch(Rectangle((start - .030 * dx, latitude - .022 * dy), end - start + .068 * dx,
                          .13 * dy, facecolor="white", edgecolor="none", alpha=.97, zorder=5))
    for left, width, color in [(start, middle - start, "black"), (middle, end - middle, "white")]:
        ax.add_patch(Rectangle((left, latitude), width, height, facecolor=color,
                              edgecolor="black", linewidth=.6, zorder=6))
    ax.text((start + end) / 2, latitude + height + .015 * dy, "20 km", ha="center", va="bottom", fontsize=10.2, zorder=7)
    return present


def legend(output, key, present, png=False):
    fig = plt.figure(figsize=(18 / 2.54, 2.5 / 2.54))
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="RdYlBu_r"),
        cax=fig.add_axes([.265, .772, .47, .115]), orientation="horizontal", ticks=np.linspace(0, 1, 6))
    bar.ax.tick_params(labelsize=9.3, length=2.5, pad=1.5, width=.5)
    bar.set_label("Mean element-response percentile", fontsize=10, labelpad=2)
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=5.7,
        markerfacecolor=color, markeredgecolor="black", markeredgewidth=.6, label=group.replace("-", "–"))
        for group, (marker, color) in SYMBOLS.items() if group in present]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .02), ncol=max(1, len(handles)),
               frameon=False, fontsize=10, handletextpad=.4, handlelength=1, borderpad=0, columnspacing=1.6)
    save(fig, output, f"fig6_{key}_legend", png)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--occurrences", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--crs", default="EPSG:32649")
    parser.add_argument("--cell-size", type=float, default=500)
    parser.add_argument("--extent", type=float, nargs=4, metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()
    configure(args.font_dir, args.font_family)
    grid, fields = load_npz(args.grid), load_npz(args.fields)
    if not np.array_equal(grid["grid_id"], fields["grid_id"]):
        parser.error("Field/geometry grid IDs differ")
    sites = normalize_sites(read_csv(args.occurrences), grid, args.cell_size)
    context = display_geometry(grid, load_boundary(args.boundary), args.crs, args.cell_size)
    for column, (key, targets) in enumerate(zip(("CuPbZnAg", "WSnMoBi"), MATCH_TARGETS)):
        records = [r for r in sites if set(r["primary_target_elements"].split(";")) & set(targets)]
        present = set()
        for method in METHODS:
            values = fields[method][:, column]
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                parser.error("Association percentiles must be finite within [0,1]")
            fig = plt.figure(figsize=(3.3, 2.6))
            ax = fig.add_axes([.093, .090, .814, .745])
            present |= draw(ax, values, context, records, args.crs, args.extent)
            fig.text(.5, .980, LABELS[method], ha="center", va="top", fontsize=12.5, fontweight="bold")
            save(fig, args.output, f"fig6_{key}_{method.replace('-', '_')}", args.png)
        legend(args.output, key, present, args.png)


if __name__ == "__main__":
    main()
