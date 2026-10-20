"""Render geological units, faults, grouped occurrences and shaded relief.

The geology/fault GeoJSON coordinates must already share --map-crs. Polygon
properties need a class_id from PALETTE (or a custom --legend JSON). There
are no geological-contact outlines. The boundary and province GeoJSON are
WGS84. An explicit --boundary-pipeline can fix the WGS84 -> native map datum
operation when an authoritative transformation is available.
"""
from __future__ import annotations
import argparse
import json
import math
from io import BytesIO
from pathlib import Path
import numpy as np
from matplotlib.patches import PathPatch, Patch, Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FuncFormatter
from pyproj import Transformer, CRS, Geod
from pyproj.enums import TransformDirection
from shapely.geometry import shape
from shapely.ops import transform
from evaluation.regional import read_csv
from .regional_maps import boundary_path, load_boundary, coordinate_label, SYMBOLS, DISPLAY_GROUPS, site_display_group
from .style import configure, save, plt
from .pdf_assembly import compose, one_page

PALETTE = {
    "q": ("Quaternary", "#fff2a8"), "paleogene_neogene": ("Paleogene–Neogene", "#efc46e"),
    "k": ("Cretaceous", "#a4d583"), "j": ("Jurassic", "#84ccd7"), "t": ("Triassic", "#ad94ca"),
    "p": ("Permian", "#cd967d"), "c": ("Carboniferous", "#a8beb5"),
    "dc": ("Devonian-Carboniferous", "#c3b791"), "d": ("Devonian", "#d0af7d"),
    "s": ("Silurian", "#a4d2a0"), "o": ("Ordovician", "#78b79d"), "cambrian": ("Cambrian", "#a6ad76"),
    "sinian": ("Sinian", "#dbb1a7"), "proterozoic": ("Proterozoic", "#cbb7d1"),
    "granitoids": ("Granitoids", "#f0a1a1"), "other_intrusive": ("Other intrusive rocks", "#bd7397"),
    "volcanic": ("Volcanic rocks", "#caa66f"), "dyke_rocks": ("Dykes and veins", "#918576"),
    "migmatitic": ("Migmatitic rocks", "#d28daa"), "metamorphic_undiff": ("Metamorphic rocks", "#b1a5b6"),
    "unclassified": ("Unclassified unit", "#e0e0e0"), "unmapped": ("No mapped unit", "#ffffff")}


def features(path):
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if document.get("type") != "FeatureCollection":
        raise ValueError("Geology and faults require GeoJSON FeatureCollections")
    return document["features"]


def line_parts(geometry):
    if geometry.geom_type in ("LineString", "LinearRing"):
        yield np.asarray(geometry.coords)
    elif hasattr(geometry, "geoms"):
        for member in geometry.geoms:
            yield from line_parts(member)


def shade(ax, clip, bounds, hillshade, crs, datum, strength=.62, width=1600):
    """Bilinear hillshade lookup; alpha darkens geology without hiding its hue."""
    import rasterio
    xmin, ymin, xmax, ymax = bounds
    height = max(2, int(width * (ymax - ymin) / (xmax - xmin)))
    xs = xmin + (np.arange(width) + .5) * (xmax - xmin) / width
    ys = ymax - (np.arange(height) + .5) * (ymax - ymin) / height
    result = np.full((height, width), np.nan, np.float32)
    with rasterio.open(hillshade) as src:
        if src.crs is None:
            raise ValueError("Hillshade raster must have its actual CRS")
        data, valid, inverse = src.read(1), src.read_masks(1) > 0, ~src.transform
        direct = Transformer.from_crs(crs, src.crs, always_xy=True)
        via_wgs = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        for start in range(0, height, 192):
            stop = min(start + 192, height)
            x, y = np.meshgrid(xs, ys[start:stop])
            if datum is not None:
                lon, lat = datum.transform(x, y, direction=TransformDirection.INVERSE)
                x, y = via_wgs.transform(lon, lat)
            else:
                x, y = direct.transform(x, y)
            col, row = inverse * (x, y)
            col, row = col - .5, row - .5
            ci, ri = np.floor(col).astype(int), np.floor(row).astype(int)
            inside = (ci >= 0) & (ri >= 0) & (ci + 1 < src.width) & (ri + 1 < src.height)
            c, r = np.clip(ci, 0, src.width - 2), np.clip(ri, 0, src.height - 2)
            dx, dy = col - ci, row - ri
            usable = inside & valid[r, c] & valid[r + 1, c] & valid[r, c + 1] & valid[r + 1, c + 1]
            value = ((1 - dy) * (1 - dx) * data[r, c] + dy * (1 - dx) * data[r + 1, c]
                     + (1 - dy) * dx * data[r, c + 1] + dy * dx * data[r + 1, c + 1])
            result[start:stop] = np.where(usable, value, np.nan)
    rgba = np.zeros((height, width, 4), np.float32)
    rgba[:, :, 3] = np.where(np.isfinite(result), np.clip((180 - result) / 180 * strength, 0, .58), 0)
    artist = ax.imshow(rgba, extent=(xmin, xmax, ymin, ymax), origin="upper", interpolation="bilinear", zorder=3.2, aspect="auto")
    artist.set_clip_path(clip)


def north_arrow(ax):
    x, top, bottom, mid, width = .935, .91, .775, .805, .015
    ax.add_patch(Polygon([[x, top], [x - width, bottom], [x, mid]], transform=ax.transAxes,
                        facecolor="black", edgecolor="black", linewidth=.6, zorder=12))
    ax.add_patch(Polygon([[x, top], [x + width, bottom], [x, mid]], transform=ax.transAxes,
                        facecolor="white", edgecolor="black", linewidth=.6, zorder=12))
    ax.text(x, .922, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, zorder=12)


def locator(ax, province, study):
    projection = Transformer.from_crs(4326, 32649, always_xy=True)
    province, study = transform(projection.transform, province), transform(projection.transform, study)
    ax.add_patch(PathPatch(boundary_path(province), facecolor="#c5c7c9", edgecolor="#6a6a6a", linewidth=.4))
    ax.add_patch(PathPatch(boundary_path(study), facecolor="#d4252a", edgecolor="#bb1720", linewidth=.4))
    xmin, ymin, xmax, ymax = province.bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set(xlim=(xmin - .065 * dx, xmax + .065 * dx),
           ylim=(ymin - .18 * dy, ymax + .055 * dy), xticks=[], yticks=[])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(.7)
    ax.figure.canvas.draw()
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    ax.text(.5, .017, "Guangdong Province", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8)
    # Keep a geodesic 200 km scale while using the inset's projected coordinates.
    inverse = Transformer.from_crs(32649, 4326, always_xy=True)
    start_x, start_y = left + .44 * (right - left), bottom + .14 * (top - bottom)
    lon, lat = inverse.transform(start_x, start_y)
    geod = Geod(ellps="WGS84")
    points = [projection.transform(*geod.fwd(lon, lat, 90, distance)[:2])
              for distance in (0, 100000, 200000)]
    for i, color in enumerate(("#202020", "#9c9c9c")):
        ax.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]],
                color=color, linewidth=1.5, zorder=9)
    for x, y in points:
        ax.plot([x, x], [y - .008 * (top - bottom), y + .008 * (top - bottom)],
                color="#202020", linewidth=.5, zorder=9)
    for index, label in ((0, "0"), (2, "200 km")):
        x, y = points[index]
        ax.text(x, y + .021 * (top - bottom), label, fontsize=5.8, ha="center", va="bottom")


def aligned_legends(fig, ax, palette):
    # Keep six rows and four columns so the mineral legend shares their origins.
    # Text advances are measured at the publication export resolution; using
    # the interactive 100 dpi renderer shifts the later columns fractionally.
    fig.set_dpi(600)
    items = [Patch(facecolor=color, edgecolor="black", linewidth=.4, label=label) for label, color in palette.values()]
    items += [Line2D([], [], color="#d4252a", linewidth=.8, label="Fault"),
              Line2D([], [], color="black", linewidth=.6, label="Study-area boundary")]
    ncols = 4
    nrows = int(np.ceil(len(items) / ncols))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    position = ax.get_position()
    legend_ax = fig.add_axes([position.x0, .016, position.width, .187]); legend_ax.axis("off")
    fp = plt.matplotlib.font_manager.FontProperties(family=plt.rcParams["font.family"], size=8.5)
    width_px = legend_ax.get_window_extent(renderer).width
    height_px = legend_ax.get_window_extent(renderer).height
    final_labels = [item.get_label() for item in items[(ncols - 1) * nrows:]]
    if not final_labels:
        final_labels = [item.get_label() for item in items]
    last_width = max(renderer.get_text_width_height_descent(label, fp, False)[0] for label in final_labels)
    handle_px, gap_px = 14.025 * fig.dpi / 72, 5.525 * fig.dpi / 72
    handle_height = .95 * 8.5 * fig.dpi / 72
    starts = np.linspace(0, (width_px - handle_px - gap_px - last_width) / width_px, ncols)
    for j, item in enumerate(items):
        column, row = divmod(j, nrows)
        x, y = starts[column], .92 - row * .76 / max(1, nrows - 1)
        if isinstance(item, Patch):
            legend_ax.add_patch(Rectangle((x, y - handle_height / height_px / 2),
                handle_px / width_px, handle_height / height_px,
                transform=legend_ax.transAxes, facecolor=item.get_facecolor(), edgecolor="black", linewidth=.4))
        else:
            legend_ax.plot([x, x + handle_px / width_px], [y, y], transform=legend_ax.transAxes,
                           color=item.get_color(), linewidth=item.get_linewidth())
        right_anchor = column == ncols - 1 and item.get_label() == max(final_labels,
            key=lambda label: renderer.get_text_width_height_descent(label, fp, False)[0])
        legend_ax.text(1 if right_anchor else x + (handle_px + gap_px) / width_px,
                       y, item.get_label(), transform=legend_ax.transAxes,
                       ha="right" if right_anchor else "left", va="center", fontsize=8.5)
    mineral = plt.figure(figsize=(7.6, .23), dpi=600)
    mineral_ax = mineral.add_axes([position.x0, 0, position.width, 1]); mineral_ax.axis("off")
    for column, (group, (marker, color)) in enumerate(SYMBOLS.items()):
        x = starts[column]
        mineral_ax.plot([x + (5.3 * fig.dpi / 72) / 2 / width_px], [.55],
                        linestyle="", marker=marker, markersize=5.3, markeredgewidth=.65,
                        markerfacecolor=color, markeredgecolor="black", transform=mineral_ax.transAxes)
        mineral_ax.text(x + (handle_px + gap_px) / width_px, .55, group.replace("-", "–"),
                        transform=mineral_ax.transAxes, fontsize=8.5, ha="left", va="center")
    return mineral


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--faults", type=Path, required=True)
    parser.add_argument("--occurrences", type=Path, required=True,
        help="CSV source_x,source_y,target_elements (or x,y,primary_target_elements); coordinates in map-crs")
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--province", type=Path, help="Optional WGS84 Guangdong boundary")
    parser.add_argument("--hillshade", type=Path, required=True, help="Georeferenced single-band 0–255 hillshade")
    parser.add_argument("--map-crs", default="EPSG:4214")
    parser.add_argument("--boundary-pipeline", help="PROJ pipeline text file explicitly mapping WGS84 to map-crs")
    parser.add_argument("--legend", type=Path, help="JSON object: class_id: [English label, color]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--components", action="store_true",
        help="Also retain separate original-size map, locator and occurrence-legend PDFs")
    args = parser.parse_args()
    configure(args.font_dir, args.font_family)
    crs = CRS.from_user_input(args.map_crs)
    if not crs.is_geographic:
        parser.error("This geographic cartographic layout requires a geographic map-crs")
    boundary_wgs = load_boundary(args.boundary)
    datum = Transformer.from_pipeline(Path(args.boundary_pipeline).read_text().strip()) if args.boundary_pipeline else None
    transformer = datum or Transformer.from_crs(4326, crs, always_xy=True)
    boundary = transform(transformer.transform, boundary_wgs)
    palette = json.loads(args.legend.read_text(encoding="utf-8")) if args.legend else PALETTE
    fig = plt.figure(figsize=(7.6, 6.6))
    ax = fig.add_axes([.074, .252, .852, .700])
    xmin, ymin, xmax, ymax = boundary.bounds
    # Round the map display frame, not the geological geometry or sample coordinates.
    bounds = (math.floor((xmin - .06) * 10) / 10, math.floor((ymin - .04) * 10) / 10,
              math.ceil((xmax + .06) * 10) / 10, math.ceil((ymax + .04) * 10) / 10)
    clip = PathPatch(boundary_path(boundary), transform=ax.transData)
    for feature in features(args.geology):
        geom, category = shape(feature["geometry"]), feature["properties"]["class_id"]
        if geom.geom_type not in ("Polygon", "MultiPolygon") or not geom.is_valid:
            raise ValueError("Invalid source polygon; validate the original geological geometry before plotting")
        if category not in palette:
            raise ValueError("Unmapped geological class: " + category)
        artist = PathPatch(boundary_path(geom), facecolor=palette[category][1], edgecolor="none", linewidth=0, zorder=2)
        artist.set_clip_path(clip); ax.add_patch(artist)
    shade(ax, clip, bounds, args.hillshade, crs, datum)
    for feature in features(args.faults):
        for coordinates in line_parts(shape(feature["geometry"])):
            line, = ax.plot(coordinates[:, 0], coordinates[:, 1], color="#d4252a", linewidth=.55, zorder=5)
            line.set_clip_path(clip)
    ax.add_patch(PathPatch(boundary_path(boundary), fill=False, edgecolor="black", linewidth=.6, zorder=6))
    eligible = {"Cu", "Pb", "Zn", "Ag", "W", "Sn", "Mo"}
    for record in read_csv(args.occurrences):
        targets = record.get("target_elements", record.get("primary_target_elements", ""))
        if not set(targets.split(";")) & eligible:
            continue
        record["primary_target_elements"] = targets
        marker, color = SYMBOLS[site_display_group(record)]
        ax.scatter([float(record.get("source_x", record.get("x")))], [float(record.get("source_y", record.get("y")))],
                   s=29, marker=marker, facecolor=color, edgecolor="black", linewidth=.65, zorder=10)
    ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]))
    ax.set_aspect(1 / np.cos(np.deg2rad((bounds[1] + bounds[3]) / 2)))
    ax.xaxis.set_major_locator(MultipleLocator(.5)); ax.yaxis.set_major_locator(MultipleLocator(.25))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: "" if abs(v - 112.5) < 1e-8 else coordinate_label(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: coordinate_label(v, p, "N")))
    ax.tick_params(labelsize=8.5, top=True, right=True, labeltop=True, labelright=True, direction="out", length=3, pad=4)
    for tick in ax.yaxis.get_major_ticks():
        for label in (tick.label1, tick.label2):
            label.set_rotation(90); label.set_verticalalignment("center")
    for spine in ax.spines.values():
        spine.set_color("black"); spine.set_linewidth(.7)
    north_arrow(ax)
    geod = crs.get_geod()
    start, sy = bounds[0] + .09, bounds[1] + .08
    points = [geod.fwd(start, sy, 90, d * 1000)[0] for d in (0, 10, 20, 30)]
    for i in range(3):
        ax.add_patch(Rectangle((points[i], sy), points[i + 1] - points[i], .012, facecolor="black" if i % 2 == 0 else "white", edgecolor="black", linewidth=.5, zorder=10))
    for x, distance in zip(points, (0, 10, 20, 30)):
        ax.text(x, sy - .018, str(distance), fontsize=7.5, ha="center", va="top")
    ax.text(points[-1] + .065, sy - .018, "km", fontsize=7.5, va="top")
    inset = None
    if args.province:
        province = load_boundary(args.province)
        inset = ax.inset_axes([.805, 0, .195, .22], zorder=20)
        locator(inset, province, boundary_wgs)
    mineral = aligned_legends(fig, ax, palette)
    args.output.mkdir(parents=True, exist_ok=True)
    main_stream, mineral_stream = BytesIO(), BytesIO()
    fig.savefig(main_stream, format="pdf")
    mineral.savefig(mineral_stream, format="pdf")
    main_stream.seek(0)
    mineral_stream.seek(0)
    # The complete page remains vector-based apart from the actual hillshade.
    compose(547.2, 475.2, [(one_page(main_stream), 1, 0, 0),
                          (one_page(mineral_stream), 1, 0, 0)], args.output / "Fig02.pdf")
    if args.components:
        mineral.savefig(args.output / "fig02_occurrence_legend.pdf")
        if inset is not None:
            fig.canvas.draw()
            bbox = inset.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(args.output / "fig02_a.pdf", bbox_inches=bbox, pad_inches=0)
            inset.set_visible(False)
        fig.savefig(args.output / "fig02_b.pdf")
    plt.close(mineral)
    plt.close(fig)
    if args.png:
        from .map_legends import png_preview
        png_preview(args.output / "Fig02.pdf")


if __name__ == "__main__":
    main()
