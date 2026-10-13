"""Occurrence-based evaluation on a common equal-area square grid.

Run ``python -m evaluation.regional --help`` for the file interface.
Point-background AUC compares matching occurrences to unlabelled grid cells;
the background excludes *all* supplied effective occurrences. It does not
assume that the background is a surveyed absence inventory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ELEMENTS = ("Au", "Ag", "As", "Bi", "Cu", "F", "Mo", "Pb", "Sb", "Sn", "W", "Zn")
ASSOCIATIONS = ("Cu-Pb-Zn-Ag", "W-Sn-Mo-Bi")
MEMBERS = (("Cu", "Pb", "Zn", "Ag"), ("W", "Sn", "Mo", "Bi"))
# Association membership and occurrence matching have separate meanings.
# These targets preserve the recorded-primary-commodity matching protocol.
MATCH_TARGETS = (("Cu", "Pb", "Zn", "Ag"), ("W", "Sn", "Mo"))
GROUP_KEYS = ("Cu10", "W12")  # identifiers only: counts are computed from inputs


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty evaluation table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def midrank(values):
    """Empirical midpoint percentile, including an exact half contribution for ties."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("A finite, nonempty one-dimensional field is required")
    ordered = np.sort(values)
    return (np.searchsorted(ordered, values, "left") +
            np.searchsorted(ordered, values, "right")) / (2 * len(values))


def fixed_association_fields(element_responses, elements=ELEMENTS):
    """Average grid-wide member percentiles; no occurrence labels are read."""
    response = np.asarray(element_responses, float)
    elements = list(elements)
    if response.ndim != 2 or response.shape[1] != len(elements):
        raise ValueError("element_responses must be [grid cells, elements]")
    if len(set(elements)) != len(elements) or not set(sum(MEMBERS, ())) <= set(elements):
        raise ValueError("Duplicate elements or missing fixed-association members")
    percentiles = np.column_stack([midrank(column) for column in response.T])
    return np.column_stack([percentiles[:, [elements.index(e) for e in group]].mean(1)
                            for group in MEMBERS])


def ranks(values, grid_ids):
    """Descending score, ascending grid ID as deterministic tie-breaker."""
    values, grid_ids = np.asarray(values, float), np.asarray(grid_ids)
    rank = np.empty(len(values), dtype=np.int64)
    rank[np.lexsort((grid_ids, -values))] = np.arange(1, len(values) + 1)
    return rank


def square_distance_km(points, selected_centres, cell_size=500.0):
    """Exact Euclidean distance to the union of selected square cells, in km.

    Coordinates must be in the same projected, metre-based CRS. A distance to
    the nearest cell centre would be a different metric and is not used here.
    """
    points, selected_centres = np.asarray(points, float), np.asarray(selected_centres, float)
    if selected_centres.ndim != 2 or selected_centres.shape[1] != 2 or not len(selected_centres):
        raise ValueError("At least one selected cell is needed")
    return np.asarray([np.sqrt(np.square(np.maximum(
        np.abs(selected_centres - point) - cell_size / 2, 0)).sum(1).min()) / 1000
        for point in points])


def validate_geometry(grid):
    ids, xy = np.asarray(grid["grid_id"]), np.asarray(grid["grid_xy"], float)
    if ids.ndim != 1 or len(ids) < 20 or len(np.unique(ids)) != len(ids):
        raise ValueError("At least 20 unique grid IDs are required")
    if xy.shape != (len(ids), 2) or not np.isfinite(xy).all():
        raise ValueError("grid_xy must contain finite projected metre coordinates [N,2]")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Use integer grid IDs to preserve numeric tie ordering")
    return ids, xy


def normalize_sites(records, grid, cell_size=500.0):
    """Validate occurrence-to-cell assignments instead of silently snapping labels."""
    ids, xy = validate_geometry(grid)
    lookup = {int(value): i for i, value in enumerate(ids)}
    sites = []
    for source in records:
        if "inside_effective_grid" in source and str(source["inside_effective_grid"]).lower() not in ("true", "1"):
            continue
        row = dict(source)
        row["site_id"] = str(row["site_id"])
        if "grid_id" in row and str(row["grid_id"]).strip():
            index = lookup[int(row["grid_id"])]
        elif "grid_idx" in row:
            index = int(row["grid_idx"])
        else:
            raise ValueError("Each occurrence needs an explicit grid_id")
        point = np.array([float(row.get("x", row.get("x_utm49n"))),
                          float(row.get("y", row.get("y_utm49n")))])
        if not 0 <= index < len(ids) or not np.isfinite(point).all():
            raise ValueError("Invalid occurrence coordinate or grid index")
        if (np.abs(point - xy[index]) > cell_size / 2 + 1e-6).any():
            raise ValueError("Occurrence does not lie in its assigned square cell: " + row["site_id"])
        row.update(grid_idx=index, grid_id=int(ids[index]), x=float(point[0]), y=float(point[1]))
        sites.append(row)
    if not sites or len({r["site_id"] for r in sites}) != len(sites):
        raise ValueError("Occurrence IDs must be unique and nonempty")
    if len({r["grid_idx"] for r in sites}) != len(sites):
        raise ValueError("The evaluation protocol requires one occurrence per occupied grid cell")
    return sites


def evaluate_field(values, grid, sites, matching_indices, cell_size=500.0):
    """Evaluate one fixed association without altering its score field."""
    ids, xy = validate_geometry(grid)
    values = np.asarray(values, float)
    if values.shape != ids.shape or not np.isfinite(values).all():
        raise ValueError("Every effective grid cell requires a finite score")
    positions = np.asarray([r["grid_idx"] for r in sites], dtype=int)
    selected = np.asarray(matching_indices, int)
    if not len(selected):
        raise ValueError("No matching occurrences for this association")
    background = np.ones(len(ids), dtype=bool)
    background[positions] = False
    if not background.any():
        raise ValueError("The background must contain unlabelled cells")
    bg_values = np.sort(values[background])
    site_values = values[positions]
    auc = (np.searchsorted(bg_values, site_values, "left") +
           np.searchsorted(bg_values, site_values, "right")) / (2 * background.sum())
    rank = ranks(values, ids)
    site_rank = rank[positions[selected]]
    matching_values = site_values[selected]
    budget10 = int(.1 * len(values))
    point_xy = np.asarray([[r["x"], r["y"]] for r in sites])
    distances = square_distance_km(point_xy[selected], xy[rank <= budget10], cell_size)
    n = len(selected)
    row = dict(n=n, PB_AUC=float(auc[selected].mean()),
               mean_distance10_km=float(distances.mean()))
    count = np.bincount(site_rank, minlength=len(values) + 1).cumsum()
    curve = count / n
    for pct in (5, 10, 20):
        budget = int(len(values) * pct / 100)
        cutoff = np.partition(values, len(values) - budget)[len(values) - budget]
        above, tied = int((values > cutoff).sum()), int((values == cutoff).sum())
        hit_above = int((matching_values > cutoff).sum())
        hit_tied = int((matching_values == cutoff).sum())
        slots = budget - above
        hit = int((site_rank <= budget).sum())
        row.update({f"N{pct}": hit, f"capture{pct}": hit / n,
                    f"EF{pct}": hit / n / (budget / len(values)),
                    f"N{pct}_min": hit_above + max(0, slots - tied + hit_tied),
                    f"N{pct}_max": hit_above + min(slots, hit_tied),
                    f"actual_area_fraction_{pct}": budget / len(values)})
    for pct in (50, 80):
        needed = math.ceil(n * pct / 100)
        cutoff = np.sort(matching_values)[-needed]
        above, tied = int((values > cutoff).sum()), int((values == cutoff).sum())
        remaining = needed - int((matching_values > cutoff).sum())
        tied_hits = int((matching_values == cutoff).sum())
        row.update({f"A{pct}": float(np.sort(site_rank)[needed - 1] / len(values)),
                    f"A{pct}_min": (above + remaining) / len(values),
                    f"A{pct}_max": (above + tied - tied_hits + remaining) / len(values)})
    row["density_top10_per_1000km2"] = row["N10"] / (budget10 * (cell_size / 1000) ** 2) * 1000
    per_site = []
    for local, i in enumerate(selected):
        per_site.append(dict(site_id=sites[i]["site_id"],
            primary_target_elements=sites[i]["primary_target_elements"],
            grid_idx=int(positions[i]), grid_id=int(ids[positions[i]]),
            score=float(site_values[i]), rank=int(rank[positions[i]]),
            rank_area_fraction=float(rank[positions[i]] / len(values)),
            PB_AUC_contribution=float(auc[i]), distance10_km=float(distances[local]),
            **{f"captured{pct}": int(rank[positions[i]] <= int(len(values) * pct / 100))
               for pct in (5, 10, 20)}))
    return row, per_site, curve


def evaluate_methods(fields, grid, occurrence_records, cell_size=500.0):
    ids, _ = validate_geometry(grid)
    sites = normalize_sites(occurrence_records, grid, cell_size)
    if "grid_id" in fields and not np.array_equal(fields["grid_id"], ids):
        raise ValueError("Score and geometry grid IDs/order differ")
    if "association_names" in fields and tuple(fields["association_names"].astype(str)) != ASSOCIATIONS:
        raise ValueError("association_names must be Cu-Pb-Zn-Ag, W-Sn-Mo-Bi in that order")
    methods = {key: np.asarray(value, float) for key, value in fields.items()
               if key not in ("grid_id", "association_names", "elements", "association_members")}
    if not methods:
        raise ValueError("No method fields supplied")
    rows, per_site, curves = [], [], {"area_fraction": np.arange(len(ids) + 1) / len(ids)}
    for method, values in methods.items():
        if values.shape != (len(ids), 2):
            raise ValueError(f"{method}: expected two fixed-association columns")
        for column, (name, group, targets) in enumerate(zip(ASSOCIATIONS, GROUP_KEYS, MATCH_TARGETS)):
            matching = [i for i, r in enumerate(sites)
                        if set(r["primary_target_elements"].split(";")) & set(targets)]
            metrics, individual, curve = evaluate_field(values[:, column], grid, sites, matching, cell_size)
            context = dict(model=method, association=name, site_group=group)
            rows.append(dict(**context, **metrics))
            per_site.extend(dict(**context, **record) for record in individual)
            curves[method + "__" + group] = curve
    for name in ASSOCIATIONS:
        group = [row for row in rows if row["association"] == name]
        for metric, higher in (("PB_AUC", True), ("N10", True), ("A50", False), ("mean_distance10_km", False)):
            values = np.array([row[metric] for row in group])
            for row, value in zip(group, values):
                better = values > value + 1e-12 if higher else values < value - 1e-12
                row[metric + "_rank"] = 1 + int(better.sum())
    return rows, per_site, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True, help="NPZ with integer grid_id and projected grid_xy[N,2]")
    parser.add_argument("--fields", type=Path, required=True, help="NPZ with grid_id and each method[N,2]")
    parser.add_argument("--occurrences", type=Path, required=True, help="CSV: site_id,grid_id,x,y,primary_target_elements")
    parser.add_argument("--cell-size", type=float, default=500, help="Square side length in metres")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cell_size <= 0:
        parser.error("cell-size must be positive")
    rows, sites, curves = evaluate_methods(load_npz(args.fields), load_npz(args.grid), read_csv(args.occurrences), args.cell_size)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "association_metrics.csv", rows)
    write_csv(args.output / "association_per_site.csv", sites)
    np.savez_compressed(args.output / "association_capture_curves.npz", **curves)
    (args.output / "evaluation_protocol.json").write_text(json.dumps(dict(
        associations=ASSOCIATIONS, members=MEMBERS, matching_primary_targets=MATCH_TARGETS,
        cell_size_m=args.cell_size, background="all supplied effective occurrence cells excluded",
        ties="ascending integer grid ID; AUC gives half weight to score ties",
        distance="minimum to selected square-cell union, not cell-centre distance"), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} method-association evaluations to {args.output}")


if __name__ == "__main__":
    main()
