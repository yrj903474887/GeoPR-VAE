"""Prepare portable equal-area geometry and fixed-element fields.

Geometry input CSV: grid_id,x,y, with x/y in a projected metre-based CRS.
Scoring input NPZ: grid_id, elements, and one [N,12] element-response matrix
per method. These are calibrated, spatially aggregated *signed responses*,
not raw chemical concentrations or native scalar anomaly scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.regional import ASSOCIATIONS, ELEMENTS, fixed_association_fields, load_npz, validate_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    geometry = commands.add_parser("geometry", help="Validate a supplied effective grid; do not expand its evaluation area")
    geometry.add_argument("--csv", type=Path, required=True)
    geometry.add_argument("--crs", required=True, help="Projected metre CRS, e.g. EPSG:32649")
    geometry.add_argument("--cell-size", type=float, default=500)
    geometry.add_argument("--output", type=Path, required=True)
    fields = commands.add_parser("associations", help="Create two fixed associations from element response grids")
    fields.add_argument("--responses", type=Path, required=True)
    fields.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "geometry":
        from pyproj import CRS
        crs = CRS.from_user_input(args.crs)
        if not crs.is_projected or not all(abs(axis.unit_conversion_factor - 1) < 1e-12 for axis in crs.axis_info[:2]):
            parser.error("The geometry CRS must be projected in metres")
        if args.cell_size <= 0:
            parser.error("cell-size must be positive")
        data = pd.read_csv(args.csv)
        if not {"grid_id", "x", "y"} <= set(data.columns):
            parser.error("Geometry CSV requires grid_id,x,y")
        raw_ids = data.grid_id.to_numpy()
        if not np.all(raw_ids == raw_ids.astype(np.int64)):
            parser.error("grid_id must be integer")
        grid = dict(grid_id=raw_ids.astype(np.int64), grid_xy=data[["x", "y"]].to_numpy(float),
                    crs=np.asarray(crs.to_string()), cell_size_m=np.asarray(args.cell_size))
        validate_geometry(grid)
        if len(np.unique(grid["grid_xy"], axis=0)) != len(data):
            parser.error("Repeated grid coordinates")
        for axis in (0, 1):
            fractional = (grid["grid_xy"][:, axis] - grid["grid_xy"][0, axis]) / args.cell_size
            if not np.allclose(fractional, np.rint(fractional), atol=1e-8, rtol=0):
                parser.error("Coordinates do not lie on the specified square lattice")
        np.savez_compressed(args.output, **grid)
    else:
        inputs = load_npz(args.responses)
        if "grid_id" not in inputs:
            parser.error("Responses need grid_id for alignment")
        elements = inputs.get("elements", np.asarray(ELEMENTS)).astype(str).tolist()
        output = dict(grid_id=inputs["grid_id"], association_names=np.asarray(ASSOCIATIONS))
        for method, values in inputs.items():
            if method not in ("grid_id", "elements"):
                if len(values) != len(inputs["grid_id"]):
                    parser.error("Grid length mismatch for " + method)
                output[method] = fixed_association_fields(values, elements)
        if len(output) == 2:
            parser.error("At least one method is required")
        np.savez_compressed(args.output, **output)
    print(json.dumps({"output": str(args.output), "command": args.command}))


if __name__ == "__main__":
    main()

