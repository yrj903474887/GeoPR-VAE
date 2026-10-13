"""Map patch-centre residuals to the two fixed regional associations.

Consumes the centre [16,16] residual channels from benchmarks.patch_runner.
The controlled-recovery midpoint readout is deliberately not interchangeable
with this regional scoring interface. No occurrence inventory is accepted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from evaluation.regional import ASSOCIATIONS, ELEMENTS, fixed_association_fields, load_npz, validate_geometry
from geopr_vae import scoring, mapping


def map_prediction(prediction, grid):
    query = np.asarray(prediction["query_patch16_signed_residual"], float)
    calibration = np.asarray(prediction["cal_patch16_signed_residual"], float)
    xy, cal_xy = np.asarray(prediction["query_xy"], float), np.asarray(prediction["cal_xy"], float)
    cal_ids = np.asarray(prediction["cal_ids"])
    ids, grid_xy = validate_geometry(grid)
    if "ids" in grid and not np.array_equal(grid["ids"].astype(str), prediction["query_ids"].astype(str)):
        raise ValueError("Query order differs from the interpolation support order")
    if "xy" in grid and not np.array_equal(grid["xy"], xy):
        raise ValueError("Query coordinates differ from interpolation support")
    result = scoring.score_fields(query[None], calibration, xy, cal_xy, cal_ids,
                                  "cal_mad", 10000, pool="threshold", power_correction=False)
    threshold = result["original_threshold_indices"]
    centre = np.asarray(result["diagnostics"]["center_calfit"])
    scale = np.asarray(result["diagnostics"]["scale_calfit"])
    signed = scoring.gaussian_weights(xy, 10000) @ ((query - centre) / scale)
    cal_signed = scoring.gaussian_weights(cal_xy[threshold], 10000) @ ((calibration[threshold] - centre) / scale)
    maps = [scoring.fit_affine_map(cal_signed[:, j]) for j in range(12)]
    mapped = np.column_stack([scoring.affine_map(signed[:, j], maps[j]) for j in range(12)])
    if "idw_indices" in grid and "idw_weights" in grid:
        if "ids" not in grid or "xy" not in grid:
            raise ValueError("Cached IDW requires ids and xy to verify query support alignment")
        indices, weights = grid["idw_indices"], grid["idw_weights"]
        if indices.shape != weights.shape or indices.shape[0] != len(ids):
            raise ValueError("Invalid cached IDW geometry")
        if not np.allclose(weights.sum(1), 1, atol=1e-12, rtol=0) or (weights < 0).any():
            raise ValueError("IDW weights must be nonnegative and row-normalized")
        neighbors = mapped[indices]
        anchor = neighbors[:, :1]
        grid_elements = np.round(anchor[:, 0] + np.sum(weights[..., None] * (neighbors - anchor), axis=1), 12)
    else:
        composite = np.partition(np.maximum(signed, 0), -3, axis=1)[:, -3:].mean(1)
        cal_composite = np.partition(np.maximum(cal_signed, 0), -3, axis=1)[:, -3:].mean(1)
        composite = scoring.affine_map(composite, scoring.fit_affine_map(cal_composite))
        grid_elements = mapping.grid_scores(xy, np.column_stack((composite, mapped)), grid_xy)["affine_scores"][:, 1:]
    associations = fixed_association_fields(grid_elements, ELEMENTS)
    return grid_elements, associations, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, metavar="METHOD=FILE.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grid = load_npz(args.grid)
    fields = dict(grid_id=grid["grid_id"], association_names=np.asarray(ASSOCIATIONS))
    responses = dict(grid_id=grid["grid_id"], elements=np.asarray(ELEMENTS))
    parameters = {}
    for item in args.prediction:
        if "=" not in item:
            parser.error("Use --prediction METHOD=FILE.npz")
        method, filename = item.split("=", 1)
        if not method or method in fields:
            parser.error("Method names must be nonempty and unique")
        elements, fields[method], parameters[method] = map_prediction(load_npz(filename), grid)
        responses[method] = elements
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "association_fields.npz", **fields)
    np.savez_compressed(args.output / "element_response_fields.npz", **responses)
    (args.output / "affine_calibration.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
    print(f"Mapped {len(parameters)} methods to {len(grid['grid_id'])} common cells")


if __name__ == "__main__":
    main()

