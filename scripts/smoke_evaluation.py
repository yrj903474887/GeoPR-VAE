"""Exercise evaluation with generated toy inputs, never paper data.

Write output outside the source repository, e.g.
``python -m scripts.smoke_evaluation --output ../smoke``.
The small input is for software checks, not a scientific performance result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from evaluation.regional import (ASSOCIATIONS, evaluate_methods, fixed_association_fields,
    write_csv, square_distance_km)
from evaluation.controlled import bootstrap_mean_ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Smoke outputs must be outside the source repository")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(517)
    gx, gy = np.meshgrid(np.arange(10) * 500 + 500000, np.arange(10) * 500 + 2450000)
    grid = dict(grid_id=np.arange(100), grid_xy=np.column_stack((gx.ravel(), gy.ravel())))
    indices = [2, 15, 29, 44, 68, 91]
    targets = ["Cu", "Pb;Zn", "Ag", "W", "Sn", "Mo"]
    sites = [dict(site_id=f"toy{i}", grid_id=j, x=grid["grid_xy"][j, 0], y=grid["grid_xy"][j, 1],
                  primary_target_elements=target) for i, (j, target) in enumerate(zip(indices, targets))]
    methods = ["AE", "VAE", "VAE_patch", "GANomaly", "Skip_GANomaly", "GeoPR-VAE"]
    fields = dict(grid_id=grid["grid_id"], association_names=np.asarray(ASSOCIATIONS))
    for method in methods:
        responses = rng.integers(0, 20, size=(100, 12)).astype(float)
        fields[method] = fixed_association_fields(responses)
    rows, per_site, curves = evaluate_methods(fields, grid, sites)
    background = np.setdiff1d(np.arange(100), indices)
    for row in rows:
        column = ASSOCIATIONS.index(row["association"])
        matched = indices[:3] if column == 0 else indices[3:]
        field = fields[row["model"]][:, column]
        pairwise = field[matched, None] - field[background][None, :]
        brute_auc = np.mean((pairwise > 0) + .5 * (pairwise == 0))
        np.testing.assert_allclose(row["PB_AUC"], brute_auc, atol=1e-14, rtol=0)
        curve = curves[row["model"] + "__" + row["site_group"]]
        assert row["N10"] == int(round(curve[10] * 3))
    np.testing.assert_allclose(square_distance_km([[0, 0], [1000, 0]], [[0, 0]], 500), [0, .75])
    np.testing.assert_array_equal(bootstrap_mean_ci(np.ones(10)), [1, 1])
    np.savez_compressed(output / "grid.npz", **grid)
    np.savez_compressed(output / "fields.npz", **fields)
    np.savez_compressed(output / "association_capture_curves.npz", **curves)
    write_csv(output / "occurrences.csv", sites)
    write_csv(output / "association_metrics.csv", rows)
    write_csv(output / "association_per_site.csv", per_site)
    report = dict(status="PASS", point_background_auc="independent pairwise replay",
                  square_distance="inside and outside analytic cases", tie_handling="integer-grid deterministic",
                  rows=len(rows), generated_toy_data=True)
    (output / "smoke_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
