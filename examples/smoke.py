"""Run from the repository root: python -m examples.smoke."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from geopr_vae import pipeline, reference, recovery, scoring
from geopr_vae.preprocessing import fit_preprocessor, transform
from geopr_vae.mapping import grid_scores
from geopr_vae.experiments import paired_intervention


def synthetic(seed=2718):
    """Small illustrative field, independent of the paper's confirmation data."""
    rng = np.random.default_rng(seed)
    n = 160
    xy = rng.uniform([400000., 2300000.], [480000., 2380000.], (n, 2))
    terrain = np.column_stack([rng.normal(200, 50, n), rng.uniform(0, 30, n), rng.normal(0, 10, n)])
    groups = np.arange(n) % 4
    background = 1 + .003 * terrain[:, :1] * np.linspace(.2, 1., 12) + .1 * np.sin(xy[:, :1] / 15000.)
    enrichment = np.exp(-np.square((xy - [440000., 2340000.]) / 11000.).sum(1))[:, None] * np.array([.3, .2, .2, 0, 0, 0, 0, 0, .15, 0, 0, 0])
    clean = background + enrichment
    return dict(log10=clean + rng.normal(0, .12, (n, 12)), xy=xy, terrain=terrain, groups=groups,
                ids=np.array([f"s{i:06d}" for i in range(n)]), train=np.arange(96),
                calibration=np.arange(96, 136)), clean


def run(output=None):
    data, clean = synthetic()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **data)
    # Calibration/evaluation chemistry must not affect fitted preprocessing.
    pp = fit_preprocessor(data, data["train"])
    changed = dict(data, log10=data["log10"].copy())
    changed["log10"][96:] += 100
    assert pp == fit_preprocessor(changed, data["train"])
    a = transform(data, pp)
    query = {k: data[k][136:] for k in ("log10", "terrain", "xy", "groups", "ids")}
    report = {}
    for method in pipeline.CONFIGS:
        bundle, history = pipeline.fit(data, method, data["train"], data["calibration"],
                                       seed=20260929, teacher_epochs=8, head_epochs=12)
        prediction, details = pipeline.predict(bundle, query)
        assert all(np.isfinite(v).all() for v in prediction.values() if v.dtype.kind in "fc")
        assert prediction["raw_scores"].shape == (24, 13)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            pipeline.save(bundle, path)
            again, _ = pipeline.predict(pipeline.load(path), query, batch_size=7)
            np.testing.assert_allclose(prediction["reference"], again["reference"], rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(prediction["affine_scores"], again["affine_scores"], rtol=2e-4, atol=2e-5)
        grid = grid_scores(prediction["xy"], prediction["affine_scores"], query["xy"][:6])
        np.testing.assert_allclose(grid["affine_scores"], prediction["affine_scores"][:6], atol=1e-10)
        delta = np.zeros_like(query["log10"])
        delta[:8, [0, 2, 8]] = .05
        pair, _, _ = paired_intervention(bundle, query, delta, clean[136:])
        if method == "GeoPR-VAE":
            model = reference.restore(bundle["reference"])
            altered = a["r"][:12].copy()
            altered[:, 0] += 2.
            p0 = model.predict(a["r"][:12], a["F"][:12], ids=a["ids"][:12])["reference"]
            p1 = model.predict(altered, a["F"][:12], ids=a["ids"][:12])["reference"]
            np.testing.assert_allclose(p0[:, 0], p1[:, 0], atol=1e-7)
            head = recovery.restore(bundle["head"])
            residual = a["r"][:30]
            base = head.predict(residual, a["F"][:30], a["xy"][:30])["correction"]
            shifted = head.predict(residual + .15, a["F"][:30], a["xy"][:30])["correction"]
            np.testing.assert_allclose(shifted - base, .15, atol=3e-6)
            assert history["recovery"][-1]["nuisance_null_rms"] > 0
        report[method] = dict(reference_parameters=bundle["training"]["teacher_parameter_count"],
                              head_parameters=bundle["training"]["head_parameter_count"],
                              epochs=[8, 12 if method == "GeoPR-VAE" else 0],
                              query_rows=24, checkpoint_and_batch_check=True, paired_intervention=pair)
    report["status"] = "PASS"
    report["scope"] = "Synthetic execution and structural checks only; no paper-performance replication claim"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-input", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run(args.save_input)
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
