"""Paired complete-pipeline interventions for independent controlled evaluation."""
import numpy as np
from .pipeline import predict
from .preprocessing import transform


def paired_intervention(bundle, query, delta_log10, clean_log10=None):
    """Re-infer the frozen reference and recovery for both complete fields.

    Truth is an optional evaluator-only argument; it never enters fitting.
    These compact diagnostics do not reproduce the full paper experiment grid.
    """
    delta = np.asarray(delta_log10, float)
    if delta.shape != np.asarray(query["log10"]).shape or not np.isfinite(delta).all() or (delta < 0).any():
        raise ValueError("delta_log10 must be finite, nonnegative and [N,12]")
    if not np.any(delta > 0):
        raise ValueError("A positive intervention is required")
    perturbed = dict(query, log10=np.asarray(query["log10"]) + delta)
    before, _ = predict(bundle, query)
    after, _ = predict(bundle, perturbed)
    pp = transform(query, bundle["preprocessing"])
    standardized_delta = delta / pp["sigma"]
    change = after["recovered"] - before["recovered"]
    energy = float(np.square(standardized_delta).sum())
    report = dict(amplitude_projection_gain_B2=float(np.sum(change * standardized_delta) / energy),
                  relative_pair_error_B2=float(np.sqrt(np.square(change - standardized_delta).sum() / energy)),
                  off_support_rms_B2=float(np.sqrt(np.mean(np.square(change[delta == 0])))) if (delta == 0).any() else None,
                  reference_recomputed=True, same_sample_ids_and_latent_draw_keys=True)
    if clean_log10 is not None:
        truth = np.asarray(clean_log10, float)
        if truth.shape != delta.shape or not np.isfinite(truth).all():
            raise ValueError("clean_log10 must be finite [N,12]")
        report["before_rmse_log10"] = float(np.sqrt(np.mean((before["recovered_log10"] - truth) ** 2)))
        report["after_rmse_log10"] = float(np.sqrt(np.mean((after["recovered_log10"] - truth - delta) ** 2)))
    return report, before, after
