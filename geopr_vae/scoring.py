"""M41 Gaussian scorer, adapted from frozen read-only M36 pure mathematics.

Independent calibration halves; only threshold-half rows form the calibration
smoothing graph. Supported bandwidths are0/3000/6000/10000metres, including self
and truncated at3h. Native normalizations are raw/cal_mad/negative_half.
Power correction is disabled and the full calibration-pool mode is rejected.
Field-MAD preprocessing, when requested by the M41 driver, must be applied
separately to each query field and to the two calibration halves before using
norm='cal_mad'. This adaptive preprocessing is explicitly transductive.

All numeric operations after validation reproduce M36 threshold-pool/no-power
calculations. No old source, guard or module constants are modified.
"""
from __future__ import annotations

import numpy as np
from . import scoring_base as archived
from .scoring_base import split_calibration, gaussian_weights, strict_order_threshold

NORMALIZATIONS = ("raw", "cal_mad", "negative_half")
RADII_M = (0.0, 3000.0, 6000.0, 10000.0)
SPLIT_SALT = archived.SPLIT_SALT

def score_fields(residual, cal_residual, val_xy, cal_xy, cal_ids, norm, radius_m,
                 *, pool="threshold", power_correction=False):
    """Same positional API; independent threshold pool and no row-power scaling.

    residual[ncase,nquery,12], cal_residual[ncal,12], projected XY metres.
    cal_score has threshold-half rows in original_threshold_indices order.
    No calibration-fit row joins that graph; query cases remain distinct fields.
    """
    r, cr = np.asarray(residual, dtype=np.float64), np.asarray(cal_residual, dtype=np.float64)
    vx, cx = np.asarray(val_xy, dtype=np.float64), np.asarray(cal_xy, dtype=np.float64)
    ids = np.asarray(cal_ids, dtype=object)
    if r.ndim != 3 or r.shape[-1] != 12 or r.shape[0] < 1 or r.shape[1] < 1:
        raise ValueError("residual must have nonempty shape [ncase,nquery,12]")
    if cr.ndim != 2 or cr.shape[-1] != 12 or len(cr) < 2:
        raise ValueError("cal_residual must have shape [ncal,12], ncal>=2")
    if not np.isfinite(r).all() or not np.isfinite(cr).all():
        raise ValueError("all residuals must be finite")
    if vx.shape != (r.shape[1], 2) or cx.shape != (len(cr), 2) or ids.shape != (len(cr),):
        raise ValueError("coordinate/ID shapes do not match their residual fields")
    if not np.isfinite(vx).all() or not np.isfinite(cx).all():
        raise ValueError("all coordinates must be finite projected metres")
    if norm not in NORMALIZATIONS or float(radius_m) not in RADII_M:
        raise ValueError("norm and radius_m must belong to the M41 native3x4 grid")
    if pool != "threshold":
        raise ValueError("M41 requires independent threshold calibration pool")
    if not isinstance(power_correction, (bool, np.bool_)):
        raise ValueError("power_correction must be boolean")
    if power_correction:
        raise ValueError("M41 Gaussian scoring keeps power_correction=False")
    fit_idx, original_threshold_idx = split_calibration(ids)
    if pool == "threshold":
        cal_pool_indices = original_threshold_idx.copy()
        threshold_indices = np.arange(len(original_threshold_idx), dtype=np.int64)
    else:
        cal_pool_indices = np.arange(len(cr), dtype=np.int64)
        threshold_indices = original_threshold_idx.copy()
    calibration_field = cr[cal_pool_indices]
    calibration_xy = cx[cal_pool_indices]
    fit = cr[fit_idx]
    center = np.median(fit, axis=0)
    unfloored_scale = 1.4826 * np.median(np.abs(fit - center), axis=0)
    scale = np.maximum(unfloored_scale, archived.SCALE_FLOOR)
    fit_z = (fit - center) / scale
    unfloored_energy = np.mean(np.minimum(fit_z, 0) ** 2, axis=0)
    energy_reference = np.maximum(unfloored_energy, archived.NEGATIVE_ENERGY_FLOOR)
    weight = {"raw": 0., "cal_mad": 0., "negative_half": .5, "negative_full": 1.}[norm]
    if norm == "raw":
        values, cal_values = r, calibration_field
    else:
        values, cal_values = (r - center) / scale, (calibration_field - center) / scale
    factors = np.ones((len(r), 12), dtype=np.float64)
    cal_factor = np.ones(12, dtype=np.float64)
    energy = ratio = cal_energy = cal_ratio = None
    if norm in ("negative_half", "negative_full"):
        factors, energy, ratio = archived._field_factors(values, energy_reference, weight)
        cal_factor, cal_energy, cal_ratio = archived._field_factors(cal_values, energy_reference, weight)
        values = values / factors[:, None, :]
        cal_values = cal_values / cal_factor[None, :]
    wq, wc = gaussian_weights(vx, radius_m), gaussian_weights(calibration_xy, radius_m)
    query_power, cal_power = np.sum(wq * wq, axis=1), np.sum(wc * wc, axis=1)
    # Compute row means before dividing, so full+False exactly reproduces M35.
    signed_query = np.matmul(wq, values)
    signed_cal = np.matmul(wc, cal_values)
    if power_correction:
        signed_query = signed_query / np.sqrt(query_power)[None, :, None]
        signed_cal = signed_cal / np.sqrt(cal_power)[:, None]
    scores = archived._top3_positive_mean(signed_query)
    calibration_scores = archived._top3_positive_mean(signed_cal)
    threshold, rank = strict_order_threshold(calibration_scores[threshold_indices])
    if not np.isfinite(scores).all() or not np.isfinite(calibration_scores).all():
        raise ValueError("non-finite scores; check residual magnitudes and numerical units")
    return {"score": scores, "cal_score": calibration_scores, "threshold": threshold,
            "threshold_indices": threshold_indices, "original_threshold_indices": original_threshold_idx,
            "calfit_indices": fit_idx, "cal_pool_indices": cal_pool_indices,
            "field_factor": factors, "cal_factor": cal_factor,
            "diagnostics": {
                "norm": norm, "radius_m": float(radius_m), "gaussian_cutoff_m": 3 * float(radius_m),
                "pool": pool, "power_correction": bool(power_correction),
                "production_default": pool == "threshold" and not bool(power_correction),
                "m41_fixed_pool_and_power": True,
                "ncase": len(r), "nquery": r.shape[1], "ncal_original": len(cr),
                "n_calibration_graph_pool": len(cal_pool_indices), "n_scale_fit": len(fit_idx),
                "n_threshold": len(threshold_indices), "cal_graph_includes_scale_fit": pool == "full",
                "split_salt": SPLIT_SALT,
                "center_calfit": center.tolist(), "scale_calfit": scale.tolist(),
                "scale_floor_B2_units": archived.SCALE_FLOOR,
                "mad_floor_applied_channels": int(np.sum(unfloored_scale < archived.SCALE_FLOOR)),
                "negative_energy_reference": energy_reference.tolist(),
                "negative_energy_floor": archived.NEGATIVE_ENERGY_FLOOR,
                "negative_energy_floor_applied_channels": int(np.sum(unfloored_energy < archived.NEGATIVE_ENERGY_FLOOR)),
                "negative_weight": weight, "field_ratio_bounds": list(archived.FIELD_RATIO_BOUNDS),
                "field_negative_energy": None if energy is None else energy.tolist(),
                "field_clipped_energy_ratio": None if ratio is None else ratio.tolist(),
                "cal_negative_energy": None if cal_energy is None else cal_energy.tolist(),
                "cal_clipped_energy_ratio": None if cal_ratio is None else cal_ratio.tolist(),
                "query_row_power": query_power.tolist(), "cal_row_power": cal_power.tolist(),
                "query_effective_n_range": [float((1 / query_power).min()), float((1 / query_power).max())],
                "cal_effective_n_range": [float((1 / cal_power).min()), float((1 / cal_power).max())],
                "threshold_rank_1based": rank, "threshold_level": .95,
                "threshold_rule": "ceil((m+1)*.95) order statistic; strict score > threshold; no tie splitting",
                "signed_smoothing_before_positive_top3": True, "separate_cal_query_graphs": True,
                "query_cases_separate": True, "transductive": norm.startswith("negative_") or float(radius_m) > 0,
                "no_conformal_guarantee": True,
                "limitations": [
                    "M41 does not apply row-power correction; smoothing changes marginal variance according to neighborhood geometry.",
                    "Spatially correlated noise instead has row variance w^T K w; correcting sum(w^2) need not calibrate it.",
                    "Row-stochastic Gaussian means preserve constant fields; fitted-center errors and shared noise can persist.",
                    "Field normalization and smoothing remain transductive and induce dependence; no conformal guarantee is claimed.",
                    "Even disjoint calibration pools do not remove scale estimation uncertainty or calibration-query density differences.",
                    "No score invariance to positive enrichment or to the target chemistry is claimed.",
                    "Projected metre coordinates in the same CRS are assumed, not inferable from the numerical arrays."]}}


def fit_affine_map(cal_scores):
    """Calibration-only affine score coordinate with explicit degenerate fallback.

    Scale is 1.4826 MAD when positive, then IQR/1.349, then population SD,
    then the fixed unit scale for a completely constant calibration field.
    The latter two fallbacks are reported, not represented as robust estimates.
    No score is clipped at the calibration maximum or at any other bound.
    """
    s = np.asarray(cal_scores, dtype=np.float64)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all():
        raise ValueError("cal_scores must be a finite nonempty vector")
    median = float(np.median(s))
    mad = float(1.4826 * np.median(np.abs(s - median)))
    iqr = float((np.quantile(s, .75) - np.quantile(s, .25)) / 1.349)
    sd = float(np.std(s, ddof=0))
    if mad > 0:
        scale, source = mad, "1.4826_MAD"
    elif iqr > 0:
        scale, source = iqr, "IQR_div_1.349"
    elif sd > 0:
        scale, source = sd, "population_SD_fallback"
    else:
        scale, source = 1.0, "constant_field_unit_fallback"
    return {"center": median, "scale": scale, "scale_source": source,
            "calibration_mad": mad, "calibration_iqr_scale": iqr, "calibration_sd": sd,
            "constant_calibration_field": bool(np.ptp(s) == 0), "clip": None,
            "fit_pool": "threshold_calibration_scores_only", "mineralization_probability": False}


def affine_map(scores, params):
    return (np.asarray(scores, dtype=np.float64) - params["center"]) / params["scale"]

