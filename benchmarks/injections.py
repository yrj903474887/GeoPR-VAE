"""Coordinate-only regional paired injections, with the 6x3x4x2 design."""
import hashlib,json
import numpy as np
TARGETS = ["Au", "Ag", "As", "Bi", "Cu", "F", "Mo", "Pb", "Sb", "Sn", "W", "Zn"]
SHAPES = [
    {"name": "compact", "sigma_major_m": 3000.0, "sigma_minor_m": 3000.0, "angle_degrees_ccw_from_east": 0.0},
    {"name": "broad", "sigma_major_m": 6000.0, "sigma_minor_m": 6000.0, "angle_degrees_ccw_from_east": 0.0},
    {"name": "elongated", "sigma_major_m": 8000.0, "sigma_minor_m": 2500.0, "angle_degrees_ccw_from_east": 45.0},
]
SUPPORTS = [
    {"name": "CuPbZnAg", "elements": ["Cu", "Pb", "Zn", "Ag"]},
    {"name": "WSnMoBi", "elements": ["W", "Sn", "Mo", "Bi"]},
    {"name": "As", "elements": ["As"]},
    {"name": "all12", "elements": TARGETS.copy()},
]
AMPLITUDES = [0.25, 0.5]
DESIGN={'centers_per_fold':6}
LIMITATIONS=[
    'Observed regional fields do not provide analytical clean truth.',
    'Labels represent injected geometry, not known mineralization.',
    'Cases and paired buffers are dependent; summarize within each fold first.',
    'Training-range clipping and spatial interpolation may attenuate input signals.'
]
def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def _frame_digest(ids, xy, targets):
    digest = hashlib.sha256()
    digest.update(_json(list(map(str, ids))).encode("utf-8"))
    digest.update(np.ascontiguousarray(xy, dtype="<f8").tobytes())
    digest.update(_json(list(map(str, targets))).encode("utf-8"))
    return digest.hexdigest()

def mahalanobis_squared(xy, centers, shape):
    """Return [number of centers, number of points] dimensionless squared radius."""
    xy = np.asarray(xy, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    angle = np.deg2rad(shape["angle_degrees_ccw_from_east"])
    dx = xy[None, :, 0] - centers[:, None, 0]
    dy = xy[None, :, 1] - centers[:, None, 1]
    major = (np.cos(angle) * dx + np.sin(angle) * dy) / shape["sigma_major_m"]
    minor = (-np.sin(angle) * dx + np.cos(angle) * dy) / shape["sigma_minor_m"]
    return major * major + minor * minor

def select_centers(xy):
    """Select six candidates without inspecting chemistry, labels, or scores."""
    xy = np.asarray(xy, dtype=np.float64)
    candidates = np.unique(xy, axis=0)  # lexicographic order makes ties deterministic
    eligible = np.ones(len(candidates), dtype=bool)
    support_counts = {}
    for shape in SHAPES:
        distance2 = mahalanobis_squared(xy, candidates, shape)
        core = (distance2 <= 1.0).sum(axis=1)
        negative = (distance2 >= 9.0).sum(axis=1)
        eligible &= (core >= 4) & (negative >= 100)
        support_counts[shape["name"]] = (core, negative)
    indices = np.flatnonzero(eligible)
    if len(indices) < DESIGN["centers_per_fold"]:
        raise RuntimeError(f"Fixed design infeasible: only {len(indices)} eligible centers; need 6. No performance-based adjustment is permitted.")
    eligible_xy = candidates[indices]
    centroid = xy.mean(axis=0)
    first = int(np.argmin(np.square(eligible_xy - centroid).sum(axis=1)))
    chosen = [first]
    closest2 = np.square(eligible_xy - eligible_xy[first]).sum(axis=1)
    while len(chosen) < DESIGN["centers_per_fold"]:
        closest2[chosen] = -1.0
        next_index = int(np.argmax(closest2))
        chosen.append(next_index)
        closest2 = np.minimum(closest2, np.square(eligible_xy - eligible_xy[next_index]).sum(axis=1))
    selected = indices[chosen]
    centers = candidates[selected]
    metadata = {
        "unique_coordinate_candidates": len(candidates),
        "eligible_candidates": int(eligible.sum()),
        "query_centroid_utm_m": centroid.tolist(),
        "centers_utm_m": centers.tolist(),
        "center_core_and_negative_counts": {
            name: {"positive": core[selected].astype(int).tolist(), "negative": negative[selected].astype(int).tolist()}
            for name, (core, negative) in support_counts.items()
        },
    }
    return centers, metadata

def build_injections(job, *, frame=None, centers=None, center_metadata=None):
    """Return deltas, masks and auditable case metadata; never use model results."""
    if frame is None: raise ValueError('Supply a frame with ids, xy and targets')
    frame=dict(frame)
    if list(frame['targets']) != TARGETS: raise ValueError('Target order mismatch')
    frame['sha256']=_frame_digest(frame['ids'],frame['xy'],frame['targets'])
    if centers is None:
        centers, center_metadata = select_centers(frame["xy"])
    if center_metadata is None:
        raise ValueError("Explicit centers must be accompanied by their selection metadata")
    nq = len(frame["ids"])
    deltas = np.zeros((144, nq, 12), dtype=np.float32)
    positives = np.empty((144, nq), dtype=bool)
    negatives = np.empty((144, nq), dtype=bool)
    cases = []
    for center_index, center in enumerate(centers):
        center_ids = frame["ids"][np.all(frame["xy"] == center, axis=1)].tolist()
        for shape in SHAPES:
            distance2 = mahalanobis_squared(frame["xy"], center, shape)[0]
            positive = distance2 <= 1.0
            negative = distance2 >= 9.0
            kernel = np.where(distance2 < 9.0, np.exp(-0.5 * distance2), 0.0)
            for support in SUPPORTS:
                active = [TARGETS.index(element) for element in support["elements"]]
                for amplitude in AMPLITUDES:
                    case_index = len(cases)
                    case_id = f"c{center_index + 1:02d}_{shape['name']}_{support['name']}_a{amplitude:g}"
                    # Basic-index loop avoids NumPy advanced-index axis swapping.
                    for channel in active:
                        deltas[case_index, :, channel] = amplitude * kernel
                    positives[case_index] = positive
                    negatives[case_index] = negative
                    cases.append({
                        "index": case_index, "case_id": case_id, "center_index": center_index,
                        "center_utm_m": center.tolist(), "center_query_ids": center_ids,
                        "shape": shape["name"], "support": support["name"],
                        "active_elements": support["elements"], "active_channel_indices": active,
                        "active_channel_count": len(active), "amplitude_per_active_element_B2": amplitude,
                        "positive_count": int(positive.sum()), "negative_count": int(negative.sum()),
                        "transition_count": int((~(positive | negative)).sum()),
                        "total_squared_energy_B2": float(np.square(deltas[case_index], dtype=np.float64).sum()),
                    })
    metadata = {
        "job": job, "fold": int(job.rsplit("fold", 1)[1]),
        "query_frame_sha256": frame["sha256"], "n_query": nq,
        "target_order": TARGETS, "case_count": len(cases), "delta_units": "B2 standardized units",
        "center_selection": center_metadata, "cases": cases,
        "source_fields_used": ["xy[iq]", "ids[iq]", "targets"],
        "chemistry_values_used_for_design": False, "mineral_information_used": False,
        "model_results_used_for_design": False, "network_inference_performed": False,
        "limitations": LIMITATIONS,
    }
    result = {"deltas": deltas, "positive_mask": positives, "negative_mask": negatives,
              "query_ids": frame["ids"], "query_xy": frame["xy"], "targets": frame["targets"],
              "case_ids": np.asarray([case["case_id"] for case in cases]), "metadata": metadata}
    validate_injections(result)
    return result

def validate_injections(result):
    delta, positive, negative = result["deltas"], result["positive_mask"], result["negative_mask"]
    cases = result["metadata"]["cases"]
    if delta.shape != (144, len(result["query_ids"]), 12):
        raise AssertionError(f"Unexpected delta shape: {delta.shape}")
    if positive.shape != delta.shape[:2] or negative.shape != delta.shape[:2]:
        raise AssertionError("Mask shape mismatch")
    if len(cases) != 144 or len(set(result["case_ids"].tolist())) != 144:
        raise AssertionError("Expected exactly 144 distinct cases")
    if not np.isfinite(delta).all() or (delta < 0).any():
        raise AssertionError("Delta must be finite and nonnegative")
    if (positive & negative).any() or (positive.sum(axis=1) < 4).any() or (negative.sum(axis=1) < 100).any():
        raise AssertionError("Positive/negative geometry count constraints failed")
    if np.any(delta[negative] != 0):
        raise AssertionError("Negative controls must have zero direct injection")
    expected_keys = {(center, shape["name"], support["name"], amplitude)
                     for center in range(6) for shape in SHAPES for support in SUPPORTS for amplitude in AMPLITUDES}
    actual_keys = {(case["center_index"], case["shape"], case["support"], case["amplitude_per_active_element_B2"]) for case in cases}
    if actual_keys != expected_keys:
        raise AssertionError("Incomplete factorial design")
    for case in cases:
        index = case["index"]
        active = case["active_channel_indices"]
        inactive = sorted(set(range(12)) - set(active))
        if inactive and np.any(delta[index][:, inactive] != 0):
            raise AssertionError("An inactive element received injection")
        if np.any(delta[index][positive[index]][:, active] <= 0):
            raise AssertionError("Positive core must be injected in every active element")
        if not np.allclose(delta[index][:, active].max(axis=0), case["amplitude_per_active_element_B2"], rtol=0, atol=1e-7):
            raise AssertionError("Center peak amplitude mismatch")
    # The second amplitude doubles each active delta, hence quadruples energy.
    if not np.array_equal(delta[1::2], 2 * delta[0::2]):
        raise AssertionError("Amplitude pairing mismatch")
    return {
        "status": "PASS", "case_count": 144,
        "positive_count_min": int(positive.sum(axis=1).min()),
        "negative_count_min": int(negative.sum(axis=1).min()),
        "finite_nonnegative": True, "negative_direct_delta_zero": True,
        "complete_6x3x4x2_factorial": True, "amplitude_pairing_exact": True,
    }
