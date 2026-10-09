"""Calibration splitting and Gaussian scoring mathematics."""
import hashlib
import math
import numpy as np

SPLIT_SALT = "M35-calibration-split-v1"
SCALE_FLOOR = 0.1
NEGATIVE_ENERGY_FLOOR = 1e-4
FIELD_RATIO_BOUNDS = (0.25, 9.0)
THRESHOLD_LEVEL = 0.95
CHEM_DIM = 12
TOP_K = 3


def _encoded_id(value):
    """Type-preserving stable UTF-8 ID encoding; leading-zero strings survive."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, bool):
        return b"bool:" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, float) and math.isfinite(value):
        return b"float:" + value.hex().encode("ascii")
    raise ValueError("cal_ids must be finite scalar strings, bytes, integers or floats")


def split_calibration(cal_ids):
    """Return fit/threshold row indices by salted SHA256 order, independent of rows.

    Hash bytes are SHA256(UTF8(SPLIT_SALT) + NUL + type-preserving ID bytes).
    The first floor(ncal/2) identities fit the scales. The remainder set the
    threshold. A 140-point field consequently uses 70 scale-fit and 70 threshold
    points. Returned indices follow hash order, not input row order.
    """
    ids = np.asarray(cal_ids, dtype=object)
    if ids.ndim != 1 or len(ids) < 2:
        raise ValueError("cal_ids must be a one-dimensional array with at least two IDs")
    encoded = [_encoded_id(v) for v in ids]
    if len(set(encoded)) != len(encoded):
        raise ValueError("calibration IDs must be unique for an unambiguous split")
    salted = SPLIT_SALT.encode("utf-8") + b"\x00"
    keyed = [(hashlib.sha256(salted + value).digest(), value, index)
             for index, value in enumerate(encoded)]
    order = np.asarray([row[2] for row in sorted(keyed)], dtype=np.int64)
    boundary = len(ids) // 2
    return order[:boundary], order[boundary:]


def gaussian_weights(xy, radius_m):
    """Dense row-stochastic planar Gaussian weights, cutoff 3h, including self.

    radius_m is the Gaussian bandwidth h, not the cutoff distance. XY must be
    projected coordinates in metres. h=0 returns the identity, including when
    different rows share an identical location.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 1 or not np.isfinite(xy).all():
        raise ValueError("xy must be finite projected coordinates [N,2]")
    h = float(radius_m)
    if not math.isfinite(h) or h < 0:
        raise ValueError("radius_m must be finite and nonnegative")
    if h == 0:
        return np.eye(len(xy), dtype=np.float64)
    delta = xy[:, None, :] - xy[None, :, :]
    distance2 = np.sum(delta * delta, axis=-1)
    weights = np.exp(-distance2 / (2 * h * h))
    weights[distance2 > (3 * h) ** 2] = 0
    # Diagonal exp(0)=1 guarantees a positive row sum even for isolated sites.
    return weights / weights.sum(axis=1, keepdims=True)


def _field_factors(z, reference_energy, weight):
    """z [...,N,12] -> factors [...,12]; cases remain strictly separate."""
    energy = np.mean(np.minimum(z, 0) ** 2, axis=-2)
    ratio = np.clip(energy / reference_energy, *FIELD_RATIO_BOUNDS)
    factors = np.sqrt((1 - weight) + weight * ratio)
    return factors, energy, ratio


def _top3_positive_mean(signed_residual):
    positive = np.maximum(signed_residual, 0)
    return np.partition(positive, -TOP_K, axis=-1)[..., -TOP_K:].mean(axis=-1)


def strict_order_threshold(calibration_scores, level=THRESHOLD_LEVEL):
    """Order statistic ceil((m+1)*level), used with strict score > threshold.

    For m=70 and level=.95 this is the 68th smallest calibration score. If the
    requested rank exceeds m, infinity is returned; no interpolation or tie
    splitting is performed. This rule alone does not confer conformal validity.
    """
    values = np.asarray(calibration_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a nonempty finite vector")
    if not 0 < float(level) < 1:
        raise ValueError("level must lie strictly between zero and one")
    rank = int(math.ceil((len(values) + 1) * float(level)))
    if rank > len(values):
        return float("inf"), rank
    return float(np.partition(values, rank - 1)[rank - 1]), rank


