"""Training-only robust group-shrinkage standardization."""
import numpy as np
EPS = 1e-12

def _matrix(values, width, name, finite=True):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != width or len(result) == 0:
        raise ValueError(f"{name} must be nonempty [N,{width}]")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {name}")
    return result


def _indices(indices, n):
    ix = np.asarray(indices)
    if ix.ndim != 1 or ix.dtype.kind not in "iu" or not len(ix):
        raise ValueError("G must be nonempty integer indices")
    if np.any(ix < 0) or np.any(ix >= n) or len(np.unique(ix)) != len(ix):
        raise ValueError("Invalid or repeated G index")
    return ix.astype(np.int64)


def robust_scale(train_values):
    """MAD -> linear IQR/1.349 -> sample SD; fail if no positive scale exists."""
    y = _matrix(train_values, 12, "training observations")
    if len(y) < 2:
        raise ValueError("At least two training observations required")
    location = np.median(y, axis=0)
    scale = 1.4826*np.median(np.abs(y-location), axis=0)
    iqr = np.diff(np.quantile(y, [.25,.75], axis=0, method="linear"), axis=0)[0]/1.349
    scale = np.where(scale > EPS, scale, iqr)
    sd = y.std(axis=0, ddof=1)
    scale = np.where(scale > EPS, scale, sd)
    if not np.isfinite(scale).all() or np.any(scale <= EPS):
        raise ValueError("Training global scale is unavailable")
    return scale


def fit_b2(observed_raw, groups, G):
    """Return mu/sigma/support/fallback, using only G response values to fit.

    The caller may pass all groups to address output references. Evaluation
    responses never enter a fitted statistic, including validation of scales.
    """
    y = _matrix(observed_raw, 12, "observations", finite=False)
    groups = np.asarray(groups)
    if groups.shape != (len(y),) or groups.dtype.kind not in "iu":
        raise ValueError("groups must be a row-aligned integer vector")
    G = _indices(G, len(y))
    train = y[G]
    if not np.isfinite(train).all():
        raise ValueError("Nonfinite training observations")
    global_mu = np.median(train, axis=0)
    global_sigma = robust_scale(train)
    mu = np.broadcast_to(global_mu, y.shape).copy()
    sigma = np.broadcast_to(global_sigma, y.shape).copy()
    support = np.zeros(len(y), dtype=np.int64)
    fallback = np.ones(len(y), dtype=bool)
    for group in np.unique(groups[G]):
        train_group = train[groups[G] == group]
        target = groups == group
        count = len(train_group)
        support[target] = count
        if count < 5:
            continue
        med = np.median(train_group, axis=0)
        mad = 1.4826*np.median(np.abs(train_group-med), axis=0)
        alpha = count/(count+50.)
        mu[target] = alpha*med + (1.-alpha)*global_mu
        sigma[target] = np.maximum(np.sqrt(alpha*mad**2+(1.-alpha)*global_sigma**2),
                                   .1*global_sigma)
        fallback[target] = False
    return {"mu":mu, "sigma":sigma, "support_n":support, "fallback":fallback,
            "global_mu":global_mu, "global_sigma":global_sigma}

