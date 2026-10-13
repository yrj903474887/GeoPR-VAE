"""Persistable training-only preprocessing and the fixed regional partition."""
import hashlib
import numpy as np
from .preprocessing_base import fit_b2

ELEMENTS = ["Au", "Ag", "As", "Bi", "Cu", "F", "Mo", "Pb", "Sb", "Sn", "W", "Zn"]


def validate_data(data):
    """Accept already quality-filtered, log10-transformed measurements."""
    d = {k: np.asarray(data[k]) for k in ("log10", "terrain", "xy", "groups", "ids")}
    n = len(d["log10"])
    for key, width in (("log10", 12), ("terrain", 3), ("xy", 2)):
        d[key] = d[key].astype(np.float64)
        if d[key].shape != (n, width) or not n or not np.isfinite(d[key]).all():
            raise ValueError(f"{key} must be finite [{n},{width}]")
    if d["groups"].shape != (n,) or d["groups"].dtype.kind not in "iu":
        raise ValueError("groups must be row-aligned integer statistical identifiers")
    if d["ids"].shape != (n,) or d["ids"].dtype.kind not in "US":
        raise ValueError("ids must be unique strings; preserve leading zeroes")
    d["ids"] = d["ids"].astype(str)
    if len(np.unique(d["ids"])) != n:
        raise ValueError("Sample IDs must be unique")
    return d


def regional_split(ids, xy, fraction=0.2):
    """10 km tiles, Hamilton allocation and ID hashes; chemistry is unused."""
    ids, xy = np.asarray(ids).astype(str), np.asarray(xy, float)
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie in (0,1)")
    tiles, inverse, count = np.unique(np.floor(xy / 10000.).astype(int), axis=0,
                                      return_inverse=True, return_counts=True)
    salt = "M80-global-calibration-20260929"
    number = int(np.floor(len(ids) * fraction))
    allocation = np.floor(count * fraction).astype(int)
    remainder = count * fraction - allocation
    tile_hash = [hashlib.sha256(f"{salt}|{a}|{b}".encode()).hexdigest() for a, b in tiles]
    order = sorted(range(len(tiles)), key=lambda i: (-remainder[i], tile_hash[i]))
    for i in order[:number - int(allocation.sum())]:
        allocation[i] += 1
    selected = []
    for i, size in enumerate(allocation):
        members = np.flatnonzero(inverse == i)
        members = sorted(members, key=lambda j: hashlib.sha256(f"{salt}|{ids[j]}".encode()).hexdigest())
        selected.extend(members[:size])
    calibration = np.sort(np.asarray(selected, dtype=np.int64))
    return np.setdiff1d(np.arange(len(ids)), calibration), calibration


def fit_preprocessor(data, train):
    d = validate_data(data)
    train = np.asarray(train)
    b2 = fit_b2(d["log10"], d["groups"], train)
    mean, sd = d["terrain"][train].mean(0), d["terrain"][train].std(0)
    if np.any(sd <= 0):
        raise ValueError("Each training terrain variable needs nonzero variation")
    c = np.column_stack([(d["terrain"] - mean) / sd,
                         b2["support_n"] / (b2["support_n"] + 50.), b2["fallback"].astype(float)])
    group_stats = {}
    for group in np.unique(d["groups"][train]):
        i = np.flatnonzero(d["groups"] == group)[0]
        group_stats[str(int(group))] = dict(mu=b2["mu"][i].tolist(), sigma=b2["sigma"][i].tolist(),
                                          n=int(b2["support_n"][i]), fallback=bool(b2["fallback"][i]))
    return dict(global_mu=b2["global_mu"].tolist(), global_sigma=b2["global_sigma"].tolist(),
                groups=group_stats, terrain_mean=mean.tolist(), terrain_sd=sd.tolist(),
                feature_center=d["xy"][train].mean(0).tolist(), cmean=c[train].mean(0).tolist(),
                cscale=np.maximum(c[train].std(0), .1).tolist(), elements=ELEMENTS)


def transform(data, state):
    d = validate_data(data)
    n = len(d["ids"])
    mu = np.broadcast_to(state["global_mu"], (n, 12)).copy()
    sigma = np.broadcast_to(state["global_sigma"], (n, 12)).copy()
    support, fallback = np.zeros(n), np.ones(n)
    for group, stats in state["groups"].items():
        ix = d["groups"] == int(group)
        mu[ix], sigma[ix] = stats["mu"], stats["sigma"]
        support[ix], fallback[ix] = stats["n"], stats["fallback"]
    terrain = (d["terrain"] - state["terrain_mean"]) / state["terrain_sd"]
    c = np.column_stack([terrain, support / (support + 50.), fallback])
    centered = d["xy"] - state["feature_center"]
    columns = [(c - state["cmean"]) / state["cscale"], centered / 60000.]
    for wavelength in (30000., 60000., 120000.):
        phase = 2 * np.pi * centered / wavelength
        columns.extend([np.sin(phase), np.cos(phase)])
    return dict(r=(d["log10"] - mu) / sigma, F=np.column_stack(columns).astype(np.float32),
                mu=mu, sigma=sigma, xy=d["xy"], ids=d["ids"])
