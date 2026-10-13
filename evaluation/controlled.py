"""Aggregation and base-realization uncertainty for controlled experiments.

Base realizations, not folds or intervention cases, are bootstrap units for
the controlled-recovery figures. Weak-injection figures instead display the
five fold means, each averaged over its interventions.
"""
import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 2026101207
BOOTSTRAP_REPLICATES = 10000


def bootstrap_mean_ci(values, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite realization means are required")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (replicates, len(values)))].mean(1)
    return np.quantile(means, [.025, .975])


def recovery_tables(cases):
    """Produce the three plotting tables from one row per controlled case.

    Required columns: method,buffer_m,base_seed,noise,amplitude,
    denoising_rmse,amplitude_gain_abs_error. Extra case dimensions (fold,
    covariance, seed) are averaged with equal row weight, as in the protocol.
    """
    keys = ["method", "buffer_m", "base_seed"]
    metrics = ["denoising_rmse", "amplitude_gain_abs_error"]
    required = keys + ["noise", "amplitude"] + metrics
    missing = set(required) - set(cases.columns)
    if missing:
        raise ValueError("Missing controlled-case columns: " + ", ".join(sorted(missing)))
    if cases[required].isnull().any().any() or not np.isfinite(cases[metrics].to_numpy(float)).all():
        raise ValueError("Incomplete or nonfinite controlled cases")
    return {"base_means": cases.groupby(keys, sort=True)[metrics].mean().reset_index(),
            "noise_base_means": cases.groupby(keys + ["noise"], sort=True)[metrics].mean().reset_index(),
            "amplitude_base_means": cases.groupby(keys + ["amplitude"], sort=True)[metrics].mean().reset_index()}


def weak_fold_table(cases):
    required = ["method", "buffer_km", "fold", "delta_auc", "delta_core_recall_top10"]
    missing = set(required) - set(cases.columns)
    if missing:
        raise ValueError("Missing weak-case columns: " + ", ".join(sorted(missing)))
    if cases[required].isnull().any().any():
        raise ValueError("Incomplete weak-injection cases")
    keys = required[:3]
    means = cases.groupby(keys)[required[3:]].mean()
    means["n_cases"] = cases.groupby(keys).size()
    return means.reset_index()

