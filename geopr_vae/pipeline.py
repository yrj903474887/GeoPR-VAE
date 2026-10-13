"""Portable training, independent calibration graphs, and full-field inference."""
from pathlib import Path
import numpy as np
import torch
from . import reference, recovery, scoring
from .training import train_residual_head
from .preprocessing import validate_data, fit_preprocessor, transform, regional_split

CONFIGS = {
    "AE": dict(width=96, latent_dim=4, denoising_augmentation_std=0.),
    "VAE": dict(width=64, latent_dim=4, beta=1., denoising_augmentation_std=0.),
    "GeoPR-VAE": dict(width=64, latent_dim=8, beta=1., mask_count=1,
                      prior_kind="conditional", decoder_family="gaussian", student_nu=5.,
                      protection_kind="positive", protection_weight=100., protection_kappa=.01,
                      denoising_augmentation_std=0.),
}
PARAMETER_COUNTS = {"AE": 25552, "VAE": 13984, "GeoPR-VAE": 16696}


def _indices(indices, n, name):
    a = np.asarray(indices)
    if a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0:
        raise ValueError(f"{name} must be nonempty integer indices")
    if len(np.unique(a)) != len(a) or np.any(a < 0) or np.any(a >= n):
        raise ValueError(f"Invalid or duplicate {name}")
    return a.astype(np.int64)


def fit(data, method="GeoPR-VAE", train=None, calibration=None, seed=20260929,
        teacher_epochs=600, head_epochs=300, head_config=None, head_seed=None):
    """Fit explicit training rows; head_seed defaults to the reference seed."""
    if method not in CONFIGS:
        raise ValueError(f"method must be one of {tuple(CONFIGS)}")
    if teacher_epochs < 1 or head_epochs < 1:
        raise ValueError("Epoch counts must be positive")
    recovery_seed=int(seed if head_seed is None else head_seed)
    d = validate_data(data)
    if train is None and calibration is None:
        train, calibration = regional_split(d["ids"], d["xy"])
    elif train is None or calibration is None:
        raise ValueError("Supply both train and calibration indices")
    it = _indices(train, len(d["ids"]), "train")
    ic = _indices(calibration, len(d["ids"]), "calibration")
    if len(it) < 24 or len(ic) < 4 or np.intersect1d(it, ic).size:
        raise ValueError("Need >=24 training and >=4 disjoint calibration rows")
    state = fit_preprocessor(d, it)
    a = transform(d, state)
    native = "RAMP_VAE" if method == "GeoPR-VAE" else method
    model = reference.make_model(native, CONFIGS[method], int(seed))
    if sum(p.numel() for p in model.parameters()) != PARAMETER_COUNTS[method]:
        raise RuntimeError("Unexpected parameter count")
    torch.set_num_threads(1)
    x, c = [torch.as_tensor(a[key][it], dtype=torch.float32) for key in ("r", "F")]
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 941)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    history = []
    model.train()
    for epoch in range(1, teacher_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = model.objective(x, c, generator, epoch / teacher_epochs)
        if not torch.isfinite(terms["total_loss"]):
            raise FloatingPointError("Nonfinite reference loss")
        terms["total_loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(epoch=epoch, gradient_norm=float(gradient),
                            **{k: float(v.detach()) for k, v in terms.items() if v.numel() == 1}))
    model.eval()
    pred = model.predict(a["r"], a["F"], ids=a["ids"], mc_samples=16, batch_size=128)
    residual = a["r"] - pred["reference"]
    head, head_history = None, []
    cal_carrier = residual[ic].copy()
    if method == "GeoPR-VAE":
        head, head_history = train_residual_head(a["r"][it], a["F"][it], a["xy"][it],
                                                pred["reference"][it], recovery_seed, head_config, head_epochs)
        # Recovery neighborhoods must not connect scale-fit and threshold pools.
        for indices in scoring.split_calibration(a["ids"][ic]):
            cal_carrier[indices] = head.predict(residual[ic][indices], a["F"][ic][indices],
                                                a["xy"][ic][indices])["correction"]
    bundle = dict(format="GeoPR-VAE-portable-v1", method=method, seed=int(seed),
                  head_seed=recovery_seed if head is not None else None,
                  reference=model.state(), head=None if head is None else head.state(),
                  preprocessing=state, calibration=dict(carrier=cal_carrier.tolist(),
                  xy=a["xy"][ic].tolist(), ids=a["ids"][ic].tolist()),
                  training=dict(train_ids=a["ids"][it].tolist(), calibration_ids=a["ids"][ic].tolist(),
                  teacher_epochs=teacher_epochs, head_epochs=head_epochs if head else 0,
                  seed=int(seed), head_seed=recovery_seed if head is not None else None,
                  reference_config=dict(model.config),
                  head_config=None if head is None else dict(head.config),
                  reference_optimizer=dict(name="AdamW", learning_rate=.0005, weight_decay=.001,
                                           gradient_clip=5., batch="full"),
                  teacher_parameter_count=PARAMETER_COUNTS[method],
                  head_parameter_count=0 if head is None else sum(p.numel() for p in head.parameters()),
                  training_truth_access=False))
    return bundle, dict(reference=history, recovery=head_history)


def score_carrier(carrier, xy, calibration, method):
    cr, cxy = np.asarray(calibration["carrier"]), np.asarray(calibration["xy"])
    norm = "negative_half" if method == "GeoPR-VAE" else "cal_mad"
    out = scoring.score_fields(carrier[None], cr, xy, cxy, calibration["ids"], norm, 10000.)
    diag, ci = out["diagnostics"], out["original_threshold_indices"]
    center, scale = np.asarray(diag["center_calfit"]), np.asarray(diag["scale_calfit"])
    signed = scoring.gaussian_weights(xy, 10000.) @ ((carrier - center) / scale / out["field_factor"][0])
    csigned = scoring.gaussian_weights(cxy[ci], 10000.) @ ((cr[ci] - center) / scale / out["cal_factor"])
    raw = np.column_stack([out["score"][0], signed])
    cal_raw = np.column_stack([out["cal_score"], csigned])
    maps = [scoring.fit_affine_map(cal_raw[:, j]) for j in range(13)]
    affine = np.column_stack([scoring.affine_map(raw[:, j], maps[j]) for j in range(13)])
    return dict(raw_scores=raw, affine_scores=affine, signed_channels=signed,
                threshold=np.asarray(out["threshold"]), above_threshold=raw[:, 0] > out["threshold"],
                calibration_raw_scores=cal_raw, field_factor=out["field_factor"][0]), dict(scoring=diag, affine_maps=maps)


def predict(bundle, data, batch_size=128):
    """Infer one complete query field; geographic batching changes the method."""
    if bundle.get("format") != "GeoPR-VAE-portable-v1":
        raise ValueError("Unsupported checkpoint format")
    d = validate_data(data)
    a = transform(d, bundle["preprocessing"])
    model = reference.restore(bundle["reference"])
    pred = model.predict(a["r"], a["F"], ids=a["ids"], mc_samples=16, batch_size=batch_size)
    ref = pred["reference"]
    residual = a["r"] - ref
    if bundle["head"] is not None:
        head = recovery.restore(bundle["head"])
        carrier = head.predict(residual, a["F"], a["xy"])["correction"]
        denoised = ref + carrier
    else:
        carrier = residual
        denoised = ref + .75 * residual + .25 * np.maximum(residual - .5, 0.)
    scores, details = score_carrier(carrier, a["xy"], bundle["calibration"], bundle["method"])
    arrays = dict(ids=a["ids"], xy=a["xy"], reference=ref, residual=residual, carrier=carrier,
                  recovered=denoised, reference_log10=a["mu"] + a["sigma"] * ref,
                  recovered_log10=a["mu"] + a["sigma"] * denoised, **scores)
    for key in ("reference_sample_variance", "decoder_conditional_variance", "predictive_variance"):
        if pred[key] is not None:
            arrays[key] = pred[key]
    return arrays, details


def save(bundle, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)
