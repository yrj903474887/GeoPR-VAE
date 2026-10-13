"""Conditional-input-matched AE/VAE and a target-masked variational reference.

This module performs no data/file access at import and provides no training loop.
The caller owns the split, preprocessing, optimizer, training schedule and truth-free
training arrays.  All twelve chemistry variables are in the caller's G-fitted
standardized space.  ``reference`` denotes a reconstruction/reference estimate;
identifying this estimate as uncontaminated background is an evaluation question.

RAMP uses a masked variational objective, not an exact full-data log likelihood.
Its Student-t scale is exp(log_scale), not an observation standard deviation.
Posterior Monte Carlo reference predictions average decoder locations rather than
plugging a posterior mean into a nonlinear decoder.  RAMP leaves out the current
target from every chemistry path, including its deterministic MC random numbers.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_CONFIG = {
    "width": 64,
    "latent_dim": 8,
    "beta": 1.0,
    "mask_count": 1,
    "student_nu": 5.0,
    "protection_weight": 0.1,
    "protection_kappa": 0.1,
    "prior_width": 32,
    "posterior_logvar_min": -6.0,
    "posterior_logvar_max": 6.0,
    "decoder_logvar_min": -4.0,
    "decoder_logvar_max": 4.0,
    "decoder_logscale_min": -4.0,
    "decoder_logscale_max": 4.0,
    "protection_supports": [2, 4],
    "protection_amplitudes": [0.25, 0.5],
}
METHODS = ("AE", "VAE", "RAMP_VAE")
CHEM_DIM = 12
CONDITION_DIM = 19
MC_NOISE_NAMESPACE = b"M33-paired-latent-noise-v1"


def _config(config: dict | None) -> dict:
    result = dict(DEFAULT_CONFIG)
    if config is not None:
        result.update(config)
    for name in ("width", "latent_dim", "prior_width", "mask_count"):
        if int(result[name]) != result[name]:
            raise ValueError(f"{name} must be an integer")
        result[name] = int(result[name])
    if result["width"] < 1 or result["prior_width"] < 1:
        raise ValueError("hidden widths must be positive")
    if not 1 <= result["latent_dim"] <= 32:
        raise ValueError("latent_dim must lie in [1,32]")
    if not 1 <= result["mask_count"] < CHEM_DIM:
        raise ValueError("mask_count must lie in [1,11]")
    for name in ("beta", "protection_weight", "protection_kappa"):
        result[name] = float(result[name])
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    result["student_nu"] = float(result["student_nu"])
    if not math.isfinite(result["student_nu"]) or result["student_nu"] <= 2:
        raise ValueError("student_nu must exceed two (finite predictive variance)")
    for prefix in ("posterior_logvar", "decoder_logvar", "decoder_logscale"):
        low, high = float(result[prefix + "_min"]), float(result[prefix + "_max"])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f"invalid {prefix} bounds")
        result[prefix + "_min"], result[prefix + "_max"] = low, high
    supports = [int(v) for v in result["protection_supports"]]
    if not supports or any(v < 1 or v > CHEM_DIM - result["mask_count"] for v in supports):
        raise ValueError("protection support must fit the visible channels")
    amplitudes = [float(v) for v in result["protection_amplitudes"]]
    if not amplitudes or any(not math.isfinite(v) or v <= 0 for v in amplitudes):
        raise ValueError("protection amplitudes must be finite and positive")
    result["protection_supports"] = supports
    result["protection_amplitudes"] = amplitudes
    return result


def _body(input_dim: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU())


def diagonal_gaussian_kl(q_mu: torch.Tensor, q_logvar: torch.Tensor,
                         p_mu: torch.Tensor | None = None,
                         p_logvar: torch.Tensor | None = None) -> torch.Tensor:
    """Per-row KL(q||p), summed over latent coordinates (not normalized)."""
    if p_mu is None:
        p_mu = torch.zeros_like(q_mu)
    if p_logvar is None:
        p_logvar = torch.zeros_like(q_logvar)
    return 0.5 * (p_logvar - q_logvar + torch.exp(q_logvar - p_logvar)
                  + (q_mu - p_mu).square() * torch.exp(-p_logvar) - 1).sum(-1)


def gaussian_nll(x: torch.Tensor, loc: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian negative log probability, including its constant."""
    return 0.5 * (math.log(2 * math.pi) + logvar
                  + (x - loc).square() * torch.exp(-logvar))


def student_t_nll(x: torch.Tensor, loc: torch.Tensor, log_scale: torch.Tensor,
                  nu: float = 5.0) -> torch.Tensor:
    """Elementwise Student-t NLL with scale=exp(log_scale)."""
    normalizer = (math.lgamma(nu / 2.0) - math.lgamma((nu + 1.0) / 2.0)
                  + 0.5 * math.log(nu * math.pi))
    standardized_square = ((x - loc) * torch.exp(-log_scale)).square()
    return normalizer + log_scale + 0.5 * (nu + 1) * torch.log1p(standardized_square / nu)


def _rand(shape, like: torch.Tensor, generator: torch.Generator, normal=False):
    # Generator is supplied by the caller; avoid modifying torch's global RNG.
    gen_device = torch.device(generator.device)
    fn = torch.randn if normal else torch.rand
    return fn(shape, dtype=like.dtype, device=gen_device, generator=generator).to(like.device)


def _id_key(value: Any) -> bytes:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("MC sample IDs must be finite")
        return b"float:" + value.hex().encode("ascii")
    raise TypeError("MC sample IDs must be strings, integers, finite floats or bytes")


class ConditionalReferenceModel(nn.Module):
    """All networks are conditional-input-matched; RAMP additionally masks x."""

    def __init__(self, method: str, config: dict, seed: int):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.config, self.seed = method, _config(config), int(seed)
        w, z = self.config["width"], self.config["latent_dim"]
        ramp = method == "RAMP_VAE"
        self.encoder = _body(CONDITION_DIM + CHEM_DIM * (2 if ramp else 1), w)
        self.q_mu = nn.Linear(w, z)
        self.q_logvar = None if method == "AE" else nn.Linear(w, z)
        self.decoder = _body(CONDITION_DIM + z, w)
        self.decoder_loc = nn.Linear(w, CHEM_DIM)
        self.decoder_dispersion = None if method == "AE" else nn.Linear(w, CHEM_DIM)
        if ramp:
            self.prior_body = nn.Sequential(nn.Linear(CONDITION_DIM, self.config["prior_width"]), nn.SiLU())
            self.prior_output = nn.Linear(self.config["prior_width"], 2 * z)
        else:
            self.prior_body = None
            self.prior_output = None
        self._noise_cache = OrderedDict()

    def _inputs(self, x, c, allow_empty=False):
        parameter = next(self.parameters())
        x = torch.as_tensor(x, dtype=parameter.dtype, device=parameter.device)
        c = torch.as_tensor(c, dtype=parameter.dtype, device=parameter.device)
        if x.ndim != 2 or x.shape[1] != CHEM_DIM:
            raise ValueError("x must have shape [N,12]")
        if c.ndim != 2 or c.shape != (len(x), CONDITION_DIM):
            raise ValueError("c must have shape [N,19] matching x")
        if not allow_empty and len(x) == 0:
            raise ValueError("training input cannot be empty")
        if not torch.isfinite(x).all() or not torch.isfinite(c).all():
            raise ValueError("model inputs must be finite")
        return x, c

    def posterior(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor | None = None):
        if self.method == "RAMP_VAE":
            if mask is None or mask.shape != x.shape:
                raise ValueError("RAMP posterior requires a [N,12] target mask (1=hidden)")
            inputs = torch.cat((x * (1 - mask), mask, c), dim=-1)
        else:
            inputs = torch.cat((x, c), dim=-1)
        h = self.encoder(inputs)
        mu = self.q_mu(h)
        if self.q_logvar is None:
            return mu, None
        lv = self.q_logvar(h).clamp(self.config["posterior_logvar_min"],
                                   self.config["posterior_logvar_max"])
        return mu, lv

    def prior(self, c: torch.Tensor):
        if self.method != "RAMP_VAE":
            shape = (len(c), self.config["latent_dim"])
            return c.new_zeros(shape), c.new_zeros(shape)
        mu, lv = self.prior_output(self.prior_body(c)).chunk(2, dim=-1)
        return mu, lv.clamp(self.config["posterior_logvar_min"],
                            self.config["posterior_logvar_max"])

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        h = self.decoder(torch.cat((z, c), dim=-1))
        loc = self.decoder_loc(h)
        if self.decoder_dispersion is None:
            return loc, None
        prefix = "decoder_logscale" if self.method == "RAMP_VAE" else "decoder_logvar"
        dispersion = self.decoder_dispersion(h).clamp(self.config[prefix + "_min"],
                                                       self.config[prefix + "_max"])
        return loc, dispersion

    def sample_mask(self, x: torch.Tensor, generator: torch.Generator):
        count = self.config["mask_count"]
        indices = _rand(x.shape, x, generator).topk(count, dim=-1).indices
        return torch.zeros_like(x).scatter_(1, indices, 1.0)

    def sample_protection_delta(self, x: torch.Tensor, mask: torch.Tensor,
                                generator: torch.Generator):
        """Positive visible-only sparse perturbation with a fixed L2 amplitude.

        The default draws support 2 or 4 and total L2 amplitude 0.25 or 0.5,
        uniformly. These are standardized-space training interventions, not labels.
        """
        supports = self.config["protection_supports"]
        amplitudes = self.config["protection_amplitudes"]
        choice = _rand((len(x), 2), x, generator)
        ns = torch.as_tensor(supports, device=x.device)[(choice[:, 0] * len(supports)).long()]
        amp = torch.as_tensor(amplitudes, dtype=x.dtype, device=x.device)[
            (choice[:, 1] * len(amplitudes)).long()]
        random_keys = _rand(x.shape, x, generator).masked_fill(mask.bool(), -1.0)
        order = random_keys.argsort(dim=1, descending=True)
        active_rank = torch.arange(CHEM_DIM, device=x.device)[None, :] < ns[:, None]
        support = torch.zeros_like(x).scatter_(1, order, active_rank.to(x.dtype))
        positive = (0.5 + _rand(x.shape, x, generator)) * support * (1 - mask)
        return positive / positive.norm(dim=1, keepdim=True).clamp_min(1e-12) * amp[:, None]

    def protection_terms(self, x, c, mask, eps, delta, base_loc=None):
        """Paired latent-epsilon intervention; own-target delta is exactly zero."""
        if torch.any(delta < 0) or torch.any(delta * mask != 0):
            raise ValueError("protection delta must be nonnegative and zero on targets")
        if base_loc is None:
            mu, lv = self.posterior(x, c, mask)
            base_loc, _ = self.decode(mu + torch.exp(0.5 * lv) * eps, c)
        pert_mu, pert_lv = self.posterior(x + delta, c, mask)
        pert_loc, _ = self.decode(pert_mu + torch.exp(0.5 * pert_lv) * eps, c)
        changes = (pert_loc - base_loc).abs()
        budget = self.config["protection_kappa"] * delta.norm(dim=1, keepdim=True)
        excess = torch.relu(changes - budget)
        count = mask.sum(dim=1).clamp_min(1)
        loss = ((excess.square() * mask).sum(dim=1) / count).mean()
        active = (((changes > budget).to(x.dtype) * mask).sum(dim=1) / count).mean()
        mean_change = ((changes * mask).sum(dim=1) / count).mean()
        return loss, active, mean_change

    def objective(self, x, c, generator: torch.Generator, epoch_fraction: float = 1.0):
        """Return scalar tensor terms. No optimizer, external data, or implicit annealing.

        AE: mean MSE. VAE: mean coordinate Gaussian NLL + beta*KL/12.
        RAMP: per-row masked Student-t NLL (divided by actual mask count),
        + beta*KL/12 + protection_weight*protection_loss.
        ``kl`` is mean per-row latent-summed KL; ``kl_per_dim`` is KL/12,
        where dim refers to observed chemistry dimensions, not latent dimensions.
        """
        x, c = self._inputs(x, c)
        if generator is None:
            raise ValueError("an explicit torch.Generator is required")
        if not math.isfinite(float(epoch_fraction)) or not 0 <= float(epoch_fraction) <= 1:
            raise ValueError("epoch_fraction must lie in [0,1]; no annealing is applied")
        zero = x.new_zeros(())
        terms = {"kl": zero, "kl_per_dim": zero, "protection_loss": zero,
                 "weighted_protection_loss": zero, "protection_active_fraction": zero,
                 "protection_mean_abs_change": zero, "mask_fraction": zero}
        mask = self.sample_mask(x, generator) if self.method == "RAMP_VAE" else None
        mu, lv = self.posterior(x, c, mask)
        if self.method == "AE":
            loc, _ = self.decode(mu, c)
            reconstruction = (x - loc).square().mean()
            total = reconstruction
        else:
            eps = _rand(mu.shape, mu, generator, normal=True)
            z = mu + torch.exp(0.5 * lv) * eps
            loc, dispersion = self.decode(z, c)
            p_mu, p_lv = self.prior(c)
            kl = diagonal_gaussian_kl(mu, lv, p_mu, p_lv).mean()
            terms["kl"], terms["kl_per_dim"] = kl, kl / CHEM_DIM
            if self.method == "VAE":
                reconstruction = gaussian_nll(x, loc, dispersion).mean()
            else:
                nll = student_t_nll(x, loc, dispersion, self.config["student_nu"])
                reconstruction = ((nll * mask).sum(1) / mask.sum(1)).mean()
                terms["mask_fraction"] = mask.mean()
                # Draw regardless of lambda to keep paired mask/latent streams.
                delta = self.sample_protection_delta(x, mask, generator)
                prop, active, change = self.protection_terms(x, c, mask, eps, delta, loc)
                terms.update(protection_loss=prop, protection_active_fraction=active,
                             protection_mean_abs_change=change,
                             weighted_protection_loss=self.config["protection_weight"] * prop)
            total = reconstruction + self.config["beta"] * terms["kl_per_dim"] + terms["weighted_protection_loss"]
        terms["reconstruction_loss"] = reconstruction
        terms["total_loss"] = total
        terms["beta_effective"] = x.new_tensor(self.config["beta"] if self.method != "AE" else 0)
        return terms

    def _fixed_noise(self, keys, target: int, mc_samples: int, like: torch.Tensor):
        """ID-only paired normal draws. Chemistry, method and fitted seed are absent."""
        rows = []
        for key in keys:
            cache_key = (key, target, mc_samples)
            value = self._noise_cache.get(cache_key)
            if value is None:
                digest = hashlib.blake2b(MC_NOISE_NAMESPACE + b"|" + key + b"|"
                                         + str(target).encode("ascii"), digest_size=16).digest()
                rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, "little")))
                # Fixed 32 columns preserve prefixes across allowed latent widths.
                value = rng.standard_normal((mc_samples, 32)).astype(np.float32)
                self._noise_cache[cache_key] = value
                if len(self._noise_cache) > 8192:
                    self._noise_cache.popitem(last=False)
            else:
                self._noise_cache.move_to_end(cache_key)
            rows.append(value[:, :self.config["latent_dim"]])
        return torch.as_tensor(np.stack(rows), dtype=like.dtype, device=like.device)

    def _predict_batch(self, x, c, keys, mc_samples):
        n = len(x)
        if self.method == "AE":
            return self.decode(self.posterior(x, c)[0], c)[0]
        if self.method == "VAE":
            mu, lv = self.posterior(x, c)
            eps = self._fixed_noise(keys, -1, mc_samples, x)
            z = mu[:, None, :] + torch.exp(0.5 * lv[:, None, :]) * eps
            cc = c[:, None, :].expand(n, mc_samples, CONDITION_DIM)
            locations = self.decode(z.reshape(-1, self.config["latent_dim"]),
                                    cc.reshape(-1, CONDITION_DIM))[0]
            return locations.reshape(n, mc_samples, CHEM_DIM).mean(dim=1)
        # Vectorize all twelve leave-one-target-out views. The diagonal extraction
        # below selects only the decoder component whose own x is absent.
        mask = torch.eye(CHEM_DIM, device=x.device, dtype=x.dtype)[None].expand(n, -1, -1)
        xx = x[:, None, :].expand(n, CHEM_DIM, CHEM_DIM)
        cc = c[:, None, :].expand(n, CHEM_DIM, CONDITION_DIM)
        mu, lv = self.posterior(xx.reshape(-1, CHEM_DIM), cc.reshape(-1, CONDITION_DIM),
                                mask.reshape(-1, CHEM_DIM))
        mu, lv = mu.reshape(n, CHEM_DIM, -1), lv.reshape(n, CHEM_DIM, -1)
        eps = torch.stack([self._fixed_noise(keys, j, mc_samples, x) for j in range(CHEM_DIM)], dim=1)
        z = mu[:, :, None, :] + torch.exp(0.5 * lv[:, :, None, :]) * eps
        ccc = cc[:, :, None, :].expand(n, CHEM_DIM, mc_samples, CONDITION_DIM)
        locations = self.decode(z.reshape(-1, self.config["latent_dim"]),
                                ccc.reshape(-1, CONDITION_DIM))[0]
        locations = locations.reshape(n, CHEM_DIM, mc_samples, CHEM_DIM).mean(2)
        return locations.diagonal(dim1=1, dim2=2)

    def predict(self, x, c, ids=None, mc_samples: int = 16, batch_size: int = 128):
        """Return {'reference': float32[N,12], 'diagnostics': dict}.

        Supply stable sample IDs for paired before/after scoring. Repeated IDs
        receive identical noise. If IDs are omitted every row uses the same
        shared MC table, retaining batch/order and own-target invariance.
        """
        x, c = self._inputs(x, c, allow_empty=True)
        if int(mc_samples) != mc_samples or mc_samples < 1:
            raise ValueError("mc_samples must be a positive integer")
        if int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        mc_samples, batch_size = int(mc_samples), int(batch_size)
        if ids is None:
            keys = [b"shared-no-id"] * len(x)
        else:
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().numpy()
            ids = np.asarray(ids, dtype=object)
            if ids.shape != (len(x),):
                raise ValueError("ids must have shape [N]")
            keys = [_id_key(value) for value in ids]
        was_training = self.training
        self.eval()
        values = []
        try:
            with torch.no_grad():
                for start in range(0, len(x), batch_size):
                    end = start + batch_size
                    values.append(self._predict_batch(x[start:end], c[start:end], keys[start:end], mc_samples).cpu().numpy())
        finally:
            self.train(was_training)
        reference = np.concatenate(values, axis=0) if values else np.empty((0, CHEM_DIM), dtype=np.float32)
        return {"reference": reference,
                "diagnostics": {"method": self.method, "conditional_input_matched": True,
                                "mc_samples": 0 if self.method == "AE" else mc_samples,
                                "mc_key": "sample ID + target; fixed shared table if no ID",
                                "reference": "posterior MC mean decoder location" if self.method != "AE" else "deterministic reconstruction",
                                "target_mask_at_prediction": 1 if self.method == "RAMP_VAE" else 0,
                                "training_mask_count": self.config["mask_count"] if self.method == "RAMP_VAE" else 0,
                                "parameter_count": sum(p.numel() for p in self.parameters())}}

    def state(self):
        return {"format": "M33-conditional-reference-v1", "method": self.method,
                "config": dict(self.config), "seed": self.seed,
                "state_dict": {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}}

    def save_checkpoint(self, path):
        save_checkpoint(self, path)


def make_model(method: str, config: dict | None = None, seed: int = 0):
    """Deterministic CPU initialization without changing the caller's torch RNG."""
    method = str(method).upper()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return ConditionalReferenceModel(method, _config(config), seed)


def restore(state: dict):
    if state.get("format") != "M33-conditional-reference-v1":
        raise ValueError("unsupported checkpoint format")
    model = make_model(state["method"], state["config"], state["seed"])
    model.load_state_dict(state["state_dict"], strict=True)
    return model


def save_checkpoint(model: ConditionalReferenceModel, path):
    torch.save(model.state(), Path(path))


def load_checkpoint(path):
    return restore(torch.load(Path(path), map_location="cpu", weights_only=True))
