"""Controlled M34 conditional AE/VAE extensions; M33 remains read-only.

The caller owns G-only preprocessing/training, candidate selection and scoring.
M33-compatible defaults exactly retain M33 architecture, initialization, objective
and training RNG consumption. Optional input noise corrupts the encoder input only;
the original observation is still the reconstruction target. No clean truth enters
the objective. RAMP uses a masked variational objective, not a full-data ELBO.

Predictive moments describe latent posterior and decoder observation variation at
fixed network parameters. Posterior-location variance is NOT a decomposition of
parameter epistemic uncertainty. Student-t decoder scale is not its standard
deviation. AE has no probabilistic decoder or native predictive observation scale.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


from . import reference_base as archived


METHODS = archived.METHODS
CHEM_DIM = archived.CHEM_DIM
CONDITION_DIM = archived.CONDITION_DIM
DEFAULT_CONFIG = dict(archived.DEFAULT_CONFIG,
                      protection_kind="positive", decoder_family="student_t",
                      prior_kind="conditional", denoising_augmentation_std=0.0)
diagonal_gaussian_kl = archived.diagonal_gaussian_kl
gaussian_nll = archived.gaussian_nll
student_t_nll = archived.student_t_nll


def normalize_config(method: str, config: dict | None = None):
    defaults = dict(DEFAULT_CONFIG)
    if method != "RAMP_VAE":
        defaults.update(protection_kind="none", decoder_family="gaussian", prior_kind="standard")
    if config:
        defaults.update(config)
    result = archived._config(defaults)
    if result["protection_kind"] not in ("none", "positive", "signed"):
        raise ValueError("protection_kind must be none, positive or signed")
    if result["decoder_family"] not in ("gaussian", "student_t"):
        raise ValueError("decoder_family must be gaussian or student_t")
    if result["prior_kind"] not in ("standard", "conditional"):
        raise ValueError("prior_kind must be standard or conditional")
    sigma = float(result["denoising_augmentation_std"])
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("denoising_augmentation_std must be finite and nonnegative")
    result["denoising_augmentation_std"] = sigma
    if method != "RAMP_VAE" and result["protection_kind"] != "none":
        raise ValueError("protection applies only to RAMP target-masked predictions")
    if method == "AE" and (result["prior_kind"] != "standard" or result["decoder_family"] != "gaussian"):
        raise ValueError("AE has no prior or probability decoder; keep their unused defaults")
    return result


class ConditionalReferenceModel(archived.ConditionalReferenceModel):
    def __init__(self, method: str, config: dict, seed: int):
        configuration = normalize_config(method, config)
        super().__init__(method, configuration, seed)
        # Parent construction preserves default M33 parameter initialization.
        if self.config["prior_kind"] == "standard":
            self.prior_body, self.prior_output = None, None
        elif self.prior_body is None:
            z = self.config["latent_dim"]
            self.prior_body = nn.Sequential(nn.Linear(CONDITION_DIM, self.config["prior_width"]), nn.SiLU())
            self.prior_output = nn.Linear(self.config["prior_width"], 2 * z)

    def prior(self, c):
        if self.config["prior_kind"] == "standard":
            shape = (len(c), self.config["latent_dim"])
            return c.new_zeros(shape), c.new_zeros(shape)
        mu, logvar = self.prior_output(self.prior_body(c)).chunk(2, dim=-1)
        return mu, logvar.clamp(self.config["posterior_logvar_min"], self.config["posterior_logvar_max"])

    def decode(self, z, c):
        h = self.decoder(torch.cat((z, c), dim=-1))
        loc = self.decoder_loc(h)
        if self.decoder_dispersion is None:
            return loc, None
        prefix = "decoder_logscale" if self.config["decoder_family"] == "student_t" else "decoder_logvar"
        dispersion = self.decoder_dispersion(h).clamp(self.config[prefix + "_min"], self.config[prefix + "_max"])
        return loc, dispersion

    def sample_protection_delta(self, x, mask, generator):
        """Matched support/energy for positive versus signed interventions.

        Signed perturbations use an auxiliary cloned RNG state and therefore do
        not change the main mask/latent/augmentation RNG sequence. On each visible
        support, approximately half the signs are positive and half negative.
        Absolute perturbations and total L2 energy exactly match positive draws.
        """
        sign_generator = None
        if self.config["protection_kind"] == "signed":
            sign_generator = torch.Generator(device=generator.device)
            sign_generator.set_state(generator.get_state().clone())
        delta = super().sample_protection_delta(x, mask, generator)
        if sign_generator is None:
            return delta
        keys = archived._rand(x.shape, x, sign_generator).masked_fill(delta == 0, -1)
        order = keys.argsort(dim=1, descending=True)
        support_size = (delta > 0).sum(dim=1)
        positive_rank = torch.arange(CHEM_DIM, device=x.device)[None, :] < ((support_size + 1) // 2)[:, None]
        signs_in_rank = positive_rank.to(x.dtype) * 2 - 1
        signs = torch.empty_like(x).scatter_(1, order, signs_in_rank)
        return delta * signs

    def protection_terms(self, x, c, mask, eps, delta, base_loc=None):
        kind = self.config["protection_kind"]
        if kind == "positive":
            return super().protection_terms(x, c, mask, eps, delta, base_loc)
        if kind == "none":
            zero = x.new_zeros(())
            return zero, zero, zero
        if torch.any(delta * mask != 0):
            raise ValueError("signed intervention must not expose a held-out target")
        if base_loc is None:
            mu, lv = self.posterior(x, c, mask)
            base_loc, _ = self.decode(mu + torch.exp(.5 * lv) * eps, c)
        pert_mu, pert_lv = self.posterior(x + delta, c, mask)
        pert_loc, _ = self.decode(pert_mu + torch.exp(.5 * pert_lv) * eps, c)
        change = (pert_loc - base_loc).abs()
        budget = self.config["protection_kappa"] * delta.norm(dim=1, keepdim=True)
        excess = torch.relu(change - budget)
        count = mask.sum(dim=1).clamp_min(1)
        loss = ((excess.square() * mask).sum(1) / count).mean()
        active = (((change > budget).to(x.dtype) * mask).sum(1) / count).mean()
        mean_change = ((change * mask).sum(1) / count).mean()
        return loss, active, mean_change

    def objective(self, x, c, generator: torch.Generator, epoch_fraction: float = 1.0):
        x, c = self._inputs(x, c)
        if generator is None:
            raise ValueError("an explicit torch.Generator is required")
        if not math.isfinite(float(epoch_fraction)) or not 0 <= float(epoch_fraction) <= 1:
            raise ValueError("epoch_fraction must lie in [0,1]; no implicit annealing")
        sigma = self.config["denoising_augmentation_std"]
        # No RNG call for the default zero corruption: M33 streams stay exact.
        encoder_x = x + sigma * archived._rand(x.shape, x, generator, normal=True) if sigma else x
        zero = x.new_zeros(())
        terms = {"kl": zero, "kl_per_dim": zero, "protection_loss": zero,
                 "weighted_protection_loss": zero, "protection_active_fraction": zero,
                 "protection_mean_abs_change": zero, "mask_fraction": zero}
        mask = self.sample_mask(x, generator) if self.method == "RAMP_VAE" else None
        mu, lv = self.posterior(encoder_x, c, mask)
        if self.method == "AE":
            loc, _ = self.decode(mu, c)
            reconstruction = (x - loc).square().mean()
            total = reconstruction
        else:
            eps = archived._rand(mu.shape, mu, generator, normal=True)
            loc, dispersion = self.decode(mu + torch.exp(.5 * lv) * eps, c)
            p_mu, p_lv = self.prior(c)
            kl = diagonal_gaussian_kl(mu, lv, p_mu, p_lv).mean()
            terms["kl"], terms["kl_per_dim"] = kl, kl / CHEM_DIM
            if self.config["decoder_family"] == "student_t":
                nll = student_t_nll(x, loc, dispersion, self.config["student_nu"])
            else:
                nll = gaussian_nll(x, loc, dispersion)
            if self.method == "VAE":
                reconstruction = nll.mean()
            else:
                reconstruction = ((nll * mask).sum(1) / mask.sum(1)).mean()
                terms["mask_fraction"] = mask.mean()
                # Even 'none' draws the same intervention to align subsequent RNG.
                delta = self.sample_protection_delta(encoder_x, mask, generator)
                prop, active, change = self.protection_terms(encoder_x, c, mask, eps, delta, loc)
                terms.update(protection_loss=prop, protection_active_fraction=active,
                             protection_mean_abs_change=change,
                             weighted_protection_loss=self.config["protection_weight"] * prop)
            total = reconstruction + self.config["beta"] * terms["kl_per_dim"] + terms["weighted_protection_loss"]
        terms["reconstruction_loss"] = reconstruction
        terms["total_loss"] = total
        terms["beta_effective"] = x.new_tensor(self.config["beta"] if self.method != "AE" else 0)
        terms["augmentation_std"] = x.new_tensor(sigma)
        return terms

    def _observation_variance(self, dispersion):
        if self.config["decoder_family"] == "gaussian":
            return torch.exp(dispersion)
        nu = self.config["student_nu"]
        return torch.exp(2 * dispersion) * (nu / (nu - 2))

    def _predict_batch_moments(self, x, c, keys, mc_samples):
        n = len(x)
        if self.method == "AE":
            reference = self.decode(self.posterior(x, c)[0], c)[0]
            return reference, torch.zeros_like(reference), None
        if self.method == "VAE":
            mu, lv = self.posterior(x, c)
            eps = self._fixed_noise(keys, -1, mc_samples, x)
            z = mu[:, None, :] + torch.exp(.5 * lv[:, None, :]) * eps
            cc = c[:, None, :].expand(n, mc_samples, CONDITION_DIM)
            locations, dispersion = self.decode(z.reshape(-1, self.config["latent_dim"]),
                                                cc.reshape(-1, CONDITION_DIM))
            locations = locations.reshape(n, mc_samples, CHEM_DIM)
            variances = self._observation_variance(dispersion).reshape(n, mc_samples, CHEM_DIM)
            return locations.mean(1), locations.var(1, correction=0), variances.mean(1)
        mask = torch.eye(CHEM_DIM, device=x.device, dtype=x.dtype)[None].expand(n, -1, -1)
        xx = x[:, None, :].expand(n, CHEM_DIM, CHEM_DIM)
        cc = c[:, None, :].expand(n, CHEM_DIM, CONDITION_DIM)
        mu, lv = self.posterior(xx.reshape(-1, CHEM_DIM), cc.reshape(-1, CONDITION_DIM),
                                mask.reshape(-1, CHEM_DIM))
        mu, lv = mu.reshape(n, CHEM_DIM, -1), lv.reshape(n, CHEM_DIM, -1)
        eps = torch.stack([self._fixed_noise(keys, j, mc_samples, x) for j in range(CHEM_DIM)], dim=1)
        z = mu[:, :, None, :] + torch.exp(.5 * lv[:, :, None, :]) * eps
        ccc = cc[:, :, None, :].expand(n, CHEM_DIM, mc_samples, CONDITION_DIM)
        locations, dispersion = self.decode(z.reshape(-1, self.config["latent_dim"]),
                                            ccc.reshape(-1, CONDITION_DIM))
        locations = locations.reshape(n, CHEM_DIM, mc_samples, CHEM_DIM)
        variances = self._observation_variance(dispersion).reshape(n, CHEM_DIM, mc_samples, CHEM_DIM)
        return (locations.mean(2).diagonal(dim1=1, dim2=2),
                locations.var(2, correction=0).diagonal(dim1=1, dim2=2),
                variances.mean(2).diagonal(dim1=1, dim2=2))

    def predict(self, x, c, ids=None, mc_samples=16, batch_size=128):
        """Posterior MC moments using ID+target draws, never chemistry-derived IDs.

        reference_sample_variance: variance of decoder locations across latent
        draws at fixed fitted weights (population-MC convention, not variance of
        the estimated mean). decoder_conditional_variance: mean observation
        variance across the same draws. predictive_variance: their sum.
        None for AE probability variances means unavailable, not zero uncertainty.
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
            keys = [archived._id_key(v) for v in ids]
        was_training = self.training
        self.eval()
        results = [[], [], []]
        try:
            with torch.no_grad():
                for start in range(0, len(x), batch_size):
                    end = start + batch_size
                    values = self._predict_batch_moments(x[start:end], c[start:end], keys[start:end], mc_samples)
                    for i, value in enumerate(values):
                        if value is not None:
                            results[i].append(value.cpu().numpy())
        finally:
            self.train(was_training)
        empty = np.empty((0, CHEM_DIM), dtype=np.float32)
        reference = np.concatenate(results[0]) if results[0] else empty.copy()
        sample_var = np.concatenate(results[1]) if results[1] else empty.copy()
        observation_var = (np.concatenate(results[2]) if results[2] else empty.copy()) if self.method != "AE" else None
        total_var = sample_var + observation_var if observation_var is not None else None
        return {"reference": reference, "reference_sample_variance": sample_var,
                "decoder_conditional_variance": observation_var, "predictive_variance": total_var,
                "diagnostics": {"method": self.method, "conditional_input_matched": True,
                                "mc_samples": 0 if self.method == "AE" else mc_samples,
                                "mc_key": "sample ID + target; shared table if no ID",
                                "variance_interpretation": "fixed-network latent-posterior and decoder observation components; not parameter epistemic uncertainty",
                                "mc_location_variance_divisor": "MC sample count (population moment)",
                                "decoder_family": self.config["decoder_family"] if self.method != "AE" else None,
                                "prior_kind": self.config["prior_kind"] if self.method != "AE" else None,
                                "target_mask_at_prediction": 1 if self.method == "RAMP_VAE" else 0,
                                "parameter_count": sum(p.numel() for p in self.parameters())}}

    def state(self):
        state = super().state()
        state["format"] = "M34-conditional-reference-v1"
        return state


def make_model(method: str, config: dict | None = None, seed: int = 0):
    method = str(method).upper()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return ConditionalReferenceModel(method, normalize_config(method, config), seed)


def restore(state: dict):
    if state.get("format") != "M34-conditional-reference-v1":
        raise ValueError("unsupported M34 checkpoint format")
    model = make_model(state["method"], state["config"], state["seed"])
    model.load_state_dict(state["state_dict"], strict=True)
    return model


def save_checkpoint(model, path):
    torch.save(model.state(), Path(path))


def load_checkpoint(path):
    return restore(torch.load(Path(path), map_location="cpu", weights_only=True))
