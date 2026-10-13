"""Independent convolutional comparison implementations.

Architectures follow VAE (Kingma and Welling), GANomaly (Akcay et al.),
Skip-GANomaly (Akcay et al.), and f-AnoGAN (Schlegl et al.). The shared
geochemical patch adapter and training settings are those used in this study.
These are independent implementations, not copies of unpublished baselines.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("VAE", "VAE_GAN", "VAE_WGAN", "VAE_WGANGP", "GANomaly", "Skip_GANomaly", "f_AnoGAN")


@dataclass(frozen=True)
class Config:
    image_size: int = 32
    base_channels: int = 32
    latent_dim: int = 100
    batch_size: int = 64
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    critic_steps: int = 5
    gp_weight: float = 10.0
    weight_clip: float = 0.01
    vae_beta: float = 1.0
    vae_gan_feature_weight: float = 1.0
    adversarial_weight: float = 1.0
    ganomaly_context_weight: float = 50.0
    skip_context_weight: float = 40.0
    skip_score_pixel_weight: float = 0.9
    fanogan_kappa: float = 1.0


def _name(method: str) -> str:
    aliases = {m.lower().replace("-", "_"): m for m in METHODS}
    aliases.update({"skipganomaly": "Skip_GANomaly", "fanogan": "f_AnoGAN"})
    key = method.lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    return aliases[key]


def _array(patches, channels=None, allow_empty=False):
    x = np.asarray(patches, dtype=np.float32)
    if x.ndim != 4 or x.shape[2:] != (32, 32) or x.shape[1] < 1:
        raise ValueError("Expected N,C,32,32 patches with C >= 1")
    if channels is not None and x.shape[1] != channels:
        raise ValueError("Prediction channels differ from training channels")
    if (not allow_empty and x.shape[0] < 1) or not np.isfinite(x).all():
        raise ValueError("Training data must be nonempty; all inputs must be finite")
    if x.size and (x.min() < 0 or x.max() > 1):
        raise ValueError("Inputs must already be scaled into [0,1] using training statistics")
    return np.ascontiguousarray(x)


def _block(cin, cout, batchnorm=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not batchnorm)]
    if batchnorm:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=False))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, channels, cfg, variational=False):
        super().__init__()
        b = cfg.base_channels
        self.blocks = nn.ModuleList([_block(channels, b, False), _block(b, b*2), _block(b*2, b*4)])
        self.mu = nn.Linear(b*4*4*4, cfg.latent_dim)
        self.logvar = nn.Linear(b*4*4*4, cfg.latent_dim) if variational else None

    def forward(self, x):
        levels = []
        for block in self.blocks:
            x = block(x)
            levels.append(x)
        flat = x.flatten(1)
        mu = self.mu(flat)
        # Numerical safety clamp is declared; no stochastic sampling at inference.
        lv = self.logvar(flat).clamp(-20, 10) if self.logvar is not None else None
        return mu, lv, levels


class Decoder(nn.Module):
    def __init__(self, channels, cfg, skip=False):
        super().__init__()
        self.skip = skip
        self.b = b = cfg.base_channels
        self.fc = nn.Linear(cfg.latent_dim, b*4*4*4)
        factor = 2 if skip else 1
        self.up1 = nn.Sequential(nn.ConvTranspose2d(b*4*factor, b*2, 4, 2, 1, bias=False), nn.BatchNorm2d(b*2), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(b*2*factor, b, 4, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU())
        self.up3 = nn.ConvTranspose2d(b*factor, channels, 4, 2, 1)

    def forward(self, z, levels=None):
        h = F.relu(self.fc(z)).reshape(-1, self.b*4, 4, 4)
        if self.skip:
            h = torch.cat((h, levels[2]), dim=1)
        h = self.up1(h)
        if self.skip:
            h = torch.cat((h, levels[1]), dim=1)
        h = self.up2(h)
        if self.skip:
            h = torch.cat((h, levels[0]), dim=1)
        # Common [0,1] scale is intentional; original image GANs commonly use tanh.
        return torch.sigmoid(self.up3(h))


class Discriminator(nn.Module):
    def __init__(self, channels, cfg, wasserstein=False):
        super().__init__()
        b = cfg.base_channels
        # No cross-sample normalization in Wasserstein critics / gradient penalty.
        self.features = nn.Sequential(_block(channels, b, False), _block(b, b*2, not wasserstein), _block(b*2, b*4, not wasserstein))
        self.head = nn.Linear(b*4*4*4, 1)

    def forward(self, x):
        f = self.features(x).flatten(1)
        return self.head(f).flatten(), f


def _init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1, 0.02)
        nn.init.zeros_(m.bias)


def _active(module, active):
    module.train(active)
    for parameter in module.parameters():
        parameter.requires_grad_(active)


def _sample(mu, logvar):
    return mu + torch.randn_like(mu) * torch.exp(0.5*logvar)


def _kl(mu, logvar, denominator):
    return -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean() / denominator


def _mse_samples(a, b):
    return (a-b).square().flatten(1).mean(1)


class ComparisonModel(nn.Module):
    def __init__(self, channels, method, config=None):
        super().__init__()
        self.method = _name(method)
        self.channels = int(channels)
        self.config = config or Config()
        cfg = self.config
        self.encoder = Encoder(channels, cfg, self.method.startswith("VAE"))
        self.decoder = Decoder(channels, cfg, self.method == "Skip_GANomaly")
        if self.method != "VAE":
            self.discriminator = Discriminator(channels, cfg, self.method in ("VAE_WGAN", "VAE_WGANGP", "f_AnoGAN"))
        if self.method == "GANomaly":
            self.encoder_out = Encoder(channels, cfg)
        self.apply(_init)
        self.history = []
        self.qa = {"finite_gradients": True, "optimizer_steps": {}, "max_gradient_norm": {}}
        self.metadata = {"implementation": "independent_standard_reimplementation_not_GeoHgan_authors_baselines", "method": self.method, "config": asdict(cfg), "score_selection": "fixed_before_training_no_labels", "output_scale": "unscaled_native_score", "input_scale": "[0,1]", "device": "cpu"}

    def _reconstruct(self, x, stochastic=False):
        mu, lv, levels = self.encoder(x)
        z = _sample(mu, lv) if stochastic and lv is not None else mu
        return self.decoder(z, levels), mu, lv

    def _step(self, name, loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{self.method}/{name}: nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Check every gradient before update, without silently clipping instability.
        norm2 = 0.0
        found = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    found = True
                    if not torch.isfinite(p.grad).all():
                        self.qa["finite_gradients"] = False
                        raise FloatingPointError(f"{self.method}/{name}: nonfinite gradient")
                    norm2 += float(p.grad.detach().square().sum())
        if not found:
            raise RuntimeError(f"No gradients for {name}")
        optimizer.step()
        self.qa["optimizer_steps"][name] = self.qa["optimizer_steps"].get(name, 0) + 1
        self.qa["max_gradient_norm"][name] = max(self.qa["max_gradient_norm"].get(name, 0), math.sqrt(norm2))
        return float(loss.detach())

    def _gp(self, real, fake):
        alpha = torch.rand(len(real), 1, 1, 1)
        interpolated = (alpha*real + (1-alpha)*fake.detach()).requires_grad_(True)
        output = self.discriminator(interpolated)[0]
        gradients = torch.autograd.grad(output.sum(), interpolated, create_graph=True)[0]
        return (gradients.flatten(1).norm(2, dim=1)-1).square().mean()

    def _bce_discriminator(self, x, fake, optimizer, prior=None):
        _active(self.discriminator, True)
        real_logit = self.discriminator(x)[0]
        fake_logit = self.discriminator(fake.detach())[0]
        real_loss = F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
        if prior is not None:
            prior_logit = self.discriminator(prior.detach())[0]
            fake_loss = 0.5*(fake_loss + F.binary_cross_entropy_with_logits(prior_logit, torch.zeros_like(prior_logit)))
        return self._step("D", 0.5*(real_loss+fake_loss), optimizer)

    def _train_vae(self, x, optimizers):
        cfg = self.config
        if self.method == "VAE":
            rec, mu, lv = self._reconstruct(x, True)
            loss = 0.5*F.mse_loss(rec, x) + cfg.vae_beta*_kl(mu, lv, x[0].numel())
            return {"E_G": self._step("E_G", loss, optimizers["E_G"])}
        if self.method == "VAE_GAN":
            # Larsen-style three separate objectives. No adversarial gradient to E.
            with torch.no_grad():
                rec = self._reconstruct(x, True)[0]
                prior = self.decoder(torch.randn(len(x), cfg.latent_dim))
            d = self._bce_discriminator(x, rec, optimizers["D"], prior)
            _active(self.discriminator, False)
            _active(self.decoder, False)
            _active(self.encoder, True)
            rec, mu, lv = self._reconstruct(x, True)
            real_f = self.discriminator(x)[1].detach()
            rec_f = self.discriminator(rec)[1]
            e_loss = 0.5*F.mse_loss(real_f, rec_f) + cfg.vae_beta*_kl(mu, lv, real_f.shape[1])
            e = self._step("E", e_loss, optimizers["E"])
            _active(self.encoder, False)
            _active(self.decoder, True)
            with torch.no_grad():
                mu, lv, _ = self.encoder(x)
                z = _sample(mu, lv)
            rec = self.decoder(z)
            prior = self.decoder(torch.randn(len(x), cfg.latent_dim))
            pred, features = self.discriminator(rec)
            pred_prior = self.discriminator(prior)[0]
            adv = 0.5*(F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred)) + F.binary_cross_entropy_with_logits(pred_prior, torch.ones_like(pred_prior)))
            g_loss = 0.5*cfg.vae_gan_feature_weight*F.mse_loss(features, real_f) + cfg.adversarial_weight*adv
            g = self._step("G", g_loss, optimizers["G"])
            _active(self.encoder, True)
            return {"D": d, "E": e, "G": g}
        # Explicit standard VAE+WGAN hybrids, not the identity-conditioned Kawai model
        # or the unpublished SAR architecture of Huang et al.
        ds = []
        _active(self.discriminator, True)
        for _ in range(cfg.critic_steps):
            with torch.no_grad():
                rec = self._reconstruct(x, True)[0]
                prior = self.decoder(torch.randn(len(x), cfg.latent_dim))
                # A balanced posterior/prior mixture, not an average of images.
                choose = (torch.rand(len(x), 1, 1, 1) < 0.5)
                fake = torch.where(choose, rec, prior)
            loss_d = self.discriminator(fake)[0].mean() - self.discriminator(x)[0].mean()
            if self.method == "VAE_WGANGP":
                loss_d = loss_d + cfg.gp_weight*self._gp(x, fake)
            ds.append(self._step("D", loss_d, optimizers["D"]))
            if self.method == "VAE_WGAN":
                with torch.no_grad():
                    for p in self.discriminator.parameters():
                        p.clamp_(-cfg.weight_clip, cfg.weight_clip)
        _active(self.discriminator, False)
        rec, mu, lv = self._reconstruct(x, True)
        prior = self.decoder(torch.randn(len(x), cfg.latent_dim))
        adv = -0.5*(self.discriminator(rec)[0].mean()+self.discriminator(prior)[0].mean())
        loss = 0.5*F.mse_loss(rec, x) + cfg.vae_beta*_kl(mu, lv, x[0].numel()) + cfg.adversarial_weight*adv
        return {"D": float(np.mean(ds)), "E_G": self._step("E_G", loss, optimizers["E_G"])}

    def _train_ganomaly(self, x, optimizers):
        with torch.no_grad():
            fake = self._reconstruct(x)[0]
        d = self._bce_discriminator(x, fake, optimizers["D"])
        _active(self.discriminator, False)
        rec, z, _ = self._reconstruct(x)
        real_f = self.discriminator(x)[1].detach()
        pred, fake_f = self.discriminator(rec)
        fm = F.mse_loss(fake_f, real_f)
        con = F.l1_loss(rec, x)
        if self.method == "GANomaly":
            z_out = self.encoder_out(rec)[0]
            loss = fm + self.config.ganomaly_context_weight*con + F.mse_loss(z_out, z)
        else:
            adv = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))
            loss = adv + self.config.skip_context_weight*con + fm
        return {"D": d, "E_G": self._step("E_G", loss, optimizers["E_G"])}

    def _train_fanogan(self, x, optimizers, stage):
        cfg = self.config
        if stage == "encoder":
            rec = self._reconstruct(x)[0]
            real_f = self.discriminator(x)[1].detach()
            loss = F.mse_loss(rec, x) + cfg.fanogan_kappa*F.mse_loss(self.discriminator(rec)[1], real_f)
            return {"E": self._step("E", loss, optimizers["E"])}
        _active(self.discriminator, True)
        ds = []
        for _ in range(cfg.critic_steps):
            with torch.no_grad():
                fake = self.decoder(torch.randn(len(x), cfg.latent_dim))
            loss = self.discriminator(fake)[0].mean() - self.discriminator(x)[0].mean() + cfg.gp_weight*self._gp(x, fake)
            ds.append(self._step("D", loss, optimizers["D"]))
        _active(self.discriminator, False)
        fake = self.decoder(torch.randn(len(x), cfg.latent_dim))
        g = self._step("G", -self.discriminator(fake)[0].mean(), optimizers["G"])
        return {"D": float(np.mean(ds)), "G": g}

    def fit(self, train_patches, seed, epochs=100, threads=1, progress: Callable | None = None):
        """Train final-epoch model. Model initialization should normally use fit() below."""
        data = _array(train_patches, self.channels)
        if not isinstance(epochs, int) or epochs < 1 or threads < 1:
            raise ValueError("epochs and threads must be positive integers")
        torch.set_num_threads(threads)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg = self.config
        def opt(parameters):
            return torch.optim.Adam(parameters, lr=cfg.lr, betas=cfg.betas)
        opts = {"E": opt(self.encoder.parameters()), "G": opt(self.decoder.parameters())}
        combined = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.method == "GANomaly":
            combined += list(self.encoder_out.parameters())
        opts["E_G"] = opt(combined)
        if hasattr(self, "discriminator"):
            opts["D"] = opt(self.discriminator.parameters())
        initial = {n: p.detach().clone() for n, p in self.named_parameters()}
        self.metadata.update({"seed": int(seed), "epochs_per_stage": epochs, "threads": threads, "n_train": len(data), "channels": self.channels, "parameter_count": sum(p.numel() for p in self.parameters())})
        stages = ("gan", "encoder") if self.method == "f_AnoGAN" else ("joint",)
        self.metadata["stages"] = list(stages)
        self.metadata["total_data_epochs"] = len(stages)*epochs
        start = time.perf_counter()
        tensor = torch.from_numpy(data)
        for stage in stages:
            self.train()
            if stage == "encoder":
                _active(self.decoder, False)
                _active(self.discriminator, False)
                _active(self.encoder, True)
            for epoch in range(epochs):
                epoch_start = time.perf_counter()
                totals, count = {}, 0
                for ids in torch.randperm(len(tensor)).split(cfg.batch_size):
                    x = tensor[ids]
                    if self.method.startswith("VAE"):
                        losses = self._train_vae(x, opts)
                    elif self.method == "f_AnoGAN":
                        losses = self._train_fanogan(x, opts, stage)
                    else:
                        losses = self._train_ganomaly(x, opts)
                    for k, value in losses.items():
                        totals[k] = totals.get(k, 0) + value*len(x)
                    count += len(x)
                row = {"stage": stage, "epoch": epoch+1, "seconds": time.perf_counter()-epoch_start, **{k: v/count for k, v in totals.items()}}
                self.history.append(row)
                if progress:
                    progress(dict(row))
        self.metadata["train_seconds"] = time.perf_counter()-start
        self.qa["parameter_update_l2_by_module"] = {}
        self.qa["finite_parameters"] = True
        for n, p in self.named_parameters():
            self.qa["finite_parameters"] &= bool(torch.isfinite(p).all())
            key = n.split(".")[0]
            self.qa["parameter_update_l2_by_module"][key] = self.qa["parameter_update_l2_by_module"].get(key, 0) + float((p.detach()-initial[n]).square().sum())
        self.qa["parameter_update_l2_by_module"] = {k: math.sqrt(v) for k, v in self.qa["parameter_update_l2_by_module"].items()}
        if not self.qa["finite_parameters"]:
            raise FloatingPointError("Nonfinite trained parameters")
        self.eval()
        return self

    @torch.no_grad()
    def predict(self, patches, batch_size=64):
        """Native scalar score, reconstruction and auxiliary signed center residual.

        The center for this even-sized patch is index (16,16). `center_residual`
        is x_center - reconstructed_center, never a replacement for `score`.
        No batchwise min-max scaling or label-conditioned score choice is used.
        """
        x_array = _array(patches, self.channels, allow_empty=True)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.eval()
        scores, recons, residuals = [], [], []
        for part in range(0, len(x_array), batch_size):
            x = torch.from_numpy(x_array[part:part+batch_size])
            rec, z, _ = self._reconstruct(x)
            pixel = _mse_samples(x, rec)
            if self.method == "GANomaly":
                score = _mse_samples(z, self.encoder_out(rec)[0])
            elif self.method in ("Skip_GANomaly", "f_AnoGAN"):
                feature = _mse_samples(self.discriminator(x)[1], self.discriminator(rec)[1])
                if self.method == "Skip_GANomaly":
                    w = self.config.skip_score_pixel_weight
                    score = w*pixel + (1-w)*feature
                else:
                    score = pixel + self.config.fanogan_kappa*feature
            else:
                score = pixel  # Declared common VAE-family scoring assumption.
            scores.append(score.numpy())
            recons.append(rec.numpy())
            residuals.append((x[:, :, 16, 16]-rec[:, :, 16, 16]).numpy())
        if not scores:
            return {"score": np.empty(0, np.float32), "reconstruction": np.empty_like(x_array), "center_residual": np.empty((0, self.channels), np.float32)}
        output = {"score": np.concatenate(scores), "reconstruction": np.concatenate(recons), "center_residual": np.concatenate(residuals)}
        if not all(np.isfinite(a).all() for a in output.values()):
            raise FloatingPointError("Nonfinite prediction")
        return output


def fit(train_patches, method, seed, epochs=100, threads=1, *, config=None, progress=None):
    """Reproducibly initialize and train one of the seven comparison models on CPU."""
    data = _array(train_patches)
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ComparisonModel(data.shape[1], method, config)
    return model.fit(data, seed=seed, epochs=epochs, threads=threads, progress=progress)
