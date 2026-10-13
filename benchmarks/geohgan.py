"""Array adapter for the Geo-Hgan author implementation, pinned to c2cce55.

All network classes are imported from the unchanged upstream source. Inputs are
N,C,32,32 spatial patches in [0,1], not C-dimensional point vectors. The paper's
32-pixel input replaces the repository's 64; C is supplied by the study data.
The default experiment trains each of the two stages for 100 epochs.

Usage::
    model = GeoHGAN(GeoHGANConfig(channels=12, seed=42, threads=1))
    model.fit(train_patches, progress_callback=print)
    result = model.predict(test_patches)
    model.save("model.pt")

No fit/predict API accepts evaluation labels. The only score is paper Eq. (9):
mean squared pixel error + mean squared discriminator-feature error, per patch.
The benchmark uses synthetic random data exclusively, never study evaluations.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn


UPSTREAM_COMMIT = "c2cce554b4ff8571c2f315ac8e11308f45443a2c"
UPSTREAM_URL = "https://github.com/cbn001012/Geo-Hgan-Unsupervised-anomaly-detection-in-geochemical-data-via-latent-space-learning"
UPSTREAM_FILES = {
    'nets/autoencoder.py':'f90fce71c51671ebb0924d1529ab7657c938f4b4599c04fbd5ad0a335e06c30d',
    'nets/latent_GAN.py':'08cbd85841dd9cec14176a7ffa0c78a4716d8d0e88223d4141711051c37e2f19',
    'nets/sample_GAN.py':'6949303d0e9fc88be478369ca320a6d8cd43aef4cb7855344230c9f70fad71af',
    'train_LSLM.py':'69c2230ac865b9b09977be26daf8419d0aa7c88760f497552472e5277a6ecf5e',
    'GAN_guided_train_encoder.py':'cc7a11e939c8b300933235447290a0dcbcfe81d841867faba42ae599067711f0',
    'test_anomaly_detection.py':'84ee76f85cbe80d2a9beb655df19ed2be9e78d16f56b03edf9624a38d8516f22',
}


@dataclass(frozen=True)
class GeoHGANConfig:
    channels: int = 12
    image_size: int = 32
    latent_dim: int = 100
    batch_size: int = 64
    epochs_stage1: int = 100
    epochs_stage2: int = 100
    lr_stage1: float = 0.0002
    lr_stage2: float = 0.0001
    beta1: float = 0.5
    beta2: float = 0.999
    n_critic: int = 5
    lambda_gp: float = 10.0
    lambda_rec: float = 0.001
    kappa: float = 1.0
    seed: int = 42
    threads: int = 1
    device: str = "cpu"
    upstream_dir: str | None = None


def _source_dir(specified: str | None = None) -> Path:
    if specified:
        path = Path(specified).resolve()
    else:
        raise FileNotFoundError('Provide upstream_dir pointing to the Geo-Hgan author checkout at '+UPSTREAM_COMMIT)
    for name,expected in UPSTREAM_FILES.items():
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
        content=(path/name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError(f'Geo-Hgan source differs from required snapshot {UPSTREAM_COMMIT}: {name}')
    if (path/'.git').exists():
        actual=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
        if actual!=UPSTREAM_COMMIT:raise ValueError(f'Geo-Hgan checkout is {actual}; expected {UPSTREAM_COMMIT}')
    return path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _network_classes(path: Path):
    # Load only definitions. Upstream training scripts run training on import.
    ae = _load_module(path / "nets/autoencoder.py", "_geohgan_author_ae")
    latent = _load_module(path / "nets/latent_GAN.py", "_geohgan_author_latent")
    sample = _load_module(path / "nets/sample_GAN.py", "_geohgan_author_sample")
    return ae, latent, sample


def _digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _memory() -> dict:
    """Process RSS/peak RSS and installed RAM without optional dependencies."""
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        ok = get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        if ok:
            return {"rss_bytes": counters.WorkingSetSize,
                    "peak_rss_bytes": counters.PeakWorkingSetSize,
                    "private_commit_peak_bytes": counters.PeakPagefileUsage}
    return {"rss_bytes": None, "peak_rss_bytes": None}


@contextlib.contextmanager
def _freeze_parameters(module):
    # Keep train/eval mode and all BatchNorm/spectral-normalization updates.
    original = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, required in zip(module.parameters(), original):
            p.requires_grad_(required)


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    # Exactly the author's per-channel interpolation, N,C,1,1 (not N,1,1,1).
    alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates), create_graph=True,
        retain_graph=True, only_inputs=True,
    )[0].reshape(real_samples.shape[0], -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class GeoHGAN:
    """Geo-Hgan networks and author training semantics behind a pure-array API."""

    def __init__(self, config: GeoHGANConfig | None = None):
        self.config = config or GeoHGANConfig()
        c = self.config
        if c.channels < 1 or c.image_size < 16 or c.image_size % 16:
            raise ValueError("channels must be positive; image_size must be a multiple of 16")
        if min(c.epochs_stage1, c.epochs_stage2, c.n_critic, c.threads) < 1 or c.batch_size < 2:
            raise ValueError("Positive epochs, n_critic, threads and batch_size >= 2 are required")
        if c.kappa != 1.0:
            raise ValueError("This reproduction fixes Eq. (9) to pixel MSE + feature MSE (kappa=1)")
        self.source_dir = _source_dir(c.upstream_dir)
        self.device = torch.device(c.device)
        torch.set_num_threads(c.threads)
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(c.seed)
        ae, latent, sample = _network_classes(self.source_dir)
        # Construction order is the same as train_LSLM.py.
        self.latent_generator = latent.latent_Generator(c.latent_dim).to(self.device)
        self.latent_discriminator = latent.latent_Discriminator(c.latent_dim).to(self.device)
        self.autoencoder = ae.AE((c.channels, c.image_size, c.image_size), c.latent_dim).to(self.device)
        self.generator = sample.WGAN_Generator(c.image_size, c.latent_dim, c.channels).to(self.device)
        self.discriminator = sample.WGAN_Discriminator(c.channels, c.image_size).to(self.device)
        self.encoder = self.autoencoder.encoder
        self.history_: list[dict] = []
        self.metadata_: dict = self._metadata()
        self.fitted_ = False

    def _metadata(self):
        paths = ["nets/autoencoder.py", "nets/latent_GAN.py", "nets/sample_GAN.py",
                 "train_LSLM.py", "GAN_guided_train_encoder.py", "test_anomaly_detection.py"]
        return {
            "implementation": "unchanged author network classes; array training adapter",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_identity_check": "six source files match pinned snapshot; Git HEAD also checked when available",
            "upstream_dir": str(self.source_dir),
            "source_sha256": {p: hashlib.sha256((self.source_dir / p).read_bytes()).hexdigest() for p in paths},
            "config": asdict(self.config),
            "input_axes": "N,C,H,W",
            "input_range": [0.0, 1.0],
            "score": "Eq9 = mean_CHW((G(E(x))-x)^2) + mean_features((D.f(G(E(x)))-D.f(x))^2)",
            "evaluation_labels_accepted": False,
            "score_selection": "fixed before evaluation; no AnomFilter or best-AUC metric selection",
            "parameter_count": {name: sum(p.numel() for p in getattr(self, name).parameters())
                                for name in ("autoencoder", "latent_generator", "latent_discriminator", "generator", "discriminator", "encoder")},
            "paper_code_and_adapter_notes": [
                "Paper section 4.2 says 32x32 geochemical patches; repository defaults to 64x64. Adapter default follows paper 32x32.",
                "Repository default channels=39; adapter channels=12 (or actual input channel count) without changing hidden network widths.",
                "Author AE is a fully connected patch autoencoder, despite a convolutional-autoencoder comment. Sample GAN remains spatial 2D CNN with CBAM and residual attention.",
                "Latent discriminator/generator use author least-squares GAN losses, not paper Eq2 logarithmic losses.",
                "Sample discriminator uses author WGAN-GP loss, lambda_gp=10 and decoded-fake weight lambda_rec=1e-3.",
                "Author stage1 latent D updates five times per batch; sample D once per batch; sample G at batch index i%5==0, with i reset each epoch.",
                "Author stage2 lr=1e-4; stage2 features are each divided by their global batch maximum. Inference features are not normalized.",
                "Stage2 optimizes only the pretrained AE encoder; G and D are frozen in eval mode. Loss is pixel MSE + feature MSE, as in the author script.",
                "Author BatchNorm2d(channels,0.8) passes eps=0.8, not momentum; unchanged original classes preserve this.",
                "Sample G has Tanh output while the AE decoder has Sigmoid; original mismatch with [0,1] data is preserved, with no output clipping or remapping.",
                "The array adapter removes unused gradient accumulation into networks with no optimizer step; all author forward passes and train-mode BN/spectral-normalization buffer updates are preserved.",
                "Author stage2 preview forwards E(fake) at batches_done%400==0 are retained to preserve encoder BN updates, but TIFF files are not written.",
                "Stage1 shuffles and drops incomplete batches. Stage2 is sequential, keeps incomplete batches, and moves one item from the penultimate batch to a singleton last batch if needed to satisfy BatchNorm.",
                "Center reconstruction is pixel [H//2,W//2], i.e. [16,16] for 32x32; complete reconstruction can be requested.",
                "No preprocessing is fitted here: caller must fit normalization/imputation/spatial preparation on the designated training partition and provide finite [0,1] patches.",
            ],
        }

    def _validate(self, patches, *, training=False):
        array = np.asanyarray(patches)
        expected = (self.config.channels, self.config.image_size, self.config.image_size)
        if array.ndim != 4 or tuple(array.shape[1:]) != expected:
            raise ValueError(f"Expected (N,{expected}), got {array.shape}")
        if training and len(array) < self.config.batch_size:
            raise ValueError("Stage1 drop_last=True needs at least one full batch")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("Patches must be numeric NumPy arrays")
        # Chunk validation keeps memmap-backed real-data inputs bounded in RAM.
        for start in range(0, len(array), 256):
            part = array[start:start + 256]
            if not np.isfinite(part).all() or (part.size and (part.min() < -1e-6 or part.max() > 1 + 1e-6)):
                raise ValueError("Patches must contain finite values within [0,1]")
        return array

    def _tensor(self, array, indices):
        return torch.from_numpy(np.array(array[indices], dtype=np.float32, order="C", copy=True)).to(self.device)

    def _noise(self, n):
        return torch.randn(n, self.config.latent_dim, device=self.device)

    @staticmethod
    def _finite(losses):
        result = {name: float(value.detach()) for name, value in losses.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError(f"Nonfinite training loss: {result}")
        return result

    @staticmethod
    def _stage2_slices(n, batch_size):
        starts = list(range(0, n, batch_size))
        if len(starts) > 1 and n - starts[-1] == 1:
            starts[-1] -= 1
        return [slice(start, starts[i + 1] if i + 1 < len(starts) else n) for i, start in enumerate(starts)]

    def fit(self, train_patches, progress_callback: Callable[[dict], None] | None = None):
        x = self._validate(train_patches, training=True)
        if self.fitted_:
            raise RuntimeError("Use a new GeoHGAN instance for each independent fit")
        c = self.config
        self.metadata_["training_patch_count"] = len(x)
        self.metadata_["stage1_dropped_per_epoch"] = len(x) % c.batch_size
        self.metadata_["stage2_singleton_tail_adjusted"] = len(x) % c.batch_size == 1
        optimizers = {
            name: torch.optim.Adam(module.parameters(), lr=c.lr_stage1, betas=(c.beta1, c.beta2))
            for name, module in (("ae", self.autoencoder), ("lg", self.latent_generator),
                                 ("ld", self.latent_discriminator), ("g", self.generator), ("d", self.discriminator))
        }
        mse = nn.MSELoss()
        modules = {"ae": self.autoencoder, "lg": self.latent_generator, "ld": self.latent_discriminator,
                   "g": self.generator, "d": self.discriminator}
        initial = {name: _digest(module) for name, module in modules.items()}
        for module in modules.values():
            module.train()
        shuffle_rng = torch.Generator().manual_seed(c.seed)
        stage1_start = time.perf_counter()
        batch_timings = {"with_sample_G_update": [], "without_sample_G_update": []}
        update_counts = {name: 0 for name in modules}
        for epoch in range(c.epochs_stage1):
            epoch_start = time.perf_counter()
            order = torch.randperm(len(x), generator=shuffle_rng).numpy()
            totals = {}
            n_batches = len(x) // c.batch_size
            for i in range(n_batches):
                batch_start = time.perf_counter()
                real = self._tensor(x, order[i * c.batch_size:(i + 1) * c.batch_size])
                # AE reconstruction update.
                optimizers["ae"].zero_grad(set_to_none=True)
                _, reconstructed = self.autoencoder(real)
                ae_loss = mse(reconstructed, real)
                ae_loss.backward()
                optimizers["ae"].step()
                update_counts["ae"] += 1

                # Latent LSGAN critic: preserve repeated AE and lG BN forwards.
                for _ in range(c.n_critic):
                    optimizers["ld"].zero_grad(set_to_none=True)
                    with torch.no_grad():
                        encoder_z, _ = self.autoencoder(real)
                        latent_z = self.latent_generator(self._noise(len(real)))
                    real_z_validity = self.latent_discriminator(encoder_z)
                    fake_z_validity = self.latent_discriminator(latent_z)
                    latent_d_loss = 0.5 * (real_z_validity - 1).square().mean() + 0.5 * fake_z_validity.square().mean()
                    latent_d_loss.backward()
                    optimizers["ld"].step()
                    update_counts["ld"] += 1
                optimizers["lg"].zero_grad(set_to_none=True)
                with _freeze_parameters(self.latent_discriminator):
                    latent_z = self.latent_generator(self._noise(len(real)))
                    latent_g_loss = 0.5 * (self.latent_discriminator(latent_z) - 1).square().mean()
                    latent_g_loss.backward()
                optimizers["lg"].step()
                update_counts["lg"] += 1

                # Sample WGAN-GP critic, including decoded AE samples.
                optimizers["d"].zero_grad(set_to_none=True)
                random_noise = self._noise(len(real))
                with torch.no_grad():
                    _, decoded_fake = self.autoencoder(real)
                decoded_validity = self.discriminator(decoded_fake)
                with torch.no_grad():
                    latent_z = self.latent_generator(random_noise)
                    fake = self.generator(latent_z)
                real_validity = self.discriminator(real)
                fake_validity = self.discriminator(fake)
                gp = compute_gradient_penalty(self.discriminator, real.detach(), fake.detach())
                sample_d_loss = -real_validity.mean() + fake_validity.mean() + c.lambda_rec * decoded_validity.mean() + c.lambda_gp * gp
                sample_d_loss.backward()
                optimizers["d"].step()
                update_counts["d"] += 1
                losses = {"ae_mse": ae_loss, "latent_D": latent_d_loss, "latent_G": latent_g_loss,
                          "sample_D": sample_d_loss, "gradient_penalty": gp}
                if i % c.n_critic == 0:
                    optimizers["g"].zero_grad(set_to_none=True)
                    with _freeze_parameters(self.discriminator):
                        fake = self.generator(latent_z)
                        sample_g_loss = -self.discriminator(fake).mean()
                        sample_g_loss.backward()
                    optimizers["g"].step()
                    update_counts["g"] += 1
                    losses["sample_G"] = sample_g_loss
                values = self._finite(losses)
                for key, value in values.items():
                    totals.setdefault(key, []).append(value)
                timing_key = "with_sample_G_update" if i % c.n_critic == 0 else "without_sample_G_update"
                batch_timings[timing_key].append(time.perf_counter() - batch_start)
            event = {"stage": 1, "epoch": epoch + 1, "epochs": c.epochs_stage1, "batches": n_batches,
                     "seconds": time.perf_counter() - epoch_start,
                     "loss": {key: float(np.mean(values)) for key, values in totals.items()}, **_memory()}
            self.history_.append(event)
            if progress_callback:
                progress_callback(event)
        self.metadata_["stage1_seconds"] = time.perf_counter() - stage1_start
        self.metadata_["stage1_update_counts"] = update_counts
        self.metadata_["stage1_batch_timing_seconds"] = {
            key: {"n": len(values), "mean": float(np.mean(values)) if values else None}
            for key, values in batch_timings.items()}
        self.metadata_["qa_stage1_all_networks_updated"] = {name: initial[name] != _digest(module) for name, module in modules.items()}
        # Release stage1 optimizer states before starting encoder-only training.
        for module in modules.values():
            module.zero_grad(set_to_none=True)
        del optimizers
        gc.collect()

        # GAN-guided encoder training. Only E is optimized; pretrained G/D eval.
        self.generator.eval()
        self.discriminator.eval()
        self.encoder.train()
        frozen_before = {"g": _digest(self.generator), "d": _digest(self.discriminator)}
        encoder_before = _digest(self.encoder)
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=c.lr_stage2, betas=(c.beta1, c.beta2))
        batches_done = 0
        preview_forwards = 0
        slices = self._stage2_slices(len(x), c.batch_size)
        stage2_start = time.perf_counter()
        with _freeze_parameters(self.generator), _freeze_parameters(self.discriminator):
            for epoch in range(c.epochs_stage2):
                epoch_start = time.perf_counter()
                totals = {"pixel_mse": [], "normalized_feature_mse": [], "encoder": []}
                for i, indices in enumerate(slices):
                    real = self._tensor(x, indices)
                    encoder_optimizer.zero_grad(set_to_none=True)
                    z = self.encoder(real)
                    fake = self.generator(z)
                    with torch.no_grad():
                        real_features = self.discriminator.forward_features(real)
                    fake_features = self.discriminator.forward_features(fake)
                    real_max, fake_max = real_features.max(), fake_features.max()
                    if real_max.item() == 0 or fake_max.item() == 0:
                        raise FloatingPointError("Author stage2 feature/max normalization encountered zero maximum")
                    real_features = real_features / real_max
                    fake_features = fake_features / fake_max
                    pixel_loss = mse(fake, real)
                    feature_loss = mse(fake_features, real_features)
                    encoder_loss = pixel_loss + c.kappa * feature_loss
                    encoder_loss.backward()
                    encoder_optimizer.step()
                    values = self._finite({"pixel_mse": pixel_loss, "normalized_feature_mse": feature_loss, "encoder": encoder_loss})
                    for key, value in values.items():
                        totals[key].append(value)
                    if i % c.n_critic == 0:
                        if batches_done % 400 == 0:
                            # Upstream preview updates E's BN running statistics.
                            with torch.no_grad():
                                self.generator(self.encoder(fake.detach()))
                            preview_forwards += 1
                        batches_done += c.n_critic
                event = {"stage": 2, "epoch": epoch + 1, "epochs": c.epochs_stage2, "batches": len(slices),
                         "seconds": time.perf_counter() - epoch_start,
                         "loss": {key: float(np.mean(values)) for key, values in totals.items()}, **_memory()}
                self.history_.append(event)
                if progress_callback:
                    progress_callback(event)
        self.metadata_["stage2_seconds"] = time.perf_counter() - stage2_start
        self.metadata_["stage2_encoder_steps"] = len(slices) * c.epochs_stage2
        self.metadata_["stage2_preview_encoder_forwards"] = preview_forwards
        self.metadata_["qa_stage2_G_D_unchanged"] = {"g": frozen_before["g"] == _digest(self.generator), "d": frozen_before["d"] == _digest(self.discriminator)}
        self.metadata_["qa_stage2_encoder_updated"] = encoder_before != _digest(self.encoder)
        self.metadata_["encoder_bn_num_batches_tracked"] = {key: int(value) for key, value in self.encoder.state_dict().items() if key.endswith("num_batches_tracked")}
        self.fitted_ = True
        self.encoder.eval()
        return self

    def predict(self, patches, *, return_reconstruction=False, batch_size=None):
        if not self.fitted_:
            raise RuntimeError("Call fit or load a fitted checkpoint first")
        x = self._validate(patches)
        size = batch_size or self.config.batch_size
        if size < 1:
            raise ValueError("Prediction batch_size must be positive")
        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()
        n, c, h, w = x.shape
        result = {"score": np.empty(n, np.float32), "pixel_mse": np.empty(n, np.float32),
                  "feature_mse": np.empty(n, np.float32), "reconstruction_center": np.empty((n, c), np.float32)}
        if return_reconstruction:
            result["reconstruction"] = np.empty(x.shape, np.float32)
        with torch.no_grad():
            for start in range(0, n, size):
                indices = slice(start, min(n, start + size))
                real = self._tensor(x, indices)
                fake = self.generator(self.encoder(real))
                real_features = self.discriminator.forward_features(real)
                fake_features = self.discriminator.forward_features(fake)
                pixel = (fake - real).square().flatten(1).mean(1)
                feature = (fake_features - real_features).square().mean(1)
                result["pixel_mse"][indices] = pixel.cpu().numpy()
                result["feature_mse"][indices] = feature.cpu().numpy()
                result["score"][indices] = (pixel + feature).cpu().numpy()
                result["reconstruction_center"][indices] = fake[:, :, h // 2, w // 2].cpu().numpy()
                if return_reconstruction:
                    result["reconstruction"][indices] = fake.cpu().numpy()
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite reconstruction or anomaly score")
        return result

    def save(self, path):
        if not self.fitted_:
            raise RuntimeError("Cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "metadata": self.metadata_, "history": self.history_,
                    "encoder": self.encoder.state_dict(), "generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict()}, path)

    @classmethod
    def load(cls, path, *, device="cpu", threads=1, upstream_dir=None):
        state = torch.load(path, map_location=device, weights_only=True)
        config = dict(state["config"], device=device, threads=threads)
        if upstream_dir is not None:config['upstream_dir']=str(upstream_dir)
        model = cls(GeoHGANConfig(**config))
        for name in ("encoder", "generator", "discriminator"):
            getattr(model, name).load_state_dict(state[name])
        model.metadata_.update({k:v for k,v in state['metadata'].items()
                                if k not in ('upstream_dir','upstream_commit','source_sha256','config','upstream_identity_check')})
        model.history_ = state["history"]
        model.fitted_ = True
        return model

