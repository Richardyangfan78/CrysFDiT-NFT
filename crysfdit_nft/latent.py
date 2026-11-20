"""Latent flow matching: the denoising transformer, the Euclidean interpolant
and the SDE sampler over latent tokens.

Latent layout: ``z in R^{B x (1 + N) x d}`` -- token 0 is the *global* latent
(lattice / system-level information, Mofasa-style local + global split), the
remaining N tokens are per-atom latents.  ``latent_mask`` extends the atom
mask with an always-on global slot.  Source ``N(0, I)``; linear path
``z_t = (1 - t) z_0 + t z_1``; the network predicts the velocity ``z_1 - z_0``
and receives its own previous endpoint estimate as *self-conditioning*
(FlowMol3 / Mofasa), which couples the discrete and continuous information
carried by the latents across sampling steps.

With positive noise, every transition of the SDE sampler is Gaussian with a
known mean / std, so per-step log-probabilities are available. Zero-noise ODE
transitions do not have such densities and cannot supply policy gradients.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import sinusoidal_embedding
from .model.student import AdaLNBlock
from .schema import PaddedSystemBatch

if TYPE_CHECKING:               # model.autoencoder imports this module: no runtime import cycle
    from .model.autoencoder import LatentBatch

Tensor = torch.Tensor


def latent_mask(batch: PaddedSystemBatch, global_token: bool = True) -> Tensor:
    """(B, N) atom mask -> (B, 1 + N) latent-token mask (global slot always on)."""
    if not global_token:
        return batch.atom_mask
    ones = torch.ones(batch.batch_size, 1, dtype=torch.bool, device=batch.atom_mask.device)
    return torch.cat([ones, batch.atom_mask], dim=1)


def split_latent(z: Tensor, global_token: bool = True) -> tuple[Tensor | None, Tensor]:
    """-> (global (B, d) or None, atoms (B, N, d))."""
    if not global_token:
        return None, z
    return z[:, 0], z[:, 1:]


def pool_atoms(z: Tensor, batch: PaddedSystemBatch, global_token: bool = True) -> Tensor:
    """Masked mean of the per-atom latents -> (B, d) (ignores the global slot)."""
    _, atoms = split_latent(z, global_token)
    m = batch.atom_mask.unsqueeze(-1).to(z.dtype)
    return (atoms * m).sum(1) / m.sum(1).clamp(min=1.0)


# ------------------------------------------------------- latent normalisation
LATENT_STATS_PROTOCOL = "crysfdit_latent_stats_v1"
LATENT_STD_FLOOR = 1e-6          # every stored std is at least this


def _finite_numbers(values, count: int) -> bool:
    return (isinstance(values, list) and len(values) == count
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values))


def validate_latent_stats(stats: dict) -> dict:
    """Check a latent_stats.json payload (stored in the CrysFDiT-NFT DiT checkpoint) and return it unchanged.

    Required: ``protocol``, a positive integer ``latent_dim``, a boolean ``global_token``, ``atom.mean`` and
    ``atom.std`` with ``latent_dim`` finite numbers each, and ``global.{mean,std}`` exactly when
    ``global_token``; every std at least ``LATENT_STD_FLOOR``.
    """
    if not isinstance(stats, dict) or stats.get("protocol") != LATENT_STATS_PROTOCOL:
        raise ValueError(f"latent statistics must follow protocol {LATENT_STATS_PROTOCOL!r}")
    dim = stats.get("latent_dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError("latent statistics need a positive integer latent_dim")
    global_token = stats.get("global_token")
    if not isinstance(global_token, bool):
        raise TypeError("latent statistics need a boolean global_token")
    if not global_token and stats.get("global") is not None:
        raise ValueError("latent statistics without a global token must not carry 'global' statistics")
    for kind in (("atom", "global") if global_token else ("atom",)):
        entry = stats.get(kind)
        if not isinstance(entry, dict):
            raise TypeError(f"latent statistics lack the {kind!r} token entry")
        for name in ("mean", "std"):
            if not _finite_numbers(entry.get(name), dim):
                raise ValueError(f"latent statistics {kind}.{name} must hold {dim} finite numbers")
        if any(v < LATENT_STD_FLOOR for v in entry["std"]):
            raise ValueError(f"latent statistics {kind}.std must be >= {LATENT_STD_FLOOR}")
    return stats


class LatentNormalizer:
    """Per-dimension affine map between the autoencoder's latent space and the normalised one.

    ``z' = (z - mean_k) / std_k`` and ``logvar' = logvar - 2 log std_k`` with ``k`` = global for token 0
    (when ``global_token``) and atom for every other token; the inverse is ``z = z' std_k + mean_k``. The
    statistics are a validated latent_stats.json payload (:func:`validate_latent_stats`). Inputs are
    ``(B, 1 + N, d)`` (or ``(B, N, d)`` without a global token); any further leading batch dimensions or
    none at all (``(1 + N, d)``) are accepted and the output keeps the input's shape. The constants are
    held in float64, the arithmetic runs in float64 and the result takes the input's device and dtype.
    With a ``mask`` the padded slots stay exactly zero, as the autoencoder leaves them.
    """

    def __init__(self, stats: dict, global_token: bool):
        stats = validate_latent_stats(stats)
        if bool(stats["global_token"]) != bool(global_token):
            raise ValueError(f"latent statistics were computed with global_token={stats['global_token']}, "
                             f"the autoencoder has global_token={bool(global_token)}")
        self.global_token = bool(global_token)
        self.latent_dim = int(stats["latent_dim"])
        self.atom_mean = torch.tensor(stats["atom"]["mean"], dtype=torch.float64)
        self.atom_std = torch.tensor(stats["atom"]["std"], dtype=torch.float64)
        self.global_mean = torch.tensor(stats["global"]["mean"], dtype=torch.float64) if self.global_token else None
        self.global_std = torch.tensor(stats["global"]["std"], dtype=torch.float64) if self.global_token else None
        self._constants_cache: dict[tuple[torch.device, int, int], tuple[Tensor, Tensor]] = {}

    def constants(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """float64 ``(mean, std)`` for the token layout of ``z`` on its device, shaped to broadcast with ``z``.

        ``(T, d)`` for a 2-D input, ``(1, T, d)`` for ``(B, T, d)`` and ``(1, ..., 1, T, d)`` for further
        leading dimensions: an elementwise operation with ``z`` keeps ``z``'s shape.
        """
        if z.ndim < 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"expected latents with {self.latent_dim} dimensions, got shape {tuple(z.shape)}")
        tokens = int(z.shape[-2])
        key = (z.device, tokens, int(z.ndim))
        cached = self._constants_cache.get(key)
        if cached is None:
            mean = self.atom_mean.view(1, -1).expand(tokens, -1)
            std = self.atom_std.view(1, -1).expand(tokens, -1)
            if self.global_token:
                if tokens < 1:
                    raise ValueError("latents with a global token need at least one token")
                mean = torch.cat([self.global_mean.view(1, -1), mean[1:]], dim=0)
                std = torch.cat([self.global_std.view(1, -1), std[1:]], dim=0)
            shape = (1,) * (z.ndim - 2) + (tokens, self.latent_dim)
            cached = (mean.contiguous().view(shape).to(z.device), std.contiguous().view(shape).to(z.device))
            self._constants_cache[key] = cached
        return cached

    @staticmethod
    def _masked(out: Tensor, mask: Tensor | None) -> Tensor:
        return out if mask is None else out * mask.unsqueeze(-1).to(out.dtype)

    def normalize(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        mean, std = self.constants(z)
        return self._masked(((z.to(torch.float64) - mean) / std).to(z.dtype), mask)

    def denormalize(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        mean, std = self.constants(z)
        return self._masked((z.to(torch.float64) * std + mean).to(z.dtype), mask)

    def normalize_logvar(self, logvar: Tensor, mask: Tensor | None = None) -> Tensor:
        _, std = self.constants(logvar)
        return self._masked((logvar.to(torch.float64) - 2.0 * torch.log(std)).to(logvar.dtype), mask)

    def normalize_latent_batch(self, latent: LatentBatch) -> LatentBatch:
        """``z``, ``mean`` and ``logvar`` transformed; template, mask and target coordinates unchanged."""
        if bool(latent.global_token) != self.global_token:
            raise ValueError("the latent batch and the normaliser disagree on the global token")
        mask = latent.mask
        return replace(latent, z=self.normalize(latent.z, mask), mean=self.normalize(latent.mean, mask),
                       logvar=self.normalize_logvar(latent.logvar, mask))


class NormalizedAutoencoder(nn.Module):
    """A (frozen) all-atom autoencoder seen through a :class:`LatentNormalizer`.

    ``encode`` returns normalised latents; ``decode`` / ``decode_raw`` / ``decode_heads`` denormalise their
    input before the wrapped decoder sees it. Stage 2, stage 3, sampling and evaluation consume this
    wrapper exactly like the plain autoencoder (``config``, ``global_token``, ``latent_mask``,
    ``reconstruction_objective_sums``, ``num_parameters``; ``eval`` / ``train`` / ``requires_grad_`` /
    ``to`` act on the wrapped module). It is not a training objective: ``forward`` raises.
    """

    def __init__(self, ae: nn.Module, normalizer: LatentNormalizer):
        super().__init__()
        if int(ae.config.latent_dim) != normalizer.latent_dim:
            raise ValueError(f"latent statistics have latent_dim={normalizer.latent_dim}, "
                             f"the autoencoder latent_dim={ae.config.latent_dim}")
        if bool(ae.global_token) != normalizer.global_token:
            raise ValueError("latent statistics and the autoencoder disagree on the global token")
        self.ae = ae
        self.normalizer = normalizer

    @property
    def config(self):
        return self.ae.config

    @property
    def global_token(self) -> bool:
        return self.ae.global_token

    def latent_mask(self, batch: PaddedSystemBatch) -> Tensor:
        return self.ae.latent_mask(batch)

    def encode(self, batch: PaddedSystemBatch, sample: bool = True) -> LatentBatch:
        return self.normalizer.normalize_latent_batch(self.ae.encode(batch, sample=sample))

    def decode_heads(self, z: Tensor, template: PaddedSystemBatch) -> dict[str, Tensor | None]:
        return self.ae.decode_heads(self.normalizer.denormalize(z, self.ae.latent_mask(template)), template)

    def decode_raw(self, z: Tensor, template: PaddedSystemBatch) -> tuple[Tensor, Tensor, Tensor]:
        return self.ae.decode_raw(self.normalizer.denormalize(z, self.ae.latent_mask(template)), template)

    def decode(self, z: Tensor, template: PaddedSystemBatch) -> PaddedSystemBatch:
        return self.ae.decode(self.normalizer.denormalize(z, self.ae.latent_mask(template)), template)

    def reconstruction_objective_sums(self, *args, **kwargs):
        return self.ae.reconstruction_objective_sums(*args, **kwargs)

    def num_parameters(self) -> int:
        return self.ae.num_parameters()

    def forward(self, batch: PaddedSystemBatch):
        raise NotImplementedError("NormalizedAutoencoder serves stage 2/3 encode/decode; train the plain autoencoder")


@dataclass
class LatentDiTConfig:
    latent_dim: int = 8
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    time_emb_dim: int = 256
    num_domains: int = 4
    condition_keys: list[str] = field(default_factory=list)
    cfg_drop_rate: float = 0.1
    max_atom_count_embed: int = 1024
    global_token: bool = True
    self_conditioning: bool = True


class LatentDiT(nn.Module):
    """Diffusion transformer over latent tokens (no geometric inductive bias)."""

    def __init__(self, config: LatentDiTConfig | None = None):
        super().__init__()
        cfg = config or LatentDiTConfig()
        self.config = cfg
        dim = cfg.hidden_dim
        in_dim = cfg.latent_dim * (2 if cfg.self_conditioning else 1)
        self.in_proj = nn.Linear(in_dim, dim)
        self.slot_embed = nn.Embedding(2, dim)                 # 0 = global slot, 1 = atom slot
        self.time_mlp = nn.Sequential(nn.Linear(cfg.time_emb_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.domain_embed = nn.Embedding(cfg.num_domains, dim)
        self.pbc_embed = nn.Embedding(2, dim)
        self.count_embed = nn.Embedding(cfg.max_atom_count_embed + 1, dim)
        self.cond_dim = len(cfg.condition_keys)
        if self.cond_dim:
            self.prop_mlp = nn.Sequential(nn.Linear(2 * self.cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.null_prop = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, cfg.num_heads, cfg.mlp_ratio, 0.0) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out_proj = nn.Linear(dim, cfg.latent_dim)
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def mask(self, batch: PaddedSystemBatch) -> Tensor:
        return latent_mask(batch, self.config.global_token)

    def condition(self, batch: PaddedSystemBatch, t: Tensor, force_uncond: bool = False) -> Tensor:
        cfg = self.config
        c = self.time_mlp(sinusoidal_embedding(t * 1000.0, cfg.time_emb_dim))
        c = c + self.domain_embed(batch.domain.clamp(0, cfg.num_domains - 1)) + self.pbc_embed(batch.pbc.long())
        c = c + self.count_embed(batch.num_atoms.clamp(0, cfg.max_atom_count_embed))
        if self.cond_dim:
            vals = torch.stack([batch.properties.get(k, torch.full_like(t, float("nan"))) for k in cfg.condition_keys], dim=-1)
            feat = torch.cat([torch.nan_to_num(vals), torch.isfinite(vals).to(vals.dtype)], dim=-1)
            prop = self.prop_mlp(feat)
            drop = torch.zeros(t.shape[0], dtype=torch.bool, device=t.device)
            if self.training and cfg.cfg_drop_rate > 0:
                drop = torch.rand(t.shape[0], device=t.device) < cfg.cfg_drop_rate
            if force_uncond:
                drop = torch.ones_like(drop)
            c = c + torch.where(drop.view(-1, 1), self.null_prop.expand_as(prop), prop)
        return c

    def forward(self, z_t: Tensor, t: Tensor, batch: PaddedSystemBatch, self_cond: Tensor | None = None, force_uncond: bool = False) -> Tensor:
        cfg = self.config
        bsz, n_tok, _ = z_t.shape
        mask = self.mask(batch)
        cond = self.condition(batch, t, force_uncond)
        x = z_t
        if cfg.self_conditioning:
            x = torch.cat([z_t, torch.zeros_like(z_t) if self_cond is None else self_cond], dim=-1)
        h = self.in_proj(x)
        slot = torch.ones(n_tok, dtype=torch.long, device=z_t.device)
        if cfg.global_token:
            slot[0] = 0
        h = h + self.slot_embed(slot).unsqueeze(0)
        bias = torch.zeros(bsz, cfg.num_heads, n_tok, n_tok, device=z_t.device, dtype=h.dtype)
        for block in self.blocks:
            h = block(h, cond, bias, mask)
        shift, scale = self.final_ada(cond).unsqueeze(1).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift
        return self.out_proj(h) * mask.unsqueeze(-1).to(h.dtype)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------- interpolant
@dataclass
class LatentFlowConfig:
    sigma0: float = 1.0
    time_eps: float = 1e-3
    time_sampling: str = "logit_normal"
    self_cond_prob: float = 0.5


class LatentInterpolant:
    def __init__(self, config: LatentFlowConfig | None = None):
        self.config = config or LatentFlowConfig()

    def sample_time(self, batch_size: int, device) -> Tensor:
        cfg = self.config
        if cfg.time_sampling == "uniform":
            t = torch.rand(batch_size, device=device)
        else:
            t = torch.sigmoid(torch.randn(batch_size, device=device))
        return t.clamp(cfg.time_eps, 1 - cfg.time_eps)

    def sample_source(self, mask: Tensor, latent_dim: int, generator: torch.Generator | None = None) -> Tensor:
        z = torch.randn(mask.shape[0], mask.shape[1], latent_dim, device=mask.device, generator=generator)
        return z * self.config.sigma0 * mask.unsqueeze(-1)

    def interpolate(self, z1: Tensor, t: Tensor, mask: Tensor, z0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if z0 is None:
            z0 = torch.randn_like(z1) * self.config.sigma0 * mask.unsqueeze(-1)
        tt = t.view(-1, 1, 1)
        return (1 - tt) * z0 + tt * z1, (z1 - z0) * mask.unsqueeze(-1)

    def score(self, z_t: Tensor, v: Tensor, t: Tensor) -> Tensor:
        tt = t.view(-1, 1, 1)
        z0_hat = z_t - tt * v
        return -z0_hat / ((1 - tt) * self.config.sigma0**2)

    @staticmethod
    def endpoint(z_t: Tensor, v: Tensor, t: Tensor) -> Tensor:
        return z_t + (1 - t.view(-1, 1, 1)) * v

    @staticmethod
    def velocity_from_endpoint(z_t: Tensor, z1: Tensor, t: Tensor) -> Tensor:
        return (z1 - z_t) / (1 - t.view(-1, 1, 1)).clamp(min=1e-4)

    def flow_matching_loss(self, v_pred: Tensor, v_target: Tensor, mask: Tensor) -> Tensor:
        m = mask.unsqueeze(-1).to(v_pred.dtype)
        per = ((v_pred - v_target) ** 2 * m).sum(-1).sum(-1) / mask.sum(-1).clamp(min=1).to(v_pred.dtype)
        return per.mean()

    def self_conditioning_input(self, model: LatentDiT, z_t: Tensor, t: Tensor, batch: PaddedSystemBatch) -> Tensor | None:
        """Training-time self-conditioning: with prob p feed a detached first-pass endpoint estimate."""
        if not model.config.self_conditioning or torch.rand(()) >= self.config.self_cond_prob:
            return None
        with torch.no_grad():
            v0 = model(z_t, t, batch, None)
            return self.endpoint(z_t, v0, t).detach()


# ---------------------------------------------------------------- sampler
@dataclass
class LatentSamplerConfig:
    num_steps: int = 50
    t_start: float = 1e-3
    t_end: float = 1.0 - 1e-3
    noise: float = 0.0                 # SDE noise scale g (0 => ODE)
    langevin_correction: bool = True
    guidance_scale: float = 1.0        # classifier-free guidance on properties


@dataclass
class LatentTransition:
    z: Tensor
    z_next: Tensor
    t: Tensor
    dt: float
    v: Tensor                    # student velocity at (z, t)
    std: Tensor                  # (B,) kernel std
    logp: Tensor                 # (B,)
    self_cond: Tensor | None     # self-conditioning input the student saw at this step


@dataclass
class LatentTrajectory:
    z_final: Tensor
    transitions: list[LatentTransition] = field(default_factory=list)


class LatentSampler:
    def __init__(self, interpolant: LatentInterpolant, config: LatentSamplerConfig | None = None):
        self.interpolant = interpolant
        self.config = config or LatentSamplerConfig()

    def time_grid(self, device) -> Tensor:
        cfg = self.config
        return torch.linspace(cfg.t_start, cfg.t_end, cfg.num_steps + 1, device=device)

    def std(self, t: Tensor, dt: float) -> Tensor:
        if self.config.noise <= 0:
            return torch.zeros_like(t)
        return self.config.noise * 2.0 * torch.sqrt((t * (1 - t)).clamp(min=0)) * math.sqrt(dt)

    def mean(self, z: Tensor, v: Tensor, t: Tensor, std: Tensor, dt: float, mask: Tensor) -> Tensor:
        drift = v * dt
        if self.config.langevin_correction and self.config.noise > 0:
            drift = drift + 0.5 * (std**2).view(-1, 1, 1) * self.interpolant.score(z, v, t)
        return z + drift * mask.unsqueeze(-1)

    @staticmethod
    def gaussian_logp(z_next: Tensor, mean: Tensor, std: Tensor, mask: Tensor) -> Tensor:
        s = std.view(-1, 1, 1).clamp(min=1e-8)
        lp = -0.5 * ((z_next - mean) / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)
        return (lp * mask.unsqueeze(-1).to(lp.dtype)).flatten(1).sum(-1)

    def step(self, z: Tensor, v: Tensor, t: Tensor, dt: float, mask: Tensor, generator=None, self_cond: Tensor | None = None) -> LatentTransition:
        std = self.std(t, dt)
        mean = self.mean(z, v, t, std, dt, mask)
        noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator) * std.view(-1, 1, 1) * mask.unsqueeze(-1)
        z_next = mean + noise
        logp = self.gaussian_logp(z_next, mean, std, mask) if self.config.noise > 0 else torch.zeros(z.shape[0], device=z.device)
        return LatentTransition(z=z, z_next=z_next, t=t, dt=dt, v=v, std=std, logp=logp, self_cond=self_cond)

    def logp_under(self, tr: LatentTransition, v_other: Tensor, mask: Tensor) -> Tensor:
        if self.config.noise <= 0:
            return torch.zeros_like(tr.logp)
        return self.gaussian_logp(tr.z_next, self.mean(tr.z, v_other, tr.t, tr.std, tr.dt, mask), tr.std, mask)

    def velocity(self, model: LatentDiT, z: Tensor, t: Tensor, batch: PaddedSystemBatch, self_cond: Tensor | None = None) -> Tensor:
        v = model(z, t, batch, self_cond)
        if self.config.guidance_scale != 1.0 and getattr(model, "cond_dim", 0):
            v_unc = model(z, t, batch, self_cond, force_uncond=True)
            v = v_unc + self.config.guidance_scale * (v - v_unc)
        return v

    @torch.no_grad()
    def sample(self, model: LatentDiT, z0: Tensor, batch: PaddedSystemBatch, record: bool = False, generator=None, t_start: float | None = None, self_cond: Tensor | None = None) -> LatentTrajectory:
        grid = self.time_grid(z0.device)
        if t_start is not None:
            grid = grid[grid > t_start + 1e-9]
            grid = torch.cat([torch.tensor([t_start], device=z0.device, dtype=grid.dtype), grid])
        mask = model.mask(batch)
        z = z0
        sc = self_cond
        transitions: list[LatentTransition] = []
        for i in range(grid.numel() - 1):
            t = grid[i].expand(z.shape[0])
            dt = float(grid[i + 1] - grid[i])
            v = self.velocity(model, z, t, batch, sc)
            tr = self.step(z, v, t, dt, mask, generator, self_cond=sc)
            if record:
                transitions.append(tr)
            z = tr.z_next
            sc = self.interpolant.endpoint(tr.z, v, t) if model.config.self_conditioning else None
        return LatentTrajectory(z_final=z, transitions=transitions)


__all__ = [
    "LATENT_STATS_PROTOCOL",
    "LATENT_STD_FLOOR",
    "Callable",
    "F",
    "LatentDiT",
    "LatentDiTConfig",
    "LatentFlowConfig",
    "LatentInterpolant",
    "LatentNormalizer",
    "LatentSampler",
    "LatentSamplerConfig",
    "LatentTrajectory",
    "LatentTransition",
    "NormalizedAutoencoder",
    "latent_mask",
    "pool_atoms",
    "split_latent",
    "validate_latent_stats",
]
