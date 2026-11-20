"""The all-atom transformer student.

Architecture (invariant core + equivariant read-out):

* Atom tokens carry only invariant information (type state, per-atom
  conditioning).  Absolute coordinates never enter the token stream, so the
  network is exactly translation-invariant.
* Geometry enters through a pairwise attention bias built from the minimum-
  image Cartesian distances (RBF) and, for periodic samples, Fourier features
  of the fractional displacement.  The core is therefore O(3)-invariant.
* Coordinate velocities are read out as attention-weighted sums of pairwise
  displacement vectors (EGNN-style), which makes them O(3)-equivariant.  The
  lattice velocity and the type posterior are invariant read-outs.
* Global conditioning (time, domain, periodicity, lattice, atom count, scalar
  properties) is injected with adaLN-Zero in every block (DiT style), with
  classifier-free-guidance dropout on the property vector.

The same module serves molecules (Euclidean coordinates, dummy lattice),
inorganic crystals and MOFs (fractional coordinates + lattice); a per-sample
``pbc`` flag routes the geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ..geometry import (
    lattice_parameters,
    lattice_to_log_cholesky,
    pairwise_displacements,
    remove_center_of_mass,
    sinusoidal_embedding,
)
from ..schema import ELEMENT_FEATURE_DIM, TYPE_CHANNELS, ModelOutputs, PaddedSystemBatch, element_features

Tensor = torch.Tensor


@dataclass
class StudentConfig:
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    num_rbf: int = 64
    rbf_cutoff: float = 12.0          # angstrom
    num_frac_fourier: int = 8         # Fourier modes for periodic fractional displacements
    pair_dim: int = 64
    time_emb_dim: int = 256
    num_domains: int = 4
    condition_keys: list[str] = field(default_factory=list)
    cfg_drop_rate: float = 0.1
    readout_heads: int = 8            # heads of the equivariant coordinate read-out
    coord_readout_scale: float = 1.0
    max_atom_count_embed: int = 1024


# ------------------------------------------------------------- building blocks
class RadialBasis(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        self.gamma = (num_rbf / cutoff) ** 2
        self.cutoff = cutoff

    def forward(self, dist: Tensor) -> Tensor:
        rbf = torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)
        envelope = 0.5 * (torch.cos(math.pi * dist.clamp(max=self.cutoff) / self.cutoff) + 1.0)
        return rbf * envelope.unsqueeze(-1)


class AdaLNBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning and an additive attention bias."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate="tanh"), nn.Linear(int(dim * mlp_ratio), dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, h: Tensor, cond: Tensor, bias: Tensor, key_mask: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).unsqueeze(1).chunk(6, dim=-1)
        x = self.norm1(h) * (1 + scale1) + shift1
        bsz, n, dim = x.shape
        q, k, v = self.qkv(x).view(bsz, n, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        attn_mask = bias + torch.where(key_mask.view(bsz, 1, 1, n), 0.0, -1e9).to(bias.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(bsz, n, dim)
        h = h + gate1 * self.dropout(self.proj(out))
        x = self.norm2(h) * (1 + scale2) + shift2
        h = h + gate2 * self.dropout(self.mlp(x))
        return h


class EquivariantReadout(nn.Module):
    """v_i = sum_j sum_h a_ij^h * r_h(|d_ij|) * d_ij / (|d_ij| + 1)."""

    def __init__(self, dim: int, pair_dim: int, heads: int, scale: float):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, heads * 32)
        self.k = nn.Linear(dim, heads * 32)
        self.radial = nn.Sequential(nn.Linear(pair_dim, pair_dim), nn.SiLU(), nn.Linear(pair_dim, heads))
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, heads))
        self.scale = scale
        nn.init.zeros_(self.radial[-1].weight)
        nn.init.zeros_(self.radial[-1].bias)

    def forward(self, h: Tensor, pair: Tensor, disp: Tensor, dist: Tensor, pair_mask: Tensor) -> Tensor:
        bsz, n, _ = h.shape
        q = self.q(h).view(bsz, n, self.heads, 32)
        k = self.k(h).view(bsz, n, self.heads, 32)
        logits = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(32)
        weights = torch.tanh(logits) * self.radial(pair) * pair_mask.unsqueeze(-1).to(h.dtype)   # (B, N, N, H)
        direction = disp / (dist.unsqueeze(-1) + 1.0)
        per_head = torch.einsum("bijh,bijk->bihk", weights, direction)                           # (B, N, H, 3)
        gate = self.gate(h).unsqueeze(-1)                                                          # (B, N, H, 1)
        return (per_head * gate).sum(2) * self.scale


# --------------------------------------------------------------------- model
class UnifiedStudent(nn.Module):
    def __init__(self, config: StudentConfig | None = None):
        super().__init__()
        cfg = config or StudentConfig()
        self.config = cfg
        dim = cfg.hidden_dim

        self.type_embed = nn.Linear(TYPE_CHANNELS + ELEMENT_FEATURE_DIM, dim)
        self.rbf = RadialBasis(cfg.num_rbf, cfg.rbf_cutoff)
        pair_in = cfg.num_rbf + 2 * 3 * cfg.num_frac_fourier + 2   # + pbc flag + same-atom flag
        self.pair_mlp = nn.Sequential(nn.Linear(pair_in, cfg.pair_dim), nn.SiLU(), nn.Linear(cfg.pair_dim, cfg.pair_dim), nn.SiLU())
        self.pair_bias = nn.ModuleList([nn.Linear(cfg.pair_dim, cfg.num_heads) for _ in range(cfg.num_layers)])

        self.time_mlp = nn.Sequential(nn.Linear(cfg.time_emb_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.domain_embed = nn.Embedding(cfg.num_domains, dim)
        self.pbc_embed = nn.Embedding(2, dim)
        self.count_embed = nn.Embedding(cfg.max_atom_count_embed + 1, dim)
        self.lattice_mlp = nn.Sequential(nn.Linear(12, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond_dim = len(cfg.condition_keys)
        if self.cond_dim:
            self.prop_mlp = nn.Sequential(nn.Linear(2 * self.cond_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.null_prop = nn.Parameter(torch.zeros(dim))

        self.blocks = nn.ModuleList([AdaLNBlock(dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)

        self.coord_readout = EquivariantReadout(dim, cfg.pair_dim, cfg.readout_heads, cfg.coord_readout_scale)
        self.type_head = nn.Linear(dim, TYPE_CHANNELS)
        self.lattice_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 6))
        nn.init.zeros_(self.lattice_head[-1].weight)
        nn.init.zeros_(self.lattice_head[-1].bias)

    # --------------------------------------------------------- features
    def _global_condition(self, batch: PaddedSystemBatch, t: Tensor, force_uncond: bool = False) -> Tensor:
        cfg = self.config
        temb = self.time_mlp(sinusoidal_embedding(t * 1000.0, cfg.time_emb_dim))
        cond = temb + self.domain_embed(batch.domain.clamp(0, cfg.num_domains - 1)) + self.pbc_embed(batch.pbc.long())
        cond = cond + self.count_embed(batch.num_atoms.clamp(0, cfg.max_atom_count_embed))
        lc = lattice_to_log_cholesky(batch.lattice)
        params = lattice_parameters(batch.lattice)
        lat_feat = torch.cat([lc, params[..., :3] / 10.0, params[..., 3:] / 180.0], dim=-1)
        lat_feat = lat_feat * batch.pbc.view(-1, 1).to(lat_feat.dtype)
        cond = cond + self.lattice_mlp(lat_feat)
        if self.cond_dim:
            vals = torch.stack([batch.properties.get(k, torch.full_like(t, float("nan"))) for k in cfg.condition_keys], dim=-1)
            present = torch.isfinite(vals)
            feat = torch.cat([torch.nan_to_num(vals), present.to(vals.dtype)], dim=-1)
            prop = self.prop_mlp(feat)
            drop = torch.zeros(t.shape[0], dtype=torch.bool, device=t.device)
            if self.training and cfg.cfg_drop_rate > 0:
                drop = torch.rand(t.shape[0], device=t.device) < cfg.cfg_drop_rate
            if force_uncond:
                drop = torch.ones_like(drop)
            prop = torch.where(drop.view(-1, 1), self.null_prop.expand_as(prop), prop)
            cond = cond + prop
        return cond

    def _pair_features(self, batch: PaddedSystemBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self.config
        bsz, n, _ = batch.coords.shape
        pbc = batch.pbc
        cart = torch.where(pbc.view(-1, 1, 1), torch.matmul(batch.coords, batch.lattice), batch.coords)
        disp = pairwise_displacements(batch.coords, batch.lattice, pbc)
        disp = torch.where(pbc.view(-1, 1, 1, 1), disp, cart.unsqueeze(2) - cart.unsqueeze(1))
        dist = disp.norm(dim=-1)
        rbf = self.rbf(dist)
        frac_diff = batch.coords.unsqueeze(2) - batch.coords.unsqueeze(1)
        freqs = torch.arange(1, cfg.num_frac_fourier + 1, device=cart.device, dtype=cart.dtype) * 2 * math.pi
        phase = frac_diff.unsqueeze(-1) * freqs                         # (B, N, N, 3, F)
        fourier = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)
        fourier = fourier * pbc.view(-1, 1, 1, 1).to(fourier.dtype)
        eye = torch.eye(n, device=cart.device, dtype=cart.dtype).view(1, n, n, 1).expand(bsz, -1, -1, -1)
        pbc_flag = pbc.view(-1, 1, 1, 1).to(cart.dtype).expand(bsz, n, n, 1)
        pair = self.pair_mlp(torch.cat([rbf, fourier, eye, pbc_flag], dim=-1))
        pair_mask = batch.atom_mask.unsqueeze(1) & batch.atom_mask.unsqueeze(2)
        return pair, disp, dist, pair_mask

    # ---------------------------------------------------------- forward
    def forward(self, batch: PaddedSystemBatch, t: Tensor, force_uncond: bool = False) -> ModelOutputs:
        mask = batch.atom_mask
        atom_probs = batch.atom_probs
        if atom_probs is None:
            from ..schema import one_hot_types

            atom_probs = one_hot_types(batch.atomic_numbers)
        type_in = torch.cat([atom_probs.to(self.type_embed.weight.dtype), element_features(batch.atomic_numbers)], dim=-1)
        h = self.type_embed(type_in)
        cond = self._global_condition(batch, t, force_uncond=force_uncond)
        pair, disp, dist, pair_mask = self._pair_features(batch)
        h = h + cond.unsqueeze(1)

        for block, bias_proj in zip(self.blocks, self.pair_bias):
            bias = bias_proj(pair).permute(0, 3, 1, 2)                 # (B, H, N, N)
            bias = bias.masked_fill(~pair_mask.unsqueeze(1), 0.0)
            h = block(h, cond, bias, mask)

        shift, scale = self.final_ada(cond).unsqueeze(1).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift
        h = h * mask.unsqueeze(-1).to(h.dtype)

        # coordinates: Cartesian equivariant velocity -> interpolant space
        v_cart = self.coord_readout(h, pair, disp, dist, pair_mask) * mask.unsqueeze(-1).to(h.dtype)
        v_frac = torch.matmul(v_cart, torch.linalg.inv(batch.lattice))
        v_mol = remove_center_of_mass(v_cart, mask)
        coord_velocity = torch.where(batch.pbc.view(-1, 1, 1), v_frac, v_mol)

        # lattice: pooled invariant read-out
        pooled = (h * mask.unsqueeze(-1).to(h.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1).to(h.dtype)
        lattice_velocity = self.lattice_head(torch.cat([pooled, cond], dim=-1)) * batch.pbc.view(-1, 1).to(h.dtype)

        type_logits = self.type_head(h)
        return ModelOutputs(coord_velocity=coord_velocity, lattice_velocity=lattice_velocity, type_logits=type_logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
