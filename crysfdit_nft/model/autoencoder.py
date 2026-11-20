"""All-atom variational autoencoder: structures <-> latent tokens.

Following the latent route of ADiT / Chemeleon2 / Mofasa, every atom becomes
one continuous latent token ``z_i in R^d`` (d = 8 by default) and the whole
system gets one *global* latent token ``z_G in R^d`` (Mofasa-style local +
global split: the lattice and system-level information live in ``z_G`` so the
per-atom latents do not have to encode them redundantly).  The latent
sequence ``[z_G, z_1, ..., z_N]`` is homogeneous and Euclidean, which is what
makes on-policy distillation in latent space a single-channel Gaussian problem.

Encoder tokens carry the atom type (one-hot / masked state plus continuous
periodic-table features), the *absolute* generative coordinate (Cartesian
for molecules after random-rotation augmentation, fractional for periodic
systems after random-translation augmentation) and the lattice; a pairwise
distance bias adds local geometry.  The decoder maps latent tokens back to
coordinates, types and, from the global token, the log-Cholesky lattice.
Periodic coordinates come either from wrapping a real-valued output (legacy
"wrap" head, which must tear once per axis) or from per-axis (cos, sin) pairs
decoded with atan2 (seam-free "circular" head).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..geometry import (
    _certified_batch_images,
    _periodic_vectors,
    lattice_parameters,
    lattice_to_log_cholesky,
    log_cholesky_to_lattice,
    min_image_diff,
    pairwise_displacements,
    random_rotation_matrices,
    sinusoidal_embedding,
    wrap_frac,
)
from ..latent import latent_mask, split_latent
from ..schema import (
    ELEMENT_FEATURE_DIM,
    MASK_INDEX,
    TYPE_CHANNELS,
    PaddedSystemBatch,
    element_features,
    one_hot_types,
)
from .student import AdaLNBlock, RadialBasis

Tensor = torch.Tensor


@dataclass
class AutoencoderConfig:
    latent_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_rbf: int = 32
    rbf_cutoff: float = 10.0
    num_coord_fourier: int = 16
    num_domains: int = 4
    kl_weight: float = 1e-5
    coord_scale_euclid: float = 10.0     # angstrom scale for Cartesian Fourier features
    augment_rotation: bool = True
    augment_translation: bool = True     # random fractional translation for periodic systems
    global_token: bool = True
    type_loss_weight: float = 1.0
    coord_loss_weight: float = 10.0
    lattice_loss_weight: float = 10.0
    # Optional MOF objectives. None/zero preserve the original checkpoint loss.
    periodic_coord_scale_angstrom: float | None = None
    # Unified objective: None keeps the molecular (pred - target) / coord_scale_euclid term; s gives
    # Cartesian Angstrom / s, so s = 1 shares A^2 units with periodic_coord_scale_angstrom = 1 (loss only).
    nonperiodic_coord_scale_angstrom: float | None = None
    local_distance_loss_weight: float = 0.0
    local_distance_cutoff_angstrom: float = 3.0
    type_class_weights: dict[int, float] | None = None
    # Periodic coordinate head: "wrap" = real MLP + wrap_frac (degree 0 on each axis, so it must
    # tear once per axis); "circular" = per-axis (cos, sin) decoded with atan2 (degree 1, no seam).
    periodic_coord_head: str = "wrap"
    circular_norm_loss_weight: float = 0.0   # weight of the mean (||(c, s)|| - 1)^2 penalty
    # Decoder robustness fine-tune (not used by the released VAE): in training mode the
    # decoder sees z + std_fraction * std_k * eps (eps ~ N(0, 1)) on real tokens, where std_k is the per-dimension
    # latent std of the token kind k (global token or atom token) copied from a latent_stats.json; the KL term keeps
    # the encoder's mean/logvar. {"std_fraction": f > 0, "atom_std": [latent_dim], "global_std": [latent_dim] (iff
    # global_token)}. None = no noise, the behaviour of every earlier checkpoint; eval mode never adds noise.
    latent_noise: dict | None = None

    def __post_init__(self):
        if self.periodic_coord_head not in ("wrap", "circular"):
            raise ValueError("periodic_coord_head must be 'wrap' or 'circular'")
        if not math.isfinite(self.circular_norm_loss_weight) or self.circular_norm_loss_weight < 0:
            raise ValueError("circular_norm_loss_weight must be finite and nonnegative")
        if self.periodic_coord_head == "wrap" and self.circular_norm_loss_weight != 0:
            # The wrap head has no (c, s) output: the weight would be silently unused (a v2 model).
            raise ValueError("circular_norm_loss_weight requires periodic_coord_head='circular'")
        if self.periodic_coord_scale_angstrom is not None and (not math.isfinite(self.periodic_coord_scale_angstrom) or self.periodic_coord_scale_angstrom <= 0):
            raise ValueError("periodic_coord_scale_angstrom must be finite and positive")
        scale = self.nonperiodic_coord_scale_angstrom
        if scale is not None and (isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0):
            raise ValueError("nonperiodic_coord_scale_angstrom must be finite and positive")
        if not math.isfinite(self.local_distance_loss_weight) or self.local_distance_loss_weight < 0:
            raise ValueError("local_distance_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.local_distance_cutoff_angstrom) or self.local_distance_cutoff_angstrom <= 0:
            raise ValueError("local_distance_cutoff_angstrom must be finite and positive")
        if self.type_class_weights is not None:
            for number, weight in self.type_class_weights.items():
                if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number < MASK_INDEX:
                    raise ValueError("type_class_weights keys must be atomic numbers")
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError("type_class_weights must be finite and positive")
        if self.latent_noise is not None:
            _validate_latent_noise(self.latent_noise, self.latent_dim, self.global_token)


def _positive_floats(values, length: int, name: str) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"latent_noise.{name} must list latent_dim = {length} numbers")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"latent_noise.{name} entries must be finite and positive")


def _validate_latent_noise(noise, latent_dim: int, global_token: bool) -> None:
    """Shape and positivity of ``AutoencoderConfig.latent_noise`` (see the field comment)."""
    if not isinstance(noise, dict):
        raise TypeError("latent_noise must be null or a mapping")
    unknown = set(noise) - {"std_fraction", "atom_std", "global_std"}
    if unknown:
        raise ValueError(f"latent_noise has unknown keys: {sorted(unknown)}")
    fraction = noise.get("std_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("latent_noise.std_fraction must be finite and positive")
    _positive_floats(noise.get("atom_std"), latent_dim, "atom_std")
    if global_token:
        if "global_std" not in noise:
            raise ValueError("latent_noise.global_std is required when the autoencoder has a global token")
        _positive_floats(noise["global_std"], latent_dim, "global_std")
    elif "global_std" in noise:
        # Without a global token the value would be silently unused.
        raise ValueError("latent_noise.global_std requires global_token=true")


@dataclass
class LatentBatch:
    z: Tensor                     # (B, 1 + N, d) with the global token first (or (B, N, d))
    mean: Tensor
    logvar: Tensor
    template: PaddedSystemBatch   # sizes / domains / masks (coords may be stale)
    global_token: bool = True
    target_coords: Tensor | None = None   # coordinates the encoder saw (after augmentation)

    @property
    def mask(self) -> Tensor:
        return latent_mask(self.template, self.global_token)


class _TokenTransformer(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, num_domains: int):
        super().__init__()
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads, 4.0, 0.0) for _ in range(layers)])
        self.domain_embed = nn.Embedding(num_domains, dim)
        self.pbc_embed = nn.Embedding(2, dim)
        self.count_mlp = nn.Sequential(nn.Linear(64, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def condition(self, batch: PaddedSystemBatch) -> Tensor:
        c = self.domain_embed(batch.domain.clamp(0, self.domain_embed.num_embeddings - 1)) + self.pbc_embed(batch.pbc.long())
        return c + self.count_mlp(sinusoidal_embedding(batch.num_atoms.float(), 64))

    def forward(self, h: Tensor, batch: PaddedSystemBatch, bias: Tensor, mask: Tensor) -> Tensor:
        cond = self.condition(batch)
        for block in self.blocks:
            h = block(h, cond, bias, mask)
        return self.norm(h)


class AllAtomAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig | None = None):
        super().__init__()
        cfg = config or AutoencoderConfig()
        self.config = cfg
        dim = cfg.hidden_dim
        # encoder
        self.enc_type = nn.Linear(TYPE_CHANNELS + ELEMENT_FEATURE_DIM, dim)
        self.enc_coord = nn.Linear(2 * 3 * cfg.num_coord_fourier, dim)
        self.enc_lattice = nn.Sequential(nn.Linear(12, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.enc_global = nn.Parameter(torch.zeros(dim))
        self.rbf = RadialBasis(cfg.num_rbf, cfg.rbf_cutoff)
        self.enc_pair = nn.Sequential(nn.Linear(cfg.num_rbf + 1, 32), nn.SiLU(), nn.Linear(32, cfg.num_heads))
        self.encoder = _TokenTransformer(dim, cfg.num_layers, cfg.num_heads, cfg.num_domains)
        self.to_mean = nn.Linear(dim, cfg.latent_dim)
        self.to_logvar = nn.Linear(dim, cfg.latent_dim)
        # decoder
        self.dec_in = nn.Linear(cfg.latent_dim, dim)
        self.dec_global = nn.Linear(cfg.latent_dim, dim)
        self.decoder = _TokenTransformer(dim, cfg.num_layers, cfg.num_heads, cfg.num_domains)
        self.dec_type = nn.Linear(dim, TYPE_CHANNELS)
        self.dec_coord = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3))
        self.dec_lattice = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, 6))
        self.register_buffer("freqs", torch.arange(1, cfg.num_coord_fourier + 1).float() * math.pi)
        class_weights = torch.ones(TYPE_CHANNELS)
        for number, weight in (cfg.type_class_weights or {}).items():
            class_weights[number] = weight
        # Derived from config; do not change compatibility with old state_dicts.
        self.register_buffer("type_class_weights", class_weights, persistent=False)
        if cfg.periodic_coord_head == "circular":
            # Created last and only when requested: wrap-head state_dicts and init streams are unchanged.
            self.dec_coord_circular = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 6))
        if cfg.latent_noise is not None:
            # Row 0 = global token std (never indexed without a global token), row 1 = atom token std. Non-persistent
            # (draws no RNG, not in the state_dict): the base checkpoint loads strictly into the fine-tune model and
            # the fine-tuned weights load into a model without the option.
            noise = cfg.latent_noise
            table = torch.tensor([noise.get("global_std", noise["atom_std"]), noise["atom_std"]], dtype=torch.float32)
            self.register_buffer("latent_noise_std", table, persistent=False)

    @property
    def global_token(self) -> bool:
        return self.config.global_token

    def latent_mask(self, batch: PaddedSystemBatch) -> Tensor:
        return latent_mask(batch, self.config.global_token)

    # ---------------------------------------------------------- features
    def _augmented_coords(self, batch: PaddedSystemBatch) -> Tensor:
        cfg = self.config
        coords = batch.coords
        if not self.training:
            return coords
        pbc = batch.pbc.view(-1, 1, 1)
        if cfg.augment_rotation:
            rot = random_rotation_matrices(batch.batch_size, coords.device, coords.dtype)
            coords = torch.where(pbc, coords, coords @ rot.transpose(-1, -2))
        if cfg.augment_translation:
            shift = torch.rand(batch.batch_size, 1, 3, device=coords.device, dtype=coords.dtype)
            coords = torch.where(pbc, wrap_frac(coords + shift), coords)
        return coords * batch.atom_mask.unsqueeze(-1)

    def _coord_features(self, coords: Tensor, batch: PaddedSystemBatch) -> Tensor:
        cfg = self.config
        scale = torch.where(batch.pbc.view(-1, 1, 1), torch.ones_like(coords[..., :1]), torch.full_like(coords[..., :1], 1.0 / cfg.coord_scale_euclid))
        phase = (coords * scale).unsqueeze(-1) * self.freqs * 2.0            # (B, N, 3, F)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(-2)

    def _lattice_features(self, batch: PaddedSystemBatch) -> Tensor:
        lc = lattice_to_log_cholesky(batch.lattice)
        params = lattice_parameters(batch.lattice)
        feat = torch.cat([lc, params[..., :3] / 10.0, params[..., 3:] / 180.0], dim=-1)
        return feat * batch.pbc.view(-1, 1).to(feat.dtype)

    def _pair_bias(self, coords: Tensor, batch: PaddedSystemBatch, n_tok: int) -> Tensor:
        disp = pairwise_displacements(coords, batch.lattice, batch.pbc)
        dist = disp.norm(dim=-1)
        n = dist.shape[1]
        eye = torch.eye(n, device=dist.device, dtype=dist.dtype).view(1, n, n, 1).expand(dist.shape[0], -1, -1, -1)
        bias = self.enc_pair(torch.cat([self.rbf(dist), eye], dim=-1)).permute(0, 3, 1, 2)
        pair_mask = batch.atom_mask.unsqueeze(1) & batch.atom_mask.unsqueeze(2)
        bias = bias.masked_fill(~pair_mask.unsqueeze(1), 0.0)
        if n_tok > n:                                                    # pad for the global slot
            bias = F.pad(bias, (n_tok - n, 0, n_tok - n, 0))
        return bias

    # ------------------------------------------------------------ encode
    def encode(self, batch: PaddedSystemBatch, sample: bool = True) -> LatentBatch:
        cfg = self.config
        atom_probs = batch.atom_probs if batch.atom_probs is not None else one_hot_types(batch.atomic_numbers)
        coords = self._augmented_coords(batch)
        type_in = torch.cat([atom_probs.to(self.enc_type.weight.dtype), element_features(batch.atomic_numbers)], dim=-1)
        h = self.enc_type(type_in) + self.enc_coord(self._coord_features(coords, batch))
        h = h + self.enc_lattice(self._lattice_features(batch)).unsqueeze(1)
        if cfg.global_token:
            g = (self.enc_global + self.enc_lattice(self._lattice_features(batch))).unsqueeze(1)
            h = torch.cat([g, h], dim=1)
        mask = self.latent_mask(batch)
        h = self.encoder(h, batch, self._pair_bias(coords, batch, h.shape[1]), mask)
        mean, logvar = self.to_mean(h), self.to_logvar(h).clamp(-10, 6)
        m = mask.unsqueeze(-1).to(mean.dtype)
        mean, logvar = mean * m, logvar * m
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) * m if sample else mean
        return LatentBatch(z=z, mean=mean, logvar=logvar, template=batch, global_token=cfg.global_token, target_coords=coords)

    # ------------------------------------------------------------ decode
    def decode_heads(self, z: Tensor, template: PaddedSystemBatch) -> dict[str, Tensor | None]:
        """Latents -> type_logits (B,N,T), coords (B,N,3), lattice_lc (B,6) and circle.

        ``circle`` is the raw (B,N,3,2) per-axis (c, s) output of the circular periodic head, or None
        for the wrap head. Periodic atoms decode to f = atan2(s, c) / 2pi in [0, 1); nonperiodic
        systems always use the Cartesian ``dec_coord`` head; padded atoms decode to 0.
        """
        cfg = self.config
        z_g, z_atoms = split_latent(z, cfg.global_token)
        h = self.dec_in(z_atoms)
        if z_g is not None:
            h = h + self.dec_global(z_g).unsqueeze(1)
        bsz, n, _ = z_atoms.shape
        zero_bias = torch.zeros(bsz, cfg.num_heads, n, n, device=z.device, dtype=h.dtype)
        h = self.decoder(h, template, zero_bias, template.atom_mask)
        m = template.atom_mask.unsqueeze(-1).to(h.dtype)
        type_logits = self.dec_type(h)
        coords = self.dec_coord(h) * m
        pbc = template.pbc.view(-1, 1, 1)
        circle = None
        if cfg.periodic_coord_head == "circular":
            circle = self.dec_coord_circular(h).unflatten(-1, (3, 2))
            # Unused slots (padding, nonperiodic) become (1, 0): angle 0 and exactly zero gradient.
            used = pbc & template.atom_mask.unsqueeze(-1)
            cos = torch.where(used, circle[..., 0], torch.ones_like(circle[..., 0]))
            sin = torch.where(used, circle[..., 1], torch.zeros_like(circle[..., 1]))
            # The second wrap maps a float32 round-up to exactly 1.0 back to 0.0 (same torus point).
            frac = wrap_frac(wrap_frac(torch.atan2(sin, cos) / (2.0 * math.pi)))
            coords = torch.where(pbc, frac, coords * cfg.coord_scale_euclid)
        else:
            coords = torch.where(pbc, wrap_frac(coords), coords * cfg.coord_scale_euclid)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        g = self.dec_global(z_g) if z_g is not None else torch.zeros_like(pooled)
        lattice_lc = self.dec_lattice(torch.cat([pooled, g], dim=-1))
        return {"type_logits": type_logits, "coords": coords, "lattice_lc": lattice_lc, "circle": circle}

    def decode_raw(self, z: Tensor, template: PaddedSystemBatch) -> tuple[Tensor, Tensor, Tensor]:
        """Latents -> (type_logits (B,N,T), coords (B,N,3), lattice_lc (B,6))."""
        heads = self.decode_heads(z, template)
        return heads["type_logits"], heads["coords"], heads["lattice_lc"]

    def decode(self, z: Tensor, template: PaddedSystemBatch) -> PaddedSystemBatch:
        type_logits, coords, lattice_lc = self.decode_raw(z, template)
        logits = type_logits[..., :MASK_INDEX].clone()
        logits[..., 0] = -1e9
        types = torch.where(template.atom_mask, logits.argmax(-1), torch.zeros_like(template.atomic_numbers))
        lattice = torch.where(template.pbc.view(-1, 1, 1), log_cholesky_to_lattice(lattice_lc), template.lattice)
        return template.with_updates(coords=coords, lattice=lattice, atomic_numbers=types, atom_probs=one_hot_types(types))

    # -------------------------------------------------------------- loss
    def reconstruction_objective_sums(
        self, type_logits: Tensor, coords: Tensor, lattice_lc: Tensor,
        batch: PaddedSystemBatch, target_coords: Tensor | None = None, circle: Tensor | None = None,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Objective numerators/denominators shared by training and evaluation.

        Physical coordinate errors use exact minimum images in the *reference*
        cell, so shrinking the predicted lattice cannot reduce this loss. The
        optional local-distance term uses periodic reference neighbours within
        a fixed Angstrom cutoff; it is an index-paired supervised geometry term,
        not a bond-perception algorithm. Padding and nonperiodic systems never
        contribute to that term. Self-image distances remain a lattice concern.
        Nonperiodic coordinate errors are Cartesian, divided by
        ``nonperiodic_coord_scale_angstrom`` when set (else ``coord_scale_euclid``).
        The circular head adds ``circular_norm``: (||(c, s)|| - 1)^2 over real
        periodic atoms and 3 axes (the key is absent for the wrap head).
        """
        cfg, mask = self.config, batch.atom_mask
        if cfg.periodic_coord_head == "circular" and (circle is None or circle.shape != (*coords.shape, 2)):
            raise ValueError("the circular periodic head requires its raw (B, N, 3, 2) circle output")
        target = batch.coords if target_coords is None else target_coords
        m = mask.to(coords.dtype)
        nll = F.cross_entropy(type_logits.flatten(0, 1), batch.atomic_numbers.flatten(), reduction="none").view_as(m)
        type_weights = self.type_class_weights[batch.atomic_numbers].to(coords.dtype) * m
        euclid = cfg.nonperiodic_coord_scale_angstrom or cfg.coord_scale_euclid   # a set scale is > 0
        diff = torch.where(batch.pbc.view(-1, 1, 1), min_image_diff(coords, target), (coords - target) / euclid)
        periodic = batch.pbc.nonzero(as_tuple=False).flatten()
        if cfg.periodic_coord_scale_angstrom is not None and periodic.numel():
            delta, cells = (coords - target)[periodic], batch.lattice[periodic]
            vectors, certified = _certified_batch_images(delta, cells)
            fallback = (~certified).nonzero(as_tuple=False).flatten()
            if fallback.numel():
                exact = torch.stack([_periodic_vectors(delta[i], cells[i]) for i in fallback.tolist()])
                vectors = vectors.index_copy(0, fallback, exact)
            diff = diff.index_copy(0, periodic, vectors / cfg.periodic_coord_scale_angstrom)
        pbc_w = batch.pbc.to(coords.dtype)
        lattice_errors = (lattice_lc - lattice_to_log_cholesky(batch.lattice)).square().mean(-1)
        result = {"types": ((nll * type_weights).sum(), type_weights.sum()),
                  "coords": ((diff.square().sum(-1) * m).sum(), m.sum()),
                  "lattice": ((lattice_errors * pbc_w).sum(), pbc_w.sum())}
        zero = coords.sum() * 0.0
        local_sum, local_count = zero, zero.detach()
        if cfg.local_distance_loss_weight > 0 and periodic.numel():
            cells, pbc = batch.lattice[periodic], batch.pbc[periodic]
            reference_distances = pairwise_displacements(target[periodic], cells, pbc).norm(dim=-1).detach()
            predicted_distances = pairwise_displacements(coords[periodic], cells, pbc).norm(dim=-1)
            active = mask[periodic]
            selected = active.unsqueeze(1) & active.unsqueeze(2)
            selected &= torch.ones_like(selected).triu(diagonal=1)
            selected &= (reference_distances > 0) & (reference_distances <= cfg.local_distance_cutoff_angstrom)
            local_sum = ((predicted_distances - reference_distances).square() * selected).sum()
            local_count = selected.sum().to(coords.dtype)
        result["local_distance"] = (local_sum, local_count)
        if cfg.periodic_coord_head == "circular":
            real = circle[mask & batch.pbc.unsqueeze(-1)]           # (K, 3, 2): real periodic atoms only
            radius = torch.linalg.vector_norm(real, dim=-1)
            result["circular_norm"] = ((radius - 1.0).square().sum(), zero.detach() + radius.numel())
        return result

    def forward(self, batch: PaddedSystemBatch) -> dict[str, Tensor]:
        """Training forward pass = the ELBO loss dict.

        Training must call the model (or its DistributedDataParallel wrapper) as
        ``runner(batch)``: DDP only all-reduces gradients for work that goes through
        its own ``forward``, so calling ``runner.module.loss(batch)`` would silently
        train one divergent replica per GPU.
        """
        return self.loss(batch)

    def _noisy_latent(self, lat: LatentBatch) -> Tensor:
        """``config.latent_noise`` in training mode: z + std_fraction * std_k * eps on real tokens; else z as is.

        No random numbers are drawn when the option is off or in eval mode, so those passes stay
        bit-identical to a model without the option.
        """
        noise = self.config.latent_noise
        if noise is None or not self.training:
            return lat.z
        z = lat.z
        kind = torch.ones(z.shape[1], dtype=torch.long, device=z.device)          # 1 = atom token
        if lat.global_token:
            kind[0] = 0                                                             # 0 = global token
        std = self.latent_noise_std.to(z.dtype)[kind].unsqueeze(0) * lat.mask.unsqueeze(-1).to(z.dtype)
        return z + float(noise["std_fraction"]) * std * torch.randn_like(z)

    def loss(self, batch: PaddedSystemBatch) -> dict[str, Tensor]:
        cfg = self.config
        lat = self.encode(batch, sample=True)
        # latent_noise perturbs only what the decoder sees; the KL below uses the encoder's mean/logvar.
        heads = self.decode_heads(self._noisy_latent(lat), batch)
        type_logits, coords, lattice_lc = heads["type_logits"], heads["coords"], heads["lattice_lc"]
        mask = batch.atom_mask
        m = mask.to(coords.dtype)
        # the decoder must reproduce the coordinates the encoder actually saw (after augmentation)
        target = lat.target_coords if lat.target_coords is not None else batch.coords
        sums = self.reconstruction_objective_sums(type_logits, coords, lattice_lc, batch, target,
                                                  circle=heads["circle"])
        terms = {key: part / count.clamp(min=1.0) for key, (part, count) in sums.items()}
        lm = lat.mask.to(coords.dtype)
        kl = -0.5 * (1 + lat.logvar - lat.mean**2 - lat.logvar.exp())
        kl = (kl.sum(-1) * lm).sum() / lm.sum().clamp(min=1.0)
        total = (cfg.type_loss_weight * terms["types"] + cfg.coord_loss_weight * terms["coords"]
                 + cfg.lattice_loss_weight * terms["lattice"] + cfg.local_distance_loss_weight * terms["local_distance"] + cfg.kl_weight * kl)
        if "circular_norm" in terms:
            total = total + cfg.circular_norm_loss_weight * terms["circular_norm"]
        with torch.no_grad():
            type_acc = ((type_logits.argmax(-1) == batch.atomic_numbers).float() * m).sum() / m.sum().clamp(min=1.0)
            # Diagnostics: class-unweighted CE (comparable across weightings) and the share of active
            # latent entries whose logvar sits at the encoder's clamp floor of -10.
            nll = F.cross_entropy(type_logits.flatten(0, 1), batch.atomic_numbers.flatten(), reduction="none")
            types_unweighted = (nll.view_as(m) * m).sum() / m.sum().clamp(min=1.0)
            floor = ((lat.logvar <= -9.99) & lat.mask.unsqueeze(-1)).sum().to(coords.dtype)
            logvar_floor_frac = floor / (lm.sum() * lat.logvar.shape[-1]).clamp(min=1.0)
        return {"loss": total, **terms, "kl": kl, "type_acc": type_acc, "types_unweighted": types_unweighted,
                "logvar_floor_frac": logvar_floor_frac}

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
