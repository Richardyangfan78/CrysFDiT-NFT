"""Unified representation shared by molecules, inorganic crystals and MOFs.

Design rules
------------
* One padded batch type for every domain.  A per-sample ``pbc`` flag decides
  whether ``coords`` live on the flat torus (fractional coordinates in [0, 1))
  or in Euclidean space (centred Cartesian coordinates in angstrom).
* The lattice is always present as a (B, 3, 3) tensor.  For non-periodic
  samples it is a dummy identity-scaled box that is masked out of every loss.
* Atom types are stored twice: ``atomic_numbers`` (hard labels, 0 = padding)
  and ``atom_probs`` (the continuous / masked state used by the discrete flow).
  ``atom_probs`` has ``num_types + 1`` channels; the last channel is [MASK].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any

import torch

Tensor = torch.Tensor

MAX_Z = 100                   # supported atomic numbers 1..100
NUM_TYPES = MAX_Z + 1         # index 0 is padding / "no atom"
MASK_INDEX = NUM_TYPES        # extra channel for the masked discrete-flow state
TYPE_CHANNELS = NUM_TYPES + 1


class Domain(IntEnum):
    MOLECULE = 0            # non-periodic (organic or inorganic) molecule
    CRYSTAL = 1             # periodic inorganic crystal
    MOF = 2                 # periodic metal-organic framework
    MOLECULAR_CRYSTAL = 3   # periodic organic molecular crystal

    @property
    def periodic(self) -> bool:
        return self is not Domain.MOLECULE


@dataclass
class AtomicSystem:
    """A single structure in the unified representation (CPU tensors)."""

    atomic_numbers: Tensor                 # (N,) long, values in 1..MAX_Z
    coords: Tensor                         # (N, 3): fractional if pbc else Cartesian (angstrom)
    lattice: Tensor | None                 # (3, 3) rows are lattice vectors (angstrom)
    domain: Domain
    properties: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pbc(self) -> bool:
        return self.lattice is not None

    @property
    def num_atoms(self) -> int:
        return int(self.atomic_numbers.shape[0])

    def validate(self) -> AtomicSystem:
        if self.atomic_numbers.ndim != 1:
            raise ValueError("atomic_numbers must be 1-D")
        if self.coords.shape != (self.num_atoms, 3):
            raise ValueError(f"coords must be (N, 3); got {tuple(self.coords.shape)}")
        if int(self.atomic_numbers.min()) < 1 or int(self.atomic_numbers.max()) > MAX_Z:
            raise ValueError(f"atomic numbers must be within 1..{MAX_Z}")
        if self.pbc:
            if self.lattice.shape != (3, 3):
                raise ValueError("lattice must be (3, 3)")
            if not self.domain.periodic:
                raise ValueError("periodic lattice given for a non-periodic domain")
        elif self.domain.periodic:
            raise ValueError("periodic domain requires a lattice")
        return self


@dataclass
class PaddedSystemBatch:
    """Padded mini-batch of :class:`AtomicSystem` objects.

    Shapes (B = batch, N = max atoms in batch):
        atomic_numbers  (B, N)     long, 0 = padding
        atom_probs      (B, N, TYPE_CHANNELS) float, continuous / masked type state
        coords          (B, N, 3)  fractional if pbc[b] else centred Cartesian
        atom_mask       (B, N)     bool
        lattice         (B, 3, 3)  float
        pbc             (B,)       bool
        domain          (B,)       long (Domain values)
        num_atoms       (B,)       long
    """

    atomic_numbers: Tensor
    coords: Tensor
    atom_mask: Tensor
    lattice: Tensor
    pbc: Tensor
    domain: Tensor
    num_atoms: Tensor
    atom_probs: Tensor | None = None
    properties: dict[str, Tensor] = field(default_factory=dict)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ basics
    @property
    def batch_size(self) -> int:
        return int(self.num_atoms.shape[0])

    @property
    def max_atoms(self) -> int:
        return int(self.atomic_numbers.shape[1])

    @property
    def device(self) -> torch.device:
        return self.coords.device

    def to(self, device: torch.device | str) -> PaddedSystemBatch:
        return PaddedSystemBatch(
            atomic_numbers=self.atomic_numbers.to(device),
            coords=self.coords.to(device),
            atom_mask=self.atom_mask.to(device),
            lattice=self.lattice.to(device),
            pbc=self.pbc.to(device),
            domain=self.domain.to(device),
            num_atoms=self.num_atoms.to(device),
            atom_probs=self.atom_probs.to(device) if self.atom_probs is not None else None,
            properties={k: v.to(device) for k, v in self.properties.items()},
            metadata=self.metadata,
        )

    def with_updates(self, **kwargs: Any) -> PaddedSystemBatch:
        return replace(self, **kwargs)

    def detach(self) -> PaddedSystemBatch:
        return PaddedSystemBatch(
            atomic_numbers=self.atomic_numbers,
            coords=self.coords.detach(),
            atom_mask=self.atom_mask,
            lattice=self.lattice.detach(),
            pbc=self.pbc,
            domain=self.domain,
            num_atoms=self.num_atoms,
            atom_probs=self.atom_probs.detach() if self.atom_probs is not None else None,
            properties={k: v.detach() for k, v in self.properties.items()},
            metadata=self.metadata,
        )

    def clone(self) -> PaddedSystemBatch:
        return PaddedSystemBatch(
            atomic_numbers=self.atomic_numbers.clone(),
            coords=self.coords.clone(),
            atom_mask=self.atom_mask.clone(),
            lattice=self.lattice.clone(),
            pbc=self.pbc.clone(),
            domain=self.domain.clone(),
            num_atoms=self.num_atoms.clone(),
            atom_probs=self.atom_probs.clone() if self.atom_probs is not None else None,
            properties={k: v.clone() for k, v in self.properties.items()},
            metadata=list(self.metadata),
        )

    # ------------------------------------------------------------- selection
    def __getitem__(self, index) -> PaddedSystemBatch:
        idx = torch.as_tensor(index, device=self.num_atoms.device)
        if idx.dtype == torch.bool:
            idx = idx.nonzero(as_tuple=False).flatten()
        return PaddedSystemBatch(
            atomic_numbers=self.atomic_numbers[idx],
            coords=self.coords[idx],
            atom_mask=self.atom_mask[idx],
            lattice=self.lattice[idx],
            pbc=self.pbc[idx],
            domain=self.domain[idx],
            num_atoms=self.num_atoms[idx],
            atom_probs=self.atom_probs[idx] if self.atom_probs is not None else None,
            properties={k: v[idx] for k, v in self.properties.items()},
            metadata=[self.metadata[int(i)] for i in idx.tolist()] if self.metadata else [],
        )

    # ------------------------------------------------------------ conversion
    def to_systems(self) -> list[AtomicSystem]:
        out: list[AtomicSystem] = []
        for b in range(self.batch_size):
            n = int(self.num_atoms[b])
            pbc = bool(self.pbc[b])
            out.append(
                AtomicSystem(
                    atomic_numbers=self.atomic_numbers[b, :n].detach().cpu().clone(),
                    coords=self.coords[b, :n].detach().cpu().clone(),
                    lattice=self.lattice[b].detach().cpu().clone() if pbc else None,
                    domain=Domain(int(self.domain[b])),
                    properties={k: float(v[b]) for k, v in self.properties.items() if v.ndim == 1},
                    metadata=self.metadata[b] if b < len(self.metadata) else {},
                )
            )
        return out


def one_hot_types(atomic_numbers: Tensor) -> Tensor:
    """(B, N) long -> (B, N, TYPE_CHANNELS) one-hot (padding rows are all-zero)."""
    probs = torch.nn.functional.one_hot(atomic_numbers.clamp(min=0), num_classes=TYPE_CHANNELS).float()
    probs[..., 0] = 0.0
    return probs


def collate_systems(
    systems: list[AtomicSystem],
    dummy_box: float = 1.0,
    property_keys: tuple[str, ...] = (),
) -> PaddedSystemBatch:
    """Pad a list of systems into one batch.

    Non-periodic samples get ``dummy_box * I`` as lattice and Cartesian
    coordinates that are re-centred at the origin.
    """
    if not systems:
        raise ValueError("cannot collate an empty list")
    bsz = len(systems)
    nmax = max(s.num_atoms for s in systems)
    atomic_numbers = torch.zeros(bsz, nmax, dtype=torch.long)
    coords = torch.zeros(bsz, nmax, 3, dtype=torch.float32)
    atom_mask = torch.zeros(bsz, nmax, dtype=torch.bool)
    lattice = torch.eye(3, dtype=torch.float32).mul(dummy_box).unsqueeze(0).repeat(bsz, 1, 1)
    pbc = torch.zeros(bsz, dtype=torch.bool)
    domain = torch.zeros(bsz, dtype=torch.long)
    num_atoms = torch.zeros(bsz, dtype=torch.long)
    properties = {k: torch.zeros(bsz, dtype=torch.float32) for k in property_keys}
    metadata: list[dict[str, Any]] = []
    for b, s in enumerate(systems):
        s.validate()
        n = s.num_atoms
        atomic_numbers[b, :n] = s.atomic_numbers.long()
        c = s.coords.float()
        if s.pbc:
            c = c - torch.floor(c)
            lattice[b] = s.lattice.float()
        else:
            c = c - c.mean(dim=0, keepdim=True)
        coords[b, :n] = c
        atom_mask[b, :n] = True
        pbc[b] = s.pbc
        domain[b] = int(s.domain)
        num_atoms[b] = n
        for k in property_keys:
            properties[k][b] = float(s.properties.get(k, float("nan")))
        metadata.append(dict(s.metadata))
    return PaddedSystemBatch(
        atomic_numbers=atomic_numbers,
        coords=coords,
        atom_mask=atom_mask,
        lattice=lattice,
        pbc=pbc,
        domain=domain,
        num_atoms=num_atoms,
        atom_probs=one_hot_types(atomic_numbers),
        properties=properties,
        metadata=metadata,
    )


@dataclass
class ModelOutputs:
    """What the student (and every teacher adapter) returns for a noisy state.

    coord_velocity   (B, N, 3)  d(coords)/dt in the interpolant's coordinate space
    lattice_velocity (B, 6)     d(log-Cholesky lattice)/dt
    type_logits      (B, N, TYPE_CHANNELS) posterior over the clean atom type
    """

    coord_velocity: Tensor
    lattice_velocity: Tensor
    type_logits: Tensor

    def detach(self) -> ModelOutputs:
        return ModelOutputs(self.coord_velocity.detach(), self.lattice_velocity.detach(), self.type_logits.detach())


# ------------------------------------------------- chemistry-structured features
ELEMENT_PERIOD = torch.tensor([0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])
ELEMENT_GROUP = torch.tensor([0, 1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
ELEMENT_FEATURE_DIM = 5


def element_features(atomic_numbers: Tensor) -> Tensor:
    """(...,) atomic numbers -> (..., 5) continuous periodic-table features.

    [period/7, group/18, f-block flag, log(Z+1)/log(101), non-real flag]; used as a
    chemistry-structured complement to the one-hot / masked type state
    (CrystalDiT-style periodic-table coordinates) so that the same embedding
    generalises across organic, inorganic and metal-organic chemistry.

    Non-real slots -- padding (Z=0) and the discrete-flow [MASK] state
    (Z=MASK_INDEX>MAX_Z) -- get an all-zero periodic-table vector with the last
    flag set to 1, so a still-masked atom is NOT silently assigned the chemistry
    of element MAX_Z (Fermium) by clamping.
    """
    real = (atomic_numbers >= 1) & (atomic_numbers <= MAX_Z)
    z = atomic_numbers.clamp(0, MAX_Z)
    period = ELEMENT_PERIOD.to(z.device)[z].float() / 7.0
    group = ELEMENT_GROUP.to(z.device)[z].float() / 18.0
    fblock = (((z >= 57) & (z <= 71)) | ((z >= 89) & (z <= 100))).float()
    logz = torch.log(z.float() + 1.0) / math.log(101.0)
    realf = real.float()
    feat = torch.stack([period * realf, group * realf, fblock * realf, logz * realf], dim=-1)
    nonreal = (~real).float().unsqueeze(-1)
    return torch.cat([feat, nonreal], dim=-1)
