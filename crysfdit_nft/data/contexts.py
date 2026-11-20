"""Context sampling for unconditional / conditional generation.

A *context* fixes everything the flow does not generate: the domain, the
periodicity, the number of atoms and optional conditioning properties.  Atom
counts are drawn from per-domain empirical tables estimated from the
training set.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..schema import MASK_INDEX, TYPE_CHANNELS, AtomicSystem, Domain, PaddedSystemBatch


@dataclass
class SizeTable:
    counts: dict[int, torch.Tensor]   # domain -> histogram over atom counts (index = N)

    @classmethod
    def from_systems(cls, systems: list[AtomicSystem], max_atoms: int = 1024) -> SizeTable:
        counts = {int(d): torch.zeros(max_atoms + 1) for d in Domain}
        for s in systems:
            n = min(s.num_atoms, max_atoms)
            counts[int(s.domain)][n] += 1
        return cls(counts)

    def sample(self, domain: Domain, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        hist = self.counts[int(domain)]
        if hist.sum() <= 0:
            raise ValueError(f"no size statistics for domain {domain.name}")
        return torch.multinomial(hist / hist.sum(), batch_size, replacement=True, generator=generator)

    def state_dict(self) -> dict:
        return {str(k): v for k, v in self.counts.items()}

    @classmethod
    def from_state_dict(cls, state: dict) -> SizeTable:
        # always CPU: checkpoints are loaded with map_location=<cuda device>, and a CUDA
        # histogram would make ContextSampler build CUDA atom counts next to CPU masks
        return cls({int(k): torch.as_tensor(v).detach().cpu() for k, v in state.items()})


class ContextSampler:
    def __init__(self, size_table: SizeTable, domain_weights: dict[Domain, float] | None = None, dummy_box: float = 1.0):
        self.size_table = size_table
        weights = domain_weights or {d: 1.0 for d in Domain if size_table.counts[int(d)].sum() > 0}
        self.domains = list(weights.keys())
        w = torch.tensor([weights[d] for d in self.domains], dtype=torch.float)
        self.probs = w / w.sum()
        self.dummy_box = dummy_box

    def sample(self, batch_size: int, device="cpu", generator: torch.Generator | None = None, domain: Domain | None = None, properties: dict[str, torch.Tensor] | None = None) -> PaddedSystemBatch:
        if domain is None:
            choice = torch.multinomial(self.probs, batch_size, replacement=True, generator=generator)
            domains = [self.domains[int(c)] for c in choice]
        else:
            domains = [domain] * batch_size
        num_atoms = torch.stack([self.size_table.sample(d, 1, generator)[0] for d in domains]).cpu()
        nmax = int(num_atoms.max())
        atom_mask = torch.arange(nmax).unsqueeze(0) < num_atoms.unsqueeze(1)
        atomic_numbers = torch.where(atom_mask, torch.full((batch_size, nmax), MASK_INDEX, dtype=torch.long), torch.zeros(batch_size, nmax, dtype=torch.long))
        atom_probs = torch.zeros(batch_size, nmax, TYPE_CHANNELS)
        atom_probs[..., MASK_INDEX] = 1.0
        atom_probs = atom_probs * atom_mask.unsqueeze(-1)
        pbc = torch.tensor([d.periodic for d in domains])
        batch = PaddedSystemBatch(
            atomic_numbers=atomic_numbers,
            coords=torch.zeros(batch_size, nmax, 3),
            atom_mask=atom_mask,
            lattice=torch.eye(3).mul(self.dummy_box).unsqueeze(0).repeat(batch_size, 1, 1),
            pbc=pbc,
            domain=torch.tensor([int(d) for d in domains]),
            num_atoms=num_atoms,
            atom_probs=atom_probs,
            properties={k: v.clone() for k, v in (properties or {}).items()},
            metadata=[{} for _ in range(batch_size)],
        )
        return batch.to(device)
