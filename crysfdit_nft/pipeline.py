"""Checkpoint loading and unconditional crystal generation for CrysFDiT-NFT.

``generate`` follows the sampling path the reported numbers were produced with: one global ``torch.manual_seed`` per
run, a 50-step Euler ODE in the normalised latent space (noise 0, guidance 1, float32), atom counts fixed per sample,
batches taken in order, and every attempt kept (no validity filter, no resampling).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .model.autoencoder import AllAtomAutoencoder, AutoencoderConfig
from .latent import (
    LatentDiT,
    LatentDiTConfig,
    LatentFlowConfig,
    LatentInterpolant,
    LatentNormalizer,
    LatentSampler,
    LatentSamplerConfig,
    NormalizedAutoencoder,
    validate_latent_stats,
)
from .data.contexts import ContextSampler, SizeTable
from .schema import Domain

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DIT_FORMAT = "crysfdit-nft/dit-v1"
VAE_FORMAT = "crysfdit-nft/vae-v1"


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vae(path: str | Path, device: torch.device, latent_stats: dict) -> NormalizedAutoencoder:
    """The frozen all-atom VAE, wrapped so that ``decode`` takes normalised latents."""
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("format") != VAE_FORMAT:
        raise ValueError(f"{path} is not a CrysFDiT-NFT VAE checkpoint (format {ckpt.get('format')!r})")
    ae = AllAtomAutoencoder(AutoencoderConfig(**ckpt["autoencoder"]))
    ae.load_state_dict(ckpt["state_dict"])
    ae = ae.to(device).eval()
    normalizer = LatentNormalizer(validate_latent_stats(latent_stats), ae.global_token)
    return NormalizedAutoencoder(ae, normalizer).to(device).eval()


def load_dit(path: str | Path, device: torch.device) -> tuple[LatentDiT, dict]:
    """The latent DiT and its checkpoint record (flow settings, atom-count table, latent statistics)."""
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("format") != DIT_FORMAT:
        raise ValueError(f"{path} is not a CrysFDiT-NFT DiT checkpoint (format {ckpt.get('format')!r})")
    model = LatentDiT(LatentDiTConfig(**ckpt["latent_dit"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_atom_count_histogram(path: str | Path | None = None) -> dict[int, int]:
    raw = json.loads(Path(path or ASSETS / "mp20_train_atom_counts.json").read_text())
    return {int(n): int(c) for n, c in raw["counts"].items()}


def draw_atom_counts(seed: int, num_samples: int, histogram: dict[int, int] | None = None) -> list[int]:
    """Atom count of every sample: ``random.Random(seed).choices`` over the MP-20 training histogram."""
    histogram = histogram or load_atom_count_histogram()
    sizes = sorted(histogram)
    return random.Random(seed).choices(sizes, weights=[histogram[n] for n in sizes], k=num_samples)


class SequenceSizeTable(SizeTable):
    """A size table whose ``sample`` hands out the next entries of a fixed atom-count sequence (no random draw)."""

    def __init__(self, table: SizeTable, sequence: list[int]):
        super().__init__(table.counts)
        self.sequence = [int(c) for c in sequence]
        self.position = 0

    def sample(self, domain: Domain, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        end = self.position + batch_size
        if end > len(self.sequence):
            raise RuntimeError(f"atom-count sequence exhausted after {self.position} of {len(self.sequence)}")
        taken = self.sequence[self.position:end]
        self.position = end
        return torch.tensor(taken, dtype=torch.long)


def record(system, seed: int, index: int) -> dict:
    """One JSONL row: fractional ``coords`` (N x 3), ``lattice`` rows in angstrom, ``atomic_numbers``."""
    return {"id": f"seed{seed}-{index:08d}", "domain": "crystal", "source": "CrysFDiT-NFT",
            "atomic_numbers": system.atomic_numbers.tolist(), "coords": system.coords.tolist(),
            "lattice": system.lattice.tolist(),
            "metadata": {**system.metadata, "sample_seed": seed, "sample_index": index}}


@torch.no_grad()
def generate(dit: LatentDiT, ckpt: dict, vae: NormalizedAutoencoder, atom_counts: list[int], seed: int,
             batch_size: int = 250, num_steps: int = 50, device: torch.device | None = None) -> list[dict]:
    """Generate ``len(atom_counts)`` crystals; sample ``i`` has ``atom_counts[i]`` atoms."""
    device = device or next(dit.parameters()).device
    interp = LatentInterpolant(LatentFlowConfig(**ckpt["latent_flow"]))
    size_table = SequenceSizeTable(SizeTable.from_state_dict(ckpt["size_table"]), atom_counts)
    ctx = ContextSampler(size_table)
    sampler = LatentSampler(interp, LatentSamplerConfig(num_steps=num_steps, noise=0.0, guidance_scale=1.0))
    n = len(atom_counts)
    rows: list[dict] = []
    torch.manual_seed(seed)
    for start in range(0, n, batch_size):
        batch = ctx.sample(min(batch_size, n - start), device=device, domain=Domain.CRYSTAL)
        z0 = interp.sample_source(dit.mask(batch), dit.config.latent_dim)
        z1 = sampler.sample(dit, z0, batch).z_final
        decoded = vae.decode(z1, batch).to("cpu")
        for i, system in enumerate(decoded.to_systems()):
            rows.append(record(system, seed, start + i))
    if [len(r["atomic_numbers"]) for r in rows] != list(atom_counts):
        raise RuntimeError("decoded atom counts differ from the requested ones")
    return rows
