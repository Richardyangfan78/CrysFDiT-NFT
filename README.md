# CrysFDiT-NFT

**CrysFDiT-NFT** (Crystal Flow Diffusion Transformer with Negative-aware Fine-Tuning) is an unconditional generative model for inorganic crystals with up to 20 atoms per cell, trained on MP-20. It is post-trained to raise the fraction of generated crystals that are metastable, unique and novel (MSUN).

This repository has the inference code, the model weights and the chemical-substitution kernel that was used to produce the reported numbers.

## Model

- **All-atom VAE** (77.7 M parameters). Every atom becomes one 8-dimensional latent token. One extra global token carries the lattice.
- **Latent DiT** (130.3 M parameters; 12 layers, width 768, 12 heads). It generates the latent tokens by flow matching with self-conditioning. Sampling integrates a 50-step Euler ODE.
- **Training** happened in three stages:
  1. The VAE and the DiT were trained from scratch on the 27,136 crystals of the MP-20 training split.
  2. The DiT was fine-tuned with Flow-DPO.
  3. The DiT was then trained for five rounds of DiffusionNFT on verified rewards. Every rollout was relaxed and labelled. Its reward is computed in three steps:
     - It starts as valid × novel (against LeMat-Bulk) × an energy score.
     - The energy score is 1 up to 0.085 eV/atom above the convex hull (mean of three machine-learned interatomic potentials) and falls linearly to 0 at 0.15 eV/atom.
     - The reward is divided by the square root of the number of rollouts with the same reduced formula, then standardised and centred within strata of atom count, arity and element class.

     The released weights are the raw weights after round 5.
- **Sall substitution kernel**. Samples pass through a frozen kernel (`crysfdit_nft/kernel.py`, rules in `assets/element_rules.json`). It swaps whole species for compatible elements of the same group and block, and scales the lattice by the ratio of mean atomic radii. It keeps the geometry, atom count and arity of every crystal, uses no energy or novelty oracle, and returns one output for every input.

## Results

MSUN counts every generated attempt in the denominator. The development screen below uses a lighter three-MLIP evaluator (single points on the relaxed geometry) on two cohorts of 2,500 attempts (seeds 2026114001-02).

| Model (+ Sall kernel) | Development MSUN (5,000 attempts) |
|---|---|
| CrysFDiT + Flow-DPO (before NFT) | 23.36 % |
| **CrysFDiT-NFT** | **30.98 %** |

The full LeMat-GenBench evaluation of CrysFDiT-NFT (three cohorts of 2,500 attempts, seeds 2026115001-03, all eight metric families) is running. This section will be updated with its result. Under that full evaluation, CrysFDiT + Flow-DPO scored 22.23 % (mean of three cohorts).

For the benchmark, structures were pre-relaxed before evaluation (not included in this repository):

1. MACE-MP-0b3 (medium), 50 steps.
2. MACE-MP-0b3 (medium), 150 steps, with a new optimizer.
3. UMA-s-1 (omat), 50 steps.

All passes use BFGSLineSearch with FrechetCellFilter and fmax 0.02 eV/Å. A slot whose relaxation fails keeps its geometry.

## Installation

```bash
git clone https://github.com/Richardyangfan78/CrysFDiT-NFT.git
cd CrysFDiT-NFT
pip install -e .            # torch >= 2.1, numpy
pip install pymatgen        # only for CIF export
```

Download the weights from the [v1.0.0 release](https://github.com/Richardyangfan78/CrysFDiT-NFT/releases/tag/v1.0.0):

```bash
mkdir -p checkpoints
gh release download v1.0.0 --repo Richardyangfan78/CrysFDiT-NFT --dir checkpoints
```

| File | Content | Size |
|---|---|---|
| `crysfdit_nft_dit.pt` | Latent DiT weights, atom-count table, latent statistics | 521 MB |
| `crysfdit_mp20_vae.pt` | All-atom VAE weights | 311 MB |

## Usage

```bash
python scripts/sample.py --dit checkpoints/crysfdit_nft_dit.pt --vae checkpoints/crysfdit_mp20_vae.pt \
    --num-samples 2500 --seed 0 --output-dir samples/seed0
python scripts/export_cif.py samples/seed0/samples.jsonl samples/seed0/cif.zip
```

`sample.py` writes three files:

- `samples_dit.jsonl`: the DiT outputs.
- `samples.jsonl`: the same crystals after the Sall kernel (proposal seed = seed + 50, at most 20 swaps). These are the structures behind the numbers above.
- `sampling_info.json`: the sampling settings.

Each row holds `atomic_numbers`, fractional `coords` and `lattice` (rows in Å). By default the atom count of each sample is drawn from the MP-20 training histogram with `random.Random(seed)`; `--atom-counts` fixes the counts from a JSON list instead. `--kernel off` skips the substitution kernel.

Sampling 2,500 crystals, model loading included, takes about 7 s on one NVIDIA GH200 with PyTorch 2.11. The same seed, batch size and hardware give the same crystals. On that setup, `--seed 2026115001` and `--seed 2026115002` reproduce the crystals of two formal-evaluation cohorts bit for bit (2 × 2,500), both as DiT outputs and after the Sall kernel.

```python
import torch
from crysfdit_nft.pipeline import draw_atom_counts, generate, load_dit, load_vae

device = torch.device("cuda")
dit, ckpt = load_dit("checkpoints/crysfdit_nft_dit.pt", device)
vae = load_vae("checkpoints/crysfdit_mp20_vae.pt", device, ckpt["latent_stats"])
rows = generate(dit, ckpt, vae, draw_atom_counts(seed=0, num_samples=100), seed=0)
```

## Acknowledgements

- The latent route (one latent token per atom plus a global token) follows ADiT, Chemeleon2 and Mofasa.
- Post-training uses Flow-DPO and DiffusionNFT.
- Evaluation follows LeMat-GenBench, with the LeMat-Bulk reference for novelty.
- The training data are the MP-20 split of the Materials Project.

## License

MIT (see `LICENSE`).
