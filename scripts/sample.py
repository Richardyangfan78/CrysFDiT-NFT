#!/usr/bin/env python
"""Generate crystals with CrysFDiT-NFT.

  python scripts/sample.py --dit checkpoints/crysfdit_nft_dit.pt --vae checkpoints/crysfdit_mp20_vae.pt \
      --num-samples 2500 --seed 0 --output-dir samples/seed0

Writes ``samples_dit.jsonl`` (the DiT outputs) and, unless ``--kernel off``, ``samples.jsonl`` (the same crystals after
the frozen Sall substitution kernel, proposal seed = seed + 50). Rows hold ``atomic_numbers``, fractional ``coords``
and ``lattice`` (rows in angstrom); ``scripts/export_cif.py`` turns them into CIF files. Every attempt is kept.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crysfdit_nft.kernel import ChemicalKernel  # noqa: E402
from crysfdit_nft.pipeline import ASSETS, default_device, draw_atom_counts, generate, load_dit, load_vae  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, allow_nan=False) + "\n" for row in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dit", type=Path, required=True, help="CrysFDiT-NFT DiT checkpoint")
    ap.add_argument("--vae", type=Path, required=True, help="CrysFDiT MP-20 VAE checkpoint")
    ap.add_argument("--output-dir", type=Path, required=True, help="new or empty directory")
    ap.add_argument("--num-samples", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed; also seeds the atom counts")
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--steps", type=int, default=50, help="Euler steps of the latent ODE")
    ap.add_argument("--atom-counts", type=Path, default=None,
                    help="JSON list fixing the atom count of every sample (default: drawn from the MP-20 training "
                         "histogram with random.Random(seed))")
    ap.add_argument("--kernel", choices=("sall", "off"), default="sall")
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    args = ap.parse_args()

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else default_device()
    if args.atom_counts:
        counts = [int(c) for c in json.loads(args.atom_counts.read_text())]
        if len(counts) != args.num_samples:
            raise SystemExit("--num-samples must equal the length of --atom-counts")
    else:
        counts = draw_atom_counts(args.seed, args.num_samples)
    if not all(1 <= c <= 20 for c in counts):
        raise SystemExit("the model was trained on 1-20 atoms per cell")

    started = time.perf_counter()
    dit, ckpt = load_dit(args.dit, device)
    vae = load_vae(args.vae, device, ckpt["latent_stats"])
    rows = generate(dit, ckpt, vae, counts, args.seed, batch_size=args.batch_size, num_steps=args.steps, device=device)
    write_jsonl(out / "samples_dit.jsonl", rows)
    info = {"model": ckpt.get("name"), "seed": args.seed, "num_samples": len(rows), "steps": args.steps,
            "batch_size": args.batch_size, "device": str(device), "torch": torch.__version__,
            "atom_counts": "file" if args.atom_counts else "MP-20 training histogram, random.Random(seed)",
            "kernel": args.kernel, "seconds": None}
    if args.kernel == "sall":
        kernel = ChemicalKernel(json.loads((ASSETS / "element_rules.json").read_text()))
        edited = [kernel.apply(row, i, args.seed + 50, 20, "Sall") for i, row in enumerate(rows)]
        write_jsonl(out / "samples.jsonl", edited)
        info["kernel_proposal_seed"] = args.seed + 50
        info["kernel_max_edits"] = 20
    info["seconds"] = round(time.perf_counter() - started, 1)
    (out / "sampling_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
