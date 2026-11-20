#!/usr/bin/env python
"""Write one CIF per row of a CrysFDiT-NFT samples JSONL (needs pymatgen).

  python scripts/export_cif.py samples/seed0/samples.jsonl samples/seed0/cif
  python scripts/export_cif.py samples/seed0/samples.jsonl samples/seed0/cif.zip   # a zip, e.g. for LeMat-GenBench
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


def cif_text(row: dict) -> str:
    structure = Structure(Lattice(row["lattice"]), row["atomic_numbers"], row["coords"], coords_are_cartesian=False)
    return str(CifWriter(structure))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", type=Path)
    ap.add_argument("output", type=Path, help="a new directory, or a path ending in .zip")
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    names = [f"{i:05d}_{row['id']}.cif" for i, row in enumerate(rows)]
    if args.output.suffix == ".zip":
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, row in zip(names, rows):
                archive.writestr(name, cif_text(row))
    else:
        args.output.mkdir(parents=True)
        for name, row in zip(names, rows):
            (args.output / name).write_text(cif_text(row))
    print(f"wrote {len(rows)} CIFs to {args.output}")


if __name__ == "__main__":
    main()
