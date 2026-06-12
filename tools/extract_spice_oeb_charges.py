#!/usr/bin/env python
"""Extract EspalomaCharge SPICE OEB charges into open NPZ shards.

This is intentionally a one-way ingestion tool. It may require a licensed
OpenEye/OEChem installation to read ``spice.oeb``, but the emitted NPZ shards
can be consumed later with only NumPy, RDKit, and Torch.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from itertools import zip_longest
from pathlib import Path

import numpy as np
from rdkit import Chem


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def open_sdf_supplier(path: Path):
    if str(path).endswith(".gz"):
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    return handle, Chem.ForwardSDMolSupplier(handle, removeHs=False)


def iter_oeb_molecules(path: Path):
    try:
        from openeye import oechem
    except ImportError as exc:
        raise RuntimeError(
            "OpenEye/OEChem is required only for OEB extraction. Install "
            "`openeye-toolkits` in a conversion environment, then rerun."
        ) from exc

    if not oechem.OEChemIsLicensed():
        raise RuntimeError(
            "OEChem is installed but unlicensed. Set OE_LICENSE or place "
            "oe_license.txt where OpenEye can find it, then rerun this "
            "one-time extraction."
        )

    stream = oechem.oemolistream()
    if not stream.open(str(path)):
        raise RuntimeError(f"could not open OEB file: {path}")

    yield from stream.GetOEMols()


def flush_shard(out_dir: Path, shard_index: int, rows: list[dict], q_refs: list[np.ndarray]):
    if not rows:
        return None

    atom_offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for idx, q_ref in enumerate(q_refs):
        atom_offsets[idx + 1] = atom_offsets[idx] + q_ref.shape[0]

    q_ref_flat = np.concatenate(q_refs).astype(np.float32, copy=False)
    path = out_dir / f"spice_charges_{shard_index:05d}.npz"
    np.savez_compressed(
        path,
        format_version=np.array([1], dtype=np.int64),
        q_ref=q_ref_flat,
        atom_offsets=atom_offsets,
        source_indices=np.array([row["source_index"] for row in rows], dtype=np.int64),
        titles=np.array([row["title"] for row in rows], dtype=str),
        smiles=np.array([row["smiles"] for row in rows], dtype=str),
        n_atoms=np.array([row["n_atoms"] for row in rows], dtype=np.int32),
        formal_charge=np.array([row["formal_charge"] for row in rows], dtype=np.int32),
        total_charge=np.array([row["total_charge"] for row in rows], dtype=np.float32),
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oeb", type=Path, required=True, help="Input spice.oeb path.")
    parser.add_argument(
        "--sdf",
        type=Path,
        required=True,
        help="Matching structure SDF/SDF.GZ used to validate order and atom counts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for NPZ shards and manifest.csv.",
    )
    parser.add_argument("--shard-size", type=int, default=25_000)
    parser.add_argument("--max-mols", type=int, default=None)
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Skip mismatches instead of raising. Use only for exploratory extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    strict = not args.no_strict
    sdf_handle, sdf_supplier = open_sdf_supplier(args.sdf)

    shard_rows: list[dict] = []
    shard_q_refs: list[np.ndarray] = []
    manifest_rows: list[dict] = []
    failures: list[dict] = []
    shard_index = 0

    try:
        paired = zip_longest(iter_oeb_molecules(args.oeb), sdf_supplier)
        for source_index, (oe_mol, rd_mol) in enumerate(paired):
            if args.max_mols is not None and source_index >= args.max_mols:
                break

            if oe_mol is None or rd_mol is None:
                message = "OEB/SDF molecule counts differ"
                if strict:
                    raise RuntimeError(message)
                failures.append({"source_index": source_index, "error": message})
                continue

            title = oe_mol.GetTitle()
            sdf_title = rd_mol.GetProp("_Name") if rd_mol.HasProp("_Name") else ""
            charges = np.array(
                [atom.GetPartialCharge() for atom in oe_mol.GetAtoms()],
                dtype=np.float32,
            )

            if charges.shape[0] != rd_mol.GetNumAtoms():
                message = (
                    f"atom count mismatch for index {source_index}: "
                    f"OEB={charges.shape[0]} SDF={rd_mol.GetNumAtoms()}"
                )
                if strict:
                    raise RuntimeError(message)
                failures.append({"source_index": source_index, "error": message})
                continue

            if title != sdf_title:
                message = (
                    f"title mismatch for index {source_index}: "
                    f"OEB={title!r} SDF={sdf_title!r}"
                )
                if strict:
                    raise RuntimeError(message)
                failures.append({"source_index": source_index, "error": message})

            smiles = Chem.MolToSmiles(rd_mol, isomericSmiles=True)
            row = {
                "source_index": source_index,
                "title": title,
                "sdf_title": sdf_title,
                "smiles": smiles,
                "n_atoms": rd_mol.GetNumAtoms(),
                "formal_charge": Chem.GetFormalCharge(rd_mol),
                "total_charge": float(charges.sum(dtype=np.float64)),
            }
            shard_rows.append(row)
            shard_q_refs.append(charges)
            manifest_rows.append(row | {"shard": shard_index})

            if len(shard_rows) >= args.shard_size:
                flush_shard(args.out_dir, shard_index, shard_rows, shard_q_refs)
                shard_index += 1
                shard_rows = []
                shard_q_refs = []
    finally:
        sdf_handle.close()

    flush_shard(args.out_dir, shard_index, shard_rows, shard_q_refs)

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as file:
        fieldnames = [
            "shard",
            "source_index",
            "title",
            "sdf_title",
            "smiles",
            "n_atoms",
            "formal_charge",
            "total_charge",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if failures:
        failures_path = args.out_dir / "failures.csv"
        with failures_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["source_index", "error"])
            writer.writeheader()
            writer.writerows(failures)

    metadata_path = args.out_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "value"])
        writer.writerow(["oeb", str(args.oeb)])
        writer.writerow(["oeb_sha256", sha256_file(args.oeb)])
        writer.writerow(["sdf", str(args.sdf)])
        writer.writerow(["sdf_sha256", sha256_file(args.sdf)])
        writer.writerow(["n_molecules", len(manifest_rows)])
        writer.writerow(["n_failures", len(failures)])

    print(f"wrote {len(manifest_rows)} molecules to {args.out_dir}")
    if failures:
        print(f"recorded {len(failures)} failures")


if __name__ == "__main__":
    main()
