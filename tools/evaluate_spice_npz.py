#!/usr/bin/env python
"""Evaluate an EspalomaCharge model against extracted SPICE NPZ charges."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from espaloma_charge.models import (
    batch_graphs,
    install_legacy_dgl_pickle_shim,
    unbatch_graphs,
)
from espaloma_charge.utils import from_rdkit_mol


def iter_charge_shards(path: Path):
    if path.is_dir():
        yield from sorted(path.glob("spice_charges_*.npz"))
    elif any(char in str(path) for char in "*?[]"):
        yield from sorted(path.parent.glob(path.name))
    else:
        yield path


def open_sdf_supplier(path: Path):
    if str(path).endswith(".gz"):
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
    return handle, Chem.ForwardSDMolSupplier(handle, removeHs=False)


def next_indexed_molecule(supplier, state: dict, target_index: int):
    while state["index"] <= target_index:
        mol = next(supplier, None)
        current = state["index"]
        state["index"] += 1
        if current == target_index:
            return mol
    raise ValueError(
        f"charge shard requested source index {target_index}, "
        f"but SDF stream is already at {state['index']}"
    )


def load_model(path: Path, device: torch.device):
    install_legacy_dgl_pickle_shim()
    return torch.load(path, map_location=device, weights_only=False).to(device).eval()


def graph_from_molecule_and_qref(mol, q_ref: np.ndarray):
    if mol is None:
        raise ValueError("SDF molecule parse failed")
    if mol.GetNumAtoms() != q_ref.shape[0]:
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        raise ValueError(
            f"atom count mismatch for {name!r}: "
            f"SDF={mol.GetNumAtoms()} q_ref={q_ref.shape[0]}"
        )
    graph = from_rdkit_mol(mol)
    graph.ndata["q_ref"] = torch.as_tensor(q_ref, dtype=torch.float32).reshape(-1, 1)
    return graph


def update_metrics(pred: torch.Tensor, ref: torch.Tensor, metrics: dict):
    diff = pred - ref
    metrics["sse"] += float(torch.sum(diff * diff).cpu())
    metrics["sae"] += float(torch.sum(torch.abs(diff)).cpu())
    metrics["max_abs"] = max(metrics["max_abs"], float(torch.max(torch.abs(diff)).cpu()))
    metrics["n_atoms"] += int(diff.numel())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charges", type=Path, required=True, help="NPZ shard, glob, or shard directory.")
    parser.add_argument("--sdf", type=Path, required=True, help="Matching spice.sdf.gz path.")
    parser.add_argument("--model", type=Path, required=True, help="EspalomaCharge model.pt path.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-mols", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional per-molecule metrics CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.model, device)

    sdf_handle, sdf_supplier = open_sdf_supplier(args.sdf)
    sdf_state = {"index": 0}

    metrics = {"sse": 0.0, "sae": 0.0, "max_abs": 0.0, "n_atoms": 0, "n_mols": 0}
    per_mol_rows = []
    batch_graphs_in = []
    batch_rows = []

    def flush_batch():
        nonlocal batch_graphs_in, batch_rows
        if not batch_graphs_in:
            return
        graph = batch_graphs(batch_graphs_in).to(device)
        with torch.no_grad():
            graph = model(graph)
        graph = graph.to("cpu")
        for row, mol_graph in zip(batch_rows, unbatch_graphs(graph)):
            pred = mol_graph.ndata["q"].flatten()
            ref = mol_graph.ndata["q_ref"].flatten()
            update_metrics(pred, ref, metrics)
            diff = pred - ref
            rmse = float(torch.mean(diff * diff).sqrt())
            mae = float(torch.mean(torch.abs(diff)))
            per_mol_rows.append(
                row
                | {
                    "rmse": rmse,
                    "mae": mae,
                    "pred_total_charge": float(pred.sum()),
                    "ref_total_charge": float(ref.sum()),
                }
            )
        metrics["n_mols"] += len(batch_graphs_in)
        batch_graphs_in = []
        batch_rows = []

    processed = 0
    try:
        for shard_path in iter_charge_shards(args.charges):
            data = np.load(shard_path, allow_pickle=False)
            offsets = data["atom_offsets"]
            source_indices = data["source_indices"]
            titles = data["titles"]
            smiles = data["smiles"]
            q_ref_all = data["q_ref"]
            for local_index, source_index in enumerate(source_indices.tolist()):
                if args.max_mols is not None and processed >= args.max_mols:
                    break
                start = int(offsets[local_index])
                end = int(offsets[local_index + 1])
                q_ref = q_ref_all[start:end]
                mol = next_indexed_molecule(sdf_supplier, sdf_state, int(source_index))
                graph = graph_from_molecule_and_qref(mol, q_ref)
                batch_graphs_in.append(graph)
                batch_rows.append(
                    {
                        "source_index": int(source_index),
                        "title": str(titles[local_index]),
                        "smiles": str(smiles[local_index]),
                        "n_atoms": int(q_ref.shape[0]),
                    }
                )
                processed += 1
                if len(batch_graphs_in) >= args.batch_size:
                    flush_batch()
            if args.max_mols is not None and processed >= args.max_mols:
                break
        flush_batch()
    finally:
        sdf_handle.close()

    rmse = (metrics["sse"] / metrics["n_atoms"]) ** 0.5
    mae = metrics["sae"] / metrics["n_atoms"]
    print(f"molecules: {metrics['n_mols']}")
    print(f"atoms:     {metrics['n_atoms']}")
    print(f"RMSE/e:   {rmse:.8f}")
    print(f"MAE/e:    {mae:.8f}")
    print(f"max |e|:  {metrics['max_abs']:.8f}")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as file:
            fieldnames = [
                "source_index",
                "title",
                "smiles",
                "n_atoms",
                "rmse",
                "mae",
                "pred_total_charge",
                "ref_total_charge",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_mol_rows)


if __name__ == "__main__":
    main()
