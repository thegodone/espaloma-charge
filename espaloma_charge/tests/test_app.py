def test_charge():
    from rdkit import Chem; from espaloma_charge import charge
    molecule = Chem.MolFromSmiles("N#N")
    charge(molecule)

def test_charge_batch():
    from rdkit import Chem
    from espaloma_charge import charge
    molecule = Chem.MolFromSmiles("N#N")
    charge([molecule, molecule])


def test_charge_respects_total_charge():
    import numpy as np
    from rdkit import Chem
    from espaloma_charge import charge

    molecule = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
    charges = charge(molecule, total_charge=1.0)

    assert np.isclose(charges.sum(), 1.0, atol=1.0e-5)


def test_run_model_passes_total_charge_to_charge_equilibrium():
    import torch
    from espaloma_charge.app import _run_model

    class Graph:
        pass

    class Identity(torch.nn.Module):
        def forward(self, graph):
            graph.seen_identity = True
            return graph

    class ChargeEquilibrium(torch.nn.Module):
        def forward(self, graph, total_charge=None):
            graph.seen_total_charge = total_charge
            return graph

    graph = _run_model(
        torch.nn.Sequential(Identity(), ChargeEquilibrium()),
        Graph(),
        total_charge=1.0,
    )

    assert graph.seen_identity
    assert graph.seen_total_charge == 1.0


def test_charge_forwards_sequence_arguments(monkeypatch):
    from espaloma_charge import app

    calls = {}

    def fake_charge_multiple(molecules, total_charges=None, model_url=None):
        calls["molecules"] = molecules
        calls["total_charges"] = total_charges
        calls["model_url"] = model_url
        return "charges"

    monkeypatch.setattr(app, "charge_multiple", fake_charge_multiple)

    molecules = [object(), object()]
    result = app.charge(molecules, total_charge=[1.0, -1.0], model_url="model.pt")

    assert result == "charges"
    assert calls == {
        "molecules": molecules,
        "total_charges": [1.0, -1.0],
        "model_url": "model.pt",
    }
