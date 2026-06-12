import pandas as pd
import torch
from rdkit import Chem
from openff.toolkit.topology import Molecule

def run():
    from espaloma_charge.utils import from_rdkit_mol
    from espaloma_charge.models import install_legacy_dgl_pickle_shim
    molecule = Molecule.from_smiles("CCl")
    molecule.assign_partial_charges("am1bcc")
    print(molecule.partial_charges)
    g = from_rdkit_mol(molecule.to_rdkit())
    install_legacy_dgl_pickle_shim()
    model = torch.load("model.pt", weights_only=False)
    model(g)
    print(g.ndata['q'])


if __name__ == "__main__":
    run()
