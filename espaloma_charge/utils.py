import torch

SUPPORTED_ELEMENTS = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]


def _explicit_valence(atom):
    from rdkit import Chem

    if hasattr(atom, "GetValence") and hasattr(Chem, "ValenceType"):
        return atom.GetValence(Chem.ValenceType.EXPLICIT)
    return atom.GetExplicitValence()


def fp_rdkit(atom):
    from rdkit import Chem

    element = atom.GetSymbol()
    if element not in SUPPORTED_ELEMENTS:
        raise ValueError(f"Element {element} is not supported.")

    hybridization_fallback = torch.zeros(5, dtype=torch.get_default_dtype())
    HYBRIDIZATION_RDKIT = {
        Chem.rdchem.HybridizationType.SP: torch.tensor(
            [1, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP2: torch.tensor(
            [0, 1, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3: torch.tensor(
            [0, 0, 1, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D: torch.tensor(
            [0, 0, 0, 1, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.SP3D2: torch.tensor(
            [0, 0, 0, 0, 1],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.S: torch.tensor(
            [0, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
        Chem.rdchem.HybridizationType.UNSPECIFIED: torch.tensor(
            [0, 0, 0, 0, 0],
            dtype=torch.get_default_dtype(),
        ),
    }
    return torch.cat(
        [
            torch.tensor(
                [
                    atom.GetTotalDegree(),
                    atom.GetTotalValence(),
                    _explicit_valence(atom),
                    # atom.GetFormalCharge(),
                    atom.GetIsAromatic() * 1.0,
                    atom.GetMass(),
                    atom.IsInRingSize(3) * 1.0,
                    atom.IsInRingSize(4) * 1.0,
                    atom.IsInRingSize(5) * 1.0,
                    atom.IsInRingSize(6) * 1.0,
                    atom.IsInRingSize(7) * 1.0,
                    atom.IsInRingSize(8) * 1.0,
                ],
                dtype=torch.get_default_dtype(),
            ),
            HYBRIDIZATION_RDKIT.get(atom.GetHybridization(), hybridization_fallback),
        ],
        dim=0,
    )


def from_rdkit_mol(mol, use_fp=True):
    from .models import MoleculeGraph

    # enter nodes
    n_atoms = mol.GetNumAtoms()
    atom_type = torch.tensor(
        [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    )
    q_ref = torch.tensor(
        [[atom.GetFormalCharge()] for atom in mol.GetAtoms()]
    )
    h_v = torch.zeros(atom_type.shape[0], 100, dtype=torch.float32)

    h_v[
        torch.arange(atom_type.shape[0]),
        torch.squeeze(atom_type).long(),
    ] = 1.0

    h_v_fp = torch.stack([fp_rdkit(atom) for atom in mol.GetAtoms()], axis=0)

    if use_fp == True:
        h_v = torch.cat([h_v, h_v_fp], dim=-1)  # (n_atoms, 117)

    # enter bonds
    bonds = list(mol.GetBonds())
    bonds_begin_idxs = [bond.GetBeginAtomIdx() for bond in bonds]
    bonds_end_idxs = [bond.GetEndAtomIdx() for bond in bonds]
    src = bonds_begin_idxs + bonds_end_idxs
    dst = bonds_end_idxs + bonds_begin_idxs
    edges = torch.tensor([src, dst], dtype=torch.long)

    return MoleculeGraph(
        ndata={
            "type": atom_type,
            "q_ref": q_ref,
            "h0": h_v,
        },
        edges=edges,
    )
