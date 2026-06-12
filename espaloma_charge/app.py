import os
from urllib import request
from pathlib import Path
from rdkit import Chem
import torch
import numpy as np
from .utils import from_rdkit_mol
from .models import ChargeEquilibrium, batch_graphs, unbatch_graphs, install_legacy_dgl_pickle_shim
from typing import Sequence


# TODO: Do we really want to define this at file level, 
# rather than within some kind of class?
MODEL_URL = """
https://github.com/choderalab/espaloma_charge/releases/download/v0.0.8/model.pt
"""

MODEL_PATH = ".espaloma_charge_model.pt"


def _default_model_path() -> Path:
    model_path = os.getenv("ESPALOMA_CHARGE_MODEL_PATH")
    if model_path:
        return Path(model_path).expanduser()

    cache_root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache_root / "espaloma_charge" / "model.pt"


def _model_path_or_url(model_url: str = None) -> Path:
    if model_url is None:
        model_url = MODEL_URL

    model_url = str(model_url).strip()
    local_path = Path(model_url).expanduser()
    if local_path.exists():
        return local_path

    model_path = _default_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        request.urlretrieve(model_url, model_path)

    return model_path


def _load_model(model_url: str = None, device=None):
    model_path = _model_path_or_url(model_url)
    load_kwargs = {"map_location": device}
    install_legacy_dgl_pickle_shim()
    try:
        return torch.load(model_path, weights_only=False, **load_kwargs)
    except TypeError:
        return torch.load(model_path, **load_kwargs)


def _is_charge_equilibrium(layer) -> bool:
    return isinstance(layer, ChargeEquilibrium) or layer.__class__.__name__ == "ChargeEquilibrium"


def _run_model(model, graph, total_charge=None):
    try:
        iterator = iter(model)
    except TypeError:
        return model(graph)

    for layer in iterator:
        if _is_charge_equilibrium(layer):
            graph = layer(graph, total_charge=total_charge)
        else:
            graph = layer(graph)
    return graph


def _charges_from_graph(graph) -> np.ndarray:
    return graph.ndata["q"].cpu().detach().flatten().numpy().astype(np.float64, copy=False)


def charge(
        molecule,
        total_charge: float = None,
        model_url: str = None,
    ) -> np.ndarray:
    """Assign machine-learned AM1-BCC partial charges to a molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        Input molecule.

    total_charge : float = 0.0


    model_url : str, optional, default=None
        URL or filepath to retrieve the model from.
        If None, the default MODEL_URL (defined at file level) is used

    Returns
    -------
    np.ndarray : (n_atoms, ) array of partial charges.

    """
    if isinstance(molecule, Sequence):
        return charge_multiple(molecule, total_charges=total_charge, model_url=model_url)


    if model_url is None:
        model_url = MODEL_URL

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_url, device=device)

    if total_charge is None:
        total_charge = Chem.GetFormalCharge(molecule)
    graph = from_rdkit_mol(molecule)

    graph = graph.to(device)
    model = model.to(device).eval()

    with torch.no_grad():
        graph = _run_model(model, graph, total_charge=total_charge)
    return _charges_from_graph(graph)


def charge_multiple(
        molecules,
        total_charges=None,
        model_url: str = None,
    ) -> np.ndarray:
    """Assign machine-learned AM1-BCC partial charges to a molecule.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        Input molecule.

    model_url : str, optional, default=None
        URL or filepath to retrieve the model from.
        If None, the default MODEL_URL (defined at file level) is used

    Returns
    -------
    np.ndarray : (n_atoms, ) array of partial charges.

    """
    if model_url is None:
        model_url = MODEL_URL

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_url, device=device)

    graphs = [from_rdkit_mol(molecule) for molecule in molecules]
    if total_charges is None:
        total_charges = [Chem.GetFormalCharge(molecule) for molecule in molecules]

    graph = batch_graphs(graphs)
    
    graph = graph.to(device)
    model = model.to(device).eval()

    with torch.no_grad():
        graph = _run_model(model, graph, total_charge=total_charges)
    graph = graph.to("cpu")
    graphs = unbatch_graphs(graph)

    return [_charges_from_graph(graph) for graph in graphs]
