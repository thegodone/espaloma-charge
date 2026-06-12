"""
Espaloma Charge Toolkit Wrapper.
"""
__all__ = ("EspalomaChargeToolkitWrapper",)

from openff.units import unit

from openff.toolkit.utils import base_wrapper, RDKitToolkitWrapper
from openff.toolkit.utils.exceptions import ChargeMethodUnavailableError
from openff.toolkit.utils.utils import inherit_docstrings

from espaloma_charge import charge


def _molecule_total_charge(molecule):
    try:
        return float(molecule.total_charge.m_as(unit.elementary_charge))
    except AttributeError:
        try:
            return float(molecule.total_charge / unit.elementary_charge)
        except AttributeError:
            return None


@inherit_docstrings
class EspalomaChargeToolkitWrapper(base_wrapper.ToolkitWrapper):
    """
    .. warning :: This API is experimental and subject to change.

    The ``model_url`` attribute can be used to modify the URL or filepath to use for charging (default: ``None``).
    """

    _toolkit_name = "Espaloma Charge Toolkit"
    _toolkit_installation_instructions = (
        "pip install espaloma_charge"
    )

    def __init__(self):
        super().__init__()

        self._toolkit_file_read_formats = []
        self._toolkit_file_write_formats = []

        # Allow the model URL to be configurable after initialization.
        self.model_url = None

        # Store an instance of an RDKitToolkitWrapper for file I/O
        self._rdkit_toolkit_wrapper = RDKitToolkitWrapper()

    def assign_partial_charges(
        self,
        molecule,
        partial_charge_method=None,
        use_conformers=None,
        strict_n_conformers=False,
        normalize_partial_charges=True,
        _cls=None,
    ):
        """

        Compute partial charges with the espaloma charge toolkit.


        .. warning :: This API is experimental and subject to change.

        Parameters
        ----------
        molecule : openff.toolkit.topology.Molecule
            Molecule for which partial charges are to be computed
        partial_charge_method: str, optional, default=None
            The charge model to use. One of ['espaloma-am1bcc']. If None, 'espaloma-am1bcc'
            will be used.
        use_conformers : iterable of unit-wrapped numpy arrays, each with shape
            (n_atoms, 3) and dimension of distance. Optional, default = None
            Ignored; espaloma-am1bcc is a graph-only charge method.
        strict_n_conformers : bool, default=False
            Ignored; conformers are accepted but not used.
        normalize_partial_charges : bool, default=True
            Whether to offset partial charges so that they sum to the total formal charge of the molecule.
            This is used to prevent accumulation of rounding errors when the partial charge generation method has
            low precision.
        _cls : class
            Molecule constructor

        Raises
        ------
        ChargeMethodUnavailableError if this toolkit cannot handle the requested charge method

        IncorrectNumConformersError if strict_n_conformers is True and use_conformers is provided
        and specifies an invalid number of conformers for the requested method

        ChargeCalculationError if the charge calculation is supported by this toolkit, but fails
        """

        PARTIAL_CHARGE_METHODS = {
            "espaloma-am1bcc": {"rec_confs": 0, "min_confs": 0, "max_confs": 0},
        }

        if partial_charge_method is None:
            partial_charge_method = "espaloma-am1bcc"

        if _cls is None:
            from openff.toolkit.topology.molecule import Molecule

            _cls = Molecule

        mol_copy = _cls(molecule)
        mol_copy._conformers = None

        partial_charge_method = partial_charge_method.lower()
        if partial_charge_method not in PARTIAL_CHARGE_METHODS:
            raise ChargeMethodUnavailableError(
                f'Partial charge method "{partial_charge_method}"" is not supported by '
                f"the Built-in toolkit. Available charge methods are "
                f"{list(PARTIAL_CHARGE_METHODS.keys())}"
            )

        if partial_charge_method == "espaloma-am1bcc":
            rdmol = self._rdkit_toolkit_wrapper.to_rdkit(mol_copy)
            partial_charges = charge(
                rdmol,
                total_charge=_molecule_total_charge(molecule),
                model_url=self.model_url,
            )
            # Due to https://github.com/choderalab/espaloma_charge/issues/7 promote to float64
            import numpy as np
            partial_charges = np.array(partial_charges, dtype=np.float64)


        molecule.partial_charges = unit.Quantity(
            partial_charges, unit.elementary_charge
        )

        if normalize_partial_charges:
            molecule._normalize_partial_charges()
