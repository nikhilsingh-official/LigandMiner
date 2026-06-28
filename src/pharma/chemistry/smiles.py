import logging
from time import perf_counter

from rdkit import Chem
from rdkit.Chem import AddHs


logger = logging.getLogger(__name__)


def smiles_to_mol(smiles: str):
    started = perf_counter()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("Invalid SMILES stage=smiles_parsing smiles=%s", smiles)
        raise ValueError(f"Invalid SMILES: {smiles}")
    logger.debug(
        "Parsed SMILES atoms=%d elapsed_ms=%.3f",
        mol.GetNumAtoms(),
        (perf_counter() - started) * 1000,
    )
    return mol
