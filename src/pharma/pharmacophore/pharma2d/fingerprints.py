import logging
from time import perf_counter

from rdkit.Chem import Mol
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from src.pharma.utils.fingerprint_conversions import sparse_to_explicit


logger = logging.getLogger(__name__)


def mol_to_pharm2d_fp(mol: Mol, sig_factory: SigFactory) -> ExplicitBitVect:
    started = perf_counter()
    fp = Gen2DFingerprint(mol, sig_factory)
    if fp is None:
        logger.warning("Pharm2D fingerprint generation returned no fingerprint")
        raise ValueError("Failed to generate pharmacophore fingerprint.")
    # Use the sig_factory's expected size to ensure consistent fingerprint lengths
    result = sparse_to_explicit(fp, sig_factory.GetSigSize())
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Generated Pharm2D fingerprint dimensions=%d on_bits=%d elapsed_ms=%.3f",
            result.GetNumBits(),
            result.GetNumOnBits(),
            (perf_counter() - started) * 1000,
        )
    return result
