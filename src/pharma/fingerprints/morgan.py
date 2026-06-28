import logging
from time import perf_counter

from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64, GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


logger = logging.getLogger(__name__)


def init_morgan_factory(radius: int = 2) -> FingerprintGenerator64:
    started = perf_counter()
    morgan_factory = GetMorganGenerator(radius=radius)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Initialized Morgan fingerprint factory radius=%d elapsed_ms=%.3f",
            radius,
            (perf_counter() - started) * 1000,
        )
    return morgan_factory


def mol_to_morgan_fp(mol: Mol, morgan_factory: FingerprintGenerator64) -> ExplicitBitVect:
    started = perf_counter()
    fingerprint = morgan_factory.GetFingerprint(mol)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Generated Morgan fingerprint dimensions=%d on_bits=%d elapsed_ms=%.3f",
            fingerprint.GetNumBits(),
            fingerprint.GetNumOnBits(),
            (perf_counter() - started) * 1000,
        )
    return fingerprint
