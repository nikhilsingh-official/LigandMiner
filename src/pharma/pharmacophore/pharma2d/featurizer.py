import logging
from time import perf_counter

from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64

from src.domain.ligand import LigandMol
from src.domain.pharm import Fingerprint, FingerprintDict
from src.pharma.fingerprints.morgan import mol_to_morgan_fp
from src.pharma.pharmacophore.pharma2d.fingerprints import mol_to_pharm2d_fp
from src.pharma.utils.hashing import make_single_fp_id


logger = logging.getLogger(__name__)


def _featurize_ligand(ligand: LigandMol, sig_factory: SigFactory, morgan_factory: FingerprintGenerator64):
    started = perf_counter()
    pharm2d = mol_to_pharm2d_fp(ligand.mol, sig_factory)
    morgan = mol_to_morgan_fp(ligand.mol, morgan_factory)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Generated fingerprints ligand_id=%s pharm2d_bits=%d pharm2d_on_bits=%d morgan_bits=%d morgan_on_bits=%d elapsed_ms=%.3f",
            ligand.record.ligand_id,
            pharm2d.GetNumBits(),
            pharm2d.GetNumOnBits(),
            morgan.GetNumBits(),
            morgan.GetNumOnBits(),
            (perf_counter() - started) * 1000,
        )

    return FingerprintDict(
        pharm2d=Fingerprint(fp_id=make_single_fp_id(ligand.record.ligand_id, "pharm2d"), bits=pharm2d, source_id=ligand.record.ligand_id),
        morgan=Fingerprint(fp_id=make_single_fp_id(ligand.record.ligand_id, "morgan"), bits=morgan, source_id=ligand.record.ligand_id)
    )
