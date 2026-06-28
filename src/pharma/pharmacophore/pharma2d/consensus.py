import logging
from time import perf_counter
from typing import List

from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64

from src.domain.ligand import LigandMol
from src.domain.pharm import ConsensusFingerprint, ConsensusFingerprintDict
from src.pharma.fingerprints.consensus import fps_to_consensus
from src.pharma.pharmacophore.pharma2d.featurizer import _featurize_ligand
from src.pharma.utils.hashing import make_consensus_fp_id


logger = logging.getLogger(__name__)


def ligands_to_consensus_fp(ligands: List[LigandMol], sig_factory: SigFactory, morgan_factory: FingerprintGenerator64, threshold_ratio: float = 0.8):
    started = perf_counter()
    logger.info("Building Pharm2D consensus from %d ligands", len(ligands))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Pharm2D consensus threshold_ratio=%.6f", threshold_ratio)
    fingerprints = []
    ids = []

    for ligand in ligands:
        fp = _featurize_ligand(ligand, sig_factory, morgan_factory)
        fingerprints.append(fp)
        ids.append(ligand.record.ligand_id)

    known_pharm2d_fps = [fp.pharm2d.bits for fp in fingerprints]
    known_morgan_fps = [fp.morgan.bits for fp in fingerprints]

    pharm2d_consensus = fps_to_consensus(known_pharm2d_fps, threshold_ratio)
    morgan_consensus = fps_to_consensus(known_morgan_fps, threshold_ratio)

    fingerprint_dict = ConsensusFingerprintDict(
        pharm2d=ConsensusFingerprint(fp_id=make_consensus_fp_id(ids, "pharm2d"), bits=pharm2d_consensus, source_id=ids),
        morgan=ConsensusFingerprint(fp_id=make_consensus_fp_id(ids, "morgan"), bits=morgan_consensus, source_id=ids)
    )

    elapsed_ms = (perf_counter() - started) * 1000
    logger.info(
        "Built Pharm2D consensus ligands=%d pharm2d_on_bits=%d morgan_on_bits=%d elapsed_ms=%.3f",
        len(ligands),
        pharm2d_consensus.GetNumOnBits(),
        morgan_consensus.GetNumOnBits(),
        elapsed_ms,
    )

    return fingerprint_dict
