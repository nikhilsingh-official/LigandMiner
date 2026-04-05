from typing import List

from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64

from src.domain.ligand import LigandMol
from src.domain.pharm import ConsensusFingerprint, ConsensusFingerprintDict
from src.pharma.fingerprints.consensus import fps_to_consensus
from src.pharma.pharmacophore.pharma2d.featurizer import _featurize_ligand
from src.pharma.utils.hashing import make_consensus_fp_id


def ligands_to_consensus_fp(ligands: List[LigandMol], sig_factory: SigFactory, morgan_factory: FingerprintGenerator64):
    fingerprints = []
    ids = []

    for ligand in ligands:
        fp = _featurize_ligand(ligand, sig_factory, morgan_factory)
        fingerprints.append(fp)
        ids.append(ligand.record.ligand_id)

    known_pharm2d_fps = [fp.pharm2d.bits for fp in fingerprints]
    known_morgan_fps = [fp.morgan.bits for fp in fingerprints]

    pharm2d_consensus = fps_to_consensus(known_pharm2d_fps)
    morgan_consensus = fps_to_consensus(known_morgan_fps)

    fingerprint_dict = ConsensusFingerprintDict(
        pharm2d=ConsensusFingerprint(fp_id=make_consensus_fp_id(ids, "pharm2d"), bits=pharm2d_consensus, source_id=ids),
        morgan=ConsensusFingerprint(fp_id=make_consensus_fp_id(ids, "morgan"), bits=morgan_consensus, source_id=ids)
    )

    return fingerprint_dict
