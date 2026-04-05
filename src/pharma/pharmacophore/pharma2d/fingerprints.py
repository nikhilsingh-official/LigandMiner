from rdkit.Chem import Mol
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from src.pharma.utils.fingerprint_conversions import sparse_to_explicit


def mol_to_pharm2d_fp(mol: Mol, sig_factory: SigFactory) -> ExplicitBitVect:
    fp = Gen2DFingerprint(mol, sig_factory)
    if fp is None:
        raise ValueError("Failed to generate pharmacophore fingerprint.")
    # Use the sig_factory's expected size to ensure consistent fingerprint lengths
    return sparse_to_explicit(fp, sig_factory.GetSigSize())
