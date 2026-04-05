from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64, GetMorganGenerator
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def init_morgan_factory(radius: int = 2) -> FingerprintGenerator64:
    morgan_factory = GetMorganGenerator(radius=radius)
    return morgan_factory


def mol_to_morgan_fp(mol: Mol, morgan_factory: FingerprintGenerator64) -> ExplicitBitVect:
    return morgan_factory.GetFingerprint(mol)