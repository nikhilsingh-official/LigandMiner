from dataclasses import dataclass
from typing import List

from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore

from src.domain.common import AffinityType
from rdkit.Chem import Mol


@dataclass
class Conformer:
    conf_id: int
    convergence_success: bool
    energy: float

@dataclass
class LigandRecord:

    protein_id: str
    inchi_key: str
    ligand_id: str
    smiles: str
    affinity_type: str
    affinity: float

@dataclass
class Ligand:
    record: LigandRecord

@dataclass
class LigandMol(Ligand):
    mol: Mol

@dataclass
class LigandConformers(LigandMol):
    conformers: List[Conformer]

@dataclass
class LigandPharmacophore(LigandConformers):
    pharmacophore: Pharmacophore

class Ligand2DScored(LigandPharmacophore):
    pharm2d_similarity: float
    morgan_similarity: float

class Ligand3DScored(Ligand2DScored):
    pharm3d_similarity: float