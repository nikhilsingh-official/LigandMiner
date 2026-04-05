from dataclasses import dataclass, field
from typing import Optional, List

from src.domain.ligand import Ligand
from src.domain.uniprot import SequenceAnnotations
from src.domain.structure import Structure

@dataclass
class ProteinData:
    uniprot_id: str
    uniparc_id: str
    other_accessions: List[str]

    annotations: Optional[SequenceAnnotations] = None

    structures: List[Structure] = field(default_factory=list)
    best_structure: Optional[Structure] = None

    probis_score: Optional[float] = None

    ligands: List[Ligand] = field(default_factory=list)
    ligand_count: int = 0

@dataclass
class PipelineState:

    target_protein: ProteinData

    interpro_group_id: Optional[str] = None
    candidate_proteins: List[ProteinData] = field(default_factory=list)
    filtered_proteins: List[ProteinData] = field(default_factory=list)

    ranked_proteins: List[ProteinData] = field(default_factory=list)
    ranked_ligands: List[Ligand] = field(default_factory=list)
