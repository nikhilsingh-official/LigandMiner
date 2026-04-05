from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
from src.domain.common import StructureDB, MethodType

@dataclass
class Chain:
    chain: str
    start: int
    end: int

@dataclass
class Pocket:
    pocket_id: int

    residues: List[Tuple[str, int, str]] = field(default_factory=list)
    uniprot_indices: List[Tuple[str, int, str]] = field(default_factory=list)

    score: float = 0.0
    druggability_score: float = 0.0
    number_of_alpha_spheres: float = 0.0
    total_sasa: float = 0.0
    polar_sasa: float = 0.0
    apolar_sasa: float = 0.0
    volume: float = 0.0
    mean_local_hydrophobic_density: float = 0.0
    mean_alpha_sphere_radius: float = 0.0
    mean_alpha_sphere_solvent_access: float = 0.0
    apolar_alpha_sphere_proportion: float = 0.0
    hydrophobicity_score: float = 0.0
    volume_score: float = 0.0
    polarity_score: float = 0.0
    charge_score: float = 0.0
    proportion_of_polar_atoms: float = 0.0
    alpha_sphere_density: float = 0.0
    center_of_mass_alpha_sphere_max_dist: float = 0.0
    flexibility: float = 0.0

    binding_site_score: float = 0.0
    active_site_coverage: float = 0.0
    pocket_domain_purity: float = 0.0

@dataclass
class BaseStructure:
    struct_id: str
    database: StructureDB
    chains: list[Chain]

    cif: Optional[str] = None
    raw_structure_path: Optional[Path] = None
    structure_path: Optional[Path] = None
    fpocket_out: Optional[Path] = None
    fpocket_pockets: Optional[Path] = None
    fpocket_txt: Optional[Path] = None
    prank_out: Optional[Path] = None
    prank_csv: Optional[Path] = None
    probis_out: Optional[Path] = None
    probis_srf: Optional[Path] = None
    probis_compare: Optional[Path] = None
    probis_nosql: Optional[Path] = None
    probis_results: Optional[Path] = None

    true_pocket: Optional[list[(str, int, str)]] = None
    true_pocket_certainty: Optional[float] = None

    structural_plausibility: float = 0.0
    fpockets: list[Pocket] = field(default_factory=list)

    domain_coverage: float = 0.0
    active_site_coverage:float = 0.0

@dataclass
class ExperimentalStructure(BaseStructure):
    method: Optional[MethodType] = None
    resolution: Optional[float] = None
    sifts_mapping: Optional[dict] = None

@dataclass
class AlphaFoldStructure(BaseStructure):
    confidence_type: Optional[str] = None
    avg_local_score: Optional[float] = None
    provider: Optional[str] = None
    created_date: Optional[str] = None
    coverage: Optional[float] = None

@dataclass
class HomologyStructure(BaseStructure):
    coverage: Optional[float] = None
    gmqe: Optional[float] = None
    identity: Optional[float] = None
    similarity: Optional[float] = None

    avg_local_score: Optional[float] = None
    qmean6_z: Optional[float] = None

    provider: Optional[str] = None
    template: Optional[str] = None
    created_date: Optional[str] = None
    method: Optional[str] = None

Structure = Union[
    ExperimentalStructure,
    AlphaFoldStructure,
    HomologyStructure
]