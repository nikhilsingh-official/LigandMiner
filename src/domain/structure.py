from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from src.domain.common import MethodType, StructureDB


ResidueRef: TypeAlias = tuple[str, int, str]


@dataclass
class Chain:
    chain: str
    start: int
    end: int


@dataclass
class Pocket:
    pocket_id: int

    residues: list[ResidueRef] = field(default_factory=list)
    uniprot_indices: list[ResidueRef] = field(default_factory=list)

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
class StructurePaths:
    raw_structure_path: Path | None = None
    structure_path: Path | None = None
    fpocket_out: Path | None = None
    fpocket_pockets: Path | None = None
    fpocket_txt: Path | None = None
    prank_out: Path | None = None
    prank_csv: Path | None = None
    probis_out: Path | None = None
    probis_srf: Path | None = None
    probis_compare: Path | None = None
    probis_nosql: Path | None = None
    probis_results: Path | None = None


@dataclass
class StructureAnalysis:
    sifts_mapping: dict | None = None
    true_pocket: list[ResidueRef] | None = None
    true_pocket_certainty: float | None = None
    structural_plausibility: float = 0.0
    fpockets: list[Pocket] = field(default_factory=list)
    domain_coverage: float = 0.0
    active_site_coverage: float = 0.0


@dataclass
class StructureMetadata:
    pass


@dataclass
class ExperimentalStructureMetadata(StructureMetadata):
    method: MethodType | None = None
    resolution: float | None = None


@dataclass
class AlphaFoldStructureMetadata(StructureMetadata):
    confidence_type: str | None = None
    avg_local_score: float | None = None
    provider: str | None = None
    created_date: str | None = None
    coverage: float | None = None


@dataclass
class HomologyStructureMetadata(StructureMetadata):
    coverage: float | None = None
    gmqe: float | None = None
    identity: float | None = None
    similarity: float | None = None
    avg_local_score: float | None = None
    qmean6_z: float | None = None
    provider: str | None = None
    template: str | None = None
    created_date: str | None = None


@dataclass
class BaseStructure:
    struct_id: str
    database: StructureDB
    chains: list[Chain] = field(default_factory=list)
    cif: str | None = None
    paths: StructurePaths = field(default_factory=StructurePaths)
    analysis: StructureAnalysis = field(default_factory=StructureAnalysis)


@dataclass
class ExperimentalStructure(BaseStructure):
    metadata: ExperimentalStructureMetadata = field(default_factory=ExperimentalStructureMetadata)


@dataclass
class AlphaFoldStructure(BaseStructure):
    metadata: AlphaFoldStructureMetadata = field(default_factory=AlphaFoldStructureMetadata)


@dataclass
class HomologyStructure(BaseStructure):
    metadata: HomologyStructureMetadata = field(default_factory=HomologyStructureMetadata)

@dataclass
class PropertyData:
    metadata: ExperimentalStructureMetadata
    chains: list[Chain]

Structure: TypeAlias = ExperimentalStructure | AlphaFoldStructure | HomologyStructure
