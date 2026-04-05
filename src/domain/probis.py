from dataclasses import dataclass
from typing import Optional

@dataclass
class PocketMatch:
    pocket_id: int
    overlap_fraction: float
    alignment_in_pocket_fraction: float
    binding_site_score: float
    post_score: float

@dataclass
class ProBiSResult:
    aligned_vertices: int
    sva: float
    rmsd: float
    alignment_score: float
    z_score: float

    best_pocket_match: Optional[PocketMatch] = None
