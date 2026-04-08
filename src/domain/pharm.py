from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Set

import numpy as np
from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore
from rdkit.Chem.rdChemicalFeatures import FreeChemicalFeature
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeature
from rdkit.DataStructs import SparseBitVect
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Geometry import Point3D
from src.domain.ligand import Ligand

if TYPE_CHECKING:
    from src.app.progress import ProgressSink


@dataclass(frozen=True)
class Pharm2DConfig:
    # Configuration for 2D pharmacophore + Morgan fingerprint screening.
    # Controls fingerprint generation and Pharm2D feature enumeration.
    morgan_radius: int = 2          # Radius for circular (ECFP-like) Morgan fingerprint
    morgan_nbits: int = 2048        # Bit length of Morgan fingerprint
    pharm2d_min_features: int = 2   # Minimum feature count per Pharm2D pattern
    pharm2d_max_features: int = 3   # Maximum feature count per Pharm2D pattern
    pharm2d_bin_step: float = 1.0           # Bin width for discretizing inter-feature distances in Pharm2D fingerprint
    db_path: str = "/home/nikhils/Programming/LigandMiner/resources/pharm2d_cache.db"

@dataclass(frozen=True)
class Pharm3DConfig:
    # Configuration controlling 3D pharmacophore consensus generation.
    progress: "ProgressSink | None" = None
    num_conformers: int = 10        # Conformers generated per ligand
    energy_window: float = 5.0      # kcal/mol window for conformer filtering
    feature_consensus_ratio: float = 0.7 # fraction of training ligands required for a feature cluster to become a consensus pharmacophore feature
    pair_distance_support_ratio: float = 0.3 # fraction of training ligands that must contain both features for their pairwise distance distribution to be learned
    tolerance: float = 0.5          # Distance tolerance used in embedding/matching
    proximity: float = 1.5          # Clustering distance threshold for features
    pairwise_bounds_k: float = 2.0  # Multiplier for stddev to set upper bound on pairwise distances during scoring/explainability
    max_dist: float = 10.0          # Fallback distance for pairs with insufficient supporting ligands according to pair_distance_support_ratio. This converts it to a unary signal, where the feature must exist but no geometric constraints.

@dataclass(frozen=True)
class Pharm2DResult:
    # Output of Pharm2D screening for a single ligand.
    pharm2d_similarity: float       # Similarity vs consensus Pharm2D fingerprint
    morgan_similarity: float        # Similarity vs consensus Morgan fingerprint

@dataclass(frozen=True)
class PairDiagnostics:
    # Candidate-specific projection onto the trained pairwise model
    observed_distances: np.ndarray   # shape (n_pairs,)  d_ij from embedding
    z_scores: np.ndarray             # shape (n_pairs,)  (d - μ) / σ
    nll_contributions: np.ndarray    # shape (n_pairs,)  per-pair Gaussian NLL
    def __str__(self):
        if len(self.observed_distances) == 0:
            return "PairDiagnostics: <no pairs>"

        lines = ["PairDiagnostics:"]
        for i, (d, z, nll) in enumerate(zip(
            self.observed_distances,
            self.z_scores,
            self.nll_contributions
        )):
            lines.append(
                f"  pair[{i}]  d={d:.4f}  z={z:.3f}  nll={nll:.4f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Embedding:
    score: float
    diagnostics: PairDiagnostics
    conformer_id: int
    features: List[MolChemicalFeature]

@dataclass(frozen=True)
class Pharm3DResult:
    # Summary of screening a single ligand against the consensus pharmacophore
    embeddings: List[Embedding]
    # embedding[i] corresponds to the ligand feature assigned to pharmacophore feature i
    mappings: List[List[MolChemicalFeature]]
    # Raw output of MatchPharmacophoreToMol:
    # mappings[i] = all ligand features that match pharmacophore feature i (pre-geometry filtering)
    scoring_function: str            # e.g. "gaussian_nll"

    def _format_embedding(self, emb: Embedding) -> str:
        lines = []
        for i, f in enumerate(emb.features):
            p = f.GetPos()
            lines.append(
                f"  [{i}] {f.GetFamily()}/{f.GetType()} "
                f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )
        return "\n".join(lines)

    def _format_mappings(self):
        if not self.mappings:
            return "  <no mappings>"
        lines = []
        for i, feats in enumerate(self.mappings):
            if not feats:
                lines.append(f"  [{i}] <none>")
                continue
            fams = [f"{f.GetFamily()}/{f.GetType()}" for f in feats]
            lines.append(f"  [{i}] {', '.join(fams)}")
        return "\n".join(lines)

    def __str__(self):
        if not self.embeddings:
            return "=== Pharm3DResult ===\n<no valid embeddings>"

        best = min(self.embeddings, key=lambda e: e.score)

        lines = [
            "=== Pharm3DResult ===",
            f"scoring_function={self.scoring_function}",
            f"num_embeddings={len(self.embeddings)}",
            f"best_score={best.score:.4f}",
            f"best_conformer={best.conformer_id}",
            "",
            "Best embedding:",
            self._format_embedding(best),
            "",
            "Feature mappings:",
            self._format_mappings(),
            "",
            "Pair diagnostics:",
            str(best.diagnostics)
        ]
        return "\n".join(lines)

@dataclass(frozen=True)
class PairwiseModel:
    idx_a: np.ndarray   # shape (n_pairs,)
    idx_b: np.ndarray
    means: np.ndarray
    raw_stds: np.ndarray
    model_stds: np.ndarray

    def __str__(self):
        lines = ["PairwiseModel:"]
        for i, j, mu, raw_sigma, model_sigma in zip(self.idx_a, self.idx_b, self.means, self.raw_stds, self.model_stds):
            lines.append(
                f"  ({int(i)}, {int(j)}): "
                f"mean={mu:.4f}  raw std={raw_sigma:.4f}  model std={model_sigma:.4f}"
            )
        if len(self.idx_a) == 0:
            lines.append("  <no pairwise constraints>")
        return "\n".join(lines)


@dataclass(frozen=True)
class ConsensusFeature:
    # Represents a single pharmacophore feature in the learned consensus model.
    id: int                         # Index matching Pharmacophore feature ordering
    family: str                     # RDKit feature family (Donor, Acceptor, Hydrophobe, etc.)
    type: str                       # RDKit feature subtype
    centroid: Point3D               # Mean 3D position across training ligands
    radius: float                   # Spatial variability (k * stddev) for scoring/explainability
    support: float                   # Fraction of training ligands contributing to this feature cluster

    def __str__(self):
        c = self.centroid
        return (
            f"[{self.id}] "
            f"{self.family}/{self.type} "
            f"centroid=({c.x:.3f}, {c.y:.3f}, {c.z:.3f}) "
            f"radius={self.radius:.4f} "
            f"support={self.support:.3f}"
        )

type FeatureLigandMap = Dict[int, List[str]]
# Maps consensus feature index → ligand IDs that contributed to that feature cluster.


def format_rdkit_pharmacophore(pcophore: Pharmacophore) -> str:
    lines = []
    n = len(pcophore.getFeatures())

    lines.append(f"  Num features: {n}")
    lines.append("  Features:")

    for i in range(n):
        f = pcophore.getFeature(i)
        p = f.GetPos()
        lines.append(
            f"    [{i}] {f.GetFamily()}/{f.GetType()} "
            f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )

    lines.append("  Pairwise Bounds:")

    for i in range(n):
        for j in range(i + 1, n):
            L = pcophore.getLowerBound(i, j)
            U = pcophore.getUpperBound(i, j)
            if L != 0 or U != 0:
                lines.append(
                    f"    ({i},{j})  L={L:.4f}  U={U:.4f}"
                )

    return "\n".join(lines)


@dataclass(frozen=True)
class Pharm3DConsensus:
    # Full trained 3D pharmacophore model.
    features: list[ConsensusFeature]        # Metadata representation of consensus features
    pairwise_stats: PairwiseModel  # Pairwise distance statistics between consensus features across training ligands
    feature_to_ligands: FeatureLigandMap    # Provenance of each consensus feature
    pharmacophore: Pharmacophore            # RDKit Pharmacophore used for matching/embedding
    num_training_ligands: int

    def __str__(self):
        lines = []
        lines.append("=== Pharm3DConsensus ===")
        lines.append(f"Training ligands: {self.num_training_ligands}")
        lines.append("")

        lines.append("Features:")
        for f, support in zip(self.features, self.feature_support):
            lines.append(f"  {str(f)}  support={support:.3f}")

        lines.append("")
        lines.append(str(self.pairwise_stats))
        lines.append("")
        lines.append("RDKit Pharmacophore:")
        lines.append(format_rdkit_pharmacophore(self.pharmacophore))

        return "\n".join(lines)

@dataclass(frozen=True)
class ConsensusFingerprint:
    # Fingerprint representing a consensus across multiple ligands.
    fp_id: str                      # Identifier of fingerprint type
    bits: ExplicitBitVect             # Combined fingerprint bit vector
    source_id: List[str]            # Ligands used to construct consensus fingerprint


@dataclass(frozen=True)
class Fingerprint:
    # Fingerprint for a single ligand.
    fp_id: str                      # Fingerprint type identifier
    bits: ExplicitBitVect             # Bit vector
    source_id: str                  # Ligand ID


@dataclass
class FingerprintDict:
    # Holds fingerprints for a single ligand.
    pharm2d: Optional[Fingerprint] = None
    morgan: Optional[Fingerprint] = None


@dataclass
class ConsensusFingerprintDict:
    # Holds consensus fingerprints derived from training ligands.
    pharm2d: Optional[ConsensusFingerprint] = None
    morgan: Optional[ConsensusFingerprint] = None


@dataclass
class ClusterFeature:
    # A feature instance extracted from a ligand during training.
    ligand_id: str                  # Ligand from which feature originated
    feature: FreeChemicalFeature    # RDKit feature object (position + type)
    def __str__(self):
        p = self.feature.GetPos()
        return (
            f"ligand={self.ligand_id} "
            f"{self.feature.GetFamily()}/{self.feature.GetType()} "
            f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )

@dataclass
class Cluster:
    # Intermediate structure used during consensus training.
    # Groups spatially similar features across ligands.
    ligand_ids: Set[str]            # Ligands contributing to this cluster
    features: List[ClusterFeature]  # All member features
    centroid: Point3D               # Running mean position of cluster
    stddev: float                   # Spatial spread of cluster
    family: str                     # Feature family shared by cluster
    type: str                       # Feature subtype shared by cluster
    M2: float                       # Running sum of squared deviations (Welford algorithm)
    def __str__(self):
        c = self.centroid
        lines = [
            f"Cluster {self.family}/{self.type}",
            f"  ligands={len(self.ligand_ids)}",
            f"  centroid=({c.x:.3f}, {c.y:.3f}, {c.z:.3f})",
            f"  stddev={self.stddev:.4f}",
            "  members:"
        ]
        for f in self.features:
            lines.append(f"    {f}")
        return "\n".join(lines)

@dataclass
class LigandFingerprint:
    ligand: Ligand
    fps: FingerprintDict
