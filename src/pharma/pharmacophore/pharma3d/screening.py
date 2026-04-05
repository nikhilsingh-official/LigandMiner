import heapq
import itertools
import math
from typing import List

import numpy as np
from rdkit.Chem import rdMolChemicalFeatures
from rdkit.Chem.Pharm3D import EmbedLib
from src.domain.ligand import Ligand, LigandConformers
from src.domain.pharm import Pharm3DConfig, Pharm3DConsensus, PairwiseModel, Pharm3DResult, PairDiagnostics, Embedding
from src.pharma.math.gaussian_nll import gaussian_nll_vectorized
from src.pharma.pharmacophore.common.feature_factory import init_feature_factory
from src.pharma.pharmacophore.pharma3d.consensus_builder import ConsensusBuilder
from src.pharma.preparation.ligand_preparation import prepare_ligands_geometry


#TODO: implement a choice of scoring algorithms, between Mahalanobis distance, Hinge loss, Gaussian negative log-likelihood, etc.


class PharmacophoreMatcher:
    def __init__(self, feat_factory):
        self.feat_factory = feat_factory

    def get_feature_candidates(self, ligand: LigandConformers, pcophore):
        success, matches = EmbedLib.MatchPharmacophoreToMol(
            ligand.mol,
            self.feat_factory,
            pcophore
        )

        return matches

def fmt_feature(f):
    p = f.GetPos()
    atoms = ",".join(map(str, f.GetAtomIds()))
    return (
        f"{f.GetFamily()}/{f.GetType()} "
        f"atoms=[{atoms}] "
        f"pos=({p.x:.2f}, {p.y:.2f}, {p.z:.2f})"
    )

def embedding_pair_distances(embedding, model: PairwiseModel):
    coords = np.array([[f.GetPos().x, f.GetPos().y, f.GetPos().z] for f in embedding])

    if len(coords) < 2:
        return np.array([])

    diffs = coords[model.idx_a] - coords[model.idx_b]

    diffs = np.atleast_2d(diffs)

    return np.linalg.norm(diffs, axis=1)


W_GEOM   = 0.55
W_PRES   = 0.25
W_RADIAL = 0.20

TAU_GEOM   = 10.0
TAU_RADIAL = 10.0
MISSING_FEATURE_PENALTY = 4.0

def partial_geom_score(embedding, pairwise_stats):
    """
    Lower bound geometric NLL using only pairs that exist
    in the current partial embedding.
    """

    m = len(embedding)
    if m < 2:
        return 0.0

    # global pair indices from consensus model
    idx_a = pairwise_stats.idx_a
    idx_b = pairwise_stats.idx_b

    # keep only pairs fully inside the partial embedding
    mask = (idx_a < m) & (idx_b < m)

    if not np.any(mask):
        return 0.0

    idx_a_sub = idx_a[mask]
    idx_b_sub = idx_b[mask]
    means_sub = pairwise_stats.means[mask]
    stds_sub  = pairwise_stats.model_stds[mask]

    # compute coordinates of partial embedding
    coords = np.array([[f.GetPos().x, f.GetPos().y, f.GetPos().z] for f in embedding])

    diffs = coords[idx_a_sub] - coords[idx_b_sub]
    observed = np.linalg.norm(diffs, axis=1)

    total, _, _ = gaussian_nll_vectorized(observed, means_sub, stds_sub)
    return total


def enumerate_valid_embeddings(matches, pairwise_stats, max_embeddings=200):
        num_features = len(matches)

        counter = itertools.count()
        queue = [(0.0, next(counter), [])]

        full_embeddings = []
        best_full_score = float("inf")

        while queue:
            partial_score, _, embedding = heapq.heappop(queue)

            if partial_score >= best_full_score:
                continue

            depth = len(embedding)

            if depth == num_features:
                full_embeddings.append(embedding)
                best_full_score = partial_score
                if len(full_embeddings) >= max_embeddings:
                    break
                continue

            for feat in matches[depth]:
                new_embedding = embedding + [feat]

                if len(rdMolChemicalFeatures.GetAtomMatch(new_embedding)) == 0:
                    continue

                new_score = partial_geom_score(new_embedding, pairwise_stats)

                if new_score >= best_full_score:
                    continue

                heapq.heappush(queue, (new_score, next(counter), new_embedding))

        return full_embeddings

def compute_presence_likelihood(matches, consensus, consensus_features):
    presence_score = 0.0
    max_presence_score = 0.0

    for i, rd_feat in enumerate(consensus.getFeatures()):
        fid = rd_feat.GetId()
        p_i = consensus_features[fid].support

        p_i = min(max(p_i, 1e-6), 1 - 1e-6)

        present = len(matches[i]) > 0

        term_present = -math.log(p_i)
        term_absent  = -math.log(1.0 - p_i)

        presence_score += term_present if present else term_absent
        max_presence_score += max(term_present, term_absent)

    presence_norm = (
        presence_score / max_presence_score
        if max_presence_score > 0 else 1.0
    )

    return presence_score, presence_norm

def compute_radial_likelihood(matches, ligand, consensus, consensus_features):
    radial_score = 0.0

    for i, rd_feat in enumerate(consensus.getFeatures()):
        fid = rd_feat.GetId()
        cons_feat = consensus_features[fid]
        sigma = max(cons_feat.radius, 1e-3)

        if len(matches[i]) == 0:
            radial_score += MISSING_FEATURE_PENALTY
            continue

        best_r = np.inf
        for conformer in ligand.conformers:
            for feat in matches[i]:
                feat.SetActiveConformer(conformer.conf_id)
                r = feat.GetPos().Distance(cons_feat.centroid)
                best_r = min(best_r, r)

        radial_score += 0.5 * (best_r / sigma) ** 2

    radial_norm = 1.0 - math.exp(-radial_score / TAU_RADIAL)
    return radial_score, radial_norm

def best_geometric_score_for_embedding(embedding, ligand, pairwise_stats):
    best_score = np.inf
    best_diag = None
    best_conf = None

    for conformer in ligand.conformers:
        for f in embedding:
            f.SetActiveConformer(conformer.conf_id)

        observed = embedding_pair_distances(embedding, pairwise_stats)
        total, z, nll = gaussian_nll_vectorized(
            observed, pairwise_stats.means, pairwise_stats.model_stds
        )

        if total < best_score:
            best_score = total
            best_conf = conformer.conf_id
            best_diag = PairDiagnostics(observed, z, nll)

    geom_norm = 1.0 - math.exp(-best_score / TAU_GEOM)
    return best_score, geom_norm, best_diag, best_conf

def score_embedding(
    embedding,
    ligand,
    pairwise_stats,
    presence_norm,
    radial_norm
):
    geom_raw, geom_norm, diag, conf = \
        best_geometric_score_for_embedding(embedding, ligand, pairwise_stats)

    final_score = (
        W_GEOM   * geom_norm +
        W_PRES   * presence_norm +
        W_RADIAL * radial_norm
    )

    return Embedding(
        score=final_score,
        diagnostics=diag,
        conformer_id=conf,
        features=list(embedding)
    )

def get_ranked_embeddings(
    matches,
    ligand,
    pairwise_stats,
    consensus,
    consensus_features
):

    valid_embeddings = enumerate_valid_embeddings(matches, pairwise_stats)

    _, presence_norm = compute_presence_likelihood(
        matches, consensus, consensus_features
    )

    _, radial_norm = compute_radial_likelihood(
        matches, ligand, consensus, consensus_features
    )

    if not valid_embeddings:
        return []

    scored = [
        score_embedding(
            emb,
            ligand,
            pairwise_stats,
            presence_norm,
            radial_norm
        )
        for emb in valid_embeddings
    ]

    scored.sort(key=lambda e: e.score)
    return scored

MAX_3D_TRAINING = 400

class Pharm3DScreening:

    def __init__(self, config: Pharm3DConfig):
        self.config = config
        self.feat_factory = init_feature_factory()
        self.progress = config.progress

        self.matcher = PharmacophoreMatcher(self.feat_factory)

        self.builder = ConsensusBuilder(
            self.feat_factory,
            config.feature_consensus_ratio,
            config.pair_distance_support_ratio,
            config.tolerance,
            config.proximity,
            config.pairwise_bounds_k,
            config.max_dist
        )

        self.consensus: Pharm3DConsensus | None = None
        self.screen_dataset: List[LigandConformers] | None = None

    def set_consensus(self, ligands: List[Ligand]):

        task = None
        if self.progress:
            task = self.progress.add_task(
                f"   ↳ Pharm3D: preparing {len(ligands)} training ligands",
                total=len(ligands)
            )

        prepped_ligands = []

        for lig in ligands:
            conf = prepare_ligands_geometry([lig])
            if conf:
                prepped_ligands.extend(conf)

            if task:
                self.progress.advance(task)

        if task:
            self.progress.update(task, description="   ✔ Pharm3D training ligands ready")

        if not prepped_ligands:
            print("[ERROR] No valid ligands provided for consensus building.")
            self.consensus = None
            return

        if self.progress:
            build_task = self.progress.add_task(
                f"   ↳ Pharm3D: building consensus",
                total=1
            )

        self.consensus = self.builder.run(prepped_ligands)
        print("Number of consensus features: ", len(self.consensus.features), " ")

        if self.progress:
            self.progress.advance(build_task)

    def set_screen_dataset(self, ligands: List[Ligand]):

        task = None
        if self.progress:
            task = self.progress.add_task(
                f"   ↳ Pharm3D: preparing screening ligands",
                total=len(ligands)
            )

        results = []

        for lig in ligands:
            conf = prepare_ligands_geometry([lig])
            if conf:
                results.extend(conf)

            if task:
                self.progress.advance(task)

        self.screen_dataset = results

        if task:
            self.progress.update(task, description="   ✔ Pharm3D screening dataset ready")

    def compare_one_to_many(
        self,
        reference: Ligand,
        candidates: List[Ligand],
    ):
        self.set_consensus([reference])
        self.set_screen_dataset(candidates)
        return self.compare_many_to_many()

    def compare_one_to_one(
        self,
        reference: Ligand,
        candidate: Ligand,
    ):
        self.set_consensus([reference])
        prepared = prepare_ligands_geometry([candidate])
        if not prepared:
            return candidate, Pharm3DResult([], [], "weighted_sum(geom, presence, radial)")

        result = self._screen_prepared(prepared[0])
        return candidate, result

    def _screen_prepared(self, ligand: LigandConformers) -> Pharm3DResult:
        matches = self.matcher.get_feature_candidates(
            ligand,
            self.consensus.pharmacophore
        )

        if matches is None:
            return Pharm3DResult([], [], "weighted_sum(geom, presence, radial)")

        embeddings = get_ranked_embeddings(
            matches,
            ligand,
            self.consensus.pairwise_stats,
            self.consensus.pharmacophore,
            self.consensus.features
        )

        return Pharm3DResult(
            embeddings=embeddings,
            mappings=matches,
            scoring_function="weighted_sum(geom, presence, radial)",
        )

    def compare_many_to_many(self):

        if self.consensus is None:
            print("No consensus - screening failed.")
            return None

        if self.screen_dataset is None:
            print("No dataset - screening failed.")
            return None

        task = None
        if self.progress:
            task = self.progress.add_task(
                f"   ↳ Pharm3D: screening {len(self.screen_dataset)} ligands",
                total=len(self.screen_dataset)
            )

        results = []

        for lig in self.screen_dataset:
            results.append((lig, self._screen_prepared(lig)))

            if task:
                self.progress.advance(task)

        if task:
            self.progress.update(task, description="   ✔ Pharm3D screening complete")

        return results
