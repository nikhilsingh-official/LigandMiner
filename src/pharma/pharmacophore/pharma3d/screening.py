import heapq
import itertools
import logging
import math
from time import perf_counter
from typing import List

import numpy as np
from rdkit.Chem import rdMolChemicalFeatures
from rdkit.Chem.Pharm3D import EmbedLib
from src.app.progress import NullProgressSink
from src.domain.ligand import Ligand, LigandConformers
from src.domain.pharm import Pharm3DConfig, Pharm3DConsensus, PairwiseModel, Pharm3DResult, PairDiagnostics, Embedding
from src.pharma.math.gaussian_nll import gaussian_nll_vectorized
from src.pharma.pharmacophore.common.feature_factory import init_feature_factory
from src.pharma.pharmacophore.pharma3d.consensus_builder import ConsensusBuilder
from src.pharma.preparation.ligand_preparation import prepare_ligands_geometry


logger = logging.getLogger(__name__)


#TODO: implement a suite of scoring algorithms, including Mahalanobis distance, Hinge loss, Gaussian negative log-likelihood, etc.


class PharmacophoreMatcher:
    def __init__(self, feat_factory):
        self.feat_factory = feat_factory

    def get_feature_candidates(self, ligand: LigandConformers, pcophore):
        started = perf_counter()
        success, matches = EmbedLib.MatchPharmacophoreToMol(
            ligand.mol,
            self.feat_factory,
            pcophore
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Matched Pharm3D candidate features ligand_id=%s success=%s consensus_features=%d candidate_counts=%s elapsed_ms=%.3f",
                ligand.record.ligand_id,
                success,
                len(pcophore.getFeatures()),
                [len(candidates) for candidates in matches] if matches is not None else None,
                (perf_counter() - started) * 1000,
            )

        return matches

def embedding_pair_distances(embedding, model: PairwiseModel):
    started = perf_counter()
    coords = np.array([[f.GetPos().x, f.GetPos().y, f.GetPos().z] for f in embedding])

    if len(coords) < 2:
        logger.debug("Embedding has fewer than two features; no pair distances generated")
        return np.array([])

    diffs = coords[model.idx_a] - coords[model.idx_b]

    diffs = np.atleast_2d(diffs)

    distances = np.linalg.norm(diffs, axis=1)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Computed Pharm3D embedding pair distances features=%d pairs=%d elapsed_ms=%.3f",
            len(embedding),
            len(distances),
            (perf_counter() - started) * 1000,
        )
    return distances


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
        started = perf_counter()
        num_features = len(matches)
        attempted = 0
        rejected_atom_overlap = 0
        pruned_by_score = 0

        counter = itertools.count()
        queue = [(0.0, next(counter), [])]

        full_embeddings = []
        best_full_score = float("inf")

        while queue:
            partial_score, _, embedding = heapq.heappop(queue)

            if partial_score >= best_full_score:
                pruned_by_score += 1
                continue

            depth = len(embedding)

            if depth == num_features:
                full_embeddings.append(embedding)
                best_full_score = partial_score
                if len(full_embeddings) >= max_embeddings:
                    break
                continue

            for feat in matches[depth]:
                attempted += 1
                new_embedding = embedding + [feat]

                if len(rdMolChemicalFeatures.GetAtomMatch(new_embedding)) == 0:
                    rejected_atom_overlap += 1
                    continue

                new_score = partial_geom_score(new_embedding, pairwise_stats)

                if new_score >= best_full_score:
                    pruned_by_score += 1
                    continue

                heapq.heappush(queue, (new_score, next(counter), new_embedding))

        logger.debug(
            "Enumerated Pharm3D embeddings consensus_features=%d attempted_extensions=%d accepted_embeddings=%d rejected_atom_overlap=%d pruned_by_score=%d max_embeddings=%d elapsed_ms=%.3f",
            num_features,
            attempted,
            len(full_embeddings),
            rejected_atom_overlap,
            pruned_by_score,
            max_embeddings,
            (perf_counter() - started) * 1000,
        )
        return full_embeddings

def compute_presence_likelihood(matches, consensus, consensus_features):
    started = perf_counter()
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

    logger.debug(
        "Computed Pharm3D presence likelihood consensus_features=%d present_features=%d raw_score=%.6f normalized_score=%.6f elapsed_ms=%.3f",
        len(consensus.getFeatures()),
        sum(bool(candidates) for candidates in matches),
        presence_score,
        presence_norm,
        (perf_counter() - started) * 1000,
    )
    return presence_score, presence_norm

def compute_radial_likelihood(matches, ligand, consensus, consensus_features):
    started = perf_counter()
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
    logger.debug(
        "Computed Pharm3D radial likelihood ligand_id=%s consensus_features=%d conformers=%d raw_score=%.6f normalized_score=%.6f elapsed_ms=%.3f",
        ligand.record.ligand_id,
        len(consensus.getFeatures()),
        len(ligand.conformers),
        radial_score,
        radial_norm,
        (perf_counter() - started) * 1000,
    )
    return radial_score, radial_norm

def best_geometric_score_for_embedding(embedding, ligand, pairwise_stats):
    started = perf_counter()
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
    logger.debug(
        "Selected Pharm3D conformer geometry ligand_id=%s conformers=%d pairs=%d best_conformer_id=%s raw_score=%.6f normalized_score=%.6f elapsed_ms=%.3f",
        ligand.record.ligand_id,
        len(ligand.conformers),
        len(pairwise_stats.means),
        best_conf,
        best_score,
        geom_norm,
        (perf_counter() - started) * 1000,
    )
    return best_score, geom_norm, best_diag, best_conf

def score_embedding(
    embedding,
    ligand,
    pairwise_stats,
    presence_norm,
    radial_norm
):
    started = perf_counter()
    geom_raw, geom_norm, diag, conf = \
        best_geometric_score_for_embedding(embedding, ligand, pairwise_stats)

    final_score = (
        W_GEOM   * geom_norm +
        W_PRES   * presence_norm +
        W_RADIAL * radial_norm
    )

    result = Embedding(
        score=final_score,
        diagnostics=diag,
        conformer_id=conf,
        features=list(embedding)
    )
    logger.debug(
        "Scored Pharm3D embedding ligand_id=%s features=%d conformer_id=%s geom_normalized=%.6f presence_normalized=%.6f radial_normalized=%.6f final_score=%.6f elapsed_ms=%.3f",
        ligand.record.ligand_id,
        len(embedding),
        conf,
        geom_norm,
        presence_norm,
        radial_norm,
        final_score,
        (perf_counter() - started) * 1000,
    )
    return result

def get_ranked_embeddings(
    matches,
    ligand,
    pairwise_stats,
    consensus,
    consensus_features
):
    started = perf_counter()
    valid_embeddings = enumerate_valid_embeddings(matches, pairwise_stats)

    _, presence_norm = compute_presence_likelihood(
        matches, consensus, consensus_features
    )

    _, radial_norm = compute_radial_likelihood(
        matches, ligand, consensus, consensus_features
    )

    if not valid_embeddings:
        logger.debug(
            "No valid Pharm3D embeddings ligand_id=%s elapsed_ms=%.3f",
            ligand.record.ligand_id,
            (perf_counter() - started) * 1000,
        )
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
    logger.debug(
        "Ranked Pharm3D embeddings ligand_id=%s accepted_embeddings=%d best_score=%.6f elapsed_ms=%.3f",
        ligand.record.ligand_id,
        len(scored),
        scored[0].score,
        (perf_counter() - started) * 1000,
    )
    return scored

MAX_3D_TRAINING = 400

class Pharm3DScreening:

    def __init__(self, config: Pharm3DConfig):
        started = perf_counter()
        self.config = config
        self.feat_factory = init_feature_factory()
        self.progress = config.progress or NullProgressSink()

        self.matcher = PharmacophoreMatcher(self.feat_factory)

        self.builder = ConsensusBuilder(
            self.feat_factory,
            config.feature_consensus_ratio,
            config.pair_distance_support_ratio,
            config.proximity,
            config.tolerance,
            config.pairwise_bounds_k,
            config.max_dist
        )

        self.consensus: Pharm3DConsensus | None = None
        self.screen_dataset: List[LigandConformers] | None = None
        logger.debug(
            "Initialized Pharm3D screening feature_consensus_ratio=%.6f pair_distance_support_ratio=%.6f tolerance=%.6f proximity=%.6f pairwise_bounds_k=%.6f max_dist=%.6f num_conformers=%d energy_window=%.6f elapsed_ms=%.3f",
            config.feature_consensus_ratio,
            config.pair_distance_support_ratio,
            config.tolerance,
            config.proximity,
            config.pairwise_bounds_k,
            config.max_dist,
            config.num_conformers,
            config.energy_window,
            (perf_counter() - started) * 1000,
        )

    def set_consensus(self, ligands: List[Ligand]):
        started = perf_counter()
        logger.info("Building Pharm3D consensus ligands_received=%d", len(ligands))
        self.progress.start(
            "prepare_training_ligands",
            f"Pharm3D: preparing {len(ligands)} training ligands",
            total=len(ligands),
        )

        preparation_started = perf_counter()
        try:
            prepped_ligands = prepare_ligands_geometry(
                ligands,
                "prepare_training_ligands",
                progress=self.progress,
                num_conformers=self.config.num_conformers,
                energy_window=self.config.energy_window,
            )
        except Exception as exc:
            logger.error("Pharm3D training ligand preparation failed ligands=%d stage=ligand_preparation exception_type=%s", len(ligands), type(exc).__name__)
            raise
        logger.debug(
            "Prepared Pharm3D training ligands received=%d successful=%d elapsed_ms=%.3f",
            len(ligands),
            len(prepped_ligands),
            (perf_counter() - preparation_started) * 1000,
        )

        self.progress.finish("prepare_training_ligands", "Pharm3D training ligands ready")

        if not prepped_ligands:
            logger.warning("Pharm3D consensus could not be built because no valid training ligands were prepared")
            self.consensus = None
            return

        self.progress.start("build_consensus", "Pharm3D: building consensus", total=1)

        consensus_started = perf_counter()
        try:
            self.consensus = self.builder.run(prepped_ligands)
        except Exception as exc:
            logger.exception(
                "Pharm3D consensus generation failed prepared_ligands=%d stage=consensus_generation exception_type=%s",
                len(prepped_ligands),
                type(exc).__name__,
            )
            raise
        logger.debug("Pharm3D consensus generation elapsed_ms=%.3f", (perf_counter() - consensus_started) * 1000)
        logger.info(
            "Pharm3D consensus contains features=%d pairwise_constraints=%d training_ligands=%d elapsed_ms=%.3f",
            len(self.consensus.features),
            len(self.consensus.pairwise_stats.means),
            self.consensus.num_training_ligands,
            (perf_counter() - started) * 1000,
        )
        self.progress.advance("build_consensus")
        self.progress.finish("build_consensus", "Pharm3D consensus built")

    def set_screen_dataset(self, ligands: List[Ligand]):
        started = perf_counter()
        logger.info("Preparing Pharm3D screening dataset ligands=%d", len(ligands))
        self.progress.start(
            "prepare_screening_ligands",
            "Pharm3D: preparing screening ligands",
            total=len(ligands),
        )

        try:
            prepped_ligands = prepare_ligands_geometry(
                ligands,
                "prepare_screening_ligands",
                progress=self.progress,
                num_conformers=self.config.num_conformers,
                energy_window=self.config.energy_window,
            )
        except Exception as exc:
            logger.error("Pharm3D screening ligand preparation failed ligands=%d stage=ligand_preparation exception_type=%s", len(ligands), type(exc).__name__)
            raise

        self.screen_dataset = prepped_ligands
        self.progress.finish("prepare_screening_ligands", "Pharm3D screening dataset ready")
        logger.info(
            "Prepared Pharm3D screening dataset received=%d successful=%d elapsed_ms=%.3f",
            len(ligands),
            len(prepped_ligands),
            (perf_counter() - started) * 1000,
        )

    def compare_one_to_many(
        self,
        reference: Ligand,
        candidates: List[Ligand],
    ):
        logger.info("Starting Pharm3D one-to-many screening candidates=%d", len(candidates))
        self.set_consensus([reference])
        self.set_screen_dataset(candidates)
        return self.compare_many_to_many()

    def compare_one_to_one(
        self,
        reference: Ligand,
        candidate: Ligand,
    ):
        started = perf_counter()
        logger.info("Starting Pharm3D one-to-one screening candidate_id=%s", candidate.record.ligand_id)
        self.set_consensus([reference])
        prepared = prepare_ligands_geometry(
            [candidate],
            num_conformers=self.config.num_conformers,
            energy_window=self.config.energy_window,
        )
        if not prepared:
            logger.warning(
                "Pharm3D candidate skipped ligand_id=%s smiles=%s stage=ligand_preparation reason=no_conformers",
                candidate.record.ligand_id,
                candidate.record.smiles,
            )
            return candidate, Pharm3DResult([], [], "weighted_sum(geom, presence, radial)")

        result = self._screen_prepared(prepared[0])
        logger.info(
            "Finished Pharm3D one-to-one screening candidate_id=%s embeddings=%d best_score=%s elapsed_ms=%.3f",
            candidate.record.ligand_id,
            len(result.embeddings),
            result.embeddings[0].score if result.embeddings else None,
            (perf_counter() - started) * 1000,
        )
        return candidate, result

    def _screen_prepared(self, ligand: LigandConformers) -> Pharm3DResult:
        started = perf_counter()
        ligand_id = ligand.record.ligand_id
        try:
            matches = self.matcher.get_feature_candidates(
                ligand,
                self.consensus.pharmacophore
            )
        except Exception as exc:
            logger.exception(
                "Pharm3D matching failed ligand_id=%s smiles=%s stage=feature_matching exception_type=%s",
                ligand_id,
                ligand.record.smiles,
                type(exc).__name__,
            )
            raise

        if matches is None:
            logger.warning(
                "Pharm3D candidate has no feature mapping ligand_id=%s smiles=%s stage=feature_matching",
                ligand_id,
                ligand.record.smiles,
            )
            return Pharm3DResult([], [], "weighted_sum(geom, presence, radial)")

        candidate_assignments = math.prod(len(candidates) for candidates in matches)
        try:
            embeddings = get_ranked_embeddings(
                matches,
                ligand,
                self.consensus.pairwise_stats,
                self.consensus.pharmacophore,
                self.consensus.features
            )
        except Exception as exc:
            logger.exception(
                "Pharm3D embedding failed ligand_id=%s smiles=%s stage=embedding candidate_assignments=%d exception_type=%s",
                ligand_id,
                ligand.record.smiles,
                candidate_assignments,
                type(exc).__name__,
            )
            raise

        logger.debug(
            "Screened Pharm3D candidate ligand_id=%s attempted_assignments=%d accepted_embeddings=%d best_score=%s elapsed_ms=%.3f",
            ligand_id,
            candidate_assignments,
            len(embeddings),
            embeddings[0].score if embeddings else None,
            (perf_counter() - started) * 1000,
        )

        return Pharm3DResult(
            embeddings=embeddings,
            mappings=matches,
            scoring_function="weighted_sum(geom, presence, radial)",
        )

    def compare_many_to_many(self):
        started = perf_counter()
        if self.consensus is None:
            logger.error("Pharm3D screening cannot continue because consensus is not set")
            return None

        if self.screen_dataset is None:
            logger.error("Pharm3D screening cannot continue because screen dataset is not set")
            return None

        logger.info("Screening %d ligands with Pharm3D", len(self.screen_dataset))

        self.progress.start(
            "screen_many_to_many",
            f"Pharm3D: screening {len(self.screen_dataset)} ligands",
            total=len(self.screen_dataset),
        )

        results = []

        for lig in self.screen_dataset:
            results.append((lig, self._screen_prepared(lig)))

            self.progress.advance("screen_many_to_many")

        self.progress.finish("screen_many_to_many", "Pharm3D screening complete")

        elapsed_ms = (perf_counter() - started) * 1000
        scored = [
            (ligand, result.embeddings[0].score)
            for ligand, result in results
            if result.embeddings
        ]
        if scored:
            top_ligand, top_score = min(scored, key=lambda item: item[1])
            logger.info(
                "Finished Pharm3D screening ligands=%d scored_ligands=%d average_best_score=%.6f top_ligand_id=%s top_score=%.6f elapsed_ms=%.3f",
                len(results),
                len(scored),
                sum(score for _, score in scored) / len(scored),
                top_ligand.record.ligand_id,
                top_score,
                elapsed_ms,
            )
        else:
            logger.info(
                "Finished Pharm3D screening ligands=%d scored_ligands=0 elapsed_ms=%.3f",
                len(results),
                elapsed_ms,
            )

        return results
