import logging
from time import perf_counter
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore
from rdkit.Chem.rdChemicalFeatures import FreeChemicalFeature

from src.domain.ligand import Ligand, LigandPharmacophore, LigandConformers
from src.domain.pharm import ConsensusFeature, Pharm3DConsensus, FeatureLigandMap, Cluster, PairwiseModel
from src.pharma.pharmacophore.pharma3d.feature_clustering import build_feature_clusters
from src.pharma.pharmacophore.pharma3d.ligand_features import ligands_to_pharmacophore


logger = logging.getLogger(__name__)

def build_consensus(consensus_features: List[ConsensusFeature]):
    started = perf_counter()
    pcophore_feats = []

    for f in consensus_features:
        p_feat = FreeChemicalFeature(
            f.family,
            f.type,
            f.centroid
        )
        p_feat.SetId(f.id)
        pcophore_feats.append(p_feat)

    pharmacophore = Pharmacophore(pcophore_feats)
    logger.debug(
        "Built RDKit consensus feature container features=%d elapsed_ms=%.3f",
        len(pcophore_feats),
        (perf_counter() - started) * 1000,
    )
    return pharmacophore


def filter_clusters(feature_clusters: List[Cluster], min_support: int):
     filtered = [c for c in feature_clusters if len(c.ligand_ids) >= min_support]
     logger.debug(
         "Filtered Pharm3D feature clusters input_clusters=%d retained_clusters=%d minimum_support=%d",
         len(feature_clusters),
         len(filtered),
         min_support,
     )
     return filtered

import math
import numpy as np

def compute_pairwise_stats(
    clusters: List[Cluster],
    ligands: List[Ligand],
    min_support: int,
    min_std: float = 0.3
) -> PairwiseModel:
    started = perf_counter()
    logger.debug(
        "Computing Pharm3D pairwise statistics clusters=%d ligands=%d minimum_support=%d minimum_std=%.6f",
        len(clusters),
        len(ligands),
        min_support,
        min_std,
    )
    idx_a = []
    idx_b = []
    means = []
    stds  = []
    model_stds = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):

            distances = []
            supporting_ligands = 0

            for ligand in ligands:
                f1s = [cf for cf in clusters[i].features if cf.ligand_id == ligand.record.ligand_id]
                f2s = [cf for cf in clusters[j].features if cf.ligand_id == ligand.record.ligand_id]

                if not f1s or not f2s:
                    continue

                supporting_ligands += 1

                for f1 in f1s:
                    for f2 in f2s:
                        distances.append(
                            f1.feature.GetPos().Distance(
                                f2.feature.GetPos()
                            )
                        )

            if supporting_ligands < min_support:
                continue

            n = len(distances)
            if n < 2:
                continue

            mean_d = sum(distances) / n
            var = sum((d - mean_d) ** 2 for d in distances) / (n - 1)
            std_d = math.sqrt(var)
            model_std = max(std_d, min_std)

            idx_a.append(i)
            idx_b.append(j)
            means.append(mean_d)
            stds.append(std_d)
            model_stds.append(model_std)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Generated Pharm3D pairwise constraint feature_a=%d feature_b=%d supporting_ligands=%d observations=%d mean=%.6f raw_std=%.6f model_std=%.6f",
                    i,
                    j,
                    supporting_ligands,
                    n,
                    mean_d,
                    std_d,
                    model_std,
                )

    model = PairwiseModel(
        idx_a=np.array(idx_a, dtype=int),
        idx_b=np.array(idx_b, dtype=int),
        means=np.array(means, dtype=float),
        raw_stds=np.array(stds, dtype=float),
        model_stds=np.array(model_stds, dtype=float)
    )
    logger.debug(
        "Finished Pharm3D pairwise statistics constraints=%d elapsed_ms=%.3f",
        len(model.means),
        (perf_counter() - started) * 1000,
    )
    return model


class FeatureClusterer:
    def __init__(self, proximity: float):
        self.proximity = proximity

    def run(self, ligands: List[LigandPharmacophore]) -> List[Cluster]:
        started = perf_counter()
        clusters = build_feature_clusters(ligands, self.proximity)
        logger.debug(
            "FeatureClusterer produced clusters=%d ligands=%d proximity=%.6f elapsed_ms=%.3f",
            len(clusters),
            len(ligands),
            self.proximity,
            (perf_counter() - started) * 1000,
        )
        return clusters


class PharmacophoreStatistics:
    def __init__(self, feature_consensus_ratio: float, pair_distance_support_ratio: float, feature_radius_k: float):
        self.feature_consensus_ratio = feature_consensus_ratio
        self.pair_distance_support_ratio = pair_distance_support_ratio
        self.feature_radius_k = feature_radius_k

    def _clusters_to_consensus_data(
        self,
        clusters: List[Cluster],
        n_training: int
    ) -> Tuple[List[ConsensusFeature], FeatureLigandMap]:

        features = []
        feature_ligand_map: FeatureLigandMap = {}

        for i, cluster in enumerate(clusters):
            radius = self.feature_radius_k * cluster.stddev

            ligand_ids = list(cluster.ligand_ids)
            feature_ligand_map[i] = ligand_ids

            feature = ConsensusFeature(
                id=i,
                family=cluster.family,
                type=cluster.type,
                centroid=cluster.centroid,
                radius=radius,
                support=len(cluster.ligand_ids) / n_training
            )

            features.append(feature)

        return features, feature_ligand_map

    def build(
        self,
        clusters: List[Cluster],
        ligands: List[LigandConformers],
    ) -> tuple[
        List[ConsensusFeature],
        PairwiseModel,
        FeatureLigandMap,
    ]:

        started = perf_counter()
        n_training = len(ligands)
        min_feature_consensus_support = math.ceil(n_training * self.feature_consensus_ratio)
        min_pairwise_support = math.ceil(n_training * self.pair_distance_support_ratio)

        logger.debug(
            "Building Pharm3D statistics training_ligands=%d clusters=%d feature_consensus_ratio=%.6f feature_min_support=%d pair_support_ratio=%.6f pair_min_support=%d feature_radius_k=%.6f",
            n_training,
            len(clusters),
            self.feature_consensus_ratio,
            min_feature_consensus_support,
            self.pair_distance_support_ratio,
            min_pairwise_support,
            self.feature_radius_k,
        )

        clusters = filter_clusters(clusters, min_feature_consensus_support)

        pairwise_stats = compute_pairwise_stats(clusters, ligands, min_pairwise_support)

        features, ligand_map = self._clusters_to_consensus_data(
            clusters,
            n_training
        )

        logger.debug(
            "Built Pharm3D statistics consensus_features=%d pairwise_constraints=%d elapsed_ms=%.3f",
            len(features),
            len(pairwise_stats.means),
            (perf_counter() - started) * 1000,
        )
        return features, pairwise_stats, ligand_map


class RDKitPharmacophoreBuilder:
    def __init__(self, pairwise_bounds_k: float, max_dist: float):
        self.k = pairwise_bounds_k
        self.max_dist = max_dist

    def enrich_pcophore_with_pairwise_stats(
        self,
        pcophore: Pharmacophore,
        model: PairwiseModel
    ) -> None:

        started = perf_counter()
        n = len(pcophore.getFeatures())

        for i in range(n):
            for j in range(i + 1, n):
                pcophore.setLowerBound(i, j, 0.0)
                pcophore.setUpperBound(i, j, self.max_dist)

        LOWER_EPS = 0.1

        for i, j, mu, sigma in zip(
            model.idx_a,
            model.idx_b,
            model.means,
            model.model_stds
        ):
            lower = max(LOWER_EPS, mu - self.k * sigma)
            upper = mu + self.k * sigma

            pcophore.setLowerBound(int(i), int(j), float(lower))
            pcophore.setUpperBound(int(i), int(j), float(upper))

        logger.debug(
            "Applied Pharm3D pairwise bounds features=%d default_pairs=%d statistical_pairs=%d k=%.6f max_dist=%.6f elapsed_ms=%.3f",
            n,
            n * (n - 1) // 2,
            len(model.means),
            self.k,
            self.max_dist,
            (perf_counter() - started) * 1000,
        )

    def build(self, features, pairwise_stats) -> Pharmacophore:
        started = perf_counter()
        pcophore = build_consensus(features)
        self.enrich_pcophore_with_pairwise_stats(pcophore, pairwise_stats)
        logger.debug(
            "Built RDKit Pharm3D pharmacophore features=%d pairwise_constraints=%d elapsed_ms=%.3f",
            len(features),
            len(pairwise_stats.means),
            (perf_counter() - started) * 1000,
        )
        return pcophore


class ConsensusBuilder:
    def __init__(self, feat_factory, feature_consensus_ratio, pair_distance_support_ratio, proximity, feature_k, pairwise_k, max_dist):
        self.feat_factory = feat_factory
        self.clusterer = FeatureClusterer(proximity)
        self.stats = PharmacophoreStatistics(feature_consensus_ratio, pair_distance_support_ratio, feature_k)
        self.rdkit = RDKitPharmacophoreBuilder(pairwise_k, max_dist)

    @staticmethod
    def _lowest_energy_conformer_id(ligand: LigandConformers) -> int:
        return min(ligand.conformers, key=lambda c: c.energy).conf_id

    @staticmethod
    def _keep_only_conformer(mol, conf_id: int):
        mol = Chem.Mol(mol)
        for existing_id in [conf.GetId() for conf in mol.GetConformers()]:
            if existing_id != conf_id:
                mol.RemoveConformer(existing_id)
        return mol

    def _align_training_ligands(
        self,
        ligands: List[LigandConformers]
    ) -> List[LigandConformers]:
        started = perf_counter()
        if not ligands:
            logger.warning("Cannot align an empty Pharm3D training set")
            return []

        #Select ligand with conformer with loewst global energy
        reference = min(
            ligands,
            key=lambda lig: min(conf.energy for conf in lig.conformers)
        )
        #Extract the ID of that low-energy conformer
        reference_conf_id = self._lowest_energy_conformer_id(reference)
        logger.debug(
            "Selected Pharm3D alignment reference ligand_id=%s conformer_id=%d",
            reference.record.ligand_id,
            reference_conf_id,
        )
        #Delete all other conformers
        reference_mol = self._keep_only_conformer(reference.mol, reference_conf_id)

        aligned = []

        for ligand in ligands:
            #Keep lowest energy lowest conformer for each ligand
            conf_id = self._lowest_energy_conformer_id(ligand)
            aligned_mol = self._keep_only_conformer(ligand.mol, conf_id)

            if ligand is not reference:
                try:
                    #Along conformer to reference conformer
                    o3a = AllChem.GetO3A(
                        aligned_mol,
                        reference_mol,
                        prbCid=conf_id,
                        refCid=reference_conf_id,
                    )
                    o3a.Align()
                except Exception as exc:
                    # Fall back to the original coordinates if O3A cannot build an atom map.
                    logger.warning(
                        "Pharm3D alignment fallback ligand_id=%s reference_ligand_id=%s exception_type=%s explanation=%s",
                        ligand.record.ligand_id,
                        reference.record.ligand_id,
                        type(exc).__name__,
                        exc,
                    )
                    pass

            aligned.append(
                LigandConformers(
                    record=ligand.record,
                    mol=aligned_mol,
                    conformers=[conf for conf in ligand.conformers if conf.conf_id == conf_id],
                )
            )

        logger.debug(
            "Aligned Pharm3D training ligands input_ligands=%d aligned_ligands=%d elapsed_ms=%.3f",
            len(ligands),
            len(aligned),
            (perf_counter() - started) * 1000,
        )
        return aligned

    def run(self, ligands: List[LigandConformers]) -> Pharm3DConsensus:
        started = perf_counter()
        logger.info("Building Pharm3D consensus ligands=%d", len(ligands))
        #Align ligands lowest-energy conformers to each other - this also deletes other conformers
        alignment_started = perf_counter()
        aligned_ligands = self._align_training_ligands(ligands)
        logger.debug("Pharm3D consensus alignment elapsed_ms=%.3f", (perf_counter() - alignment_started) * 1000)
        #Convert ligands to LigandPharmacophore objects
        extraction_started = perf_counter()
        pcophore_ligands = ligands_to_pharmacophore(aligned_ligands, self.feat_factory)
        logger.debug("Pharm3D feature extraction elapsed_ms=%.3f", (perf_counter() - extraction_started) * 1000)

        clustering_started = perf_counter()
        clusters = self.clusterer.run(pcophore_ligands)
        logger.debug("Pharm3D consensus clustering clusters=%d elapsed_ms=%.3f", len(clusters), (perf_counter() - clustering_started) * 1000)

        statistics_started = perf_counter()
        (
            features,
            pairwise_stats,
            ligand_map
        ) = self.stats.build(clusters, aligned_ligands)
        logger.debug(
            "Pharm3D consensus statistics features=%d pairwise_constraints=%d elapsed_ms=%.3f",
            len(features),
            len(pairwise_stats.means),
            (perf_counter() - statistics_started) * 1000,
        )

        build_started = perf_counter()
        pharmacophore = self.rdkit.build(features, pairwise_stats)
        logger.debug("Pharm3D RDKit pharmacophore build elapsed_ms=%.3f", (perf_counter() - build_started) * 1000)

        consensus = Pharm3DConsensus(
            features=features,
            pairwise_stats=pairwise_stats,
            feature_to_ligands=ligand_map,
            pharmacophore=pharmacophore,
            num_training_ligands=len(ligands),
        )
        logger.info(
            "Built Pharm3D consensus ligands=%d features=%d clusters=%d pairwise_constraints=%d elapsed_ms=%.3f",
            len(ligands),
            len(features),
            len(clusters),
            len(pairwise_stats.means),
            (perf_counter() - started) * 1000,
        )
        return consensus
