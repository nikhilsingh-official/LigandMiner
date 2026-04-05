from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore
from rdkit.Chem.rdChemicalFeatures import FreeChemicalFeature

from src.domain.ligand import Ligand, LigandPharmacophore, LigandConformers
from src.domain.pharm import ConsensusFeature, Pharm3DConsensus, FeatureLigandMap, Cluster, PairwiseModel
from src.pharma.pharmacophore.pharma3d.feature_clustering import build_feature_clusters
from src.pharma.pharmacophore.pharma3d.ligand_features import ligands_to_pharmacophore

def build_consensus(consensus_features: List[ConsensusFeature]):

    pcophore_feats = []

    for f in consensus_features:
        p_feat = FreeChemicalFeature(
            f.family,
            f.type,
            f.centroid
        )
        p_feat.SetId(f.id)
        pcophore_feats.append(p_feat)

    return Pharmacophore(pcophore_feats)


def filter_clusters(feature_clusters: List[Cluster], min_support: int):
     return [c for c in feature_clusters if len(c.ligand_ids) >= min_support]

import math
import numpy as np

def compute_pairwise_stats(
    clusters: List[Cluster],
    ligands: List[Ligand],
    min_support: int,
    min_std: float = 0.3
) -> PairwiseModel:

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

    return PairwiseModel(
        idx_a=np.array(idx_a, dtype=int),
        idx_b=np.array(idx_b, dtype=int),
        means=np.array(means, dtype=float),
        raw_stds=np.array(stds, dtype=float),
        model_stds=np.array(model_stds, dtype=float)
    )


class FeatureClusterer:
    def __init__(self, proximity: float):
        self.proximity = proximity

    def run(self, ligands: List[LigandPharmacophore]) -> List[Cluster]:
        return build_feature_clusters(ligands, self.proximity)


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

        n_training = len(ligands)
        min_feature_consensus_support = math.ceil(n_training * self.feature_consensus_ratio)
        min_pairwise_support = math.ceil(n_training * self.pair_distance_support_ratio)

        clusters = filter_clusters(clusters, min_feature_consensus_support)

        pairwise_stats = compute_pairwise_stats(clusters, ligands, min_pairwise_support)

        features, ligand_map = self._clusters_to_consensus_data(
            clusters,
            n_training
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

    def build(self, features, pairwise_stats) -> Pharmacophore:
        pcophore = build_consensus(features)
        self.enrich_pcophore_with_pairwise_stats(pcophore, pairwise_stats)
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
        if not ligands:
            return []

        reference = min(
            ligands,
            key=lambda lig: min(conf.energy for conf in lig.conformers)
        )
        reference_conf_id = self._lowest_energy_conformer_id(reference)
        reference_mol = self._keep_only_conformer(reference.mol, reference_conf_id)

        aligned = []

        for ligand in ligands:
            conf_id = self._lowest_energy_conformer_id(ligand)
            aligned_mol = self._keep_only_conformer(ligand.mol, conf_id)

            if ligand is not reference:
                try:
                    o3a = AllChem.GetO3A(
                        aligned_mol,
                        reference_mol,
                        prbCid=conf_id,
                        refCid=reference_conf_id,
                    )
                    o3a.Align()
                except Exception:
                    # Fall back to the original coordinates if O3A cannot build an atom map.
                    pass

            aligned.append(
                LigandConformers(
                    record=ligand.record,
                    mol=aligned_mol,
                    conformers=[conf for conf in ligand.conformers if conf.conf_id == conf_id],
                )
            )

        return aligned

    def run(self, ligands: List[LigandConformers]) -> Pharm3DConsensus:
        aligned_ligands = self._align_training_ligands(ligands)
        pcophore_ligands = ligands_to_pharmacophore(aligned_ligands, self.feat_factory)
        clusters = self.clusterer.run(pcophore_ligands)

        (
            features,
            pairwise_stats,
            ligand_map
        ) = self.stats.build(clusters, aligned_ligands)

        pharmacophore = self.rdkit.build(features, pairwise_stats)

        return Pharm3DConsensus(
            features=features,
            pairwise_stats=pairwise_stats,
            feature_to_ligands=ligand_map,
            pharmacophore=pharmacophore,
            num_training_ligands=len(ligands),
        )
