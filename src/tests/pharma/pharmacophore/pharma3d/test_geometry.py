import unittest

import numpy as np
from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore

from src.domain.ligand import LigandPharmacophore
from src.pharma.pharmacophore.pharma3d.feature_clustering import (
    build_feature_clusters,
    get_updated_params,
    is_near_to_centroid,
    is_same_feature,
)
from src.pharma.pharmacophore.pharma3d.geometry import embedding_pair_distances

from src.tests.pharma.pharmacophore.pharma3d._helpers import (
    cluster,
    feature,
    pairwise_model,
    point_tuple,
    prepared_ligand,
)


class TestGeometry(unittest.TestCase):
    def test_extracts_distances_for_model_pairs_in_model_order(self):
        embedding = [feature(0, 0, 0), feature(3, 4, 0), feature(3, 4, 12)]
        model = pairwise_model(idx_a=[0, 1], idx_b=[1, 2])

        np.testing.assert_allclose(embedding_pair_distances(embedding, model), [5.0, 12.0])

    def test_empty_pair_model_produces_empty_distance_vector(self):
        distances = embedding_pair_distances([feature(0)], pairwise_model())
        self.assertEqual(distances.shape, (0,))

    def test_online_centroid_and_population_spread_are_exact(self):
        base = cluster("Acceptor", "Acceptor", [("a", (0.0, 0.0, 0.0))])

        centroid, stddev, m2 = get_updated_params(base, feature(2.0).GetPos())

        self.assertEqual(point_tuple(centroid), (1.0, 0.0, 0.0))
        self.assertEqual(stddev, 1.0)
        self.assertEqual(m2, 2.0)

    def test_proximity_is_strict_and_feature_identity_uses_family_and_type(self):
        base = cluster("Acceptor", "Acceptor", [("a", (0.0, 0.0, 0.0))])

        self.assertTrue(is_near_to_centroid(feature(0.99), base, proximity=1.0))
        self.assertFalse(is_near_to_centroid(feature(1.0), base, proximity=1.0))
        self.assertTrue(is_same_feature(feature(0.0), base))
        self.assertFalse(is_same_feature(feature(0.0, family="Donor"), base))

    def test_clustering_merges_nearby_matching_features_and_tracks_unique_support(self):
        first = prepared_ligand("CCO", "same-id")
        second = prepared_ligand("CCO", "same-id")
        ligands = [
            LigandPharmacophore(**first.__dict__, pharmacophore=Pharmacophore([feature(0.0)])),
            LigandPharmacophore(**second.__dict__, pharmacophore=Pharmacophore([feature(1.0)])),
        ]

        clusters = build_feature_clusters(ligands, proximity=2.0)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0].features), 2)
        self.assertEqual(clusters[0].ligand_ids, {"same-id"})
        self.assertEqual(point_tuple(clusters[0].centroid), (0.5, 0.0, 0.0))
        self.assertEqual(clusters[0].stddev, 0.5)

    def test_clustering_separates_distant_or_different_features(self):
        prepared = prepared_ligand("CCO", "lig")
        pharmacophore = Pharmacophore(
            [feature(0.0), feature(2.0), feature(0.1, family="Donor")]
        )
        ligand = LigandPharmacophore(**prepared.__dict__, pharmacophore=pharmacophore)

        clusters = build_feature_clusters([ligand], proximity=1.0)

        self.assertEqual(len(clusters), 3)


if __name__ == "__main__":
    unittest.main()
