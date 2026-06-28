import math
import unittest

import numpy as np
from rdkit.Chem import Conformer as RDConformer
from rdkit.Geometry import Point3D

from src.domain.ligand import Conformer
from src.domain.pharm import ConsensusFeature
from src.pharma.pharmacophore.common.feature_factory import init_feature_factory
from src.pharma.pharmacophore.pharma3d.consensus_builder import (
    ConsensusBuilder,
    PharmacophoreStatistics,
    RDKitPharmacophoreBuilder,
    build_consensus,
    compute_pairwise_stats,
    filter_clusters,
)

from src.tests.pharma.pharmacophore.pharma3d._helpers import (
    cluster,
    ligand,
    pairwise_model,
    point_tuple,
    prepared_ligand,
)


def consensus_feature(feature_id, x, family="Acceptor", radius=0.5, support=1.0):
    return ConsensusFeature(
        id=feature_id,
        family=family,
        type=family,
        centroid=Point3D(float(x), 0.0, 0.0),
        radius=radius,
        support=support,
    )


class TestConsensusStatistics(unittest.TestCase):
    def test_build_consensus_preserves_feature_order_metadata_and_ids(self):
        features = [consensus_feature(7, 0.0), consensus_feature(9, 3.0, "Donor")]

        pharmacophore = build_consensus(features)
        rd_features = pharmacophore.getFeatures()

        self.assertEqual([(f.GetId(), f.GetFamily()) for f in rd_features], [(7, "Acceptor"), (9, "Donor")])
        self.assertEqual(point_tuple(rd_features[1].GetPos()), (3.0, 0.0, 0.0))

    def test_support_filter_is_inclusive_and_keeps_input_order(self):
        clusters = [
            cluster("Acceptor", "Acceptor", [("a", (0, 0, 0))]),
            cluster("Donor", "Donor", [("a", (0, 0, 0)), ("b", (0, 0, 0))]),
            cluster("Aromatic", "Aromatic", [("a", (0, 0, 0)), ("b", (0, 0, 0)), ("c", (0, 0, 0))]),
        ]

        self.assertEqual(filter_clusters(clusters, 2), clusters[1:])

    def test_pairwise_mean_and_sample_standard_deviation(self):
        clusters = [
            cluster("Acceptor", "Acceptor", [("a", (0, 0, 0)), ("b", (0, 0, 0))]),
            cluster("Donor", "Donor", [("a", (3, 0, 0)), ("b", (5, 0, 0))]),
        ]

        model = compute_pairwise_stats(clusters, [ligand("CCO", "a"), ligand("CCO", "b")], 2)

        np.testing.assert_array_equal(model.idx_a, [0])
        np.testing.assert_array_equal(model.idx_b, [1])
        np.testing.assert_allclose(model.means, [4.0])
        np.testing.assert_allclose(model.raw_stds, [math.sqrt(2.0)])
        np.testing.assert_allclose(model.model_stds, [math.sqrt(2.0)])

    def test_pairwise_model_applies_minimum_standard_deviation_floor(self):
        clusters = [
            cluster("Acceptor", "Acceptor", [("a", (0, 0, 0)), ("b", (1, 0, 0))]),
            cluster("Donor", "Donor", [("a", (3, 0, 0)), ("b", (4, 0, 0))]),
        ]

        model = compute_pairwise_stats(
            clusters, [ligand("CCO", "a"), ligand("CCO", "b")], 2, min_std=0.4
        )

        np.testing.assert_allclose(model.raw_stds, [0.0])
        np.testing.assert_allclose(model.model_stds, [0.4])

    def test_pairwise_model_omits_pairs_below_ligand_support(self):
        clusters = [
            cluster("Acceptor", "Acceptor", [("a", (0, 0, 0)), ("b", (0, 0, 0))]),
            cluster("Donor", "Donor", [("a", (3, 0, 0)), ("a", (4, 0, 0))]),
        ]

        model = compute_pairwise_stats(clusters, [ligand("CCO", "a"), ligand("CCO", "b")], 2)

        self.assertEqual(model.means.shape, (0,))

    def test_pairwise_statistics_are_deterministic(self):
        clusters = [
            cluster("Acceptor", "Acceptor", [("a", (0, 0, 0)), ("b", (0, 0, 0))]),
            cluster("Donor", "Donor", [("a", (3, 0, 0)), ("b", (5, 0, 0))]),
        ]
        ligands = [ligand("CCO", "a"), ligand("CCO", "b")]

        first = compute_pairwise_stats(clusters, ligands, 2)
        second = compute_pairwise_stats(clusters, ligands, 2)

        for field in ("idx_a", "idx_b", "means", "raw_stds", "model_stds"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))

    def test_statistics_apply_ceil_support_and_compute_radius(self):
        clusters = [
            cluster("Acceptor", "Acceptor", [("a", (0, 0, 0)), ("b", (2, 0, 0))]),
            cluster("Donor", "Donor", [("a", (0, 0, 0))]),
        ]
        stats = PharmacophoreStatistics(0.5, 1.0, feature_radius_k=2.5)

        features, pairs, ligand_map = stats.build(
            clusters,
            [prepared_ligand("CCO", "a"), prepared_ligand("CCO", "b"), prepared_ligand("CCO", "c")],
        )

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].support, 2 / 3)
        self.assertEqual(features[0].radius, 2.5)
        self.assertCountEqual(ligand_map[0], ["a", "b"])
        self.assertEqual(pairs.means.shape, (0,))


class TestPharmacophoreBounds(unittest.TestCase):
    def test_unmodelled_pairs_receive_default_bounds(self):
        builder = RDKitPharmacophoreBuilder(pairwise_bounds_k=2.0, max_dist=12.0)

        pharmacophore = builder.build(
            [consensus_feature(0, 0), consensus_feature(1, 3), consensus_feature(2, 6)],
            pairwise_model(),
        )

        for i, j in ((0, 1), (0, 2), (1, 2)):
            self.assertEqual(pharmacophore.getLowerBound(i, j), 0.0)
            self.assertEqual(pharmacophore.getUpperBound(i, j), 12.0)

    def test_modelled_pairs_receive_mean_plus_or_minus_k_sigma(self):
        builder = RDKitPharmacophoreBuilder(pairwise_bounds_k=2.0, max_dist=12.0)

        pharmacophore = builder.build(
            [consensus_feature(0, 0), consensus_feature(1, 3)],
            pairwise_model([0], [1], [4.0], [0.5], [0.5]),
        )

        self.assertEqual(pharmacophore.getLowerBound(0, 1), 3.0)
        self.assertEqual(pharmacophore.getUpperBound(0, 1), 5.0)

    def test_statistical_lower_bound_has_positive_floor(self):
        builder = RDKitPharmacophoreBuilder(pairwise_bounds_k=2.0, max_dist=12.0)

        pharmacophore = builder.build(
            [consensus_feature(0, 0), consensus_feature(1, 0.2)],
            pairwise_model([0], [1], [0.2], [0.3], [0.3]),
        )

        self.assertAlmostEqual(pharmacophore.getLowerBound(0, 1), 0.1)
        self.assertAlmostEqual(pharmacophore.getUpperBound(0, 1), 0.8)


class TestConsensusBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = ConsensusBuilder(init_feature_factory(), 0.5, 0.5, 2.0, 1.0, 2.0, 10.0)

    def test_empty_training_set_builds_empty_internal_consensus(self):
        consensus = self.builder.run([])

        self.assertEqual(consensus.features, [])
        self.assertEqual(consensus.num_training_ligands, 0)
        self.assertEqual(consensus.pairwise_stats.means.shape, (0,))
        self.assertEqual(len(consensus.pharmacophore.getFeatures()), 0)

    def test_single_ligand_builds_supported_features(self):
        consensus = self.builder.run([prepared_ligand("CCO", "one")])

        self.assertEqual(consensus.num_training_ligands, 1)
        self.assertGreater(len(consensus.features), 0)
        self.assertTrue(all(item.support == 1.0 for item in consensus.features))

    def test_duplicate_structures_with_distinct_ids_count_as_distinct_support(self):
        consensus = self.builder.run(
            [prepared_ligand("CCO", "one"), prepared_ligand("CCO", "two")]
        )

        self.assertGreater(len(consensus.features), 0)
        self.assertTrue(any(len(ids) == 2 for ids in consensus.feature_to_ligands.values()))

    def test_repeated_build_is_deterministic_for_prepared_input(self):
        training = [prepared_ligand("CCO", "one"), prepared_ligand("CCO", "two")]

        first = self.builder.run(training)
        second = self.builder.run(training)

        first_signature = [
            (f.family, f.type, point_tuple(f.centroid), f.radius, f.support) for f in first.features
        ]
        second_signature = [
            (f.family, f.type, point_tuple(f.centroid), f.radius, f.support) for f in second.features
        ]
        self.assertEqual(first_signature, second_signature)

    def test_alignment_keeps_only_each_ligands_lowest_energy_conformer(self):
        prepared = prepared_ligand("CCO", "multi")
        second_conf = RDConformer(prepared.mol.GetConformer(0))
        second_id = prepared.mol.AddConformer(second_conf, assignId=True)
        prepared.conformers = [
            Conformer(0, True, 5.0),
            Conformer(second_id, True, -1.0),
        ]

        aligned = self.builder._align_training_ligands([prepared])

        self.assertEqual([c.conf_id for c in aligned[0].conformers], [second_id])
        self.assertEqual([c.GetId() for c in aligned[0].mol.GetConformers()], [second_id])


if __name__ == "__main__":
    unittest.main()
