import unittest

import numpy as np
from rdkit.Chem import rdMolChemicalFeatures

from src.pharma.pharmacophore.common.feature_factory import init_feature_factory
from src.pharma.pharmacophore.pharma3d.embedding import score_embedding_with_diagnostics
from src.pharma.pharmacophore.pharma3d.screening import (
    embedding_pair_distances,
    enumerate_valid_embeddings,
    partial_geom_score,
)

from src.tests.pharma.pharmacophore.pharma3d._helpers import (
    feature,
    pairwise_model,
    prepared_ligand,
)


class TestEmbeddingGeometry(unittest.TestCase):
    def test_pair_distances_follow_pairwise_indices(self):
        embedding = [feature(0, 0, 0), feature(0, 3, 4), feature(12, 3, 4)]
        model = pairwise_model([0, 1], [1, 2])

        np.testing.assert_allclose(embedding_pair_distances(embedding, model), [5.0, 12.0])

    def test_single_feature_has_no_pair_distances(self):
        distances = embedding_pair_distances([feature(0)], pairwise_model())
        self.assertEqual(distances.shape, (0,))

    def test_diagnostic_score_is_exact_gaussian_nll(self):
        embedding = [feature(0), feature(5)]
        model = pairwise_model([0], [1], [4.0], [2.0], [2.0])

        score, diagnostics = score_embedding_with_diagnostics(embedding, model)

        self.assertEqual(score, 0.125)
        np.testing.assert_allclose(diagnostics.observed_distances, [5.0])
        np.testing.assert_allclose(diagnostics.z_scores, [0.5])
        np.testing.assert_allclose(diagnostics.nll_contributions, [0.125])

    def test_empty_pair_model_scores_zero(self):
        score, diagnostics = score_embedding_with_diagnostics([feature(0)], pairwise_model())

        self.assertEqual(score, 0.0)
        self.assertEqual(diagnostics.observed_distances.shape, (0,))

    def test_partial_score_uses_only_pairs_whose_features_are_present(self):
        model = pairwise_model(
            [0, 0, 1], [1, 2, 2], [3.0, 50.0, 50.0], [1, 1, 1], [1, 1, 1]
        )

        self.assertEqual(partial_geom_score([feature(0), feature(4)], model), 0.5)

    def test_scoring_is_deterministic(self):
        embedding = [feature(0), feature(5)]
        model = pairwise_model([0], [1], [4.0], [2.0], [2.0])

        first = score_embedding_with_diagnostics(embedding, model)
        second = score_embedding_with_diagnostics(embedding, model)

        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1].z_scores, second[1].z_scores)


class TestEmbeddingEnumeration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = init_feature_factory().GetFeaturesForMol(
            prepared_ligand("OCCO", "diol").mol
        )

    def test_finds_an_atom_disjoint_embedding(self):
        donors = [f for f in self.features if f.GetFamily() == "Donor"]
        acceptors = [f for f in self.features if f.GetFamily() == "Acceptor"]

        embeddings = enumerate_valid_embeddings([donors, acceptors], pairwise_model())

        self.assertEqual(len(embeddings), 1)
        self.assertGreater(len(rdMolChemicalFeatures.GetAtomMatch(embeddings[0])), 0)

    def test_rejects_embedding_that_reuses_the_same_atom(self):
        donor = next(f for f in self.features if f.GetFamily() == "Donor")
        acceptor = next(
            f
            for f in self.features
            if f.GetFamily() == "Acceptor" and f.GetAtomIds() == donor.GetAtomIds()
        )

        embeddings = enumerate_valid_embeddings([[donor], [acceptor]], pairwise_model())

        self.assertEqual(embeddings, [])

    def test_missing_feature_candidates_make_full_embedding_impossible(self):
        donor = next(f for f in self.features if f.GetFamily() == "Donor")

        self.assertEqual(enumerate_valid_embeddings([[donor], []], pairwise_model()), [])


if __name__ == "__main__":
    unittest.main()
