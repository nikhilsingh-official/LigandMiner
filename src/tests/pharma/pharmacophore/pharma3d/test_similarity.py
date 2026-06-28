import math
import unittest

import numpy as np
from rdkit.Geometry import Point3D

from src.domain.pharm import ConsensusFeature
from src.pharma.math.gaussian_nll import gaussian_nll, gaussian_nll_vectorized, nll_to_unit_interval
from src.pharma.pharmacophore.pharma3d.consensus_builder import build_consensus
from src.pharma.pharmacophore.pharma3d.embedding import score_embedding_with_diagnostics
from src.pharma.pharmacophore.pharma3d.screening import compute_presence_likelihood

from src.tests.pharma.pharmacophore.pharma3d._helpers import feature, pairwise_model


class TestSimilarityScoring(unittest.TestCase):
    def test_identical_pair_geometry_has_zero_dissimilarity(self):
        model = pairwise_model([0], [1], [4.0], [0.5], [0.5])

        score, _ = score_embedding_with_diagnostics([feature(0), feature(4)], model)

        self.assertEqual(score, 0.0)

    def test_clearly_different_geometry_scores_worse(self):
        model = pairwise_model([0], [1], [4.0], [0.5], [0.5])

        identical, _ = score_embedding_with_diagnostics([feature(0), feature(4)], model)
        different, _ = score_embedding_with_diagnostics([feature(0), feature(8)], model)

        self.assertGreater(different, identical)

    def test_unit_interval_transform_is_bounded_and_monotonic(self):
        values = [nll_to_unit_interval(value, tau=10.0) for value in (0.0, 1.0, 100.0)]

        self.assertEqual(values[0], 0.0)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertEqual(nll_to_unit_interval(math.inf, 10.0), 1.0)

    def test_zero_standard_deviation_is_numerically_floored(self):
        scalar = gaussian_nll(1.0, 0.0, 0.0)
        total, z_scores, contributions = gaussian_nll_vectorized([1.0], [0.0], [0.0])

        self.assertTrue(math.isfinite(scalar))
        self.assertTrue(math.isfinite(total))
        self.assertTrue(np.all(np.isfinite(z_scores)))
        self.assertTrue(np.all(np.isfinite(contributions)))

    def test_presence_likelihood_is_normalized_to_unit_interval(self):
        metadata = {
            0: ConsensusFeature(0, "Acceptor", "Acceptor", Point3D(), 1.0, 0.8),
            1: ConsensusFeature(1, "Donor", "Donor", Point3D(), 1.0, 0.6),
        }
        pharmacophore = build_consensus(list(metadata.values()))

        raw, normalized = compute_presence_likelihood(
            [[feature(0)], []], pharmacophore, metadata
        )

        self.assertGreater(raw, 0.0)
        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 1.0)


if __name__ == "__main__":
    unittest.main()
