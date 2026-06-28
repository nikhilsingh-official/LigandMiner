import unittest
from unittest.mock import patch

from src.app.progress import NullProgressSink
from src.domain.pharm import Pharm3DConfig
from src.pharma.pharmacophore.pharma3d.screening import Pharm3DScreening

from src.tests.pharma.pharmacophore.pharma3d._helpers import ligand


class TestPharm3DConstruction(unittest.TestCase):
    def test_initializes_empty_screening_state(self):
        screening = Pharm3DScreening(Pharm3DConfig())

        self.assertIsNone(screening.consensus)
        self.assertIsNone(screening.screen_dataset)
        self.assertIsInstance(screening.progress, NullProgressSink)
        self.assertIs(screening.matcher.feat_factory, screening.feat_factory)

    def test_propagates_consensus_configuration(self):
        config = Pharm3DConfig(
            feature_consensus_ratio=0.61,
            pair_distance_support_ratio=0.42,
            tolerance=0.73,
            proximity=1.17,
            pairwise_bounds_k=2.4,
            max_dist=14.0,
        )

        builder = Pharm3DScreening(config).builder

        self.assertEqual(builder.clusterer.proximity, config.proximity)
        self.assertEqual(builder.stats.feature_consensus_ratio, config.feature_consensus_ratio)
        self.assertEqual(builder.stats.pair_distance_support_ratio, config.pair_distance_support_ratio)
        self.assertEqual(builder.stats.feature_radius_k, config.tolerance)
        self.assertEqual(builder.rdkit.k, config.pairwise_bounds_k)
        self.assertEqual(builder.rdkit.max_dist, config.max_dist)

    def test_propagates_geometry_configuration_to_preparation(self):
        config = Pharm3DConfig(num_conformers=3, energy_window=1.25)
        screening = Pharm3DScreening(config)

        with patch(
            "src.pharma.pharmacophore.pharma3d.screening.prepare_ligands_geometry",
            return_value=[],
        ) as prepare:
            screening.set_screen_dataset([ligand("CCO", "candidate")])

        self.assertEqual(prepare.call_args.kwargs["num_conformers"], 3)
        self.assertEqual(prepare.call_args.kwargs["energy_window"], 1.25)

    def test_rejects_unknown_configuration_fields(self):
        with self.assertRaises(TypeError):
            Pharm3DConfig(does_not_exist=True)

    def test_rejects_non_configuration_constructor_input(self):
        with self.assertRaises(AttributeError):
            Pharm3DScreening(None)


if __name__ == "__main__":
    unittest.main()
