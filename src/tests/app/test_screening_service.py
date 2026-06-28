import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app.ligand_services import LigandService, ScreeningService
from src.app.progress import NullProgressSink
from src.domain.ligand import Ligand, LigandRecord
from src.domain.pharm import Pharm2DConfig, Pharm3DConfig
from src.domain.pipeline import PipelineState, ProteinData, ProteinScores


def ligand(smiles, ligand_id, affinity):
    return Ligand(
        LigandRecord("P1", f"K-{ligand_id}", ligand_id, smiles, "IC50", affinity)
    )


class TestTrainingSetSelection(unittest.TestCase):
    def test_requires_at_least_two_ligands(self):
        service = ScreeningService()
        self.assertEqual(service.select_pharm3d_training_set([]), [])
        self.assertEqual(service.select_pharm3d_training_set([ligand("CCO", "1", 1)]), [])

    def test_selects_strongest_thirty_percent_with_bounds(self):
        training = [ligand("CCO", str(index), float(index)) for index in range(1, 41)]

        selected = ScreeningService().select_pharm3d_training_set(training)

        self.assertEqual(len(selected), 12)
        self.assertEqual([item.record.affinity for item in selected], [float(index) for index in range(1, 13)])


class ScreeningDatabaseCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "resources").mkdir()
        self.db = self.root / "resources" / "ligand.db"
        connection = sqlite3.connect(self.db)
        connection.executescript(
            """
            CREATE TABLE ligand (ligand_id TEXT, inchikey TEXT, smiles TEXT);
            CREATE TABLE activity (
                protein_id TEXT, ligand_id TEXT, affinity_type TEXT,
                affinity_value REAL, p_affinity REAL
            );
            """
        )
        connection.commit()
        connection.close()
        self.ligand_service = LigandService(base_dir=self.root)

    def tearDown(self):
        self.temp.cleanup()

    def add_training(self, rows):
        connection = sqlite3.connect(self.db)
        for ligand_id, smiles, affinity in rows:
            connection.execute("INSERT INTO ligand VALUES (?, ?, ?)", (ligand_id, f"K-{ligand_id}", smiles))
            connection.execute(
                "INSERT INTO activity VALUES (?, ?, ?, ?, ?)",
                ("P1", ligand_id, "IC50", affinity, 9.0 - affinity / 1000),
            )
        connection.commit()
        connection.close()

    @staticmethod
    def pipeline():
        protein = ProteinData("P1", "UPI", [], scores=ProteinScores(probis_score=4.2))
        return PipelineState(target_protein=ProteinData("TARGET", "UPI-T", []), ranked_proteins=[protein])


class TestScreeningOrchestration(ScreeningDatabaseCase):
    def test_empty_ranked_protein_set_returns_empty_results(self):
        pipeline = PipelineState(target_protein=ProteinData("TARGET", "UPI", []))
        self.assertEqual(
            ScreeningService(self.ligand_service).screen_ranked_proteins(
                pipeline, [ligand("CCO", "FDA", 1)], NullProgressSink()
            ),
            [],
        )

    def test_protein_without_training_ligands_is_skipped(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch(
                "src.app.ligand_services.Pharm2DConfig",
                return_value=Pharm2DConfig(db_path=str(Path(cache) / "cache.db")),
            ):
                results = ScreeningService(self.ligand_service).screen_ranked_proteins(
                    self.pipeline(), [ligand("CCO", "FDA", 1)], NullProgressSink()
                )
        self.assertEqual(results, [])

    def test_single_training_ligand_runs_2d_and_preserves_training_affinity(self):
        self.add_training([("L1", "CCO", 10.0)])
        candidates = [ligand("CCO", "L1", 0), ligand("c1ccccc1", "FDA2", 0)]

        with tempfile.TemporaryDirectory() as cache, patch(
            "src.app.ligand_services.Pharm2DConfig",
            return_value=Pharm2DConfig(db_path=str(Path(cache) / "cache.db")),
        ):
            results = ScreeningService(self.ligand_service).screen_ranked_proteins(
                self.pipeline(), candidates, NullProgressSink()
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["protein"], "P1")
        self.assertEqual(results[0]["num_training_ligands"], 1)
        self.assertEqual(len(results[0]["hits"]), 1)
        hit = results[0]["hits"][0]
        self.assertEqual(hit["ligand_id"], "L1")
        self.assertEqual(hit["training_affinity_value"], 10.0)
        self.assertIsNone(hit["pharm3d_score"])

    def test_two_training_ligands_execute_real_3d_handoff(self):
        self.add_training([("T1", "CCO", 10.0), ("T2", "CCO", 20.0)])
        candidates = [ligand("CCO", "FDA-HIT", 0), ligand("c1ccccc1", "FDA-MISS", 0)]

        with tempfile.TemporaryDirectory() as cache, patch(
            "src.app.ligand_services.Pharm2DConfig",
            return_value=Pharm2DConfig(db_path=str(Path(cache) / "cache.db")),
        ), patch(
            "src.app.ligand_services.Pharm3DConfig",
            side_effect=lambda **kwargs: Pharm3DConfig(
                num_conformers=1,
                feature_consensus_ratio=kwargs["feature_consensus_ratio"],
                proximity=kwargs["proximity"],
                pair_distance_support_ratio=kwargs["pair_distance_support_ratio"],
                pairwise_bounds_k=kwargs["pairwise_bounds_k"],
                energy_window=kwargs["energy_window"],
                progress=kwargs["progress"],
            ),
        ):
            results = ScreeningService(self.ligand_service).screen_ranked_proteins(
                self.pipeline(), candidates, NullProgressSink()
            )

        self.assertEqual(len(results[0]["hits"]), 1)
        self.assertEqual(results[0]["hits"][0]["ligand_id"], "FDA-HIT")
        self.assertIsNotNone(results[0]["hits"][0]["pharm3d_score"])


if __name__ == "__main__":
    unittest.main()
