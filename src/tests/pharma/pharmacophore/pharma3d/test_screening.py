import contextlib
import io
import math
import unittest

from src.domain.ligand import LigandConformers
from src.domain.pharm import Pharm3DConfig, Pharm3DResult
from src.pharma.pharmacophore.pharma3d.screening import Pharm3DScreening

from src.tests.pharma.pharmacophore.pharma3d._helpers import ligand


class TestPharm3DScreening(unittest.TestCase):
    def setUp(self):
        self.config = Pharm3DConfig(
            num_conformers=1,
            energy_window=5.0,
            feature_consensus_ratio=0.5,
            pair_distance_support_ratio=0.5,
            proximity=2.0,
        )

    def screening(self):
        return Pharm3DScreening(self.config)

    def test_empty_public_training_set_clears_consensus(self):
        screening = self.screening()

        with contextlib.redirect_stdout(io.StringIO()):
            screening.set_consensus([])

        self.assertIsNone(screening.consensus)

    def test_empty_screening_dataset_produces_empty_results(self):
        screening = self.screening()
        screening.set_consensus([ligand("CCO", "reference-empty")])
        screening.set_screen_dataset([])

        self.assertEqual(screening.compare_many_to_many(), [])

    def test_compare_many_requires_consensus(self):
        screening = self.screening()
        screening.set_screen_dataset([])

        with contextlib.redirect_stdout(io.StringIO()):
            result = screening.compare_many_to_many()

        self.assertIsNone(result)

    def test_compare_many_requires_screen_dataset(self):
        screening = self.screening()
        screening.set_consensus([ligand("CCO", "reference-no-data")])

        with contextlib.redirect_stdout(io.StringIO()):
            result = screening.compare_many_to_many()

        self.assertIsNone(result)

    def test_one_to_one_returns_original_candidate_and_result(self):
        screening = self.screening()
        candidate = ligand("CCO", "candidate-one")

        returned, result = screening.compare_one_to_one(
            ligand("CCO", "reference-one"), candidate
        )

        self.assertIs(returned, candidate)
        self.assertIsInstance(result, Pharm3DResult)
        self.assertGreater(len(result.embeddings), 0)

    def test_one_to_many_preserves_candidate_order(self):
        screening = self.screening()
        candidates = [
            ligand("c1ccccc1", "aromatic-first"),
            ligand("CCO", "alcohol-second"),
        ]

        results = screening.compare_one_to_many(
            ligand("CCO", "reference-many"), candidates
        )

        self.assertEqual(
            [prepared.record.ligand_id for prepared, _ in results],
            ["aromatic-first", "alcohol-second"],
        )
        self.assertTrue(all(isinstance(prepared, LigandConformers) for prepared, _ in results))

    def test_duplicate_structures_are_screened_independently(self):
        screening = self.screening()
        candidates = [ligand("CCO", "duplicate-a"), ligand("CCO", "duplicate-b")]

        results = screening.compare_one_to_many(
            ligand("CCO", "reference-duplicates"), candidates
        )

        self.assertEqual([item.record.ligand_id for item, _ in results], ["duplicate-a", "duplicate-b"])
        self.assertTrue(all(result.embeddings for _, result in results))

    def test_lower_scores_rank_matching_ligand_before_absent_features(self):
        screening = self.screening()
        candidates = [ligand("c1ccccc1", "different"), ligand("CCO", "matching")]

        results = screening.compare_one_to_many(
            ligand("CCO", "reference-ranking"), candidates
        )
        scores = {
            item.record.ligand_id: (
                result.embeddings[0].score if result.embeddings else math.inf
            )
            for item, result in results
        }

        self.assertLess(scores["matching"], scores["different"])

    def test_embedding_scores_are_sorted_and_bounded(self):
        screening = self.screening()

        _, result = screening.compare_one_to_one(
            ligand("CCO", "reference-score"), ligand("OCCO", "candidate-score")
        )
        scores = [embedding.score for embedding in result.embeddings]

        self.assertEqual(scores, sorted(scores))
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))

    def test_repeated_screening_with_cached_ligands_is_deterministic(self):
        reference = ligand("CCO", "reference-repeat")
        candidate = ligand("CCO", "candidate-repeat")

        first = self.screening().compare_one_to_one(reference, candidate)[1]
        second = self.screening().compare_one_to_one(reference, candidate)[1]

        self.assertEqual(
            [embedding.score for embedding in first.embeddings],
            [embedding.score for embedding in second.embeddings],
        )

    def test_invalid_smiles_raises_explicit_value_error(self):
        screening = self.screening()

        with self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            screening.set_screen_dataset([ligand("not a smiles", "invalid")])

    def test_none_dataset_is_invalid_input(self):
        with self.assertRaises(TypeError):
            self.screening().set_screen_dataset(None)


class TestPharm3DEndToEnd(unittest.TestCase):
    def test_complete_pipeline_builds_consensus_and_screens_small_dataset(self):
        screening = Pharm3DScreening(
            Pharm3DConfig(
                num_conformers=1,
                feature_consensus_ratio=1.0,
                pair_distance_support_ratio=1.0,
                proximity=2.0,
            )
        )
        training = [ligand("CCO", "train-a"), ligand("CCO", "train-b")]
        candidates = [ligand("CCO", "hit"), ligand("c1ccccc1", "miss")]

        screening.set_consensus(training)
        screening.set_screen_dataset(candidates)
        results = screening.compare_many_to_many()

        self.assertEqual(screening.consensus.num_training_ligands, 2)
        self.assertGreater(len(screening.consensus.features), 0)
        self.assertEqual([item.record.ligand_id for item, _ in results], ["hit", "miss"])
        by_id = {item.record.ligand_id: result for item, result in results}
        self.assertGreater(len(by_id["hit"].embeddings), 0)
        self.assertEqual(by_id["miss"].embeddings, [])


if __name__ == "__main__":
    unittest.main()
