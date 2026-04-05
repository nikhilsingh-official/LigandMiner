import pathlib
import sys
import unittest

from rich.progress import Progress

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.ligand import Ligand, LigandRecord
from src.domain.pharm import Pharm3DConfig, Pharm3DResult
from src.pharma.pharmacophore.pharma3d.screening import Pharm3DScreening


def L(smiles, i):
    return Ligand(
        record=LigandRecord(
            protein_id="test",
            ligand_id=str(i),
            affinity_type="IC50",
            affinity=100.0,
            smiles=smiles,
            inchi_key=str(i),
        )
    )


class TestPharm3DConsensusAlignment(unittest.TestCase):
    def test_single_ligand_training_builds_consensus(self):
        screening = Pharm3DScreening(
            Pharm3DConfig(
                progress=Progress(transient=True),
                feature_consensus_ratio=0.7,
                pair_distance_support_ratio=0.3,
                tolerance=1.0,
                proximity=0.5,
                pairwise_bounds_k=3,
            )
        )

        screening.set_consensus([L("CCO", 1)])

        self.assertIsNotNone(screening.consensus)
        self.assertEqual(screening.consensus.num_training_ligands, 1)
        self.assertGreater(len(screening.consensus.features), 0)

    def test_repeated_ligand_training_does_not_collapse_to_empty_consensus(self):
        training = [L("CCO", i) for i in range(20)]
        candidates = [L("CCO", 1000 + i) for i in range(5)]

        screening = Pharm3DScreening(
            Pharm3DConfig(
                progress=Progress(transient=True),
                feature_consensus_ratio=0.7,
                pair_distance_support_ratio=0.3,
                tolerance=1.0,
                proximity=0.5,
                pairwise_bounds_k=3,
            )
        )

        screening.set_consensus(training)

        self.assertGreater(len(screening.consensus.features), 0)
        self.assertGreater(len(screening.consensus.pairwise_stats.idx_a), 0)

        screening.set_screen_dataset(candidates)
        results = screening.compare_many_to_many()

        scores = [res.embeddings[0].score for _, res in results if res.embeddings]

        self.assertTrue(scores)
        self.assertTrue(all(score < 0.25 for score in scores))

    def test_compare_one_to_many(self):
        screening = Pharm3DScreening(
            Pharm3DConfig(
                progress=Progress(transient=True),
                feature_consensus_ratio=0.7,
                pair_distance_support_ratio=0.3,
                tolerance=1.0,
                proximity=0.5,
                pairwise_bounds_k=3,
            )
        )

        results = screening.compare_one_to_many(
            L("CCO", 2000),
            [L("CCO", 2001), L("c1ccccc1", 2002)],
        )

        self.assertEqual(len(results), 2)
        scores = {}
        for lig, res in results:
            scores[lig.record.ligand_id] = res.embeddings[0].score if res.embeddings else float("inf")

        self.assertLess(scores["2001"], scores["2002"])

    def test_compare_one_to_one(self):
        screening = Pharm3DScreening(
            Pharm3DConfig(
                progress=Progress(transient=True),
                feature_consensus_ratio=0.7,
                pair_distance_support_ratio=0.3,
                tolerance=1.0,
                proximity=0.5,
                pairwise_bounds_k=3,
            )
        )

        lig, res = screening.compare_one_to_one(
            L("CCO", 4000),
            L("CCO", 4001),
        )

        self.assertEqual(lig.record.ligand_id, "4001")
        self.assertIsInstance(res, Pharm3DResult)


if __name__ == "__main__":
    unittest.main()
