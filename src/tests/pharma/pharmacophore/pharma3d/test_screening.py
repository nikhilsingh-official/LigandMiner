import pathlib
import sys
import unittest
import math
import time
from collections import defaultdict
from contextlib import contextmanager

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.ligand import Ligand, LigandRecord
from src.domain.pharm import Pharm3DConfig
from src.pharma.pharmacophore.pharma3d.screening import Pharm3DScreening
from rich.progress import Progress

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


class StageTimer:
    def __init__(self):
        self.times = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def track(self, name):
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.times[name] += dt
        self.calls[name] += 1

    def report(self):
        print("\n==============================")
        print("TIMING BREAKDOWN")
        print("==============================")
        for k in sorted(self.times, key=self.times.get, reverse=True):
            total = self.times[k]
            calls = self.calls[k]
            avg = total / calls
            print(f"{k:30s} | total={total:8.3f}s | calls={calls:5d} | avg={avg:7.4f}s")


class TestPharm3DScreeningDiagnostics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = Pharm3DConfig(
            progress=Progress(transient=True),
            feature_consensus_ratio=0.2,
            pair_distance_support_ratio=0.1,
            tolerance=1.0,
            proximity=0.5,
            pairwise_bounds_k=3
        )

        cls.timer = StageTimer()

        # Base chemistry motifs
        base_smiles = [
            "CO","CCO","CCCO","CCCCO","CCCCCO","CCCCCCO",
            "CCCOC","CCCCOC","CCCOCCO","CCOCCO",
            "CC(C)O","CCC(C)O","CC(C)(C)O","CC(C)CO","CC(C)(C)CO",
            "CC(C)C(C)O","CCC(C)(C)O",
            "OCCO","OCCCO","OCCCCO","CC(O)CO","CC(O)C(O)C",
            "C(CO)CO","C(CO)(CO)O",
            "OCCN","OCCCN","OCCNCCO","NCCO","NCCCO",
            "CCNCO","CCN(CC)CO",
            "CC(O)C=O","O=CCCO","CC(O)CC=O","O=CC(O)CO",
            "CC(O)C(=O)O","CC(O)C(=O)N",
            "C1CCOCC1","C1CCCOC1","C1CC(O)CC1","C1CCC(O)CC1",
            "COCCO","CCOCCCO","COCCCO","COCCOCCO"
        ]

        # Realistic training set (~200 ligands)
        cls.training = [
            L(base_smiles[i % len(base_smiles)], i)
            for i in range(200)
        ]

        # Realistic candidate library (~500 ligands)
        candidate_smiles = base_smiles + [
            "CCN", "CCCN", "CCN(CC)C",
            "CC(=O)O", "CC(=O)N", "CC(=O)Cl",
            "c1ccccc1", "Clc1ccccc1", "c1ccncc1",
            "CCOC(=O)C", "CCCOC(=O)C",
            "CC(C)CO", "CC(C)C(C)O",
        ]

        cls.candidates = [
            L(candidate_smiles[i % len(candidate_smiles)], 1000 + i)
            for i in range(500)
        ]

    def test_run_and_profile_pipeline(self):
        screening = Pharm3DScreening(self.config)
        screening = Pharm3DScreening(self.config)

        with self.timer.track("consensus_build"):
            screening.set_consensus(self.training)

        with self.timer.track("dataset_preparation"):
            screening.set_screen_dataset(self.candidates)

        with self.timer.track("screen_total"):
            results = screening.compare_many_to_many()

        # Score extraction timing
        with self.timer.track("score_extraction"):
            scored = []
            for lig, res in results:
                if not res.embeddings:
                    score = float("inf")
                else:
                    score = res.embeddings[0].score
                scored.append((lig.record.ligand_id, lig.record.smiles, score))

        scored.sort(key=lambda x: x[2])

        print("\n==============================")
        print("TOP 15 RESULTS")
        print("==============================")

        for rank, (lig_id, smi, score) in enumerate(scored[:15], 1):
            print(f"{rank:02d} | score={score:.4f} | {smi}")

        best_score = scored[0][2]
        worst_score = scored[-1][2]

        self.assertTrue(math.isfinite(best_score))
        self.assertTrue(best_score < worst_score)

        self.timer.report()

    def test_score_distribution(self):
        screening = Pharm3DScreening(self.config)

        with self.timer.track("consensus_build"):
            screening.set_consensus(self.training)

        with self.timer.track("dataset_preparation"):
            screening.set_screen_dataset(self.candidates)

        with self.timer.track("screen_total"):
            results = screening.compare_many_to_many()

        with self.timer.track("score_distribution_calc"):
            scores = []
            for _, res in results:
                if res.embeddings:
                    scores.append(res.embeddings[0].score)

        print("\nScore distribution:", scores[:20], "...")

        self.assertGreater(max(scores) - min(scores), 0.05)

        self.timer.report()


if __name__ == "__main__":
    unittest.main()
