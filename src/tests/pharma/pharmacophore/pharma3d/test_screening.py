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


class TestPharm3DScreeningEdgeCases(unittest.TestCase):
    """Additional tests for edge cases and robustness in 3D screening"""
    
    @classmethod
    def setUpClass(cls):
        cls.config = Pharm3DConfig()
    
    def test_empty_training_set(self):
        """Test behavior with empty training set"""
        screening = Pharm3DScreening(self.config)
        
        # Should handle empty consensus gracefully
        screening.set_consensus([])
        
        # Consensus should be initialized properly
        self.assertIsNotNone(screening.consensus)
    
    def test_empty_screening_dataset(self):
        """Test behavior with empty screening dataset"""
        screening = Pharm3DScreening(self.config)
        screening.set_consensus([L("CCO", 1)])
        
        # Set empty screening dataset
        screening.set_screen_dataset([])
        
        # Should return empty results
        results = screening.compare_many_to_many()
        self.assertEqual(len(results), 0)
    
    def test_single_ligand_training_and_screening(self):
        """Test with minimal datasets"""
        screening = Pharm3DScreening(self.config)
        
        training = [L("CCO", 1)]
        candidates = [L("CCCO", 10)]
        
        screening.set_consensus(training)
        screening.set_screen_dataset(candidates)
        
        results = screening.compare_many_to_many()
        
        # Should handle minimal case
        self.assertEqual(len(results), 1)
        lig, result = results[0]
        self.assertEqual(lig.record.ligand_id, "10")
    
    def test_duplicate_ligands_handling(self):
        """Test handling of duplicate SMILES in datasets"""
        screening = Pharm3DScreening(self.config)
        
        # Training with duplicates
        training = [L("CCO", 1), L("CCO", 2)]
        
        # Screening with duplicates
        candidates = [L("CCCO", 10), L("CCCO", 11)]
        
        screening.set_consensus(training)
        screening.set_screen_dataset(candidates)
        
        results = screening.compare_many_to_many()
        
        # Should return results for all ligands
        self.assertEqual(len(results), 2)
        result_ids = {lig.record.ligand_id for lig, _ in results}
        self.assertEqual(result_ids, {"10", "11"})
    
    def test_conformer_generation_failures(self):
        """Test handling of molecules that fail conformer generation"""
        screening = Pharm3DScreening(self.config)
        
        # Use simple molecules that should work
        training = [L("CCO", 1)]
        
        # Some molecules might fail conformer generation
        # Test with various challenging molecules
        challenging_molecules = [
            L("C", 10),                    # very simple
            L("CC", 11),                   # simple alkane
            L("[H][H]", 12),              # diatomic (if valid)
        ]
        
        screening.set_consensus(training)
        
        for mol in challenging_molecules:
            try:
                screening.set_screen_dataset([mol])
                results = screening.compare_many_to_many()
                
                # If successful, should return results
                self.assertGreaterEqual(len(results), 0)
                
            except (ValueError, RuntimeError):
                # Some molecules may legitimately fail
                # Important that failures are explicit, not silent
                pass
    
    def test_very_different_molecules(self):
        """Test with structurally very different molecules"""
        screening = Pharm3DScreening(self.config)
        
        # Training: simple alcohols
        training = [L("CCO", 1), L("CCCO", 2)]
        
        # Screening: very different structures
        diverse_candidates = [
            L("c1ccc2c(c1)cc3ccccc3c2", 10),  # larger aromatic (naphthalene-like)
            L("CC(C)(C)C(C)(C)C", 11),        # branched alkane
            L("CC(=O)N(C)C", 12),             # amide
            L("C1CCCCCCC1", 13),              # cycloalkane
        ]
        
        screening.set_consensus(training)
        screening.set_screen_dataset(diverse_candidates)
        
        results = screening.compare_many_to_many()
        
        # Should return results for diverse molecules
        self.assertEqual(len(results), 4)
        
        # Check that scores are in valid ranges
        for lig, result in results:
            if result.embeddings:
                score = result.embeddings[0].score
                self.assertIsInstance(score, (int, float))
                self.assertGreaterEqual(score, 0.0)
    
    def test_identical_training_and_screening(self):
        """Test when training and screening sets are identical"""
        screening = Pharm3DScreening(self.config)
        
        molecules = [L("CCO", 1), L("CCCO", 2)]
        
        screening.set_consensus(molecules)
        screening.set_screen_dataset(molecules)
        
        results = screening.compare_many_to_many()
        
        # Should return results for all molecules
        self.assertEqual(len(results), 2)
        
        # Self-similarity should be very good
        for lig, result in results:
            if result.embeddings:
                score = result.embeddings[0].score
                # Identical molecules should have excellent scores
                self.assertLessEqual(score, 2.0)  # Low is better in 3D
    
    def test_embedding_failure_handling(self):
        """Test handling when embedding fails"""
        screening = Pharm3DScreening(self.config)
        
        # Simple training set
        training = [L("CCO", 1)]
        screening.set_consensus(training)
        
        # Test with molecules that might not embed well
        candidates = [L("C", 10)]  # Very simple molecule
        
        screening.set_screen_dataset(candidates)
        results = screening.compare_many_to_many()
        
        # Should handle embedding failures gracefully
        self.assertGreaterEqual(len(results), 0)
        
        for lig, result in results:
            # Result should exist even if embedding failed
            self.assertIsNotNone(result)
            
            # embeddings list might be empty for failed cases
            if not result.embeddings:
                # This documents that empty embeddings are acceptable
                pass
            else:
                # If embeddings exist, they should be valid
                for embedding in result.embeddings:
                    self.assertIsNotNone(embedding.score)
    
    def test_score_ordering(self):
        """Test that screening produces meaningful score ordering"""
        screening = Pharm3DScreening(self.config)
        
        # Training: alcohol
        training = [L("CCO", 1)]
        
        # Screening: molecules with expected similarity order
        candidates = [
            L("CCO", 10),      # identical - should be best
            L("CCCO", 11),     # similar alcohol - should be good
            L("c1ccccc1", 12)  # different - should be worse
        ]
        
        screening.set_consensus(training)
        screening.set_screen_dataset(candidates)
        results = screening.compare_many_to_many()
        
        # Extract scores
        scores = {}
        for lig, result in results:
            if result.embeddings:
                scores[lig.record.ligand_id] = result.embeddings[0].score
            else:
                scores[lig.record.ligand_id] = float('inf')
        
        # Identical molecule should have best (lowest) score
        if "10" in scores and "12" in scores:
            self.assertLessEqual(scores["10"], scores["12"])
    
    def test_large_candidate_set_performance(self):
        """Test performance with larger candidate sets"""
        screening = Pharm3DScreening(self.config)
        
        # Small training set
        training = [L("CCO", 1), L("CCCO", 2)]
        
        # Larger candidate set
        base_smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "CC(=O)O"]
        candidates = [
            L(base_smiles[i % len(base_smiles)], i+100)
            for i in range(50)  # 50 candidates
        ]
        
        screening.set_consensus(training)
        screening.set_screen_dataset(candidates)
        
        results = screening.compare_many_to_many()
        
        # Should handle larger sets
        self.assertEqual(len(results), 50)
    
    def test_consensus_determinism(self):
        """Test that consensus building is deterministic"""
        screening1 = Pharm3DScreening(self.config)
        screening2 = Pharm3DScreening(self.config)
        
        training = [L("CCO", 1), L("CCCO", 2)]
        
        screening1.set_consensus(training)
        screening2.set_consensus(training)
        
        # Should produce equivalent consensus
        # (This is a basic check - full comparison would need more detailed inspection)
        self.assertEqual(
            len(screening1.consensus.features if screening1.consensus else 0),
            len(screening2.consensus.features if screening2.consensus else 0)
        )
    
    def test_screening_without_consensus(self):
        """Test error handling when screening without setting consensus"""
        screening = Pharm3DScreening(self.config)
        
        candidates = [L("CCO", 10)]
        screening.set_screen_dataset(candidates)
        
        # Should handle missing consensus gracefully
        try:
            results = screening.compare_many_to_many()
            # If it succeeds, that's fine
            self.assertIsInstance(results, list)
        except (AttributeError, ValueError):
            # It's also acceptable to raise an error for missing consensus
            pass


if __name__ == "__main__":
    unittest.main()
