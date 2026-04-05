import time
import unittest

from src.domain.ligand import Ligand, LigandRecord
from src.domain.pharm import Pharm2DConfig, Pharm2DResult
from src.pharma.pharmacophore.pharma2d.screening import Pharm2DScreening


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


class TestPharm2DPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = Pharm2DConfig()

        cls.training = [L("CCO", 1), L("CCCO", 2), L("CCCCO", 3)]

        cls.fda = [
            L(s, i+100) for i, s in enumerate([
                "CCN","CC(C)O","CCCCO","CC(C)CO","CCOC","CCN(CC)CC",
                "c1ccccc1","c1ccncc1","c1ccccc1O","c1ccccc1Cl",
                "CC(=O)O","CC(=O)N","CC(C)C(=O)O","CC(C)C(=O)N",
                "C1CCCCC1","C1CCNCC1","O=C(O)C(O)C","OCCN","NCCO","CC(C)(C)O",
            ])
        ]

    def _init_screener_with_dataset(self):
        screener = Pharm2DScreening(self.config)

        t0 = time.perf_counter()
        screener.set_screen_dataset(self.fda)
        dt = time.perf_counter() - t0

        print(f"\n[PERF] Dataset preload: {dt:.4f} sec for {len(self.fda)} ligands")
        return screener

    def test_timing_consensus_build(self):
        screener = self._init_screener_with_dataset()

        t0 = time.perf_counter()
        screener.set_consensus(self.training)
        dt = time.perf_counter() - t0

        print(f"[PERF] Consensus build: {dt:.4f} sec for {len(self.training)} ligands")

    def test_timing_screen(self):
        screener = self._init_screener_with_dataset()
        screener.set_consensus(self.training)

        t0 = time.perf_counter()
        screener.compare_many_to_many()
        dt = time.perf_counter() - t0

        per_ligand = dt / len(self.fda)

        print(
            f"[PERF] Screen: {dt:.4f} sec for {len(self.fda)} ligands "
            f"({per_ligand*1000:.3f} ms/ligand)"
        )

    def test_scaling_dataset_preload(self):
        def make_dataset(n):
            base = [
                "CCO","CCCO","CCCCO","CCN","CC(C)O","CC(C)CO","CCOC",
                "CCN(CC)CC","CC(=O)O","CC(=O)N","C1CCCCC1","C1CCNCC1",
                "OCCN","NCCO","CC(C)(C)O","c1ccccc1","c1ccncc1",
            ]
            smiles = (base * (n // len(base) + 1))[:n]
            return [L(s, i+5000) for i, s in enumerate(smiles)]

        sizes = [10, 50, 100, 500, 1000]

        for n in sizes:
            screener = Pharm2DScreening(self.config)
            dataset = make_dataset(n)

            t0 = time.perf_counter()
            screener.set_screen_dataset(dataset)
            dt = time.perf_counter() - t0

            print(
                f"[PERF] Dataset preload scaling: {n:4d} ligands -> "
                f"{dt:.3f} sec ({dt/n*1000:.3f} ms/ligand)"
            )

    def test_scaling_consensus(self):
        screener = self._init_screener_with_dataset()

        def make_training_set(n):
            base = [
                "CCO","CCCO","CCCCO","CCN","CC(C)O","CC(C)CO","CCOC",
                "CCN(CC)CC","CC(=O)O","CC(=O)N","C1CCCCC1","C1CCNCC1",
                "OCCN","NCCO","CC(C)(C)O","c1ccccc1","c1ccncc1",
            ]
            smiles = (base * (n // len(base) + 1))[:n]
            return [L(s, i+10000) for i, s in enumerate(smiles)]

        sizes = [10, 50, 100, 500, 1000]

        for n in sizes:
            ligands = make_training_set(n)

            t0 = time.perf_counter()
            screener.set_consensus(ligands)
            dt = time.perf_counter() - t0

            print(
                f"[PERF] Consensus scaling: {n:4d} ligands -> "
                f"{dt:.3f} sec ({dt/n*1000:.3f} ms/ligand)"
            )

    def test_scaling_screen(self):
        screener = self._init_screener_with_dataset()
        screener.set_consensus(self.training)

        repeats = [10, 50, 100, 500]

        for n in repeats:
            t0 = time.perf_counter()
            for _ in range(n):
                screener.compare_many_to_many()
            dt = time.perf_counter() - t0

            print(
                f"[PERF] Screen repeated {n:4d} runs -> "
                f"{dt:.3f} sec ({dt/n:.4f} sec/run)"
            )

    def test_single_ligand_consensus_is_kept(self):
        screener = Pharm2DScreening(self.config)
        reference = L("CCO", 9000)

        screener.set_consensus([reference])

        self.assertEqual(screener.consensus.pharm2d.source_id, ["9000"])
        self.assertEqual(screener.consensus.morgan.source_id, ["9000"])

    def test_compare_one_to_many(self):
        screener = Pharm2DScreening(self.config)
        reference = L("CCO", 9100)
        candidates = [L("CCO", 9101), L("CCCC", 9102)]

        results = screener.compare_one_to_many(reference, candidates)

        self.assertEqual(len(results), 2)
        scores = {lig.record.ligand_id: res for lig, res in results}
        self.assertGreater(scores["9101"].pharm2d_similarity, scores["9102"].pharm2d_similarity)
        self.assertGreater(scores["9101"].morgan_similarity, scores["9102"].morgan_similarity)

    def test_compare_one_to_one(self):
        screener = Pharm2DScreening(self.config)
        reference = L("CCO", 9200)
        candidate = L("CCCC", 9201)

        lig, result = screener.compare_one_to_one(reference, candidate)

        self.assertEqual(lig.record.ligand_id, "9201")
        self.assertIsInstance(result, Pharm2DResult)
