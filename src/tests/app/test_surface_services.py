import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app.progress import NullProgressSink
from src.app.surface_services import SurfaceService
from src.domain.structure import Chain, ExperimentalStructure, Pocket, StructurePaths


def structures(root):
    target = ExperimentalStructure(
        "TARGET", "PDB", chains=[Chain("B", 1, 10), Chain("A", 1, 10)],
        paths=StructurePaths(
            structure_path=root / "target.pdb",
            probis_out=root / "target_probis",
            probis_srf=root / "target_probis" / "target.srf",
        ),
    )
    similar = ExperimentalStructure(
        "HOMOLOG", "PDB", chains=[Chain("A", 1, 10)],
        paths=StructurePaths(
            structure_path=root / "homolog.pdb",
            probis_out=root / "homolog_probis",
            probis_srf=root / "homolog_probis" / "homolog.srf",
            probis_compare=root / "compare",
            probis_nosql=root / "compare" / "probis.nosql",
            probis_results=root / "compare" / "results.json",
        ),
    )
    return target, similar


class TestSurfaceResultAlgorithms(unittest.TestCase):
    def setUp(self):
        self.service = SurfaceService()

    def test_parse_aligned_pair_returns_target_and_similar_residue_refs(self):
        self.assertEqual(
            self.service.parse_aligned_pair("ALA:10:A=GLY:20:B"),
            (("A", 10, " "), ("B", 20, " ")),
        )

    def test_similarity_without_pocket_match_is_halved(self):
        parsed = {"aligned_vertices": 10, "sva": 1, "alignment_score": 2, "best_pocket_match": None}
        base = 0.6 * 10 + 0.25 * 2 + 0.15 / 2
        self.assertEqual(self.service.probis_similarity_score(parsed), base * 0.5)

    def test_similarity_with_pocket_match_applies_overlap_multiplier(self):
        parsed = {
            "aligned_vertices": 10,
            "sva": 1,
            "alignment_score": 2,
            "best_pocket_match": {
                "overlap_fraction": 0.5,
                "alignment_in_pocket_fraction": 0.25,
                "binding_site_score": 0.8,
            },
        }
        base = 0.6 * 10 + 0.25 * 2 + 0.15 / 2
        multiplier = 1 + 0.5 * 0.5 + 0.3 * 0.25 + 0.2 * 0.8
        self.assertAlmostEqual(self.service.probis_similarity_score(parsed), base * multiplier)

    def test_parse_results_selects_most_aligned_result_and_best_pocket_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, similar = structures(Path(tmp))
            similar.paths.probis_results.parent.mkdir()
            target.analysis.fpockets = [
                Pocket(1, residues=[("A", 10, " "), ("A", 11, " ")], binding_site_score=0.8),
                Pocket(2, residues=[("A", 99, " ")], binding_site_score=1.0),
            ]
            payload = [
                {
                    "alignment": [
                        {"scores": {"aligned_vertices": 1}, "aligned_residues": []},
                        {
                            "scores": {"aligned_vertices": 2, "sva": 1.5, "rmsd": 0.8, "alignment_score": 4.0, "z_score": 3.0},
                            "aligned_residues": [
                                {"a": "ALA:10:A=GLY:20:A"},
                                {"a": "ALA:12:A=GLY:22:A"},
                            ],
                        },
                    ]
                }
            ]
            similar.paths.probis_results.write_text(json.dumps(payload))

            parsed = self.service.parse_results(target, similar, NullProgressSink())

            self.assertEqual(parsed["aligned_vertices"], 2)
            self.assertEqual(parsed["aligned_target_residues"], {("A", 10, " "), ("A", 12, " ")})
            self.assertEqual(parsed["best_pocket_match"]["pocket_id"], 1)
            self.assertEqual(parsed["best_pocket_match"]["overlap_fraction"], 0.5)

    def test_empty_result_document_is_a_recoverable_no_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, similar = structures(Path(tmp))
            similar.paths.probis_results.parent.mkdir()
            similar.paths.probis_results.write_text("[]")
            self.assertIsNone(self.service.parse_results(target, similar, NullProgressSink()))


class TestSurfaceExternalBoundaries(unittest.TestCase):
    def setUp(self):
        self.service = SurfaceService()

    def test_surface_extraction_reuses_nonempty_cached_surface(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run") as run:
            target, _ = structures(Path(tmp))
            target.paths.probis_srf.parent.mkdir()
            target.paths.probis_srf.write_text("surface")

            returned = self.service.surface_extraction(target, NullProgressSink())

            self.assertEqual(returned, target.paths.probis_srf)
            run.assert_not_called()

    def test_surface_extraction_requires_prepared_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, _ = structures(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                self.service.surface_extraction(target, NullProgressSink())

    def test_surface_extraction_invokes_probis_with_sorted_chains(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run") as run:
            target, _ = structures(Path(tmp))
            target.paths.structure_path.write_text("END\n")

            self.service.surface_extraction(target, NullProgressSink())

            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["./probis", "-extract"])
            self.assertEqual(command[command.index("-c1") + 1], "A,B")

    def test_compare_reuses_existing_nosql_result(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run") as run:
            target, similar = structures(Path(tmp))
            target.paths.probis_srf.parent.mkdir()
            similar.paths.probis_srf.parent.mkdir()
            target.paths.probis_srf.write_text("target")
            similar.paths.probis_srf.write_text("similar")
            similar.paths.probis_compare.mkdir()
            similar.paths.probis_nosql.write_text("cached")

            self.service.compare_surfaces(target, similar, NullProgressSink())

            run.assert_not_called()

    def test_compare_requires_both_surfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, similar = structures(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                self.service.compare_surfaces(target, similar, NullProgressSink())

    def test_get_results_reuses_existing_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run") as run:
            target, similar = structures(Path(tmp))
            target.paths.structure_path.write_text("END\n")
            similar.paths.structure_path.write_text("END\n")
            similar.paths.probis_compare.mkdir()
            similar.paths.probis_results.write_text("[]")

            self.service.get_results(target, similar, NullProgressSink())

            run.assert_not_called()

    def test_get_results_invokes_probis_when_cache_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run") as run:
            target, similar = structures(Path(tmp))
            target.paths.structure_path.write_text("END\n")
            similar.paths.structure_path.write_text("END\n")

            self.service.get_results(target, similar, NullProgressSink())

            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["./probis", "-results"])
            self.assertEqual(command[command.index("-json") + 1], "results.json")


if __name__ == "__main__":
    unittest.main()
