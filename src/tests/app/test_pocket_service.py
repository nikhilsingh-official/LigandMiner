import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.app.services.pocket_service import PocketService
from src.domain.pipeline import ProteinAnnotations, ProteinData, ProteinStructures
from src.domain.structure import Chain, ExperimentalStructure, Pocket, StructurePaths
from src.domain.uniprot import Region, SequenceAnnotations


def protein_with_structure(item, domains=None, active_sites=None):
    return ProteinData(
        "P1",
        "UPI",
        [],
        annotations=ProteinAnnotations(SequenceAnnotations(domains=domains or [], active_sites=active_sites or [])),
        structures=ProteinStructures([item]),
    )


class TestPocketParsingAndExecution(unittest.TestCase):
    def setUp(self):
        self.mapper = Mock()
        self.service = PocketService(self.mapper)

    def test_residues_to_custom_pocket_format_preserves_insertion_codes(self):
        self.assertEqual(
            self.service.residues_to_fpocket_p([("A", 10, ""), ("B", 20, "C")]),
            "10:-:A.20:C:B",
        )

    def test_parse_fpocket_info_maps_known_numeric_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = Path(tmp) / "sample_info.txt"
            info.write_text(
                "Pocket 1\nDruggability Score: 0.75\nVolume: 320\nIgnored: text\n"
                "Pocket 2\nScore: 1.5\n"
            )

            pockets = self.service.parse_fpocket_info(Path(tmp))

            self.assertEqual([p.pocket_id for p in pockets], [1, 2])
            self.assertEqual(pockets[0].druggability_score, 0.75)
            self.assertEqual(pockets[0].volume, 320.0)
            self.assertEqual(pockets[1].score, 1.5)

    def test_parse_fpocket_info_rejects_missing_info_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "No fpocket info"):
                self.service.parse_fpocket_info(Path(tmp))

    def test_run_fpocket_moves_generated_output_to_configured_location(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run") as run:
            root = Path(tmp)
            pdb = root / "protein.pdb"
            pdb.write_text("MODEL\nEND\n")
            generated = root / "protein_out"
            generated.mkdir()
            (generated / "protein_info.txt").write_text("Pocket 1\nScore: 1\n")
            desired = root / "fpocket_out"
            item = ExperimentalStructure(
                "1ABC", "PDB", paths=StructurePaths(structure_path=pdb, fpocket_out=desired)
            )
            item.analysis.true_pocket = [("A", 10, "")]

            returned = self.service.run_fpocket(item)

            self.assertEqual(returned, desired)
            self.assertTrue((desired / "protein_info.txt").exists())
            self.assertEqual(run.call_args.args[0][:3], ["fpocket", "-f", "protein.pdb"])

    def test_missing_structure_is_recoverable_and_clears_pockets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = ExperimentalStructure(
                "1ABC",
                "PDB",
                paths=StructurePaths(
                    structure_path=root / "missing.pdb",
                    fpocket_txt=root / "out" / "info.txt",
                    fpocket_out=root / "out",
                ),
            )
            item.analysis.fpockets = [Pocket(1)]

            result = self.service.fpocket_enrichment(protein_with_structure(item))

            self.assertEqual(result.structures.structures[0].analysis.fpockets, [])


class TestPocketMappingAndScoring(unittest.TestCase):
    def setUp(self):
        self.mapper = Mock()
        self.service = PocketService(self.mapper)

    def test_mapping_prefers_sifts_then_uses_chain_offset_fallback(self):
        item = ExperimentalStructure("1ABC", "PDB", chains=[Chain("A", 100, 200), Chain("B", 50, 150)])
        item.analysis.sifts_mapping = {"chain_id": "A"}
        item.analysis.fpockets = [Pocket(1, residues=[("A", 5, ""), ("B", 2, "")])]
        self.mapper.map_pdb_to_uniprot.return_value = 500

        result = self.service.map_residues_to_uniprot(protein_with_structure(item))

        self.assertEqual(result.structures.structures[0].analysis.fpockets[0].uniprot_indices, [("A", 500, ""), ("B", 51, "")])

    def test_mapping_missing_chain_raises(self):
        item = ExperimentalStructure("1ABC", "PDB", chains=[Chain("A", 1, 20)])
        item.analysis.fpockets = [Pocket(1, residues=[("Z", 2, "")])]

        with self.assertRaises(StopIteration):
            self.service.map_residues_to_uniprot(protein_with_structure(item))

    def test_score_pockets_uses_domain_active_site_druggability_and_surface_fraction(self):
        item = ExperimentalStructure("1ABC", "PDB")
        residues = [("A", index, "") for index in range(1, 11)]
        item.analysis.fpockets = [Pocket(1, uniprot_indices=residues, druggability_score=0.8)]
        protein = protein_with_structure(item, [Region(1, 100)], [Region(5, 5)])

        result = self.service.score_pockets(protein)

        pocket = result.structures.structures[0].analysis.fpockets[0]
        self.assertEqual(pocket.pocket_domain_purity, 1.0)
        self.assertEqual(pocket.active_site_coverage, 1.0)
        self.assertEqual(pocket.binding_site_score, 0.95)

    def test_score_pockets_leaves_small_or_unmapped_pockets_unscored(self):
        item = ExperimentalStructure("1ABC", "PDB")
        item.analysis.fpockets = [Pocket(1, uniprot_indices=[("A", 1, "")])]

        result = self.service.score_pockets(protein_with_structure(item, [Region(1, 10)]))

        self.assertEqual(result.structures.structures[0].analysis.fpockets[0].binding_site_score, 0.0)

    def test_prank_returns_highest_probability_row_and_parses_insertion_code(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run"):
            root = Path(tmp)
            csv = root / "predictions.csv"
            csv.write_text("probability,residue_ids\n0.2,A_1\n0.9,A_10 B_20C\n")
            item = ExperimentalStructure(
                "1ABC", "PDB", paths=StructurePaths(structure_path=root / "in.pdb", prank_out=root / "out", prank_csv=csv)
            )

            probability, residues = self.service.get_pocket_from_prank(item)

            self.assertEqual(probability, 0.9)
            self.assertEqual(residues, [("A", 10, ""), ("B", 20, "C")])


if __name__ == "__main__":
    unittest.main()
