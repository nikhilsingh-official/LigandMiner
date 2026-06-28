import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.app.progress import NullProgressSink
from src.app.protein_services import StructureMetadataService
from src.app.services.pocket_service import PocketService
from src.app.services.structure_services import StructureSelectionService
from src.domain.pipeline import ProteinAnnotations, ProteinData, ProteinStructures
from src.domain.structure import Chain, ExperimentalStructure, ExperimentalStructureMetadata, Pocket
from src.domain.uniprot import Region, SequenceAnnotations


MINIMAL_MMCIF = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . ALA A 1 1 ? 0 0 0 1 10 ? 1 ALA A CA 1
"""


class FileBoundaryDownloader:
    def download_structures(self, protein, progress, task_key, root):
        for structure in protein.structures.structures:
            structure.paths.raw_structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure.paths.raw_structure_path.write_text(MINIMAL_MMCIF)
        return list(protein.structures.structures)

    def clean_structure(self, input_cif, output_pdb, structure):
        output_pdb.write_text("END\n")


class DeterministicPocketBoundary:
    def __init__(self):
        self.scorer = PocketService(Mock())

    def get_pocket_from_prank(self, structure):
        return 0.8, [("A", index, "") for index in range(1, 11)]

    def fpocket_enrichment(self, protein):
        for structure in protein.structures.structures:
            structure.analysis.fpockets = [Pocket(1, druggability_score=0.8)]
        return protein

    def pocket_residue_enrichment(self, protein):
        residues = [("A", index, "") for index in range(1, 11)]
        for structure in protein.structures.structures:
            structure.analysis.fpockets[0].residues = residues
            structure.analysis.fpockets[0].uniprot_indices = residues
        return protein

    def score_pockets(self, protein):
        return self.scorer.score_pockets(protein)


class TestStructureSelectionAlgorithms(unittest.TestCase):
    def setUp(self):
        self.service = StructureSelectionService(
            session_obj=Mock(),
            metadata_service=StructureMetadataService(Mock()),
            pocket_service=DeterministicPocketBoundary(),
            structure_downloader=FileBoundaryDownloader(),
        )

    def test_shortlist_orders_descending_and_truncates(self):
        structures = [ExperimentalStructure(str(index), "PDB") for index in range(4)]
        for structure, score in zip(structures, [0.1, 0.9, 0.4, 0.7]):
            structure.analysis.structural_plausibility = score

        selected = self.service.shortlist_structures(structures, top_n=2)

        self.assertEqual([item.struct_id for item in selected], ["1", "3"])

    def test_motif_chain_extraction_is_sorted_and_rejects_missing_chains(self):
        self.assertEqual(self.service.get_chains_from_motif("[:B and 2 or :A and 1]"), "A,B")
        with self.assertRaisesRegex(ValueError, "No chains"):
            self.service.get_chains_from_motif("[1-10]")

    def test_residues_to_motif_compacts_contiguous_ranges(self):
        data = {
            "pockets": [
                {
                    "binding_site_score": 1.0,
                    "residues": [("A", 1, ""), ("A", 2, ""), ("A", 4, ""), ("B", 7, "")],
                }
            ]
        }
        self.assertEqual(self.service.residues_to_motif(data), "[:A and (1-2,4) or :B and (7)]")

    def test_structure_without_hetero_ligand_has_no_holo_pocket(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "apo.cif"
            path.write_text(MINIMAL_MMCIF)
            self.assertIsNone(self.service.is_holo(path, [Chain("A", 1, 10)]))


class TestStructureSelectionPipeline(unittest.TestCase):
    def test_selects_highest_combined_structure_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structures = [
                ExperimentalStructure(
                    "BEST", "PDB", chains=[Chain("A", 1, 100)],
                    metadata=ExperimentalStructureMetadata(method="X-ray", resolution=1.5),
                ),
                ExperimentalStructure(
                    "OTHER", "PDB", chains=[Chain("A", 1, 100)],
                    metadata=ExperimentalStructureMetadata(method="X-ray", resolution=3.0),
                ),
            ]
            protein = ProteinData(
                "P1", "UPI", [],
                annotations=ProteinAnnotations(SequenceAnnotations(domains=[Region(1, 100)])),
                structures=ProteinStructures(structures),
            )
            service = StructureSelectionService(
                session_obj=Mock(),
                metadata_service=StructureMetadataService(Mock()),
                pocket_service=DeterministicPocketBoundary(),
                structure_downloader=FileBoundaryDownloader(),
            )

            with patch("src.app.protein_services.STRUCTURES_DIR", root / "structures"), patch(
                "src.app.protein_services.COMPARISON_DIR", root / "compare"
            ):
                selected = service.select_best_structure(protein, NullProgressSink())

            self.assertEqual(selected.struct_id, "BEST")
            self.assertGreater(selected.analysis.structural_plausibility, structures[1].analysis.structural_plausibility)
            self.assertGreater(selected.analysis.fpockets[0].binding_site_score, 0.0)


if __name__ == "__main__":
    unittest.main()
