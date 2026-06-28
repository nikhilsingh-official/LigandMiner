import unittest
from unittest.mock import Mock

from src.app.protein_services import StructureMetadataService
from src.domain.structure import (
    AlphaFoldStructure,
    AlphaFoldStructureMetadata,
    Chain,
    ExperimentalStructure,
    ExperimentalStructureMetadata,
    HomologyStructure,
    HomologyStructureMetadata,
    Pocket,
)
from src.domain.uniprot import Region


class TestStructureProviderParsing(unittest.TestCase):
    def setUp(self):
        self.service = StructureMetadataService(Mock())

    def test_swiss_model_parser_filters_methods_and_maps_metadata(self):
        payload = {
            "result": {
                "structures": [
                    {"method": "NOT A MODEL"},
                    {
                        "method": "HOMOLOGY MODELLING",
                        "from": 10,
                        "to": 90,
                        "chains": [{"id": "A"}],
                        "coverage": 0.8,
                        "gmqe": 0.7,
                        "identity": 45.0,
                        "similarity": 0.6,
                        "qmean": {"avg_local_score": 0.75, "qmean6_z_score": -1.0},
                        "provider": "SWISS-MODEL",
                        "template": "1abc.1.A",
                        "modelcif": "https://example/model.cif",
                    },
                ]
            }
        }

        structures = self.service.parse_swiss_model_intermediate_json(payload, "P1")

        self.assertEqual(len(structures), 1)
        structure = structures[0]
        self.assertEqual(structure.struct_id, "SM-P1-1")
        self.assertEqual((structure.chains[0].chain, structure.chains[0].start, structure.chains[0].end), ("A", 10, 90))
        self.assertEqual(structure.metadata.gmqe, 0.7)
        self.assertEqual(structure.cif, "https://example/model.cif")

    def test_alphafold_parser_filters_other_providers(self):
        payload = {
            "structures": [
                {"summary": {"provider": "Other"}},
                {
                    "summary": {
                        "provider": "AlphaFold DB",
                        "entities": [{"chain_ids": ["A", "B"]}],
                        "uniprot_start": 1,
                        "uniprot_end": 120,
                        "coverage": 0.9,
                        "confidence_type": "pLDDT",
                        "confidence_avg_local_score": 80.0,
                        "model_url": "https://example/af.cif",
                    }
                },
            ]
        }

        structures = self.service.parse_alpha_fold_intermediate_json(payload, "P1")

        self.assertEqual(len(structures), 1)
        self.assertEqual([chain.chain for chain in structures[0].chains], ["A", "B"])
        self.assertEqual(structures[0].metadata.avg_local_score, 80.0)


class TestStructureScoring(unittest.TestCase):
    def setUp(self):
        self.service = StructureMetadataService(Mock())

    def test_pdb_xray_score_uses_resolution_and_method_weights(self):
        structure = ExperimentalStructure(
            "1ABC", "PDB", metadata=ExperimentalStructureMetadata(method="X-ray", resolution=2.0)
        )

        scored = self.service.score_pdb_structure(structure)

        raw = 0.7 * ((1 / 2.0) / (1 / 1.2)) + 0.3 * 1.0
        self.assertEqual(scored.analysis.structural_plausibility, round(raw ** 0.75, 4))

    def test_pdb_nmr_score_does_not_require_resolution(self):
        structure = ExperimentalStructure(
            "1ABC", "PDB", metadata=ExperimentalStructureMetadata(method="NMR", resolution=None)
        )
        self.assertEqual(
            self.service.score_pdb_structure(structure).analysis.structural_plausibility,
            round(0.6 ** 0.75, 4),
        )

    def test_alphafold_score_is_weighted_calibrated_and_multiplied(self):
        structure = AlphaFoldStructure(
            "AF-P1", "AlphaFoldDB", metadata=AlphaFoldStructureMetadata(avg_local_score=80, coverage=0.5)
        )

        score = self.service.score_alpha_fold_structure(structure).analysis.structural_plausibility

        self.assertEqual(score, round((0.7 * 0.8 + 0.3 * 0.5) ** 1.25 * 0.6, 4))

    def test_swiss_model_score_includes_all_available_terms(self):
        structure = HomologyStructure(
            "SM-P1", "SMR",
            metadata=HomologyStructureMetadata(
                gmqe=0.8, identity=50, similarity=0.7, avg_local_score=0.6, qmean6_z=-3
            ),
        )
        structure.analysis.domain_coverage = 0.75

        score = self.service.score_swiss_model_structure(structure).analysis.structural_plausibility

        raw = 0.25 * 0.8 + 0.25 * 0.75 + 0.15 * 0.5 + 0.15 * 0.7 + 0.10 * 0.6 + 0.10 * 0.5
        self.assertEqual(score, round(raw ** 1.05 * 0.8, 4))

    def test_structure_selection_score_requires_both_structure_and_pocket(self):
        structure = ExperimentalStructure("1ABC", "PDB")
        structure.analysis.structural_plausibility = 0.8
        structure.analysis.true_pocket_certainty = 0.9

        self.assertEqual(self.service.structure_selection_score(structure), 0.0)

        structure.analysis.fpockets = [Pocket(1, binding_site_score=0.5)]
        expected = (0.8 ** 0.7) * (0.5 ** 0.3) * 0.9
        self.assertAlmostEqual(self.service.structure_selection_score(structure), expected)


class TestStructureEnrichment(unittest.TestCase):
    def setUp(self):
        self.service = StructureMetadataService(Mock())

    def test_feature_enrichment_uses_largest_domain_and_chain_coverage(self):
        structure = ExperimentalStructure(
            "1ABC", "PDB", chains=[Chain("A", 20, 80), Chain("B", 1, 10)]
        )

        result = self.service.feature_structure_enrichment(
            structure,
            [Region(1, 10), Region(20, 100)],
            [Region(25, 25), Region(90, 90)],
        )

        self.assertAlmostEqual(result.analysis.domain_coverage, 61 / 81)
        self.assertEqual(result.analysis.active_site_coverage, 0.5)

    def test_feature_enrichment_without_chains_leaves_analysis_unchanged(self):
        structure = ExperimentalStructure("1ABC", "PDB")
        structure.analysis.domain_coverage = 0.4

        result = self.service.feature_structure_enrichment(structure, [Region(1, 10)], [])

        self.assertEqual(result.analysis.domain_coverage, 0.4)

    def test_add_paths_builds_all_artifact_locations(self):
        structure = ExperimentalStructure("1ABC", "PDB")

        result = self.service.add_paths("P1", structure)

        self.assertEqual(str(result.paths.raw_structure_path), "structures/P1/1ABC/1ABC_raw.cif")
        self.assertEqual(str(result.paths.probis_srf), "structures/P1/1ABC/probis_out/1ABC.srf")
        self.assertEqual(str(result.paths.probis_results), "probis_compare/P1/1ABC/results.json")

    def test_swiss_model_metadata_enrichment_scores_computed_domain_coverage(self):
        payload = {
            "result": {
                "structures": [
                    {
                        "method": "HOMOLOGY MODELLING",
                        "from": 1,
                        "to": 100,
                        "chains": [{"id": "A"}],
                        "coverage": 1.0,
                        "gmqe": 0.8,
                        "identity": 50.0,
                        "similarity": 0.7,
                        "qmean": {"avg_local_score": 0.6},
                        "modelcif": "https://example/model.cif",
                    }
                ]
            }
        }
        session = Mock()
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = payload
        service = StructureMetadataService(session)

        result = service.metadata_structure_enrichment(
            HomologyStructure("P1", "SMR"), [Region(1, 100)], [], "P1"
        )

        raw = 0.25 * 0.8 + 0.25 * 1.0 + 0.15 * 0.5 + 0.15 * 0.7 + 0.10 * 0.6
        self.assertEqual(result.analysis.domain_coverage, 1.0)
        self.assertEqual(result.analysis.structural_plausibility, round(raw ** 1.05 * 0.8, 4))


if __name__ == "__main__":
    unittest.main()
