import unittest
from unittest.mock import Mock

from src.app.progress import NullProgressSink
from src.app.protein_services import UniProtService
from src.domain.pipeline import ProteinData
from src.domain.structure import AlphaFoldStructure, ExperimentalStructure, HomologyStructure
from src.domain.uniprot import (
    ExtraAttributes,
    Feature,
    FeatureLocation,
    FeatureLocationRange,
    Property,
    UniProtEntry,
    UniProtKBCrossReference,
)


def entry(accession="P12345", cross_references=None, features=None):
    return UniProtEntry(
        entryType="UniProtKB reviewed (Swiss-Prot)",
        primaryAccession=accession,
        extraAttributes=ExtraAttributes(uniParcId=f"UPI-{accession}"),
        uniProtKBCrossReferences=cross_references,
        features=features,
    )


def response(*, status=200, payload=None, headers=None, ok=None):
    result = Mock()
    result.status_code = status
    result.ok = status < 400 if ok is None else ok
    result.json.return_value = payload if payload is not None else {}
    result.headers = headers or {}
    result.raise_for_status.side_effect = None if status < 400 else RuntimeError("http error")
    return result


class TestUniProtRetrieval(unittest.TestCase):
    def test_fetch_by_id_validates_response_into_domain_model(self):
        session = Mock()
        session.get.return_value = response(payload=entry().model_dump())

        result = UniProtService(session).fetch_uniprot_json_from_id("P12345")

        self.assertIsInstance(result, UniProtEntry)
        self.assertEqual(result.primaryAccession, "P12345")
        self.assertIn("P12345.json", session.get.call_args.args[0])

    def test_fetch_interpro_members_follows_pagination_and_preserves_order(self):
        first = response(
            payload={"results": [entry("A").model_dump()]},
            headers={"Link": '<https://next.example/page>; rel="next"'},
        )
        second = response(payload={"results": [entry("B").model_dump()]})
        session = Mock()
        session.get.side_effect = [first, second]

        results = UniProtService(session).fetch_uniprot_json_from_interpro(
            "IPR1", NullProgressSink(), reviewed=True, page_size=2
        )

        self.assertEqual([item.primaryAccession for item in results], ["A", "B"])
        self.assertEqual(session.get.call_args_list[0].kwargs["params"]["query"], "(interpro-IPR1 AND reviewed:true)")
        self.assertIsNone(session.get.call_args_list[1].kwargs["params"])

    def test_fetch_interpro_members_raises_on_failed_page(self):
        session = Mock()
        session.get.return_value = response(status=503)

        with self.assertRaisesRegex(RuntimeError, "503"):
            UniProtService(session).fetch_uniprot_json_from_interpro("IPR1", NullProgressSink())

    def test_uniparc_fetch_and_parse_filters_non_uniprot_databases(self):
        payload = {
            "uniParcCrossReferences": [
                {"database": "UniProtKB/Swiss-Prot", "id": "P1"},
                {"database": "UniProtKB/TrEMBL", "id": "Q2"},
                {"database": "RefSeq", "id": "NP_1"},
            ]
        }
        session = Mock()
        session.get.return_value = response(payload=payload)
        service = UniProtService(session)

        parsed = service.parse_uniparc_json(service.get_uniparc_json("UPI1"))

        self.assertEqual(parsed, ["P1", "Q2"])


class TestUniProtParsing(unittest.TestCase):
    def setUp(self):
        self.service = UniProtService(Mock())

    def test_parse_chains_expands_shared_ranges_and_ignores_malformed_groups(self):
        parsed = self.service.parse_chains("A/B=10-20, malformed, C=30-x, D=40-50")
        self.assertEqual(
            parsed,
            [
                {"chain": "A", "start": 10, "end": 20},
                {"chain": "B", "start": 10, "end": 20},
                {"chain": "D", "start": 40, "end": 50},
            ],
        )

    def test_map_pdb_to_uniprot_is_inclusive_at_boundaries(self):
        mapping = {
            "start": {"residue_number": 10},
            "end": {"residue_number": 20},
            "unp_start": 100,
        }

        self.assertEqual(self.service.map_pdb_to_uniprot(10, mapping), 100)
        self.assertEqual(self.service.map_pdb_to_uniprot(20, mapping), 110)
        self.assertIsNone(self.service.map_pdb_to_uniprot(9, mapping))

    def test_cross_references_produce_annotations_and_all_structure_types(self):
        refs = [
            UniProtKBCrossReference(database="InterPro", id="IPR1"),
            UniProtKBCrossReference(
                database="PDB",
                id="1ABC",
                properties=[
                    Property(key="Method", value="X-ray"),
                    Property(key="Resolution", value="2.0 A"),
                    Property(key="Chains", value="A/B=1-100"),
                ],
            ),
            UniProtKBCrossReference(database="SMR", id="P12345"),
            UniProtKBCrossReference(database="AlphaFoldDB", id="P12345"),
            UniProtKBCrossReference(database="GO", id="GO:1"),
        ]

        annotations, structures = self.service.parse_uniprot_cross_references(entry(cross_references=refs))

        self.assertEqual(annotations.interpro_ids, ["IPR1"])
        self.assertEqual(len(structures), 3)
        pdb = next(item for item in structures if isinstance(item, ExperimentalStructure))
        self.assertEqual(pdb.metadata.method, "X-ray")
        self.assertEqual(pdb.metadata.resolution, 2.0)
        self.assertEqual([chain.chain for chain in pdb.chains], ["A", "B"])
        self.assertTrue(any(isinstance(item, HomologyStructure) for item in structures))
        self.assertTrue(any(isinstance(item, AlphaFoldStructure) for item in structures))

    def test_parse_features_extracts_only_domains_and_active_sites(self):
        features = [
            Feature(type="Domain", location=FeatureLocation(start=FeatureLocationRange(value=2), end=FeatureLocationRange(value=20))),
            Feature(type="Active site", location=FeatureLocation(start=FeatureLocationRange(value=7), end=FeatureLocationRange(value=7))),
            Feature(type="Binding site", location=FeatureLocation(start=FeatureLocationRange(value=8), end=FeatureLocationRange(value=8))),
        ]

        annotations, _ = self.service.parse_uniprot_json(entry(features=features))

        self.assertEqual([(region.start, region.end) for region in annotations.domains], [(2, 20)])
        self.assertEqual([(region.start, region.end) for region in annotations.active_sites], [(7, 7)])

    def test_sifts_parser_prefers_first_matching_accession_and_chain(self):
        structure = ExperimentalStructure(
            struct_id="1ABC", database="PDB", chains=[]
        )
        structure.chains = [Mock(chain="A")]
        wanted = {"struct_asym_id": "A", "chain_id": "X", "start": {"residue_number": 1}, "end": {"residue_number": 2}, "unp_start": 5}
        payload = {"1abc": {"UniProt": {"OTHER": {"mappings": []}, "P1": {"mappings": [wanted]}}}}

        self.assertIs(self.service.parse_sifts_json(payload, structure, ["OTHER", "P1"]), wanted)

    def test_sifts_parser_raises_for_missing_pdb_entry(self):
        structure = ExperimentalStructure(struct_id="1ABC", database="PDB")
        with self.assertRaisesRegex(ValueError, "No SIFTS entry"):
            self.service.parse_sifts_json({}, structure, ["P1"])


class TestProteinMetadataAndInterPro(unittest.TestCase):
    def test_prepare_metadata_populates_annotations_structures_and_sifts(self):
        refs = [
            UniProtKBCrossReference(database="InterPro", id="IPR1"),
            UniProtKBCrossReference(
                database="PDB",
                id="1ABC",
                properties=[Property(key="Chains", value="A=1-100")],
            ),
        ]
        mapping = {"struct_asym_id": "A", "chain_id": "A", "start": {"residue_number": 1}, "end": {"residue_number": 100}, "unp_start": 1}
        session = Mock()
        session.get.return_value = response(payload={"1abc": {"UniProt": {"P1": {"mappings": [mapping]}}}})
        protein = ProteinData("P1", "UPI-P1", [])

        UniProtService(session).prepare_protein_metadata(protein, entry("P1", refs), NullProgressSink())

        self.assertEqual(protein.annotations.annotations.interpro_ids, ["IPR1"])
        self.assertEqual(len(protein.structures.structures), 1)
        self.assertIs(protein.structures.structures[0].analysis.sifts_mapping, mapping)

    def test_interpro_selection_prefers_family_over_earlier_domain(self):
        session = Mock()
        session.get.side_effect = [
            response(payload={"metadata": {"type": "domain"}}),
            response(payload={"metadata": {"type": "family"}}),
        ]

        result = UniProtService(session).fetch_least_abstracted_interpro_id(
            ["IPR-D", "IPR-F"], NullProgressSink()
        )

        self.assertEqual(result, {"id": "IPR-F", "type": "family"})

    def test_interpro_selection_returns_first_domain_when_no_family_exists(self):
        session = Mock()
        session.get.side_effect = [
            response(payload={"metadata": {"type": "domain"}}),
            response(payload={"metadata": {"type": "other"}}),
        ]

        result = UniProtService(session).fetch_least_abstracted_interpro_id(
            ["IPR-D", "IPR-O"], NullProgressSink()
        )

        self.assertEqual(result, {"id": "IPR-D", "type": "domain"})

    def test_empty_interpro_set_has_no_selection(self):
        result = UniProtService(Mock()).fetch_least_abstracted_interpro_id([], NullProgressSink())
        self.assertIsNone(result)

    def test_manage_interpro_returns_empty_when_no_group_can_be_selected(self):
        result = UniProtService(Mock()).manage_interpro([], "P1", NullProgressSink())
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
