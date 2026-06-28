import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.app.application import LigandMinerApplication
from src.app.progress import NullProgressSink
from src.app.protein_services import UniProtService
from src.domain.ligand import LigandRecord
from src.domain.structure import ExperimentalStructure
from src.domain.uniprot import ExtraAttributes, UniProtEntry, UniProtKBCrossReference


def entry(accession, interpro="IPR1"):
    return UniProtEntry(
        entryType="UniProtKB reviewed (Swiss-Prot)",
        primaryAccession=accession,
        extraAttributes=ExtraAttributes(uniParcId=f"UPI-{accession}"),
        uniProtKBCrossReferences=[
            UniProtKBCrossReference(database="InterPro", id=interpro),
            UniProtKBCrossReference(database="AlphaFoldDB", id=accession),
        ],
    )


class RoutedSession:
    def __init__(self):
        self.target = entry("TARGET")
        self.homologs = [entry("LOW"), entry("HIGH"), entry("NO-LIGAND")]

    @staticmethod
    def _response(payload, status=200, headers=None):
        result = Mock()
        result.status_code = status
        result.ok = status < 400
        result.json.return_value = payload
        result.headers = headers or {}
        result.raise_for_status.return_value = None
        return result

    def get(self, url, params=None, timeout=None):
        if "/uniprotkb/TARGET.json" in url:
            return self._response(self.target.model_dump())
        if "/uniparc/" in url:
            return self._response({"uniParcCrossReferences": []})
        if "/entry/interpro/" in url:
            return self._response({"metadata": {"type": "family"}})
        if url.endswith("/uniprotkb/search"):
            return self._response({"results": [item.model_dump() for item in self.homologs]})
        raise AssertionError(f"Unexpected URL: {url}")


class DeterministicStructureSelection:
    def select_best_structure(self, protein, progress):
        return ExperimentalStructure(f"S-{protein.uniprot_id}", "PDB")


class DeterministicSurfaceService:
    def surface_extraction(self, structure, progress):
        return None

    def compare_surfaces(self, target, similar, progress):
        return None

    def manage_results(self, target, similar, progress):
        scores = {"S-LOW": 1.0, "S-HIGH": 3.0, "S-NO-LIGAND": 10.0}
        return {"score": scores[similar.struct_id]}

    def probis_similarity_score(self, parsed):
        return parsed["score"]


class DeterministicLigandService:
    def get_ligands_for_proteins(self, uniprot_ids):
        return [
            LigandRecord("LOW", "K1", "L1", "CCO", "IC50", 10.0),
            LigandRecord("HIGH", "K2", "L2", "CCN", "Ki", 20.0),
        ]

    def get_fda_approved_ligands(self):
        return [LigandRecord("FDA", "KF", "FDA1", "CCO", "IC50", 1.0)]


class ObservingScreeningService:
    def __init__(self):
        self.ranked_ids = None

    def screen_ranked_proteins(self, pipeline, fda_approved_ligands, progress):
        self.ranked_ids = [protein.uniprot_id for protein in pipeline.ranked_proteins]
        return [
            {"protein": protein.uniprot_id, "probis_score": protein.probis_score, "hits": []}
            for protein in pipeline.ranked_proteins
        ]


class TestApplicationPipeline(unittest.TestCase):
    def test_public_main_filters_unliganded_homologs_ranks_and_exports(self):
        app = LigandMinerApplication()
        app.uniprot_service = UniProtService(RoutedSession())
        app.structure_selection_service = DeterministicStructureSelection()
        app.surface_service = DeterministicSurfaceService()
        app.ligand_service = DeterministicLigandService()
        app.screening_service = ObservingScreeningService()

        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                returned = app.main("TARGET", NullProgressSink())
                output = json.loads(Path("ranked_protein_virtual_screening.json").read_text())
            finally:
                os.chdir(previous)

        self.assertIsNone(returned)
        self.assertEqual(app.screening_service.ranked_ids, ["HIGH", "LOW"])
        self.assertEqual([record["protein"] for record in output], ["HIGH", "LOW"])
        self.assertEqual([record["probis_score"] for record in output], [3.0, 1.0])


if __name__ == "__main__":
    unittest.main()
