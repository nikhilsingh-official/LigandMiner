import json
import os
import tempfile
import unittest

from src.app.application import LigandMinerApplication
from src.app.progress import NullProgressSink, ProgressEvent
from src.domain.ligand import Ligand, LigandRecord
from src.domain.pipeline import (
    PipelineState,
    ProteinAnnotations,
    ProteinData,
    ProteinLigands,
    ProteinScores,
    ProteinStructures,
)
from src.domain.structure import ExperimentalStructure
from src.domain.uniprot import ExtraAttributes, SequenceAnnotations, UniProtEntry


def make_entry(accession: str, interpro_id: str = "IPR0001") -> UniProtEntry:
    return UniProtEntry(
        entryType="UniProtKB reviewed (Swiss-Prot)",
        primaryAccession=accession,
        extraAttributes=ExtraAttributes(uniParcId=f"UPI-{accession}"),
    )


def make_structure(accession: str) -> ExperimentalStructure:
    return ExperimentalStructure(struct_id=f"S-{accession}", database="PDB")


def make_ligand(protein_id: str, ligand_id: str, affinity: float = 10.0) -> LigandRecord:
    return LigandRecord(
        protein_id=protein_id,
        inchi_key=f"KEY-{ligand_id}",
        ligand_id=ligand_id,
        smiles="CCO",
        affinity_type="IC50",
        affinity=affinity,
    )


class RecordingProgressSink(NullProgressSink):
    def __init__(self):
        self.events = []
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited = True
        return False

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def start(self, key: str, message: str, total: int | None = None) -> None:
        self.emit(ProgressEvent("start", key, message=message, total=total))

    def update(
        self,
        key: str,
        message: str,
        *,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        self.emit(ProgressEvent("update", key, message=message, total=total, completed=completed))

    def advance(self, key: str, step: int = 1, message: str | None = None) -> None:
        self.emit(ProgressEvent("advance", key, message=message, advance=step))

    def finish(self, key: str, message: str | None = None) -> None:
        self.emit(ProgressEvent("finish", key, message=message))

    def child(self, prefix: str):
        return self


class StubUniProtService:
    def __init__(self, homolog_ids=None, interpro_ids=None, fail_on_fetch=False):
        self.homolog_ids = list(homolog_ids or [])
        self.interpro_ids = list(interpro_ids or ["IPR0001"])
        self.fail_on_fetch = fail_on_fetch
        self.prepared = []
        self.managed_interpro_ids = None

    def fetch_uniprot_json_from_id(self, uniprot_id: str) -> UniProtEntry:
        if self.fail_on_fetch:
            raise RuntimeError("UniProt unavailable")
        return make_entry(uniprot_id)

    def get_uniparc_json(self, uniparc_id: str):
        return {"uniParcCrossReferences": []}

    def parse_uniparc_json(self, uniparc_json):
        return ["ALT-1"]

    def prepare_protein_metadata(self, protein: ProteinData, uniprot_json: UniProtEntry, progress):
        protein.annotations = ProteinAnnotations(
            SequenceAnnotations(interpro_ids=list(self.interpro_ids), domains=[], active_sites=[])
        )
        protein.structures = ProteinStructures(structures=[make_structure(protein.uniprot_id)])
        self.prepared.append(protein.uniprot_id)

    def manage_interpro(self, interpro_ids, target_uniprot_id, progress):
        self.managed_interpro_ids = list(interpro_ids)
        return [make_entry(accession) for accession in self.homolog_ids]


class StubStructureSelectionService:
    def __init__(self):
        self.selected = []

    def select_best_structure(self, protein: ProteinData, progress):
        self.selected.append(protein.uniprot_id)
        return protein.structures.structures[0]


class StubSurfaceService:
    def __init__(self, scores=None, missing_results=None, fail_on_extract=False):
        self.scores = scores or {}
        self.missing_results = set(missing_results or [])
        self.fail_on_extract = fail_on_extract
        self.extracted = []
        self.compared = []

    def surface_extraction(self, structure, progress):
        if self.fail_on_extract:
            raise RuntimeError("ProBiS extraction failed")
        self.extracted.append(structure.struct_id)

    def compare_surfaces(self, target, similar, progress):
        self.compared.append((target.struct_id, similar.struct_id))

    def manage_results(self, target, similar, progress):
        if similar.struct_id in self.missing_results:
            return None
        return {"score": self.scores.get(similar.struct_id, 1.0)}

    def probis_similarity_score(self, parsed):
        return parsed["score"]


class StubLigandService:
    def __init__(self, records=None, fda_records=None):
        self.records = list(records or [])
        self.fda_records = list(fda_records or [make_ligand("FDA", "FDA-1", affinity=1.0)])
        self.requested_protein_ids = None

    def get_ligands_for_proteins(self, uniprot_ids):
        self.requested_protein_ids = list(uniprot_ids)
        requested = set(uniprot_ids)
        return [record for record in self.records if record.protein_id in requested]

    def get_fda_approved_ligands(self):
        return list(self.fda_records)


class RecordingScreeningService:
    def __init__(self):
        self.pipeline = None
        self.fda_ligands = None

    def screen_ranked_proteins(self, pipeline, fda_approved_ligands, progress):
        self.pipeline = pipeline
        self.fda_ligands = fda_approved_ligands
        return [
            {
                "protein": protein.uniprot_id,
                "probis_score": protein.probis_score,
                "ligand_count": protein.ligand_count,
            }
            for protein in pipeline.ranked_proteins
        ]


class LigandMinerApplicationIntegrationTests(unittest.TestCase):
    def build_app(
        self,
        *,
        homolog_ids=None,
        ligand_records=None,
        surface_scores=None,
        missing_surface_results=None,
        fail_uniprot=False,
        fail_surface=False,
    ):
        app = LigandMinerApplication()
        app.uniprot_service = StubUniProtService(homolog_ids=homolog_ids, fail_on_fetch=fail_uniprot)
        app.structure_selection_service = StubStructureSelectionService()
        app.surface_service = StubSurfaceService(
            scores=surface_scores,
            missing_results=missing_surface_results,
            fail_on_extract=fail_surface,
        )
        app.ligand_service = StubLigandService(records=ligand_records)
        app.screening_service = RecordingScreeningService()
        return app

    def run_main_in_temp_dir(self, app, progress=None):
        progress = progress or RecordingProgressSink()
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                result = app.main("TARGET", progress)
                with open("ranked_protein_virtual_screening.json") as handle:
                    output = json.load(handle)
            finally:
                os.chdir(previous)
        return result, output, progress

    def test_main_enriches_filters_ranks_screens_and_exports_results(self):
        app = self.build_app(
            homolog_ids=["LOW", "HIGH", "NO-LIGAND"],
            ligand_records=[
                make_ligand("LOW", "L-LOW", affinity=20.0),
                make_ligand("HIGH", "L-HIGH", affinity=5.0),
            ],
            surface_scores={"S-LOW": 1.0, "S-HIGH": 3.0},
        )

        result, output, progress = self.run_main_in_temp_dir(app)

        self.assertIsNone(result)
        self.assertTrue(progress.entered)
        self.assertTrue(progress.exited)
        self.assertEqual(app.uniprot_service.prepared, ["TARGET", "LOW", "HIGH", "NO-LIGAND"])
        self.assertEqual(app.uniprot_service.managed_interpro_ids, ["IPR0001"])
        self.assertEqual(app.ligand_service.requested_protein_ids, ["LOW", "HIGH", "NO-LIGAND"])
        self.assertEqual([protein.uniprot_id for protein in app.screening_service.pipeline.ranked_proteins], ["HIGH", "LOW"])
        self.assertEqual([record["protein"] for record in output], ["HIGH", "LOW"])
        self.assertEqual([record["probis_score"] for record in output], [3.0, 1.0])
        self.assertEqual([event.action for event in progress.events if event.key == "main"], [
            "start",
            "advance",
            "advance",
            "advance",
            "advance",
            "advance",
            "advance",
            "advance",
            "finish",
        ])

    def test_main_handles_pipeline_with_no_known_homolog_ligands(self):
        app = self.build_app(
            homolog_ids=["HOMOLOG"],
            ligand_records=[],
            surface_scores={"S-HOMOLOG": 2.0},
        )

        result, output, progress = self.run_main_in_temp_dir(app)

        self.assertIsNone(result)
        self.assertEqual(output, [])
        self.assertEqual(app.screening_service.pipeline.filtered_proteins, [])
        self.assertEqual(app.screening_service.pipeline.ranked_proteins, [])
        self.assertTrue(progress.exited)

    def test_main_omits_homologs_without_surface_similarity_results(self):
        app = self.build_app(
            homolog_ids=["MATCH", "NO-SURFACE-HIT"],
            ligand_records=[
                make_ligand("MATCH", "L-MATCH"),
                make_ligand("NO-SURFACE-HIT", "L-MISS"),
            ],
            surface_scores={"S-MATCH": 2.5},
            missing_surface_results={"S-NO-SURFACE-HIT"},
        )

        _, output, _ = self.run_main_in_temp_dir(app)

        self.assertEqual([record["protein"] for record in output], ["MATCH"])
        self.assertEqual(app.surface_service.compared, [
            ("S-TARGET", "S-MATCH"),
            ("S-TARGET", "S-NO-SURFACE-HIT"),
        ])

    def test_critical_uniprot_failures_propagate(self):
        app = self.build_app(fail_uniprot=True)
        progress = RecordingProgressSink()

        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                with self.assertRaisesRegex(RuntimeError, "UniProt unavailable"):
                    app.main("TARGET", progress)
            finally:
                os.chdir(previous)

        self.assertTrue(progress.entered)
        self.assertTrue(progress.exited)

    def test_critical_surface_failures_propagate(self):
        app = self.build_app(fail_surface=True)
        progress = RecordingProgressSink()

        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                with self.assertRaisesRegex(RuntimeError, "ProBiS extraction failed"):
                    app.main("TARGET", progress)
            finally:
                os.chdir(previous)

        self.assertTrue(progress.entered)
        self.assertTrue(progress.exited)


class PipelineDomainIntegrationTests(unittest.TestCase):
    def test_protein_data_preserves_identity_metadata_structures_ligands_and_scores(self):
        annotations = ProteinAnnotations(
            SequenceAnnotations(interpro_ids=["IPR0001"], domains=[], active_sites=[])
        )
        structures = ProteinStructures(structures=[make_structure("TARGET")])
        ligands = ProteinLigands(ligands=[Ligand(make_ligand("TARGET", "L1"))], ligand_count=1)
        scores = ProteinScores(probis_score=4.2)

        protein = ProteinData(
            uniprot_id="TARGET",
            uniparc_id="UPI-TARGET",
            other_accessions=["ALT-1"],
            annotations=annotations,
            structures=structures,
            ligands=ligands,
            scores=scores,
        )

        self.assertEqual(protein.uniprot_id, "TARGET")
        self.assertEqual(protein.uniparc_id, "UPI-TARGET")
        self.assertEqual(protein.other_accessions, ["ALT-1"])
        self.assertIs(protein.annotations, annotations)
        self.assertEqual(protein.structures.structures[0].struct_id, "S-TARGET")
        self.assertEqual(protein.ligand_count, 1)
        self.assertEqual(protein.probis_score, 4.2)

    def test_pipeline_state_tracks_candidates_filtered_and_ranked_proteins(self):
        target = ProteinData("TARGET", "UPI-TARGET", [])
        low = ProteinData("LOW", "UPI-LOW", [], scores=ProteinScores(probis_score=1.0))
        high = ProteinData("HIGH", "UPI-HIGH", [], scores=ProteinScores(probis_score=3.0))

        pipeline = PipelineState(
            target_protein=target,
            candidate_proteins=[low, high],
            filtered_proteins=[low, high],
            ranked_proteins=sorted([low, high], key=lambda protein: protein.probis_score, reverse=True),
        )

        self.assertEqual([protein.uniprot_id for protein in pipeline.candidate_proteins], ["LOW", "HIGH"])
        self.assertEqual([protein.uniprot_id for protein in pipeline.filtered_proteins], ["LOW", "HIGH"])
        self.assertEqual([protein.uniprot_id for protein in pipeline.ranked_proteins], ["HIGH", "LOW"])


class LigandMinerApplicationLifecycleTests(unittest.TestCase):
    def test_application_instances_own_independent_service_objects(self):
        first = LigandMinerApplication()
        second = LigandMinerApplication()

        self.assertIsNot(first, second)
        self.assertIsNot(first.uniprot_service, second.uniprot_service)
        self.assertIsNot(first.ligand_service, second.ligand_service)
        self.assertIsNot(first.screening_service, second.screening_service)
        self.assertIsNot(first.structure_selection_service, second.structure_selection_service)
        self.assertIsNot(first.surface_service, second.surface_service)


if __name__ == "__main__":
    unittest.main(verbosity=2)
