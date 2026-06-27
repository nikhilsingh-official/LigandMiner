from __future__ import annotations

import json
from collections import defaultdict
from typing import List

from src.domain.ligand import Ligand, LigandRecord
from src.domain.pipeline import PipelineState, ProteinData, ProteinLigands
from src.domain.uniprot import UniProtEntry

from .ligand_services import LigandService, ScreeningService
from .progress import ProgressSink
from .protein_services import UniProtService
from .services.structure_services import StructureSelectionService
from .surface_services import SurfaceService


class LigandMinerApplication:
    def __init__(self):
        self.uniprot_service = UniProtService()
        self.ligand_service = LigandService()
        self.screening_service = ScreeningService(self.ligand_service)
        self.structure_selection_service = StructureSelectionService()
        self.surface_service = SurfaceService()

    def main(self, uniprot_id: str, progress: ProgressSink):
        with progress:
            progress.start("main", f"[bold]Finding candidate ligands for {uniprot_id}...", total=7)

            pipeline = self._initialize_target(uniprot_id, progress)
            progress.advance("main")

            self._find_candidate_homologs(pipeline, progress)
            progress.advance("main")

            self._attach_known_ligands(pipeline, progress)
            progress.advance("main")

            self._process_homologs(pipeline, progress)
            progress.advance("main")

            self._rank_homologs(pipeline, progress)
            progress.advance("main")

            final_results = self._virtual_screen(pipeline, progress)
            progress.advance("main")

            self._export_results(final_results, progress)
            progress.advance("main")

            progress.finish("main", f"[bold]Completed ligand search for {uniprot_id}")

    def _initialize_target(self, uniprot_id: str, progress: ProgressSink) -> PipelineState:
        uniprot_json: UniProtEntry = self.uniprot_service.fetch_uniprot_json_from_id(uniprot_id)

        other_accessions = self.uniprot_service.parse_uniparc_json(
            self.uniprot_service.get_uniparc_json(uniprot_json.extraAttributes.uniParcId)
        )

        target_protein = ProteinData(
            uniprot_id=uniprot_id,
            uniparc_id=uniprot_json.extraAttributes.uniParcId,
            other_accessions=other_accessions,
        )

        self.uniprot_service.prepare_protein_metadata(
            uniprot_json,
            progress.child("target_metadata"),
        )

        pipeline = PipelineState(target_protein=target_protein)

        pipeline.target_protein.best_structure = self.structure_selection_service.select_best_structure(
            pipeline.target_protein,
            progress.child("target_best_structure"),
        )

        self.surface_service.surface_extraction(
            pipeline.target_protein.best_structure,
            progress.child("target_surface"),
        )
        return pipeline

    def _find_candidate_homologs(self, pipeline: PipelineState, progress: ProgressSink):
        results: List[UniProtEntry] = self.uniprot_service.manage_interpro(
            pipeline.target_protein.annotations.annotations.interpro_ids,
            pipeline.target_protein.uniprot_id,
            progress.child("interpro"),
        )

        progress.start(
            "enrich_homologs",
            f"Enriching {len(results)} homolog proteins with metadata...",
            total=len(results),
        )

        for entry in results:
            progress.update("enrich_homologs", f"Enriching homolog {entry.primaryAccession} with metadata...")

            other_accessions = self.uniprot_service.parse_uniparc_json(
                self.uniprot_service.get_uniparc_json(entry.extraAttributes.uniParcId)
            )

            protein = ProteinData(
                uniprot_id=entry.primaryAccession,
                uniparc_id=entry.extraAttributes.uniParcId,
                other_accessions=other_accessions,
            )

            self.uniprot_service.prepare_protein_metadata(
                protein,
                entry,
                progress.child(f"homolog_metadata_{entry.primaryAccession}"),
            )
            pipeline.candidate_proteins.append(protein)
            progress.advance("enrich_homologs")

        progress.finish("enrich_homologs", "Enriched homolog proteins with metadata")

    def _attach_known_ligands(self, pipeline: PipelineState, progress: ProgressSink):
        progress.start(
            "find_ligands",
            f"Finding ligands for {len(pipeline.candidate_proteins)} homolog proteins...",
            total=len(pipeline.candidate_proteins),
        )

        uniprot_ids = [p.uniprot_id for p in pipeline.candidate_proteins]

        progress.update("find_ligands", "Querying ligand database (batch)...")

        ligand_records: list[LigandRecord] = self.ligand_service.get_ligands_for_proteins(uniprot_ids)

        print("[DEBUG] Retrieved ligand records:", len(ligand_records))

        ligands_by_protein: dict[str, list[Ligand]] = defaultdict(list)

        for record in ligand_records:
            ligand_obj = Ligand(record=record)
            ligands_by_protein[record.protein_id].append(ligand_obj)

        filtered_proteins = []

        for entry in pipeline.candidate_proteins:
            accession = entry.uniprot_id

            progress.update("find_ligands", f"Mapping ligands to homolog {accession}...")

            ligands_for_protein = ligands_by_protein.get(accession, [])

            if not ligands_for_protein:
                progress.advance("find_ligands")
                continue

            entry.ligands = ProteinLigands(
                ligands=ligands_for_protein,
                ligand_count=len(ligands_for_protein),
            )
            filtered_proteins.append(entry)
            progress.advance("find_ligands")

        pipeline.filtered_proteins = filtered_proteins
        progress.finish("find_ligands", "Found ligands for homolog proteins")

    def _process_homologs(self, pipeline: PipelineState, progress: ProgressSink):
        progress.start(
            "process_homologs",
            f"Processing {len(pipeline.filtered_proteins)} homolog proteins...",
            total=len(pipeline.filtered_proteins),
        )

        for similar_protein in pipeline.filtered_proteins:
            similar_accession = similar_protein.uniprot_id
            homolog_progress = progress.child(f"homolog_{similar_accession}")

            if not similar_protein.structures or not similar_protein.structures.structures:
                progress.advance("process_homologs")
                continue

            progress.update(
                "process_homologs",
                (
                    f"   ↳ Processing homolog {similar_accession} "
                    f"(selecting best structure from {len(similar_protein.structures.structures)} candidates)..."
                ),
            )

            similar_protein.best_structure = self.structure_selection_service.select_best_structure(
                similar_protein,
                homolog_progress.child("best_structure"),
            )

            progress.update("process_homologs", f"Processing homolog {similar_accession} (extracting surface of best structure)...")

            self.surface_service.surface_extraction(
                similar_protein.best_structure,
                homolog_progress.child("surface"),
            )

            progress.update("process_homologs", f"Processing homolog {similar_accession} (comparing surfaces)...")

            self.surface_service.compare_surfaces(
                pipeline.target_protein.best_structure,
                similar_protein.best_structure,
                homolog_progress.child("compare"),
            )

            progress.update("process_homologs", f"Processing homolog {similar_accession} (obtaining results)...")

            parsed = self.surface_service.manage_results(
                pipeline.target_protein.best_structure,
                similar_protein.best_structure,
                homolog_progress.child("results"),
            )

            if not parsed:
                progress.advance("process_homologs")
                continue

            similar_protein.probis_score = self.surface_service.probis_similarity_score(parsed)
            progress.advance("process_homologs")

    def _rank_homologs(self, pipeline: PipelineState, progress: ProgressSink):
        scored = [p for p in pipeline.filtered_proteins if p.probis_score is not None]
        pipeline.ranked_proteins = sorted(
            scored,
            key=lambda item: item.probis_score,
            reverse=True,
        )

    def _virtual_screen(self, pipeline: PipelineState, progress: ProgressSink):
        print("[INFO] Loading FDA-approved ligands...")
        fda_records = self.ligand_service.get_fda_approved_ligands()
        fda_approved_ligands = [Ligand(record=r) for r in fda_records]
        print(f"[INFO] Loaded {len(fda_approved_ligands)} FDA-approved ligands")
        print(f"[INFO] Screening {len(pipeline.ranked_proteins)} ranked proteins...")

        return self.screening_service.screen_ranked_proteins(
            pipeline,
            fda_approved_ligands,
            progress.child("screening"),
        )

    def _export_results(self, final_results, progress: ProgressSink):
        with open("ranked_protein_virtual_screening.json", "w") as f:
            json.dump(final_results, f, indent=4)
        print("[INFO] Results written to ranked_protein_virtual_screening.json")
