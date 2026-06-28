# Homolog Selection and Main Pipeline Architecture

This document describes the current working-tree implementation. It distinguishes the literal executable path from downstream code that is statically wired but currently unreachable through the public entry point because of source-level inconsistencies.

# 1. High-Level Pipeline

## 1.1 Literal entry path and blockers

```text
CLI: main.py
  main(uniprot_id)
      |
      v
src.app.LigandMinerApplication.main(uniprot_id, progress)
      |
      v
LigandMinerApplication._initialize_target(...)
      |
      +--> UniProtService.fetch_uniprot_json_from_id()
      +--> UniProtService.get_uniparc_json()
      +--> UniProtService.parse_uniparc_json()
      +--> ProteinData(...)
      |
      +--> UniProtService.prepare_protein_metadata(...)
             |
             +--> CURRENT BLOCKER: called with the UniProtEntry as `protein`
                  and the ProgressSink as `entry`; the implementation expects
                  (ProteinData, UniProtEntry, ProgressSink | None).
```

Even if that call is supplied the objects implied by its signature, `prepare_protein_metadata()` calls `parse_uniprot_json()`, whose `parse_uniprot_cross_references()` currently returns an undefined name, `annotations`. The downstream pipeline below is therefore a static call-flow description, not a claim that `main.py` currently completes.

## 1.2 Statically wired end-to-end flow

```text
main.py:main(uniprot_id)
  |
  v
LigandMinerApplication.main()
  |
  +-- 1. _initialize_target()
  |     UniProt + UniParc retrieval
  |     -> ProteinData target
  |     -> annotation/structure enrichment
  |     -> best-structure selection
  |     -> ProBiS target-surface extraction
  |
  +-- 2. _find_candidate_homologs()
  |     InterPro family/domain selection
  |     -> reviewed UniProt members
  |     -> ProteinData candidate_proteins
  |
  +-- 3. _attach_known_ligands()
  |     local ligand.db query
  |     -> ProteinLigands attached by protein ID
  |     -> filtered_proteins (only candidates with ligands)
  |
  +-- 4. _process_homologs()
  |     per homolog:
  |       structure enrichment/ranking
  |       download + cleaning
  |       holo-pocket or P2Rank pocket selection
  |       fpocket analysis and pocket scoring
  |       ProBiS surface extraction/comparison/results
  |       -> ProteinData.probis_score
  |
  +-- 5. _rank_homologs()
  |     descending stable sort by probis_score
  |     -> ranked_proteins
  |
  +-- 6. _virtual_screen()
  |     local FDA ligand retrieval
  |     -> ScreeningService.screen_ranked_proteins()
  |          per ranked protein:
  |            training ligands from ligand.db
  |            Pharm2D consensus
  |            FDA Pharm2D/Morgan screening
  |            0.6/0.4 combined score
  |            retain top 25% (at least one)
  |            optional Pharm3D consensus and screening
  |            ascending Pharm3D score ranking
  |          -> list[dict] grouped by protein
  |
  +-- 7. _export_results()
        json.dump(..., "ranked_protein_virtual_screening.json")
        -> final JSON file; main() returns None
```

## 1.3 Stage inventory

| Stage | Primary function/class | File | Purpose | Input | Output |
|---|---|---|---|---|---|
| CLI entry | `main(uniprot_id)` | `main.py` | Construct the application and invoke the pipeline with the shared Rich progress sink. | UniProt accession string | No return value; downstream file side effect |
| Pipeline orchestration | `LigandMinerApplication.main()` | `src/app/application.py` | Execute seven sequential phases under a progress context. | UniProt accession, `ProgressSink` | `None` |
| Target retrieval | `UniProtService.fetch_uniprot_json_from_id()` | `src/app/protein_services.py` | Fetch selected UniProtKB fields and validate them with Pydantic. | UniProt accession | `UniProtEntry` |
| UniParc aliases | `get_uniparc_json()`, `parse_uniparc_json()` | `src/app/protein_services.py` | Retrieve cross-referenced UniProtKB accessions. | UniParc ID | `list[str]` |
| Protein metadata | `prepare_protein_metadata()` | `src/app/protein_services.py` | Populate annotations, candidate structures, and PDB SIFTS mappings. | `ProteinData`, `UniProtEntry` | Mutates `ProteinData`; returns `None` |
| Structure metadata/scoring | `StructureMetadataService.metadata_structures_enrichment()` | `src/app/protein_services.py` | Expand SMR/AlphaFold references, score structures, compute feature coverage, and assign paths. | Structures, domains, active sites, UniProt ID | Enriched `list[Structure]` |
| Structure selection | `StructureSelectionService.select_best_structure()` | `src/app/services/structure_services.py` | Shortlist, download, clean, identify pockets, score pockets, and select a structure. | Annotated `ProteinData` | One `Structure` |
| Pocket identification | `is_holo()` or `PocketService.get_pocket_from_prank()` | structure/pocket service modules | Prefer a sufficiently large pocket around a bound non-artifact ligand; otherwise invoke P2Rank. | Raw structure and chain coverage | Pocket residue list and certainty |
| fpocket enrichment | `PocketService.fpocket_enrichment()` | `src/app/services/pocket_service.py` | Run or reuse custom-pocket fpocket output. | Protein structures with a true pocket | Mutated structure pocket lists |
| Pocket mapping/scoring | `pocket_residue_enrichment()`, `score_pockets()` | `src/app/services/pocket_service.py` | Map pocket residues to UniProt coordinates and assign binding-site scores. | Annotated structures and fpocket output | Mutated `ProteinData` |
| Target surface | `SurfaceService.surface_extraction()` | `src/app/surface_services.py` | Run ProBiS `-extract` or reuse a cached `.srf`. | Selected target structure | `.srf` path/file side effect |
| InterPro selection | `UniProtService.fetch_least_abstracted_interpro_id()` | `src/app/protein_services.py` | Return the first family entry, otherwise the first domain entry. | Target InterPro IDs | `{"id": str, "type": "family"|"domain"}` or `None` |
| Homolog retrieval | `UniProtService.fetch_uniprot_json_from_interpro()` | `src/app/protein_services.py` | Page through reviewed UniProtKB members of the selected InterPro entry. | InterPro ID | `list[UniProtEntry]` |
| Homolog enrichment | `_find_candidate_homologs()` | `src/app/application.py` | Build `ProteinData` objects and attach metadata to every returned member. | `PipelineState` | Mutates `candidate_proteins` |
| Known-ligand retrieval | `LigandService.get_ligands_for_proteins()` | `src/app/ligand_services.py` | Query local activity and ligand tables for supported affinity measurements. | Candidate UniProt IDs | `list[LigandRecord]` |
| Ligand-based filtering | `_attach_known_ligands()` | `src/app/application.py` | Attach `ProteinLigands`; retain only proteins with at least one ligand. | Candidate proteins, ligand records | Mutates `filtered_proteins` |
| Homolog surfaces | `_process_homologs()` | `src/app/application.py` | Select homolog structures, extract/compare surfaces, parse results, and assign ProBiS scores. | Target structure and filtered homologs | Mutated `probis_score` fields |
| ProBiS comparison | `compare_surfaces()`, `manage_results()` | `src/app/surface_services.py` | Run `-compare`/`-results`, parse the best alignment, and measure pocket overlap. | Target and homolog structures | Parsed result dictionary or `None` |
| Homolog ranking | `_rank_homologs()` | `src/app/application.py` | Drop proteins without a score and sort remaining proteins descending. | `filtered_proteins` | `ranked_proteins` |
| FDA retrieval | `LigandService.get_fda_approved_ligands()` | `src/app/ligand_services.py` | Select the strongest stored activity row per FDA ligand. | Local SQLite database | `list[LigandRecord]` |
| Pharm2D screening | `Pharm2DScreening` | `src/pharma/pharmacophore/pharma2d/screening.py` | Build consensus Pharm2D/Morgan fingerprints and score FDA candidates. | Training and candidate `Ligand` objects | `(Ligand, Pharm2DResult)` pairs |
| Pharm3D screening | `Pharm3DScreening` | `src/pharma/pharmacophore/pharma3d/screening.py` | Build a spatial consensus and rank valid feature embeddings by weighted dissimilarity. | Selected training ligands and top 2D hits | `(LigandConformers, Pharm3DResult)` pairs |
| Result assembly | `ScreeningService.screen_ranked_proteins()` | `src/app/ligand_services.py` | Produce per-protein hit records in final ranking order. | Ranked proteins and FDA ligands | `list[dict]` |
| Export | `_export_results()` | `src/app/application.py` | Serialize the result list. | `list[dict]` | `ranked_protein_virtual_screening.json` |

There is no BLAST stage in the active call graph. Homolog candidates come from reviewed UniProtKB members of one selected InterPro family/domain. The modules under `src/clients/`, `src/pockets/`, `src/surfaces/`, `src/scoring/`, and `src/io/` are not imported by this entry path.

# 2. Stage Breakdown

### Entry and orchestration

- **Purpose:** validate the presence of a CLI argument and execute the seven application stages in fixed order.
- **Inputs:** a UniProt accession string and the `RichProgressSink` exported from `src.app`.
- **Outputs:** no Python return value. The intended terminal artifact is a JSON file.
- **Primary public function:** `main.main(uniprot_id)` and `LigandMinerApplication.main(uniprot_id, progress)`.
- **Internal helpers:** `_initialize_target`, `_find_candidate_homologs`, `_attach_known_ligands`, `_process_homologs`, `_rank_homologs`, `_virtual_screen`, `_export_results`.
- **Configuration:** fixed progress total of seven; fixed output filename.
- **External dependencies:** Rich progress rendering and the filesystem.
- **Determinism:** the orchestration order is deterministic; results are not because most stages depend on remote services, mutable local databases, external binaries, and Pharm3D conformer generation.

### UniProt, UniParc, SIFTS, and annotation preparation

- **Purpose:** retrieve the target or homolog entry, collect aliases, parse domains/active sites/InterPro IDs, construct structure records, and attach PDB-to-UniProt residue mappings.
- **Inputs:** UniProt accession or InterPro ID; HTTP session; `UniProtEntry` Pydantic objects.
- **Outputs:** `UniProtEntry`, alias strings, `SequenceAnnotations`, `ProteinStructures`, and mutations to `ProteinData`.
- **Primary public functions:** `fetch_uniprot_json_from_id`, `fetch_uniprot_json_from_interpro`, `prepare_protein_metadata`, `manage_interpro`.
- **Internal helpers:** `parse_uniparc_json`, `parse_uniprot_cross_references`, `parse_uniprot_features`, `fetch_sifts_json`, `parse_sifts_json`, `parse_chains`, `map_pdb_to_uniprot`, `fetch_least_abstracted_interpro_id`.
- **Configuration:** UniProt member requests default to reviewed entries and pages of 500. Only PDB, SMR, and AlphaFoldDB cross-references are retained.
- **External dependencies:** UniProt REST, UniParc REST, InterPro REST, PDBe SIFTS, `requests`, Pydantic.
- **Determinism:** parsing is deterministic for fixed payloads. Retrieval contents, order, pagination, and availability are external-state dependent.

### Structure metadata and preliminary scoring

- **Purpose:** turn structure cross-references into concrete PDB, Swiss-Model, or AlphaFold candidates; calculate structural plausibility; compute domain/active-site coverage; assign all artifact paths.
- **Inputs:** `Structure` objects, annotations, UniProt accession.
- **Outputs:** enriched `Structure` objects with metadata, chains, paths, and `analysis.structural_plausibility`.
- **Primary public function:** `StructureMetadataService.metadata_structures_enrichment()`.
- **Internal helpers:** provider fetch/parse methods; `score_pdb_structure`, `score_swiss_model_structure`, `score_alpha_fold_structure`; `feature_structure_enrichment`; `add_paths`.
- **Configuration:** hard-coded score weights from `src/domain/scoring.py`; hard-coded database multipliers; output roots `structures/` and `probis_compare/`.
- **External dependencies:** RCSB URLs embedded in records; Swiss-Model Repository; AlphaFold API.
- **Determinism:** scoring and path construction are deterministic for fixed metadata. Provider candidate selection depends on remote payloads.

### Structure download, cleaning, pocket discovery, and selection

- **Purpose:** retain the top 100 preliminary structures, download raw CIF files, convert selected chains to PDB, identify a true pocket, run fpocket, map residues, score pockets, and select the maximum combined structure score.
- **Inputs:** annotated `ProteinData` with candidate structures.
- **Outputs:** one selected `Structure`; many intermediate files and mutations to all candidates.
- **Primary public function:** `StructureSelectionService.select_best_structure()`.
- **Internal helpers:** `shortlist_structures`, `StructureDownloader.download_structures`, `clean_structure`, `is_holo`, `ligand_passes_basic_filters`, `get_pocket_residues`, `PocketService.get_pocket_from_prank`, `fpocket_enrichment`, `pocket_residue_enrichment`, `score_pockets`, and `StructureMetadataService.structure_selection_score`.
- **Configuration:** shortlist 100; download workers 6; holo contact distance 5 Å; minimum six pocket residues; ligand artifact blacklist; minimum molecular weight 200; minimum 10 heavy atoms; occupancy at least 0.6; structure/pocket exponents 0.7/0.3.
- **External dependencies:** HTTP downloads, `gemmi` executable, gemmi Python package, BioPython `NeighborSearch`, NumPy, P2Rank `prank`, pandas, fpocket.
- **Determinism:** filtering/scoring is deterministic for fixed structures and tool outputs. Concurrent download completion order and external tool versions are not controlled. Structure ordering is mutated in place by `shortlist_structures`.

### Homolog discovery and ligand filtering

- **Purpose:** choose one InterPro group, retrieve its reviewed members, enrich each member, query known ligands, and discard members without ligands.
- **Inputs:** target annotations and local `resources/ligand.db`.
- **Outputs:** `candidate_proteins`, then `filtered_proteins` with `ProteinLigands`.
- **Primary public functions:** `UniProtService.manage_interpro()`, `LigandService.get_ligands_for_proteins()`.
- **Internal helpers:** `fetch_least_abstracted_interpro_id`, `fetch_uniprot_json_from_interpro`, `_find_candidate_homologs`, `_attach_known_ligands`.
- **Configuration:** first InterPro family wins; otherwise first domain; only `Ki`, `Kd`, `IC50`, and `EC50` rows with non-null values are queried.
- **External dependencies:** InterPro, UniProt, UniParc, SIFTS, local SQLite.
- **Determinism:** database grouping is deterministic for a fixed query result. Candidate order follows remote pagination. The code does not explicitly remove the target accession from the returned InterPro member list.

### Surface extraction, comparison, and homolog ranking

- **Purpose:** create ProBiS surfaces, compare target/homolog surfaces, parse the best alignment by aligned vertices, calculate target-pocket overlap, derive a scalar similarity, and rank homologs.
- **Inputs:** selected target and homolog `Structure` objects with chains, paths, and target pockets.
- **Outputs:** ProBiS files, parsed dictionaries, `ProteinData.probis_score`, and `ranked_proteins`.
- **Primary public functions:** `SurfaceService.surface_extraction`, `compare_surfaces`, `manage_results`, `probis_similarity_score`.
- **Internal helpers:** `get_results`, `parse_results`, `parse_aligned_pair`, `_process_homologs`, `_rank_homologs`.
- **Configuration:** commands use local `./probis` and `parameters.imp`. Base score is `0.6*aligned_vertices + 0.25*alignment_score + 0.15/(1+sva)`. No pocket match halves the base score; otherwise it is multiplied by `1 + 0.5*overlap + 0.3*alignment-in-pocket + 0.2*binding-score`.
- **External dependencies:** ProBiS executable, JSON/filesystem.
- **Determinism:** parsing, score calculation, and Python's stable descending sort are deterministic for fixed ProBiS output. ProBiS execution is an external dependency.

### Pharm2D consensus and FDA screening

- **Purpose:** generate Pharm2D and Morgan fingerprints, form threshold consensuses, cache candidate fingerprints in SQLite, compute two Tanimoto similarities, and retain the strongest combined candidates.
- **Inputs:** homolog training `Ligand` objects and the FDA ligand collection.
- **Outputs:** `ConsensusFingerprintDict`, cached `LigandFingerprint` objects, `Pharm2DResult`, and `ScreeningHit` objects.
- **Primary public functions:** `Pharm2DScreening.set_consensus`, `set_screen_dataset`, `compare_many_to_many`.
- **Internal helpers:** ligand molecule attachment, factory initialization, `_featurize_ligand`, `fps_to_consensus`, `_get_or_compute_fp`, cache load/store, RDKit bulk Tanimoto.
- **Configuration:** `Pharm2DConfig`; `ScreeningService` uses defaults. Combined score is `0.6*pharm2d_similarity + 0.4*morgan_similarity`; candidates are sorted descending and truncated to `max(1, floor(25%))`.
- **External dependencies:** RDKit and SQLite cache/database files.
- **Determinism:** deterministic for fixed RDKit version, configuration, molecule inputs, and valid cache state.

### Pharm3D consensus and screening

- **Purpose:** prepare conformers, align lowest-energy training conformers, extract and cluster chemical features, learn pair distances, construct an RDKit pharmacophore, enumerate atom-disjoint matches, and score embeddings.
- **Inputs:** a selected training subset and the top Pharm2D FDA hits.
- **Outputs:** `Pharm3DConsensus` and `Pharm3DResult` objects; lower scores rank earlier.
- **Primary public functions:** `Pharm3DScreening.set_consensus`, `set_screen_dataset`, `compare_many_to_many`.
- **Internal helpers:** `prepare_ligands_geometry`, conformer generation/filtering, `ConsensusBuilder.run`, feature clustering/statistics/bounds, `MatchPharmacophoreToMol`, embedding enumeration, presence/radial/geometric scoring.
- **Configuration:** screening service sets feature consensus 0.4, proximity 2.5, pair support 0.2, bounds multiplier 3, 20 conformers, and 10 kcal/mol energy window. Final embedding score is 0.55 geometry + 0.25 presence + 0.20 radial.
- **External dependencies:** RDKit ETKDG, MMFF/UFF, O3A, Pharm3D matching, NumPy.
- **Determinism:** not guaranteed across uncached first runs because ETKDG uses random coordinates without an explicit seed. The in-process conformer cache is keyed only by ligand ID and can make repeated calls reuse coordinates.

### Final result assembly and export

- **Purpose:** return per-protein ranked hit records and write them to disk.
- **Inputs:** ranked homologs, per-hit Pharm2D/Pharm3D results, training ligands.
- **Outputs:** a list of dictionaries and `ranked_protein_virtual_screening.json`.
- **Primary public function:** `ScreeningService.screen_ranked_proteins()`; export is internal `_export_results()`.
- **Internal helpers:** `select_pharm3d_training_set`, local `get_pharm3d_score`, dictionary assembly.
- **Configuration:** Pharm3D training size is `min(max(10, floor(30% of n)), 30, n)` when at least two ligands exist.
- **External dependencies:** filesystem JSON export.
- **Determinism:** deterministic for fixed upstream results. Hits without Pharm3D embeddings receive an internal sort key of infinity and preserve prior 2D order because sorting is stable.

# 3. Data Flow

### 3.1 Protein and homolog state

```text
UniProt accession: str
  |
  +--> UniProtEntry (Pydantic)
  +--> UniParc JSON --> list[str] alternate accessions
  |
  v
ProteinData
  identity: ProteinIdentity
  annotations: None
  structures: None
  ligands: None
  scores: None
  |
  +--> SequenceAnnotations
  |      interpro_ids, domains, active_sites
  |    wrapped as ProteinAnnotations
  |
  +--> list[ExperimentalStructure | HomologyStructure | AlphaFoldStructure]
  |    wrapped as ProteinStructures
  |
  v
PipelineState.target_protein
  |
  +--> best_structure
  +--> ProBiS surface file
```

```text
selected InterPro ID
  |
  v
list[UniProtEntry]
  |
  v
list[ProteinData] candidate_proteins
  |
  +--> ligand.db rows --> LigandRecord --> Ligand
  +--> ProteinLigands attached
  |
  v
filtered_proteins
  |
  +--> best_structure
  +--> ProBiS parsed result dict
  +--> ProteinScores.probis_score
  |
  v
ranked_proteins (descending probis_score)
```

`PipelineState.interpro_group_id` and `PipelineState.ranked_ligands` are defined but are not populated by the active orchestration.

### 3.2 Structure evolution

```text
UniProtKBCrossReference
  |
  v
Base Structure record
  ExperimentalStructure | HomologyStructure | AlphaFoldStructure
  |
  +--> provider-specific metadata and Chain coverage
  +--> StructureAnalysis.structural_plausibility
  +--> StructurePaths
  |
  v
raw CIF file
  |
  +--> gemmi CLI chain selection/cleaning
  v
clean PDB file
  |
  +--> bound-ligand pocket OR P2Rank predicted pocket
  v
StructureAnalysis.true_pocket + true_pocket_certainty
  |
  +--> fpocket Pocket records
  +--> residue-to-UniProt mapping
  +--> Pocket.binding_site_score
  v
selected best_structure
  |
  +--> ProBiS .srf/.nosql/results.json
  v
parsed alignment dict --> ProteinData.probis_score
```

### 3.3 Ligand and Pharm2D evolution

```text
SQLite activity + ligand row
  |
  v
LigandRecord
  |
  v
Ligand
  |
  +--> SMILES parsing
  v
LigandMol (RDKit Mol)
  |
  +--> Pharm2D ExplicitBitVect
  +--> Morgan ExplicitBitVect
  v
FingerprintDict
  |
  +--> LigandFingerprint for screening/cache
  +--> many training fingerprints
           |
           v
      ConsensusFingerprintDict
  |
  v
Pharm2DResult(pharm2d_similarity, morgan_similarity)
  |
  v
ScreeningHit(combined_score=0.6/0.4 weighted similarity)
```

### 3.4 Pharm3D evolution

```text
Ligand
  |
  +--> SMILES -> RDKit Mol -> explicit hydrogens
  +--> ETKDG conformers -> MMFF/UFF optimization -> energy filtering
  v
LigandConformers
  |
  +--> lowest-energy conformer selection and O3A alignment
  +--> RDKit feature extraction/deduplication
  v
LigandPharmacophore
  |
  +--> greedy spatial Cluster records
  +--> support filtering
  v
ConsensusFeature list + FeatureLigandMap
  |
  +--> PairwiseModel (means, raw stds, floored model stds)
  +--> RDKit Pharmacophore with bounds
  v
Pharm3DConsensus
  |
  +--> candidate feature mappings
  +--> atom-disjoint embeddings
  +--> conformer-specific diagnostics and weighted score
  v
Pharm3DResult
```

### 3.5 Final JSON shape

```json
[
  {
    "protein": "<homolog UniProt ID>",
    "probis_score": 0.0,
    "num_training_ligands": 0,
    "hits": [
      {
        "ligand_id": "...",
        "smiles": "...",
        "training_affinity_type": null,
        "training_affinity_value": null,
        "pharm2d_similarity": 0.0,
        "morgan_similarity": 0.0,
        "combined_2d_score": 0.0,
        "pharm3d_score": null
      }
    ]
  }
]
```

This is a per-homolog result list, not one globally ranked ligand list.

# 4. Dependency Graph

### 4.1 Active modules

| Module | Direct internal imports | Imported by | Role |
|---|---|---|---|
| `main.py` | `src.app` | CLI/runtime | Entry point |
| `src/app/__init__.py` | `context.progress`; lazy `application` | `main.py`, package imports | Infrastructure facade |
| `src/app/context.py` | `progress.RichProgressSink` | protein, ligand, structure, downloader services | Infrastructure: HTTP session, retry policy, progress instance, logging handler |
| `src/app/progress.py` | none beyond Rich | application and services | Infrastructure |
| `src/app/application.py` | ligand, protein, structure-selection, surface services; pipeline/domain models | lazy app facade | Pipeline orchestration |
| `src/app/protein_services.py` | context/progress; pipeline, scoring, structure, UniProt domains | application, structure service, pocket service | External wrappers + parsing + structure algorithms |
| `src/app/services/structure_services.py` | protein metadata, pocket, downloader, context/progress, structure domains | application | Stateful orchestration + structure algorithms |
| `src/app/services/structure_downloader.py` | context/progress, structure/pipeline domains | structure service | External I/O wrapper |
| `src/app/services/pocket_service.py` | protein service and structure/pipeline domains | structure service | External tool wrapper + pocket algorithms |
| `src/app/surface_services.py` | progress and structure domain | application | External ProBiS wrapper + scoring logic |
| `src/app/ligand_services.py` | context/progress, pipeline/ligand/pharm domains, Pharm2D/3D screeners | application | SQLite wrapper + screening orchestration |
| `src/domain/common.py` | none | domain/scoring/structure | Foundational types |
| `src/domain/uniprot.py` | Pydantic/dataclasses | protein service, application | API/domain models |
| `src/domain/structure.py` | common | pipeline and structure/surface services | Domain models |
| `src/domain/pipeline.py` | ligand, structure, UniProt | application/services | Mutable pipeline state |
| `src/domain/ligand.py` | common, RDKit | ligand service and pharmacophore pipeline | Domain models |
| `src/domain/pharm.py` | ligand, RDKit, NumPy | Pharm2D/3D and ligand service | Configuration/result/intermediate models |
| `src/domain/scoring.py` | common | protein service | Algorithm constants |
| `src/pharma/pharmacophore/pharma2d/screening.py` | consensus, factories, featurizer, preparation, hashing | ligand service | Stateful algorithm + SQLite cache |
| `src/pharma/pharmacophore/pharma2d/consensus.py` | generic consensus, featurizer, hashing | Pharm2D screening | Algorithmic logic |
| `src/pharma/pharmacophore/pharma2d/factories.py` | Morgan, common feature factory, signature factory | Pharm2D screening | Factory infrastructure |
| `src/pharma/pharmacophore/pharma2d/featurizer.py` | Pharm2D/Morgan fingerprint functions, hashing | consensus and screening | Algorithmic logic |
| `src/pharma/pharmacophore/pharma2d/fingerprints.py` | fingerprint conversion | featurizer | Algorithmic logic |
| `src/pharma/pharmacophore/pharma2d/sig_factory.py` | RDKit | factory builder | Configuration/factory logic |
| `src/pharma/fingerprints/consensus.py` | NumPy/RDKit | Pharm2D consensus | Pure algorithm |
| `src/pharma/fingerprints/morgan.py` | RDKit | Pharm2D factories/featurizer | Factory + algorithm wrapper |
| `src/pharma/utils/fingerprint_conversions.py` | RDKit | Pharm2D fingerprints | Pure conversion |
| `src/pharma/utils/hashing.py` | RDKit/hashlib | Pharm2D consensus/featurizer/screening | Identity/cache serialization utilities |
| `src/pharma/pharmacophore/pharma3d/screening.py` | progress, domains, Gaussian NLL, feature factory, consensus builder, preparation | ligand service | Stateful algorithm/orchestration |
| `src/pharma/pharmacophore/pharma3d/consensus_builder.py` | clustering, ligand features, domains, RDKit/NumPy | Pharm3D screening | Algorithmic logic |
| `src/pharma/pharmacophore/pharma3d/feature_clustering.py` | ligand/pharm domains, RDKit | consensus builder | Stateful greedy algorithm |
| `src/pharma/pharmacophore/pharma3d/ligand_features.py` | ligand domain, RDKit | consensus builder | Algorithmic conversion |
| `src/pharma/pharmacophore/common/feature_factory.py` | RDKit | Pharm2D and Pharm3D factories | Infrastructure factory |
| `src/pharma/preparation/ligand_preparation.py` | chemistry SMILES/conformers, ligand domain | Pharm2D/3D screening | Stateful preparation |
| `src/pharma/chemistry/smiles.py` | RDKit | ligand preparation | External-library wrapper |
| `src/pharma/chemistry/mol.py` | conformer generation | ligand preparation | External-library wrapper |
| `src/pharma/preparation/conformer_generation.py` | ligand domain, RDKit | chemistry/mol | Stateful algorithm + in-memory cache |
| `src/pharma/math/gaussian_nll.py` | NumPy | Pharm3D screening | Pure algorithm |

### 4.2 Modules present but not in the active call graph

- `src/clients/*` contains separate fetch/parse modules for UniProt, InterPro, SIFTS, Swiss-Model, AlphaFold, ChEMBL, and BindingDB. The active application services perform their own HTTP calls instead.
- `src/pockets/*`, `src/surfaces/*`, `src/scoring/*`, and `src/io/download.py` are parallel implementations not imported by `main.py` or the application services.
- `src/domain/probis.py` defines typed ProBiS results, but `SurfaceService` returns dictionaries.
- `src/pharma/pharmacophore/pharma3d/embedding.py`, `geometry.py`, `tolerance.py`, and `similarity.py` are not called by the main Pharm3D screening path. `screening.py` contains its own distance/scoring helpers.
- `src/bindingdb.py` is not used by the runtime ligand queries.

### 4.3 Cycles and architectural smells observed

- No direct active import cycle remains in the current tree. `src.app.__init__` loads `LigandMinerApplication` lazily, which prevents application imports from recursively re-entering the app package during Pharm3D imports.
- `src/app/services/structure_services.py` imports `PocketService` and `UniProtService`; `PocketService` also imports `UniProtService`. This is not a cycle, but couples pocket algorithms to the broad protein-service module.
- `src/pharma/pharmacophore/pharma3d/screening.py` imports `src.app.progress.NullProgressSink`, so algorithm code depends on the application layer.
- Active service modules duplicate the unused client, pocket, surface, scoring, and download module families listed above.
- Structure acquisition, metadata parsing, scoring constants, provider parsing, and UniProt/InterPro/SIFTS access are combined in the 774-line `protein_services.py` module.
- Result representation crosses typed dataclasses and ad hoc dictionaries: `ProBiSResult` is unused, while the surface and final screening stages exchange dictionaries.
- Several service methods mutate passed domain objects and also return them; the application relies primarily on mutation.

# 5. Public APIs

Python modules do not declare a formal stability policy. The following are the externally invoked facades and service interfaces in the active pipeline; underscore-prefixed application stages are implementation details.

### Entry and orchestration

| API | Expected input | Output | Side effects |
|---|---|---|---|
| `main.main(uniprot_id)` | UniProt accession `str` | `None` | Runs complete application |
| `LigandMinerApplication()` | No arguments | Application instance | Constructs five service objects |
| `LigandMinerApplication.main(uniprot_id: str, progress: ProgressSink)` | Accession and progress sink/context manager | `None` | Network, subprocess, SQLite, directories/files, JSON export, domain mutation |

### Protein and structure services

| API signature | Expected input | Output | Side effects |
|---|---|---|---|
| `UniProtService(session_obj=session)` | requests-compatible session | Service | Retains session |
| `fetch_uniprot_json_from_id(uniprot_id: str) -> UniProtEntry` | Accession | Validated entry | HTTP GET |
| `fetch_uniprot_json_from_interpro(interpro_id, progress, reviewed=True, page_size=500) -> list[UniProtEntry]` | InterPro ID and sink | Validated entries | Paged HTTP GET, progress updates |
| `get_uniparc_json(uniparc_id)` / `parse_uniparc_json(json_data)` | UniParc ID/payload | JSON / aliases | HTTP on fetch |
| `prepare_protein_metadata(protein, entry, progress=None) -> None` | `ProteinData`, `UniProtEntry` | `None` | Mutates protein; SIFTS HTTP calls |
| `manage_interpro(interpro_ids, uniprot_id, progress=None)` | IDs and target label | `list[UniProtEntry]` | InterPro/UniProt HTTP calls, progress |
| `StructureMetadataService(session_obj=session)` | HTTP session | Service | Retains session |
| `metadata_structures_enrichment(structures, domain, active_sites, uniprot)` | Structure/annotation lists | `list[Structure]` | HTTP for model providers; mutates structures |
| `structure_selection_score(structure, w_structure=0.7, w_pocket=0.3)` | Enriched structure | `float` | None |
| `StructureSelectionService(...)` | Optional injected services/session | Service | Constructs defaults |
| `select_best_structure(protein, progress=None) -> Structure` | Annotated protein with structures | Selected structure | Downloads, subprocesses, files, mutations |
| `StructureDownloader.download_structures(protein, progress, task_key, root="./", max_workers=6)` | Protein and progress | Downloaded structures | Concurrent HTTP/file writes |
| `StructureDownloader.clean_structure(input_cif, output_pdb, structure)` | Paths and structure | `None` | Runs gemmi executable, writes PDB |
| `PocketService(uniprot_service)` | UniProt service | Service | Retains mapper |
| `fpocket_enrichment(data)` | Object with structures | Same object | Runs fpocket, reads/writes/deletes directories, mutates pockets |
| `pocket_residue_enrichment(protein)` | Identified structures | Same protein | Reads pocket files, mutates mappings |
| `score_pockets(protein, *, residue_sasa=None, min_surface_residues=6, min_surface_sasa=5.0)` | Annotated structures | Same protein | Mutates pocket scores |
| `get_pocket_from_prank(structure)` | Prepared structure | `(probability, residues)` | Runs P2Rank, reads CSV |

### Surface and ligand services

| API signature | Expected input | Output | Side effects |
|---|---|---|---|
| `SurfaceService.surface_extraction(structure, progress=None)` | Prepared structure | Cached path or `None` | Runs ProBiS, writes `.srf` |
| `compare_surfaces(target_structure, similar_structure, progress=None)` | Two structures with surfaces | `None` | Runs ProBiS, writes `.nosql` |
| `manage_results(target_structure, similar_structure, progress=None)` | Compared structures | Parsed `dict` or `None` | Runs ProBiS results command, reads JSON |
| `probis_similarity_score(parsed)` | Parsed result dictionary | `float` | None |
| `LigandService(session_obj=session, base_dir=None)` | Session and repository base | Service | Resolves `resources/ligand.db` |
| `get_ligands_for_proteins(uniprot_ids) -> list[LigandRecord]` | Accessions | Ligand records | SQLite read; connection is not explicitly closed |
| `get_fda_approved_ligands() -> list[LigandRecord]` | None | FDA ligand records | SQLite read; connection is not explicitly closed |
| `fetch_ligands_for_protein(uniprot_id) -> list[Ligand]` | Accession | Deduplicated ligands | SQLite read; closes connection |
| `get_ligands_from_chembl(uniprot)` | Accession | Parsed dictionary or `None` | ChEMBL HTTP calls; not used by main path |
| `ScreeningService(ligand_service=None)` | Optional ligand service | Service | Constructs default if absent |
| `select_pharm3d_training_set(training_ligands)` | `list[Ligand]` | Selected list or empty | None |
| `screen_ranked_proteins(pipeline, fda_approved_ligands, progress=None) -> list[dict]` | Ranked state and FDA ligands | Final grouped results | SQLite and pharmacophore caches; extensive domain/library work |

### Pharmacophore facades

| API signature | Expected input | Output | Side effects |
|---|---|---|---|
| `Pharm2DScreening(config: Pharm2DConfig)` | Config | Stateful screener | Opens/initializes SQLite cache |
| `set_consensus(ligands)` | Training `Ligand` list | `None` | Parses molecules, updates screener state |
| `set_screen_dataset(ligands)` | Candidate list | `None` | Fingerprint generation/cache writes |
| `compare_one_to_one(reference, candidate)` | Two ligands | `(Ligand, Pharm2DResult)` | Consensus/cache changes |
| `compare_one_to_many(reference, candidates)` | Reference and candidates | Result pairs | Consensus/cache changes |
| `compare_many_to_many()` | Previously set state | Result pairs | Bulk RDKit scoring |
| `close()` | None | `None` | Commits/closes cache; suppresses close exceptions |
| `Pharm3DScreening(config: Pharm3DConfig)` | Config | Stateful screener | Initializes RDKit feature factory |
| `set_consensus(ligands)` | Training ligands | `None` | Conformer generation/cache and consensus mutation |
| `set_screen_dataset(ligands)` | Candidate ligands | `None` | Conformer generation/cache and state mutation |
| `compare_one_to_one(reference, candidate)` | Two ligands | `(Ligand, Pharm3DResult)` | Rebuilds consensus and prepares candidate |
| `compare_one_to_many(reference, candidates)` | Reference and candidates | Result pairs or `None` | Rebuilds consensus/dataset |
| `compare_many_to_many()` | Previously set state | Result pairs or `None` | Updates active conformers on RDKit features |

### Core public data contracts

- Pipeline: `PipelineState`, `ProteinData`, `ProteinAnnotations`, `ProteinStructures`, `ProteinLigands`, `ProteinScores`.
- Structure: `ExperimentalStructure`, `HomologyStructure`, `AlphaFoldStructure`, `StructurePaths`, `StructureAnalysis`, `Pocket`, `Chain`.
- Ligand: `LigandRecord`, `Ligand`, `LigandMol`, `LigandConformers`, `LigandPharmacophore`, `Conformer`.
- Pharm2D: `Pharm2DConfig`, `Fingerprint`, `FingerprintDict`, `ConsensusFingerprint`, `ConsensusFingerprintDict`, `LigandFingerprint`, `Pharm2DResult`.
- Pharm3D: `Pharm3DConfig`, `Cluster`, `ConsensusFeature`, `PairwiseModel`, `Pharm3DConsensus`, `PairDiagnostics`, `Embedding`, `Pharm3DResult`.
- Infrastructure: `ProgressSink`, `NullProgressSink`, `RichProgressSink`, `ProgressEvent`.

# 6. Failure Modes

| Location | Condition | Handling | Pipeline effect | Reporting |
|---|---|---|---|---|
| `main.py` | No CLI accession | Does not call application | Stops before pipeline | `print` usage line |
| `fetch_uniprot_json_from_id` | HTTP error | `raise_for_status()` exception | Aborts | No local logging |
| `get_uniparc_json` | Non-OK response | `RuntimeError` | Aborts | Exception text includes ID/status |
| `_initialize_target` | Current `prepare_protein_metadata` argument mismatch | Attribute/type failure when method accesses expected objects | Aborts before structure selection | No local catch |
| `parse_uniprot_cross_references` | Reached with valid entry | Returns undefined `annotations` | `NameError`; aborts | No local catch |
| `fetch_interpro_json` | Status not 200 | Returns `None` | Current ID is ignored; loop continues | Prints invalid response |
| `fetch_least_abstracted_interpro_id` | No usable family/domain or empty input | Dereferences `least_abstracted` while finishing | Aborts | No local catch |
| `fetch_uniprot_json_from_interpro` | Page status not 200 | `RuntimeError` | Aborts homolog discovery | No local logging |
| SIFTS retrieval | HTTP error | `raise_for_status()` | Aborts metadata preparation | No local catch |
| SIFTS parse | PDB key absent | `ValueError` caught by `prepare_protein_metadata` | Sets mapping to `None`; continues | No log |
| `StructureSelectionService.select_best_structure` | Current progress sink lacks `group()` | `AttributeError` | Aborts structure selection | No local catch |
| `StructureSelectionService.is_holo` | Current module references `gemmi` without importing it | `NameError` | Aborts structure preparation | No local catch |
| Provider metadata fetch | Non-200 Swiss-Model/AlphaFold response | `RuntimeError` | Aborts structure enrichment | No local catch |
| Provider parsing | No usable structures | `max([])` in best-model selection | `ValueError`; aborts | No local catch |
| Concurrent structure download | Individual HTTP/download failure | Caught per future; omitted from returned `downloaded` list | Download loop continues | Prints `[DOWNLOAD FAIL]` |
| Selection after download | A failed download remains in `protein.structures.structures` because returned list is ignored | Later clean/read can raise | Aborts selection | No surrounding catch |
| Structure cleaning | gemmi executable failure | `CalledProcessError` | Aborts selection | Subprocess output suppressed |
| P2Rank | Command failure, missing CSV, malformed row | Propagates subprocess/pandas/parsing exception | Aborts selection | Output suppressed; no catch |
| fpocket missing PDB | Missing prepared file | Sets `fpockets=[]`; continues structure loop | Recoverable for that structure | Prints `[FPOCKET SKIP]` |
| Cached fpocket parse | Parse failure | Clears pockets, deletes/re-runs output | Recoverable if rerun works | Prints cache failure |
| fpocket run/parse | Any exception | Sets `fpockets=[]`; continues | Recoverable for that structure | Prints run failure |
| Pocket residue mapping | Structure chain absent | Re-raises `StopIteration` | Aborts selection | Prints detailed chain context |
| Best structure selection | No structures | `max([])` | Aborts | No local catch |
| Application homolog processing | Homolog has no structures | Explicit skip | Homolog remains unscored | Progress only |
| Surface extraction/comparison | Missing input/surface | `FileNotFoundError` | Aborts whole run | No local catch |
| ProBiS command | Non-zero exit | `CalledProcessError` | Aborts whole run | Output suppressed |
| ProBiS results | Empty JSON list | Returns `None` | Homolog is skipped and remains unscored | Prints skip message |
| ProBiS JSON shape | Missing alignment/scores fields | Key/index/max exception | Aborts | No local catch |
| Homolog ranking | `probis_score is None` | Protein filtered out | Continues | No log |
| Ligand SQLite queries | Missing DB/table, lock, malformed row | SQLite/conversion exception | Aborts | No local catch |
| Ranked proteins empty | `screen_ranked_proteins` guard | Returns `[]` | Export writes empty list | Progress not started |
| No training ligands for one protein | Explicit guard | Skips protein without appending a result record | Continues next protein | Progress only |
| Pharm2D construction in `screen_ranked_proteins` | Screener is immediately closed before `set_screen_dataset` | Subsequent cache operation uses closed connection | `sqlite3.ProgrammingError`; screening aborts | Pharm2D error logging may report cache operation |
| Pharm2D consensus | Empty training fingerprints or invalid threshold/dimensions | `ValueError` | Aborts current screening call | Warning/error logs |
| Invalid ligand SMILES | RDKit returns no molecule | `ValueError` re-raised through preparation | Aborts Pharm2D batch; Pharm3D preparation batch also re-raises | Ligand ID/SMILES/error logs |
| Pharm3D conformer embedding | No conformers | Returns `None`; batch omits that ligand | Recoverable for that ligand | Warnings |
| Pharm3D training preparation | No valid prepared ligands | Sets consensus to `None`; returns | Later many-to-many returns `None` | Warning |
| Pharm3D feature matching | No mapping | Returns empty result | Candidate remains with no embeddings | Warning |
| Pharm3D screening state missing | No consensus or no dataset | Returns `None` | Caller assumptions may fail | Error log |
| Export | File open/write or JSON failure | Propagates | Aborts final phase | Success print occurs only after dump |

The application has no top-level exception handler. Except where the table identifies an explicit per-item skip/catch, exceptions propagate out of `LigandMinerApplication.main()`.

# 7. Testability Assessment

| Stage | Classification | Unit tests | Integration tests | Mocks/fakes | Implicit contracts to preserve |
|---|---|---|---|---|---|
| CLI/application sequencing | Pipeline orchestration | Useful for call order and state handoff | Required for complete workflow | Injected services, progress, filesystem | Seven fixed stages; mutation order; `main()` returns `None` |
| UniProt/UniParc/InterPro/SIFTS retrieval | External service wrapper | Payload/status/pagination parsing | Required against API contracts | HTTP session and progress sink | Pydantic validation; reviewed-only query; Link pagination; first-family selection |
| UniProt feature/cross-reference parsing | Pure algorithm | Best fit | Optional with recorded payloads | None | Exact feature type strings; database allowlist; chain/range conventions |
| Structure provider parsing/scoring | Pure algorithm plus external wrapper | Best fit for parsing and score formulas | Provider contract checks | HTTP for provider fetches | Calibration exponents, weights, rounding, maximum-provider-model selection |
| Structure selection | Pipeline orchestration | Useful for shortlist and selection decisions | Required for actual artifacts/tools | Downloader, metadata, pocket service, progress | In-place plausibility sort; top 100; maximum combined score |
| Holo-pocket detection | Stateful algorithm | Useful with small structure fixtures | Useful with real gemmi/BioPython structures | PubChem method is not called by selection | Artifact filters, 5 Å contacts, six-residue minimum, largest pocket |
| P2Rank/fpocket | External service wrapper | Parser/command construction units | Required for executable/file formats | `subprocess`, filesystem, pandas | Cache reuse rules; custom-pocket encoding; failures clear pocket list |
| Pocket residue mapping/scoring | Stateful algorithm | Best fit | Useful with SIFTS/fpocket fixtures | SIFTS mapping fixture | SIFTS preferred for matching chain; offset fallback; main-domain choice; score weights |
| Structure download/clean | External service wrapper | Command and cache path units | Required with HTTP/gemmi | HTTP, executor, subprocess, filesystem | Existing raw file is reused; per-download failures are caught |
| ProBiS parsing/scoring | Pure algorithm | Best fit | Required for executable JSON shape | JSON fixtures | Best alignment by aligned vertices; residue tuple format; exact score formulas |
| ProBiS execution | External service wrapper | Command/cache decision units | Required with executable | subprocess and filesystem | `.srf`, `.nosql`, and `results.json` cache checks |
| Homolog ranking | Pure algorithm | Best fit | Covered through orchestration | None | Exclude `None`; descending stable order |
| Ligand database access | External service wrapper | Query/row mapping tests | Required with schema-compatible SQLite | Temporary SQLite DB | Supported affinity types; strongest activity subqueries; exact record fields |
| Screening service | Pipeline orchestration | Useful with fake screeners/services | Required for Pharm2D→Pharm3D handoff | Ligand service and screeners | 0.6/0.4 score; top 25%; training subset size; lower Pharm3D score wins |
| Pharm2D fingerprint/consensus math | Pure algorithm | Best fit | Useful with real RDKit molecules | Usually none | Fixed vector dimensions; threshold uses `ceil`; Tanimoto range/order |
| Pharm2D screening/cache | Stateful algorithm | Best fit with temporary SQLite | Useful across reopen/version cases | Temporary DB | Cache key is ligand ID + config hash; dataset order preserved |
| Pharm3D preparation/consensus | Stateful algorithm | Useful with deterministic prepared ligands | Required with real RDKit conformers | Expensive preparation may be isolated | lowest-energy alignment; greedy feature order; support ceilings; std floor |
| Pharm3D matching/scoring | Stateful algorithm | Best fit with small RDKit features | Useful end-to-end | Usually none | atom-disjoint embeddings; lower weighted score is better; empty mappings allowed |
| JSON result assembly/export | Pipeline orchestration | Shape/order and serialization units | Final artifact integration | Temporary filesystem | Per-protein grouping; nullable Pharm3D/training affinity; fixed output filename |

# 8. Open Questions

These questions cannot be resolved from the active source without choosing intended behavior.

1. `LigandMinerApplication._initialize_target()` calls `prepare_protein_metadata()` with objects that do not match its signature. It is unclear whether the caller or the service signature represents the intended contract.
2. `parse_uniprot_cross_references()` builds `InterProIDs` but returns undefined `annotations`. It is unclear whether it should construct `SequenceAnnotations(interpro_ids=...)` or use another representation.
3. `uniprot_cross_references_to_structures()` is a second, unused parser and references `ref` from outside the database-group loop. It is unclear which cross-reference parser is authoritative and where PDB properties/chains are intended to be populated.
4. `StructureSelectionService.select_best_structure()` requires `progress.group()`, but neither `ProgressSink`, `NullProgressSink`, nor `RichProgressSink` defines it. No alternate progress implementation providing this method exists in the repository.
5. `StructureSelectionService.is_holo()` uses `gemmi` without importing it in that module. The intended source of that name is not present.
6. Failed concurrent downloads are collected and reported, but the returned successful list is ignored. It is unclear whether later stages should process all original structures or only successful downloads.
7. The InterPro member query does not exclude the target accession even though `manage_interpro()` receives it. It is unclear whether self-inclusion is intentional.
8. If none of the target InterPro IDs resolves to a family or domain, the code dereferences `None`. No fallback homolog-selection policy is defined.
9. `select_pharm3d_training_set()` sorts raw affinity values descending, while ligand SQL ranking elsewhere uses descending `p_affinity` and lower raw `Ki`/`Kd`/`IC50`/`EC50` conventionally represents stronger activity. The intended selection direction is not stated in source.
10. `ScreeningService.screen_ranked_proteins()` closes its Pharm2D screener immediately after construction and then reuses it. The intended lifetime of the cache connection is unclear.
11. The final output is grouped per homolog and sorted primarily by Pharm3D score after a 2D cutoff. `PipelineState.ranked_ligands` remains empty, so no intended global ligand-ranking contract can be derived.
12. Structure, pocket, surface, scoring, download, and external-client implementations exist both under `src/app` and separate package directories, but only the app-service versions are wired. The intended status of the parallel modules is undocumented.
13. External executable versions and exact supported output schemas for gemmi CLI, P2Rank (`prank`), fpocket, and ProBiS are not pinned in the repository.
14. The existing integration test describes database tables and output fields that differ from the active queries and result assembly; it does not resolve which schema is authoritative.
15. The active cross-reference parser creates PDB structures without applying UniProt `Method`, `Resolution`, or `Chains` properties, while an unused parser contains property handling. The intended PDB metadata path is unclear.
16. Swiss-Model scoring reads `structure.analysis.domain_coverage` before `feature_structure_enrichment()` computes that coverage. The intended ordering of coverage enrichment and structural scoring is not stated.
