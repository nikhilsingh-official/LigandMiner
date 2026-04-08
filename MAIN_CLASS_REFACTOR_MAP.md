# `main.py` Class Refactor Map

This refactor keeps the pipeline flow intact and moves the former `main.py` statements into grouped service classes, following the same idea as `Pharm3DScreening`: related state and behavior now live together in a class.

## New class layout

### `LigandMinerApplication`
File: `src/app/application.py`

Purpose:
- Owns the full top-level pipeline that used to live in `main(uniprot_id)`.
- Wires the service classes together and runs the same stages in the same order.

Statements moved here:
- The full body of the old `main(uniprot_id)` function.
- The orchestration for:
  - target UniProt fetch
  - target metadata preparation
  - best-structure selection
  - InterPro homolog discovery
  - homolog metadata enrichment
  - ligand batch lookup
  - surface comparison and ProBiS scoring
  - FDA ligand loading
  - pharmacophore screening
  - JSON result writing

### `UniProtService`
File: `src/app/protein_services.py`

Purpose:
- Holds UniProt / InterPro / UniParc / SIFTS fetching and parsing logic.
- Builds `ProteinData.annotations` and initial `ProteinData.structures`.

Statements moved here:
- `fetch_interpro_json`
- `map_pdb_to_uniprot`
- `map_uniprot_chains_to_pdb`
- `fetch_uniprot_json_from_id`
- `fetch_uniprot_json_from_interpro`
- `parse_uniprot_cross_references`
- `parse_uniprot_features`
- `parse_uniprot_json`
- `parse_chains`
- `get_uniparc_json`
- `parse_uniparc_json`
- `fetch_sifts_json`
- `parse_sifts_json`
- `prepare_protein_metadata`
- `manage_interpro`
- `fetch_least_abstracted_interpro_id`

### `StructureMetadataService`
File: `src/app/protein_services.py`

Purpose:
- Keeps structure metadata expansion, structure-type parsing, scoring, and path assignment together.
- Handles PDB, SwissModel, and AlphaFold structural metadata using the same formulas as before.

Statements moved here:
- `get_swiss_model_intermediate`
- `get_alpha_fold_intermediate`
- `parse_swiss_model_intermediate_json`
- `parse_alpha_fold_intermediate_json`
- `score_alpha_fold_structure`
- `score_alpha_fold_structures`
- `score_pdb_structure`
- `score_swiss_model_structure`
- `score_swiss_model_structures`
- `structure_selection_score`
- `feature_structure_enrichment`
- `find_best_swiss_model_structure`
- `find_best_alpha_fold_structure`
- `add_paths`
- `metadata_structure_enrichment`
- `metadata_structures_enrichment`
- `pdb_specific_enrichment`
- `pdb_specific_filtration`

### `StructureService`
File: `src/app/structure_services.py`

Purpose:
- Holds structure preparation, holo-pocket detection, fpocket execution, residue mapping, and best-structure selection.
- This is the main structural-processing class.

Statements moved here:
- `clean_structure`
- `build_neighbor_search`
- `ligand_passes_basic_filters`
- `get_pocket_residues`
- `search_pubchem_by_smiles`
- `is_holo`
- `download_structure`
- `download_structures`
- `residues_to_fpocket_P`
- `run_fpocket`
- `parse_fpocket_info`
- `format_chains`
- `pdb_residue_exists`
- `has_sidechain`
- `normalize_resolution`
- `has_valid_binding_pocket`
- `extract_pocket_residues`
- `get_chains_from_motif`
- `expand_residues_spatially`
- `residues_to_motif`
- `fpocket_enrichment`
- `parse_fpocket_atm_pdb`
- `geometry_score`
- `physical_sanity_score`
- `compute_binding_site_score`
- `score_pockets`
- `pocket_residue_enrichment`
- `map_residues_to_uniprot`
- `get_pocket_from_prank`
- `shortlist_structures`
- `select_best_structure`

Supporting types moved here:
- `AminoAcid`
- `StructureResidue`
- `StructureBindingSite`

### `SurfaceService`
File: `src/app/surface_services.py`

Purpose:
- Encapsulates ProBiS extraction, comparison, result parsing, and similarity scoring.

Statements moved here:
- `probis_similarity_score`
- `parse_aligned_pair`
- `parse_results`
- `manage_results`
- `surface_extraction`
- `compare_surfaces`
- `get_results`

### `LigandService`
File: `src/app/ligand_services.py`

Purpose:
- Holds ligand-source access logic: ChEMBL-derived ligand gathering and SQLite-based ligand retrieval.

Statements moved here:
- `chembl_targets_from_uniprot`
- `chembl_activities_from_targets`
- `parse_chembl_activities`
- `get_ligands_from_chembl`
- `merge_ligand_outputs`
- `get_ligands_for_proteins`
- `get_fda_approved_ligands`
- `fetch_ligands_for_protein`

### `ScreeningService`
File: `src/app/ligand_services.py`

Purpose:
- Owns the ranked-protein virtual screening flow.
- Keeps the original two-stage Pharm2D -> Pharm3D screening logic unchanged.

Statements moved here:
- `screen_ranked_proteins`
- `select_pharm3d_training_set`

Supporting type moved here:
- `ScreeningHit`

### `NullProgress`
File: `src/app/protein_services.py`

Purpose:
- Preserves the no-op progress behavior that several original functions relied on when `progress` was absent.

Statements moved here:
- the full `NullProgress` class

## Remaining `main.py`

`main.py` is now only the CLI entrypoint:
- imports `LigandMinerApplication`
- defines `main(uniprot_id)` as a thin wrapper
- preserves the existing `if __name__ == "__main__":` behavior

## Shared non-class context

File: `src/app/context.py`

Moved here to avoid duplicating globals across classes:
- the global `Progress` configuration
- the `requests.Session` setup with retry adapter
- the `pharm2d` file logger setup

These are shared infrastructure objects, not behavior classes, so they were kept as module-level context.
