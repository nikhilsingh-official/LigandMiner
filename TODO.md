# TODO

## Pocket-Centric Structure Selection

### Goal
- Replace domain-based structure relevance heuristics with pocket-centric structure selection.
- Prefer structures that actually preserve the binding pocket neighborhood.
- Reject terminal fragments and truncated structures that are structurally irrelevant to pocket analysis.

### Core idea
- Detect candidate pocket evidence from each structure.
- Map pocket residues into UniProt coordinates.
- Build a cross-structure pocket consensus.
- Score each structure by how well it preserves that consensus pocket, plus structural quality terms.

### Proposed workflow
1. Process every candidate structure.
2. Classify each structure as holo or apo.
3. Extract pocket evidence from each structure.
4. Map pocket residues to UniProt residue indices.
5. Build a weighted consensus pocket from all structures.
6. Score each structure against the consensus pocket.
7. Use that score as the primary selector for best structure.

### Pocket evidence sources
- Holo ligand-contact residues from detected bound ligands.
- `fpocket` predicted pocket residues.
- `prank` predicted pocket residues.
- UniProt active-site / binding-site annotations as secondary support.

### Consensus pocket design
- Build the consensus in UniProt coordinates.
- Weight pocket evidence by confidence.
- Suggested evidence weighting:
  - holo ligand-contact pocket: highest confidence
  - holo structure with plausible predicted pocket: medium-high confidence
  - apo predicted pocket: medium confidence
  - weak/noisy pocket predictions: low confidence

### Pocket consensus outputs
- `consensus_pocket_core`
  - residues with repeated strong support across structures
- `consensus_pocket_extended`
  - moderately supported neighboring residues
- `consensus_pocket_confidence`
  - per-residue support values across all structures

### Structure scoring against consensus
- `consensus_pocket_core_coverage`
  - fraction of core pocket residues present in the structure
- `consensus_pocket_extended_coverage`
  - fraction of extended pocket residues present
- `consensus_pocket_overlap`
  - overlap between the structure's own detected pocket and the consensus pocket
- `pocket_continuity_score`
  - how intact the pocket region is
- `pocket_fragmentation_penalty`
  - penalty for internal gaps or breaks through the pocket region
- `structural_plausibility`
  - existing structure-quality score

### Pocket continuity and fragmentation
- Continuity means the pocket neighborhood is present as one intact region.
- Fragmentation means internal portions of the pocket neighborhood are missing.
- This should detect cases where a structure contains some pocket residues but is missing the middle of the binding region.

### Initial continuity / fragmentation metrics
- largest contiguous run of consensus residues present
- number of internal gaps through the consensus core
- maximum internal gap length
- number of missing core residues
- optional 3D connectedness check for the pocket residue cluster

### Selection policy
- Promote pocket-centric completeness to the primary structure-selection signal.
- Keep `active_site_coverage` as a secondary support term.
- Demote `domain_coverage` to a fallback or tiebreaker.

### Replacement plan
- Replace domain-driven relevance in best-structure selection.
- Keep existing structure-quality terms.
- Keep annotation overlap as a weak supporting feature.


## Method For Detecting The Correct Pocket From `fpocket` Results

### Goal
- Stop assuming the top `fpocket` pocket is the correct one.
- Select the `fpocket` pocket that best matches binding evidence.

### Pocket selection candidates
- all `fpocket` pockets for the structure
- holo ligand-contact residues when available
- `prank` predicted pocket residues
- UniProt active-site / binding-site residues
- cross-structure consensus pocket in UniProt coordinates

### Proposed `fpocket` pocket ranking strategy
- Score each `fpocket` pocket using:
  - overlap with holo ligand-contact residues
  - overlap with UniProt active-site residues
  - overlap with consensus pocket residues
  - pocket druggability score
  - pocket size / residue count
  - surface-exposure sanity
  - residue continuity around the pocket

### Priority logic
- If holo ligand contacts exist:
  - choose the `fpocket` pocket with strongest overlap to holo-contact residues
- Else if consensus pocket exists:
  - choose the `fpocket` pocket with strongest overlap to consensus pocket
- Else:
  - choose the best-scoring physically plausible `fpocket` pocket using druggability + geometry + annotation support

### Required outputs
- `selected_fpocket_id`
- `selected_fpocket_score`
- `fpocket_selection_reason`
- `fpocket_candidate_scores`

### Open design question
- decide whether pocket identity should be structure-local only or whether pocket IDs should also be aligned across homologous structures via mapped UniProt residues


## Alternative Input Methods

### Goal
- Support input modes beyond a single UniProt accession.

### Planned input modes
- UniProt ID
- PDB ID + chain selection
- PDB ID + residue range
- local PDB/mmCIF file + chain selection
- local PDB/mmCIF file + residue range
- custom binding-site residue list
- ligand-bound structure + explicit ligand identifier
- existing structure + user-specified pocket residues

### Questions to resolve
- how to normalize all inputs into a single internal target structure model
- how to map custom residue numbering into UniProt coordinates when UniProt is known
- how to support structures with author numbering vs canonical numbering
- how to handle partial inputs with no UniProt accession

### Implementation target
- define one normalized target-input object for all ingestion paths
- all downstream pipeline stages should accept the normalized target-input object rather than raw CLI-specific values


## Standardized Result Type

### Goal
- Replace ad hoc dictionaries and mixed return shapes with standardized result objects.

### Motivation
- many pipeline stages currently return raw dicts, tuples, mutated objects, or `None`
- standardized result types would make scoring, debugging, serialization, and downstream composition cleaner

### Proposed result families
- `PocketDetectionResult`
- `PocketConsensusResult`
- `StructureSelectionResult`
- `SurfaceComparisonResult`
- `HomologRankingResult`
- `LigandScreeningResult`
- `PipelineRunResult`

### Each result type should support
- typed fields
- confidence / score fields
- provenance metadata
- error / warning fields
- optional debug payloads
- serialization to JSON

### Standardization targets
- pocket detection output
- ProBiS parsing output
- structure scoring output
- homolog ranking output
- screening hit output
- final pipeline output

### Specific cleanup targets
- replace bare dicts in result parsing
- replace tuple returns where meaning is implicit
- define explicit success / empty / failed states
- unify score names and confidence semantics


## Follow-Up Engineering Tasks

### Short term
- implement pocket-consensus data model
- add consensus-based structure scoring
- add `fpocket` pocket-selection scoring
- add debug reporting for why a structure was selected or rejected

### Medium term
- add standardized result objects across major pipeline stages
- add alternative input methods and normalize them early
- add tests for truncated-structure rejection and pocket completeness scoring

### Long term
- compare apo vs holo structure selection quality on benchmark targets
- benchmark consensus-pocket selection against current domain-based selection
- add explainable selection summaries to final outputs

## ProBiS-style Binding Site Analysis

### Goal
- Use ProBiS to identify structurally similar binding sites and their known ligands.

### Proposed workflow
1. For each detected pocket in the target protein, use ProBiS to find structurally similar pockets in a database of known binding sites.
2. Retrieve the ligands known to bind to these similar pockets.
3. Rank these "transferred" ligands based on the similarity of the binding sites and other factors.
4. Add these transferred ligands to the list of candidate ligands for virtual screening.

### Integration
- This stage would fit naturally after the "Binding Site Identification" and "Binding Site Analysis" stages in the intended high-level workflow.
- It would extend the `SurfaceService` to handle these new ProBiS features.
- The results would be a new source of ligands for the "Ligand Collection" stage.

## Ligand Filtering

### Goal
- Implement a flexible ligand filtering pipeline to remove undesirable compounds before virtual screening.

### Proposed filters
- Drug-likeness (e.g., Lipinski's Rule of Five)
- ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties
- PAINS (Pan-Assay Interference Compounds)
- User-defined filters

### Integration
- This would be a new stage in the pipeline, after "Ligand Collection" and before "Pharmacophore Modelling".
- It would likely involve a new `FilteringService` and new domain models for representing filter criteria and results.

## Molecular Dynamics

### Goal
- Add support for running molecular dynamics simulations to validate docking poses and estimate binding free energies.

### Proposed workflow
1. System preparation (e.g., using GROMACS or AMBER).
2. Equilibration simulations.
3. Production simulations.
4. Trajectory analysis (RMSD, RMSF, hydrogen bonds).
5. Binding free energy calculations (e.g., MM/PBSA).

### Integration
- This would be a major new component of the pipeline, likely implemented as a new `MDService`.
- It would be one of the final stages, after "Docking".

## Post-analysis

### Goal
- Provide a comprehensive analysis of the simulation results to help the user interpret the data.

### Proposed analyses
- RMSD and RMSF plots.
- Hydrogen bond analysis.
- MM/PBSA results.
- Stability analysis.
- Final ranking of candidates based on all available data.

### Integration
- This would be the final stage of the pipeline, producing the final interactive report.
- It would likely involve a new `AnalysisService`.

## Refactoring and Cleanup

### Goal
- Improve the long-term maintainability and usability of the codebase.

### Tasks
- **Refactor `StructureService`**: This class is currently very large and handles too many responsibilities. It should be broken down into smaller, more focused services (e.g., a `PocketService`, a `StructureDownloadService`, etc.).
- **Configuration Management**: Externalize hardcoded paths and parameters into a configuration file (e.g., `config.yaml` or similar).
- **Improve Logging**: The current logging is minimal. A more structured and informative logging setup would be beneficial for debugging.
- **Add Unit Tests**: The test coverage is sparse. More unit tests are needed, especially for the critical service classes.
