# LigandMiner Architecture Document

## 1. Overview

LigandMiner is a modular, end-to-end computational drug discovery platform. It identifies candidate ligands for a target protein by analyzing the protein's homologs, their structures, and their known ligands. The application is architected as a sequential pipeline, orchestrated by a central `LigandMinerApplication` class, with distinct tasks handled by specialized `Service` classes.

## 2. Core Architecture

The application follows a service-oriented architecture. The main components are:

*   **Application (`src/app/application.py`)**: The `LigandMinerApplication` class is the entry point and acts as the central controller, orchestrating the execution of the various services.
*   **Services (`src/app/*.py`)**: A collection of service classes, each responsible for a specific part of the pipeline (e.g., `UniProtService` for fetching protein data, `StructureService` for handling protein structures, `SurfaceService` for ProBiS-based surface analysis, `LigandService` for ligand data, and `ScreeningService` for pharmacophore screening).
*   **Domain Models (`src/domain/*.py`)**: A set of data classes that define the core entities of the application, such as `ProteinData`, `Structure`, `Ligand`, and `Pocket`. These models are used to pass data between the services.
*   **Clients (`src/clients/*.py`)**: A collection of modules for interacting with external APIs (UniProt, InterPro, etc.).
*   **Pharmacophore Engine (`src/pharma/`)**: A dedicated package for 2D and 3D pharmacophore modeling and screening.

## 3. Execution Flow

The application is executed from `main.py`, which takes a UniProt ID as input. The `LigandMinerApplication.main()` method then executes the following high-level workflow:

1.  **Target Protein Analysis**:
    *   Fetches metadata for the target protein from UniProt.
    *   Identifies homologous proteins using the target's InterPro domains.
    *   Selects the "best" representative 3D structure for the target protein. This is a complex process involving downloading, cleaning, and scoring multiple candidate structures from the PDB, AlphaFold DB, and Swiss-Model.
    *   Extracts the molecular surface of the selected structure using ProBiS.

2.  **Homolog Processing**:
    *   Iterates through the list of homologous proteins found in the previous step.
    *   For each homolog, it performs the same structure selection and surface extraction process as for the target protein.
    *   It then compares the homolog's surface to the target's surface using ProBiS and calculates a similarity score.

3.  **Ranking and Screening**:
    *   Ranks the homologous proteins based on their ProBiS surface similarity score.
    *   For each of the top-ranked homologs, it fetches their known ligands from a local database.
    *   It then uses these ligands to build 2D and 3D pharmacophore models.
    *   Finally, it screens a library of FDA-approved ligands against these pharmacophore models to identify potential candidates.

4.  **Output**:
    *   The results of the virtual screening are saved to a JSON file (`ranked_protein_virtual_screening.json`).

## 4. Major Components

### 4.1. Services

*   **`UniProtService`**: Handles all interactions with the UniProt and InterPro APIs. It's responsible for fetching protein metadata and finding homologous proteins.
*   **`StructureService`**: A large and critical component that manages the entire lifecycle of protein structures. This includes downloading, cleaning (with `gemmi`), running pocket detection (`fpocket`, `prank`), scoring, and selecting the best structure.
*   **`SurfaceService`**: Manages all ProBiS-related tasks. It uses the `probis` executable to extract surfaces, compare them, and parse the results.
*   **`LigandService`**: Responsible for querying the local SQLite database (`ligand.db`) to retrieve ligand information.
*   **`ScreeningService`**: Orchestrates the 2D and 3D pharmacophore screening process, using the `pharma` package to perform the underlying calculations.

### 4.2. Domain Models

*   **`PipelineState`**: The central data object that is passed through the pipeline and progressively populated by the services.
*   **`ProteinData`**: A container for all data related to a single protein, including its annotations, structures, ligands, and scores.
*   **`Structure`**: A hierarchical set of data classes representing protein structures from different sources (PDB, AlphaFold, Swiss-Model).
*   **`Ligand`**: A data class for ligands, which is decorated with additional information (e.g., conformers, pharmacophores) as it moves through the screening process.

## 5. Data Flow

The data flows through the application in a well-defined manner:

*   The `LigandMinerApplication` initializes a `PipelineState` object.
*   Each service is called in sequence, taking the `PipelineState` object as input, modifying it, and returning it.
*   The `PipelineState` object grows in size and complexity as it moves through the pipeline. For example, the `UniProtService` adds protein annotations, the `StructureService` adds structural information, and so on.
*   The final `PipelineState` object contains all the information generated during the run, which is then used to generate the final output.

## 6. Assumptions and Limitations

*   **ProBiS Dependency**: The application has a hard dependency on the `probis` executable being in the root directory.
*   **Local Database**: The application requires a pre-built SQLite database (`ligand.db`) containing ligand information.
*   **Input**: The application currently only accepts a UniProt ID as input.
*   **Pocket Selection**: The current method for selecting binding pockets is relatively simple and is a key area for improvement, as detailed in the `TODO.md` file.

## 7. Current Status and Roadmap

### 7.1. Completed

*   The core pipeline for identifying and ranking homologous proteins based on surface similarity is in place.
*   The pharmacophore screening module is functional.
*   The basic service-oriented architecture is well-established.

### 7.2. Unfinished / TODO

The `TODO.md` file provides a detailed roadmap. Here's a summary of the most important items:

#### Critical

*   **Pocket-Centric Structure Selection**: The current structure selection is based on domain coverage. The plan is to move to a more sophisticated method that prioritizes the completeness of the binding pocket.
*   **Improved Pocket Detection**: The current method of selecting the top-scoring `fpocket` pocket is a heuristic. A more robust method that considers evidence from multiple sources (holo-ligand contacts, annotations, consensus) is needed.

#### Planned Features

*   **Flexible Structure Acquisition**: Support for additional input types, such as PDB IDs and uploaded files.
*   **Standardized Result Types**: Replace ad-hoc dictionaries with well-defined data classes for all pipeline outputs to improve type safety and maintainability.

#### Nice Improvements

*   **Refactoring `StructureService`**: This class is currently very large and could be broken down into smaller, more focused components.
*   **Configuration Management**: The application currently has some hardcoded paths and parameters that could be externalized into a configuration file.

#### Future Work

*   **ProBiS-style Binding Site Analysis**: Use ProBiS to identify structurally similar binding sites and their known ligands, and use these "transferred" ligands as candidates for virtual screening.
*   **Ligand Filtering**: Implement a flexible ligand filtering pipeline to remove undesirable compounds before virtual screening.
*   **Molecular Dynamics**: Add support for running molecular dynamics simulations to validate docking poses and estimate binding free energies.
*   **Post-analysis**: Provide a comprehensive analysis of the simulation results to help the user interpret the data.

## 8. Where to Start

*   **To understand the overall flow**: Start with `main.py` and `src/app/application.py`.
*   **To modify the structure selection logic**: Look at `src/app/structure_services.py` and the scoring functions in `src/app/protein_services.py`.
*   **To change the surface comparison**: See `src/app/surface_services.py`.
*   **To work on pharmacophore screening**: Dive into `src/app/ligand_services.py` and the `src/pharma` package.
*   **For the data models**: The `src/domain` package is the place to go.
