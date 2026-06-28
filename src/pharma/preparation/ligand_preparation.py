from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, List

from rdkit.Chem import AddHs

from src.pharma.chemistry.mol import ligand_to_conformers
from src.pharma.chemistry.smiles import smiles_to_mol
from src.domain.ligand import Ligand, LigandConformers, LigandMol

if TYPE_CHECKING:
    from src.app.progress import ProgressSink


logger = logging.getLogger(__name__)


def add_hydrogens_to_mol(ligand: LigandMol) -> LigandMol:
    started = perf_counter()
    if ligand.mol is None:
        logger.error("Cannot add hydrogens ligand_id=%s stage=hydrogen_addition reason=missing_molecule", ligand.record.ligand_id)
        raise ValueError("Ligand must have mol before adding hydrogens")

    mol_with_h = AddHs(ligand.mol)
    logger.debug(
        "Added hydrogens ligand_id=%s input_atoms=%d output_atoms=%d elapsed_ms=%.3f",
        ligand.record.ligand_id,
        ligand.mol.GetNumAtoms(),
        mol_with_h.GetNumAtoms(),
        (perf_counter() - started) * 1000,
    )
    return LigandMol(record=ligand.record, mol=mol_with_h)

def add_hydrogens_to_mols(ligands: List[LigandMol]) -> List[LigandMol]:
    started = perf_counter()
    results = [add_hydrogens_to_mol(l) for l in ligands]
    logger.debug("Added hydrogens to ligand batch ligands=%d elapsed_ms=%.3f", len(ligands), (perf_counter() - started) * 1000)
    return results

def attach_mol(ligand: Ligand) -> LigandMol:
    started = perf_counter()
    try:
        mol = smiles_to_mol(ligand.record.smiles)
    except Exception as exc:
        logger.error(
            "Ligand molecule parsing failed ligand_id=%s smiles=%s stage=smiles_parsing exception_type=%s",
            ligand.record.ligand_id,
            ligand.record.smiles,
            type(exc).__name__,
        )
        raise
    ligand_mol = LigandMol(ligand.record, mol)
    logger.debug(
        "Attached molecule ligand_id=%s smiles=%s atoms=%d elapsed_ms=%.3f",
        ligand.record.ligand_id,
        ligand.record.smiles,
        mol.GetNumAtoms(),
        (perf_counter() - started) * 1000,
    )
    return ligand_mol

def attach_mols(ligands: list[Ligand]) -> list[LigandMol]:
    started = perf_counter()
    results = [attach_mol(l) for l in ligands]
    logger.debug("Attached molecules to ligand batch ligands=%d elapsed_ms=%.3f", len(ligands), (perf_counter() - started) * 1000)
    return results

def attach_conformers(
    ligand: LigandMol,
    num_conformers: int = 8,
    energy_window: float = 5.0,
) -> LigandConformers | None:
    started = perf_counter()
    if ligand.mol is None:
        logger.error("Cannot generate conformers ligand_id=%s stage=conformer_generation reason=missing_molecule", ligand.record.ligand_id)
        raise ValueError("Ligand must have mol before conformer generation")

    conformers = ligand_to_conformers(ligand, num_conformers, energy_window)
    if not conformers:
        logger.warning(
            "Ligand has no retained conformers ligand_id=%s smiles=%s stage=conformer_generation",
            ligand.record.ligand_id,
            ligand.record.smiles,
        )
        return None
    ligand_conformers = LigandConformers(ligand.record, ligand.mol, conformers)
    logger.debug(
        "Attached conformers ligand_id=%s conformers=%d requested_conformers=%d energy_window=%.6f elapsed_ms=%.3f",
        ligand.record.ligand_id,
        len(conformers),
        num_conformers,
        energy_window,
        (perf_counter() - started) * 1000,
    )
    return ligand_conformers

def prepare_ligand_geometry(
    ligand: Ligand,
    num_conformers: int = 8,
    energy_window: float = 5.0,
) -> LigandConformers | None:
    started = perf_counter()
    ligand_mol = attach_mol(ligand)
    ligand_mol = add_hydrogens_to_mol(ligand_mol)

    ligand_confs = attach_conformers(ligand_mol, num_conformers, energy_window)

    logger.debug(
        "Prepared ligand geometry ligand_id=%s successful=%s elapsed_ms=%.3f",
        ligand.record.ligand_id,
        ligand_confs is not None,
        (perf_counter() - started) * 1000,
    )
    return ligand_confs

def prepare_ligands_geometry(
    ligands: list[Ligand],
    key: str | None = None,
    progress: ProgressSink | None = None,
    num_conformers: int = 8,
    energy_window: float = 5.0,
) -> List[LigandConformers]:
    started = perf_counter()
    logger.debug(
        "Preparing ligand geometry batch ligands=%d requested_conformers=%d energy_window=%.6f",
        len(ligands),
        num_conformers,
        energy_window,
    )
    results = []

    for ligand in ligands:
        try:
            ligand_mol = attach_mol(ligand)
            ligand_mol = add_hydrogens_to_mol(ligand_mol)
            ligand_confs = attach_conformers(ligand_mol, num_conformers, energy_window)
        except Exception as exc:
            logger.exception(
                "Ligand geometry preparation failed ligand_id=%s smiles=%s stage=geometry_preparation exception_type=%s",
                ligand.record.ligand_id,
                ligand.record.smiles,
                type(exc).__name__,
            )
            raise

        if ligand_confs is not None:
            results.append(ligand_confs)

        if progress and key:
            progress.advance(key)

    logger.debug(
        "Prepared ligand geometry batch received=%d successful=%d elapsed_ms=%.3f",
        len(ligands),
        len(results),
        (perf_counter() - started) * 1000,
    )
    return results
