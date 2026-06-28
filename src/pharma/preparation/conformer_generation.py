import logging
from time import perf_counter
from typing import List
from rdkit.Chem import Mol, Conformer as RDConformer
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFOptimizeMoleculeConfs,
    MMFFHasAllMoleculeParams,
    UFFOptimizeMoleculeConfs,
)

from src.domain.ligand import Conformer


logger = logging.getLogger(__name__)


def prefer_converged_conformers(conformers: List[Conformer]) -> List[Conformer]:
    converged = [c for c in conformers if c.convergence_success]
    logger.debug(
        "Selected converged conformers input_conformers=%d converged_conformers=%d fallback_to_all=%s",
        len(conformers),
        len(converged),
        not bool(converged),
    )
    return converged if converged else conformers


def prefer_lower_e_conformers(
    conformers: List[Conformer], energy_window: float = 5.0
) -> List[Conformer]:
    if not conformers:
        return []

    best = min(conformers, key=lambda c: c.energy)
    best_energy = best.energy
    preferred = [c for c in conformers if c.energy <= best_energy + energy_window]
    logger.debug(
        "Filtered conformers by energy input_conformers=%d retained_conformers=%d best_energy=%.6f energy_window=%.6f",
        len(conformers),
        len(preferred),
        best_energy,
        energy_window,
    )
    return preferred


def prune_mol_conformers(mol: Mol, keep_conf_ids: list[int]) -> None:
    keep = set(keep_conf_ids)

    existing_ids = [conf.GetId() for conf in mol.GetConformers()]

    for cid in existing_ids:
        if cid not in keep:
            mol.RemoveConformer(cid)
    logger.debug(
        "Pruned molecule conformers input_conformers=%d retained_conformers=%d",
        len(existing_ids),
        mol.GetNumConformers(),
    )


def filter_conformers(
    conformers: List[Conformer], energy_window: float = 5.0
) -> List[Conformer] | None:

    if not conformers:
        logger.warning("Conformer filtering received no conformers")
        return None

    converged = prefer_converged_conformers(conformers)
    preferred = prefer_lower_e_conformers(converged, energy_window)

    return preferred


_CONFORMER_CACHE: dict[str, tuple[list[Conformer], Mol]] = {}


def conformer_generation(mol: Mol, num_confs: int = 10) -> List[Conformer]:
    started = perf_counter()
    logger.debug("Starting conformer generation requested_conformers=%d atoms=%d", num_confs, mol.GetNumAtoms())

    if mol.GetNumConformers():
        logger.debug("Removing existing conformers count=%d before generation", mol.GetNumConformers())
        mol.RemoveAllConformers()

    params = ETKDGv3()
    params.pruneRmsThresh = 0.5
    params.numThreads = 0
    params.useRandomCoords = True

    conf_ids = list(EmbedMultipleConfs(mol, numConfs=num_confs, params=params))

    if not conf_ids:
        logger.warning(
            "Conformer embedding generated no conformers requested_conformers=%d atoms=%d elapsed_ms=%.3f",
            num_confs,
            mol.GetNumAtoms(),
            (perf_counter() - started) * 1000,
        )
        return []

    if MMFFHasAllMoleculeParams(mol):
        force_field = "MMFF"
        results = MMFFOptimizeMoleculeConfs(mol)
    else:
        force_field = "UFF"
        results = UFFOptimizeMoleculeConfs(mol)

    if len(conf_ids) != len(results):
        logger.error(
            "Conformer optimization result mismatch embedded_conformers=%d optimized_conformers=%d force_field=%s",
            len(conf_ids),
            len(results),
            force_field,
        )
        raise RuntimeError("Embedding/MMFF mismatch")

    conformers: List[Conformer] = []

    for conf_id, res in zip(conf_ids, results):
        conformers.append(
            Conformer(
                conf_id=conf_id,
                convergence_success=(res[0] == 0),
                energy=res[1],
            )
        )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Generated and optimized conformers requested_conformers=%d embedded_conformers=%d converged_conformers=%d force_field=%s elapsed_ms=%.3f",
            num_confs,
            len(conformers),
            sum(conformer.convergence_success for conformer in conformers),
            force_field,
            (perf_counter() - started) * 1000,
        )
    return conformers


def generate_filtered_conformers(
    mol: Mol,
    ligand_id: str,
    num_confs: int = 10,
    energy_window: float = 5.0,
) -> list[Conformer] | None:
    started = perf_counter()
    if ligand_id in _CONFORMER_CACHE:
        cached_confs, cached_mol = _CONFORMER_CACHE[ligand_id]

        mol.RemoveAllConformers()

        for conf in cached_mol.GetConformers():
            new_conf = RDConformer(conf)
            mol.AddConformer(new_conf, assignId=False)

        logger.debug(
            "Conformer cache hit ligand_id=%s conformers=%d elapsed_ms=%.3f",
            ligand_id,
            len(cached_confs),
            (perf_counter() - started) * 1000,
        )
        return cached_confs

    logger.debug("Conformer cache miss ligand_id=%s", ligand_id)

    try:
        conformers = conformer_generation(mol, num_confs)
    except Exception as exc:
        logger.exception(
            "Conformer generation failed ligand_id=%s stage=conformer_generation requested_conformers=%d exception_type=%s",
            ligand_id,
            num_confs,
            type(exc).__name__,
        )
        raise
    filtered = filter_conformers(conformers, energy_window)

    if not filtered:
        logger.warning(
            "No conformers retained ligand_id=%s generated_conformers=%d energy_window=%.6f",
            ligand_id,
            len(conformers),
            energy_window,
        )
        return None

    keep_ids = [c.conf_id for c in filtered]
    prune_mol_conformers(mol, keep_ids)

    _CONFORMER_CACHE[ligand_id] = (filtered, Mol(mol))

    logger.debug(
        "Stored conformer cache entry ligand_id=%s retained_conformers=%d elapsed_ms=%.3f",
        ligand_id,
        len(filtered),
        (perf_counter() - started) * 1000,
    )
    return filtered
