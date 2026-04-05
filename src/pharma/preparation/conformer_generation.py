from typing import List
from rdkit.Chem import Mol, Conformer as RDConformer
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFOptimizeMoleculeConfs,
    MMFFHasAllMoleculeParams,
    UFFOptimizeMoleculeConfs,
)

from src.domain.ligand import Conformer


def prefer_converged_conformers(conformers: List[Conformer]) -> List[Conformer]:
    converged = [c for c in conformers if c.convergence_success]
    return converged if converged else conformers


def prefer_lower_e_conformers(
    conformers: List[Conformer], energy_window: float = 5.0
) -> List[Conformer]:
    if not conformers:
        return []

    best = min(conformers, key=lambda c: c.energy)
    best_energy = best.energy

    return [c for c in conformers if c.energy <= best_energy + energy_window]


def prune_mol_conformers(mol: Mol, keep_conf_ids: list[int]) -> None:
    keep = set(keep_conf_ids)

    existing_ids = [conf.GetId() for conf in mol.GetConformers()]

    for cid in existing_ids:
        if cid not in keep:
            mol.RemoveConformer(cid)


def filter_conformers(
    conformers: List[Conformer], energy_window: float = 5.0
) -> List[Conformer] | None:

    if not conformers:
        print("No conformers for molecule.")
        return None

    converged = prefer_converged_conformers(conformers)
    preferred = prefer_lower_e_conformers(converged, energy_window)

    return preferred


_CONFORMER_CACHE: dict[str, tuple[list[Conformer], Mol]] = {}


def conformer_generation(mol: Mol, num_confs: int = 10) -> List[Conformer]:

    if mol.GetNumConformers():
        mol.RemoveAllConformers()

    params = ETKDGv3()
    params.pruneRmsThresh = 0.5
    params.numThreads = 0
    params.useRandomCoords = True

    conf_ids = list(EmbedMultipleConfs(mol, numConfs=num_confs, params=params))

    if not conf_ids:
        return []

    if MMFFHasAllMoleculeParams(mol):
        results = MMFFOptimizeMoleculeConfs(mol)
    else:
        results = UFFOptimizeMoleculeConfs(mol)

    if len(conf_ids) != len(results):
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

    return conformers


def generate_filtered_conformers(
    mol: Mol,
    ligand_id: str,
    num_confs: int = 10,
    energy_window: float = 5.0,
) -> list[Conformer] | None:

    if ligand_id in _CONFORMER_CACHE:
        cached_confs, cached_mol = _CONFORMER_CACHE[ligand_id]

        mol.RemoveAllConformers()

        for conf in cached_mol.GetConformers():
            new_conf = RDConformer(conf)
            mol.AddConformer(new_conf, assignId=False)

        return cached_confs

    conformers = conformer_generation(mol, num_confs)
    filtered = filter_conformers(conformers, energy_window)

    if not filtered:
        return None

    keep_ids = [c.conf_id for c in filtered]
    prune_mol_conformers(mol, keep_ids)

    _CONFORMER_CACHE[ligand_id] = (filtered, Mol(mol))

    return filtered