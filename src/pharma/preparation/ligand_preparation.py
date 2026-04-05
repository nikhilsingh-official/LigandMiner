from typing import List

from rdkit.Chem import AddHs

from src.pharma.chemistry.mol import ligand_to_conformers
from src.pharma.chemistry.smiles import smiles_to_mol
from src.domain.ligand import Ligand, LigandConformers, LigandMol


def add_hydrogens_to_mol(ligand: LigandMol) -> LigandMol:
    if ligand.mol is None:
        raise ValueError("Ligand must have mol before adding hydrogens")

    mol_with_h = AddHs(ligand.mol)
    return LigandMol(record=ligand.record, mol=mol_with_h)

def add_hydrogens_to_mols(ligands: List[LigandMol]) -> List[LigandMol]:
    return [add_hydrogens_to_mol(l) for l in ligands]

def attach_mol(ligand: Ligand) -> LigandMol:
    mol = smiles_to_mol(ligand.record.smiles)
    ligand_mol = LigandMol(ligand.record, mol)
    return ligand_mol

def attach_mols(ligands: list[Ligand]) -> list[LigandMol]:
    return [attach_mol(l) for l in ligands]

def attach_conformers(ligand: LigandMol) -> LigandConformers | None:
    if ligand.mol is None:
        raise ValueError("Ligand must have mol before conformer generation")

    conformers = ligand_to_conformers(ligand)
    if not conformers:
        return None
    ligand_conformers = LigandConformers(ligand.record, ligand.mol, conformers)
    return ligand_conformers

def prepare_ligand_geometry(ligand: Ligand) -> LigandConformers | None:
    ligand_mol = attach_mol(ligand)
    ligand_mol = add_hydrogens_to_mol(ligand_mol)

    ligand_confs = attach_conformers(ligand_mol)

    return ligand_confs

def prepare_ligands_geometry(
    ligands: list[Ligand]
) -> List[LigandConformers]:

    results = []

    for ligand in ligands:
        ligand_mol = attach_mol(ligand)
        ligand_mol = add_hydrogens_to_mol(ligand_mol)

        ligand_confs = attach_conformers(ligand_mol)

        if ligand_confs is not None:
            results.append(ligand_confs)

    return results