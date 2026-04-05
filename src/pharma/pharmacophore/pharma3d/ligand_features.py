from typing import List

from rdkit.Chem import Mol
from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore

from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory

from src.domain.ligand import LigandMol, LigandPharmacophore, LigandConformers

FEATURE_PRIORITY = [
    "Donor",
    "Acceptor",
    "PosIonizable",
    "NegIonizable",
    "Aromatic",
    "Hydrophobe",
    "LumpedHydrophobe",
]


def ligand_to_pcophore_ligand(ligand: LigandConformers, feat_factory: MolChemicalFeatureFactory) -> LigandPharmacophore:
    pharmacophore = mol_to_pharmacophore(ligand.mol, feat_factory)
    return LigandPharmacophore(
        record=ligand.record,
        mol=ligand.mol,
        conformers=ligand.conformers,
        pharmacophore=pharmacophore
    )


def ligands_to_pharmacophore(ligands: List[LigandConformers], feat_factory: MolChemicalFeatureFactory) -> List[LigandPharmacophore]:
    pcophore_ligands = []
    for ligand in ligands:
        pcophore_ligands.append(ligand_to_pcophore_ligand(ligand, feat_factory))
    return pcophore_ligands


def mol_to_pharmacophore(mol: Mol, feat_factory: MolChemicalFeatureFactory) -> Pharmacophore:
    """
    :param mol: Molecule to extract pharmacophore features from.
    :param feat_factory: Used to generate features of mol.
    :return: Pharmacophore with features from the molecule, deduplicated by atom set and prioritized by FEATURE_PRIORITY.

    The deduplication step is necessary to pass the internal atomMatch flag in GetAllPharmacophoreMatches, which assumes a one-to-one mapping between features and atoms. In cases where multiple features are generated for the same set of atoms (e.g. a feature that is both aromatic and a hydrophobe), we retain only the feature with the highest priority according to FEATURE_PRIORITY.
    The feature priority is designed with the assumption that more specific features (e.g. Donor) are more likely to be relevant for binding than less specific features (e.g. Aromatic), and thus should be retained in cases of conflict.
    """
    feats = feat_factory.GetFeaturesForMol(mol)

    by_atomset = {}

    for f in feats:
        atom_ids = tuple(sorted(f.GetAtomIds()))
        fam = f.GetFamily()

        if atom_ids not in by_atomset:
            by_atomset[atom_ids] = f
            continue

        existing = by_atomset[atom_ids]

        if FEATURE_PRIORITY.index(fam) < FEATURE_PRIORITY.index(existing.GetFamily()):
            by_atomset[atom_ids] = f

    deduped = list(by_atomset.values())
    return Pharmacophore(deduped)
