import logging
from time import perf_counter
from typing import List

from rdkit.Chem import Mol
from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore

from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory

from src.domain.ligand import LigandMol, LigandPharmacophore, LigandConformers


logger = logging.getLogger(__name__)

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
    started = perf_counter()
    pharmacophore = mol_to_pharmacophore(ligand.mol, feat_factory)
    logger.debug(
        "Extracted Pharm3D ligand pharmacophore ligand_id=%s features=%d conformers=%d elapsed_ms=%.3f",
        ligand.record.ligand_id,
        len(pharmacophore.getFeatures()),
        len(ligand.conformers),
        (perf_counter() - started) * 1000,
    )
    return LigandPharmacophore(
        record=ligand.record,
        mol=ligand.mol,
        conformers=ligand.conformers,
        pharmacophore=pharmacophore
    )


def ligands_to_pharmacophore(ligands: List[LigandConformers], feat_factory: MolChemicalFeatureFactory) -> List[LigandPharmacophore]:
    started = perf_counter()
    pcophore_ligands = []
    for ligand in ligands:
        pcophore_ligands.append(ligand_to_pcophore_ligand(ligand, feat_factory))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Extracted Pharm3D pharmacophores ligands=%d total_features=%d elapsed_ms=%.3f",
            len(pcophore_ligands),
            sum(len(ligand.pharmacophore.getFeatures()) for ligand in pcophore_ligands),
            (perf_counter() - started) * 1000,
        )
    return pcophore_ligands


def mol_to_pharmacophore(mol: Mol, feat_factory: MolChemicalFeatureFactory) -> Pharmacophore:
    """
    :param mol: Molecule to extract pharmacophore features from.
    :param feat_factory: Used to generate features of mol.
    :return: Pharmacophore with features from the molecule, deduplicated by atom set and prioritized by FEATURE_PRIORITY.

    The deduplication step is necessary to pass the internal atomMatch flag in GetAllPharmacophoreMatches, which assumes a one-to-one mapping between features and atoms. In cases where multiple features are generated for the same set of atoms (e.g. a feature that is both aromatic and a hydrophobe), we retain only the feature with the highest priority according to FEATURE_PRIORITY.
    The feature priority is designed with the assumption that more specific features (e.g. Donor) are more likely to be relevant for binding than less specific features (e.g. Aromatic), and thus should be retained in cases of conflict.
    """
    started = perf_counter()
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
    if not deduped:
        logger.warning("Molecule produced no supported Pharm3D features")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Generated Pharm3D molecular features raw_features=%d deduplicated_features=%d elapsed_ms=%.3f",
            len(feats),
            len(deduped),
            (perf_counter() - started) * 1000,
        )
    return Pharmacophore(deduped)
