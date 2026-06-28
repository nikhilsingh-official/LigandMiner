import unittest

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore
from rdkit.Geometry import Point3D

from src.pharma.pharmacophore.common.feature_factory import init_feature_factory
from src.pharma.pharmacophore.pharma3d.ligand_features import (
    ligand_to_pcophore_ligand,
    ligands_to_pharmacophore,
    mol_to_pharmacophore,
)
from src.pharma.pharmacophore.pharma3d.screening import PharmacophoreMatcher

from src.tests.pharma.pharmacophore.pharma3d._helpers import feature, prepared_ligand


class TestFeatureExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.factory = init_feature_factory()

    @staticmethod
    def molecule(smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if mol.GetNumAtoms():
            AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
        return mol

    def test_real_feature_generation_is_deterministic(self):
        mol = self.molecule("CCO")

        first = mol_to_pharmacophore(mol, self.factory).getFeatures()
        second = mol_to_pharmacophore(mol, self.factory).getFeatures()
        signature = lambda items: [
            (f.GetFamily(), f.GetType(), tuple(f.GetPos())) for f in items
        ]

        self.assertEqual(signature(first), signature(second))
        self.assertGreater(len(first), 0)

    def test_deduplicates_features_sharing_an_atom_set_by_priority(self):
        mol = self.molecule("c1ccccc1O")

        features = mol_to_pharmacophore(mol, self.factory).getFeatures()
        families = [f.GetFamily() for f in features]

        self.assertIn("Aromatic", families)
        self.assertNotIn("LumpedHydrophobe", families)

    def test_empty_molecule_has_no_features(self):
        pharmacophore = mol_to_pharmacophore(Chem.MolFromSmiles(""), self.factory)
        self.assertEqual(len(pharmacophore.getFeatures()), 0)

    def test_unsupported_atom_has_no_features(self):
        pharmacophore = mol_to_pharmacophore(Chem.MolFromSmiles("[He]"), self.factory)
        self.assertEqual(len(pharmacophore.getFeatures()), 0)

    def test_none_is_not_a_valid_molecule(self):
        with self.assertRaises(Exception):
            mol_to_pharmacophore(None, self.factory)

    def test_ligand_conversion_preserves_ligand_data(self):
        prepared = prepared_ligand("CCO", "ethanol")

        converted = ligand_to_pcophore_ligand(prepared, self.factory)

        self.assertIs(converted.record, prepared.record)
        self.assertIs(converted.mol, prepared.mol)
        self.assertIs(converted.conformers, prepared.conformers)
        self.assertGreater(len(converted.pharmacophore.getFeatures()), 0)

    def test_batch_conversion_preserves_order_and_duplicates(self):
        prepared = [prepared_ligand("CCO", "one"), prepared_ligand("CCO", "two")]

        converted = ligands_to_pharmacophore(prepared, self.factory)

        self.assertEqual([item.record.ligand_id for item in converted], ["one", "two"])


class TestPharmacophoreMatching(unittest.TestCase):
    def test_matcher_returns_candidates_for_each_consensus_feature(self):
        factory = init_feature_factory()
        prepared = prepared_ligand("CCO", "candidate")
        pharmacophore = Pharmacophore(
            [
                feature(0, family="Donor", feature_type="SingleAtomDonor"),
                feature(0, family="Acceptor", feature_type="SingleAtomAcceptor"),
            ]
        )

        matches = PharmacophoreMatcher(factory).get_feature_candidates(prepared, pharmacophore)

        self.assertEqual(len(matches), 2)
        self.assertTrue(all(len(candidates) == 1 for candidates in matches))
        self.assertEqual(matches[0][0].GetFamily(), "Donor")
        self.assertEqual(matches[1][0].GetFamily(), "Acceptor")

    def test_matcher_reports_no_candidates_for_absent_feature_family(self):
        factory = init_feature_factory()
        prepared = prepared_ligand("CCO", "candidate")
        pharmacophore = Pharmacophore(
            [feature(0, family="Aromatic", feature_type="Arom6")]
        )

        matches = PharmacophoreMatcher(factory).get_feature_candidates(prepared, pharmacophore)

        self.assertIsNone(matches)


if __name__ == "__main__":
    unittest.main()
