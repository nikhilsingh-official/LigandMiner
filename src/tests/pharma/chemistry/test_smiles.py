import unittest

from src.pharma.chemistry.smiles import smiles_to_mol


class TestSmilesParsing(unittest.TestCase):
    def test_valid_smiles_returns_rdkit_molecule_with_expected_atoms(self):
        molecule = smiles_to_mol("CCO")
        self.assertEqual(molecule.GetNumAtoms(), 3)

    def test_empty_smiles_is_a_valid_empty_molecule(self):
        molecule = smiles_to_mol("")
        self.assertEqual(molecule.GetNumAtoms(), 0)

    def test_invalid_smiles_raises_explicit_value_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            smiles_to_mol("not a smiles")


if __name__ == "__main__":
    unittest.main()
