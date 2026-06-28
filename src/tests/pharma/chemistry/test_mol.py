import unittest

from rdkit import Chem

from src.domain.ligand import LigandMol, LigandRecord
from src.pharma.chemistry.mol import ligand_to_conformers
from src.pharma.preparation.conformer_generation import _CONFORMER_CACHE


class TestLigandConformerConversion(unittest.TestCase):
    def setUp(self):
        _CONFORMER_CACHE.clear()

    def test_generates_requested_filtered_conformers_for_real_molecule(self):
        record = LigandRecord("P", "K", "ethanol", "CCO", "IC50", 1.0)
        ligand = LigandMol(record, Chem.AddHs(Chem.MolFromSmiles("CCO")))

        conformers = ligand_to_conformers(ligand, num_conformers=2, energy_window=100.0)

        self.assertGreaterEqual(len(conformers), 1)
        self.assertLessEqual(len(conformers), 2)
        self.assertEqual(ligand.mol.GetNumConformers(), len(conformers))


if __name__ == "__main__":
    unittest.main()
