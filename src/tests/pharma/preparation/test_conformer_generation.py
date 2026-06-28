import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from src.domain.ligand import Conformer
from src.pharma.preparation.conformer_generation import (
    _CONFORMER_CACHE,
    conformer_generation,
    filter_conformers,
    generate_filtered_conformers,
    prefer_converged_conformers,
    prefer_lower_e_conformers,
    prune_mol_conformers,
)


class TestConformerFiltering(unittest.TestCase):
    def test_prefers_converged_entries_but_falls_back_when_none_converged(self):
        values = [Conformer(0, False, 1.0), Conformer(1, True, 2.0)]
        self.assertEqual(prefer_converged_conformers(values), [values[1]])
        self.assertEqual(prefer_converged_conformers([values[0]]), [values[0]])

    def test_energy_window_is_inclusive_relative_to_minimum(self):
        values = [Conformer(0, True, 1.0), Conformer(1, True, 6.0), Conformer(2, True, 6.1)]
        self.assertEqual(prefer_lower_e_conformers(values, 5.0), values[:2])

    def test_filter_empty_input_returns_none(self):
        self.assertIsNone(filter_conformers([]))

    def test_filter_prefers_convergence_before_energy(self):
        unconverged_low = Conformer(0, False, 0.0)
        converged_high = Conformer(1, True, 10.0)
        self.assertEqual(filter_conformers([unconverged_low, converged_high]), [converged_high])


class TestConformerGeneration(unittest.TestCase):
    def setUp(self):
        _CONFORMER_CACHE.clear()

    @staticmethod
    def molecule():
        return Chem.AddHs(Chem.MolFromSmiles("CCO"))

    def test_generation_replaces_existing_conformers_and_returns_matching_metadata(self):
        molecule = self.molecule()
        AllChem.EmbedMolecule(molecule, randomSeed=7)

        conformers = conformer_generation(molecule, num_confs=2)

        self.assertEqual(molecule.GetNumConformers(), len(conformers))
        self.assertGreaterEqual(len(conformers), 1)
        self.assertTrue(all(isinstance(item.energy, float) for item in conformers))

    def test_zero_requested_conformers_returns_empty(self):
        molecule = self.molecule()
        self.assertEqual(conformer_generation(molecule, num_confs=0), [])

    def test_prune_retains_only_requested_conformer_ids(self):
        molecule = self.molecule()
        ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=3, randomSeed=7))

        prune_mol_conformers(molecule, [ids[1]])

        self.assertEqual([conf.GetId() for conf in molecule.GetConformers()], [ids[1]])

    def test_cached_generation_restores_conformers_to_new_molecule(self):
        first = self.molecule()
        first_result = generate_filtered_conformers(first, "same", num_confs=1, energy_window=5)
        second = self.molecule()

        second_result = generate_filtered_conformers(second, "same", num_confs=5, energy_window=0)

        self.assertEqual(second_result, first_result)
        self.assertEqual(second.GetNumConformers(), first.GetNumConformers())


if __name__ == "__main__":
    unittest.main()
