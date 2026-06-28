import unittest
from unittest.mock import Mock

from rdkit import Chem

from src.domain.ligand import Ligand, LigandMol, LigandRecord
from src.pharma.preparation.conformer_generation import _CONFORMER_CACHE
from src.pharma.preparation.ligand_preparation import (
    add_hydrogens_to_mol,
    attach_conformers,
    attach_mol,
    attach_mols,
    prepare_ligand_geometry,
    prepare_ligands_geometry,
)


def ligand(smiles, ligand_id):
    return Ligand(LigandRecord("P", f"K-{ligand_id}", ligand_id, smiles, "IC50", 1.0))


class TestLigandPreparation(unittest.TestCase):
    def setUp(self):
        _CONFORMER_CACHE.clear()

    def test_attach_molecule_preserves_record_and_parses_smiles(self):
        source = ligand("CCO", "one")
        result = attach_mol(source)
        self.assertIs(result.record, source.record)
        self.assertEqual(result.mol.GetNumAtoms(), 3)

    def test_attach_molecules_preserves_input_order_and_duplicates(self):
        result = attach_mols([ligand("CCO", "a"), ligand("CCO", "b")])
        self.assertEqual([item.record.ligand_id for item in result], ["a", "b"])

    def test_add_hydrogens_requires_molecule(self):
        with self.assertRaisesRegex(ValueError, "must have mol"):
            add_hydrogens_to_mol(LigandMol(ligand("CCO", "x").record, None))

    def test_add_hydrogens_returns_new_ligand_molecule(self):
        source = attach_mol(ligand("CCO", "x"))
        result = add_hydrogens_to_mol(source)
        self.assertGreater(result.mol.GetNumAtoms(), source.mol.GetNumAtoms())
        self.assertIs(result.record, source.record)

    def test_attach_conformers_returns_none_when_embedding_produces_none(self):
        source = add_hydrogens_to_mol(attach_mol(ligand("CCO", "x")))
        self.assertIsNone(attach_conformers(source, num_conformers=0))

    def test_prepare_single_ligand_runs_complete_geometry_pipeline(self):
        result = prepare_ligand_geometry(ligand("CCO", "single"), num_conformers=1)
        self.assertEqual(result.record.ligand_id, "single")
        self.assertGreater(result.mol.GetNumConformers(), 0)
        self.assertEqual(len(result.conformers), result.mol.GetNumConformers())

    def test_batch_omits_ligands_without_conformers_and_advances_each_item(self):
        progress = Mock()
        result = prepare_ligands_geometry(
            [ligand("CCO", "a"), ligand("CCN", "b")],
            key="prepare",
            progress=progress,
            num_conformers=0,
        )
        self.assertEqual(result, [])
        self.assertEqual(progress.advance.call_count, 2)

    def test_invalid_smiles_aborts_with_explicit_value_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            prepare_ligands_geometry([ligand("bad smiles", "bad")], num_conformers=1)


if __name__ == "__main__":
    unittest.main()
