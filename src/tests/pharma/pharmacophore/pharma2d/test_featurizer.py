import unittest
from dataclasses import asdict

from rdkit import Chem
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.pharma.utils.hashing import make_single_fp_id
from src.pharma.pharmacophore.pharma2d.factories import build_pharm2d_factories
from src.domain.ligand import Ligand, LigandRecord, LigandMol
from src.domain.pharm import FingerprintDict, Fingerprint, Pharm2DConfig
from src.pharma.pharmacophore.pharma2d.featurizer import _featurize_ligand


class TestPharm2DFeaturizer(unittest.TestCase):
    
    def setUp(self):
        """Set up using the same initialization as production code"""
        # Use actual config initialization like Pharm2DScreening does
        self.config = Pharm2DConfig()

        # Create factories the same way as production code
        self.sig_factory, self.morgan_factory = build_pharm2d_factories(self.config)
    
    def _create_ligand_mol(self, smiles: str, ligand_id: str) -> LigandMol:
        """Helper to create LigandMol objects for testing"""
        record = LigandRecord(
            protein_id="test_protein",
            ligand_id=ligand_id,
            affinity_type="IC50",
            affinity=100.0,
            smiles=smiles,
            inchi_key=f"test_inchi_key_{ligand_id}"
        )
        ligand = Ligand(record=record)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return LigandMol(record=ligand.record, mol=mol)
    
    def test_different_molecules_different_fingerprints(self):
        """Test that different molecules produce different fingerprints"""
        ligand1 = self._create_ligand_mol("CCO", "ethanol")
        ligand2 = self._create_ligand_mol("CCCO", "propanol")
        
        fp1 = _featurize_ligand(ligand1, self.sig_factory, self.morgan_factory)
        fp2 = _featurize_ligand(ligand2, self.sig_factory, self.morgan_factory)
        
        # Fingerprints should be different
        self.assertNotEqual(fp1.pharm2d.bits.ToBitString(), 
                          fp2.pharm2d.bits.ToBitString())
        self.assertNotEqual(fp1.morgan.bits.ToBitString(), 
                          fp2.morgan.bits.ToBitString())
        
        # But source IDs should be correct
        self.assertEqual(fp1.pharm2d.source_id, "ethanol")
        self.assertEqual(fp2.pharm2d.source_id, "propanol")
    
    def test_identical_molecules_identical_fingerprints(self):
        """Test that identical molecules produce identical fingerprints"""
        ligand1 = self._create_ligand_mol("CCO", "test_A")
        ligand2 = self._create_ligand_mol("CCO", "test_B")
        
        fp1 = _featurize_ligand(ligand1, self.sig_factory, self.morgan_factory)
        fp2 = _featurize_ligand(ligand2, self.sig_factory, self.morgan_factory)
        
        # Fingerprint bits should be identical
        self.assertEqual(fp1.pharm2d.bits.ToBitString(), 
                        fp2.pharm2d.bits.ToBitString())
        self.assertEqual(fp1.morgan.bits.ToBitString(), 
                        fp2.morgan.bits.ToBitString())
        
        # But IDs should reflect different source IDs
        self.assertEqual(fp1.pharm2d.source_id, "test_A")
        self.assertEqual(fp2.pharm2d.source_id, "test_B")
        self.assertNotEqual(fp1.pharm2d.fp_id, fp2.pharm2d.fp_id)
    
    def test_aromatic_molecules(self):
        """Test featurization of aromatic molecules"""
        ligand_mol = self._create_ligand_mol("c1ccccc1", "benzene")
        
        fp_dict = _featurize_ligand(ligand_mol, self.sig_factory, self.morgan_factory)
        
        # Should handle aromatic molecules correctly
        self.assertIsInstance(fp_dict, FingerprintDict)
        self.assertEqual(fp_dict.pharm2d.source_id, "benzene")
        self.assertIsNotNone(fp_dict.pharm2d.bits)
        self.assertIsNotNone(fp_dict.morgan.bits)
    
    def test_complex_pharmacophore_features(self):
        """Test featurization of molecules with diverse pharmacophore features"""
        # Molecule with multiple features: aromatic ring, H-bond donor/acceptor, etc.
        ligand_mol = self._create_ligand_mol("CC(=O)Oc1ccccc1C(=O)O", "aspirin")
        
        fp_dict = _featurize_ligand(ligand_mol, self.sig_factory, self.morgan_factory)
        
        # Should successfully featurize complex molecules
        self.assertIsInstance(fp_dict, FingerprintDict)
        self.assertEqual(fp_dict.pharm2d.source_id, "aspirin")
        
        # Complex molecules should have non-trivial fingerprints
        self.assertGreater(fp_dict.pharm2d.bits.GetNumOnBits(), 0)
        self.assertGreater(fp_dict.morgan.bits.GetNumOnBits(), 0)
    
    def test_fingerprint_consistency_across_calls(self):
        """Test that repeated featurization gives consistent results"""
        ligand_mol = self._create_ligand_mol("CCO", "consistency_test")
        
        fp1 = _featurize_ligand(ligand_mol, self.sig_factory, self.morgan_factory)
        fp2 = _featurize_ligand(ligand_mol, self.sig_factory, self.morgan_factory)
        
        # Should be completely identical
        self.assertEqual(fp1.pharm2d.bits.ToBitString(), 
                        fp2.pharm2d.bits.ToBitString())
        self.assertEqual(fp1.morgan.bits.ToBitString(),
                        fp2.morgan.bits.ToBitString())
        self.assertEqual(fp1.pharm2d.source_id, fp2.pharm2d.source_id)
        self.assertEqual(fp1.morgan.source_id, fp2.morgan.source_id)

    def test_make_single_fp_id_is_deterministic(self):
        self.assertEqual(
            make_single_fp_id("abc", "pharm2d"),
            make_single_fp_id("abc", "pharm2d"),
        )

    def test_make_single_fp_id_differs_for_inputs(self):
        self.assertNotEqual(
            make_single_fp_id("abc", "pharm2d"),
            make_single_fp_id("def", "pharm2d"),
        )

    def test_make_single_fp_id_prefix(self):
        self.assertTrue(
            make_single_fp_id("abc", "pharm2d").startswith("pharm2d_")
        )
    
    def test_invalid_molecule_handling(self):
        """Test handling of invalid molecules"""
        # Create ligand with None mol
        record = LigandRecord(
            protein_id="test",
            ligand_id="invalid",
            affinity_type="IC50",
            affinity=100.0,
            smiles="invalid_smiles",
            inchi_key="invalid_key"
        )
        ligand = Ligand(record=record)
        ligand_mol = LigandMol(record=ligand.record, mol=None)
        
        # Should raise an appropriate exception
        with self.assertRaises((ValueError, AttributeError, TypeError)):
            _featurize_ligand(ligand_mol, self.sig_factory, self.morgan_factory)
    
    def test_edge_case_molecules(self):
        """Test featurization of edge case molecules"""
        edge_cases = [
            ("C", "methane"),           # Simple alkane
            ("O", "water"),             # Very simple molecule
            ("[H][H]", "hydrogen"),     # Diatomic if valid
        ]
        
        for smiles, name in edge_cases:
            try:
                ligand_mol = self._create_ligand_mol(smiles, name)
                fp_dict = _featurize_ligand(ligand_mol, self.sig_factory, self.morgan_factory)
                
                # Should handle edge cases gracefully
                self.assertIsInstance(fp_dict, FingerprintDict)
                self.assertEqual(fp_dict.pharm2d.source_id, name)
                
            except ValueError:
                # Some edge cases might not be valid in RDKit
                # This documents expected behavior
                pass
    
    def test_large_molecule_featurization(self):
        """Test featurization of larger, drug-like molecules"""
        # Ibuprofen
        large_mol = self._create_ligand_mol("CC(C)Cc1ccc(cc1)C(C(=O)O)C", "ibuprofen")
        
        fp_dict = _featurize_ligand(large_mol, self.sig_factory, self.morgan_factory)
        
        # Should handle larger molecules
        self.assertIsInstance(fp_dict, FingerprintDict)
        self.assertEqual(fp_dict.pharm2d.source_id, "ibuprofen")
        
        # Larger molecules should have richer fingerprints
        self.assertGreater(fp_dict.pharm2d.bits.GetNumOnBits(), 5)
        self.assertGreater(fp_dict.morgan.bits.GetNumOnBits(), 10)


if __name__ == '__main__':
    unittest.main()
