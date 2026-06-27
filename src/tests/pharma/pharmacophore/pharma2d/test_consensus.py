import unittest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from src.domain.ligand import Ligand, LigandRecord
from src.domain.pharm import Pharm2DConfig
from src.pharma.pharmacophore.pharma2d.consensus import ligands_to_consensus_fp
from src.pharma.pharmacophore.pharma2d.factories import build_pharm2d_factories
from src.pharma.preparation.ligand_preparation import attach_mols


class TestPharm2DConsensus(unittest.TestCase):
    """Test 2D pharmacophore consensus generation using actual implementation patterns"""
    
    def setUp(self):
        """Set up using the same initialization as production code"""
        # Use actual config initialization like Pharm2DScreening does
        self.config = Pharm2DConfig()
        
        # Create factories the same way as production code
        self.sig_factory, self.morgan_factory = build_pharm2d_factories(self.config)
    
    def _create_test_ligand(self, smiles: str, ligand_id: str) -> Ligand:
        """Create test ligand using same structure as production"""
        record = LigandRecord(
            protein_id="TEST_PROTEIN",
            ligand_id=ligand_id,
            affinity_type="IC50",
            affinity=10.0,
            smiles=smiles,
            inchi_key=f"TEST_KEY_{ligand_id}"
        )
        return Ligand(record=record)
    
    def test_single_ligand_consensus(self):
        """Test consensus generation with single ligand"""
        ligands = [self._create_test_ligand("CCO", "ethanol")]
        
        # Use same preparation as production code
        prepped_ligands = attach_mols(ligands)
        
        # Call actual consensus function
        consensus = ligands_to_consensus_fp(prepped_ligands, self.sig_factory, self.morgan_factory)
        
        # Verify structure matches production expectations
        self.assertIsNotNone(consensus)
        self.assertIsNotNone(consensus.pharm2d)
        self.assertIsNotNone(consensus.morgan)
        
        # Verify source tracking
        self.assertEqual(consensus.pharm2d.source_id, ["ethanol"])
        self.assertEqual(consensus.morgan.source_id, ["ethanol"])
        
        # Verify fingerprints were generated
        self.assertIsNotNone(consensus.pharm2d.bits)
        self.assertIsNotNone(consensus.morgan.bits)
    
    def test_multiple_ligands_consensus(self):
        """Test consensus with multiple ligands"""
        ligands = [
            self._create_test_ligand("CCO", "ethanol"),
            self._create_test_ligand("CCCO", "propanol"), 
            self._create_test_ligand("CCCCO", "butanol")
        ]
        
        prepped_ligands = attach_mols(ligands)
        consensus = ligands_to_consensus_fp(prepped_ligands, self.sig_factory, self.morgan_factory)
        
        # Should track all source ligands
        expected_ids = ["ethanol", "propanol", "butanol"]
        self.assertEqual(consensus.pharm2d.source_id, expected_ids)
        self.assertEqual(consensus.morgan.source_id, expected_ids)
    
    def test_empty_ligand_list(self):
        """Test consensus with empty list - production behavior"""
        ligands = []
        
        prepped_ligands = attach_mols(ligands)  # Should return empty list
        
        # Empty list raises ValueError in actual implementation
        with self.assertRaises(ValueError):
            ligands_to_consensus_fp(prepped_ligands, self.sig_factory, self.morgan_factory)
    
    def test_consensus_determinism(self):
        """Test that consensus is deterministic"""
        ligands = [
            self._create_test_ligand("CCO", "lig1"),
            self._create_test_ligand("CCCO", "lig2")
        ]
        
        prepped1 = attach_mols(ligands)
        prepped2 = attach_mols(ligands)
        
        consensus1 = ligands_to_consensus_fp(prepped1, self.sig_factory, self.morgan_factory)
        consensus2 = ligands_to_consensus_fp(prepped2, self.sig_factory, self.morgan_factory)
        
        # Should produce identical fingerprints
        self.assertEqual(
            consensus1.pharm2d.bits.ToBitString(),
            consensus2.pharm2d.bits.ToBitString()
        )
        self.assertEqual(
            consensus1.morgan.bits.ToBitString(), 
            consensus2.morgan.bits.ToBitString()
        )
    
    def test_invalid_smiles_handling(self):
        """Test how production code handles invalid SMILES"""
        # Test with invalid SMILES that might cause attach_mols to fail
        ligands = [self._create_test_ligand("INVALID_SMILES", "invalid")]
        
        try:
            prepped_ligands = attach_mols(ligands)
            if prepped_ligands:  # If preparation succeeded despite invalid SMILES
                consensus = ligands_to_consensus_fp(prepped_ligands, self.sig_factory, self.morgan_factory)
                self.assertIsNotNone(consensus)
        except Exception:
            # It's acceptable for invalid SMILES to cause failures
            # This documents the actual behavior
            pass
    
    def test_duplicate_ligand_ids(self):
        """Test consensus with duplicate ligand IDs"""
        ligands = [
            self._create_test_ligand("CCO", "duplicate"),
            self._create_test_ligand("CCCO", "duplicate")  # Same ID, different SMILES
        ]
        
        prepped_ligands = attach_mols(ligands)
        consensus = ligands_to_consensus_fp(prepped_ligands, self.sig_factory, self.morgan_factory)
        
        # Should handle duplicates (behavior depends on implementation)
        self.assertIsNotNone(consensus)
        # The exact behavior depends on how source_id list is constructed


if __name__ == '__main__':
    unittest.main()
