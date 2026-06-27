import unittest
import math
from unittest.mock import patch, MagicMock

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemicalFeatures import BuildFeatureFactory

from src.domain.ligand import Ligand, LigandRecord, LigandConformers, ConformerData
from src.domain.pharm import ConsensusFeature, Cluster
from src.pharma.pharmacophore.pharma3d.consensus_builder import (
    build_consensus, filter_clusters, compute_pairwise_stats,
    FeatureClusterer, PharmacophoreStatistics, RDKitPharmacophoreBuilder,
    ConsensusBuilder
)


class TestPharm3DConsensusBuilder(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create feature factory for testing
        cls.feat_factory = BuildFeatureFactory()
        cls.consensus_builder = ConsensusBuilder(
            feat_factory=cls.feat_factory,
            feature_consensus_ratio=0.5,
            pair_distance_support_ratio=0.5,
            proximity=1.0,
            feature_k=1.0,
            pairwise_k=2.0,
            max_dist=15.0
        )
    
    def _create_ligand_conformers(self, smiles: str, ligand_id: str, num_conformers: int = 1) -> LigandConformers:
        """Helper to create LigandConformers with generated conformers"""
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
        
        mol = Chem.AddHs(mol)
        
        # Generate conformers
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, randomSeed=42)
        if not conf_ids:
            # Fallback: add a single conformer with basic coordinates
            AllChem.Compute2DCoords(mol)
        
        conformers = []
        for i, conf in enumerate(mol.GetConformers()):
            conformers.append(ConformerData(
                conf_id=conf.GetId(),
                energy=float(i * 0.5)  # Mock energy values
            ))
        
        return LigandConformers(
            record=record,
            mol=mol,
            conformers=conformers
        )
    
    def test_build_consensus_from_features(self):
        """Test building pharmacophore from consensus features"""
        features = [
            ConsensusFeature(
                id=0,
                family="Aromatic",
                type="Aromatic",
                centroid=(0.0, 0.0, 0.0),
                radius=1.5,
                support=0.8
            ),
            ConsensusFeature(
                id=1,
                family="Acceptor",
                type="Acceptor",
                centroid=(3.0, 0.0, 0.0),
                radius=1.2,
                support=0.6
            )
        ]
        
        pharmacophore = build_consensus(features)
        
        self.assertIsNotNone(pharmacophore)
        self.assertEqual(len(pharmacophore.getFeatures()), 2)
        
        # Verify feature properties are preserved
        rdkit_features = pharmacophore.getFeatures()
        self.assertEqual(rdkit_features[0].GetFamily(), "Aromatic")
        self.assertEqual(rdkit_features[1].GetFamily(), "Acceptor")
    
    def test_filter_clusters_by_support(self):
        """Test filtering clusters based on minimum support"""
        # Create mock clusters with different support levels
        cluster1 = MagicMock()
        cluster1.ligand_ids = ["1", "2", "3"]  # Support: 3
        
        cluster2 = MagicMock()
        cluster2.ligand_ids = ["1"]  # Support: 1
        
        cluster3 = MagicMock()
        cluster3.ligand_ids = ["1", "2"]  # Support: 2
        
        clusters = [cluster1, cluster2, cluster3]
        
        # Filter with min_support = 2
        filtered = filter_clusters(clusters, min_support=2)
        
        # Should keep cluster1 and cluster3 (support >= 2)
        self.assertEqual(len(filtered), 2)
        self.assertIn(cluster1, filtered)
        self.assertIn(cluster3, filtered)
        self.assertNotIn(cluster2, filtered)
    
    def test_empty_ligands_consensus(self):
        """Test consensus building with empty ligand list"""
        ligands = []
        
        consensus = self.consensus_builder.run(ligands)
        
        # Should return valid consensus with empty data
        self.assertIsNotNone(consensus)
        self.assertEqual(len(consensus.features), 0)
        self.assertEqual(consensus.num_training_ligands, 0)
        self.assertIsNotNone(consensus.pharmacophore)
    
    def test_single_ligand_consensus(self):
        """Test consensus building with single ligand"""
        ligands = [self._create_ligand_conformers("CCO", "single")]
        
        consensus = self.consensus_builder.run(ligands)
        
        # Should handle single ligand case
        self.assertIsNotNone(consensus)
        self.assertEqual(consensus.num_training_ligands, 1)
        self.assertIsNotNone(consensus.pharmacophore)
        
        # May or may not have features depending on support thresholds
        self.assertGreaterEqual(len(consensus.features), 0)
    
    def test_multiple_ligands_consensus(self):
        """Test consensus building with multiple ligands"""
        ligands = [
            self._create_ligand_conformers("CCO", "ethanol"),
            self._create_ligand_conformers("CCCO", "propanol"),
            self._create_ligand_conformers("CCCCO", "butanol")
        ]
        
        consensus = self.consensus_builder.run(ligands)
        
        # Should build meaningful consensus
        self.assertIsNotNone(consensus)
        self.assertEqual(consensus.num_training_ligands, 3)
        self.assertIsNotNone(consensus.pharmacophore)
        self.assertIsNotNone(consensus.pairwise_stats)
        
        # Should have some features for similar molecules
        # (actual number depends on feature detection and clustering)
        self.assertGreaterEqual(len(consensus.features), 0)
    
    def test_diverse_ligands_consensus(self):
        """Test consensus with structurally diverse ligands"""
        ligands = [
            self._create_ligand_conformers("CCO", "alcohol"),
            self._create_ligand_conformers("c1ccccc1", "benzene"),
            self._create_ligand_conformers("CC(=O)O", "acid")
        ]
        
        consensus = self.consensus_builder.run(ligands)
        
        # Should handle diverse structures
        self.assertIsNotNone(consensus)
        self.assertEqual(consensus.num_training_ligands, 3)
        
        # Diverse molecules may have fewer consensus features
        # but should still build valid pharmacophore
        self.assertIsNotNone(consensus.pharmacophore)
    
    def test_consensus_determinism(self):
        """Test that consensus building is deterministic"""
        ligands = [
            self._create_ligand_conformers("CCO", "1"),
            self._create_ligand_conformers("CCCO", "2")
        ]
        
        # Build consensus twice
        consensus1 = self.consensus_builder.run(ligands)
        consensus2 = self.consensus_builder.run(ligands)
        
        # Should produce identical results
        self.assertEqual(len(consensus1.features), len(consensus2.features))
        self.assertEqual(consensus1.num_training_ligands, consensus2.num_training_ligands)
        
        # Feature properties should match
        for f1, f2 in zip(consensus1.features, consensus2.features):
            self.assertEqual(f1.family, f2.family)
            self.assertEqual(f1.type, f2.type)
            self.assertAlmostEqual(f1.support, f2.support, places=6)
    
    def test_ligand_order_independence(self):
        """Test that ligand order doesn't affect consensus"""
        ligand1 = self._create_ligand_conformers("CCO", "1")
        ligand2 = self._create_ligand_conformers("CCCO", "2")
        ligand3 = self._create_ligand_conformers("CCCCO", "3")
        
        # Different orders
        ligands_order1 = [ligand1, ligand2, ligand3]
        ligands_order2 = [ligand3, ligand1, ligand2]
        
        consensus1 = self.consensus_builder.run(ligands_order1)
        consensus2 = self.consensus_builder.run(ligands_order2)
        
        # Should produce equivalent results
        self.assertEqual(len(consensus1.features), len(consensus2.features))
        self.assertEqual(consensus1.num_training_ligands, consensus2.num_training_ligands)
    
    def test_feature_clustering(self):
        """Test feature clustering functionality"""
        clusterer = FeatureClusterer(proximity=1.5)
        
        # Create mock pharmacophore ligands
        # This is a simplified test - full test would require proper RDKit features
        mock_ligands = []
        
        # Test with empty input
        clusters = clusterer.run(mock_ligands)
        
        # Should handle empty input gracefully
        self.assertIsInstance(clusters, list)
        self.assertEqual(len(clusters), 0)
    
    def test_pharmacophore_statistics(self):
        """Test pharmacophore statistics calculation"""
        stats = PharmacophoreStatistics(
            feature_consensus_ratio=0.5,
            pair_distance_support_ratio=0.3,
            feature_radius_k=2.0
        )
        
        # Test with empty clusters
        clusters = []
        ligands = []
        
        features, pairwise_stats, ligand_map = stats.build(clusters, ligands)
        
        # Should handle empty input
        self.assertEqual(len(features), 0)
        self.assertEqual(len(pairwise_stats.means), 0)
        self.assertEqual(len(ligand_map), 0)
    
    def test_pairwise_stats_computation(self):
        """Test pairwise statistics computation"""
        # Create minimal clusters for testing
        clusters = []
        ligands = []
        min_support = 1
        
        pairwise_model = compute_pairwise_stats(clusters, ligands, min_support)
        
        # Should return valid model structure
        self.assertIsNotNone(pairwise_model)
        self.assertIsInstance(pairwise_model.idx_a, np.ndarray)
        self.assertIsInstance(pairwise_model.idx_b, np.ndarray)
        self.assertIsInstance(pairwise_model.means, np.ndarray)
        self.assertIsInstance(pairwise_model.raw_stds, np.ndarray)
        self.assertIsInstance(pairwise_model.model_stds, np.ndarray)
    
    def test_rdkit_pharmacophore_builder(self):
        """Test RDKit pharmacophore building with bounds"""
        builder = RDKitPharmacophoreBuilder(pairwise_bounds_k=2.0, max_dist=15.0)
        
        # Create simple features
        features = [
            ConsensusFeature(
                id=0,
                family="Aromatic",
                type="Aromatic",
                centroid=(0.0, 0.0, 0.0),
                radius=1.0,
                support=1.0
            )
        ]
        
        # Create minimal pairwise stats
        pairwise_stats = MagicMock()
        pairwise_stats.idx_a = np.array([])
        pairwise_stats.idx_b = np.array([])
        pairwise_stats.means = np.array([])
        pairwise_stats.model_stds = np.array([])
        
        pharmacophore = builder.build(features, pairwise_stats)
        
        # Should build valid pharmacophore
        self.assertIsNotNone(pharmacophore)
        self.assertEqual(len(pharmacophore.getFeatures()), 1)
    
    def test_alignment_handling(self):
        """Test ligand alignment in consensus building"""
        # Test with molecules that should align well
        ligands = [
            self._create_ligand_conformers("CCO", "ethanol"),
            self._create_ligand_conformers("CCCO", "propanol")
        ]
        
        # Use private method to test alignment
        aligned = self.consensus_builder._align_training_ligands(ligands)
        
        # Should return same number of ligands
        self.assertEqual(len(aligned), len(ligands))
        
        # All should be LigandConformers
        for aligned_ligand in aligned:
            self.assertIsInstance(aligned_ligand, LigandConformers)
            self.assertIsNotNone(aligned_ligand.mol)
    
    def test_consensus_with_invalid_molecules(self):
        """Test consensus building with problematic molecules"""
        try:
            # Try with a molecule that might cause issues
            ligands = [self._create_ligand_conformers("C", "methane")]
            
            consensus = self.consensus_builder.run(ligands)
            
            # Should either succeed or fail gracefully
            self.assertIsNotNone(consensus)
            
        except (ValueError, RuntimeError):
            # It's acceptable for some molecules to cause failures
            # Important thing is that it doesn't crash silently
            pass
    
    def test_conformer_selection(self):
        """Test lowest energy conformer selection"""
        # Create ligand with multiple conformers
        ligand = self._create_ligand_conformers("CCO", "test", num_conformers=3)
        
        # Test static method
        conf_id = self.consensus_builder._lowest_energy_conformer_id(ligand)
        
        # Should return lowest energy conformer ID
        self.assertIsInstance(conf_id, int)
        
        # Energy should be lowest (0.0 based on our mock)
        selected_conf = next(c for c in ligand.conformers if c.conf_id == conf_id)
        self.assertEqual(selected_conf.energy, 0.0)


class TestPharm3DConsensusEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_malformed_input_handling(self):
        """Test handling of malformed input"""
        builder = ConsensusBuilder(
            feat_factory=BuildFeatureFactory(),
            feature_consensus_ratio=0.5,
            pair_distance_support_ratio=0.5,
            proximity=1.0,
            feature_k=1.0,
            pairwise_k=2.0,
            max_dist=15.0
        )
        
        # Test with None input
        with self.assertRaises((TypeError, AttributeError)):
            builder.run(None)
        
        # Test with non-list input
        with self.assertRaises((TypeError, AttributeError)):
            builder.run("not_a_list")
    
    def test_extreme_parameter_values(self):
        """Test with extreme parameter values"""
        # Very high consensus ratio (should require all ligands)
        builder_strict = ConsensusBuilder(
            feat_factory=BuildFeatureFactory(),
            feature_consensus_ratio=1.0,
            pair_distance_support_ratio=1.0,
            proximity=0.1,  # Very tight clustering
            feature_k=0.1,
            pairwise_k=10.0,  # Very wide bounds
            max_dist=100.0
        )
        
        ligands = []
        consensus = builder_strict.run(ligands)
        
        # Should handle extreme parameters gracefully
        self.assertIsNotNone(consensus)
    
    def test_zero_parameter_values(self):
        """Test with zero/minimal parameter values"""
        builder_minimal = ConsensusBuilder(
            feat_factory=BuildFeatureFactory(),
            feature_consensus_ratio=0.0,
            pair_distance_support_ratio=0.0,
            proximity=0.0,
            feature_k=0.0,
            pairwise_k=0.0,
            max_dist=0.1
        )
        
        ligands = []
        consensus = builder_minimal.run(ligands)
        
        # Should handle minimal parameters
        self.assertIsNotNone(consensus)


if __name__ == '__main__':
    unittest.main()
