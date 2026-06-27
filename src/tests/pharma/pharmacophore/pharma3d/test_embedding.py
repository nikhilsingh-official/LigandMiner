import unittest
import numpy as np
from unittest.mock import MagicMock

from src.domain.pharm import PairwiseModel, PairDiagnostics
from src.pharma.pharmacophore.pharma3d.embedding import score_embedding_with_diagnostics


class TestPharm3DEmbedding(unittest.TestCase):
    """Test 3D pharmacophore embedding scoring functionality"""
    
    def test_score_embedding_with_valid_data(self):
        """Test embedding scoring with valid input data"""
        # Create mock embedding
        mock_embedding = MagicMock()
        
        # Create realistic pairwise model
        model = PairwiseModel(
            idx_a=np.array([0, 1]),
            idx_b=np.array([1, 2]),
            means=np.array([3.5, 5.2]),
            raw_stds=np.array([0.8, 1.2]),
            model_stds=np.array([0.8, 1.2])
        )
        
        # Mock the embedding_pair_distances function to return realistic distances
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([3.4, 5.1])  # Close to model means
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Should return valid score and diagnostics
            self.assertIsInstance(score, (int, float))
            self.assertIsInstance(diagnostics, PairDiagnostics)
            
            # Score should be reasonable (low for good match)
            self.assertGreater(score, 0.0)
            self.assertLess(score, 100.0)
            
            # Diagnostics should have correct structure
            self.assertEqual(len(diagnostics.observed_distances), 2)
            self.assertEqual(len(diagnostics.z_scores), 2)
            self.assertEqual(len(diagnostics.nll_contributions), 2)
    
    def test_score_embedding_perfect_match(self):
        """Test scoring when embedding perfectly matches model"""
        mock_embedding = MagicMock()
        
        model = PairwiseModel(
            idx_a=np.array([0]),
            idx_b=np.array([1]),
            means=np.array([4.0]),
            raw_stds=np.array([0.5]),
            model_stds=np.array([0.5])
        )
        
        # Perfect match: observed distance equals mean
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([4.0])  # Exactly matches mean
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Perfect match should have very low score
            self.assertLess(score, 1.0)
            
            # Z-score should be close to 0 for perfect match
            self.assertAlmostEqual(diagnostics.z_scores[0], 0.0, places=3)
    
    def test_score_embedding_poor_match(self):
        """Test scoring when embedding poorly matches model"""
        mock_embedding = MagicMock()
        
        model = PairwiseModel(
            idx_a=np.array([0]),
            idx_b=np.array([1]),
            means=np.array([3.0]),
            raw_stds=np.array([0.2]),
            model_stds=np.array([0.2])
        )
        
        # Poor match: observed distance far from mean
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([6.0])  # Far from mean (3.0)
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Poor match should have higher score
            self.assertGreater(score, 10.0)
            
            # Z-score should be large for poor match
            self.assertGreater(abs(diagnostics.z_scores[0]), 10.0)
    
    def test_score_embedding_multiple_pairs(self):
        """Test scoring with multiple feature pairs"""
        mock_embedding = MagicMock()
        
        model = PairwiseModel(
            idx_a=np.array([0, 1, 2]),
            idx_b=np.array([1, 2, 3]),
            means=np.array([2.0, 4.0, 6.0]),
            raw_stds=np.array([0.5, 0.8, 1.0]),
            model_stds=np.array([0.5, 0.8, 1.0])
        )
        
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([2.1, 3.8, 6.2])  # Close matches
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Should handle multiple pairs
            self.assertEqual(len(diagnostics.observed_distances), 3)
            self.assertEqual(len(diagnostics.z_scores), 3)
            self.assertEqual(len(diagnostics.nll_contributions), 3)
            
            # All z-scores should be reasonable for close matches
            for z_score in diagnostics.z_scores:
                self.assertLess(abs(z_score), 1.0)
    
    def test_score_embedding_empty_model(self):
        """Test behavior with empty pairwise model"""
        mock_embedding = MagicMock()
        
        # Empty model
        model = PairwiseModel(
            idx_a=np.array([]),
            idx_b=np.array([]),
            means=np.array([]),
            raw_stds=np.array([]),
            model_stds=np.array([])
        )
        
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([])
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Should handle empty model gracefully
            self.assertIsInstance(score, (int, float))
            self.assertEqual(len(diagnostics.observed_distances), 0)
            self.assertEqual(len(diagnostics.z_scores), 0)
            self.assertEqual(len(diagnostics.nll_contributions), 0)
    
    def test_score_embedding_single_pair(self):
        """Test scoring with single feature pair"""
        mock_embedding = MagicMock()
        
        model = PairwiseModel(
            idx_a=np.array([0]),
            idx_b=np.array([1]),
            means=np.array([5.0]),
            raw_stds=np.array([1.0]),
            model_stds=np.array([1.0])
        )
        
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([4.5])
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Single pair should work correctly
            self.assertIsInstance(score, (int, float))
            self.assertEqual(len(diagnostics.observed_distances), 1)
            self.assertAlmostEqual(diagnostics.observed_distances[0], 4.5)
    
    def test_score_embedding_varying_tolerances(self):
        """Test scoring with different model tolerances"""
        mock_embedding = MagicMock()
        
        # Model with varying tolerances
        model = PairwiseModel(
            idx_a=np.array([0, 1]),
            idx_b=np.array([1, 2]),
            means=np.array([3.0, 3.0]),
            raw_stds=np.array([0.1, 2.0]),  # Very tight vs very loose
            model_stds=np.array([0.1, 2.0])
        )
        
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([3.2, 3.2])  # Same deviation from mean
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Tight tolerance should have large z-score, loose tolerance should have small z-score
            self.assertGreater(abs(diagnostics.z_scores[0]), abs(diagnostics.z_scores[1]))
    
    def test_score_embedding_edge_case_distances(self):
        """Test with edge case distance values"""
        mock_embedding = MagicMock()
        
        model = PairwiseModel(
            idx_a=np.array([0]),
            idx_b=np.array([1]),
            means=np.array([1.0]),
            raw_stds=np.array([0.1]),
            model_stds=np.array([0.1])
        )
        
        # Test with zero distance
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([0.0])
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Should handle zero distance
            self.assertIsInstance(score, (int, float))
            self.assertIsInstance(diagnostics.z_scores[0], (int, float))
    
    def test_diagnostics_structure_validation(self):
        """Test that diagnostics have the expected structure"""
        mock_embedding = MagicMock()
        
        model = PairwiseModel(
            idx_a=np.array([0, 1]),
            idx_b=np.array([1, 2]), 
            means=np.array([2.0, 4.0]),
            raw_stds=np.array([0.5, 0.8]),
            model_stds=np.array([0.5, 0.8])
        )
        
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([2.1, 3.9])
            
            score, diagnostics = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Validate diagnostics structure
            self.assertIsInstance(diagnostics.observed_distances, np.ndarray)
            self.assertIsInstance(diagnostics.z_scores, np.ndarray)
            self.assertIsInstance(diagnostics.nll_contributions, np.ndarray)
            
            # All arrays should have same length
            self.assertEqual(len(diagnostics.observed_distances), len(diagnostics.z_scores))
            self.assertEqual(len(diagnostics.z_scores), len(diagnostics.nll_contributions))
            
            # Values should be finite
            self.assertTrue(np.all(np.isfinite(diagnostics.observed_distances)))
            self.assertTrue(np.all(np.isfinite(diagnostics.z_scores)))
            self.assertTrue(np.all(np.isfinite(diagnostics.nll_contributions)))
    
    def test_score_consistency(self):
        """Test that identical inputs produce identical scores"""
        mock_embedding = MagicMock()
        
        model = PairwiseModel(
            idx_a=np.array([0]),
            idx_b=np.array([1]),
            means=np.array([3.0]),
            raw_stds=np.array([0.5]),
            model_stds=np.array([0.5])
        )
        
        with unittest.mock.patch('src.pharma.pharmacophore.pharma3d.embedding.embedding_pair_distances') as mock_distances:
            mock_distances.return_value = np.array([2.8])
            
            # Score twice with identical inputs
            score1, diagnostics1 = score_embedding_with_diagnostics(mock_embedding, model)
            score2, diagnostics2 = score_embedding_with_diagnostics(mock_embedding, model)
            
            # Should be identical
            self.assertEqual(score1, score2)
            np.testing.assert_array_equal(diagnostics1.observed_distances, diagnostics2.observed_distances)
            np.testing.assert_array_equal(diagnostics1.z_scores, diagnostics2.z_scores)
            np.testing.assert_array_equal(diagnostics1.nll_contributions, diagnostics2.nll_contributions)


if __name__ == '__main__':
    unittest.main()
