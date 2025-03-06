"""
Unit tests for threshold cointegration modules.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.threshold import ThresholdCointegration, calculate_asymmetric_adjustment
from src.models.threshold_vecm import ThresholdVECM, calculate_half_lives, calculate_regime_transition_matrix
from src.utils import ModelError, ValidationError


class TestThresholdCointegration(unittest.TestCase):
    """Tests for the ThresholdCointegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate cointegrated series with threshold effect
        # First create a common trend
        self.n = 200
        trend = np.cumsum(np.random.normal(0, 1, self.n))
        
        # Equilibrium relationship with threshold effect
        eq_error = np.random.normal(0, 1, self.n)
        
        # Create two regimes for different adjustment speeds
        threshold = 0.0  # arbitrary threshold
        below = eq_error <= threshold
        above = ~below
        
        # Series 1 adjusts faster in upper regime
        self.series1 = np.zeros(self.n)
        for t in range(1, self.n):
            if above[t-1]:
                self.series1[t] = 2 * trend[t] + 0.6 * eq_error[t-1] + np.random.normal(0, 0.5)
            else:
                self.series1[t] = 2 * trend[t] + 0.2 * eq_error[t-1] + np.random.normal(0, 0.5)
        
        # Series 2 is simpler
        self.series2 = trend + np.random.normal(0, 0.5, self.n)
        
        # Non-cointegrated series
        self.series3 = np.cumsum(np.random.normal(0, 1, self.n))
        self.series4 = np.cumsum(np.random.normal(0, 1, self.n))
    
    def test_initialization(self):
        """Test initialization of threshold cointegration model."""
        # Valid initialization
        model = ThresholdCointegration(self.series1, self.series2)
        self.assertEqual(len(model.data1), self.n)
        self.assertEqual(len(model.data2), self.n)
        
        # Test invalid inputs
        with self.assertRaises(ValidationError):
            # Too short series
            ThresholdCointegration(self.series1[:10], self.series2[:10])
        
        with self.assertRaises(ValidationError):
            # Unequal length series
            ThresholdCointegration(self.series1, self.series2[:-5])
    
    def test_cointegration_estimation(self):
        """Test estimation of cointegration relationship."""
        model = ThresholdCointegration(self.series1, self.series2)
        results = model.estimate_cointegration()
        
        # Check result structure
        self.assertIn('statistic', results)
        self.assertIn('pvalue', results)
        self.assertIn('cointegrated', results)
        self.assertIn('beta0', results)
        self.assertIn('beta1', results)
        
        # Series should be cointegrated
        self.assertTrue(results['cointegrated'])
        
        # Check if equilibrium errors are computed
        self.assertIsNotNone(model.eq_errors)
        self.assertEqual(len(model.eq_errors), self.n)
    
    def test_threshold_estimation(self):
        """Test threshold parameter estimation."""
        model = ThresholdCointegration(self.series1, self.series2)
        model.estimate_cointegration()
        
        # Test threshold estimation
        results = model.estimate_threshold(n_grid=50)
        
        # Check result structure
        self.assertIn('threshold', results)
        self.assertIn('ssr', results)
        self.assertIn('all_thresholds', results)
        self.assertIn('all_ssrs', results)
        
        # Check threshold is set in model
        self.assertIsNotNone(model.threshold)
    
    def test_tvecm_estimation(self):
        """Test threshold VECM estimation."""
        model = ThresholdCointegration(self.series1, self.series2)
        model.estimate_cointegration()
        model.estimate_threshold()
        
        # Test TVECM estimation
        results = model.estimate_tvecm()
        
        # Check result structure
        self.assertIn('equation1', results)
        self.assertIn('equation2', results)
        self.assertIn('adjustment_below_1', results)
        self.assertIn('adjustment_above_1', results)
        self.assertIn('adjustment_below_2', results)
        self.assertIn('adjustment_above_2', results)
        self.assertIn('threshold', results)
        self.assertIn('cointegration_beta', results)
    
    def test_asymmetric_adjustment(self):
        """Test calculating asymmetric adjustment metrics."""
        model = ThresholdCointegration(self.series1, self.series2)
        model.estimate_cointegration()
        model.estimate_threshold()
        tvecm_results = model.estimate_tvecm()
        
        # Calculate asymmetric adjustment
        adjustment = calculate_asymmetric_adjustment(tvecm_results)
        
        # Check result structure
        self.assertIn('half_life_below_1', adjustment)
        self.assertIn('half_life_above_1', adjustment)
        self.assertIn('asymmetry_1', adjustment)
        self.assertIn('asymmetry_2', adjustment)
        
        # Check if half-lives are meaningful
        self.assertGreater(adjustment['half_life_below_1'], 0)
        self.assertGreater(adjustment['half_life_above_1'], 0)


class TestThresholdVECM(unittest.TestCase):
    """Tests for the ThresholdVECM class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate cointegrated series in a system
        self.n = 150
        
        # Common stochastic trend
        trend = np.cumsum(np.random.normal(0, 1, self.n))
        
        # Generate three series with cointegration relationship
        noise1 = np.random.normal(0, 0.5, self.n)
        noise2 = np.random.normal(0, 0.5, self.n)
        
        self.y1 = trend + noise1
        self.y2 = 2 * trend + noise2
        
        # Create a DataFrame
        self.data = pd.DataFrame({
            'y1': self.y1,
            'y2': self.y2
        })
    
    def test_initialization(self):
        """Test initialization of threshold VECM model."""
        # Valid initialization
        model = ThresholdVECM(self.data)
        self.assertEqual(model.data.shape, (self.n, 2))
        
        # Test invalid inputs
        with self.assertRaises(ValidationError):
            # Single variable
            ThresholdVECM(pd.DataFrame({'x': range(100)}))
    
    def test_linear_vecm(self):
        """Test estimation of linear VECM."""
        model = ThresholdVECM(self.data)
        results = model.estimate_linear_vecm()
        
        # Check results
        self.assertIsNotNone(results)
        self.assertTrue(hasattr(results, 'beta'))
        self.assertTrue(hasattr(results, 'alpha'))
        self.assertTrue(hasattr(results, 'llf'))
    
    def test_threshold_search(self):
        """Test threshold parameter grid search."""
        model = ThresholdVECM(self.data)
        model.estimate_linear_vecm()
        
        # Test grid search
        results = model.grid_search_threshold(n_grid=20)
        
        # Check result structure
        self.assertIn('threshold', results)
        self.assertIn('llf', results)
        self.assertIn('all_thresholds', results)
        self.assertIn('all_llfs', results)
        
        # Check threshold is set in model
        self.assertIsNotNone(model.threshold)
    
    def test_tvecm_estimation(self):
        """Test threshold VECM estimation."""
        model = ThresholdVECM(self.data)
        model.estimate_linear_vecm()
        model.grid_search_threshold()
        
        # Test TVECM estimation
        results = model.estimate_tvecm()
        
        # Check result structure
        self.assertIn('threshold', results)
        self.assertIn('cointegration_beta', results)
        self.assertIn('alpha_below', results)
        self.assertIn('alpha_above', results)
        self.assertIn('below_regime', results)
        self.assertIn('above_regime', results)
        self.assertIn('llf', results)
        
        # Check dimension of alpha vectors
        self.assertEqual(len(results['alpha_below']), 2)
        self.assertEqual(len(results['alpha_above']), 2)
    
    def test_half_lives(self):
        """Test calculation of half-lives."""
        model = ThresholdVECM(self.data)
        model.estimate_linear_vecm()
        model.grid_search_threshold()
        results = model.estimate_tvecm()
        
        # Calculate half-lives
        half_lives = calculate_half_lives(results)
        
        # Check result structure
        self.assertIn('below_regime', half_lives)
        self.assertIn('above_regime', half_lives)
        
        # Check dimensions
        self.assertEqual(len(half_lives['below_regime']), 2)
        self.assertEqual(len(half_lives['above_regime']), 2)
    
    def test_transition_matrix(self):
        """Test calculation of regime transition matrix."""
        model = ThresholdVECM(self.data)
        model.estimate_linear_vecm()
        model.grid_search_threshold()
        results = model.estimate_tvecm()
        
        # Calculate transition matrix
        trans_matrix = calculate_regime_transition_matrix(
            results['equilibrium_errors'], 
            results['threshold']
        )
        
        # Check dimensions
        self.assertEqual(trans_matrix.shape, (2, 2))
        
        # Check row sums (should be close to 1)
        self.assertAlmostEqual(trans_matrix.iloc[0].sum(), 1.0, places=6)
        self.assertAlmostEqual(trans_matrix.iloc[1].sum(), 1.0, places=6)
        
        # Check values between 0 and 1
        self.assertTrue((trans_matrix >= 0).all().all())
        self.assertTrue((trans_matrix <= 1).all().all())


if __name__ == '__main__':
    unittest.main()