"""
Unit tests for the threshold models module.

This module tests the threshold autoregressive (TAR) and momentum threshold 
autoregressive (M-TAR) models in the threshold.py module.
"""
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from unittest.mock import patch, MagicMock

from src.models.threshold import ThresholdCointegration
from src.utils.validation import validate_time_series


class TestThresholdCointegration(unittest.TestCase):
    """Test cases for the ThresholdCointegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        np.random.seed(42)  # For reproducibility
        
        # Sample size
        n = 200
        
        # Create cointegrated series
        # Start with a random walk
        random_walk = np.cumsum(np.random.normal(0, 1, n))
        
        # Create a second series that is cointegrated with the first (same stochastic trend)
        # y_t = beta * x_t + e_t where e_t is stationary (mean-reverting)
        beta = 1.5
        e_t = np.random.normal(0, 1, n) * 0.3  # Error term (stationary)
        
        # Introduce threshold behavior
        # e_t has faster mean reversion if above a threshold and slower below
        # This creates asymmetric adjustment
        threshold = 0.5
        for i in range(1, n):
            if e_t[i-1] > threshold:
                e_t[i] = 0.5 * e_t[i-1] + np.random.normal(0, 0.2)  # Fast adjustment
            elif e_t[i-1] < -threshold:
                e_t[i] = 0.8 * e_t[i-1] + np.random.normal(0, 0.2)  # Slow adjustment
            else:
                e_t[i] = 0.7 * e_t[i-1] + np.random.normal(0, 0.2)  # Medium adjustment
        
        # Generate final series
        self.series1 = random_walk
        self.series2 = beta * random_walk + e_t
        
        # Create model
        self.model = ThresholdCointegration(self.series1, self.series2)
    
    def test_initialization(self):
        """Test initialization of ThresholdCointegration class."""
        # Test with default parameters
        model = ThresholdCointegration(self.series1, self.series2)
        
        # Check that series are stored as numpy arrays
        self.assertIsInstance(model.series1, np.ndarray)
        self.assertIsInstance(model.series2, np.ndarray)
        self.assertEqual(len(model.series1), len(self.series1))
        self.assertEqual(len(model.series2), len(self.series2))
        
        # Test with custom names
        model_named = ThresholdCointegration(
            self.series1, self.series2, 
            market1_name="Market1", market2_name="Market2"
        )
        self.assertEqual(model_named.market1_name, "Market1")
        self.assertEqual(model_named.market2_name, "Market2")
        
        # Test with pandas Series
        series1_pd = pd.Series(self.series1)
        series2_pd = pd.Series(self.series2)
        model_pd = ThresholdCointegration(series1_pd, series2_pd)
        np.testing.assert_array_equal(model_pd.series1, self.series1)
        np.testing.assert_array_equal(model_pd.series2, self.series2)
        
        # Test validation
        with self.assertRaises(ValueError):
            # Different lengths
            ThresholdCointegration(self.series1[:100], self.series2)
            
        with self.assertRaises(ValueError):
            # Non-numeric series
            ThresholdCointegration(
                np.array(['a', 'b', 'c']), 
                self.series2[:3]
            )
    
    def test_estimate_cointegration(self):
        """Test estimation of cointegration relationship."""
        # Calculate cointegration
        result = self.model.estimate_cointegration()
        
        # Check that results dictionary has required keys
        required_keys = ['cointegrated', 'test_statistic', 'critical_value', 'p_value']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that cointegrating vector is estimated
        self.assertIsNotNone(self.model.beta)
        
        # Check that residuals are calculated
        self.assertIsNotNone(self.model.residuals)
        self.assertEqual(len(self.model.residuals), len(self.series1))
        
        # Check result types
        self.assertIsInstance(result['cointegrated'], bool)
        self.assertIsInstance(result['test_statistic'], float)
        self.assertIsInstance(result['critical_value'], float)
        self.assertIsInstance(result['p_value'], float)
    
    @patch('src.models.threshold.sm.tsa.stattools.coint')
    def test_cointegration_with_mock(self, mock_coint):
        """Test cointegration estimation with mocked statsmodels function."""
        # Set up mock to return evidence of cointegration
        mock_coint.return_value = (-4.5, 0.01, np.array([-3.0, -3.5, -4.0]))
        
        # Calculate cointegration
        result = self.model.estimate_cointegration()
        
        # Check that mock was called
        mock_coint.assert_called_once()
        
        # Check that we get expected results
        self.assertTrue(result['cointegrated'])
        self.assertEqual(result['test_statistic'], -4.5)
        self.assertEqual(result['p_value'], 0.01)
        self.assertEqual(result['critical_value'], -3.5)  # 5% critical value
    
    def test_estimate_threshold(self):
        """Test estimation of threshold parameter."""
        # First need to estimate cointegration
        self.model.estimate_cointegration()
        
        # Now estimate threshold
        result = self.model.estimate_threshold()
        
        # Check that results dictionary has required keys
        required_keys = ['threshold', 'significant', 'threshold_min', 'threshold_max']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that threshold is set
        self.assertIsNotNone(self.model.threshold)
        
        # Check that threshold is within the range of residuals
        min_residual = np.min(self.model.residuals)
        max_residual = np.max(self.model.residuals)
        self.assertGreaterEqual(self.model.threshold, min_residual)
        self.assertLessEqual(self.model.threshold, max_residual)
        
        # Check result types
        self.assertIsInstance(result['threshold'], float)
        self.assertIsInstance(result['significant'], bool)
        self.assertIsInstance(result['threshold_min'], float)
        self.assertIsInstance(result['threshold_max'], float)
    
    def test_estimate_tar(self):
        """Test estimation of TAR model."""
        # First need to estimate cointegration and threshold
        self.model.estimate_cointegration()
        self.model.estimate_threshold()
        
        # Now estimate TAR model
        result = self.model.estimate_tar()
        
        # Check that results dictionary has required keys
        required_keys = ['asymmetric', 'adjustment_below', 'adjustment_middle', 'adjustment_above']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check result types
        self.assertIsInstance(result['asymmetric'], bool)
        self.assertIsInstance(result['adjustment_below'], float)
        self.assertIsInstance(result['adjustment_middle'], float)
        self.assertIsInstance(result['adjustment_above'], float)
        
        # Adjustment speeds should be negative for stable system
        self.assertLessEqual(result['adjustment_below'], 0)
        self.assertLessEqual(result['adjustment_middle'], 0)
        self.assertLessEqual(result['adjustment_above'], 0)
    
    def test_estimate_mtar(self):
        """Test estimation of M-TAR model."""
        # First need to estimate cointegration
        self.model.estimate_cointegration()
        
        # Now estimate M-TAR model
        result = self.model.estimate_mtar()
        
        # Check that results dictionary has required keys
        required_keys = ['asymmetric', 'adjustment_negative', 'adjustment_positive']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check result types
        self.assertIsInstance(result['asymmetric'], bool)
        self.assertIsInstance(result['adjustment_negative'], float)
        self.assertIsInstance(result['adjustment_positive'], float)
        
        # Adjustment speeds should be negative for stable system
        self.assertLessEqual(result['adjustment_negative'], 0)
        self.assertLessEqual(result['adjustment_positive'], 0)
    
    def test_calculate_half_lives(self):
        """Test calculation of half-lives for adjustment speeds."""
        # Mock the model to have known adjustment speeds
        self.model.results = {
            'adjustment_below': -0.5,
            'adjustment_middle': -0.2,
            'adjustment_above': -0.4
        }
        
        # Calculate half-lives
        half_lives = self.model.calculate_half_lives()
        
        # Check that half-lives are calculated correctly
        # Half-life = ln(0.5) / ln(1 + adjustment) = -0.693 / ln(1 + adjustment)
        expected_below = -np.log(0.5) / np.log(1 - 0.5)
        expected_middle = -np.log(0.5) / np.log(1 - 0.2)
        expected_above = -np.log(0.5) / np.log(1 - 0.4)
        
        self.assertAlmostEqual(half_lives['half_life_below'], expected_below, places=5)
        self.assertAlmostEqual(half_lives['half_life_middle'], expected_middle, places=5)
        self.assertAlmostEqual(half_lives['half_life_above'], expected_above, places=5)
    
    def test_predict_adjustment(self):
        """Test prediction of price adjustment."""
        # Mock the model to have known parameters
        self.model.beta = 1.5
        self.model.threshold = 10.0
        self.model.results = {
            'adjustment_below': -0.5,
            'adjustment_middle': -0.2,
            'adjustment_above': -0.4
        }
        
        # Test adjustment when price differential is above threshold
        adjustment = self.model.predict_adjustment(100, 80)  # diff = 100 - 1.5*80 = -20, which is < -10
        self.assertAlmostEqual(adjustment, -0.5 * -20)  # adjustment = -0.5 * deviation
        
        # Test adjustment when price differential is in the middle regime
        adjustment = self.model.predict_adjustment(95, 60)  # diff = 95 - 1.5*60 = 5, which is |5| < 10
        self.assertAlmostEqual(adjustment, -0.2 * 5)  # adjustment = -0.2 * deviation
        
        # Test adjustment when price differential is below negative threshold
        adjustment = self.model.predict_adjustment(80, 60)  # diff = 80 - 1.5*60 = -10, which is > -10
        self.assertAlmostEqual(adjustment, -0.2 * -10)  # adjustment = -0.2 * deviation


if __name__ == '__main__':
    unittest.main()
