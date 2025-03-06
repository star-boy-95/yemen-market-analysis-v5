"""
Unit tests for model modules.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unit_root import UnitRootTester, determine_integration_order
from src.models.cointegration import CointegrationTester, estimate_cointegration_vector, calculate_half_life
from src.utils import ModelError, ValidationError


class TestUnitRootTester(unittest.TestCase):
    """Tests for the UnitRootTester class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = UnitRootTester()
        
        # Create test time series
        np.random.seed(42)
        # Non-stationary series (random walk)
        self.nonstationary = np.cumsum(np.random.normal(0, 1, 100))
        # Stationary series
        self.stationary = np.random.normal(0, 1, 100)
    
    def test_adf_test(self):
        """Test ADF test."""
        # Test on nonstationary series
        result_ns = self.tester.test_adf(self.nonstationary)
        
        # Test on stationary series
        result_s = self.tester.test_adf(self.stationary)
        
        # Check expected outcomes
        self.assertFalse(result_ns['stationary'])
        self.assertTrue(result_s['stationary'])
        
        # Check result structure
        for result in [result_ns, result_s]:
            self.assertIn('trace_statistics', result)
            self.assertIn('trace_critical_values', result)
            self.assertIn('max_statistics', result)
            self.assertIn('max_critical_values', result)
            self.assertIn('rank_trace', result)
            self.assertIn('rank_max', result)
            self.assertIn('cointegration_vectors', result)
    
    def test_calculate_half_life(self):
        """Test calculating half-life of deviations."""
        # Create an AR(1) process with known coefficient
        ar_coef = 0.8
        n = 100
        e = np.random.normal(0, 1, n)
        y = np.zeros(n)
        
        # Generate AR(1) process: y_t = ar_coef * y_{t-1} + e_t
        for t in range(1, n):
            y[t] = ar_coef * y[t-1] + e[t]
        
        # Calculate half-life
        half_life = calculate_half_life(y)
        
        # Expected half-life: log(0.5) / log(|ar_coefficient|)
        expected_half_life = np.log(0.5) / np.log(abs(ar_coef))
        
        # Check result is close to expected
        self.assertAlmostEqual(half_life, expected_half_life, delta=1.0)
    
    def test_estimate_cointegration_vector(self):
        """Test estimating cointegration vector."""
        # Create cointegrated series with known relationship: y = 2*x + 3 + noise
        x = np.cumsum(np.random.normal(0, 1, 100))
        y = 2 * x + 3 + np.random.normal(0, 0.5, 100)
        
        # Estimate cointegration vector
        beta, residuals = estimate_cointegration_vector(y, x)
        
        # Check coefficients are close to true values
        self.assertAlmostEqual(beta[0], 3.0, delta=1.0)  # Intercept
        self.assertAlmostEqual(beta[1], 2.0, delta=0.5)  # Slope
        
        # Check residuals properties
        self.assertEqual(len(residuals), 100)
        self.assertLess(np.std(residuals), 1.0)  # Residuals should have small std


class TestIntegration(unittest.TestCase):
    """Integration tests between unit root and cointegration modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test series for integration testing
        np.random.seed(42)
        
        # Common random walk component
        random_walk = np.cumsum(np.random.normal(0, 1, 100))
        
        # Create cointegrated series
        self.y1 = random_walk + np.random.normal(0, 0.5, 100)
        self.y2 = 2 * random_walk + 3 + np.random.normal(0, 0.5, 100)
        
        # Create non-cointegrated series
        self.z1 = np.cumsum(np.random.normal(0, 1, 100))
        self.z2 = np.cumsum(np.random.normal(0, 1, 100))
        
        # Initialize testers
        self.unit_root_tester = UnitRootTester()
        self.cointegration_tester = CointegrationTester()
    
    def test_integrated_series_cointegration(self):
        """Test integration between unit root and cointegration testing."""
        # Check individual series are non-stationary (I(1))
        y1_order = determine_integration_order(self.y1)
        y2_order = determine_integration_order(self.y2)
        
        # Both should be I(1)
        self.assertEqual(y1_order, 1)
        self.assertEqual(y2_order, 1)
        
        # Test for cointegration
        result = self.cointegration_tester.test_engle_granger(self.y1, self.y2)
        
        # Series should be cointegrated
        self.assertTrue(result['cointegrated'])
        
        # Validate residuals are stationary
        residuals = result['residuals']
        residuals_stationary = self.unit_root_tester.test_adf(residuals)['stationary']
        
        # Residuals should be stationary if cointegrated
        self.assertTrue(residuals_stationary)
    
    def test_non_cointegrated_series(self):
        """Test non-cointegrated series."""
        # Check individual series are non-stationary (I(1))
        z1_order = determine_integration_order(self.z1)
        z2_order = determine_integration_order(self.z2)
        
        # Both should be I(1)
        self.assertEqual(z1_order, 1)
        self.assertEqual(z2_order, 1)
        
        # Test for cointegration
        result = self.cointegration_tester.test_engle_granger(self.z1, self.z2)
        
        # Series should not be cointegrated
        self.assertFalse(result['cointegrated'])
        
        # Residuals should not be stationary
        residuals = result['residuals']
        residuals_stationary = self.unit_root_tester.test_adf(residuals)['stationary']
        
        # Residuals should not be stationary if not cointegrated
        self.assertFalse(residuals_stationary)assertIn('statistic', result)
            self.assertIn('pvalue', result)
            self.assertIn('usedlag', result)
            self.assertIn('critical_values', result)
    
    def test_kpss_test(self):
        """Test KPSS test."""
        # Test on nonstationary series
        result_ns = self.tester.test_kpss(self.nonstationary)
        
        # Test on stationary series
        result_s = self.tester.test_kpss(self.stationary)
        
        # Check expected outcomes
        self.assertFalse(result_ns['stationary'])
        self.assertTrue(result_s['stationary'])
        
        # Check result structure
        for result in [result_ns, result_s]:
            self.assertIn('statistic', result)
            self.assertIn('pvalue', result)
            self.assertIn('critical_values', result)
    
    def test_run_all_tests(self):
        """Test running all tests."""
        all_results = self.tester.run_all_tests(self.stationary)
        
        # Check if all test results are present
        self.assertIn('adf', all_results)
        self.assertIn('adf_gls', all_results)
        self.assertIn('kpss', all_results)
        self.assertIn('zivot_andrews', all_results)
    
    def test_too_short_series(self):
        """Test error handling for too short series."""
        short_series = np.random.normal(0, 1, 5)
        
        with self.assertRaises(ValidationError):
            self.tester.test_adf(short_series)
            
    def test_integration_order(self):
        """Test determining integration order."""
        # I(0) series - already stationary
        i0_series = np.random.normal(0, 1, 100)
        
        # I(1) series - random walk
        i1_series = np.cumsum(np.random.normal(0, 1, 100))
        
        # I(2) series - double integration
        i2_series = np.cumsum(np.cumsum(np.random.normal(0, 1, 100)))
        
        # Test results
        self.assertEqual(determine_integration_order(i0_series), 0)
        self.assertEqual(determine_integration_order(i1_series), 1)
        self.assertEqual(determine_integration_order(i2_series), 2)


class TestCointegrationTester(unittest.TestCase):
    """Tests for the CointegrationTester class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = CointegrationTester()
        
        # Create cointegrated series
        np.random.seed(42)
        # Common trend
        common_trend = np.cumsum(np.random.normal(0, 1, 100))
        # Two series with the same trend but different noise
        self.series1 = common_trend + np.random.normal(0, 0.5, 100)
        self.series2 = 2 * common_trend + np.random.normal(0, 0.5, 100)
        
        # Create non-cointegrated series
        self.series3 = np.cumsum(np.random.normal(0, 1, 100))
        self.series4 = np.cumsum(np.random.normal(0, 1, 100))
    
    def test_engle_granger(self):
        """Test Engle-Granger cointegration test."""
        # Test cointegrated series
        result_co = self.tester.test_engle_granger(self.series1, self.series2)
        
        # Test non-cointegrated series
        result_nonco = self.tester.test_engle_granger(self.series3, self.series4)
        
        # Check expected outcomes
        self.assertTrue(result_co['cointegrated'])
        self.assertFalse(result_nonco['cointegrated'])
        
        # Check result structure
        for result in [result_co, result_nonco]:
            self.assertIn('statistic', result)
            self.assertIn('pvalue', result)
            self.assertIn('critical_values', result)
            self.assertIn('beta', result)
            self.assertIn('residuals', result)
    
    def test_johansen(self):
        """Test Johansen cointegration test."""
        # Create data matrix
        data_co = np.column_stack([self.series1, self.series2])
        data_nonco = np.column_stack([self.series3, self.series4])
        
        # Test cointegrated series
        result_co = self.tester.test_johansen(data_co)
        
        # Test non-cointegrated series
        result_nonco = self.tester.test_johansen(data_nonco)
        
        # Check expected outcomes
        self.assertTrue(result_co['cointegrated'])  # Should find cointegration
        self.assertFalse(result_nonco['cointegrated'])  # Should not find cointegration
        
        # Check result structure
        for result in [result_co, result_nonco]:
            self.assertIn('trace_statistics', result)
            self.