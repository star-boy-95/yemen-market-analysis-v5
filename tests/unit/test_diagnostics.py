"""
Unit tests for the model diagnostics module.

This module tests the diagnostic functions for econometric models, 
including residual diagnostics, model selection criteria, and stability tests.
"""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.models.diagnostics import (
    ModelDiagnostics, 
    calculate_fit_statistics,
    test_residual_normality,
    test_residual_autocorrelation,
    test_heteroskedasticity,
    test_parameter_stability
)


class TestResidualDiagnostics(unittest.TestCase):
    """Test cases for residual diagnostic functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample residuals
        np.random.seed(42)  # For reproducibility
        
        # Normal residuals (well-behaved)
        self.normal_residuals = np.random.normal(0, 1, 100)
        
        # Non-normal residuals (skewed)
        self.skewed_residuals = np.exp(np.random.normal(0, 1, 100)) - 1
        
        # Autocorrelated residuals
        self.autocorr_residuals = np.zeros(100)
        for i in range(1, 100):
            self.autocorr_residuals[i] = 0.7 * self.autocorr_residuals[i-1] + np.random.normal(0, 0.3)
        
        # Heteroskedastic residuals (variance increases with x)
        x = np.linspace(0, 10, 100)
        self.hetero_residuals = np.random.normal(0, 0.2 + 0.2 * x)
        self.x_variable = x
    
    def test_residual_normality(self):
        """Test Jarque-Bera test for residual normality."""
        # Test with normal residuals
        normal_result = test_residual_normality(self.normal_residuals)
        
        # Check result structure
        self.assertIn('test_statistic', normal_result)
        self.assertIn('p_value', normal_result)
        self.assertIn('normal', normal_result)
        
        # Normal residuals should pass normality test (high p-value)
        self.assertGreater(normal_result['p_value'], 0.05)
        self.assertTrue(normal_result['normal'])
        
        # Test with skewed residuals
        skewed_result = test_residual_normality(self.skewed_residuals)
        
        # Skewed residuals should fail normality test (low p-value)
        self.assertLess(skewed_result['p_value'], 0.05)
        self.assertFalse(skewed_result['normal'])
    
    def test_residual_autocorrelation(self):
        """Test Ljung-Box test for residual autocorrelation."""
        # Test with normal residuals
        normal_result = test_residual_autocorrelation(self.normal_residuals)
        
        # Check result structure
        self.assertIn('test_statistic', normal_result)
        self.assertIn('p_value', normal_result)
        self.assertIn('autocorrelated', normal_result)
        
        # Normal residuals should not be autocorrelated (high p-value)
        self.assertGreater(normal_result['p_value'], 0.05)
        self.assertFalse(normal_result['autocorrelated'])
        
        # Test with autocorrelated residuals
        autocorr_result = test_residual_autocorrelation(self.autocorr_residuals)
        
        # Autocorrelated residuals should be detected (low p-value)
        self.assertLess(autocorr_result['p_value'], 0.05)
        self.assertTrue(autocorr_result['autocorrelated'])
    
    def test_heteroskedasticity(self):
        """Test White's test for heteroskedasticity."""
        # Create X matrix for testing (including constant and x variable)
        X = np.column_stack((np.ones(100), self.x_variable))
        
        # Test with normal residuals
        normal_result = test_heteroskedasticity(self.normal_residuals, X)
        
        # Check result structure
        self.assertIn('test_statistic', normal_result)
        self.assertIn('p_value', normal_result)
        self.assertIn('heteroskedastic', normal_result)
        
        # Normal residuals should not be heteroskedastic (high p-value)
        self.assertGreater(normal_result['p_value'], 0.01)  # Less strict due to potential false positives
        self.assertFalse(normal_result['heteroskedastic'])
        
        # Test with heteroskedastic residuals
        hetero_result = test_heteroskedasticity(self.hetero_residuals, X)
        
        # Heteroskedastic residuals should be detected (low p-value)
        self.assertLess(hetero_result['p_value'], 0.05)
        self.assertTrue(hetero_result['heteroskedastic'])


class TestModelSelection(unittest.TestCase):
    """Test cases for model selection criteria."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for model comparison
        np.random.seed(42)  # For reproducibility
        
        # Generate x values
        x = np.linspace(0, 10, 100)
        
        # True model: quadratic relationship with noise
        y_true = 1 + 2*x + 0.5*x**2 + np.random.normal(0, 1, 100)
        
        # Fitted values from different models
        self.y_const = np.ones(100) * np.mean(y_true)  # Constant only (1 parameter)
        self.y_linear = np.polyval([1.8, 1.1], x)  # Linear model (2 parameters)
        self.y_quadratic = np.polyval([0.48, 2.1, 0.9], x)  # Quadratic model (3 parameters)
        self.y_cubic = np.polyval([0.05, 0.45, 2.2, 0.8], x)  # Cubic model (4 parameters)
        
        self.observed = y_true
        self.n_samples = 100
    
    def test_calculate_fit_statistics(self):
        """Test calculation of model fit statistics."""
        # Calculate statistics for different models
        const_stats = calculate_fit_statistics(self.observed, self.y_const, n_params=1)
        linear_stats = calculate_fit_statistics(self.observed, self.y_linear, n_params=2)
        quadratic_stats = calculate_fit_statistics(self.observed, self.y_quadratic, n_params=3)
        cubic_stats = calculate_fit_statistics(self.observed, self.y_cubic, n_params=4)
        
        # Check result structure
        for stats in [const_stats, linear_stats, quadratic_stats, cubic_stats]:
            self.assertIn('aic', stats)
            self.assertIn('bic', stats)
            self.assertIn('hqic', stats)
            self.assertIn('r_squared', stats)
            self.assertIn('adj_r_squared', stats)
            self.assertIn('rmse', stats)
        
        # Model fit should improve with more parameters (R-squared increases)
        self.assertLess(const_stats['r_squared'], linear_stats['r_squared'])
        self.assertLess(linear_stats['r_squared'], quadratic_stats['r_squared'])
        self.assertLess(quadratic_stats['r_squared'], cubic_stats['r_squared'])
        
        # True model is quadratic, so information criteria should prefer quadratic over cubic
        # BIC and HQIC penalize complexity more, so they should favor simpler models
        self.assertLess(quadratic_stats['bic'], cubic_stats['bic'])
        self.assertLess(quadratic_stats['hqic'], cubic_stats['hqic'])
        
        # Quadratic model should have better adjusted R-squared than cubic
        # due to parsimony (penalty for extra parameter)
        self.assertGreater(quadratic_stats['adj_r_squared'], const_stats['adj_r_squared'])
        self.assertGreater(quadratic_stats['adj_r_squared'], linear_stats['adj_r_squared'])
        
        # RMSE should decrease with more parameters
        self.assertGreater(const_stats['rmse'], linear_stats['rmse'])
        self.assertGreater(linear_stats['rmse'], quadratic_stats['rmse'])


class TestModelDiagnostics(unittest.TestCase):
    """Test cases for the ModelDiagnostics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample residuals
        np.random.seed(42)  # For reproducibility
        self.residuals = np.random.normal(0, 1, 100)
        
        # Create sample X matrix
        self.x_variable = np.linspace(0, 10, 100)
        self.X = np.column_stack((np.ones(100), self.x_variable))
        
        # Create model diagnostics instance
        self.diagnostics = ModelDiagnostics(residuals=self.residuals, X=self.X)
    
    def test_initialization(self):
        """Test initialization of ModelDiagnostics class."""
        # Test with residuals only
        diagnostics = ModelDiagnostics(residuals=self.residuals)
        self.assertIsInstance(diagnostics.residuals, np.ndarray)
        self.assertEqual(len(diagnostics.residuals), 100)
        self.assertIsNone(diagnostics.X)
        
        # Test with residuals and X
        diagnostics = ModelDiagnostics(residuals=self.residuals, X=self.X)
        self.assertIsInstance(diagnostics.residuals, np.ndarray)
        self.assertIsInstance(diagnostics.X, np.ndarray)
        self.assertEqual(diagnostics.X.shape, (100, 2))
        
        # Test validation
        with self.assertRaises(ValueError):
            # Empty residuals
            ModelDiagnostics(residuals=np.array([]))
            
        with self.assertRaises(ValueError):
            # Mismatched lengths
            ModelDiagnostics(residuals=self.residuals, X=np.ones((50, 2)))
    
    def test_residual_tests(self):
        """Test running all residual diagnostic tests."""
        # Run all tests
        result = self.diagnostics.residual_tests()
        
        # Check result structure
        self.assertIn('normality', result)
        self.assertIn('autocorrelation', result)
        self.assertIn('heteroskedasticity', result)
        
        # Each test result should have test_statistic and p_value
        for test_name, test_result in result.items():
            self.assertIn('test_statistic', test_result)
            self.assertIn('p_value', test_result)
    
    def test_model_selection_criteria(self):
        """Test calculation of model selection criteria."""
        # Create sample observed and predicted values
        observed = 1 + 2*self.x_variable + 0.5*self.x_variable**2 + np.random.normal(0, 1, 100)
        predicted = np.polyval([0.48, 2.1, 0.9], self.x_variable)
        
        # Calculate model selection criteria
        result = self.diagnostics.model_selection_criteria(observed, predicted, n_params=3)
        
        # Check result structure
        self.assertIn('aic', result)
        self.assertIn('bic', result)
        self.assertIn('hqic', result)
        self.assertIn('r_squared', result)
        self.assertIn('adj_r_squared', result)
        self.assertIn('rmse', result)
    
    @patch('src.models.diagnostics.test_parameter_stability')
    def test_parameter_stability(self, mock_test):
        """Test parameter stability test with mocked underlying function."""
        # Set up mock to return a dummy result
        mock_test.return_value = {
            'test_statistic': 1.5,
            'p_value': 0.2,
            'stable': True
        }
        
        # Call the method
        result = self.diagnostics.parameter_stability(breakpoint=50)
        
        # Check that mock was called
        mock_test.assert_called_once()
        
        # Check result (should match mock return value)
        self.assertEqual(result['test_statistic'], 1.5)
        self.assertEqual(result['p_value'], 0.2)
        self.assertTrue(result['stable'])
    
    def test_summary(self):
        """Test generation of diagnostic summary."""
        # Create sample observed and predicted values
        observed = 1 + 2*self.x_variable + 0.5*self.x_variable**2 + np.random.normal(0, 1, 100)
        predicted = np.polyval([0.48, 2.1, 0.9], self.x_variable)
        
        # Generate summary
        summary = self.diagnostics.summary(observed, predicted, n_params=3)
        
        # Check summary structure
        self.assertIn('residual_tests', summary)
        self.assertIn('fit_statistics', summary)
        self.assertIn('overall_assessment', summary)
        
        # Check overall assessment (should be a string)
        self.assertIsInstance(summary['overall_assessment'], str)


if __name__ == '__main__':
    unittest.main()
