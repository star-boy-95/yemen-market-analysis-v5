"""
Johansen cointegration test module for Yemen Market Analysis.

This module provides the JohansenTester class for testing cointegration
using the Johansen procedure.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class JohansenTester:
    """
    Johansen cointegration tester for Yemen Market Analysis.
    
    This class provides methods for testing cointegration using the Johansen
    procedure, which can detect multiple cointegrating relationships.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
    """
    
    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the Johansen tester.
        
        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.cointegration.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.cointegration.max_lags', 4)
    
    @handle_errors
    def test(
        self, data: pd.DataFrame, columns: List[str], det_order: int = 0,
        k_ar_diff: int = 1, max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Johansen test for cointegration.
        
        The Johansen test is a multivariate generalization of the ADF test that can
        detect multiple cointegrating relationships.
        
        Args:
            data: DataFrame containing the data.
            columns: List of column names to include in the test.
            det_order: Deterministic order. 0: no deterministic terms, 1: constant,
                      2: constant and trend.
            k_ar_diff: Number of lagged differences in the VECM.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the test fails.
        """
        logger.info(f"Performing Johansen test with det_order={det_order}, k_ar_diff={k_ar_diff}")
        
        # Check if columns exist
        for column in columns:
            if column not in data.columns:
                logger.error(f"Column {column} not found in data")
                raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        data_subset = data[columns].dropna()
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
            
        # Handle small sample sizes
        n_obs = len(data_subset)
        min_required_obs = 4 * len(columns) * (k_ar_diff + 1)  # Rough estimate of minimum sample size
        
        if n_obs < min_required_obs:
            logger.warning(f"Sample size ({n_obs}) too small for Johansen test. Needs at least {min_required_obs}. Returning mock results.")
            return self._mock_johansen_results(columns, n_obs, det_order, k_ar_diff)
            
        # Convert deterministic order to string format for statsmodels
        det_type = 'n'
        if det_order == 1:
            det_type = 'co'
        elif det_order == 2:
            det_type = 'ci'
        elif det_order == 3:
            det_type = 'lo'
        elif det_order == 4:
            det_type = 'li'
        
        try:
            # Perform Johansen test
            vecm_model = VECM(data_subset, k_ar_diff=k_ar_diff, deterministic=det_type)
            vecm_results = vecm_model.fit()
            
            # Get cointegration rank test results
            trace_results = vecm_results.test_coint_johansen(method='trace')
            eigen_results = vecm_results.test_coint_johansen(method='maxeig')
            
            # Determine cointegration rank
            trace_rank = sum(trace_results.pvalue < self.alpha)
            eigen_rank = sum(eigen_results.pvalue < self.alpha)
            
            # Determine if the series are cointegrated
            is_cointegrated = trace_rank > 0 or eigen_rank > 0
            
            # Create results dictionary
            results = {
                'test': 'Johansen',
                'det_order': det_order,
                'k_ar_diff': k_ar_diff,
                'trace_results': {
                    'test_statistics': trace_results.test_stats,
                    'critical_values': trace_results.crit_vals,
                    'p_values': trace_results.pvalue,
                    'rank': trace_rank,
                },
                'eigen_results': {
                    'test_statistics': eigen_results.test_stats,
                    'critical_values': eigen_results.crit_vals,
                    'p_values': eigen_results.pvalue,
                    'rank': eigen_rank,
                },
                'is_cointegrated': is_cointegrated,
                'alpha': self.alpha,
                'cointegrating_vectors': vecm_results.beta,
                'loading_matrix': vecm_results.alpha,
                'n_obs': len(data_subset),
            }
            
            logger.info(f"Johansen test results: trace_rank={trace_rank}, eigen_rank={eigen_rank}, is_cointegrated={is_cointegrated}")
            return results
        except Exception as e:
            logger.error(f"Error performing Johansen test: {e}")
            raise YemenAnalysisError(f"Error performing Johansen test: {e}")
            
    def _mock_johansen_results(self, columns: List[str], n_obs: int, det_order: int, k_ar_diff: int) -> Dict[str, Any]:
        """
        Create mock Johansen test results for small sample sizes.
        
        Args:
            columns: List of column names that were tested.
            n_obs: Number of observations.
            det_order: Deterministic order that was used.
            k_ar_diff: Number of lagged differences that was used.
            
        Returns:
            Dictionary containing mock Johansen test results.
        """
        # Create mock test statistics and p-values for trace and eigenvalue tests
        n_cols = len(columns)
        mock_test_stats = np.array([10.0, 2.0]) if n_cols == 2 else np.ones(n_cols) * 5.0
        mock_crit_vals = np.array([[15.0, 3.8], [20.0, 6.5], [24.0, 12.0]]) if n_cols == 2 else np.ones((3, n_cols)) * 15.0
        mock_pvalues = np.array([0.2, 0.5]) if n_cols == 2 else np.ones(n_cols) * 0.3
        
        # Create mock cointegrating vectors and loading matrix
        mock_beta = np.eye(n_cols) if n_cols > 1 else np.array([[1.0]])
        mock_alpha = np.ones((n_cols, n_cols)) * 0.1 if n_cols > 1 else np.array([[0.1]])
        
        return {
            'test': 'Johansen',
            'det_order': det_order,
            'k_ar_diff': k_ar_diff,
            'trace_results': {
                'test_statistics': mock_test_stats,
                'critical_values': mock_crit_vals,
                'p_values': mock_pvalues,
                'rank': 0,  # Assuming no cointegration for safety
            },
            'eigen_results': {
                'test_statistics': mock_test_stats,
                'critical_values': mock_crit_vals,
                'p_values': mock_pvalues,
                'rank': 0,  # Assuming no cointegration for safety
            },
            'is_cointegrated': False,  # Assuming no cointegration for safety
            'alpha': self.alpha,
            'cointegrating_vectors': mock_beta,
            'loading_matrix': mock_alpha,
            'n_obs': n_obs,
            'mock_result': True  # Flag to indicate this is a mock result
        }
