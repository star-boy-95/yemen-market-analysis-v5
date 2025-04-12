"""
Engle-Granger cointegration test module for Yemen Market Analysis.

This module provides the EngleGrangerTester class for testing cointegration
using the Engle-Granger two-step procedure.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class EngleGrangerTester:
    """
    Engle-Granger cointegration tester for Yemen Market Analysis.
    
    This class provides methods for testing cointegration using the Engle-Granger
    two-step procedure.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
    """
    
    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the Engle-Granger tester.
        
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
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        trend: str = 'c', max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Engle-Granger test for cointegration.
        
        The Engle-Granger test is a two-step procedure:
        1. Estimate the cointegrating relationship: y = a + b*x + e
        2. Test the residuals for stationarity using an ADF test
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            trend: Trend to include in the test. Options are 'c' (constant),
                  'ct' (constant and trend), 'ctt' (constant, linear and quadratic trend),
                  and 'n' (no trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the test fails.
        """
        logger.info(f"Performing Engle-Granger test with trend={trend}")
        
        # Check if columns exist
        if y_col not in y.columns:
            logger.error(f"Column {y_col} not found in y data")
            raise YemenAnalysisError(f"Column {y_col} not found in y data")
        
        if x_col not in x.columns:
            logger.error(f"Column {x_col} not found in x data")
            raise YemenAnalysisError(f"Column {x_col} not found in x data")
        
        # Get column data
        y_data = y[y_col].dropna()
        x_data = x[x_col].dropna()
        
        # Ensure the series have the same length
        common_index = y_data.index.intersection(x_data.index)
        y_data = y_data.loc[common_index]
        x_data = x_data.loc[common_index]
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
        
        try:
            # Perform Engle-Granger test
            coint_result = coint(y_data, x_data, trend=trend, maxlag=max_lags, autolag=None)
            
            # Extract results
            test_statistic = coint_result[0]
            p_value = coint_result[1]
            critical_values = coint_result[2]
            
            # Determine if the series are cointegrated
            is_cointegrated = p_value < self.alpha
            
            # Estimate the cointegrating relationship
            model = OLS(y_data, sm.add_constant(x_data))
            results = model.fit()
            
            # Get the residuals
            residuals = results.resid
            
            # Create results dictionary
            results_dict = {
                'test': 'Engle-Granger',
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': {
                    '1%': critical_values[0],
                    '5%': critical_values[1],
                    '10%': critical_values[2],
                },
                'is_cointegrated': is_cointegrated,
                'alpha': self.alpha,
                'cointegrating_vector': {
                    'constant': results.params[0],
                    'coefficient': results.params[1],
                },
                'residuals': residuals,
                'n_obs': len(y_data),
            }
            
            logger.info(f"Engle-Granger test results: test_statistic={test_statistic:.4f}, p_value={p_value:.4f}, is_cointegrated={is_cointegrated}")
            return results_dict
        except Exception as e:
            logger.error(f"Error performing Engle-Granger test: {e}")
            raise YemenAnalysisError(f"Error performing Engle-Granger test: {e}")
