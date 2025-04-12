"""
Cointegration testing module for Yemen Market Analysis.

This module provides functions for testing cointegration relationships between time series.
It includes implementations of various cointegration tests, including Engle-Granger,
Johansen, and Gregory-Hansen tests.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.regression.linear_model import OLS

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class CointegrationTester:
    """
    Cointegration tester for Yemen Market Analysis.
    
    This class provides methods for testing cointegration relationships between time series.
    It includes implementations of various cointegration tests, including Engle-Granger,
    Johansen, and Gregory-Hansen tests.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
    """
    
    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the cointegration tester.
        
        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.cointegration.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.cointegration.max_lags', 4)
    
    @handle_errors
    def test_engle_granger(
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
    
    @handle_errors
    def test_johansen(
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
        
        try:
            # Perform Johansen test
            vecm_model = VECM(data_subset, k_ar_diff=k_ar_diff, deterministic=det_order)
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
    
    @handle_errors
    def test_gregory_hansen(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        model: str = 'c', max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Gregory-Hansen test for cointegration with structural breaks.
        
        The Gregory-Hansen test extends the Engle-Granger test to allow for a structural
        break in the cointegrating relationship.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            model: Model type. Options are 'c' (level shift), 'ct' (level shift with trend),
                  and 'cshift' (regime shift).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the test fails.
        """
        logger.info(f"Performing Gregory-Hansen test with model={model}")
        
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
            # Implement Gregory-Hansen test
            # This is a simplified implementation that searches for a break point
            # and performs the Engle-Granger test at each potential break point
            
            n = len(y_data)
            min_size = int(0.15 * n)  # Trimming parameter
            
            # Initialize variables to store results
            min_adf = float('inf')
            break_idx = None
            break_date = None
            
            # Loop through potential break points
            for i in range(min_size, n - min_size):
                # Create dummy variable for break
                dummy = np.zeros(n)
                dummy[i:] = 1
                
                # Create interaction term for regime shift model
                if model == 'cshift':
                    x_with_break = pd.DataFrame({
                        'x': x_data,
                        'dummy': dummy,
                        'x_dummy': x_data * dummy
                    })
                    # Estimate model with regime shift
                    reg = OLS(y_data, sm.add_constant(x_with_break)).fit()
                elif model == 'ct':
                    # Create trend variable
                    trend = np.arange(n)
                    x_with_break = pd.DataFrame({
                        'x': x_data,
                        'trend': trend,
                        'dummy': dummy
                    })
                    # Estimate model with level shift and trend
                    reg = OLS(y_data, sm.add_constant(x_with_break)).fit()
                else:  # model == 'c'
                    x_with_break = pd.DataFrame({
                        'x': x_data,
                        'dummy': dummy
                    })
                    # Estimate model with level shift
                    reg = OLS(y_data, sm.add_constant(x_with_break)).fit()
                
                # Get residuals and perform ADF test
                resid = reg.resid
                adf_result = adfuller(resid, maxlag=max_lags, regression='c', autolag=None)
                
                # Update minimum ADF statistic
                if adf_result[0] < min_adf:
                    min_adf = adf_result[0]
                    break_idx = i
                    break_date = y_data.index[i]
                    best_reg = reg
            
            # Get critical values for Gregory-Hansen test
            # These are different from standard ADF critical values
            gh_critical_values = {
                'c': {
                    '1%': -5.13,
                    '5%': -4.61,
                    '10%': -4.34
                },
                'ct': {
                    '1%': -5.45,
                    '5%': -4.99,
                    '10%': -4.72
                },
                'cshift': {
                    '1%': -5.47,
                    '5%': -4.95,
                    '10%': -4.68
                }
            }
            
            # Determine if the series are cointegrated
            critical_value_5pct = gh_critical_values[model]['5%']
            is_cointegrated = min_adf < critical_value_5pct
            
            # Create results dictionary
            results = {
                'test': 'Gregory-Hansen',
                'model': model,
                'test_statistic': min_adf,
                'critical_values': gh_critical_values[model],
                'is_cointegrated': is_cointegrated,
                'break_index': break_idx,
                'break_date': break_date,
                'alpha': self.alpha,
                'cointegrating_vector': dict(zip(best_reg.params.index, best_reg.params)),
                'residuals': best_reg.resid,
                'n_obs': n,
            }
            
            logger.info(f"Gregory-Hansen test results: test_statistic={min_adf:.4f}, is_cointegrated={is_cointegrated}, break_date={break_date}")
            return results
        except Exception as e:
            logger.error(f"Error performing Gregory-Hansen test: {e}")
            raise YemenAnalysisError(f"Error performing Gregory-Hansen test: {e}")
    
    @handle_errors
    def run_all_tests(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        trend: str = 'c', max_lags: Optional[int] = None, include_johansen: bool = True,
        include_gh: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all cointegration tests.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            trend: Trend to include in the tests.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            include_johansen: Whether to include the Johansen test.
            include_gh: Whether to include the Gregory-Hansen test.
            
        Returns:
            Dictionary mapping test names to test results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or any of the tests fail.
        """
        logger.info(f"Running all cointegration tests for {y_col} and {x_col}")
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
        
        # Run tests
        results = {}
        
        # Engle-Granger test
        eg_results = self.test_engle_granger(y, x, y_col, x_col, trend, max_lags)
        results['Engle-Granger'] = eg_results
        
        # Johansen test
        if include_johansen:
            # Combine data for Johansen test
            combined_data = pd.DataFrame({
                'y': y[y_col],
                'x': x[x_col]
            })
            
            # Determine deterministic order based on trend
            det_order = 0 if trend == 'n' else 1 if trend == 'c' else 2
            
            johansen_results = self.test_johansen(combined_data, ['y', 'x'], det_order, 1, max_lags)
            results['Johansen'] = johansen_results
        
        # Gregory-Hansen test
        if include_gh:
            gh_results = self.test_gregory_hansen(y, x, y_col, x_col, trend, max_lags)
            results['Gregory-Hansen'] = gh_results
        
        # Determine overall cointegration
        is_cointegrated = results['Engle-Granger']['is_cointegrated']
        
        if include_johansen:
            is_cointegrated = is_cointegrated or results['Johansen']['is_cointegrated']
        
        if include_gh:
            is_cointegrated = is_cointegrated or results['Gregory-Hansen']['is_cointegrated']
        
        # Add overall result
        results['overall'] = {
            'is_cointegrated': is_cointegrated,
            'alpha': self.alpha,
        }
        
        logger.info(f"Overall cointegration result: {is_cointegrated}")
        return results
    
    @handle_errors
    def estimate_error_correction_model(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate an error correction model (ECM) for cointegrated series.
        
        The ECM is estimated as:
        Δy_t = α + β*Δx_t + γ*z_{t-1} + ε_t
        
        where z_{t-1} is the lagged residual from the cointegrating regression.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the ECM results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the estimation fails.
        """
        logger.info(f"Estimating error correction model for {y_col} and {x_col}")
        
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
            # First, test for cointegration
            coint_results = self.test_engle_granger(y, x, y_col, x_col, 'c', max_lags)
            
            if not coint_results['is_cointegrated']:
                logger.warning(f"{y_col} and {x_col} are not cointegrated, ECM may not be appropriate")
            
            # Get the residuals from the cointegrating regression
            residuals = coint_results['residuals']
            
            # Create first differences
            dy = y_data.diff().dropna()
            dx = x_data.diff().dropna()
            
            # Align the data
            common_index = dy.index.intersection(dx.index).intersection(residuals.index[:-1])
            dy = dy.loc[common_index]
            dx = dx.loc[common_index]
            z_lag = residuals.shift(1).loc[common_index]
            
            # Estimate the ECM
            X = sm.add_constant(pd.DataFrame({'dx': dx, 'z_lag': z_lag}))
            model = OLS(dy, X)
            results = model.fit()
            
            # Extract results
            alpha = results.params[0]  # Constant
            beta = results.params[1]   # Short-run effect
            gamma = results.params[2]  # Speed of adjustment
            
            # Create results dictionary
            ecm_results = {
                'model': 'Error Correction Model',
                'params': {
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                },
                'std_errors': {
                    'alpha': results.bse[0],
                    'beta': results.bse[1],
                    'gamma': results.bse[2],
                },
                'p_values': {
                    'alpha': results.pvalues[0],
                    'beta': results.pvalues[1],
                    'gamma': results.pvalues[2],
                },
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'aic': results.aic,
                'bic': results.bic,
                'residuals': results.resid,
                'n_obs': len(dy),
                'cointegration_results': coint_results,
            }
            
            logger.info(f"ECM results: gamma={gamma:.4f} (speed of adjustment), beta={beta:.4f} (short-run effect)")
            return ecm_results
        except Exception as e:
            logger.error(f"Error estimating error correction model: {e}")
            raise YemenAnalysisError(f"Error estimating error correction model: {e}")
