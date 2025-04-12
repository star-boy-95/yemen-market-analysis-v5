"""
Error Correction Model module for Yemen Market Analysis.

This module provides the ErrorCorrectionModel class for estimating
error correction models for cointegrated series.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data
from src.models.cointegration.engle_granger import EngleGrangerTester

# Initialize logger
logger = logging.getLogger(__name__)

class ErrorCorrectionModel:
    """
    Error Correction Model for Yemen Market Analysis.
    
    This class provides methods for estimating error correction models for
    cointegrated series.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        eg_tester (EngleGrangerTester): Engle-Granger test implementation.
    """
    
    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the Error Correction Model.
        
        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.cointegration.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.cointegration.max_lags', 4)
        self.eg_tester = EngleGrangerTester(alpha=self.alpha, max_lags=self.max_lags)
    
    @handle_errors
    def estimate(
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
            coint_results = self.eg_tester.test(y, x, y_col, x_col, 'c', max_lags)
            
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
    
    @handle_errors
    def estimate_asymmetric(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate an asymmetric error correction model (AECM) for cointegrated series.
        
        The AECM is estimated as:
        Δy_t = α + β*Δx_t + γ⁺*z⁺_{t-1} + γ⁻*z⁻_{t-1} + ε_t
        
        where z⁺_{t-1} and z⁻_{t-1} are the positive and negative parts of the
        lagged residual from the cointegrating regression.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the AECM results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the estimation fails.
        """
        logger.info(f"Estimating asymmetric error correction model for {y_col} and {x_col}")
        
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
            coint_results = self.eg_tester.test(y, x, y_col, x_col, 'c', max_lags)
            
            if not coint_results['is_cointegrated']:
                logger.warning(f"{y_col} and {x_col} are not cointegrated, AECM may not be appropriate")
            
            # Get the residuals from the cointegrating regression
            residuals = coint_results['residuals']
            
            # Create first differences
            dy = y_data.diff().dropna()
            dx = x_data.diff().dropna()
            
            # Create positive and negative parts of the residuals
            z_lag = residuals.shift(1)
            z_pos = z_lag.copy()
            z_pos[z_pos < 0] = 0
            z_neg = z_lag.copy()
            z_neg[z_neg > 0] = 0
            
            # Align the data
            common_index = dy.index.intersection(dx.index).intersection(z_lag.index)
            dy = dy.loc[common_index]
            dx = dx.loc[common_index]
            z_pos = z_pos.loc[common_index]
            z_neg = z_neg.loc[common_index]
            
            # Estimate the AECM
            X = sm.add_constant(pd.DataFrame({
                'dx': dx,
                'z_pos': z_pos,
                'z_neg': z_neg
            }))
            model = OLS(dy, X)
            results = model.fit()
            
            # Extract results
            alpha = results.params[0]    # Constant
            beta = results.params[1]     # Short-run effect
            gamma_pos = results.params[2]  # Speed of adjustment (positive)
            gamma_neg = results.params[3]  # Speed of adjustment (negative)
            
            # Test for asymmetry
            from scipy import stats
            
            # Wald test for gamma_pos = gamma_neg
            r_matrix = np.array([[0, 0, 1, -1]])  # Test gamma_pos = gamma_neg
            wald_test = results.f_test(r_matrix)
            
            # Create results dictionary
            aecm_results = {
                'model': 'Asymmetric Error Correction Model',
                'params': {
                    'alpha': alpha,
                    'beta': beta,
                    'gamma_pos': gamma_pos,
                    'gamma_neg': gamma_neg,
                },
                'std_errors': {
                    'alpha': results.bse[0],
                    'beta': results.bse[1],
                    'gamma_pos': results.bse[2],
                    'gamma_neg': results.bse[3],
                },
                'p_values': {
                    'alpha': results.pvalues[0],
                    'beta': results.pvalues[1],
                    'gamma_pos': results.pvalues[2],
                    'gamma_neg': results.pvalues[3],
                },
                'asymmetry_test': {
                    'f_statistic': wald_test.fvalue,
                    'p_value': wald_test.pvalue,
                    'is_asymmetric': wald_test.pvalue < self.alpha,
                },
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'aic': results.aic,
                'bic': results.bic,
                'residuals': results.resid,
                'n_obs': len(dy),
                'cointegration_results': coint_results,
            }
            
            logger.info(f"AECM results: gamma_pos={gamma_pos:.4f}, gamma_neg={gamma_neg:.4f}, asymmetry_p_value={wald_test.pvalue:.4f}")
            return aecm_results
        except Exception as e:
            logger.error(f"Error estimating asymmetric error correction model: {e}")
            raise YemenAnalysisError(f"Error estimating asymmetric error correction model: {e}")
