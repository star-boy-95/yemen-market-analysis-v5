"""
Threshold Autoregressive (TAR) model module for Yemen Market Analysis.

This module provides the ThresholdAutoregressive class for estimating
TAR models for cointegrated time series.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data
from src.models.cointegration.engle_granger import EngleGrangerTester

# Initialize logger
logger = logging.getLogger(__name__)

class ThresholdAutoregressive:
    """
    Threshold Autoregressive (TAR) model for Yemen Market Analysis.
    
    This class provides methods for estimating TAR models for cointegrated time series.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        eg_tester (EngleGrangerTester): Engle-Granger test implementation.
    """
    
    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the TAR model.
        
        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.threshold.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.threshold.max_lags', 4)
        self.eg_tester = EngleGrangerTester(alpha=self.alpha, max_lags=self.max_lags)
    
    @handle_errors
    def estimate(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        fixed_threshold: Optional[float] = None, max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate a TAR model for cointegrated series.
        
        The TAR model is estimated as:
        Δz_t = ρ₁*z_{t-1}*I(z_{t-1} ≥ τ) + ρ₂*z_{t-1}*I(z_{t-1} < τ) + ∑ γ_i*Δz_{t-i} + ε_t
        
        where z_t is the residual from the cointegrating regression, τ is the threshold,
        and I() is the indicator function.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            fixed_threshold: Fixed threshold value. If None, the threshold is estimated.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the TAR model results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the estimation fails.
        """
        logger.info(f"Estimating TAR model for {y_col} and {x_col}")
        
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
                logger.warning(f"{y_col} and {x_col} are not cointegrated, TAR model may not be appropriate")
            
            # Get the residuals from the cointegrating regression
            residuals = coint_results['residuals']
            
            # Create lagged residuals and differences
            z_lag = residuals.shift(1)
            dz = residuals.diff()
            
            # Create lagged differences
            dz_lags = pd.DataFrame()
            for i in range(1, max_lags + 1):
                dz_lags[f'dz_lag_{i}'] = dz.shift(i)
            
            # Align the data
            common_index = dz.index.intersection(z_lag.index).intersection(dz_lags.index)
            dz = dz.loc[common_index]
            z_lag = z_lag.loc[common_index]
            dz_lags = dz_lags.loc[common_index]
            
            # If fixed threshold is provided, use it
            if fixed_threshold is not None:
                threshold = fixed_threshold
                logger.info(f"Using fixed threshold: {threshold}")
            else:
                # Estimate the threshold
                # Get trimming parameter from config
                trim = config.get('analysis.threshold.trim', 0.15)
                n_grid = config.get('analysis.threshold.n_grid', 300)
                
                # Sort residuals for grid search
                sorted_residuals = np.sort(z_lag)
                n = len(sorted_residuals)
                
                # Define grid points
                lower_idx = int(n * trim)
                upper_idx = int(n * (1 - trim))
                grid_points = sorted_residuals[lower_idx:upper_idx]
                
                # If grid is too large, subsample
                if len(grid_points) > n_grid:
                    step = len(grid_points) // n_grid
                    grid_points = grid_points[::step]
                
                # Initialize variables for grid search
                min_ssr = float('inf')
                best_threshold = None
                
                # Grid search for threshold
                for threshold_candidate in grid_points:
                    # Create indicator variables
                    above_threshold = (z_lag >= threshold_candidate).astype(int)
                    below_threshold = (z_lag < threshold_candidate).astype(int)
                    
                    # Create interaction terms
                    z_above = z_lag * above_threshold
                    z_below = z_lag * below_threshold
                    
                    # Create design matrix
                    X = pd.DataFrame({
                        'z_above': z_above,
                        'z_below': z_below
                    })
                    
                    # Add lagged differences
                    X = pd.concat([X, dz_lags], axis=1)
                    
                    # Estimate model
                    model = OLS(dz, sm.add_constant(X))
                    results = model.fit()
                    
                    # Update if SSR is lower
                    if results.ssr < min_ssr:
                        min_ssr = results.ssr
                        best_threshold = threshold_candidate
                
                threshold = best_threshold
                logger.info(f"Estimated threshold: {threshold}")
            
            # Create indicator variables with the best threshold
            above_threshold = (z_lag >= threshold).astype(int)
            below_threshold = (z_lag < threshold).astype(int)
            
            # Create interaction terms
            z_above = z_lag * above_threshold
            z_below = z_lag * below_threshold
            
            # Create design matrix
            X = pd.DataFrame({
                'z_above': z_above,
                'z_below': z_below
            })
            
            # Add lagged differences
            X = pd.concat([X, dz_lags], axis=1)
            
            # Estimate final model
            model = OLS(dz, sm.add_constant(X))
            results = model.fit()
            
            # Extract results
            rho_above = results.params[1]  # Adjustment coefficient above threshold
            rho_below = results.params[2]  # Adjustment coefficient below threshold
            
            # Test for threshold effect
            r_matrix = np.zeros((1, len(results.params)))
            r_matrix[0, 1] = 1
            r_matrix[0, 2] = -1
            
            wald_test = results.f_test(r_matrix)
            
            # Create results dictionary
            tar_results = {
                'model': 'TAR',
                'threshold': threshold,
                'fixed_threshold': fixed_threshold is not None,
                'params': {
                    'rho_above': rho_above,
                    'rho_below': rho_below,
                    'constant': results.params[0],
                    'lag_coefficients': results.params[3:].tolist(),
                },
                'std_errors': {
                    'rho_above': results.bse[1],
                    'rho_below': results.bse[2],
                    'constant': results.bse[0],
                    'lag_coefficients': results.bse[3:].tolist(),
                },
                'p_values': {
                    'rho_above': results.pvalues[1],
                    'rho_below': results.pvalues[2],
                    'constant': results.pvalues[0],
                    'lag_coefficients': results.pvalues[3:].tolist(),
                },
                'threshold_test': {
                    'f_statistic': wald_test.fvalue,
                    'p_value': wald_test.pvalue,
                    'is_threshold_significant': wald_test.pvalue < self.alpha,
                },
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'aic': results.aic,
                'bic': results.bic,
                'residuals': results.resid,
                'n_obs': len(dz),
                'cointegration_results': coint_results,
            }
            
            logger.info(f"TAR model results: rho_above={rho_above:.4f}, rho_below={rho_below:.4f}, threshold_p_value={wald_test.pvalue:.4f}")
            return tar_results
        except Exception as e:
            logger.error(f"Error estimating TAR model: {e}")
            raise YemenAnalysisError(f"Error estimating TAR model: {e}")
    
    @handle_errors
    def bootstrap_threshold_test(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        max_lags: Optional[int] = None, n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform bootstrap test for threshold effect.
        
        This method implements a bootstrap test for the significance of the threshold
        effect in the TAR model.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            n_bootstrap: Number of bootstrap replications.
            
        Returns:
            Dictionary containing the bootstrap test results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the test fails.
        """
        logger.info(f"Performing bootstrap test for threshold effect with {n_bootstrap} replications")
        
        # First, estimate the TAR model
        tar_results = self.estimate(y, x, y_col, x_col, max_lags=max_lags)
        
        # Extract the test statistic
        f_statistic = tar_results['threshold_test']['f_statistic']
        
        try:
            # Get the residuals from the cointegrating regression
            residuals = tar_results['cointegration_results']['residuals']
            
            # Create lagged residuals and differences
            z_lag = residuals.shift(1)
            dz = residuals.diff()
            
            # Create lagged differences
            dz_lags = pd.DataFrame()
            for i in range(1, max_lags + 1):
                dz_lags[f'dz_lag_{i}'] = dz.shift(i)
            
            # Align the data
            common_index = dz.index.intersection(z_lag.index).intersection(dz_lags.index)
            dz = dz.loc[common_index]
            z_lag = z_lag.loc[common_index]
            dz_lags = dz_lags.loc[common_index]
            
            # Estimate the linear model (no threshold)
            X_linear = pd.DataFrame({'z_lag': z_lag})
            X_linear = pd.concat([X_linear, dz_lags], axis=1)
            
            linear_model = OLS(dz, sm.add_constant(X_linear))
            linear_results = linear_model.fit()
            
            # Get residuals from the linear model
            linear_residuals = linear_results.resid
            
            # Initialize bootstrap distribution
            bootstrap_f_stats = np.zeros(n_bootstrap)
            
            # Bootstrap loop
            for i in range(n_bootstrap):
                # Generate bootstrap sample
                bootstrap_residuals = np.random.choice(linear_residuals, size=len(linear_residuals))
                bootstrap_dz = linear_results.predict() + bootstrap_residuals
                
                # Estimate TAR model on bootstrap sample
                # Create indicator variables with the threshold from the original model
                threshold = tar_results['threshold']
                above_threshold = (z_lag >= threshold).astype(int)
                below_threshold = (z_lag < threshold).astype(int)
                
                # Create interaction terms
                z_above = z_lag * above_threshold
                z_below = z_lag * below_threshold
                
                # Create design matrix
                X_tar = pd.DataFrame({
                    'z_above': z_above,
                    'z_below': z_below
                })
                
                # Add lagged differences
                X_tar = pd.concat([X_tar, dz_lags], axis=1)
                
                # Estimate TAR model
                tar_model = OLS(bootstrap_dz, sm.add_constant(X_tar))
                tar_bootstrap_results = tar_model.fit()
                
                # Estimate linear model
                linear_model = OLS(bootstrap_dz, sm.add_constant(X_linear))
                linear_bootstrap_results = linear_model.fit()
                
                # Compute F-statistic
                ssr_linear = linear_bootstrap_results.ssr
                ssr_tar = tar_bootstrap_results.ssr
                
                f_stat = ((ssr_linear - ssr_tar) / 1) / (ssr_tar / (len(bootstrap_dz) - len(tar_bootstrap_results.params)))
                bootstrap_f_stats[i] = f_stat
            
            # Compute bootstrap p-value
            bootstrap_p_value = np.mean(bootstrap_f_stats > f_statistic)
            
            # Create results dictionary
            bootstrap_results = {
                'test': 'Bootstrap Threshold Test',
                'f_statistic': f_statistic,
                'bootstrap_p_value': bootstrap_p_value,
                'is_threshold_significant': bootstrap_p_value < self.alpha,
                'n_bootstrap': n_bootstrap,
                'bootstrap_distribution': bootstrap_f_stats,
                'alpha': self.alpha,
            }
            
            logger.info(f"Bootstrap test results: f_statistic={f_statistic:.4f}, bootstrap_p_value={bootstrap_p_value:.4f}")
            return bootstrap_results
        except Exception as e:
            logger.error(f"Error performing bootstrap test: {e}")
            raise YemenAnalysisError(f"Error performing bootstrap test: {e}")
