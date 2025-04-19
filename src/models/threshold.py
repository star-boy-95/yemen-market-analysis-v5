"""
Threshold modeling module for Yemen Market Analysis.

This module provides the ThresholdCointegration class for estimating
threshold cointegration models for time series data.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
import matplotlib.pyplot as plt

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data
from src.models.cointegration.engle_granger import EngleGrangerTester

# Initialize logger
logger = logging.getLogger(__name__)

class ThresholdCointegration:
    """
    Threshold cointegration model for Yemen Market Analysis.
    
    This class provides methods for estimating threshold cointegration models for time series data,
    including TAR and M-TAR models.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        trim (float): Trimming parameter for threshold estimation.
        n_grid (int): Number of grid points for threshold estimation.
        n_bootstrap (int): Number of bootstrap replications for hypothesis tests.
        eg_tester (EngleGrangerTester): Engle-Granger test implementation.
    """
    
    def __init__(
        self, alpha: float = None, max_lags: int = None,
        trim: float = None, n_grid: int = None, n_bootstrap: int = None
    ):
        """
        Initialize the threshold cointegration model.
        
        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
            trim: Trimming parameter for threshold estimation. If None, uses the value
                 from config.
            n_grid: Number of grid points for threshold estimation. If None, uses the
                   value from config.
            n_bootstrap: Number of bootstrap replications for hypothesis tests. If None,
                        uses the value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.threshold.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.threshold.max_lags', 4)
        self.trim = trim if trim is not None else config.get('analysis.threshold.trim', 0.15)
        self.n_grid = n_grid if n_grid is not None else config.get('analysis.threshold.n_grid', 300)
        self.n_bootstrap = n_bootstrap if n_bootstrap is not None else config.get('analysis.threshold.n_bootstrap', 1000)
        self.eg_tester = EngleGrangerTester(alpha=self.alpha, max_lags=self.max_lags)
    
    @handle_errors
    def estimate_cointegration(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        trend: str = 'c', max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate cointegrating relationship.
        
        This method estimates the cointegrating relationship between y and x using the
        Engle-Granger test.
        
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
            Dictionary containing the cointegration results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the estimation fails.
        """
        logger.info(f"Estimating cointegrating relationship for {y_col} and {x_col}")
        
        # Use Engle-Granger test to estimate cointegrating relationship
        return self.eg_tester.test(y, x, y_col, x_col, trend, max_lags)
    
    @handle_errors
    def estimate_threshold(
        self, residuals: pd.Series, model_type: str = 'tar',
        fixed_threshold: Optional[float] = None, max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate a threshold autoregressive model for residuals.
        
        Args:
            residuals: Residuals from the cointegrating regression.
            model_type: Type of threshold model. Options are 'tar' and 'mtar'.
            fixed_threshold: Fixed threshold value. If None, the threshold is estimated.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the threshold model results.
            
        Raises:
            YemenAnalysisError: If the estimation fails or the model type is invalid.
        """
        logger.info(f"Estimating {model_type.upper()} model for residuals")
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
        
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
        
        # Create momentum term if model_type is 'mtar'
        if model_type == 'mtar':
            dz_lag = dz.shift(1).loc[common_index]
        
        try:
            # If fixed threshold is provided, use it
            if fixed_threshold is not None:
                threshold = fixed_threshold
                logger.info(f"Using fixed threshold: {threshold}")
            else:
                # Estimate the threshold
                # Get trimming parameter from class
                trim = self.trim
                n_grid = self.n_grid
                
                # Sort values for grid search
                if model_type == 'tar':
                    sorted_values = np.sort(z_lag)
                else:  # model_type == 'mtar'
                    sorted_values = np.sort(dz_lag)
                
                n = len(sorted_values)
                
                # Define grid points
                lower_idx = int(n * trim)
                upper_idx = int(n * (1 - trim))
                grid_points = sorted_values[lower_idx:upper_idx]
                
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
                    if model_type == 'tar':
                        above_threshold = (z_lag >= threshold_candidate).astype(int)
                        below_threshold = (z_lag < threshold_candidate).astype(int)
                    else:  # model_type == 'mtar'
                        above_threshold = (dz_lag >= threshold_candidate).astype(int)
                        below_threshold = (dz_lag < threshold_candidate).astype(int)
                    
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
            if model_type == 'tar':
                above_threshold = (z_lag >= threshold).astype(int)
                below_threshold = (z_lag < threshold).astype(int)
            else:  # model_type == 'mtar'
                above_threshold = (dz_lag >= threshold).astype(int)
                below_threshold = (dz_lag < threshold).astype(int)
            
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
            # Null hypothesis: rho_above = rho_below
            r_matrix = np.zeros((1, len(results.params)))
            r_matrix[0, 1] = 1
            r_matrix[0, 2] = -1
            
            wald_test = results.f_test(r_matrix)
            
            # Create results dictionary
            threshold_results = {
                'model': model_type.upper(),
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
                'max_lags': max_lags,
                'trim': self.trim,
            }
            
            logger.info(f"{model_type.upper()} model results: rho_above={rho_above:.4f}, rho_below={rho_below:.4f}, threshold_p_value={wald_test.pvalue:.4f}")
            return threshold_results
        except Exception as e:
            logger.error(f"Error estimating {model_type.upper()} model: {e}")
            raise YemenAnalysisError(f"Error estimating {model_type.upper()} model: {e}")
    
    @handle_errors
    def bootstrap_threshold_test(
        self, residuals: pd.Series, model_type: str = 'tar',
        threshold: Optional[float] = None, max_lags: Optional[int] = None,
        n_bootstrap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform bootstrap test for threshold effect.
        
        Args:
            residuals: Residuals from the cointegrating regression.
            model_type: Type of threshold model. Options are 'tar' and 'mtar'.
            threshold: Threshold value. If None, uses the estimated threshold.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            n_bootstrap: Number of bootstrap replications. If None, uses the value
                        from the class.
            
        Returns:
            Dictionary containing the bootstrap test results.
            
        Raises:
            YemenAnalysisError: If the test fails or the model type is invalid.
        """
        logger.info(f"Performing bootstrap test for threshold effect in {model_type.upper()} model")
        
        # Set max_lags and n_bootstrap
        if max_lags is None:
            max_lags = self.max_lags
        
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
        
        # First, estimate the threshold model to get the test statistic
        threshold_results = self.estimate_threshold(residuals, model_type, threshold, max_lags)
        
        # Extract the test statistic
        f_statistic = threshold_results['threshold_test']['f_statistic']
        threshold_value = threshold_results['threshold']
        
        try:
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
            
            # Create momentum term if model_type is 'mtar'
            if model_type == 'mtar':
                dz_lag = dz.shift(1).loc[common_index]
            
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
                
                # Estimate threshold model on bootstrap sample
                # Create indicator variables with the threshold from the original model
                if model_type == 'tar':
                    above_threshold = (z_lag >= threshold_value).astype(int)
                    below_threshold = (z_lag < threshold_value).astype(int)
                else:  # model_type == 'mtar'
                    above_threshold = (dz_lag >= threshold_value).astype(int)
                    below_threshold = (dz_lag < threshold_value).astype(int)
                
                # Create interaction terms
                z_above = z_lag * above_threshold
                z_below = z_lag * below_threshold
                
                # Create design matrix
                X_threshold = pd.DataFrame({
                    'z_above': z_above,
                    'z_below': z_below
                })
                
                # Add lagged differences
                X_threshold = pd.concat([X_threshold, dz_lags], axis=1)
                
                # Estimate threshold model
                threshold_model = OLS(bootstrap_dz, sm.add_constant(X_threshold))
                threshold_bootstrap_results = threshold_model.fit()
                
                # Estimate linear model
                linear_model = OLS(bootstrap_dz, sm.add_constant(X_linear))
                linear_bootstrap_results = linear_model.fit()
                
                # Compute F-statistic
                ssr_linear = linear_bootstrap_results.ssr
                ssr_threshold = threshold_bootstrap_results.ssr
                
                f_stat = ((ssr_linear - ssr_threshold) / 1) / (ssr_threshold / (len(bootstrap_dz) - len(threshold_bootstrap_results.params)))
                bootstrap_f_stats[i] = f_stat
            
            # Compute bootstrap p-value
            bootstrap_p_value = np.mean(bootstrap_f_stats > f_statistic)
            
            # Create results dictionary
            bootstrap_results = {
                'test': 'Bootstrap Threshold Test',
                'model': model_type.upper(),
                'threshold': threshold_value,
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
    
    @handle_errors
    def test_asymmetric_adjustment(
        self, rho_above: float, rho_below: float,
        se_above: float, se_below: float
    ) -> Dict[str, Any]:
        """
        Test for asymmetric adjustment in threshold model.
        
        Args:
            rho_above: Adjustment coefficient above threshold.
            rho_below: Adjustment coefficient below threshold.
            se_above: Standard error of rho_above.
            se_below: Standard error of rho_below.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the test fails.
        """
        logger.info("Testing for asymmetric adjustment")
        
        try:
            # Calculate t-statistic
            t_stat = (rho_above - rho_below) / np.sqrt(se_above**2 + se_below**2)
            
            # Calculate p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=np.inf))
            
            # Create results dictionary
            results = {
                'test': 'Asymmetric Adjustment',
                'rho_above': rho_above,
                'rho_below': rho_below,
                'se_above': se_above,
                'se_below': se_below,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_asymmetric': p_value < self.alpha,
                'alpha': self.alpha,
            }
            
            logger.info(f"Asymmetric adjustment test results: t_statistic={t_stat:.4f}, p_value={p_value:.4f}, is_asymmetric={results['is_asymmetric']}")
            return results
        except Exception as e:
            logger.error(f"Error testing for asymmetric adjustment: {e}")
            raise YemenAnalysisError(f"Error testing for asymmetric adjustment: {e}")
    
    @handle_errors
    def plot_regimes(
        self, residuals: pd.Series, threshold: float, dates: Optional[pd.Series] = None,
        model_type: str = 'tar', title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a plot of regimes in threshold model.
        
        Args:
            residuals: Residuals from the cointegrating regression.
            threshold: Threshold value.
            dates: Dates for the residuals. If None, uses index values.
            model_type: Type of threshold model. Options are 'tar' and 'mtar'.
            title: Title for the plot.
            
        Returns:
            Matplotlib figure.
            
        Raises:
            YemenAnalysisError: If the plot cannot be created.
        """
        logger.info(f"Creating regimes plot for {model_type.upper()} model")
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # Create lagged residuals and differences
            z_lag = residuals.shift(1)
            dz = residuals.diff()
            
            # Create momentum term if model_type is 'mtar'
            if model_type == 'mtar':
                regime_var = dz.shift(1)
                var_name = "Δz_{t-1}"
            else:  # model_type == 'tar'
                regime_var = z_lag
                var_name = "z_{t-1}"
            
            # Align the data
            common_index = residuals.index.intersection(z_lag.index).intersection(dz.index)
            if model_type == 'mtar':
                common_index = common_index.intersection(regime_var.index)
            
            residuals = residuals.loc[common_index]
            z_lag = z_lag.loc[common_index]
            dz = dz.loc[common_index]
            regime_var = regime_var.loc[common_index]
            
            # Create regimes
            above_regime = regime_var >= threshold
            below_regime = regime_var < threshold
            
            # X-axis values
            if dates is not None:
                dates = dates.loc[common_index]
                x = dates
                ax.set_xlabel("Date")
            else:
                x = np.arange(len(residuals))
                ax.set_xlabel("Index")
            
            # Plot regimes
            ax.plot(x, residuals, 'k-', alpha=0.5, label="Residuals")
            ax.scatter(x[above_regime], residuals[above_regime], color='blue', label=f"{var_name} ≥ {threshold:.4f}")
            ax.scatter(x[below_regime], residuals[below_regime], color='red', label=f"{var_name} < {threshold:.4f}")
            
            # Add horizontal lines
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(y=threshold, color='g', linestyle='--', alpha=0.7, label=f"Threshold = {threshold:.4f}")
            
            # Set title and labels
            if title is None:
                title = f"Regimes in {model_type.upper()} Model"
            ax.set_title(title)
            ax.set_ylabel("Residuals")
            
            # Add legend
            ax.legend()
            
            # Rotate x-axis labels if dates are provided
            if dates is not None:
                fig.autofmt_xdate()
            
            logger.info(f"Created regimes plot for {model_type.upper()} model")
            return fig
        except Exception as e:
            logger.error(f"Error creating regimes plot: {e}")
            raise YemenAnalysisError(f"Error creating regimes plot: {e}")
    
    @handle_errors
    def run(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        model_type: str = 'tar', trend: str = 'c', fixed_threshold: Optional[float] = None,
        max_lags: Optional[int] = None, bootstrap: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete threshold cointegration analysis.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            model_type: Type of threshold model. Options are 'tar' and 'mtar'.
            trend: Trend to include in the cointegration test. Options are 'c' (constant),
                 'ct' (constant and trend), 'ctt' (constant, linear and quadratic trend),
                 and 'n' (no trend).
            fixed_threshold: Fixed threshold value. If None, the threshold is estimated.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            bootstrap: Whether to perform bootstrap tests.
            
        Returns:
            Dictionary containing the complete analysis results.
            
        Raises:
            YemenAnalysisError: If the analysis fails or the model type is invalid.
        """
        logger.info(f"Running complete threshold cointegration analysis with {model_type.upper()} model")
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
        
        try:
            # Step 1: Estimate cointegrating relationship
            coint_results = self.estimate_cointegration(y, x, y_col, x_col, trend, max_lags)
            residuals = coint_results['residuals']
            
            # Step 2: Estimate threshold model
            threshold_results = self.estimate_threshold(residuals, model_type, fixed_threshold, max_lags)
            
            # Step 3: Perform bootstrap test if requested
            bootstrap_results = None
            if bootstrap:
                bootstrap_results = self.bootstrap_threshold_test(
                    residuals, model_type, threshold_results['threshold'], max_lags
                )
            
            # Step 4: Test for asymmetric adjustment
            asymmetric_results = self.test_asymmetric_adjustment(
                threshold_results['params']['rho_above'],
                threshold_results['params']['rho_below'],
                threshold_results['std_errors']['rho_above'],
                threshold_results['std_errors']['rho_below']
            )
            
            # Step 5: Create regimes plot
            # Extract dates if available
            dates = None
            if 'date' in y.columns:
                dates = y['date']
            
            regimes_plot = self.plot_regimes(residuals, threshold_results['threshold'], dates, model_type)
            
            # Create complete results dictionary
            results = {
                'model': model_type.upper(),
                'cointegration': coint_results,
                'threshold': threshold_results,
                'bootstrap': bootstrap_results,
                'asymmetry': asymmetric_results,
                'regimes_plot': regimes_plot,
                'y_variable': y_col,
                'x_variable': x_col,
                'trend': trend,
                'max_lags': max_lags,
                'fixed_threshold': fixed_threshold,
            }
            
            logger.info(f"Complete threshold cointegration analysis completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error running complete threshold cointegration analysis: {e}")
            raise YemenAnalysisError(f"Error running complete threshold cointegration analysis: {e}")