"""
Threshold Autoregressive (TAR) model module for Yemen Market Analysis.

This module provides the ThresholdAutoregressive class for estimating
TAR models for cointegrated time series.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
from scipy.optimize import minimize_scalar

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

        # Handle small sample sizes
        n_obs = len(common_index)
        min_required_obs = 2 * (max_lags + 2) + 3  # Need observations for residuals, lags, and differencing

        if n_obs < min_required_obs:
            logger.warning(f"Sample size ({n_obs}) too small for TAR model. Needs at least {min_required_obs}.")
            raise YemenAnalysisError(f"Sample size ({n_obs}) too small for TAR model. Need at least {min_required_obs} observations.")

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
                # Estimate the threshold using the specified method
                threshold = self.estimate_threshold(
                    residuals=z_lag,
                    dz=dz,
                    dz_lags=dz_lags,
                    trim=config.get('analysis.threshold.trim', 0.15),
                    n_grid=config.get('analysis.threshold.n_grid', 300),
                    method=config.get('analysis.threshold.method', 'hansen_seo')
                )
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
    def estimate_threshold(
        self,
        residuals: pd.Series,
        dz: pd.Series = None,
        dz_lags: pd.DataFrame = None,
        trim: float = 0.15,
        n_grid: int = 300,
        method: str = 'hansen_seo'
    ) -> float:
        """
        Estimate threshold using improved grid search algorithms.
        
        This method implements several approaches to threshold estimation:
        1. 'grid_search': Basic grid search over sorted residuals (original method)
        2. 'hansen_seo': Hansen & Seo's concentrated maximum likelihood approach
        3. 'adaptive': Adaptive grid search with finer grid in promising regions
        
        Args:
            residuals: Series of lagged residuals from cointegrating regression
            dz: Series of differenced residuals (required for some methods)
            dz_lags: DataFrame of lagged differenced residuals (required for some methods)
            trim: Trimming parameter for grid search (excludes trim% from each tail)
            n_grid: Number of grid points to evaluate
            method: Method for threshold estimation ('grid_search', 'hansen_seo', 'adaptive')
            
        Returns:
            Estimated threshold value
            
        Raises:
            YemenAnalysisError: If the estimation fails or parameters are invalid
        """
        logger.info(f"Estimating threshold using {method} method")
        
        # Validate inputs
        if method in ['hansen_seo', 'adaptive'] and (dz is None or dz_lags is None):
            logger.error("dz and dz_lags are required for hansen_seo and adaptive methods")
            raise YemenAnalysisError("dz and dz_lags are required for hansen_seo and adaptive methods")
            
        # Sort residuals for grid search
        sorted_residuals = np.sort(residuals)
        n = len(sorted_residuals)
        
        # Define grid points based on trimming
        lower_idx = int(n * trim)
        upper_idx = int(n * (1 - trim))
        grid_points = sorted_residuals[lower_idx:upper_idx]
        
        # If grid is too large, subsample
        if len(grid_points) > n_grid and method != 'adaptive':
            step = len(grid_points) // n_grid
            grid_points = grid_points[::step]
            
        if method == 'grid_search':
            # Basic grid search (original implementation)
            return self._grid_search_threshold(residuals, dz, dz_lags, grid_points)
            
        elif method == 'hansen_seo':
            # Hansen & Seo's concentrated maximum likelihood approach
            return self._hansen_seo_threshold(residuals, dz, dz_lags, grid_points)
            
        elif method == 'adaptive':
            # Adaptive grid search with finer grid in promising regions
            return self._adaptive_grid_search_threshold(residuals, dz, dz_lags, grid_points, n_grid)
            
        else:
            logger.warning(f"Unknown method '{method}', falling back to grid_search")
            return self._grid_search_threshold(residuals, dz, dz_lags, grid_points)
    
    def _grid_search_threshold(
        self,
        residuals: pd.Series,
        dz: pd.Series,
        dz_lags: pd.DataFrame,
        grid_points: np.ndarray
    ) -> float:
        """
        Basic grid search for threshold estimation.
        
        Args:
            residuals: Series of lagged residuals
            dz: Series of differenced residuals
            dz_lags: DataFrame of lagged differenced residuals
            grid_points: Array of threshold candidates to evaluate
            
        Returns:
            Threshold value that minimizes SSR
        """
        # Initialize variables for grid search
        min_ssr = float('inf')
        best_threshold = None
        
        # Grid search for threshold
        for threshold_candidate in grid_points:
            # Create indicator variables
            above_threshold = (residuals >= threshold_candidate).astype(int)
            below_threshold = (residuals < threshold_candidate).astype(int)
            
            # Create interaction terms
            z_above = residuals * above_threshold
            z_below = residuals * below_threshold
            
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
                
        return best_threshold
    
    def _hansen_seo_threshold(
        self,
        residuals: pd.Series,
        dz: pd.Series,
        dz_lags: pd.DataFrame,
        grid_points: np.ndarray
    ) -> float:
        """
        Hansen & Seo's concentrated maximum likelihood approach for threshold estimation.
        
        This method maximizes the likelihood function over the threshold parameter,
        which is equivalent to minimizing the determinant of the residual covariance matrix.
        For univariate models, this simplifies to minimizing the SSR.
        
        Args:
            residuals: Series of lagged residuals
            dz: Series of differenced residuals
            dz_lags: DataFrame of lagged differenced residuals
            grid_points: Array of threshold candidates to evaluate
            
        Returns:
            Threshold value that maximizes the likelihood function
        """
        # Initialize variables
        max_likelihood = float('-inf')
        best_threshold = None
        
        # Define function to compute negative log likelihood for a given threshold
        def compute_neg_log_likelihood(threshold):
            # Create indicator variables
            above_threshold = (residuals >= threshold).astype(int)
            below_threshold = (residuals < threshold).astype(int)
            
            # Create interaction terms
            z_above = residuals * above_threshold
            z_below = residuals * below_threshold
            
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
            
            # Compute log likelihood (negative for minimization)
            # Hansen & Seo use the determinant of the residual covariance matrix
            # For univariate case, this is equivalent to SSR
            log_likelihood = -np.log(results.ssr / len(dz))
            
            return -log_likelihood  # Return negative for minimization
        
        # Grid search for initial estimate
        for threshold_candidate in grid_points:
            likelihood = -compute_neg_log_likelihood(threshold_candidate)
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                best_threshold = threshold_candidate
        
        # Refine estimate using numerical optimization
        # Use the best grid point as starting value
        result = minimize_scalar(
            compute_neg_log_likelihood,
            bracket=[grid_points[0], best_threshold, grid_points[-1]],
            method='brent',
            options={'maxiter': 100, 'xtol': 1e-5}
        )
        
        if result.success:
            # Use optimized value if successful
            return result.x
        else:
            # Fall back to grid search result
            logger.warning("Numerical optimization failed, using grid search result")
            return best_threshold
    
    def _adaptive_grid_search_threshold(
        self,
        residuals: pd.Series,
        dz: pd.Series,
        dz_lags: pd.DataFrame,
        initial_grid: np.ndarray,
        n_grid: int
    ) -> float:
        """
        Adaptive grid search with finer grid in promising regions.
        
        This method performs an initial coarse grid search, then refines the search
        in the most promising regions to efficiently find the optimal threshold.
        
        Args:
            residuals: Series of lagged residuals
            dz: Series of differenced residuals
            dz_lags: DataFrame of lagged differenced residuals
            initial_grid: Initial array of threshold candidates
            n_grid: Target number of grid points to evaluate
            
        Returns:
            Threshold value that minimizes SSR
        """
        # Function to evaluate SSR for a threshold
        def evaluate_threshold(threshold):
            # Create indicator variables
            above_threshold = (residuals >= threshold).astype(int)
            below_threshold = (residuals < threshold).astype(int)
            
            # Create interaction terms
            z_above = residuals * above_threshold
            z_below = residuals * below_threshold
            
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
            
            return threshold, results.ssr
        
        # Initial coarse grid search (use ~20% of total grid points)
        coarse_n = max(20, n_grid // 5)
        step = len(initial_grid) // coarse_n
        coarse_grid = initial_grid[::step]
        
        # Evaluate coarse grid
        threshold_ssr_pairs = [evaluate_threshold(t) for t in coarse_grid]
        threshold_ssr_pairs.sort(key=lambda x: x[1])  # Sort by SSR
        
        # Select top candidates (lowest SSR)
        top_n = min(5, len(threshold_ssr_pairs))
        top_candidates = [pair[0] for pair in threshold_ssr_pairs[:top_n]]
        
        # Create refined grid around top candidates
        refined_grid = []
        for candidate in top_candidates:
            # Find candidate index in original grid
            idx = np.searchsorted(initial_grid, candidate)
            
            # Define window around candidate
            window_size = len(initial_grid) // (n_grid // top_n)
            lower_idx = max(0, idx - window_size // 2)
            upper_idx = min(len(initial_grid), idx + window_size // 2)
            
            # Add points from window to refined grid
            refined_grid.extend(initial_grid[lower_idx:upper_idx])
        
        # Remove duplicates and sort
        refined_grid = sorted(list(set(refined_grid)))
        
        # If refined grid is still too large, subsample
        if len(refined_grid) > n_grid:
            step = len(refined_grid) // n_grid
            refined_grid = refined_grid[::step]
        
        # Final grid search on refined grid
        min_ssr = float('inf')
        best_threshold = None
        
        for threshold in refined_grid:
            _, ssr = evaluate_threshold(threshold)
            if ssr < min_ssr:
                min_ssr = ssr
                best_threshold = threshold
                
        return best_threshold

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
            
    @handle_errors
    def test_threshold_significance(
        self,
        residuals: pd.Series,
        threshold: float,
        dz: pd.Series = None,
        dz_lags: pd.DataFrame = None,
        max_lags: int = 1,
        n_bootstrap: int = 1000,
        method: str = 'sup_lm'
    ) -> Dict[str, Any]:
        """
        Test for threshold significance using sup-LM test or other methods.
        
        This method implements several approaches to test the significance of the threshold effect:
        1. 'sup_lm': Hansen's supremum LM test
        2. 'wald': Wald test for threshold effect
        3. 'lr': Likelihood ratio test
        
        Args:
            residuals: Series of residuals from cointegrating regression
            threshold: Estimated threshold value
            dz: Series of differenced residuals (required for some methods)
            dz_lags: DataFrame of lagged differenced residuals (required for some methods)
            max_lags: Maximum number of lags for the test
            n_bootstrap: Number of bootstrap replications
            method: Testing method ('sup_lm', 'wald', 'lr')
            
        Returns:
            Dictionary with test results
            
        Raises:
            YemenAnalysisError: If the test fails or parameters are invalid
        """
        logger.info(f"Testing threshold significance using {method} method")
        
        # Validate inputs
        if dz is None or dz_lags is None:
            logger.error("dz and dz_lags are required for threshold significance tests")
            raise YemenAnalysisError("dz and dz_lags are required for threshold significance tests")
        
        if method == 'sup_lm':
            # Implement Hansen's supremum LM test
            return self._sup_lm_test(residuals, threshold, dz, dz_lags, max_lags, n_bootstrap)
        elif method == 'wald':
            # Implement Wald test
            return self._wald_test(residuals, threshold, dz, dz_lags)
        elif method == 'lr':
            # Implement likelihood ratio test
            return self._lr_test(residuals, threshold, dz, dz_lags, max_lags)
        else:
            logger.warning(f"Unknown method '{method}', falling back to sup_lm")
            return self._sup_lm_test(residuals, threshold, dz, dz_lags, max_lags, n_bootstrap)
    
    def _sup_lm_test(
        self,
        residuals: pd.Series,
        threshold: float,
        dz: pd.Series,
        dz_lags: pd.DataFrame,
        max_lags: int,
        n_bootstrap: int
    ) -> Dict[str, Any]:
        """
        Implement Hansen's supremum LM test for threshold significance.
        
        The sup-LM test computes the supremum of the LM statistics over all possible
        threshold values and uses bootstrap to determine critical values.
        
        Args:
            residuals: Series of residuals
            threshold: Estimated threshold value
            dz: Series of differenced residuals
            dz_lags: DataFrame of lagged differenced residuals
            max_lags: Maximum number of lags
            n_bootstrap: Number of bootstrap replications
            
        Returns:
            Dictionary with test results
        """
        # Step 1: Estimate restricted model (linear model without threshold)
        X_linear = pd.DataFrame({'z_lag': residuals})
        X_linear = pd.concat([X_linear, dz_lags], axis=1)
        
        linear_model = OLS(dz, sm.add_constant(X_linear))
        linear_results = linear_model.fit()
        
        # Get residuals from the linear model
        linear_residuals = linear_results.resid
        ssr_linear = linear_results.ssr
        
        # Step 2: Estimate unrestricted model (TAR model with threshold)
        # Create indicator variables
        above_threshold = (residuals >= threshold).astype(int)
        below_threshold = (residuals < threshold).astype(int)
        
        # Create interaction terms
        z_above = residuals * above_threshold
        z_below = residuals * below_threshold
        
        # Create design matrix
        X_tar = pd.DataFrame({
            'z_above': z_above,
            'z_below': z_below
        })
        
        # Add lagged differences
        X_tar = pd.concat([X_tar, dz_lags], axis=1)
        
        # Estimate TAR model
        tar_model = OLS(dz, sm.add_constant(X_tar))
        tar_results = tar_model.fit()
        
        ssr_tar = tar_results.ssr
        
        # Step 3: Compute sup-LM statistic
        # For homoskedastic errors, LM statistic is proportional to (SSR_0 - SSR_1)/SSR_0
        n = len(dz)
        lm_stat = n * (ssr_linear - ssr_tar) / ssr_linear
        
        # Step 4: Bootstrap to get critical values
        bootstrap_lm_stats = self._bootstrap_sup_lm(
            residuals, dz, dz_lags, linear_results, n_bootstrap
        )
        
        # Compute bootstrap p-value
        bootstrap_p_value = np.mean(bootstrap_lm_stats > lm_stat)
        
        # Get critical values
        critical_values = {
            '1%': np.percentile(bootstrap_lm_stats, 99),
            '5%': np.percentile(bootstrap_lm_stats, 95),
            '10%': np.percentile(bootstrap_lm_stats, 90)
        }
        
        # Create results dictionary
        test_results = {
            'test': 'Sup-LM Test for Threshold Effect',
            'lm_statistic': lm_stat,
            'bootstrap_p_value': bootstrap_p_value,
            'is_threshold_significant': bootstrap_p_value < self.alpha,
            'critical_values': critical_values,
            'n_bootstrap': n_bootstrap,
            'bootstrap_distribution': bootstrap_lm_stats,
            'alpha': self.alpha,
        }
        
        logger.info(f"Sup-LM test results: lm_statistic={lm_stat:.4f}, bootstrap_p_value={bootstrap_p_value:.4f}")
        return test_results
    
    def _bootstrap_sup_lm(
        self,
        residuals: pd.Series,
        dz: pd.Series,
        dz_lags: pd.DataFrame,
        linear_results: Any,
        n_bootstrap: int
    ) -> np.ndarray:
        """
        Bootstrap procedure for sup-LM test.
        
        Args:
            residuals: Series of residuals
            dz: Series of differenced residuals
            dz_lags: DataFrame of lagged differenced residuals
            linear_results: Results from linear model estimation
            n_bootstrap: Number of bootstrap replications
            
        Returns:
            Array of bootstrap sup-LM statistics
        """
        # Get residuals from the linear model
        linear_residuals = linear_results.resid
        
        # Initialize bootstrap distribution
        bootstrap_lm_stats = np.zeros(n_bootstrap)
        
        # Sort residuals for grid search
        sorted_residuals = np.sort(residuals)
        n = len(sorted_residuals)
        
        # Define grid points based on trimming
        trim = 0.15  # Use same trimming as in threshold estimation
        lower_idx = int(n * trim)
        upper_idx = int(n * (1 - trim))
        grid_points = sorted_residuals[lower_idx:upper_idx]
        
        # If grid is too large, subsample
        if len(grid_points) > 100:  # Use smaller grid for bootstrap
            step = len(grid_points) // 100
            grid_points = grid_points[::step]
        
        # Bootstrap loop
        for i in range(n_bootstrap):
            # Generate bootstrap sample
            bootstrap_residuals = np.random.choice(linear_residuals, size=len(linear_residuals))
            bootstrap_dz = linear_results.predict() + bootstrap_residuals
            
            # Compute sup-LM statistic for this bootstrap sample
            # First, estimate linear model
            X_linear = pd.DataFrame({'z_lag': residuals})
            X_linear = pd.concat([X_linear, dz_lags], axis=1)
            
            linear_model = OLS(bootstrap_dz, sm.add_constant(X_linear))
            bootstrap_linear_results = linear_model.fit()
            
            ssr_linear = bootstrap_linear_results.ssr
            
            # Find sup-LM statistic over all threshold values
            max_lm_stat = 0
            
            for threshold_candidate in grid_points:
                # Create indicator variables
                above_threshold = (residuals >= threshold_candidate).astype(int)
                below_threshold = (residuals < threshold_candidate).astype(int)
                
                # Create interaction terms
                z_above = residuals * above_threshold
                z_below = residuals * below_threshold
                
                # Create design matrix
                X_tar = pd.DataFrame({
                    'z_above': z_above,
                    'z_below': z_below
                })
                
                # Add lagged differences
                X_tar = pd.concat([X_tar, dz_lags], axis=1)
                
                # Estimate TAR model
                tar_model = OLS(bootstrap_dz, sm.add_constant(X_tar))
                bootstrap_tar_results = tar_model.fit()
                
                ssr_tar = bootstrap_tar_results.ssr
                
                # Compute LM statistic
                lm_stat = n * (ssr_linear - ssr_tar) / ssr_linear
                
                # Update max LM statistic
                max_lm_stat = max(max_lm_stat, lm_stat)
            
            bootstrap_lm_stats[i] = max_lm_stat
        
        return bootstrap_lm_stats
    
    def _wald_test(
        self,
        residuals: pd.Series,
        threshold: float,
        dz: pd.Series,
        dz_lags: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Wald test for threshold effect.
        
        This test examines whether the adjustment coefficients above and below
        the threshold are significantly different.
        
        Args:
            residuals: Series of residuals
            threshold: Estimated threshold value
            dz: Series of differenced residuals
            dz_lags: DataFrame of lagged differenced residuals
            
        Returns:
            Dictionary with test results
        """
        # Create indicator variables
        above_threshold = (residuals >= threshold).astype(int)
        below_threshold = (residuals < threshold).astype(int)
        
        # Create interaction terms
        z_above = residuals * above_threshold
        z_below = residuals * below_threshold
        
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
        
        # Test for threshold effect
        r_matrix = np.zeros((1, len(results.params)))
        r_matrix[0, 1] = 1
        r_matrix[0, 2] = -1
        
        wald_test = results.f_test(r_matrix)
        
        # Create results dictionary
        test_results = {
            'test': 'Wald Test for Threshold Effect',
            'f_statistic': wald_test.fvalue,
            'p_value': wald_test.pvalue,
            'is_threshold_significant': wald_test.pvalue < self.alpha,
            'df_num': wald_test.df_num,
            'df_denom': wald_test.df_denom,
        }
        
        logger.info(f"Wald test results: f_statistic={wald_test.fvalue:.4f}, p_value={wald_test.pvalue:.4f}")
        return test_results
    
    def _lr_test(
        self,
        residuals: pd.Series,
        threshold: float,
        dz: pd.Series,
        dz_lags: pd.DataFrame,
        max_lags: int
    ) -> Dict[str, Any]:
        """
        Likelihood ratio test for threshold effect.
        
        This test compares the likelihood of the TAR model to that of a linear model.
        
        Args:
            residuals: Series of residuals
            threshold: Estimated threshold value
            dz: Series of differenced residuals
            dz_lags: DataFrame of lagged differenced residuals
            max_lags: Maximum number of lags
            
        Returns:
            Dictionary with test results
        """
        # Estimate restricted model (linear model without threshold)
        X_linear = pd.DataFrame({'z_lag': residuals})
        X_linear = pd.concat([X_linear, dz_lags], axis=1)
        
        linear_model = OLS(dz, sm.add_constant(X_linear))
        linear_results = linear_model.fit()
        
        ssr_linear = linear_results.ssr
        
        # Estimate unrestricted model (TAR model with threshold)
        # Create indicator variables
        above_threshold = (residuals >= threshold).astype(int)
        below_threshold = (residuals < threshold).astype(int)
        
        # Create interaction terms
        z_above = residuals * above_threshold
        z_below = residuals * below_threshold
        
        # Create design matrix
        X_tar = pd.DataFrame({
            'z_above': z_above,
            'z_below': z_below
        })
        
        # Add lagged differences
        X_tar = pd.concat([X_tar, dz_lags], axis=1)
        
        # Estimate TAR model
        tar_model = OLS(dz, sm.add_constant(X_tar))
        tar_results = tar_model.fit()
        
        ssr_tar = tar_results.ssr
        
        # Compute LR statistic
        n = len(dz)
        lr_stat = n * np.log(ssr_linear / ssr_tar)
        
        # Compute p-value using chi-squared distribution
        # The degrees of freedom is the difference in the number of parameters
        df = 1  # One additional parameter in TAR model (separate adjustment coefficients)
        p_value = 1 - stats.chi2.cdf(lr_stat, df)
        
        # Create results dictionary
        test_results = {
            'test': 'Likelihood Ratio Test for Threshold Effect',
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'is_threshold_significant': p_value < self.alpha,
            'df': df,
        }
        
        logger.info(f"LR test results: lr_statistic={lr_stat:.4f}, p_value={p_value:.4f}")
        return test_results
    
    @handle_errors
    def bootstrap_threshold_test_enhanced(
        self,
        residuals: pd.Series,
        dz: pd.Series = None,
        dz_lags: pd.DataFrame = None,
        n_bootstrap: int = 1000,
        method: str = 'parametric',
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced bootstrap test for threshold effect with multiple methods and parallel processing.
        
        This method implements several bootstrap approaches:
        1. 'parametric': Parametric bootstrap assuming normal errors
        2. 'nonparametric': Nonparametric bootstrap resampling residuals
        3. 'wild': Wild bootstrap for heteroskedastic errors
        
        Args:
            residuals: Series of residuals from cointegrating regression
            dz: Series of differenced residuals
            dz_lags: DataFrame of lagged differenced residuals
            n_bootstrap: Number of bootstrap replications
            method: Bootstrap method ('parametric', 'nonparametric', 'wild')
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with bootstrap test results
            
        Raises:
            YemenAnalysisError: If the bootstrap fails or parameters are invalid
        """
        logger.info(f"Performing enhanced bootstrap test with {method} method and {n_bootstrap} replications")
        
        # Validate inputs
        if dz is None or dz_lags is None:
            logger.error("dz and dz_lags are required for enhanced bootstrap test")
            raise YemenAnalysisError("dz and dz_lags are required for enhanced bootstrap test")
        
        # Estimate threshold
        threshold = self.estimate_threshold(residuals, dz, dz_lags)
        
        # Estimate linear model (no threshold)
        X_linear = pd.DataFrame({'z_lag': residuals})
        X_linear = pd.concat([X_linear, dz_lags], axis=1)
        
        linear_model = OLS(dz, sm.add_constant(X_linear))
        linear_results = linear_model.fit()
        
        # Estimate TAR model with threshold
        # Create indicator variables
        above_threshold = (residuals >= threshold).astype(int)
        below_threshold = (residuals < threshold).astype(int)
        
        # Create interaction terms
        z_above = residuals * above_threshold
        z_below = residuals * below_threshold
        
        # Create design matrix
        X_tar = pd.DataFrame({
            'z_above': z_above,
            'z_below': z_below
        })
        
        # Add lagged differences
        X_tar = pd.concat([X_tar, dz_lags], axis=1)
        
        # Estimate TAR model
        tar_model = OLS(dz, sm.add_constant(X_tar))
        tar_results = tar_model.fit()
        
        # Compute test statistic (F-statistic)
        ssr_linear = linear_results.ssr
        ssr_tar = tar_results.ssr
        n = len(dz)
        df_diff = 1  # Difference in degrees of freedom
        df_denom = n - len(tar_results.params)
        
        f_stat = ((ssr_linear - ssr_tar) / df_diff) / (ssr_tar / df_denom)
        
        # Define bootstrap function
        def bootstrap_iteration(i):
            # Generate bootstrap sample based on method
            if method == 'parametric':
                # Parametric bootstrap assuming normal errors
                sigma = np.std(tar_results.resid)
                bootstrap_errors = np.random.normal(0, sigma, size=n)
                bootstrap_dz = tar_results.predict() + bootstrap_errors
            elif method == 'nonparametric':
                # Nonparametric bootstrap resampling residuals
                bootstrap_residuals = np.random.choice(tar_results.resid, size=n)
                bootstrap_dz = tar_results.predict() + bootstrap_residuals
            elif method == 'wild':
                # Wild bootstrap for heteroskedastic errors
                # Rademacher distribution: random variable taking values -1 or 1 with probability 0.5
                rademacher = np.random.choice([-1, 1], size=n)
                bootstrap_dz = tar_results.predict() + tar_results.resid * rademacher
            else:
                # Default to nonparametric
                bootstrap_residuals = np.random.choice(tar_results.resid, size=n)
                bootstrap_dz = tar_results.predict() + bootstrap_residuals
            
            # Estimate linear model on bootstrap sample
            bootstrap_linear_model = OLS(bootstrap_dz, sm.add_constant(X_linear))
            bootstrap_linear_results = bootstrap_linear_model.fit()
            
            # Estimate TAR model on bootstrap sample
            bootstrap_tar_model = OLS(bootstrap_dz, sm.add_constant(X_tar))
            bootstrap_tar_results = bootstrap_tar_model.fit()
            
            # Compute F-statistic
            bootstrap_ssr_linear = bootstrap_linear_results.ssr
            bootstrap_ssr_tar = bootstrap_tar_results.ssr
            
            bootstrap_f_stat = ((bootstrap_ssr_linear - bootstrap_ssr_tar) / df_diff) / (bootstrap_ssr_tar / df_denom)
            
            return bootstrap_f_stat
        
        # Perform bootstrap
        bootstrap_f_stats = np.zeros(n_bootstrap)
        
        if parallel and n_bootstrap >= 100:
            # Use parallel processing for large number of replications
            try:
                # Determine number of workers (use at most 4 cores)
                import os
                n_workers = min(4, os.cpu_count() or 1)
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all bootstrap iterations
                    futures = [executor.submit(bootstrap_iteration, i) for i in range(n_bootstrap)]
                    
                    # Collect results as they complete
                    for i, future in enumerate(as_completed(futures)):
                        bootstrap_f_stats[i] = future.result()
                        
                logger.info(f"Completed parallel bootstrap with {n_workers} workers")
            except Exception as e:
                logger.warning(f"Parallel bootstrap failed: {e}. Falling back to sequential.")
                # Fall back to sequential bootstrap
                for i in range(n_bootstrap):
                    bootstrap_f_stats[i] = bootstrap_iteration(i)
        else:
            # Sequential bootstrap
            for i in range(n_bootstrap):
                bootstrap_f_stats[i] = bootstrap_iteration(i)
        
        # Compute bootstrap p-value
        bootstrap_p_value = np.mean(bootstrap_f_stats > f_stat)
        
        # Get critical values
        critical_values = {
            '1%': np.percentile(bootstrap_f_stats, 99),
            '5%': np.percentile(bootstrap_f_stats, 95),
            '10%': np.percentile(bootstrap_f_stats, 90)
        }
        
        # Create results dictionary
        bootstrap_results = {
            'test': f'Enhanced Bootstrap Test ({method})',
            'f_statistic': f_stat,
            'bootstrap_p_value': bootstrap_p_value,
            'is_threshold_significant': bootstrap_p_value < self.alpha,
            'critical_values': critical_values,
            'n_bootstrap': n_bootstrap,
            'bootstrap_distribution': bootstrap_f_stats,
            'alpha': self.alpha,
            'method': method,
            'parallel': parallel,
        }
        
        logger.info(f"Enhanced bootstrap test results: f_statistic={f_stat:.4f}, bootstrap_p_value={bootstrap_p_value:.4f}")
        return bootstrap_results
