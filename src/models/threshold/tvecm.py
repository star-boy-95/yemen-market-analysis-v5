"""
Threshold Vector Error Correction Model (TVECM) module for Yemen Market Analysis.

This module provides the ThresholdVECM class for estimating
TVECM models for cointegrated time series.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.regression.linear_model import OLS
from scipy import stats

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data
from src.models.cointegration.johansen import JohansenTester

# Initialize logger
logger = logging.getLogger(__name__)

class ThresholdVECM:
    """
    Threshold Vector Error Correction Model (TVECM) for Yemen Market Analysis.

    This class provides methods for estimating TVECM models for cointegrated time series.

    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        johansen_tester (JohansenTester): Johansen test implementation.
    """

    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the TVECM model.

        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.threshold.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.threshold.max_lags', 4)
        self.johansen_tester = JohansenTester(alpha=self.alpha, max_lags=self.max_lags)

    @handle_errors
    def estimate(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        k_ar_diff: int = 2, deterministic: str = 'ci', coint_rank: int = 1,
        fixed_threshold: Optional[float] = None, max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate a TVECM model for cointegrated series.

        The TVECM model is a multivariate extension of the TAR model that allows for
        threshold effects in the error correction mechanism.

        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            k_ar_diff: Number of lagged differences in the VECM.
            deterministic: Deterministic terms to include. Options are 'nc' (no constant),
                          'co' (constant outside cointegration), 'ci' (constant inside
                          cointegration), 'lo' (linear trend outside), 'li' (linear trend
                          inside), and 'cili' (constant and linear trend inside).
            coint_rank: Cointegration rank.
            fixed_threshold: Fixed threshold value. If None, the threshold is estimated.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.

        Returns:
            Dictionary containing the TVECM model results.

        Raises:
            YemenAnalysisError: If the columns are not found or the estimation fails.
        """
        logger.info(f"Estimating TVECM model for {y_col} and {x_col}")

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
        min_required_obs = 4 * (k_ar_diff + 1) + 5  # VECM requires more observations due to its multivariate nature

        if n_obs < min_required_obs:
            logger.warning(f"Sample size ({n_obs}) too small for TVECM model. Needs at least {min_required_obs}.")
            raise YemenAnalysisError(f"Sample size ({n_obs}) too small for TVECM model. Need at least {min_required_obs} observations.")

        try:
            # Create combined data
            combined_data = pd.DataFrame({
                'y': y_data,
                'x': x_data
            })

            # First, test for cointegration using Johansen test
            det_order = 0 if deterministic == 'nc' else 1 if deterministic in ['co', 'ci'] else 2
            coint_results = self.johansen_tester.test(combined_data, ['y', 'x'], det_order, k_ar_diff, max_lags)

            if not coint_results['is_cointegrated']:
                logger.warning(f"{y_col} and {x_col} are not cointegrated, TVECM model may not be appropriate")

            # Estimate standard VECM
            vecm_model = VECM(combined_data, k_ar_diff=k_ar_diff, deterministic=deterministic, coint_rank=coint_rank)
            vecm_results = vecm_model.fit()

            # Get the error correction term
            beta = vecm_results.beta  # Cointegrating vector
            z = np.dot(combined_data, beta)  # Error correction term

            # Create lagged error correction term
            z_lag = pd.Series(z[:-1], index=combined_data.index[1:])

            # Create first differences
            dy = y_data.diff().dropna()
            dx = x_data.diff().dropna()

            # Create lagged differences
            dy_lags = pd.DataFrame()
            dx_lags = pd.DataFrame()

            for i in range(1, k_ar_diff):
                dy_lags[f'dy_lag_{i}'] = dy.shift(i)
                dx_lags[f'dx_lag_{i}'] = dx.shift(i)

            # Align the data
            common_index = dy.index.intersection(dx.index).intersection(z_lag.index).intersection(dy_lags.index).intersection(dx_lags.index)
            dy = dy.loc[common_index]
            dx = dx.loc[common_index]
            z_lag = z_lag.loc[common_index]
            dy_lags = dy_lags.loc[common_index]
            dx_lags = dx_lags.loc[common_index]

            # If fixed threshold is provided, use it
            if fixed_threshold is not None:
                threshold = fixed_threshold
                logger.info(f"Using fixed threshold: {threshold}")
            else:
                # Estimate the threshold
                # Get trimming parameter from config
                trim = config.get('analysis.threshold.trim', 0.15)
                n_grid = config.get('analysis.threshold.n_grid', 300)

                # Sort error correction term for grid search
                sorted_z = np.sort(z_lag)
                n = len(sorted_z)

                # Define grid points
                lower_idx = int(n * trim)
                upper_idx = int(n * (1 - trim))
                grid_points = sorted_z[lower_idx:upper_idx]

                # If grid is too large, subsample
                if len(grid_points) > n_grid:
                    step = len(grid_points) // n_grid
                    grid_points = grid_points[::step]

                # Initialize variables for grid search
                min_ssr_y = float('inf')
                min_ssr_x = float('inf')
                best_threshold_y = None
                best_threshold_x = None

                # Grid search for threshold
                for threshold_candidate in grid_points:
                    # Create indicator variables
                    above_threshold = (z_lag >= threshold_candidate).astype(int)
                    below_threshold = (z_lag < threshold_candidate).astype(int)

                    # Create interaction terms
                    z_above = z_lag * above_threshold
                    z_below = z_lag * below_threshold

                    # Create design matrices
                    X_y = pd.DataFrame({
                        'z_above': z_above,
                        'z_below': z_below
                    })

                    X_x = pd.DataFrame({
                        'z_above': z_above,
                        'z_below': z_below
                    })

                    # Add lagged differences
                    X_y = pd.concat([X_y, dy_lags, dx_lags], axis=1)
                    X_x = pd.concat([X_x, dy_lags, dx_lags], axis=1)

                    # Estimate models
                    model_y = OLS(dy, sm.add_constant(X_y))
                    results_y = model_y.fit()

                    model_x = OLS(dx, sm.add_constant(X_x))
                    results_x = model_x.fit()

                    # Update if SSR is lower
                    if results_y.ssr < min_ssr_y:
                        min_ssr_y = results_y.ssr
                        best_threshold_y = threshold_candidate

                    if results_x.ssr < min_ssr_x:
                        min_ssr_x = results_x.ssr
                        best_threshold_x = threshold_candidate

                # Use the threshold that minimizes the sum of SSRs
                if min_ssr_y + min_ssr_x < min_ssr_y + min_ssr_x:
                    threshold = best_threshold_y
                else:
                    threshold = best_threshold_x

                logger.info(f"Estimated threshold: {threshold}")

            # Create indicator variables with the best threshold
            above_threshold = (z_lag >= threshold).astype(int)
            below_threshold = (z_lag < threshold).astype(int)

            # Create interaction terms
            z_above = z_lag * above_threshold
            z_below = z_lag * below_threshold

            # Create design matrices
            X_y = pd.DataFrame({
                'z_above': z_above,
                'z_below': z_below
            })

            X_x = pd.DataFrame({
                'z_above': z_above,
                'z_below': z_below
            })

            # Add lagged differences
            X_y = pd.concat([X_y, dy_lags, dx_lags], axis=1)
            X_x = pd.concat([X_x, dy_lags, dx_lags], axis=1)

            # Estimate final models
            model_y = OLS(dy, sm.add_constant(X_y))
            results_y = model_y.fit()

            model_x = OLS(dx, sm.add_constant(X_x))
            results_x = model_x.fit()

            # Extract results
            alpha_y_above = results_y.params[1]  # Adjustment coefficient for y above threshold
            alpha_y_below = results_y.params[2]  # Adjustment coefficient for y below threshold

            alpha_x_above = results_x.params[1]  # Adjustment coefficient for x above threshold
            alpha_x_below = results_x.params[2]  # Adjustment coefficient for x below threshold

            # Test for threshold effect in y equation
            r_matrix_y = np.zeros((1, len(results_y.params)))
            r_matrix_y[0, 1] = 1
            r_matrix_y[0, 2] = -1

            wald_test_y = results_y.f_test(r_matrix_y)

            # Test for threshold effect in x equation
            r_matrix_x = np.zeros((1, len(results_x.params)))
            r_matrix_x[0, 1] = 1
            r_matrix_x[0, 2] = -1

            wald_test_x = results_x.f_test(r_matrix_x)

            # Create results dictionary
            tvecm_results = {
                'model': 'TVECM',
                'threshold': threshold,
                'fixed_threshold': fixed_threshold is not None,
                'y_equation': {
                    'params': {
                        'alpha_above': alpha_y_above,
                        'alpha_below': alpha_y_below,
                        'constant': results_y.params[0],
                        'lag_coefficients': results_y.params[3:].tolist(),
                    },
                    'std_errors': {
                        'alpha_above': results_y.bse[1],
                        'alpha_below': results_y.bse[2],
                        'constant': results_y.bse[0],
                        'lag_coefficients': results_y.bse[3:].tolist(),
                    },
                    'p_values': {
                        'alpha_above': results_y.pvalues[1],
                        'alpha_below': results_y.pvalues[2],
                        'constant': results_y.pvalues[0],
                        'lag_coefficients': results_y.pvalues[3:].tolist(),
                    },
                    'threshold_test': {
                        'f_statistic': wald_test_y.fvalue,
                        'p_value': wald_test_y.pvalue,
                        'is_threshold_significant': wald_test_y.pvalue < self.alpha,
                    },
                    'r_squared': results_y.rsquared,
                    'adj_r_squared': results_y.rsquared_adj,
                    'aic': results_y.aic,
                    'bic': results_y.bic,
                    'residuals': results_y.resid,
                },
                'x_equation': {
                    'params': {
                        'alpha_above': alpha_x_above,
                        'alpha_below': alpha_x_below,
                        'constant': results_x.params[0],
                        'lag_coefficients': results_x.params[3:].tolist(),
                    },
                    'std_errors': {
                        'alpha_above': results_x.bse[1],
                        'alpha_below': results_x.bse[2],
                        'constant': results_x.bse[0],
                        'lag_coefficients': results_x.bse[3:].tolist(),
                    },
                    'p_values': {
                        'alpha_above': results_x.pvalues[1],
                        'alpha_below': results_x.pvalues[2],
                        'constant': results_x.pvalues[0],
                        'lag_coefficients': results_x.pvalues[3:].tolist(),
                    },
                    'threshold_test': {
                        'f_statistic': wald_test_x.fvalue,
                        'p_value': wald_test_x.pvalue,
                        'is_threshold_significant': wald_test_x.pvalue < self.alpha,
                    },
                    'r_squared': results_x.rsquared,
                    'adj_r_squared': results_x.rsquared_adj,
                    'aic': results_x.aic,
                    'bic': results_x.bic,
                    'residuals': results_x.resid,
                },
                'system': {
                    'is_threshold_significant': (wald_test_y.pvalue < self.alpha) or (wald_test_x.pvalue < self.alpha),
                    'aic': results_y.aic + results_x.aic,
                    'bic': results_y.bic + results_x.bic,
                    'n_obs': len(dy),
                },
                'cointegration_results': coint_results,
                'vecm_results': {
                    'beta': vecm_results.beta.tolist(),
                    'alpha': vecm_results.alpha.tolist(),
                    'deterministic': deterministic,
                    'k_ar_diff': k_ar_diff,
                    'coint_rank': coint_rank,
                },
            }

            logger.info(f"TVECM model results: alpha_y_above={alpha_y_above:.4f}, alpha_y_below={alpha_y_below:.4f}, alpha_x_above={alpha_x_above:.4f}, alpha_x_below={alpha_x_below:.4f}")
            return tvecm_results
        except Exception as e:
            logger.error(f"Error estimating TVECM model: {e}")
            raise YemenAnalysisError(f"Error estimating TVECM model: {e}")



    def bootstrap_threshold_test(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        k_ar_diff: int = 2, deterministic: str = 'ci', coint_rank: int = 1,
        max_lags: Optional[int] = None, n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform bootstrap test for threshold effect in TVECM.

        This method implements a bootstrap test for the significance of the threshold
        effect in the TVECM model.

        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            k_ar_diff: Number of lagged differences in the VECM.
            deterministic: Deterministic terms to include.
            coint_rank: Cointegration rank.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            n_bootstrap: Number of bootstrap replications.

        Returns:
            Dictionary containing the bootstrap test results.

        Raises:
            YemenAnalysisError: If the columns are not found or the test fails.
        """
        logger.info(f"Performing bootstrap test for threshold effect with {n_bootstrap} replications")

        # First, estimate the TVECM model
        tvecm_results = self.estimate(
            y, x, y_col, x_col, k_ar_diff, deterministic, coint_rank, max_lags=max_lags
        )

        # Extract the test statistics
        f_statistic_y = tvecm_results['y_equation']['threshold_test']['f_statistic']
        f_statistic_x = tvecm_results['x_equation']['threshold_test']['f_statistic']

        # Use the maximum of the two F-statistics as the test statistic
        f_statistic = max(f_statistic_y, f_statistic_x)

        try:
            # Create combined data
            y_data = y[y_col].dropna()
            x_data = x[x_col].dropna()

            # Ensure the series have the same length
            common_index = y_data.index.intersection(x_data.index)
            y_data = y_data.loc[common_index]
            x_data = x_data.loc[common_index]

            combined_data = pd.DataFrame({
                'y': y_data,
                'x': x_data
            })

            # Estimate standard VECM
            vecm_model = VECM(combined_data, k_ar_diff=k_ar_diff, deterministic=deterministic, coint_rank=coint_rank)
            vecm_results = vecm_model.fit()

            # Get the error correction term
            beta = vecm_results.beta  # Cointegrating vector
            z = np.dot(combined_data, beta)  # Error correction term

            # Create lagged error correction term
            z_lag = pd.Series(z[:-1], index=combined_data.index[1:])

            # Create first differences
            dy = y_data.diff().dropna()
            dx = x_data.diff().dropna()

            # Create lagged differences
            dy_lags = pd.DataFrame()
            dx_lags = pd.DataFrame()

            for i in range(1, k_ar_diff):
                dy_lags[f'dy_lag_{i}'] = dy.shift(i)
                dx_lags[f'dx_lag_{i}'] = dx.shift(i)

            # Align the data
            common_index = dy.index.intersection(dx.index).intersection(z_lag.index).intersection(dy_lags.index).intersection(dx_lags.index)
            dy = dy.loc[common_index]
            dx = dx.loc[common_index]
            z_lag = z_lag.loc[common_index]
            dy_lags = dy_lags.loc[common_index]
            dx_lags = dx_lags.loc[common_index]

            # Estimate linear VECM equations
            X_linear_y = pd.DataFrame({'z_lag': z_lag})
            X_linear_y = pd.concat([X_linear_y, dy_lags, dx_lags], axis=1)

            X_linear_x = pd.DataFrame({'z_lag': z_lag})
            X_linear_x = pd.concat([X_linear_x, dy_lags, dx_lags], axis=1)

            linear_model_y = OLS(dy, sm.add_constant(X_linear_y))
            linear_results_y = linear_model_y.fit()

            linear_model_x = OLS(dx, sm.add_constant(X_linear_x))
            linear_results_x = linear_model_x.fit()

            # Get residuals from the linear models
            linear_residuals_y = linear_results_y.resid
            linear_residuals_x = linear_results_x.resid

            # Initialize bootstrap distribution
            bootstrap_f_stats = np.zeros(n_bootstrap)

            # Bootstrap loop
            for i in range(n_bootstrap):
                # Generate bootstrap samples
                bootstrap_residuals_y = np.random.choice(linear_residuals_y, size=len(linear_residuals_y))
                bootstrap_residuals_x = np.random.choice(linear_residuals_x, size=len(linear_residuals_x))

                bootstrap_dy = linear_results_y.predict() + bootstrap_residuals_y
                bootstrap_dx = linear_results_x.predict() + bootstrap_residuals_x

                # Estimate TVECM on bootstrap sample
                # Create indicator variables with the threshold from the original model
                threshold = tvecm_results['threshold']
                above_threshold = (z_lag >= threshold).astype(int)
                below_threshold = (z_lag < threshold).astype(int)

                # Create interaction terms
                z_above = z_lag * above_threshold
                z_below = z_lag * below_threshold

                # Create design matrices
                X_tvecm_y = pd.DataFrame({
                    'z_above': z_above,
                    'z_below': z_below
                })

                X_tvecm_x = pd.DataFrame({
                    'z_above': z_above,
                    'z_below': z_below
                })

                # Add lagged differences
                X_tvecm_y = pd.concat([X_tvecm_y, dy_lags, dx_lags], axis=1)
                X_tvecm_x = pd.concat([X_tvecm_x, dy_lags, dx_lags], axis=1)

                # Estimate TVECM models
                tvecm_model_y = OLS(bootstrap_dy, sm.add_constant(X_tvecm_y))
                tvecm_bootstrap_results_y = tvecm_model_y.fit()

                tvecm_model_x = OLS(bootstrap_dx, sm.add_constant(X_tvecm_x))
                tvecm_bootstrap_results_x = tvecm_model_x.fit()

                # Estimate linear models
                linear_model_y = OLS(bootstrap_dy, sm.add_constant(X_linear_y))
                linear_bootstrap_results_y = linear_model_y.fit()

                linear_model_x = OLS(bootstrap_dx, sm.add_constant(X_linear_x))
                linear_bootstrap_results_x = linear_model_x.fit()

                # Compute F-statistics
                ssr_linear_y = linear_bootstrap_results_y.ssr
                ssr_tvecm_y = tvecm_bootstrap_results_y.ssr

                ssr_linear_x = linear_bootstrap_results_x.ssr
                ssr_tvecm_x = tvecm_bootstrap_results_x.ssr

                f_stat_y = ((ssr_linear_y - ssr_tvecm_y) / 1) / (ssr_tvecm_y / (len(bootstrap_dy) - len(tvecm_bootstrap_results_y.params)))
                f_stat_x = ((ssr_linear_x - ssr_tvecm_x) / 1) / (ssr_tvecm_x / (len(bootstrap_dx) - len(tvecm_bootstrap_results_x.params)))

                # Use the maximum of the two F-statistics
                bootstrap_f_stats[i] = max(f_stat_y, f_stat_x)

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
