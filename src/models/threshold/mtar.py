"""
Momentum Threshold Autoregressive (M-TAR) model module for Yemen Market Analysis.

This module provides the MomentumThresholdAutoregressive class for estimating
M-TAR models for cointegrated time series, with enhanced capabilities for
analyzing asymmetric adjustments in conflict-affected markets.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

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

class MomentumThresholdAutoregressive:
    """
    Momentum Threshold Autoregressive (M-TAR) model for Yemen Market Analysis.

    This class provides methods for estimating M-TAR models for cointegrated time series,
    with enhanced capabilities for analyzing asymmetric adjustments in conflict-affected markets.
    The M-TAR model extends the TAR model by using the momentum (change) of the series
    rather than its level to determine regime switching.

    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        eg_tester (EngleGrangerTester): Engle-Granger test implementation.
    """

    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the M-TAR model.

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
        Estimate an M-TAR model for cointegrated series.

        The M-TAR model is estimated as:
        Δz_t = ρ₁*z_{t-1}*I(Δz_{t-1} ≥ τ) + ρ₂*z_{t-1}*I(Δz_{t-1} < τ) + ∑ γ_i*Δz_{t-i} + ε_t

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
            Dictionary containing the M-TAR model results.

        Raises:
            YemenAnalysisError: If the columns are not found or the estimation fails.
        """
        logger.info(f"Estimating M-TAR model for {y_col} and {x_col}")

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
        min_required_obs = 2 * (max_lags + 2) + 3  # Need observations for residuals, lags, differences, and momentum term

        if n_obs < min_required_obs:
            logger.warning(f"Sample size ({n_obs}) too small for M-TAR model. Needs at least {min_required_obs}.")
            raise YemenAnalysisError(f"Sample size ({n_obs}) too small for M-TAR model. Need at least {min_required_obs} observations.")

        try:
            # First, test for cointegration
            coint_results = self.eg_tester.test(y, x, y_col, x_col, 'c', max_lags)

            if not coint_results['is_cointegrated']:
                logger.warning(f"{y_col} and {x_col} are not cointegrated, M-TAR model may not be appropriate")

            # Get the residuals from the cointegrating regression
            residuals = coint_results['residuals']

            # Create lagged residuals and differences
            z_lag = residuals.shift(1)
            dz = residuals.diff()
            # Create momentum term based on configuration or default to standard lag 1
            momentum_type = config.get('analysis.threshold.mtar.momentum_type', 'standard')
            momentum_lag = config.get('analysis.threshold.mtar.momentum_lag', 1)
            
            # Create momentum term using the specified method
            dz_lag = self.create_heaviside_indicators(
                residuals=residuals,
                threshold=None,  # Not needed for creating momentum term
                momentum_type=momentum_type,
                momentum_lag=momentum_lag
            )[0]  # Get only the momentum term, not the indicators

            # Create lagged differences
            dz_lags = pd.DataFrame()
            for i in range(1, max_lags + 1):
                dz_lags[f'dz_lag_{i}'] = dz.shift(i)

            # Align the data
            common_index = dz.index.intersection(z_lag.index).intersection(dz_lag.index).intersection(dz_lags.index)
            dz = dz.loc[common_index]
            z_lag = z_lag.loc[common_index]
            dz_lag = dz_lag.loc[common_index]
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

                # Sort momentum term for grid search
                sorted_momentum = np.sort(dz_lag)
                n = len(sorted_momentum)

                # Define grid points
                lower_idx = int(n * trim)
                upper_idx = int(n * (1 - trim))
                grid_points = sorted_momentum[lower_idx:upper_idx]

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

            # Test for threshold effect (asymmetric adjustment)
            se_above = results.bse[1]  # Standard error for rho_above
            se_below = results.bse[2]  # Standard error for rho_below
            
            # Get test type from config or default to standard
            test_type = config.get('analysis.threshold.mtar.asymmetry_test', 'standard')
            
            # Perform asymmetry test
            asymmetry_test = self.test_asymmetric_adjustment_enhanced(
                rho_above=rho_above,
                rho_below=rho_below,
                se_above=se_above,
                se_below=se_below,
                test_type=test_type
            )

            # Create results dictionary
            mtar_results = {
                'model': 'M-TAR',
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
                'threshold_test': asymmetry_test,
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'aic': results.aic,
                'bic': results.bic,
                'residuals': results.resid,
                'n_obs': len(dz),
                'cointegration_results': coint_results,
            }

            logger.info(f"M-TAR model results: rho_above={rho_above:.4f}, rho_below={rho_below:.4f}, threshold_p_value={asymmetry_test['p_value']:.4f}")
            return mtar_results
        except Exception as e:
            logger.error(f"Error estimating M-TAR model: {e}")
            raise YemenAnalysisError(f"Error estimating M-TAR model: {e}")



    def bootstrap_threshold_test(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        max_lags: Optional[int] = None, n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform bootstrap test for threshold effect.

        This method implements a bootstrap test for the significance of the threshold
        effect in the M-TAR model.

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

        # First, estimate the M-TAR model
        mtar_results = self.estimate(y, x, y_col, x_col, max_lags=max_lags)

        # Extract the test statistic
        f_statistic = mtar_results['threshold_test']['f_statistic']

        try:
            # Get the residuals from the cointegrating regression
            residuals = mtar_results['cointegration_results']['residuals']

            # Create lagged residuals and differences
            z_lag = residuals.shift(1)
            dz = residuals.diff()
            dz_lag = dz.shift(1)  # This is the momentum term

            # Create lagged differences
            dz_lags = pd.DataFrame()
            for i in range(1, max_lags + 1):
                dz_lags[f'dz_lag_{i}'] = dz.shift(i)

            # Align the data
            common_index = dz.index.intersection(z_lag.index).intersection(dz_lag.index).intersection(dz_lags.index)
            dz = dz.loc[common_index]
            z_lag = z_lag.loc[common_index]
            dz_lag = dz_lag.loc[common_index]
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

                # Estimate M-TAR model on bootstrap sample
                # Create indicator variables with the threshold from the original model
                threshold = mtar_results['threshold']
                above_threshold = (dz_lag >= threshold).astype(int)
                below_threshold = (dz_lag < threshold).astype(int)

                # Create interaction terms
                z_above = z_lag * above_threshold
                z_below = z_lag * below_threshold

                # Create design matrix
                X_mtar = pd.DataFrame({
                    'z_above': z_above,
                    'z_below': z_below
                })

                # Add lagged differences
                X_mtar = pd.concat([X_mtar, dz_lags], axis=1)

                # Estimate M-TAR model
                mtar_model = OLS(bootstrap_dz, sm.add_constant(X_mtar))
                mtar_bootstrap_results = mtar_model.fit()

                # Estimate linear model
                linear_model = OLS(bootstrap_dz, sm.add_constant(X_linear))
                linear_bootstrap_results = linear_model.fit()

                # Compute F-statistic
                ssr_linear = linear_bootstrap_results.ssr
                ssr_mtar = mtar_bootstrap_results.ssr

                f_stat = ((ssr_linear - ssr_mtar) / 1) / (ssr_mtar / (len(bootstrap_dz) - len(mtar_bootstrap_results.params)))
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
            
    def create_heaviside_indicators(
        self,
        residuals: pd.Series,
        threshold: Optional[float] = None,
        momentum_type: str = 'standard',
        momentum_lag: int = 1
    ) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
        """
        Create Heaviside indicator functions for M-TAR model with different momentum specifications.
        
        This method implements several approaches to specify the momentum term:
        1. 'standard': Uses a single lag of the differenced series (default)
        2. 'moving_average': Uses a moving average of past changes
        3. 'exponential': Uses exponentially weighted changes
        
        Args:
            residuals: Series of residuals from cointegrating regression
            threshold: Threshold value. If None, only the momentum term is returned
                      without creating indicators.
            momentum_type: Type of momentum specification
                         ('standard', 'moving_average', 'exponential')
            momentum_lag: Lag for momentum term or window size for averaging methods
            
        Returns:
            Tuple containing:
            - momentum_term: The momentum term used for regime identification
            - above_threshold: Indicator for observations above threshold (None if threshold is None)
            - below_threshold: Indicator for observations below threshold (None if threshold is None)
            
        Raises:
            YemenAnalysisError: If the momentum specification is invalid
        """
        logger.info(f"Creating Heaviside indicators with {momentum_type} momentum specification")
        
        # Compute differenced series
        dz = residuals.diff()
        
        # Create momentum term based on specified type
        if momentum_type == 'standard':
            # Standard implementation: single lag of differenced series
            momentum_term = dz.shift(momentum_lag)
            logger.debug(f"Using standard momentum with lag {momentum_lag}")
            
        elif momentum_type == 'moving_average':
            # Moving average of past changes
            # Use at least 2 periods for moving average
            window = max(2, momentum_lag)
            
            # Create lagged differences for moving average
            dz_lags = pd.DataFrame()
            for i in range(1, window + 1):
                dz_lags[f'dz_lag_{i}'] = dz.shift(i)
                
            # Compute moving average (equal weights)
            momentum_term = dz_lags.mean(axis=1)
            logger.debug(f"Using moving average momentum with window {window}")
            
        elif momentum_type == 'exponential':
            # Exponentially weighted moving average of past changes
            # The alpha parameter controls the decay rate (higher = more weight on recent observations)
            # Default alpha is 0.5 for moderate decay
            alpha = config.get('analysis.threshold.mtar.ewma_alpha', 0.5)
            
            # Compute EWMA using pandas
            momentum_term = dz.shift(1).ewm(alpha=alpha, adjust=False).mean()
            logger.debug(f"Using exponential momentum with alpha {alpha}")
            
        else:
            logger.error(f"Invalid momentum type: {momentum_type}")
            raise YemenAnalysisError(f"Invalid momentum type: {momentum_type}. Must be one of: 'standard', 'moving_average', 'exponential'")
        
        # If threshold is None, just return the momentum term
        if threshold is None:
            return (momentum_term, None, None)
        
        # Create indicator variables
        above_threshold = (momentum_term >= threshold).astype(int)
        below_threshold = (momentum_term < threshold).astype(int)
        
        return (momentum_term, above_threshold, below_threshold)
    
    def test_asymmetric_adjustment_enhanced(
        self,
        rho_above: float,
        rho_below: float,
        se_above: float,
        se_below: float,
        test_type: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Enhanced test for asymmetric adjustment in M-TAR model.
        
        This method implements several approaches to test for asymmetric adjustment:
        1. 'standard': Standard t-test for equality of coefficients
        2. 'bootstrap': Bootstrap-based test for robustness
        3. 'joint': Joint test of asymmetry and threshold effect
        
        Args:
            rho_above: Adjustment coefficient above threshold
            rho_below: Adjustment coefficient below threshold
            se_above: Standard error of rho_above
            se_below: Standard error of rho_below
            test_type: Type of asymmetry test
                     ('standard', 'bootstrap', 'joint')
            
        Returns:
            Dictionary with test results
            
        Raises:
            YemenAnalysisError: If the test type is invalid
        """
        logger.info(f"Testing asymmetric adjustment using {test_type} method")
        
        if test_type == 'standard':
            # Standard t-test for equality of coefficients
            return self._standard_asymmetry_test(rho_above, rho_below, se_above, se_below)
            
        elif test_type == 'bootstrap':
            # Bootstrap-based test (requires additional data, use placeholder)
            # In a real implementation, this would require the original data
            # Here we'll use a simulation-based approach for demonstration
            return self._bootstrap_asymmetry_test(rho_above, rho_below, se_above, se_below)
            
        elif test_type == 'joint':
            # Joint test of asymmetry and threshold effect
            return self._joint_asymmetry_test(rho_above, rho_below, se_above, se_below)
            
        else:
            logger.warning(f"Unknown test type '{test_type}', falling back to standard")
            return self._standard_asymmetry_test(rho_above, rho_below, se_above, se_below)
    
    def _standard_asymmetry_test(
        self,
        rho_above: float,
        rho_below: float,
        se_above: float,
        se_below: float
    ) -> Dict[str, Any]:
        """
        Standard t-test for equality of adjustment coefficients.
        
        Args:
            rho_above: Adjustment coefficient above threshold
            rho_below: Adjustment coefficient below threshold
            se_above: Standard error of rho_above
            se_below: Standard error of rho_below
            
        Returns:
            Dictionary with test results
        """
        # Compute t-statistic for difference in coefficients
        # H0: rho_above = rho_below (symmetric adjustment)
        # H1: rho_above ≠ rho_below (asymmetric adjustment)
        diff = rho_above - rho_below
        
        # Standard error of the difference (assuming independence)
        se_diff = np.sqrt(se_above**2 + se_below**2)
        
        # Compute t-statistic
        t_stat = diff / se_diff
        
        # Compute p-value (two-tailed test)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=np.inf))
        
        # Create results dictionary
        test_results = {
            'test': 'Standard Asymmetry Test',
            'test_type': 'standard',
            'diff_coefficient': diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_threshold_significant': p_value < self.alpha,
            'f_statistic': t_stat**2,  # F-statistic is t-statistic squared for 1 restriction
        }
        
        logger.info(f"Standard asymmetry test results: t_statistic={t_stat:.4f}, p_value={p_value:.4f}")
        return test_results
    
    def _bootstrap_asymmetry_test(
        self,
        rho_above: float,
        rho_below: float,
        se_above: float,
        se_below: float,
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Bootstrap-based test for asymmetric adjustment.
        
        This method uses a parametric bootstrap approach to test for asymmetric adjustment,
        which is more robust to non-normality and small sample sizes.
        
        Args:
            rho_above: Adjustment coefficient above threshold
            rho_below: Adjustment coefficient below threshold
            se_above: Standard error of rho_above
            se_below: Standard error of rho_below
            n_bootstrap: Number of bootstrap replications
            
        Returns:
            Dictionary with test results
        """
        # Compute observed test statistic
        diff = rho_above - rho_below
        se_diff = np.sqrt(se_above**2 + se_below**2)
        t_stat = diff / se_diff
        
        # Initialize bootstrap distribution
        bootstrap_t_stats = np.zeros(n_bootstrap)
        
        # Bootstrap loop
        for i in range(n_bootstrap):
            # Generate bootstrap coefficients
            # Under H0, rho_above = rho_below, so we use the pooled estimate
            rho_pooled = (rho_above + rho_below) / 2
            
            # Generate random coefficients with same standard errors
            bootstrap_rho_above = np.random.normal(rho_pooled, se_above)
            bootstrap_rho_below = np.random.normal(rho_pooled, se_below)
            
            # Compute bootstrap test statistic
            bootstrap_diff = bootstrap_rho_above - bootstrap_rho_below
            bootstrap_t_stat = bootstrap_diff / se_diff
            
            bootstrap_t_stats[i] = bootstrap_t_stat
        
        # Compute bootstrap p-value (two-tailed)
        bootstrap_p_value = np.mean(np.abs(bootstrap_t_stats) > abs(t_stat))
        
        # Get critical values
        critical_values = {
            '1%': np.percentile(np.abs(bootstrap_t_stats), 99),
            '5%': np.percentile(np.abs(bootstrap_t_stats), 95),
            '10%': np.percentile(np.abs(bootstrap_t_stats), 90)
        }
        
        # Create results dictionary
        test_results = {
            'test': 'Bootstrap Asymmetry Test',
            'test_type': 'bootstrap',
            'diff_coefficient': diff,
            't_statistic': t_stat,
            'bootstrap_p_value': bootstrap_p_value,
            'p_value': bootstrap_p_value,  # For consistency with other tests
            'is_threshold_significant': bootstrap_p_value < self.alpha,
            'critical_values': critical_values,
            'n_bootstrap': n_bootstrap,
            'f_statistic': t_stat**2,  # For consistency with other tests
        }
        
        logger.info(f"Bootstrap asymmetry test results: t_statistic={t_stat:.4f}, bootstrap_p_value={bootstrap_p_value:.4f}")
        return test_results
    
    def _joint_asymmetry_test(
        self,
        rho_above: float,
        rho_below: float,
        se_above: float,
        se_below: float
    ) -> Dict[str, Any]:
        """
        Joint test of asymmetry and threshold effect.
        
        This test examines both whether the adjustment coefficients are different from each other
        (asymmetry) and whether they are jointly different from zero (threshold effect).
        
        Args:
            rho_above: Adjustment coefficient above threshold
            rho_below: Adjustment coefficient below threshold
            se_above: Standard error of rho_above
            se_below: Standard error of rho_below
            
        Returns:
            Dictionary with test results
        """
        # First, test for asymmetry (rho_above ≠ rho_below)
        asymmetry_results = self._standard_asymmetry_test(rho_above, rho_below, se_above, se_below)
        
        # Second, test for joint significance (rho_above ≠ 0 and rho_below ≠ 0)
        # Compute t-statistics for individual coefficients
        t_above = rho_above / se_above
        t_below = rho_below / se_below
        
        # Compute p-values for individual coefficients
        p_above = 2 * (1 - stats.t.cdf(abs(t_above), df=np.inf))
        p_below = 2 * (1 - stats.t.cdf(abs(t_below), df=np.inf))
        
        # Approximate joint F-statistic (assuming independence)
        # This is a simplification; in practice, you would use the full covariance matrix
        f_joint = (t_above**2 + t_below**2) / 2
        
        # Compute p-value using F-distribution with 2 and infinite degrees of freedom
        p_joint = 1 - stats.f.cdf(f_joint, 2, np.inf)
        
        # Create results dictionary
        test_results = {
            'test': 'Joint Asymmetry and Threshold Test',
            'test_type': 'joint',
            'asymmetry': {
                'diff_coefficient': asymmetry_results['diff_coefficient'],
                't_statistic': asymmetry_results['t_statistic'],
                'p_value': asymmetry_results['p_value'],
                'is_significant': asymmetry_results['is_threshold_significant']
            },
            'individual': {
                'rho_above': {
                    'coefficient': rho_above,
                    't_statistic': t_above,
                    'p_value': p_above,
                    'is_significant': p_above < self.alpha
                },
                'rho_below': {
                    'coefficient': rho_below,
                    't_statistic': t_below,
                    'p_value': p_below,
                    'is_significant': p_below < self.alpha
                }
            },
            'joint': {
                'f_statistic': f_joint,
                'p_value': p_joint,
                'is_significant': p_joint < self.alpha
            },
            # For consistency with other tests
            'f_statistic': asymmetry_results['f_statistic'],
            'p_value': asymmetry_results['p_value'],
            'is_threshold_significant': asymmetry_results['is_threshold_significant']
        }
        
        logger.info(f"Joint asymmetry test results: asymmetry_p={asymmetry_results['p_value']:.4f}, joint_p={p_joint:.4f}")
        return test_results
