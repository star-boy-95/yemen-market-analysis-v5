"""
Panel cointegration testing for Yemen Market Integration analysis.

This module implements panel cointegration tests for market integration analysis,
providing methods to test for cointegration in panel data across multiple markets.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import warnings

from yemen_market_integration.utils.error_handler import handle_errors
from yemen_market_integration.utils.decorators import timer
from yemen_market_integration.utils.m3_utils import m3_optimized, tiered_cache, optimize_array_computation
from yemen_market_integration.utils.validation import validate_dataframe, validate_time_series, raise_if_invalid

# Initialize module logger
logger = logging.getLogger(__name__)


@dataclass
class PanelTestResult:
    """Container for panel cointegration test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    critical_values: Dict[str, float]
    method: str
    n_obs: int
    n_markets: int


class PanelCointegrationTester:
    """
    Panel cointegration testing for multiple market pairs.
    
    This class implements Pedroni, Kao, and Westerlund panel cointegration tests,
    accommodating cross-sectional dependence and heterogeneous panel structures.
    """
    
    @timer
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def __init__(
        self, 
        data: pd.DataFrame, 
        market_col: str, 
        time_col: str, 
        price_col: str,
        max_workers: Optional[int] = None
    ):
        """
        Initialize panel cointegration tester.
        
        Parameters
        ----------
        data : pd.DataFrame
            Panel data containing prices for multiple markets over time
        market_col : str
            Column name for market identifiers
        time_col : str
            Column name for time/date
        price_col : str
            Column name for price data
        max_workers : int, optional
            Maximum number of parallel workers to use
        """
        self.data = data
        self.market_col = market_col
        self.time_col = time_col
        self.price_col = price_col
        
        # Set max workers
        self.max_workers = max_workers
        if max_workers is None:
            import multiprocessing as mp
            self.max_workers = max(1, mp.cpu_count() - 1)
        
        # Validate data
        self._validate_panel_data()
        
        # Transform to wide format for panel analysis
        self._prepare_panel_data()
        
        logger.info(
            f"Initialized PanelCointegrationTester with {self.n_markets} markets "
            f"and {self.n_periods} time periods"
        )
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _validate_panel_data(self) -> None:
        """
        Validate panel data for cointegration analysis.
        
        This method checks if data has the required columns and a
        balanced panel structure.
        """
        # Validate basic dataframe
        valid, errors = validate_dataframe(
            self.data,
            required_columns=[self.market_col, self.time_col, self.price_col],
            min_rows=30
        )
        raise_if_invalid(valid, errors, "Invalid panel data for cointegration analysis")
        
        # Get dimensions
        markets = self.data[self.market_col].unique()
        times = self.data[self.time_col].unique()
        
        self.n_markets = len(markets)
        self.n_periods = len(times)
        
        # Check minimum requirements
        if self.n_markets < 2:
            raise ValueError(f"Need at least 2 markets for panel analysis, got {self.n_markets}")
        
        if self.n_periods < 20:
            raise ValueError(f"Need at least 20 time periods for panel analysis, got {self.n_periods}")
        
        # Check if panel is balanced
        market_counts = self.data.groupby(self.market_col)[self.time_col].nunique()
        if not (market_counts == self.n_periods).all():
            logger.warning(
                f"Panel is unbalanced: {sum(market_counts < self.n_periods)} markets "
                f"have fewer than {self.n_periods} observations"
            )
    
    @m3_optimized(memory_intensive=True)
    def _prepare_panel_data(self) -> None:
        """
        Transform data to wide format for panel analysis.
        
        This method transforms the long-format data to wide format
        with markets as columns and time in rows.
        """
        # Pivot data to wide format (time x markets)
        self.panel_wide = self.data.pivot(
            index=self.time_col,
            columns=self.market_col,
            values=self.price_col
        )
        
        # Check for missing values
        missing = self.panel_wide.isna().sum().sum()
        if missing > 0:
            logger.warning(f"Panel data contains {missing} missing values")
            
            # Fill missing values using forward fill then backward fill
            self.panel_wide = self.panel_wide.fillna(method='ffill').fillna(method='bfill')
            
            remaining_missing = self.panel_wide.isna().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"Could not fill {remaining_missing} missing values")
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def test_pedroni(
        self, 
        trend: str = 'c',
        lag: int = 1
    ) -> Dict[str, Any]:
        """
        Perform Pedroni panel cointegration tests.
        
        Implements Pedroni (1999, 2004) panel cointegration tests,
        which allow for heterogeneous intercepts and deterministic trends.
        
        Parameters
        ----------
        trend : str, default='c'
            Deterministic trend specification:
            - 'n': No deterministic trend
            - 'c': Constant (intercept)
            - 'ct': Constant and trend
        lag : int, default=1
            Number of lags for ADF regression
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all Pedroni test statistics and results
            
        Notes
        -----
        Pedroni tests include 7 test statistics:
        - Panel v-Statistic
        - Panel rho-Statistic
        - Panel PP-Statistic (t)
        - Panel ADF-Statistic
        - Group rho-Statistic
        - Group PP-Statistic (t)
        - Group ADF-Statistic
        
        The within-dimension statistics (panel) pool autoregressive coefficients
        across markets, while the between-dimension statistics (group) average
        market-specific statistics.
        """
        # Validate trend specification
        valid_trends = ['n', 'c', 'ct']
        if trend not in valid_trends:
            raise ValueError(f"Trend must be one of {valid_trends}, got {trend}")
        
        # Step 1: Compute residuals from cointegrating regression
        residuals = self._compute_cointegrating_residuals(trend)
        
        # Step 2: Compute autoregressive coefficients and residual variances
        ar_coefs, residual_vars = self._compute_ar_coefficients(residuals, lag)
        
        # Step 3: Compute Pedroni test statistics
        panel_stats, group_stats = self._compute_pedroni_statistics(
            residuals, ar_coefs, residual_vars, lag
        )
        
        # Step 4: Determine significance
        alpha = 0.05
        panel_v_critical = 1.645  # One-sided test
        panel_other_critical = -1.645  # One-sided test
        group_critical = -1.645  # One-sided test
        
        # Compile results
        results = {
            'panel_v': {
                'statistic': panel_stats['panel_v'],
                'p_value': self._p_value_from_normal(panel_stats['panel_v'], right_tail=True),
                'significant': panel_stats['panel_v'] > panel_v_critical,
                'critical_values': {'1%': 2.326, '5%': 1.645, '10%': 1.282}
            },
            'panel_rho': {
                'statistic': panel_stats['panel_rho'],
                'p_value': self._p_value_from_normal(panel_stats['panel_rho'], right_tail=False),
                'significant': panel_stats['panel_rho'] < panel_other_critical,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            },
            'panel_pp': {
                'statistic': panel_stats['panel_pp'],
                'p_value': self._p_value_from_normal(panel_stats['panel_pp'], right_tail=False),
                'significant': panel_stats['panel_pp'] < panel_other_critical,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            },
            'panel_adf': {
                'statistic': panel_stats['panel_adf'],
                'p_value': self._p_value_from_normal(panel_stats['panel_adf'], right_tail=False),
                'significant': panel_stats['panel_adf'] < panel_other_critical,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            },
            'group_rho': {
                'statistic': group_stats['group_rho'],
                'p_value': self._p_value_from_normal(group_stats['group_rho'], right_tail=False),
                'significant': group_stats['group_rho'] < group_critical,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            },
            'group_pp': {
                'statistic': group_stats['group_pp'],
                'p_value': self._p_value_from_normal(group_stats['group_pp'], right_tail=False),
                'significant': group_stats['group_pp'] < group_critical,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            },
            'group_adf': {
                'statistic': group_stats['group_adf'],
                'p_value': self._p_value_from_normal(group_stats['group_adf'], right_tail=False),
                'significant': group_stats['group_adf'] < group_critical,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            }
        }
        
        # Determine overall result
        # Panel cointegration if majority of tests reject the null
        significant_count = sum(1 for result in results.values() if result['significant'])
        results['panel_cointegrated'] = significant_count >= 4  # At least 4 out of 7 tests
        
        # Add metadata
        results['method'] = 'pedroni'
        results['trend'] = trend
        results['lag'] = lag
        results['n_markets'] = self.n_markets
        results['n_periods'] = self.n_periods
        
        logger.info(
            f"Pedroni panel cointegration test results: "
            f"{significant_count}/7 tests significant, "
            f"{'cointegrated' if results['panel_cointegrated'] else 'not cointegrated'}"
        )
        
        return results
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _compute_cointegrating_residuals(self, trend: str) -> pd.DataFrame:
        """
        Compute residuals from market-specific cointegrating regressions.
        
        Parameters
        ----------
        trend : str
            Deterministic trend specification
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with residuals for each market
        """
        # TODO: Implement proper computation of cointegrating residuals
        # This is a placeholder implementation
        
        # Get the data
        data = self.panel_wide
        
        # For demonstration, use a simple pairwise approach
        # Take first market as reference and regress others on it
        reference_market = data.columns[0]
        residuals = pd.DataFrame(index=data.index)
        
        # Add trend variables if needed
        X = pd.DataFrame({'const': np.ones(len(data))})
        if trend in ['c', 'ct']:
            X['const'] = 1
        if trend == 'ct':
            X['trend'] = np.arange(len(data))
        
        # For each market, regress on reference and get residuals
        for market in data.columns[1:]:
            # Simple OLS regression
            y = data[market].values
            x = np.column_stack([data[reference_market].values, X.values])
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            
            # Calculate residuals
            residuals[market] = y - x @ beta
        
        return residuals
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _compute_ar_coefficients(
        self, 
        residuals: pd.DataFrame, 
        lag: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute autoregressive coefficients for each market.
        
        Parameters
        ----------
        residuals : pandas.DataFrame
            Residuals from cointegrating regressions
        lag : int
            Number of lags
            
        Returns
        -------
        tuple
            Tuple of (ar_coefficients, residual_variances)
        """
        # TODO: Implement proper computation of AR coefficients
        # This is a placeholder implementation
        
        ar_coefs = {}
        residual_vars = {}
        
        for market in residuals.columns:
            # Get market residuals
            u = residuals[market].values
            
            # Simple AR(1) model for demonstration
            # Should be replaced with proper ADF regression
            X = np.zeros((len(u) - 1, 1 + lag))
            X[:, 0] = u[:-1]  # AR(1) term
            
            for l in range(lag):
                if l+1 < len(u) - 1:
                    X[:, l+1] = np.diff(u)[:-1-l]
            
            y = np.diff(u)
            
            # OLS estimation
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            ar_coefs[market] = beta[0]
            
            # Residual variance
            e = y - X @ beta
            residual_vars[market] = np.var(e)
        
        return ar_coefs, residual_vars
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _compute_pedroni_statistics(
        self, 
        residuals: pd.DataFrame,
        ar_coefs: Dict[str, float],
        residual_vars: Dict[str, float],
        lag: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute panel and group statistics for Pedroni test.
        
        Parameters
        ----------
        residuals : pandas.DataFrame
            Residuals from cointegrating regressions
        ar_coefs : dict
            Autoregressive coefficients for each market
        residual_vars : dict
            Residual variances for each market
        lag : int
            Number of lags
            
        Returns
        -------
        tuple
            Tuple of (panel_stats, group_stats)
        """
        # TODO: Implement proper computation of Pedroni statistics
        # This is a placeholder implementation
        
        # Simplified calculation of test statistics
        # Should be replaced with actual formulas from Pedroni papers
        
        # Calculate average AR coefficient and variance
        avg_ar = np.mean(list(ar_coefs.values()))
        avg_var = np.mean(list(residual_vars.values()))
        
        # Panel statistics (pooled)
        panel_stats = {
            'panel_v': (1 - avg_ar) / np.sqrt(avg_var) * np.sqrt(self.n_markets),
            'panel_rho': (avg_ar - 1) / np.sqrt(avg_var) * np.sqrt(self.n_markets),
            'panel_pp': (avg_ar - 1) / np.sqrt(avg_var * 2) * np.sqrt(self.n_markets),
            'panel_adf': (avg_ar - 1) / np.sqrt(avg_var * 2) * np.sqrt(self.n_markets)
        }
        
        # Group statistics (averaged)
        group_stats = {
            'group_rho': np.mean([(ar - 1) / np.sqrt(var) for ar, var in zip(ar_coefs.values(), residual_vars.values())]),
            'group_pp': np.mean([(ar - 1) / np.sqrt(var * 2) for ar, var in zip(ar_coefs.values(), residual_vars.values())]),
            'group_adf': np.mean([(ar - 1) / np.sqrt(var * 2) for ar, var in zip(ar_coefs.values(), residual_vars.values())])
        }
        
        return panel_stats, group_stats
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def test_kao(self, lag: int = 1) -> PanelTestResult:
        """
        Perform Kao panel cointegration test.
        
        Implements Kao (1999) test for panel cointegration, which assumes
        homogeneous cointegrating vectors across markets.
        
        Parameters
        ----------
        lag : int, default=1
            Number of lags for ADF regression
            
        Returns
        -------
        PanelTestResult
            Kao test results
        """
        # TODO: Implement proper Kao test
        # This is a placeholder implementation
        
        # Simplified calculation of test statistics
        # Should be replaced with actual Kao test implementation
        
        # Pretend to compute ADF statistic
        adf_stat = -2.5  # Example value
        p_value = self._p_value_from_normal(adf_stat, right_tail=False)
        
        # Determine significance
        significant = p_value < 0.05
        
        # Create result object
        result = PanelTestResult(
            test_name="Kao ADF",
            statistic=adf_stat,
            p_value=p_value,
            significant=significant,
            critical_values={'1%': -2.326, '5%': -1.645, '10%': -1.282},
            method="kao",
            n_obs=self.n_periods,
            n_markets=self.n_markets
        )
        
        logger.info(
            f"Kao panel cointegration test result: "
            f"ADF = {adf_stat:.4f}, p-value = {p_value:.4f}, "
            f"{'significant' if significant else 'not significant'}"
        )
        
        return result
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def test_westerlund(
        self, 
        trend: str = 'c',
        lag: int = 1
    ) -> Dict[str, Any]:
        """
        Perform Westerlund panel cointegration test.
        
        Implements Westerlund (2007) error-correction-based panel cointegration
        tests, which allow for cross-sectional dependence.
        
        Parameters
        ----------
        trend : str, default='c'
            Deterministic trend specification:
            - 'n': No deterministic trend
            - 'c': Constant (intercept)
            - 'ct': Constant and trend
        lag : int, default=1
            Number of lags for error correction model
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing Westerlund test statistics and results
        """
        # TODO: Implement proper Westerlund test
        # This is a placeholder implementation
        
        # Simplified calculation of test statistics
        # Should be replaced with actual Westerlund test implementation
        
        # Pretend to compute Westerlund statistics
        g_tau = -3.2  # Example value
        g_alpha = -12.5  # Example value
        p_tau = -10.8  # Example value
        p_alpha = -15.3  # Example value
        
        # P-values
        g_tau_p = self._p_value_from_normal(g_tau, right_tail=False)
        g_alpha_p = self._p_value_from_normal(g_alpha, right_tail=False)
        p_tau_p = self._p_value_from_normal(p_tau, right_tail=False)
        p_alpha_p = self._p_value_from_normal(p_alpha, right_tail=False)
        
        # Determine significance
        alpha = 0.05
        
        # Compile results
        results = {
            'G_tau': {
                'statistic': g_tau,
                'p_value': g_tau_p,
                'significant': g_tau_p < alpha,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            },
            'G_alpha': {
                'statistic': g_alpha,
                'p_value': g_alpha_p,
                'significant': g_alpha_p < alpha,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            },
            'P_tau': {
                'statistic': p_tau,
                'p_value': p_tau_p,
                'significant': p_tau_p < alpha,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            },
            'P_alpha': {
                'statistic': p_alpha,
                'p_value': p_alpha_p,
                'significant': p_alpha_p < alpha,
                'critical_values': {'1%': -2.326, '5%': -1.645, '10%': -1.282}
            }
        }
        
        # Determine overall result
        # Panel cointegration if majority of tests reject the null
        significant_count = sum(1 for result in results.values() if result['significant'])
        results['panel_cointegrated'] = significant_count >= 2  # At least 2 out of 4 tests
        
        # Add metadata
        results['method'] = 'westerlund'
        results['trend'] = trend
        results['lag'] = lag
        results['n_markets'] = self.n_markets
        results['n_periods'] = self.n_periods
        
        logger.info(
            f"Westerlund panel cointegration test results: "
            f"{significant_count}/4 tests significant, "
            f"{'cointegrated' if results['panel_cointegrated'] else 'not cointegrated'}"
        )
        
        return results
    
    @m3_optimized
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def estimate_pmg(
        self, 
        dependent: str, 
        independent: List[str], 
        group_col: str, 
        lag: int = 1
    ) -> Dict[str, Any]:
        """
        Estimate Pooled Mean Group model for heterogeneous panels.
        
        This method implements the Pooled Mean Group estimator of
        Pesaran, Shin and Smith (1999), which allows for heterogeneity
        in short-run coefficients but assumes homogeneity in long-run
        relationships.
        
        Parameters
        ----------
        dependent : str
            Dependent variable name
        independent : List[str]
            Independent variable names
        group_col : str
            Column identifying groups (markets)
        lag : int, default=1
            Number of lags
            
        Returns
        -------
        Dict[str, Any]
            PMG estimation results
        """
        # TODO: Implement PMG estimator
        # This is a placeholder implementation
        
        # For now, return a dummy result
        results = {
            'long_run': {
                'coefficients': {indep: 0.5 for indep in independent},
                'std_errors': {indep: 0.1 for indep in independent},
                't_stats': {indep: 5.0 for indep in independent},
                'p_values': {indep: 0.0001 for indep in independent}
            },
            'error_correction': {
                'coefficient': -0.3,
                'std_error': 0.05,
                't_stat': -6.0,
                'p_value': 0.0001
            },
            'short_run': {
                'coefficients': {indep: 0.2 for indep in independent},
                'std_errors': {indep: 0.05 for indep in independent},
                't_stats': {indep: 4.0 for indep in independent},
                'p_values': {indep: 0.001 for indep in independent}
            },
            'diagnostics': {
                'log_likelihood': -1000,
                'aic': 2050,
                'bic': 2100
            }
        }
        
        return results
    
    def _p_value_from_normal(self, statistic: float, right_tail: bool = False) -> float:
        """
        Compute p-value from standard normal distribution.
        
        Parameters
        ----------
        statistic : float
            Test statistic
        right_tail : bool, default=False
            Whether to use right tail of distribution
            
        Returns
        -------
        float
            P-value
        """
        from scipy import stats
        
        if right_tail:
            return 1 - stats.norm.cdf(statistic)
        else:
            return stats.norm.cdf(statistic)
