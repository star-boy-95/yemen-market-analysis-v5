"""
Unified threshold model implementation for Yemen market integration analysis.

This module provides a comprehensive implementation of threshold cointegration
models with support for different operational modes:

- 'standard': Standard threshold cointegration (TAR)
- 'fixed': Fixed threshold implementation with fallback mechanisms
- 'vecm': Threshold Vector Error Correction Model (Hansen & Seo)
- 'mtar': Momentum Threshold Autoregressive model

The unified approach simplifies the API, reduces code duplication, and
ensures consistent behavior across all threshold model variants.
"""
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Any, Optional, Union, List, Tuple
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller, coint
from datetime import datetime

from yemen_market_integration.utils.error_handler import handle_errors, ModelError, ValidationError, capture_error
from yemen_market_integration.utils.validation import validate_time_series, validate_model_inputs, validate_dataframe, raise_if_invalid
from yemen_market_integration.utils.performance_utils import (
    timer, m1_optimized, memory_usage_decorator, disk_cache, parallelize_dataframe,
    optimize_dataframe, configure_system_for_performance
)
from yemen_market_integration.utils.data_utils import fill_missing_values, create_lag_features
from yemen_market_integration.utils.stats_utils import (
    test_stationarity, test_cointegration, bootstrap_confidence_interval,
    test_linearity, calculate_half_life, test_structural_break, test_white_noise
)
from yemen_market_integration.utils.plotting_utils import (
    set_plotting_style, format_date_axis, plot_time_series, 
    plot_dual_axis, save_plot, add_annotations
)
from yemen_market_integration.utils.config import config

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values
DEFAULT_ALPHA = config.get('analysis.threshold.alpha', 0.05)
DEFAULT_TRIM = config.get('analysis.threshold.trim', 0.15)
DEFAULT_N_GRID = config.get('analysis.threshold.n_grid', 300)
DEFAULT_MAX_LAGS = config.get('analysis.threshold.max_lags', 4)
DEFAULT_N_BOOTSTRAP = config.get('analysis.threshold.n_bootstrap', 1000)
DEFAULT_MTAR_THRESHOLD = config.get('analysis.threshold.mtar_default_threshold', 0.0)
DEFAULT_VECM_K_AR_DIFF = config.get('analysis.threshold_vecm.k_ar_diff', 2)

# Define module-level functions for multiprocessing
def process_threshold(threshold_candidate, model_instance):
    """Process a single threshold candidate."""
    return (threshold_candidate, model_instance._compute_ssr_for_threshold(threshold_candidate))

def process_chunk(df_chunk, model_instance):
    """Process a chunk of threshold candidates."""
    return df_chunk.apply(
        lambda row: process_threshold(row['threshold'], model_instance), axis=1)


class ThresholdModel:
    """
    Unified threshold model implementation with multiple operational modes.
    
    This class provides a comprehensive implementation of threshold cointegration
    models with support for different operational modes:
    
    - 'standard': Standard threshold cointegration (TAR)
    - 'fixed': Fixed threshold implementation with fallback mechanisms
    - 'vecm': Threshold Vector Error Correction Model (Hansen & Seo)
    - 'mtar': Momentum Threshold Autoregressive model
    
    Parameters
    ----------
    data1 : array-like
        First time series (typically price series from first market)
    data2 : array-like
        Second time series (typically price series from second market)
    mode : str, optional
        Operational mode ('standard', 'fixed', 'vecm', or 'mtar')
    max_lags : int, optional
        Maximum number of lags to consider
    market1_name : str, optional
        Name of the first market (for plotting and reporting)
    market2_name : str, optional
        Name of the second market (for plotting and reporting)
    **kwargs : dict
        Additional parameters for specific modes
    """
    
    def __init__(
        self, 
        data1, 
        data2, 
        mode: str = "standard",
        max_lags: int = DEFAULT_MAX_LAGS,
        market1_name: str = "Market 1",
        market2_name: str = "Market 2",
        **kwargs
    ):
        """Initialize the threshold model."""
        # Validate mode
        valid_modes = ["standard", "fixed", "vecm", "mtar"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        
        self.mode = mode
        
        # Optimize system for performance
        configure_system_for_performance()
        
        # Validate inputs
        self._validate_inputs(data1, data2)
        
        # Store input time series
        if isinstance(data1, pd.Series):
            self.data1 = data1.copy()
            self.index = data1.index
        else:
            self.data1 = np.asarray(data1)
            self.index = None
        
        if isinstance(data2, pd.Series):
            self.data2 = data2.copy()
            if self.index is None:
                self.index = data2.index
        else:
            self.data2 = np.asarray(data2)
        
        # Store market names
        self.market1_name = market1_name
        self.market2_name = market2_name
        
        # Validate max_lags
        valid, errors = validate_model_inputs(
            model_name="threshold",
            params={"max_lags": max_lags},
            required_params={"max_lags"},
            param_validators={
                "max_lags": lambda x: isinstance(x, int) and x > 0
            }
        )
        raise_if_invalid(valid, errors, "Invalid max_lags parameter")
        
        self.max_lags = max_lags
        self.results = None
        self.beta0 = None
        self.beta1 = None
        self.eq_errors = None
        self.threshold = None
        self.ssr = None
        
        # VECM-specific attributes
        self.linear_model = None
        self.below_model = None
        self.above_model = None
        self.ec_term = None
        
        # Store additional parameters
        self.params = kwargs
        
        # Set mode-specific defaults
        if self.mode == "vecm":
            self.k_ar_diff = kwargs.get('k_ar_diff', DEFAULT_VECM_K_AR_DIFF)
            self.deterministic = kwargs.get('deterministic', "ci")
            self.coint_rank = kwargs.get('coint_rank', 1)
        
        logger.info(f"Initialized ThresholdModel in '{mode}' mode with {len(self.data1)} observations")
    
    def _validate_inputs(self, data1, data2):
        """
        Validate input time series.
        
        Parameters
        ----------
        data1 : array-like
            First time series
        data2 : array-like
            Second time series
            
        Raises
        ------
        ValidationError
            If inputs are invalid
        """
        for i, data in enumerate([data1, data2], 1):
            valid, errors = validate_time_series(
                data,
                min_length=30,
                max_nulls=0,
                check_constant=True
            )
            raise_if_invalid(valid, errors, f"Invalid time series for data{i}")
        
        # Check equal lengths
        if len(data1) != len(data2):
            raise ValidationError(
                f"Time series must have equal length, got {len(data1)} and {len(data2)}"
            )
    
    @disk_cache(cache_dir=".cache/yemen_market_integration/threshold")
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def estimate_cointegration(self) -> Dict[str, Any]:
        """
        Estimate the cointegration relationship.
        
        Tests for long-run equilibrium relationship between two market prices
        using the Engle-Granger procedure.
        
        Returns
        -------
        dict
            Cointegration test results including:
            - statistic: Test statistic value
            - pvalue: P-value of the test
            - critical_values: Critical values at different significance levels
            - cointegrated: Boolean indicating if markets are cointegrated
            - beta0: Intercept of cointegrating relationship
            - beta1: Slope coefficient
            - equilibrium_errors: Residuals representing deviations from equilibrium
            
        Notes
        -----
        In the Yemen market context, cointegration indicates price linkages between markets.
        The beta1 coefficient shows the long-run price transmission elasticity,
        while equilibrium errors represent deviations due to transaction costs and barriers.
        """
        # Use project's test_cointegration utility
        result = test_cointegration(
            self.data1, 
            self.data2, 
            method='engle-granger', 
            trend='c', 
            lags=self.max_lags,
            alpha=DEFAULT_ALPHA
        )
        
        # Store the cointegration vector from the results
        self.beta0 = result['beta'][0]  # Intercept
        self.beta1 = result['beta'][1]  # Slope
        
        # Calculate the equilibrium errors using the cointegration vector
        self.eq_errors = self.data1 - (self.beta0 + self.beta1 * self.data2)
        
        logger.info(
            f"Cointegration test: statistic={result['statistic']:.4f}, "
            f"p-value={result['pvalue']:.4f}, "
            f"beta0={self.beta0:.4f}, beta1={self.beta1:.4f}"
        )
        
        # Format result
        return {
            'statistic': result['statistic'],
            'pvalue': result['pvalue'],
            'critical_values': result['critical_values'],
            'cointegrated': result['cointegrated'],
            'beta0': self.beta0,
            'beta1': self.beta1,
            'equilibrium_errors': self.eq_errors,
            'long_run_relationship': f"{self.market1_name} = {self.beta0:.4f} + {self.beta1:.4f} × {self.market2_name}",
            'residuals': self.eq_errors
        }
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def estimate_threshold(
        self, 
        n_grid: int = None, 
        trim: float = None
    ) -> Dict[str, Any]:
        """
        Estimate the threshold parameter using grid search.
        
        The threshold represents transaction costs between markets. When price
        differentials exceed this threshold, arbitrage becomes profitable and
        prices adjust more quickly.
        
        Parameters
        ----------
        n_grid : int, optional
            Number of grid points to evaluate
        trim : float, optional
            Trimming percentage for excluding extreme values
            
        Returns
        -------
        dict
            Threshold estimation results including:
            - threshold: Optimal threshold value
            - ssr: Sum of squared residuals at optimal threshold
            - all_thresholds: All evaluated thresholds
            - all_ssrs: SSRs for all thresholds
            
        Notes
        -----
        In Yemen's fragmented markets, the threshold estimates transaction costs that
        include conflict-related barriers (checkpoints, security risks), transportation
        costs, and dual exchange rate effects.
        """
        # Use mode-specific defaults if not provided
        if n_grid is None:
            n_grid = DEFAULT_N_GRID if self.mode != "fixed" else 30
        
        if trim is None:
            trim = DEFAULT_TRIM
        
        # Dispatch to mode-specific implementation
        if self.mode == "standard":
            return self._estimate_standard_threshold(n_grid, trim)
        elif self.mode == "fixed":
            return self._estimate_fixed_threshold(n_grid, trim)
        elif self.mode == "vecm":
            return self._estimate_vecm_threshold(n_grid, trim)
        elif self.mode == "mtar":
            return self._estimate_mtar_threshold(n_grid, trim)
    
    def _estimate_standard_threshold(self, n_grid, trim):
        """
        Standard threshold estimation with robust multiprocessing.
        
        Parameters
        ----------
        n_grid : int
            Number of grid points to evaluate
        trim : float
            Trimming percentage for excluding extreme values
            
        Returns
        -------
        dict
            Threshold estimation results
        """
        # Validate parameters
        valid, errors = validate_model_inputs(
            model_name="threshold",
            params={"n_grid": n_grid, "trim": trim},
            required_params={"n_grid", "trim"},
            param_validators={
                "n_grid": lambda x: isinstance(x, int) and x > 0,
                "trim": lambda x: 0.0 < x < 0.5
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold estimation parameters")
        
        # Make sure we have cointegration results
        if self.eq_errors is None:
            logger.info("Running cointegration estimation first")
            self.estimate_cointegration()
        
        # Save trim for later use
        self.trim = trim
        
        # Get threshold candidates
        candidates = self._get_threshold_candidates(trim, n_grid)
        
        # Grid search with parallelization for better performance
        logger.info(f"Starting grid search with {len(candidates)} threshold candidates")
        
        try:
            # Try parallel processing with module-level functions
            df_candidates = pd.DataFrame({'threshold': candidates})
            results_df = parallelize_dataframe(
                df_candidates,
                lambda chunk: process_chunk(chunk, self)
            )
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            # Fall back to sequential processing
            results = []
            for threshold in candidates:
                ssr = self._compute_ssr_for_threshold(threshold)
                results.append((threshold, ssr))
            results_df = pd.DataFrame({0: results})
        
        # Process results
        best_threshold, best_ssr, thresholds, ssrs = self._process_threshold_results(results_df)
        
        self.threshold = best_threshold
        self.ssr = best_ssr
        
        logger.info(f"Threshold estimation complete: threshold={best_threshold:.4f}, ssr={best_ssr:.4f}")
        
        return {
            'threshold': best_threshold,
            'ssr': best_ssr,
            'all_thresholds': thresholds,
            'all_ssrs': ssrs,
            'proportion_below': np.mean(self.eq_errors <= best_threshold),
            'proportion_above': np.mean(self.eq_errors > best_threshold)
        }
    
    def _estimate_fixed_threshold(self, n_grid, trim):
        """
        Fixed threshold estimation with fallback to sequential processing.
        
        Parameters
        ----------
        n_grid : int
            Number of grid points to evaluate
        trim : float
            Trimming percentage for excluding extreme values
            
        Returns
        -------
        dict
            Threshold estimation results
        """
        # Validate inputs
        valid, errors = validate_model_inputs(
            model_name="ThresholdEstimation",
            params={
                "trim": trim,
                "n_grid": n_grid
            },
            required_params={"trim", "n_grid"},
            param_validators={
                "n_grid": lambda x: isinstance(x, int) and x > 0,
                "trim": lambda x: 0.0 < x < 0.5
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold estimation parameters")
        
        # Make sure we have cointegration results
        if self.eq_errors is None:
            logger.info("Running cointegration estimation first")
            self.estimate_cointegration()
        
        logger.info("Starting threshold estimation")
        
        # Get threshold candidates
        logger.info(f"Getting threshold candidates with trim={trim}, n_grid={n_grid}")
        candidates = self._get_threshold_candidates(trim, n_grid)
        
        # Grid search with sequential processing
        logger.info(f"Starting grid search with {len(candidates)} threshold candidates")
        
        # Use sequential processing by default for fixed mode
        results = []
        for threshold in candidates:
            try:
                ssr = self._compute_ssr_for_threshold(threshold)
                results.append((threshold, ssr))
            except Exception as e:
                logger.warning(f"Error computing SSR for threshold {threshold}: {e}")
                results.append((threshold, np.inf))
        
        # Create a DataFrame with a single column that contains tuples
        results_df = pd.DataFrame({0: results})
        logger.info(f"Sequential processing complete, got {len(results_df)} results")
        
        # Process results
        logger.info("Processing threshold results")
        best_threshold, best_ssr, thresholds, ssrs = self._process_threshold_results(results_df)
        logger.info(f"Found best threshold: {best_threshold:.4f} with SSR: {best_ssr:.4f}")
        
        self.threshold = best_threshold
        self.ssr = best_ssr
        
        logger.info(f"Threshold estimation complete: threshold={best_threshold:.4f}, ssr={best_ssr:.4f}")
        
        return {
            'threshold': best_threshold,
            'ssr': best_ssr,
            'all_thresholds': thresholds,
            'all_ssrs': ssrs,
            'proportion_below': np.mean(self.eq_errors <= best_threshold),
            'proportion_above': np.mean(self.eq_errors > best_threshold)
        }
    
    def _get_threshold_candidates(self, trim, n_grid, values=None):
        """
        Get candidate threshold values for grid search.
        
        Parameters
        ----------
        trim : float
            Trimming parameter to exclude extreme values
        n_grid : int
            Number of grid points
        values : array-like, optional
            Values to use for threshold candidates (defaults to eq_errors)
            
        Returns
        -------
        np.ndarray
            Array of threshold candidates
        """
        # Use provided values or equilibrium errors
        if values is None:
            if self.eq_errors is None:
                raise ModelError("Must run estimate_cointegration before estimating threshold")
            values = self.eq_errors
        
        # Sort values
        sorted_values = np.sort(values)
        
        # Determine range with trimming
        n = len(sorted_values)
        lower_idx = int(n * trim)
        upper_idx = int(n * (1 - trim))
        
        # Get candidates
        candidates = sorted_values[lower_idx:upper_idx]
        
        # If too many candidates, select a subset
        if len(candidates) > n_grid:
            step = len(candidates) // n_grid
            candidates = candidates[::step]
        
        return candidates
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _compute_ssr_for_threshold(self, threshold: float) -> float:
        """
        Compute sum of squared residuals for a given threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value to evaluate
            
        Returns
        -------
        float
            Sum of squared residuals
        """
        # Indicator function
        below = self.eq_errors <= threshold
        above = ~below
        
        # Count observations in each regime
        n_below = np.sum(below)
        n_above = np.sum(above)
        
        # Check if enough observations in each regime
        min_obs = 5
        if n_below < min_obs or n_above < min_obs:
            return np.inf
        
        # Prepare data for regression
        y = np.diff(self.data1)
        X1 = np.column_stack([
            self.eq_errors[:-1] * below[:-1],
            self.eq_errors[:-1] * above[:-1]
        ])
        
        # Add constant to X1
        X1 = sm.add_constant(X1)
        
        # Fit the model and return SSR
        return sm.OLS(y, X1).fit().ssr
    
    def _process_threshold_results(self, results_df) -> Tuple[float, float, List[float], List[float]]:
        """
        Process threshold grid search results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with threshold candidates and evaluation results
            
        Returns
        -------
        Tuple[float, float, List[float], List[float]]
            Best threshold, best evaluation metric, all thresholds, all metrics
        """
        best_metric = -np.inf if self.mode == "vecm" else np.inf
        best_threshold = None
        thresholds = []
        metrics = []
        
        for i, row in results_df.iterrows():
            threshold, metric = row[0]
            thresholds.append(threshold)
            metrics.append(metric)
            
            if self.mode == "vecm":
                # For VECM, we maximize likelihood
                if metric > best_metric:
                    best_metric = metric
                    best_threshold = threshold
            else:
                # For other modes, we minimize SSR
                if metric < best_metric:
                    best_metric = metric
                    best_threshold = threshold
        
        return best_threshold, best_metric, thresholds, metrics
    
    def generate_report(
        self, 
        format: str = "markdown", 
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a standardized report of model results.
        
        Parameters
        ----------
        format : str, optional
            Report format ('markdown', 'json', or 'latex')
        output_path : str, optional
            Path to save the report
            
        Returns
        -------
        dict
            Report content and metadata
        """
        # Ensure we have results
        if self.threshold is None:
            logger.info("Running full analysis first")
            self.run_full_analysis()
        
        # Create reporter instance
        from yemen_market_integration.models.threshold_reporter import ThresholdReporter
        reporter = ThresholdReporter(self, format, output_path)
        
        # Generate report
        return reporter.generate_report()
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete threshold analysis workflow.
        
        Performs the full analytical sequence based on the selected mode.
        
        Returns
        -------
        dict
            Complete analysis results
        """
        logger.info(f"Starting complete threshold analysis in '{self.mode}' mode")
        
        # Common steps for all modes
        coint_results = self.estimate_cointegration()
        
        # Check if series are cointegrated
        if not coint_results['cointegrated']:
            logger.warning("Series are not cointegrated, threshold estimation may not be valid")
        
        # Estimate threshold
        threshold_results = self.estimate_threshold()
        
        # Mode-specific analysis
        if self.mode == "standard":
            # Estimate TVECM
            tvecm_results = self.estimate_tvecm(run_diagnostics=True)
            
            # Compile results
            full_results = {
                'cointegration': coint_results,
                'threshold': threshold_results,
                'tvecm': tvecm_results,
                'summary': {
                    'cointegrated': coint_results['cointegrated'],
                    'threshold': self.threshold,
                    'mode': self.mode
                }
            }
        elif self.mode == "fixed":
            # Estimate TVECM with fixed approach
            tvecm_results = self.estimate_tvecm()
            
            # Compile results
            full_results = {
                'cointegration': coint_results,
                'threshold': threshold_results,
                'tvecm': tvecm_results,
                'summary': {
                    'cointegrated': coint_results['cointegrated'],
                    'threshold': self.threshold,
                    'mode': self.mode
                }
            }
        elif self.mode == "vecm":
            # Estimate linear VECM first
            linear_results = self.estimate_linear_vecm()
            
            # Estimate TVECM
            tvecm_results = self.estimate_tvecm(run_diagnostics=True)
            
            # Compile results
            full_results = {
                'cointegration': coint_results,
                'linear_vecm': linear_results,
                'threshold': threshold_results,
                'tvecm': tvecm_results,
                'summary': {
                    'cointegrated': coint_results['cointegrated'],
                    'threshold': self.threshold,
                    'mode': self.mode
                }
            }
        elif self.mode == "mtar":
            # Estimate M-TAR model
            mtar_results = self.estimate_mtar(run_diagnostics=True)
            
            # Compile results
            full_results = {
                'cointegration': coint_results,
                'threshold': threshold_results,
                'mtar': mtar_results,
                'summary': {
                    'cointegrated': coint_results['cointegrated'],
                    'threshold': self.threshold,
                    'mode': self.mode
                }
            }
        
        logger.info(f"Full analysis complete for mode '{self.mode}'")
        
        return full_results
        
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(Exception,), reraise=True)
    def estimate_tvecm(self, run_diagnostics=False):
        """
        Estimate Threshold Vector Error Correction Model (TVECM).
        
        Parameters
        ----------
        run_diagnostics : bool, optional
            Whether to run model diagnostics
            
        Returns
        -------
        dict
            TVECM estimation results
        """
        logger.info(f"Estimating TVECM for mode '{self.mode}'")
        
        # Make sure we have threshold results
        if self.threshold is None:
            logger.info("Estimating threshold first")
            self.estimate_threshold()
            
        if self.threshold is None:
            logger.warning("Could not estimate threshold, skipping TVECM estimation")
            return {
                'success': False,
                'error': 'No threshold available'
            }
            
        # Estimate regime models
        self.estimate_regime_models()
        
        # Compile results
        results = {
            'success': True,
            'threshold': self.threshold,
            'below_model': {
                'coefficients': self.below_model.params.tolist() if hasattr(self, 'below_model') else None,
                'ssr': self.below_model.ssr if hasattr(self, 'below_model') else None
            },
            'above_model': {
                'coefficients': self.above_model.params.tolist() if hasattr(self, 'above_model') else None,
                'ssr': self.above_model.ssr if hasattr(self, 'above_model') else None
            }
        }
        
        # Run diagnostics if requested
        if run_diagnostics and hasattr(self, 'below_model') and hasattr(self, 'above_model'):
            results['diagnostics'] = {
                'below': {
                    'r_squared': self.below_model.rsquared,
                    'adj_r_squared': self.below_model.rsquared_adj,
                    'aic': self.below_model.aic,
                    'bic': self.below_model.bic
                },
                'above': {
                    'r_squared': self.above_model.rsquared,
                    'adj_r_squared': self.above_model.rsquared_adj,
                    'aic': self.above_model.aic,
                    'bic': self.above_model.bic
                }
            }
            
        logger.info(f"TVECM estimation complete for mode '{self.mode}'")
        return results
        
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(Exception,), reraise=True)
    def estimate_regime_models(self):
        """
        Estimate separate models for each regime (below and above threshold).
        
        Returns
        -------
        tuple
            (below_model, above_model) - OLS model results for each regime
        """
        logger.info(f"Estimating regime models for threshold {self.threshold:.4f}")
        
        # Make sure we have threshold results
        if self.threshold is None:
            logger.info("Estimating threshold first")
            self.estimate_threshold()
            
        if self.threshold is None:
            logger.warning("Could not estimate threshold, skipping regime model estimation")
            return None, None
            
        # Create indicator variables for regimes
        below = self.eq_errors <= self.threshold
        above = ~below
        
        # Count observations in each regime
        n_below = np.sum(below)
        n_above = np.sum(above)
        
        logger.info(f"Observations in regimes: below={n_below}, above={n_above}")
        
        # Prepare data for regression
        y = np.diff(self.data1)
        
        # Below threshold model
        if n_below > 5:  # Minimum observations for regression
            X_below = sm.add_constant(self.eq_errors[:-1][below[:-1]])
            self.below_model = sm.OLS(y[below[:-1]], X_below).fit()
            # Get the adjustment coefficient (second parameter, index might not be 1)
            adjustment_below = self.below_model.params.iloc[1] if len(self.below_model.params) > 1 else 0
            logger.info(f"Below threshold model: adjustment={adjustment_below:.4f}")
        else:
            logger.warning(f"Not enough observations below threshold ({n_below})")
            self.below_model = None
            
        # Above threshold model
        if n_above > 5:  # Minimum observations for regression
            X_above = sm.add_constant(self.eq_errors[:-1][above[:-1]])
            self.above_model = sm.OLS(y[above[:-1]], X_above).fit()
            # Get the adjustment coefficient (second parameter, index might not be 1)
            adjustment_above = self.above_model.params.iloc[1] if len(self.above_model.params) > 1 else 0
            logger.info(f"Above threshold model: adjustment={adjustment_above:.4f}")
        else:
            logger.warning(f"Not enough observations above threshold ({n_above})")
            self.above_model = None
            
        return self.below_model, self.above_model
        
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(Exception,), reraise=True)
    def estimate_linear_vecm(self):
        """
        Estimate linear Vector Error Correction Model (VECM).
        
        Returns
        -------
        dict
            Linear VECM estimation results
        """
        logger.info("Estimating linear VECM")
        
        # Make sure we have cointegration results
        if self.eq_errors is None:
            logger.info("Running cointegration estimation first")
            self.estimate_cointegration()
            
        # Prepare data
        y1 = np.diff(self.data1)
        y2 = np.diff(self.data2)
        
        # Create design matrices
        X = sm.add_constant(self.eq_errors[:-1])
        
        # Fit models for each equation
        model1 = sm.OLS(y1, X).fit()
        model2 = sm.OLS(y2, X).fit()
        
        # Extract adjustment speeds
        alpha1 = model1.params[1]
        alpha2 = model2.params[1]
        
        logger.info(f"Linear VECM adjustment speeds: alpha1={alpha1:.4f}, alpha2={alpha2:.4f}")
        
        # Compile results
        results = {
            'alpha1': alpha1,
            'alpha2': alpha2,
            'model1': {
                'coefficients': model1.params.tolist(),
                'std_errors': model1.bse.tolist(),
                'r_squared': model1.rsquared,
                'ssr': model1.ssr
            },
            'model2': {
                'coefficients': model2.params.tolist(),
                'std_errors': model2.bse.tolist(),
                'r_squared': model2.rsquared,
                'ssr': model2.ssr
            }
        }
        
        return results
        
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(Exception,), reraise=True)
    def estimate_mtar(self, run_diagnostics=False):
        """
        Estimate Momentum Threshold Autoregressive (M-TAR) model.
        
        Parameters
        ----------
        run_diagnostics : bool, optional
            Whether to run model diagnostics
            
        Returns
        -------
        dict
            M-TAR estimation results
        """
        logger.info("Estimating M-TAR model")
        
        # Make sure we have cointegration results
        if self.eq_errors is None:
            logger.info("Running cointegration estimation first")
            self.estimate_cointegration()
            
        # Calculate momentum (changes in residuals)
        momentum = np.diff(self.eq_errors)
        
        # Use threshold from threshold estimation or default to 0
        threshold = self.threshold if self.threshold is not None else 0.0
        
        # Create indicator variables for regimes
        positive = momentum > threshold
        negative = ~positive
        
        # Count observations in each regime
        n_positive = np.sum(positive)
        n_negative = np.sum(negative)
        
        logger.info(f"Observations in regimes: positive={n_positive}, negative={n_negative}")
        
        # Prepare data for regression
        y = np.diff(momentum)  # Second difference
        
        # Positive momentum model
        if n_positive > 5:  # Minimum observations for regression
            X_positive = sm.add_constant(self.eq_errors[1:-1][positive[:-1]])
            positive_model = sm.OLS(y[positive[:-1]], X_positive).fit()
            logger.info(f"Positive momentum model: adjustment={positive_model.params[1]:.4f}")
        else:
            logger.warning(f"Not enough observations with positive momentum ({n_positive})")
            positive_model = None
            
        # Negative momentum model
        if n_negative > 5:  # Minimum observations for regression
            X_negative = sm.add_constant(self.eq_errors[1:-1][negative[:-1]])
            negative_model = sm.OLS(y[negative[:-1]], X_negative).fit()
            logger.info(f"Negative momentum model: adjustment={negative_model.params[1]:.4f}")
        else:
            logger.warning(f"Not enough observations with negative momentum ({n_negative})")
            negative_model = None
            
        # Compile results
        results = {
            'threshold': threshold,
            'positive_model': {
                'coefficients': positive_model.params.tolist() if positive_model is not None else None,
                'ssr': positive_model.ssr if positive_model is not None else None
            },
            'negative_model': {
                'coefficients': negative_model.params.tolist() if negative_model is not None else None,
                'ssr': negative_model.ssr if negative_model is not None else None
            }
        }
        
        # Run diagnostics if requested
        if run_diagnostics:
            results['diagnostics'] = {
                'positive': {
                    'r_squared': positive_model.rsquared if positive_model is not None else None,
                    'adj_r_squared': positive_model.rsquared_adj if positive_model is not None else None,
                    'aic': positive_model.aic if positive_model is not None else None,
                    'bic': positive_model.bic if positive_model is not None else None
                },
                'negative': {
                    'r_squared': negative_model.rsquared if negative_model is not None else None,
                    'adj_r_squared': negative_model.rsquared_adj if negative_model is not None else None,
                    'aic': negative_model.aic if negative_model is not None else None,
                    'bic': negative_model.bic if negative_model is not None else None
                }
            }
            
        logger.info("M-TAR estimation complete")
        return results
        
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(Exception,), reraise=True)
    def calculate_half_lives(self):
        """
        Calculate half-lives of deviations from equilibrium.
        
        Returns
        -------
        dict
            Half-life calculation results
        """
        logger.info("Calculating half-lives")
        
        # Make sure we have regime models
        if not hasattr(self, 'below_model') or not hasattr(self, 'above_model'):
            logger.info("Estimating regime models first")
            self.estimate_regime_models()
            
        # Initialize results
        half_lives = {
            'below': float('inf'),
            'above': float('inf'),
            'ratio': float('inf'),
            'average': {
                'below': float('inf'),
                'above': float('inf')
            }
        }
        
        # Calculate half-life for below threshold regime
        if self.below_model is not None:
            # Get the adjustment coefficient (second parameter, index might not be 1)
            adjustment_below = self.below_model.params.iloc[1] if len(self.below_model.params) > 1 else 0
            if adjustment_below < 0:
                half_lives['below'] = np.log(0.5) / np.log(1 + adjustment_below)
                half_lives['average']['below'] = half_lives['below']
                logger.info(f"Below threshold half-life: {half_lives['below']:.4f}")
            else:
                logger.warning("Adjustment coefficient for below threshold is positive or zero, half-life is infinite")
                
        # Calculate half-life for above threshold regime
        if self.above_model is not None:
            # Get the adjustment coefficient (second parameter, index might not be 1)
            adjustment_above = self.above_model.params.iloc[1] if len(self.above_model.params) > 1 else 0
            if adjustment_above < 0:
                half_lives['above'] = np.log(0.5) / np.log(1 + adjustment_above)
                half_lives['average']['above'] = half_lives['above']
                logger.info(f"Above threshold half-life: {half_lives['above']:.4f}")
            else:
                logger.warning("Adjustment coefficient for above threshold is positive or zero, half-life is infinite")
                
        # Calculate half-life ratio
        if half_lives['below'] != float('inf') and half_lives['above'] != float('inf'):
            if half_lives['above'] != 0:
                half_lives['ratio'] = half_lives['below'] / half_lives['above']
                logger.info(f"Half-life ratio (below/above): {half_lives['ratio']:.4f}")
            else:
                logger.warning("Above threshold half-life is zero, ratio is infinite")
        else:
            logger.warning("At least one half-life is infinite, ratio is undefined")
            
        return half_lives
        
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(Exception,), reraise=True)
    def test_asymmetric_adjustment(self):
        """
        Test for asymmetric adjustment in threshold model.
        
        Returns
        -------
        dict
            Asymmetric adjustment test results
        """
        logger.info("Testing for asymmetric adjustment")
        
        # Make sure we have regime models
        if not hasattr(self, 'below_model') or not hasattr(self, 'above_model'):
            logger.info("Estimating regime models first")
            self.estimate_regime_models()
            
        # Initialize results
        results = {
            'asymmetric': False,
            'p_value': 1.0,
            'f_statistic': 0.0,
            'adjustment_below': None,
            'adjustment_above': None
        }
        
        # Check if we have valid models
        if self.below_model is None or self.above_model is None:
            logger.warning("Cannot test for asymmetric adjustment, missing regime models")
            return results
            
        # Extract adjustment speeds
        adjustment_below = self.below_model.params.iloc[1] if len(self.below_model.params) > 1 else 0
        adjustment_above = self.above_model.params.iloc[1] if len(self.above_model.params) > 1 else 0
        
        results['adjustment_below'] = adjustment_below
        results['adjustment_above'] = adjustment_above
        
        # Calculate asymmetry
        asymmetry = adjustment_above - adjustment_below
        results['asymmetry'] = asymmetry
        
        logger.info(f"Adjustment speeds: below={adjustment_below:.4f}, above={adjustment_above:.4f}, asymmetry={asymmetry:.4f}")
        
        # Prepare data for regression
        y = np.diff(self.data1)
        
        # Create indicator variables for regimes
        below = self.eq_errors[:-1] <= self.threshold
        above = ~below
        
        # Count observations in each regime
        n_below = np.sum(below)
        n_above = np.sum(above)
        
        results['n_below'] = n_below
        results['n_above'] = n_above
        
        # Create design matrix for unrestricted model
        X_unrestricted = np.column_stack([
            np.ones(len(y)),
            self.eq_errors[:-1] * below,
            self.eq_errors[:-1] * above
        ])
        
        # Create design matrix for restricted model
        X_restricted = np.column_stack([
            np.ones(len(y)),
            self.eq_errors[:-1]
        ])
        
        # Fit unrestricted model
        unrestricted_model = sm.OLS(y, X_unrestricted).fit()
        
        # Fit restricted model
        restricted_model = sm.OLS(y, X_restricted).fit()
        
        # Calculate F-statistic
        ssr_restricted = restricted_model.ssr
        ssr_unrestricted = unrestricted_model.ssr
        
        # Degrees of freedom
        df1 = 1  # Number of restrictions
        df2 = len(y) - 3  # Degrees of freedom for unrestricted model
        
        # Calculate F-statistic
        f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
        
        # Calculate p-value
        from scipy import stats
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        results['f_statistic'] = f_stat
        results['p_value'] = p_value
        results['asymmetric'] = p_value < 0.05
        
        logger.info(f"Asymmetric adjustment test: F={f_stat:.4f}, p-value={p_value:.4f}, asymmetric={results['asymmetric']}")
        
        return results
