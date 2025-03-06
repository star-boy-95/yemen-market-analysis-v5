"""
Threshold cointegration module for market integration analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import statsmodels.api as sm
from arch.unitroot.cointegration import engle_granger

from src.utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_time_series, validate_model_inputs, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, disk_cache, parallelize,
    
    # Data processing
    fill_missing_values, create_lag_features,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values
DEFAULT_ALPHA = config.get('analysis.threshold.alpha', 0.05)
DEFAULT_TRIM = config.get('analysis.threshold.trim', 0.15)
DEFAULT_MAX_LAGS = config.get('analysis.threshold.max_lags', 4)


class ThresholdCointegration:
    """Threshold cointegration model implementation."""
    
    def __init__(
        self, 
        data1: Union[pd.Series, np.ndarray], 
        data2: Union[pd.Series, np.ndarray], 
        max_lags: int = DEFAULT_MAX_LAGS
    ):
        """
        Initialize the threshold cointegration model.
        
        Parameters
        ----------
        data1 : array_like
            First time series
        data2 : array_like
            Second time series
        max_lags : int, optional
            Maximum number of lags to consider
        """
        # Validate input time series
        self._validate_inputs(data1, data2)
        
        # Store validated inputs as numpy arrays
        self.data1 = np.asarray(data1)
        self.data2 = np.asarray(data2)
        
        # Validate max_lags
        valid, errors = validate_model_inputs(
            model_name="threshold",
            params={"max_lags": max_lags},
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
        
        logger.info(f"Initialized ThresholdCointegration with {len(data1)} observations")
    
    def _validate_inputs(self, data1, data2):
        """Validate input time series."""
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
    
    @disk_cache(cache_dir='.cache/threshold')
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def estimate_cointegration(self) -> Dict[str, Any]:
        """
        Estimate the cointegration relationship.
        
        Returns
        -------
        dict
            Cointegration test results
        """
        # Run Engle-Granger test
        result = engle_granger(self.data1, self.data2, trend='c', lags=self.max_lags)
        
        # Store the cointegration vector
        self.beta0 = result.coef[0]  # Intercept
        self.beta1 = result.coef[1]  # Slope
        
        # Calculate the equilibrium errors
        self.eq_errors = self.data1 - (self.beta0 + self.beta1 * self.data2)
        
        logger.info(
            f"Cointegration test: statistic={result.stat:.4f}, p-value={result.pvalue:.4f}, "
            f"beta0={self.beta0:.4f}, beta1={self.beta1:.4f}"
        )
        
        # Format result
        return {
            'statistic': result.stat,
            'pvalue': result.pvalue,
            'critical_values': result.critical_values,
            'cointegrated': result.pvalue < DEFAULT_ALPHA,
            'beta0': self.beta0,
            'beta1': self.beta1,
            'equilibrium_errors': self.eq_errors
        }
    
    @timer
    @memory_usage_decorator
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def estimate_threshold(
        self, 
        n_grid: int = 300, 
        trim: float = DEFAULT_TRIM
    ) -> Dict[str, Any]:
        """
        Estimate the threshold parameter using grid search.
        
        Parameters
        ----------
        n_grid : int, optional
            Number of grid points
        trim : float, optional
            Trimming percentage
            
        Returns
        -------
        dict
            Threshold estimation results
        """
        # Make sure we have cointegration results
        if self.eq_errors is None:
            logger.info("Running cointegration estimation first")
            self.estimate_cointegration()
        
        # Validate parameters
        valid, errors = validate_model_inputs(
            model_name="threshold",
            params={"n_grid": n_grid, "trim": trim},
            param_validators={
                "n_grid": lambda x: isinstance(x, int) and x > 0,
                "trim": lambda x: 0.0 < x < 0.5
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold estimation parameters")
        
        # Identify candidates for threshold
        sorted_errors = np.sort(self.eq_errors)
        lower_idx = int(len(sorted_errors) * trim)
        upper_idx = int(len(sorted_errors) * (1 - trim))
        candidates = sorted_errors[lower_idx:upper_idx]
        
        # Grid search for threshold
        if len(candidates) > n_grid:
            step = len(candidates) // n_grid
            candidates = candidates[::step]
        
        # Initialize variables for grid search
        best_ssr = np.inf
        best_threshold = None
        ssrs = []
        thresholds = []
        
        # Grid search
        logger.info(f"Starting grid search with {len(candidates)} threshold candidates")
        for threshold in candidates:
            ssr = self._compute_ssr_for_threshold(threshold)
            ssrs.append(ssr)
            thresholds.append(threshold)
            
            if ssr < best_ssr:
                best_ssr = ssr
                best_threshold = threshold
        
        self.threshold = best_threshold
        self.ssr = best_ssr
        
        logger.info(f"Threshold estimation complete: threshold={best_threshold:.4f}, ssr={best_ssr:.4f}")
        
        return {
            'threshold': best_threshold,
            'ssr': best_ssr,
            'all_thresholds': thresholds,
            'all_ssrs': ssrs
        }
    
    @timer
    @memory_usage_decorator
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def _compute_ssr_for_threshold(self, threshold: float) -> float:
        """
        Compute sum of squared residuals for a given threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value
            
        Returns
        -------
        float
            Sum of squared residuals
        """
        # Indicator function
        below = self.eq_errors <= threshold
        above = ~below
        
        # Prepare data for regression
        y = np.diff(self.data1)
        X1 = np.column_stack([
            self.eq_errors[:-1] * below[:-1],
            self.eq_errors[:-1] * above[:-1]
        ])
        
        # Add lagged differences using project utilities
        lag_diffs = create_lag_features(
            pd.DataFrame({
                'd1': np.diff(self.data1),
                'd2': np.diff(self.data2)
            }), 
            cols=['d1', 'd2'], 
            lags=list(range(1, min(self.max_lags + 1, len(y))))
        ).iloc[self.max_lags:].fillna(0).values
        
        # Combine features and add constant
        X1 = np.column_stack([X1, lag_diffs])
        X1 = sm.add_constant(X1)
        
        # Fit the model and return SSR
        return sm.OLS(y[self.max_lags:], X1).fit().ssr
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def estimate_tvecm(self) -> Dict[str, Any]:
        """
        Estimate the Threshold Vector Error Correction Model.
        
        Returns
        -------
        dict
            TVECM estimation results
        """
        # Make sure we have a threshold
        if self.threshold is None:
            logger.info("Estimating threshold first")
            self.estimate_threshold()
        
        # Estimate models for each regime
        results = self._estimate_regime_models()
        
        # Extract and format results
        self.results = self._format_tvecm_results(results)
        
        logger.info(
            f"TVECM estimation complete: threshold={self.threshold:.4f}, "
            f"adjustment_below_1={self.results['adjustment_below_1']:.4f}, "
            f"adjustment_above_1={self.results['adjustment_above_1']:.4f}"
        )
        
        return self.results
        
    @m1_optimized()
    def _estimate_regime_models(self) -> Dict[str, Any]:
        """Estimate OLS models for each regime."""
        # Indicator function
        below = self.eq_errors <= self.threshold
        above = ~below
        
        # Prepare data
        y1 = np.diff(self.data1)
        y2 = np.diff(self.data2)
        
        # Create design matrices using project utilities
        X = self._create_regime_design_matrix(below, above)
        
        # Fit models
        model1 = sm.OLS(y1[self.max_lags:], X)
        model2 = sm.OLS(y2[self.max_lags:], X)
        
        return {
            'results1': model1.fit(),
            'results2': model2.fit()
        }
    
    def _create_regime_design_matrix(self, below, above) -> np.ndarray:
        """Create design matrix for regime-specific models."""
        # Create regime-specific terms
        regime_terms = np.column_stack([
            self.eq_errors[:-1] * below[:-1],
            self.eq_errors[:-1] * above[:-1]
        ])
        
        # Add lagged differences using project utilities
        lag_diffs = create_lag_features(
            pd.DataFrame({
                'd1': np.diff(self.data1),
                'd2': np.diff(self.data2)
            }), 
            cols=['d1', 'd2'], 
            lags=list(range(1, min(self.max_lags + 1, len(self.data1)-1)))
        ).iloc[self.max_lags:].fillna(0).values
        
        # Combine and add constant
        X = np.column_stack([regime_terms[self.max_lags:], lag_diffs])
        return sm.add_constant(X)
    
    def _format_tvecm_results(self, model_results) -> Dict[str, Any]:
        """Extract and format TVECM results."""
        results1 = model_results['results1']
        results2 = model_results['results2']
        
        # Extract adjustment speeds
        adj_below_1 = results1.params[1]
        adj_above_1 = results1.params[2]
        adj_below_2 = results2.params[1]
        adj_above_2 = results2.params[2]
        
        return {
            'equation1': results1,
            'equation2': results2,
            'adjustment_below_1': adj_below_1,
            'adjustment_above_1': adj_above_1,
            'adjustment_below_2': adj_below_2,
            'adjustment_above_2': adj_above_2,
            'threshold': self.threshold,
            'cointegration_beta': self.beta1
        }


@timer
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def calculate_asymmetric_adjustment(
    model_results: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate asymmetric adjustment metrics from TVECM results.
    
    Parameters
    ----------
    model_results : dict
        Results from threshold_vecm estimation
        
    Returns
    -------
    dict
        Asymmetric adjustment metrics
    """
    # Extract adjustment parameters
    adj_below_1 = model_results['adjustment_below_1']
    adj_above_1 = model_results['adjustment_above_1']
    adj_below_2 = model_results['adjustment_below_2']
    adj_above_2 = model_results['adjustment_above_2']
    
    # Calculate half-lives
    if adj_below_1 < 0:
        half_life_below_1 = np.log(0.5) / np.log(1 + adj_below_1)
    else:
        half_life_below_1 = float('inf')
        
    if adj_above_1 < 0:
        half_life_above_1 = np.log(0.5) / np.log(1 + adj_above_1)
    else:
        half_life_above_1 = float('inf')
    
    # Calculate asymmetry measures
    asymmetry_1 = abs(adj_above_1) - abs(adj_below_1)
    asymmetry_2 = abs(adj_above_2) - abs(adj_below_2)
    
    # Calculate relative adjustment speeds
    if abs(adj_below_1) + abs(adj_below_2) > 0:
        rel_adj_below = abs(adj_below_1) / (abs(adj_below_1) + abs(adj_below_2))
    else:
        rel_adj_below = float('nan')
    
    if abs(adj_above_1) + abs(adj_above_2) > 0:
        rel_adj_above = abs(adj_above_1) / (abs(adj_above_1) + abs(adj_above_2))
    else:
        rel_adj_above = float('nan')
    
    return {
        'half_life_below_1': half_life_below_1,
        'half_life_above_1': half_life_above_1,
        'asymmetry_1': asymmetry_1,
        'asymmetry_2': asymmetry_2,
        'relative_adjustment_below': rel_adj_below,
        'relative_adjustment_above': rel_adj_above
    }