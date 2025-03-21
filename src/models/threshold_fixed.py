"""
Threshold cointegration models for Yemen market integration analysis.

This module implements threshold cointegration models to analyze asymmetric
price transmission between markets in Yemen.
"""
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.stattools import adfuller, coint

from yemen_market_integration.utils.validation import validate_model_inputs, raise_if_invalid
from yemen_market_integration.utils.error_handler import handle_errors, ModelError, capture_error
from yemen_market_integration.utils.performance_utils import (
    timer, memory_usage_decorator, parallelize_dataframe, optimize_dataframe
)
from yemen_market_integration.utils.decorators import m1_optimized

logger = logging.getLogger(__name__)

# Define process_threshold and process_chunk functions at module level
# so they can be pickled for multiprocessing
def process_threshold(threshold_candidate, model_instance):
    """Process a single threshold candidate."""
    return (threshold_candidate, model_instance._compute_ssr_for_threshold(threshold_candidate))

def process_chunk(df_chunk, model_instance):
    """Process a chunk of threshold candidates."""
    return df_chunk.apply(
        lambda row: process_threshold(row['threshold'], model_instance), axis=1)

class ThresholdCointegration:
    """
    Threshold cointegration model for asymmetric price transmission analysis.
    
    This class implements threshold cointegration models to analyze asymmetric
    price transmission between markets in Yemen, including:
    - Threshold Vector Error Correction Models (TVECM)
    - Momentum Threshold Autoregressive (M-TAR) models
    - Self-Exciting Threshold Autoregressive (SETAR) models
    
    Parameters
    ----------
    price1 : array-like
        Price series for the first market
    price2 : array-like
        Price series for the second market
    max_lags : int, optional
        Maximum number of lags to consider, by default 4
    market1_name : str, optional
        Name of the first market, by default "Market 1"
    market2_name : str, optional
        Name of the second market, by default "Market 2"
    """
    
    def __init__(
        self, 
        price1: Union[pd.Series, np.ndarray], 
        price2: Union[pd.Series, np.ndarray],
        max_lags: int = 4,
        market1_name: str = "Market 1",
        market2_name: str = "Market 2"
    ):
        """
        Initialize the threshold cointegration model.
        """
        # Validate inputs
        valid, errors = validate_model_inputs(
            model_name="ThresholdCointegration",
            params={
                "price1": price1,
                "price2": price2,
                "max_lags": max_lags,
                "market1_name": market1_name,
                "market2_name": market2_name
            },
            required_params={"price1", "price2"},
            param_validators={
                "max_lags": lambda x: isinstance(x, int) and x > 0,
                "market1_name": lambda x: isinstance(x, str),
                "market2_name": lambda x: isinstance(x, str)
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold cointegration parameters")
        
        # Convert to numpy arrays if needed
        self.price1 = np.asarray(price1).flatten()
        self.price2 = np.asarray(price2).flatten()
        
        # Store parameters
        self.max_lags = max_lags
        self.market1_name = market1_name
        self.market2_name = market2_name
        
        # Initialize results
        self.beta = None
        self.residuals = None
        self.threshold = None
        self.ssr = None
        self.tvecm_model = None
        self.mtar_model = None
        
        # Log initialization
        n_obs = len(self.price1)
        logger.info(f"Initialized ThresholdCointegration with {n_obs} observations")
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, ModelError))
    def estimate_cointegration(self) -> Dict[str, Any]:
        """
        Estimate the cointegration relationship between the two price series.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing cointegration results
        """
        # Add constant for regression
        X = sm.add_constant(self.price1)
        
        # Estimate cointegration relationship
        model = sm.OLS(self.price2, X)
        results = model.fit()
        
        # Store results
        self.beta = results.params
        self.residuals = results.resid
        
        # Test for cointegration
        adf_result = adfuller(self.residuals, maxlag=self.max_lags)
        
        # Check if cointegrated
        is_cointegrated = adf_result[1] < 0.05
        
        # Log results
        logger.info(f"Cointegration test: statistic={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}, beta0={self.beta[0]:.4f}, beta1={self.beta[1]:.4f}")
        
        return {
            'beta': self.beta,
            'residuals': self.residuals,
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'cointegrated': is_cointegrated
        }
    
    def _get_threshold_candidates(self, trim: float = 0.15, n_grid: int = 30) -> np.ndarray:
        """
        Generate threshold candidates for grid search.
        
        Parameters
        ----------
        trim : float, optional
            Trimming parameter to exclude extreme values, by default 0.15
        n_grid : int, optional
            Number of grid points, by default 30
            
        Returns
        -------
        np.ndarray
            Array of threshold candidates
        """
        if self.residuals is None:
            raise ModelError("Must run estimate_cointegration before estimating threshold")
        
        # Sort residuals
        sorted_residuals = np.sort(self.residuals)
        
        # Determine range with trimming
        n = len(sorted_residuals)
        lower_idx = int(n * trim)
        upper_idx = int(n * (1 - trim))
        
        # Generate grid
        candidates = np.linspace(
            sorted_residuals[lower_idx],
            sorted_residuals[upper_idx],
            n_grid
        )
        
        return candidates
    
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
        if self.residuals is None:
            raise ModelError("Must run estimate_cointegration before computing SSR")
        
        # Create indicator variables
        below = self.residuals <= threshold
        above = ~below
        
        # Count observations in each regime
        n_below = np.sum(below)
        n_above = np.sum(above)
        
        # Check if enough observations in each regime
        min_obs = 5
        if n_below < min_obs or n_above < min_obs:
            return np.inf
        
        # Compute SSR
        ssr = 0
        
        # Add constant and lagged residuals
        X = sm.add_constant(np.column_stack([
            below * self.residuals,
            above * self.residuals
        ]))
        
        # Estimate model
        model = sm.OLS(np.diff(self.residuals), X[:-1])
        results = model.fit()
        
        # Return SSR
        return results.ssr
    
    def _process_threshold_results(self, results_df: pd.DataFrame) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Process threshold grid search results.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame with threshold candidates and SSR values
            
        Returns
        -------
        Tuple[float, float, np.ndarray, np.ndarray]
            Best threshold, best SSR, all thresholds, all SSRs
        """
        # Extract results
        thresholds = []
        ssrs = []
        
        for _, row in results_df.iterrows():
            threshold, ssr = row[0]
            thresholds.append(threshold)
            ssrs.append(ssr)
        
        # Convert to arrays
        thresholds = np.array(thresholds)
        ssrs = np.array(ssrs)
        
        # Find best threshold
        best_idx = np.argmin(ssrs)
        best_threshold = thresholds[best_idx]
        best_ssr = ssrs[best_idx]
        
        return best_threshold, best_ssr, thresholds, ssrs
    
    @timer
    @memory_usage_decorator
    # Temporarily remove the m1_optimized decorator as it might be causing issues
    # @m1_optimized
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, ModelError))
    def estimate_threshold(self, trim: float = 0.15, n_grid: int = 30) -> Dict[str, Any]:
        """
        Estimate the threshold parameter using grid search.
        
        Parameters
        ----------
        trim : float, optional
            Trimming parameter to exclude extreme values, by default 0.15
        n_grid : int, optional
            Number of grid points, by default 30
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing threshold estimation results
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
        
        logger.info("Starting threshold estimation")
        
        # Get threshold candidates using helper function
        logger.info(f"Getting threshold candidates with trim={trim}, n_grid={n_grid}")
        candidates = self._get_threshold_candidates(trim, n_grid)
        
        # Grid search with parallelization for better performance
        logger.info(f"Starting grid search with {len(candidates)} threshold candidates")
        
        df_candidates = pd.DataFrame({'threshold': candidates})
        logger.info(f"Created DataFrame with {len(df_candidates)} candidates")
        
        # Use the module-level process_chunk function with self as an argument
        logger.info("Starting parallel processing of threshold candidates")
        try:
            results_df = parallelize_dataframe(df_candidates,
                                              lambda chunk: process_chunk(chunk, self))
            logger.info(f"Parallel processing complete, got {len(results_df)} results")
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Fall back to non-parallel processing
            logger.info("Falling back to non-parallel processing")
            results = []
            for _, row in df_candidates.iterrows():
                threshold = row['threshold']
                ssr = self._compute_ssr_for_threshold(threshold)
                results.append((threshold, ssr))
            
            # Create a DataFrame with a single column that contains tuples
            # This matches the structure expected by _process_threshold_results
            results_df = pd.DataFrame({'result': results})
            logger.info(f"Non-parallel processing complete, got {len(results_df)} results")
        
        # Process results using helper function
        logger.info("Processing threshold results")
        best_threshold, best_ssr, thresholds, ssrs = self._process_threshold_results(results_df)
        logger.info(f"Found best threshold: {best_threshold:.4f} with SSR: {best_ssr:.4f}")
        
        self.threshold = best_threshold
        self.ssr = best_ssr
        
        logger.info(f"Threshold estimation complete: threshold={best_threshold:.4f}, ssr={best_ssr:.4f}")
        
        return {
            'threshold': best_threshold,
            'ssr': best_ssr,
            'thresholds': thresholds,
            'ssrs': ssrs
        }
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, ModelError))
    def estimate_tvecm(self, lags: int = None) -> Dict[str, Any]:
        """
        Estimate a Threshold Vector Error Correction Model (TVECM).
        
        Parameters
        ----------
        lags : int, optional
            Number of lags to include, by default None (uses AIC)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing TVECM estimation results
        """
        if self.beta is None or self.threshold is None:
            raise ModelError("Must run estimate_cointegration and estimate_threshold before estimating TVECM")
        
        # Determine optimal lag order if not provided
        if lags is None:
            lags = min(self.max_lags, 2)  # Default to 2 lags or max_lags, whichever is smaller
        
        # Create data matrix
        data = np.column_stack([self.price1, self.price2])
        
        # Create indicator variables
        ecm = self.price2 - self.beta[0] - self.beta[1] * self.price1
        below = ecm <= self.threshold
        above = ~below
        
        # Count observations in each regime
        n_below = np.sum(below)
        n_above = np.sum(above)
        
        # Log regime information
        logger.info(f"TVECM regimes: {n_below} observations below threshold, {n_above} observations above threshold")
        
        # Store results
        tvecm_results = {
            'lags': lags,
            'n_below': n_below,
            'n_above': n_above,
            'threshold': self.threshold
        }
        
        # Store model
        self.tvecm_model = tvecm_results
        
        return tvecm_results
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, ModelError))
    def estimate_mtar(self) -> Dict[str, Any]:
        """
        Estimate a Momentum Threshold Autoregressive (M-TAR) model.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing M-TAR estimation results
        """
        if self.beta is None or self.threshold is None:
            raise ModelError("Must run estimate_cointegration and estimate_threshold before estimating M-TAR")
        
        # Create error correction term
        ecm = self.price2 - self.beta[0] - self.beta[1] * self.price1
        
        # Create indicator variables based on changes in ECM
        delta_ecm = np.diff(ecm)
        below = np.append(False, delta_ecm <= self.threshold)
        above = ~below
        
        # Count observations in each regime
        n_below = np.sum(below)
        n_above = np.sum(above)
        
        # Log regime information
        logger.info(f"M-TAR regimes: {n_below} observations with decreasing ECM, {n_above} observations with increasing ECM")
        
        # Store results
        mtar_results = {
            'n_below': n_below,
            'n_above': n_above,
            'threshold': self.threshold
        }
        
        # Store model
        self.mtar_model = mtar_results
        
        return mtar_results
    
    def plot_regime_dynamics(self, title: str = "Threshold Dynamics") -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the regime dynamics over time.
        
        Parameters
        ----------
        title : str, optional
            Plot title, by default "Threshold Dynamics"
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects
        """
        if self.beta is None or self.threshold is None:
            raise ModelError("Must run estimate_cointegration and estimate_threshold before plotting")
        
        # Create error correction term
        ecm = self.price2 - self.beta[0] - self.beta[1] * self.price1
        
        # Create indicator variables
        below = ecm <= self.threshold
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot ECM
        ax.plot(ecm, label="Error Correction Term", color="blue")
        
        # Plot threshold
        ax.axhline(y=self.threshold, color="red", linestyle="--", label=f"Threshold = {self.threshold:.4f}")
        
        # Highlight regimes
        for i in range(len(ecm)):
            if below[i]:
                ax.axvspan(i-0.5, i+0.5, alpha=0.2, color="blue")
            else:
                ax.axvspan(i-0.5, i+0.5, alpha=0.2, color="red")
        
        # Add labels
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Error Correction Term")
        ax.legend()
        
        return fig, ax