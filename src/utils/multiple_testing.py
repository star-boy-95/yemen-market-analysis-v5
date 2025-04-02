"""
Multiple testing correction utilities for Yemen Market Integration analysis.

This module provides functions for controlling family-wise error rate (FWER)
or false discovery rate (FDR) when performing multiple hypothesis tests,
which is critical when analyzing many market pairs simultaneously.
"""
import numpy as np
import pandas as pd
import logging
from typing import Union, List, Tuple, Dict, Optional, Any
import warnings

from yemen_market_integration.utils.error_handler import handle_errors
from yemen_market_integration.utils.m3_utils import m3_optimized

# Initialize module logger
logger = logging.getLogger(__name__)


@m3_optimized
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def apply_multiple_testing_correction(
    p_values: Union[List[float], np.ndarray, pd.Series],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Tuple[Union[List[bool], np.ndarray], Union[List[float], np.ndarray]]:
    """
    Apply multiple testing correction to p-values.
    
    Implements various correction methods for controlling
    family-wise error rate or false discovery rate when
    performing multiple hypothesis tests.
    
    Parameters
    ----------
    p_values : array-like
        List, array, or Series of p-values to correct
    method : str, default='fdr_bh'
        Correction method to use:
        - 'bonferroni': Bonferroni correction (FWER)
        - 'holm': Holm-Bonferroni step-down method (FWER)
        - 'fdr_bh': Benjamini-Hochberg FDR correction
        - 'fdr_by': Benjamini-Yekutieli FDR correction
        - 'holm-sidak': Holm-Sidak step-down method (FWER)
        - 'sidak': Sidak correction (FWER)
    alpha : float, default=0.05
        Significance level
        
    Returns
    -------
    Tuple[array-like, array-like]
        (reject, corrected_p_values) where:
        - reject is a boolean array indicating which tests to reject
        - corrected_p_values are the corrected p-values
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        logger.error("statsmodels is required for multiple testing correction")
        raise ImportError("statsmodels is required for multiple testing correction")
    
    # Convert to numpy array
    if isinstance(p_values, pd.Series):
        original_index = p_values.index
        p_values = p_values.values
    else:
        original_index = None
    
    if not isinstance(p_values, np.ndarray):
        p_values = np.array(p_values)
    
    # Check for valid p-values
    if np.any((p_values < 0) | (p_values > 1)):
        logger.warning("Some p-values are outside the valid range [0, 1]")
        p_values = np.clip(p_values, 0, 1)
    
    # Handle NaN values if present
    nan_mask = np.isnan(p_values)
    if np.any(nan_mask):
        logger.warning(f"Found {np.sum(nan_mask)} NaN p-values, will be treated as non-significant")
        p_values_no_nan = p_values[~nan_mask]
    else:
        p_values_no_nan = p_values
    
    # Check for valid method
    valid_methods = ['bonferroni', 'holm', 'fdr_bh', 'fdr_by', 'holm-sidak', 'sidak']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got {method}")
    
    # Apply correction to non-NaN values
    if len(p_values_no_nan) > 0:
        reject, corrected_p, _, _ = multipletests(p_values_no_nan, alpha=alpha, method=method)
        
        # Reconstruct full arrays if there were NaNs
        if np.any(nan_mask):
            full_reject = np.zeros(len(p_values), dtype=bool)
            full_corrected_p = np.ones(len(p_values)) * np.nan
            
            full_reject[~nan_mask] = reject
            full_corrected_p[~nan_mask] = corrected_p
            
            reject = full_reject
            corrected_p = full_corrected_p
    else:
        # All p-values are NaN
        reject = np.zeros(len(p_values), dtype=bool)
        corrected_p = np.ones(len(p_values)) * np.nan
    
    # Return as pandas Series if input was a Series
    if original_index is not None:
        reject = pd.Series(reject, index=original_index)
        corrected_p = pd.Series(corrected_p, index=original_index)
    
    return reject, corrected_p


@m3_optimized
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def apply_multiple_testing_to_results(
    results: Dict[str, Dict[str, Any]],
    p_value_key_path: List[str],
    method: str = 'fdr_bh',
    alpha: float = 0.05,
    significance_key_path: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Apply multiple testing correction to p-values in a results dictionary.
    
    This function is designed to work with the nested dictionary structures
    typically returned by the analysis modules, where each key is a market pair
    and the value is a dictionary of results.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Dictionary of results, keyed by market pair
    p_value_key_path : List[str]
        Path to the p-value in each result dictionary, e.g. ['cointegration', 'p_value']
    method : str, default='fdr_bh'
        Correction method (see apply_multiple_testing_correction for options)
    alpha : float, default=0.05
        Significance level
    significance_key_path : List[str], optional
        Path to the significance flag in each result dictionary
        If provided, this will be updated based on the corrected p-values
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Updated results dictionary with corrected p-values and significance flags
    """
    # Extract p-values from results
    p_values = {}
    market_pairs = []
    
    for market_pair, result in results.items():
        # Navigate to the p-value through the key path
        current = result
        valid_path = True
        
        for key in p_value_key_path:
            if key not in current:
                logger.warning(f"Key path {p_value_key_path} not found for {market_pair}")
                valid_path = False
                break
            current = current[key]
        
        if valid_path:
            p_values[market_pair] = current
            market_pairs.append(market_pair)
    
    # Convert to array for correction
    p_array = np.array([p_values[pair] for pair in market_pairs])
    
    # Apply correction
    reject, corrected_p = apply_multiple_testing_correction(
        p_array, method=method, alpha=alpha
    )
    
    # Update results with corrected p-values and significance
    for i, market_pair in enumerate(market_pairs):
        # Create a copy of the result to avoid modifying the original
        result = results[market_pair]
        
        # Navigate to the p-value's parent dictionary
        current = result
        parent_key = p_value_key_path[-1]
        for key in p_value_key_path[:-1]:
            current = current[key]
        
        # Update p-value with corrected value
        current[f"corrected_{parent_key}"] = corrected_p[i]
        current[f"passed_correction"] = bool(reject[i])
        
        # Update significance flag if provided
        if significance_key_path:
            # Navigate to significance flag
            sig_current = result
            sig_parent_key = significance_key_path[-1]
            valid_sig_path = True
            
            for key in significance_key_path[:-1]:
                if key not in sig_current:
                    valid_sig_path = False
                    break
                sig_current = sig_current[key]
            
            if valid_sig_path and sig_parent_key in sig_current:
                # Store original significance
                sig_current[f"original_{sig_parent_key}"] = sig_current[sig_parent_key]
                # Update with corrected significance
                sig_current[sig_parent_key] = bool(reject[i])
    
    # Add summary information
    n_total = len(market_pairs)
    n_significant = np.sum(reject)
    
    logger.info(
        f"Multiple testing correction ({method}): "
        f"{n_significant}/{n_total} tests remain significant "
        f"after correction at alpha={alpha}"
    )
    
    return results


@m3_optimized
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def correct_threshold_cointegration_tests(
    results: Dict[str, Dict[str, Any]],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Apply multiple testing correction to threshold cointegration test results.
    
    Convenience function specifically designed for the threshold cointegration
    test results produced by the Yemen Market Integration analysis.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Dictionary of threshold cointegration results, keyed by market pair
    method : str, default='fdr_bh'
        Correction method (see apply_multiple_testing_correction for options)
    alpha : float, default=0.05
        Significance level
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Updated results with corrected p-values and significance flags
    """
    # Apply correction to cointegration tests
    results = apply_multiple_testing_to_results(
        results,
        p_value_key_path=['cointegration', 'p_value'],
        method=method,
        alpha=alpha,
        significance_key_path=['cointegration', 'cointegrated']
    )
    
    # Apply correction to threshold effect tests if present
    threshold_present = any(
        'threshold' in result and 
        'threshold_effect' in result['threshold'] and
        'p_value' in result['threshold']['threshold_effect']
        for result in results.values()
    )
    
    if threshold_present:
        results = apply_multiple_testing_to_results(
            results,
            p_value_key_path=['threshold', 'threshold_effect', 'p_value'],
            method=method,
            alpha=alpha,
            significance_key_path=['threshold', 'threshold_effect', 'significant']
        )
    
    return results


@m3_optimized
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def estimate_false_discovery_proportion(
    p_values: Union[List[float], np.ndarray],
    method: str = 'storey',
    lambda_param: float = 0.5
) -> float:
    """
    Estimate the proportion of false discoveries.
    
    This is useful for understanding the expected rate of false positives
    in the results, beyond just applying corrections.
    
    Parameters
    ----------
    p_values : array-like
        List or array of p-values
    method : str, default='storey'
        Method to use:
        - 'storey': Storey's method
        - 'lb': Lower bound estimate
    lambda_param : float, default=0.5
        Tuning parameter for Storey's method, in range [0, 1]
        
    Returns
    -------
    float
        Estimated proportion of false discoveries
    """
    if not isinstance(p_values, np.ndarray):
        p_values = np.array(p_values)
    
    # Remove NaN values
    p_values = p_values[~np.isnan(p_values)]
    
    # Ensure p-values are in [0, 1]
    if np.any((p_values < 0) | (p_values > 1)):
        logger.warning("Some p-values outside [0, 1], clipping")
        p_values = np.clip(p_values, 0, 1)
    
    if method == 'storey':
        # Storey's method
        m = len(p_values)
        if m == 0:
            return 0.0
            
        # Count p-values greater than lambda
        w_lambda = np.sum(p_values > lambda_param)
        
        # Estimate pi0 (proportion of true nulls)
        pi0 = w_lambda / (m * (1 - lambda_param))
        pi0 = min(1.0, pi0)  # Ensure pi0 <= 1
        
        # Ensure we don't get NaN if lambda_param = 1
        if lambda_param == 1.0:
            pi0 = 1.0
        
        return pi0
        
    elif method == 'lb':
        # Lower bound method
        m = len(p_values)
        if m == 0:
            return 0.0
            
        # Sort p-values
        p_values = np.sort(p_values)
        
        # Calculate expected distribution under null
        expected = np.arange(1, m + 1) / m
        
        # Find maximum deviation
        d = np.max(expected - p_values)
        
        # Lower bound estimate of pi0
        pi0_lb = 1.0 - d
        
        return max(0.0, pi0_lb)
    
    else:
        raise ValueError(f"Unknown method: {method}")


@m3_optimized
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def calibrate_testing_threshold(
    p_values: Union[List[float], np.ndarray],
    target_fdr: float = 0.05,
    method: str = 'bhq'
) -> float:
    """
    Calibrate a testing threshold to achieve a target false discovery rate.
    
    This is useful for determining a more appropriate threshold for
    controlling the false discovery rate in large-scale testing.
    
    Parameters
    ----------
    p_values : array-like
        List or array of p-values from previous tests
    target_fdr : float, default=0.05
        Target false discovery rate
    method : str, default='bhq'
        Method to use:
        - 'bhq': Benjamini-Hochberg q-value calibration
        - 'adaptive': Adaptive procedure based on estimate of pi0
        
    Returns
    -------
    float
        Calibrated threshold to achieve target FDR
    """
    if not isinstance(p_values, np.ndarray):
        p_values = np.array(p_values)
    
    # Remove NaN values
    p_values = p_values[~np.isnan(p_values)]
    
    # Ensure p-values are in [0, 1]
    if np.any((p_values < 0) | (p_values > 1)):
        logger.warning("Some p-values outside [0, 1], clipping")
        p_values = np.clip(p_values, 0, 1)
    
    m = len(p_values)
    if m == 0:
        logger.warning("No valid p-values provided, returning target_fdr as threshold")
        return target_fdr
    
    if method == 'bhq':
        # Sort p-values
        p_sorted = np.sort(p_values)
        
        # Apply BH procedure
        i = np.arange(1, m + 1)
        threshold_candidates = p_sorted[i * target_fdr / m >= p_sorted]
        
        if len(threshold_candidates) > 0:
            return threshold_candidates[-1]
        else:
            return target_fdr
            
    elif method == 'adaptive':
        # Estimate proportion of true nulls
        pi0 = estimate_false_discovery_proportion(p_values)
        
        # Adjust target based on pi0
        adjusted_target = target_fdr / pi0 if pi0 > 0 else target_fdr
        
        # Sort p-values
        p_sorted = np.sort(p_values)
        
        # Apply adjusted BH procedure
        i = np.arange(1, m + 1)
        threshold_candidates = p_sorted[i * adjusted_target / m >= p_sorted]
        
        if len(threshold_candidates) > 0:
            return threshold_candidates[-1]
        else:
            return min(adjusted_target, 0.1)  # Cap at 0.1 for safety
    
    else:
        raise ValueError(f"Unknown method: {method}")
