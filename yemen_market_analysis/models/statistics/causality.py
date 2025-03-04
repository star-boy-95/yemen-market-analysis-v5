"""
Granger causality testing for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from core.decorators import error_handler, performance_tracker
from core.exceptions import StatisticalTestError

logger = logging.getLogger(__name__)


@error_handler(fallback_value=(False, {"error": "Granger causality test failed"}))
@performance_tracker()
def run_granger_causality(
    y: Union[np.ndarray, pd.Series],
    x: Union[np.ndarray, pd.Series],
    max_lags: int = 8,
    significance_level: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run Granger causality test to determine if x Granger-causes y.
    
    Args:
        y: Dependent variable (effect)
        x: Independent variable (potential cause)
        max_lags: Maximum number of lags to test
        significance_level: Significance level for test
        
    Returns:
        Tuple of (causality_exists, test_results)
    """
    # Convert to pandas Series
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    
    # Ensure series are aligned
    if hasattr(y, 'index') and hasattr(x, 'index'):
        common_index = y.index.intersection(x.index)
        y = y.loc[common_index]
        x = x.loc[common_index]
    
    # Check input dimensions
    if len(y) != len(x):
        raise StatisticalTestError(f"Series have different lengths: {len(y)} vs {len(x)}")
    
    if len(y) < 3 * max_lags:
        # Reduce max_lags if insufficient observations
        effective_max_lags = max(1, len(y) // 4)
        logger.warning(
            f"Insufficient observations for {max_lags} lags. "
            f"Reducing to {effective_max_lags}."
        )
        max_lags = effective_max_lags
    
    # Combine into dataframe for grangercausalitytests
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Run test
    try:
        gc_results = grangercausalitytests(data, maxlag=max_lags, verbose=False)
        
        # Process results
        lags_significant = []
        min_p_value = 1.0
        
        for lag, result in gc_results.items():
            p_value = result[0]['ssr_ftest'][1]
            
            if p_value < significance_level:
                lags_significant.append(lag)
                
            if p_value < min_p_value:
                min_p_value = p_value
        
        # Determine causality
        causality = len(lags_significant) > 0
        
        # Compile results
        results = {
            'causality': causality,
            'min_p_value': float(min_p_value),
            'significant_lags': lags_significant,
            'max_lags_tested': max_lags,
            'n_obs': len(y)
        }
        
        return causality, results
    
    except Exception as e:
        logger.error(f"Error in Granger causality test: {str(e)}")
        raise StatisticalTestError(f"Granger causality test failed: {str(e)}")


@error_handler(fallback_value={"error": "Bidirectional Granger causality test failed"})
@performance_tracker()
def run_bidirectional_granger_causality(
    north_series: pd.Series, 
    south_series: pd.Series, 
    max_lags: int = 8,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Run Granger causality tests in both directions.
    
    Args:
        north_series: North market prices
        south_series: South market prices
        max_lags: Maximum number of lags to test
        significance_level: Significance level for test
        
    Returns:
        Dictionary with test results in both directions
    """
    # Check for sufficient observations
    min_obs = 3 * max_lags
    
    if len(north_series) < min_obs or len(south_series) < min_obs:
        effective_max_lags = max(1, min(len(north_series), len(south_series)) // 4)
        logger.warning(
            f"Insufficient observations for {max_lags} lags. "
            f"Reducing to {effective_max_lags}."
        )
        max_lags = effective_max_lags
    
    # Test north -> south causality
    north_to_south_causality, north_to_south_results = run_granger_causality(
        y=south_series,
        x=north_series,
        max_lags=max_lags,
        significance_level=significance_level
    )
    
    # Test south -> north causality
    south_to_north_causality, south_to_north_results = run_granger_causality(
        y=north_series,
        x=south_series,
        max_lags=max_lags,
        significance_level=significance_level
    )
    
    # Determine dominant direction
    n2s_causality = north_to_south_results.get('causality', False)
    s2n_causality = south_to_north_results.get('causality', False)
    
    if n2s_causality and not s2n_causality:
        dominant_direction = "north_to_south"
    elif s2n_causality and not n2s_causality:
        dominant_direction = "south_to_north"
    elif n2s_causality and s2n_causality:
        # Both directions are significant, determine stronger effect
        n2s_p_value = north_to_south_results.get('min_p_value', 1.0)
        s2n_p_value = south_to_north_results.get('min_p_value', 1.0)
        
        if n2s_p_value < s2n_p_value:
            dominant_direction = "north_to_south_stronger"
        else:
            dominant_direction = "south_to_north_stronger"
    else:
        # No significant causality in either direction
        dominant_direction = "none"
    
    # Compile results
    results = {
        "north_to_south": north_to_south_results,
        "south_to_north": south_to_north_results,
        "dominant_direction": dominant_direction,
        "bidirectional": n2s_causality and s2n_causality,
        "unidirectional": (n2s_causality and not s2n_causality) or (s2n_causality and not n2s_causality),
        "any_causality": n2s_causality or s2n_causality,
        "n_obs": min(len(north_series), len(south_series)),
        "max_lags": max_lags
    }
    
    return results


@error_handler(fallback_value={"error": "Rolling Granger causality test failed"})
def run_rolling_granger_causality(
    north_series: pd.Series,
    south_series: pd.Series,
    window_size: int = 24,
    step_size: int = 3,
    max_lags: int = 4,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Run rolling Granger causality tests to detect evolving relationships.
    
    Args:
        north_series: North market prices
        south_series: South market prices
        window_size: Size of rolling window
        step_size: Step size between windows
        max_lags: Maximum number of lags to test
        significance_level: Significance level for test
        
    Returns:
        Dictionary with rolling test results
    """
    # Ensure series are aligned
    common_index = north_series.index.intersection(south_series.index)
    north = north_series.loc[common_index]
    south = south_series.loc[common_index]
    
    # Check for sufficient observations
    min_obs = window_size + max_lags
    if len(north) < min_obs:
        return {"error": f"Insufficient observations ({len(north)}) for window size {window_size}"}
    
    # Initialize results
    rolling_results = []
    dates = []
    
    # Run rolling tests
    for start_idx in range(0, len(north) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Extract window
        north_window = north.iloc[start_idx:end_idx]
        south_window = south.iloc[start_idx:end_idx]
        
        # Run bidirectional test
        window_result = run_bidirectional_granger_causality(
            north_window, south_window, max_lags, significance_level
        )
        
        # Store results
        rolling_results.append(window_result)
        dates.append(north.index[start_idx + window_size // 2])  # Middle of window
    
    # Compile time series of causality indicators
    n2s_causality = [r.get('north_to_south', {}).get('causality', False) for r in rolling_results]
    s2n_causality = [r.get('south_to_north', {}).get('causality', False) for r in rolling_results]
    dominant_directions = [r.get('dominant_direction', 'none') for r in rolling_results]
    
    # Calculate summary statistics
    results = {
        "dates": dates,
        "n2s_causality": n2s_causality,
        "s2n_causality": s2n_causality,
        "dominant_directions": dominant_directions,
        "n2s_percentage": np.mean(n2s_causality) * 100 if n2s_causality else 0,
        "s2n_percentage": np.mean(s2n_causality) * 100 if s2n_causality else 0,
        "bidirectional_percentage": np.mean([r.get('bidirectional', False) for r in rolling_results]) * 100,
        "window_size": window_size,
        "step_size": step_size,
        "max_lags": max_lags,
        "n_windows": len(rolling_results)
    }
    
    return results