"""
DEPRECATED: Threshold cointegration model for market integration analysis.

This module is DEPRECATED and will be removed in a future version.
Please use the unified ThresholdModel with mode='standard' instead.

Example:
    # Old code:
    from src.models.threshold import ThresholdCointegration
    model = ThresholdCointegration(data1, data2)
    
    # New code:
    from src.models.threshold_model import ThresholdModel
    model = ThresholdModel(data1, data2, mode='standard')
"""
import warnings

# Emit a deprecation warning when the module is imported
warnings.warn(
    "The 'threshold' module is deprecated and will be removed in a future version. "
    "Use 'threshold_model' with mode='standard' instead.",
    DeprecationWarning,
    stacklevel=2
)
import warnings
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd

# Import directly from local module to avoid circular imports
from src.utils.config import config

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values
DEFAULT_ALPHA = config.get('analysis.threshold.alpha', 0.05)
DEFAULT_TRIM = config.get('analysis.threshold.trim', 0.15)
DEFAULT_N_GRID = config.get('analysis.threshold.n_grid', 300)
DEFAULT_MAX_LAGS = config.get('analysis.threshold.max_lags', 4)


def ThresholdCointegration(
    data1, 
    data2, 
    max_lags=DEFAULT_MAX_LAGS,
    market1_name="Market 1",
    market2_name="Market 2",
    **kwargs
):
    """
    Backward compatibility wrapper for ThresholdCointegration.
    
    This function is deprecated and will be removed in a future version.
    Use ThresholdModel with mode='standard' instead.
    
    Parameters
    ----------
    data1 : array-like
        First time series (typically price series from first market)
    data2 : array-like
        Second time series (typically price series from second market)
    max_lags : int, optional
        Maximum number of lags to consider
    market1_name : str, optional
        Name of the first market (for plotting and reporting)
    market2_name : str, optional
        Name of the second market (for plotting and reporting)
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    ThresholdModel
        Initialized threshold model instance in 'standard' mode
    """
    warnings.warn(
        "ThresholdCointegration is deprecated. Use ThresholdModel with mode='standard' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import here to avoid circular imports
    from src.models.threshold_model import ThresholdModel
    
    return ThresholdModel(
        data1, 
        data2, 
        mode="standard",
        max_lags=max_lags,
        market1_name=market1_name,
        market2_name=market2_name,
        **kwargs
    )


def calculate_asymmetric_adjustment(
    adjustment_below: float, 
    adjustment_above: float
) -> Dict[str, float]:
    """
    Calculate asymmetric adjustment metrics.
    
    This function is maintained for backward compatibility.
    
    Parameters
    ----------
    adjustment_below : float
        Adjustment speed below threshold
    adjustment_above : float
        Adjustment speed above threshold
        
    Returns
    -------
    dict
        Dictionary containing asymmetric adjustment metrics
    """
    warnings.warn(
        "calculate_asymmetric_adjustment is deprecated. Use ThresholdModel methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Calculate asymmetry
    asymmetry = adjustment_above - adjustment_below
    
    # Calculate half-lives
    if adjustment_below >= 0:
        half_life_below = float('inf')
    else:
        half_life_below = np.log(0.5) / np.log(1 + adjustment_below)
        
    if adjustment_above >= 0:
        half_life_above = float('inf')
    else:
        half_life_above = np.log(0.5) / np.log(1 + adjustment_above)
    
    # Calculate half-life ratio
    if np.isinf(half_life_below) or np.isinf(half_life_above):
        half_life_ratio = float('inf')
    elif half_life_below == 0 or half_life_above == 0:
        half_life_ratio = float('inf')
    else:
        half_life_ratio = half_life_below / half_life_above
    
    return {
        'asymmetry_1': asymmetry,
        'half_life_below_1': half_life_below,
        'half_life_above_1': half_life_above,
        'half_life_ratio_1': half_life_ratio
    }


def calculate_half_life(residuals: np.ndarray) -> Dict[str, float]:
    """
    Calculate half-life of deviations from equilibrium.
    
    This function is maintained for backward compatibility.
    
    Parameters
    ----------
    residuals : array_like
        Residuals from cointegration equation
        
    Returns
    -------
    dict
        Dictionary containing half-life metrics
    """
    warnings.warn(
        "calculate_half_life is deprecated. Use ThresholdModel methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Fit AR(1) model to residuals
    y = residuals[1:]
    X = residuals[:-1]
    X = np.column_stack([np.ones(len(X)), X])
    
    # Estimate coefficients
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Extract AR coefficient
    ar_coef = beta[1]
    
    # Calculate half-life
    if ar_coef >= 1:
        half_life = float('inf')
    else:
        half_life = np.log(0.5) / np.log(ar_coef)
    
    return {
        'overall': half_life,
        'ar_coefficient': ar_coef
    }


def test_asymmetric_adjustment(
    eq_errors: np.ndarray, 
    threshold: float, 
    alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """
    Test for asymmetric adjustment in threshold model.
    
    This function is maintained for backward compatibility.
    
    Parameters
    ----------
    eq_errors : array_like
        Equilibrium errors from cointegration equation
    threshold : float
        Threshold value
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Dictionary containing test results
    """
    warnings.warn(
        "test_asymmetric_adjustment is deprecated. Use ThresholdModel methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Create indicator variables
    below = eq_errors <= threshold
    above = ~below
    
    # Count observations in each regime
    n_below = np.sum(below)
    n_above = np.sum(above)
    
    # Prepare data for regression
    y = np.diff(eq_errors)
    X = np.column_stack([
        eq_errors[:-1] * below[:-1],
        eq_errors[:-1] * above[:-1]
    ])
    
    # Add constant
    X = np.column_stack([np.ones(len(X)), X])
    
    # Fit the model
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Extract adjustment speeds
    adjustment_below = beta[1]
    adjustment_above = beta[2]
    
    # Calculate residuals and SSR
    residuals = y - X @ beta
    ssr = np.sum(residuals**2)
    
    # Calculate F-statistic for H0: adjustment_below = adjustment_above
    restriction = np.zeros((1, 3))
    restriction[0, 1] = 1
    restriction[0, 2] = -1
    
    # Calculate F-statistic
    r = np.array([0])
    q = X.shape[0] - 3  # degrees of freedom
    
    # Calculate restricted model
    X_r = np.column_stack([np.ones(len(X)), eq_errors[:-1]])
    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
    residuals_r = y - X_r @ beta_r
    ssr_r = np.sum(residuals_r**2)
    
    # Calculate F-statistic
    f_stat = ((ssr_r - ssr) / 1) / (ssr / q)
    
    # Calculate p-value
    from scipy import stats
    p_value = 1 - stats.f.cdf(f_stat, 1, q)
    
    return {
        'asymmetric': p_value < alpha,
        'p_value': p_value,
        'f_statistic': f_stat,
        'adjustment_below': adjustment_below,
        'adjustment_above': adjustment_above,
        'n_below': n_below,
        'n_above': n_above
    }


def test_mtar_adjustment(
    eq_errors: np.ndarray, 
    threshold: float = 0.0, 
    alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """
    Test for asymmetric adjustment using Momentum-TAR model.
    
    This function is maintained for backward compatibility.
    
    Parameters
    ----------
    eq_errors : array_like
        Equilibrium errors from cointegration equation
    threshold : float, optional
        Threshold value for momentum (default: 0.0)
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Dictionary containing test results
    """
    warnings.warn(
        "test_mtar_adjustment is deprecated. Use ThresholdModel methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Calculate changes in residuals (momentum)
    d_residuals = np.diff(eq_errors)
    
    # Create indicator variables based on momentum
    positive = d_residuals > threshold
    negative = ~positive
    
    # Count observations in each regime
    n_positive = np.sum(positive)
    n_negative = np.sum(negative)
    
    # Prepare data for regression
    y = np.diff(d_residuals)  # Second difference
    X = np.column_stack([
        eq_errors[1:-1] * positive[:-1],
        eq_errors[1:-1] * negative[:-1]
    ])
    
    # Add constant
    X = np.column_stack([np.ones(len(X)), X])
    
    # Fit the model
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Extract adjustment speeds
    adjustment_positive = beta[1]
    adjustment_negative = beta[2]
    
    # Calculate residuals and SSR
    residuals = y - X @ beta
    ssr = np.sum(residuals**2)
    
    # Calculate F-statistic for H0: adjustment_positive = adjustment_negative
    restriction = np.zeros((1, 3))
    restriction[0, 1] = 1
    restriction[0, 2] = -1
    
    # Calculate F-statistic
    r = np.array([0])
    q = X.shape[0] - 3  # degrees of freedom
    
    # Calculate restricted model
    X_r = np.column_stack([np.ones(len(X)), eq_errors[1:-1]])
    beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
    residuals_r = y - X_r @ beta_r
    ssr_r = np.sum(residuals_r**2)
    
    # Calculate F-statistic
    f_stat = ((ssr_r - ssr) / 1) / (ssr / q)
    
    # Calculate p-value
    from scipy import stats
    p_value = 1 - stats.f.cdf(f_stat, 1, q)
    
    # Calculate half-lives
    if adjustment_positive >= 0:
        half_life_positive = float('inf')
    else:
        half_life_positive = np.log(0.5) / np.log(1 + adjustment_positive)
        
    if adjustment_negative >= 0:
        half_life_negative = float('inf')
    else:
        half_life_negative = np.log(0.5) / np.log(1 + adjustment_negative)
    
    return {
        'asymmetric': p_value < alpha,
        'p_value': p_value,
        'f_statistic': f_stat,
        'adjustment_positive': adjustment_positive,
        'adjustment_negative': adjustment_negative,
        'half_life_positive': half_life_positive,
        'half_life_negative': half_life_negative,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'threshold': threshold
    }