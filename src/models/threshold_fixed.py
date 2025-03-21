"""
DEPRECATED: Fixed implementation of threshold cointegration with simplified approach.

This module is DEPRECATED and will be removed in a future version.
Please use the unified ThresholdModel with mode='fixed' instead.

Example:
    # Old code:
    from src.models.threshold_fixed import ThresholdFixed
    model = ThresholdFixed(data1, data2)
    
    # New code:
    from src.models.threshold_model import ThresholdModel
    model = ThresholdModel(data1, data2, mode='fixed')
"""
import warnings

# Emit a deprecation warning when the module is imported
warnings.warn(
    "The 'threshold_fixed' module is deprecated and will be removed in a future version. "
    "Use 'threshold_model' with mode='fixed' instead.",
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
DEFAULT_N_GRID = 30  # Smaller grid for fixed implementation
DEFAULT_MAX_LAGS = config.get('analysis.threshold.max_lags', 4)


def ThresholdFixed(
    data1, 
    data2, 
    max_lags=DEFAULT_MAX_LAGS,
    market1_name="Market 1",
    market2_name="Market 2",
    **kwargs
):
    """
    Backward compatibility wrapper for ThresholdFixed.
    
    This function is deprecated and will be removed in a future version.
    Use ThresholdModel with mode='fixed' instead.
    
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
        Initialized threshold model instance in 'fixed' mode
    """
    warnings.warn(
        "ThresholdFixed is deprecated. Use ThresholdModel with mode='fixed' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import here to avoid circular imports
    from src.models.threshold_model import ThresholdModel
    
    return ThresholdModel(
        data1, 
        data2, 
        mode="fixed",
        max_lags=max_lags,
        market1_name=market1_name,
        market2_name=market2_name,
        **kwargs
    )