"""
DEPRECATED: Threshold Vector Error Correction Model (TVECM) for market integration analysis.

This module is DEPRECATED and will be removed in a future version.
Please use the unified ThresholdModel with mode='vecm' instead.

Example:
    # Old code:
    from src.models.threshold_vecm import ThresholdVECM
    model = ThresholdVECM(data)
    
    # New code:
    from src.models.threshold_model import ThresholdModel
    model = ThresholdModel(data1, data2, mode='vecm')
"""
import warnings

# Emit a deprecation warning when the module is imported
warnings.warn(
    "The 'threshold_vecm' module is deprecated and will be removed in a future version. "
    "Use 'threshold_model' with mode='vecm' instead.",
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
DEFAULT_ALPHA = config.get('analysis.threshold_vecm.alpha', 0.05)
DEFAULT_TRIM = config.get('analysis.threshold_vecm.trim', 0.15)
DEFAULT_N_GRID = config.get('analysis.threshold_vecm.n_grid', 300)
DEFAULT_K_AR_DIFF = config.get('analysis.threshold_vecm.k_ar_diff', 2)
DEFAULT_BOOTSTRAP_REPS = config.get('analysis.threshold_vecm.bootstrap_reps', 1000)


def ThresholdVECM(
    data, 
    k_ar_diff=DEFAULT_K_AR_DIFF,
    deterministic="ci",
    coint_rank=1,
    market_names=None,
    **kwargs
):
    """
    Backward compatibility wrapper for ThresholdVECM.
    
    This function is deprecated and will be removed in a future version.
    Use ThresholdModel with mode='vecm' instead.
    
    Parameters
    ----------
    data : array_like
        Multivariate time series data (markets as columns)
    k_ar_diff : int, optional
        Number of lagged differences
    deterministic : str, optional
        Deterministic term specification ('n', 'c', 'ct', 'ci', 'cit')
    coint_rank : int, optional
        Cointegration rank
    market_names : list of str, optional
        Names of markets for each column
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    ThresholdModel
        Initialized threshold model instance in 'vecm' mode
    """
    warnings.warn(
        "ThresholdVECM is deprecated. Use ThresholdModel with mode='vecm' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import here to avoid circular imports
    from src.models.threshold_model import ThresholdModel
    
    # Handle the different input format for VECM
    if isinstance(data, pd.DataFrame):
        if data.shape[1] < 2:
            raise ValueError("Data must have at least 2 columns for TVECM estimation")
        
        # Extract first two columns for bivariate analysis
        data1 = data.iloc[:, 0]
        data2 = data.iloc[:, 1]
        
        # Use column names as market names if not provided
        if market_names is None:
            market1_name = data.columns[0]
            market2_name = data.columns[1]
        else:
            market1_name = market_names[0] if len(market_names) > 0 else "Market 1"
            market2_name = market_names[1] if len(market_names) > 1 else "Market 2"
    else:
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Data must be 2-dimensional with at least 2 columns for TVECM estimation")
            
            # Extract first two columns for bivariate analysis
            data1 = data[:, 0]
            data2 = data[:, 1]
            
            # Use default market names if not provided
            if market_names is None:
                market1_name = "Market 1"
                market2_name = "Market 2"
            else:
                market1_name = market_names[0] if len(market_names) > 0 else "Market 1"
                market2_name = market_names[1] if len(market_names) > 1 else "Market 2"
        else:
            raise ValueError("Data must be a pandas DataFrame or numpy ndarray")
    
    return ThresholdModel(
        data1, 
        data2, 
        mode="vecm",
        max_lags=k_ar_diff,  # Use k_ar_diff as max_lags
        market1_name=market1_name,
        market2_name=market2_name,
        k_ar_diff=k_ar_diff,
        deterministic=deterministic,
        coint_rank=coint_rank,
        **kwargs
    )


def combine_tvecm_results(
    models: Dict[str, Any],
    market_pairs: Dict[str, Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Combine results from multiple TVECM models for market comparison.
    
    This function is maintained for backward compatibility.
    
    Parameters
    ----------
    models : dict
        Dictionary of ThresholdModel instances
    market_pairs : dict
        Dictionary mapping model keys to market pair names
        
    Returns
    -------
    dict
        Combined analysis results
    """
    warnings.warn(
        "combine_tvecm_results is deprecated. Use the new reporting mechanism instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Check inputs
    if not models:
        raise ValueError("No models provided")
    
    # Initialize result structure
    result = {
        'thresholds': {},
        'adjustment_speeds': {
            'below': {},
            'above': {}
        },
        'half_lives': {
            'below': {},
            'above': {},
            'ratio': {}
        },
        'market_pairs': market_pairs
    }
    
    # Extract thresholds
    for key, model in models.items():
        if model.threshold is not None:
            result['thresholds'][key] = model.threshold
    
    # Extract adjustment speeds (alpha)
    for key, model in models.items():
        if hasattr(model, 'below_model') and hasattr(model, 'above_model'):
            if model.below_model is not None and model.above_model is not None:
                # Use first loading coefficient (alpha) for first cointegration relation
                result['adjustment_speeds']['below'][key] = model.below_model.alpha[0, 0]
                result['adjustment_speeds']['above'][key] = model.above_model.alpha[0, 0]
    
    # Extract half-lives
    for key, model in models.items():
        if hasattr(model, 'calculate_half_lives'):
            half_lives = model.calculate_half_lives()
            result['half_lives']['below'][key] = half_lives['average']['below']
            result['half_lives']['above'][key] = half_lives['average']['above']
            result['half_lives']['ratio'][key] = half_lives['ratio']
    
    # Market ranking by integration strength
    # Lower half-lives indicate better integration
    sorted_markets = sorted(
        result['half_lives']['above'].items(),
        key=lambda x: x[1]
    )
    
    result['market_ranking'] = [
        {'market_pair': key, 'half_life': half_life}
        for key, half_life in sorted_markets
    ]
    
    # Add summary statistics
    result['summary'] = {
        'avg_threshold': np.mean(list(result['thresholds'].values())) if result['thresholds'] else np.nan,
        'min_threshold': np.min(list(result['thresholds'].values())) if result['thresholds'] else np.nan,
        'max_threshold': np.max(list(result['thresholds'].values())) if result['thresholds'] else np.nan,
        'avg_half_life_below': np.mean([x for x in result['half_lives']['below'].values() if not np.isinf(x)]) if result['half_lives']['below'] else np.nan,
        'avg_half_life_above': np.mean([x for x in result['half_lives']['above'].values() if not np.isinf(x)]) if result['half_lives']['above'] else np.nan,
        'best_integrated_pair': sorted_markets[0][0] if sorted_markets else None,
        'worst_integrated_pair': sorted_markets[-1][0] if sorted_markets else None
    }
    
    return result