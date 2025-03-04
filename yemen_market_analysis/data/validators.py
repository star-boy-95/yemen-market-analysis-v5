"""
Data validation functions for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

from core.exceptions import ValidationError
from core.decorators import error_handler

logger = logging.getLogger(__name__)


@error_handler(fallback_value=(None, None, {"error": "Validation failed"}))
def validate_data_series(
    y: pd.Series,
    x: pd.Series
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Basic validation of two time series."""
    # Check for empty series
    if y.empty or x.empty:
        raise ValidationError("Empty time series provided")
    
    # Ensure series are aligned
    if not y.index.equals(x.index):
        common_index = y.index.intersection(x.index)
        if len(common_index) == 0:
            raise ValidationError("No common dates between time series")
        
        y = y.loc[common_index]
        x = x.loc[common_index]
        
    # Check for sufficient observations
    if len(y) < 20:  # Minimum required for meaningful analysis
        raise ValidationError(f"Insufficient observations: {len(y)} < 20")
    
    # Check for missing values
    y_missing = y.isna().sum()
    x_missing = x.isna().sum()
    
    if y_missing > 0 or x_missing > 0:
        logger.warning(f"Missing values detected: y={y_missing}, x={x_missing}")
        
        # Drop missing values
        valid_mask = y.notna() & x.notna()
        y = y[valid_mask]
        x = x[valid_mask]
        
        if len(y) < 20:
            raise ValidationError(f"Insufficient valid observations after removing NAs: {len(y)} < 20")
    
    # Check for constant values
    if y.std() == 0 or x.std() == 0:
        raise ValidationError("Constant values detected in time series")
    
    return y, x, {
        "n_observations": len(y),
        "y_missing": y_missing,
        "x_missing": x_missing,
        "y_mean": y.mean(),
        "x_mean": x.mean(),
        "y_std": y.std(),
        "x_std": x.std()
    }


@error_handler(fallback_value=(None, None, {"error": "Economic validation failed"}))
def validate_data_series_with_economic_constraints(
    y: pd.Series, 
    x: pd.Series, 
    min_transport_cost: float = 0.05, 
    max_price_ratio: float = 5.0
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Enhanced validation for time series data with economic constraints."""
    # Perform basic validation first
    y, x, basic_stats = validate_data_series(y, x)
    
    if "error" in basic_stats:
        return None, None, basic_stats
    
    # Calculate minimum price at each point
    y_x_min = pd.concat([y, x], axis=1).min(axis=1)
    
    # Calculate price difference as percentage
    price_diff_pct = abs(y - x) / y_x_min
    
    # Identify constraint violations
    too_small = price_diff_pct < min_transport_cost
    too_large = price_diff_pct > max_price_ratio
    
    # Count violations
    small_violations = too_small.sum()
    large_violations = too_large.sum()
    total_violations = small_violations + large_violations
    
    # Calculate violation percentage
    violation_pct = (total_violations / len(y)) * 100
    
    # Create validation results
    validation_results = {
        **basic_stats,
        "economic_violations": {
            "too_small": small_violations,
            "too_large": large_violations,
            "total": total_violations,
            "violation_pct": violation_pct
        }
    }
    
    # Warning if high violation percentage
    if violation_pct > 20:
        logger.warning(f"High economic constraint violation rate: {violation_pct:.1f}%")
    
    # Filter out extreme violations if requested (disabled by default)
    filter_violations = False
    if filter_violations and total_violations > 0:
        valid_mask = ~(too_small | too_large)
        y = y[valid_mask]
        x = x[valid_mask]
        
        if len(y) < 20:
            raise ValidationError(f"Insufficient observations after economic filtering: {len(y)} < 20")
    
    return y, x, validation_results


@error_handler(fallback_value=(False, {"error": "USD price validation failed"}))
def validate_usd_price(
    df: pd.DataFrame,
    commodity: str,
    regime: str,
    exchange_rate_col: str = 'exchange_rate',
    price_col: str = 'price',
    usd_price_col: str = 'usdprice',
    max_deviation: float = 0.2
) -> Tuple[bool, Dict[str, Any]]:
    """Validate USD prices using exchange rates."""
    # Filter data for specific commodity and regime
    subset = df[(df['commodity'] == commodity) & (df['exchange_rate_regime'] == regime)]
    
    if subset.empty:
        return False, {"error": f"No data for {commodity} in {regime} regime"}
    
    # Check required columns
    required_cols = [exchange_rate_col, price_col, usd_price_col]
    missing_cols = [col for col in required_cols if col not in subset.columns]
    
    if missing_cols:
        return False, {"error": f"Missing columns: {', '.join(missing_cols)}"}
    
    # Calculate expected USD prices
    subset = subset.copy()
    subset['expected_usd'] = subset[price_col] / subset[exchange_rate_col]
    
    # Calculate deviation
    subset['deviation'] = abs(subset[usd_price_col] - subset['expected_usd']) / subset['expected_usd']
    
    # Identify problematic conversions
    problematic = subset[subset['deviation'] > max_deviation]
    problem_count = len(problematic)
    problem_pct = (problem_count / len(subset)) * 100 if len(subset) > 0 else 0
    
    results = {
        "total_observations": len(subset),
        "problematic_conversions": problem_count,
        "problem_percentage": problem_pct,
        "max_deviation": subset['deviation'].max() if not subset.empty else 0,
        "mean_deviation": subset['deviation'].mean() if not subset.empty else 0
    }
    
    if problem_pct > 10:
        logger.warning(f"High USD price conversion deviation for {commodity} in {regime} regime: {problem_pct:.1f}%")
        return False, {**results, "status": "failed"}
    
    return True, {**results, "status": "passed"}


@error_handler(fallback_value={})
def validate_regime_price_consistency(
    df: pd.DataFrame,
    commodity: str,
    regime: str,
    price_col: str = 'usdprice',
    window: int = 5,
    mad_threshold: float = 3.0
) -> Dict[str, Any]:
    """Validate price consistency within a regime using robust statistics."""
    # Filter data
    subset = df[(df['commodity'] == commodity) & (df['exchange_rate_regime'] == regime)]
    
    if subset.empty:
        return {"status": "error", "message": f"No data for {commodity} in {regime} regime"}
    
    # Sort by date
    subset = subset.sort_values('date')
    
    # Calculate rolling median and MAD (Median Absolute Deviation)
    subset = subset.copy()
    subset['rolling_median'] = subset[price_col].rolling(window, center=True, min_periods=2).median()
    
    # Calculate absolute deviations from median
    subset['abs_deviation'] = (subset[price_col] - subset['rolling_median']).abs()
    
    # Calculate MAD
    subset['mad'] = subset['abs_deviation'].rolling(window, center=True, min_periods=2).median()
    
    # Identify outliers (using MAD, which is more robust than standard deviation)
    subset['is_outlier'] = (subset['abs_deviation'] > (mad_threshold * subset['mad']))
    
    # Count outliers
    outlier_count = subset['is_outlier'].sum()
    outlier_pct = (outlier_count / len(subset)) * 100 if len(subset) > 0 else 0
    
    # Calculate statistics
    result = {
        "commodity": commodity,
        "regime": regime,
        "total_observations": len(subset),
        "outliers": outlier_count,
        "outlier_percentage": outlier_pct,
        "median_price": subset[price_col].median(),
        "price_volatility": subset[price_col].std() / subset[price_col].mean() if subset[price_col].mean() != 0 else 0
    }
    
    if outlier_pct > 10:
        logger.warning(f"High outlier percentage for {commodity} in {regime} regime: {outlier_pct:.1f}%")
        result["status"] = "warning"
    else:
        result["status"] = "passed"
    
    return result


@error_handler(fallback_value=False)
def check_minimum_observations(
    north_series: pd.Series,
    south_series: pd.Series,
    min_obs: int = 20
) -> bool:
    """Check if there are sufficient observations for analysis."""
    # Check non-empty series
    if north_series.empty or south_series.empty:
        logger.warning("Empty price series")
        return False
    
    # Check series lengths
    if len(north_series) < min_obs or len(south_series) < min_obs:
        logger.warning(f"Insufficient observations: north={len(north_series)}, south={len(south_series)}, min={min_obs}")
        return False
    
    # Check for constant values
    if north_series.std() == 0 or south_series.std() == 0:
        logger.warning("Constant values in price series")
        return False
    
    return True