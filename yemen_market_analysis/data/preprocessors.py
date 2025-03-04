"""
Data preprocessing pipelines for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

from core.decorators import error_handler, performance_tracker
from core.exceptions import DataProcessingError
from .cleaners import (
    handle_missing_values, remove_outliers, 
    apply_regime_specific_cleaning, fix_usd_prices
)
from .validators import (
    validate_data_series_with_economic_constraints,
    validate_usd_price, validate_regime_price_consistency,
    check_minimum_observations
)

logger = logging.getLogger(__name__)


@error_handler(fallback_value=None)
@performance_tracker()
def preprocess_market_data(
    df: pd.DataFrame,
    handle_missing: bool = True,
    max_gap: int = 3,
    fix_usd: bool = True,
    max_deviation: float = 0.2,
    clean_outliers: bool = True,
    window: int = 5,
    mad_threshold: float = 3.0,
    commodities: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Preprocess market data with configurable cleaning steps.
    
    Args:
        df: Input DataFrame
        handle_missing: Whether to interpolate missing values
        max_gap: Maximum gap size for interpolation
        fix_usd: Whether to fix inconsistent USD prices
        max_deviation: Maximum allowed deviation for USD prices
        clean_outliers: Whether to remove outliers
        window: Window size for outlier detection
        mad_threshold: MAD threshold for outlier detection
        commodities: List of commodities to process (None = all)
        
    Returns:
        Tuple of (cleaned_df, results_dict)
    """
    if df.empty:
        raise DataProcessingError("Empty DataFrame provided")
    
    # Create copy to avoid modifying the original
    df_clean = df.copy()
    
    # Track changes
    results = {
        "original_rows": len(df),
        "processing_steps": []
    }
    
    # Get list of commodities if not specified
    if commodities is None:
        commodities = df_clean['commodity'].unique().tolist()
    
    # Step 1: Handle missing values
    if handle_missing:
        df_clean = handle_missing_values(df_clean, max_gap=max_gap)
        missing_before = df.isna().sum().sum()
        missing_after = df_clean.isna().sum().sum()
        
        results["processing_steps"].append({
            "step": "handle_missing_values",
            "missing_before": int(missing_before),
            "missing_after": int(missing_after),
            "interpolated_values": int(missing_before - missing_after)
        })
    
    # Step 2: Fix USD prices
    if fix_usd:
        df_clean = fix_usd_prices(
            df_clean,
            max_deviation=max_deviation,
            commodities=commodities
        )
        
        # Count corrections
        usd_checks = []
        for commodity in commodities:
            for regime in ['north', 'south']:
                validation = validate_usd_price(
                    df_clean, commodity, regime, max_deviation=max_deviation
                )
                usd_checks.append({
                    "commodity": commodity,
                    "regime": regime,
                    "status": validation[1].get("status", "unknown"),
                    "problem_percentage": validation[1].get("problem_percentage", 0)
                })
        
        results["processing_steps"].append({
            "step": "fix_usd_prices",
            "usd_validations": usd_checks
        })
    
    # Step 3: Clean outliers for each commodity
    if clean_outliers:
        cleaning_results = {}
        
        for commodity in commodities:
            df_clean, commodity_results = apply_regime_specific_cleaning(
                df_clean,
                commodity=commodity,
                price_col='usdprice',
                window=window,
                mad_threshold=mad_threshold
            )
            
            cleaning_results[commodity] = commodity_results
        
        results["processing_steps"].append({
            "step": "clean_outliers",
            "commodities": cleaning_results
        })
    
    # Final statistics
    results["final_rows"] = len(df_clean)
    results["missing_values"] = int(df_clean.isna().sum().sum())
    
    return df_clean, results


@error_handler(fallback_value=(None, None, {}))
@performance_tracker()
def prepare_commodity_series(
    df: pd.DataFrame,
    commodity: str,
    price_col: str = 'usdprice',
    min_obs: int = 20,
    handle_missing: bool = True,
    max_gap: int = 3,
    clean_outliers: bool = True,
    min_transport_cost: float = 0.05,
    max_price_ratio: float = 5.0
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Prepare north/south price series for a specific commodity.
    
    Args:
        df: Input DataFrame
        commodity: Commodity to process
        price_col: Column containing prices
        min_obs: Minimum observations required
        handle_missing: Whether to interpolate missing values
        max_gap: Maximum gap size for interpolation
        clean_outliers: Whether to remove outliers
        min_transport_cost: Minimum transport cost as fraction
        max_price_ratio: Maximum price ratio between regimes
        
    Returns:
        Tuple of (north_series, south_series, results_dict)
    """
    # Filter for commodity
    commodity_df = df[df['commodity'] == commodity].copy()
    
    if commodity_df.empty:
        logger.warning(f"No data found for commodity: {commodity}")
        return None, None, {"error": f"No data for {commodity}"}
    
    # Create results dictionary
    results = {
        "commodity": commodity,
        "total_observations": len(commodity_df)
    }
    
    # Split by regime
    north_df = commodity_df[commodity_df['exchange_rate_regime'] == 'north']
    south_df = commodity_df[commodity_df['exchange_rate_regime'] == 'south']
    
    results["north_observations"] = len(north_df)
    results["south_observations"] = len(south_df)
    
    if north_df.empty or south_df.empty:
        logger.warning(f"Missing regime data for {commodity}")
        return None, None, {**results, "error": "Missing regime data"}
    
    # Handle missing values if requested
    if handle_missing:
        north_df = handle_missing_values(north_df, max_gap=max_gap)
        south_df = handle_missing_values(south_df, max_gap=max_gap)
    
    # Clean outliers if requested
    if clean_outliers:
        north_results = validate_regime_price_consistency(df, commodity, 'north', price_col=price_col)
        south_results = validate_regime_price_consistency(df, commodity, 'south', price_col=price_col)
        
        results["validation"] = {
            "north": north_results,
            "south": south_results
        }
        
        if north_results.get("status") == "warning":
            north_series = north_df.sort_values('date').set_index('date')[price_col]
            north_series = remove_outliers(north_series)
            north_df = north_df.copy()
            north_df.loc[:, price_col] = north_series.values
        
        if south_results.get("status") == "warning":
            south_series = south_df.sort_values('date').set_index('date')[price_col]
            south_series = remove_outliers(south_series)
            south_df = south_df.copy()
            south_df.loc[:, price_col] = south_series.values
    
    # Extract price series
    north_df = north_df.sort_values('date')
    south_df = south_df.sort_values('date')
    
    north_series = north_df.groupby('date')[price_col].mean()
    south_series = south_df.groupby('date')[price_col].mean()
    
    # Align indices
    common_dates = north_series.index.intersection(south_series.index)
    if len(common_dates) == 0:
        logger.warning(f"No common dates for {commodity}")
        return None, None, {**results, "error": "No common dates"}
    
    north_series = north_series[common_dates]
    south_series = south_series[common_dates]
    
    # Check minimum observations
    if not check_minimum_observations(north_series, south_series, min_obs=min_obs):
        return None, None, {**results, "error": "Insufficient observations"}
    
    # Validate with economic constraints
    north_series, south_series, eco_results = validate_data_series_with_economic_constraints(
        north_series, south_series, 
        min_transport_cost=min_transport_cost,
        max_price_ratio=max_price_ratio
    )
    
    # Update results
    results.update({
        "final_observations": len(north_series),
        "economic_validation": eco_results
    })
    
    return north_series, south_series, results


@error_handler(fallback_value=None)
def add_conflict_data(
    df: pd.DataFrame,
    conflict_df: pd.DataFrame
) -> pd.DataFrame:
    """Add conflict data to market data."""
    if df.empty or conflict_df.empty:
        return df
    
    # Make a copy
    df_result = df.copy()
    
    # Ensure date columns are datetime
    if not pd.api.types.is_datetime64_dtype(df_result['date']):
        df_result['date'] = pd.to_datetime(df_result['date'])
        
    if not pd.api.types.is_datetime64_dtype(conflict_df['date']):
        conflict_df['date'] = pd.to_datetime(conflict_df['date'])
    
    # Create a mapping from admin1-date to conflict intensity
    conflict_map = {}
    for _, row in conflict_df.iterrows():
        key = (row['admin1'], row['date'])
        conflict_map[key] = row['conflict_intensity']
    
    # Add conflict intensity to market data
    df_result['conflict_intensity'] = np.nan
    
    for idx, row in df_result.iterrows():
        key = (row['admin1'], row['date'])
        if key in conflict_map:
            df_result.at[idx, 'conflict_intensity'] = conflict_map[key]
    
    # Add lagged conflict intensity
    for lag in [1, 2, 3]:
        df_result[f'conflict_intensity_lag{lag}'] = df_result.groupby('admin1')['conflict_intensity'].shift(lag)
    
    return df_result