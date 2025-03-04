"""
Data cleaning utilities for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

from core.decorators import error_handler, performance_tracker
from .validators import validate_regime_price_consistency

logger = logging.getLogger(__name__)


@error_handler(fallback_value=None)
def handle_missing_values(
    df: pd.DataFrame,
    max_gap: int = 3,
    method: str = 'linear'
) -> pd.DataFrame:
    """Handle missing values in time series data."""
    if df.empty:
        return df
        
    # Create copy to avoid modifying the original
    df_clean = df.copy()
    
    # Get numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # Handle missing values for each numeric column
    for col in numeric_cols:
        # Only interpolate if not too many consecutive missing values
        if df_clean[col].isna().sum() > 0:
            # Mark large gaps that shouldn't be interpolated
            mask = df_clean[col].isna()
            gap_starts = np.where((~mask[:-1]) & mask[1:])[0] + 1
            gap_ends = np.where(mask[:-1] & (~mask[1:]))[0] + 1
            
            # Add start/end of series if needed
            if mask.iloc[0]:
                gap_starts = np.insert(gap_starts, 0, 0)
            if mask.iloc[-1]:
                gap_ends = np.append(gap_ends, len(mask))
                
            # Find gap sizes
            gap_sizes = gap_ends - gap_starts
            
            # Identify gaps that are too large to interpolate
            large_gaps = []
            for start, size in zip(gap_starts, gap_sizes):
                if size > max_gap:
                    large_gaps.extend(range(start, start + size))
            
            # Only interpolate small gaps
            if len(large_gaps) < len(mask[mask]):
                # Temporarily set large gaps to non-NA so they won't be interpolated
                temp_vals = df_clean[col].copy()
                temp_vals.iloc[large_gaps] = -9999
                
                # Interpolate
                df_clean[col] = temp_vals.interpolate(method=method, limit=max_gap)
                
                # Reset large gaps back to NA
                df_clean.loc[df_clean[col] == -9999, col] = np.nan
    
    return df_clean


@error_handler(fallback_value=None)
def remove_outliers(
    series: pd.Series,
    window: int = 5,
    threshold: float = 3.0,
    method: str = 'mad'
) -> pd.Series:
    """Detect and remove outliers from a time series."""
    if series.empty:
        return series
        
    # Create copy to avoid modifying the original
    clean_series = series.copy()
    
    if method == 'mad':
        # Calculate rolling median
        rolling_median = clean_series.rolling(window, center=True, min_periods=2).median()
        
        # Calculate absolute deviations
        abs_dev = (clean_series - rolling_median).abs()
        
        # Calculate MAD (Median Absolute Deviation)
        mad = abs_dev.rolling(window, center=True, min_periods=2).median()
        
        # Identify outliers
        outliers = abs_dev > (threshold * mad)
        
        # Replace outliers with NaN
        clean_series[outliers] = np.nan
        
    elif method == 'zscore':
        # Calculate rolling mean and std
        rolling_mean = clean_series.rolling(window, center=True, min_periods=2).mean()
        rolling_std = clean_series.rolling(window, center=True, min_periods=2).std()
        
        # Calculate z-scores
        z_scores = (clean_series - rolling_mean) / rolling_std
        
        # Identify outliers
        outliers = z_scores.abs() > threshold
        
        # Replace outliers with NaN
        clean_series[outliers] = np.nan
        
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")
    
    # Count replaced outliers
    outlier_count = outliers.sum()
    if outlier_count > 0:
        logger.info(f"Removed {outlier_count} outliers from time series")
    
    return clean_series


@error_handler(fallback_value=(None, {}))
@performance_tracker()
def apply_regime_specific_cleaning(
    df: pd.DataFrame,
    commodity: str,
    regimes: List[str] = ['north', 'south'],
    price_col: str = 'usdprice',
    window: int = 5,
    mad_threshold: float = 3.0,
    replace_outliers: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply regime-specific cleaning to a commodity."""
    if df.empty:
        return df, {"status": "error", "message": "Empty DataFrame"}
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Results dictionary
    results = {"commodity": commodity, "regimes": {}}
    
    # Process each regime
    for regime in regimes:
        # Validate price consistency
        validation = validate_regime_price_consistency(
            df_clean, commodity, regime, price_col=price_col,
            window=window, mad_threshold=mad_threshold
        )
        
        results["regimes"][regime] = validation
        
        # Apply cleaning if needed
        if validation.get("status") == "warning" and replace_outliers:
            # Get the indices for this regime
            regime_mask = (df_clean['commodity'] == commodity) & (df_clean['exchange_rate_regime'] == regime)
            
            if regime_mask.sum() == 0:
                continue
                
            # Extract series for this regime
            regime_series = df_clean.loc[regime_mask, price_col]
            
            # Remove outliers
            clean_series = remove_outliers(
                regime_series, 
                window=window, 
                threshold=mad_threshold, 
                method='mad'
            )
            
            # Update the dataframe
            df_clean.loc[regime_mask, price_col] = clean_series
            
            # Update results
            results["regimes"][regime]["cleaned"] = True
            results["regimes"][regime]["outliers_removed"] = regime_series.isna().sum() - clean_series.isna().sum()
    
    return df_clean, results


@error_handler(fallback_value=None)
def fix_usd_prices(
    df: pd.DataFrame,
    exchange_rate_col: str = 'exchange_rate',
    price_col: str = 'price',
    usd_price_col: str = 'usdprice',
    max_deviation: float = 0.2,
    commodities: Optional[List[str]] = None
) -> pd.DataFrame:
    """Fix inconsistent USD prices using exchange rates."""
    if df.empty:
        return df
        
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Get list of commodities if not specified
    if commodities is None:
        commodities = df_clean['commodity'].unique().tolist()
    
    # Track corrections
    corrections = 0
    
    # Process each commodity and regime combination
    for commodity in commodities:
        for regime in ['north', 'south']:
            # Get subset
            mask = (df_clean['commodity'] == commodity) & (df_clean['exchange_rate_regime'] == regime)
            subset = df_clean.loc[mask]
            
            if subset.empty:
                continue
                
            # Check for required columns
            required_cols = [exchange_rate_col, price_col, usd_price_col]
            if not all(col in subset.columns for col in required_cols):
                continue
                
            # Calculate expected USD prices
            expected_usd = subset[price_col] / subset[exchange_rate_col]
            
            # Calculate deviation
            deviation = abs(subset[usd_price_col] - expected_usd) / expected_usd
            
            # Identify problematic conversions
            problematic = deviation > max_deviation
            
            # Fix problematic prices
            if problematic.sum() > 0:
                df_clean.loc[mask & problematic, usd_price_col] = (
                    df_clean.loc[mask & problematic, price_col] / 
                    df_clean.loc[mask & problematic, exchange_rate_col]
                )
                corrections += problematic.sum()
    
    if corrections > 0:
        logger.info(f"Fixed {corrections} inconsistent USD prices")
    
    return df_clean


@error_handler(fallback_value=None)
def clean_price_spikes(
    df: pd.DataFrame,
    commodity: str,
    price_col: str = 'usdprice',
    window: int = 5,
    threshold: float = 3.0
) -> pd.DataFrame:
    """Clean price spikes from a commodity's price data."""
    if df.empty:
        return df
        
    # Filter for the specified commodity
    commodity_df = df[df['commodity'] == commodity].copy()
    
    if commodity_df.empty:
        return df
    
    # Group by regime and clean each separately
    result_dfs = []
    for regime, regime_df in commodity_df.groupby('exchange_rate_regime'):
        if regime_df.empty:
            continue
            
        # Sort by date
        regime_df = regime_df.sort_values('date')
        
        # Calculate rolling median and MAD
        regime_df['rolling_median'] = regime_df[price_col].rolling(window, center=True, min_periods=2).median()
        abs_dev = (regime_df[price_col] - regime_df['rolling_median']).abs()
        mad = abs_dev.rolling(window, center=True, min_periods=2).median()
        
        # Identify spikes
        spikes = abs_dev > (threshold * mad)
        
        # Replace spikes with rolling median
        regime_df.loc[spikes, price_col] = regime_df.loc[spikes, 'rolling_median']
        
        # Drop temporary columns
        regime_df = regime_df.drop('rolling_median', axis=1)
        
        # Add to results
        result_dfs.append(regime_df)
    
    # Combine cleaned regime dataframes
    cleaned_commodity_df = pd.concat(result_dfs)
    
    # Update the original dataframe
    df_clean = df.copy()
    df_clean.loc[df_clean['commodity'] == commodity, price_col] = cleaned_commodity_df[price_col]
    
    return df_clean