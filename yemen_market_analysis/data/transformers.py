"""
Data transformation utilities for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, List, Optional, Tuple, Any, Union

from core.decorators import error_handler, performance_tracker

logger = logging.getLogger(__name__)


class TimeSeriesTransformer:
    """Utility class for time series transformations."""
    
    @staticmethod
    @error_handler(fallback_value=None)
    def difference(series: pd.Series, order: int = 1) -> pd.Series:
        """Calculate differences of a time series."""
        if series.empty:
            return series
        
        return series.diff(order).dropna()
    
    @staticmethod
    @error_handler(fallback_value=None)
    def log_transform(series: pd.Series) -> pd.Series:
        """Apply log transformation to a time series."""
        if series.empty:
            return series
        
        # Check for non-positive values
        if (series <= 0).any():
            logger.warning("Non-positive values detected in log transformation")
            return series
        
        return np.log(series)
    
    @staticmethod
    @error_handler(fallback_value=None)
    def percentage_change(series: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate percentage change in a time series."""
        if series.empty:
            return series
        
        return series.pct_change(periods=periods).dropna()
    
    @staticmethod
    @error_handler(fallback_value=None)
    def scale(
        series: pd.Series, 
        method: str = 'standard'
    ) -> pd.Series:
        """Scale a time series."""
        if series.empty:
            return series
        
        if method == 'standard':
            # Z-score normalization
            return (series - series.mean()) / series.std()
        
        elif method == 'minmax':
            # Min-max scaling
            min_val = series.min()
            max_val = series.max()
            
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
                
            return (series - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = series.median()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                return pd.Series(0, index=series.index)
                
            return (series - median) / iqr
        
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
    
    @staticmethod
    @error_handler(fallback_value=(None, {}))
    def remove_seasonality(
        series: pd.Series, 
        period: int = 12, 
        model: str = 'additive'
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Remove seasonality from a time series."""
        if series.empty or len(series) < period * 2:
            return series, {"error": "Insufficient data for seasonal decomposition"}
        
        # Ensure series is regularly spaced
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
        
        # Check for missing values
        if series.isna().any():
            series = series.interpolate(method='linear')
        
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(
                series, 
                model=model, 
                period=period, 
                extrapolate_trend='freq'
            )
            
            # Extract components
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            # Create seasonally adjusted series
            if model == 'additive':
                adjusted = series - seasonal
            else:  # multiplicative
                adjusted = series / seasonal
            
            # Return result
            return adjusted, {
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual,
                "model": model,
                "period": period
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {str(e)}")
            return series, {"error": f"Seasonal decomposition failed: {str(e)}"}
    
    @staticmethod
    @error_handler(fallback_value=None)
    def rolling_mean(series: pd.Series, window: int = 3) -> pd.Series:
        """Calculate rolling mean of a time series."""
        if series.empty or len(series) < window:
            return series
        
        return series.rolling(window=window, center=True).mean()
    
    @staticmethod
    @error_handler(fallback_value=None)
    def rolling_std(series: pd.Series, window: int = 3) -> pd.Series:
        """Calculate rolling standard deviation of a time series."""
        if series.empty or len(series) < window:
            return series
        
        return series.rolling(window=window, center=True).std()
    
    @staticmethod
    @error_handler(fallback_value=None)
    def detect_structural_breaks(
        series: pd.Series, 
        window: int = 20, 
        threshold: float = 3.0
    ) -> List[pd.Timestamp]:
        """Detect potential structural breaks in a time series."""
        if series.empty or len(series) < window * 2:
            return []
        
        # Calculate rolling mean and standard deviation
        rolling_mean = series.rolling(window=window).mean().dropna()
        rolling_std = series.rolling(window=window).std().dropna()
        
        # Calculate z-scores for each point relative to previous window
        z_scores = pd.Series(index=series.index[window:])
        
        for i in range(window, len(series)):
            prev_mean = rolling_mean.iloc[i-window]
            prev_std = rolling_std.iloc[i-window]
            
            if prev_std > 0:
                z_scores.iloc[i-window] = abs((series.iloc[i] - prev_mean) / prev_std)
            else:
                z_scores.iloc[i-window] = 0
        
        # Find potential breaks where z-score exceeds threshold
        potential_breaks = z_scores[z_scores > threshold].index.tolist()
        
        # Filter breaks to avoid clusters
        if not potential_breaks:
            return []
            
        filtered_breaks = [potential_breaks[0]]
        
        for date in potential_breaks[1:]:
            # Check if this break is too close to the previous one
            if (date - filtered_breaks[-1]).days > window:
                filtered_breaks.append(date)
        
        return filtered_breaks


@error_handler(fallback_value=None)
@performance_tracker()
def calculate_price_differentials(
    north_series: pd.Series,
    south_series: pd.Series
) -> pd.DataFrame:
    """Calculate price differentials between north and south."""
    if north_series.empty or south_series.empty:
        return None
    
    # Ensure series are aligned
    common_index = north_series.index.intersection(south_series.index)
    if len(common_index) == 0:
        return None
    
    north = north_series[common_index]
    south = south_series[common_index]
    
    # Create result dataframe
    result = pd.DataFrame({
        'date': common_index,
        'north': north.values,
        'south': south.values
    }).set_index('date')
    
    # Calculate differentials
    result['diff'] = result['north'] - result['south']
    result['diff_pct'] = (result['diff'] / result[['north', 'south']].min(axis=1)) * 100
    result['abs_diff'] = result['diff'].abs()
    result['abs_diff_pct'] = result['diff_pct'].abs()
    
    # Calculate log prices
    result['log_north'] = np.log(result['north'])
    result['log_south'] = np.log(result['south'])
    result['log_diff'] = result['log_north'] - result['log_south']
    
    return result


@error_handler(fallback_value=None)
def identify_arbitrage_opportunities(
    price_diff_df: pd.DataFrame,
    threshold: float = 0.15
) -> pd.DataFrame:
    """Identify potential arbitrage opportunities."""
    if price_diff_df is None or price_diff_df.empty:
        return None
    
    # Create copy
    result = price_diff_df.copy()
    
    # Identify arbitrage opportunities in each direction
    result['north_to_south'] = (result['diff'] > 0) & (result['diff_pct'] > threshold)
    result['south_to_north'] = (result['diff'] < 0) & (result['diff_pct'] < -threshold)
    
    # Any arbitrage opportunity
    result['arbitrage'] = result['north_to_south'] | result['south_to_north']
    
    # Calculate opportunity sizes
    result['opportunity_size'] = 0.0
    
    mask = result['north_to_south']
    if mask.any():
        result.loc[mask, 'opportunity_size'] = (
            result.loc[mask, 'diff_pct'] - threshold
        )
    
    mask = result['south_to_north']
    if mask.any():
        result.loc[mask, 'opportunity_size'] = (
            -result.loc[mask, 'diff_pct'] - threshold
        )
    
    return result