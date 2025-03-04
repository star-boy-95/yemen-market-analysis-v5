"""
Data loading functions for Yemen Market Analysis.
"""
import os
import json
import logging
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Union, Tuple, Any

from core.exceptions import DataProcessingError
from core.decorators import error_handler, performance_tracker

logger = logging.getLogger(__name__)


@error_handler(fallback_value={})
@performance_tracker()
def load_commodity_config(config_path: str) -> Dict[str, Any]:
    """Load commodity-specific configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


@error_handler(fallback_value=pd.DataFrame())
@performance_tracker()
def load_market_data(file_path: str, as_gdf: bool = False) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Load market data from CSV or GeoJSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, parse_dates=['date'])
        if as_gdf and 'latitude' in df.columns and 'longitude' in df.columns:
            try:
                import geopandas as gpd
                from shapely.geometry import Point
                
                geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
                return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            except ImportError:
                logger.warning("GeoPandas not available, returning DataFrame")
                return df
        return df
        
    elif file_path.endswith('.geojson'):
        return gpd.read_file(file_path)
        
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


@error_handler(fallback_value=pd.DataFrame())
@performance_tracker()
def load_unified_data(file_path: str) -> pd.DataFrame:
    """Load unified market data from GeoJSON or CSV and process to standard format."""
    df = load_market_data(file_path)
    
    # Ensure required columns exist
    required_columns = ['admin1', 'commodity', 'date', 'price', 'usdprice', 'exchange_rate_regime']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise DataProcessingError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Convert date column if needed
    if not pd.api.types.is_datetime64_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Ensure price columns are numeric
    for col in ['price', 'usdprice']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


@error_handler(fallback_value=(pd.DataFrame(), pd.DataFrame()))
@performance_tracker()
def split_by_regime(df: pd.DataFrame, commodity: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by exchange rate regime for a specific commodity."""
    # Filter for specific commodity
    commodity_df = df[df['commodity'] == commodity].copy()
    
    if commodity_df.empty:
        logger.warning(f"No data found for commodity: {commodity}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Split by regime
    north_df = commodity_df[commodity_df['exchange_rate_regime'] == 'north']
    south_df = commodity_df[commodity_df['exchange_rate_regime'] == 'south']
    
    # Check for empty dataframes
    if north_df.empty:
        logger.warning(f"No data for 'north' regime in commodity: {commodity}")
    if south_df.empty:
        logger.warning(f"No data for 'south' regime in commodity: {commodity}")
    
    return north_df, south_df


@error_handler(fallback_value=(pd.Series(), pd.Series()))
@performance_tracker()
def extract_price_series(
    north_df: pd.DataFrame, 
    south_df: pd.DataFrame,
    price_col: str = 'usdprice'
) -> Tuple[pd.Series, pd.Series]:
    """Extract and align north/south price series."""
    # Group by date to get average prices per date (if multiple locations per regime)
    if not north_df.empty:
        north_series = north_df.groupby('date')[price_col].mean()
    else:
        north_series = pd.Series(dtype='float64')
        
    if not south_df.empty:
        south_series = south_df.groupby('date')[price_col].mean()
    else:
        south_series = pd.Series(dtype='float64')
        
    # Align series to common dates
    if not north_series.empty and not south_series.empty:
        common_dates = north_series.index.intersection(south_series.index)
        if len(common_dates) > 0:
            north_series = north_series[common_dates]
            south_series = south_series[common_dates]
        else:
            logger.warning("No common dates between north and south regime data")
            return pd.Series(), pd.Series()
            
    return north_series, south_series


@error_handler(fallback_value=None)
def load_conflict_data(file_path: str) -> pd.DataFrame:
    """Load conflict intensity data."""
    if not os.path.exists(file_path):
        logger.warning(f"Conflict data file not found: {file_path}")
        return None
        
    df = pd.read_csv(file_path, parse_dates=['date'])
    
    # Check for required columns
    required_columns = ['date', 'admin1', 'conflict_intensity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing columns in conflict data: {', '.join(missing_columns)}")
        return None
        
    return df


@error_handler(fallback_value=pd.Series())
def extract_conflict_series(
    conflict_df: pd.DataFrame,
    admin1: Optional[List[str]] = None
) -> pd.Series:
    """Extract conflict intensity time series for specified admin1 regions."""
    if conflict_df is None or conflict_df.empty:
        return pd.Series()
        
    # Filter by admin1 if specified
    if admin1:
        conflict_df = conflict_df[conflict_df['admin1'].isin(admin1)]
        
    if conflict_df.empty:
        return pd.Series()
        
    # Aggregate by date
    conflict_series = conflict_df.groupby('date')['conflict_intensity'].mean()
    return conflict_series