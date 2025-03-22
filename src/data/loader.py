"""
Data loading utilities for Yemen market integration analysis.
"""
import geopandas as gpd
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Union, Tuple

# Use absolute imports for better module resolution
from yemen_market_integration.utils.error_handler import handle_errors, DataError
from yemen_market_integration.utils.validation import validate_geodataframe, raise_if_invalid
from yemen_market_integration.utils.file_utils import read_geojson, write_geojson
from yemen_market_integration.utils.performance_utils import memory_usage_decorator, optimize_dataframe

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for GeoJSON market data."""
    
    def __init__(self, data_path: Union[str, Path] = "./data"):
        """Initialize the data loader."""
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
    
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, OSError), reraise=True)
    def load_geojson(self, filename: str) -> gpd.GeoDataFrame:
        """Load GeoJSON data file into a GeoDataFrame."""
        file_path = self.raw_path / filename
        
        if not file_path.exists():
            raise DataError(f"GeoJSON file not found: {file_path}. Please ensure the file exists and you have read permissions.")
        
        gdf = read_geojson(file_path)
        
        # Optimize GeoDataFrame for memory efficiency
        gdf = optimize_dataframe(gdf)
        
        valid, errors = validate_geodataframe(
            gdf,
            required_columns=["admin1", "commodity", "date", "price"],
            check_nulls=False  # Don't check for null values
        )
        
        # Log warnings about null values but don't fail
        for error in errors:
            if "null values" in error:
                logger.warning(error)
            else:
                logger.error(error)
        
        raise_if_invalid(valid, errors)
        
        return gdf
    
    @handle_errors(logger=logger, error_type=(ValueError, KeyError), reraise=True)
    def filter_by_date(self, gdf: gpd.GeoDataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None) -> gpd.GeoDataFrame:
        """Filter GeoDataFrame by date range."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("Input must be a GeoDataFrame")
        
        # Convert date column to datetime if it's not already
        if 'date' in gdf.columns and not pd.api.types.is_datetime64_any_dtype(gdf['date']):
            gdf['date'] = pd.to_datetime(gdf['date'])
        
        # Apply date filters
        if start_date:
            start_date = pd.to_datetime(start_date)
            gdf = gdf[gdf['date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            gdf = gdf[gdf['date'] <= end_date]
        
        return gdf
    
    @handle_errors(logger=logger, error_type=(ValueError, KeyError), reraise=True)
    def preprocess_commodity_data(self, gdf: gpd.GeoDataFrame, commodity: str) -> pd.DataFrame:
        """Preprocess commodity data for analysis."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("Input must be a GeoDataFrame")
        
        # Filter by commodity if specified
        if commodity and 'commodity' in gdf.columns:
            gdf = gdf[gdf['commodity'] == commodity]
        
        # Check if we have data
        if len(gdf) == 0:
            raise DataError(f"No data found for commodity: {commodity}")
        
        # Convert to time series format
        df = gdf.pivot_table(
            index='date',
            columns='admin1',
            values='price',
            aggfunc='mean'
        )
        
        # Sort by date
        df = df.sort_index()
        
        # Handle missing values
        df = df.interpolate(method='linear')
        
        return df
    
    @handle_errors(logger=logger, error_type=(PermissionError, OSError), reraise=True)
    def save_processed_data(self, gdf: gpd.GeoDataFrame, filename: str) -> None:
        """Save processed data to the processed directory."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise DataError(f"Expected GeoDataFrame, got {type(gdf)}")
        
        file_path = self.processed_path / filename
        
        # Ensure the directory exists
        self.processed_path.mkdir(exist_ok=True, parents=True)
        
        # Save the data
        write_geojson(gdf, file_path)
        
        logger.info(f"Saved processed data to {file_path}")