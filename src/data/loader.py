"""
Data loading utilities for Yemen market integration analysis.
"""
import geopandas as gpd
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Union, Tuple

from utils import handle_errors, validate_geodataframe, raise_if_invalid
from utils import read_geojson, write_geojson, DataError

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for GeoJSON market data."""
    
    def __init__(self, data_path: Union[str, Path] = "./data"):
        """Initialize the data loader."""
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
    
    @handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, OSError))
    def load_geojson(self, filename: str) -> gpd.GeoDataFrame:
        """Load GeoJSON data file into a GeoDataFrame."""
        file_path = self.raw_path / filename
        
        if not file_path.exists():
            raise DataError(f"GeoJSON file not found: {file_path}")
        
        gdf = read_geojson(file_path)
        
        valid, errors = validate_geodataframe(
            gdf, 
            required_columns=["admin1", "commodity", "date", "price"]
        )
        raise_if_invalid(valid, errors, f"Invalid GeoJSON file: {filename}")
        
        logger.info(f"Loaded GeoJSON from {file_path}: {len(gdf)} features")
        return gdf
    
    @handle_errors(logger=logger, error_type=(PermissionError, OSError))
    def save_processed_data(self, gdf: gpd.GeoDataFrame, filename: str) -> None:
        """Save processed data to the processed directory."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise DataError(f"Expected GeoDataFrame, got {type(gdf)}")
        
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        output_path = self.processed_path / filename
        write_geojson(gdf, output_path)
        
        logger.info(f"Saved processed data to {output_path}: {len(gdf)} features")
    
    @handle_errors(logger=logger, error_type=ValueError)
    def split_by_exchange_regime(self, gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Split data by exchange rate regime."""
        if 'exchange_rate_regime' not in gdf.columns:
            raise ValueError("Column 'exchange_rate_regime' not found in GeoDataFrame")
        
        north = gdf[gdf['exchange_rate_regime'] == 'north'].copy()
        south = gdf[gdf['exchange_rate_regime'] == 'south'].copy()
        
        logger.info(f"Split data by exchange regime: {len(north)} north, {len(south)} south")
        return north, south
    
    @handle_errors(logger=logger, error_type=ValueError)
    def get_time_series(self, gdf: gpd.GeoDataFrame, admin_region: str, commodity: str) -> gpd.GeoDataFrame:
        """Extract time series for specific region and commodity."""
        required_cols = ['admin1', 'commodity', 'date']
        for col in required_cols:
            if col not in gdf.columns:
                raise ValueError(f"Required column '{col}' not found in GeoDataFrame")
        
        mask = (gdf['admin1'] == admin_region) & (gdf['commodity'] == commodity)
        result = gdf[mask].sort_values('date').copy()
        
        logger.info(f"Extracted time series for {admin_region}, {commodity}: {len(result)} observations")
        return result
    
    @handle_errors(logger=logger, error_type=ValueError)
    def get_commodity_list(self, gdf: gpd.GeoDataFrame) -> List[str]:
        """Get list of available commodities in the data."""
        if 'commodity' not in gdf.columns:
            raise ValueError("Column 'commodity' not found in GeoDataFrame")
        
        commodities = sorted(gdf['commodity'].unique())
        
        logger.info(f"Found {len(commodities)} unique commodities")
        return commodities
    
    @handle_errors(logger=logger, error_type=ValueError)
    def get_region_list(self, gdf: gpd.GeoDataFrame) -> List[str]:
        """Get list of available administrative regions."""
        if 'admin1' not in gdf.columns:
            raise ValueError("Column 'admin1' not found in GeoDataFrame")
        
        regions = sorted(gdf['admin1'].unique())
        
        logger.info(f"Found {len(regions)} unique administrative regions")
        return regions
    
    @handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
    def load_multiple_periods(self, filenames: List[str]) -> gpd.GeoDataFrame:
        """Load and combine multiple GeoJSON files representing different time periods."""
        gdfs = []
        
        for filename in filenames:
            gdf = self.load_geojson(filename)
            gdfs.append(gdf)
        
        combined_gdf = pd.concat(gdfs, ignore_index=True)
        
        logger.info(f"Combined {len(filenames)} files: {len(combined_gdf)} total observations")
        return combined_gdf
    
    @handle_errors(logger=logger, error_type=ValueError)
    def filter_data(self, 
                   gdf: gpd.GeoDataFrame,
                   commodities: Optional[List[str]] = None,
                   regions: Optional[List[str]] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   exchange_regime: Optional[str] = None) -> gpd.GeoDataFrame:
        """Filter data by various criteria."""
        filtered_gdf = gdf.copy()
        
        if commodities:
            filtered_gdf = filtered_gdf[filtered_gdf['commodity'].isin(commodities)]
        
        if regions:
            filtered_gdf = filtered_gdf[filtered_gdf['admin1'].isin(regions)]
        
        if start_date:
            filtered_gdf = filtered_gdf[filtered_gdf['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            filtered_gdf = filtered_gdf[filtered_gdf['date'] <= pd.to_datetime(end_date)]
        
        if exchange_regime:
            if exchange_regime not in ['north', 'south']:
                raise ValueError("exchange_regime must be 'north' or 'south'")
            filtered_gdf = filtered_gdf[filtered_gdf['exchange_rate_regime'] == exchange_regime]
        
        logger.info(f"Filtered data: {len(filtered_gdf)} observations remaining")
        return filtered_gdf