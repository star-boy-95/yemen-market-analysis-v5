"""
Data integration utilities for combining data from different sources.
"""
import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
from typing import List, Optional, Union

from src.utils import handle_errors, validate_geodataframe, raise_if_invalid

logger = logging.getLogger(__name__)


class DataIntegrator:
    """Integrate data from multiple sources."""
    
    def __init__(self, data_path: Union[str, Path] = "./data"):
        """
        Initialize the data integrator.
        
        Parameters
        ----------
        data_path : str or Path
            Path to the data directory
        """
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
    
    @handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
    def integrate_conflict_data(
        self, market_gdf: gpd.GeoDataFrame, conflict_file: str
    ) -> gpd.GeoDataFrame:
        """
        Integrate conflict data with market data.
        
        Parameters
        ----------
        market_gdf : geopandas.GeoDataFrame
            Market data
        conflict_file : str
            Filename for conflict data
            
        Returns
        -------
        geopandas.GeoDataFrame
            Integrated dataset
        """
        # Read conflict data
        conflict_path = self.raw_path / conflict_file
        conflict_gdf = gpd.read_file(conflict_path)
        
        # Validate conflict data
        valid, errors = validate_geodataframe(
            conflict_gdf,
            required_columns=["admin1", "date", "events", "fatalities"]
        )
        raise_if_invalid(valid, errors, f"Invalid conflict data: {conflict_file}")
        
        # Ensure dates are in datetime format
        market_gdf['date'] = pd.to_datetime(market_gdf['date'])
        conflict_gdf['date'] = pd.to_datetime(conflict_gdf['date'])
        
        # Aggregate conflict data to admin region and month level
        conflict_gdf['yearmonth'] = conflict_gdf['date'].dt.strftime('%Y-%m')
        conflict_monthly = conflict_gdf.groupby(['admin1', 'yearmonth']).agg({
            'events': 'sum',
            'fatalities': 'sum'
        }).reset_index()
        
        # Prepare market data for merge
        if 'yearmonth' not in market_gdf.columns:
            market_gdf['yearmonth'] = market_gdf['date'].dt.strftime('%Y-%m')
        
        # Merge data
        result = pd.merge(
            market_gdf,
            conflict_monthly,
            on=['admin1', 'yearmonth'],
            how='left',
            suffixes=('', '_new')
        )
        
        # Update conflict data where new values are available
        for col in ['events', 'fatalities']:
            if f'{col}_new' in result.columns:
                mask = ~result[f'{col}_new'].isna()
                result.loc[mask, col] = result.loc[mask, f'{col}_new']
                result.drop(columns=[f'{col}_new'], inplace=True)
        
        # Fill any remaining missing values
        for col in ['events', 'fatalities']:
            if result[col].isna().any():
                result[col] = result[col].fillna(0)
        
        logger.info(f"Integrated conflict data, {len(result)} records in result")
        return result
    
    @handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
    def integrate_exchange_rates(
        self, market_gdf: gpd.GeoDataFrame, exchange_file: str
    ) -> gpd.GeoDataFrame:
        """
        Integrate exchange rate data with market data.
        
        Parameters
        ----------
        market_gdf : geopandas.GeoDataFrame
            Market data
        exchange_file : str
            Filename for exchange rate data
            
        Returns
        -------
        geopandas.GeoDataFrame
            Integrated dataset
        """
        # Read exchange rate data
        exchange_path = self.raw_path / exchange_file
        exchange_df = pd.read_csv(exchange_path)
        
        # Validate exchange rate data
        if not all(col in exchange_df.columns for col in ['date', 'north_rate', 'south_rate']):
            raise ValueError(f"Exchange rate data must have date, north_rate, and south_rate columns")
        
        # Ensure dates are in datetime format
        market_gdf['date'] = pd.to_datetime(market_gdf['date'])
        exchange_df['date'] = pd.to_datetime(exchange_df['date'])
        
        # Merge data
        result = pd.merge(
            market_gdf,
            exchange_df,
            on='date',
            how='left'
        )
        
        # Apply exchange rates based on regime
        if 'exchange_rate_regime' in result.columns:
            # Calculate USD price if needed
            if 'usdprice' not in result.columns:
                result['usdprice'] = 0.0
            
            # Update USD prices based on regime
            north_mask = result['exchange_rate_regime'] == 'north'
            south_mask = result['exchange_rate_regime'] == 'south'
            
            # Apply rates
            result.loc[north_mask, 'usdprice'] = result.loc[north_mask, 'price'] / result.loc[north_mask, 'north_rate']
            result.loc[south_mask, 'usdprice'] = result.loc[south_mask, 'price'] / result.loc[south_mask, 'south_rate']
        
        logger.info(f"Integrated exchange rate data, {len(result)} records in result")
        return result
    
    @handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
    def get_spatial_boundaries(self, boundary_file: str) -> gpd.GeoDataFrame:
        """
        Load administrative boundaries for spatial analysis.
        
        Parameters
        ----------
        boundary_file : str
            Filename for boundary data
            
        Returns
        -------
        geopandas.GeoDataFrame
            Administrative boundaries
        """
        boundary_path = self.raw_path / boundary_file
        boundaries = gpd.read_file(boundary_path)
        
        # Validate boundaries data
        valid, errors = validate_geodataframe(
            boundaries,
            required_columns=["admin1", "geometry"]
        )
        raise_if_invalid(valid, errors, f"Invalid boundary data: {boundary_file}")
        
        logger.info(f"Loaded boundary data with {len(boundaries)} regions")
        return boundaries