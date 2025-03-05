"""
Data preprocessing utilities for Yemen market integration analysis.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

from src.utils import handle_errors, validate_geodataframe, raise_if_invalid

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess raw GeoJSON market data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    @handle_errors(logger=logger, error_type=(ValueError, KeyError))
    def preprocess_geojson(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Preprocess the raw GeoJSON data.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Raw GeoJSON data
            
        Returns
        -------
        geopandas.GeoDataFrame
            Preprocessed data
        """
        # Validate input data
        valid, errors = validate_geodataframe(
            gdf, 
            required_columns=["admin1", "commodity", "date", "price"]
        )
        raise_if_invalid(valid, errors, "Invalid input data for preprocessing")
        
        # Make a copy to avoid modifying the original
        processed = gdf.copy()
        
        # Convert date strings to datetime objects
        processed['date'] = pd.to_datetime(processed['date'])
        
        # Handle missing values
        processed = self._handle_missing_values(processed)
        
        # Create additional features
        processed = self._create_features(processed)
        
        logger.info(f"Preprocessed {len(processed)} records")
        return processed
    
    def _handle_missing_values(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Handle missing values in the data.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        geopandas.GeoDataFrame
            DataFrame with handled missing values
        """
        # Check for missing values
        missing_values = gdf.isnull().sum()
        
        # Forward fill date-based missing values (for time series)
        for col in ['price', 'usdprice']:
            if missing_values[col] > 0:
                gdf[col] = gdf.groupby(['admin1', 'commodity'])[col].fillna(method='ffill')
                # If still missing, backward fill
                gdf[col] = gdf.groupby(['admin1', 'commodity'])[col].fillna(method='bfill')
        
        # For conflict data, fill remaining NAs with zeros
        conflict_cols = [col for col in gdf.columns if 'conflict' in col]
        for col in conflict_cols:
            if col in missing_values and missing_values[col] > 0:
                gdf[col] = gdf[col].fillna(0)
        
        # Log missing values that couldn't be filled
        remaining_missing = gdf.isnull().sum()
        if remaining_missing.sum() > 0:
            for col in remaining_missing[remaining_missing > 0].index:
                logger.warning(f"Column {col} still has {remaining_missing[col]} missing values")
        
        return gdf
    
    def _create_features(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create additional features for analysis.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        geopandas.GeoDataFrame
            DataFrame with additional features
        """
        # Extract year and month
        gdf['year'] = gdf['date'].dt.year
        gdf['month'] = gdf['date'].dt.month
        gdf['yearmonth'] = gdf['date'].dt.strftime('%Y-%m')
        
        # Create price log returns for volatility analysis
        gdf['price_log'] = np.log(gdf['price'])
        
        # Group by admin1, commodity, and date, then calculate log returns
        gdf = gdf.sort_values(['admin1', 'commodity', 'date'])
        gdf['price_return'] = gdf.groupby(['admin1', 'commodity'])['price_log'].diff()
        
        # Create price volatility measure (rolling std of returns)
        gdf['price_volatility'] = gdf.groupby(['admin1', 'commodity'])['price_return'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        logger.info("Created additional features for analysis")
        return gdf
    
    @handle_errors(logger=logger, error_type=(ValueError, KeyError))
    def calculate_price_differentials(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Calculate price differentials between north and south exchange rate regimes.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with price differentials
        """
        # Validate input data
        if 'exchange_rate_regime' not in gdf.columns:
            raise ValueError("Input data must have 'exchange_rate_regime' column")
        
        # Get unique commodities and dates
        commodities = gdf['commodity'].unique()
        dates = gdf['date'].unique()
        
        # Create empty list to store differentials
        differentials = []
        
        # For each commodity and date, calculate north-south differential
        for commodity in commodities:
            for date in dates:
                # Filter data for this commodity and date
                mask = (gdf['commodity'] == commodity) & (gdf['date'] == date)
                data = gdf[mask]
                
                # Skip if no data for this combination
                if len(data) == 0:
                    continue
                
                # Get average prices by regime
                north_price = data[data['exchange_rate_regime'] == 'north']['price'].mean()
                south_price = data[data['exchange_rate_regime'] == 'south']['price'].mean()
                
                # Skip if one regime is missing
                if pd.isna(north_price) or pd.isna(south_price):
                    continue
                
                # Calculate differential
                differential = {
                    'commodity': commodity,
                    'date': date,
                    'north_price': north_price,
                    'south_price': south_price,
                    'price_diff': north_price - south_price,
                    'price_diff_pct': (north_price - south_price) / south_price * 100,
                    'log_price_ratio': np.log(north_price / south_price)
                }
                differentials.append(differential)
        
        result = pd.DataFrame(differentials)
        logger.info(f"Calculated price differentials: {len(result)} records")
        return result