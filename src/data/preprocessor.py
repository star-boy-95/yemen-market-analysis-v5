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
from src.utils import clean_column_names, convert_dates, fill_missing_values
from src.utils import normalize_columns, create_date_features, create_lag_features
from src.utils import validate_dataframe

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
        
        # Clean column names using utility
        processed = clean_column_names(processed)
        
        # Convert date strings to datetime objects using utility
        processed = convert_dates(processed, date_cols=['date'])
        
        # Handle missing values using utility
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
        # Use the fill_missing_values utility
        filled_gdf = fill_missing_values(
            gdf,
            numeric_strategy='median',
            group_cols=['admin1', 'commodity'],
            date_strategy='forward'
        )
        
        # For conflict data, fill remaining NAs with zeros
        conflict_cols = [col for col in filled_gdf.columns if 'conflict' in col]
        for col in conflict_cols:
            if filled_gdf[col].isna().any():
                filled_gdf[col] = filled_gdf[col].fillna(0)
                logger.info(f"Filled missing values in {col} with zeros")
        
        # Log missing values that couldn't be filled
        remaining_missing = filled_gdf.isnull().sum()
        if remaining_missing.sum() > 0:
            for col in remaining_missing[remaining_missing > 0].index:
                logger.warning(f"Column {col} still has {remaining_missing[col]} missing values")
        
        return filled_gdf
    
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
        # Create date features using utility
        gdf = create_date_features(
            gdf, 
            date_col='date',
            features=['year', 'month', 'weekofyear']
        )
        
        # Add yearmonth manually as it's specific to this application
        gdf['yearmonth'] = gdf['date'].dt.strftime('%Y-%m')
        
        # Create price log for returns calculation
        gdf['price_log'] = np.log(gdf['price'])
        
        # Create lagged features using utility
        gdf = create_lag_features(
            gdf,
            cols=['price', 'price_log'],
            lags=[1],
            group_cols=['admin1', 'commodity']
        )
        
        # Calculate price returns manually
        gdf = gdf.sort_values(['admin1', 'commodity', 'date'])
        gdf['price_return'] = gdf.groupby(['admin1', 'commodity'])['price_log'].diff()
        
        # Create rolling features for volatility
        gdf['price_volatility'] = gdf.groupby(['admin1', 'commodity'])['price_return'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        # Normalize conflict intensity if present
        if 'conflict_intensity' in gdf.columns and 'conflict_intensity_normalized' not in gdf.columns:
            gdf = normalize_columns(
                gdf, 
                columns=['conflict_intensity'],
                method='minmax'
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
        valid, errors = validate_dataframe(
            gdf,
            required_columns=['commodity', 'date', 'price', 'exchange_rate_regime']
        )
        raise_if_invalid(valid, errors, "Invalid data for price differential calculation")
        
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