"""
Data preprocessing utilities for Yemen market integration analysis.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

from yemen_market_integration.utils import handle_errors, validate_geodataframe, raise_if_invalid
from yemen_market_integration.utils import clean_column_names, convert_dates, fill_missing_values
from yemen_market_integration.utils import normalize_columns, create_date_features, create_lag_features
from yemen_market_integration.utils import validate_dataframe

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess raw GeoJSON market data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    @handle_errors(logger=logger, error_type=(ValueError, KeyError), reraise=True)
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
            required_columns=["admin1", "commodity", "date", "price"],
            check_nulls=False  # Don't check for null values
        )
        
        # Log warnings about null values but don't fail
        for error in errors:
            if "null values" in error:
                logger.warning(error)
            else:
                # Only raise for critical errors
                raise_if_invalid(False, [error], "Invalid input data for preprocessing")
        
        # Make a copy to avoid modifying the original
        processed = gdf.copy()
        
        # Clean column names using utility
        processed = clean_column_names(processed)
        
        # Convert date strings to datetime objects using utility
        processed = convert_dates(processed, date_columns=['date'])
        
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
        # Use the fill_missing_values utility (without specifying columns)
        filled_gdf = fill_missing_values(
            gdf,
            numeric_strategy='median',
            group_columns=['admin1', 'commodity'],
            date_strategy='forward'
        )
        
        # Fill missing values in 'population' column separately
        if 'population' in filled_gdf.columns:
            filled_gdf['population'] = filled_gdf.groupby(['admin1', 'commodity'])['population'].transform(
                lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
            )
            logger.info("Filled missing values in 'population' column with group medians")
        
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
            date_column='date',
            features=['year', 'month', 'weekofyear']
        )
        
        # Add yearmonth manually as it's specific to this application
        gdf['yearmonth'] = gdf['date'].dt.strftime('%Y-%m')
        
        # Create price log for returns calculation
        gdf['price_log'] = np.log(gdf['price'])
        
        # Create lagged features using utility
        gdf = create_lag_features(
            gdf,
            columns=['price', 'price_log'],
            lags=[1],
            group_columns=['admin1', 'commodity']
        )
        
        # Fill missing values in lagged price columns
        for col in ['price_lag1', 'price_log_lag1']:
            if col in gdf.columns:
                gdf[col] = gdf.groupby(['admin1', 'commodity'])[col].transform(
                    lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
                )
                logger.info(f"Filled missing values in '{col}' with group medians")
        
        # Calculate price returns manually
        gdf = gdf.sort_values(['admin1', 'commodity', 'date'])
        gdf['price_return'] = gdf.groupby(['admin1', 'commodity'])['price_log'].diff()
        
        # Fill missing price returns with 0 (no change) for first observations in each group
        missing_returns = gdf['price_return'].isna().sum()
        if missing_returns > 0:
            logger.info(f"Filling {missing_returns} missing price returns with 0")
            gdf['price_return'] = gdf['price_return'].fillna(0)
        
        # Create rolling features for volatility with more lenient min_periods
        gdf['price_volatility'] = gdf.groupby(['admin1', 'commodity'])['price_return'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std() if len(x) > 0 else np.nan
        )
        
        # Fill any remaining missing volatility values with the group median
        missing_volatility = gdf['price_volatility'].isna().sum()
        if missing_volatility > 0:
            logger.info(f"Filling {missing_volatility} missing volatility values with group medians")
            gdf['price_volatility'] = gdf.groupby(['admin1', 'commodity'])['price_volatility'].transform(
                lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 0)
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
    
    @handle_errors(logger=logger, error_type=(ValueError, KeyError), reraise=True)
    def calculate_price_differentials(self, gdf: gpd.GeoDataFrame, commodity: str = None) -> pd.DataFrame:
        """
        Calculate price differentials between north and south exchange rate regimes.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
        commodity : str, optional
            Commodity to filter by
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with price differentials
        """
        # Validate input data
        valid, errors = validate_dataframe(
            gdf,
            required_columns=['commodity', 'date', 'price', 'exchange_rate_regime'],
            check_nulls=False  # Don't fail on null values
        )
        
        # Log null values but don't fail
        null_counts = gdf.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if not columns_with_nulls.empty:
            for col, count in columns_with_nulls.items():
                logger.warning(f"Column '{col}' has {count} null values")
        
        # Only validate required columns exist, not null values
        raise_if_invalid(valid, [e for e in errors if "null values" not in e],
                        "Invalid data for price differential calculation")
        
        # Filter by commodity if provided
        if commodity:
            filtered_gdf = gdf[gdf['commodity'] == commodity]
        else:
            filtered_gdf = gdf
            
        # Get unique commodities and dates
        commodities = filtered_gdf['commodity'].unique()
        dates = filtered_gdf['date'].unique()
        
        # Create empty list to store differentials
        differentials = []
        
        # For each commodity and date, calculate north-south differential
        for commodity_name in commodities:
            for date in dates:
                # Filter data for this commodity and date
                mask = (filtered_gdf['commodity'] == commodity_name) & (filtered_gdf['date'] == date)
                data = filtered_gdf[mask]
                
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
                    'commodity': commodity_name,
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