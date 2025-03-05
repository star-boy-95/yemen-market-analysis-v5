"""
Data preprocessing utilities for Yemen market integration analysis.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
import logging

from src.utils import handle_errors, DataError, ValidationError
from src.utils import (
    clean_column_names, 
    convert_dates, 
    fill_missing_values,
    normalize_columns
)
from src.utils import validate_dataframe, raise_if_invalid, m1_optimized

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess raw GeoJSON market data with optimized methods."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    @handle_errors(logger=logger, error_type=(DataError, ValueError))
    def preprocess_geojson(self, gdf):
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
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise DataError(f"Expected GeoDataFrame but got {type(gdf)}")
        
        # Make a copy to avoid modifying the original
        processed = gdf.copy()
        
        # Clean column names using utility function
        processed = clean_column_names(processed)
        
        # Convert date strings to datetime objects using utility function
        processed = convert_dates(processed, date_cols=['date'])
        
        # Handle missing values
        processed = self._handle_missing_values(processed)
        
        # Create additional features
        processed = self._create_features(processed)
        
        logger.info(f"Completed preprocessing of {len(processed)} records")
        return processed
    
    @handle_errors(logger=logger, error_type=ValueError)
    def _handle_missing_values(self, gdf):
        """
        Handle missing values in the data using utility function.
        
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
        total_missing = missing_values.sum()
        
        if total_missing > 0:
            logger.info(f"Found {total_missing} missing values across {sum(missing_values > 0)} columns")
            
            # Use utility function for filling missing values
            filled_gdf = fill_missing_values(
                gdf,
                numeric_strategy='median',
                group_cols=['admin1', 'commodity'],
                date_strategy='ffill'
            )
            
            # For conflict data, fill remaining NAs with zeros
            conflict_cols = [col for col in filled_gdf.columns if 'conflict' in col]
            for col in conflict_cols:
                if filled_gdf[col].isnull().sum() > 0:
                    filled_gdf[col] = filled_gdf[col].fillna(0)
                    logger.info(f"Filled remaining NAs in {col} with zeros")
            
            logger.info("Successfully handled missing values")
            return filled_gdf
        else:
            logger.info("No missing values found")
            return gdf
    
    @m1_optimized(use_numba=True)
    @handle_errors(logger=logger, error_type=ValueError)
    def _create_features(self, gdf):
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
        # Make a copy to avoid modifying the input
        df = gdf.copy()
        
        # Extract year and month
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Create price log returns for volatility analysis
        df['price_log'] = np.log(df['price'])
        
        # Normalize conflict intensity if not already normalized
        if 'conflict_intensity' in df.columns and 'conflict_intensity_normalized' not in df.columns:
            df = normalize_columns(df, columns=['conflict_intensity'], new_names=['conflict_intensity_normalized'])
        
        # Group by admin1, commodity, and date, then calculate log returns
        df = df.sort_values(['admin1', 'commodity', 'date'])
        groups = df.groupby(['admin1', 'commodity'])
        
        # Calculate price returns (more efficient than apply)
        price_returns = []
        for name, group in groups:
            group = group.copy()
            group['price_return'] = group['price_log'].diff()
            price_returns.append(group)
        
        if price_returns:
            df = pd.concat(price_returns)
            logger.info("Created features: year, month, price_log, price_return")
        
        return df
    
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(DataError, ValueError))
    def calculate_price_differentials(self, gdf):
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
        # Validate input
        required_cols = ['commodity', 'date', 'price', 'exchange_rate_regime']
        valid, errors = validate_dataframe(gdf, required_columns=required_cols)
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
                
                # Get average prices by regime
                north_price = data[data['exchange_rate_regime'] == 'north']['price'].mean()
                south_price = data[data['exchange_rate_regime'] == 'south']['price'].mean()
                
                # Calculate differential
                if not (np.isnan(north_price) or np.isnan(south_price)):
                    differential = {
                        'commodity': commodity,
                        'date': date,
                        'north_price': north_price,
                        'south_price': south_price,
                        'price_diff': north_price - south_price,
                        'price_diff_pct': (north_price - south_price) / south_price * 100
                    }
                    differentials.append(differential)
        
        result = pd.DataFrame(differentials)
        logger.info(f"Calculated price differentials for {len(result)} commodity-date combinations")
        return result