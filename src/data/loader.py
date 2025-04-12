"""
Data loader module for Yemen Market Analysis.

This module provides functions for loading and preprocessing data for the Yemen
Market Analysis package. It includes functions for loading GeoJSON data, filtering
by commodity and market, and preprocessing data for analysis.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for Yemen Market Analysis.
    
    This class provides methods for loading and preprocessing data for the Yemen
    Market Analysis package.
    
    Attributes:
        cache (Dict[str, Any]): Cache for loaded data.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.cache: Dict[str, Any] = {}
    
    @handle_errors
    def load_geojson(self, file_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Load GeoJSON data from a file.
        
        Args:
            file_path: Path to the GeoJSON file.
            
        Returns:
            GeoDataFrame containing the loaded data.
            
        Raises:
            YemenAnalysisError: If the file cannot be loaded or is not valid GeoJSON.
        """
        file_path = Path(file_path)
        
        # Check if data is already cached
        cache_key = f"geojson_{file_path}"
        if cache_key in self.cache:
            logger.info(f"Using cached data for {file_path}")
            return self.cache[cache_key]
        
        logger.info(f"Loading GeoJSON data from {file_path}")
        
        try:
            # Load GeoJSON data
            gdf = gpd.read_file(file_path)
            
            # Validate data
            validate_data(gdf, data_type="geojson")
            
            # Cache data
            self.cache[cache_key] = gdf
            
            logger.info(f"Loaded GeoJSON data with {len(gdf)} rows")
            return gdf
        except Exception as e:
            logger.error(f"Error loading GeoJSON data: {e}")
            raise YemenAnalysisError(f"Error loading GeoJSON data: {e}")
    
    @handle_errors
    def filter_by_commodity(
        self, data: gpd.GeoDataFrame, commodity: str
    ) -> gpd.GeoDataFrame:
        """
        Filter data by commodity.
        
        Args:
            data: GeoDataFrame containing the data.
            commodity: Commodity to filter by.
            
        Returns:
            GeoDataFrame containing the filtered data.
            
        Raises:
            YemenAnalysisError: If the commodity is not found in the data.
        """
        logger.info(f"Filtering data by commodity: {commodity}")
        
        # Check if commodity exists in data
        if 'commodity' not in data.columns:
            logger.error("Commodity column not found in data")
            raise YemenAnalysisError("Commodity column not found in data")
        
        # Filter data
        filtered_data = data[data['commodity'] == commodity]
        
        if len(filtered_data) == 0:
            logger.error(f"No data found for commodity: {commodity}")
            raise YemenAnalysisError(f"No data found for commodity: {commodity}")
        
        logger.info(f"Filtered data has {len(filtered_data)} rows")
        return filtered_data
    
    @handle_errors
    def filter_by_markets(
        self, data: gpd.GeoDataFrame, markets: List[str]
    ) -> gpd.GeoDataFrame:
        """
        Filter data by markets.
        
        Args:
            data: GeoDataFrame containing the data.
            markets: List of markets to filter by.
            
        Returns:
            GeoDataFrame containing the filtered data.
            
        Raises:
            YemenAnalysisError: If any of the markets are not found in the data.
        """
        logger.info(f"Filtering data by markets: {markets}")
        
        # Check if market column exists in data
        if 'market' not in data.columns:
            logger.error("Market column not found in data")
            raise YemenAnalysisError("Market column not found in data")
        
        # Filter data
        filtered_data = data[data['market'].isin(markets)]
        
        # Check if all markets were found
        found_markets = filtered_data['market'].unique()
        missing_markets = [m for m in markets if m not in found_markets]
        
        if missing_markets:
            logger.warning(f"Markets not found: {missing_markets}")
        
        if len(filtered_data) == 0:
            logger.error(f"No data found for markets: {markets}")
            raise YemenAnalysisError(f"No data found for markets: {markets}")
        
        logger.info(f"Filtered data has {len(filtered_data)} rows")
        return filtered_data
    
    @handle_errors
    def preprocess_data(self, data: gpd.GeoDataFrame) -> Dict[str, pd.DataFrame]:
        """
        Preprocess data for analysis.
        
        This method preprocesses the data for analysis by:
        1. Converting dates to datetime
        2. Sorting data by date
        3. Handling missing values
        4. Splitting data by market
        
        Args:
            data: GeoDataFrame containing the data.
            
        Returns:
            Dictionary mapping market names to DataFrames.
            
        Raises:
            YemenAnalysisError: If the data cannot be preprocessed.
        """
        logger.info("Preprocessing data")
        
        try:
            # Convert dates to datetime
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # Sort data by date
            data = data.sort_values('date')
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Split data by market
            market_data = self._split_by_market(data)
            
            logger.info(f"Preprocessed data for {len(market_data)} markets")
            return market_data
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise YemenAnalysisError(f"Error preprocessing data: {e}")
    
    def _handle_missing_values(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: GeoDataFrame containing the data.
            
        Returns:
            GeoDataFrame with missing values handled.
        """
        logger.info("Handling missing values")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        logger.info(f"Missing values before handling: {missing_values}")
        
        # Handle missing values in price column
        if 'price' in data.columns and data['price'].isnull().any():
            # Interpolate missing prices
            data = data.sort_values(['market', 'date'])
            data['price'] = data.groupby('market')['price'].transform(
                lambda x: x.interpolate(method='linear')
            )
        
        # Handle missing values in other columns
        for col in data.columns:
            if col != 'price' and data[col].isnull().any():
                if data[col].dtype == 'object':
                    # Fill missing categorical values with mode
                    mode = data[col].mode()[0]
                    data[col] = data[col].fillna(mode)
                else:
                    # Fill missing numerical values with median
                    median = data[col].median()
                    data[col] = data[col].fillna(median)
        
        # Check for remaining missing values
        missing_values = data.isnull().sum()
        logger.info(f"Missing values after handling: {missing_values}")
        
        return data
    
    def _split_by_market(self, data: gpd.GeoDataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data by market.
        
        Args:
            data: GeoDataFrame containing the data.
            
        Returns:
            Dictionary mapping market names to DataFrames.
        """
        logger.info("Splitting data by market")
        
        # Check if market column exists
        if 'market' not in data.columns:
            logger.error("Market column not found in data")
            raise YemenAnalysisError("Market column not found in data")
        
        # Get unique markets
        markets = data['market'].unique()
        logger.info(f"Found {len(markets)} unique markets")
        
        # Split data by market
        market_data = {}
        for market in markets:
            market_df = data[data['market'] == market].copy()
            market_df = market_df.sort_values('date')
            market_df = market_df.reset_index(drop=True)
            market_data[market] = market_df
        
        return market_data
    
    @handle_errors
    def combine_market_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine data from multiple markets.
        
        Args:
            market_data: Dictionary mapping market names to DataFrames.
            
        Returns:
            DataFrame containing the combined data.
        """
        logger.info("Combining market data")
        
        # Combine data
        combined_data = pd.concat(market_data.values(), ignore_index=True)
        
        logger.info(f"Combined data has {len(combined_data)} rows")
        return combined_data
    
    @handle_errors
    def preprocess_commodity_data(
        self, data: gpd.GeoDataFrame, commodity: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Preprocess data for a specific commodity.
        
        This is a convenience method that filters data by commodity and then
        preprocesses it.
        
        Args:
            data: GeoDataFrame containing the data.
            commodity: Commodity to filter by.
            
        Returns:
            Dictionary mapping market names to DataFrames.
        """
        # Filter data by commodity
        filtered_data = self.filter_by_commodity(data, commodity)
        
        # Preprocess data
        return self.preprocess_data(filtered_data)
    
    @handle_errors
    def save_processed_data(
        self, data: Dict[str, pd.DataFrame], output_dir: Union[str, Path]
    ) -> None:
        """
        Save processed data to files.
        
        Args:
            data: Dictionary mapping market names to DataFrames.
            output_dir: Directory to save the data.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to {output_dir}")
        
        for market, df in data.items():
            file_path = output_dir / f"{market}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved data for {market} to {file_path}")
    
    @handle_errors
    def load_processed_data(
        self, input_dir: Union[str, Path]
    ) -> Dict[str, pd.DataFrame]:
        """
        Load processed data from files.
        
        Args:
            input_dir: Directory containing the data files.
            
        Returns:
            Dictionary mapping market names to DataFrames.
        """
        input_dir = Path(input_dir)
        
        logger.info(f"Loading processed data from {input_dir}")
        
        # Find all CSV files in the directory
        csv_files = list(input_dir.glob("*.csv"))
        
        if not csv_files:
            logger.error(f"No CSV files found in {input_dir}")
            raise YemenAnalysisError(f"No CSV files found in {input_dir}")
        
        # Load data from each file
        market_data = {}
        for file_path in csv_files:
            market = file_path.stem
            df = pd.read_csv(file_path)
            
            # Convert dates to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            market_data[market] = df
            logger.info(f"Loaded data for {market} from {file_path}")
        
        return market_data
