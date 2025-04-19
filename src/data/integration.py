"""Data integration module for Yemen Market Analysis.

This module provides functions for integrating data from different sources,
including merging market data with conflict data, integrating time series data
with spatial data, and combining data from different commodities.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)


class DataIntegrator:
    """
    Data integrator for Yemen Market Analysis.

    This class provides methods for integrating data from different sources.

    Attributes:
        cache (Dict[str, Any]): Cache for storing integrated data.
    """

    def __init__(self):
        """
        Initialize the data integrator.
        """
        self.cache = {}

    @handle_errors
    def merge_market_conflict_data(
        self, market_data: gpd.GeoDataFrame, conflict_data: gpd.GeoDataFrame,
        distance_threshold: float = 10.0, conflict_column: str = 'conflict_intensity',
        conflict_weight: float = 0.5, cache_key: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Merge market data with conflict data based on spatial proximity.

        This method merges market data with conflict data by calculating the
        weighted average of conflict intensity for each market based on the
        distance to conflict events.

        Args:
            market_data: GeoDataFrame containing market data.
            conflict_data: GeoDataFrame containing conflict data.
            distance_threshold: Maximum distance (in km) to consider for conflict events.
            conflict_column: Column in conflict_data containing conflict intensity.
            conflict_weight: Weight to apply to conflict intensity based on distance.
            cache_key: Key to use for caching the result. If None, the result is not cached.

        Returns:
            GeoDataFrame containing merged market and conflict data.

        Raises:
            YemenAnalysisError: If the data cannot be merged.
        """
        logger.info("Merging market data with conflict data")

        # Check if result is already cached
        if cache_key and cache_key in self.cache:
            logger.info(f"Using cached result for {cache_key}")
            return self.cache[cache_key]

        # Validate input data
        validate_data(market_data, data_type='geojson')
        validate_data(conflict_data, data_type='geojson')

        try:
            # Ensure both GeoDataFrames have the same CRS
            if market_data.crs != conflict_data.crs:
                logger.info("Converting conflict data to match market data CRS")
                conflict_data = conflict_data.to_crs(market_data.crs)

            # Create a copy of market data to avoid modifying the original
            result = market_data.copy()

            # Initialize conflict intensity column
            result[f'{conflict_column}_weighted'] = 0.0

            # Calculate distance-weighted conflict intensity for each market
            for idx, market in result.iterrows():
                # Calculate distances to all conflict events
                distances = conflict_data.geometry.distance(market.geometry)

                # Convert distances to kilometers (assuming CRS is in meters)
                distances_km = distances / 1000.0

                # Filter conflict events within the distance threshold
                mask = distances_km <= distance_threshold
                nearby_conflicts = conflict_data[mask]
                nearby_distances = distances_km[mask]

                if len(nearby_conflicts) > 0:
                    # Calculate distance weights (inverse of distance)
                    weights = 1.0 / (1.0 + nearby_distances * conflict_weight)

                    # Normalize weights
                    weights = weights / weights.sum()

                    # Calculate weighted average of conflict intensity
                    weighted_intensity = (nearby_conflicts[conflict_column] * weights).sum()

                    # Assign weighted conflict intensity to the market
                    result.at[idx, f'{conflict_column}_weighted'] = weighted_intensity

            # Normalize weighted conflict intensity to [0, 1] range
            max_intensity = result[f'{conflict_column}_weighted'].max()
            if max_intensity > 0:
                result[f'{conflict_column}_normalized'] = result[f'{conflict_column}_weighted'] / max_intensity
            else:
                result[f'{conflict_column}_normalized'] = 0.0

            # Cache result if cache_key is provided
            if cache_key:
                self.cache[cache_key] = result

            logger.info(f"Merged market data with conflict data, resulting in {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error merging market data with conflict data: {e}")
            raise YemenAnalysisError(f"Error merging market data with conflict data: {e}")

    @handle_errors
    def integrate_time_series_spatial(
        self, time_series_data: Dict[str, pd.DataFrame], spatial_data: gpd.GeoDataFrame,
        market_column: str = 'market', date_column: str = 'date',
        cache_key: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Integrate time series data with spatial data.

        This method integrates time series data with spatial data by merging
        the time series data for each market with the corresponding spatial data.

        Args:
            time_series_data: Dictionary mapping market names to DataFrames.
            spatial_data: GeoDataFrame containing spatial data.
            market_column: Column in spatial_data containing market names.
            date_column: Column in time_series_data containing dates.
            cache_key: Key to use for caching the result. If None, the result is not cached.

        Returns:
            GeoDataFrame containing integrated time series and spatial data.

        Raises:
            YemenAnalysisError: If the data cannot be integrated.
        """
        logger.info("Integrating time series data with spatial data")

        # Check if result is already cached
        if cache_key and cache_key in self.cache:
            logger.info(f"Using cached result for {cache_key}")
            return self.cache[cache_key]

        # Validate spatial data
        validate_data(spatial_data, data_type='geojson')

        try:
            # Create a list to store DataFrames for each market
            market_dfs = []

            # Process each market's time series data
            for market_name, market_data in time_series_data.items():
                # Validate time series data
                validate_data(market_data, data_type='dataframe')

                # Find the corresponding spatial data for this market
                market_spatial = spatial_data[spatial_data[market_column] == market_name]

                if len(market_spatial) == 0:
                    logger.warning(f"No spatial data found for market: {market_name}")
                    continue

                # Create a copy of the time series data
                market_df = market_data.copy()

                # Add market name column if not already present
                if market_column not in market_df.columns:
                    market_df[market_column] = market_name

                # Add geometry and other spatial attributes to the time series data
                for col in market_spatial.columns:
                    if col not in market_df.columns and col != 'geometry':
                        market_df[col] = market_spatial.iloc[0][col]

                # Add geometry column
                market_df['geometry'] = market_spatial.iloc[0].geometry

                # Append to the list of market DataFrames
                market_dfs.append(market_df)

            # Combine all market DataFrames
            if not market_dfs:
                logger.error("No markets could be integrated")
                raise YemenAnalysisError("No markets could be integrated")

            # Concatenate all market DataFrames
            combined_df = pd.concat(market_dfs, ignore_index=True)

            # Convert to GeoDataFrame
            result = gpd.GeoDataFrame(combined_df, geometry='geometry', crs=spatial_data.crs)

            # Sort by market and date
            result = result.sort_values([market_column, date_column])

            # Cache result if cache_key is provided
            if cache_key:
                self.cache[cache_key] = result

            logger.info(f"Integrated time series data with spatial data, resulting in {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error integrating time series data with spatial data: {e}")
            raise YemenAnalysisError(f"Error integrating time series data with spatial data: {e}")

    @handle_errors
    def combine_commodity_data(
        self, commodity_data: Dict[str, Dict[str, pd.DataFrame]],
        common_columns: Optional[List[str]] = None,
        cache_key: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Combine data from different commodities.

        This method combines data from different commodities into a single
        dataset for each market.

        Args:
            commodity_data: Dictionary mapping commodity names to dictionaries
                           mapping market names to DataFrames.
            common_columns: Columns to include in the combined data. If None,
                          includes all columns that are common across commodities.
            cache_key: Key to use for caching the result. If None, the result is not cached.

        Returns:
            Dictionary mapping market names to DataFrames containing combined commodity data.

        Raises:
            YemenAnalysisError: If the data cannot be combined.
        """
        logger.info("Combining data from different commodities")

        # Check if result is already cached
        if cache_key and cache_key in self.cache:
            logger.info(f"Using cached result for {cache_key}")
            return self.cache[cache_key]

        try:
            # Get all market names across all commodities
            all_markets = set()
            for commodity_markets in commodity_data.values():
                all_markets.update(commodity_markets.keys())

            # Create a dictionary to store combined data for each market
            combined_data = {}

            # Process each market
            for market_name in all_markets:
                # Get data for this market from each commodity
                market_commodity_data = {}
                for commodity, markets in commodity_data.items():
                    if market_name in markets:
                        market_commodity_data[commodity] = markets[market_name]

                if not market_commodity_data:
                    logger.warning(f"No data found for market: {market_name}")
                    continue

                # Determine common columns if not provided
                if common_columns is None:
                    # Find columns that are common across all commodities
                    common_cols = set(market_commodity_data[list(market_commodity_data.keys())[0]].columns)
                    for commodity_df in market_commodity_data.values():
                        common_cols &= set(commodity_df.columns)
                    common_columns = list(common_cols)

                # Create a list to store DataFrames for each commodity
                commodity_dfs = []

                # Process each commodity's data for this market
                for commodity, df in market_commodity_data.items():
                    # Create a copy of the DataFrame with only common columns
                    commodity_df = df[common_columns].copy()

                    # Add commodity column
                    commodity_df['commodity'] = commodity

                    # Append to the list of commodity DataFrames
                    commodity_dfs.append(commodity_df)

                # Combine all commodity DataFrames for this market
                if not commodity_dfs:
                    logger.warning(f"No data could be combined for market: {market_name}")
                    continue

                # Concatenate all commodity DataFrames
                combined_df = pd.concat(commodity_dfs, ignore_index=True)

                # Store combined data for this market
                combined_data[market_name] = combined_df

            # Cache result if cache_key is provided
            if cache_key:
                self.cache[cache_key] = combined_data

            logger.info(f"Combined data from {len(commodity_data)} commodities for {len(combined_data)} markets")
            return combined_data
        except Exception as e:
            logger.error(f"Error combining commodity data: {e}")
            raise YemenAnalysisError(f"Error combining commodity data: {e}")

    @handle_errors
    def aggregate_data(
        self, data: pd.DataFrame, group_columns: List[str],
        agg_columns: Dict[str, Union[str, List[str]]],
        cache_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate data at different levels.

        This method aggregates data at different levels based on the specified
        grouping columns and aggregation functions.

        Args:
            data: DataFrame containing the data to aggregate.
            group_columns: Columns to group by.
            agg_columns: Dictionary mapping column names to aggregation functions.
            cache_key: Key to use for caching the result. If None, the result is not cached.

        Returns:
            DataFrame containing aggregated data.

        Raises:
            YemenAnalysisError: If the data cannot be aggregated.
        """
        logger.info(f"Aggregating data by {group_columns}")

        # Check if result is already cached
        if cache_key and cache_key in self.cache:
            logger.info(f"Using cached result for {cache_key}")
            return self.cache[cache_key]

        # Validate input data
        validate_data(data, data_type='dataframe')

        try:
            # Check if all group columns exist in the data
            missing_columns = [col for col in group_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing group columns: {missing_columns}")
                raise YemenAnalysisError(f"Missing group columns: {missing_columns}")

            # Check if all aggregation columns exist in the data
            missing_columns = [col for col in agg_columns.keys() if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing aggregation columns: {missing_columns}")
                raise YemenAnalysisError(f"Missing aggregation columns: {missing_columns}")

            # Group data by the specified columns
            grouped = data.groupby(group_columns)

            # Aggregate data using the specified functions
            result = grouped.agg(agg_columns).reset_index()

            # Cache result if cache_key is provided
            if cache_key:
                self.cache[cache_key] = result

            logger.info(f"Aggregated data, resulting in {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error aggregating data: {e}")
            raise YemenAnalysisError(f"Error aggregating data: {e}")

    @handle_errors
    def clear_cache(self) -> None:
        """
        Clear the cache.
        """
        self.cache.clear()
        logger.info("Cache cleared")