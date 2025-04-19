"""
Data preprocessor module for Yemen Market Analysis.

This module provides functions for preprocessing data for the Yemen Market Analysis
package. It includes functions for handling missing values, outlier detection,
and feature engineering.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessor for Yemen Market Analysis.
    
    This class provides methods for preprocessing data for the Yemen Market Analysis
    package.
    
    Attributes:
        scalers (Dict[str, Any]): Dictionary of fitted scalers.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scalers: Dict[str, Any] = {}
    
    @handle_errors
    def detect_outliers(
        self, data: pd.DataFrame, column: str, method: str = 'zscore', threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in a column.
        
        Args:
            data: DataFrame containing the data.
            column: Column to detect outliers in.
            method: Method to use for outlier detection. Options are 'zscore', 'iqr',
                   and 'modified_zscore'.
            threshold: Threshold for outlier detection.
            
        Returns:
            Boolean Series indicating outliers.
            
        Raises:
            YemenAnalysisError: If the column is not found or the method is invalid.
        """
        logger.info(f"Detecting outliers in {column} using {method} method")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        col_data = data[column]
        
        # Detect outliers using the specified method
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            outliers = z_scores > threshold
        elif method == 'iqr':
            # IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
        elif method == 'modified_zscore':
            # Modified Z-score method
            median = col_data.median()
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * np.abs(col_data - median) / mad
            outliers = modified_z_scores > threshold
        else:
            logger.error(f"Invalid outlier detection method: {method}")
            raise YemenAnalysisError(f"Invalid outlier detection method: {method}")
        
        logger.info(f"Detected {outliers.sum()} outliers in {column}")
        return outliers
    
    @handle_errors
    def handle_outliers(
        self, data: pd.DataFrame, column: str, method: str = 'zscore',
        threshold: float = 3.0, handling: str = 'winsorize'
    ) -> pd.DataFrame:
        """
        Handle outliers in a column.
        
        Args:
            data: DataFrame containing the data.
            column: Column to handle outliers in.
            method: Method to use for outlier detection. Options are 'zscore', 'iqr',
                   and 'modified_zscore'.
            threshold: Threshold for outlier detection.
            handling: Method to use for handling outliers. Options are 'winsorize',
                     'trim', and 'mean'.
            
        Returns:
            DataFrame with outliers handled.
            
        Raises:
            YemenAnalysisError: If the column is not found or the method is invalid.
        """
        logger.info(f"Handling outliers in {column} using {handling} method")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Detect outliers
        outliers = self.detect_outliers(data_copy, column, method, threshold)
        
        # Handle outliers using the specified method
        if handling == 'winsorize':
            # Winsorize outliers
            if method == 'iqr':
                q1 = data_copy[column].quantile(0.25)
                q3 = data_copy[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                data_copy.loc[data_copy[column] < lower_bound, column] = lower_bound
                data_copy.loc[data_copy[column] > upper_bound, column] = upper_bound
            else:
                # For z-score and modified z-score, use percentiles
                lower_bound = data_copy[column].quantile(0.01)
                upper_bound = data_copy[column].quantile(0.99)
                data_copy.loc[outliers, column] = data_copy.loc[outliers, column].clip(
                    lower=lower_bound, upper=upper_bound
                )
        elif handling == 'trim':
            # Trim outliers (set to NaN)
            data_copy.loc[outliers, column] = np.nan
        elif handling == 'mean':
            # Replace outliers with mean
            mean_value = data_copy.loc[~outliers, column].mean()
            data_copy.loc[outliers, column] = mean_value
        else:
            logger.error(f"Invalid outlier handling method: {handling}")
            raise YemenAnalysisError(f"Invalid outlier handling method: {handling}")
        
        logger.info(f"Handled {outliers.sum()} outliers in {column}")
        return data_copy
    
    @handle_errors
    def normalize_data(
        self, data: pd.DataFrame, columns: List[str], method: str = 'standard',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize data in specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to normalize.
            method: Method to use for normalization. Options are 'standard' and 'minmax'.
            fit: Whether to fit a new scaler or use a previously fitted one.
            
        Returns:
            DataFrame with normalized data.
            
        Raises:
            YemenAnalysisError: If any of the columns are not found or the method is invalid.
        """
        logger.info(f"Normalizing columns {columns} using {method} method")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Check if columns exist
        for column in columns:
            if column not in data_copy.columns:
                logger.error(f"Column {column} not found in data")
                raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Normalize data using the specified method
        if method == 'standard':
            # Standardization (z-score normalization)
            if fit:
                scaler = StandardScaler()
                data_copy[columns] = scaler.fit_transform(data_copy[columns])
                self.scalers['standard'] = scaler
            else:
                if 'standard' not in self.scalers:
                    logger.error("No fitted standard scaler found")
                    raise YemenAnalysisError("No fitted standard scaler found")
                data_copy[columns] = self.scalers['standard'].transform(data_copy[columns])
        elif method == 'minmax':
            # Min-max normalization
            if fit:
                scaler = MinMaxScaler()
                data_copy[columns] = scaler.fit_transform(data_copy[columns])
                self.scalers['minmax'] = scaler
            else:
                if 'minmax' not in self.scalers:
                    logger.error("No fitted min-max scaler found")
                    raise YemenAnalysisError("No fitted min-max scaler found")
                data_copy[columns] = self.scalers['minmax'].transform(data_copy[columns])
        else:
            logger.error(f"Invalid normalization method: {method}")
            raise YemenAnalysisError(f"Invalid normalization method: {method}")
        
        logger.info(f"Normalized {len(columns)} columns")
        return data_copy
    
    @handle_errors
    def create_time_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create time-based features from a date column.
        
        Args:
            data: DataFrame containing the data.
            date_column: Column containing dates.
            
        Returns:
            DataFrame with time-based features added.
            
        Raises:
            YemenAnalysisError: If the date column is not found or is not a datetime column.
        """
        logger.info(f"Creating time features from {date_column}")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Check if date column exists
        if date_column not in data_copy.columns:
            logger.error(f"Column {date_column} not found in data")
            raise YemenAnalysisError(f"Column {date_column} not found in data")
        
        # Check if date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data_copy[date_column]):
            try:
                data_copy[date_column] = pd.to_datetime(data_copy[date_column])
            except Exception as e:
                logger.error(f"Error converting {date_column} to datetime: {e}")
                raise YemenAnalysisError(f"Error converting {date_column} to datetime: {e}")
        
        # Create time-based features
        data_copy['year'] = data_copy[date_column].dt.year
        data_copy['month'] = data_copy[date_column].dt.month
        data_copy['day'] = data_copy[date_column].dt.day
        data_copy['day_of_week'] = data_copy[date_column].dt.dayofweek
        data_copy['quarter'] = data_copy[date_column].dt.quarter
        data_copy['is_month_start'] = data_copy[date_column].dt.is_month_start.astype(int)
        data_copy['is_month_end'] = data_copy[date_column].dt.is_month_end.astype(int)
        data_copy['is_quarter_start'] = data_copy[date_column].dt.is_quarter_start.astype(int)
        data_copy['is_quarter_end'] = data_copy[date_column].dt.is_quarter_end.astype(int)
        data_copy['is_year_start'] = data_copy[date_column].dt.is_year_start.astype(int)
        data_copy['is_year_end'] = data_copy[date_column].dt.is_year_end.astype(int)
        
        logger.info("Created time features")
        return data_copy
    
    @handle_errors
    def create_lag_features(
        self, data: pd.DataFrame, columns: List[str], lags: List[int],
        group_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create lag features for specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to create lag features for.
            lags: List of lag values.
            group_column: Column to group by when creating lags. If None, no grouping is used.
            
        Returns:
            DataFrame with lag features added.
            
        Raises:
            YemenAnalysisError: If any of the columns are not found.
        """
        logger.info(f"Creating lag features for columns {columns} with lags {lags}")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Check if columns exist
        for column in columns:
            if column not in data_copy.columns:
                logger.error(f"Column {column} not found in data")
                raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Check if group column exists
        if group_column is not None and group_column not in data_copy.columns:
            logger.error(f"Group column {group_column} not found in data")
            raise YemenAnalysisError(f"Group column {group_column} not found in data")
        
        # Create lag features
        for column in columns:
            for lag in lags:
                lag_name = f"{column}_lag_{lag}"
                if group_column is not None:
                    data_copy[lag_name] = data_copy.groupby(group_column)[column].shift(lag)
                else:
                    data_copy[lag_name] = data_copy[column].shift(lag)
        
        logger.info(f"Created {len(columns) * len(lags)} lag features")
        return data_copy
    
    @handle_errors
    def create_rolling_features(
        self, data: pd.DataFrame, columns: List[str], windows: List[int],
        functions: List[str] = ['mean', 'std', 'min', 'max'],
        group_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features for specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to create rolling features for.
            windows: List of window sizes.
            functions: List of functions to apply to rolling windows.
            group_column: Column to group by when creating rolling features.
                         If None, no grouping is used.
            
        Returns:
            DataFrame with rolling features added.
            
        Raises:
            YemenAnalysisError: If any of the columns are not found or the functions are invalid.
        """
        logger.info(f"Creating rolling features for columns {columns} with windows {windows}")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Check if columns exist
        for column in columns:
            if column not in data_copy.columns:
                logger.error(f"Column {column} not found in data")
                raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Check if group column exists
        if group_column is not None and group_column not in data_copy.columns:
            logger.error(f"Group column {group_column} not found in data")
            raise YemenAnalysisError(f"Group column {group_column} not found in data")
        
        # Create rolling features
        for column in columns:
            for window in windows:
                for function in functions:
                    feature_name = f"{column}_roll_{window}_{function}"
                    if group_column is not None:
                        if function == 'mean':
                            data_copy[feature_name] = data_copy.groupby(group_column)[column].transform(
                                lambda x: x.rolling(window=window, min_periods=1).mean()
                            )
                        elif function == 'std':
                            data_copy[feature_name] = data_copy.groupby(group_column)[column].transform(
                                lambda x: x.rolling(window=window, min_periods=1).std()
                            )
                        elif function == 'min':
                            data_copy[feature_name] = data_copy.groupby(group_column)[column].transform(
                                lambda x: x.rolling(window=window, min_periods=1).min()
                            )
                        elif function == 'max':
                            data_copy[feature_name] = data_copy.groupby(group_column)[column].transform(
                                lambda x: x.rolling(window=window, min_periods=1).max()
                            )
                        else:
                            logger.error(f"Invalid rolling function: {function}")
                            raise YemenAnalysisError(f"Invalid rolling function: {function}")
                    else:
                        if function == 'mean':
                            data_copy[feature_name] = data_copy[column].rolling(window=window, min_periods=1).mean()
                        elif function == 'std':
                            data_copy[feature_name] = data_copy[column].rolling(window=window, min_periods=1).std()
                        elif function == 'min':
                            data_copy[feature_name] = data_copy[column].rolling(window=window, min_periods=1).min()
                        elif function == 'max':
                            data_copy[feature_name] = data_copy[column].rolling(window=window, min_periods=1).max()
                        else:
                            logger.error(f"Invalid rolling function: {function}")
                            raise YemenAnalysisError(f"Invalid rolling function: {function}")
        
        logger.info(f"Created {len(columns) * len(windows) * len(functions)} rolling features")
        return data_copy
    
    @handle_errors
    def create_difference_features(
        self, data: pd.DataFrame, columns: List[str], periods: List[int] = [1],
        group_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create difference features for specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to create difference features for.
            periods: List of periods for differencing.
            group_column: Column to group by when creating differences.
                         If None, no grouping is used.
            
        Returns:
            DataFrame with difference features added.
            
        Raises:
            YemenAnalysisError: If any of the columns are not found.
        """
        logger.info(f"Creating difference features for columns {columns} with periods {periods}")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Check if columns exist
        for column in columns:
            if column not in data_copy.columns:
                logger.error(f"Column {column} not found in data")
                raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Check if group column exists
        if group_column is not None and group_column not in data_copy.columns:
            logger.error(f"Group column {group_column} not found in data")
            raise YemenAnalysisError(f"Group column {group_column} not found in data")
        
        # Create difference features
        for column in columns:
            for period in periods:
                diff_name = f"{column}_diff_{period}"
                if group_column is not None:
                    data_copy[diff_name] = data_copy.groupby(group_column)[column].diff(period)
                else:
                    data_copy[diff_name] = data_copy[column].diff(period)
        
        logger.info(f"Created {len(columns) * len(periods)} difference features")
        return data_copy
    
    @handle_errors
    def create_pct_change_features(
        self, data: pd.DataFrame, columns: List[str], periods: List[int] = [1],
        group_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create percentage change features for specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to create percentage change features for.
            periods: List of periods for percentage change.
            group_column: Column to group by when creating percentage changes.
                         If None, no grouping is used.
            
        Returns:
            DataFrame with percentage change features added.
            
        Raises:
            YemenAnalysisError: If any of the columns are not found.
        """
        logger.info(f"Creating percentage change features for columns {columns} with periods {periods}")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Check if columns exist
        for column in columns:
            if column not in data_copy.columns:
                logger.error(f"Column {column} not found in data")
                raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Check if group column exists
        if group_column is not None and group_column not in data_copy.columns:
            logger.error(f"Group column {group_column} not found in data")
            raise YemenAnalysisError(f"Group column {group_column} not found in data")
        
        # Create percentage change features
        for column in columns:
            for period in periods:
                pct_name = f"{column}_pct_{period}"
                if group_column is not None:
                    data_copy[pct_name] = data_copy.groupby(group_column)[column].pct_change(period)
                else:
                    data_copy[pct_name] = data_copy[column].pct_change(period)
        
        logger.info(f"Created {len(columns) * len(periods)} percentage change features")
        return data_copy
    
    @handle_errors
    def create_interaction_features(
        self, data: pd.DataFrame, columns: List[List[str]], operations: List[str] = ['multiply']
    ) -> pd.DataFrame:
        """
        Create interaction features between pairs of columns.
        
        Args:
            data: DataFrame containing the data.
            columns: List of column pairs to create interaction features for.
            operations: List of operations to apply. Options are 'multiply', 'divide',
                       'add', and 'subtract'.
            
        Returns:
            DataFrame with interaction features added.
            
        Raises:
            YemenAnalysisError: If any of the columns are not found or the operations are invalid.
        """
        logger.info(f"Creating interaction features for column pairs {columns}")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Check if columns exist
        for col_pair in columns:
            if len(col_pair) != 2:
                logger.error(f"Column pair must have exactly 2 columns: {col_pair}")
                raise YemenAnalysisError(f"Column pair must have exactly 2 columns: {col_pair}")
            
            for column in col_pair:
                if column not in data_copy.columns:
                    logger.error(f"Column {column} not found in data")
                    raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Create interaction features
        for col_pair in columns:
            col1, col2 = col_pair
            for operation in operations:
                if operation == 'multiply':
                    feature_name = f"{col1}_mul_{col2}"
                    data_copy[feature_name] = data_copy[col1] * data_copy[col2]
                elif operation == 'divide':
                    feature_name = f"{col1}_div_{col2}"
                    # Avoid division by zero
                    data_copy[feature_name] = data_copy[col1] / data_copy[col2].replace(0, np.nan)
                elif operation == 'add':
                    feature_name = f"{col1}_add_{col2}"
                    data_copy[feature_name] = data_copy[col1] + data_copy[col2]
                elif operation == 'subtract':
                    feature_name = f"{col1}_sub_{col2}"
                    data_copy[feature_name] = data_copy[col1] - data_copy[col2]
                else:
                    logger.error(f"Invalid interaction operation: {operation}")
                    raise YemenAnalysisError(f"Invalid interaction operation: {operation}")
        
        logger.info(f"Created {len(columns) * len(operations)} interaction features")
        return data_copy
    
    @handle_errors
    def encode_categorical_features(
        self, data: pd.DataFrame, columns: List[str], method: str = 'one_hot',
        drop_first: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to encode.
            method: Method to use for encoding. Options are 'one_hot' and 'label'.
            drop_first: Whether to drop the first category in one-hot encoding.
            
        Returns:
            DataFrame with encoded features.
            
        Raises:
            YemenAnalysisError: If any of the columns are not found or the method is invalid.
        """
        logger.info(f"Encoding categorical features {columns} using {method} method")
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Check if columns exist
        for column in columns:
            if column not in data_copy.columns:
                logger.error(f"Column {column} not found in data")
                raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Encode categorical features
        if method == 'one_hot':
            # One-hot encoding
            for column in columns:
                dummies = pd.get_dummies(data_copy[column], prefix=column, drop_first=drop_first)
                data_copy = pd.concat([data_copy, dummies], axis=1)
                data_copy = data_copy.drop(column, axis=1)
        elif method == 'label':
            # Label encoding
            for column in columns:
                data_copy[column] = data_copy[column].astype('category').cat.codes
        else:
            logger.error(f"Invalid encoding method: {method}")
            raise YemenAnalysisError(f"Invalid encoding method: {method}")
        
        logger.info(f"Encoded {len(columns)} categorical features")
        return data_copy
    
    @handle_errors
    def detect_conflict_outliers(
        self, data: pd.DataFrame, price_column: str,
        conflict_column: Optional[str] = None,
        threshold: float = 2.0
    ) -> pd.Series:
        """
        Detect outliers in price data with conflict-aware methods.
        
        This method implements specialized outlier detection for conflict-affected price data,
        which often exhibits structural shifts and unusual volatility patterns. When a conflict
        column is provided, the method adjusts thresholds based on conflict intensity.
        
        Args:
            data: DataFrame containing the price and optional conflict data.
            price_column: Column containing price data to analyze for outliers.
            conflict_column: Optional column containing conflict intensity data.
                            If provided, thresholds are adjusted based on conflict intensity.
            threshold: Base threshold for outlier detection. This value is adjusted
                     based on conflict intensity when conflict_column is provided.
                     
        Returns:
            Boolean Series indicating outliers (True for outliers).
            
        Raises:
            YemenAnalysisError: If the price_column or conflict_column is not found in data.
        """
        logger.info(f"Detecting conflict-aware outliers in {price_column}")
        
        # Check if price column exists
        if price_column not in data.columns:
            logger.error(f"Price column {price_column} not found in data")
            raise YemenAnalysisError(f"Price column {price_column} not found in data")
        
        # Check if conflict column exists if provided
        if conflict_column is not None and conflict_column not in data.columns:
            logger.error(f"Conflict column {conflict_column} not found in data")
            raise YemenAnalysisError(f"Conflict column {conflict_column} not found in data")
        
        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Get price data
        price_data = data_copy[price_column]
        
        if conflict_column is not None:
            # Conflict-aware outlier detection with dynamic thresholds
            
            # Normalize conflict intensity to a 0-1 scale if not already
            conflict_data = data_copy[conflict_column]
            if conflict_data.max() > 1:
                # Assuming conflict data is on a different scale, normalize it
                conflict_data = (conflict_data - conflict_data.min()) / (conflict_data.max() - conflict_data.min())
            
            # Create adjusted thresholds based on conflict intensity
            # Higher conflict intensity allows for more volatility (higher threshold)
            # This accounts for the fact that prices are more volatile during conflict
            adjusted_thresholds = threshold * (1 + conflict_data)
            
            # Segment the data into windows to account for structural shifts
            # Use rolling windows to calculate local statistics
            window_size = min(30, len(price_data) // 4)  # Adaptive window size
            if window_size < 5:
                window_size = 5  # Minimum window size
            
            # Calculate rolling median and MAD (Median Absolute Deviation)
            rolling_median = price_data.rolling(window=window_size, center=True, min_periods=3).median()
            rolling_mad = (price_data - rolling_median).abs().rolling(window=window_size, center=True, min_periods=3).median()
            
            # Fill NaN values at the edges with the first/last valid values
            rolling_median = rolling_median.fillna(method='bfill').fillna(method='ffill')
            rolling_mad = rolling_mad.fillna(method='bfill').fillna(method='ffill')
            
            # Calculate modified Z-scores with conflict-adjusted thresholds
            modified_z_scores = 0.6745 * np.abs(price_data - rolling_median) / (rolling_mad + 1e-8)  # Add small constant to avoid division by zero
            
            # Identify outliers using adjusted thresholds
            outliers = modified_z_scores > adjusted_thresholds
        else:
            # If no conflict data is provided, use a robust method that's less sensitive to structural shifts
            
            # Use LOWESS (Locally Weighted Scatterplot Smoothing) to identify the trend
            # This helps account for structural shifts without explicit conflict data
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Create a time index
            time_index = np.arange(len(price_data))
            
            # Apply LOWESS smoothing
            # The frac parameter controls the size of the local window (as a fraction of the total number of data points)
            smoothed = lowess(price_data.values, time_index, frac=0.2, return_sorted=False)
            
            # Calculate residuals from the trend
            residuals = price_data.values - smoothed
            
            # Calculate robust statistics on the residuals
            median_residual = np.median(residuals)
            mad_residual = np.median(np.abs(residuals - median_residual))
            
            # Calculate modified Z-scores on the residuals
            modified_z_scores = 0.6745 * np.abs(residuals - median_residual) / (mad_residual + 1e-8)
            
            # Identify outliers
            outliers = pd.Series(modified_z_scores > threshold, index=price_data.index)
        
        logger.info(f"Detected {outliers.sum()} conflict-aware outliers in {price_column}")
        return outliers
    
    @handle_errors
    def adjust_dual_exchange_rates(
        self, data: pd.DataFrame, price_column: str,
        exchange_rate_columns: List[str],
        regime_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Adjust prices based on multiple exchange rate regimes.
        
        In conflict-affected regions, multiple exchange rate regimes often exist simultaneously.
        This method adjusts price data to account for these different regimes, either using
        an explicit regime indicator or by inferring the appropriate regime.
        
        Args:
            data: DataFrame containing price and exchange rate data.
            price_column: Column containing price data to adjust.
            exchange_rate_columns: List of columns containing different exchange rates.
            regime_column: Optional column indicating which exchange rate regime applies.
                          If None, the method attempts to infer the appropriate regime.
                          
        Returns:
            DataFrame with adjusted price data.
            
        Raises:
            YemenAnalysisError: If any of the required columns are not found in data.
        """
        logger.info(f"Adjusting prices in {price_column} using multiple exchange rate regimes")
        
        # Check if price column exists
        if price_column not in data.columns:
            logger.error(f"Price column {price_column} not found in data")
            raise YemenAnalysisError(f"Price column {price_column} not found in data")
        
        # Check if exchange rate columns exist
        for column in exchange_rate_columns:
            if column not in data.columns:
                logger.error(f"Exchange rate column {column} not found in data")
                raise YemenAnalysisError(f"Exchange rate column {column} not found in data")
        
        # Check if regime column exists if provided
        if regime_column is not None and regime_column not in data.columns:
            logger.error(f"Regime column {regime_column} not found in data")
            raise YemenAnalysisError(f"Regime column {regime_column} not found in data")
        
        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Create a new column for adjusted prices
        adjusted_price_column = f"{price_column}_adjusted"
        
        if regime_column is not None:
            # Use explicit regime indicator
            for i, exchange_rate_column in enumerate(exchange_rate_columns):
                # For each exchange rate, adjust prices where the regime matches
                regime_mask = (data_copy[regime_column] == i)
                if regime_mask.any():
                    # Convert prices to a common currency using the appropriate exchange rate
                    # We're assuming prices are in local currency and exchange rates are local/reference
                    data_copy.loc[regime_mask, adjusted_price_column] = (
                        data_copy.loc[regime_mask, price_column] /
                        data_copy.loc[regime_mask, exchange_rate_column]
                    )
        else:
            # Infer regime based on structural breaks in exchange rates
            
            # First, check for missing values in exchange rates
            for column in exchange_rate_columns:
                if data_copy[column].isna().any():
                    # Fill missing values with forward fill then backward fill
                    data_copy[column] = data_copy[column].fillna(method='ffill').fillna(method='bfill')
            
            # Initialize adjusted price column
            data_copy[adjusted_price_column] = np.nan
            
            # Create a time index for structural break detection
            time_index = np.arange(len(data_copy))
            
            # Detect structural breaks in exchange rate differences
            if len(exchange_rate_columns) == 2:
                # For two exchange rates, calculate the ratio between them
                exchange_ratio = data_copy[exchange_rate_columns[0]] / data_copy[exchange_rate_columns[1]]
                
                # Use rolling statistics to identify regime changes
                # A significant change in the ratio indicates a regime change
                rolling_mean = exchange_ratio.rolling(window=10, min_periods=1).mean()
                rolling_std = exchange_ratio.rolling(window=10, min_periods=1).std()
                
                # Identify potential regime changes (where ratio changes significantly)
                z_scores = np.abs((exchange_ratio - rolling_mean) / (rolling_std + 1e-8))
                regime_changes = z_scores > 3.0  # Threshold for significant change
                
                # Create regime indicators
                regime_indicators = np.zeros(len(data_copy), dtype=int)
                current_regime = 0
                
                for i in range(1, len(regime_indicators)):
                    if regime_changes.iloc[i]:
                        current_regime = 1 - current_regime  # Toggle between 0 and 1
                    regime_indicators[i] = current_regime
                
                # Apply adjustments based on inferred regimes
                for i, exchange_rate_column in enumerate(exchange_rate_columns):
                    regime_mask = (regime_indicators == i)
                    if regime_mask.any():
                        data_copy.loc[regime_mask, adjusted_price_column] = (
                            data_copy.loc[regime_mask, price_column] /
                            data_copy.loc[regime_mask, exchange_rate_column]
                        )
            else:
                # For more than two exchange rates, use a more complex approach
                # We'll select the exchange rate that minimizes price volatility in each period
                
                # Create a window for local volatility calculation
                window_size = min(30, len(data_copy) // 4)
                if window_size < 5:
                    window_size = 5  # Minimum window size
                
                # For each point, calculate adjusted prices using all exchange rates
                adjusted_prices = {}
                for exchange_rate_column in exchange_rate_columns:
                    adjusted_prices[exchange_rate_column] = (
                        data_copy[price_column] / data_copy[exchange_rate_column]
                    )
                
                # For each point, select the exchange rate that minimizes local volatility
                for i in range(len(data_copy)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(data_copy), i + window_size // 2 + 1)
                    
                    min_volatility = float('inf')
                    best_column = exchange_rate_columns[0]
                    
                    for exchange_rate_column in exchange_rate_columns:
                        # Calculate local volatility for this exchange rate
                        local_prices = adjusted_prices[exchange_rate_column].iloc[start_idx:end_idx]
                        volatility = local_prices.std() / local_prices.mean() if local_prices.mean() != 0 else float('inf')
                        
                        if volatility < min_volatility:
                            min_volatility = volatility
                            best_column = exchange_rate_column
                    
                    # Use the exchange rate that minimizes volatility
                    data_copy.loc[i, adjusted_price_column] = adjusted_prices[best_column].iloc[i]
        
        logger.info(f"Created adjusted price column {adjusted_price_column}")
        return data_copy
    
    @handle_errors
    def transform_conflict_affected_data(
        self, data: pd.DataFrame, price_column: str,
        conflict_column: str, method: str = 'robust_scaling'
    ) -> pd.DataFrame:
        """
        Transform conflict-affected price data using robust methods.
        
        This method applies specialized transformations to price data affected by conflict,
        accounting for structural shifts, volatility clusters, and other anomalies common
        in conflict-affected time series.
        
        Args:
            data: DataFrame containing price and conflict data.
            price_column: Column containing price data to transform.
            conflict_column: Column containing conflict intensity data.
            method: Transformation method to use. Options are:
                  - 'robust_scaling': Uses median and IQR instead of mean and std
                  - 'segmented_scaling': Applies different scaling to different conflict periods
                  - 'winsorized_scaling': Winsorizes data before scaling
                  - 'log_transform': Applies log transformation to reduce impact of extreme values
                  
        Returns:
            DataFrame with transformed price data.
            
        Raises:
            YemenAnalysisError: If any of the required columns are not found or the method is invalid.
        """
        logger.info(f"Transforming conflict-affected data in {price_column} using {method} method")
        
        # Check if price column exists
        if price_column not in data.columns:
            logger.error(f"Price column {price_column} not found in data")
            raise YemenAnalysisError(f"Price column {price_column} not found in data")
        
        # Check if conflict column exists
        if conflict_column not in data.columns:
            logger.error(f"Conflict column {conflict_column} not found in data")
            raise YemenAnalysisError(f"Conflict column {conflict_column} not found in data")
        
        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Create a new column for transformed prices
        transformed_column = f"{price_column}_transformed"
        
        # Get price and conflict data
        price_data = data_copy[price_column]
        conflict_data = data_copy[conflict_column]
        
        # Normalize conflict intensity to a 0-1 scale if not already
        if conflict_data.max() > 1:
            conflict_data = (conflict_data - conflict_data.min()) / (conflict_data.max() - conflict_data.min())
        
        if method == 'robust_scaling':
            # Robust scaling using median and IQR instead of mean and std
            # This is less sensitive to outliers and structural shifts
            median = price_data.median()
            q1 = price_data.quantile(0.25)
            q3 = price_data.quantile(0.75)
            iqr = q3 - q1
            
            # Avoid division by zero
            if iqr == 0:
                iqr = 1.0
                
            data_copy[transformed_column] = (price_data - median) / iqr
            
        elif method == 'segmented_scaling':
            # Segmented scaling based on conflict intensity
            # Define conflict thresholds for segmentation
            low_conflict = conflict_data <= 0.3
            medium_conflict = (conflict_data > 0.3) & (conflict_data <= 0.7)
            high_conflict = conflict_data > 0.7
            
            # Initialize transformed column
            data_copy[transformed_column] = np.nan
            
            # Apply different scaling to each segment
            for segment_mask, segment_name in [
                (low_conflict, "low conflict"),
                (medium_conflict, "medium conflict"),
                (high_conflict, "high conflict")
            ]:
                if segment_mask.any():
                    segment_data = price_data[segment_mask]
                    segment_median = segment_data.median()
                    segment_iqr = segment_data.quantile(0.75) - segment_data.quantile(0.25)
                    
                    # Avoid division by zero
                    if segment_iqr == 0:
                        segment_iqr = 1.0
                    
                    data_copy.loc[segment_mask, transformed_column] = (
                        (segment_data - segment_median) / segment_iqr
                    )
                    
                    logger.info(f"Transformed {segment_mask.sum()} points in {segment_name} segment")
            
        elif method == 'winsorized_scaling':
            # Winsorize data before scaling to reduce impact of extreme values
            # The winsorization level is adjusted based on conflict intensity
            
            # Calculate winsorization levels based on conflict intensity
            # Higher conflict intensity means more winsorization
            winsor_level = 0.01 + 0.04 * conflict_data  # Ranges from 0.01 to 0.05
            
            # Apply winsorization point by point
            winsorized_data = price_data.copy()
            
            for i in range(len(price_data)):
                level = winsor_level.iloc[i]
                lower_bound = price_data.quantile(level)
                upper_bound = price_data.quantile(1 - level)
                
                if price_data.iloc[i] < lower_bound:
                    winsorized_data.iloc[i] = lower_bound
                elif price_data.iloc[i] > upper_bound:
                    winsorized_data.iloc[i] = upper_bound
            
            # Scale the winsorized data
            mean = winsorized_data.mean()
            std = winsorized_data.std()
            
            # Avoid division by zero
            if std == 0:
                std = 1.0
                
            data_copy[transformed_column] = (winsorized_data - mean) / std
            
        elif method == 'log_transform':
            # Log transformation to reduce impact of extreme values
            # Add a small constant to avoid log(0)
            min_value = price_data.min()
            offset = 1.0 if min_value >= 0 else abs(min_value) + 1.0
            
            data_copy[transformed_column] = np.log(price_data + offset)
            
        else:
            logger.error(f"Invalid transformation method: {method}")
            raise YemenAnalysisError(f"Invalid transformation method: {method}")
        
        logger.info(f"Created transformed price column {transformed_column}")
        return data_copy
