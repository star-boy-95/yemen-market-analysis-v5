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
