"""
Data manipulation utilities optimized for the Yemen Market Integration Project.
Provides functions for transforming, cleaning, and preparing data.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from typing import Union, List, Dict, Any, Optional, Tuple, Callable
import logging
from pathlib import Path
from datetime import datetime
import re

from src.utils.error_handler import handle_errors, DataError
from src.utils.decorators import timer, m1_optimized

logger = logging.getLogger(__name__)

@handle_errors(logger=logger)
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names: lowercase, replace spaces with underscores
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns to clean
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned column names
    """
    df = df.copy()
    df.columns = [re.sub(r'[^\w\s]', '', col).lower().replace(' ', '_') for col in df.columns]
    return df

@handle_errors(logger=logger)
def convert_dates(
    df: pd.DataFrame, 
    date_cols: List[str], 
    format: Optional[str] = None,
    errors: str = 'coerce'
) -> pd.DataFrame:
    """
    Convert columns to datetime
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with date columns
    date_cols : list
        List of column names to convert
    format : str, optional
        Date format string
    errors : str, optional
        How to handle parsing errors
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with converted date columns
    """
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=format, errors=errors)
    return df

@handle_errors(logger=logger)
def fill_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'mode',
    date_strategy: str = 'nearest',
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fill missing values based on column types
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with missing values
    numeric_strategy : str, optional
        Strategy for numeric columns ('mean', 'median', 'zero', 'none')
    categorical_strategy : str, optional
        Strategy for categorical columns ('mode', 'none')
    date_strategy : str, optional
        Strategy for date columns ('nearest', 'forward', 'backward', 'none')
    group_cols : list, optional
        Columns to group by before filling
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with filled missing values
    """
    df = df.copy()
    
    # Get column types
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns
    
    # Fill missing values by group if specified
    if group_cols:
        # Filter group_cols to those that exist in the dataframe
        valid_group_cols = [col for col in group_cols if col in df.columns]
        
        if valid_group_cols:
            # Handle numeric columns
            if numeric_strategy != 'none' and len(numeric_cols) > 0:
                for col in numeric_cols:
                    if numeric_strategy == 'mean':
                        df[col] = df.groupby(valid_group_cols)[col].transform(
                            lambda x: x.fillna(x.mean())
                        )
                    elif numeric_strategy == 'median':
                        df[col] = df.groupby(valid_group_cols)[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                    elif numeric_strategy == 'zero':
                        df[col] = df.groupby(valid_group_cols)[col].transform(
                            lambda x: x.fillna(0)
                        )
            
            # Handle categorical columns
            if categorical_strategy == 'mode' and len(categorical_cols) > 0:
                for col in categorical_cols:
                    df[col] = df.groupby(valid_group_cols)[col].transform(
                        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else None)
                    )
            
            # Handle date columns
            if date_strategy != 'none' and len(date_cols) > 0:
                for col in date_cols:
                    if date_strategy == 'nearest':
                        df[col] = df.groupby(valid_group_cols)[col].transform(
                            lambda x: x.interpolate(method='nearest')
                        )
                    elif date_strategy == 'forward':
                        df[col] = df.groupby(valid_group_cols)[col].transform(
                            lambda x: x.fillna(method='ffill')
                        )
                    elif date_strategy == 'backward':
                        df[col] = df.groupby(valid_group_cols)[col].transform(
                            lambda x: x.fillna(method='bfill')
                        )
        else:
            logger.warning("No valid group columns found, filling without grouping")
            # Fall back to non-grouped filling
            return fill_missing_values(
                df, 
                numeric_strategy=numeric_strategy,
                categorical_strategy=categorical_strategy,
                date_strategy=date_strategy,
                group_cols=None
            )
    else:
        # Fill missing values without grouping
        
        # Handle numeric columns
        if numeric_strategy != 'none' and len(numeric_cols) > 0:
            if numeric_strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif numeric_strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif numeric_strategy == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Handle categorical columns
        if categorical_strategy == 'mode' and len(categorical_cols) > 0:
            for col in categorical_cols:
                if df[col].isna().any():
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
        
        # Handle date columns
        if date_strategy != 'none' and len(date_cols) > 0:
            if date_strategy == 'nearest':
                df[date_cols] = df[date_cols].interpolate(method='nearest')
            elif date_strategy == 'forward':
                df[date_cols] = df[date_cols].fillna(method='ffill')
            elif date_strategy == 'backward':
                df[date_cols] = df[date_cols].fillna(method='bfill')
    
    return df

@handle_errors(logger=logger)
def detect_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers in specified columns
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    columns : list
        Columns to check for outliers
    method : str, optional
        Detection method ('zscore', 'iqr', 'percentile')
    threshold : float, optional
        Threshold for outlier detection
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with outlier flag columns
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric, skipping outlier detection")
            continue
        
        # Create outlier flag column
        flag_col = f"{col}_outlier"
        
        # Detect outliers based on specified method
        if method == 'zscore':
            z_scores = stats.zscore(df[col], nan_policy='omit')
            result[flag_col] = abs(z_scores) > threshold
        
        elif method == 'iqr':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            result[flag_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'percentile':
            lower = df[col].quantile(threshold / 100)
            upper = df[col].quantile(1 - threshold / 100)
            result[flag_col] = (df[col] < lower) | (df[col] > upper)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    return result

@timer
@handle_errors(logger=logger)
def normalize_columns(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'zscore'
) -> pd.DataFrame:
    """
    Normalize specified columns
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    columns : list
        Columns to normalize
    method : str, optional
        Normalization method ('zscore', 'minmax', 'robust')
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized columns
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric, skipping normalization")
            continue
        
        # Create normalized column
        norm_col = f"{col}_norm"
        
        # Apply normalization based on specified method
        if method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                logger.warning(f"Standard deviation is zero for column {col}, setting normalized values to zero")
                result[norm_col] = 0
            else:
                result[norm_col] = (df[col] - mean) / std
        
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val == min_val:
                logger.warning(f"Min and max are equal for column {col}, setting normalized values to 0.5")
                result[norm_col] = 0.5
            else:
                result[norm_col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                logger.warning(f"IQR is zero for column {col}, setting normalized values to zero")
                result[norm_col] = 0
            else:
                result[norm_col] = (df[col] - median) / iqr
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return result

@m1_optimized(use_numba=True)
def compute_price_differentials(
    north_prices: np.ndarray,
    south_prices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute price differentials between north and south regions
    
    Parameters
    ----------
    north_prices : numpy.ndarray
        Array of prices from north region
    south_prices : numpy.ndarray
        Array of prices from south region
        
    Returns
    -------
    tuple
        (absolute_diff, percentage_diff)
    """
    # Compute absolute difference
    absolute_diff = north_prices - south_prices
    
    # Compute percentage difference
    # Avoid division by zero
    percentage_diff = np.zeros_like(absolute_diff)
    mask = south_prices != 0
    percentage_diff[mask] = (absolute_diff[mask] / south_prices[mask]) * 100
    
    return absolute_diff, percentage_diff

@handle_errors(logger=logger)
def aggregate_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: List[str],
    freq: str = 'M',
    agg_func: str = 'mean',
    fillna: bool = True
) -> pd.DataFrame:
    """
    Aggregate time series data to a specified frequency
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    date_col : str
        Column containing dates
    value_cols : list
        Columns containing values to aggregate
    freq : str, optional
        Frequency for aggregation ('D', 'W', 'M', 'Q', 'Y')
    agg_func : str, optional
        Aggregation function ('mean', 'median', 'sum', 'min', 'max')
    fillna : bool, optional
        Whether to fill missing values after aggregation
        
    Returns
    -------
    pandas.DataFrame
        Aggregated time series data
    """
    # Check if date column is datetime type, convert if not
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Map aggregation function string to actual function
    if agg_func == 'mean':
        func = np.mean
    elif agg_func == 'median':
        func = np.median
    elif agg_func == 'sum':
        func = np.sum
    elif agg_func == 'min':
        func = np.min
    elif agg_func == 'max':
        func = np.max
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")
    
    # Set date column as index
    df_indexed = df.set_index(date_col)
    
    # Aggregate data
    df_agg = df_indexed[value_cols].resample(freq).agg(func)
    
    # Fill missing values if requested
    if fillna:
        df_agg = df_agg.fillna(method='ffill').fillna(method='bfill')
    
    # Reset index to get date as a column
    df_agg = df_agg.reset_index()
    
    return df_agg

@handle_errors(logger=logger)
def create_lag_features(
    df: pd.DataFrame,
    cols: List[str],
    lags: List[int],
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create lagged features for time series analysis
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    cols : list
        Columns to create lags for
    lags : list
        List of lag values
    group_cols : list, optional
        Columns to group by before creating lags
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with lag features added
    """
    result = df.copy()
    
    # Sort by group columns and date if provided
    if group_cols:
        # Filter group_cols to those that exist in the dataframe
        valid_group_cols = [col for col in group_cols if col in df.columns]
        if valid_group_cols:
            result = result.sort_values(valid_group_cols)
    
    # Create lag features
    for col in cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            
            if group_cols:
                # Create lags within groups
                valid_group_cols = [col for col in group_cols if col in df.columns]
                if valid_group_cols:
                    result[lag_col] = result.groupby(valid_group_cols)[col].shift(lag)
            else:
                # Create lags without grouping
                result[lag_col] = result[col].shift(lag)
    
    return result

@handle_errors(logger=logger)
def create_rolling_features(
    df: pd.DataFrame,
    cols: List[str],
    windows: List[int],
    stats: List[str] = ['mean', 'std'],
    min_periods: int = 1,
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create rolling window features for time series analysis
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    cols : list
        Columns to create rolling features for
    windows : list
        List of window sizes
    stats : list, optional
        Statistics to compute for each window
    min_periods : int, optional
        Minimum number of observations required
    group_cols : list, optional
        Columns to group by before creating features
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with rolling features added
    """
    result = df.copy()
    
    # Sort by group columns if provided
    if group_cols:
        # Filter group_cols to those that exist in the dataframe
        valid_group_cols = [col for col in group_cols if col in df.columns]
        if valid_group_cols:
            result = result.sort_values(valid_group_cols)
    
    # Create rolling features
    for col in cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric, skipping rolling features")
            continue
        
        for window in windows:
            for stat in stats:
                feat_col = f"{col}_roll{window}_{stat}"
                
                if group_cols:
                    # Create features within groups
                    valid_group_cols = [col for col in group_cols if col in df.columns]
                    if valid_group_cols:
                        if stat == 'mean':
                            result[feat_col] = result.groupby(valid_group_cols)[col].transform(
                                lambda x: x.rolling(window, min_periods=min_periods).mean()
                            )
                        elif stat == 'std':
                            result[feat_col] = result.groupby(valid_group_cols)[col].transform(
                                lambda x: x.rolling(window, min_periods=min_periods).std()
                            )
                        elif stat == 'min':
                            result[feat_col] = result.groupby(valid_group_cols)[col].transform(
                                lambda x: x.rolling(window, min_periods=min_periods).min()
                            )
                        elif stat == 'max':
                            result[feat_col] = result.groupby(valid_group_cols)[col].transform(
                                lambda x: x.rolling(window, min_periods=min_periods).max()
                            )
                        elif stat == 'median':
                            result[feat_col] = result.groupby(valid_group_cols)[col].transform(
                                lambda x: x.rolling(window, min_periods=min_periods).median()
                            )
                        else:
                            logger.warning(f"Unknown rolling statistic: {stat}")
                else:
                    # Create features without grouping
                    if stat == 'mean':
                        result[feat_col] = result[col].rolling(window, min_periods=min_periods).mean()
                    elif stat == 'std':
                        result[feat_col] = result[col].rolling(window, min_periods=min_periods).std()
                    elif stat == 'min':
                        result[feat_col] = result[col].rolling(window, min_periods=min_periods).min()
                    elif stat == 'max':
                        result[feat_col] = result[col].rolling(window, min_periods=min_periods).max()
                    elif stat == 'median':
                        result[feat_col] = result[col].rolling(window, min_periods=min_periods).median()
                    else:
                        logger.warning(f"Unknown rolling statistic: {stat}")
    
    return result

@handle_errors(logger=logger)
def convert_exchange_rates(
    df: pd.DataFrame,
    price_col: str,
    exchange_rate_col: str,
    target_currency: str = 'USD',
    new_col_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert prices from local currency to target currency
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    price_col : str
        Column containing prices in local currency
    exchange_rate_col : str
        Column containing exchange rates
    target_currency : str, optional
        Target currency code
    new_col_name : str, optional
        Name for the new column, defaults to {price_col}_{target_currency}
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with converted prices
    """
    result = df.copy()
    
    # Check if columns exist
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    if exchange_rate_col not in df.columns:
        raise ValueError(f"Exchange rate column '{exchange_rate_col}' not found in DataFrame")
    
    # Set default new column name if not provided
    if new_col_name is None:
        new_col_name = f"{price_col}_{target_currency.lower()}"
    
    # Convert prices
    result[new_col_name] = result[price_col] / result[exchange_rate_col]
    
    return result

@handle_errors(logger=logger)
def calculate_price_changes(
    df: pd.DataFrame,
    price_col: str,
    date_col: str,
    method: str = 'pct',
    periods: List[int] = [1],
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate price changes over specified periods
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    price_col : str
        Column containing prices
    date_col : str
        Column containing dates
    method : str, optional
        Change calculation method ('pct', 'diff', 'log')
    periods : list, optional
        List of periods to calculate changes for
    group_cols : list, optional
        Columns to group by
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with price changes added
    """
    result = df.copy()
    
    # Check if columns exist
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    # Sort by date and group columns if provided
    sort_cols = [date_col]
    if group_cols:
        sort_cols = [col for col in group_cols if col in df.columns] + sort_cols
    
    result = result.sort_values(sort_cols)
    
    # Calculate changes for each period
    for period in periods:
        # Create column name based on method and period
        if method == 'pct':
            change_col = f"{price_col}_pct_change_{period}"
        elif method == 'diff':
            change_col = f"{price_col}_diff_{period}"
        elif method == 'log':
            change_col = f"{price_col}_log_change_{period}"
        else:
            raise ValueError(f"Unknown change calculation method: {method}")
        
        # Calculate changes
        if group_cols:
            # Calculate changes within groups
            valid_group_cols = [col for col in group_cols if col in df.columns]
            if valid_group_cols:
                if method == 'pct':
                    result[change_col] = result.groupby(valid_group_cols)[price_col].pct_change(periods=period)
                elif method == 'diff':
                    result[change_col] = result.groupby(valid_group_cols)[price_col].diff(periods=period)
                elif method == 'log':
                    result[change_col] = result.groupby(valid_group_cols)[price_col].apply(
                        lambda x: np.log(x) - np.log(x.shift(period))
                    )
        else:
            # Calculate changes without grouping
            if method == 'pct':
                result[change_col] = result[price_col].pct_change(periods=period)
            elif method == 'diff':
                result[change_col] = result[price_col].diff(periods=period)
            elif method == 'log':
                result[change_col] = np.log(result[price_col]) - np.log(result[price_col].shift(period))
    
    return result

@handle_errors(logger=logger)
def pivot_data(
    df: pd.DataFrame,
    index_cols: List[str],
    column_col: str,
    value_col: str,
    agg_func: str = 'mean',
    fill_value: Optional[Any] = None
) -> pd.DataFrame:
    """
    Pivot data from long to wide format
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data in long format
    index_cols : list
        Columns to use as index
    column_col : str
        Column to use for new columns
    value_col : str
        Column to use for values
    agg_func : str, optional
        Aggregation function if multiple values per cell
    fill_value : any, optional
        Value to fill missing cells
        
    Returns
    -------
    pandas.DataFrame
        Pivoted data in wide format
    """
    # Map aggregation function string to actual function
    if agg_func == 'mean':
        func = np.mean
    elif agg_func == 'median':
        func = np.median
    elif agg_func == 'sum':
        func = np.sum
    elif agg_func == 'min':
        func = np.min
    elif agg_func == 'max':
        func = np.max
    elif agg_func == 'first':
        func = 'first'
    elif agg_func == 'last':
        func = 'last'
    elif agg_func == 'count':
        func = 'count'
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")
    
    # Pivot data
    result = df.pivot_table(
        index=index_cols,
        columns=column_col,
        values=value_col,
        aggfunc=func,
        fill_value=fill_value
    )
    
    # Reset index to get index columns back
    result = result.reset_index()
    
    return result

@handle_errors(logger=logger)
def unpivot_data(
    df: pd.DataFrame,
    id_vars: List[str],
    value_name: str = 'value',
    var_name: str = 'variable'
) -> pd.DataFrame:
    """
    Unpivot data from wide to long format
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data in wide format
    id_vars : list
        Columns to keep as is
    value_name : str, optional
        Name for value column
    var_name : str, optional
        Name for variable column
        
    Returns
    -------
    pandas.DataFrame
        Unpivoted data in long format
    """
    # Get value columns (all columns not in id_vars)
    value_cols = [col for col in df.columns if col not in id_vars]
    
    # Unpivot data
    result = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_cols,
        var_name=var_name,
        value_name=value_name
    )
    
    return result

@handle_errors(logger=logger)
def merge_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: Optional[List[str]] = None,
    left_on: Optional[List[str]] = None,
    right_on: Optional[List[str]] = None,
    how: str = 'inner',
    suffixes: Tuple[str, str] = ('_x', '_y'),
    validate: Optional[str] = None
) -> pd.DataFrame:
    """
    Safely merge two DataFrames with validation
    
    Parameters
    ----------
    left_df : pandas.DataFrame
        Left DataFrame
    right_df : pandas.DataFrame
        Right DataFrame
    on : list, optional
        Columns to join on (must be in both DataFrames)
    left_on : list, optional
        Columns from left DataFrame to join on
    right_on : list, optional
        Columns from right DataFrame to join on
    how : str, optional
        Type of merge ('inner', 'outer', 'left', 'right')
    suffixes : tuple, optional
        Suffixes for overlapping columns
    validate : str, optional
        Validation mode ('one_to_one', 'one_to_many', 'many_to_one', 'many_to_many')
        
    Returns
    -------
    pandas.DataFrame
        Merged DataFrame
    """
    # Check for empty DataFrames
    if left_df.empty:
        logger.warning("Left DataFrame is empty")
        if how in ['left', 'inner']:
            return pd.DataFrame()
        return right_df.copy()
    
    if right_df.empty:
        logger.warning("Right DataFrame is empty")
        if how in ['right', 'inner']:
            return pd.DataFrame()
        return left_df.copy()
    
    # Log merge info
    log_msg = f"Merging DataFrames with {len(left_df)} and {len(right_df)} rows"
    if on:
        log_msg += f" on columns {on}"
    elif left_on and right_on:
        log_msg += f" on columns {left_on} and {right_on}"
    
    log_msg += f" using {how} join"
    logger.info(log_msg)
    
    # Perform merge
    result = pd.merge(
        left_df,
        right_df,
        on=on,
        left_on=left_on,
        right_on=right_on,
        how=how,
        suffixes=suffixes,
        validate=validate
    )
    
    # Log result info
    logger.info(f"Merge result has {len(result)} rows")
    
    return result

@handle_errors(logger=logger)
def bin_numeric_column(
    df: pd.DataFrame,
    column: str,
    bins: Union[int, List],
    labels: Optional[List[str]] = None,
    right: bool = True,
    include_lowest: bool = True,
    new_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Bin a numeric column into categories
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    column : str
        Column to bin
    bins : int or list
        Number of bins or bin edges
    labels : list, optional
        Labels for bins
    right : bool, optional
        Whether bin intervals include the right edge
    include_lowest : bool, optional
        Whether first interval includes the lowest value
    new_column : str, optional
        Name for new column, defaults to {column}_bin
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with binned column added
    """
    result = df.copy()
    
    # Check if column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    # Set default new column name if not provided
    if new_column is None:
        new_column = f"{column}_bin"
    
    # Create bins
    result[new_column] = pd.cut(
        result[column],
        bins=bins,
        labels=labels,
        right=right,
        include_lowest=include_lowest
    )
    
    return result

@handle_errors(logger=logger)
def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'onehot',
    drop_first: bool = False,
    prefix_sep: str = '_'
) -> pd.DataFrame:
    """
    Encode categorical variables
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    columns : list
        Categorical columns to encode
    method : str, optional
        Encoding method ('onehot', 'label', 'ordinal')
    drop_first : bool, optional
        Whether to drop first category in one-hot encoding
    prefix_sep : str, optional
        Separator for category names
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded variables
    """
    result = df.copy()
    
    # Check if columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Columns {missing_cols} not found in DataFrame")
        columns = [col for col in columns if col in df.columns]
    
    # Encode each column
    if method == 'onehot':
        # Use pandas get_dummies for one-hot encoding
        encoded = pd.get_dummies(
            result[columns],
            drop_first=drop_first,
            prefix_sep=prefix_sep
        )
        # Add encoded columns to result
        result = pd.concat([result, encoded], axis=1)
        
    elif method == 'label':
        # Use sklearn LabelEncoder for label encoding
        from sklearn.preprocessing import LabelEncoder
        
        for col in columns:
            le = LabelEncoder()
            result[f"{col}_encoded"] = le.fit_transform(result[col].astype(str))
            
    elif method == 'ordinal':
        # Use pandas factorize for ordinal encoding
        for col in columns:
            result[f"{col}_encoded"], _ = pd.factorize(result[col])
            
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return result

@handle_errors(logger=logger)
def winsorize_columns(
    df: pd.DataFrame,
    columns: List[str],
    limits: Tuple[float, float] = (0.01, 0.01)
) -> pd.DataFrame:
    """
    Winsorize columns to deal with outliers
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    columns : list
        Columns to winsorize
    limits : tuple, optional
        (lower, upper) percentiles to winsorize at
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with winsorized columns
    """
    from scipy.stats import mstats
    
    result = df.copy()
    
    # Check if columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Columns {missing_cols} not found in DataFrame")
        columns = [col for col in columns if col in df.columns]
    
    # Winsorize each column
    for col in columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric, skipping winsorization")
            continue
        
        # Winsorize column
        result[col] = mstats.winsorize(df[col], limits=limits)
    
    return result

@handle_errors(logger=logger)
def explode_geojson_features(geojson_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Read GeoJSON and explode multi-part geometries into single parts
    
    Parameters
    ----------
    geojson_path : str or Path
        Path to GeoJSON file
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with exploded geometries
    """
    # Read GeoJSON
    gdf = gpd.read_file(geojson_path)
    
    # Check if there are any multi-part geometries
    has_multi = any(gdf.geometry.type.str.startswith('Multi'))
    
    if has_multi:
        # Explode multi-part geometries
        gdf = gdf.explode(index_parts=True)
        gdf = gdf.reset_index(drop=True)
    
    return gdf

@handle_errors(logger=logger)
def create_date_features(
    df: pd.DataFrame,
    date_col: str,
    features: List[str] = ['year', 'month', 'day', 'dayofweek', 'quarter']
) -> pd.DataFrame:
    """
    Create features from a date column
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data
    date_col : str
        Date column
    features : list, optional
        Date features to create
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with date features added
    """
    result = df.copy()
    
    # Check if column exists
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        result[date_col] = pd.to_datetime(df[date_col])
    
    # Create features
    for feature in features:
        feature_col = f"{date_col}_{feature}"
        
        if feature == 'year':
            result[feature_col] = result[date_col].dt.year
        elif feature == 'month':
            result[feature_col] = result[date_col].dt.month
        elif feature == 'day':
            result[feature_col] = result[date_col].dt.day
        elif feature == 'dayofweek':
            result[feature_col] = result[date_col].dt.dayofweek
        elif feature == 'quarter':
            result[feature_col] = result[date_col].dt.quarter
        elif feature == 'weekofyear':
            result[feature_col] = result[date_col].dt.isocalendar().week
        elif feature == 'dayofyear':
            result[feature_col] = result[date_col].dt.dayofyear
        elif feature == 'weekend':
            result[feature_col] = result[date_col].dt.dayofweek >= 5
        elif feature == 'month_start':
            result[feature_col] = result[date_col].dt.is_month_start
        elif feature == 'month_end':
            result[feature_col] = result[date_col].dt.is_month_end
        else:
            logger.warning(f"Unknown date feature: {feature}")
    
    return result

@handle_errors(logger=logger)
def calculate_distance_matrix(
    gdf: gpd.GeoDataFrame,
    id_col: str = 'admin1',
    method: str = 'euclidean',
    crs: Optional[str] = 'EPSG:32638'  # UTM zone 38N (Yemen)
) -> pd.DataFrame:
    """
    Calculate distance matrix between points in a GeoDataFrame
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Spatial points data
    id_col : str, optional
        Column to use as identifier
    method : str, optional
        Distance calculation method ('euclidean', 'great_circle')
    crs : str, optional
        Coordinate reference system for projection
        
    Returns
    -------
    pandas.DataFrame
        Distance matrix
    """
    # Check if GeoDataFrame has points
    if not all(gdf.geometry.type == 'Point'):
        raise ValueError("All geometries must be points")
    
    # Extract unique points (one per identifier)
    unique_gdf = gdf.drop_duplicates(subset=[id_col])
    
    # Convert to appropriate CRS if needed
    if crs is not None and unique_gdf.crs != crs:
        unique_gdf = unique_gdf.to_crs(crs)
    
    # Extract point coordinates
    coords = np.array([(p.x, p.y) for p in unique_gdf.geometry])
    ids = unique_gdf[id_col].values
    
    # Calculate distances
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    
    if method == 'euclidean':
        # Euclidean distance (faster)
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(((coords[i] - coords[j]) ** 2).sum())
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    elif method == 'great_circle':
        # Great circle distance (more accurate for long distances)
        from geopy.distance import great_circle
        
        # Reproject to WGS84 for great circle calculation
        wgs84_gdf = unique_gdf.to_crs('EPSG:4326')
        coords = np.array([(p.y, p.x) for p in wgs84_gdf.geometry])  # Note: lat, lon order
        
        for i in range(n):
            for j in range(i+1, n):
                dist = great_circle(coords[i], coords[j]).meters
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    else:
        raise ValueError(f"Unknown distance calculation method: {method}")
    
    # Create DataFrame with row and column labels
    distance_df = pd.DataFrame(distance_matrix, index=ids, columns=ids)
    
    return distance_df