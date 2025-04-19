"""
Essential data transformation utilities for Yemen Market Analysis.

This module provides core functions for transforming data to improve model performance.
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

@handle_errors
def log_transform(data, column, add_constant=0.01):
    """
    Apply log transformation to a column.
    
    Args:
        data: DataFrame containing the data
        column: Column name to transform
        add_constant: Value to add before taking log (for zero values)
        
    Returns:
        DataFrame with new log-transformed column
    """
    logger.info(f"Applying log transformation to {column}")
    
    # Create a copy of the data
    result = data.copy()
    
    # Check for non-positive values
    min_val = result[column].min()
    if min_val <= 0:
        constant = add_constant + abs(min_val) + 1
        logger.warning(f"Column {column} contains non-positive values. Adding constant {constant}")
    else:
        constant = add_constant
    
    # Apply transformation
    result[f"{column}_log"] = np.log(result[column] + constant)
    
    return result

@handle_errors
def difference(data, column, periods=1):
    """
    Apply differencing to a column.
    
    Args:
        data: DataFrame containing the data
        column: Column name to transform
        periods: Number of periods to difference
        
    Returns:
        DataFrame with new differenced column
    """
    logger.info(f"Applying {periods}-period differencing to {column}")
    
    # Create a copy of the data
    result = data.copy()
    
    # Apply differencing
    result[f"{column}_diff{periods}"] = result[column].diff(periods=periods)
    
    return result

@handle_errors
def remove_outliers(data, column, method='zscore', threshold=3.0):
    """
    Identify and remove outliers from a column.
    
    Args:
        data: DataFrame containing the data
        column: Column name to clean
        method: Method to use ('zscore', 'iqr', or 'modified_zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers replaced with NaN
    """
    logger.info(f"Removing outliers from {column} using {method} method")
    
    # Create a copy of the data
    result = data.copy()
    
    # Detect outliers
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(result[column].dropna()))
        outliers = z_scores > threshold
        outlier_indices = result[column].dropna().index[outliers]
    elif method == 'iqr':
        q1 = result[column].quantile(0.25)
        q3 = result[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (result[column] < lower_bound) | (result[column] > upper_bound)
        outlier_indices = result.index[outliers]
    elif method == 'modified_zscore':
        median = result[column].median()
        mad = np.median(np.abs(result[column] - median))
        modified_z_scores = 0.6745 * np.abs(result[column] - median) / mad
        outliers = modified_z_scores > threshold
        outlier_indices = result.index[outliers]
    else:
        raise YemenAnalysisError(f"Invalid outlier detection method: {method}")
    
    # Replace outliers with median
    result.loc[outlier_indices, column] = result[column].median()
    
    logger.info(f"Removed {len(outlier_indices)} outliers from {column}")
    return result

@handle_errors
def check_stationarity(data, column, alpha=0.05):
    """
    Check if a time series is stationary using ADF and KPSS tests.
    
    Args:
        data: DataFrame containing the data
        column: Column name to check
        alpha: Significance level
        
    Returns:
        Dictionary with test results and stationarity conclusion
    """
    logger.info(f"Checking stationarity of {column}")
    
    # Get the series
    series = data[column].dropna()
    
    # Perform ADF test
    adf_result = adfuller(series)
    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_stationary = adf_pvalue < alpha
    
    # Perform KPSS test
    kpss_result = kpss(series)
    kpss_statistic = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_stationary = kpss_pvalue > alpha
    
    # Determine stationarity
    if adf_stationary and kpss_stationary:
        conclusion = "Stationary"
    elif adf_stationary and not kpss_stationary:
        conclusion = "Difference stationary"
    elif not adf_stationary and kpss_stationary:
        conclusion = "Trend stationary"
    else:
        conclusion = "Non-stationary"
    
    return {
        'adf_test': {'statistic': adf_statistic, 'pvalue': adf_pvalue, 'stationary': adf_stationary},
        'kpss_test': {'statistic': kpss_statistic, 'pvalue': kpss_pvalue, 'stationary': kpss_stationary},
        'conclusion': conclusion
    }

@handle_errors
def make_stationary(data, column):
    """
    Transform a column to make it stationary.
    
    Args:
        data: DataFrame containing the data
        column: Column name to transform
        
    Returns:
        DataFrame with transformed column and the transformation applied
    """
    logger.info(f"Making {column} stationary")
    
    # Create a copy of the data
    result = data.copy()
    
    # Check original stationarity
    original_check = check_stationarity(result, column)
    
    # If already stationary, return original
    if original_check['conclusion'] == "Stationary":
        logger.info(f"{column} is already stationary")
        return result, "none"
    
    # Try log transformation
    if result[column].min() > 0:
        log_result = log_transform(result, column)
        log_check = check_stationarity(log_result, f"{column}_log")
        
        if log_check['conclusion'] == "Stationary":
            logger.info(f"Log transformation made {column} stationary")
            return log_result, "log"
    
    # Try first difference
    diff_result = difference(result, column)
    diff_check = check_stationarity(diff_result, f"{column}_diff1")
    
    if diff_check['conclusion'] == "Stationary":
        logger.info(f"First difference made {column} stationary")
        return diff_result, "diff1"
    
    # Try log + first difference
    if result[column].min() > 0:
        log_diff_result = difference(log_result, f"{column}_log")
        log_diff_check = check_stationarity(log_diff_result, f"{column}_log_diff1")
        
        if log_diff_check['conclusion'] == "Stationary":
            logger.info(f"Log + first difference made {column} stationary")
            return log_diff_result, "log_diff1"
    
    # Try second difference
    diff2_result = difference(diff_result, f"{column}_diff1")
    diff2_check = check_stationarity(diff2_result, f"{column}_diff1_diff1")
    
    if diff2_check['conclusion'] == "Stationary":
        logger.info(f"Second difference made {column} stationary")
        return diff2_result, "diff2"
    
    # If nothing worked, return first difference as default
    logger.warning(f"Could not make {column} stationary, using first difference")
    return diff_result, "diff1"

@handle_errors
def prepare_price_pairs(data, market1, market2, price_col='price', clean_outliers=True):
    """
    Prepare price pairs for threshold model analysis.
    
    Args:
        data: DataFrame containing the data
        market1: First market name
        market2: Second market name
        price_col: Column name for prices
        clean_outliers: Whether to clean outliers
        
    Returns:
        DataFrame with prepared price pairs
    """
    logger.info(f"Preparing price pairs for {market1} and {market2}")
    
    # Filter data for each market
    if 'market' in data.columns:
        market_col = 'market'
    elif 'market_name' in data.columns:
        market_col = 'market_name'
    else:
        raise YemenAnalysisError("No market column found in data")
    
    market1_data = data[data[market_col] == market1].copy()
    market2_data = data[data[market_col] == market2].copy()
    
    # Clean outliers if requested
    if clean_outliers:
        market1_data = remove_outliers(market1_data, price_col)
        market2_data = remove_outliers(market2_data, price_col)
    
    # Make prices stationary
    market1_data, trans1 = make_stationary(market1_data, price_col)
    market2_data, trans2 = make_stationary(market2_data, price_col)
    
    # Get transformed column names
    if trans1 == "none":
        col1 = price_col
    else:
        col1 = f"{price_col}_{trans1}" if trans1 != "diff1" else f"{price_col}_diff1"
    
    if trans2 == "none":
        col2 = price_col
    else:
        col2 = f"{price_col}_{trans2}" if trans2 != "diff1" else f"{price_col}_diff1"
    
    # Merge data on date
    if 'date' not in market1_data.columns or 'date' not in market2_data.columns:
        raise YemenAnalysisError("No date column found in data")
    
    market1_data = market1_data[['date', col1]].rename(columns={col1: f"{market1}_{col1}"})
    market2_data = market2_data[['date', col2]].rename(columns={col2: f"{market2}_{col2}"})
    
    # Merge
    merged = pd.merge(market1_data, market2_data, on='date', how='inner')
    
    logger.info(f"Prepared {len(merged)} price pairs for {market1} and {market2}")
    return merged, col1, col2
