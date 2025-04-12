"""
Data validation module for Yemen Market Analysis.

This module provides functions for validating data used in the Yemen Market Analysis
package. It includes functions for validating GeoJSON data, time series data, and
model parameters.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd

from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

@handle_errors
def validate_data(
    data: Union[pd.DataFrame, gpd.GeoDataFrame], data_type: str = 'dataframe'
) -> bool:
    """
    Validate data for use in Yemen Market Analysis.

    Args:
        data: Data to validate.
        data_type: Type of data to validate. Options are 'dataframe', 'geojson',
                  'time_series', and 'spatial'.

    Returns:
        True if the data is valid.

    Raises:
        YemenAnalysisError: If the data is invalid.
    """
    logger.info(f"Validating {data_type} data")

    # Check if data is empty
    if data is None or len(data) == 0:
        logger.error("Data is empty")
        raise YemenAnalysisError("Data is empty")

    # Validate based on data type
    if data_type == 'dataframe':
        return _validate_dataframe(data)
    elif data_type == 'geojson':
        return _validate_geojson(data)
    elif data_type == 'time_series':
        return _validate_time_series(data)
    elif data_type == 'spatial':
        return _validate_spatial(data)
    else:
        logger.error(f"Invalid data type: {data_type}")
        raise YemenAnalysisError(f"Invalid data type: {data_type}")


def _validate_dataframe(data: pd.DataFrame) -> bool:
    """
    Validate a pandas DataFrame.

    Args:
        data: DataFrame to validate.

    Returns:
        True if the DataFrame is valid.

    Raises:
        YemenAnalysisError: If the DataFrame is invalid.
    """
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        logger.error("Data is not a pandas DataFrame")
        raise YemenAnalysisError("Data is not a pandas DataFrame")

    # Check if data has any rows
    if len(data) == 0:
        logger.error("DataFrame has no rows")
        raise YemenAnalysisError("DataFrame has no rows")

    # Check if data has any columns
    if len(data.columns) == 0:
        logger.error("DataFrame has no columns")
        raise YemenAnalysisError("DataFrame has no columns")

    # Check for duplicate indices
    if data.index.duplicated().any():
        logger.warning("DataFrame has duplicate indices")

    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"DataFrame has missing values: {missing_values}")

    logger.info("DataFrame validation successful")
    return True


def _validate_geojson(data: gpd.GeoDataFrame) -> bool:
    """
    Validate a GeoJSON GeoDataFrame.

    Args:
        data: GeoDataFrame to validate.

    Returns:
        True if the GeoDataFrame is valid.

    Raises:
        YemenAnalysisError: If the GeoDataFrame is invalid.
    """
    # Check if data is a GeoDataFrame
    if not isinstance(data, gpd.GeoDataFrame):
        logger.error("Data is not a GeoDataFrame")
        raise YemenAnalysisError("Data is not a GeoDataFrame")

    # Check if data has any rows
    if len(data) == 0:
        logger.error("GeoDataFrame has no rows")
        raise YemenAnalysisError("GeoDataFrame has no rows")

    # Check if data has any columns
    if len(data.columns) == 0:
        logger.error("GeoDataFrame has no columns")
        raise YemenAnalysisError("GeoDataFrame has no columns")

    # Check if data has a geometry column
    if 'geometry' not in data.columns:
        logger.error("GeoDataFrame has no geometry column")
        raise YemenAnalysisError("GeoDataFrame has no geometry column")

    # Check if geometry column has valid geometries
    if data.geometry.isna().any():
        logger.warning("GeoDataFrame has missing geometries")

    # Check for required columns for Yemen Market Analysis
    required_columns = ['market', 'date', 'price', 'commodity']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        logger.warning(f"GeoDataFrame is missing required columns: {missing_columns}")

    logger.info("GeoDataFrame validation successful")
    return True


def _validate_time_series(data: pd.DataFrame) -> bool:
    """
    Validate a time series DataFrame.

    Args:
        data: DataFrame to validate.

    Returns:
        True if the DataFrame is valid.

    Raises:
        YemenAnalysisError: If the DataFrame is invalid.
    """
    # First, validate as a DataFrame
    _validate_dataframe(data)

    # Check if data has a date column
    if 'date' not in data.columns:
        logger.error("Time series data has no date column")
        raise YemenAnalysisError("Time series data has no date column")

    # Check if date column is a datetime
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        logger.warning("Date column is not a datetime, attempting to convert")
        try:
            data['date'] = pd.to_datetime(data['date'])
        except Exception as e:
            logger.error(f"Error converting date column to datetime: {e}")
            raise YemenAnalysisError(f"Error converting date column to datetime: {e}")

    # Check if data has a price column
    if 'price' not in data.columns:
        logger.error("Time series data has no price column")
        raise YemenAnalysisError("Time series data has no price column")

    # Check if price column is numeric
    if not pd.api.types.is_numeric_dtype(data['price']):
        logger.error("Price column is not numeric")
        raise YemenAnalysisError("Price column is not numeric")

    # Check for negative prices
    if (data['price'] < 0).any():
        logger.warning("Time series data has negative prices")

    # Check for duplicate dates
    if data.duplicated(subset=['date']).any():
        logger.warning("Time series data has duplicate dates")

    # Check for gaps in time series
    date_diff = data['date'].diff().dropna()
    if len(date_diff.unique()) > 1:
        logger.warning("Time series data has irregular time intervals")

    logger.info("Time series validation successful")
    return True


def _validate_spatial(data: gpd.GeoDataFrame) -> bool:
    """
    Validate spatial data.

    Args:
        data: GeoDataFrame to validate.

    Returns:
        True if the GeoDataFrame is valid.

    Raises:
        YemenAnalysisError: If the GeoDataFrame is invalid.
    """
    # First, validate as a GeoDataFrame
    _validate_geojson(data)

    # Check if data has a coordinate reference system (CRS)
    if data.crs is None:
        logger.warning("Spatial data has no coordinate reference system (CRS)")

    # Check for required columns for spatial analysis
    required_columns = ['market', 'date', 'price', 'commodity', 'geometry']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        logger.warning(f"Spatial data is missing required columns: {missing_columns}")

    # Check for conflict data if available
    if 'conflict_intensity' in data.columns:
        if not pd.api.types.is_numeric_dtype(data['conflict_intensity']):
            logger.warning("Conflict intensity column is not numeric")

    logger.info("Spatial data validation successful")
    return True


@handle_errors
def validate_model_parameters(
    model_type: str, params: Dict[str, Any]
) -> bool:
    """
    Validate model parameters.

    Args:
        model_type: Type of model. Options are 'unit_root', 'cointegration',
                   'threshold', 'spatial', and 'vecm'.
        params: Dictionary of model parameters.

    Returns:
        True if the parameters are valid.

    Raises:
        YemenAnalysisError: If the parameters are invalid.
    """
    logger.info(f"Validating {model_type} model parameters")

    # Validate based on model type
    if model_type == 'unit_root':
        return _validate_unit_root_params(params)
    elif model_type == 'cointegration':
        return _validate_cointegration_params(params)
    elif model_type == 'threshold':
        return _validate_threshold_params(params)
    elif model_type == 'spatial':
        return _validate_spatial_params(params)
    elif model_type == 'vecm':
        return _validate_vecm_params(params)
    else:
        logger.error(f"Invalid model type: {model_type}")
        raise YemenAnalysisError(f"Invalid model type: {model_type}")


def _validate_unit_root_params(params: Dict[str, Any]) -> bool:
    """
    Validate unit root test parameters.

    Args:
        params: Dictionary of unit root test parameters.

    Returns:
        True if the parameters are valid.

    Raises:
        YemenAnalysisError: If the parameters are invalid.
    """
    # Check for required parameters
    required_params = ['trend', 'max_lags']
    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        logger.error(f"Unit root test is missing required parameters: {missing_params}")
        raise YemenAnalysisError(f"Unit root test is missing required parameters: {missing_params}")

    # Validate trend parameter
    valid_trends = ['c', 'ct', 'ctt', 'n']
    if params['trend'] not in valid_trends:
        logger.error(f"Invalid trend parameter: {params['trend']}")
        raise YemenAnalysisError(f"Invalid trend parameter: {params['trend']}")

    # Validate max_lags parameter
    if not isinstance(params['max_lags'], int) or params['max_lags'] < 0:
        logger.error(f"Invalid max_lags parameter: {params['max_lags']}")
        raise YemenAnalysisError(f"Invalid max_lags parameter: {params['max_lags']}")

    logger.info("Unit root test parameters validation successful")
    return True


def _validate_cointegration_params(params: Dict[str, Any]) -> bool:
    """
    Validate cointegration test parameters.

    Args:
        params: Dictionary of cointegration test parameters.

    Returns:
        True if the parameters are valid.

    Raises:
        YemenAnalysisError: If the parameters are invalid.
    """
    # Check for required parameters
    required_params = ['trend', 'max_lags']
    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        logger.error(f"Cointegration test is missing required parameters: {missing_params}")
        raise YemenAnalysisError(f"Cointegration test is missing required parameters: {missing_params}")

    # Validate trend parameter
    valid_trends = ['c', 'ct', 'ctt', 'n']
    if params['trend'] not in valid_trends:
        logger.error(f"Invalid trend parameter: {params['trend']}")
        raise YemenAnalysisError(f"Invalid trend parameter: {params['trend']}")

    # Validate max_lags parameter
    if not isinstance(params['max_lags'], int) or params['max_lags'] < 0:
        logger.error(f"Invalid max_lags parameter: {params['max_lags']}")
        raise YemenAnalysisError(f"Invalid max_lags parameter: {params['max_lags']}")

    # Validate test parameter if present
    if 'test' in params:
        valid_tests = ['eg', 'johansen', 'gh']
        if params['test'] not in valid_tests:
            logger.error(f"Invalid test parameter: {params['test']}")
            raise YemenAnalysisError(f"Invalid test parameter: {params['test']}")

    logger.info("Cointegration test parameters validation successful")
    return True


def _validate_threshold_params(params: Dict[str, Any]) -> bool:
    """
    Validate threshold model parameters.

    Args:
        params: Dictionary of threshold model parameters.

    Returns:
        True if the parameters are valid.

    Raises:
        YemenAnalysisError: If the parameters are invalid.
    """
    # Check for required parameters
    required_params = ['max_lags', 'trim']
    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        logger.error(f"Threshold model is missing required parameters: {missing_params}")
        raise YemenAnalysisError(f"Threshold model is missing required parameters: {missing_params}")

    # Validate max_lags parameter
    if not isinstance(params['max_lags'], int) or params['max_lags'] < 0:
        logger.error(f"Invalid max_lags parameter: {params['max_lags']}")
        raise YemenAnalysisError(f"Invalid max_lags parameter: {params['max_lags']}")

    # Validate trim parameter
    if not isinstance(params['trim'], (int, float)) or params['trim'] <= 0 or params['trim'] >= 0.5:
        logger.error(f"Invalid trim parameter: {params['trim']}")
        raise YemenAnalysisError(f"Invalid trim parameter: {params['trim']}")

    # Validate model parameter if present
    if 'model' in params:
        valid_models = ['tar', 'mtar', 'tvecm']
        if params['model'] not in valid_models:
            logger.error(f"Invalid model parameter: {params['model']}")
            raise YemenAnalysisError(f"Invalid model parameter: {params['model']}")

    # Validate threshold parameter if present
    if 'threshold' in params and params['threshold'] is not None:
        if not isinstance(params['threshold'], (int, float)):
            logger.error(f"Invalid threshold parameter: {params['threshold']}")
            raise YemenAnalysisError(f"Invalid threshold parameter: {params['threshold']}")

    logger.info("Threshold model parameters validation successful")
    return True


def _validate_spatial_params(params: Dict[str, Any]) -> bool:
    """
    Validate spatial model parameters.

    Args:
        params: Dictionary of spatial model parameters.

    Returns:
        True if the parameters are valid.

    Raises:
        YemenAnalysisError: If the parameters are invalid.
    """
    # Check for required parameters
    required_params = ['conflict_column', 'price_column']
    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        logger.error(f"Spatial model is missing required parameters: {missing_params}")
        raise YemenAnalysisError(f"Spatial model is missing required parameters: {missing_params}")

    # Validate conflict_weight parameter if present
    if 'conflict_weight' in params:
        if not isinstance(params['conflict_weight'], (int, float)) or params['conflict_weight'] < 0 or params['conflict_weight'] > 1:
            logger.error(f"Invalid conflict_weight parameter: {params['conflict_weight']}")
            raise YemenAnalysisError(f"Invalid conflict_weight parameter: {params['conflict_weight']}")

    # Validate conflict_reduction parameter if present
    if 'conflict_reduction' in params:
        if not isinstance(params['conflict_reduction'], (int, float)) or params['conflict_reduction'] < 0 or params['conflict_reduction'] > 1:
            logger.error(f"Invalid conflict_reduction parameter: {params['conflict_reduction']}")
            raise YemenAnalysisError(f"Invalid conflict_reduction parameter: {params['conflict_reduction']}")

    logger.info("Spatial model parameters validation successful")
    return True


def _validate_vecm_params(params: Dict[str, Any]) -> bool:
    """
    Validate VECM parameters.

    Args:
        params: Dictionary of VECM parameters.

    Returns:
        True if the parameters are valid.

    Raises:
        YemenAnalysisError: If the parameters are invalid.
    """
    # Check for required parameters
    required_params = ['k_ar_diff', 'deterministic', 'coint_rank']
    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        logger.error(f"VECM is missing required parameters: {missing_params}")
        raise YemenAnalysisError(f"VECM is missing required parameters: {missing_params}")

    # Validate k_ar_diff parameter
    if not isinstance(params['k_ar_diff'], int) or params['k_ar_diff'] < 1:
        logger.error(f"Invalid k_ar_diff parameter: {params['k_ar_diff']}")
        raise YemenAnalysisError(f"Invalid k_ar_diff parameter: {params['k_ar_diff']}")

    # Validate deterministic parameter
    valid_deterministics = ['nc', 'co', 'ci', 'lo', 'li', 'cili']
    if params['deterministic'] not in valid_deterministics:
        logger.error(f"Invalid deterministic parameter: {params['deterministic']}")
        raise YemenAnalysisError(f"Invalid deterministic parameter: {params['deterministic']}")

    # Validate coint_rank parameter
    if not isinstance(params['coint_rank'], int) or params['coint_rank'] < 0:
        logger.error(f"Invalid coint_rank parameter: {params['coint_rank']}")
        raise YemenAnalysisError(f"Invalid coint_rank parameter: {params['coint_rank']}")

    logger.info("VECM parameters validation successful")
    return True