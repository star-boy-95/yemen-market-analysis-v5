"""
Data validation utilities for the Yemen Market Integration Project.
Provides functions to validate input data and model parameters.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
import logging
from pathlib import Path
import json
import re

from .error_handler import ValidationError

logger = logging.getLogger(__name__)

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, type]] = None,
    min_rows: int = 1,
    check_nulls: bool = True,
    custom_validators: Optional[Dict[str, Callable[[pd.Series], bool]]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a pandas DataFrame
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate
    required_columns : list of str, optional
        Required column names
    column_types : dict, optional
        Mapping of column names to expected types
    min_rows : int, optional
        Minimum number of rows
    check_nulls : bool, optional
        Whether to check for null values
    custom_validators : dict, optional
        Custom validation functions for specific columns
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    errors = []
    
    # Check if object is a DataFrame
    if not isinstance(df, pd.DataFrame):
        errors.append(f"Expected pandas DataFrame, got {type(df)}")
        return False, errors
    
    # Check minimum number of rows
    if len(df) < min_rows:
        errors.append(f"DataFrame has {len(df)} rows, expected at least {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check column types
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                # Handle special case for datetime
                if expected_type == pd.Timestamp:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        errors.append(f"Column '{col}' is not datetime type")
                else:
                    # For other types, check using isinstance and handle special cases
                    if expected_type == int:
                        if not pd.api.types.is_integer_dtype(df[col]):
                            errors.append(f"Column '{col}' is not integer type")
                    elif expected_type == float:
                        if not pd.api.types.is_float_dtype(df[col]):
                            errors.append(f"Column '{col}' is not float type")
                    elif expected_type == str:
                        if not pd.api.types.is_string_dtype(df[col]):
                            errors.append(f"Column '{col}' is not string type")
                    else:
                        # For other types, check each value
                        invalid_values = df[~df[col].apply(lambda x: isinstance(x, expected_type))][col]
                        if len(invalid_values) > 0:
                            errors.append(f"Column '{col}' has {len(invalid_values)} values not of type {expected_type.__name__}")
            else:
                errors.append(f"Column '{col}' not found in DataFrame")
    
    # Check for null values
    if check_nulls:
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        if not columns_with_nulls.empty:
            for col, count in columns_with_nulls.items():
                errors.append(f"Column '{col}' has {count} null values")
    
    # Apply custom validators
    if custom_validators:
        for col, validator in custom_validators.items():
            if col in df.columns:
                try:
                    if not validator(df[col]):
                        errors.append(f"Custom validation failed for column '{col}'")
                except Exception as e:
                    errors.append(f"Error during custom validation for column '{col}': {str(e)}")
            else:
                errors.append(f"Column '{col}' not found for custom validation")
    
    return len(errors) == 0, errors

def validate_exchange_rate_regime(value: str) -> bool:
    """
    Validate exchange rate regime value
    
    Parameters
    ----------
    value : str
        Exchange rate regime to validate
        
    Returns
    -------
    bool
        True if valid
    """
    return value in ['north', 'south']

def validate_admin_region(region: str, valid_regions: Optional[List[str]] = None) -> bool:
    """
    Validate an administrative region in Yemen
    
    Parameters
    ----------
    region : str
        Region to validate
    valid_regions : list, optional
        List of valid regions
        
    Returns
    -------
    bool
        True if valid
    """
    # Get regions from configuration if not provided
    if not valid_regions:
        from src.utils import config
        north = config.get('regions.north', [])
        south = config.get('regions.south', [])
        valid_regions = north + south
    
    # Normalize input
    region = region.lower().strip()
    valid_regions = [r.lower().strip() for r in valid_regions]
    
    return region in valid_regions

def validate_geodataframe(
    gdf: gpd.GeoDataFrame,
    crs: Optional[str] = None,
    geometry_type: Optional[str] = None,
    **kwargs
) -> Tuple[bool, List[str]]:
    """
    Validate a GeoDataFrame
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to validate
    crs : str, optional
        Expected coordinate reference system
    geometry_type : str, optional
        Expected geometry type (Point, LineString, Polygon, etc.)
    **kwargs : dict
        Additional arguments to pass to validate_dataframe
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    # First validate as a DataFrame
    is_valid, errors = validate_dataframe(gdf, **kwargs)
    
    # Check if object is a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        errors.append(f"Expected GeoDataFrame, got {type(gdf)}")
        return False, errors
    
    # Check if geometry column exists
    if 'geometry' not in gdf.columns:
        errors.append("Missing 'geometry' column")
    
    # Check CRS
    if crs and gdf.crs:
        if str(gdf.crs) != str(crs):
            errors.append(f"CRS mismatch: expected {crs}, got {gdf.crs}")
    
    # Check geometry type
    if geometry_type and len(gdf) > 0:
        geom_types = gdf.geometry.type.unique()
        if geometry_type not in geom_types:
            errors.append(f"Expected geometry type {geometry_type}, got {', '.join(geom_types)}")
    
    return len(errors) == 0, errors

def validate_time_series(
    series: Union[pd.Series, np.ndarray],
    min_length: int = 30,
    max_nulls: int = 0,
    check_stationarity: bool = False,
    check_constant: bool = True,
    custom_validators: Optional[List[Callable[[Union[pd.Series, np.ndarray]], bool]]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a time series for econometric analysis
    
    Parameters
    ----------
    series : pandas.Series or numpy.ndarray
        Time series to validate
    min_length : int, optional
        Minimum length of the series
    max_nulls : int, optional
        Maximum allowed null values
    check_stationarity : bool, optional
        Whether to check for stationarity
    check_constant : bool, optional
        Whether to check for constant values
    custom_validators : list of callable, optional
        List of custom validation functions that take a series and return a boolean
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    errors = []
    
    # Convert to numpy array if needed
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = series
    
    # Check length
    if len(values) < min_length:
        errors.append(f"Time series length is {len(values)}, expected at least {min_length}")
    
    # Check nulls
    null_count = np.isnan(values).sum()
    if null_count > max_nulls:
        errors.append(f"Time series has {null_count} null values, maximum allowed is {max_nulls}")
    
    # Check for constant values
    if check_constant and len(values) > 1:
        non_nan_values = values[~np.isnan(values)]
        if len(non_nan_values) > 0 and np.all(non_nan_values == non_nan_values[0]):
            errors.append("Time series has constant values")
    
    # Check stationarity
    if check_stationarity and len(values) >= 10:
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(values, regression='c')
            pvalue = result[1]
            if pvalue > 0.05:
                errors.append(f"Time series may not be stationary (ADF test p-value: {pvalue:.4f})")
        except Exception as e:
            errors.append(f"Error checking stationarity: {str(e)}")
    
    # Apply custom validators
    if custom_validators:
        for i, validator in enumerate(custom_validators):
            try:
                if not validator(series):
                    errors.append(f"Custom validation {i+1} failed")
            except Exception as e:
                errors.append(f"Error in custom validation {i+1}: {str(e)}")
    
    return len(errors) == 0, errors

def validate_model_inputs(
    model_name: str,
    params: Dict[str, Any],
    required_params: Set[str],
    param_validators: Dict[str, Callable[[Any], bool]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate inputs for a model
    
    Parameters
    ----------
    model_name : str
        Name of the model for error reporting
    params : dict
        Model parameters
    required_params : set
        Required parameter names
    param_validators : dict, optional
        Validators for specific parameters
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    errors = []
    
    # Check required parameters
    missing_params = required_params - set(params.keys())
    if missing_params:
        errors.append(f"Missing required parameters for {model_name}: {', '.join(missing_params)}")
    
    # Validate parameter values
    if param_validators:
        for param, validator in param_validators.items():
            if param in params:
                try:
                    if not validator(params[param]):
                        errors.append(f"Invalid value for parameter '{param}': {params[param]}")
                except Exception as e:
                    errors.append(f"Error validating parameter '{param}': {str(e)}")
    
    return len(errors) == 0, errors

def validate_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
        
    Returns
    -------
    bool
        True if the file exists
    """
    return Path(file_path).is_file()

def validate_dir_exists(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory exists
    
    Parameters
    ----------
    dir_path : str or Path
        Path to the directory
        
    Returns
    -------
    bool
        True if the directory exists
    """
    return Path(dir_path).is_dir()

def validate_is_json(text: str) -> bool:
    """
    Check if a string is valid JSON
    
    Parameters
    ----------
    text : str
        Text to validate
        
    Returns
    -------
    bool
        True if the text is valid JSON
    """
    try:
        json.loads(text)
        return True
    except ValueError:
        return False

def validate_geojson(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate GeoJSON data
    
    Parameters
    ----------
    data : dict
        GeoJSON data
        
    Returns
    -------
    tuple
        (is_valid, error_messages)
    """
    errors = []
    
    # Check if data is a dictionary
    if not isinstance(data, dict):
        errors.append(f"Expected dictionary, got {type(data)}")
        return False, errors
    
    # Check if type is specified
    if 'type' not in data:
        errors.append("Missing 'type' field")
        return False, errors
    
    # Check type value
    if data['type'] not in ['FeatureCollection', 'Feature', 'Point', 'LineString', 
                           'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon', 
                           'GeometryCollection']:
        errors.append(f"Invalid type: {data['type']}")
    
    # Check for FeatureCollection
    if data['type'] == 'FeatureCollection':
        if 'features' not in data:
            errors.append("FeatureCollection missing 'features' array")
        elif not isinstance(data['features'], list):
            errors.append("'features' field is not an array")
    
    # Check for Feature
    if data['type'] == 'Feature':
        if 'geometry' not in data:
            errors.append("Feature missing 'geometry' field")
        if 'properties' not in data:
            errors.append("Feature missing 'properties' field")
    
    # Check for geometry types
    if data['type'] in ['Point', 'LineString', 'Polygon', 'MultiPoint', 
                       'MultiLineString', 'MultiPolygon']:
        if 'coordinates' not in data:
            errors.append(f"{data['type']} missing 'coordinates' field")
    
    return len(errors) == 0, errors

def validate_exchange_rate_regime(value: str) -> bool:
    """
    Validate exchange rate regime value
    
    Parameters
    ----------
    value : str
        Exchange rate regime to validate
        
    Returns
    -------
    bool
        True if valid
    """
    return value in ['north', 'south']

def validate_date_string(date_str: str, format: str = "%Y-%m-%d") -> bool:
    """
    Validate a date string
    
    Parameters
    ----------
    date_str : str
        Date string to validate
    format : str, optional
        Expected date format
        
    Returns
    -------
    bool
        True if valid
    """
    try:
        pd.to_datetime(date_str, format=format)
        return True
    except ValueError:
        return False

def validate_admin_region(region: str, valid_regions: List[str]) -> bool:
    """
    Validate an administrative region
    
    Parameters
    ----------
    region : str
        Region to validate
    valid_regions : list
        List of valid regions
        
    Returns
    -------
    bool
        True if valid
    """
    return region in valid_regions

def validate_commodity(commodity: str, valid_commodities: List[str]) -> bool:
    """
    Validate a commodity
    
    Parameters
    ----------
    commodity : str
        Commodity to validate
    valid_commodities : list
        List of valid commodities
        
    Returns
    -------
    bool
        True if valid
    """
    return commodity in valid_commodities

def validate_phone_number(phone: str) -> bool:
    """
    Validate a phone number format
    
    Parameters
    ----------
    phone : str
        Phone number to validate
        
    Returns
    -------
    bool
        True if valid
    """
    pattern = r'^\+?[0-9]{8,15}$'
    return bool(re.match(pattern, phone))

def validate_email(email: str) -> bool:
    """
    Validate an email address
    
    Parameters
    ----------
    email : str
        Email to validate
        
    Returns
    -------
    bool
        True if valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_latitude(lat: float) -> bool:
    """
    Validate a latitude value
    
    Parameters
    ----------
    lat : float
        Latitude to validate
        
    Returns
    -------
    bool
        True if valid
    """
    return -90 <= lat <= 90

def validate_longitude(lon: float) -> bool:
    """
    Validate a longitude value
    
    Parameters
    ----------
    lon : float
        Longitude to validate
        
    Returns
    -------
    bool
        True if valid
    """
    return -180 <= lon <= 180

def validate_percentage(value: float) -> bool:
    """
    Validate a percentage value
    
    Parameters
    ----------
    value : float
        Percentage to validate
        
    Returns
    -------
    bool
        True if valid
    """
    return 0 <= value <= 100

def validate_data(gdf: gpd.GeoDataFrame, logger: logging.Logger) -> bool:
    """
    Validate the input data for the Yemen Market Integration analysis.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the market data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    bool
        True if data is valid, False otherwise
    """
    logger.info("Validating input data")
    
    # Check if it's a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        logger.error("Input data is not a GeoDataFrame")
        return False
    
    # Check required columns
    required_columns = ['date', 'commodity', 'price', 'admin1', 'geometry']
    missing_columns = set(required_columns) - set(gdf.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        return False
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(gdf['date']):
        logger.error("'date' column is not datetime type")
        return False
    
    if not pd.api.types.is_numeric_dtype(gdf['price']):
        logger.error("'price' column is not numeric type")
        return False
    
    # Check for null values in critical columns
    for col in ['date', 'commodity', 'price', 'admin1']:
        if col in gdf.columns:
            null_count = gdf[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Column '{col}' has {null_count} null values")
    
    # Check for valid geometry
    if gdf.geometry.isna().any():
        logger.warning(f"Found {gdf.geometry.isna().sum()} rows with missing geometry")
    
    # Check for sufficient data
    if len(gdf) < 10:
        logger.warning(f"Limited data available: only {len(gdf)} observations")
    
    logger.info("Data validation completed")
    return True

def raise_if_invalid(is_valid: bool, errors: List[str], error_msg: str = "Validation failed") -> None:
    """
    Raise ValidationError if validation failed
    
    Parameters
    ----------
    is_valid : bool
        Validation result
    errors : list
        List of error messages
    error_msg : str, optional
        Main error message
        
    Raises
    ------
    ValidationError
        If validation failed
    """
    if not is_valid:
        detailed_msg = f"{error_msg}: {'; '.join(errors)}"
        raise ValidationError(detailed_msg)