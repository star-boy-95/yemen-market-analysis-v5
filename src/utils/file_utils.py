"""
File utilities module for Yemen Market Analysis.

This module provides utilities for file operations, including loading and saving
data in various formats, handling common file operations, and managing paths.
"""
import logging
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, BinaryIO

import pandas as pd
import numpy as np
import geopandas as gpd
import yaml

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

@handle_errors
def create_directory(directory: Union[str, Path], overwrite: bool = False) -> Path:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Path to the directory.
        overwrite: Whether to overwrite an existing directory.
        
    Returns:
        Path to the created directory.
        
    Raises:
        YemenAnalysisError: If the directory cannot be created.
    """
    logger.info(f"Creating directory: {directory}")
    
    # Convert to Path object
    directory = Path(directory)
    
    try:
        # Check if directory exists
        if directory.exists():
            if not directory.is_dir():
                logger.error(f"{directory} exists but is not a directory")
                raise YemenAnalysisError(f"{directory} exists but is not a directory")
            
            if overwrite:
                logger.warning(f"Overwriting directory: {directory}")
                import shutil
                shutil.rmtree(directory)
                directory.mkdir(parents=True)
            else:
                logger.info(f"Directory already exists: {directory}")
        else:
            # Create directory
            directory.mkdir(parents=True)
            logger.info(f"Created directory: {directory}")
        
        return directory
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise YemenAnalysisError(f"Error creating directory {directory}: {e}")

@handle_errors
def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        File extension without the dot.
        
    Raises:
        YemenAnalysisError: If the file path is invalid.
    """
    logger.debug(f"Getting file extension for: {file_path}")
    
    try:
        # Convert to Path object
        file_path = Path(file_path)
        
        # Get extension
        extension = file_path.suffix.lower()
        
        # Remove dot
        if extension.startswith('.'):
            extension = extension[1:]
        
        return extension
    except Exception as e:
        logger.error(f"Error getting file extension for {file_path}: {e}")
        raise YemenAnalysisError(f"Error getting file extension for {file_path}: {e}")

@handle_errors
def load_data(
    file_path: Union[str, Path], file_type: Optional[str] = None, **kwargs
) -> Union[pd.DataFrame, gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Load data from a file.
    
    Args:
        file_path: Path to the file.
        file_type: Type of file. If None, inferred from the file extension.
        **kwargs: Additional arguments to pass to the loader function.
        
    Returns:
        Loaded data.
        
    Raises:
        YemenAnalysisError: If the file cannot be loaded or the file type is invalid.
    """
    logger.info(f"Loading data from: {file_path}")
    
    # Convert to Path object
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        raise YemenAnalysisError(f"File does not exist: {file_path}")
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = get_file_extension(file_path)
    
    try:
        # Load data based on file type
        if file_type in ['csv', 'txt']:
            # Load CSV or text file
            return pd.read_csv(file_path, **kwargs)
        elif file_type == 'excel' or file_type in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
            # Load Excel file
            return pd.read_excel(file_path, **kwargs)
        elif file_type == 'json':
            # Load JSON file
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_type == 'geojson':
            # Load GeoJSON file
            return gpd.read_file(file_path, **kwargs)
        elif file_type == 'shp':
            # Load Shapefile
            return gpd.read_file(file_path, **kwargs)
        elif file_type == 'pickle' or file_type == 'pkl':
            # Load Pickle file
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_type == 'parquet':
            # Load Parquet file
            return pd.read_parquet(file_path, **kwargs)
        elif file_type == 'feather':
            # Load Feather file
            return pd.read_feather(file_path, **kwargs)
        elif file_type == 'hdf5' or file_type == 'h5':
            # Load HDF5 file
            return pd.read_hdf(file_path, **kwargs)
        elif file_type == 'yaml' or file_type == 'yml':
            # Load YAML file
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise YemenAnalysisError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise YemenAnalysisError(f"Error loading data from {file_path}: {e}")

@handle_errors
def save_data(
    data: Union[pd.DataFrame, gpd.GeoDataFrame, Dict[str, Any]],
    file_path: Union[str, Path], file_type: Optional[str] = None,
    create_dirs: bool = True, **kwargs
) -> None:
    """
    Save data to a file.
    
    Args:
        data: Data to save.
        file_path: Path to the file.
        file_type: Type of file. If None, inferred from the file extension.
        create_dirs: Whether to create directories if they don't exist.
        **kwargs: Additional arguments to pass to the saver function.
        
    Raises:
        YemenAnalysisError: If the data cannot be saved or the file type is invalid.
    """
    logger.info(f"Saving data to: {file_path}")
    
    # Convert to Path object
    file_path = Path(file_path)
    
    # Create directories if requested
    if create_dirs:
        create_directory(file_path.parent)
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = get_file_extension(file_path)
    
    try:
        # Save data based on file type
        if file_type in ['csv', 'txt']:
            # Save to CSV or text file
            if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
                data.to_csv(file_path, **kwargs)
            else:
                pd.DataFrame(data).to_csv(file_path, **kwargs)
        elif file_type == 'excel' or file_type in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
            # Save to Excel file
            if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
                data.to_excel(file_path, **kwargs)
            else:
                pd.DataFrame(data).to_excel(file_path, **kwargs)
        elif file_type == 'json':
            # Save to JSON file
            if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
                data.to_json(file_path, **kwargs)
            else:
                with open(file_path, 'w') as f:
                    json.dump(data, f, **kwargs)
        elif file_type == 'geojson':
            # Save to GeoJSON file
            if isinstance(data, gpd.GeoDataFrame):
                data.to_file(file_path, driver='GeoJSON', **kwargs)
            else:
                logger.error("Data is not a GeoDataFrame, cannot save as GeoJSON")
                raise YemenAnalysisError("Data is not a GeoDataFrame, cannot save as GeoJSON")
        elif file_type == 'shp':
            # Save to Shapefile
            if isinstance(data, gpd.GeoDataFrame):
                data.to_file(file_path, **kwargs)
            else:
                logger.error("Data is not a GeoDataFrame, cannot save as Shapefile")
                raise YemenAnalysisError("Data is not a GeoDataFrame, cannot save as Shapefile")
        elif file_type == 'pickle' or file_type == 'pkl':
            # Save to Pickle file
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        elif file_type == 'parquet':
            # Save to Parquet file
            if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
                data.to_parquet(file_path, **kwargs)
            else:
                pd.DataFrame(data).to_parquet(file_path, **kwargs)
        elif file_type == 'feather':
            # Save to Feather file
            if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
                data.to_feather(file_path, **kwargs)
            else:
                pd.DataFrame(data).to_feather(file_path, **kwargs)
        elif file_type == 'hdf5' or file_type == 'h5':
            # Save to HDF5 file
            if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
                data.to_hdf(file_path, key='data', **kwargs)
            else:
                pd.DataFrame(data).to_hdf(file_path, key='data', **kwargs)
        elif file_type == 'yaml' or file_type == 'yml':
            # Save to YAML file
            with open(file_path, 'w') as f:
                yaml.dump(data, f, **kwargs)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise YemenAnalysisError(f"Unsupported file type: {file_type}")
        
        logger.info(f"Data saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise YemenAnalysisError(f"Error saving data to {file_path}: {e}")

@handle_errors
def list_files(
    directory: Union[str, Path], pattern: str = '*',
    recursive: bool = False, sort: bool = True
) -> List[Path]:
    """
    List files in a directory.
    
    Args:
        directory: Path to the directory.
        pattern: Pattern to match files.
        recursive: Whether to search recursively.
        sort: Whether to sort the files.
        
    Returns:
        List of file paths.
        
    Raises:
        YemenAnalysisError: If the directory does not exist or is not a directory.
    """
    logger.info(f"Listing files in: {directory}")
    
    # Convert to Path object
    directory = Path(directory)
    
    # Check if directory exists
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        raise YemenAnalysisError(f"Directory does not exist: {directory}")
    
    # Check if it's a directory
    if not directory.is_dir():
        logger.error(f"{directory} is not a directory")
        raise YemenAnalysisError(f"{directory} is not a directory")
    
    try:
        # List files
        if recursive:
            files = list(directory.glob(f"**/{pattern}"))
        else:
            files = list(directory.glob(pattern))
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        # Sort files
        if sort:
            files.sort()
        
        logger.info(f"Found {len(files)} files in {directory}")
        return files
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        raise YemenAnalysisError(f"Error listing files in {directory}: {e}")

@handle_errors
def merge_csv_files(
    input_files: List[Union[str, Path]], output_file: Union[str, Path],
    id_cols: Optional[List[str]] = None, **kwargs
) -> pd.DataFrame:
    """
    Merge multiple CSV files into a single CSV file.
    
    Args:
        input_files: List of input file paths.
        output_file: Path to the output file.
        id_cols: Columns to use as identifiers for merging. If None, all columns
                are used.
        **kwargs: Additional arguments to pass to the pandas merge function.
        
    Returns:
        Merged DataFrame.
        
    Raises:
        YemenAnalysisError: If the files cannot be merged.
    """
    logger.info(f"Merging {len(input_files)} CSV files into {output_file}")
    
    try:
        # Load the first file
        if not input_files:
            logger.error("No input files provided")
            raise YemenAnalysisError("No input files provided")
        
        # Load all dataframes
        dfs = []
        for file_path in input_files:
            df = load_data(file_path, file_type='csv')
            dfs.append(df)
        
        # Merge dataframes
        if id_cols is None:
            # Concatenate dataframes
            merged_df = pd.concat(dfs, ignore_index=True)
        else:
            # Merge dataframes on id_cols
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on=id_cols, **kwargs)
        
        # Save merged dataframe
        save_data(merged_df, output_file, file_type='csv')
        
        logger.info(f"Merged {len(input_files)} CSV files into {output_file}")
        return merged_df
    except Exception as e:
        logger.error(f"Error merging CSV files: {e}")
        raise YemenAnalysisError(f"Error merging CSV files: {e}")

@handle_errors
def get_data_path(file_name: str, data_type: str = 'raw') -> Path:
    """
    Get the path to a data file.
    
    Args:
        file_name: Name of the file.
        data_type: Type of data. Options are 'raw', 'processed', and 'results'.
        
    Returns:
        Path to the data file.
        
    Raises:
        YemenAnalysisError: If the data type is invalid.
    """
    logger.debug(f"Getting path for {data_type} data file: {file_name}")
    
    try:
        # Get data path from config
        if data_type == 'raw':
            data_path = config.get('data.raw_path', './data/raw')
        elif data_type == 'processed':
            data_path = config.get('data.processed_path', './data/processed')
        elif data_type == 'results':
            data_path = config.get('paths.output_dir', './results')
        else:
            logger.error(f"Invalid data type: {data_type}")
            raise YemenAnalysisError(f"Invalid data type: {data_type}")
        
        # Create path
        file_path = Path(data_path) / file_name
        
        return file_path
    except Exception as e:
        logger.error(f"Error getting data path for {file_name}: {e}")
        raise YemenAnalysisError(f"Error getting data path for {file_name}: {e}")

@handle_errors
def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a configuration file.
    
    Args:
        file_path: Path to the configuration file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        YemenAnalysisError: If the configuration file cannot be loaded.
    """
    logger.info(f"Loading configuration from: {file_path}")
    
    # Convert to Path object
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        logger.error(f"Configuration file does not exist: {file_path}")
        raise YemenAnalysisError(f"Configuration file does not exist: {file_path}")
    
    try:
        # Get file type
        file_type = get_file_extension(file_path)
        
        # Load configuration based on file type
        if file_type == 'json':
            # Load JSON configuration
            with open(file_path, 'r') as f:
                config_data = json.load(f)
        elif file_type in ['yaml', 'yml']:
            # Load YAML configuration
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            logger.error(f"Unsupported configuration file type: {file_type}")
            raise YemenAnalysisError(f"Unsupported configuration file type: {file_type}")
        
        logger.info(f"Loaded configuration from: {file_path}")
        return config_data
    except Exception as e:
        logger.error(f"Error loading configuration from {file_path}: {e}")
        raise YemenAnalysisError(f"Error loading configuration from {file_path}: {e}")

@handle_errors
def save_config_file(
    config_data: Dict[str, Any], file_path: Union[str, Path],
    file_type: Optional[str] = None
) -> None:
    """
    Save a configuration file.
    
    Args:
        config_data: Configuration data to save.
        file_path: Path to the configuration file.
        file_type: Type of configuration file. If None, inferred from the file extension.
        
    Raises:
        YemenAnalysisError: If the configuration file cannot be saved.
    """
    logger.info(f"Saving configuration to: {file_path}")
    
    # Convert to Path object
    file_path = Path(file_path)
    
    # Create directories if needed
    create_directory(file_path.parent)
    
    # Infer file type from extension if not provided
    if file_type is None:
        file_type = get_file_extension(file_path)
    
    try:
        # Save configuration based on file type
        if file_type == 'json':
            # Save JSON configuration
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        elif file_type in ['yaml', 'yml']:
            # Save YAML configuration
            with open(file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        else:
            logger.error(f"Unsupported configuration file type: {file_type}")
            raise YemenAnalysisError(f"Unsupported configuration file type: {file_type}")
        
        logger.info(f"Saved configuration to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {file_path}: {e}")
        raise YemenAnalysisError(f"Error saving configuration to {file_path}: {e}")