import fiona
"""
File operations utilities for the Yemen Market Integration Project.
"""
import os
import json
import yaml
import csv
import pickle
import gzip
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO, Tuple, Set, Iterator
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime

from .error_handler import handle_errors, DataError
from .decorators import timer

logger = logging.getLogger(__name__)

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IsADirectoryError), reraise=True)
def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists
    
    Parameters
    ----------
    directory : str or Path
        Directory path
        
    Returns
    -------
    Path
        Path to the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError), reraise=True)
def delete_file(file_path: Union[str, Path]) -> bool:
    """
    Delete a file
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
        
    Returns
    -------
    bool
        True if successful
    """
    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()
        return True
    return False

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError), reraise=True)
def move_file(source: Union[str, Path], destination: Union[str, Path], 
             overwrite: bool = False) -> bool:
    """
    Move a file
    
    Parameters
    ----------
    source : str or Path
        Source file path
    destination : str or Path
        Destination file path
    overwrite : bool, optional
        Whether to overwrite existing file
        
    Returns
    -------
    bool
        True if successful
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination file already exists: {destination}")
    
    # Ensure destination directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Move the file
    shutil.move(str(source), str(destination))
    return True

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError), reraise=True)
def copy_file(source: Union[str, Path], destination: Union[str, Path], 
             overwrite: bool = False) -> bool:
    """
    Copy a file
    
    Parameters
    ----------
    source : str or Path
        Source file path
    destination : str or Path
        Destination file path
    overwrite : bool, optional
        Whether to overwrite existing file
        
    Returns
    -------
    bool
        True if successful
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination file already exists: {destination}")
    
    # Ensure destination directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the file
    shutil.copy2(str(source), str(destination))
    return True

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, json.JSONDecodeError), reraise=True)
def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read JSON from a file
    
    Parameters
    ----------
    file_path : str or Path
        Path to the JSON file
        
    Returns
    -------
    dict
        JSON data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Write JSON to a file
    
    Parameters
    ----------
    data : dict
        JSON data
    file_path : str or Path
        Path to the output file
    indent : int, optional
        Indentation level
        
    Returns
    -------
    bool
        True if successful
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)
    
    return True

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, yaml.YAMLError), reraise=True)
def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read YAML from a file
    
    Parameters
    ----------
    file_path : str or Path
        Path to the YAML file
        
    Returns
    -------
    dict
        YAML data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, yaml.YAMLError), reraise=True)
def write_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Write YAML to a file
    
    Parameters
    ----------
    data : dict
        YAML data
    file_path : str or Path
        Path to the output file
        
    Returns
    -------
    bool
        True if successful
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return True

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, csv.Error, pd.errors.EmptyDataError), reraise=True)
def read_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read CSV file with enhanced error handling
    
    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file
    **kwargs : dict
        Additional arguments for pandas.read_csv
        
    Returns
    -------
    pandas.DataFrame
        CSV data
    """
    file_path = Path(file_path)
    return pd.read_csv(file_path, **kwargs)

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def write_csv(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
    """
    Write DataFrame to CSV
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to write
    file_path : str or Path
        Path to the output file
    **kwargs : dict
        Additional arguments for pandas.to_csv
        
    Returns
    -------
    bool
        True if successful
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(file_path, **kwargs)
    return True

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def read_geojson(file_path: Union[str, Path], **kwargs) -> gpd.GeoDataFrame:
    """
    Read GeoJSON file
    
    Parameters
    ----------
    file_path : str or Path
        Path to the GeoJSON file
    **kwargs : dict
        Additional arguments for geopandas.read_file
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoJSON data
    """
    file_path = Path(file_path)
    return gpd.read_file(file_path, **kwargs)

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def write_geojson(gdf: gpd.GeoDataFrame, file_path: Union[str, Path], **kwargs) -> bool:
    """
    Write GeoDataFrame to GeoJSON
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Data to write
    file_path : str or Path
        Path to the output file
    **kwargs : dict
        Additional arguments for geopandas.to_file
        
    Returns
    -------
    bool
        True if successful
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set default driver to GeoJSON if not specified
    if 'driver' not in kwargs:
        kwargs['driver'] = 'GeoJSON'
    
    gdf.to_file(file_path, **kwargs)
    return True

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, pickle.PickleError), reraise=True)
def read_pickle(file_path: Union[str, Path]) -> Any:
    """
    Read pickle file
    
    Parameters
    ----------
    file_path : str or Path
        Path to the pickle file
        
    Returns
    -------
    object
        Unpickled object
    """
    file_path = Path(file_path)
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, pickle.PickleError), reraise=True)
def write_pickle(obj: Any, file_path: Union[str, Path]) -> bool:
    """
    Write object to pickle file
    
    Parameters
    ----------
    obj : object
        Object to pickle
    file_path : str or Path
        Path to the output file
        
    Returns
    -------
    bool
        True if successful
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    return True

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
        
    Returns
    -------
    int
        File size in bytes
    """
    file_path = Path(file_path)
    return file_path.stat().st_size

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate file hash
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
    algorithm : str, optional
        Hash algorithm to use
        
    Returns
    -------
    str
        File hash
    """
    file_path = Path(file_path)
    
    # Choose hash algorithm
    if algorithm.lower() == 'md5':
        hash_func = hashlib.md5()
    elif algorithm.lower() == 'sha1':
        hash_func = hashlib.sha1()
    elif algorithm.lower() == 'sha256':
        hash_func = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    # Calculate hash
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def list_files(directory: Union[str, Path], pattern: str = '*', 
              recursive: bool = False) -> List[Path]:
    """
    List files in a directory
    
    Parameters
    ----------
    directory : str or Path
        Directory path
    pattern : str, optional
        Glob pattern for filtering files
    recursive : bool, optional
        Whether to search recursively
        
    Returns
    -------
    list
        List of file paths
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.glob(f'**/{pattern}'))
    else:
        return list(directory.glob(pattern))

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def compress_file(file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Compress a file using gzip
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file to compress
    output_path : str or Path, optional
        Path for the compressed file, defaults to file_path + '.gz'
        
    Returns
    -------
    Path
        Path to the compressed file
    """
    file_path = Path(file_path)
    
    if output_path is None:
        output_path = Path(str(file_path) + '.gz')
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_path

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError, gzip.BadGzipFile), reraise=True)
def decompress_file(file_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Decompress a gzip file
    
    Parameters
    ----------
    file_path : str or Path
        Path to the compressed file
    output_path : str or Path, optional
        Path for the decompressed file, defaults to file_path without '.gz'
        
    Returns
    -------
    Path
        Path to the decompressed file
    """
    file_path = Path(file_path)
    
    if output_path is None:
        # Remove .gz extension if present
        if str(file_path).endswith('.gz'):
            output_path = Path(str(file_path)[:-3])
        else:
            output_path = Path(str(file_path) + '.uncompressed')
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    return output_path

class AtomicFileWriter:
    """Context manager for atomic file writing"""
    
    def __init__(self, file_path: Union[str, Path], mode: str = 'w', **kwargs):
        """
        Initialize atomic file writer
        
        Parameters
        ----------
        file_path : str or Path
            Path to the output file
        mode : str, optional
            File open mode ('w', 'wb')
        **kwargs : dict
            Additional arguments for open()
        """
        self.file_path = Path(file_path)
        self.mode = mode
        self.kwargs = kwargs
        self.temp_file = None
        self.file_obj = None
    
    def __enter__(self):
        # Create directory if needed
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary file
        prefix = self.file_path.name + '.'
        suffix = '.tmp'
        dir_path = self.file_path.parent
        
        self.temp_file = tempfile.NamedTemporaryFile(
            mode=self.mode, prefix=prefix, suffix=suffix, 
            dir=dir_path, delete=False, **self.kwargs
        )
        
        return self.temp_file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the file
        self.temp_file.close()
        
        # If no exception, rename the temporary file to the target file
        if exc_type is None:
            shutil.move(self.temp_file.name, self.file_path)
        else:
            # On error, clean up the temporary file
            try:
                os.unlink(self.temp_file.name)
            except OSError:
                pass

@timer
@handle_errors(logger=logger, error_type=(IOError, ValueError, UnicodeDecodeError), reraise=True)
def read_large_file_chunks(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """
    Read a large file in chunks
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
    chunk_size : int, optional
        Size of each chunk in bytes
        
    Returns
    -------
    iterator
        Iterator of file chunks
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

@timer
@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def read_large_csv_chunks(file_path: Union[str, Path], chunk_size: int = 10000, **kwargs) -> Iterator[pd.DataFrame]:
    """
    Read a large CSV file in chunks
    
    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file
    chunk_size : int, optional
        Number of rows per chunk
    **kwargs : dict
        Additional arguments for pandas.read_csv
        
    Returns
    -------
    iterator
        Iterator of DataFrame chunks
    """
    file_path = Path(file_path)
    
    # Use pandas chunking mechanism
    return pd.read_csv(file_path, chunksize=chunk_size, **kwargs)

@timer
@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError, fiona.errors.DriverError), reraise=True)
def read_large_geojson_chunks(file_path: Union[str, Path], chunk_size: int = 1000) -> Iterator[gpd.GeoDataFrame]:
    """
    Read a large GeoJSON file in chunks
    
    Parameters
    ----------
    file_path : str or Path
        Path to the GeoJSON file
    chunk_size : int, optional
        Number of features per chunk
        
    Returns
    -------
    iterator
        Iterator of GeoDataFrame chunks
    """
    file_path = Path(file_path)
    
    # Load the whole file but parse in chunks
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    features = data.get('features', [])
    
    # Process features in chunks
    for i in range(0, len(features), chunk_size):
        chunk = features[i:i + chunk_size]
        chunk_data = {
            'type': 'FeatureCollection',
            'features': chunk
        }
        if 'crs' in data:
            chunk_data['crs'] = data['crs']
        
        # Convert chunk to GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(chunk_data)
        yield gdf

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def create_backup(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a backup of a file
    
    Parameters
    ----------
    file_path : str or Path
        Path to the file
    backup_dir : str or Path, optional
        Directory for backup, defaults to same directory as file
        
    Returns
    -------
    Path
        Path to the backup file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine backup directory
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_filename
    
    # Copy the file
    shutil.copy2(file_path, backup_path)
    
    return backup_path

@handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, IOError), reraise=True)
def clear_directory(directory: Union[str, Path], pattern: str = '*', 
                   recursive: bool = False) -> int:
    """
    Clear files in a directory
    
    Parameters
    ----------
    directory : str or Path
        Directory path
    pattern : str, optional
        Glob pattern for filtering files
    recursive : bool, optional
        Whether to search recursively
        
    Returns
    -------
    int
        Number of files deleted
    """
    directory = Path(directory)
    
    files = list_files(directory, pattern, recursive)
    
    count = 0
    for file_path in files:
        try:
            file_path.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")
    
    return count