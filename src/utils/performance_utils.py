"""
Performance optimization utilities for the Yemen Market Integration Project.
"""
import os
import sys
import logging
import multiprocessing
import pandas as pd
import numpy as np
from typing import Callable, Any, Optional, List, Dict, Union
import platform
import psutil
from functools import wraps
import time

from src.utils.decorators import timer
from src.utils.error_handler import handle_errors

logger = logging.getLogger(__name__)

# Check if running on Apple Silicon
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and 
    platform.machine().startswith(("arm", "aarch"))
)

@handle_errors(logger=logger)
def get_system_info() -> Dict[str, Any]:
    """
    Get system information including hardware and memory.
    
    Returns
    -------
    dict
        System information
    """
    cpu_count = multiprocessing.cpu_count()
    mem = psutil.virtual_memory()
    
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": cpu_count,
        "total_memory_gb": round(mem.total / (1024**3), 2),
        "available_memory_gb": round(mem.available / (1024**3), 2),
        "is_apple_silicon": IS_APPLE_SILICON
    }

@handle_errors(logger=logger)
def configure_system_for_performance() -> None:
    """
    Configure the system for optimal performance based on hardware.
    """
    sys_info = get_system_info()
    logger.info(f"Configuring system for performance: {sys_info}")
    
    # Set number of threads for NumPy operations based on CPU count
    cpu_count = sys_info["cpu_count"]
    
    # Apple Silicon specific optimizations
    if sys_info["is_apple_silicon"]:
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
        logger.info(f"Configured for Apple Silicon with {cpu_count} cores")
        
        # Try to enable MPS acceleration if available
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.backends.mps.enable_mps_device = True
                logger.info("MPS acceleration enabled for PyTorch")
        except (ImportError, AttributeError):
            pass
    
    # Configure NumPy to use multiple threads
    try:
        import numpy as np
        np.set_printoptions(precision=6, suppress=True)
        logger.info("NumPy configured for better output formatting")
    except ImportError:
        pass

@handle_errors(logger=logger)
@timer
def parallelize_dataframe(
    df: pd.DataFrame, 
    func: Callable[[pd.DataFrame], pd.DataFrame], 
    n_workers: Optional[int] = None,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply a function to a DataFrame in parallel using multiprocessing.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    func : callable
        Function to apply to each chunk
    n_workers : int, optional
        Number of worker processes
    chunk_size : int, optional
        Size of each chunk
        
    Returns
    -------
    pandas.DataFrame
        Processed DataFrame
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Calculate chunk size if not provided
    if chunk_size is None:
        chunk_size = max(1, len(df) // n_workers)
    
    # Split DataFrame into chunks
    df_split = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    logger.info(f"Processing DataFrame in parallel: {len(df_split)} chunks with {n_workers} workers")
    
    # Process chunks in parallel
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(func, df_split)
    
    # Combine results
    return pd.concat(results)

@handle_errors(logger=logger)
def optimize_dataframe(
    df: pd.DataFrame,
    downcast: bool = True,
    category_min_size: int = 50
) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by adjusting dtypes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    downcast : bool, optional
        Whether to downcast numeric columns
    category_min_size : int, optional
        Convert string columns with fewer unique values to category
        
    Returns
    -------
    pandas.DataFrame
        Memory-optimized DataFrame
    """
    result = df.copy()
    start_mem = result.memory_usage(deep=True).sum() / (1024 ** 2)
    
    # Process each column
    for column in result.columns:
        column_type = result[column].dtype
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(column_type) and downcast:
            # Integers
            if pd.api.types.is_integer_dtype(column_type):
                result[column] = pd.to_numeric(result[column], downcast='integer')
            # Floats
            elif pd.api.types.is_float_dtype(column_type):
                result[column] = pd.to_numeric(result[column], downcast='float')
        
        # String columns
        elif pd.api.types.is_object_dtype(column_type):
            n_unique = result[column].nunique()
            n_total = len(result[column])
            
            # Convert to category if low cardinality
            if n_unique / n_total < 0.5 and n_unique < category_min_size:
                result[column] = result[column].astype('category')
    
    # Calculate memory savings
    end_mem = result.memory_usage(deep=True).sum() / (1024 ** 2)
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.2f}%)")
    
    return result

def memory_usage_decorator(func):
    """
    Decorator to track memory usage of a function.
    
    Parameters
    ----------
    func : callable
        Function to track
        
    Returns
    -------
    callable
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 2)
        
        # Run the function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Get final memory usage
        mem_after = process.memory_info().rss / (1024 ** 2)
        
        logger.info(
            f"Function {func.__name__} used {mem_after - mem_before:.2f} MB "
            f"(total: {mem_after:.2f} MB) and took {execution_time:.2f} seconds"
        )
        
        return result
    
    return wrapper