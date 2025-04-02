"""
Performance optimization utilities for the Yemen Market Integration Project.

Enhanced for M3 Pro hardware with asymmetric core optimizations and memory
management strategies tailored for large-scale market integration analysis.
"""
import os
import sys
import logging
import multiprocessing
import pandas as pd
import numpy as np
from typing import Callable, Any, Optional, List, Dict, Union, Tuple, Iterator
import platform
import psutil
from functools import wraps
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

from .decorators import timer, m1_optimized, disk_cache
from .m3_utils import m3_optimized, tiered_cache, configure_system_for_m3_performance
from .error_handler import handle_errors

logger = logging.getLogger(__name__)

# Check if running on Apple Silicon
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and 
    platform.machine().startswith(("arm", "aarch"))
)

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
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
    
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": cpu_count,
        "total_memory_gb": round(mem.total / (1024**3), 2),
        "available_memory_gb": round(mem.available / (1024**3), 2),
        "is_apple_silicon": IS_APPLE_SILICON
    }
    
    # Try to detect GPU
    try:
        import torch
        info["has_cuda"] = torch.cuda.is_available()
        info["has_mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if info["has_cuda"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["has_cuda"] = False
        info["has_mps"] = False
    
    return info

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
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
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
        
        logger.info(f"Configured for Apple Silicon with {cpu_count} cores")
        
        # Try to enable MPS acceleration if available
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.backends.mps.enable_mps_device = True
                logger.info("MPS acceleration enabled for PyTorch")
        except (ImportError, AttributeError):
            pass
    
    # Configure NumPy to use multiple threads and optimize memory layout
    try:
        import numpy as np
        np.set_printoptions(precision=6, suppress=True)
        
        # Use C order for arrays (better for row operations in Yemen market analysis)
        np.config.add_link_function('order', 'C')
        
        # Use float32 when appropriate to save memory
        np.config.add_link_function('floatX', 'float32')
        
        logger.info("NumPy configured for better performance and memory usage")
    except (ImportError, AttributeError):
        pass
    
    # Configure pandas for parallel operations when supported
    try:
        import pandas as pd
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        logger.info("Pandas configured for optimized computation")
    except ImportError:
        pass
    
    # Force garbage collection to start with clean memory
    gc.collect()

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
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

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def parallelize_array_processing(
    array: np.ndarray,
    func: Callable[[np.ndarray], Any],
    n_workers: Optional[int] = None,
    axis: int = 0,
    combine_func: Optional[Callable] = None
) -> Any:
    """
    Process NumPy array in parallel across specified axis.
    
    Parameters
    ----------
    array : numpy.ndarray
        Input array
    func : callable
        Function to apply to each sub-array
    n_workers : int, optional
        Number of worker processes
    axis : int, default=0
        Axis along which to split the array
    combine_func : callable, optional
        Function to combine results (default: np.vstack or np.hstack)
        
    Returns
    -------
    numpy.ndarray or list
        Combined results
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Split array along specified axis
    array_splits = np.array_split(array, n_workers, axis=axis)
    
    logger.info(f"Processing array in parallel: {len(array_splits)} splits with {n_workers} workers")
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(func, array_splits))
    
    # Combine results
    if combine_func is not None:
        return combine_func(results)
    else:
        try:
            if isinstance(results[0], np.ndarray):
                if axis == 0:
                    return np.vstack(results)
                else:
                    return np.hstack(results)
            return results
        except (ValueError, TypeError):
            return results

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def determine_optimal_chunk_size(
    data_size: int, 
    memory_headroom_mb: int = 1000,
    size_per_item_bytes: int = 1000,
    min_chunk_size: int = 100,
    max_chunk_size: Optional[int] = None
) -> int:
    """
    Determine the optimal chunk size for processing large datasets based on available system memory.
    
    This function calculates an appropriate chunk size that balances memory usage with processing
    efficiency. It accounts for system-specific characteristics like available memory and
    adapts calculation for Apple Silicon systems with shared memory.
    
    Parameters
    ----------
    data_size : int
        Total number of items in the dataset
    memory_headroom_mb : int, default=1000
        Memory safety margin in MB to reserve for other processes
    size_per_item_bytes : int, default=1000
        Estimated memory usage per item in bytes
    min_chunk_size : int, default=100
        Minimum chunk size regardless of memory constraints
    max_chunk_size : int, optional
        Maximum chunk size (defaults to data_size if not specified)
        
    Returns
    -------
    int
        Optimal chunk size for processing
        
    Notes
    -----
    Memory calculation is performed by:
    1. Determining available system memory
    2. Reserving headroom memory
    3. Dividing usable memory by per-item memory requirements
    4. Applying system-specific adjustments
    5. Enforcing minimum and maximum bounds
    
    Example
    -------
    >>> # Example usage
    >>> chunk_size = determine_optimal_chunk_size(
    ...     len(large_dataset), 
    ...     memory_headroom_mb=2000,
    ...     size_per_item_bytes=large_dataset.memory_usage(deep=True).sum() / len(large_dataset)
    ... )
    """
    # Get available system memory
    mem = psutil.virtual_memory()
    available_memory_bytes = mem.available
    
    # Calculate usable memory (available minus headroom)
    usable_memory_bytes = available_memory_bytes - (memory_headroom_mb * 1024 * 1024)
    
    # Account for system-specific memory characteristics
    if IS_APPLE_SILICON:
        # Apple Silicon has unified memory architecture, be more conservative
        # to account for memory shared with GPU
        usable_memory_bytes = usable_memory_bytes * 0.8
    
    # Calculate how many items we can process at once
    max_items = int(usable_memory_bytes / size_per_item_bytes)
    
    # Apply min/max constraints
    if max_chunk_size is None:
        max_chunk_size = data_size
    
    chunk_size = min(max(min_chunk_size, max_items), max_chunk_size, data_size)
    
    # Log the determined chunk size
    available_memory_mb = available_memory_bytes / (1024 * 1024)
    usable_memory_mb = usable_memory_bytes / (1024 * 1024)
    
    logger.info(
        f"Determined optimal chunk size: {chunk_size} items "
        f"(available memory: {available_memory_mb:.1f} MB, "
        f"usable memory: {usable_memory_mb:.1f} MB, "
        f"estimated memory per chunk: {(chunk_size * size_per_item_bytes) / (1024 * 1024):.1f} MB)"
    )
    
    return chunk_size

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def process_in_batches(
    data: Union[pd.DataFrame, np.ndarray],
    batch_func: Callable,
    batch_size: int,
    show_progress: bool = True
) -> List[Any]:
    """
    Process large data in batches to control memory usage.
    
    Parameters
    ----------
    data : DataFrame or ndarray
        Input data
    batch_func : callable
        Function to apply to each batch
    batch_size : int
        Number of items per batch
    show_progress : bool, default=True
        Whether to show progress information
        
    Returns
    -------
    list
        List of batch results
    """
    n_items = len(data)
    n_batches = math.ceil(n_items / batch_size)
    results = []
    
    if show_progress:
        logger.info(f"Processing {n_items} items in {n_batches} batches")
    
    start_time = time.time()
    
    for i in range(0, n_items, batch_size):
        batch = data[i:i+batch_size]
        batch_result = batch_func(batch)
        results.append(batch_result)
        
        if show_progress and (i // batch_size) % max(1, n_batches // 10) == 0:
            progress = (i + len(batch)) / n_items * 100
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + len(batch))) * (n_items - i - len(batch)) if i > 0 else 0
            logger.info(f"Progress: {progress:.1f}% - Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s")
    
    return results

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def optimize_dataframe(
    df: pd.DataFrame,
    downcast: bool = True,
    category_min_size: int = 50,
    convert_datetimes: bool = True,
    deep_copy: bool = True
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
    convert_datetimes : bool, optional
        Convert string dates to datetime
    deep_copy : bool, optional
        Whether to create a deep copy of the DataFrame
        
    Returns
    -------
    pandas.DataFrame
        Memory-optimized DataFrame
    """
    result = df.copy() if deep_copy else df
    start_mem = result.memory_usage(deep=True).sum() / (1024 ** 2)
    
    # Process each column
    for column in result.columns:
        column_type = result[column].dtype
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(column_type) and downcast:
            # Integers
            if pd.api.types.is_integer_dtype(column_type):
                # Check if column has nulls, if so convert to float
                if result[column].isna().sum() > 0:
                    result[column] = pd.to_numeric(result[column], downcast='float')
                else:
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
            
            # Try to convert date strings to datetime
            elif convert_datetimes:
                try:
                    # Check if column contains dates
                    if result[column].str.match(r'^\d{4}-\d{2}-\d{2}').any():
                        result[column] = pd.to_datetime(result[column], errors='ignore')
                except (AttributeError, TypeError):
                    pass
    
    # Calculate memory savings
    end_mem = result.memory_usage(deep=True).sum() / (1024 ** 2)
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.2f}%)")
    
    return result

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def optimize_numpy_array(
    arr: np.ndarray, 
    convert_to_float32: bool = True
) -> np.ndarray:
    """
    Optimize NumPy array memory usage.
    
    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    convert_to_float32 : bool, default=True
        Whether to convert float64 to float32
        
    Returns
    -------
    numpy.ndarray
        Memory-optimized array
    """
    start_mem = arr.nbytes / (1024 ** 2)
    
    # For float64 arrays, convert to float32 to save memory
    if convert_to_float32 and arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    
    # For int64 arrays, downcast if possible
    elif arr.dtype == np.int64:
        # Check min/max values to determine smallest possible type
        min_val = arr.min()
        max_val = arr.max()
        
        if min_val >= 0:
            if max_val < 256:
                arr = arr.astype(np.uint8)
            elif max_val < 65536:
                arr = arr.astype(np.uint16)
            elif max_val < 4294967296:
                arr = arr.astype(np.uint32)
        else:
            if min_val > -128 and max_val < 128:
                arr = arr.astype(np.int8)
            elif min_val > -32768 and max_val < 32768:
                arr = arr.astype(np.int16)
            elif min_val > -2147483648 and max_val < 2147483648:
                arr = arr.astype(np.int32)
    
    end_mem = arr.nbytes / (1024 ** 2)
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    logger.info(f"Array memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.2f}%)")
    
    return arr

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

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def track_memory_usage(name: str = "Memory usage"):
    """
    Function to explicitly track current memory usage.
    
    Parameters
    ----------
    name : str, optional
        Label for the log entry
        
    Returns
    -------
    float
        Current memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 ** 2)
    logger.info(f"{name}: {memory_mb:.2f} MB")
    return memory_mb

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def read_large_array_chunks(
    file_path: str, 
    chunk_size: int = 1000, 
    dtype: Any = None
) -> Iterator[np.ndarray]:
    """
    Read large NumPy arrays in chunks.
    
    Parameters
    ----------
    file_path : str
        Path to .npy or .npz file
    chunk_size : int, optional
        Number of rows per chunk
    dtype : numpy dtype, optional
        Data type for the array
        
    Yields
    ------
    numpy.ndarray
        Array chunks
    """
    # Memory-map the file
    if file_path.endswith('.npy'):
        arr = np.load(file_path, mmap_mode='r')
        shape = arr.shape
        
        # Yield chunks
        for i in range(0, shape[0], chunk_size):
            yield arr[i:i+chunk_size].copy()
    
    elif file_path.endswith('.npz'):
        with np.load(file_path) as data:
            # Process each array in the npz file
            for name in data.files:
                arr = data[name]
                shape = arr.shape
                
                for i in range(0, shape[0], chunk_size):
                    yield arr[i:i+chunk_size].copy()
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def force_gc():
    """
    Force garbage collection and report memory usage.
    
    Returns
    -------
    float
        Memory usage after collection in MB
    """
    before = track_memory_usage("Memory before GC")
    gc.collect()
    after = track_memory_usage("Memory after GC")
    
    logger.info(f"Garbage collection freed {before - after:.2f} MB")
    return after
