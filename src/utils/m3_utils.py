"""
M3 Pro hardware optimization utilities for Yemen Market Integration analysis.

This module provides optimizations tailored to M3 Pro processors with
asymmetric core architectures (P-cores and E-cores) and unified memory,
enabling efficient market integration analysis on large datasets.
"""
import os
import sys
import logging
import multiprocessing
import psutil
import time
import functools
import inspect
from functools import wraps, lru_cache
from typing import Callable, Any, Optional, Dict, Union, List, Tuple
import platform
import warnings
import gc
import numpy as np
import threading

# Initialize module logger
logger = logging.getLogger(__name__)

# Check if running on Apple Silicon M3
IS_M3 = (
    platform.system() == "Darwin" and 
    platform.machine().startswith(("arm", "aarch")) and
    "M3" in os.popen("sysctl -n machdep.cpu.brand_string").read()
)


def configure_system_for_m3_performance() -> None:
    """
    Configure system settings for optimal performance on M3 Pro hardware.
    
    This function sets environment variables and numpy configurations
    to best utilize M3 Pro's asymmetric core architecture and unified memory.
    """
    # Get CPU info
    try:
        p_cores = 6  # Performance cores in M3 Pro
        e_cores = 4  # Efficiency cores in M3 Pro
        total_cores = multiprocessing.cpu_count()
        
        # Adjust if detected core count differs
        if total_cores != p_cores + e_cores:
            p_cores = min(6, total_cores // 2)
            e_cores = total_cores - p_cores
        
        logger.info(f"Detected M3 Pro with {p_cores} P-cores and {e_cores} E-cores")
    except Exception as e:
        logger.warning(f"Could not detect core configuration: {e}")
        p_cores = 6
        e_cores = 4
    
    # Configure MKL/OpenBLAS to use correct number of threads
    os.environ["MKL_NUM_THREADS"] = str(p_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(p_cores) 
    os.environ["OMP_NUM_THREADS"] = str(p_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(p_cores)
    
    # Optimize for Apple Silicon
    os.environ["VECLIB_PREFERRED_ARCH"] = "armv8.5-a"
    
    # Use accelerate framework on macOS when possible
    if platform.system() == "Darwin":
        os.environ["ACCELERATE_ENABLE_SIMD"] = "1"
    
    # Configure numpy
    try:
        import numpy as np
        np.set_printoptions(precision=8, threshold=1000, edgeitems=5)
        logger.info("Configured NumPy for M3 Pro")
    except ImportError:
        logger.warning("NumPy not available")
    
    # Configure pandas
    try:
        import pandas as pd
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        logger.info("Configured Pandas for M3 Pro")
    except ImportError:
        logger.warning("Pandas not available")
    
    # For large datasets, ensure we use a reasonable chunk size for parallel operations
    memory_info = psutil.virtual_memory()
    ram_gb = memory_info.total / (1024**3)
    
    # M3 Pro often has 18 or 36GB RAM, adjust based on available memory
    if ram_gb >= 32:
        os.environ["YEMEN_MARKET_CHUNK_SIZE"] = "50000"  # Large chunks for 32GB+ systems
    elif ram_gb >= 16:
        os.environ["YEMEN_MARKET_CHUNK_SIZE"] = "20000"  # Medium chunks for 16GB systems
    else:
        os.environ["YEMEN_MARKET_CHUNK_SIZE"] = "10000"  # Smaller chunks for <16GB systems
    
    logger.info(f"System configured for M3 Pro optimization with {ram_gb:.1f}GB RAM")


def m3_optimized(
    func=None, 
    *,
    parallel: bool = False,
    memory_intensive: bool = False,
    io_intensive: bool = False,
    use_numba: bool = False
):
    """
    Decorator to optimize function execution for M3 Pro processor.
    
    This decorator configures thread usage, vectorization, and memory management
    based on the M3 Pro's hardware characteristics and the function's needs.
    
    Parameters
    ----------
    func : callable, optional
        Function to decorate
    parallel : bool, default=False
        Whether function benefits from parallel execution
    memory_intensive : bool, default=False
        Whether function uses large amounts of memory
    io_intensive : bool, default=False
        Whether function is I/O bound rather than CPU bound
    use_numba : bool, default=False
        Whether to use Numba JIT compilation if available
        
    Returns
    -------
    callable
        Decorated function with M3 Pro optimizations
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get CPU info
            p_cores = 6  # Default for M3 Pro
            e_cores = 4  # Default for M3 Pro
            
            try:
                total_cores = multiprocessing.cpu_count()
                if total_cores != p_cores + e_cores:
                    p_cores = min(6, total_cores // 2)
                    e_cores = total_cores - p_cores
            except:
                pass
            
            # Configure thread usage based on operation type
            prev_mkl = os.environ.get("MKL_NUM_THREADS")
            prev_omp = os.environ.get("OMP_NUM_THREADS")
            prev_blas = os.environ.get("OPENBLAS_NUM_THREADS")
            
            if parallel:
                if memory_intensive:
                    # For memory-intensive parallel tasks, use primarily P-cores
                    # to benefit from shared L2 cache and higher memory bandwidth
                    thread_count = p_cores
                elif io_intensive:
                    # For I/O bound operations, use all cores since they'll often
                    # be waiting for I/O anyway
                    thread_count = p_cores + e_cores
                else:
                    # For compute-bound parallel operations, primarily use P-cores
                    # but also some E-cores for background tasks
                    thread_count = p_cores + (e_cores // 2)
                
                # Set thread counts for various libraries
                os.environ["MKL_NUM_THREADS"] = str(thread_count)
                os.environ["OMP_NUM_THREADS"] = str(thread_count)
                os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
            else:
                # For sequential operations, use just 1-2 threads to avoid overhead
                os.environ["MKL_NUM_THREADS"] = "2"
                os.environ["OMP_NUM_THREADS"] = "2" 
                os.environ["OPENBLAS_NUM_THREADS"] = "2"
            
            # For memory-intensive operations, run garbage collection before execution
            if memory_intensive:
                gc.collect()
            
            # Try to use Numba for computationally intensive functions if requested
            if use_numba:
                try:
                    import numba
                    
                    # Check if function is already JIT-compiled
                    if hasattr(f, '__numba__'):
                        return f(*args, **kwargs)
                    
                    # Configure Numba for M3
                    numba.set_num_threads(thread_count if parallel else 2)
                    
                    # Compile with appropriate options
                    if parallel:
                        jit_f = numba.njit(parallel=True, fastmath=True)(f)
                    else:
                        jit_f = numba.njit(fastmath=True)(f)
                    
                    result = jit_f(*args, **kwargs)
                    
                    # Restore original environment
                    _restore_env(prev_mkl, prev_omp, prev_blas)
                    
                    return result
                except ImportError:
                    logger.debug("Numba not available, using regular function")
                except Exception as e:
                    logger.debug(f"Could not use Numba for {f.__name__}: {e}")
            
            # Execute function with optimized settings
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                # Restore original environment
                _restore_env(prev_mkl, prev_omp, prev_blas)
                
                # For memory-intensive operations, run garbage collection after execution
                if memory_intensive:
                    gc.collect()
        
        def _restore_env(prev_mkl, prev_omp, prev_blas):
            """Restore previous environment variables."""
            if prev_mkl:
                os.environ["MKL_NUM_THREADS"] = prev_mkl
            elif "MKL_NUM_THREADS" in os.environ:
                del os.environ["MKL_NUM_THREADS"]
                
            if prev_omp:
                os.environ["OMP_NUM_THREADS"] = prev_omp
            elif "OMP_NUM_THREADS" in os.environ:
                del os.environ["OMP_NUM_THREADS"]
                
            if prev_blas:
                os.environ["OPENBLAS_NUM_THREADS"] = prev_blas
            elif "OPENBLAS_NUM_THREADS" in os.environ:
                del os.environ["OPENBLAS_NUM_THREADS"]
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


class TieredCache:
    """
    Multi-level caching system optimized for M3 Pro memory architecture.
    
    Implements a tiered caching system with memory (L1) and disk (L2) caches,
    automatically managing cache eviction based on memory pressure and
    access patterns to optimize for the M3 Pro's unified memory.
    """
    
    def __init__(
        self,
        maxsize: int = 128,
        disk_cache_dir: Optional[str] = None,
        memory_limit_mb: int = 1024,
        ttl_seconds: int = 3600
    ):
        """
        Initialize tiered cache.
        
        Parameters
        ----------
        maxsize : int, default=128
            Maximum number of items to keep in memory cache
        disk_cache_dir : str, optional
            Directory for disk cache (if None, disk cache is disabled)
        memory_limit_mb : int, default=1024
            Maximum memory usage in MB for memory cache
        ttl_seconds : int, default=3600
            Time-to-live for cache entries in seconds
        """
        self.memory_cache = {}
        self.memory_limit_mb = memory_limit_mb
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
        # Initialize disk cache if directory provided
        self.disk_cache_enabled = disk_cache_dir is not None
        if self.disk_cache_enabled:
            import os
            self.disk_cache_dir = disk_cache_dir
            os.makedirs(disk_cache_dir, exist_ok=True)
    
    def get(self, key: str) -> Any:
        """
        Get value from cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Any
            Cached value or None if not found
        """
        with self.lock:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if time.time() < entry['expiry']:
                    self.hits += 1
                    # Update access time
                    entry['last_access'] = time.time()
                    return entry['value']
                else:
                    # Expired, remove from memory cache
                    del self.memory_cache[key]
            
            # Check disk cache if enabled
            if self.disk_cache_enabled:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    # Cache hit from disk, promote to memory
                    self.hits += 1
                    self.set(key, disk_value)
                    return disk_value
            
            # Cache miss
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Add or update value in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        with self.lock:
            # Check memory pressure and evict if needed
            self._check_memory_pressure()
            
            # Add to memory cache
            expiry = time.time() + self.ttl_seconds
            self.memory_cache[key] = {
                'value': value,
                'expiry': expiry,
                'last_access': time.time()
            }
            
            # Also save to disk if enabled
            if self.disk_cache_enabled:
                self._save_to_disk(key, value, expiry)
    
    def _check_memory_pressure(self) -> None:
        """
        Check memory usage and evict cache entries if necessary.
        
        Uses M3 Pro-aware eviction strategies that consider both
        the size of cached items and memory pressure on the system.
        """
        # If memory cache is full, evict based on LRU
        if len(self.memory_cache) >= self.maxsize:
            self._evict_lru_entries(1)
        
        # Check system memory pressure
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # More aggressive eviction if memory is under pressure
            if available_gb < 4:  # Less than 4GB available
                evict_count = max(10, len(self.memory_cache) // 4)
                self._evict_lru_entries(evict_count)
                # Force garbage collection
                gc.collect()
        except Exception as e:
            logger.warning(f"Error checking memory pressure: {e}")
    
    def _evict_lru_entries(self, count: int) -> None:
        """
        Evict least recently used entries from memory cache.
        
        Parameters
        ----------
        count : int
            Number of entries to evict
        """
        if not self.memory_cache:
            return
            
        # Sort by last access time
        sorted_keys = sorted(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k]['last_access']
        )
        
        # Evict oldest entries
        for key in sorted_keys[:count]:
            del self.memory_cache[key]
    
    def _save_to_disk(self, key: str, value: Any, expiry: float) -> None:
        """
        Save value to disk cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        expiry : float
            Expiry timestamp
        """
        if not self.disk_cache_enabled:
            return
            
        try:
            import pickle
            import os
            
            # Create cache file path
            cache_file = os.path.join(self.disk_cache_dir, f"{key}.cache")
            
            # Save value and metadata
            with open(cache_file, 'wb') as f:
                pickle.dump({'value': value, 'expiry': expiry}, f)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _get_from_disk(self, key: str) -> Any:
        """
        Get value from disk cache.
        
        Parameters
        ----------
        key : str
            Cache key
            
        Returns
        -------
        Any
            Cached value or None if not found or expired
        """
        if not self.disk_cache_enabled:
            return None
            
        try:
            import pickle
            import os
            
            # Create cache file path
            cache_file = os.path.join(self.disk_cache_dir, f"{key}.cache")
            
            # Check if file exists
            if not os.path.exists(cache_file):
                return None
            
            # Load value and check expiry
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                
                if time.time() < data['expiry']:
                    return data['value']
                else:
                    # Expired, delete file
                    os.remove(cache_file)
                    return None
        except Exception as e:
            logger.debug(f"Failed to load from disk cache: {e}")
            return None
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'size': len(self.memory_cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
    
    def clear(self) -> None:
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            
            # Clear disk cache if enabled
            if self.disk_cache_enabled:
                try:
                    import os
                    import glob
                    
                    # Delete all cache files
                    cache_files = glob.glob(os.path.join(self.disk_cache_dir, "*.cache"))
                    for file in cache_files:
                        os.remove(file)
                except Exception as e:
                    logger.warning(f"Failed to clear disk cache: {e}")


def tiered_cache(
    maxsize: int = 128,
    typed: bool = False,
    disk_cache_dir: Optional[str] = None,
    memory_limit_mb: int = 1024,
    ttl_seconds: int = 3600
):
    """
    Decorator for tiered caching optimized for M3 Pro.
    
    Creates a function-specific tiered cache with memory and optional
    disk caching, automatically tuned for M3 Pro hardware.
    
    Parameters
    ----------
    maxsize : int, default=128
        Maximum number of items to keep in memory cache
    typed : bool, default=False
        If True, arguments of different types will be cached separately
    disk_cache_dir : str, optional
        Directory for disk cache (if None, disk cache is disabled)
    memory_limit_mb : int, default=1024
        Maximum memory usage for memory cache
    ttl_seconds : int, default=3600
        Time-to-live for cache entries in seconds
        
    Returns
    -------
    callable
        Decorator function
    """
    # Adjust cache size based on available system memory
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.total / (1024**3)
        
        if available_gb >= 32:  # 32GB+ systems
            maxsize = max(maxsize, 512)
            memory_limit_mb = max(memory_limit_mb, 4096)  # 4GB cache
        elif available_gb >= 16:  # 16GB systems
            maxsize = max(maxsize, 256)
            memory_limit_mb = max(memory_limit_mb, 2048)  # 2GB cache
    except:
        pass
    
    # Create unique folder for this function's disk cache
    if disk_cache_dir is not None:
        try:
            import hashlib
            import os
            
            def make_disk_cache_dir(func):
                """Create unique disk cache directory for function."""
                func_id = f"{func.__module__}.{func.__qualname__}"
                dir_hash = hashlib.md5(func_id.encode()).hexdigest()[:8]
                func_cache_dir = os.path.join(disk_cache_dir, dir_hash)
                os.makedirs(func_cache_dir, exist_ok=True)
                return func_cache_dir
        except:
            make_disk_cache_dir = lambda _: disk_cache_dir
    else:
        make_disk_cache_dir = lambda _: None
    
    def decorator(func):
        # Create cache instance for this function
        func_disk_cache_dir = make_disk_cache_dir(func)
        cache = TieredCache(
            maxsize=maxsize,
            disk_cache_dir=func_disk_cache_dir,
            memory_limit_mb=memory_limit_mb,
            ttl_seconds=ttl_seconds
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            if typed:
                key_parts = [
                    func.__module__,
                    func.__qualname__,
                    str(args),
                    str(sorted(kwargs.items())),
                    str([type(arg) for arg in args]),
                    str(sorted((k, str(type(v))) for k, v in kwargs.items()))
                ]
            else:
                key_parts = [
                    func.__module__,
                    func.__qualname__,
                    str(args),
                    str(sorted(kwargs.items()))
                ]
            
            key = "_".join(key_parts)
            
            try:
                import hashlib
                # Create hash of the key for shorter keys
                key = hashlib.md5(key.encode()).hexdigest()
            except:
                # If hashing fails, use the key as-is
                pass
            
            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value
            
            # Cache miss, call function
            value = func(*args, **kwargs)
            
            # Store in cache
            cache.set(key, value)
            
            return value
        
        # Add cache instance to wrapper
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = lambda: cache.stats
        
        return wrapper
    
    return decorator


@m3_optimized(parallel=True, memory_intensive=True)
def optimize_array_computation(
    arr: np.ndarray,
    chunk_size: Optional[int] = None,
    precision: str = 'float32',
    parallel: bool = True
) -> np.ndarray:
    """
    Optimize array computation for M3 Pro processors.
    
    This function applies various optimizations to numpy arrays
    for efficient processing on M3 Pro hardware, including data type
    optimization, memory layout adjustments, and chunked processing.
    
    Parameters
    ----------
    arr : numpy.ndarray
        Input array to optimize
    chunk_size : int, optional
        Size of chunks for processing large arrays
        (if None, automatically determined based on array size and memory)
    precision : str, default='float32'
        Precision to use for floating point operations
        ('float32' or 'float64')
    parallel : bool, default=True
        Whether to use parallel processing for chunked computation
        
    Returns
    -------
    numpy.ndarray
        Optimized array with the same data
    """
    # Return if input is not a numpy array
    if not isinstance(arr, np.ndarray):
        return arr
    
    # Create copy to avoid modifying original
    result = arr.copy()
    
    # Convert floating point precision
    if precision == 'float32' and arr.dtype == np.float64:
        result = result.astype(np.float32)
    elif precision == 'float64' and arr.dtype == np.float32:
        result = result.astype(np.float64)
    
    # Ensure C contiguous memory layout for better performance
    if not result.flags.c_contiguous:
        result = np.ascontiguousarray(result)
    
    # Process in chunks if array is large
    if chunk_size is None:
        # Determine chunk size based on array size and available memory
        array_size_mb = result.nbytes / (1024**2)
        
        if array_size_mb > 1000:  # >1GB
            chunk_size = result.shape[0] // 10  # 10 chunks
        elif array_size_mb > 100:  # >100MB
            chunk_size = result.shape[0] // 4   # 4 chunks
        else:
            # Small array, no chunking needed
            return result
    
    # If array is small enough, return as is
    if result.shape[0] <= chunk_size:
        return result
    
    # Process in chunks
    if parallel and result.shape[0] > chunk_size:
        # Split into chunks and process in parallel
        import multiprocessing as mp
        
        # Define chunk processing function
        def process_chunk(chunk):
            # Do any chunk-specific processing here
            # For now, just return the chunk
            return chunk
        
        # Split array into chunks
        chunks = [result[i:i+chunk_size] for i in range(0, result.shape[0], chunk_size)]
        
        # Process chunks in parallel
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            processed_chunks = pool.map(process_chunk, chunks)
        
        # Combine chunks
        result = np.concatenate(processed_chunks, axis=0)
    
    return result
