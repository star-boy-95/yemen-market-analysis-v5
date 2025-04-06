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
from typing import Callable, Any, Optional, Dict, Union, List, Tuple, Generator, Iterator
import platform
import warnings
import gc
import numpy as np
import threading
import tempfile
import shutil
import mmap
import weakref
from pathlib import Path

# Initialize module logger
logger = logging.getLogger(__name__)

# Check if running on Apple Silicon M3
IS_M3 = (
    platform.system() == "Darwin" and 
    platform.machine().startswith(("arm", "aarch")) and
    "M3" in os.popen("sysctl -n machdep.cpu.brand_string").read()
)

# Create a directory for memory-mapped files
MMAP_DIR = Path(tempfile.gettempdir()) / "yemen_market_mmap"
os.makedirs(MMAP_DIR, exist_ok=True)


def is_m3_processor() -> bool:
    """
    Check if the system is running on an Apple M3 processor.
    
    Returns
    -------
    bool
        True if running on an M3 processor, False otherwise
    """
    return IS_M3


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
    
    # Configure numpy to reduce memory usage
    try:
        import numpy as np
        np.set_printoptions(precision=8, threshold=1000, edgeitems=5)
        # Enable numpy automatic garbage collection for memory-mapped files
        if hasattr(np, 'set_auto_garbage_collection'):
            np.set_auto_garbage_collection(True)
        logger.info("Configured NumPy for M3 Pro")
    except ImportError:
        logger.warning("NumPy not available")
    
    # Configure pandas
    try:
        import pandas as pd
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        # Reduce memory usage in pandas operations
        pd.set_option('mode.chained_assignment', None)
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
    
    # Configure chunk size for matrix operations
    os.environ["YEMEN_MARKET_MATRIX_CHUNK"] = str(min(5000, int(ram_gb * 500)))
    
    # Enable adaptive chunk size
    os.environ["YEMEN_MARKET_ADAPTIVE_CHUNKS"] = "1"
    
    logger.info(f"System configured for M3 Pro optimization with {ram_gb:.1f}GB RAM")


def m3_optimized(
    func=None, 
    *,
    parallel: bool = False,
    memory_intensive: bool = False,
    io_intensive: bool = False,
    use_numba: bool = False,
    chunk_size: Optional[int] = None,
    mmap_threshold_mb: Optional[int] = 1000
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
    chunk_size : int, optional
        Size of chunks for processing large arrays
        (if None, automatically determined based on memory)
    mmap_threshold_mb : int, optional
        Size threshold in MB for using memory mapping for array operations
        (if None, memory mapping is disabled)
        
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
            
            # Check if inputs are large arrays that could benefit from memory mapping
            large_array_args = []
            if mmap_threshold_mb is not None:
                for i, arg in enumerate(args):
                    if isinstance(arg, np.ndarray) and arg.nbytes > mmap_threshold_mb * 1024 * 1024:
                        # Create memory-mapped copy
                        args = list(args)
                        args[i], mmap_file = create_mmap_array(arg)
                        large_array_args.append((i, mmap_file))
                
                for key, value in list(kwargs.items()):
                    if isinstance(value, np.ndarray) and value.nbytes > mmap_threshold_mb * 1024 * 1024:
                        # Create memory-mapped copy
                        kwargs[key], mmap_file = create_mmap_array(value)
                        large_array_args.append((key, mmap_file))
            
            # Execute function with optimized settings
            try:
                if chunk_size is not None and ('chunk_size' in inspect.signature(f).parameters):
                    # Add chunk_size parameter if function supports it
                    kwargs['chunk_size'] = chunk_size
                
                result = f(*args, **kwargs)
                return result
            finally:
                # Restore original environment
                _restore_env(prev_mkl, prev_omp, prev_blas)
                
                # Clean up memory-mapped files
                for arg_id, mmap_file in large_array_args:
                    try:
                        if hasattr(mmap_file, 'close'):
                            mmap_file.close()
                        if hasattr(mmap_file, 'filename') and os.path.exists(mmap_file.filename):
                            os.unlink(mmap_file.filename)
                    except Exception as e:
                        logger.warning(f"Failed to clean up memory-mapped file: {e}")
                
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
        ttl_seconds: int = 3600,
        prefetch_factor: float = 0.3
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
        prefetch_factor : float, default=0.3
            Fraction of cache entries to prefetch when accessing disk cache
        """
        self.memory_cache = {}
        self.memory_limit_mb = memory_limit_mb
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.prefetch_factor = prefetch_factor
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        self.access_history = []  # Track access patterns for prefetching
        self.access_count = {}  # Count accesses for each key
        
        # Track memory usage of cached items
        self.memory_usage = {}
        
        # Initialize disk cache if directory provided
        self.disk_cache_enabled = disk_cache_dir is not None
        if self.disk_cache_enabled:
            import os
            self.disk_cache_dir = disk_cache_dir
            os.makedirs(disk_cache_dir, exist_ok=True)
            # Create index file for disk cache
            self.index_file = os.path.join(disk_cache_dir, "cache_index.json")
            self._load_disk_index()
    
    def _load_disk_index(self):
        """Load disk cache index if available."""
        import json
        import os
        
        self.disk_index = {}
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    self.disk_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load disk cache index: {e}")
    
    def _save_disk_index(self):
        """Save disk cache index."""
        import json
        
        if not self.disk_cache_enabled:
            return
            
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.disk_index, f)
        except Exception as e:
            logger.warning(f"Failed to save disk cache index: {e}")
    
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
                    # Update access time and pattern
                    entry['last_access'] = time.time()
                    self._update_access_pattern(key)
                    return entry['value']
                else:
                    # Expired, remove from memory cache
                    del self.memory_cache[key]
                    if key in self.memory_usage:
                        del self.memory_usage[key]
            
            # Check disk cache if enabled
            if self.disk_cache_enabled:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    # Cache hit from disk, promote to memory
                    self.hits += 1
                    self._update_access_pattern(key)
                    self.set(key, disk_value)
                    
                    # Prefetch related items
                    self._prefetch_related_items(key)
                    
                    return disk_value
            
            # Cache miss
            self.misses += 1
            return None
    
    def _update_access_pattern(self, key: str):
        """
        Update access pattern tracking for prefetching.
        
        Parameters
        ----------
        key : str
            Cache key being accessed
        """
        # Add to access history with timestamp
        current_time = time.time()
        self.access_history.append((key, current_time))
        
        # Maintain history size
        max_history = min(1000, self.maxsize * 5)
        if len(self.access_history) > max_history:
            self.access_history = self.access_history[-max_history:]
        
        # Update access count
        self.access_count[key] = self.access_count.get(key, 0) + 1
    
    def _prefetch_related_items(self, trigger_key: str):
        """
        Prefetch items related to the currently accessed key.
        
        Parameters
        ----------
        trigger_key : str
            Key that triggered prefetching
        """
        if not self.disk_cache_enabled or self.prefetch_factor <= 0:
            return
            
        # Find keys that are frequently accessed after the trigger key
        related_keys = []
        trigger_indices = [i for i, (k, _) in enumerate(self.access_history) if k == trigger_key]
        
        if not trigger_indices:
            return
            
        # Examine keys accessed after trigger key
        follow_counts = {}
        for idx in trigger_indices:
            if idx + 1 < len(self.access_history):
                next_key = self.access_history[idx + 1][0]
                if next_key != trigger_key and next_key not in self.memory_cache:
                    follow_counts[next_key] = follow_counts.get(next_key, 0) + 1
        
        # Prioritize by frequency
        if follow_counts:
            candidates = sorted(follow_counts.items(), key=lambda x: x[1], reverse=True)
            prefetch_count = max(1, int(self.maxsize * self.prefetch_factor))
            related_keys = [k for k, _ in candidates[:prefetch_count]]
            
            # Prefetch in a background thread to avoid blocking
            if related_keys:
                prefetch_thread = threading.Thread(
                    target=self._background_prefetch,
                    args=(related_keys,),
                    daemon=True
                )
                prefetch_thread.start()
    
    def _background_prefetch(self, keys):
        """
        Prefetch multiple keys in background.
        
        Parameters
        ----------
        keys : list
            Keys to prefetch
        """
        for key in keys:
            try:
                # Check if already in memory cache (may have been added since decision to prefetch)
                with self.lock:
                    if key in self.memory_cache:
                        continue
                        
                    # Get from disk and add to memory cache
                    disk_value = self._get_from_disk(key)
                    if disk_value is not None:
                        # Add to memory cache if there's room
                        if len(self.memory_cache) < self.maxsize:
                            self.set(key, disk_value)
            except Exception as e:
                logger.debug(f"Error prefetching key {key}: {e}")
                continue
    
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
            
            # Estimate memory usage of value
            try:
                mem_usage = self._estimate_size(value)
                self.memory_usage[key] = mem_usage
            except Exception:
                self.memory_usage[key] = 0
            
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
    
    def _estimate_size(self, obj) -> int:
        """
        Estimate memory size of an object.
        
        Parameters
        ----------
        obj : Any
            Object to estimate size of
            
        Returns
        -------
        int
            Estimated size in bytes
        """
        if hasattr(obj, 'nbytes'):
            # Numpy arrays, pandas Series/DataFrame
            return obj.nbytes
        elif isinstance(obj, (list, tuple, dict, set)):
            try:
                import sys
                return sys.getsizeof(obj)
            except Exception:
                pass
        
        # Fall back to a rough estimate
        return 1024  # 1KB default if we can't determine
    
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
            
            # Calculate current cache size
            cache_size_mb = sum(self.memory_usage.values()) / (1024 * 1024)
            
            # More aggressive eviction if memory is under pressure
            if available_gb < 4 or cache_size_mb > self.memory_limit_mb:  # Less than 4GB available or exceeding cache limit
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
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.memory_usage:
                del self.memory_usage[key]
    
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
            
            # Update index
            self.disk_index[key] = {
                'file': f"{key}.cache",
                'expiry': expiry,
                'size': os.path.getsize(cache_file)
            }
            
            # Periodically save index
            if len(self.disk_index) % 10 == 0:
                self._save_disk_index()
                
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
                    # Remove from index
                    if key in self.disk_index:
                        del self.disk_index[key]
                    return None
        except Exception as e:
            logger.debug(f"Failed to load from disk cache: {e}")
            return None
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        stats = {
            'size': len(self.memory_cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_limit_mb': self.memory_limit_mb
        }
        
        # Add disk cache stats if enabled
        if self.disk_cache_enabled:
            try:
                import os
                disk_size = sum(os.path.getsize(os.path.join(self.disk_cache_dir, f))
                               for f in os.listdir(self.disk_cache_dir)
                               if f.endswith('.cache'))
                stats.update({
                    'disk_entries': len(self.disk_index),
                    'disk_size_mb': disk_size / (1024 * 1024)
                })
            except Exception:
                pass
        
        return stats
    
    def clear(self) -> None:
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            self.memory_usage.clear()
            self.access_history.clear()
            self.access_count.clear()
            
            # Clear disk cache if enabled
            if self.disk_cache_enabled:
                try:
                    import os
                    import glob
                    
                    # Delete all cache files
                    cache_files = glob.glob(os.path.join(self.disk_cache_dir, "*.cache"))
                    for file in cache_files:
                        os.remove(file)
                    
                    # Clear and save index
                    self.disk_index.clear()
                    self._save_disk_index()
                except Exception as e:
                    logger.warning(f"Failed to clear disk cache: {e}")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns
        -------
        int
            Number of entries removed
        """
        with self.lock:
            # Check memory cache
            current_time = time.time()
            expired_keys = [k for k, v in self.memory_cache.items()
                           if current_time >= v['expiry']]
            
            # Remove expired entries
            for key in expired_keys:
                del self.memory_cache[key]
                if key in self.memory_usage:
                    del self.memory_usage[key]
            
            # Check disk cache
            disk_expired = 0
            if self.disk_cache_enabled:
                try:
                    import os
                    
                    # Get expired disk entries
                    disk_expired_keys = [k for k, v in self.disk_index.items()
                                        if current_time >= v['expiry']]
                    
                    # Remove expired files
                    for key in disk_expired_keys:
                        cache_file = os.path.join(self.disk_cache_dir, self.disk_index[key]['file'])
                        if os.path.exists(cache_file):
                            os.remove(cache_file)
                        del self.disk_index[key]
                    
                    disk_expired = len(disk_expired_keys)
                    
                    # Save index if changed
                    if disk_expired > 0:
                        self._save_disk_index()
                except Exception as e:
                    logger.warning(f"Error cleaning up disk cache: {e}")
            
            return len(expired_keys) + disk_expired


def tiered_cache(
    maxsize: int = 128,
    typed: bool = False,
    disk_cache_dir: Optional[str] = None,
    memory_limit_mb: int = 1024,
    ttl_seconds: int = 3600,
    prefetch_factor: float = 0.3
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
    prefetch_factor : float, default=0.3
        Fraction of cache entries to prefetch when accessing disk cache
        
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
            ttl_seconds=ttl_seconds,
            prefetch_factor=prefetch_factor
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
        wrapper.cleanup_expired = cache.cleanup_expired
        
        return wrapper
    
    return decorator


def create_mmap_array(array: np.ndarray) -> Tuple[np.ndarray, Any]:
    """
    Create a memory-mapped copy of a numpy array.
    
    Parameters
    ----------
    array : numpy.ndarray
        Input array to memory-map
        
    Returns
    -------
    tuple
        (memory-mapped array, file handle)
    """
    # Create a temporary file for the memory map
    filename = os.path.join(MMAP_DIR, f"mmap_array_{id(array)}_{time.time()}.dat")
    
    # Ensure array is C-contiguous
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    
    # Create memory-mapped array
    mmap_array = np.memmap(filename, dtype=array.dtype, mode='w+', shape=array.shape)
    
    # Copy data
    mmap_array[:] = array[:]
    mmap_array.flush()
    
    # Create file handle with cleanup
    class MmapFile:
        def __init__(self, filename, arr):
            self.filename = filename
            self._arr_ref = weakref.ref(arr)
            
        def close(self):
            # Get array reference
            arr = self._arr_ref()
            if arr is not None:
                arr._mmap.close()
            
            # Remove the file
            if os.path.exists(self.filename):
                try:
                    os.unlink(self.filename)
                except Exception:
                    pass
    
    return mmap_array, MmapFile(filename, mmap_array)


def chunk_iterator(data, chunk_size=None) -> Generator:
    """
    Creates an iterator that yields chunks of a large dataset.
    
    Parameters
    ----------
    data : array-like
        Data to chunk (numpy array, pandas DataFrame, etc.)
    chunk_size : int, optional
        Size of chunks (if None, uses system default)
        
    Yields
    ------
    chunk : same type as data
        Chunk of data
    """
    if chunk_size is None:
        # Get system default chunk size
        chunk_size = int(os.environ.get("YEMEN_MARKET_CHUNK_SIZE", "20000"))
        
        # Adjust based on data size and system memory
        if hasattr(data, 'shape') and len(data.shape) > 0:
            data_size = data.shape[0]
            memory_info = psutil.virtual_memory()
            available_mb = memory_info.available / (1024 * 1024)
            
            # Adaptive chunking based on data size and available memory
            if os.environ.get("YEMEN_MARKET_ADAPTIVE_CHUNKS", "0") == "1":
                if data_size > 1000000:  # Very large dataset
                    chunk_size = min(chunk_size, max(1000, int(available_mb * 5)))
                elif data_size > 100000:  # Large dataset
                    chunk_size = min(chunk_size, max(5000, int(available_mb * 20)))
    
    # Handle different data types
    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            # 1D array
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
        else:
            # Multi-dimensional array
            for i in range(0, data.shape[0], chunk_size):
                yield data[i:i + chunk_size]
    elif hasattr(data, 'iloc'):
        # pandas DataFrame or Series
        for i in range(0, len(data), chunk_size):
            yield data.iloc[i:i + chunk_size]
    elif isinstance(data, (list, tuple)):
        # Basic sequence
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    else:
        # Fallback - just yield the entire data
        yield data


def process_in_chunks(
    data, 
    process_func, 
    chunk_size=None, 
    parallel=False, 
    n_jobs=None,
    reduce_func=None
) -> Any:
    """
    Process large data in chunks, optionally in parallel.
    
    Parameters
    ----------
    data : array-like
        Data to process
    process_func : callable
        Function to apply to each chunk
    chunk_size : int, optional
        Size of chunks
    parallel : bool, default=False
        Whether to process chunks in parallel
    n_jobs : int, optional
        Number of parallel jobs (if None, uses all available cores)
    reduce_func : callable, optional
        Function to combine results from chunks
        
    Returns
    -------
    Any
        Processed data (combined if reduce_func provided)
    """
    # Get chunks
    chunks = list(chunk_iterator(data, chunk_size))
    
    if not chunks:
        return None
    
    # Process chunks
    if parallel and len(chunks) > 1:
        try:
            import multiprocessing as mp
            
            if n_jobs is None:
                # Use all cores for default, slightly reduced for large numbers
                total_cores = mp.cpu_count()
                n_jobs = max(1, total_cores - 1 if total_cores > 4 else total_cores)
            
            # Process chunks in parallel
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.map(process_func, chunks)
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}. Falling back to sequential.")
            # Fall back to sequential processing
            results = [process_func(chunk) for chunk in chunks]
    else:
        # Sequential processing
        results = [process_func(chunk) for chunk in chunks]
    
    # Combine results if reduce function provided
    if reduce_func is not None:
        return reduce_func(results)
    
    return results


@m3_optimized(parallel=True, memory_intensive=True)
def optimize_array_computation(
    arr: np.ndarray,
    chunk_size: Optional[int] = None,
    precision: str = 'float32',
    parallel: bool = True,
    operation: Optional[Callable] = None,
    memory_map: bool = False
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
    operation : callable, optional
        Operation to apply to each chunk
        (if None, just returns the optimized array)
    memory_map : bool, default=False
        Whether to use memory mapping for very large arrays
        
    Returns
    -------
    numpy.ndarray
        Optimized array with the same data
    """
    # Return if input is not a numpy array
    if not isinstance(arr, np.ndarray):
        return arr
    
    # Use memory mapping for very large arrays
    if memory_map and arr.nbytes > 1 * 1024 * 1024 * 1024:  # > 1GB
        logger.info(f"Using memory mapping for large array: {arr.nbytes / (1024**2):.1f} MB")
        mmap_arr, mmap_file = create_mmap_array(arr)
        # Process the memory-mapped array
        result = optimize_array_computation(
            mmap_arr, chunk_size, precision, parallel, operation, memory_map=False
        )
        # Clean up memory map
        if hasattr(mmap_file, 'close'):
            mmap_file.close()
        return result
    
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
    
    # If no operation provided, just return the optimized array
    if operation is None:
        return result
    
    # Process in chunks if array is large
    if chunk_size is None:
        # Determine chunk size based on array size and available memory
        array_size_mb = result.nbytes / (1024**2)
        memory_info = psutil.virtual_memory()
        available_mb = memory_info.available / (1024**2)
        
        # Adaptive chunking
        if array_size_mb > 1000:  # >1GB
            chunk_size = min(result.shape[0] // 10, max(1000, int(available_mb * 0.1)))
        elif array_size_mb > 100:  # >100MB
            chunk_size = min(result.shape[0] // 4, max(10000, int(available_mb * 0.2)))
        else:
            # Small array, no chunking needed
            return operation(result) if operation else result
    
    # If array is small enough, process directly
    if result.shape[0] <= chunk_size:
        return operation(result) if operation else result
    
    # Process in chunks
    def process_chunk(chunk):
        return operation(chunk) if operation else chunk
    
    # Use the process_in_chunks utility
    processed_chunks = process_in_chunks(
        result, process_chunk, chunk_size, parallel
    )
    
    # Combine chunks
    if operation:
        try:
            return np.concatenate(processed_chunks, axis=0)
        except Exception:
            # If concatenation fails, just return the list of chunks
            return processed_chunks
    else:
        return result


def monitor_memory_usage(interval=1.0, log_level=logging.DEBUG):
    """
    Start a background thread that monitors memory usage.
    
    Parameters
    ----------
    interval : float, default=1.0
        Monitoring interval in seconds
    log_level : int, default=logging.DEBUG
        Logging level for memory usage messages
        
    Returns
    -------
    threading.Thread
        Monitoring thread
    """
    def memory_monitor():
        """Background thread for monitoring memory usage."""
        logger.log(log_level, "Starting memory usage monitoring")
        while not stop_flag.is_set():
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                rss_mb = memory_info.rss / (1024 * 1024)
                vms_mb = memory_info.vms / (1024 * 1024)
                system_memory = psutil.virtual_memory()
                available_mb = system_memory.available / (1024 * 1024)
                used_percent = system_memory.percent
                
                logger.log(log_level, 
                          f"Memory usage: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB, "
                          f"System: {available_mb:.1f}MB available ({used_percent:.1f}% used)")
            except Exception as e:
                logger.warning(f"Error in memory monitor: {e}")
            
            # Wait for next check
            time.sleep(interval)
    
    # Create stop flag for graceful shutdown
    stop_flag = threading.Event()
    
    # Create and start monitor thread
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()
    
    # Add stop method
    def stop_monitor():
        stop_flag.set()
        monitor_thread.join(timeout=2.0)
    
    monitor_thread.stop = stop_monitor
    
    return monitor_thread


def memory_profile(func=None, *, detailed=False, log_level=logging.INFO):
    """
    Decorator to profile memory usage of a function.
    
    Parameters
    ----------
    func : callable, optional
        Function to profile
    detailed : bool, default=False
        Whether to show detailed memory usage by line
    log_level : int, default=logging.INFO
        Logging level for memory usage messages
        
    Returns
    -------
    callable
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get initial memory use
            process = psutil.Process(os.getpid())
            gc.collect()  # Force garbage collection
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Create a monitor if detailed profiling requested
            if detailed:
                monitor = monitor_memory_usage(interval=0.5, log_level=log_level)
            
            # Start timing
            start_time = time.time()
            
            try:
                # Call function
                result = f(*args, **kwargs)
                
                # End timing
                end_time = time.time()
                
                # Get final memory use
                gc.collect()  # Force garbage collection
                final_memory = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Calculate and log statistics
                elapsed_time = end_time - start_time
                memory_change = final_memory - initial_memory
                
                logger.log(log_level, 
                          f"Memory profile for {f.__name__}: "
                          f"Initial: {initial_memory:.1f}MB, "
                          f"Final: {final_memory:.1f}MB, "
                          f"Change: {memory_change:+.1f}MB, "
                          f"Time: {elapsed_time:.3f}s")
                
                return result
            finally:
                # Stop monitor if detailed profiling was used
                if detailed and 'monitor' in locals():
                    monitor.stop()
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


# Cleanup function to remove temporary memory-mapped files
def cleanup_mmap_files():
    """Clean up temporary memory-mapped files."""
    try:
        # Remove all files in the mmap directory
        for file in os.listdir(MMAP_DIR):
            try:
                os.unlink(os.path.join(MMAP_DIR, file))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Error cleaning up mmap files: {e}")

# Register cleanup function to run at exit
import atexit
atexit.register(cleanup_mmap_files)
