"""
Utility decorators for the Yemen Market Integration Project.
"""
import time
import logging
import functools
import hashlib
import inspect
import warnings
import os
from datetime import datetime
from typing import Callable, TypeVar, Any, Dict, Optional, Union, List, Tuple
from functools import lru_cache

T = TypeVar('T')
logger = logging.getLogger(__name__)

def timer(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to time function execution
    
    Parameters
    ----------
    func : callable
        Function to time
        
    Returns
    -------
    callable
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        # Get function signature for better logging
        func_name = func.__qualname__
        logger.debug(f"Function {func_name} executed in {duration:.4f} seconds")
        
        return result
    return wrapper

def m1_optimized(use_numba: bool = True, parallel: bool = True) -> Callable:
    """
    Decorator to optimize functions for M1 Mac
    
    Parameters
    ----------
    use_numba : bool, optional
        Whether to use numba JIT if available
    parallel : bool, optional
        Whether to enable parallel execution
        
    Returns
    -------
    callable
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Try to import numba if requested
        if use_numba:
            try:
                import numba
                # Check if we're on Apple Silicon
                is_m1 = 'arm' in os.uname().machine.lower()
                
                # Apply appropriate JIT decorator
                if is_m1:
                    if parallel:
                        return numba.njit(parallel=True)(func)
                    else:
                        return numba.njit()(func)
                
            except ImportError:
                # If numba is not available, fall back to regular function
                warnings.warn("Numba not available, M1 optimization not applied")
                pass
        
        # If numba is not requested or not available, return the original function
        return func
    
    return decorator

def disk_cache(
    cache_dir: str = '.cache',
    expiration_seconds: Optional[int] = None
) -> Callable:
    """
    Decorator to cache function results to disk
    
    Parameters
    ----------
    cache_dir : str, optional
        Directory to store cache files
    expiration_seconds : int, optional
        Cache expiration time in seconds
        
    Returns
    -------
    callable
        Decorator function
    """
    import pickle
    import os
    from pathlib import Path
    
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(exist_ok=True, parents=True)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hash of the function name and arguments
            func_name = func.__qualname__
            arg_hash = hashlib.md5(str(args).encode() + str(sorted(kwargs.items())).encode()).hexdigest()
            cache_file = cache_dir_path / f"{func_name}_{arg_hash}.pkl"
            
            # Check if cache file exists and is not expired
            if cache_file.exists():
                file_age = time.time() - os.path.getmtime(cache_file)
                if expiration_seconds is None or file_age < expiration_seconds:
                    try:
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
                    except (pickle.PickleError, EOFError):
                        # Handle corrupted cache file
                        cache_file.unlink(missing_ok=True)
            
            # Execute function if no cache or expired
            result = func(*args, **kwargs)
            
            # Save result to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except (pickle.PickleError, IOError) as e:
                logger.warning(f"Failed to cache result for {func_name}: {e}")
            
            return result
        return wrapper
    return decorator

def memoize(func: Callable[..., T]) -> Callable[..., T]:
    """
    Simple in-memory memoization decorator
    
    Parameters
    ----------
    func : callable
        Function to memoize
        
    Returns
    -------
    callable
        Memoized function
    """
    cache: Dict[str, Any] = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from the function arguments
        key_parts = [repr(arg) for arg in args]
        key_parts.extend(f"{k}={repr(v)}" for k, v in sorted(kwargs.items()))
        key = "||".join(key_parts)
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    # Add clear_cache method to wrapper
    wrapper.clear_cache = lambda: cache.clear()
    
    return wrapper

def rate_limited(max_per_second: float) -> Callable:
    """
    Decorator to rate limit function calls
    
    Parameters
    ----------
    max_per_second : float
        Maximum calls per second
        
    Returns
    -------
    callable
        Decorator function
    """
    min_interval = 1.0 / max_per_second
    last_called = [0.0]
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            sleep_time = min_interval - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def log_calls(level: int = logging.DEBUG) -> Callable:
    """
    Decorator to log function calls
    
    Parameters
    ----------
    level : int, optional
        Logging level
        
    Returns
    -------
    callable
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            args_str = ", ".join([repr(a) for a in args])
            kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))
            
            call_signature = f"{func_name}({all_args})"
            logger.log(level, f"Calling: {call_signature}")
            
            result = func(*args, **kwargs)
            
            logger.log(level, f"Returned from: {func_name}")
            return result
        
        return wrapper
    
    return decorator

def retry(
    max_attempts: int = 3, 
    delay: float = 1.0, 
    backoff_factor: float = 2.0,
    exceptions: Union[type, Tuple[type, ...]] = Exception
) -> Callable:
    """
    Decorator to retry a function on exception
    
    Parameters
    ----------
    max_attempts : int, optional
        Maximum number of attempts
    delay : float, optional
        Initial delay between attempts in seconds
    backoff_factor : float, optional
        Factor to increase delay by after each attempt
    exceptions : Exception type or tuple of types, optional
        Types of exceptions to catch and retry
        
    Returns
    -------
    callable
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} for {func.__qualname__} "
                        f"failed with error: {str(e)}. Retrying in {current_delay:.2f}s..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    attempt += 1
        
        return wrapper
    
    return decorator

def validate_args(**validators) -> Callable:
    """
    Decorator to validate function arguments
    
    Parameters
    ----------
    **validators : callable
        Validation functions for each parameter
        
    Returns
    -------
    callable
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind arguments to the function signature
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument with its validator
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for parameter '{param_name}': {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def deprecated(
    reason: str = "This function is deprecated and will be removed in a future version."
) -> Callable:
    """
    Decorator to mark functions as deprecated
    
    Parameters
    ----------
    reason : str, optional
        Reason for deprecation
        
    Returns
    -------
    callable
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def singleton(cls):
    """
    Decorator to convert a class into a singleton
    
    Parameters
    ----------
    cls : class
        Class to convert to singleton
        
    Returns
    -------
    class
        Singleton class
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance