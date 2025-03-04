"""
Common decorators for Yemen Market Analysis.
"""
import time
import logging
import functools
from typing import Any, Callable, Dict, Optional, TypeVar, cast, Union

from .exceptions import YemenMarketError

F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')
logger = logging.getLogger(__name__)


def error_handler(
    fallback_value: Any = None,
    log_level: str = "error",
    include_traceback: bool = True,
    exception_types: tuple = (Exception,)
) -> Callable[[F], F]:
    """Universal error handler with configurable fallback."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(f"Error in {func.__name__}: {str(e)}", exc_info=include_traceback)
                return fallback_value() if callable(fallback_value) else fallback_value
        return cast(F, wrapper)
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exception_types: tuple = (Exception,)
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exception_types as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"Failed after {attempts} attempts: {str(e)}")
                        raise
                    
                    logger.warning(f"Attempt {attempts} failed: {str(e)}. Retrying in {current_delay:.2f}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
        return cast(F, wrapper)
    return decorator


def performance_tracker(name: Optional[str] = None, level: str = "debug") -> Callable[[F], F]:
    """Track function execution time."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log_method = getattr(logger, level.lower(), logger.debug)
                log_method(f"{func_name} completed in {elapsed:.3f} seconds")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.debug(f"{func_name} failed after {elapsed:.3f} seconds: {str(e)}")
                raise
                
        return cast(F, wrapper)
    return decorator


def validate_inputs(**validators: Callable[[Any], bool]) -> Callable[[F], F]:
    """Validate function inputs against provided validators."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function parameter names
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            
            # Create dictionary of args and kwargs
            all_args = dict(zip(arg_names, args))
            all_args.update(kwargs)
            
            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in all_args:
                    value = all_args[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for parameter '{param_name}': {value}")
            
            return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator


class performance_context:
    """Context manager for performance tracking."""
    
    def __init__(self, name: str, level: str = "debug"):
        self.name = name
        self.level = level
        self.start_time = 0.0
        
    def __enter__(self) -> 'performance_context':
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = time.time() - self.start_time
        log_method = getattr(logger, self.level.lower(), logger.debug)
        
        if exc_type:
            log_method(f"{self.name} failed after {elapsed:.3f} seconds: {str(exc_val)}")
        else:
            log_method(f"{self.name} completed in {elapsed:.3f} seconds")