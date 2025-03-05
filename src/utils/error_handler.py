"""
Centralized error handling for the Yemen Market Integration Project.
Provides consistent error handling patterns across the application.
"""
import logging
import functools
import traceback
from typing import Callable, TypeVar, Any, Optional, Type, Dict, Union

T = TypeVar('T')

class MarketIntegrationError(Exception):
    """Base exception for all application errors"""
    pass

class DataError(MarketIntegrationError):
    """Errors related to data loading or processing"""
    pass

class ModelError(MarketIntegrationError):
    """Errors related to model estimation or prediction"""
    pass

class VisualizationError(MarketIntegrationError):
    """Errors related to visualization generation"""
    pass

class ConfigError(MarketIntegrationError):
    """Errors related to configuration"""
    pass

class ValidationError(MarketIntegrationError):
    """Errors related to data validation"""
    pass

# Error registry for categorizing errors
ERROR_REGISTRY: Dict[Type[Exception], Type[MarketIntegrationError]] = {
    # Data errors
    FileNotFoundError: DataError,
    PermissionError: DataError,
    IsADirectoryError: DataError,
    
    # Analysis errors
    ValueError: ModelError,
    ZeroDivisionError: ModelError,
    
    # General errors that need contextual mapping
    TypeError: None,  # Will be mapped based on context
    KeyError: None,   # Will be mapped based on context
    AttributeError: None  # Will be mapped based on context
}

def handle_errors(
    logger: Optional[logging.Logger] = None,
    error_type: Union[Type[Exception], tuple] = Exception,
    default_return: Any = None,
    reraise: bool = False,
    error_map: Optional[Dict[Type[Exception], Type[MarketIntegrationError]]] = None
) -> Callable:
    """
    Decorator for centralized error handling
    
    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use for error reporting
    error_type : Type[Exception] or tuple, optional
        Exception type(s) to catch
    default_return : Any, optional
        Default value to return if an exception occurs
    reraise : bool, optional
        Whether to reraise the exception after logging
    error_map : Dict[Type[Exception], Type[MarketIntegrationError]], optional
        Mapping to convert standard exceptions to application exceptions
        
    Returns
    -------
    Callable
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal logger
            
            if logger is None:
                # Get module-level logger if none provided
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except error_type as e:
                # Get full traceback info
                tb = traceback.format_exc()
                
                # Get function signature for better error context
                func_name = func.__qualname__
                args_str = ", ".join([str(a) for a in args])
                kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                func_call = f"{func_name}({args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str})"
                
                # Log the error with context
                logger.error(f"Error in {func_call}: {str(e)}\n{tb}")
                
                # Map standard exception to application exception if needed
                combined_map = dict(ERROR_REGISTRY)
                if error_map:
                    combined_map.update(error_map)
                
                mapped_error_class = combined_map.get(type(e))
                if mapped_error_class and reraise:
                    raise mapped_error_class(str(e)) from e
                elif reraise:
                    raise
                
                return default_return
                
        return wrapper
    return decorator

def capture_error(
    e: Exception, 
    context: str = "",
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Utility function to capture and log errors outside decorators
    
    Parameters
    ----------
    e : Exception
        The exception to capture
    context : str, optional
        Additional context information
    logger : logging.Logger, optional
        Logger to use, defaults to root logger
    """
    if logger is None:
        logger = logging.getLogger()
        
    tb = traceback.format_exc()
    logger.error(f"Error in {context}: {str(e)}\n{tb}")