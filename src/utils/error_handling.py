"""
Error handling module for Yemen Market Analysis.

This module provides error handling utilities for the Yemen Market Analysis package.
It includes a custom exception class and a decorator for handling errors.
"""
import logging
import functools
import traceback
from typing import Any, Callable, TypeVar, cast

# Type variable for generic type hints
F = TypeVar('F', bound=Callable[..., Any])

# Initialize logger
logger = logging.getLogger(__name__)

class YemenAnalysisError(Exception):
    """
    Custom exception class for Yemen Market Analysis.

    This exception is raised when an error occurs during the analysis.

    Attributes:
        message (str): Error message.
        original_error (Exception, optional): Original exception that caused this error.
    """

    def __init__(self, message: str, original_error: Exception = None):
        """
        Initialize the exception.

        Args:
            message: Error message.
            original_error: Original exception that caused this error.
        """
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Return a string representation of the exception.

        Returns:
            String representation of the exception.
        """
        if self.original_error:
            return f"{self.message} (Original error: {str(self.original_error)})"
        return self.message


def handle_errors(func: F) -> F:
    """
    Decorator for handling errors in functions.

    This decorator catches exceptions raised by the decorated function and
    wraps them in a YemenAnalysisError with additional context information.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except YemenAnalysisError:
            # Re-raise YemenAnalysisError without wrapping
            raise
        except Exception as e:
            # Get function name and module
            func_name = func.__name__
            module_name = func.__module__

            # Get traceback
            tb = traceback.format_exc()

            # Log error
            logger.error(f"Error in {module_name}.{func_name}: {str(e)}\n{tb}")

            # Wrap exception in YemenAnalysisError
            raise YemenAnalysisError(
                f"Error in {module_name}.{func_name}: {str(e)}",
                original_error=e
            ) from e

    return cast(F, wrapper)


def log_execution(func: F) -> F:
    """
    Decorator for logging function execution.

    This decorator logs the start and end of function execution, as well as
    any errors that occur during execution.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Get function name and module
        func_name = func.__name__
        module_name = func.__module__

        # Log start of execution
        logger.info(f"Starting {module_name}.{func_name}")

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Log end of execution
            logger.info(f"Completed {module_name}.{func_name}")

            return result
        except Exception as e:
            # Log error
            logger.error(f"Error in {module_name}.{func_name}: {str(e)}")

            # Re-raise exception
            raise

    return cast(F, wrapper)