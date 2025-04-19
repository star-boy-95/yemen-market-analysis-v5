"""
Logging module for Yemen Market Analysis.

This module provides functions for setting up logging for the Yemen Market Analysis package.
It includes functions for configuring loggers, creating log handlers, and setting up
logging for different components of the package.
"""
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, TextIO, Tuple
import logging.config

from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.file_utils import create_directory

# Default logger for this module
logger = logging.getLogger(__name__)

@handle_errors
def setup_logging(
    config_path: Optional[Union[str, Path]] = None,
    default_level: int = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    handlers: Optional[List[str]] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        config_path: Path to the logging configuration file. If None, default configuration is used.
        default_level: Default logging level.
        log_dir: Directory for log files. If None, logs directory in the current directory is used.
        handlers: List of handlers to enable. If None, all handlers are enabled.
        
    Raises:
        YemenAnalysisError: If logging cannot be set up.
    """
    try:
        # Set up log directory
        if log_dir is None:
            log_dir = Path("logs")
        else:
            log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        create_directory(log_dir)
        
        # Load configuration from file if provided
        if config_path is not None:
            config_path = Path(config_path)
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    try:
                        config = yaml.safe_load(f.read())
                        
                        # Update log directory in file handlers
                        if 'handlers' in config:
                            for handler_name, handler_config in config['handlers'].items():
                                if 'filename' in handler_config:
                                    handler_config['filename'] = os.path.join(
                                        log_dir, os.path.basename(handler_config['filename'])
                                    )
                        
                        # Enable only specified handlers if provided
                        if handlers is not None and 'handlers' in config:
                            for handler_name in list(config['handlers'].keys()):
                                if handler_name not in handlers:
                                    # Remove handler from config
                                    del config['handlers'][handler_name]
                                    
                                    # Remove handler from root logger
                                    if 'root' in config and 'handlers' in config['root']:
                                        if handler_name in config['root']['handlers']:
                                            config['root']['handlers'].remove(handler_name)
                        
                        # Apply configuration
                        logging.config.dictConfig(config)
                        logger.info(f"Loaded logging configuration from {config_path}")
                        return
                    except Exception as e:
                        logger.warning(f"Error loading logging configuration from {config_path}: {e}")
                        # Fall back to default configuration
                        pass
        
        # Default configuration
        # Create a default configuration dictionary
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout',
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'level': 'DEBUG',
                    'formatter': 'detailed',
                    'filename': os.path.join(log_dir, 'yemen_market_analysis.log'),
                    'encoding': 'utf8',
                },
                'error_file': {
                    'class': 'logging.FileHandler',
                    'level': 'ERROR',
                    'formatter': 'detailed',
                    'filename': os.path.join(log_dir, 'yemen_market_analysis_error.log'),
                    'encoding': 'utf8',
                },
            },
            'loggers': {
                'src': {
                    'level': default_level,
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False,
                },
            },
            'root': {
                'level': default_level,
                'handlers': ['console', 'file', 'error_file'],
            }
        }
        
        # Enable only specified handlers if provided
        if handlers is not None:
            for handler_name in list(config['handlers'].keys()):
                if handler_name not in handlers:
                    # Remove handler from config
                    del config['handlers'][handler_name]
                    
                    # Remove handler from loggers
                    for logger_name, logger_config in config['loggers'].items():
                        if 'handlers' in logger_config and handler_name in logger_config['handlers']:
                            logger_config['handlers'].remove(handler_name)
                    
                    # Remove handler from root logger
                    if 'handlers' in config['root'] and handler_name in config['root']['handlers']:
                        config['root']['handlers'].remove(handler_name)
        
        # Apply configuration
        logging.config.dictConfig(config)
        logger.info("Applied default logging configuration")
    except Exception as e:
        # Print to stderr as logging might not be set up
        print(f"Error setting up logging: {e}", file=sys.stderr)
        raise YemenAnalysisError(f"Error setting up logging: {e}")

@handle_errors
def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Name of the logger.
        level: Logging level. If None, the logger's current level is used.
        
    Returns:
        Logger with the specified name and level.
        
    Raises:
        YemenAnalysisError: If the logger cannot be created.
    """
    try:
        # Get logger
        logger = logging.getLogger(name)
        
        # Set level if provided
        if level is not None:
            logger.setLevel(level)
        
        return logger
    except Exception as e:
        # Print to stderr as logging might not be set up
        print(f"Error getting logger {name}: {e}", file=sys.stderr)
        raise YemenAnalysisError(f"Error getting logger {name}: {e}")

@handle_errors
def add_file_handler(
    logger_name: str, file_path: Union[str, Path], level: int = logging.DEBUG,
    formatter: Optional[logging.Formatter] = None
) -> None:
    """
    Add a file handler to a logger.
    
    Args:
        logger_name: Name of the logger.
        file_path: Path to the log file.
        level: Logging level for the handler.
        formatter: Formatter for the handler. If None, a default formatter is used.
        
    Raises:
        YemenAnalysisError: If the handler cannot be added.
    """
    try:
        # Get logger
        logger = logging.getLogger(logger_name)
        
        # Create directory for log file if it doesn't exist
        create_directory(Path(file_path).parent)
        
        # Create file handler
        handler = logging.FileHandler(file_path)
        handler.setLevel(level)
        
        # Set formatter
        if formatter is None:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        logger.info(f"Added file handler to logger {logger_name}")
    except Exception as e:
        # Print to stderr as logging might not be set up
        print(f"Error adding file handler to logger {logger_name}: {e}", file=sys.stderr)
        raise YemenAnalysisError(f"Error adding file handler to logger {logger_name}: {e}")

@handle_errors
def log_function_call(func: Any) -> Any:
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate.
        
    Returns:
        Decorated function.
        
    Raises:
        YemenAnalysisError: If the function call cannot be logged.
    """
    def wrapper(*args, **kwargs):
        """Wrapper function for logging."""
        # Get function name and module
        func_name = func.__name__
        module_name = func.__module__
        
        # Get logger
        logger = logging.getLogger(module_name)
        
        # Log function call
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
        
        try:
            # Call function
            result = func(*args, **kwargs)
            
            # Log function return
            logger.debug(f"{func_name} returned {result}")
            
            return result
        except Exception as e:
            # Log error
            logger.error(f"Error in {func_name}: {e}")
            
            # Re-raise exception
            raise
    
    return wrapper

@handle_errors
def capture_stdout_stderr(
    logger: logging.Logger = None, stdout_level: int = logging.INFO,
    stderr_level: int = logging.ERROR
) -> Tuple[TextIO, TextIO]:
    """
    Capture stdout and stderr and redirect them to a logger.
    
    Args:
        logger: Logger to use. If None, the root logger is used.
        stdout_level: Logging level for stdout.
        stderr_level: Logging level for stderr.
        
    Returns:
        Tuple containing the original stdout and stderr.
        
    Raises:
        YemenAnalysisError: If stdout and stderr cannot be captured.
    """
    class LoggerWriter:
        """Writer class that redirects writes to a logger."""
        
        def __init__(self, logger: logging.Logger, level: int):
            """
            Initialize the logger writer.
            
            Args:
                logger: Logger to use.
                level: Logging level.
            """
            self.logger = logger
            self.level = level
            self.buffer = []
        
        def write(self, message: str) -> int:
            """
            Write a message to the logger.
            
            Args:
                message: Message to write.
                
            Returns:
                Number of characters written.
            """
            if message and not message.isspace():
                # Strip trailing newlines and spaces
                message = message.rstrip()
                
                # Log message
                self.logger.log(self.level, message)
            
            return len(message)
        
        def flush(self) -> None:
            """Flush the writer."""
            for message in self.buffer:
                self.write(message)
            
            self.buffer = []
    
    try:
        # Use root logger if not provided
        if logger is None:
            logger = logging.getLogger()
        
        # Save original stdout and stderr
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        
        # Create logger writers
        stdout_writer = LoggerWriter(logger, stdout_level)
        stderr_writer = LoggerWriter(logger, stderr_level)
        
        # Redirect stdout and stderr
        sys.stdout = stdout_writer  # type: ignore
        sys.stderr = stderr_writer  # type: ignore
        
        logger.info("Captured stdout and stderr")
        return orig_stdout, orig_stderr
    except Exception as e:
        # Print to stderr as logging might not be set up
        print(f"Error capturing stdout and stderr: {e}", file=sys.stderr)
        raise YemenAnalysisError(f"Error capturing stdout and stderr: {e}")

@handle_errors
def restore_stdout_stderr(stdout: TextIO, stderr: TextIO) -> None:
    """
    Restore stdout and stderr.
    
    Args:
        stdout: Original stdout.
        stderr: Original stderr.
        
    Raises:
        YemenAnalysisError: If stdout and stderr cannot be restored.
    """
    try:
        # Restore stdout and stderr
        sys.stdout = stdout
        sys.stderr = stderr
        
        logger.info("Restored stdout and stderr")
    except Exception as e:
        # Print to stderr as logging might not be set up
        print(f"Error restoring stdout and stderr: {e}", file=sys.stderr)
        raise YemenAnalysisError(f"Error restoring stdout and stderr: {e}")

@handle_errors
def create_log_file_name(name: str, log_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a log file name.
    
    Args:
        name: Base name for the log file.
        log_dir: Directory for log files. If None, logs directory in the current directory is used.
        
    Returns:
        Path to the log file.
        
    Raises:
        YemenAnalysisError: If the log file name cannot be created.
    """
    try:
        from datetime import datetime
        
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set log directory
        if log_dir is None:
            log_dir = Path("logs")
        else:
            log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        create_directory(log_dir)
        
        # Create log file name
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        return log_file
    except Exception as e:
        # Print to stderr as logging might not be set up
        print(f"Error creating log file name: {e}", file=sys.stderr)
        raise YemenAnalysisError(f"Error creating log file name: {e}")

def create_default_logging_config(
    log_dir: Union[str, Path], level: int = logging.INFO
) -> Dict[str, Any]:
    """
    Create a default logging configuration.
    
    Args:
        log_dir: Directory for log files.
        level: Default logging level.
        
    Returns:
        Default logging configuration.
    """
    # Create log directory path
    log_dir = Path(log_dir)
    
    # Create configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'yemen_market_analysis.log'),
                'encoding': 'utf8',
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, 'yemen_market_analysis_error.log'),
                'encoding': 'utf8',
            },
        },
        'loggers': {
            'src': {
                'level': level,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False,
            },
        },
        'root': {
            'level': level,
            'handlers': ['console', 'file', 'error_file'],
        }
    }
    
    return config

def save_logging_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save a logging configuration to a file.
    
    Args:
        config: Logging configuration.
        file_path: Path to the output file.
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    create_directory(file_path.parent)
    
    # Save configuration
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)