"""
Logging configuration for the Yemen Market Integration Project.
Provides consistent logging setup across the application.
"""
import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import json
import datetime

class MaxLevelFilter(logging.Filter):
    """Filter that only passes records at or below a specific level"""
    
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level
        
    def filter(self, record):
        return record.levelno <= self.max_level

class ContextAdapter(logging.LoggerAdapter):
    """Adapter that adds context information to log records"""
    
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        if 'context' in self.extra:
            kwargs['extra']['context'] = self.extra['context']
        return msg, kwargs

def setup_logging(
    log_dir: str = 'logs',
    log_level: int = logging.INFO,
    log_file: str = 'yemen_analysis.log',
    error_file: str = 'errors.log',
    console: bool = True,
    log_format: Optional[str] = None,
    rotation: str = 'daily',
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 30,
    capture_warnings: bool = True,
    add_timestamp_to_files: bool = True
) -> logging.Logger:
    """
    Configure logging for the application
    
    Parameters
    ----------
    log_dir : str, optional
        Directory for log files
    log_level : int, optional
        Overall log level
    log_file : str, optional
        Main log file name
    error_file : str, optional
        Error log file name
    console : bool, optional
        Whether to log to console
    log_format : str, optional
        Log format string
    rotation : str, optional
        Log rotation strategy ('daily', 'size', or 'none')
    max_bytes : int, optional
        Maximum log file size for size-based rotation
    backup_count : int, optional
        Number of backup files to keep
    capture_warnings : bool, optional
        Whether to capture Python warnings
    add_timestamp_to_files : bool, optional
        Whether to add timestamp to log file names
        
    Returns
    -------
    logging.Logger
        Root logger
    """
    root_logger = logging.getLogger()
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set overall log level
    root_logger.setLevel(log_level)
    
    # Capture warnings if requested
    if capture_warnings:
        logging.captureWarnings(True)
    
    # Create log directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Default format if none provided
    if not log_format:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if 'context' in logging.Logger.manager.loggerDict:
            log_format = '%(asctime)s - [%(context)s] - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    
    # Add timestamp to filenames if requested
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if add_timestamp_to_files:
        name, ext = os.path.splitext(log_file)
        log_file = f"{name}_{timestamp}{ext}"
        
        name, ext = os.path.splitext(error_file)
        error_file = f"{name}_{timestamp}{ext}"
    
    # Set up log file handler
    if rotation == 'daily':
        file_handler = logging.handlers.TimedRotatingFileHandler(
            str(log_dir_path / log_file),
            when='midnight',
            interval=1,
            backupCount=backup_count
        )
    elif rotation == 'size':
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_dir_path / log_file),
            maxBytes=max_bytes,
            backupCount=backup_count
        )
    else:  # 'none'
        file_handler = logging.FileHandler(str(log_dir_path / log_file))
    
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Set up error file handler
    error_handler = logging.FileHandler(str(log_dir_path / error_file))
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)
    
    # Set up console handler if requested
    if console:
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # Info and debug to stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(console_formatter)
        stdout_handler.setLevel(log_level)
        stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
        root_logger.addHandler(stdout_handler)
        
        # Warnings and errors to stderr
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(console_formatter)
        stderr_handler.setLevel(logging.WARNING)
        root_logger.addHandler(stderr_handler)
    
    return root_logger

def get_logger_with_context(name: str, context: Dict[str, Any]) -> logging.LoggerAdapter:
    """
    Get a logger with added context information
    
    Parameters
    ----------
    name : str
        Logger name
    context : dict
        Context dictionary
        
    Returns
    -------
    logging.LoggerAdapter
        Logger adapter with context
    """
    logger = logging.getLogger(name)
    
    # Format context as a string for the log
    context_str = " ".join([f"{k}={v}" for k, v in context.items()])
    
    return ContextAdapter(logger, {'context': context_str})

class JsonFileHandler(logging.FileHandler):
    """Custom handler that writes logs as JSON objects"""
    
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        
    def emit(self, record):
        try:
            record_dict = {
                'timestamp': self.formatter.formatTime(record),
                'name': record.name,
                'level': record.levelname,
                'message': record.getMessage()
            }
            
            # Add extra attributes
            for key, value in record.__dict__.items():
                if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                              'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                              'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                              'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'):
                    try:
                        # Try to serialize to ensure it's JSON compatible
                        json.dumps({key: value})
                        record_dict[key] = value
                    except (TypeError, OverflowError):
                        record_dict[key] = str(value)
            
            # Add exception info if available
            if record.exc_info:
                record_dict['exception'] = self.formatter.formatException(record.exc_info)
            
            # Write as JSON
            self.stream.write(json.dumps(record_dict) + '\n')
            self.flush()
        except Exception:
            self.handleError(record)

def add_json_logging(
    log_dir: str = 'logs',
    json_log_file: str = 'yemen_analysis.json',
    log_level: int = logging.INFO
) -> None:
    """
    Add JSON logging to the application
    
    Parameters
    ----------
    log_dir : str, optional
        Directory for log files
    json_log_file : str, optional
        JSON log file name
    log_level : int, optional
        Log level for the JSON handler
    """
    # Create log directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Set up JSON file handler
    json_handler = JsonFileHandler(str(log_dir_path / json_log_file))
    json_handler.setLevel(log_level)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(json_handler)

def log_start_stop(logger: logging.Logger) -> None:
    """
    Log application start/stop events
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to use
    """
    import atexit
    
    logger.info("===== Application starting =====")
    
    def log_exit():
        logger.info("===== Application shutting down =====")
    
    atexit.register(log_exit)