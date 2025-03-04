"""
Logging configuration for Yemen Market Analysis.
"""
import os
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .config import config


class JsonFormatter(logging.Formatter):
    """Formatter for JSON-structured log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
            
        # Add extra attributes
        if hasattr(record, 'data'):
            log_data['data'] = record.data
            
        return json.dumps(log_data)


def ensure_log_directory(log_dir: str) -> str:
    """Ensure log directory exists."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    json_format: bool = False,
    verbose_libraries: Optional[Dict[str, str]] = None
) -> None:
    """Set up logging configuration."""
    # Determine log directory
    if log_dir is None:
        log_dir = config.get('logging.logs_dir', 'logs')
    
    log_dir = ensure_log_directory(log_dir)
    
    # Set up basic configuration
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # File handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"market_analysis_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Select formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set levels for verbose libraries
    if verbose_libraries:
        for lib, lib_level in verbose_libraries.items():
            lib_logger = logging.getLogger(lib)
            lib_logger.setLevel(getattr(logging, lib_level.upper(), logging.WARNING))
    
    logging.getLogger(__name__).info(f"Logging initialized at level {log_level}")


def setup_logging_from_config() -> None:
    """Initialize logging from application configuration."""
    log_level = config.get('logging.log_level', 'INFO')
    log_dir = config.get('logging.logs_dir')
    verbose_libraries = config.get('logging.verbose_libraries', {})
    
    setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        verbose_libraries=verbose_libraries
    )