# Yemen Market Integration Utilities Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
   - [Project Environment Setup](#project-environment-setup)
   - [Quick Start](#quick-start)
   - [Core Principles](#core-principles)
3. [Utility Modules Overview](#utility-modules-overview)
   - [Module Organization](#module-organization)
   - [Dependency Relationships](#dependency-relationships)
   - [Import Patterns](#import-patterns)
4. [Error Handling](#error-handling)
   - [Error Handler Decorator](#error-handler-decorator)
   - [Custom Exception Hierarchy](#custom-exception-hierarchy)
   - [Error Logging and Recovery](#error-logging-and-recovery)
   - [Error Registry System](#error-registry-system)
5. [Configuration Management](#configuration-management)
   - [Configuration Setup](#configuration-setup)
   - [Accessing Configuration](#accessing-configuration)
   - [Environment Variables](#environment-variables)
   - [Model-Specific Configuration](#model-specific-configuration)
6. [Logging](#logging)
   - [Setting Up Logging](#setting-up-logging)
   - [Context-Aware Logging](#context-aware-logging)
   - [Logging Best Practices](#logging-best-practices)
   - [JSON Logging for Analysis](#json-logging-for-analysis)
7. [Performance Optimization](#performance-optimization)
   - [M1/M2 Optimization](#m1m2-optimization)
   - [Parallel Processing](#parallel-processing)
   - [Memory Optimization](#memory-optimization)
   - [Caching Strategies](#caching-strategies)
8. [Data Validation](#data-validation)
   - [DataFrame Validation](#dataframe-validation)
   - [GeoDataFrame Validation](#geodataframe-validation)
   - [Time Series Validation](#time-series-validation)
   - [Model Input Validation](#model-input-validation)
9. [File Operations](#file-operations)
   - [Reading and Writing Files](#reading-and-writing-files)
   - [Processing Large Files](#processing-large-files)
   - [Atomic File Operations](#atomic-file-operations)
   - [File Backups and Versioning](#file-backups-and-versioning)
10. [Data Processing](#data-processing)
    - [Cleaning and Normalization](#cleaning-and-normalization)
    - [Feature Engineering](#feature-engineering)
    - [Time Series Processing](#time-series-processing)
    - [Missing Data Handling](#missing-data-handling)
11. [Statistical Analysis](#statistical-analysis)
    - [Unit Root Testing](#unit-root-testing)
    - [Cointegration Testing](#cointegration-testing)
    - [Threshold Models](#threshold-models)
    - [Statistical Utilities](#statistical-utilities)
12. [Spatial Analysis](#spatial-analysis)
    - [GeoDataFrame Operations](#geodataframe-operations)
    - [Spatial Weight Matrices](#spatial-weight-matrices)
    - [Distance Calculations](#distance-calculations)
    - [Market Catchment Analysis](#market-catchment-analysis)
13. [Visualization](#visualization)
    - [Time Series Visualization](#time-series-visualization)
    - [Spatial Visualization](#spatial-visualization)
    - [Advanced Plot Types](#advanced-plot-types)
    - [Publication-Ready Graphics](#publication-ready-graphics)
14. [Testing and Quality Assurance](#testing-and-quality-assurance)
    - [Unit Testing](#unit-testing)
    - [Integration Testing](#integration-testing)
    - [Performance Testing](#performance-testing)
15. [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Debugging Strategies](#debugging-strategies)
    - [Support Resources](#support-resources)
16. [Quick Reference](#quick-reference)
    - [Key Functions Summary](#key-functions-summary)
    - [Common Patterns](#common-patterns)

## Introduction

The Yemen Market Integration project utilities package provides a comprehensive set of tools designed specifically for analyzing market integration in conflict-affected Yemen. These utilities are optimized for Apple Silicon (M1/M2) hardware and implement best practices for econometric analysis, spatial data processing, and high-performance computing.

This guide serves as the definitive reference for using these utilities effectively throughout the project. Following these patterns will ensure consistency, reliability, and performance across all project components.

## Getting Started

### Project Environment Setup

To initialize the project environment with standardized configuration and logging:

```python
from src.utils import setup_project_environment

# Initialize project environment with a single function call
config, logger = setup_project_environment(
    config_file='config/settings.yaml',
    log_dir='logs'
)

# Log system information to verify setup
logger.info(f"Starting analysis with configuration: {config.get('analysis_settings')}")
logger.info(f"System information: {get_system_info()}")
```

The `setup_project_environment` function handles:

- Loading and validating configuration
- Setting up logging with appropriate handlers
- Optimizing Python for M1/M2 hardware if available
- Initializing environment variables
- Setting up exception handlers
- Returning the configuration and logger objects for immediate use

### Quick Start

For a new module, follow this template to ensure proper integration with the utilities system:

```python
"""
Module description and purpose.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

from src.utils import (
    # Error handling
    handle_errors, DataError, ModelError,
    
    # Data validation
    validate_dataframe, validate_geodataframe, raise_if_invalid,
    
    # File operations
    read_csv, write_csv, read_geojson, write_geojson,
    
    # Data processing
    clean_column_names, convert_dates, fill_missing_values
)

# Initialize module logger
logger = logging.getLogger(__name__)

@handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
def example_function(input_path: str) -> pd.DataFrame:
    """
    Function description with detailed docstring.
    
    Parameters
    ----------
    input_path : str
        Path to input file
        
    Returns
    -------
    pandas.DataFrame
        Processed data
    """
    # Load data using utility function
    df = read_csv(input_path, parse_dates=['date'])
    
    # Validate input data
    valid, errors = validate_dataframe(
        df, 
        required_columns=['date', 'price', 'region']
    )
    raise_if_invalid(valid, errors, f"Invalid input data from {input_path}")
    
    # Process data using utility functions
    df = clean_column_names(df)
    df = convert_dates(df, date_cols=['date'])
    df = fill_missing_values(df, numeric_strategy='median')
    
    logger.info(f"Successfully processed {len(df)} records from {input_path}")
    return df
```

### Core Principles

The utilities package follows these core principles:

1. **Comprehensive error handling**: All functions use the `handle_errors` decorator for consistent error management
2. **Strict validation**: Input validation occurs before any processing
3. **Performance optimization**: Compute-intensive functions use the `m1_optimized` decorator
4. **Contextual logging**: All operations log appropriate information for monitoring and debugging
5. **Vectorized operations**: Data transformations use vectorized operations instead of loops where possible
6. **Memory efficiency**: Large data processing uses chunking and memory optimization techniques
7. **Configurable behavior**: Functions accept configuration parameters rather than hard-coding values

## Utility Modules Overview

### Module Organization

The utilities package contains the following modules:

- **error_handler.py**: Centralized error handling with custom exceptions
  - Core functions: `handle_errors`, `capture_error`
  - Custom exceptions: `MarketIntegrationError`, `DataError`, `ModelError`, `ValidationError`

- **config.py**: Configuration management with environment variables support
  - Core functions: `initialize_config`, `Config.get`, `Config.load_from_file`
  - Singleton pattern: `config` global instance

- **logging_setup.py**: Comprehensive logging with context-aware adapters
  - Core functions: `setup_logging`, `get_logger_with_context`, `add_json_logging`
  - Custom classes: `ContextAdapter`, `JsonFileHandler`

- **decorators.py**: Performance-focused decorators
  - Core decorators: `timer`, `m1_optimized`, `disk_cache`, `memoize`, `retry`
  - Advanced decorators: `validate_args`, `rate_limited`, `singleton`

- **validation.py**: Data validation for DataFrames and spatial data
  - Core functions: `validate_dataframe`, `validate_geodataframe`, `validate_time_series`
  - Helper functions: `raise_if_invalid`, `validate_model_inputs`

- **file_utils.py**: Optimized file operations with chunked processing
  - Core functions: `read_csv`, `write_csv`, `read_geojson`, `write_geojson`
  - Advanced functions: `read_large_file_chunks`, `read_large_csv_chunks`, `AtomicFileWriter`

- **data_utils.py**: Data manipulation with vectorized operations
  - Core functions: `clean_column_names`, `convert_dates`, `fill_missing_values`
  - Advanced functions: `create_lag_features`, `create_rolling_features`, `detect_outliers`

- **stats_utils.py**: Statistical analysis with threshold cointegration tests
  - Core functions: `test_stationarity`, `test_cointegration`, `fit_threshold_vecm`
  - Advanced functions: `test_causality_granger`, `calculate_half_life`, `bootstrap_confidence_interval`

- **spatial_utils.py**: GIS operations optimized for geospatial analysis
  - Core functions: `reproject_gdf`, `calculate_distances`, `create_spatial_weight_matrix`
  - Advanced functions: `find_nearest_points`, `compute_accessibility_index`, `create_market_catchments`

- **plotting_utils.py**: Data visualization utilities with sensible defaults
  - Core functions: `plot_time_series`, `plot_multiple_time_series`, `plot_time_series_by_group`
  - Advanced functions: `plot_heatmap`, `plot_dual_axis`, `add_annotations`

- **performance_utils.py**: M1-specific optimizations and parallel processing
  - Core functions: `configure_system_for_performance`, `parallelize_dataframe`, `optimize_dataframe`
  - Helper functions: `chunked_file_reader`, `memory_usage_decorator`, `get_system_info`

### Dependency Relationships

The utilities modules have the following dependency hierarchy (from lowest to highest level):

1. **error_handler.py** (no internal dependencies)
2. **config.py** (depends on error_handler)
3. **logging_setup.py** (depends on error_handler)
4. **decorators.py** (depends on error_handler, logging_setup)
5. **validation.py** (depends on error_handler, decorators)
6. **file_utils.py** (depends on error_handler, decorators, validation)
7. **data_utils.py** (depends on error_handler, decorators, validation)
8. **performance_utils.py** (depends on error_handler, decorators)
9. **stats_utils.py** (depends on error_handler, decorators, validation, data_utils)
10. **spatial_utils.py** (depends on error_handler, decorators, validation, data_utils)
11. **plotting_utils.py** (depends on error_handler, decorators)

When importing, always follow this hierarchy to avoid circular imports.

### Import Patterns

For consistency across the project, follow these import patterns:

1. **Standard library imports** first:

```python
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
```

2. **Third-party library imports** next:

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.api as sm
```

3. **Project imports** last, with specific function imports:

```python
from src.utils import handle_errors, validate_dataframe, read_csv
from src.utils.error_handler import DataError
from src.utils.spatial_utils import calculate_distances
```

4. **Utility bundle imports** for related functionality:

```python
from src.utils import (
    # File operations
    read_csv, write_csv, read_geojson, write_geojson,
    
    # Data processing
    clean_column_names, convert_dates, fill_missing_values
)
```

## Error Handling

### Error Handler Decorator

The `handle_errors` decorator provides centralized error handling throughout the application:

```python
from src.utils import handle_errors, DataError
import logging

logger = logging.getLogger(__name__)

@handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
def load_market_data(file_path):
    """
    Load market data from a CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file
        
    Returns
    -------
    pandas.DataFrame
        Loaded market data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Load data
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    required_cols = ['date', 'price', 'market']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in {file_path}")
            
    return df
```

Key benefits of this approach:

- Consistent error handling across the codebase
- Automatic logging of errors with context
- Optional conversion of standard exceptions to application-specific ones
- Control over re-raising or returning a default value

#### Advanced Usage

You can customize the behavior of the error handler:

```python
@handle_errors(
    logger=logger,              # Logger to use for error reporting
    error_type=(IOError, KeyError),  # Specific exceptions to catch
    default_return=pd.DataFrame(),   # Value to return on error (if not reraising)
    reraise=True,               # Whether to reraise as application exception
    error_map={                 # Custom mapping from standard to app exceptions
        ValueError: ValidationError,
        KeyError: ConfigError
    }
)
def process_config(config_section):
    # Implementation
```

### Custom Exception Hierarchy

The utilities package defines a hierarchy of exceptions for specific error types:

```
MarketIntegrationError
  ├── DataError           # Data loading, parsing, or format errors
  ├── ModelError          # Statistical model estimation or prediction errors
  ├── ValidationError     # Input validation failures
  ├── ConfigError         # Configuration issues
  └── VisualizationError  # Plot generation problems
```

Use these specific exceptions to indicate the error type:

```python
from src.utils import DataError, ModelError, ValidationError

# Raise specific exceptions for better error tracking
if not file_exists:
    raise DataError(f"Required data file not found: {file_path}")

if np.isnan(data).any():
    raise ValidationError("Input data contains NaN values")

if not model.converged:
    raise ModelError("Threshold model failed to converge")
```

### Error Logging and Recovery

The error handling system logs comprehensive information for debugging:

```
ERROR [2023-08-15 14:23:45] - Error in load_market_data('data/prices.csv'): Required column price not found in data/prices.csv
Traceback (most recent call last):
  File "src/utils/error_handler.py", line 45, in wrapper
    return func(*args, **kwargs)
  File "src/data/loader.py", line 32, in load_market_data
    raise ValueError(f"Required column {col} not found in {file_path}")
ValueError: Required column price not found in data/prices.csv
```

For manual error capturing outside decorators, use the `capture_error` function:

```python
from src.utils import capture_error

try:
    # Risky operation
    result = complex_operation()
except Exception as e:
    capture_error(e, context="running complex_operation", logger=logger)
    # Recovery logic
    result = fallback_operation()
```

### Error Registry System

The error handler uses a registry to map standard exceptions to application exceptions:

```python
# Default mappings in ERROR_REGISTRY
ERROR_REGISTRY = {
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
```

You can supplement this registry with more specific mappings in the `error_map` parameter.

## Configuration Management

### Configuration Setup

Initialize configuration at the start of your application:

```python
from src.utils import initialize_config

# Initialize with multiple sources
config = initialize_config(
    config_file='config/settings.yaml',  # Primary config file
    env_prefix="YEMEN_",                # Environment variable prefix
    defaults={                          # Default fallback values
        'analysis': {
            'max_lags': 4,
            'alpha': 0.05,
            'trim': 0.15
        },
        'data': {
            'start_date': '2020-01-01',
            'end_date': '2023-12-31'
        }
    }
)
```

Configuration priority (highest to lowest):

1. Environment variables with the specified prefix
2. Values from the config file
3. Default values provided in the initialization

### Accessing Configuration

The singleton `config` object provides access to configuration values throughout your application:

```python
from src.utils import config

# Access top-level settings
data_path = config.get('data_path', 'data')  # With default fallback value

# Access nested settings with dot notation
alpha = config.get('analysis.alpha')
trim = config.get('analysis.threshold.trim', 0.15)  # With default fallback

# Get an entire section as a dictionary
analysis_config = config.get_section('analysis')

# For frequently accessed values, unpack at the module level
MAX_LAGS = config.get('analysis.max_lags', 4)
ALPHA = config.get('analysis.alpha', 0.05)
```

### Environment Variables

The configuration system automatically loads from environment variables with the specified prefix:

```bash
# Set in .env file or export in shell
export YEMEN_ANALYSIS__MAX_LAGS=6
export YEMEN_DATA__START_DATE=2021-01-01
export YEMEN_LOG_LEVEL=DEBUG
```

These are parsed and converted to the appropriate nested structure:

- `YEMEN_ANALYSIS__MAX_LAGS=6` becomes `config['analysis']['max_lags'] = 6`
- `YEMEN_DATA__START_DATE=2021-01-01` becomes `config['data']['start_date'] = '2021-01-01'`
- `YEMEN_LOG_LEVEL=DEBUG` becomes `config['log_level'] = 'DEBUG'`

Values are automatically converted to appropriate types:

- `true`/`false` → boolean
- Numeric strings → int/float
- Lists (comma-separated) → Python lists

### Model-Specific Configuration

For econometric models, use the specialized getter:

```python
tvecm_params = config.get_model_params('tvecm')
# Equivalent to config.get('models.tvecm')

print(tvecm_params)
# {
#   'k_ar_diff': 2,
#   'coint_rank': 1,
#   'deterministic': 'ci',
#   'trim': 0.15,
#   'n_grid': 300
# }
```

This pattern helps standardize model parameters across the application.

## Logging

### Setting Up Logging

Set up logging at the beginning of your application:

```python
from src.utils import setup_logging
import logging

# Initialize application-wide logging
logger = setup_logging(
    log_dir='logs',               # Directory for log files
    log_level=logging.INFO,       # Overall log level
    log_file='yemen_analysis.log', # Main log file name
    error_file='errors.log',      # Separate file for errors
    console=True,                 # Also log to console
    rotation='daily',             # Rotate logs daily
    backup_count=30,              # Keep 30 days of logs
    capture_warnings=True         # Capture Python warnings
)

# Log application startup
logger.info("===== Yemen Market Integration Analysis Starting =====")
```

For each module, get a module-specific logger:

```python
import logging

# Get logger for current module
logger = logging.getLogger(__name__)
```

### Context-Aware Logging

For complex analyses, use context-aware logging to add metadata to log messages:

```python
from src.utils import get_logger_with_context

# Create logger with specific context
context_logger = get_logger_with_context(__name__, {
    'region': 'abyan', 
    'commodity': 'beans',
    'model': 'threshold_var'
})

# Log with context included
context_logger.info("Starting model estimation")
# Output: 2023-08-15 14:30:12 - [region=abyan commodity=beans model=threshold_var] - data_loader - INFO - Starting model estimation
```

This approach helps filter and analyze logs by specific dimensions.

### Logging Best Practices

Follow these guidelines for effective logging:

1. **Use appropriate log levels**:
   - `DEBUG`: Detailed information, typically of interest only when diagnosing problems
   - `INFO`: Confirmation that things are working as expected
   - `WARNING`: Indication that something unexpected happened, but the application still works
   - `ERROR`: Due to a more serious problem, the application has not been able to perform a function
   - `CRITICAL`: A serious error, indicating that the application itself may be unable to continue running

2. **Include actionable information**:

   ```python
   # Good - specific and actionable
   logger.error(f"Failed to read file {file_path}: {str(e)}")
   
   # Bad - vague and not actionable
   logger.error("An error occurred")
   ```

3. **Log start and end of key processes**:

   ```python
   logger.info(f"Starting data import from {file_path}")
   # ... process data ...
   logger.info(f"Completed data import: {len(df)} records processed")
   ```

4. **Include metrics where helpful**:

   ```python
   logger.info(f"Model converged after {iterations} iterations, AIC={aic:.4f}")
   ```

### JSON Logging for Analysis

For programmatic log analysis, enable JSON logging:

```python
from src.utils import add_json_logging

# Add JSON logging in addition to regular logging
add_json_logging(
    log_dir='logs',
    json_log_file='yemen_analysis.json',
    log_level=logging.INFO
)
```

This creates structured logs in JSON format:

```json
{"timestamp": "2023-08-15 14:32:45", "name": "src.data.loader", "level": "INFO", "message": "Loaded 2500 records", "region": "abyan", "commodity": "beans"}
```

These logs can be easily analyzed using pandas:

```python
import pandas as pd

# Load logs into a DataFrame
logs_df = pd.read_json('logs/yemen_analysis.json', lines=True)

# Analyze errors by module
error_counts = logs_df[logs_df['level'] == 'ERROR'].groupby('name').size()

# Filter logs by context
abyan_logs = logs_df[logs_df['region'] == 'abyan']
```

## Performance Optimization

### M1/M2 Optimization

For compute-intensive functions, use the `m1_optimized` decorator:

```python
from src.utils import m1_optimized
import numpy as np

@m1_optimized(use_numba=True, parallel=True)
def calculate_pairwise_distances(points):
    """
    Calculate pairwise distances between all points.
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of points with shape (n, 2)
        
    Returns
    -------
    numpy.ndarray
        Distance matrix with shape (n, n)
    """
    n = len(points)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(((points[i] - points[j]) ** 2).sum())
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances
```

This decorator provides:

- Automatic detection of Apple Silicon (M1/M2) hardware
- JIT compilation with Numba if available (dramatically faster loops)
- Parallel execution across available cores
- Graceful fallback if Numba is not available

For highly optimized code, focus on array operations rather than explicit loops:

```python
# Vectorized version (much faster, especially on M1/M2)
@m1_optimized()
def calculate_pairwise_distances_vectorized(points):
    # Calculate squared distances using broadcasting
    squared_dists = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
    # Take square root
    return np.sqrt(squared_dists)
```

### Parallel Processing

For data-parallel operations, use the `parallelize_dataframe` function:

```python
from src.utils import parallelize_dataframe

def process_chunk(df_chunk):
    """Process a chunk of the DataFrame."""
    # Complex processing logic
    result = df_chunk.copy()
    result['transformed'] = df_chunk['value'] ** 2 + df_chunk['other'] * 3
    return result

# Process large DataFrame in parallel
result_df = parallelize_dataframe(
    large_df,            # Input DataFrame
    process_chunk,       # Function to apply to each chunk
    n_workers=4,         # Number of parallel workers
    chunk_size=10000     # Rows per chunk
)
```

This approach:

- Splits the DataFrame into chunks
- Processes each chunk in a separate process
- Combines the results back into a single DataFrame
- Optimizes memory usage by not keeping the entire DataFrame in memory at once

For operations that don't fit the DataFrame pattern, use the `parallelize` function:

```python
from src.utils import parallelize

def process_file(file_path):
    """Process a single file."""
    # Implementation
    return result

# Process multiple files in parallel
results = parallelize(
    process_file,              # Function to apply
    file_paths,                # List of arguments
    n_workers=4,               # Number of parallel workers
    progress_bar=True          # Show progress bar
)
```

### Memory Optimization

For large DataFrames, optimize memory usage:

```python
from src.utils import optimize_dataframe

# Optimize memory usage by choosing appropriate dtypes
optimized_df = optimize_dataframe(
    df,                 # Input DataFrame
    downcast=True,      # Downcast numeric columns to smallest possible type
    category_min_size=10 # Convert string columns with ≤10 unique values to category
)
```

This approach can reduce memory usage by 50-90% in many cases, particularly for data with:

- Integer columns that don't need 64 bits
- Float columns that can use less precision
- String columns with repeated values
- Boolean columns

For large file processing, use chunked reading:

```python
from src.utils import chunked_file_reader, read_large_csv_chunks

# Process a large text file in chunks
for chunk in chunked_file_reader('large_file.txt', chunk_size=1024*1024):
    # Process each chunk of text
    process_text_chunk(chunk)

# Process a large CSV file in chunks
for chunk_df in read_large_csv_chunks('large_file.csv', chunk_size=10000):
    # Process each chunk of rows
    process_df_chunk(chunk_df)
```

### Caching Strategies

For expensive calculations that may be repeated, use memoization:

```python
from src.utils import memoize

@memoize
def calculate_complex_statistic(time_series):
    """Calculate a complex statistic of a time series."""
    # Expensive calculation
    return result
```

For calculations that should persist across sessions, use disk caching:

```python
from src.utils import disk_cache

@disk_cache(cache_dir='.cache', expiration_seconds=86400)  # 24 hour expiration
def fetch_external_data(source_id, start_date, end_date):
    """Fetch data from an external source."""
    # Expensive API call or database query
    return data
```

## Data Validation

### DataFrame Validation

Validate DataFrames before processing to catch issues early:

```python
from src.utils import validate_dataframe, raise_if_invalid
import pandas as pd

def process_price_data(df):
    """Process price data."""
    # Validate input DataFrame
    valid, errors = validate_dataframe(
        df,
        required_columns=['date', 'price', 'commodity', 'admin1'],
        column_types={
            'date': pd.Timestamp,
            'price': float,
            'commodity': str,
            'admin1': str
        },
        min_rows=1,
        check_nulls=True
    )
    
    # Raise exception if validation fails
    raise_if_invalid(valid, errors, "Invalid price data")
    
    # Safe to proceed with processing now
    # ...
```

For more complex validation, use custom validators:

```python
from src.utils import validate_dataframe

def validate_price_range(price_series):
    """Check that prices are within reasonable range."""
    return (price_series > 0).all() and (price_series < 1e6).all()

def validate_date_range(date_series):
    """Check that dates are within expected range."""
    return (date_series >= pd.Timestamp('2020-01-01')).all() and \
           (date_series <= pd.Timestamp('2023-12-31')).all()

# Use custom validators
valid, errors = validate_dataframe(
    df,
    required_columns=['date', 'price'],
    custom_validators={
        'price': validate_price_range,
        'date': validate_date_range
    }
)
```

### GeoDataFrame Validation

For spatial data, use specialized validation:

```python
from src.utils import validate_geodataframe, raise_if_invalid

# Validate spatial data
valid, errors = validate_geodataframe(
    markets_gdf,
    required_columns=['admin1', 'market_name', 'latitude', 'longitude'],
    crs="EPSG:4326",          # Check coordinate reference system
    geometry_type="Point"     # Ensure all geometries are points
)
raise_if_invalid(valid, errors, "Invalid market locations data")
```

### Time Series Validation

For econometric analysis, validate time series properties:

```python
from src.utils import validate_time_series, raise_if_invalid

# Validate time series before analysis
valid, errors = validate_time_series(
    price_series,
    min_length=30,             # Minimum length for reliable analysis
    max_nulls=0,               # Maximum allowed null values
    check_stationarity=True,   # Check if series is stationary
    check_constant=True        # Check if series has constant values
)
raise_if_invalid(valid, errors, "Invalid price time series")
```

### Model Input Validation

Validate model parameters before estimation:

```python
from src.utils import validate_model_inputs, raise_if_invalid

# Validate model parameters
valid, errors = validate_model_inputs(
    model_name="tvecm",
    params={
        'k_ar_diff': 2,
        'coint_rank': 1,
        'deterministic': 'ci',
        'trim': 0.15
    },
    required_params={'k_ar_diff', 'coint_rank', 'deterministic'},
    param_validators={
        'k_ar_diff': lambda x: isinstance(x, int) and x > 0,
        'trim': lambda x: 0 < x < 0.5
    }
)
raise_if_invalid(valid, errors, "Invalid TVECM model parameters")
```

## File Operations

### Reading and Writing Files

Use the utility functions for safe file operations:

```python
from src.utils import (
    read_csv, write_csv, 
    read_geojson, write_geojson,
    read_json, write_json,
    read_yaml, write_yaml,
    read_pickle, write_pickle
)

# Reading files with enhanced error handling
df = read_csv('data/prices.csv', parse_dates=['date'])
gdf = read_geojson('data/admin_boundaries.geojson')
config = read_yaml('config/settings.yaml')
model = read_pickle('models/tvecm_model.pkl')

# Writing files with directory creation and atomic operations
write_csv(results_df, 'results/price_analysis.csv', index=False)
write_geojson(markets_gdf, 'results/market_locations.geojson')
write_json(analysis_results, 'results/summary.json', indent=2)
write_pickle(fitted_model, 'models/fitted_tvecm.pkl')
```

These functions provide:

- Comprehensive error handling
- Path creation if needed
- Appropriate type conversions
- Logging of operations
- Consistent behavior across file types

### Processing Large Files

For files too large to fit in memory, use chunked processing:

```python
from src.utils import (
    read_large_csv_chunks,
    read_large_geojson_chunks,
    read_large_file_chunks
)

# Process a large CSV file in chunks
total_rows = 0
for chunk_df in read_large_csv_chunks('large_prices.csv', chunk_size=10000):
    # Process each chunk
    process_chunk(chunk_df)
    total_rows += len(chunk_df)

# Process a large GeoJSON file in chunks
for chunk_gdf in read_large_geojson_chunks('large_admin.geojson', chunk_size=1000):
    # Process each chunk
    process_spatial_chunk(chunk_gdf)

# Process a large text file in chunks
for text_chunk in read_large_file_chunks('logs.txt', chunk_size=1024*1024):
    # Process each chunk
    process_text_chunk(text_chunk)
```

This approach keeps memory usage constant regardless of file size.

### Atomic File Operations

For critical files, use atomic operations to prevent corruption:

```python
from src.utils import AtomicFileWriter

# Write a file atomically (all or nothing)
with AtomicFileWriter('important_results.csv', mode='w') as f:
    # Write to the file
    f.write("date,value\n")
    for date, value in results:
        f.write(f"{date},{value}\n")
    
    # If this block completes without error, the file is replaced atomically
    # If an error occurs, the original file is untouched
```

This approach ensures that the file is either completely written or not modified at all, preventing partial updates.

### File Backups and Versioning

Before modifying important files, create automatic backups:

```python
from src.utils import create_backup, file_hash

# Create a timestamped backup
backup_path = create_backup('important_model.pkl')
print(f"Backup created at {backup_path}")

# Calculate file hash for verification
original_hash = file_hash('important_model.pkl', algorithm='sha256')
backup_hash = file_hash(backup_path, algorithm='sha256')
assert original_hash == backup_hash, "Backup verification failed"
```

For versioning files during analysis:

```python
from src.utils import write_csv
from datetime import datetime

# Add timestamp to filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
write_csv(results_df, f'results/analysis_{timestamp}.csv')
```

## Data Processing

### Cleaning and Normalization

Clean and standardize data before analysis:

```python
from src.utils import (
    clean_column_names,
    convert_dates,
    fill_missing_values,
    normalize_columns,
    detect_outliers,
    winsorize_columns
)

# Clean column names
df = clean_column_names(df)  # Lowercase, replace spaces with underscores, etc.

# Convert date columns
df = convert_dates(df, date_cols=['date', 'report_date'])

# Fill missing values appropriately
df = fill_missing_values(
    df,
    numeric_strategy='median',  # Use median for numeric columns
    categorical_strategy='mode', # Use mode for categorical columns
    date_strategy='nearest',    # Use nearest valid date for date columns
    group_cols=['admin1', 'commodity']  # Group by these columns when filling
)

# Normalize numeric columns for comparison
df = normalize_columns(
    df,
    columns=['price', 'quantity', 'conflict_intensity'],
    method='zscore'  # Standardize to mean=0, std=1
)

# Detect outliers for investigation
df = detect_outliers(
    df,
    columns=['price', 'quantity'],
    method='zscore',
    threshold=3.0
)

# Handle extreme values
df = winsorize_columns(
    df,
    columns=['price', 'quantity'],
    limits=(0.01, 0.01)  # Cap at 1st and 99th percentiles
)
```

### Feature Engineering

Create derived features for analysis:

```python
from src.utils import (
    create_lag_features,
    create_rolling_features,
    create_date_features,
    bin_numeric_column,
    calculate_price_changes,
    encode_categorical
)

# Create lag features for time series analysis
df = create_lag_features(
    df,
    cols=['price', 'quantity'],
    lags=[1, 3, 6, 12],  # Monthly lags up to a year
    group_cols=['admin1', 'commodity']
)

# Create rolling window features
df = create_rolling_features(
    df,
    cols=['price'],
    windows=[3, 6, 12],
    stats=['mean', 'std', 'min', 'max'],
    group_cols=['admin1', 'commodity']
)

# Extract features from dates
df = create_date_features(
    df,
    date_col='date',
    features=['year', 'month', 'quarter', 'dayofyear', 'weekend']
)

# Bin continuous values into categories
df = bin_numeric_column(
    df,
    column='price',
    bins=5,  # 5 equal-width bins
    labels=['very_low', 'low', 'medium', 'high', 'very_high']
)

# Calculate price changes
df = calculate_price_changes(
    df,
    price_col='price',
    date_col='date',
    method='pct',  # Percentage change
    periods=[1, 3, 6],  # 1, 3, and 6-month changes
    group_cols=['admin1', 'commodity']
)

# Encode categorical variables for modeling
df = encode_categorical(
    df,
    columns=['admin1', 'commodity', 'exchange_rate_regime'],
    method='onehot',
    drop_first=True
)
```

### Time Series Processing

Process time series data for econometric analysis:

```python
from src.utils import (
    aggregate_time_series,
    compute_rolling_correlation,
    compute_variance_ratio,
    test_white_noise,
    test_autocorrelation
)

# Aggregate time series to a different frequency
monthly_df = aggregate_time_series(
    df,
    date_col='date',
    value_cols=['price', 'quantity'],
    freq='M',  # Monthly
    agg_func='mean'
)

# Compute rolling correlation between series
rolling_corr = compute_rolling_correlation(
    north_prices,
    south_prices,
    window=12  # 12-month rolling correlation
)

# Check if price behaves like a random walk
vr_results = compute_variance_ratio(
    price_series,
    periods=[2, 5, 10, 20],
    overlapping=True
)
print(f"Random walk hypothesis rejected: {vr_results['random_walk_rejected']}")

# Test if residuals are white noise
wn_results = test_white_noise(residuals)
print(f"Residuals are white noise: {wn_results['is_white_noise']}")

# Test for autocorrelation
ac_results = test_autocorrelation(residuals, lags=12)
print(f"Autocorrelation detected: {ac_results['has_autocorrelation']}")
```

### Missing Data Handling

Handle missing data appropriately:

```python
from src.utils import fill_missing_values

# Simple approach: fill missing values with basic strategies
df = fill_missing_values(
    df,
    numeric_strategy='median',
    categorical_strategy='mode',
    date_strategy='nearest'
)

# Group-based approach: fill using values from the same group
df = fill_missing_values(
    df,
    numeric_strategy='median',
    categorical_strategy='mode',
    group_cols=['admin1', 'commodity']
)

# Time series approach: handle missing dates
df = fill_missing_values(
    df,
    date_strategy='forward',  # Forward-fill (last observation carried forward)
    group_cols=['admin1', 'commodity']
)

# Advanced approach: different strategies for different columns
df_filled = df.copy()

# Prices: fill with median by region and commodity
df_filled = fill_missing_values(
    df_filled,
    columns=['price'],
    numeric_strategy='median',
    group_cols=['admin1', 'commodity']
)

# Quantities: forward-fill then backward-fill
df_filled = fill_missing_values(
    df_filled,
    columns=['quantity'],
    date_strategy='forward'
)
df_filled = fill_missing_values(
    df_filled,
    columns=['quantity'],
    date_strategy='backward'
)

# Conflict data: fill with zeros
df_filled = fill_missing_values(
    df_filled,
    columns=['events', 'fatalities'],
    numeric_strategy='zero'
)
```

## Statistical Analysis

### Unit Root Testing

Test for stationarity in time series data:

```python
from src.utils import test_stationarity

# Basic ADF test
adf_result = test_stationarity(
    price_series,
    test='adf',          # Augmented Dickey-Fuller test
    regression='c',      # Include constant
    lags=None,           # Automatic lag selection
    alpha=0.05           # Significance level
)
print(f"Series is stationary: {adf_result['stationary']}")

# Run multiple tests
for series_name, data in series_dict.items():
    for test_type in ['adf', 'kpss', 'pp', 'dfgls', 'za']:
        result = test_stationarity(data, test=test_type)
        status = "stationary" if result['stationary'] else "non-stationary"
        print(f"{series_name} is {status} according to {test_type.upper()} test (p={result['pvalue']:.4f})")

# Test with structural break
za_result = test_stationarity(
    price_series,
    test='za',           # Zivot-Andrews test (accounts for structural breaks)
    regression='ct'      # Include constant and trend
)
if za_result['stationary']:
    print(f"Series is stationary with a break at {za_result['breakpoint']}")
```

### Cointegration Testing

Test for cointegration between price series:

```python
from src.utils import test_cointegration

# Engle-Granger two-step test
eg_result = test_cointegration(
    north_prices,
    south_prices,
    method='engle-granger',
    trend='c',           # Include constant
    lags=None,           # Automatic lag selection
    alpha=0.05           # Significance level
)
print(f"Series are cointegrated: {eg_result['cointegrated']}")

# If cointegrated, print the cointegrating relationship
if eg_result['cointegrated']:
    beta0, beta1 = eg_result['beta']
    print(f"Cointegrating relationship: price1 = {beta0:.4f} + {beta1:.4f} * price2")
    print(f"Half-life of deviations: {calculate_half_life(eg_result['residuals']):.2f} periods")

# Johansen test for multiple series
johansen_result = test_cointegration(
    market_data,          # DataFrame with multiple price series
    method='johansen',
    trend='c',           # Include constant
    lags=4               # Lag order
)
print(f"Number of cointegrating relations: {johansen_result['rank_trace']}")
```

### Threshold Models

Estimate threshold models for market integration analysis:

```python
from src.utils import fit_threshold_vecm, test_linearity

# First test linearity against threshold alternative
linearity_result = test_linearity(
    north_prices,
    south_prices,
    lags=4,
    method='hansen'  # Hansen (1999) test
)
print(f"Linearity rejected: {linearity_result['linearity_rejected']}")

# Estimate threshold VECM if linearity is rejected
if linearity_result['linearity_rejected']:
    tvecm_result = fit_threshold_vecm(
        market_data,     # DataFrame with both price series
        k_ar_diff=4,     # Lag order
        coint_rank=1,    # Cointegration rank
        deterministic='ci'  # Constant inside cointegration relation
    )
    
    # Analyze regime-specific adjustment speeds
    print("Threshold value:", tvecm_result['threshold'])
    print("Adjustment speed (below threshold):", tvecm_result['below_regime']['alpha'])
    print("Adjustment speed (above threshold):", tvecm_result['above_regime']['alpha'])
    
    # Calculate half-lives for each regime
    below_half_life = calculate_half_life(tvecm_result['below_regime']['alpha'][0], regime='threshold')
    above_half_life = calculate_half_life(tvecm_result['above_regime']['alpha'][0], regime='threshold')
    print(f"Half-life below threshold: {below_half_life:.2f} periods")
    print(f"Half-life above threshold: {above_half_life:.2f} periods")
```

### Statistical Utilities

Additional statistical utilities for econometric analysis:

```python
from src.utils import (
    test_granger_causality,
    test_causality_granger,
    bootstrap_confidence_interval,
    test_structural_break
)

# Test for Granger causality
gc_result = test_granger_causality(
    north_prices,
    south_prices,
    max_lags=6
)
print(f"Granger causality detected: {gc_result['causality_detected']}")
print(f"Optimal lag: {gc_result['optimal_lag']}")

# Test bidirectional causality
causality_matrix = test_causality_granger(
    [north_prices, south_prices, center_prices],
    maxlag=4,
    names=['North', 'South', 'Center']
)
print("Causality matrix:")
print(causality_matrix)

# Bootstrap confidence interval for threshold
threshold_ci = bootstrap_confidence_interval(
    market_data,
    statistic_func=estimate_threshold,
    alpha=0.05,
    n_bootstrap=1000
)
print(f"Threshold: {threshold_ci['statistic']:.4f}")
print(f"95% CI: [{threshold_ci['lower_bound']:.4f}, {threshold_ci['upper_bound']:.4f}]")

# Test for structural breaks
break_test = test_structural_break(
    price_series,
    method='quandt',  # Quandt likelihood ratio test
    trim=0.15         # Trimming percentage
)
print(f"Structural break detected: {break_test['significant']}")
if break_test['significant']:
    break_date = price_series.index[break_test['break_date']]
    print(f"Break date: {break_date}")
```

## Spatial Analysis

### GeoDataFrame Operations

Process spatial data for market analysis:

```python
from src.utils import (
    reproject_gdf,
    reproject_geometry,
    create_point_from_coords,
    create_buffer,
    overlay_layers,
    extract_area_of_interest,
    explode_geojson_features
)

# Reproject GeoDataFrame to appropriate CRS for Yemen
markets_gdf = reproject_gdf(
    markets_gdf,
    to_crs=32638  # UTM Zone 38N for Yemen
)

# Create point geometry from coordinates
market_point = create_point_from_coords(
    x=45.325,  # Longitude
    y=15.369,  # Latitude
    crs="EPSG:4326"  # WGS84
)

# Create buffer around markets
buffer_gdf = create_buffer(
    markets_gdf,
    distance=10000,  # 10 km buffer
    unit='meters'
)

# Overlay market buffers with admin regions
market_admins = overlay_layers(
    buffer_gdf,
    admin_gdf,
    how='intersection',
    keep_columns=['admin1', 'population']
)

# Extract data for a specific area
abyan_markets = extract_area_of_interest(
    markets_gdf,
    area_name='abyan',
    area_col='admin1'
)

# Handle multi-part geometries
exploded_gdf = explode_geojson_features(
    'data/admin_boundaries.geojson'
)
```

### Spatial Weight Matrices

Create and use spatial weight matrices for spatial econometrics:

```python
from src.utils import (
    create_spatial_weight_matrix,
    calculate_distances,
    calculate_distance_matrix,
    calculate_market_isolation
)

# Create spatial weight matrix
w = create_spatial_weight_matrix(
    markets_gdf,
    method='knn',          # k-nearest neighbors
    k=5,                   # Number of neighbors
    conflict_col='conflict_intensity',  # Adjust weights for conflict
    conflict_weight=0.5    # Weight of conflict in adjustment
)

# Calculate distances between all markets
distance_matrix = calculate_distance_matrix(
    markets_gdf,
    id_col='market_id',
    method='euclidean',
    crs=32638  # UTM Zone 38N
)

# Calculate market isolation index
isolation_gdf = calculate_market_isolation(
    markets_gdf,
    transport_network_gdf=roads_gdf,
    population_gdf=population_gdf,
    conflict_col='conflict_intensity',
    max_distance=50000  # 50 km
)
```

### Distance Calculations

Calculate and analyze distances between markets:

```python
from src.utils import (
    find_nearest_points,
    calculate_distances,
    aggregate_points_to_polygons
)

# Find nearest facility for each market
nearest_facilities = find_nearest_points(
    markets_gdf,
    facilities_gdf,
    target_col='facility_type',
    max_distance=100000  # 100 km
)
print(f"Average distance to nearest facility: {nearest_facilities['distance'].mean():.2f} meters")

# Calculate all-pairs distances
all_distances = calculate_distances(
    markets_gdf,
    markets_gdf,
    'market_id',
    'market_id'
)

# Aggregate market data to admin regions
admin_summary = aggregate_points_to_polygons(
    markets_gdf,
    admin_gdf,
    value_col='price',
    agg_func='mean',
    polygon_id_col='admin1'
)
```

### Market Catchment Analysis

Analyze market catchments and accessibility:

```python
from src.utils import (
    compute_accessibility_index,
    create_market_catchments,
    create_exchange_regime_boundaries,
    assign_exchange_rate_regime
)

# Compute market accessibility index
accessibility_gdf = compute_accessibility_index(
    markets_gdf,
    population_gdf,
    max_distance=50000,  # 50 km
    distance_decay=2.0,  # Square of distance
    weight_col='population'
)

# Create market catchment areas
catchments_gdf = create_market_catchments(
    markets_gdf,
    population_gdf,
    market_id_col='market_id',
    population_weight_col='population',
    max_distance=100000,  # 100 km
    distance_decay=2.0
)

# Create exchange rate regime boundaries
regime_boundaries = create_exchange_regime_boundaries(
    admin_gdf,
    regime_col='exchange_rate_regime',
    dissolve=True,
    simplify_tolerance=100  # Simplify geometries by 100 meters
)

# Assign exchange rate regime to markets
markets_with_regime = assign_exchange_rate_regime(
    markets_gdf,
    regime_polygons_gdf=regime_boundaries,
    regime_col='exchange_rate_regime',
    default_regime=None  # Markets outside boundaries will have no regime
)
```

## Visualization

### Time Series Visualization

Create time series visualizations for market analysis:

```python
from src.utils import (
    set_plotting_style,
    plot_time_series,
    plot_multiple_time_series,
    plot_time_series_by_group,
    plot_dual_axis,
    format_date_axis,
    format_currency_axis,
    save_plot
)

# Set consistent plotting style
set_plotting_style()

# Plot price time series for a single commodity
fig, ax = plot_time_series(
    df,
    x='date',
    y='price',
    title='Wheat Prices Over Time',
    xlabel='Date',
    ylabel='Price (YER)',
    date_format='%Y-%m',
    interval='month',
    marker='o',
    linestyle='-',
    color='blue'
)
format_currency_axis(ax, axis='y', symbol='YER')
save_plot(fig, 'figures/wheat_prices.png', dpi=300)

# Plot multiple commodities
fig, ax = plot_multiple_time_series(
    df,
    x='date',
    y_columns=['wheat_price', 'rice_price', 'beans_price'],
    labels=['Wheat', 'Rice', 'Beans'],
    title='Commodity Prices Comparison',
    xlabel='Date',
    ylabel='Price (YER)'
)

# Plot prices by exchange rate regime
fig, ax = plot_time_series_by_group(
    df,
    x='date',
    y='price',
    group='exchange_rate_regime',
    title='Prices by Exchange Rate Regime',
    xlabel='Date',
    ylabel='Price (YER)',
    palette=['blue', 'red']  # Blue for north, red for south
)

# Plot price and conflict on dual axes
fig, (ax1, ax2) = plot_dual_axis(
    df,
    x='date',
    y1='price',
    y2='conflict_intensity',
    color1='blue',
    color2='red',
    label1='Price',
    label2='Conflict Intensity',
    title='Price and Conflict Over Time',
    xlabel='Date',
    ylabel1='Price (YER)',
    ylabel2='Conflict Intensity'
)
```

### Spatial Visualization

Create maps and spatial visualizations:

```python
from src.utils import (
    plot_static_map,
    create_interactive_map,
    plot_price_heatmap
)

# Create static map
fig = plot_static_map(
    markets_gdf,
    column='price',
    cmap='viridis',
    figsize=(12, 10),
    title='Market Prices Across Yemen',
    add_basemap=True,
    scheme='quantiles',
    k=5,
    legend=True
)
save_plot(fig, 'figures/market_prices_map.png', dpi=300)

# Create interactive map
m = create_interactive_map(
    markets_gdf,
    column='price',
    popup_cols=['market_name', 'admin1', 'price', 'date'],
    title='Interactive Market Prices Map',
    tiles='OpenStreetMap'
)
m.save('figures/interactive_map.html')

# Create price heatmap
fig = plot_price_heatmap(
    markets_gdf,
    commodity='beans',
    date='2022-06-01',
    price_col='price',
    cmap='YlOrRd',
    title='Bean Prices - June 2022'
)
```

### Advanced Plot Types

Create specialized visualizations for in-depth analysis:

```python
from src.utils import (
    plot_heatmap,
    plot_scatter,
    plot_histogram,
    plot_bar_chart,
    plot_boxplot,
    plot_stacked_bar,
    add_annotations
)

# Create correlation heatmap
fig, ax = plot_heatmap(
    df.corr(),
    cmap='coolwarm',
    title='Correlation Matrix',
    annot=True,
    fmt='.2f',
    linewidths=0.5,
    mask_upper=True,
    center=0
)

# Create scatter plot with regression line
fig, ax = plot_scatter(
    df,
    x='north_price',
    y='south_price',
    title='North vs South Prices',
    xlabel='North Price (YER)',
    ylabel='South Price (YER)',
    alpha=0.7,
    fit_line=True,
    fit_order=1,
    size='market_size',
    size_scale=100,
    annotate='market_name'
)

# Create price distribution histogram
fig, ax = plot_histogram(
    df,
    column='price',
    bins=20,
    title='Price Distribution',
    xlabel='Price (YER)',
    ylabel='Frequency',
    vertical_line=df['price'].median(),
    vertical_line_label='Median'
)

# Create bar chart of prices by region
fig, ax = plot_bar_chart(
    df.groupby('admin1')['price'].mean().reset_index(),
    x='admin1',
    y='price',
    title='Average Prices by Region',
    xlabel='Region',
    ylabel='Average Price (YER)',
    sort_values=True,
    rotation=45
)

# Create box plot of prices by commodity
fig, ax = plot_boxplot(
    df,
    column='price',
    by='commodity',
    title='Price Distribution by Commodity',
    xlabel='Commodity',
    ylabel='Price (YER)',
    showfliers=True
)

# Create stacked bar chart of market types by region
fig, ax = plot_stacked_bar(
    df_counts,
    x='admin1',
    y_columns=['small_market', 'medium_market', 'large_market'],
    title='Market Types by Region',
    xlabel='Region',
    ylabel='Number of Markets',
    rotation=45,
    legend_title='Market Type'
)

# Add annotations to a plot
annotations = {
    (x1, y1): "Price spike due to conflict",
    (x2, y2): "Exchange rate policy change"
}
add_annotations(
    ax,
    annotations,
    fontsize=9,
    alpha=0.8,
    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 3}
)
```

### Publication-Ready Graphics

Prepare plots for publication:

```python
from src.utils import (
    configure_axes_for_print,
    plotting_context,
    save_plot
)

# Use context manager for temporary style changes
with plotting_context(style='whitegrid', context='paper', font_scale=1.2):
    fig, ax = plot_time_series(
        df,
        x='date',
        y='price',
        title='Wheat Prices (2020-2023)',
        xlabel='Date',
        ylabel='Price (YER)'
    )
    
    # Configure for publication
    configure_axes_for_print(
        ax,
        fontsize_title=14,
        fontsize_labels=12,
        fontsize_ticks=10,
        linewidth=1.0
    )
    
    # Save in publication quality
    save_plot(
        fig,
        'figures/wheat_prices_publication.png',
        dpi=600,
        bbox_inches='tight',
        transparent=False,
        facecolor='white'
    )
    
    # Also save as vector format for journal submission
    save_plot(
        fig,
        'figures/wheat_prices_publication.pdf',
        bbox_inches='tight'
    )
```

## Testing and Quality Assurance

### Unit Testing

Write unit tests for your modules:

```python
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.utils import DataError

class TestDataLoader(unittest.TestCase):
    """Tests for the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader('./data')
        
        # Create test data
        self.test_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'price': [100, 110, 120, 130, 140],
            'admin1': ['abyan'] * 5,
            'commodity': ['wheat'] * 5
        })
    
    @patch('src.utils.read_csv')
    def test_load_csv(self, mock_read_csv):
        """Test loading CSV file."""
        # Configure mock
        mock_read_csv.return_value = self.test_data
        
        # Call method
        result = self.loader.load_csv('test.csv')
        
        # Verify mock was called
        mock_read_csv.assert_called_once_with('data/raw/test.csv', parse_dates=['date'])
        
        # Verify result
        self.assertEqual(len(result), 5)
        
    def test_split_by_exchange_regime(self):
        """Test splitting data by exchange rate regime."""
        # Add exchange rate regime column
        df = self.test_data.copy()
        df['exchange_rate_regime'] = ['north', 'north', 'south', 'south', 'north']
        
        # Call method
        north, south = self.loader.split_by_exchange_regime(df)
        
        # Verify results
        self.assertEqual(len(north), 3)
        self.assertEqual(len(south), 2)
        
    def test_get_commodity_list(self):
        """Test getting commodity list."""
        # Add another commodity
        df = pd.concat([
            self.test_data,
            pd.DataFrame({
                'date': pd.date_range('2020-01-01', periods=3),
                'price': [200, 210, 220],
                'admin1': ['aden'] * 3,
                'commodity': ['rice'] * 3
            })
        ])
        
        # Call method
        commodities = self.loader.get_commodity_list(df)
        
        # Verify results
        self.assertEqual(len(commodities), 2)
        self.assertIn('wheat', commodities)
        self.assertIn('rice', commodities)
```

### Integration Testing

Write integration tests to verify component interactions:

```python
import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from src.data import load_market_data, preprocess_data, calculate_price_differentials

class TestDataPipeline(unittest.TestCase):
    """Integration tests for the data pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary CSV file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = os.path.join(self.temp_dir.name, 'test_data.csv')
        
        # Create test data
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'price': np.linspace(100, 200, 10),
            'admin1': ['abyan', 'aden'] * 5,
            'commodity': ['wheat'] * 10,
            'exchange_rate_regime': ['north', 'south'] * 5
        })
        
        # Save to CSV
        df.to_csv(self.temp_file, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_full_pipeline(self):
        """Test the full data processing pipeline."""
        # Load data
        df = load_market_data(self.temp_file)
        self.assertEqual(len(df), 10)
        
        # Preprocess data
        processed_df = preprocess_data(df)
        self.assertIn('price_log', processed_df.columns)
        self.assertIn('year', processed_df.columns)
        
        # Calculate price differentials
        differentials = calculate_price_differentials(processed_df)
        self.assertGreater(len(differentials), 0)
        self.assertIn('north_price', differentials.columns)
        self.assertIn('south_price', differentials.columns)
        self.assertIn('price_diff', differentials.columns)
```

### Performance Testing

Test performance characteristics of your code:

```python
import unittest
import pandas as pd
import numpy as np
import time
from src.utils import parallelize_dataframe, m1_optimized

class TestPerformance(unittest.TestCase):
    """Performance tests."""
    
    def test_parallelization_speedup(self):
        """Test that parallelization improves performance."""
        # Create a large DataFrame
        n_rows = 1000000
        df = pd.DataFrame({
            'value': np.random.rand(n_rows),
            'group': np.random.choice(['A', 'B', 'C', 'D'], n_rows)
        })
        
        # Define processing function
        def process_chunk(chunk):
            # Simulate some work
            result = chunk.copy()
            result['transformed'] = np.sqrt(chunk['value']) * 10
            # Group operations are often slow
            result['group_mean'] = chunk.groupby('group')['value'].transform('mean')
            return result
        
        # Time sequential processing
        start_time = time.time()
        sequential_result = process_chunk(df)
        sequential_time = time.time() - start_time
        
        # Time parallel processing
        start_time = time.time()
        parallel_result = parallelize_dataframe(df, process_chunk, n_workers=4)
        parallel_time = time.time() - start_time
        
        # Verify results are the same
        pd.testing.assert_frame_equal(sequential_result, parallel_result)
        
        # Verify speedup
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup: {sequential_time / parallel_time:.2f}x")
        
        # Should be faster with parallelization
        self.assertLess(parallel_time, sequential_time)
```

## Troubleshooting

### Common Issues

**1. Module Import Errors**

Problem: `ImportError: cannot import name 'X' from 'src.utils'`

Solution:

- Check that the function is properly exported in `src/utils/__init__.py`
- Make sure you're not creating circular imports
- If importing from a specific sub-module, use `from src.utils.submodule import X`

**2. Memory Issues with Large Data**

Problem: `MemoryError` when processing large files

Solution:

- Use chunked reading: `read_large_csv_chunks`, `read_large_geojson_chunks`
- Use `optimize_dataframe` to reduce memory usage
- Process data in parallel with `parallelize_dataframe`
- Use the `disk_cache` decorator for intermediate results

**3. Performance Issues**

Problem: Functions are running slowly

Solution:

- Apply the `@m1_optimized` decorator to compute-intensive functions
- Use vectorized operations instead of loops where possible
- Use the `@timer` decorator to identify bottlenecks
- Consider using `memoize` for repeated calculations
- Ensure you're using the latest version of libraries optimized for M1/M2

**4. Unexpected Data Types**

Problem: Functions failing due to unexpected data types

Solution:

- Always validate inputs with `validate_dataframe` or `validate_geodataframe`
- Convert date columns explicitly with `convert_dates`
- Use `raise_if_invalid` to catch issues early

### Debugging Strategies

**1. Enable Debug Logging**

```python
from src.utils import setup_logging
import logging

# Temporarily increase log level
logger = setup_logging(log_level=logging.DEBUG)
```

**2. Add Debug Decorators**

```python
from src.utils import log_calls, timer

@log_calls(level=logging.DEBUG)  # Log all function calls with arguments
@timer  # Time function execution
def problematic_function(arg1, arg2):
    # Implementation
```

**3. Inspect Data at Each Step**

```python
# Add debug logging inside functions
logger.debug(f"DataFrame shape: {df.shape}")
logger.debug(f"DataFrame columns: {df.columns.tolist()}")
logger.debug(f"First few rows:\n{df.head()}")
logger.debug(f"Column types:\n{df.dtypes}")
```

**4. Use Memory Profiling**

```python
from src.utils import memory_usage_decorator

@memory_usage_decorator
def memory_intensive_function(large_data):
    # Implementation
```

### Support Resources

If you encounter persistent issues:

1. Check the project documentation in the `docs/` directory
2. Review the tests in the `tests/` directory for usage examples
3. Consult the source code of the utility functions for detailed behavior
4. Contact the project maintainers via the issue tracker

## Quick Reference

### Key Functions Summary

**Error Handling**

- `handle_errors`: Decorator for consistent error handling
- `raise_if_invalid`: Raise exception if validation fails
- Error types: `DataError`, `ModelError`, `ValidationError`

**Configuration**

- `initialize_config`: Set up configuration
- `config.get`: Access configuration values

**Logging**

- `setup_logging`: Initialize logging system
- `get_logger_with_context`: Logger with context metadata

**Performance**

- `m1_optimized`: Optimize for M1/M2 Mac
- `parallelize_dataframe`: Process DataFrame in parallel
- `optimize_dataframe`: Reduce memory usage
- `memoize`, `disk_cache`: Cache results

**Data Validation**

- `validate_dataframe`: Validate DataFrame properties
- `validate_geodataframe`: Validate GeoDataFrame properties
- `validate_time_series`: Validate time series properties

**File Operations**

- `read_csv`, `write_csv`: Read/write CSV files
- `read_geojson`, `write_geojson`: Read/write GeoJSON files
- `read_large_csv_chunks`: Process large CSV files

**Data Processing**

- `clean_column_names`: Standardize column names
- `convert_dates`: Convert string dates to datetime
- `fill_missing_values`: Handle missing data
- `create_lag_features`: Generate lag features

**Statistical Analysis**

- `test_stationarity`: Test for unit roots
- `test_cointegration`: Test for cointegration
- `fit_threshold_vecm`: Estimate threshold VECM

**Spatial Analysis**

- `reproject_gdf`: Reproject GeoDataFrame
- `calculate_distances`: Calculate distance matrix
- `create_spatial_weight_matrix`: Create weight matrix

**Visualization**

- `plot_time_series`: Plot time series data
- `plot_multiple_time_series`: Plot multiple series
- `plot_static_map`: Create static map
- `create_interactive_map`: Create interactive map

### Common Patterns

**Standard Module Template**

```python
"""
Module description and purpose.
"""
import logging
from typing import Dict, List, Optional, Union

# Import utilities in logical groups
from src.utils import (
    # Error handling
    handle_errors, DataError,
    
    # Data validation
    validate_dataframe, raise_if_invalid,
    
    # File operations
    read_csv, write_csv
)

# Initialize module logger
logger = logging.getLogger(__name__)

@handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
def function_name(arg1, arg2):
    """
    Function docstring.
    
    Parameters
    ----------
    arg1 : type
        Description
    arg2 : type
        Description
        
    Returns
    -------
    type
        Description
    """
    # Implementation
    pass
```

**Standard Data Processing Pattern**

```python
# 1. Load data with validation
data = read_csv('input.csv')
valid, errors = validate_dataframe(data, required_columns=['date', 'price'])
raise_if_invalid(valid, errors, "Invalid input data")

# 2. Clean and prepare data
data = clean_column_names(data)
data = convert_dates(data, date_cols=['date'])
data = fill_missing_values(data, numeric_strategy='median')

# 3. Create features
data = create_lag_features(data, cols=['price'], lags=[1, 2, 3])

# 4. Analyze data
results = calculate_statistics(data)

# 5. Visualize results
fig, ax = plot_time_series(data, x='date', y='price')
save_plot(fig, 'output.png')

# 6. Save results
write_csv(results, 'results.csv')
```

**Standard Statistical Analysis Pattern**

```python
# 1. Test for stationarity
stationary_result = test_stationarity(price_series)
if not stationary_result['stationary']:
    # Take first difference
    price_diff = price_series.diff(1).dropna()
    
# 2. Test for cointegration
cointegration_result = test_cointegration(north_prices, south_prices)
if cointegration_result['cointegrated']:
    # Test for threshold effects
    linearity_result = test_linearity(north_prices, south_prices)
    if linearity_result['linearity_rejected']:
        # Estimate threshold model
        tvecm_result = fit_threshold_vecm(market_data)
```

**Standard Spatial Analysis Pattern**

```python
# 1. Prepare spatial data
markets_gdf = reproject_gdf(markets_gdf, to_crs=32638)

# 2. Calculate spatial relationships
distances = calculate_distances(markets_gdf, markets_gdf)
w = create_spatial_weight_matrix(markets_gdf, method='knn', k=5)

# 3. Analyze spatial patterns
isolation = calculate_market_isolation(markets_gdf)

# 4. Visualize spatial data
map_fig = plot_static_map(markets_gdf, column='price')
interactive_map = create_interactive_map(markets_gdf)
```
