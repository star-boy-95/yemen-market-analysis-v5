# Yemen Market Integration Utilities Guide

This guide provides instructions and best practices for using the utility modules in the Yemen Market Integration project. These utilities are optimized for M1/M2 MacBook Pro hardware and designed to handle the specific requirements of econometric analysis for Yemen's market integration.

## Table of Contents
- [Getting Started](#getting-started)
- [Utility Modules Overview](#utility-modules-overview)
- [Error Handling](#error-handling)
- [Configuration Management](#configuration-management)
- [Logging](#logging)
- [Performance Optimization](#performance-optimization)
- [Data Validation](#data-validation)
- [File Operations](#file-operations)
- [Data Processing](#data-processing)
- [Statistical Analysis](#statistical-analysis)
- [Spatial Analysis](#spatial-analysis)
- [Visualization](#visualization)

## Getting Started

To set up the project environment with standard configuration and logging:

```python
from src.utils import setup_project_environment

# Initialize project environment
config, logger = setup_project_environment(
    config_file='config/settings.yaml',
    log_dir='logs'
)

# Log system information
logger.info(f"Starting analysis with config: {config.get('analysis_settings')}")
```

To see the best practices guide:

```python
from src.utils import show_best_practices

show_best_practices()
```

## Utility Modules Overview

The utilities package provides the following modules:

- **error_handler**: Centralized error handling with custom exceptions
- **config**: Configuration management with environment variables support
- **logging_setup**: Comprehensive logging with context-aware adapters
- **decorators**: Performance-focused decorators (caching, timing, M1 optimization)
- **validation**: Data validation for DataFrames and spatial data
- **file_utils**: Optimized file operations with chunked processing
- **plotting_utils**: Data visualization utilities with sensible defaults
- **data_utils**: Data manipulation with vectorized operations
- **stats_utils**: Statistical analysis with threshold cointegration tests
- **spatial_utils**: GIS operations optimized for geospatial analysis
- **performance_utils**: M1-specific optimizations and parallel processing

## Error Handling

Use the `handle_errors` decorator for consistent error handling across the application:

```python
from src.utils import handle_errors, DataError
import logging

logger = logging.getLogger(__name__)

@handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
def load_market_data(file_path):
    # Function implementation
    # If an error occurs, it will be logged and handled consistently
```

Custom exceptions for specific error types:

```python
from src.utils import DataError, ModelError, ValidationError

# Raise specific exceptions for better error tracking
if not file_exists:
    raise DataError(f"Required data file not found: {file_path}")

if model_failed:
    raise ModelError("Failed to converge threshold model")
```

## Configuration Management

Initialize and access configuration from anywhere in the application:

```python
from src.utils import config, initialize_config

# Initialize at the start of your application
initialize_config(
    config_file='config.yaml',
    env_prefix="YEMEN_",
    defaults={
        'analysis': {
            'max_lags': 4,
            'alpha': 0.05
        }
    }
)

# Access configuration anywhere in your code
max_lags = config.get('analysis.max_lags')
alpha = config.get('analysis.alpha')
```

## Logging

Set up logging once at application start:

```python
from src.utils import setup_logging

# Initialize logging
logger = setup_logging(
    log_dir='logs',
    log_level=logging.INFO,
    log_file='yemen_analysis.log',
    rotation='daily'
)
```

Get module-specific loggers with context:

```python
from src.utils import get_logger_with_context
import logging

# Simple logger
logger = logging.getLogger(__name__)

# Logger with context
context_logger = get_logger_with_context(__name__, {
    'region': 'north', 
    'commodity': 'wheat'
})

context_logger.info("Processing market data")  # Will include region and commodity in log
```

## Performance Optimization

Optimize functions for M1/M2 Mac hardware:

```python
from src.utils import m1_optimized

@m1_optimized(use_numba=True, parallel=True)
def compute_intensive_calculation(data):
    # Computationally intensive function
    # Will be optimized for M1/M2 Mac if available
```

Process large DataFrames in parallel:

```python
from src.utils import parallelize_dataframe

def process_chunk(df_chunk):
    # Process a chunk of the DataFrame
    return processed_chunk

# Process large DataFrame in parallel
result = parallelize_dataframe(
    large_df, 
    process_chunk, 
    n_workers=4
)
```

Optimize memory usage:

```python
from src.utils import optimize_dataframe

# Reduce memory usage by choosing appropriate dtypes
optimized_df = optimize_dataframe(df)
```

## Data Validation

Validate inputs early in your process:

```python
from src.utils import validate_dataframe, validate_geodataframe, raise_if_invalid

# Validate regular DataFrame
valid, errors = validate_dataframe(
    df, 
    required_columns=['date', 'price', 'region'],
    column_types={'price': float, 'date': pd.Timestamp}
)
raise_if_invalid(valid, errors, "Invalid market data")

# Validate GeoDataFrame
valid, errors = validate_geodataframe(
    gdf,
    crs="EPSG:32638",  # UTM Zone 38N for Yemen
    geometry_type="Point"
)
raise_if_invalid(valid, errors, "Invalid spatial data")
```

## File Operations

Safely read and write files with optimized error handling:

```python
from src.utils import read_csv, write_csv, read_geojson, write_geojson

# Read CSV with improved error handling
df = read_csv('data/market_prices.csv', parse_dates=['date'])

# Write results safely
write_csv(results_df, 'results/market_analysis.csv', index=False)

# Read GeoJSON data
admin_regions = read_geojson('data/admin_boundaries.geojson')

# Write spatial data
write_geojson(market_locations, 'results/market_locations.geojson')
```

Process large files in chunks:

```python
from src.utils.performance_utils import chunked_file_reader

# Process a large CSV file in chunks
for chunk in chunked_file_reader('large_file.csv', chunk_size=100000):
    # Process each chunk
    processed_chunk = process_data(chunk)
    # Append results or process further
```

## Data Processing

Clean and prepare data:

```python
from src.utils import (
    clean_column_names, 
    convert_dates, 
    fill_missing_values,
    create_lag_features
)

# Clean column names
df = clean_column_names(df)

# Convert date columns
df = convert_dates(df, date_cols=['date', 'report_date'])

# Fill missing values
df = fill_missing_values(
    df,
    numeric_strategy='median',
    group_cols=['region', 'commodity']
)

# Create lag features for time series analysis
df = create_lag_features(
    df, 
    cols=['price'], 
    lags=[1, 2, 3], 
    group_cols=['region', 'commodity']
)
```

## Statistical Analysis

Perform statistical tests for market integration analysis:

```python
from src.utils import (
    test_stationarity,
    test_cointegration,
    test_granger_causality,
    fit_threshold_vecm
)

# Test for stationarity
result = test_stationarity(price_series, test='adf', regression='ct')
print(f"Is price series stationary? {result['stationary']}")

# Test for cointegration between two price series
coint_result = test_cointegration(
    north_prices, 
    south_prices, 
    method='engle-granger'
)
print(f"Are prices cointegrated? {coint_result['cointegrated']}")

# Test for Granger causality
causality = test_granger_causality(
    north_prices, 
    south_prices, 
    maxlag=4
)
print(f"Causality detected: {causality['causality_detected']}")

# Fit threshold VECM model
tvecm_result = fit_threshold_vecm(
    market_data,
    k_ar_diff=2,
    coint_rank=1
)
```

## Spatial Analysis

Perform spatial operations for market analysis:

```python
from src.utils import (
    reproject_gdf,
    calculate_distances,
    create_spatial_weight_matrix,
    assign_exchange_rate_regime
)

# Reproject GeoDataFrame to appropriate CRS for Yemen
markets_gdf = reproject_gdf(markets_gdf, to_crs=32638)  # UTM Zone 38N

# Calculate distances between markets
distance_matrix = calculate_distances(
    markets_gdf, 
    markets_gdf,
    'market_id', 
    'market_id'
)

# Create spatial weight matrix with conflict adjustment
w = create_spatial_weight_matrix(
    markets_gdf,
    method='knn',
    k=5,
    conflict_col='conflict_intensity'
)

# Assign exchange rate regime based on location
markets_gdf = assign_exchange_rate_regime(
    markets_gdf,
    regime_polygons_gdf,
    regime_col='exchange_rate_regime'
)
```

## Visualization

Create standardized visualizations:

```python
from src.utils import (
    set_plotting_style,
    plot_time_series,
    plot_multiple_time_series,
    plot_time_series_by_group,
    save_plot
)

# Set consistent plot style
set_plotting_style()

# Plot single time series
fig, ax = plot_time_series(
    df, 
    x='date', 
    y='price',
    title='Wheat Prices Over Time',
    ylabel='Price (YER)'
)

# Plot multiple commodities
fig, ax = plot_multiple_time_series(
    df,
    x='date',
    y_columns=['wheat_price', 'rice_price', 'beans_price'],
    labels=['Wheat', 'Rice', 'Beans']
)

# Plot prices by exchange rate regime
fig, ax = plot_time_series_by_group(
    df,
    x='date',
    y='price',
    group='exchange_rate_regime',
    title='Prices by Exchange Rate Regime'
)

# Save plot with consistent settings
save_plot(fig, 'figures/price_analysis.png', dpi=300)
```