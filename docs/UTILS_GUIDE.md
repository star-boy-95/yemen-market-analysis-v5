# Yemen Market Integration Project: Utilities Guide

This guide documents the utility modules available in the `src.utils` package, designed to streamline development for the Yemen Market Integration Project with a focus on performance, error handling, and consistent patterns.

## Core Principles

1. **Error Handling**: All functions use centralized error handling
2. **Performance Optimization**: Specialized for M1 Mac architecture
3. **Memory Efficiency**: Tools for working with large datasets
4. **Validation**: Input validation throughout the pipeline
5. **Logging**: Consistent logging patterns

## Quick Reference

```python
# Import common utilities at package level
from src.utils import (
    # Configuration
    config, initialize_config,
    
    # Error handling
    handle_errors, DataError, ModelError,
    
    # Validation
    validate_dataframe, validate_geodataframe, raise_if_invalid,
    
    # File operations
    read_csv, write_csv, read_geojson, write_geojson,
    
    # Data processing
    clean_column_names, normalize_columns, compute_price_differentials,
    
    # Performance
    configure_system_for_performance, parallelize_dataframe,
    
    # Spatial
    reproject_gdf, calculate_distances, assign_exchange_rate_regime,
    
    # Statistical
    test_stationarity, test_cointegration, fit_threshold_vecm
)
```

## Module: `error_handler`

Provides centralized error handling through decorators and custom exception types.

```python
from src.utils.error_handler import handle_errors, DataError

# Apply to functions that might raise exceptions
@handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
def process_file(file_path):
    # Implementation...
    
# Raise custom exceptions for better error categorization
raise DataError("Missing price column in market data")
```

### Key Components:

- **Custom Exception Types**: `MarketIntegrationError`, `DataError`, `ModelError`, `ValidationError`, `ConfigError`
- **`handle_errors` Decorator**: Catches exceptions, logs with context, and optionally remaps to custom types
- **`capture_error` Function**: Manual error capture outside decorator context

## Module: `validation`

Tools for validating data inputs and parameters.

```python
from src.utils.validation import validate_dataframe, raise_if_invalid

# Validate DataFrame structure
valid, errors = validate_dataframe(
    df, 
    required_columns=['date', 'price', 'market'],
    column_types={'date': pd.Timestamp, 'price': float},
    min_rows=10
)

# Abort processing if validation fails
raise_if_invalid(valid, errors, "Market data validation failed")
```

### Key Functions:

- **`validate_dataframe`**: Checks column presence, types, and null values
- **`validate_geodataframe`**: Additional checks for spatial data
- **`validate_time_series`**: Tests for stationarity, length, and null values
- **`validate_model_inputs`**: Verifies parameters for statistical models

## Module: `file_utils`

Enhanced file operations with consistent error handling and path management.

```python
from src.utils.file_utils import read_csv, write_json, ensure_dir

# Read with enhanced error handling
df = read_csv('market_prices.csv')

# Safe file operations
output_dir = ensure_dir('results/models')
write_json(model_results, output_dir / 'model_output.json')

# Atomic file writing to prevent corruption
with AtomicFileWriter('important_results.csv', 'w') as f:
    df.to_csv(f)
```

### Key Functions:

- **File Operations**: `read_csv`, `write_csv`, `read_json`, `write_json`, `read_geojson`, `write_geojson`
- **Path Management**: `ensure_dir`, `delete_file`, `move_file`, `copy_file`
- **Large Files**: `read_large_csv_chunks`, `read_large_geojson_chunks`
- **Safety Features**: `create_backup`, `file_hash`, `AtomicFileWriter`

## Module: `config`

Configuration management with a singleton pattern for global access.

```python
from src.utils import config, initialize_config

# Initialize from multiple sources
initialize_config(
    config_file='settings.yaml',
    env_prefix='YEMEN_',
    defaults={'analysis': {'alpha': 0.05}}
)

# Access configuration values
alpha = config.get('analysis.alpha', 0.05)
market_file = config.get('data.market_file')

# Get a section
model_params = config.get_section('models')
```

### Key Features:

- **Multiple Sources**: Load from files, environment variables, and code
- **Hierarchical Access**: Use dot notation for nested configuration
- **Type Conversion**: Automatically parse types from environment variables
- **Defaults**: Supply defaults for missing configuration

## Module: `decorators`

Utility decorators for common patterns.

```python
from src.utils.decorators import timer, m1_optimized, disk_cache

# Time function execution
@timer
def process_large_dataset():
    # Implementation...

# Optimize for M1 Mac
@m1_optimized(use_numba=True, parallel=True)
def compute_intensive_calculation(data):
    # Implementation...

# Cache results to disk
@disk_cache(cache_dir='.cache', expiration_seconds=3600)
def fetch_external_data(url):
    # Implementation...
```

### Key Decorators:

- **Performance**: `timer`, `m1_optimized`, `memoize`, `disk_cache`
- **Reliability**: `retry`, `rate_limited`
- **Debugging**: `log_calls`, `deprecated`
- **Validation**: `validate_args`

## Module: `performance_utils`

Utilities for optimizing performance, especially on M1 hardware.

```python
from src.utils.performance_utils import (
    configure_system_for_performance, 
    parallelize_dataframe,
    optimize_dataframe
)

# Configure system before processing
configure_system_for_performance()

# Process DataFrame in parallel
def process_chunk(chunk):
    # Process a DataFrame chunk
    return processed_chunk

result_df = parallelize_dataframe(large_df, process_chunk, n_workers=4)

# Reduce memory usage
optimized_df = optimize_dataframe(df, downcast=True, category_min_size=50)
```

### Key Functions:

- **`configure_system_for_performance`**: Optimizes NumPy and other libraries for M1
- **`parallelize_dataframe`**: Applies a function to DataFrame chunks in parallel
- **`optimize_dataframe`**: Reduces memory usage by adjusting datatypes
- **`get_system_info`**: Reports available hardware and memory
- **`memory_usage_decorator`**: Tracks memory usage of functions

## Module: `data_utils`

Tools for data cleaning, transformation, and feature engineering.

```python
from src.utils.data_utils import (
    clean_column_names,
    convert_dates,
    fill_missing_values,
    create_lag_features
)

# Clean and prepare data
df = clean_column_names(df)
df = convert_dates(df, date_columns=['date'])
df = fill_missing_values(df, numeric_strategy='median', group_columns=['region'])

# Create features for time series analysis
df = create_lag_features(df, columns=['price'], lags=[1, 2, 3], group_columns=['market'])
```

### Key Functions:

- **Cleaning**: `clean_column_names`, `fill_missing_values`, `detect_outliers`
- **Transformation**: `normalize_columns`, `convert_exchange_rates`, `winsorize_columns`
- **Time Series**: `create_lag_features`, `create_rolling_features`, `calculate_price_changes`
- **Aggregation**: `aggregate_time_series`, `pivot_data`, `unpivot_data`

## Module: `stats_utils`

Statistical tests and econometric models specific to market integration analysis.

```python
from src.utils.stats_utils import (
    test_stationarity,
    test_cointegration,
    fit_threshold_vecm
)

# Test time series properties
stationarity = test_stationarity(price_series, test='adf')
print(f"Series is stationary: {stationarity['stationary']}")

# Test for market integration
result = test_cointegration(
    north_prices, 
    south_prices, 
    method='engle-granger'
)
print(f"Markets are cointegrated: {result['cointegrated']}")

# Fit threshold model for non-linear price transmission
model = fit_threshold_vecm(
    price_data,
    k_ar_diff=2,
    coint_rank=1
)
```

### Key Functions:

- **Tests**: `test_stationarity`, `test_cointegration`, `test_causality_granger`, `test_linearity`
- **Models**: `fit_var_model`, `fit_vecm_model`, `fit_threshold_vecm`, `estimate_threshold_tar`
- **Diagnostics**: `test_white_noise`, `test_autocorrelation`, `compute_variance_ratio`
- **Bootstrap**: `bootstrap_confidence_interval`, `calculate_threshold_ci`

## Module: `spatial_utils`

Utilities for spatial operations and GIS analysis.

```python
from src.utils.spatial_utils import (
    reproject_gdf,
    calculate_distances,
    assign_exchange_rate_regime
)

# Ensure consistent coordinate system
gdf = reproject_gdf(gdf, to_crs=32638)  # UTM Zone 38N for Yemen

# Calculate distance matrix between markets
distance_matrix = calculate_distances(
    origin_gdf=markets_gdf,
    destination_gdf=markets_gdf,
    origin_id_col='market_id',
    dest_id_col='market_id'
)

# Assign exchange rate regimes based on location
markets_with_regimes = assign_exchange_rate_regime(
    markets_gdf,
    regime_polygons_gdf,
    regime_col='exchange_rate_regime'
)
```

### Key Functions:

- **Transformations**: `reproject_gdf`, `create_buffer`, `reproject_geometry`
- **Analysis**: `find_nearest_points`, `overlay_layers`, `calculate_distances`
- **Market Integration**: `calculate_market_isolation`, `create_market_catchments`, `assign_exchange_rate_regime`
- **Weights**: `create_spatial_weight_matrix`, `create_exchange_regime_boundaries`

## Module: `plotting_utils`

Specialized visualization tools for market integration analysis.

```python
from src.utils.plotting_utils import (
    plot_time_series_by_group,
    plot_dual_axis,
    set_plotting_style
)

# Set consistent styling
set_plotting_style()

# Plot time series by exchange rate regime
fig, ax = plot_time_series_by_group(
    df,
    x='date',
    y='price',
    group='exchange_rate_regime',
    title='Price Trends by Exchange Rate Regime'
)

# Plot prices and exchange rates
fig, (ax1, ax2) = plot_dual_axis(
    df,
    x='date',
    y1='price',
    y2='exchange_rate',
    title='Prices and Exchange Rates'
)

# Save with consistent settings
save_plot(fig, 'results/price_analysis.png', dpi=300)
```

### Key Functions:

- **Time Series**: `plot_time_series`, `plot_multiple_time_series`, `plot_time_series_by_group`
- **Statistical**: `plot_scatter`, `plot_boxplot`, `plot_histogram`, `plot_heatmap`
- **Comparisons**: `plot_bar_chart`, `plot_stacked_bar`, `plot_dual_axis`
- **Formatting**: `format_date_axis`, `format_currency_axis`, `configure_axes_for_print`

## Module: `logging_setup`

Configures consistent logging across the application.

```python
from src.utils.logging_setup import setup_logging, get_logger_with_context

# Set up application-wide logging
root_logger = setup_logging(
    log_dir='logs',
    log_level=logging.INFO,
    rotation='daily'
)

# Get logger with context for a specific module
logger = get_logger_with_context(
    __name__,
    {'region': 'north', 'commodity': 'wheat'}
)

# Log with context
logger.info("Processing market data")  # Will include region and commodity
```

### Key Functions:

- **`setup_logging`**: Configures root logger with file and console handlers
- **`get_logger_with_context`**: Creates a logger that adds context to all messages
- **`add_json_logging`**: Adds JSON-formatted logging for machine processing
- **`log_start_stop`**: Automatically logs application startup and shutdown

## Best Practices

1. **Use handle_errors everywhere**:
   ```python
   @handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
   def my_function():
       # Implementation
   ```

2. **Always validate inputs**:
   ```python
   valid, errors = validate_dataframe(df, required_columns=['price'])
   raise_if_invalid(valid, errors)
   ```

3. **Optimize computation-heavy functions**:
   ```python
   @m1_optimized(use_numba=True, parallel=True)
   def intensive_calculation():
       # Implementation
   ```

4. **Process large data in chunks**:
   ```python
   for chunk in read_large_csv_chunks('large_file.csv', chunk_size=10000):
       process_chunk(chunk)
   ```

5. **Use the config singleton**:
   ```python
   threshold = config.get('analysis.threshold', 0.05)
   ```

6. **Add context to logs**:
   ```python
   logger = get_logger_with_context(__name__, {'task': 'data_loading'})
   ```

7. **Use specialized plotting functions**:
   ```python
   fig, ax = plot_time_series_by_group(df, x='date', y='price', group='region')
   ```

8. **Save atomic file writes**:
   ```python
   with AtomicFileWriter('results.csv') as f:
       df.to_csv(f)
   ```

9. **For time series analysis**:
   ```python
   # Test before modeling
   result = test_stationarity(series)
   if not result['stationary']:
       series = np.diff(series)
   ```

10. **For spatial operations**:
    ```python
    # Always ensure consistent CRS
    gdf = reproject_gdf(gdf, to_crs=32638)
    ```
