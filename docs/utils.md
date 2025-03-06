# Yemen Market Integration Utilities Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Error Handling](#error-handling)
3. [Configuration Management](#configuration-management)
4. [Logging](#logging)
5. [Performance Optimization](#performance-optimization)
6. [Data Validation](#data-validation)
7. [File Operations](#file-operations)
8. [Data Processing](#data-processing)
9. [Spatial Analysis](#spatial-analysis)
10. [Visualization](#visualization)
11. [Common Usage Patterns](#common-usage-patterns)
12. [Examples](#examples)

## Introduction

The Yemen Market Integration Project utilities provide a comprehensive set of tools for robust data processing, analysis, and visualization. These utilities handle error management, configuration, performance optimization, spatial operations, and more, optimized specifically for working with Yemen market data on Apple Silicon hardware.

## Error Handling

The `error_handler.py` module provides centralized error handling with custom exceptions.

### Key Functions and Classes

- `handle_errors` decorator - Wraps functions with consistent error handling
- `MarketIntegrationError` - Base exception class
- `DataError`, `ModelError`, `ValidationError` - Specific exception types
- `ERROR_REGISTRY` - Maps standard exceptions to custom ones

### Basic Usage

```python
from src.utils import handle_errors, DataError
import logging

logger = logging.getLogger(__name__)

@handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
def load_data(file_path):
    """Load data with robust error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Validate data format
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise DataError(f"Unsupported file format: {file_path}")
```

### Advanced Error Handling

```python
@handle_errors(
    logger=logger,
    error_type=(IOError, KeyError),
    default_return=pd.DataFrame(),   # Return empty DataFrame on error
    reraise=True,                    # Reraise as application exception
    error_map={                      # Custom mapping from standard to app exceptions
        ValueError: ValidationError,
        KeyError: ConfigError
    }
)
def process_config(config_section):
    # Implementation
```

## Configuration Management

The `config.py` module provides a singleton configuration object for managing settings.

### Key Functions and Classes

- `Config` class - Singleton configuration manager
- `config` - Global configuration instance
- `initialize_config` - Initialize configuration from multiple sources

### Basic Usage

```python
from src.utils import config, initialize_config

# Initialize with file, environment variables, and defaults
config = initialize_config(
    config_file='config/settings.yaml',
    env_prefix="YEMEN_",
    defaults={'analysis': {'max_lags': 4}}
)

# Access configuration values
data_path = config.get('data_path', 'data')  # With default value
alpha = config.get('analysis.alpha')         # Nested access with dot notation
analysis_config = config.get_section('analysis')  # Get entire section
```

## Logging

The `logging_setup.py` module provides comprehensive logging configuration.

### Key Functions and Classes

- `setup_logging` - Initialize logging system
- `get_logger_with_context` - Create context-aware logger
- `add_json_logging` - Enable JSON format logging

### Basic Usage

```python
from src.utils import setup_logging, get_logger_with_context
import logging

# Initialize logging
logger = setup_logging(
    log_dir='logs',
    log_level=logging.INFO,
    log_file='market_analysis.log',
    rotation='daily'
)

# Module-specific logger
module_logger = logging.getLogger(__name__)

# Context-aware logging
context_logger = get_logger_with_context(__name__, {
    'region': 'abyan', 
    'commodity': 'wheat'
})

context_logger.info("Starting analysis")  
# Output: 2023-08-15 14:30:12 - [region=abyan commodity=wheat] - module_name - INFO - Starting analysis
```

## Performance Optimization

The `performance_utils.py` and `decorators.py` modules provide tools for optimizing performance.

### Key Functions and Classes

- `m1_optimized` decorator - Optimize for Apple Silicon
- `timer` decorator - Measure function execution time
- `parallelize_dataframe` - Process DataFrame in parallel
- `optimize_dataframe` - Reduce memory usage
- `memoize`, `disk_cache` - Cache function results

### Basic Usage

```python
from src.utils import m1_optimized, timer, parallelize_dataframe, optimize_dataframe

@m1_optimized(use_numba=True, parallel=True)
def compute_pairwise_distances(points):
    """Numba-accelerated distance calculation."""
    # Implementation optimized for M1/M2 Mac

@timer
def run_analysis():
    """Time this function execution."""
    # Implementation

# Process large DataFrame in parallel
result_df = parallelize_dataframe(
    large_df,
    process_chunk,
    n_workers=4
)

# Optimize memory usage
optimized_df = optimize_dataframe(
    df,
    downcast=True,      # Downcast numeric columns
    category_min_size=50  # Convert string columns with few unique values to category
)
```

## Data Validation

The `validation.py` module provides data validation utilities.

### Key Functions and Classes

- `validate_dataframe` - Validate DataFrame properties
- `validate_geodataframe` - Validate GeoDataFrame properties
- `validate_time_series` - Validate time series properties
- `raise_if_invalid` - Raise exception on validation failure

### Basic Usage

```python
from src.utils import validate_dataframe, validate_geodataframe, raise_if_invalid

# Validate DataFrame
valid, errors = validate_dataframe(
    df,
    required_columns=['date', 'price', 'admin1'],
    column_types={'date': pd.Timestamp, 'price': float},
    min_rows=1,
    check_nulls=True
)
raise_if_invalid(valid, errors, "Invalid market data")

# Validate GeoDataFrame
valid, errors = validate_geodataframe(
    markets_gdf,
    required_columns=['admin1', 'market_name'],
    crs="EPSG:4326",
    geometry_type="Point"
)
raise_if_invalid(valid, errors, "Invalid market locations")

# Validate time series
valid, errors = validate_time_series(
    price_series,
    min_length=30,
    check_stationarity=True
)
raise_if_invalid(valid, errors, "Invalid price time series")
```

## File Operations

The `file_utils.py` module provides robust file operation utilities.

### Key Functions and Classes

- `read_csv`, `write_csv` - Safe CSV file handling
- `read_geojson`, `write_geojson` - GeoJSON file handling
- `read_large_csv_chunks` - Process large CSV files in chunks
- `AtomicFileWriter` - Atomic file writing context manager

### Basic Usage

```python
from src.utils import read_csv, write_csv, read_geojson, read_large_csv_chunks, AtomicFileWriter

# Safe file reading with enhanced error handling
df = read_csv('data/prices.csv', parse_dates=['date'])
gdf = read_geojson('data/admin_boundaries.geojson')

# Process large CSV in chunks to manage memory
total_rows = 0
for chunk_df in read_large_csv_chunks('large_data.csv', chunk_size=10000):
    # Process each chunk
    process_chunk(chunk_df)
    total_rows += len(chunk_df)

# Atomic file writing (all or nothing)
with AtomicFileWriter('important_results.csv') as f:
    # Write data
    f.write("date,value\n")
    for date, value in results:
        f.write(f"{date},{value}\n")
```

## Data Processing

The `data_utils.py` module provides data manipulation utilities.

### Key Functions and Classes

- `clean_column_names` - Standardize column names
- `convert_dates` - Convert date strings to datetime
- `fill_missing_values` - Handle missing values
- `create_lag_features` - Generate lag features for time series
- `normalize_columns` - Standardize or normalize numeric columns

### Basic Usage

```python
from src.utils import (
    clean_column_names, convert_dates, fill_missing_values,
    create_lag_features, normalize_columns
)

# Clean and standardize data
df = clean_column_names(df)  # Convert to lowercase, replace spaces with underscores
df = convert_dates(df, date_columns=['date', 'report_date'])

# Handle missing values
df = fill_missing_values(
    df,
    numeric_strategy='median',
    categorical_strategy='mode',
    date_strategy='nearest',
    group_columns=['admin1', 'commodity']
)

# Create features for time series analysis
df = create_lag_features(
    df,
    columns=['price'],
    lags=[1, 3, 6, 12],
    group_columns=['admin1', 'commodity']
)

# Normalize numeric columns
df = normalize_columns(
    df,
    columns=['price', 'quantity'],
    method='zscore'  # Standardize to mean=0, std=1
)
```

## Spatial Analysis

The `spatial_utils.py` module provides GIS operations for spatial analysis.

### Key Functions and Classes

- `reproject_gdf` - Reproject GeoDataFrame to different CRS
- `calculate_distances` - Calculate distances between points
- `create_spatial_weight_matrix` - Create weight matrix for spatial econometrics
- `compute_accessibility_index` - Calculate market accessibility

### Basic Usage

```python
from src.utils import (
    reproject_gdf, calculate_distances, create_spatial_weight_matrix,
    compute_accessibility_index
)

# Reproject GeoDataFrame to appropriate CRS for Yemen
markets_gdf = reproject_gdf(
    markets_gdf,
    to_crs=32638  # UTM Zone 38N for Yemen
)

# Calculate distances between markets
distance_matrix = calculate_distances(
    markets_gdf,
    markets_gdf,
    origin_id_col='market_id',
    dest_id_col='market_id'
)

# Create spatial weight matrix with conflict adjustment
w = create_spatial_weight_matrix(
    markets_gdf,
    method='knn',
    k=5,
    conflict_col='conflict_intensity',
    conflict_weight=0.5
)

# Calculate market accessibility index
accessibility = compute_accessibility_index(
    markets_gdf,
    population_gdf,
    max_distance=50000,  # 50km
    weight_col='population'
)
```

## Visualization

The `plotting_utils.py` module provides visualization utilities.

### Key Functions and Classes

- `plot_time_series` - Plot time series data
- `plot_multiple_time_series` - Plot multiple time series
- `plot_time_series_by_group` - Plot time series grouped by category
- `format_date_axis`, `format_currency_axis` - Format plot axes
- `save_plot` - Save plots with consistent settings

### Basic Usage

```python
from src.utils import (
    plot_time_series, plot_multiple_time_series, plot_time_series_by_group, 
    format_date_axis, format_currency_axis, save_plot
)

# Create time series plot
fig, ax = plot_time_series(
    df,
    x='date',
    y='price',
    title='Wheat Prices (2020-2023)',
    xlabel='Date',
    ylabel='Price (YER)',
    color='blue',
    marker='o'
)
format_currency_axis(ax, axis='y', symbol='YER')
save_plot(fig, 'figures/wheat_prices.png', dpi=300)

# Plot multiple commodities
fig, ax = plot_multiple_time_series(
    df,
    x='date',
    y_columns=['wheat_price', 'rice_price', 'beans_price'],
    labels=['Wheat', 'Rice', 'Beans'],
    title='Commodity Prices Comparison'
)

# Plot prices by exchange rate regime
fig, ax = plot_time_series_by_group(
    df,
    x='date',
    y='price',
    group='exchange_rate_regime',
    title='Prices by Exchange Rate Regime',
    palette=['blue', 'red']  # Blue for north, red for south
)
```

## Common Usage Patterns

### Standard Data Processing Pattern

```python
from src.utils import (
    read_csv, validate_dataframe, raise_if_invalid,
    clean_column_names, convert_dates, fill_missing_values,
    create_lag_features, write_csv
)

# 1. Load data with validation
data = read_csv('input.csv')
valid, errors = validate_dataframe(data, required_columns=['date', 'price'])
raise_if_invalid(valid, errors, "Invalid input data")

# 2. Clean and prepare data
data = clean_column_names(data)
data = convert_dates(data, date_columns=['date'])
data = fill_missing_values(data, numeric_strategy='median')

# 3. Create features
data = create_lag_features(data, columns=['price'], lags=[1, 2, 3])

# 4. Save results
write_csv(data, 'processed_data.csv')
```

### Standard Module Template

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
    Function docstring with Parameters and Returns sections.
    
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
```

## Examples

### Example 1: Complete Data Processing Pipeline

```python
"""
Example of a complete data processing pipeline using the utility functions.
"""
import os
import logging
from src.utils import (
    # Project setup
    setup_logging, initialize_config,
    
    # Error handling
    handle_errors, DataError,
    
    # File operations
    read_csv, write_csv, ensure_dir,
    
    # Data processing
    clean_column_names, convert_dates, fill_missing_values,
    create_lag_features, create_rolling_features,
    
    # Validation
    validate_dataframe, raise_if_invalid,
    
    # Performance
    timer, parallelize_dataframe
)

# Initialize logging and configuration
logger = setup_logging(log_dir='logs', log_level=logging.INFO)
config = initialize_config(config_file='config/settings.yaml')

@timer
@handle_errors(logger=logger)
def process_market_data(input_file, output_file):
    """Process market data with comprehensive error handling and validation."""
    logger.info(f"Processing market data from {input_file}")
    
    # Ensure output directory exists
    ensure_dir(os.path.dirname(output_file))
    
    # Load data
    data = read_csv(input_file, parse_dates=['date'])
    
    # Validate input data
    valid, errors = validate_dataframe(
        data,
        required_columns=['date', 'price', 'admin1', 'commodity'],
        column_types={'date': 'datetime', 'price': float},
        min_rows=1
    )
    raise_if_invalid(valid, errors, f"Invalid market data in {input_file}")
    
    # Clean and prepare data
    data = clean_column_names(data)
    data = convert_dates(data, date_columns=['date'])
    data = fill_missing_values(
        data,
        numeric_strategy='median',
        group_columns=['admin1', 'commodity']
    )
    
    # Create time series features
    data = create_lag_features(
        data,
        columns=['price'],
        lags=[1, 3, 6, 12],
        group_columns=['admin1', 'commodity']
    )
    
    data = create_rolling_features(
        data,
        columns=['price'],
        windows=[3, 6, 12],
        stats=['mean', 'std'],
        group_columns=['admin1', 'commodity']
    )
    
    # Save processed data
    write_csv(data, output_file, index=False)
    logger.info(f"Successfully processed {len(data)} records, saved to {output_file}")
    
    return data

if __name__ == "__main__":
    process_market_data(
        config.get('data.input_file'),
        config.get('data.output_file')
    )
```

### Example 2: Memory-Efficient Processing of Large Files

```python
"""
Example of processing a large dataset with memory efficiency.
"""
import logging
from src.utils import (
    setup_logging,
    read_large_csv_chunks,
    optimize_dataframe,
    AtomicFileWriter,
    timer
)

logger = setup_logging()

@timer
def process_large_file(input_file, output_file, chunk_size=100000):
    """Process a large file in memory-efficient chunks."""
    logger.info(f"Processing large file {input_file} in chunks of {chunk_size}")
    
    # Initialize counters and aggregates
    total_rows = 0
    sum_values = 0
    
    # Process file in chunks
    for i, chunk in enumerate(read_large_csv_chunks(input_file, chunk_size=chunk_size)):
        logger.info(f"Processing chunk {i+1} with {len(chunk)} rows")
        
        # Optimize memory usage
        chunk = optimize_dataframe(chunk, downcast=True)
        
        # Process the chunk
        processed = process_chunk(chunk)
        
        # Update aggregates
        total_rows += len(processed)
        sum_values += processed['value'].sum()
        
        # Write output incrementally
        if i == 0:
            # First chunk, create new file
            processed.to_csv(output_file, index=False, mode='w')
        else:
            # Append to existing file
            processed.to_csv(output_file, index=False, mode='a', header=False)
    
    logger.info(f"Completed processing {total_rows} total rows")
    return total_rows, sum_values

def process_chunk(df):
    """Process a single chunk of data."""
    # Implement chunk processing logic here
    return df

if __name__ == "__main__":
    process_large_file('data/large_market_data.csv', 'results/processed_large_data.csv')
```

### Example 3: Spatial Data Analysis with Conflict Adjustment

```python
"""
Example of spatial analysis with conflict-adjusted weights.
"""
import logging
import matplotlib.pyplot as plt
from src.utils import (
    setup_logging,
    read_geojson,
    reproject_gdf,
    create_spatial_weight_matrix,
    calculate_market_isolation,
    plot_static_map,
    save_plot
)

logger = setup_logging()

def analyze_market_isolation(geojson_file, output_dir):
    """Analyze market isolation with conflict adjustment."""
    logger.info(f"Analyzing market isolation from {geojson_file}")
    
    # Load and prepare data
    markets_gdf = read_geojson(geojson_file)
    markets_gdf = reproject_gdf(markets_gdf, to_crs=32638)  # UTM Zone 38N
    
    # Create spatial weight matrices
    logger.info("Creating spatial weight matrices")
    w_standard = create_spatial_weight_matrix(
        markets_gdf, method='knn', k=5, conflict_adjusted=False
    )
    
    w_conflict = create_spatial_weight_matrix(
        markets_gdf, method='knn', k=5, conflict_col='conflict_intensity', 
        conflict_weight=0.5
    )
    
    # Calculate market isolation
    logger.info("Calculating market isolation")
    isolation_gdf = calculate_market_isolation(
        markets_gdf,
        transport_network_gdf=read_geojson('data/roads.geojson'),
        conflict_col='conflict_intensity'
    )
    
    # Visualize isolation index
    logger.info("Creating visualizations")
    fig = plot_static_map(
        isolation_gdf,
        column='isolation_index',
        cmap='YlOrRd',
        scheme='quantiles',
        k=5,
        title='Market Isolation Index'
    )
    save_plot(fig, f'{output_dir}/market_isolation_map.png', dpi=300)
    
    # Additional analysis using the weight matrices can be added here
    
    return isolation_gdf

if __name__ == "__main__":
    analyze_market_isolation(
        'data/market_locations.geojson',
        'results/spatial_analysis'
    )
```

### Example 4: Time Series Analysis with Stationarity Testing

```python
"""
Example of time series analysis using stats utilities.
"""
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import (
    read_csv, test_stationarity, test_cointegration, 
    test_causality_granger, calculate_half_life
)

def analyze_price_dynamics(price_file, commodity="wheat"):
    """Analyze time series properties of price data."""
    # Load and prepare data
    df = read_csv(price_file)
    commodity_data = df[df['commodity'] == commodity]
    
    # Extract north and south prices
    north = commodity_data[commodity_data['exchange_rate_regime'] == 'north']
    south = commodity_data[commodity_data['exchange_rate_regime'] == 'south']
    
    # Aggregate to monthly averages
    north_monthly = north.groupby(pd.Grouper(key='date', freq='M'))['price'].mean()
    south_monthly = south.groupby(pd.Grouper(key='date', freq='M'))['price'].mean()
    
    # Test for stationarity
    north_adf = test_stationarity(north_monthly, test='adf')
    south_adf = test_stationarity(south_monthly, test='adf')
    
    # If non-stationary, test for cointegration
    if not north_adf['stationary'] and not south_adf['stationary']:
        coint_result = test_cointegration(north_monthly, south_monthly)
        
        if coint_result['cointegrated']:
            print(f"Markets are cointegrated (p-value: {coint_result['pvalue']:.4f})")
            
            # Test for Granger causality
            north_to_south = test_causality_granger(south_monthly, north_monthly, maxlag=4)
            south_to_north = test_causality_granger(north_monthly, south_monthly, maxlag=4)
            
            print(f"North Granger-causes South: {north_to_south['causality_detected']}")
            print(f"South Granger-causes North: {south_to_north['causality_detected']}")
            
            # Calculate half-life of deviations
            ec_term = north_monthly - coint_result['beta'][0] - coint_result['beta'][1] * south_monthly
            half_life = calculate_half_life(ec_term)
            print(f"Half-life of price deviations: {half_life:.2f} periods")
        else:
            print("Markets are not cointegrated")
    else:
        print("One or both series are stationary - cointegration not applicable")
    
    return {
        'north_stationary': north_adf,
        'south_stationary': south_adf
    }

if __name__ == "__main__":
    results = analyze_price_dynamics('data/monthly_prices.csv')
```

### Example 5: Integrated Workflow for Policy Impact Analysis

```python
"""
Example of an integrated workflow for policy impact analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import (
    # Setup and configuration
    setup_logging, initialize_config,
    
    # Data operations
    read_geojson, write_geojson, read_csv, write_csv,
    
    # Data processing
    fill_missing_values, create_lag_features, normalize_columns,
    
    # Time series analysis
    test_cointegration, fit_threshold_vecm, test_linearity,
    
    # Spatial analysis
    create_spatial_weight_matrix, reproject_gdf,
    
    # Visualization
    plot_time_series_by_group, plot_static_map, save_plot
)

def analyze_policy_impact(pre_data_file, post_data_file, output_dir):
    """Analyze impact of policy intervention with before-after comparison."""
    logger = setup_logging()
    config = initialize_config('config/analysis.yaml')
    
    # Load pre and post-intervention data
    pre_data = read_geojson(pre_data_file)
    post_data = read_geojson(post_data_file)
    
    # Ensure consistent projection
    pre_data = reproject_gdf(pre_data, to_crs=32638)
    post_data = reproject_gdf(post_data, to_crs=32638)
    
    # 1. Compare price gaps before and after
    pre_gaps = calculate_price_differentials(pre_data)
    post_gaps = calculate_price_differentials(post_data)
    
    # Visualize changes in price differentials
    fig, ax = plot_time_series_by_group(
        pd.concat([
            pre_gaps.assign(period='pre-intervention'), 
            post_gaps.assign(period='post-intervention')
        ]),
        x='date',
        y='price_diff_pct',
        group='period',
        title='Price Differentials Before and After Intervention'
    )
    save_plot(fig, f'{output_dir}/price_diff_comparison.png')
    
    # 2. Compare market integration speed before and after
    # Threshold cointegration analysis
    for commodity in config.get('analysis.commodities'):
        # Extract and prepare data
        pre_commodity = extract_commodity_data(pre_data, commodity)
        post_commodity = extract_commodity_data(post_data, commodity)
        
        # Test for cointegration and estimate threshold models
        pre_tvecm = fit_threshold_vecm(pre_commodity, k_ar_diff=2)
        post_tvecm = fit_threshold_vecm(post_commodity, k_ar_diff=2)
        
        # Compare adjustment speeds
        pre_speed = abs(pre_tvecm['adjustment_above'][0])
        post_speed = abs(post_tvecm['adjustment_above'][0])
        
        print(f"{commodity}: Adjustment speed before: {pre_speed:.4f}, after: {post_speed:.4f}")
        print(f"Change: {(post_speed-pre_speed)/pre_speed*100:.1f}%")
    
    # 3. Compare spatial market connectivity 
    # Create spatial weight matrices with conflict adjustment
    pre_weights = create_spatial_weight_matrix(pre_data, method='knn', k=5, 
                                              conflict_col='conflict_intensity')
    post_weights = create_spatial_weight_matrix(post_data, method='knn', k=5, 
                                               conflict_col='conflict_intensity')
    
    # Calculate connectivity metrics
    pre_connectivity = calculate_connectivity(pre_weights)
    post_connectivity = calculate_connectivity(post_weights)
    
    # Map changes in connectivity
    connectivity_change = post_data.copy()
    connectivity_change['conn_change'] = post_connectivity - pre_connectivity
    
    fig = plot_static_map(
        connectivity_change,
        column='conn_change',
        cmap='RdYlGn',  # Red (negative) to Green (positive)
        scheme='fisher_jenks',
        k=5,
        title='Change in Market Connectivity'
    )
    save_plot(fig, f'{output_dir}/connectivity_change_map.png')
    
    # 4. Save summary results
    write_csv(pd.DataFrame({
        'metric': ['avg_price_diff', 'adjustment_speed', 'connectivity'],
        'pre_intervention': [pre_gaps['price_diff_pct'].mean(), pre_speed, pre_connectivity.mean()],
        'post_intervention': [post_gaps['price_diff_pct'].mean(), post_speed, post_connectivity.mean()],
        'pct_change': [
            (post_gaps['price_diff_pct'].mean() - pre_gaps['price_diff_pct'].mean()) / 
            pre_gaps['price_diff_pct'].mean() * 100,
            (post_speed - pre_speed) / pre_speed * 100,
            (post_connectivity.mean() - pre_connectivity.mean()) / 
            pre_connectivity.mean() * 100
        ]
    }), f'{output_dir}/impact_summary.csv')
    
    return {
        'pre_data': pre_data,
        'post_data': post_data,
        'pre_gaps': pre_gaps,
        'post_gaps': post_gaps
    }

# Helper functions
def calculate_price_differentials(gdf):
    """Calculate price differentials between north and south."""
    north = gdf[gdf['exchange_rate_regime'] == 'north']
    south = gdf[gdf['exchange_rate_regime'] == 'south']
    
    # Group by date and commodity
    north_avg = north.groupby(['date', 'commodity'])['price'].mean().reset_index()
    south_avg = south.groupby(['date', 'commodity'])['price'].mean().reset_index()
    
    # Merge and calculate differentials
    merged = pd.merge(
        north_avg, south_avg, 
        on=['date', 'commodity'], 
        suffixes=('_north', '_south')
    )
    
    merged['price_diff'] = merged['price_north'] - merged['price_south']
    merged['price_diff_pct'] = (merged['price_diff'] / merged['price_south']) * 100
    
    return merged

def extract_commodity_data(gdf, commodity):
    """Extract and prepare data for a specific commodity."""
    commodity_data = gdf[gdf['commodity'] == commodity]
    
    # Prepare data matrix for VECM
    north = commodity_data[commodity_data['exchange_rate_regime'] == 'north']
    south = commodity_data[commodity_data['exchange_rate_regime'] == 'south']
    
    north_monthly = north.groupby(pd.Grouper(key='date', freq='M'))['price'].mean()
    south_monthly = south.groupby(pd.Grouper(key='date', freq='M'))['price'].mean()
    
    # Align dates
    common_dates = north_monthly.index.intersection(south_monthly.index)
    data_matrix = np.column_stack([
        north_monthly.loc[common_dates].values,
        south_monthly.loc[common_dates].values
    ])
    
    return data_matrix

def calculate_connectivity(weights):
    """Calculate market connectivity from spatial weights."""
    # Calculate node degree (number of connections)
    connectivity = np.array([len(neighbors) for neighbors in weights.neighbors.values()])
    
    # Weight by connection strength
    for i, neighbors in enumerate(weights.neighbors.values()):
        connectivity[i] = sum(weights.weights[i])
    
    return connectivity

if __name__ == "__main__":
    results = analyze_policy_impact(
        'data/pre_intervention.geojson',
        'data/post_intervention.geojson',
        'results/policy_impact'
    )
```

## Troubleshooting Common Issues

### Memory Issues

If you encounter memory errors when processing large files:

1. Use chunked file reading with `read_large_csv_chunks`
2. Apply `optimize_dataframe` to reduce memory usage
3. Process data in parallel with `parallelize_dataframe`
4. Use the `@disk_cache` decorator for intermediate results

```python
# Example memory optimization
from src.utils import optimize_dataframe, read_large_csv_chunks

# Process large file in chunks
results = []
for chunk in read_large_csv_chunks('large_file.csv', chunk_size=10000):
    # Optimize memory usage
    chunk = optimize_dataframe(chunk, downcast=True, category_min_size=100)
    # Process the chunk
    processed_chunk = process_function(chunk)
    results.append(processed_chunk)

# Combine results
final_result = pd.concat(results)
```

### Performance Issues

If your code is running slowly:

1. Apply the `@m1_optimized` decorator to compute-intensive functions
2. Use the `@timer` decorator to identify bottlenecks
3. Ensure you're using vectorized operations instead of loops
4. Consider using `memoize` for repeated calculations
5. Parallelize data processing where possible

```python
# Example performance optimization
from src.utils import m1_optimized, timer, parallelize_dataframe

@timer
def analyze_all_regions(df):
    """Process all regions with progress tracking."""
    regions = df['admin1'].unique()
    results = []
    
    for i, region in enumerate(regions):
        print(f"Processing region {i+1}/{len(regions)}: {region}")
        region_data = df[df['admin1'] == region]
        results.append(process_region(region_data))
    
    return pd.concat(results)

# Instead, parallelize:
@timer
def analyze_all_regions_parallel(df):
    """Process all regions in parallel."""
    def process_chunk(chunk_df):
        results = []
        for region, group in chunk_df.groupby('admin1'):
            results.append(process_region(group))
        return pd.concat(results)
    
    return parallelize_dataframe(df, process_chunk)
```

### Data Quality Issues

If you have data quality problems:

1. Use validation functions (`validate_dataframe`, `validate_geodataframe`)
2. Apply `fill_missing_values` with appropriate strategies
3. Check for outliers with `detect_outliers`
4. Normalize/standardize columns with `normalize_columns`

```python
# Example data quality handling
from src.utils import validate_dataframe, fill_missing_values, detect_outliers

# Validate data
valid, errors = validate_dataframe(df, required_columns=['date', 'price', 'admin1'])
if not valid:
    print(f"Data validation failed: {errors}")
    # Handle validation failure
    # Clean data and try again

# Handle missing values
df = fill_missing_values(
    df,
    numeric_strategy='median',
    categorical_strategy='mode',
    group_columns=['admin1', 'commodity']
)

# Detect and handle outliers
df = detect_outliers(df, columns=['price'], method='zscore', threshold=3.0)
df_clean = df[~df['price_outlier']]  # Filter out outliers
```

## Best Practices

1. **Always validate input data** before processing
2. **Use appropriate error handling** with the `handle_errors` decorator
3. **Log important operations** for debugging and monitoring
4. **Optimize memory usage** for large datasets
5. **Use vectorized operations** instead of loops
6. **Structure code in a modular way** for reusability
7. **Document functions thoroughly** with docstrings
8. **Write unit tests** for all utility functions
9. **Monitor performance** with the `timer` decorator
10. **Use version control** for all code and configuration files