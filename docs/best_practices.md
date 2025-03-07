# Yemen Market Integration Project: Best Practices

This document outlines recommended patterns and practices for developing with the Yemen Market Integration Project utilities.

## Project Setup

```python
from src.utils import (
    config, initialize_config, setup_logging, 
    configure_system_for_performance
)

def main():
    # Initialize configuration from multiple sources
    initialize_config(
        config_file='settings.yaml',
        env_prefix='YEMEN_',
        defaults={'analysis': {'alpha': 0.05}}
    )
    
    # Configure logging
    logger = setup_logging(
        log_dir=config.get('logging.log_dir', 'logs'),
        log_level=config.get('logging.level', 'INFO')
    )
    
    # Optimize performance for M1 Mac
    configure_system_for_performance()
    
    # Start processing pipeline
    logger.info("Starting analysis pipeline")
    # ...
```

## Error Handling

Apply consistent error handling to all functions that might fail.

```python
import logging
from src.utils import handle_errors, DataError

logger = logging.getLogger(__name__)

@handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
def load_market_data(file_path):
    """Load market data from CSV file."""
    if not file_path.endswith('.csv'):
        raise ValueError(f"Expected CSV file, got {file_path}")
    
    # Implementation that might raise FileNotFoundError
    return read_csv(file_path)
```

## Data Processing Pipeline

Follow a consistent data processing pattern:

```python
from src.utils import (
    read_csv, clean_column_names, validate_dataframe,
    fill_missing_values, create_lag_features
)

def process_market_data(file_path):
    # 1. Load the data
    df = read_csv(file_path)
    
    # 2. Clean column names
    df = clean_column_names(df)
    
    # 3. Validate structure
    valid, errors = validate_dataframe(
        df, 
        required_columns=['date', 'market', 'commodity', 'price'],
        column_types={'date': pd.Timestamp, 'price': float}
    )
    if not valid:
        logger.error(f"Invalid data format: {'; '.join(errors)}")
        return None
    
    # 4. Clean and transform
    df = fill_missing_values(df, numeric_strategy='median')
    
    # 5. Create features
    df = create_lag_features(df, columns=['price'], lags=[1, 2, 3])
    
    return df
```

## Memory Optimization

For large datasets, use these patterns:

```python
from src.utils import (
    read_large_csv_chunks, parallelize_dataframe, 
    optimize_dataframe
)

def process_large_file(file_path):
    results = []
    
    # Process in chunks
    for chunk in read_large_csv_chunks(file_path, chunk_size=10000):
        # Optimize memory usage
        chunk = optimize_dataframe(chunk, downcast=True)
        
        # Process chunk
        processed = process_chunk(chunk)
        results.append(processed)
    
    # Combine results
    return pd.concat(results)

def process_in_parallel(df):
    # Define function to process a single chunk
    def process_market_group(market_df):
        # Process a single market's data
        return processed_df
    
    # Process all markets in parallel
    return parallelize_dataframe(
        df, 
        process_market_group, 
        n_workers=config.get('performance.n_workers', 4)
    )
```

## Time Series Analysis

For market integration analysis:

```python
from src.utils import (
    test_stationarity, test_cointegration, 
    fit_threshold_vecm, calculate_half_life
)

def analyze_market_integration(north_prices, south_prices):
    results = {}
    
    # 1. Test stationarity
    north_stationary = test_stationarity(north_prices)
    south_stationary = test_stationarity(south_prices)
    
    # 2. If not stationary, difference the series
    if not north_stationary['stationary']:
        north_prices = np.diff(north_prices)
        
    if not south_stationary['stationary']:
        south_prices = np.diff(south_prices)
    
    # 3. Test for cointegration
    coint_result = test_cointegration(
        north_prices, 
        south_prices, 
        method=config.get('analysis.cointegration.method', 'engle-granger')
    )
    
    # 4. If cointegrated, fit threshold model
    if coint_result['cointegrated']:
        model = fit_threshold_vecm(
            np.column_stack([north_prices, south_prices]),
            k_ar_diff=config.get('analysis.threshold_vecm.k_ar_diff', 2),
            coint_rank=1
        )
        
        # 5. Calculate half-life of shocks
        half_life = calculate_half_life(
            model['below_regime']['alpha'][0], 
            regime='threshold'
        )
        
        results.update({
            'threshold': model['threshold'],
            'half_life': half_life,
            'alpha_below': model['below_regime']['alpha'],
            'alpha_above': model['above_regime']['alpha']
        })
    
    results.update({
        'north_stationary': north_stationary['stationary'],
        'south_stationary': south_stationary['stationary'],
        'cointegrated': coint_result['cointegrated']
    })
    
    return results
```

## Spatial Analysis

For spatial operations:

```python
from src.utils import (
    read_geojson, reproject_gdf, calculate_distances,
    assign_exchange_rate_regime, create_spatial_weight_matrix
)

def analyze_spatial_relationships(markets_path, regions_path):
    # 1. Load spatial data
    markets_gdf = read_geojson(markets_path)
    regions_gdf = read_geojson(regions_path)
    
    # 2. Ensure consistent CRS
    crs = config.get('spatial.crs', 32638)  # UTM Zone 38N for Yemen
    markets_gdf = reproject_gdf(markets_gdf, to_crs=crs)
    regions_gdf = reproject_gdf(regions_gdf, to_crs=crs)
    
    # 3. Assign regions to markets
    markets_gdf = assign_exchange_rate_regime(
        markets_gdf,
        regions_gdf,
        regime_col='exchange_rate_regime'
    )
    
    # 4. Calculate distance matrix
    distances = calculate_distances(
        markets_gdf,
        markets_gdf,
        origin_id_col='market_id',
        dest_id_col='market_id'
    )
    
    # 5. Create spatial weight matrix
    weights = create_spatial_weight_matrix(
        markets_gdf,
        method='knn',
        k=config.get('spatial.knn', 5),
        conflict_col='conflict_intensity'
    )
    
    return markets_gdf, distances, weights
```

## Visualization

For consistent visualization:

```python
from src.utils import (
    set_plotting_style, plot_time_series_by_group,
    format_date_axis, save_plot
)

def visualize_price_trends(df, output_dir):
    # 1. Set consistent style
    set_plotting_style()
    
    # 2. Create plot
    fig, ax = plot_time_series_by_group(
        df,
        x='date',
        y='price',
        group='exchange_rate_regime',
        title='Price Trends by Exchange Rate Regime',
        ylabel='Price (YER)'
    )
    
    # 3. Format axes
    format_date_axis(
        ax, 
        date_format=config.get('visualization.date_format', '%Y-%m'),
        interval=config.get('visualization.date_interval', 'month')
    )
    
    # 4. Save with consistent settings
    save_plot(
        fig,
        f"{output_dir}/price_trends.png",
        dpi=config.get('visualization.figure_dpi', 300)
    )
```

## Configuration

Use the configuration system for all parameters:

```python
from src.utils import config

# Get with default
alpha = config.get('analysis.alpha', 0.05)

# Get nested section
model_params = config.get_section('analysis.threshold_vecm')

# Override for testing
config.set('analysis.bootstrap_reps', 100)
```

## Logging

Use consistent logging patterns:

```python
import logging
from src.utils import get_logger_with_context

def process_market_data(market, commodity):
    # Create logger with context
    logger = get_logger_with_context(
        __name__, 
        {'market': market, 'commodity': commodity}
    )
    
    # Log with context (will include market and commodity)
    logger.info("Processing market data")
    
    try:
        # Processing logic
        result = calculate_something()
        logger.debug("Calculated result: %.2f", result)
        
    except Exception as e:
        logger.error("Processing failed: %s", str(e))
        raise
    
    logger.info("Processing complete")
```

## File Operations

Use safe file operations:

```python
from src.utils import (
    ensure_dir, read_csv, write_csv, 
    AtomicFileWriter, create_backup
)

def save_processed_data(df, output_path):
    # Ensure directory exists
    output_dir = ensure_dir(Path(output_path).parent)
    
    # Create backup if file exists
    if Path(output_path).exists():
        backup_path = create_backup(output_path)
        logger.info(f"Created backup at {backup_path}")
    
    # Atomic write to prevent corruption
    with AtomicFileWriter(output_path, 'w') as f:
        df.to_csv(f, index=False)
    
    logger.info(f"Saved processed data to {output_path}")
```

## Testing

Follow these testing patterns:

```python
import pytest
from src.utils import validate_dataframe

def test_market_data_validation():
    # Create test data
    test_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=10),
        'price': np.random.rand(10) * 100,
        'market': ['Market A'] * 10
    })
    
    # Test validation
    valid, errors = validate_dataframe(
        test_df,
        required_columns=['date', 'price', 'market'],
        column_types={'date': pd.Timestamp, 'price': float}
    )
    
    assert valid
    assert len(errors) == 0
    
    # Test validation failure
    invalid_df = test_df.drop(columns=['price'])
    valid, errors = validate_dataframe(
        invalid_df,
        required_columns=['date', 'price', 'market']
    )
    
    assert not valid
    assert any('price' in error for error in errors)
```

## Command-Line Tools

For command-line tools, use this pattern:

```python
import argparse
from src.utils import config, initialize_config, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Market Integration Analysis')
    parser.add_argument('--config', '-c', help='Path to config file')
    parser.add_argument('--input', '-i', required=True, help='Input data file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize configuration
    initialize_config(config_file=args.config)
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else config.get('logging.level', 'INFO')
    logger = setup_logging(
        log_dir=config.get('logging.log_dir', 'logs'),
        log_level=log_level
    )
    
    logger.info(f"Processing {args.input}")
    
    # Implement processing logic
    # ...
    
    logger.info("Processing complete")

if __name__ == '__main__':
    main()
```

## Performance Tips

1. **Always profile first**: Use `@timer` to identify bottlenecks
2. **Cache expensive operations**: Use `@disk_cache` for network or compute-intensive functions
3. **Prefer chunking over loading entire files**: Use chunked readers for all large files
4. **Optimize DataFrame memory usage early**: Apply `optimize_dataframe` before processing
5. **Use parallel processing for independent operations**: Apply `parallelize_dataframe` for operations on grouped data

Remember that these utilities are designed to work together as an integrated system. Use them consistently across the project for maintainable, efficient, and robust code.