"""
Yemen Market Integration Project utilities package.

This package provides optimized utility functions for data analysis,
statistical modeling, spatial operations, and performance enhancements
specifically for working with Yemen market integration data on Apple
Silicon (M1/M2) hardware.

Usage examples and best practices are documented in each module.
"""

# Import key utilities to make them available at package level
from .error_handler import (
    handle_errors, 
    MarketIntegrationError, 
    DataError, 
    ModelError, 
    ValidationError
)

from .config import (
    config,
    initialize_config
)

from .logging_setup import (
    setup_logging,
    get_logger_with_context,
    add_json_logging
)

from .decorators import (
    timer,
    m1_optimized,
    disk_cache,
    memoize,
    retry,
    singleton
)

from .validation import (
    validate_dataframe,
    validate_geodataframe,
    validate_time_series,
    validate_model_inputs,
    raise_if_invalid
)

from .file_utils import (
    read_json,
    write_json,
    read_csv,
    write_csv,
    read_geojson,
    write_geojson
)

from .data_utils import (
    clean_column_names,
    convert_dates,
    fill_missing_values,
    normalize_columns,
    compute_price_differentials,
    aggregate_time_series,
    create_lag_features
)

from .stats_utils import (
    test_stationarity,
    test_cointegration,
    test_granger_causality,
    fit_threshold_vecm,
    test_causality_granger
)

from .spatial_utils import (
    reproject_gdf,
    calculate_distances,
    create_spatial_weight_matrix,
    assign_exchange_rate_regime
)

from .performance_utils import (
    configure_system_for_performance,
    parallelize_dataframe,
    optimize_dataframe,
    get_system_info
)

from .plotting_utils import (
    set_plotting_style,
    plot_time_series,
    plot_multiple_time_series,
    plot_time_series_by_group
)

# Initialize performance optimizations for M1 Mac
from .performance_utils import IS_APPLE_SILICON
if IS_APPLE_SILICON:
    from .performance_utils import configure_system_for_performance
    configure_system_for_performance()

# Package metadata
__version__ = '0.1.0'
__author__ = 'Yemen Market Integration Team'


# Best practices guide
BEST_PRACTICES = """
Yemen Market Integration Utilities Best Practices
================================================

1. Error Handling
----------------
- Use the handle_errors decorator for consistent error handling
- Raise specific exceptions (DataError, ModelError) for better error tracking
- Example: @handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))

2. Configuration
---------------
- Access configuration via the singleton 'config' object
- Initialize with project defaults: initialize_config(config_file='config.yaml')
- Override with environment variables using YEMEN_ prefix

3. Logging
---------
- Set up logging once at application start: setup_logging(log_dir='logs')
- Get module-specific loggers: logger = logging.getLogger(__name__)
- Add context to logs: logger = get_logger_with_context(__name__, {'region': 'north'})

4. Optimization
-------------
- Use the @m1_optimized decorator for compute-intensive functions
- Process large DataFrames in parallel with parallelize_dataframe()
- Optimize memory usage with optimize_dataframe()

5. Data Validation
----------------
- Validate inputs early: valid, errors = validate_dataframe(df, required_columns=[...])
- Abort processing on validation failure: raise_if_invalid(valid, errors)

6. Spatial Operations
-------------------
- Always check and ensure consistent CRS before spatial operations
- Use optimized spatial joins for large datasets
- Create spatial weight matrices with conflict adjustment

7. Visualization
--------------
- Set style once at the start of analysis: set_plotting_style()
- Use specialized plotting functions for time series and spatial data
- Save plots with consistent settings: save_plot(fig, 'output.png', dpi=300)

8. Statistical Analysis
--------------------
- Test for stationarity before time series analysis
- Use threshold models for data with regime shifts
- Validate model inputs and outputs

9. Parallelization
----------------
- Parallelize CPU-bound tasks with @m1_optimized(parallel=True)
- Process large files in chunks with chunked_file_reader()
- Monitor memory usage with memory_usage_decorator
"""

def show_best_practices():
    """Print the best practices guide for using Yemen Market Integration utilities."""
    print(BEST_PRACTICES)


def setup_project_environment(config_file=None, log_dir='logs'):
    """
    Set up the project environment with standard configuration and logging.
    
    Parameters
    ----------
    config_file : str, optional
        Path to config file
    log_dir : str, optional
        Directory for log files
    
    Returns
    -------
    tuple
        (config object, root logger)
    """
    # Initialize configuration
    cfg = initialize_config(config_file=config_file)
    
    # Set up logging
    logger = setup_logging(log_dir=log_dir)
    
    # Configure system for performance
    configure_system_for_performance()
    
    return cfg, logger