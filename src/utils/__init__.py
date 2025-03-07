"""
Yemen Market Integration Project utilities package.

This package provides optimized utility functions for data analysis,
statistical modeling, spatial operations, and performance enhancements
specifically for working with Yemen market integration data on Apple
Silicon (M1/M2) hardware.

Usage examples and best practices are documented in each module.
"""

# Error handling utilities
from .error_handler import (
    handle_errors, 
    capture_error,
    MarketIntegrationError, 
    DataError, 
    ModelError, 
    ValidationError,
    ConfigError,
    VisualizationError
)

# Configuration utilities
from .config import (
    config,
    initialize_config,
    Config
)

# Logging utilities
from .logging_setup import (
    setup_logging,
    get_logger_with_context,
    add_json_logging,
    log_start_stop,
    ContextAdapter,
    JsonFileHandler
)

# Decorator utilities
from .decorators import (
    timer,
    m1_optimized,
    disk_cache,
    memoize,
    retry,
    singleton,
    rate_limited,
    log_calls,
    validate_args,
    deprecated
)

# Validation utilities
from .validation import (
    validate_dataframe,
    validate_geodataframe,
    validate_time_series,
    validate_model_inputs,
    validate_geojson,
    validate_email,
    validate_phone_number,
    validate_latitude,
    validate_longitude,
    validate_percentage,
    validate_date_string,
    raise_if_invalid
)

# File handling utilities
from .file_utils import (
    ensure_dir,
    read_json,
    write_json,
    read_yaml,
    write_yaml,
    read_csv,
    write_csv,
    read_geojson,
    write_geojson,
    read_pickle,
    write_pickle,
    move_file,
    copy_file,
    delete_file,
    file_size,
    file_hash,
    list_files,
    compress_file,
    decompress_file,
    create_backup,
    AtomicFileWriter,
    read_large_csv_chunks,
    read_large_geojson_chunks,
    read_large_file_chunks
)

# Data processing utilities
from .data_utils import (
    clean_column_names,
    convert_dates,
    fill_missing_values,
    normalize_columns,
    detect_outliers,
    compute_price_differentials,
    aggregate_time_series,
    create_lag_features,
    create_rolling_features,
    convert_exchange_rates,
    calculate_price_changes,
    create_date_features,
    pivot_data,
    unpivot_data,
    merge_dataframes,
    bin_numeric_column,
    encode_categorical,
    winsorize_columns,
    explode_geojson_features,
    calculate_distance_matrix
)

# Statistical utilities
from .stats_utils import (
    test_stationarity,
    test_cointegration,
    test_granger_causality,
    fit_var_model,
    fit_vecm_model,
    test_autocorrelation,
    test_white_noise,
    test_covariate_significance,
    compute_rolling_correlation,
    estimate_threshold_tar,
    calculate_threshold_ci,
    test_linearity,
    fit_threshold_vecm,
    calculate_half_life,
    bootstrap_confidence_interval,
    compute_variance_ratio,
    test_causality_granger,
    test_structural_break
)

# Spatial utilities
from .spatial_utils import (
    reproject_gdf,
    reproject_geometry,
    create_point_from_coords,
    create_buffer,
    find_nearest_points,
    overlay_layers,
    calculate_distances,
    calculate_distance_matrix,
    create_spatial_weight_matrix,
    extract_area_of_interest,
    aggregate_points_to_polygons,
    compute_accessibility_index,
    create_exchange_regime_boundaries,
    calculate_market_isolation,
    assign_exchange_rate_regime,
    create_market_catchments
)

# Performance utilities
from .performance_utils import (
    configure_system_for_performance,
    parallelize_dataframe,
    optimize_dataframe,
    get_system_info,
    memory_usage_decorator,
    IS_APPLE_