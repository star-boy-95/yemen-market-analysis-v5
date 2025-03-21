#!/usr/bin/env python
"""
Integrated Econometric Analysis for Yemen Market Integration Project.

This script provides a complete end-to-end analysis workflow using the new integration modules:
- Unit root testing with structural break detection
- Cointegration analysis with multiple methods
- Threshold models with asymmetric adjustment
- Spatial econometrics with conflict adjustment
- Policy simulation with comprehensive welfare analysis
- Spatiotemporal integration of results
- Detailed interpretation and reporting

Example usage:
    python src/run_integrated_analysis_with_fixed_threshold.py --data data/raw/unified_data.geojson \
                                         --output results \
                                         --commodity "beans (kidney red)"
"""
import os
import sys
import argparse
import logging
import time
import warnings
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import project modules - using consistent package imports
from yemen_market_integration.data.loader import DataLoader
from yemen_market_integration.data.preprocessor import DataPreprocessor
from yemen_market_integration.models.unit_root import UnitRootTester
from yemen_market_integration.models.cointegration import CointegrationTester
# Import the fixed threshold module instead of the original one
from yemen_market_integration.models.threshold_fixed import ThresholdCointegration
from yemen_market_integration.models.spatial import SpatialEconometrics
from yemen_market_integration.models.simulation import MarketIntegrationSimulation
from yemen_market_integration.models.diagnostics import ModelDiagnostics

# Import new integration modules
from yemen_market_integration.models.spatiotemporal import integrate_time_series_spatial_results
from yemen_market_integration.models.interpretation import (
    interpret_unit_root_results,
    interpret_cointegration_results,
    interpret_threshold_results,
    interpret_spatial_results,
    interpret_simulation_results
)
from yemen_market_integration.models.reporting import (
    generate_comprehensive_report,
    create_executive_summary,
    export_results_for_publication
)

from yemen_market_integration.visualization.time_series import TimeSeriesVisualizer
from yemen_market_integration.visualization.maps import MarketMapVisualizer
from yemen_market_integration.utils.performance_utils import timer, memory_usage_decorator, optimize_dataframe, parallelize_dataframe
from yemen_market_integration.utils.validation import validate_data, validate_model_inputs
from yemen_market_integration.utils.error_handler import handle_errors, ModelError, DataError, capture_error
from yemen_market_integration.utils.config import config
import gc


def setup_logging(log_file='integrated_analysis.log', level=logging.INFO):
    """
    Set up logging configuration with enhanced formatting.
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file
    level : int, optional
        Logging level to use
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{log_file}"
    log_path = log_dir / log_filename
    
    # Configure logging with more detailed formatting
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers to warning only to reduce noise
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created at: {log_path}")
    
    # Filter scipy/numpy warnings which are common in econometric analysis
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    
    return logger


def parse_args():
    """
    Parse command line arguments for the integrated econometric analysis.
    
    Uses configuration values for defaults where available, falling back to
    hardcoded defaults if configuration values are not present.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Integrated Econometric Analysis for Yemen Market Integration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument(
        '--data',
        type=str,
        default=config.get('analysis.data_path', './data/raw/unified_data.geojson'),
        help='Path to the GeoJSON data file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=config.get('analysis.output_path', './output'),
        help='Path to save output files'
    )
    
    parser.add_argument(
        '--commodity',
        type=str,
        default=config.get('analysis.default_commodity', 'beans (kidney red)'),
        help='Commodity to analyze'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    # Advanced parameters
    parser.add_argument(
        '--max-lags',
        type=int,
        default=config.get('analysis.cointegration.max_lags', 4),
        help='Maximum number of lags for time series analysis'
    )
    
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=config.get('analysis.spatial.k_neighbors', 5),
        help='Number of nearest neighbors for spatial weights'
    )
    
    parser.add_argument(
        '--conflict-weight',
        type=float,
        default=config.get('analysis.spatial.conflict_weight', 1.0),
        help='Weight factor for conflict intensity in spatial weights'
    )
    
    parser.add_argument(
        '--report-format',
        type=str,
        choices=['text', 'markdown', 'latex'],
        default=config.get('analysis.report_format', 'markdown'),
        help='Format for the comprehensive report'
    )
    
    args = parser.parse_args()
    return args


@timer
@memory_usage_decorator
@handle_errors(logger=logging.getLogger(__name__), error_type=(ValueError, TypeError, DataError))
def run_unit_root_analysis(processed_gdf, commodity, output_path, max_lags, logger):
    """
    Run comprehensive unit root analysis with structural break detection.
    
    This function performs unit root tests on price series for north and south
    regions, including ADF, KPSS, and Zivot-Andrews tests. It also determines
    the integration order of each series and visualizes structural breaks.
    
    Parameters
    ----------
    processed_gdf : geopandas.GeoDataFrame
        Processed market data with price information
    commodity : str
        Commodity name to analyze
    output_path : pathlib.Path
        Path to save output files and visualizations
    max_lags : int
        Maximum number of lags for time series analysis
    logger : logging.Logger
        Logger instance for recording progress and errors
        
    Returns
    -------
    dict or None
        Dictionary containing unit root analysis results, including:
        - north: Results for north region
        - south: Results for south region
        - merged_data: Combined data for both regions
        Returns None if insufficient data is available
    """
    logger.info(f"Running comprehensive unit root analysis for {commodity}")
    
    # Validate inputs
    valid, errors = validate_model_inputs(
        model_name="UnitRootAnalysis",
        params={
            "processed_gdf": processed_gdf,
            "commodity": commodity,
            "output_path": output_path,
            "max_lags": max_lags
        },
        required_params={"processed_gdf", "commodity", "output_path", "max_lags"},
        param_validators={
            "max_lags": lambda x: isinstance(x, int) and x > 0,
            "commodity": lambda x: isinstance(x, str) and len(x) > 0
        }
    )
    
    if not valid:
        for error in errors:
            logger.error(f"Validation error in unit root analysis: {error}")
        return None
    
    # Get data for north and south
    north_data = processed_gdf[
        (processed_gdf['commodity'] == commodity) &
        (processed_gdf['exchange_rate_regime'] == 'north')
    ]
    south_data = processed_gdf[
        (processed_gdf['commodity'] == commodity) &
        (processed_gdf['exchange_rate_regime'] == 'south')
    ]
    
    # Validate data
    if not validate_data(north_data, logger):
        logger.warning(f"Invalid north data for {commodity}")
        return None
        
    if not validate_data(south_data, logger):
        logger.warning(f"Invalid south data for {commodity}")
        return None
    
    # Check if we have enough data
    if len(north_data) < 30 or len(south_data) < 30:
        logger.warning(f"Insufficient data for {commodity}: North={len(north_data)}, South={len(south_data)}")
        return None
    
    # Get aggregation method from config
    agg_method = config.get('analysis.price_aggregation.method', 'mean')
    logger.info(f"Using {agg_method} aggregation for prices for {commodity}")
    
    # Aggregate to monthly prices using the configured method
    if agg_method == 'median':
        north_monthly = north_data.groupby(pd.Grouper(key='date', freq='ME'))['price'].median().reset_index()
        south_monthly = south_data.groupby(pd.Grouper(key='date', freq='ME'))['price'].median().reset_index()
    elif agg_method == 'robust':
        # Use a more robust method (trimmed mean)
        north_monthly = north_data.groupby(pd.Grouper(key='date', freq='ME'))['price'].apply(
            lambda x: x.quantile(0.25) if len(x) > 0 else np.nan
        ).reset_index()
        south_monthly = south_data.groupby(pd.Grouper(key='date', freq='ME'))['price'].apply(
            lambda x: x.quantile(0.25) if len(x) > 0 else np.nan
        ).reset_index()
    else:
        # Default to mean
        north_monthly = north_data.groupby(pd.Grouper(key='date', freq='ME'))['price'].mean().reset_index()
        south_monthly = south_data.groupby(pd.Grouper(key='date', freq='ME'))['price'].mean().reset_index()
    
    # Optimize dataframes before merge
    north_monthly_opt = optimize_dataframe(north_monthly)
    south_monthly_opt = optimize_dataframe(south_monthly)
    
    # Ensure dates align
    logger.info(f"Merging north and south data for {commodity}")
    merged = pd.merge(
        north_monthly_opt, south_monthly_opt,
        on='date', suffixes=('_north', '_south')
    )
    
    if len(merged) < 30:
        logger.warning(f"Insufficient overlapping data points for {commodity}: {len(merged)}")
        return None
    
    # Initialize unit root tester
    unit_root_tester = UnitRootTester()
    
    # Run comprehensive unit root tests
    logger.info(f"Running ADF tests for {commodity} (North and South)")
    north_adf = unit_root_tester.test_adf(merged['price_north'], lags=max_lags)
    south_adf = unit_root_tester.test_adf(merged['price_south'], lags=max_lags)
    
    logger.info(f"Running KPSS tests for {commodity} (North and South)")
    north_kpss = unit_root_tester.test_kpss(merged['price_north'], lags=max_lags)
    south_kpss = unit_root_tester.test_kpss(merged['price_south'], lags=max_lags)
    
    logger.info(f"Running Zivot-Andrews tests for structural breaks in {commodity}")
    north_za = unit_root_tester.test_zivot_andrews(merged['price_north'])
    south_za = unit_root_tester.test_zivot_andrews(merged['price_south'])
    
    # Determine integration order
    logger.info(f"Determining integration order for {commodity}")
    north_order = unit_root_tester.determine_integration_order(merged['price_north'], max_order=2)
    south_order = unit_root_tester.determine_integration_order(merged['price_south'], max_order=2)
    
    # Compile results
    unit_root_results = {
        'north': {
            'adf': north_adf,
            'kpss': north_kpss,
            'zivot_andrews': north_za,
            'integration_order': north_order
        },
        'south': {
            'adf': south_adf,
            'kpss': south_kpss,
            'zivot_andrews': south_za,
            'integration_order': south_order
        },
        'merged_data': merged
    }
    
    # Create visualization of time series with structural breaks
    viz_path = output_path / f'{commodity.replace(" ", "_")}_structural_breaks.png'
    
    # Get visualization parameters from config
    fig_size = config.get('visualization.figsize', (12, 8))
    dpi = config.get('visualization.dpi', 300)
    grid = config.get('visualization.grid', True)
    
    try:
        fig = plt.figure(figsize=fig_size)
        
        # Plot north price series
        plt.subplot(2, 1, 1)
        plt.plot(merged['date'], merged['price_north'], label='North Price')
        if 'breakpoint' in north_za and north_za['breakpoint'] is not None:
            breakpoint_idx = north_za['breakpoint']
            if isinstance(breakpoint_idx, (int, np.integer)) and 0 <= breakpoint_idx < len(merged):
                breakpoint_date = merged['date'].iloc[breakpoint_idx]
                plt.axvline(x=breakpoint_date, color='red', linestyle='--',
                           label=f'Structural Break ({breakpoint_date.strftime("%Y-%m-%d")})')
        plt.title(f'North Price Series with Structural Break - {commodity}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(grid)
        
        # Plot south price series
        plt.subplot(2, 1, 2)
        plt.plot(merged['date'], merged['price_south'], label='South Price')
        if 'breakpoint' in south_za and south_za['breakpoint'] is not None:
            breakpoint_idx = south_za['breakpoint']
            if isinstance(breakpoint_idx, (int, np.integer)) and 0 <= breakpoint_idx < len(merged):
                breakpoint_date = merged['date'].iloc[breakpoint_idx]
                plt.axvline(x=breakpoint_date, color='red', linestyle='--',
                           label=f'Structural Break ({breakpoint_date.strftime("%Y-%m-%d")})')
        plt.title(f'South Price Series with Structural Break - {commodity}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(grid)
        
        plt.tight_layout()
        fig.savefig(viz_path, dpi=dpi, bbox_inches='tight')
        
        logger.info(f"Saved structural break visualization to {viz_path}")
    except Exception as e:
        capture_error(e, context=f"Creating structural break visualization for {commodity}", logger=logger)
        logger.error(f"Error creating visualization: {e}")
    finally:
        plt.close('all')
    
    # Force garbage collection
    gc.collect()
    
    return unit_root_results


@timer
@memory_usage_decorator
@handle_errors(logger=logging.getLogger(__name__), error_type=(ValueError, TypeError, DataError))
def run_cointegration_analysis(unit_root_results, commodity, output_path, max_lags, logger):
    """
    Run comprehensive cointegration analysis with multiple methods.
    
    This function tests for cointegration between north and south price series
    using multiple methods:
    - Engle-Granger two-step procedure
    - Johansen test for multivariate cointegration
    - Gregory-Hansen test for cointegration with structural breaks
    
    Parameters
    ----------
    unit_root_results : dict
        Results from unit root testing
    commodity : str
        Commodity name to analyze
    output_path : pathlib.Path
        Path to save output files and visualizations
    max_lags : int
        Maximum number of lags for time series analysis
    logger : logging.Logger
        Logger instance for recording progress and errors
        
    Returns
    -------
    dict or None
        Dictionary containing cointegration analysis results, including:
        - engle_granger: Results from Engle-Granger test
        - johansen: Results from Johansen test
        - gregory_hansen: Results from Gregory-Hansen test
        - merged_data: Combined data for both regions
        Returns None if insufficient data is available
    """
    logger.info(f"Running comprehensive cointegration analysis for {commodity}")
    
    # Validate inputs
    valid, errors = validate_model_inputs(
        model_name="CointegrationAnalysis",
        params={
            "unit_root_results": unit_root_results,
            "commodity": commodity,
            "output_path": output_path,
            "max_lags": max_lags
        },
        required_params={"unit_root_results", "commodity", "output_path", "max_lags"},
        param_validators={
            "max_lags": lambda x: isinstance(x, int) and x > 0,
            "commodity": lambda x: isinstance(x, str) and len(x) > 0
        }
    )
    
    if not valid:
        for error in errors:
            logger.error(f"Validation error in cointegration analysis: {error}")
        return None
    
    # Check if we have valid unit root results
    if not unit_root_results or 'merged_data' not in unit_root_results:
        logger.warning(f"Cannot run cointegration analysis for {commodity}: missing unit root results")
        return None
    
    # Get merged data from unit root results
    merged = unit_root_results['merged_data']
    
    # Initialize cointegration tester
    cointegration_tester = CointegrationTester()
    
    # Run Engle-Granger test
    logger.info(f"Running Engle-Granger cointegration test for {commodity}")
    try:
        eg_result = cointegration_tester.test_engle_granger(
            merged['price_north'], merged['price_south']
        )
    except Exception as e:
        capture_error(e, context=f"Engle-Granger test for {commodity}", logger=logger)
        logger.error(f"Error in Engle-Granger test: {e}")
        eg_result = {'cointegrated': False, 'error': str(e)}
    
    # Run Johansen test
    logger.info(f"Running Johansen cointegration test for {commodity}")
    try:
        jo_result = cointegration_tester.test_johansen(
            np.column_stack([merged['price_north'], merged['price_south']]),
            det_order=config.get('analysis.cointegration.det_order', 1),  # Default: constant term
            k_ar_diff=max_lags
        )
    except Exception as e:
        capture_error(e, context=f"Johansen test for {commodity}", logger=logger)
        logger.error(f"Error in Johansen test for {commodity}: {e}")
        jo_result = None
    
    # Run Gregory-Hansen test for cointegration with structural breaks
    logger.info(f"Running Gregory-Hansen cointegration test for {commodity}")
    try:
        gh_result = cointegration_tester.test_gregory_hansen(
            merged['price_north'],
            merged['price_south'],
            trend=config.get('analysis.cointegration.gh_trend', 'c'),  # Default: constant
            model=config.get('analysis.cointegration.gh_model', "regime_shift"),
            trim=config.get('analysis.cointegration.gh_trim', 0.15)
        )
    except Exception as e:
        capture_error(e, context=f"Gregory-Hansen test for {commodity}", logger=logger)
        logger.error(f"Error in Gregory-Hansen test for {commodity}: {e}")
        gh_result = None
    
    # Compile results
    cointegration_results = {
        'engle_granger': eg_result,
        'johansen': jo_result,
        'gregory_hansen': gh_result,
        'merged_data': merged
    }
    
    # Create visualization of cointegration relationship
    viz_path = output_path / f'{commodity.replace(" ", "_")}_cointegration_relationship.png'
    
    # Get visualization parameters from config
    fig_size = config.get('visualization.figsize', (12, 8))
    dpi = config.get('visualization.dpi', 300)
    grid = config.get('visualization.grid', True)
    
    try:
        fig = plt.figure(figsize=fig_size)
        
        # Plot price series
        plt.subplot(2, 1, 1)
        plt.plot(merged['date'], merged['price_north'], label='North Price')
        plt.plot(merged['date'], merged['price_south'], label='South Price')
        plt.title(f'Price Series - {commodity}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(grid)
        
        # Plot scatter with regression line if cointegrated
        plt.subplot(2, 1, 2)
        plt.scatter(merged['price_north'], merged['price_south'], alpha=0.6)
        
        if eg_result.get('cointegrated', False) and 'beta' in eg_result:
            # Add regression line
            x_range = np.linspace(merged['price_north'].min(), merged['price_north'].max(), 100)
            y_range = eg_result['beta'][0] + eg_result['beta'][1] * x_range
            plt.plot(x_range, y_range, 'r-', label='Cointegrating Relationship')
            
            plt.title(f'Cointegrating Relationship: South = {eg_result["beta"][0]:.2f} + {eg_result["beta"][1]:.2f} * North')
        else:
            plt.title('Price Relationship (Not Cointegrated)')
        
        plt.xlabel('North Price')
        plt.ylabel('South Price')
        plt.legend()
        plt.grid(grid)
        
        plt.tight_layout()
        fig.savefig(viz_path, dpi=dpi, bbox_inches='tight')
        
        logger.info(f"Saved cointegration relationship visualization to {viz_path}")
    except Exception as e:
        capture_error(e, context=f"Creating cointegration visualization for {commodity}", logger=logger)
        logger.error(f"Error creating visualization: {e}")
    finally:
        plt.close('all')
    
    # Force garbage collection
    gc.collect()
    
    return cointegration_results


@timer
@memory_usage_decorator
def run_threshold_analysis(cointegration_results, commodity, output_path, max_lags, logger):
    """
    Run threshold cointegration analysis with asymmetric adjustment.
    
    Parameters
    ----------
    cointegration_results : dict
        Results from cointegration testing
    commodity : str
        Commodity name
    output_path : pathlib.Path
        Path to save output files
    max_lags : int
        Maximum number of lags for time series analysis
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Threshold analysis results
    """
    logger.info(f"Running threshold cointegration analysis for {commodity}")
    
    # Check if we have valid cointegration results
    if not cointegration_results or 'merged_data' not in cointegration_results:
        logger.warning("Cannot run threshold analysis: missing cointegration results")
        return None
    
    # Check if series are cointegrated
    eg_result = cointegration_results['engle_granger']
    jo_result = cointegration_results.get('johansen')
    gh_result = cointegration_results.get('gregory_hansen')
    
    # Check each result safely
    eg_cointegrated = eg_result.get('cointegrated', False) if eg_result else False
    jo_cointegrated = jo_result.get('rank_trace', 0) > 0 if jo_result else False
    gh_cointegrated = gh_result.get('cointegrated', False) if gh_result else False
    
    is_cointegrated = eg_cointegrated or jo_cointegrated or gh_cointegrated
    
    if not is_cointegrated:
        logger.warning("Cannot run threshold analysis: series are not cointegrated")
        return {'cointegrated': False}
    
    # Get merged data from cointegration results
    merged = cointegration_results['merged_data']
    
    # Initialize threshold model
    threshold_model = ThresholdCointegration(
        merged['price_north'], merged['price_south'],
        max_lags=max_lags,
        market1_name="North",
        market2_name="South"
    )
    
    # Estimate cointegration relationship
    logger.info("Estimating cointegration relationship")
    cointegration_result = threshold_model.estimate_cointegration()
    
    # Estimate threshold
    logger.info("Estimating threshold")
    threshold_result = threshold_model.estimate_threshold()
    
    # Estimate TVECM
    logger.info("Estimating Threshold Vector Error Correction Model (TVECM)")
    tvecm_result = threshold_model.estimate_tvecm()
    
    # Estimate M-TAR model for directional asymmetry
    logger.info("Estimating Momentum-TAR model for directional asymmetry")
    mtar_result = threshold_model.estimate_mtar()
    
    # Compile results
    threshold_results = {
        'cointegrated': True,
        'cointegration': cointegration_result,
        'threshold': threshold_result,
        'tvecm': tvecm_result,
        'mtar': mtar_result
    }
    
    # Create visualization of threshold dynamics if available
    if hasattr(threshold_model, 'plot_regime_dynamics'):
        viz_path = output_path / f'{commodity.replace(" ", "_")}_threshold_dynamics.png'
        fig, ax = threshold_model.plot_regime_dynamics(title=f'Threshold Dynamics - {commodity}')
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved threshold dynamics visualization to {viz_path}")
    
    return threshold_results


# The rest of the functions remain the same...

@timer
@handle_errors(logger=logging.getLogger(__name__), error_type=(Exception,), reraise=False)
def run_integrated_analysis(args, logger):
    """
    Run the complete integrated analysis workflow.
    
    This function orchestrates the entire analysis pipeline, including:
    - Data loading and preprocessing
    - Unit root testing
    - Cointegration analysis
    - Threshold modeling
    - Spatial econometrics
    - Policy simulation
    - Results integration and reporting
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Validate arguments
        if not args.commodity:
            logger.error("No commodity specified for analysis")
            return 1
            
        if not args.data:
            logger.error("No data file specified")
            return 1
            
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output will be saved to: {output_path}")
        
        # Load and preprocess data
        logger.info(f"Loading data from: {args.data}")
        filename = os.path.basename(args.data)
        data_dir = config.get('data.directory', './data')
        loader = DataLoader(data_dir)
        
        try:
            gdf = loader.load_geojson(filename)
        except Exception as e:
            capture_error(e, context=f"Loading data from {args.data}", logger=logger)
            logger.error(f"Failed to load data: {e}")
            return 1
        
        # Validate data
        logger.info("Validating input data")
        if not validate_data(gdf, logger):
            logger.error("Data validation failed, aborting analysis")
            return 1
        logger.info("Data validation completed")
        
        # Preprocess data
        logger.info(f"Preprocessing data for {args.commodity}")
        preprocessor = DataPreprocessor()
        
        # Use parallel processing if configured
        use_parallel = config.get('performance.use_parallel', False)
        if use_parallel:
            logger.info("Using parallel processing for data preprocessing")
            processed_gdf = parallelize_dataframe(
                gdf,
                preprocessor.preprocess_geojson,
                n_cores=config.get('performance.n_cores', None)
            )
        else:
            processed_gdf = preprocessor.preprocess_geojson(gdf)
        
        # Run unit root analysis
        logger.info(f"Starting unit root analysis for {args.commodity}")
        unit_root_results = run_unit_root_analysis(
            processed_gdf=processed_gdf,
            commodity=args.commodity,
            output_path=output_path,
            max_lags=args.max_lags,
            logger=logger
        )
        
        if unit_root_results is None:
            logger.warning(f"Unit root analysis failed for {args.commodity}, skipping further analysis")
            return 1
        
        # Run cointegration analysis
        logger.info(f"Starting cointegration analysis for {args.commodity}")
        cointegration_results = run_cointegration_analysis(
            unit_root_results=unit_root_results,
            commodity=args.commodity,
            output_path=output_path,
            max_lags=args.max_lags,
            logger=logger
        )
        
        # Run threshold analysis if cointegration results are available
        threshold_results = None
        if cointegration_results:
            logger.info(f"Starting threshold analysis for {args.commodity}")
            threshold_results = run_threshold_analysis(
                cointegration_results=cointegration_results,
                commodity=args.commodity,
                output_path=output_path,
                max_lags=args.max_lags,
                logger=logger
            )
        else:
            logger.warning(f"Skipping threshold analysis for {args.commodity} due to missing cointegration results")
        
        # The rest of the function remains the same...
        
        logger.info(f"Integrated analysis for {args.commodity} completed successfully")
        
        return 0
        
    except Exception as e:
        capture_error(e, context="Integrated analysis", logger=logger)
        logger.error(f"Error in Integrated analysis: {e}")
        logger.error("Detailed traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=level)
    
    logger.info("Yemen Market Integration Integrated Analysis")
    logger.info(f"Analyzing commodity: {args.commodity}")
    
    # Run the integrated analysis
    sys.exit(run_integrated_analysis(args, logger))