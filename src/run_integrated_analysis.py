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
    python src/run_integrated_analysis.py --data data/raw/unified_data.geojson \
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

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester
from src.models.threshold import ThresholdCointegration
from src.models.spatial import SpatialEconometrics
from src.models.simulation import MarketIntegrationSimulation
from src.models.diagnostics import ModelDiagnostics

# Import new integration modules
from src.models.spatiotemporal import integrate_time_series_spatial_results
from src.models.interpretation import (
    interpret_unit_root_results,
    interpret_cointegration_results,
    interpret_threshold_results,
    interpret_spatial_results,
    interpret_simulation_results
)
from src.models.reporting import (
    generate_comprehensive_report,
    create_executive_summary,
    export_results_for_publication
)

from src.visualization.time_series import TimeSeriesVisualizer
from src.visualization.maps import MarketMapVisualizer
from src.utils.performance_utils import timer, memory_usage_decorator
from src.utils.validation import validate_data


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
        default='./data/raw/unified_data.geojson',
        help='Path to the GeoJSON data file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Path to save output files'
    )
    
    parser.add_argument(
        '--commodity',
        type=str,
        default='beans (kidney red)',
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
        default=4,
        help='Maximum number of lags for time series analysis'
    )
    
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=5,
        help='Number of nearest neighbors for spatial weights'
    )
    
    parser.add_argument(
        '--conflict-weight',
        type=float,
        default=1.0,
        help='Weight factor for conflict intensity in spatial weights'
    )
    
    parser.add_argument(
        '--report-format',
        type=str,
        choices=['text', 'markdown', 'latex'],
        default='markdown',
        help='Format for the comprehensive report'
    )
    
    args = parser.parse_args()
    return args


@timer
@memory_usage_decorator
def run_unit_root_analysis(processed_gdf, commodity, output_path, max_lags, logger):
    """
    Run comprehensive unit root analysis with structural break detection.
    
    Parameters
    ----------
    processed_gdf : geopandas.GeoDataFrame
        Processed market data
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
        Unit root analysis results
    """
    logger.info(f"Running comprehensive unit root analysis for {commodity}")
    
    # Get data for north and south
    north_data = processed_gdf[
        (processed_gdf['commodity'] == commodity) & 
        (processed_gdf['exchange_rate_regime'] == 'north')
    ]
    south_data = processed_gdf[
        (processed_gdf['commodity'] == commodity) & 
        (processed_gdf['exchange_rate_regime'] == 'south')
    ]
    
    # Check if we have enough data
    if len(north_data) < 30 or len(south_data) < 30:
        logger.warning(f"Insufficient data for {commodity}: North={len(north_data)}, South={len(south_data)}")
        return None
    
    # Aggregate to monthly average prices
    logger.info("Aggregating to monthly average prices")
    north_monthly = north_data.groupby(pd.Grouper(key='date', freq='ME'))['price'].mean().reset_index()
    south_monthly = south_data.groupby(pd.Grouper(key='date', freq='ME'))['price'].mean().reset_index()
    
    # Ensure dates align
    logger.info("Merging north and south data")
    merged = pd.merge(
        north_monthly, south_monthly,
        on='date', suffixes=('_north', '_south')
    )
    
    if len(merged) < 30:
        logger.warning(f"Insufficient overlapping data points: {len(merged)}")
        return None
    
    # Initialize unit root tester
    unit_root_tester = UnitRootTester()
    
    # Run comprehensive unit root tests
    logger.info("Running ADF tests")
    north_adf = unit_root_tester.test_adf(merged['price_north'], lags=max_lags)
    south_adf = unit_root_tester.test_adf(merged['price_south'], lags=max_lags)
    
    logger.info("Running KPSS tests")
    north_kpss = unit_root_tester.test_kpss(merged['price_north'], lags=max_lags)
    south_kpss = unit_root_tester.test_kpss(merged['price_south'], lags=max_lags)
    
    logger.info("Running Zivot-Andrews tests for structural breaks")
    north_za = unit_root_tester.test_zivot_andrews(merged['price_north'])
    south_za = unit_root_tester.test_zivot_andrews(merged['price_south'])
    
    # Determine integration order
    logger.info("Determining integration order")
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
    
    plt.figure(figsize=(12, 8))
    
    # Plot north price series
    plt.subplot(2, 1, 1)
    plt.plot(merged['date'], merged['price_north'], label='North Price')
    if 'breakpoint' in north_za and north_za['breakpoint'] is not None:
        breakpoint_idx = north_za['breakpoint']
        if isinstance(breakpoint_idx, (int, np.integer)) and 0 <= breakpoint_idx < len(merged):
            breakpoint_date = merged['date'].iloc[breakpoint_idx]
            plt.axvline(x=breakpoint_date, color='red', linestyle='--', label=f'Structural Break ({breakpoint_date.strftime("%Y-%m-%d")})')
    plt.title(f'North Price Series with Structural Break - {commodity}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot south price series
    plt.subplot(2, 1, 2)
    plt.plot(merged['date'], merged['price_south'], label='South Price')
    if 'breakpoint' in south_za and south_za['breakpoint'] is not None:
        breakpoint_idx = south_za['breakpoint']
        if isinstance(breakpoint_idx, (int, np.integer)) and 0 <= breakpoint_idx < len(merged):
            breakpoint_date = merged['date'].iloc[breakpoint_idx]
            plt.axvline(x=breakpoint_date, color='red', linestyle='--', label=f'Structural Break ({breakpoint_date.strftime("%Y-%m-%d")})')
    plt.title(f'South Price Series with Structural Break - {commodity}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved structural break visualization to {viz_path}")
    
    return unit_root_results


@timer
@memory_usage_decorator
def run_cointegration_analysis(unit_root_results, commodity, output_path, max_lags, logger):
    """
    Run comprehensive cointegration analysis with multiple methods.
    
    Parameters
    ----------
    unit_root_results : dict
        Results from unit root testing
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
        Cointegration analysis results
    """
    logger.info(f"Running comprehensive cointegration analysis for {commodity}")
    
    # Check if we have valid unit root results
    if not unit_root_results or 'merged_data' not in unit_root_results:
        logger.warning("Cannot run cointegration analysis: missing unit root results")
        return None
    
    # Get merged data from unit root results
    merged = unit_root_results['merged_data']
    
    # Initialize cointegration tester
    cointegration_tester = CointegrationTester()
    
    # Run Engle-Granger test
    logger.info("Running Engle-Granger cointegration test")
    eg_result = cointegration_tester.test_engle_granger(
        merged['price_north'], merged['price_south']
    )
    
    # Run Johansen test
    logger.info("Running Johansen cointegration test")
    try:
        jo_result = cointegration_tester.test_johansen(
            np.column_stack([merged['price_north'], merged['price_south']]),
            det_order=1,  # Default: constant term
            k_ar_diff=max_lags
        )
    except Exception as e:
        logger.error(f"Error in Johansen test: {e}")
        jo_result = None
    
    # Run Gregory-Hansen test for cointegration with structural breaks
    logger.info("Running Gregory-Hansen cointegration test")
    try:
        gh_result = cointegration_tester.test_gregory_hansen(
            merged['price_north'],
            merged['price_south'],
            trend='c',  # Default: constant
            model="regime_shift",
            trim=0.15
        )
    except Exception as e:
        logger.error(f"Error in Gregory-Hansen test: {e}")
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
    
    plt.figure(figsize=(12, 8))
    
    # Plot price series
    plt.subplot(2, 1, 1)
    plt.plot(merged['date'], merged['price_north'], label='North Price')
    plt.plot(merged['date'], merged['price_south'], label='South Price')
    plt.title(f'Price Series - {commodity}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot scatter with regression line if cointegrated
    plt.subplot(2, 1, 2)
    plt.scatter(merged['price_north'], merged['price_south'], alpha=0.6)
    
    if eg_result['cointegrated'] and 'beta' in eg_result:
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
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved cointegration relationship visualization to {viz_path}")
    
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


@timer
@memory_usage_decorator
def run_spatial_analysis(processed_gdf, commodity, output_path, k_neighbors, conflict_weight, logger):
    """
    Run spatial econometric analysis with conflict adjustment.
    
    Parameters
    ----------
    processed_gdf : geopandas.GeoDataFrame
        Processed market data
    commodity : str
        Commodity name
    output_path : pathlib.Path
        Path to save output files
    k_neighbors : int
        Number of nearest neighbors for spatial weights
    conflict_weight : float
        Weight factor for conflict intensity in spatial weights
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Spatial analysis results
    """
    logger.info(f"Running spatial econometric analysis for {commodity}")
    
    # Filter data for the commodity on the latest date
    latest_date = processed_gdf['date'].max()
    latest_data = processed_gdf[
        (processed_gdf['commodity'] == commodity) & 
        (processed_gdf['date'] == latest_date)
    ]
    
    if len(latest_data) < 10:
        logger.warning(f"Insufficient spatial data for {commodity}: {len(latest_data)} markets")
        return None
    
    # Initialize spatial econometrics model
    logger.info("Initializing spatial econometrics model")
    spatial_model = SpatialEconometrics(latest_data)
    
    # Create spatial weights matrix with conflict adjustment
    logger.info(f"Creating spatial weights matrix with k={k_neighbors} and conflict_weight={conflict_weight}")
    spatial_model.create_weight_matrix(
        k=k_neighbors,
        conflict_adjusted=True,
        conflict_col='conflict_intensity_normalized',
        conflict_weight=conflict_weight
    )
    
    # Run global spatial autocorrelation test
    logger.info("Testing for global spatial autocorrelation")
    global_moran = spatial_model.moran_i_test(variable='price')
    
    # Run local spatial autocorrelation test
    logger.info("Testing for local spatial autocorrelation")
    local_moran = spatial_model.local_moran_test(variable='price')
    
    # Check which columns are available in the data
    available_cols = [col for col in ['usdprice', 'conflict_intensity_normalized', 'distance_to_port']
                     if col in latest_data.columns]
    
    logger.info(f"Available columns for spatial model: {available_cols}")
    
    # Estimate spatial lag model
    logger.info("Estimating spatial lag model")
    lag_model = spatial_model.spatial_lag_model(
        y_col='price',
        x_cols=available_cols,
    )
    
    # Estimate spatial error model
    logger.info("Estimating spatial error model")
    error_model = spatial_model.spatial_error_model(
        y_col='price',
        x_cols=available_cols,
    )
    
    # Calculate direct and indirect effects
    logger.info("Calculating direct and indirect effects")
    spillover_effects = spatial_model.calculate_impacts(model_type='lag')
    
    # Compile results
    spatial_results = {
        'model': spatial_model,
        'global_moran': global_moran,
        'local_moran': local_moran,
        'lag_model': lag_model,
        'error_model': error_model,
        'spillover_effects': spillover_effects
    }
    
    # Create visualization of spatial patterns
    viz_path = output_path / f'{commodity.replace(" ", "_")}_spatial_patterns.png'
    
    # Create map visualization if possible
    map_vis = MarketMapVisualizer()
    fig, ax = map_vis.plot_static_map(
        latest_data,
        column='price',
        title=f'Price Distribution of {commodity} ({latest_date.strftime("%Y-%m-%d")})',
        legend=True
    )
    fig.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved spatial patterns visualization to {viz_path}")
    
    # Save local indicators as GeoJSON for mapping
    local_indicators_path = output_path / f'{commodity.replace(" ", "_")}_local_moran.geojson'
    if hasattr(spatial_model, 'gdf'):
        spatial_model.gdf.to_file(local_indicators_path, driver='GeoJSON')
        logger.info(f"Saved local indicators to {local_indicators_path}")
    
    return spatial_results


@timer
@memory_usage_decorator
def run_simulation_analysis(processed_gdf, threshold_results, spatial_results, commodity, output_path, logger):
    """
    Run policy simulation analysis with comprehensive welfare analysis.
    
    Parameters
    ----------
    processed_gdf : geopandas.GeoDataFrame
        Processed market data
    threshold_results : dict
        Results from threshold analysis
    spatial_results : dict
        Results from spatial analysis
    commodity : str
        Commodity name
    output_path : pathlib.Path
        Path to save output files
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Simulation analysis results
    """
    logger.info(f"Running policy simulation analysis for {commodity}")
    
    # Filter data for the specified commodity
    commodity_data = processed_gdf[processed_gdf['commodity'] == commodity]
    
    if len(commodity_data) < 50:
        logger.warning(f"Limited data for simulation: {len(commodity_data)} observations")
    
    # Check if required columns exist for simulation
    required_cols = ['exchange_rate']
    missing_cols = [col for col in required_cols if col not in commodity_data.columns]
    
    if missing_cols:
        logger.warning(f"Missing required columns for simulation: {missing_cols}")
        logger.info("Adding dummy exchange_rate column for simulation")
        # Add dummy exchange_rate column based on exchange_rate_regime
        commodity_data['exchange_rate'] = commodity_data['exchange_rate_regime'].map({
            'north': 250.0,  # Example value for north
            'south': 300.0   # Example value for south
        })
    
    # Initialize simulation model
    logger.info("Initializing market integration simulation model")
    simulation_model = MarketIntegrationSimulation(
        data=commodity_data,
        threshold_model=threshold_results.get('tvecm') if threshold_results else None,
        spatial_model=spatial_results.get('model') if spatial_results else None
    )
    
    # Run exchange rate unification simulation
    logger.info("Simulating exchange rate unification")
    try:
        exchange_unification = simulation_model.simulate_exchange_rate_unification()
    except Exception as e:
        logger.error(f"Error in exchange rate unification simulation: {e}")
        exchange_unification = {'welfare_gain': 0, 'error': str(e)}
    
    # Run conflict reduction simulation
    logger.info("Simulating conflict reduction")
    conflict_reduction = simulation_model.simulate_improved_connectivity()
    
    # Run combined policy simulation
    logger.info("Simulating combined policies")
    combined_policies = simulation_model.simulate_combined_policy()
    
    # Calculate welfare effects
    logger.info("Calculating welfare effects")
    welfare_effects = simulation_model.calculate_welfare_effects()
    
    # Compile results
    simulation_results = {
        'exchange_unification': exchange_unification,
        'conflict_reduction': conflict_reduction,
        'combined_policies': combined_policies,
        'welfare_effects': welfare_effects
    }
    
    # Create visualization of policy impacts
    viz_path = output_path / f'{commodity.replace(" ", "_")}_policy_impacts.png'
    
    # Create bar chart of welfare effects
    plt.figure(figsize=(12, 8))
    
    # Extract welfare gains
    policies = ['Exchange Rate Unification', 'Conflict Reduction', 'Combined Policies']
    welfare_gains = [
        exchange_unification.get('welfare_gain', 0),
        conflict_reduction.get('welfare_gain', 0),
        combined_policies.get('welfare_gain', 0)
    ]
    
    plt.bar(policies, welfare_gains, color=['blue', 'green', 'purple'])
    plt.title(f'Welfare Gains from Policy Interventions - {commodity}')
    plt.xlabel('Policy')
    plt.ylabel('Welfare Gain')
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved policy impacts visualization to {viz_path}")
    
    return simulation_results


@timer
def run_integrated_analysis(args, logger):
    """
    Run the complete integrated analysis workflow.
    
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
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output will be saved to: {output_path}")
        
        # Load and preprocess data
        logger.info(f"Loading data from: {args.data}")
        filename = os.path.basename(args.data)
        loader = DataLoader("./data")
        gdf = loader.load_geojson(filename)
        
        # Validate data
        validate_data(gdf, logger)
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        processed_gdf = preprocessor.preprocess_geojson(gdf)
        
        # Run unit root analysis
        logger.info("Starting unit root analysis")
        unit_root_results = run_unit_root_analysis(
            processed_gdf=processed_gdf,
            commodity=args.commodity,
            output_path=output_path,
            max_lags=args.max_lags,
            logger=logger
        )
        
        # Run cointegration analysis
        logger.info("Starting cointegration analysis")
        cointegration_results = run_cointegration_analysis(
            unit_root_results=unit_root_results,
            commodity=args.commodity,
            output_path=output_path,
            max_lags=args.max_lags,
            logger=logger
        )
        
        # Run threshold analysis
        logger.info("Starting threshold analysis")
        threshold_results = run_threshold_analysis(
            cointegration_results=cointegration_results,
            commodity=args.commodity,
            output_path=output_path,
            max_lags=args.max_lags,
            logger=logger
        )
        
        # Run spatial analysis
        logger.info("Starting spatial analysis")
        spatial_results = run_spatial_analysis(
            processed_gdf=processed_gdf,
            commodity=args.commodity,
            output_path=output_path,
            k_neighbors=args.k_neighbors,
            conflict_weight=args.conflict_weight,
            logger=logger
        )
        
        # Run simulation analysis
        logger.info("Starting simulation analysis")
        simulation_results = run_simulation_analysis(
            processed_gdf=processed_gdf,
            threshold_results=threshold_results,
            spatial_results=spatial_results,
            commodity=args.commodity,
            output_path=output_path,
            logger=logger
        )
        
        # Integrate time series and spatial results
        logger.info("Integrating time series and spatial results")
        integrated_results = integrate_time_series_spatial_results(
            time_series_results={
                'unit_root': unit_root_results,
                'cointegration': cointegration_results,
                'tvecm': threshold_results.get('tvecm') if threshold_results else None
            },
            spatial_results=spatial_results,
            commodity=args.commodity
        )
        
        # Compile all results
        all_results = {
            'unit_root_results': unit_root_results,
            'cointegration_results': cointegration_results,
            'threshold_results': threshold_results,
            'spatial_results': spatial_results,
            'simulation_results': simulation_results,
            'integrated_results': integrated_results
        }
        
        # Generate comprehensive report
        logger.info("Generating comprehensive report")
        report_path = generate_comprehensive_report(
            all_results=all_results,
            commodity=args.commodity,
            output_path=output_path,
            logger=logger
        )
        
        # Create executive summary
        logger.info("Creating executive summary")
        summary_path = create_executive_summary(
            all_results=all_results,
            commodity=args.commodity,
            output_path=output_path,
            logger=logger
        )
        
        # Export results for publication if requested
        if args.report_format == 'latex':
            logger.info("Exporting results for publication")
            publication_path = export_results_for_publication(
                all_results=all_results,
                commodity=args.commodity,
                output_path=output_path,
                logger=logger
            )
        
        logger.info("Integrated analysis completed successfully")
        logger.info(f"Comprehensive report saved to: {report_path}")
        logger.info(f"Executive summary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        logger.exception("Detailed traceback:")
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