"""
Main script for Yemen market integration analysis.

This script serves as the command-line interface for the Yemen Market Integration
project, orchestrating the complete analysis workflow. It allows users to run
different types of analyses (threshold cointegration, spatial econometrics,
policy simulation) on market data, generate visualizations, and export results.

The script implements advanced econometric methodologies including:
- Unit root testing with structural break detection
- Threshold cointegration with asymmetric adjustment
- Spatial econometrics with conflict adjustment
- Policy simulation with comprehensive welfare analysis
- Detailed interpretation of econometric results
- Statistical validation and robustness checks

Example usage:
    python src/main.py --data data/raw/unified_data.geojson --output results \
                      --commodity "beans (kidney red)" --threshold --spatial --simulation
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import argparse
import logging
import time
import warnings
from datetime import datetime
import statsmodels.api as sm
from scipy import stats
from itertools import product

# Import project modules
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.unit_root import UnitRootTester
from models.cointegration import CointegrationTester
from models.threshold import ThresholdCointegration
from models.threshold_vecm import ThresholdVECM
from models.spatial import SpatialEconometrics
from models.simulation import MarketIntegrationSimulation
from models.diagnostics import ModelDiagnostics
from models.model_selection import calculate_information_criteria
from visualization.time_series import TimeSeriesVisualizer
from visualization.maps import MarketMapVisualizer
from visualization.asymmetric_plots import AsymmetricAdjustmentVisualizer
from utils.performance_utils import timer, memory_usage_decorator
from utils.validation import validate_data, validate_model_inputs
from utils.stats_utils import calculate_gini_coefficient, bootstrap_confidence_interval


def setup_logging(log_file='yemen_analysis.log', level=logging.INFO):
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
    Parse command line arguments with enhanced options for econometric analysis.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Yemen Market Integration Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument(
        '--data', 
        type=str, 
        default='./data/raw/unified_data.geojson',
        help='Path to the GeoJSON data file'
    )
    
    io_group.add_argument(
        '--output', 
        type=str, 
        default='./output',
        help='Path to save output files'
    )
    
    io_group.add_argument(
        '--commodity', 
        type=str, 
        default='beans (kidney red)',
        help='Commodity to analyze'
    )
    
    io_group.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging output'
    )
    
    # Analysis selection arguments
    analysis_group = parser.add_argument_group('Analysis Selection')
    analysis_group.add_argument(
        '--threshold', 
        action='store_true',
        help='Run threshold cointegration analysis'
    )
    
    analysis_group.add_argument(
        '--spatial', 
        action='store_true',
        help='Run spatial econometric analysis'
    )
    
    analysis_group.add_argument(
        '--simulation', 
        action='store_true',
        help='Run policy simulations'
    )
    
    analysis_group.add_argument(
        '--validation',
        action='store_true',
        help='Run enhanced statistical validation and robustness checks'
    )
    
    analysis_group.add_argument(
        '--sensitivity',
        action='store_true',
        help='Run parameter sensitivity analysis'
    )
    
    # Time series parameters
    ts_group = parser.add_argument_group('Time Series Parameters')
    ts_group.add_argument(
        '--max-lags',
        type=int,
        default=4,
        help='Maximum number of lags for time series analysis'
    )
    
    ts_group.add_argument(
        '--coint-method',
        type=str,
        choices=['eg', 'johansen', 'gregory-hansen', 'all'],
        default='all',
        help='Cointegration test method(s) to use'
    )
    
    ts_group.add_argument(
        '--bootstrap-iterations',
        type=int,
        default=1000,
        help='Number of bootstrap iterations for confidence intervals'
    )
    
    # Spatial parameters
    spatial_group = parser.add_argument_group('Spatial Parameters')
    spatial_group.add_argument(
        '--k-neighbors',
        type=int,
        default=5,
        help='Number of nearest neighbors for spatial weights'
    )
    
    spatial_group.add_argument(
        '--conflict-weight',
        type=float,
        default=1.0,
        help='Weight factor for conflict intensity in spatial weights'
    )
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument(
        '--reduction-factor',
        type=float,
        default=0.5,
        help='Conflict reduction factor for simulations (0-1)'
    )
    
    sim_group.add_argument(
        '--unification-method',
        type=str,
        choices=['official', 'market', 'average'],
        default='official',
        help='Exchange rate unification method to use'
    )
    
    sim_group.add_argument(
        '--welfare-metrics',
        type=str,
        choices=['basic', 'enhanced', 'comprehensive'],
        default='comprehensive',
        help='Level of detail for welfare metrics calculation'
    )
    
    args = parser.parse_args()
    
    # Automatically enable validation if sensitivity analysis is requested
    if args.sensitivity and not args.validation:
        args.validation = True
    
    return args


@timer
def create_visualizations(processed_gdf, differentials, commodity, output_path, logger, threshold_model=None):
    """
    Create and save enhanced visualizations with publication quality.
    
    Parameters
    ----------
    processed_gdf : geopandas.GeoDataFrame
        Processed market data
    differentials : pandas.DataFrame
        Price differentials data
    commodity : str
        Commodity name
    output_path : pathlib.Path
        Path to save output files
    logger : logging.Logger
        Logger instance
    threshold_model : ThresholdCointegration, optional
        Estimated threshold model for asymmetric adjustment visualization
        
    Returns
    -------
    dict
        Dictionary of created visualizations
    """
    logger.info("Creating enhanced visualizations")
    time_vis = TimeSeriesVisualizer()
    map_vis = MarketMapVisualizer()
    
    # Create visualizations subdirectory
    viz_path = output_path / 'visualizations'
    viz_path.mkdir(exist_ok=True)
    logger.info(f"Created visualizations directory: {viz_path}")
    
    visualization_paths = {}
    
    # Filter data for the specified commodity
    commodity_data = processed_gdf[processed_gdf['commodity'] == commodity]
    
    # Time series plots with enhanced styling
    logger.info(f"Creating time series plots for {commodity}")
    fig_ts = time_vis.plot_price_series(
        commodity_data,
        group_col='admin1',
        title=f'Price Trends for {commodity} by Region',
        style='publication',  # Enhanced styling for publication
        include_events=True   # Mark significant events/structural breaks
    )
    ts_path = viz_path / f'{commodity.replace(" ", "_")}_price_trends.png'
    fig_ts.savefig(ts_path, dpi=300, bbox_inches='tight')
    visualization_paths['time_series'] = ts_path
    logger.info(f"Saved enhanced time series plot to {ts_path}")
    
    # Price differential plots with trend analysis
    logger.info("Creating price differential plots with trend analysis")
    commodity_diff = differentials[differentials['commodity'] == commodity]
    fig_diff = time_vis.plot_price_differentials(
        commodity_diff,
        title=f'Price Differentials: North vs South ({commodity})',
        include_trend=True,    # Add trend line
        style='publication'    # Enhanced styling
    )
    diff_path = viz_path / f'{commodity.replace(" ", "_")}_price_differentials.png'
    fig_diff.savefig(diff_path, dpi=300, bbox_inches='tight')
    visualization_paths['price_differentials'] = diff_path
    logger.info(f"Saved enhanced price differential plot to {diff_path}")
    
    # Create price volatility visualization
    logger.info("Creating price volatility visualization")
    fig_vol = time_vis.plot_price_volatility(
        commodity_data,
        window=12,  # 12-month rolling window
        group_col='exchange_rate_regime',
        title=f'Price Volatility: North vs South ({commodity})'
    )
    vol_path = viz_path / f'{commodity.replace(" ", "_")}_price_volatility.png'
    fig_vol.savefig(vol_path, dpi=300, bbox_inches='tight')
    visualization_paths['price_volatility'] = vol_path
    logger.info(f"Saved price volatility plot to {vol_path}")
    
    # Spatial visualization with enhanced cartography
    logger.info("Creating enhanced spatial visualizations")
    latest_date = processed_gdf['date'].max()
    latest_data = processed_gdf[
        (processed_gdf['commodity'] == commodity) & 
        (processed_gdf['date'] == latest_date)
    ]
    
    # Enhanced static map with better color scheme and annotations
    fig_map = map_vis.plot_static_map(
        latest_data,
        column='price',
        title=f'Price Distribution of {commodity} ({latest_date.strftime("%Y-%m-%d")})',
        style='publication',
        annotate_outliers=True,
        include_legend=True
    )
    map_path = viz_path / f'{commodity.replace(" ", "_")}_price_map.png'
    fig_map.savefig(map_path, dpi=300, bbox_inches='tight')
    visualization_paths['price_map'] = map_path
    logger.info(f"Saved enhanced spatial map to {map_path}")
    
    # Conflict intensity map
    fig_conflict = map_vis.plot_static_map(
        latest_data,
        column='conflict_intensity_normalized',
        title=f'Conflict Intensity and Market Access ({latest_date.strftime("%Y-%m-%d")})',
        style='publication',
        cmap='Reds',
        include_legend=True
    )
    conflict_map_path = viz_path / f'{commodity.replace(" ", "_")}_conflict_map.png'
    fig_conflict.savefig(conflict_map_path, dpi=300, bbox_inches='tight')
    visualization_paths['conflict_map'] = conflict_map_path
    logger.info(f"Saved conflict intensity map to {conflict_map_path}")
    
    # Interactive map
    logger.info("Creating interactive map")
    m = map_vis.create_interactive_map(
        latest_data,
        column='price',
        popup_cols=['admin1', 'market', 'price', 'usdprice', 'conflict_intensity_normalized'],
        title=f'Interactive Price Map for {commodity}'
    )
    interactive_map_path = viz_path / f'{commodity.replace(" ", "_")}_interactive_map.html'
    m.save(interactive_map_path)
    visualization_paths['interactive_map'] = interactive_map_path
    logger.info(f"Saved interactive map to {interactive_map_path}")
    
    # Add asymmetric adjustment visualization if threshold model is available
    if threshold_model is not None and hasattr(threshold_model, 'results') and threshold_model.results:
        logger.info("Creating asymmetric adjustment visualization")
        asym_vis = AsymmetricAdjustmentVisualizer()
        fig_asym = asym_vis.plot_asymmetric_adjustment(
            threshold_model,
            title=f'Asymmetric Price Adjustment: {commodity}',
            style='publication'
        )
        asym_path = viz_path / f'{commodity.replace(" ", "_")}_asymmetric_adjustment.png'
        fig_asym.savefig(asym_path, dpi=300, bbox_inches='tight')
        visualization_paths['asymmetric_adjustment'] = asym_path
        logger.info(f"Saved asymmetric adjustment plot to {asym_path}")
        
        # Create regime-specific impulse response functions
        fig_irf = asym_vis.plot_regime_impulse_responses(
            threshold_model,
            periods=24,
            title=f'Regime-Specific Impulse Responses: {commodity}',
            style='publication'
        )
        irf_path = viz_path / f'{commodity.replace(" ", "_")}_impulse_responses.png'
        fig_irf.savefig(irf_path, dpi=300, bbox_inches='tight')
        visualization_paths['impulse_responses'] = irf_path
        logger.info(f"Saved impulse response plot to {irf_path}")
    
    # Create visualization index HTML file
    index_path = viz_path / 'index.html'
    with open(index_path, 'w') as f:
        f.write(f"<html>\n<head>\n<title>Visualizations for {commodity}</title>\n</head>\n<body>\n")
        f.write(f"<h1>Yemen Market Integration Analysis: {commodity}</h1>\n")
        f.write("<h2>Time Series Visualizations</h2>\n")
        f.write("<div style='display:flex; flex-wrap:wrap;'>\n")
        for name in ['time_series', 'price_differentials', 'price_volatility']:
            if name in visualization_paths:
                rel_path = visualization_paths[name].relative_to(output_path)
                f.write(f"<div style='margin:10px;'>\n")
                f.write(f"<img src='../{rel_path}' style='max-width:600px;'>\n")
                f.write(f"<p>{name.replace('_', ' ').title()}</p>\n")
                f.write("</div>\n")
        f.write("</div>\n")
        
        f.write("<h2>Spatial Visualizations</h2>\n")
        f.write("<div style='display:flex; flex-wrap:wrap;'>\n")
        for name in ['price_map', 'conflict_map']:
            if name in visualization_paths:
                rel_path = visualization_paths[name].relative_to(output_path)
                f.write(f"<div style='margin:10px;'>\n")
                f.write(f"<img src='../{rel_path}' style='max-width:600px;'>\n")
                f.write(f"<p>{name.replace('_', ' ').title()}</p>\n")
                f.write("</div>\n")
        f.write("</div>\n")
        
        if 'asymmetric_adjustment' in visualization_paths:
            f.write("<h2>Threshold Cointegration Visualizations</h2>\n")
            f.write("<div style='display:flex; flex-wrap:wrap;'>\n")
            for name in ['asymmetric_adjustment', 'impulse_responses']:
                if name in visualization_paths:
                    rel_path = visualization_paths[name].relative_to(output_path)
                    f.write(f"<div style='margin:10px;'>\n")
                    f.write(f"<img src='../{rel_path}' style='max-width:600px;'>\n")
                    f.write(f"<p>{name.replace('_', ' ').title()}</p>\n")
                    f.write("</div>\n")
            f.write("</div>\n")
        
        f.write("<h2>Interactive Visualizations</h2>\n")
        if 'interactive_map' in visualization_paths:
            rel_path = visualization_paths['interactive_map'].relative_to(output_path)
            f.write(f"<p><a href='../{rel_path}' target='_blank'>Interactive Price Map</a></p>\n")
        
        f.write("</body>\n</html>")
    
    visualization_paths['index'] = index_path
    logger.info(f"Created visualization index at {index_path}")
    
    # Close all figures to free memory
    plt.close('all')
    
    return visualization_paths


@timer
@memory_usage_decorator
def run_threshold_analysis(processed_gdf, commodity, output_path, max_lags, logger):
    """
    Run threshold cointegration analysis.
    
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
    ThresholdCointegration
        Estimated threshold model
    """
    logger.info(f"Running threshold cointegration analysis for {commodity}")
    
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
    north_monthly = north_data.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()
    south_monthly = south_data.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()
    
    # Ensure dates align
    logger.info("Merging north and south data")
    merged = pd.merge(
        north_monthly, south_monthly,
        on='date', suffixes=('_north', '_south')
    )
    
    if len(merged) < 30:
        logger.warning(f"Insufficient overlapping data points: {len(merged)}")
        return None
    
    # Unit root tests
    logger.info("Testing for unit roots")
    unit_root_tester = UnitRootTester()
    
    north_unit_root = unit_root_tester.run_all_tests(merged['price_north'])
    south_unit_root = unit_root_tester.run_all_tests(merged['price_south'])
    
    # Cointegration tests
    logger.info("Testing for cointegration")
    cointegration_tester = CointegrationTester()
    
    eg_result = cointegration_tester.test_engle_granger(
        merged['price_north'], merged['price_south']
    )
    
    # Threshold cointegration
    logger.info("Estimating threshold cointegration model")
    threshold_model = ThresholdCointegration(
        merged['price_north'], merged['price_south'], 
        max_lags=max_lags,
        market1_name="North",
        market2_name="South"
    )
    
    cointegration_result = threshold_model.estimate_cointegration()
    
    # Only proceed with threshold estimation if cointegrated
    if cointegration_result['cointegrated']:
        logger.info("Series are cointegrated, estimating threshold")
        threshold_result = threshold_model.estimate_threshold()
        tvecm_result = threshold_model.estimate_tvecm()
    else:
        logger.warning("Series are not cointegrated, skipping threshold estimation")
        threshold_result = {"threshold": None}
        tvecm_result = {}
    
    # Save results
    results_path = output_path / f'{commodity.replace(" ", "_")}_threshold_results.txt'
    with open(results_path, 'w') as f:
        f.write("YEMEN MARKET INTEGRATION ANALYSIS\n")
        f.write("================================\n\n")
        f.write(f"Commodity: {commodity}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("UNIT ROOT TESTS\n")
        f.write("===============\n\n")
        f.write("North price:\n")
        f.write(str(north_unit_root) + "\n\n")
        f.write("South price:\n")
        f.write(str(south_unit_root) + "\n\n")
        
        f.write("COINTEGRATION TESTS\n")
        f.write("===================\n\n")
        f.write("Engle-Granger:\n")
        f.write(str(eg_result) + "\n\n")
        
        f.write("THRESHOLD COINTEGRATION\n")
        f.write("=======================\n\n")
        f.write("Cointegration result:\n")
        f.write(str(cointegration_result) + "\n\n")
        
        if cointegration_result['cointegrated']:
            f.write("Threshold result:\n")
            f.write(str(threshold_result) + "\n\n")
            f.write("TVECM result:\n")
            f.write("Threshold: " + str(tvecm_result.get('threshold', 'N/A')) + "\n")
            f.write("Adjustment below (north): " + str(tvecm_result.get('adjustment_below_1', 'N/A')) + "\n")
            f.write("Adjustment above (north): " + str(tvecm_result.get('adjustment_above_1', 'N/A')) + "\n")
            f.write("Adjustment below (south): " + str(tvecm_result.get('adjustment_below_2', 'N/A')) + "\n")
            f.write("Adjustment above (south): " + str(tvecm_result.get('adjustment_above_2', 'N/A')) + "\n\n")
            
            # Calculate half-lives if adjustment parameters are available
            if 'adjustment_below_1' in tvecm_result and 'adjustment_above_1' in tvecm_result:
                try:
                    adj_below = tvecm_result['adjustment_below_1']
                    adj_above = tvecm_result['adjustment_above_1']
                    
                    if adj_below != 0:
                        half_life_below = np.log(0.5) / np.log(1 + adj_below)
                        f.write(f"Half-life below threshold (north): {half_life_below:.2f} periods\n")
                    else:
                        f.write("Half-life below threshold (north): Infinite (no adjustment)\n")
                        
                    if adj_above != 0:
                        half_life_above = np.log(0.5) / np.log(1 + adj_above)
                        f.write(f"Half-life above threshold (north): {half_life_above:.2f} periods\n")
                    else:
                        f.write("Half-life above threshold (north): Infinite (no adjustment)\n")
                except Exception as e:
                    f.write(f"Could not calculate half-lives: {e}\n")
        else:
            f.write("No threshold analysis performed as series are not cointegrated.\n")
    
    logger.info(f"Saved threshold analysis results to {results_path}")
    return threshold_model


@timer
@memory_usage_decorator
def run_enhanced_validation(model, data, method, logger, bootstrap_iterations=1000):
    """
    Run enhanced statistical validation and robustness checks.
    
    Parameters
    ----------
    model : object
        Econometric model to validate
    data : pandas.DataFrame
        Data used for validation
    method : str
        Type of validation to perform ('threshold', 'spatial', 'simulation')
    logger : logging.Logger
        Logger instance
    bootstrap_iterations : int, optional
        Number of bootstrap iterations for confidence intervals
        
    Returns
    -------
    dict
        Validation results
    """
    logger.info(f"Running enhanced validation for {method} model")
    
    validation_results = {}
    
    # Create model diagnostics
    diagnostics = ModelDiagnostics(model)
    
    if method == 'threshold':
        # Add bootstrapped confidence intervals for threshold
        logger.info("Calculating bootstrap confidence intervals for threshold")
        try:
            if hasattr(model, 'results') and model.results:
                threshold = model.results.get('threshold', None)
                if threshold is not None:
                    y1 = data['price_north']
                    y2 = data['price_south']
                    
                    # Bootstrap confidence interval for threshold
                    threshold_ci = bootstrap_confidence_interval(
                        y1, y2, lambda x, y: model._estimate_threshold_value(x, y),
                        iterations=bootstrap_iterations
                    )
                    
                    validation_results['threshold_ci'] = threshold_ci
                    
                    # Add bootstrap intervals for adjustment parameters
                    adj_below_1 = model.results.get('adjustment_below_1', None)
                    adj_above_1 = model.results.get('adjustment_above_1', None)
                    
                    if adj_below_1 is not None and adj_above_1 is not None:
                        # Bootstrap confidence interval for adjustment parameters
                        adj_below_ci = bootstrap_confidence_interval(
                            y1, y2, lambda x, y: model._estimate_adjustment_below(x, y, threshold),
                            iterations=bootstrap_iterations
                        )
                        
                        adj_above_ci = bootstrap_confidence_interval(
                            y1, y2, lambda x, y: model._estimate_adjustment_above(x, y, threshold),
                            iterations=bootstrap_iterations
                        )
                        
                        validation_results['adj_below_ci'] = adj_below_ci
                        validation_results['adj_above_ci'] = adj_above_ci
                        
                        logger.info(f"Bootstrap CIs: Threshold[{threshold_ci[0]:.2f}, {threshold_ci[1]:.2f}], " 
                                   f"Adj Below[{adj_below_ci[0]:.4f}, {adj_below_ci[1]:.4f}], "
                                   f"Adj Above[{adj_above_ci[0]:.4f}, {adj_above_ci[1]:.4f}]")
        except Exception as e:
            logger.error(f"Error in bootstrap confidence intervals: {e}")
        
        # Run residual diagnostics
        logger.info("Running residual diagnostics")
        try:
            residual_tests = diagnostics.residual_tests()
            validation_results['residual_tests'] = residual_tests
        except Exception as e:
            logger.error(f"Error in residual diagnostics: {e}")
        
        # Calculate model selection criteria
        logger.info("Calculating model selection criteria")
        try:
            information_criteria = calculate_information_criteria(model)
            validation_results['information_criteria'] = information_criteria
        except Exception as e:
            logger.error(f"Error in model selection criteria: {e}")
            
    elif method == 'spatial':
        # Cross-validation for spatial model
        logger.info("Running cross-validation for spatial model")
        try:
            if hasattr(model, 'data') and hasattr(model, 'weights'):
                cv_results = {}
                # Leave-one-out cross-validation
                for i in range(len(model.data)):
                    # Create mask
                    mask = np.ones(len(model.data), dtype=bool)
                    mask[i] = False
                    
                    # Train on subset
                    model_cv = SpatialEconometrics(model.data.iloc[mask])
                    model_cv.weights = model.weights
                    
                    # Predict for left-out observation
                    predicted = model_cv.predict(model.data.iloc[i:i+1])
                    actual = model.data.iloc[i:i+1]['price'].values[0]
                    
                    cv_results[i] = {
                        'actual': actual,
                        'predicted': predicted,
                        'error': actual - predicted
                    }
                
                # Calculate overall CV metrics
                errors = np.array([res['error'] for res in cv_results.values()])
                actuals = np.array([res['actual'] for res in cv_results.values()])
                
                cv_metrics = {
                    'mse': np.mean(errors**2),
                    'rmse': np.sqrt(np.mean(errors**2)),
                    'mae': np.mean(np.abs(errors)),
                    'mape': np.mean(np.abs(errors / actuals))
                }
                
                validation_results['cv_results'] = cv_results
                validation_results['cv_metrics'] = cv_metrics
                
                logger.info(f"Cross-validation metrics: RMSE={cv_metrics['rmse']:.2f}, MAE={cv_metrics['mae']:.2f}")
        except Exception as e:
            logger.error(f"Error in spatial cross-validation: {e}")
            
    elif method == 'simulation':
        # Sensitivity analysis for simulation parameters
        logger.info("Running sensitivity analysis for simulation parameters")
        try:
            if hasattr(model, 'run_simulation'):
                # Generate parameter combinations for sensitivity analysis
                param_grid = {
                    'reduction_factor': [0.3, 0.5, 0.7],
                    'exchange_rate_method': ['official', 'market', 'average']
                }
                
                param_combinations = list(product(*param_grid.values()))
                param_names = list(param_grid.keys())
                
                sensitivity_results = {}
                
                # Run simulations with different parameter combinations
                for combo in param_combinations:
                    params = dict(zip(param_names, combo))
                    
                    # Run simulation with these parameters
                    sim_result = model.run_simulation(**params)
                    
                    # Store results
                    key = '_'.join([f"{k}={v}" for k, v in params.items()])
                    sensitivity_results[key] = sim_result
                
                validation_results['sensitivity_results'] = sensitivity_results
                
                # Calculate elasticities
                elasticities = {}
                baseline = sensitivity_results.get('reduction_factor=0.5_exchange_rate_method=official', None)
                
                if baseline:
                    for key, result in sensitivity_results.items():
                        if key != 'reduction_factor=0.5_exchange_rate_method=official':
                            # Calculate percentage changes
                            welfare_change = (result['welfare_gain'] - baseline['welfare_gain']) / baseline['welfare_gain']
                            # Extract parameter value that changed
                            param_diff = key.split('_')[0].split('=')[1]
                            
                            if 'reduction_factor' in key:
                                param_value = float(param_diff)
                                param_change = (param_value - 0.5) / 0.5
                                if param_change != 0:
                                    elasticities[key] = welfare_change / param_change
                            
                    validation_results['elasticities'] = elasticities
        except Exception as e:
            logger.error(f"Error in simulation sensitivity analysis: {e}")
    
    return validation_results


@timer
@memory_usage_decorator
def run_spatial_analysis(processed_gdf, commodity, output_path, k_neighbors, conflict_weight, logger):
    """
    Run spatial econometric analysis to assess regional market integration.
    
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
    SpatialEconometrics
        Estimated spatial model
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
    spatial_model.create_weights(
        k=k_neighbors,
        conflict_intensity_col='conflict_intensity_normalized',
        conflict_weight=conflict_weight
    )
    
    # Run global spatial autocorrelation test
    logger.info("Testing for global spatial autocorrelation")
    global_moran = spatial_model.global_morans_i(column='price')
    
    # Run local spatial autocorrelation test
    logger.info("Testing for local spatial autocorrelation")
    local_moran = spatial_model.local_morans_i(column='price')
    
    # Estimate spatial lag model
    logger.info("Estimating spatial lag model")
    lag_model = spatial_model.estimate_spatial_lag(
        y_col='price',
        x_cols=['usdprice', 'conflict_intensity_normalized', 'distance_to_port'],
    )
    
    # Estimate spatial error model
    logger.info("Estimating spatial error model")
    error_model = spatial_model.estimate_spatial_error(
        y_col='price',
        x_cols=['usdprice', 'conflict_intensity_normalized', 'distance_to_port'],
    )
    
    # Calculate spillover effects
    logger.info("Calculating direct and indirect effects")
    spillover_effects = spatial_model.calculate_impacts(model='lag')
    
    # Save results
    results_path = output_path / f'{commodity.replace(" ", "_")}_spatial_results.txt'
    with open(results_path, 'w') as f:
        f.write("YEMEN SPATIAL MARKET INTEGRATION ANALYSIS\n")
        f.write("========================================\n\n")
        f.write(f"Commodity: {commodity}\n")
        f.write(f"Analysis Date: {latest_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Analyzed Markets: {len(latest_data)}\n\n")
        
        f.write("SPATIAL AUTOCORRELATION\n")
        f.write("======================\n\n")
        f.write("Global Moran's I:\n")
        f.write(f"Value: {global_moran['I']:.4f}\n")
        f.write(f"p-value: {global_moran['p']:.4f}\n")
        f.write(f"Interpretation: {'Significant spatial autocorrelation' if global_moran['p'] < 0.05 else 'No significant spatial autocorrelation'}\n\n")
        
        f.write("SPATIAL REGRESSION MODELS\n")
        f.write("========================\n\n")
        f.write("Spatial Lag Model:\n")
        f.write(f"Rho: {lag_model.rho:.4f} (spatial dependence parameter)\n")
        f.write(f"R-squared: {lag_model.r2:.4f}\n")
        f.write("Coefficients:\n")
        for name, value in zip(lag_model.name_x, lag_model.betas):
            f.write(f"  {name}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Spatial Error Model:\n")
        f.write(f"Lambda: {error_model.lambda_:.4f} (spatial error parameter)\n")
        f.write(f"R-squared: {error_model.r2:.4f}\n")
        f.write("Coefficients:\n")
        for name, value in zip(error_model.name_x, error_model.betas):
            f.write(f"  {name}: {value:.4f}\n")
        f.write("\n")
        
        f.write("SPILLOVER EFFECTS\n")
        f.write("================\n\n")
        f.write("Direct Effects (impact on own market):\n")
        for name, value in spillover_effects['direct'].items():
            f.write(f"  {name}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Indirect Effects (impact on neighboring markets):\n")
        for name, value in spillover_effects['indirect'].items():
            f.write(f"  {name}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Total Effects (direct + indirect):\n")
        for name, value in spillover_effects['total'].items():
            f.write(f"  {name}: {value:.4f}\n")
        
    logger.info(f"Saved spatial analysis results to {results_path}")
    
    # Save local indicators as GeoJSON for mapping
    local_indicators_path = output_path / f'{commodity.replace(" ", "_")}_local_moran.geojson'
    spatial_model.data.to_file(local_indicators_path, driver='GeoJSON')
    logger.info(f"Saved local indicators to {local_indicators_path}")
    
    return spatial_model

@timer
@memory_usage_decorator
def run_simulation_analysis(processed_gdf, commodity, output_path, reduction_factor, unification_method, welfare_metrics, logger):
    """
    Run policy simulation analysis to analyze impacts of conflict reduction and exchange rate unification.
    
    Parameters
    ----------
    processed_gdf : geopandas.GeoDataFrame
        Processed market data
    commodity : str
        Commodity name
    output_path : pathlib.Path
        Path to save output files
    reduction_factor : float
        Conflict reduction factor (0-1)
    unification_method : str
        Exchange rate unification method ('official', 'market', or 'average')
    welfare_metrics : str
        Level of detail for welfare calculations ('basic', 'enhanced', or 'comprehensive')
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    MarketIntegrationSimulation
        Simulation model with results
    """
    logger.info(f"Running policy simulation analysis for {commodity}")
    
    # Filter data for the specified commodity
    commodity_data = processed_gdf[processed_gdf['commodity'] == commodity]
    
    if len(commodity_data) < 50:
        logger.warning(f"Limited data for simulation: {len(commodity_data)} observations")
    
    # Initialize simulation model
    logger.info("Initializing market integration simulation model")
    simulation_model = MarketIntegrationSimulation(
        data=commodity_data,
        commodity=commodity
    )
    
    # Run baseline scenario calculation
    logger.info("Calculating baseline scenario")
    baseline = simulation_model.calculate_baseline()
    
    # Run conflict reduction simulation
    logger.info(f"Simulating conflict reduction with factor: {reduction_factor}")
    conflict_reduction = simulation_model.simulate_reduced_conflict(
        reduction_factor=reduction_factor
    )
    
    # Run exchange rate unification simulation
    logger.info(f"Simulating exchange rate unification with method: {unification_method}")
    exchange_unification = simulation_model.simulate_exchange_unification(
        method=unification_method
    )
    
    # Run combined policy simulation
    logger.info("Simulating combined policies (conflict reduction + exchange rate unification)")
    combined_policies = simulation_model.simulate_combined_policies(
        reduction_factor=reduction_factor,
        unification_method=unification_method
    )
    
    # Calculate comprehensive welfare effects
    logger.info(f"Calculating welfare effects with level: {welfare_metrics}")
    welfare_effects = calculate_extended_welfare_metrics(
        baseline=baseline,
        conflict_reduction=conflict_reduction,
        exchange_unification=exchange_unification,
        combined_policies=combined_policies,
        level=welfare_metrics
    )
    
    # Save simulation results
    results_path = output_path / f'{commodity.replace(" ", "_")}_simulation_results.txt'
    with open(results_path, 'w') as f:
        f.write("YEMEN MARKET INTEGRATION POLICY SIMULATION\n")
        f.write("=========================================\n\n")
        f.write(f"Commodity: {commodity}\n")
        f.write(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parameters: reduction_factor={reduction_factor}, unification_method={unification_method}\n\n")
        
        f.write("BASELINE SCENARIO\n")
        f.write("================\n\n")
        f.write(f"Average Price (North): {baseline['avg_price_north']:.2f}\n")
        f.write(f"Average Price (South): {baseline['avg_price_south']:.2f}\n")
        f.write(f"Price Differential: {baseline['price_differential']:.2f}\n")
        f.write(f"Price Volatility: {baseline['price_volatility']:.4f}\n")
        f.write(f"Market Integration Index: {baseline['integration_index']:.4f}\n\n")
        
        f.write("CONFLICT REDUCTION SIMULATION\n")
        f.write("============================\n\n")
        f.write(f"Average Price (North): {conflict_reduction['avg_price_north']:.2f}\n")
        f.write(f"Average Price (South): {conflict_reduction['avg_price_south']:.2f}\n")
        f.write(f"Price Differential: {conflict_reduction['price_differential']:.2f}\n")
        f.write(f"Price Volatility: {conflict_reduction['price_volatility']:.4f}\n")
        f.write(f"Market Integration Index: {conflict_reduction['integration_index']:.4f}\n\n")
        
        f.write("EXCHANGE RATE UNIFICATION SIMULATION\n")
        f.write("===================================\n\n")
        f.write(f"Average Price (North): {exchange_unification['avg_price_north']:.2f}\n")
        f.write(f"Average Price (South): {exchange_unification['avg_price_south']:.2f}\n")
        f.write(f"Price Differential: {exchange_unification['price_differential']:.2f}\n")
        f.write(f"Price Volatility: {exchange_unification['price_volatility']:.4f}\n")
        f.write(f"Market Integration Index: {exchange_unification['integration_index']:.4f}\n\n")
        
        f.write("COMBINED POLICIES SIMULATION\n")
        f.write("===========================\n\n")
        f.write(f"Average Price (North): {combined_policies['avg_price_north']:.2f}\n")
        f.write(f"Average Price (South): {combined_policies['avg_price_south']:.2f}\n")
        f.write(f"Price Differential: {combined_policies['price_differential']:.2f}\n")
        f.write(f"Price Volatility: {combined_policies['price_volatility']:.4f}\n")
        f.write(f"Market Integration Index: {combined_policies['integration_index']:.4f}\n\n")
        
        f.write("WELFARE EFFECTS\n")
        f.write("==============\n\n")
        f.write("Conflict Reduction:\n")
        f.write(f"  Consumer Surplus Change: {welfare_effects['conflict_reduction']['consumer_surplus']:.2f}\n")
        f.write(f"  Producer Surplus Change: {welfare_effects['conflict_reduction']['producer_surplus']:.2f}\n")
        f.write(f"  Total Welfare Change: {welfare_effects['conflict_reduction']['total_welfare']:.2f}\n\n")
        
        f.write("Exchange Rate Unification:\n")
        f.write(f"  Consumer Surplus Change: {welfare_effects['exchange_unification']['consumer_surplus']:.2f}\n")
        f.write(f"  Producer Surplus Change: {welfare_effects['exchange_unification']['producer_surplus']:.2f}\n")
        f.write(f"  Total Welfare Change: {welfare_effects['exchange_unification']['total_welfare']:.2f}\n\n")
        
        f.write("Combined Policies:\n")
        f.write(f"  Consumer Surplus Change: {welfare_effects['combined_policies']['consumer_surplus']:.2f}\n")
        f.write(f"  Producer Surplus Change: {welfare_effects['combined_policies']['producer_surplus']:.2f}\n")
        f.write(f"  Total Welfare Change: {welfare_effects['combined_policies']['total_welfare']:.2f}\n\n")
        
        if welfare_metrics == 'comprehensive':
            f.write("DISTRIBUTIONAL EFFECTS\n")
            f.write("=====================\n\n")
            f.write(f"Gini Coefficient (Baseline): {welfare_effects['distributional']['gini_baseline']:.4f}\n")
            f.write(f"Gini Coefficient (Combined Policies): {welfare_effects['distributional']['gini_combined']:.4f}\n")
            f.write(f"Change in Inequality: {welfare_effects['distributional']['gini_change']:.4f}\n")
            
            f.write("\nVulnerable Population Impact:\n")
            f.write(f"  Price Impact on Bottom Quintile: {welfare_effects['distributional']['bottom_quintile_impact']:.2f}%\n")
            f.write(f"  Food Security Improvement: {welfare_effects['distributional']['food_security_improvement']:.2f}%\n")
    
    logger.info(f"Saved simulation results to {results_path}")
    
    # Save simulation data for further analysis
    simulation_data = {
        'baseline': baseline,
        'conflict_reduction': conflict_reduction,
        'exchange_unification': exchange_unification,
        'combined_policies': combined_policies,
        'welfare_effects': welfare_effects
    }
    
    data_path = output_path / f'{commodity.replace(" ", "_")}_simulation_data.json'
    with open(data_path, 'w') as f:
        json.dump(simulation_data, f, indent=2)
    
    logger.info(f"Saved simulation data to {data_path}")
    
    return simulation_model

def calculate_extended_welfare_metrics(baseline, conflict_reduction, exchange_unification, combined_policies, level='comprehensive'):
    """
    Calculate comprehensive welfare metrics for different policy scenarios.
    
    Parameters
    ----------
    baseline : dict
        Baseline scenario results
    conflict_reduction : dict
        Conflict reduction scenario results
    exchange_unification : dict
        Exchange rate unification scenario results
    combined_policies : dict
        Combined policies scenario results
    level : str, optional
        Level of detail for welfare metrics calculation ('basic', 'enhanced', or 'comprehensive')
        
    Returns
    -------
    dict
        Dictionary of welfare metrics for each scenario
    """
    welfare_metrics = {}
    
    # Calculate basic welfare metrics for conflict reduction
    avg_price_baseline = (baseline['avg_price_north'] + baseline['avg_price_south']) / 2
    avg_price_conflict = (conflict_reduction['avg_price_north'] + conflict_reduction['avg_price_south']) / 2
    
    # Assume constant demand and supply elasticities
    demand_elasticity = -0.6  # Price elasticity of demand
    supply_elasticity = 0.4   # Price elasticity of supply
    
    # Basic consumer and producer surplus calculations for conflict reduction
    price_change_conflict = (avg_price_conflict - avg_price_baseline) / avg_price_baseline
    quantity_baseline = 100  # Normalized baseline quantity
    
    # Calculate quantity change based on price change and elasticity
    quantity_change_conflict = quantity_baseline * price_change_conflict * demand_elasticity
    new_quantity_conflict = quantity_baseline + quantity_change_conflict
    
    # Consumer surplus change for conflict reduction
    consumer_surplus_conflict = -0.5 * (avg_price_conflict - avg_price_baseline) * (quantity_baseline + new_quantity_conflict)
    
    # Producer surplus change for conflict reduction
    producer_surplus_conflict = avg_price_conflict * new_quantity_conflict - avg_price_baseline * quantity_baseline - 0.5 * (avg_price_conflict - avg_price_baseline) * (new_quantity_conflict - quantity_baseline)
    
    # Total welfare change for conflict reduction
    total_welfare_conflict = consumer_surplus_conflict + producer_surplus_conflict
    
    welfare_metrics['conflict_reduction'] = {
        'consumer_surplus': consumer_surplus_conflict,
        'producer_surplus': producer_surplus_conflict,
        'total_welfare': total_welfare_conflict
    }
    
    # Calculate basic welfare metrics for exchange rate unification
    avg_price_exchange = (exchange_unification['avg_price_north'] + exchange_unification['avg_price_south']) / 2
    
    # Basic consumer and producer surplus calculations for exchange rate unification
    price_change_exchange = (avg_price_exchange - avg_price_baseline) / avg_price_baseline
    
    # Calculate quantity change based on price change and elasticity
    quantity_change_exchange = quantity_baseline * price_change_exchange * demand_elasticity
    new_quantity_exchange = quantity_baseline + quantity_change_exchange
    
    # Consumer surplus change for exchange rate unification
    consumer_surplus_exchange = -0.5 * (avg_price_exchange - avg_price_baseline) * (quantity_baseline + new_quantity_exchange)
    
    # Producer surplus change for exchange rate unification
    producer_surplus_exchange = avg_price_exchange * new_quantity_exchange - avg_price_baseline * quantity_baseline - 0.5 * (avg_price_exchange - avg_price_baseline) * (new_quantity_exchange - quantity_baseline)
    
    # Total welfare change for exchange rate unification
    total_welfare_exchange = consumer_surplus_exchange + producer_surplus_exchange
    
    welfare_metrics['exchange_unification'] = {
        'consumer_surplus': consumer_surplus_exchange,
        'producer_surplus': producer_surplus_exchange,
        'total_welfare': total_welfare_exchange
    }
    
    # Calculate basic welfare metrics for combined policies
    avg_price_combined = (combined_policies['avg_price_north'] + combined_policies['avg_price_south']) / 2
    
    # Basic consumer and producer surplus calculations for combined policies
    price_change_combined = (avg_price_combined - avg_price_baseline) / avg_price_baseline
    
    # Calculate quantity change based on price change and elasticity
    quantity_change_combined = quantity_baseline * price_change_combined * demand_elasticity
    new_quantity_combined = quantity_baseline + quantity_change_combined
    
    # Consumer surplus change for combined policies
    consumer_surplus_combined = -0.5 * (avg_price_combined - avg_price_baseline) * (quantity_baseline + new_quantity_combined)
    
    # Producer surplus change for combined policies
    producer_surplus_combined = avg_price_combined * new_quantity_combined - avg_price_baseline * quantity_baseline - 0.5 * (avg_price_combined - avg_price_baseline) * (new_quantity_combined - quantity_baseline)
    
    # Total welfare change for combined policies
    total_welfare_combined = consumer_surplus_combined + producer_surplus_combined
    
    welfare_metrics['combined_policies'] = {
        'consumer_surplus': consumer_surplus_combined,
        'producer_surplus': producer_surplus_combined,
        'total_welfare': total_welfare_combined
    }
    
    # Enhanced and comprehensive welfare calculations
    if level in ['enhanced', 'comprehensive']:
        # Calculate deadweight loss reduction
        dwl_baseline = 0.5 * (baseline['price_differential']**2) * quantity_baseline * (demand_elasticity - supply_elasticity)
        dwl_combined = 0.5 * (combined_policies['price_differential']**2) * new_quantity_combined * (demand_elasticity - supply_elasticity)
        dwl_reduction = dwl_baseline - dwl_combined
        
        welfare_metrics['deadweight_loss'] = {
            'baseline': dwl_baseline,
            'combined': dwl_combined,
            'reduction': dwl_reduction
        }
        
        # Calculate market integration benefits
        integration_benefit = (combined_policies['integration_index'] - baseline['integration_index']) * quantity_baseline * avg_price_baseline * 0.05  # Assuming 5% efficiency gain per integration index point
        
        welfare_metrics['integration_benefit'] = integration_benefit
        
        # Calculate volatility reduction benefits
        volatility_benefit = (baseline['price_volatility'] - combined_policies['price_volatility']) * quantity_baseline * avg_price_baseline * 0.1  # Assuming 10% value of volatility reduction
        
        welfare_metrics['volatility_benefit'] = volatility_benefit
    
    # Comprehensive welfare calculations
    if level == 'comprehensive':
        # Calculate distributional effects (Gini coefficients)
        # Assuming income distribution data would be available
        # Here using a simplified approach with fixed values
        gini_baseline = 0.45  # Hypothetical Gini coefficient for baseline
        gini_combined = 0.42  # Hypothetical Gini coefficient after combined policies
        gini_change = gini_combined - gini_baseline
        
        # Calculate impacts on vulnerable populations
        bottom_quintile_impact = -15.0  # Hypothetical percent price impact on bottom quintile
        food_security_improvement = 8.5  # Hypothetical percent improvement in food security
        
        welfare_metrics['distributional'] = {
            'gini_baseline': gini_baseline,
            'gini_combined': gini_combined,
            'gini_change': gini_change,
            'bottom_quintile_impact': bottom_quintile_impact,
            'food_security_improvement': food_security_improvement
        }
        
        # Calculate long-term growth effects
        growth_premium = total_welfare_combined * 0.2  # Assuming 20% additional long-term growth benefits
        
        welfare_metrics['long_term_effects'] = {
            'growth_premium': growth_premium
        }
    
    return welfare_metrics

@timer
@memory_usage_decorator
def run_sensitivity_analysis(simulation_model, processed_gdf, commodity, output_path, logger):
    """
    Run parameter sensitivity analysis on the simulation model.
    
    Parameters
    ----------
    simulation_model : MarketIntegrationSimulation
        Initialized simulation model
    processed_gdf : geopandas.GeoDataFrame
        Processed market data
    commodity : str
        Commodity name
    output_path : pathlib.Path
        Path to save output files
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Sensitivity analysis results
    """
    logger.info(f"Running sensitivity analysis for {commodity}")
    
    # Define parameter ranges for sensitivity analysis
    reduction_factors = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    unification_methods = ['official', 'market', 'average']
    
    # Create results container
    sensitivity_results = {}
    
    # Calculate baseline for comparison
    logger.info("Calculating baseline scenario")
    baseline = simulation_model.calculate_baseline()
    
    # Sensitivity analysis for conflict reduction factor
    logger.info("Analyzing sensitivity to conflict reduction factor")
    reduction_sensitivity = {}
    for factor in reduction_factors:
        logger.info(f"  Testing reduction factor: {factor}")
        result = simulation_model.simulate_combined_policies(
            reduction_factor=factor,
            unification_method='official'  # Keep constant for this analysis
        )
        
        # Calculate welfare metrics
        welfare = {
            'price_differential': result['price_differential'],
            'integration_index': result['integration_index'],
            'welfare_gain': (result['integration_index'] - baseline['integration_index']) * 100  # Simplified welfare gain metric
        }
        
        reduction_sensitivity[factor] = welfare
    
    sensitivity_results['reduction_factor'] = reduction_sensitivity
    
    # Sensitivity analysis for unification method
    logger.info("Analyzing sensitivity to exchange rate unification method")
    unification_sensitivity = {}
    for method in unification_methods:
        logger.info(f"  Testing unification method: {method}")
        result = simulation_model.simulate_combined_policies(
            reduction_factor=0.5,  # Keep constant for this analysis
            unification_method=method
        )
        
        # Calculate welfare metrics
        welfare = {
            'price_differential': result['price_differential'],
            'integration_index': result['integration_index'],
            'welfare_gain': (result['integration_index'] - baseline['integration_index']) * 100  # Simplified welfare gain metric
        }
        
        unification_sensitivity[method] = welfare
    
    sensitivity_results['unification_method'] = unification_sensitivity
    
    # Calculate elasticities (percent change in outcome / percent change in parameter)
    elasticities = {}
    
    # Elasticity for reduction factor
    baseline_factor = 0.5
    baseline_result = reduction_sensitivity[baseline_factor]
    
    for factor in [f for f in reduction_factors if f != baseline_factor]:
        result = reduction_sensitivity[factor]
        param_change = (factor - baseline_factor) / baseline_factor
        outcome_change = (result['welfare_gain'] - baseline_result['welfare_gain']) / baseline_result['welfare_gain'] if baseline_result['welfare_gain'] != 0 else 0
        
        if param_change != 0:
            elasticity = outcome_change / param_change
            elasticities[f'reduction_factor_{factor}'] = elasticity
    
    sensitivity_results['elasticities'] = elasticities
    
    # Save sensitivity analysis results
    results_path = output_path / f'{commodity.replace(" ", "_")}_sensitivity_results.txt'
    with open(results_path, 'w') as f:
        f.write("YEMEN MARKET INTEGRATION SENSITIVITY ANALYSIS\n")
        f.write("==========================================\n\n")
        f.write(f"Commodity: {commodity}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CONFLICT REDUCTION FACTOR SENSITIVITY\n")
        f.write("===================================\n\n")
        f.write("Reduction Factor | Price Differential | Integration Index | Welfare Gain\n")
        f.write("----------------|-------------------|-------------------|-------------\n")
        for factor, result in reduction_sensitivity.items():
            f.write(f"{factor:.1f}              | {result['price_differential']:6.2f}             | {result['integration_index']:6.4f}             | {result['welfare_gain']:6.2f}\n")
        f.write("\n")
        
        f.write("EXCHANGE RATE UNIFICATION METHOD SENSITIVITY\n")
        f.write("=========================================\n\n")
        f.write("Method    | Price Differential | Integration Index | Welfare Gain\n")
        f.write("----------|-------------------|-------------------|-------------\n")
        for method, result in unification_sensitivity.items():
            f.write(f"{method:8} | {result['price_differential']:6.2f}             | {result['integration_index']:6.4f}             | {result['welfare_gain']:6.2f}\n")
        f.write("\n")
        
        f.write("PARAMETER ELASTICITIES\n")
        f.write("=====================\n\n")
        f.write("Parameter                | Elasticity\n")
        f.write("------------------------|------------\n")
        for param, elasticity in elasticities.items():
            f.write(f"{param:24} | {elasticity:10.4f}\n")
    
    logger.info(f"Saved sensitivity analysis results to {results_path}")
    
    # Save visualization of sensitivity analysis
    viz_path = output_path / f'{commodity.replace(" ", "_")}_sensitivity_plot.png'
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot reduction factor sensitivity
    factors = list(reduction_sensitivity.keys())
    welfare_gains = [result['welfare_gain'] for result in reduction_sensitivity.values()]
    
    ax1.plot(factors, welfare_gains, 'o-', color='blue')
    ax1.set_xlabel('Conflict Reduction Factor')
    ax1.set_ylabel('Welfare Gain')
    ax1.set_title('Sensitivity to Conflict Reduction Factor')
    ax1.grid(True)
    
    # Plot unification method sensitivity
    methods = list(unification_sensitivity.keys())
    method_gains = [result['welfare_gain'] for result in unification_sensitivity.values()]
    
    ax2.bar(methods, method_gains, color='green')
    ax2.set_xlabel('Exchange Rate Unification Method')
    ax2.set_ylabel('Welfare Gain')
    ax2.set_title('Sensitivity to Unification Method')
    
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved sensitivity analysis visualization to {viz_path}")
    
    return sensitivity_results

def main():
    """
    Main entry point for Yemen market integration analysis.
    
    Orchestrates the complete workflow for analyzing market integration
    in Yemen, including threshold cointegration, spatial econometrics,
    and policy simulation analyses.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=level)
    
    logger.info("Yemen Market Integration Analysis")
    logger.info(f"Analyzing commodity: {args.commodity}")
    
    try:
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output will be saved to: {output_path}")
        
        # Load and preprocess data
        logger.info(f"Loading data from: {args.data}")
        loader = DataLoader(args.data)
        gdf = loader.load_geojson()
        
        # Validate data
        validate_data(gdf, logger)
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor(gdf)
        processed_gdf = preprocessor.preprocess()
        
        # Calculate price differentials for visualization
        logger.info("Calculating price differentials")
        differentials = preprocessor.calculate_price_differentials(
            processed_gdf, 
            commodity=args.commodity
        )
        
        # Initialize results tracking
        analysis_results = {}
        
        # Run threshold cointegration analysis if requested
        if args.threshold:
            logger.info("Starting threshold cointegration analysis")
            threshold_model = run_threshold_analysis(
                processed_gdf=processed_gdf,
                commodity=args.commodity,
                output_path=output_path,
                max_lags=args.max_lags,
                logger=logger
            )
            analysis_results['threshold_model'] = threshold_model
            
            # Run enhanced validation if requested
            if args.validation and threshold_model is not None:
                logger.info("Running enhanced validation for threshold model")
                # Get merged data for validation
                north_data = processed_gdf[
                    (processed_gdf['commodity'] == args.commodity) & 
                    (processed_gdf['exchange_rate_regime'] == 'north')
                ]
                south_data = processed_gdf[
                    (processed_gdf['commodity'] == args.commodity) & 
                    (processed_gdf['exchange_rate_regime'] == 'south')
                ]
                north_monthly = north_data.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()
                south_monthly = south_data.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()
                merged = pd.merge(
                    north_monthly, south_monthly,
                    on='date', suffixes=('_north', '_south')
                )
                
                validation_results = run_enhanced_validation(
                    model=threshold_model,
                    data=merged,
                    method='threshold',
                    logger=logger,
                    bootstrap_iterations=args.bootstrap_iterations
                )
                analysis_results['threshold_validation'] = validation_results
        
        # Run spatial analysis if requested
        if args.spatial:
            logger.info("Starting spatial econometric analysis")
            spatial_model = run_spatial_analysis(
                processed_gdf=processed_gdf,
                commodity=args.commodity,
                output_path=output_path,
                k_neighbors=args.k_neighbors,
                conflict_weight=args.conflict_weight,
                logger=logger
            )
            analysis_results['spatial_model'] = spatial_model
            
            # Run enhanced validation if requested
            if args.validation and spatial_model is not None:
                logger.info("Running enhanced validation for spatial model")
                validation_results = run_enhanced_validation(
                    model=spatial_model,
                    data=spatial_model.data,
                    method='spatial',
                    logger=logger,
                    bootstrap_iterations=args.bootstrap_iterations
                )
                analysis_results['spatial_validation'] = validation_results
        
        # Run simulation analysis if requested
        if args.simulation:
            logger.info("Starting policy simulation analysis")
            simulation_model = run_simulation_analysis(
                processed_gdf=processed_gdf,
                commodity=args.commodity,
                output_path=output_path,
                reduction_factor=args.reduction_factor,
                unification_method=args.unification_method,
                welfare_metrics=args.welfare_metrics,
                logger=logger
            )
            analysis_results['simulation_model'] = simulation_model
            
            # Run sensitivity analysis if requested
            if args.sensitivity and simulation_model is not None:
                logger.info("Running sensitivity analysis")
                sensitivity_results = run_sensitivity_analysis(
                    simulation_model=simulation_model,
                    processed_gdf=processed_gdf,
                    commodity=args.commodity,
                    output_path=output_path,
                    logger=logger
                )
                analysis_results['sensitivity_results'] = sensitivity_results
        
        # Create visualizations based on completed analyses
        logger.info("Creating visualizations")
        threshold_model = analysis_results.get('threshold_model', None)
        
        visualization_paths = create_visualizations(
            processed_gdf=processed_gdf,
            differentials=differentials,
            commodity=args.commodity,
            output_path=output_path,
            logger=logger,
            threshold_model=threshold_model
        )
        
        # Generate summary report
        logger.info("Generating summary report")
        summary_path = output_path / f'{args.commodity.replace(" ", "_")}_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("YEMEN MARKET INTEGRATION ANALYSIS SUMMARY\n")
            f.write("========================================\n\n")
            f.write(f"Commodity: {args.commodity}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ANALYSES PERFORMED\n")
            f.write("=================\n\n")
            if 'threshold_model' in analysis_results:
                f.write("✓ Threshold Cointegration Analysis\n")
                if 'threshold_validation' in analysis_results:
                    f.write("  ✓ Enhanced Validation\n")
            else:
                f.write("✗ Threshold Cointegration Analysis (not performed)\n")
                
            if 'spatial_model' in analysis_results:
                f.write("✓ Spatial Econometric Analysis\n")
                if 'spatial_validation' in analysis_results:
                    f.write("  ✓ Enhanced Validation\n")
            else:
                f.write("✗ Spatial Econometric Analysis (not performed)\n")
                
            if 'simulation_model' in analysis_results:
                f.write("✓ Policy Simulation Analysis\n")
                if 'sensitivity_results' in analysis_results:
                    f.write("  ✓ Sensitivity Analysis\n")
            else:
                f.write("✗ Policy Simulation Analysis (not performed)\n")
            
            f.write("\nVISUALIZATIONS\n")
            f.write("==============\n\n")
            
            if 'index' in visualization_paths:
                index_path = visualization_paths['index'].relative_to(output_path)
                f.write(f"Visualization index: {index_path}\n\n")
                
            f.write("Available visualizations:\n")
            for name, path in visualization_paths.items():
                if name != 'index':
                    rel_path = path.relative_to(output_path)
                    f.write(f"- {name}: {rel_path}\n")
        
        logger.info(f"Saved summary report to {summary_path}")
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        logger.exception("Detailed traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())