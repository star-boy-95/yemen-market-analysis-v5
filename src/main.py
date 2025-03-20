"""
Main script for Yemen market integration analysis.

This script serves as the command-line interface for the Yemen Market Integration
project, orchestrating the complete analysis workflow. It allows users to run
different types of analyses (threshold cointegration, spatial econometrics,
policy simulation) on market data, generate visualizations, and export results.

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
import argparse
import logging
import time
from datetime import datetime

# Import project modules
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.unit_root import UnitRootTester
from models.cointegration import CointegrationTester
from models.threshold import ThresholdCointegration
from models.threshold_vecm import ThresholdVECM
from models.spatial import SpatialEconometrics
from models.simulation import MarketIntegrationSimulation
from visualization.time_series import TimeSeriesVisualizer
from visualization.maps import MarketMapVisualizer
from utils.performance_utils import timer, memory_usage_decorator


def setup_logging(log_file='yemen_analysis.log'):
    """
    Set up logging configuration.
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create log file path
    log_path = log_dir / log_file
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Yemen Market Integration Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
        '--threshold', 
        action='store_true',
        help='Run threshold cointegration analysis'
    )
    
    parser.add_argument(
        '--spatial', 
        action='store_true',
        help='Run spatial econometric analysis'
    )
    
    parser.add_argument(
        '--simulation', 
        action='store_true',
        help='Run policy simulations'
    )
    
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
        '--reduction-factor',
        type=float,
        default=0.5,
        help='Conflict reduction factor for simulations (0-1)'
    )
    
    return parser.parse_args()


@timer
def create_visualizations(processed_gdf, differentials, commodity, output_path, logger):
    """
    Create and save visualizations.
    
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
        
    Returns
    -------
    dict
        Dictionary of created visualizations
    """
    logger.info("Creating visualizations")
    time_vis = TimeSeriesVisualizer()
    map_vis = MarketMapVisualizer()
    
    # Filter data for the specified commodity
    commodity_data = processed_gdf[processed_gdf['commodity'] == commodity]
    
    # Time series plots
    logger.info(f"Creating time series plots for {commodity}")
    fig_ts = time_vis.plot_price_series(
        commodity_data,
        group_col='admin1',
        title=f'Price Trends for {commodity} by Region'
    )
    ts_path = output_path / f'{commodity.replace(" ", "_")}_price_trends.png'
    fig_ts.savefig(ts_path)
    logger.info(f"Saved time series plot to {ts_path}")
    
    # Price differential plots
    logger.info("Creating price differential plots")
    commodity_diff = differentials[differentials['commodity'] == commodity]
    fig_diff = time_vis.plot_price_differentials(
        commodity_diff,
        title=f'Price Differentials: North vs South ({commodity})'
    )
    diff_path = output_path / f'{commodity.replace(" ", "_")}_price_differentials.png'
    fig_diff.savefig(diff_path)
    logger.info(f"Saved price differential plot to {diff_path}")
    
    # Spatial visualization
    logger.info("Creating spatial visualizations")
    latest_date = processed_gdf['date'].max()
    latest_data = processed_gdf[
        (processed_gdf['commodity'] == commodity) & 
        (processed_gdf['date'] == latest_date)
    ]
    
    fig_map = map_vis.plot_static_map(
        latest_data,
        column='price',
        title=f'Price Distribution of {commodity} ({latest_date.strftime("%Y-%m-%d")})'
    )
    map_path = output_path / f'{commodity.replace(" ", "_")}_price_map.png'
    fig_map.savefig(map_path)
    logger.info(f"Saved spatial map to {map_path}")
    
    # Interactive map
    logger.info("Creating interactive map")
    m = map_vis.create_interactive_map(
        latest_data,
        column='price',
        popup_cols=['admin1', 'price', 'usdprice', 'conflict_intensity_normalized'],
        title=f'Interactive Price Map for {commodity}'
    )
    interactive_map_path = output_path / f'{commodity.replace(" ", "_")}_interactive_map.html'
    m.save(interactive_map_path)
    logger.info(f"Saved interactive map to {interactive_map_path}")
    
    # Close all figures to free memory
    plt.close('all')
    
    return {
        'time_series': ts_path,
        'price_differentials': diff_path,
        'price_map': map_path,
        'interactive_map': interactive_map_path
    }


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
def run_spatial_analysis(processed_gdf, commodity, output_path, k_neighbors, logger):
    """
    Run spatial econometric analysis.
    
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
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    SpatialEconometrics
        Spatial econometrics model
    """
    logger.info(f"Running spatial econometric analysis for {commodity}")
    
    # Get latest data for spatial analysis
    latest_date = processed_gdf['date'].max()
    spatial_data = processed_gdf[
        (processed_gdf['commodity'] == commodity) & 
        (processed_gdf['date'] == latest_date)
    ]
    
    if len(spatial_data) < 5:
        logger.warning(f"Insufficient spatial data points: {len(spatial_data)}")
        return None
    
    # Create spatial econometrics model
    logger.info("Creating spatial econometrics model")
    spatial_model = SpatialEconometrics(spatial_data)
    
    # Create weight matrices
    logger.info("Creating spatial weight matrices")
    w_standard = spatial_model.create_weight_matrix(
        k=k_neighbors, conflict_adjusted=False
    )
    w_conflict = spatial_model.create_weight_matrix(
        k=k_neighbors, conflict_adjusted=True
    )
    
    # Test for spatial autocorrelation
    logger.info("Testing for spatial autocorrelation with standard weights")
    moran_standard = spatial_model.moran_i_test('price')
    
    # Reset weights to conflict-adjusted
    spatial_model.weights = w_conflict
    logger.info("Testing for spatial autocorrelation with conflict-adjusted weights")
    moran_conflict = spatial_model.moran_i_test('price')
    
    # Estimate spatial lag model
    logger.info("Estimating spatial lag model")
    x_vars = ['conflict_intensity_normalized']
    if 'exchange_rate_regime' in spatial_data.columns:
        # Create dummy for exchange rate regime
        spatial_data['north_regime'] = (spatial_data['exchange_rate_regime'] == 'north').astype(int)
        x_vars.append('north_regime')
    
    try:
        spatial_lag = spatial_model.spatial_lag_model('price', x_vars)
        spatial_error = spatial_model.spatial_error_model('price', x_vars)
    except Exception as e:
        logger.error(f"Error estimating spatial models: {e}")
        spatial_lag = None
        spatial_error = None
    
    # Save results
    results_path = output_path / f'{commodity.replace(" ", "_")}_spatial_results.txt'
    with open(results_path, 'w') as f:
        f.write("YEMEN MARKET INTEGRATION ANALYSIS\n")
        f.write("================================\n\n")
        f.write(f"Commodity: {commodity}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Date: {latest_date.strftime('%Y-%m-%d')}\n\n")
        
        f.write("SPATIAL AUTOCORRELATION\n")
        f.write("======================\n\n")
        f.write("Standard weights:\n")
        f.write(f"Moran's I: {moran_standard['I']:.4f}\n")
        f.write(f"p-value: {moran_standard['p_norm']:.4f}\n")
        f.write(f"z-score: {moran_standard['z_norm']:.4f}\n")
        f.write(f"Significant: {moran_standard['p_norm'] < 0.05}\n\n")
        
        f.write("Conflict-adjusted weights:\n")
        f.write(f"Moran's I: {moran_conflict['I']:.4f}\n")
        f.write(f"p-value: {moran_conflict['p_norm']:.4f}\n")
        f.write(f"z-score: {moran_conflict['z_norm']:.4f}\n")
        f.write(f"Significant: {moran_conflict['p_norm'] < 0.05}\n\n")
        
        f.write("SPATIAL LAG MODEL\n")
        f.write("================\n\n")
        if spatial_lag:
            f.write(str(spatial_lag.summary) + "\n\n")
            
            # Add interpretation
            f.write("INTERPRETATION:\n")
            f.write(f"Spatial autoregressive parameter (rho): {spatial_lag.rho:.4f}\n")
            if spatial_lag.rho > 0 and spatial_lag.z_stat[-1] < 0.05:
                f.write("Significant positive spatial dependence detected. ")
                f.write("This indicates that prices in neighboring markets influence each other.\n\n")
            elif spatial_lag.rho < 0 and spatial_lag.z_stat[-1] < 0.05:
                f.write("Significant negative spatial dependence detected. ")
                f.write("This indicates a competitive relationship between neighboring markets.\n\n")
            else:
                f.write("No significant spatial dependence detected.\n\n")
        else:
            f.write("Could not estimate spatial lag model.\n\n")
        
        f.write("SPATIAL ERROR MODEL\n")
        f.write("==================\n\n")
        if spatial_error:
            f.write(str(spatial_error.summary) + "\n\n")
            
            # Add interpretation
            f.write("INTERPRETATION:\n")
            f.write(f"Spatial error parameter (lambda): {spatial_error.lam:.4f}\n")
            if spatial_error.lam > 0 and spatial_error.z_stat[-1] < 0.05:
                f.write("Significant positive spatial error correlation detected. ")
                f.write("This suggests that unobserved factors affecting prices are spatially correlated.\n\n")
            elif spatial_error.lam < 0 and spatial_error.z_stat[-1] < 0.05:
                f.write("Significant negative spatial error correlation detected. ")
                f.write("This suggests that unobserved factors have opposing effects in neighboring markets.\n\n")
            else:
                f.write("No significant spatial error correlation detected.\n\n")
        else:
            f.write("Could not estimate spatial error model.\n\n")
        
        # Add conflict impact analysis
        f.write("CONFLICT IMPACT ANALYSIS\n")
        f.write("=======================\n\n")
        f.write("Comparison of spatial autocorrelation with and without conflict adjustment:\n")
        moran_diff = moran_standard['I'] - moran_conflict['I']
        f.write(f"Difference in Moran's I: {moran_diff:.4f}\n")
        if abs(moran_diff) > 0.1:
            f.write("Substantial difference detected, indicating that conflict significantly ")
            f.write("alters the spatial structure of market relationships.\n")
            if moran_diff > 0:
                f.write("Conflict appears to weaken spatial price relationships.\n")
            else:
                f.write("Conflict appears to strengthen spatial price relationships, possibly ")
                f.write("due to increased reliance on specific trade routes.\n")
        else:
            f.write("Minimal difference detected, suggesting that conflict has limited impact ")
            f.write("on the spatial structure of market relationships for this commodity.\n")
    
    logger.info(f"Saved spatial analysis results to {results_path}")
    return spatial_model


@timer
@memory_usage_decorator
def run_simulations(processed_gdf, threshold_model, spatial_model, commodity, 
                   output_path, reduction_factor, logger):
    """
    Run policy simulations.
    
    Parameters
    ----------
    processed_gdf : geopandas.GeoDataFrame
        Processed market data
    threshold_model : ThresholdCointegration
        Estimated threshold model
    spatial_model : SpatialEconometrics
        Spatial econometrics model
    commodity : str
        Commodity name
    output_path : pathlib.Path
        Path to save output files
    reduction_factor : float
        Conflict reduction factor (0-1)
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Simulation results
    """
    logger.info(f"Running policy simulations for {commodity}")
    
    # Filter data for the specified commodity
    commodity_data = processed_gdf[processed_gdf['commodity'] == commodity].copy()
    
    # Create simulation model
    logger.info("Creating simulation model")
    sim_model = MarketIntegrationSimulation(
        commodity_data, threshold_model=threshold_model, spatial_model=spatial_model
    )
    
    # Initialize results dictionary
    simulation_results = {}
    
    # Simulate exchange rate unification
    if threshold_model is not None and threshold_model.results is not None:
        logger.info("Simulating exchange rate unification")
        try:
            unified_result = sim_model.simulate_exchange_rate_unification()
            simulation_results['exchange_rate_unification'] = unified_result
            
            # Save unified data
            unified_data = unified_result['data']
            unified_file = output_path / f'{commodity.replace(" ", "_")}_unified_exchange_rate_data.geojson'
            unified_data.to_file(unified_file, driver='GeoJSON')
            logger.info(f"Saved unified exchange rate data to {unified_file}")
        except Exception as e:
            logger.error(f"Error in exchange rate unification simulation: {e}")
    else:
        logger.warning("Skipping exchange rate unification simulation: No valid threshold model")
    
    # Simulate improved connectivity
    if spatial_model is not None:
        logger.info(f"Simulating improved connectivity with reduction factor {reduction_factor}")
        try:
            connectivity_result = sim_model.simulate_improved_connectivity(
                reduction_factor=reduction_factor
            )
            simulation_results['improved_connectivity'] = connectivity_result
            
            # Save connectivity data
            connectivity_data = connectivity_result['simulated_data']
            connectivity_file = output_path / f'{commodity.replace(" ", "_")}_improved_connectivity_data.geojson'
            connectivity_data.to_file(connectivity_file, driver='GeoJSON')
            logger.info(f"Saved improved connectivity data to {connectivity_file}")
        except Exception as e:
            logger.error(f"Error in improved connectivity simulation: {e}")
    else:
        logger.warning("Skipping improved connectivity simulation: No valid spatial model")
    
    # Simulate combined policy if both models are available
    if threshold_model is not None and spatial_model is not None:
        logger.info("Simulating combined policy intervention")
        try:
            combined_result = sim_model.simulate_combined_policy(
                exchange_rate_target='official',
                conflict_reduction=reduction_factor
            )
            simulation_results['combined_policy'] = combined_result
            
            # Save combined policy data
            combined_data = combined_result['simulated_data']
            combined_file = output_path / f'{commodity.replace(" ", "_")}_combined_policy_data.geojson'
            combined_data.to_file(combined_file, driver='GeoJSON')
            logger.info(f"Saved combined policy data to {combined_file}")
        except Exception as e:
            logger.error(f"Error in combined policy simulation: {e}")
    else:
        logger.warning("Skipping combined policy simulation: Missing threshold or spatial model")
    
    # Calculate welfare effects
    logger.info("Calculating welfare effects")
    welfare_results = {}
    
    for policy_name, result in simulation_results.items():
        try:
            welfare = sim_model.calculate_welfare_effects(policy_name)
            welfare_results[policy_name] = welfare
        except Exception as e:
            logger.error(f"Error calculating welfare effects for {policy_name}: {e}")
    
    # Save simulation results
    results_path = output_path / f'{commodity.replace(" ", "_")}_simulation_results.txt'
    with open(results_path, 'w') as f:
        f.write("YEMEN MARKET INTEGRATION ANALYSIS\n")
        f.write("================================\n\n")
        f.write(f"Commodity: {commodity}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("POLICY SIMULATION RESULTS\n")
        f.write("========================\n\n")
        
        # Exchange rate unification
        f.write("1. EXCHANGE RATE UNIFICATION SIMULATION\n")
        f.write("-------------------------------------\n\n")
        if 'exchange_rate_unification' in simulation_results:
            result = simulation_results['exchange_rate_unification']
            f.write(f"Unified exchange rate: {result.get('unified_rate', 'N/A')}\n\n")
            
            if 'exchange_rate_unification' in welfare_results:
                welfare = welfare_results['exchange_rate_unification']
                f.write("Welfare effects:\n")
                f.write(f"Price convergence: {welfare.get('price_convergence', 'N/A')}\n")
                f.write(f"Price dispersion reduction: {welfare.get('dispersion_reduction', 'N/A')}\n\n")
        else:
            f.write("Simulation not performed or failed.\n\n")
        
        # Improved connectivity
        f.write("2. IMPROVED CONNECTIVITY SIMULATION\n")
        f.write("----------------------------------\n\n")
        if 'improved_connectivity' in simulation_results:
            result = simulation_results['improved_connectivity']
            f.write(f"Conflict reduction factor: {reduction_factor}\n\n")
            
            if 'improved_connectivity' in welfare_results:
                welfare = welfare_results['improved_connectivity']
                f.write("Welfare effects:\n")
                f.write(f"Market accessibility improvement: {welfare.get('accessibility_improvement', 'N/A')}\n")
                f.write(f"Spatial integration improvement: {welfare.get('integration_improvement', 'N/A')}\n\n")
        else:
            f.write("Simulation not performed or failed.\n\n")
        
        # Combined policy
        f.write("3. COMBINED POLICY SIMULATION\n")
        f.write("----------------------------\n\n")
        if 'combined_policy' in simulation_results:
            result = simulation_results['combined_policy']
            f.write(f"Unified exchange rate: {result.get('unified_rate', 'N/A')}\n")
            f.write(f"Conflict reduction factor: {reduction_factor}\n\n")
            
            if 'combined_policy' in welfare_results:
                welfare = welfare_results['combined_policy']
                f.write("Welfare effects:\n")
                f.write(f"Price convergence: {welfare.get('price_convergence', 'N/A')}\n")
                f.write(f"Market accessibility improvement: {welfare.get('accessibility_improvement', 'N/A')}\n")
                f.write(f"Overall welfare improvement: {welfare.get('overall_improvement', 'N/A')}\n\n")
        else:
            f.write("Simulation not performed or failed.\n\n")
        
        # Policy comparison
        if len(welfare_results) > 1:
            f.write("4. POLICY COMPARISON\n")
            f.write("-------------------\n\n")
            f.write("Comparison of welfare effects across policies:\n\n")
            
            # Create a simple comparison table
            f.write("Policy                    | Price Convergence | Market Accessibility | Overall\n")
            f.write("-------------------------|------------------|---------------------|--------\n")
            
            for policy, welfare in welfare_results.items():
                price_conv = welfare.get('price_convergence', 'N/A')
                access_imp = welfare.get('accessibility_improvement', 'N/A')
                overall = welfare.get('overall_improvement', 'N/A')
                
                f.write(f"{policy.ljust(25)} | {str(price_conv).ljust(18)} | {str(access_imp).ljust(21)} | {overall}\n")
            
            f.write("\nRECOMMENDATION:\n")
            # Simple recommendation logic
            if 'combined_policy' in welfare_results:
                f.write("The combined policy approach generally yields the most comprehensive benefits,\n")
                f.write("addressing both exchange rate disparities and conflict-related market fragmentation.\n")
            elif 'exchange_rate_unification' in welfare_results and 'improved_connectivity' in welfare_results:
                er_welfare = welfare_results['exchange_rate_unification'].get('overall_improvement', 0)
                ic_welfare = welfare_results['improved_connectivity'].get('overall_improvement', 0)
                
                if er_welfare > ic_welfare:
                    f.write("Exchange rate unification appears to yield greater welfare benefits than\n")
                    f.write("conflict reduction alone, suggesting prioritization of monetary policy.\n")
                else:
                    f.write("Conflict reduction appears to yield greater welfare benefits than\n")
                    f.write("exchange rate unification alone, suggesting prioritization of security improvements.\n")
    
    logger.info(f"Saved simulation results to {results_path}")
    return simulation_results


@timer
def main():
    """Main entry point for the analysis."""
    # Record start time
    start_time = time.time()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Yemen market integration analysis")
    
    # Parse arguments
    args = parse_args()
    logger.info(f"Command line arguments: {args}")
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    loader = DataLoader()
    try:
        gdf = loader.load_geojson(args.data)
        logger.info(f"Loaded data with {len(gdf)} records")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    try:
        processed_gdf = preprocessor.preprocess_geojson(gdf)
        logger.info(f"Preprocessed data with {len(processed_gdf)} records")
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return
    
    # Save processed data
    processed_file = output_path / 'processed_data.geojson'
    processed_gdf.to_file(processed_file, driver='GeoJSON')
    logger.info(f"Saved processed data to {processed_file}")
    
    # Calculate price differentials
    logger.info("Calculating price differentials between north and south")
    try:
        differentials = preprocessor.calculate_price_differentials(processed_gdf)
        diff_file = output_path / 'price_differentials.csv'
        differentials.to_csv(diff_file, index=False)
        logger.info(f"Saved price differentials to {diff_file}")
    except Exception as e:
        logger.error(f"Error calculating price differentials: {e}")
        differentials = pd.DataFrame()
    
    # Create visualizations
    try:
        viz_results = create_visualizations(
            processed_gdf, differentials, args.commodity, output_path, logger
        )
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        viz_results = {}
    
    # Run threshold cointegration analysis if requested
    threshold_model = None
    if args.threshold:
        try:
            threshold_model = run_threshold_analysis(
                processed_gdf, args.commodity, output_path, args.max_lags, logger
            )
        except Exception as e:
            logger.error(f"Error in threshold analysis: {e}")
    
    # Run spatial analysis if requested
    spatial_model = None
    if args.spatial:
        try:
            spatial_model = run_spatial_analysis(
                processed_gdf, args.commodity, output_path, args.k_neighbors, logger
            )
        except Exception as e:
            logger.error(f"Error in spatial analysis: {e}")
    
    # Run policy simulations if requested
    if args.simulation:
        try:
            simulation_results = run_simulations(
                processed_gdf, threshold_model, spatial_model, 
                args.commodity, output_path, args.reduction_factor, logger
            )
        except Exception as e:
            logger.error(f"Error in policy simulations: {e}")
    
    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(f"Analysis completed in {execution_time:.2f} seconds")
    
    # Create summary report
    summary_path = output_path / f'{args.commodity.replace(" ", "_")}_analysis_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("YEMEN MARKET INTEGRATION ANALYSIS SUMMARY\n")
        f.write("========================================\n\n")
        f.write(f"Commodity: {args.commodity}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        
        f.write("ANALYSIS COMPONENTS\n")
        f.write("==================\n\n")
        f.write(f"Data Processing: {'✓' if processed_file.exists() else '✗'}\n")
        f.write(f"Price Differentials: {'✓' if len(differentials) > 0 else '✗'}\n")
        f.write(f"Visualizations: {'✓' if viz_results else '✗'}\n")
        f.write(f"Threshold Analysis: {'✓' if threshold_model is not None else '✗'}\n")
        f.write(f"Spatial Analysis: {'✓' if spatial_model is not None else '✗'}\n")
        f.write(f"Policy Simulations: {'✓' if args.simulation else '✗'}\n\n")
        
        f.write("OUTPUT FILES\n")
        f.write("============\n\n")
        for file in output_path.glob(f'*{args.commodity.replace(" ", "_")}*'):
            f.write(f"- {file.name}\n")
    
    logger.info(f"Saved analysis summary to {summary_path}")
    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main() 