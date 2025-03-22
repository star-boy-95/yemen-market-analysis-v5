#!/usr/bin/env python
"""
Yemen Market Analysis - Unified Analysis Script

This script provides a comprehensive econometric analysis of market integration in Yemen,
merging the functionality of run_yemen_market_analysis.py and run_comprehensive_analysis.py
with enhanced reporting features aligned with both World Bank standards and academic
publication requirements.

Features:
- Unified data loading and preprocessing
- Comprehensive model selection and comparison
- Enhanced reporting with formal hypothesis testing
- Cross-commodity comparative analysis
- Publication-quality visualizations with statistical significance indicators
- Support for multiple threshold model types (standard, fixed, VECM, MTAR)

Example usage:
    # Analyze a single commodity
    python run_yemen_analysis_unified.py --commodity "beans (kidney red)" --output results
    
    # Analyze multiple commodities with model comparison
    python run_yemen_analysis_unified.py --commodities "wheat" "rice" --comparative-analysis --output results
    
    # Full batch analysis across all commodities and models
    python run_yemen_analysis_unified.py --commodities "wheat" "rice" --comparative-analysis --output results
    
    # Generate publication-quality reports with enhanced econometric tables
    python run_yemen_analysis_unified.py --commodity "wheat" --publication-quality --econometric-tables --output publication

DEPRECATION NOTICE: This script replaces run_yemen_market_analysis.py, run_comprehensive_analysis.py,
and run_integrated_analysis.py.
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import json
import warnings
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import project modules
from src.data.loader import DataLoader
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Importing unit root testing functionality")

# Import UnitRootTester class instead of non-existent test_unit_root function
from src.models.unit_root import UnitRootTester

# Create a wrapper function to maintain compatibility with existing code
def test_unit_root(data, config):
    """Wrapper function for UnitRootTester to maintain compatibility."""
    logger.debug(f"Creating UnitRootTester instance and running tests on data shape: {data.shape}")
    tester = UnitRootTester()
    results = {}
    
    # Test each series in the dataframe
    for column in data.columns:
        series = data[column]
        logger.debug(f"Testing series: {column}")
        # Run ADF test for each series
        adf_result = tester.test_adf(series)
        results[f"{column}_adf"] = adf_result
        results[f"{column}_stationary"] = adf_result['stationary']
    
    logger.debug(f"Unit root testing completed with results: {results.keys()}")
    return results
# Import CointegrationTester class instead of non-existent test_cointegration function
from src.models.cointegration import CointegrationTester
# Also import the actual test_cointegration function from utils
from src.utils.stats_utils import test_cointegration as stats_test_cointegration

# Create a wrapper function to maintain compatibility with existing code
def test_cointegration(data, config):
    """Wrapper function for cointegration testing to maintain compatibility."""
    logger.debug(f"Setting up cointegration testing for data shape: {data.shape}")
    
    # Initialize results dictionary
    results = {
        'cointegrated': False,
        'method': 'johansen',
        'pairs_tested': []
    }
    
    # Get list of columns (markets)
    markets = data.columns.tolist()
    
    # Test cointegration for each pair of markets
    for i in range(len(markets)):
        for j in range(i+1, len(markets)):
            market1 = markets[i]
            market2 = markets[j]
            
            logger.debug(f"Testing cointegration between {market1} and {market2}")
            
            # Use the actual test_cointegration function from stats_utils
            pair_result = stats_test_cointegration(
                data[market1],
                data[market2],
                method='engle-granger'
            )
            
            # Store results
            pair_info = {
                'market1': market1,
                'market2': market2,
                'cointegrated': pair_result['cointegrated'],
                'statistic': pair_result['statistic'],
                'pvalue': pair_result['pvalue'],
                'critical_values': pair_result['critical_values']
            }
            
            results['pairs_tested'].append(pair_info)
            
            # If any pair is cointegrated, set the overall result to True
            if pair_result['cointegrated']:
                results['cointegrated'] = True
    
    logger.debug(f"Cointegration testing completed with {len(results['pairs_tested'])} pairs tested")
    return results
# Import ThresholdModel class instead of non-existent run_threshold_analysis function
from src.models.threshold_model import ThresholdModel

# Create a wrapper function to maintain compatibility with existing code
def run_threshold_analysis(data, config):
    """Wrapper function for ThresholdModel to maintain compatibility."""
    logger.debug(f"Setting up threshold analysis for data shape: {data.shape}")
    
    # Initialize results dictionary
    results = {}
    
    # Get list of columns (markets)
    markets = data.columns.tolist()
    
    # We need at least two markets for threshold analysis
    if len(markets) < 2:
        logger.warning(f"Not enough markets for threshold analysis: {len(markets)}")
        return {'error': 'Not enough markets for threshold analysis'}
    
    # For simplicity, we'll use the first two markets
    market1 = markets[0]
    market2 = markets[1]
    
    logger.debug(f"Running threshold analysis between {market1} and {market2}")
    
    # Create ThresholdModel instance
    model = ThresholdModel(
        data[market1],
        data[market2],
        mode="standard",
        market1_name=market1,
        market2_name=market2
    )
    
    # Run full analysis
    analysis_results = model.run_full_analysis()
    
    logger.debug(f"Threshold analysis completed with results: {list(analysis_results.keys())}")
    
    return analysis_results
# Import ThresholdFixed function instead of non-existent run_fixed_threshold_analysis function
from src.models.threshold_fixed import ThresholdFixed

# Create a wrapper function to maintain compatibility with existing code
def run_fixed_threshold_analysis(data, config):
    """Wrapper function for ThresholdFixed to maintain compatibility."""
    logger.debug(f"Setting up fixed threshold analysis for data shape: {data.shape}")
    
    # Initialize results dictionary
    results = {}
    
    # Get list of columns (markets)
    markets = data.columns.tolist()
    
    # We need at least two markets for threshold analysis
    if len(markets) < 2:
        logger.warning(f"Not enough markets for fixed threshold analysis: {len(markets)}")
        return {'error': 'Not enough markets for fixed threshold analysis'}
    
    # For simplicity, we'll use the first two markets
    market1 = markets[0]
    market2 = markets[1]
    
    logger.debug(f"Running fixed threshold analysis between {market1} and {market2}")
    
    # Create ThresholdModel instance using the ThresholdFixed wrapper
    model = ThresholdFixed(
        data[market1],
        data[market2],
        market1_name=market1,
        market2_name=market2
    )
    
    # Run full analysis
    analysis_results = model.run_full_analysis()
    
    logger.debug(f"Fixed threshold analysis completed with results: {list(analysis_results.keys())}")
    
    return analysis_results
# Import ThresholdVECM function instead of non-existent run_threshold_vecm_analysis function
from src.models.threshold_vecm import ThresholdVECM

# Create a wrapper function to maintain compatibility with existing code
def run_threshold_vecm_analysis(data, config):
    """Wrapper function for ThresholdVECM to maintain compatibility."""
    logger.debug(f"Setting up threshold VECM analysis for data shape: {data.shape}")
    
    # Initialize results dictionary
    results = {}
    
    # Get list of columns (markets)
    markets = data.columns.tolist()
    
    # We need at least two markets for VECM analysis
    if len(markets) < 2:
        logger.warning(f"Not enough markets for threshold VECM analysis: {len(markets)}")
        return {'error': 'Not enough markets for threshold VECM analysis'}
    
    logger.debug(f"Running threshold VECM analysis with {len(markets)} markets")
    
    # Create ThresholdModel instance using the ThresholdVECM wrapper
    model = ThresholdVECM(
        data,
        k_ar_diff=config.get('vecm.k_ar_diff', 2),
        deterministic=config.get('vecm.deterministic', 'ci'),
        coint_rank=config.get('vecm.coint_rank', 1),
        market_names=markets
    )
    
    # Run full analysis
    analysis_results = model.run_full_analysis()
    
    logger.debug(f"Threshold VECM analysis completed with results: {list(analysis_results.keys())}")
    
    return analysis_results
# Import ModelComparer class instead of non-existent select_best_model and compare_models functions
from src.models.model_selection import ModelComparer

# Create wrapper functions to maintain compatibility with existing code
def select_best_model(models, criterion='aic'):
    """Wrapper function for ModelComparer to maintain compatibility."""
    logger.debug(f"Selecting best model using criterion: {criterion}")
    
    # Create a ModelComparer instance
    comparer = ModelComparer(
        data=None,  # We'll handle the data directly in the wrapper
        model_specs=[{'name': name, 'model_class': type(model)} for name, model in models.items()]
    )
    
    # Manually add the fitted models to the comparer
    comparer.fitted_models = models
    
    # Add results structure expected by get_best_model
    comparer.results = {
        name: {
            'model': model,
            'model_info': comparer._extract_model_info(model)
        }
        for name, model in models.items()
    }
    
    # Get the best model
    best_model_name = comparer.get_best_model(criterion=criterion, return_model=False)
    
    logger.debug(f"Selected best model: {best_model_name}")
    
    return models.get(best_model_name)

def compare_models(models, data=None, criteria=None):
    """Wrapper function for ModelComparer to maintain compatibility."""
    logger.debug(f"Comparing {len(models)} models")
    
    # Create a ModelComparer instance
    comparer = ModelComparer(
        data=data if data is not None else pd.DataFrame(),  # Empty DataFrame if no data provided
        model_specs=[{'name': name, 'model_class': type(model)} for name, model in models.items()]
    )
    
    # Manually add the fitted models to the comparer
    comparer.fitted_models = models
    
    # Add results structure expected by get_comprehensive_report
    comparer.results = {
        name: {
            'model': model,
            'model_info': comparer._extract_model_info(model)
        }
        for name, model in models.items()
    }
    
    # Get comprehensive report
    report = comparer.get_comprehensive_report()
    
    logger.debug(f"Model comparison completed with results: {list(report.keys())}")
    
    return report
# Import _run_spatial_analysis function and create a wrapper function
from src.models.spatial import _run_spatial_analysis, SpatialEconometrics

# Create a wrapper function to maintain compatibility with existing code
def run_spatial_analysis(data, config):
    """Wrapper function for spatial analysis to maintain compatibility."""
    logger.debug(f"Setting up spatial analysis for data shape: {data.shape}")
    
    # Initialize results dictionary
    results = {}
    
    # Check if we have geographic data
    if not hasattr(data, 'geometry'):
        logger.warning("Data does not have geometry column, cannot perform spatial analysis")
        return {'error': 'Data does not have geometry column'}
    
    # Create SpatialEconometrics instance
    spatial_model = SpatialEconometrics(data)
    
    # Create weight matrix
    conflict_col = config.get('spatial.conflict_column', 'conflict_intensity_normalized')
    conflict_weight = config.get('spatial.conflict_weight', 0.5)
    
    logger.debug(f"Creating weight matrix with conflict column: {conflict_col}")
    weights = spatial_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=conflict_col,
        conflict_weight=conflict_weight
    )
    
    # Get price column
    price_col = config.get('spatial.price_column', 'price')
    
    # Prepare simulation data with reduced conflict
    conflict_reduction = config.get('spatial.conflict_reduction', 0.5)
    sim_data = _prepare_simulation_data(data, conflict_reduction, conflict_col)
    
    # Run spatial analysis
    results = _run_spatial_analysis(
        data, sim_data, conflict_col, price_col,
        spatial_model, conflict_reduction, results
    )
    
    logger.debug(f"Spatial analysis completed with results: {list(results.keys())}")
    
    return results
# Import MarketIntegrationSimulation class instead of non-existent run_policy_simulation function
from src.models.simulation import MarketIntegrationSimulation

# Create a wrapper function to maintain compatibility with existing code
def run_policy_simulation(data, config):
    """Wrapper function for MarketIntegrationSimulation to maintain compatibility."""
    logger.debug(f"Setting up policy simulation for data shape: {data.shape}")
    
    # Initialize results dictionary
    results = {}
    
    # Check if we have geographic data
    if not hasattr(data, 'geometry'):
        logger.warning("Data does not have geometry column, cannot perform spatial simulation")
        return {'error': 'Data does not have geometry column'}
    
    # Create MarketIntegrationSimulation instance
    simulation = MarketIntegrationSimulation(data)
    
    # Determine which policy to simulate based on config
    policy_type = config.get('simulation.policy_type', 'exchange_rate')
    
    if policy_type == 'exchange_rate':
        # Simulate exchange rate unification
        target_rate = config.get('simulation.target_rate', 'official')
        reference_date = config.get('simulation.reference_date', None)
        
        logger.debug(f"Running exchange rate unification simulation with target rate: {target_rate}")
        results = simulation.simulate_exchange_rate_unification(
            target_rate=target_rate,
            reference_date=reference_date
        )
    elif policy_type == 'connectivity':
        # Simulate improved connectivity
        reduction_factor = config.get('simulation.reduction_factor', 0.5)
        
        logger.debug(f"Running improved connectivity simulation with reduction factor: {reduction_factor}")
        results = simulation.simulate_improved_connectivity(
            reduction_factor=reduction_factor
        )
    elif policy_type == 'combined':
        # Simulate combined policy
        target_rate = config.get('simulation.target_rate', 'official')
        reduction_factor = config.get('simulation.reduction_factor', 0.5)
        reference_date = config.get('simulation.reference_date', None)
        
        logger.debug(f"Running combined policy simulation with target rate: {target_rate} and reduction factor: {reduction_factor}")
        results = simulation.simulate_combined_policy(
            exchange_rate_target=target_rate,
            conflict_reduction=reduction_factor,
            reference_date=reference_date
        )
    else:
        logger.warning(f"Unknown policy type: {policy_type}")
        return {'error': f'Unknown policy type: {policy_type}'}
    
    # Calculate welfare effects
    welfare_results = simulation.calculate_welfare_effects()
    results['welfare'] = welfare_results
    
    logger.debug(f"Policy simulation completed with results: {list(results.keys())}")
    
    return results
from src.models.reporting import (
    generate_comprehensive_report,
    create_executive_summary,
    export_results_for_publication
)
from src.visualization.enhanced_econometric_reporting import (
    generate_enhanced_report,
    generate_cross_commodity_comparison
)
from src.utils.logging_setup import setup_logging
from src.utils.config import initialize_config as load_config
from src.utils.file_utils import ensure_dir

# Constants
DEFAULT_CONFIG_PATH = "config/settings.yaml"
DEFAULT_OUTPUT_DIR = "results"
AVAILABLE_THRESHOLD_MODES = ["standard", "fixed", "vecm", "mtar", "all"]
AVAILABLE_FORMATS = ["markdown", "latex", "html", "json"]
AVAILABLE_STYLES = ["world_bank", "academic", "policy"]

# Setup logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Yemen Market Analysis - Unified Analysis Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data selection arguments
    data_group = parser.add_argument_group("Data Selection")
    commodity_group = data_group.add_mutually_exclusive_group(required=True)
    commodity_group.add_argument(
        "--commodity",
        type=str,
        help="Single commodity to analyze"
    )
    commodity_group.add_argument(
        "--commodities",
        type=str,
        nargs="+",
        help="List of commodities to analyze"
    )
    commodity_group.add_argument(
        "--all-commodities",
        action="store_true",
        help="Analyze all available commodities"
    )
    data_group.add_argument(
        "--start-date",
        type=str,
        help="Start date for analysis (YYYY-MM-DD)"
    )
    data_group.add_argument(
        "--end-date",
        type=str,
        help="End date for analysis (YYYY-MM-DD)"
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--threshold-modes",
        type=str,
        nargs="+",
        choices=AVAILABLE_THRESHOLD_MODES,
        default=["standard"],
        help="Threshold model types to use"
    )
    analysis_group.add_argument(
        "--comparative-analysis",
        action="store_true",
        help="Perform cross-commodity comparative analysis"
    )
    analysis_group.add_argument(
        "--spatial-analysis",
        action="store_true",
        help="Include spatial analysis"
    )
    analysis_group.add_argument(
        "--policy-simulation",
        action="store_true",
        help="Run policy simulation"
    )
    analysis_group.add_argument(
        "--confidence-level",
        type=float,
        choices=[0.90, 0.95, 0.99],
        default=0.95,
        help="Confidence level for intervals"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results"
    )
    output_group.add_argument(
        "--format",
        type=str,
        choices=AVAILABLE_FORMATS,
        default="markdown",
        help="Output format for reports"
    )
    output_group.add_argument(
        "--publication-quality",
        action="store_true",
        help="Generate publication-quality outputs"
    )
    output_group.add_argument(
        "--econometric-tables",
        action="store_true",
        help="Generate formal econometric tables"
    )
    output_group.add_argument(
        "--style",
        type=str,
        choices=AVAILABLE_STYLES,
        default="world_bank",
        help="Visual style for reports"
    )
    
    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--parallel",
        action="store_true",
        help="Run analysis in parallel (for multiple commodities)"
    )
    perf_group.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes for parallel execution"
    )
    
    # Configuration options
    config_group = parser.add_argument_group("Configuration Options")
    config_group.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file"
    )
    config_group.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    config_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment for analysis."""
    # Configure logging
    log_level = "DEBUG" if args.debug else args.log_level
    log_dir = Path(args.output)
    log_file = "analysis.log"
    setup_logging(str(log_dir), log_level, log_file)
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up matplotlib for publication quality if requested
    if args.publication_quality:
        plt.style.use('seaborn-v0_8-whitegrid')  # Updated style name for newer matplotlib
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
    
    # Suppress warnings if not in debug mode
    if not args.debug:
        warnings.filterwarnings("ignore")
    
    return config, output_dir


def get_commodities(args, config):
    """Get the list of commodities to analyze."""
    if args.commodity:
        return [args.commodity]
    elif args.commodities:
        return args.commodities
    elif args.all_commodities:
        # Load all available commodities from data
        try:
            data_path = config.get('data', {}).get('processed_path', 'data/processed/prices.csv')
            df = pd.read_csv(data_path)
            return sorted(df['commodity'].unique().tolist())
        except Exception as e:
            logger.error(f"Failed to load commodities: {e}")
            return []
    else:
        return []


def expand_threshold_modes(threshold_modes):
    """Expand 'all' to include all available threshold modes."""
    if 'all' in threshold_modes:
        return [mode for mode in AVAILABLE_THRESHOLD_MODES if mode != 'all']
    return threshold_modes


def run_analysis_for_commodity(
    commodity,
    data,
    config,
    output_dir,
    threshold_modes,
    include_spatial=False,
    include_simulation=False,
    publication_quality=False,
    econometric_tables=False,
    format='markdown',
    style='world_bank',
    confidence_level=0.95,
    start_date=None,
    end_date=None,
    debug=False
):
    """Run the complete analysis pipeline for a single commodity."""
    start_time = time.time()
    logger.info(f"Starting analysis for {commodity}")
    
    # Create commodity-specific output directory
    commodity_dir = output_dir / commodity.replace(' ', '_')
    ensure_dir(commodity_dir)
    
    # Initialize results dictionary
    all_results = {
        'commodity': commodity,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'threshold_modes': threshold_modes,
            'spatial_analysis': include_spatial,
            'policy_simulation': include_simulation,
            'publication_quality': publication_quality,
            'format': format,
            'style': style,
            'confidence_level': confidence_level
        }
    }
    
    try:
        # Step 1: Preprocess data
        logger.info(f"Preprocessing data for {commodity}")
        data_loader = DataLoader(config.get('data', {}).get('path', './data'))
        
        # Filter by date if specified
        if start_date or end_date:
            data = data_loader.filter_by_date(data, start_date, end_date)
        
        # Preprocess the data
        preprocessed_data = data_loader.preprocess_commodity_data(data, commodity)
        all_results['data_summary'] = {
            'n_observations': len(preprocessed_data),
            'start_date': preprocessed_data.index.min().strftime('%Y-%m-%d'),
            'end_date': preprocessed_data.index.max().strftime('%Y-%m-%d'),
            'markets': list(preprocessed_data.columns)
        }
        
        # Step 2: Unit root testing
        logger.info(f"Running unit root tests for {commodity}")
        unit_root_results = test_unit_root(preprocessed_data, config)
        all_results['unit_root_results'] = unit_root_results
        
        # Step 3: Cointegration testing
        logger.info(f"Running cointegration tests for {commodity}")
        cointegration_results = test_cointegration(preprocessed_data, config)
        all_results['cointegration_results'] = cointegration_results
        
        # Step 4: Threshold analysis with multiple models
        logger.info(f"Running threshold analysis for {commodity} with modes: {threshold_modes}")
        threshold_results = {}
        model_comparison = {'information_criteria': {}, 'performance_metrics': {}}
        
        for mode in threshold_modes:
            logger.info(f"Running {mode} threshold model for {commodity}")
            
            if mode == 'standard':
                results = run_threshold_analysis(preprocessed_data, config)
            elif mode == 'fixed':
                results = run_fixed_threshold_analysis(preprocessed_data, config)
            elif mode == 'vecm':
                results = run_threshold_vecm_analysis(preprocessed_data, config)
            elif mode == 'mtar':
                # Placeholder for MTAR model
                results = run_threshold_analysis(preprocessed_data, config, model_type='mtar')
            else:
                logger.warning(f"Unknown threshold mode: {mode}")
                continue
            
            threshold_results[mode] = results
            
            # Extract information criteria and performance metrics for comparison
            if 'information_criteria' in results:
                model_comparison['information_criteria'][mode] = results['information_criteria']
            
            if 'performance_metrics' in results:
                model_comparison['performance_metrics'][mode] = results['performance_metrics']
        
        # Step 5: Model selection
        logger.info(f"Selecting best model for {commodity}")
        best_model = select_best_model(model_comparison)
        model_comparison['best_model'] = best_model
        
        # Add threshold results to all_results
        all_results['threshold_analysis'] = {
            'models': threshold_results,
            'model_comparison': model_comparison,
            'best_model': best_model,
            'best_model_results': threshold_results.get(best_model, {})
        }
        
        # Step 6: Spatial analysis (optional)
        if include_spatial:
            logger.info(f"Running spatial analysis for {commodity}")
            spatial_results = run_spatial_analysis(preprocessed_data, config)
            all_results['spatial_results'] = spatial_results
        
        # Step 7: Policy simulation (optional)
        if include_simulation:
            logger.info(f"Running policy simulation for {commodity}")
            simulation_results = run_policy_simulation(
                preprocessed_data,
                threshold_results.get(best_model, {}),
                config
            )
            all_results['simulation_results'] = simulation_results
        
        # Step 8: Generate reports
        logger.info(f"Generating reports for {commodity}")
        
        # Generate comprehensive report
        report_path = generate_comprehensive_report(
            all_results=all_results,
            commodity=commodity,
            output_path=commodity_dir / f"{commodity.replace(' ', '_')}_report",
            format=format,
            publication_quality=publication_quality,
            confidence_level=confidence_level,
            significance_indicators=True,
            style=style
        )
        
        # Generate executive summary
        summary_path = create_executive_summary(
            all_results=all_results,
            commodity=commodity,
            output_path=commodity_dir / f"{commodity.replace(' ', '_')}_summary",
            format=format
        )
        
        # Export publication-quality tables and figures if requested
        if econometric_tables:
            export_paths = export_results_for_publication(
                all_results=all_results,
                commodity=commodity,
                output_dir=commodity_dir / "publication",
                formats=[format, 'png', 'pdf'] if publication_quality else [format, 'png']
            )
        
        # Record report paths
        all_results['report_paths'] = {
            'comprehensive_report': str(report_path),
            'executive_summary': str(summary_path)
        }
        
        # Save all results to JSON
        with open(commodity_dir / f"{commodity.replace(' ', '_')}_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis for {commodity} completed in {elapsed_time:.2f} seconds")
        
        return commodity, all_results
    
    except Exception as e:
        logger.error(f"Error analyzing {commodity}: {e}", exc_info=True)
        all_results['error'] = str(e)
        
        # Save error results to JSON
        with open(commodity_dir / f"{commodity.replace(' ', '_')}_error.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        return commodity, {'error': str(e), 'commodity': commodity}


def run_batch_analysis(
    commodities,
    config,
    output_dir,
    threshold_modes,
    include_spatial=False,
    include_simulation=False,
    publication_quality=False,
    econometric_tables=False,
    format='markdown',
    style='world_bank',
    confidence_level=0.95,
    start_date=None,
    end_date=None,
    parallel=False,
    workers=None,
    debug=False
):
    """Run analysis for multiple commodities, optionally in parallel."""
    start_time = time.time()
    logger.info(f"Starting batch analysis for {len(commodities)} commodities")
    
    # Expand threshold modes if 'all' is specified
    threshold_modes = expand_threshold_modes(threshold_modes)
    
    # Initialize results dictionary
    all_commodity_results = {}
    # Load the unified data file
    logger.info("Loading unified data file")
    data_loader = DataLoader(config.get('data', {}).get('path', './data'))
    try:
        unified_data = data_loader.load_geojson("unified_data.geojson")
        logger.info(f"Loaded unified data with {len(unified_data)} records")
    except Exception as e:
        logger.error(f"Error loading unified data: {e}")
        return all_commodity_results
    
    if parallel and len(commodities) > 1 and workers > 1:
        logger.info(f"Running in parallel with {workers} workers")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            
            for commodity in commodities:
                # Filter data for this commodity
                commodity_data = unified_data[unified_data['commodity'] == commodity].copy()
                logger.info(f"Filtered {len(commodity_data)} records for {commodity}")
                
                future = executor.submit(
                    run_analysis_for_commodity,
                    commodity=commodity,
                    data=commodity_data,
                    config=config,
                    output_dir=output_dir,
                    threshold_modes=threshold_modes,
                    include_spatial=include_spatial,
                    include_simulation=include_simulation,
                    publication_quality=publication_quality,
                    econometric_tables=econometric_tables,
                    format=format,
                    style=style,
                    confidence_level=confidence_level,
                    start_date=start_date,
                    end_date=end_date,
                    debug=debug
                )
                futures[future] = commodity
            
            for future in as_completed(futures):
                commodity = futures[future]
                try:
                    _, results = future.result()
                    all_commodity_results[commodity] = results
                    logger.info(f"Completed analysis for {commodity}")
                except Exception as e:
                    logger.error(f"Error in parallel analysis for {commodity}: {e}")
                    all_commodity_results[commodity] = {'error': str(e), 'commodity': commodity}
    else:
        logger.info("Running sequentially")
        
        for commodity in commodities:
            # Filter data for this commodity
            commodity_data = unified_data[unified_data['commodity'] == commodity].copy()
            logger.info(f"Filtered {len(commodity_data)} records for {commodity}")
            
            _, results = run_analysis_for_commodity(
                commodity=commodity,
                data=commodity_data,
                config=config,
                output_dir=output_dir,
                threshold_modes=threshold_modes,
                include_spatial=include_spatial,
                include_simulation=include_simulation,
                publication_quality=publication_quality,
                econometric_tables=econometric_tables,
                format=format,
                style=style,
                confidence_level=confidence_level,
                start_date=start_date,
                end_date=end_date,
                debug=debug
            )
            all_commodity_results[commodity] = results
            
            # Force garbage collection to free memory
            gc.collect()
            gc.collect()
    
    # Generate cross-commodity comparison if requested
    if len(commodities) > 1:
        logger.info("Generating cross-commodity comparison")
        
        comparison_path = generate_cross_commodity_comparison(
            all_results=all_commodity_results,
            output_dir=output_dir,
            format=format,
            publication_quality=publication_quality
        )
        
        logger.info(f"Cross-commodity comparison saved to {comparison_path}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Batch analysis completed in {elapsed_time:.2f} seconds")
    
    return all_commodity_results


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment
    config, output_dir = setup_environment(args)
    
    # Get commodities to analyze
    commodities = get_commodities(args, config)
    
    if not commodities:
        logger.error("No commodities specified for analysis")
        sys.exit(1)
    
    logger.info(f"Starting Yemen Market Analysis for {len(commodities)} commodities")
    logger.info(f"Output directory: {output_dir}")
    
    # Run batch analysis
    all_results = run_batch_analysis(
        commodities=commodities,
        config=config,
        output_dir=output_dir,
        threshold_modes=args.threshold_modes,
        include_spatial=args.spatial_analysis,
        include_simulation=args.policy_simulation,
        publication_quality=args.publication_quality,
        econometric_tables=args.econometric_tables,
        format=args.format,
        style=args.style,
        confidence_level=args.confidence_level,
        start_date=args.start_date,
        end_date=args.end_date,
        parallel=args.parallel,
        workers=args.workers,
        debug=args.debug
    )
    
    # Save summary of all results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'commodities': list(all_results.keys()),
        'successful': sum(1 for r in all_results.values() if 'error' not in r),
        'failed': sum(1 for r in all_results.values() if 'error' in r),
        'config': {
            'threshold_modes': args.threshold_modes,
            'spatial_analysis': args.spatial_analysis,
            'policy_simulation': args.policy_simulation,
            'publication_quality': args.publication_quality,
            'format': args.format,
            'style': args.style
        }
    }
    
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    logger.info(f"Successful analyses: {summary['successful']}")
    logger.info(f"Failed analyses: {summary['failed']}")


if __name__ == "__main__":
    main()

# Add the parent directory to the Python path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import project modules
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester
from src.models.threshold_model import ThresholdModel
from src.models.spatial import SpatialEconometrics
from src.models.simulation import MarketIntegrationSimulation
from src.models.diagnostics import ModelDiagnostics
from src.models.model_selection import ModelComparer, calculate_information_criteria

# Import integration modules
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

# Import visualization modules
from src.visualization.time_series import TimeSeriesVisualizer
from src.visualization.maps import MarketMapVisualizer
from src.visualization.econometric_reporting import generate_econometric_report
from src.visualization.dashboard_components import DashboardCreator
from src.visualization.enhanced_econometric_reporting import (
    generate_enhanced_report,
    generate_cross_commodity_comparison,
    EconometricReporter
)

# Import utility functions
from src.utils.performance_utils import timer, memory_usage_decorator, optimize_dataframe, parallelize_dataframe, configure_system_for_performance
from src.utils.validation import validate_data, validate_model_inputs
from src.utils.error_handler import handle_errors, ModelError, DataError, capture_error
from src.utils.config import config


def setup_logging(log_file='yemen_market_analysis.log', level=logging.INFO):
    """Set up logging configuration with enhanced formatting."""
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
    
    # Log the creation of the log file
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created at: {log_path}")
    
    return log_path


def parse_arguments():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description='Yemen Market Integration Analysis with Enhanced Reporting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data input options
    parser.add_argument('--data', type=str, default='data/raw/unified_data.geojson',
                        help='Path to input data file')
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save output files')
    
    # Commodity selection options
    commodity_group = parser.add_mutually_exclusive_group(required=True)
    commodity_group.add_argument('--commodity', type=str,
                               help='Single commodity to analyze')
    commodity_group.add_argument('--commodities', type=str, nargs='+',
                               help='List of commodities to analyze')
    commodity_group.add_argument('--all-commodities', action='store_true',
                               help='Analyze all available commodities')
    
    # Analysis options
    parser.add_argument('--threshold-modes', type=str, nargs='+', 
                        default=['standard'],
                        choices=['standard', 'fixed', 'vecm', 'mtar', 'all'],
                        help='Threshold model modes to use')
    parser.add_argument('--spatial-analysis', action='store_true',
                        help='Perform spatial analysis')
    parser.add_argument('--comparative-analysis', action='store_true',
                        help='Perform comparative analysis across models')
    parser.add_argument('--policy-simulation', action='store_true',
                        help='Run policy simulations')
    parser.add_argument('--cross-commodity-comparison', action='store_true',
                        help='Generate cross-commodity comparison report')
    
    # Model parameters
    parser.add_argument('--max-lags', type=int, default=4,
                        help='Maximum number of lags for time series models')
    parser.add_argument('--k-neighbors', type=int, default=5,
                        help='Number of neighbors for spatial models')
    parser.add_argument('--conflict-weight', type=float, default=1.0,
                        help='Weight for conflict intensity in spatial models')
    
    # Reporting options
    parser.add_argument('--report-format', type=str, default='markdown',
                        choices=['markdown', 'latex', 'html', 'json'],
                        help='Format for generated reports')
    parser.add_argument('--econometric-tables', action='store_true',
                        help='Generate publication-quality econometric tables')
    parser.add_argument('--publication-plots', action='store_true',
                        help='Generate publication-quality plots')
    parser.add_argument('--interactive-dashboard', action='store_true',
                        help='Generate interactive dashboard')
    parser.add_argument('--report-style', type=str, default='world_bank',
                        choices=['world_bank', 'academic', 'policy'],
                        help='Style for reports and visualizations')
    
    # Performance options
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing for analysis')
    parser.add_argument('--max-workers', type=int, default=mp.cpu_count() - 1,
                        help='Maximum number of worker processes')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    # New academic reporting options
    parser.add_argument('--publication-quality', action='store_true',
                        help='Generate publication-quality outputs')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        choices=[0.90, 0.95, 0.99],
                        help='Confidence level for intervals (0.90, 0.95, or 0.99)')
    parser.add_argument('--significance-indicators', action='store_true',
                        help='Add significance indicators (*, **, ***) to tables')
    parser.add_argument('--robust-diagnostics', action='store_true',
                        help='Perform robust diagnostic tests')
    parser.add_argument('--formal-hypothesis-testing', action='store_true',
                        help='Perform formal hypothesis testing')
    
    args = parser.parse_args()
    
    # Process 'all' in threshold modes
    if 'all' in args.threshold_modes:
        args.threshold_modes = ['standard', 'fixed', 'vecm', 'mtar']
    
    return args


def get_commodity_list(gdf, commodity=None, commodities=None, all_commodities=False):
    """Get list of commodities to analyze based on command line arguments."""
    available_commodities = sorted(gdf['commodity'].unique())
    
    if commodity:
        if commodity not in available_commodities:
            raise ValueError(f"Commodity '{commodity}' not found in data. Available commodities: {available_commodities}")
        return [commodity]
    elif commodities:
        invalid_commodities = [c for c in commodities if c not in available_commodities]
        if invalid_commodities:
            raise ValueError(f"Commodities {invalid_commodities} not found in data. Available commodities: {available_commodities}")
        return commodities
    elif all_commodities:
        return available_commodities
    else:
        raise ValueError("No commodity specified. Use --commodity, --commodities, or --all-commodities")


def setup_output_directory(output_path):
    """Set up output directory for results."""
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def load_and_preprocess_data(args):
    """Load and preprocess data from specified source."""
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading data from: {args.data}")
    data_path = Path(args.data)
    
    # Check if the path is absolute or relative
    if data_path.is_absolute():
        # If absolute path, use the parent directory as data_dir
        data_dir = data_path.parent.parent  # Go up one more level to avoid duplicate 'raw'
        filename = data_path.name
    else:
        # If relative path, use the parent directory as data_dir
        data_dir = str(data_path.parent.parent if data_path.parent.name == 'raw' else data_path.parent)
        filename = data_path.name
    
    logger.info(f"Using data directory: {data_dir}, filename: {filename}")
    
    # Initialize data loader
    loader = DataLoader(data_path=data_dir)
    
    # Load GeoJSON data
    gdf = loader.load_geojson(filename)
    
    # Validate data
    logger.info("Validating input data")
    validate_data(gdf, logger)
    
    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    processed_gdf = preprocessor.preprocess_geojson(gdf)
    
    # Get list of commodities to analyze
    available_commodities = sorted(processed_gdf['commodity'].unique())
    logger.info(f"Available commodities: {available_commodities}")
    
    commodities = get_commodity_list(
        processed_gdf,
        commodity=args.commodity,
        commodities=args.commodities,
        all_commodities=args.all_commodities
    )
    
    logger.info(f"Analyzing {len(commodities)} commodities: {commodities}")
    
    return processed_gdf, commodities


@timer
@memory_usage_decorator
@handle_errors(logger=logging.getLogger(__name__), error_type=(ValueError, RuntimeError), reraise=False)
def analyze_threshold_models(commodity, gdf, args, logger=None):
    """
    Analyze market integration using threshold models.
    
    This function performs threshold model analysis for a specific commodity,
    comparing different threshold model specifications if requested.
    
    Parameters
    ----------
    commodity : str
        Commodity name to analyze
    gdf : GeoDataFrame
        Preprocessed data
    args : argparse.Namespace
        Command line arguments
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    dict
        Analysis results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Analyzing {commodity} with threshold models")
    
    # Filter data for the specified commodity
    commodity_gdf = gdf[gdf['commodity'] == commodity].copy()
    
    # Check what values are in the admin1 column
    admin1_values = commodity_gdf['admin1'].unique()
    logger.info(f"Admin1 values for {commodity}: {admin1_values}")
    
    # Always use usdprice for analysis
    price_column = 'usdprice'
    logger.info(f"Using {price_column} column for analysis")
    
    # Determine aggregation method
    aggregation = args.aggregation if hasattr(args, 'aggregation') and args.aggregation else 'mean'
    logger.info(f"Using {aggregation} aggregation for prices")
    
    # Use exchange_rate_regime to determine regions
    commodity_gdf['region'] = commodity_gdf['exchange_rate_regime'].str.title()
    
    # Log unique regions and their counts
    region_counts = commodity_gdf['region'].value_counts()
    logger.info(f"Region mapping: {region_counts.to_dict()}")
    
    # Separate north and south data
    north_gdf = commodity_gdf[commodity_gdf['exchange_rate_regime'] == 'north'].copy()
    south_gdf = commodity_gdf[commodity_gdf['exchange_rate_regime'] == 'south'].copy()
    
    logger.info(f"North data size: {len(north_gdf)}, South data size: {len(south_gdf)}")
    
    # Aggregate prices by date
    north_prices = north_gdf.groupby('date')[price_column].agg(aggregation).reset_index()
    south_prices = south_gdf.groupby('date')[price_column].agg(aggregation).reset_index()
    
    logger.info(f"North prices after aggregation: {len(north_prices)}, South prices after aggregation: {len(south_prices)}")
    
    # Check if we have data for both regions
    if len(north_prices) == 0 or len(south_prices) == 0:
        logger.error(f"Missing data for {'North' if len(north_prices) == 0 else 'South'} region")
        return {
            'commodity': commodity,
            'success': False,
            'error': f"Missing data for {'North' if len(north_prices) == 0 else 'South'} region"
        }
    
    # Merge north and south data
    logger.info(f"Merging north and south data for {commodity}")
    merged = pd.merge(north_prices, south_prices, on='date', suffixes=('_north', '_south'))
    
    logger.info(f"Merged data size: {len(merged)}")
    
    # Check if merged data is empty
    if len(merged) == 0:
        logger.error("No matching dates found between North and South data")
        return {
            'commodity': commodity,
            'success': False,
            'error': "No matching dates found between North and South data"
        }
    
    # Define column names for clarity
    north_price_col = f"{price_column}_north"
    south_price_col = f"{price_column}_south"
    
    # Check if we should do model comparison
    if args.comparative_analysis and len(args.threshold_modes) > 1:
        logger.info(f"Setting up model comparison for {commodity}")
        
        # Create output directory
        output_path = Path(args.output) / commodity.replace(" ", "_")
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Configure model comparer
        model_specs = []
        
        # Add model specifications for each threshold mode
        for mode in args.threshold_modes:
            model_specs.append({
                'name': f'Threshold Model ({mode})',
                'model_class': ThresholdModel,
                'parameters': {
                    'price1': merged[north_price_col],
                    'price2': merged[south_price_col],
                    'mode': mode,
                    'max_lags': args.max_lags,
                    'market1_name': 'North',
                    'market2_name': 'South',
                    'index': merged['date']
                }
            })
        
        # Initialize model comparer
        model_comparer = ModelComparer(
            model_specs=model_specs,
            max_workers=1 if not args.parallel else args.max_workers
        )
        
        # Fit models
        model_comparer.fit_models()
        
        # Compare models using information criteria
        logger.info("Comparing models using information criteria")
        ic_comparison = model_comparer.compare_information_criteria()
        
        # Compare performance metrics
        performance_comparison = model_comparer.compare_performance_metrics()
        
        # Get comprehensive report
        model_report = model_comparer.get_comprehensive_report()
        
        # Get the best model
        best_model_name = model_report['best_models'].get('aic')
        if best_model_name:
            best_model = model_comparer.fitted_models.get(best_model_name)
            logger.info(f"Best model for {commodity}: {best_model_name}")
        else:
            best_model = None
            logger.warning(f"Could not determine best model for {commodity}")
        
        # Save model comparison results
        comparison_path = output_path / f"model_comparison.json"
        with open(comparison_path, 'w') as f:
            # Convert results to serializable format
            serializable_report = {}
            for key, value in model_report.items():
                if key != 'fitted_models':  # Skip fitted models which aren't JSON serializable
                    if isinstance(value, dict):
                        serializable_report[key] = {k: str(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
                    elif hasattr(value, '__dict__'):
                        serializable_report[key] = str(value)
                    else:
                        serializable_report[key] = value
            
            json.dump(serializable_report, f, indent=2, default=str)
        
        # Generate econometric report for the best model
        if best_model:
            # Run the full analysis on the best model to get detailed results
            if hasattr(best_model, 'run_full_analysis'):
                full_results = best_model.run_full_analysis()
            else:
                # For models that don't have run_full_analysis method
                full_results = {'model': best_model, 'model_name': best_model_name}
                
            # Generate report for the best model
            report_path = output_path / f"threshold_report.{args.report_format}"
            if hasattr(best_model, 'generate_report'):
                report = best_model.generate_report(
                    format=args.report_format,
                    output_path=str(report_path)
                )
            
            # Generate enhanced econometric tables if requested
            if args.econometric_tables or args.publication_quality:
                logger.info("Generating publication-quality econometric tables")
                tables_path = output_path / f"econometric_tables.{args.report_format}"
                generate_econometric_report(
                    model=best_model,
                    results=full_results,
                    output_path=str(tables_path),
                    format=args.report_format,
                    publication_quality=True,
                    confidence_level=args.confidence_level if hasattr(args, 'confidence_level') else 0.95,
                    significance_indicators=args.significance_indicators if hasattr(args, 'significance_indicators') else True
                )
        
        # Return combined results
        return {
            'commodity': commodity,
            'success': True,
            'model_comparison': {
                'best_model': best_model_name,
                'information_criteria': ic_comparison,
                'performance_metrics': performance_comparison,
                'comparison_path': str(comparison_path)
            },
            'best_model_results': full_results if best_model else None,
            'report_path': str(report_path) if best_model else None
        }
    
    else:
        # Run individual threshold model for each mode without comparison
        results = {}
        
        for mode in args.threshold_modes:
            logger.info(f"Analyzing {commodity} with {mode} threshold model")
            
            # Initialize threshold model with the current mode
            threshold_model = ThresholdModel(
                merged[north_price_col],
                merged[south_price_col],
                mode=mode,
                max_lags=args.max_lags,
                market1_name="North",
                market2_name="South",
                index=merged['date']
            )
            
            # Run full analysis
            threshold_results = threshold_model.run_full_analysis()
            
            # Generate report
            output_path = Path(args.output) / commodity.replace(" ", "_")
            output_path.mkdir(exist_ok=True, parents=True)
            report_path = output_path / f"{mode}_threshold_report.{args.report_format}"
            report = threshold_model.generate_report(
                format=args.report_format,
                output_path=str(report_path)
            )
            
            # Save results
            results[mode] = {
                'success': True,
                'report_path': str(report_path),
                'threshold': threshold_model.threshold,
                'cointegrated': threshold_results.get('cointegration', {}).get('cointegrated', False)
            }
            
            # Generate enhanced econometric tables if requested
            if args.econometric_tables or args.publication_quality:
                logger.info(f"Generating publication-quality econometric tables for {mode} model")
                tables_path = output_path / f"{mode}_econometric_tables.{args.report_format}"
                # Generate report directly from the threshold model
                report_path = output_path / f"{mode}_econometric_report.{args.report_format}"
                threshold_model.generate_report(
                    output_path=str(report_path),
                    format=args.report_format
                )
        
        return {
            'commodity': commodity,
            'success': True,
            'threshold_results': results
        }


@timer
def process_commodity(commodity, gdf, args, logger):
    """Process a single commodity with all requested analyses."""
    logger.info(f"Processing commodity: {commodity}")
    
    results = {
        'commodity': commodity,
        'timestamp': datetime.now().isoformat()
    }
    
    # Create output directory
    output_path = Path(args.output) / commodity.replace(" ", "_")
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Run threshold model analysis
    threshold_results = analyze_threshold_models(
        commodity=commodity,
        gdf=gdf,
        args=args,
        logger=logger
    )
    results['threshold_analysis'] = threshold_results
    
    # Generate enhanced report
    try:
        reporter = EconometricReporter(
            output_dir=output_path,
            publication_quality=args.publication_quality if hasattr(args, 'publication_quality') else True,
            style=args.report_style,
            format=args.report_format
        )
        
        report_path = reporter.generate_model_comparison_report(
            model_results=threshold_results.get('model_comparison', {}),
            commodity=commodity,
            output_file=f"{commodity.replace(' ', '_')}_model_comparison"
        )
        
        results['enhanced_report_path'] = str(report_path)
        logger.info(f"Enhanced report for {commodity} saved to {report_path}")
    
    except Exception as e:
        logger.error(f"Error generating enhanced report for {commodity}: {str(e)}")
        results['report_error'] = str(e)
    
    return results


@timer
def main():
    """Main entry point for Yemen Market Analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    log_path = setup_logging(level=level)
    logger = logging.getLogger(__name__)
    
    logger.info("Yemen Market Integration Analysis")
    logger.info(f"Threshold modes: {args.threshold_modes}")
    
    try:
        # Configure system for optimal performance
        configure_system_for_performance()
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Load data
        gdf, commodities = load_and_preprocess_data(args)
        
        # Process commodities
        logger.info("Processing commodities sequentially")
        all_results = {}
        
        for commodity in commodities:
            commodity_results = process_commodity(
                commodity=commodity,
                gdf=gdf,
                args=args,
                logger=logger
            )
            all_results[commodity] = commodity_results
        
        # Generate cross-commodity comparison if requested
        if args.cross_commodity_comparison and len(commodities) > 1:
            try:
                logger.info("Generating cross-commodity comparison report")
                comparison_path = generate_cross_commodity_comparison(
                    all_results=all_results,
                    output_dir=output_path,
                    format=args.report_format,
                    publication_quality=args.publication_quality if hasattr(args, 'publication_quality') else True
                )
                logger.info(f"Cross-commodity comparison saved to {comparison_path}")
            except Exception as e:
                logger.error(f"Error generating cross-commodity comparison: {str(e)}")
        
        # Save complete results
        results_file = output_path / "complete_results.json"
        try:
            with open(results_file, 'w') as f:
                # Convert to serializable format
                serializable_results = {}
                for commodity, results in all_results.items():
                    serializable_results[commodity] = {}
                    for k, v in results.items():
                        if isinstance(v, dict):
                            serializable_results[commodity][k] = {
                                sk: str(sv) if hasattr(sv, '__dict__') else sv 
                                for sk, sv in v.items()
                            }
                        elif hasattr(v, '__dict__'):
                            serializable_results[commodity][k] = str(v)
                        else:
                            serializable_results[commodity][k] = v
                
                json.dump(serializable_results, f, indent=2, default=str)
            logger.info(f"Complete results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving complete results: {str(e)}")
        
        # Generate interactive dashboard if requested
        if args.interactive_dashboard:
            try:
                logger.info("Generating interactive dashboard")
                dashboard = DashboardCreator(
                    results=all_results,
                    output_dir=output_path
                )
                dashboard_path = dashboard.create_dashboard()
                logger.info(f"Interactive dashboard saved to {dashboard_path}")
            except Exception as e:
                logger.error(f"Error generating dashboard: {str(e)}")
        
        # Report success
        logger.info(f"Analysis completed for {len(all_results)} out of {len(commodities)} commodities")
        return 0
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())