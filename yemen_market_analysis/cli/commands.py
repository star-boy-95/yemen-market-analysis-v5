"""
Command implementation for Yemen Market Analysis CLI.
"""
import os
import sys
import time
import logging
import argparse
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from core.config import initialize_config, config
from core.logging_setup import setup_logging
from core.exceptions import YemenMarketError

from data.loaders import load_unified_data, load_conflict_data
from data.preprocessors import preprocess_market_data, prepare_commodity_series

from models.threshold import ThresholdHistoryTracker
from models.threshold.hansen_seo import HansenSeoModel
from models.threshold.enders_siklos import EndersSiklosModel

from computation.device_manager import get_device_manager
from computation.parallel import ParallelProcessor
from computation.performance import optimize_computation_settings

from analysis.market_integration import analyze_market_integration, analyze_welfare_impact
from analysis.conflict_impact import analyze_conflict_impact
from analysis.economics import interpret_threshold_economics

from visualization.plot_manager import set_plot_manager
from visualization.threshold_plots import plot_threshold_regimes, plot_adjustment_speeds
from visualization.market_plots import plot_price_series, plot_market_integration

from reporting.output_manager import OutputManager
from reporting.reports import generate_html_summary_report, generate_academic_results_report
from reporting.tables import create_summary_table

from .parsers import parse_commodities

logger = logging.getLogger(__name__)


def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the market analysis with the given arguments."""
    start_time = time.time()
    
    # Initialize configuration
    initialize_config(args.config_path)
    
    # Update config with command-line arguments
    if args.output_dir:
        config.set('directories.results_dir', args.output_dir)
        
    if args.parallel:
        config.set('parameters.parallel_commodities', True)
        
    if args.max_workers:
        config.set('parameters.commodity_parallel_processes', args.max_workers)
        
    if args.gpu:
        config.set('use_gpu', True)
    elif args.cpu:
        config.set('use_gpu', False)
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO" if args.verbose else "WARNING"
    setup_logging(log_level=log_level)
    
    # Log analysis start
    logger.info("Starting Yemen Market Analysis")
    logger.info(f"Configuration loaded from: {args.config_path}")
    
    # Load data
    logger.info(f"Loading data from: {args.data_path}")
    df = load_unified_data(args.data_path)
    
    if df.empty:
        raise YemenMarketError(f"Failed to load data from {args.data_path}")
    
    logger.info(f"Loaded data with {len(df)} observations")
    
    # Load conflict data if specified
    conflict_df = None
    if args.conflict_path:
        logger.info(f"Loading conflict data from: {args.conflict_path}")
        conflict_df = load_conflict_data(args.conflict_path)
        
        if conflict_df is None:
            logger.warning(f"Failed to load conflict data from {args.conflict_path}")
    
    # Preprocess data
    logger.info("Preprocessing market data")
    clean_df, preprocess_results = preprocess_market_data(df)
    
    # Get list of commodities to analyze
    commodities = parse_commodities(args.commodities)
    if not commodities:
        # Get all available commodities
        commodities = clean_df['commodity'].unique().tolist()
        logger.info(f"Analyzing all {len(commodities)} commodities")
    else:
        logger.info(f"Analyzing {len(commodities)} specified commodities: {', '.join(commodities)}")
    
    # Initialize output manager
    output_manager = OutputManager()
    
    # Setup device manager
    device_manager = get_device_manager()
    logger.info(f"Using computation device: {device_manager.device_name}")
    
    # Initialize threshold history tracker
    threshold_tracker = ThresholdHistoryTracker()
    
    # Setup parallel processing if enabled
    if config.get('parameters.parallel_commodities', False):
        max_workers = config.get('parameters.commodity_parallel_processes', 4)
        parallel_processor = ParallelProcessor(max_workers=max_workers)
        logger.info(f"Using parallel processing with {max_workers} workers")
    else:
        parallel_processor = None
        logger.info("Parallel processing disabled")
    
    # Define function to process each commodity
    def process_commodity(commodity: str, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Processing commodity: {commodity}")
        
        try:
            # Prepare price series
            north_series, south_series, prep_results = prepare_commodity_series(
                clean_df, commodity
            )
            
            if north_series is None or south_series is None:
                logger.warning(f"Insufficient data for commodity: {commodity}")
                return {"error": f"Insufficient data for {commodity}"}
            
            # Setup model configuration
            model_config = {
                'grid_points': config.get('parameters.threshold_cointegration.grid_points', 50),
                'nboot': config.get('parameters.threshold_cointegration.nboot', 500),
                'block_size': config.get('parameters.threshold_cointegration.block_size', 5),
                'min_regime_size': config.get('parameters.threshold_cointegration.min_regime_size', 0.1),
                'max_lags': config.get('parameters.lag_periods', 8)
            }
            
            # Get threshold range for this commodity
            threshold_range = threshold_tracker.compute_robust_threshold_range(
                commodity, north_series.values - south_series.values
            )
            
            # Select and run threshold model
            if args.model == "hansen_seo" or args.model == "both":
                model = HansenSeoModel(model_config)
                results = model.fit(
                    north_series.values, south_series.values, 
                    commodity=commodity, threshold_range=threshold_range
                )
            elif args.model == "enders_siklos":
                model = EndersSiklosModel(model_config)
                results = model.fit(
                    north_series.values, south_series.values, 
                    commodity=commodity, threshold_range=threshold_range
                )
            
            # Add threshold to history
            threshold_tracker.add_threshold(
                commodity=commodity,
                threshold=results.get('threshold', 0.0),
                p_value=results.get('p_value'),
                threshold_significant=results.get('threshold_significant', False),
                model_type=args.model
            )
            
            # Run additional analyses
            # Market integration analysis
            price_diff_df = shared_data.get('price_diffs', {}).get(commodity)
            integration_results = analyze_market_integration(results, price_diff_df, commodity)
            results.update(integration_results)
            
            # Economic interpretation
            economics_results = interpret_threshold_economics(
                results.get('threshold', 0.0), commodity, results
            )
            results.update({"economics": economics_results})
            
            # Conflict impact analysis if conflict data available
            if conflict_df is not None:
                conflict_results = analyze_conflict_impact(
                    results, price_diff_df, conflict_df['conflict_intensity'], commodity
                )
                results.update({"conflict_impact": conflict_results})
            
            # Welfare impact analysis
            welfare_results = analyze_welfare_impact(
                results, north_series, south_series, commodity
            )
            results.update({"welfare_impact": welfare_results})
            
            # Save results for this commodity
            output_manager.save_json(results, commodity=commodity)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {commodity}: {str(e)}")
            return {"error": str(e), "commodity": commodity}
    
    # Process all commodities
    results_by_commodity = {}
    shared_data = {"price_diffs": {}}
    
    if parallel_processor:
        # Process in parallel
        parallel_results = parallel_processor.process(
            commodities,
            process_func=process_commodity,
            shared_data=shared_data
        )
        results_by_commodity = parallel_results.get("data", {})
    else:
        # Process sequentially
        for commodity in commodities:
            results = process_commodity(commodity, shared_data)
            if results:
                results_by_commodity[commodity] = results
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "commodities": commodities,
        "config_path": args.config_path,
        "data_path": args.data_path,
        "conflict_path": args.conflict_path,
        "model_type": args.model,
        "parallel": args.parallel,
        "n_commodities_analyzed": len(results_by_commodity),
        "total_time": time.time() - start_time
    }
    output_manager.save_metadata(metadata)
    
    # Generate summary table
    summary_df = create_summary_table(results_by_commodity)
    output_manager.save_csv(summary_df, "summary_table.csv")
    
    # Generate reports if requested
    if args.reports:
        logger.info("Generating HTML reports")
        html_report = generate_html_summary_report(results_by_commodity)
        output_manager.save_html(html_report, "summary_table.html")
        
        academic_report = generate_academic_results_report(results_by_commodity)
        output_manager.save_html(academic_report, "formal_academic_results.html")
    
    # Save manifest
    output_manager.save_manifest()
    
    # Log completion
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    
    return {
        "results_by_commodity": results_by_commodity,
        "metadata": metadata,
        "output_dir": output_manager.analysis_dir
    }


def run_visualization(args: argparse.Namespace, analysis_results: Dict[str, Any]) -> bool:
    """Run visualizations based on analysis results."""
    if not args.visualize:
        return False
    
    logger.info("Generating visualizations")
    
    # Extract results
    results_by_commodity = analysis_results.get("results_by_commodity", {})
    
    if not results_by_commodity:
        logger.warning("No results available for visualization")
        return False
    
    # Initialize output manager for visualization paths
    output_manager = OutputManager()
    
    # Configure plot manager
    viz_dir = os.path.join(output_manager.analysis_dir, "visualizations")
    set_plot_manager(
        style='seaborn-v0_8-whitegrid',
        figsize=(10, 6),
        dpi=300,
        output_dir=viz_dir
    )
    
    # Process each commodity
    for commodity, results in results_by_commodity.items():
        logger.info(f"Generating visualizations for: {commodity}")
        
        try:
            # Get commodity-specific directory
            commodity_dir = output_manager.get_commodity_viz_dir(commodity)
            
            # Extract data needed for visualizations
            threshold = results.get('threshold', 0.0)
            
            # Plot threshold regimes
            price_diff_df = analysis_results.get("price_diffs", {}).get(commodity)
            if price_diff_df is not None:
                plot_threshold_regimes(
                    price_diff_df,
                    threshold,
                    commodity=commodity,
                    filename=os.path.join(commodity_dir, "threshold_regimes.png")
                )
            
            # Plot adjustment speeds
            plot_adjustment_speeds(
                results,
                commodity=commodity,
                filename=os.path.join(commodity_dir, "adjustment_speeds.png")
            )
            
            # Plot market integration
            integration_results = results.get('integration', {})
            if integration_results:
                plot_market_integration(
                    integration_results,
                    results,
                    commodity=commodity,
                    filename=os.path.join(commodity_dir, "market_integration.png")
                )
        
        except Exception as e:
            logger.error(f"Error generating visualizations for {commodity}: {str(e)}")
    
    logger.info("Visualizations completed")
    return True