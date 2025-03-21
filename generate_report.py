#!/usr/bin/env python
"""
Generate threshold model reports in production mode.

This script runs the unified threshold model in production mode and generates
comprehensive reports and visualizations for specified commodities.

Example usage:
    python generate_report.py --commodity "beans (kidney red)" --mode standard
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import the unified threshold model directly from src
from src.models.threshold_model import ThresholdModel
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.error_handler import handle_errors, capture_error
from src.utils.validation import validate_data
from src.utils.config import config


def setup_logging(log_file='generate_report.log', level=logging.INFO):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_{log_file}"
    log_path = log_dir / log_filename
    
    # Configure logging
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
    
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate threshold model reports in production mode',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=config.get('analysis.data_path', './data/raw/unified_data.geojson'),
        help='Path to the GeoJSON data file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=config.get('analysis.output_path', './results'),
        help='Path to save output files'
    )
    
    parser.add_argument(
        '--commodity',
        type=str,
        default=config.get('analysis.default_commodity', 'beans (kidney red)'),
        help='Commodity to analyze'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['standard', 'fixed', 'vecm', 'mtar'],
        default=config.get('analysis.threshold.mode', 'standard'),
        help='Mode for threshold model analysis'
    )
    
    parser.add_argument(
        '--max-lags',
        type=int,
        default=config.get('analysis.cointegration.max_lags', 4),
        help='Maximum number of lags for time series analysis'
    )
    
    parser.add_argument(
        '--report-format',
        type=str,
        choices=['markdown', 'json', 'latex'],
        default=config.get('analysis.report_format', 'markdown'),
        help='Format for the report'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    return args


@handle_errors(logger=logging.getLogger(__name__), error_type=(Exception,), reraise=False)
def generate_threshold_report(args, logger):
    """
    Generate threshold model report for the specified commodity.
    
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
        
        # Load data
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
        if not validate_data(gdf, logger):
            logger.error("Data validation failed, aborting analysis")
            return 1
        
        # Preprocess data
        logger.info(f"Preprocessing data for {args.commodity}")
        preprocessor = DataPreprocessor()
        processed_gdf = preprocessor.preprocess_geojson(gdf)
        
        # Get data for north and south
        north_data = processed_gdf[
            (processed_gdf['commodity'] == args.commodity) &
            (processed_gdf['exchange_rate_regime'] == 'north')
        ]
        south_data = processed_gdf[
            (processed_gdf['commodity'] == args.commodity) &
            (processed_gdf['exchange_rate_regime'] == 'south')
        ]
        
        # Validate data
        if len(north_data) < 30 or len(south_data) < 30:
            logger.warning(f"Insufficient data for {args.commodity}: North={len(north_data)}, South={len(south_data)}")
            return 1
        
        # Get aggregation method from config
        agg_method = config.get('analysis.price_aggregation.method', 'mean')
        logger.info(f"Using {agg_method} aggregation for prices for {args.commodity}")
        
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
        
        # Ensure dates align
        logger.info(f"Merging north and south data for {args.commodity}")
        merged = pd.merge(
            north_monthly, south_monthly,
            on='date', suffixes=('_north', '_south')
        )
        
        if len(merged) < 30:
            logger.warning(f"Insufficient overlapping data points for {args.commodity}: {len(merged)}")
            return 1
        
        # Initialize threshold model with the specified mode
        logger.info(f"Initializing threshold model in {args.mode} mode")
        threshold_model = ThresholdModel(
            merged['price_north'], 
            merged['price_south'],
            mode=args.mode,
            max_lags=args.max_lags,
            market1_name="North",
            market2_name="South"
        )
        
        # Run full analysis
        logger.info(f"Running full threshold analysis in {args.mode} mode")
        full_results = threshold_model.run_full_analysis()
        
        # Generate report
        logger.info(f"Generating standardized report for {args.mode} threshold model")
        report_path = str(output_path / f'{args.commodity.replace(" ", "_")}_threshold_report.{args.report_format}')
        report = threshold_model.generate_report(
            format=args.report_format, 
            output_path=report_path
        )
        
        # Create visualization of threshold dynamics
        viz_path = output_path / f'{args.commodity.replace(" ", "_")}_threshold_dynamics.png'
        
        logger.info(f"Analysis complete for {args.commodity}")
        logger.info(f"Report saved to: {report_path}")
        
        return 0
        
    except Exception as e:
        capture_error(e, context="Generating threshold report", logger=logger)
        logger.error(f"Unhandled error during analysis: {e}")
        logger.exception("Detailed traceback:")
        return 1


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=level)
    
    logger.info("Yemen Market Integration Threshold Analysis")
    logger.info(f"Analyzing commodity: {args.commodity} with {args.mode} mode")
    
    # Generate the report
    exit_code = generate_threshold_report(args, logger)
    
    if exit_code == 0:
        logger.info("Report generation completed successfully")
    else:
        logger.error("Report generation failed")
    
    sys.exit(exit_code)