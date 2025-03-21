#!/usr/bin/env python
"""
Run full threshold model analysis for all commodities and save results.

This script runs the unified threshold model in production mode for all
available commodities and generates comprehensive reports, visualizations,
and analysis results.

Example usage:
    python run_full_analysis.py --mode fixed
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
import json

# Import the unified threshold model
from src.models.threshold_model import ThresholdModel
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.error_handler import handle_errors, capture_error
from src.utils.validation import validate_data
from src.utils.config import config


def setup_logging(log_file='run_full_analysis.log', level=logging.INFO):
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
        description='Run full threshold model analysis for all commodities',
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
        '--commodities',
        type=str,
        nargs='+',
        help='Specific commodities to analyze (if not specified, all available commodities will be analyzed)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    return args


def get_available_commodities(gdf):
    """Get list of available commodities in the dataset."""
    return sorted(gdf['commodity'].unique().tolist())


@handle_errors(logger=logging.getLogger(__name__), error_type=(Exception,), reraise=False)
def analyze_commodity(commodity, gdf, args, logger):
    """
    Analyze a single commodity using the threshold model.
    
    Parameters
    ----------
    commodity : str
        Commodity to analyze
    gdf : GeoDataFrame
        Preprocessed GeoDataFrame
    args : argparse.Namespace
        Command line arguments
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    dict
        Analysis results
    """
    logger.info(f"Analyzing {commodity} with {args.mode} mode")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get data for north and south
    north_data = gdf[
        (gdf['commodity'] == commodity) &
        (gdf['exchange_rate_regime'] == 'north')
    ]
    south_data = gdf[
        (gdf['commodity'] == commodity) &
        (gdf['exchange_rate_regime'] == 'south')
    ]
    
    # Validate data
    if len(north_data) < 30 or len(south_data) < 30:
        logger.warning(f"Insufficient data for {commodity}: North={len(north_data)}, South={len(south_data)}")
        return {
            'commodity': commodity,
            'success': False,
            'error': 'Insufficient data',
            'north_count': len(north_data),
            'south_count': len(south_data)
        }
    
    # Check if usdprice column exists
    price_column = 'usdprice' if 'usdprice' in north_data.columns else 'price'
    logger.info(f"Using {price_column} column for analysis")
    
    # Get aggregation method from config
    agg_method = config.get('analysis.price_aggregation.method', 'mean')
    logger.info(f"Using {agg_method} aggregation for prices for {commodity}")
    
    # Aggregate to monthly prices using the configured method
    if agg_method == 'median':
        north_monthly = north_data.groupby(pd.Grouper(key='date', freq='ME'))[price_column].median().reset_index()
        south_monthly = south_data.groupby(pd.Grouper(key='date', freq='ME'))[price_column].median().reset_index()
    elif agg_method == 'robust':
        # Use a more robust method (trimmed mean)
        north_monthly = north_data.groupby(pd.Grouper(key='date', freq='ME'))[price_column].apply(
            lambda x: x.quantile(0.25) if len(x) > 0 else np.nan
        ).reset_index()
        south_monthly = south_data.groupby(pd.Grouper(key='date', freq='ME'))[price_column].apply(
            lambda x: x.quantile(0.25) if len(x) > 0 else np.nan
        ).reset_index()
    else:
        # Default to mean
        north_monthly = north_data.groupby(pd.Grouper(key='date', freq='ME'))[price_column].mean().reset_index()
        south_monthly = south_data.groupby(pd.Grouper(key='date', freq='ME'))[price_column].mean().reset_index()
    
    # Ensure dates align
    logger.info(f"Merging north and south data for {commodity}")
    merged = pd.merge(
        north_monthly, south_monthly,
        on='date', suffixes=('_north', '_south')
    )
    
    if len(merged) < 30:
        logger.warning(f"Insufficient overlapping data points for {commodity}: {len(merged)}")
        return {
            'commodity': commodity,
            'success': False,
            'error': 'Insufficient overlapping data points',
            'data_points': len(merged)
        }
    
    # Initialize threshold model with the specified mode
    logger.info(f"Initializing threshold model in {args.mode} mode")
    
    # Use the column names from the merged dataframe
    north_price_col = f"{price_column}_north"
    south_price_col = f"{price_column}_south"
    
    threshold_model = ThresholdModel(
        merged[north_price_col],
        merged[south_price_col],
        mode=args.mode,
        max_lags=args.max_lags,
        market1_name="North",
        market2_name="South",
        index=merged['date']  # Pass dates for time series plots
    )
    
    # Run full analysis
    logger.info(f"Running full threshold analysis in {args.mode} mode")
    full_results = threshold_model.run_full_analysis()
    
    # Generate report
    logger.info(f"Generating standardized report for {args.mode} threshold model")
    report_path = str(output_path / f'{commodity.replace(" ", "_")}_threshold_report.{args.report_format}')
    report = threshold_model.generate_report(
        format=args.report_format,
        output_path=report_path
    )
    
    # The report already includes visualizations, so we'll use the paths from the report
    dynamics_path = str(output_path / f'{commodity.replace(" ", "_")}_threshold_report_regime_dynamics.png')
    
    # Save full results as JSON
    results_path = str(output_path / f'{commodity.replace(" ", "_")}_full_results.json')
    
    # Convert numpy arrays and other non-serializable objects to lists
    serializable_results = {}
    for key, value in full_results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                elif isinstance(v, pd.DataFrame):
                    serializable_results[key][k] = v.to_dict()
                elif isinstance(v, pd.Series):
                    serializable_results[key][k] = v.to_dict()
                elif isinstance(v, np.float64) or isinstance(v, np.float32):
                    serializable_results[key][k] = float(v)
                elif isinstance(v, np.int64) or isinstance(v, np.int32):
                    serializable_results[key][k] = int(v)
                elif isinstance(v, bool):
                    serializable_results[key][k] = bool(v)
                elif v is None:
                    serializable_results[key][k] = None
                else:
                    # Try to convert to a basic type
                    try:
                        serializable_results[key][k] = str(v)
                    except:
                        serializable_results[key][k] = f"Unserializable: {type(v).__name__}"
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            serializable_results[key] = value.to_dict()
        elif isinstance(value, pd.Series):
            serializable_results[key] = value.to_dict()
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            serializable_results[key] = float(value)
        elif isinstance(value, np.int64) or isinstance(value, np.int32):
            serializable_results[key] = int(value)
        elif isinstance(value, bool):
            serializable_results[key] = bool(value)
        elif value is None:
            serializable_results[key] = None
        else:
            # Try to convert to a basic type
            try:
                serializable_results[key] = str(value)
            except:
                serializable_results[key] = f"Unserializable: {type(value).__name__}"
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Analysis complete for {commodity}")
    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Dynamics plot saved to: {dynamics_path}")
    logger.info(f"Full results saved to: {results_path}")
    
    return {
        'commodity': commodity,
        'success': True,
        'report_path': report_path,
        'dynamics_path': dynamics_path,
        'results_path': results_path,
        'threshold': threshold_model.threshold,
        'cointegrated': full_results['cointegration']['cointegrated'] if 'cointegration' in full_results else False
    }


def main():
    """Main function to run the analysis."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=level)
    
    logger.info("Yemen Market Integration Threshold Analysis")
    
    # Define threshold modes
    threshold_modes = ['standard', 'fixed', 'vecm', 'mtar']
    
    # Store results for each mode
    all_mode_results = {}
    
    # Iterate through threshold modes
    for mode in threshold_modes:
        logger.info(f"Running analysis with {mode} mode")
        
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
        logger.info("Validating input data")
        if not validate_data(gdf, logger):
            logger.error("Data validation failed, aborting analysis")
            return 1
        logger.info("Data validation completed")
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        processed_gdf = preprocessor.preprocess_geojson(gdf)
        
        # Get available commodities
        available_commodities = get_available_commodities(processed_gdf)
        logger.info(f"Available commodities: {available_commodities}")
        
        # Determine which commodities to analyze
        if args.commodities:
            commodities = [c for c in args.commodities if c in available_commodities]
            if not commodities:
                logger.error(f"None of the specified commodities {args.commodities} are available in the dataset")
                return 1
        else:
            commodities = available_commodities
        
        logger.info(f"Analyzing {len(commodities)} commodities: {commodities}")
        
        # Analyze each commodity
        results = []
        for commodity in commodities:
            try:
                # Create a new set of args for each commodity
                commodity_args = argparse.Namespace(**vars(args))
                commodity_args.mode = mode  # Set the current mode
        
                result = analyze_commodity(commodity, processed_gdf, commodity_args, logger)
                # Ensure result is not None and has a success key
                if result is None:
                    result = {
                        'commodity': commodity,
                        'success': False,
                        'error': 'Unknown error - analyze_commodity returned None'
                    }
                elif 'success' not in result:
                    result['success'] = False
                    result['error'] = 'Result did not contain success status'
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {commodity}: {e}")
                logger.exception("Detailed traceback:")
                results.append({
                    'commodity': commodity,
                    'success': False,
                    'error': str(e)
                })
        
        # Store results for the current mode
        all_mode_results[mode] = results
        
        # Summarize results for the current mode
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"Analysis completed for {success_count} out of {len(commodities)} commodities in {mode} mode")
    
    # Compare results across modes
    best_mode = None
    best_success_count = -1
    
    for mode, results in all_mode_results.items():
        success_count = sum(1 for r in results if r['success'])
        if success_count > best_success_count:
            best_success_count = success_count
            best_mode = mode
    
    logger.info(f"Best mode: {best_mode} with {best_success_count} successful commodities")
    
    # Save summary report for the best mode
    summary_path = os.path.join(args.output, f"analysis_summary_{best_mode}.json")
    
    # Create a custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, (np.float64, np.float32, np.float16, np.number)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, int)):
                return int(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif obj is None:
                return None
            try:
                return super().default(obj)
            except:
                return str(obj)
    
    # Convert results to a basic Python list with serializable objects
    serializable_results = []
    for result in all_mode_results[best_mode]:
        if result is None:
            serializable_results.append(None)
            continue
        
        # Use the encoder to convert each result to a serializable dict
        result_json = json.dumps(result, cls=NumpyEncoder)
        result_dict = json.loads(result_json)
        serializable_results.append(result_dict)
    
    with open(summary_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Summary report saved to: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())