#!/usr/bin/env python
"""
Yemen Market Integration Analysis - Main Entry Point

This script provides a convenient entry point for running the full integrated
analysis of market integration in Yemen. It includes:
- Unit root testing with structural break detection
- Cointegration analysis with multiple methods
- Threshold models with asymmetric adjustment
- Spatial econometrics with conflict adjustment
- Policy simulation with comprehensive welfare analysis
- Spatiotemporal integration of results
- Detailed interpretation and reporting

Example usage:
    python run_yemen_analysis.py --commodity "beans (kidney red)" --output results
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Import the integrated analysis module using consistent package name
from yemen_market_integration.run_integrated_analysis import run_integrated_analysis, setup_logging, parse_args
from yemen_market_integration.utils.error_handler import capture_error
from yemen_market_integration.utils.config import config


def main():
    """
    Main entry point for Yemen Market Integration Analysis.
    
    Parses command line arguments and runs the integrated analysis.
    This function serves as a convenient wrapper around the core
    integrated analysis functionality.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Yemen Market Integration Analysis',
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
        
        # Validate arguments
        if not args.commodity:
            print("Error: No commodity specified for analysis")
            return 1
            
        if not args.data:
            print("Error: No data file specified")
            return 1
        
        # Set up logging
        level = logging.DEBUG if args.verbose else logging.INFO
        logger = setup_logging(log_file='yemen_analysis.log', level=level)
        
        logger.info("Yemen Market Integration Analysis")
        logger.info(f"Analyzing commodity: {args.commodity}")
        logger.info(f"Using data file: {args.data}")
        logger.info(f"Output directory: {args.output}")
        
        # Run the integrated analysis
        return run_integrated_analysis(args, logger)
    
    except Exception as e:
        # Capture any unhandled exceptions
        try:
            capture_error(e, context="Main entry point", logger=logger)
            logger.error(f"Unhandled error in main entry point: {e}")
            logger.exception("Detailed traceback:")
        except:
            # If logger is not available, print to stderr
            print(f"Critical error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())