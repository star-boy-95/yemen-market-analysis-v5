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

# Import the integrated analysis module
from src.run_integrated_analysis import run_integrated_analysis, setup_logging, parse_args


def main():
    """
    Main entry point for Yemen Market Integration Analysis.
    
    Parses command line arguments and runs the integrated analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Yemen Market Integration Analysis',
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
        default='./results',
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
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_file='yemen_analysis.log', level=level)
    
    logger.info("Yemen Market Integration Analysis")
    logger.info(f"Analyzing commodity: {args.commodity}")
    
    # Run the integrated analysis
    return run_integrated_analysis(args, logger)


if __name__ == "__main__":
    sys.exit(main())