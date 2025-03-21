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
    python src/run_integrated_analysis_fixed.py --data data/raw/unified_data.geojson \
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

# Import project modules using relative imports
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.unit_root import UnitRootTester
from models.cointegration import CointegrationTester
from models.threshold import ThresholdCointegration
from models.spatial import SpatialEconometrics
from models.simulation import MarketIntegrationSimulation
from models.diagnostics import ModelDiagnostics

# Import new integration modules
from models.spatiotemporal import integrate_time_series_spatial_results
from models.interpretation import (
    interpret_unit_root_results,
    interpret_cointegration_results,
    interpret_threshold_results,
    interpret_spatial_results,
    interpret_simulation_results
)
from models.reporting import (
    generate_comprehensive_report,
    create_executive_summary,
    export_results_for_publication
)

from visualization.time_series import TimeSeriesVisualizer
from visualization.maps import MarketMapVisualizer
from utils.performance_utils import timer, memory_usage_decorator, optimize_dataframe, parallelize_dataframe
from utils.validation import validate_data, validate_model_inputs
from utils.error_handler import handle_errors, ModelError, DataError, capture_error
from utils.config import config
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


# Rest of the file remains the same, just with the imports changed
# ...

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