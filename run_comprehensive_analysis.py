#!/usr/bin/env python
"""
Comprehensive Yemen Market Analysis Script

This script integrates all analysis components from the Yemen Market Integration project
into a single, comprehensive analysis pipeline. It serves as a unified entry point for
running the full suite of econometric analyses on market data.

Features:
- Unified data loading and preprocessing
- Comprehensive model selection and comparison
- Enhanced reporting with World Bank standards
- Cross-commodity comparative analysis
- Publication-quality visualizations

Example usage:
    # Run full analysis with all commodities and models
    python run_comprehensive_analysis.py --all-commodities --all-models --output full_report
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import warnings
import json

# Add the parent directory to the Python path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import project modules
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester
from src.models.threshold_model import ThresholdModel
from src.models.spatial import SpatialEconometrics
from src.models.model_selection import ModelComparer
from src.models.reporting import generate_comprehensive_report
from src.visualization.enhanced_econometric_reporting import EconometricReporter
from src.utils.performance_utils import configure_system_for_performance
from src.utils.logging_setup import setup_logging

# Import from run_yemen_market_analysis.py to ensure consistency
from run_yemen_market_analysis import (
    analyze_threshold_models,
    parse_arguments,
    load_and_preprocess_data,
    get_commodity_list,
    setup_output_directory
)

def main():
    """Main function to run the comprehensive analysis pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Yemen Market Integration Comprehensive Analysis")
    
    # Configure system for optimal performance
    configure_system_for_performance()
    
    # Load and preprocess data
    data, commodities = load_and_preprocess_data(args)
    
    # Set up output directory
    output_dir = setup_output_directory(args.output)
    
    # Process each commodity
    results = {}
    logger.info(f"Processing {len(commodities)} commodities sequentially")
    
    for commodity in commodities:
        logger.info(f"Processing commodity: {commodity}")
        
        # Run threshold model analysis
        threshold_results = analyze_threshold_models(
            data, 
            commodity, 
            threshold_modes=args.threshold_modes,
            price_column=args.price_column,
            aggregation=args.aggregation
        )
        
        # Store results
        results[commodity] = {
            'threshold_models': threshold_results
        }
        
        # Generate enhanced report if requested
        if args.enhanced_reporting:
            try:
                reporter = EconometricReporter(
                    output_dir=output_dir,
                    publication_quality=True,
                    style='world_bank',
                    format='markdown'
                )
                
                report_path = reporter.generate_model_comparison_report(
                    model_results=threshold_results,
                    commodity=commodity,
                    output_file=f"{commodity.replace(' ', '_')}/model_comparison"
                )
                
                logger.info(f"Enhanced report generated at {report_path}")
            except Exception as e:
                logger.error(f"Error generating enhanced report for {commodity}: {str(e)}")
    
    # Generate cross-commodity comparison if requested
    if args.comparative_analysis and len(commodities) > 1:
        try:
            reporter = EconometricReporter(
                output_dir=output_dir,
                publication_quality=True,
                style='world_bank',
                format='markdown'
            )
            
            cross_report_path = reporter.generate_cross_commodity_report(
                all_results=results,
                output_file="cross_commodity_comparison"
            )
            
            logger.info(f"Cross-commodity comparison report generated at {cross_report_path}")
        except Exception as e:
            logger.error(f"Error generating cross-commodity report: {str(e)}")
    
    # Save complete results to JSON
    results_file = output_dir / "complete_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Complete results saved to {results_file}")
    except Exception as e:
        logger.error(f"Error saving complete results: {str(e)}")
    
    logger.info("Comprehensive analysis completed")
    return results

if __name__ == "__main__":
    main()