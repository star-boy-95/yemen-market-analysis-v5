#!/usr/bin/env python
"""
app.py

Command-line application for Yemen Market Analysis Project.
"""

import sys
import os
from pathlib import Path
import logging
import typer
from typing import List, Optional
from datetime import datetime

# Add project root to path to ensure imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from yemen_market_analysis.models.threshold.repository import MarketAnalysisRepository
from yemen_market_analysis.core.logging_setup import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Create app instance
app = typer.Typer(help="Yemen Market Analysis with Threshold Cointegration")

@app.command()
def analyze(
    config: str = typer.Option('config/config.yaml', help='Configuration file path'),
    data: Optional[str] = typer.Option(None, help='Data file path (overrides config)'),
    commodities: Optional[str] = typer.Option(None, help='Comma-separated list of commodities to analyze'),
    gpu: bool = typer.Option(False, help='Force GPU acceleration if available'),
    cpu: bool = typer.Option(False, help='Force CPU computation'),
    parallel: bool = typer.Option(False, help='Enable parallel processing'),
    output: Optional[str] = typer.Option(None, help='Custom output directory'),
    verbose: bool = typer.Option(False, help='Enable verbose logging'),
    enhance_quality: bool = typer.Option(False, help='Apply enhanced data quality procedures'),
    no_conflict_data: bool = typer.Option(False, help='Disable conflict-sensitive estimation')
):
    """Run Yemen Market Analysis with Threshold Cointegration."""
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Process commodities list if provided
    commodity_list = None
    if commodities:
        commodity_list = [c.strip() for c in commodities.split(',')]
    
    try:
        # Initialize repository with configuration
        repository = MarketAnalysisRepository(config)
        
        # Use provided data path or get from config
        data_path = data or repository.config.get('data.input_file')
        if not data_path or not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise typer.Exit(code=1)
        
        # Run analysis
        start_time = datetime.now()
        logger.info(f"Analysis started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        result = repository.run_analysis(
            data_path=data_path,
            commodities=commodity_list,
            use_parallel=parallel,
            output_dir=output,
            conflict_data=not no_conflict_data,
            enhanced_quality=enhance_quality
        )
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Analysis completed in {execution_time:.2f} seconds")
        
        # Report results
        if result.get("status") == "success":
            logger.info(f"Analysis completed successfully:")
            logger.info(f"- Commodities processed: {result.get('commodities_processed')}")
            logger.info(f"- Results saved to: {result.get('saved_path')}")
            logger.info(f"- Visualizations saved to: {result.get('visualization_dir')}")
            
            if result.get('enhanced_data_quality'):
                logger.info("- Enhanced data quality procedures were applied")
            if result.get('conflict_adjusted'):
                logger.info("- Conflict-adjusted threshold analysis was applied")
        else:
            logger.error(f"Analysis failed: {result.get('error', result.get('message', 'unknown error'))}")
            raise typer.Exit(code=1)
            
    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        raise typer.Exit(code=1)

@app.command()
def visualize(
    results_path: str = typer.Argument(..., help='Path to results file'),
    output_dir: Optional[str] = typer.Option(None, help='Output directory for visualizations'),
    config: str = typer.Option('config/config.yaml', help='Configuration file path'),
    format: str = typer.Option('png', help='Output format (png, svg, pdf)'),
    dpi: int = typer.Option(300, help='Resolution for raster formats')
):
    """Generate visualizations from existing results."""
    try:
        # Initialize repository with configuration
        repository = MarketAnalysisRepository(config)
        
        # Check if results file exists
        results_path = Path(results_path)
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            raise typer.Exit(code=1)
        
        # Create output directory if not provided
        if output_dir is None:
            output_dir = repository.config.get('directories.results_dir', 'results')
            output_dir = os.path.join(output_dir, "visualizations")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        logger.info(f"Loading results from {results_path}")
        results = repository.output_manager.read_json(results_path)
        
        if not results or "data" not in results:
            logger.error("Invalid results file format")
            raise typer.Exit(code=1)
        
        # Extract results data
        results_data = results.get("data", {})
        
        # Generate visualizations
        from yemen_market_analysis.visualization.visualization import generate_all_visualizations
        
        logger.info(f"Generating visualizations in {output_dir}")
        generate_all_visualizations(
            results_data, 
            None,  # No raw data available for additional plots
            output_dir,
            file_format=format,
            dpi=dpi
        )
        
        logger.info(f"Visualizations created successfully in {output_dir}")
        
    except Exception as e:
        logger.error(f"Visualization failed with error: {e}")
        raise typer.Exit(code=1)

@app.command()
def report(
    results_path: str = typer.Argument(..., help='Path to results file'),
    output_dir: Optional[str] = typer.Option(None, help='Output directory for reports'),
    config: str = typer.Option('config/config.yaml', help='Configuration file path'),
    format: str = typer.Option('html', help='Output format (html, pdf, docx)'),
    include_plots: bool = typer.Option(True, help='Include plots in report')
):
    """Generate comprehensive report from existing results."""
    try:
        # Initialize repository with configuration
        repository = MarketAnalysisRepository(config)
        
        # Check if results file exists
        results_path = Path(results_path)
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            raise typer.Exit(code=1)
        
        # Create output directory if not provided
        if output_dir is None:
            output_dir = repository.config.get('directories.results_dir', 'results')
            output_dir = os.path.join(output_dir, "reports")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        logger.info(f"Loading results from {results_path}")
        results = repository.output_manager.read_json(results_path)
        
        if not results or "data" not in results:
            logger.error("Invalid results file format")
            raise typer.Exit(code=1)
        
        # Extract results data
        results_data = results.get("data", {})
        
        # Generate report
        from yemen_market_analysis.reporting.reports import create_comprehensive_report
        
        logger.info(f"Generating {format} report in {output_dir}")
        report_path = create_comprehensive_report(
            results_data,
            output_dir,
            format=format,
            include_plots=include_plots
        )
        
        logger.info(f"Report created successfully: {report_path}")
        
    except Exception as e:
        logger.error(f"Report generation failed with error: {e}")
        raise typer.Exit(code=1)

@app.command()
def validate_data(
    data_path: str = typer.Argument(..., help='Path to data file'),
    config: str = typer.Option('config/config.yaml', help='Configuration file path'),
    output_file: Optional[str] = typer.Option(None, help='Output file for validation report'),
    verbose: bool = typer.Option(False, help='Show detailed validation results')
):
    """Validate data quality without running full analysis."""
    try:
        # Initialize repository with configuration
        repository = MarketAnalysisRepository(config)
        
        # Check if data file exists
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise typer.Exit(code=1)
        
        # Load data without full preprocessing
        from yemen_market_analysis.data.validators import validate_data_quality
        
        logger.info(f"Validating data from {data_path}")
        df, _ = repository.load_and_preprocess_data(data_path)
        
        if df is None or df.empty:
            logger.error("Data validation failed: could not load data")
            raise typer.Exit(code=1)
        
        # Run comprehensive validation
        validation_result = validate_data_quality(df)
        
        # Print summary to console
        if validation_result.get("valid", False):
            logger.info("Data validation passed")
            logger.info(f"Total observations: {validation_result.get('total_observations', 0)}")
            
            # Print quality stats
            quality_score = validation_result.get("quality_scores", {}).get("overall_quality_score", 0)
            quality_grade = validation_result.get("quality_scores", {}).get("quality_grade", "Unknown")
            logger.info(f"Data quality score: {quality_score:.1f}/100 ({quality_grade})")
            
            # Print recommendations
            recommendations = validation_result.get("quality_recommendations", [])
            if recommendations:
                logger.info("Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"  {i}. {rec}")
        else:
            logger.error(f"Data validation failed: {validation_result.get('error', 'Unknown error')}")
            raise typer.Exit(code=1)
        
        # Save report if output file specified
        if output_file:
            from yemen_market_analysis.core.utilities import save_json
            
            logger.info(f"Saving validation report to {output_file}")
            save_json(validation_result, output_file, indent=2)
            
        # Print detailed results if verbose
        if verbose:
            # Print regime coverage
            logger.info("\nRegime Coverage:")
            regime_coverage = validation_result.get("regime_coverage", {})
            available_regimes = regime_coverage.get("available_regimes", [])
            missing_regimes = regime_coverage.get("missing_regimes", [])
            logger.info(f"  Available regimes: {', '.join(available_regimes)}")
            if missing_regimes:
                logger.info(f"  Missing regimes: {', '.join(missing_regimes)}")
            
            # Print commodity stats
            logger.info("\nCommodity Statistics:")
            commodity_stats = validation_result.get("commodity_stats", {})
            for commodity, stats in commodity_stats.items():
                logger.info(f"  {commodity}:")
                logger.info(f"    North observations: {stats.get('north_observations', 0)}")
                logger.info(f"    South observations: {stats.get('south_observations', 0)}")
                logger.info(f"    Sufficient observations: {'Yes' if stats.get('sufficient_observations', False) else 'No'}")
                logger.info(f"    Date range: {stats.get('date_range', {}).get('start')} to {stats.get('date_range', {}).get('end')}")
            
            # Print volatility analysis
            logger.info("\nVolatility Analysis:")
            volatility_by_commodity = validation_result.get("volatility_analysis", {})
            for commodity, regimes in volatility_by_commodity.items():
                logger.info(f"  {commodity}:")
                for regime, stats in regimes.items():
                    logger.info(f"    {regime} regime:")
                    logger.info(f"      Mean volatility: {stats.get('mean_volatility', 0):.2f}%")
                    logger.info(f"      Extreme changes: {stats.get('extreme_changes', 0)}")
            
            # Print outlier analysis
            logger.info("\nOutlier Analysis:")
            outliers_by_commodity = validation_result.get("outlier_analysis", {})
            for commodity, regimes in outliers_by_commodity.items():
                logger.info(f"  {commodity}:")
                for regime, stats in regimes.items():
                    logger.info(f"    {regime} regime:")
                    logger.info(f"      Outlier count: {stats.get('outlier_count', 0)}")
                    logger.info(f"      Outlier ratio: {stats.get('outlier_ratio', 0) * 100:.2f}%")
        
    except Exception as e:
        logger.error(f"Data validation failed with error: {e}")
        raise typer.Exit(code=1)

@app.command()
def extract_thresholds(
    results_path: str = typer.Argument(..., help='Path to results directory or file'),
    output_file: str = typer.Argument(..., help='Output file for threshold values'),
    config: str = typer.Option('config/config.yaml', help='Configuration file path')
):
    """Extract threshold values from analysis results."""
    try:
        # Initialize repository with configuration
        repository = MarketAnalysisRepository(config)
        
        # Check if results path exists
        results_path = Path(results_path)
        if not results_path.exists():
            logger.error(f"Results path not found: {results_path}")
            raise typer.Exit(code=1)
        
        # Load results
        threshold_values = {}
        
        if results_path.is_file():
            # Single results file
            logger.info(f"Loading results from {results_path}")
            results = repository.output_manager.read_json(results_path)
            
            if results and "data" in results:
                # Extract threshold values
                for commodity, data in results["data"].items():
                    if "unified" in data and "threshold" in data["unified"]:
                        threshold_values[commodity] = {
                            "threshold": data["unified"]["threshold"],
                            "hs_threshold": data["unified"].get("hs_threshold"),
                            "p_value": data["unified"].get("p_value"),
                            "significant": data["unified"].get("threshold_significant", False)
                        }
        else:
            # Directory of results
            logger.info(f"Scanning result files in {results_path}")
            json_files = list(results_path.glob("**/*.json"))
            
            for file_path in json_files:
                if "metadata" in file_path.name or "manifest" in file_path.name:
                    continue
                    
                try:
                    results = repository.output_manager.read_json(file_path)
                    
                    if results and "data" in results:
                        # Extract threshold values
                        for commodity, data in results["data"].items():
                            if "unified" in data and "threshold" in data["unified"]:
                                threshold_values[commodity] = {
                                    "threshold": data["unified"]["threshold"],
                                    "hs_threshold": data["unified"].get("hs_threshold"),
                                    "p_value": data["unified"].get("p_value"),
                                    "significant": data["unified"].get("threshold_significant", False)
                                }
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        
        # Check if we found any threshold values
        if not threshold_values:
            logger.error("No threshold values found in results")
            raise typer.Exit(code=1)
        
        # Save threshold values
        from yemen_market_analysis.core.utilities import save_json
        
        logger.info(f"Saving {len(threshold_values)} threshold values to {output_file}")
        save_json(threshold_values, output_file, indent=2)
        
        # Print summary
        logger.info("\nThreshold Values Summary:")
        for commodity, values in threshold_values.items():
            significance = "Significant" if values.get("significant", False) else "Not significant"
            logger.info(f"  {commodity}: {values.get('threshold', 'N/A')} ({significance})")
        
    except Exception as e:
        logger.error(f"Threshold extraction failed with error: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()