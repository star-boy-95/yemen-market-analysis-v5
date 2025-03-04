"""
Argument parsing utilities for Yemen Market Analysis CLI.
"""
import argparse
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Yemen Market Analysis - Threshold Cointegration Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        "--config", "-c",
        dest="config_path",
        help="Path to configuration file (YAML)",
        default="config/config.yaml"
    )
    
    # Input options
    parser.add_argument(
        "--data", "-d",
        dest="data_path",
        help="Path to input data file (CSV or GeoJSON)",
        default="data/unified_data.geojson"
    )
    
    parser.add_argument(
        "--conflict", 
        dest="conflict_path",
        help="Path to conflict data file (optional)",
        default=None
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        dest="output_dir",
        help="Output directory for results",
        default=None
    )
    
    # Analysis options
    parser.add_argument(
        "--commodities",
        help="Comma-separated list of commodities to analyze (default: all)",
        default=None
    )
    
    parser.add_argument(
        "--model",
        choices=["hansen_seo", "enders_siklos", "both"],
        help="Threshold model to use",
        default="hansen_seo"
    )
    
    # Computation options
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU computation (no GPU)"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU computation (if available)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Maximum number of parallel workers",
        default=None
    )
    
    # Visualization options
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Generate HTML reports"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser


def parse_commodities(commodities_str: Optional[str]) -> List[str]:
    """Parse comma-separated list of commodities."""
    if not commodities_str:
        return []
        
    return [c.strip() for c in commodities_str.split(",") if c.strip()]