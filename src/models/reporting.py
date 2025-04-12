"""Results reporting module for Yemen Market Analysis.

This module provides functionality for generating reports from analysis results.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

class ResultsReporter:
    """
    Results reporter for Yemen Market Analysis.
    
    This class provides methods for generating reports from analysis results.
    
    Attributes:
        format (str): Output format for reports (markdown, latex, html).
        style (str): Style for reports.
        confidence_level (float): Confidence level for statistical tests.
        significance_indicators (bool): Whether to include significance indicators.
    """
    
    def __init__(
        self,
        format: Optional[str] = None,
        style: Optional[str] = None,
        confidence_level: Optional[float] = None,
        significance_indicators: Optional[bool] = None
    ):
        """
        Initialize the results reporter.
        
        Args:
            format: Output format for reports (markdown, latex, html).
            style: Style for reports.
            confidence_level: Confidence level for statistical tests.
            significance_indicators: Whether to include significance indicators.
        """
        self.format = format or config.get('reporting.format', 'markdown')
        self.style = style or config.get('reporting.style', 'world_bank')
        self.confidence_level = confidence_level or config.get('reporting.confidence_level', 0.95)
        self.significance_indicators = significance_indicators \
            if significance_indicators is not None \
            else config.get('reporting.significance_indicators', True)
    
    @handle_errors
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a report from analysis results.
        
        Args:
            results: Analysis results.
            output_path: Path to save the report to.
            
        Returns:
            Generated report as a string.
        """
        logger.info(f"Generating report in {self.format} format with {self.style} style")
        
        # Placeholder for actual report generation logic
        report = f"# Yemen Market Analysis Report\n\n"
        
        # Summary section
        report += f"## Summary\n\n"
        report += f"Analysis performed with {self.confidence_level*100}% confidence level.\n\n"
        
        # Add sections for each type of result
        for section, section_results in results.items():
            report += f"## {section.title()}\n\n"
            
            # Format depends on the type of results and desired output format
            if isinstance(section_results, dict):
                for key, value in section_results.items():
                    report += f"**{key}**: {value}\n\n"
            elif isinstance(section_results, pd.DataFrame):
                if self.format == 'markdown':
                    report += section_results.to_markdown()
                elif self.format == 'latex':
                    report += section_results.to_latex()
                elif self.format == 'html':
                    report += section_results.to_html()
                report += "\n\n"
        
        # Save report if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report

# Function to make the API more convenient
def generate_report(
    results: Dict[str, Any],
    format: Optional[str] = None, 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a report from analysis results.
    
    Args:
        results: Analysis results.
        format: Output format for the report.
        output_path: Path to save the report to.
        
    Returns:
        Generated report as a string.
    """
    reporter = ResultsReporter(format=format)
    return reporter.generate_report(results, output_path)