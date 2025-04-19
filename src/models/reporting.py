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

    def _preprocess_results(self, results: Any) -> Any:
        """
        Preprocess results to ensure they can be properly formatted in reports.
        Converts NumPy types to Python native types and formats complex data structures.

        Args:
            results: Results to preprocess.

        Returns:
            Preprocessed results.
        """
        if isinstance(results, dict):
            return {k: self._preprocess_results(v) for k, v in results.items()}
        elif isinstance(results, list):
            return [self._preprocess_results(item) for item in results]
        elif isinstance(results, np.ndarray):
            return results.tolist()
        elif isinstance(results, np.number):
            return float(results)
        else:
            return results

    def _format_dataframe_summary(self, df: pd.DataFrame) -> str:
        """
        Create a concise summary of a DataFrame instead of including all the data.

        Args:
            df: DataFrame to summarize.

        Returns:
            Markdown string with DataFrame summary.
        """
        summary = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"

        # Add column information
        summary += "**Columns:** " + ", ".join(df.columns.tolist()) + "\n\n"

        # Add a small sample if the DataFrame is not too large
        if df.shape[0] > 0 and df.shape[1] > 0:
            # Get first few rows for preview (max 5)
            preview_rows = min(5, df.shape[0])
            preview_df = df.head(preview_rows)

            # Limit columns if there are too many
            if df.shape[1] > 6:
                preview_cols = list(df.columns[:5]) + ['...']
                summary += "**Preview (first 5 rows, first 5 columns):**\n"
            else:
                preview_cols = df.columns
                summary += "**Preview (first 5 rows):**\n"

            # Create a small preview table
            preview_table = preview_df[preview_df.columns[:5]].copy()
            summary += preview_table.to_markdown(index=True, tablefmt="pipe", floatfmt=".4f")
            summary += "\n\n"

        return summary

    @handle_errors
    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None,
                      publication_quality: bool = True, output_dir: Optional[str] = None) -> str:
        """
        Generate a report from analysis results.

        Args:
            results: Analysis results.
            output_path: Path to save the report to.
            publication_quality: Whether to generate publication-quality report.
            output_dir: Directory to save the report to (alternative to output_path).

        Returns:
            Generated report as a string.
        """
        logger.info(f"Generating report in {self.format} format with {self.style} style")

        # Preprocess results to handle complex data types
        processed_results = self._preprocess_results(results)

        # Start building the report
        report = f"# Yemen Market Analysis Report\n\n"

        # Summary section
        report += f"## Summary\n\n"
        report += f"Analysis performed with {self.confidence_level*100}% confidence level.\n\n"

        # Add commodity information if available
        if 'data' in processed_results and processed_results['data']:
            first_market = next(iter(processed_results['data']))
            if 'commodity' in processed_results['data'][first_market].columns:
                commodity = processed_results['data'][first_market]['commodity'].iloc[0]
                report += f"Commodity analyzed: **{commodity}**\n\n"

            # Add number of markets
            report += f"Number of markets analyzed: **{len(processed_results['data'])}**\n\n"
            report += f"Markets: {', '.join(sorted(processed_results['data'].keys()))}\n\n"

        # Add sections for each type of result (excluding raw data)
        for section, section_results in processed_results.items():
            # Skip the raw data section
            if section == 'data':
                continue

            report += f"## {section.title()}\n\n"

            # Format depends on the type of results
            if isinstance(section_results, dict):
                # For visualization results, provide more detailed information
                if section == 'visualizations':
                    for viz_type, viz_results in section_results.items():
                        report += f"### {viz_type.replace('_', ' ').title()}\n\n"
                        if isinstance(viz_results, dict) and viz_results:
                            # Create a table of visualizations with file paths
                            report += f"Generated {len(viz_results)} visualizations for the following markets:\n\n"

                            # Create a more structured table of visualizations
                            viz_table = []
                            for i, (market, file_path) in enumerate(sorted(viz_results.items())[:10], 1):
                                # Handle both string paths and nested dictionaries
                                if isinstance(file_path, str):
                                    viz_name = file_path.split('/')[-1]
                                else:
                                    # For nested dictionaries, just use a placeholder
                                    viz_name = f"Multiple visualizations"
                                viz_table.append({"Market": market, "Visualization": viz_name})

                            if viz_table:
                                viz_df = pd.DataFrame(viz_table)
                                report += viz_df.to_markdown(index=False, tablefmt="pipe")

                                if len(viz_results) > 10:
                                    report += f"\n\n*...and {len(viz_results) - 10} more visualizations*"

                            report += "\n\n"
                else:
                    # For other dictionary results, format key-value pairs
                    for key, value in section_results.items():
                        if isinstance(value, pd.DataFrame):
                            report += f"### {key}\n\n"
                            report += self._format_dataframe_summary(value)
                        elif isinstance(value, dict):
                            report += f"### {key}\n\n"
                            # Special handling for unit root and cointegration results
                            if section in ['unit_root', 'cointegration']:
                                # Extract test results and format as a table
                                test_results = {}
                                for test_name, test_data in value.items():
                                    if test_name != 'overall':
                                        # Extract key statistics from test results
                                        if isinstance(test_data, dict):
                                            # Unit root tests
                                            if 'test_statistic' in test_data:
                                                test_results[f"{test_name}_stat"] = test_data.get('test_statistic')
                                            if 'p_value' in test_data:
                                                test_results[f"{test_name}_pval"] = test_data.get('p_value')
                                            if 'is_stationary' in test_data:
                                                test_results[f"{test_name}_result"] = 'Stationary' if test_data.get('is_stationary') else 'Non-stationary'
                                            # Cointegration tests
                                            if 'is_cointegrated' in test_data:
                                                test_results[f"{test_name}_result"] = 'Cointegrated' if test_data.get('is_cointegrated') else 'Not cointegrated'

                                if test_results:
                                    # Convert to DataFrame for table formatting
                                    test_df = pd.DataFrame([test_results])
                                    report += test_df.to_markdown(index=False, tablefmt="pipe", floatfmt=".4f")
                                    report += "\n\n"
                                else:
                                    # Fallback to simple format if no test results found
                                    for subkey, subvalue in value.items():
                                        if isinstance(subvalue, (int, float, str, bool)):
                                            report += f"**{subkey}**: {subvalue}\n"
                                        else:
                                            report += f"**{subkey}**: {type(subvalue).__name__}\n"
                                    report += "\n"
                            else:
                                # Format nested dictionaries more concisely
                                # Create a small table for the dictionary values
                                if len(value) > 0:
                                    # Extract primitive values for a table
                                    table_data = {}
                                    for subkey, subvalue in value.items():
                                        if isinstance(subvalue, (int, float, str, bool)):
                                            table_data[subkey] = subvalue

                                    if table_data:
                                        # Convert to DataFrame for table formatting
                                        table_df = pd.DataFrame([table_data])
                                        report += table_df.to_markdown(index=False, tablefmt="pipe", floatfmt=".4f")
                                        report += "\n\n"
                                    else:
                                        # Just list the keys and types if no primitive values
                                        for subkey, subvalue in value.items():
                                            report += f"**{subkey}**: {type(subvalue).__name__}\n"
                                        report += "\n"
                        else:
                            report += f"**{key}**: {value}\n\n"
            elif isinstance(section_results, pd.DataFrame):
                # Format DataFrames as proper markdown tables
                if self.format == 'markdown':
                    report += self._format_dataframe_summary(section_results)
                elif self.format == 'latex':
                    report += section_results.to_latex(index=True, float_format="%.4f")
                    report += "\n\n"
                elif self.format == 'html':
                    report += section_results.to_html(index=True, float_format="%.4f")
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
    output_path: Optional[str] = None,
    publication_quality: bool = True,
    output_dir: Optional[str] = None
) -> str:
    """
    Generate a report from analysis results.

    Args:
        results: Analysis results.
        format: Output format for the report.
        output_path: Path to save the report to.
        publication_quality: Whether to generate publication-quality report.
        output_dir: Directory to save the report to (alternative to output_path).

    Returns:
        Generated report as a string.
    """
    reporter = ResultsReporter(format=format)
    return reporter.generate_report(results, output_path, publication_quality, output_dir)