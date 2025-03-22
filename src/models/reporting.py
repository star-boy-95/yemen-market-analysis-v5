"""
Reporting module for Yemen Market Integration project.

This module provides functions for generating comprehensive reports and
executive summaries based on analysis results, following World Bank standards
and academic publication requirements.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, Union, Optional, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import visualization modules
from src.visualization.enhanced_econometric_reporting import (
    generate_enhanced_report,
    generate_cross_commodity_comparison,
    EconometricReporter
)

# Import utility functions
from yemen_market_integration.utils import (
    handle_errors,
    validate_dataframe,
    validate_geodataframe,
    raise_if_invalid,
    config
)

# Create logger
logger = logging.getLogger(__name__)


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def generate_comprehensive_report(
    all_results: Dict[str, Any], 
    commodity: str, 
    output_path: Union[str, Path], 
    format: str = 'markdown',
    publication_quality: bool = False,
    confidence_level: float = 0.95,
    significance_indicators: bool = True,
    style: str = 'world_bank',
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Generate a comprehensive report of all analysis results.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all analysis results
    commodity : str
        Commodity name
    output_path : pathlib.Path or str
        Path to save the report
    format : str
        Output format ('markdown', 'latex', 'html', 'json')
    publication_quality : bool
        Whether to generate publication-quality output
    confidence_level : float
        Confidence level for intervals (0.90, 0.95, or 0.99)
    significance_indicators : bool
        Whether to add significance indicators (*, **, ***)
    style : str
        Visual style for reports ('world_bank', 'academic', 'policy')
    logger : logging.Logger, optional
        Logger instance. If None, uses the module logger.
        
    Returns
    -------
    pathlib.Path
        Path to the generated report
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        from src.models.interpretation import (
            interpret_unit_root_results,
            interpret_cointegration_results,
            interpret_threshold_results,
            interpret_spatial_results,
            interpret_simulation_results
        )
    except ImportError:
        logger.warning("Could not import interpretation modules. Some report sections may be limited.")
        
        # Create dummy interpretation functions if the real ones aren't available
        def interpret_unit_root_results(results, commodity):
            return {'summary': 'Unit root analysis results available', 'implications': ['Limited interpretation available']}
            
        def interpret_cointegration_results(results, commodity):
            return {'summary': 'Cointegration analysis results available', 'implications': ['Limited interpretation available']}
            
        def interpret_threshold_results(results, commodity):
            return {'summary': 'Threshold analysis results available', 'implications': ['Limited interpretation available']}
            
        def interpret_spatial_results(results, commodity):
            return {'summary': 'Spatial analysis results available', 'implications': ['Limited interpretation available']}
            
        def interpret_simulation_results(results, commodity):
            return {'summary': 'Simulation results available', 'implications': ['Limited interpretation available'], 'policy_recommendations': []}
    
    # Ensure output_path is a Path object
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Generate interpretations
    interpretations = {}
    
    if 'unit_root_results' in all_results:
        interpretations['unit_root'] = interpret_unit_root_results(
            all_results['unit_root_results'], commodity
        )
    
    if 'cointegration_results' in all_results:
        interpretations['cointegration'] = interpret_cointegration_results(
            all_results['cointegration_results'], commodity
        )
    
    if 'threshold_results' in all_results:
        interpretations['threshold'] = interpret_threshold_results(
            all_results['threshold_results'], commodity
        )
    
    if 'spatial_results' in all_results:
        interpretations['spatial'] = interpret_spatial_results(
            all_results['spatial_results'], commodity
        )
    
    if 'simulation_results' in all_results:
        interpretations['simulation'] = interpret_simulation_results(
            all_results['simulation_results'], commodity
        )
    
    # Add interpretations to results
    all_results['interpretations'] = interpretations
    
    # Use enhanced reporting module for publication-quality output
    if publication_quality:
        return generate_enhanced_report(
            all_results=all_results,
            commodity=commodity,
            output_dir=output_path.parent,
            format=format,
            publication_quality=True,
            style=style
        )
    
    # Generate basic report if not using publication quality
    report_path = output_path
    
    # Ensure file has correct extension
    if not str(report_path).endswith(f'.{format}'):
        report_path = Path(f"{str(report_path)}.{format}")
    
    # Generate report content based on format
    if format == 'markdown':
        content = _generate_markdown_report(all_results, commodity, interpretations)
    elif format == 'latex':
        content = _generate_latex_report(all_results, commodity, interpretations)
    elif format == 'html':
        content = _generate_html_report(all_results, commodity, interpretations)
    elif format == 'json':
        # For JSON, just save the results directly
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        return report_path
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Write report to file
    with open(report_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Comprehensive report saved to {report_path}")
    
    return report_path


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def create_executive_summary(
    all_results: Dict[str, Any],
    commodity: str,
    output_path: Union[str, Path],
    format: str = 'markdown',
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Create an executive summary of analysis results.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all analysis results
    commodity : str
        Commodity name
    output_path : pathlib.Path or str
        Path to save the summary
    format : str
        Output format ('markdown', 'latex', 'html', 'json')
    logger : logging.Logger, optional
        Logger instance. If None, uses the module logger.
        
    Returns
    -------
    pathlib.Path
        Path to the generated summary
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Ensure output_path is a Path object
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Extract key findings
    findings = _extract_key_findings(all_results, commodity)
    
    # Ensure file has correct extension
    if not str(output_path).endswith(f'.{format}'):
        output_path = Path(f"{str(output_path)}.{format}")
    
    # Generate summary content based on format
    if format == 'markdown':
        content = _generate_markdown_summary(findings)
    elif format == 'latex':
        content = _generate_latex_summary(findings)
    elif format == 'html':
        content = _generate_html_summary(findings)
    elif format == 'json':
        # For JSON, just save the findings directly
        with open(output_path, 'w') as f:
            json.dump(findings, f, indent=2, default=str)
        logger.info(f"Executive summary saved to {output_path}")
        return output_path
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Write summary to file
    with open(output_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Executive summary saved to {output_path}")
    
    return output_path


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def export_results_for_publication(
    all_results: Dict[str, Any],
    commodity: str,
    output_dir: Union[str, Path],
    formats: List[str] = ['latex', 'png'],
    dpi: int = 300
) -> Dict[str, Path]:
    """
    Export analysis results in publication-ready formats.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all analysis results
    commodity : str
        Commodity name
    output_dir : pathlib.Path or str
        Directory to save the exports
    formats : list of str
        Output formats to generate
    dpi : int
        Resolution for exported figures
        
    Returns
    -------
    dict
        Dictionary mapping export types to file paths
    """
    # Ensure output_dir is a Path object
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize exports dictionary
    exports = {}
    
    # Export comprehensive report
    if 'markdown' in formats:
        report_path = generate_comprehensive_report(
            all_results,
            commodity,
            output_dir / f"{commodity.replace(' ', '_')}_report",
            format='markdown'
        )
        exports['report_markdown'] = report_path
    
    if 'latex' in formats:
        report_path = generate_comprehensive_report(
            all_results,
            commodity,
            output_dir / f"{commodity.replace(' ', '_')}_report",
            format='latex'
        )
        exports['report_latex'] = report_path
    
    # Export model comparison report
    if 'threshold_results' in all_results and 'model_comparison' in all_results['threshold_results']:
        report_path = EconometricReporter(
            output_dir=output_dir,
            format='latex'
        ).generate_model_comparison_report(
            all_results['threshold_results']['model_comparison'],
            commodity,
            format='latex'
        )
        exports['model_comparison'] = report_path
    
    # Export tables
    if 'latex' in formats and 'threshold_results' in all_results:
        table_path = output_dir / f"{commodity.replace(' ', '_')}_tables.tex"
        _export_latex_tables(all_results['threshold_results'], commodity, table_path)
        exports['tables'] = table_path
    
    # Export figures
    if any(fmt in formats for fmt in ['png', 'pdf', 'svg']) and 'threshold_results' in all_results:
        figure_paths = _export_figures(all_results['threshold_results'], commodity, output_dir, formats, dpi)
        exports.update(figure_paths)
    
    logger.info(f"Exported {len(exports)} publication-ready files to {output_dir}")
    
    return exports


def _extract_key_findings(all_results: Dict[str, Any], commodity: str) -> Dict[str, Any]:
    """Extract key findings from analysis results."""
    findings = {
        'commodity': commodity,
        'timestamp': datetime.now().isoformat(),
        'summary': f"Market integration analysis for {commodity}",
        'key_points': []
    }
    
    # Extract unit root findings
    if 'unit_root_results' in all_results:
        unit_root = all_results['unit_root_results']
        if isinstance(unit_root, dict):
            stationary = unit_root.get('north_stationary', False) and unit_root.get('south_stationary', False)
            findings['key_points'].append(
                f"Price series {'are' if stationary else 'are not'} stationary"
            )
    
    # Extract cointegration findings
    if 'cointegration_results' in all_results:
        cointegration = all_results['cointegration_results']
        if isinstance(cointegration, dict):
            cointegrated = cointegration.get('cointegrated', False)
            findings['key_points'].append(
                f"Markets {'are' if cointegrated else 'are not'} cointegrated"
            )
    
    # Extract threshold findings
    if 'threshold_results' in all_results:
        threshold = all_results['threshold_results']
        if isinstance(threshold, dict):
            threshold_value = threshold.get('threshold')
            if threshold_value is not None:
                findings['key_points'].append(
                    f"Threshold value: {threshold_value:.4f}"
                )
            
            asymmetric = threshold.get('asymmetric_adjustment', False)
            if asymmetric:
                findings['key_points'].append(
                    "Markets exhibit asymmetric price adjustment"
                )
    
    # Extract spatial findings
    if 'spatial_results' in all_results:
        spatial = all_results['spatial_results']
        if isinstance(spatial, dict):
            spatial_autocorrelation = spatial.get('global_moran_significant', False)
            findings['key_points'].append(
                f"Spatial autocorrelation {'is' if spatial_autocorrelation else 'is not'} significant"
            )
    
    # Extract simulation findings
    if 'simulation_results' in all_results:
        simulation = all_results['simulation_results']
        if isinstance(simulation, dict):
            welfare_gain = simulation.get('welfare_gain')
            if welfare_gain is not None:
                findings['key_points'].append(
                    f"Estimated welfare gain: {welfare_gain:.2f}%"
                )
    
    # Extract interpretations
    if 'interpretations' in all_results:
        interpretations = all_results['interpretations']
        if isinstance(interpretations, dict):
            for interp_type, interp_data in interpretations.items():
                if isinstance(interp_data, dict) and 'implications' in interp_data:
                    for implication in interp_data['implications']:
                        findings['key_points'].append(implication)
    
    return findings


def _generate_markdown_report(
    all_results: Dict[str, Any],
    commodity: str,
    interpretations: Dict[str, Any]
) -> str:
    """Generate a markdown report."""
    # Start building the report
    report = [
        f"# Market Integration Analysis Report: {commodity}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Add interpretations
    if interpretations:
        for interp_type, interp_data in interpretations.items():
            if isinstance(interp_data, dict) and 'summary' in interp_data:
                report.append(f"- {interp_data['summary']}")
        
        report.append("")
    
    # Add unit root section
    if 'unit_root_results' in all_results:
        report.extend([
            "## Unit Root Analysis",
            ""
        ])
        
        unit_root = all_results['unit_root_results']
        if isinstance(unit_root, dict):
            # Add unit root test results
            report.extend([
                "### ADF Test Results",
                "",
                "| Market | Test Statistic | p-value | Critical Value (5%) | Stationary |",
                "|--------|---------------|---------|---------------------|------------|"
            ])
            
            # North market
            north_adf = unit_root.get('north_adf', {})
            if isinstance(north_adf, dict):
                stat = north_adf.get('statistic')
                pval = north_adf.get('pvalue')
                crit = north_adf.get('critical_values', {}).get('5%')
                stationary = unit_root.get('north_stationary', False)
                
                stat_str = f"{stat:.4f}" if isinstance(stat, (int, float)) else "N/A"
                pval_str = f"{pval:.4f}" if isinstance(pval, (int, float)) else "N/A"
                crit_str = f"{crit:.4f}" if isinstance(crit, (int, float)) else "N/A"
                
                report.append(f"| North | {stat_str} | {pval_str} | {crit_str} | {'Yes' if stationary else 'No'} |")
            
            # South market
            south_adf = unit_root.get('south_adf', {})
            if isinstance(south_adf, dict):
                stat = south_adf.get('statistic')
                pval = south_adf.get('pvalue')
                crit = south_adf.get('critical_values', {}).get('5%')
                stationary = unit_root.get('south_stationary', False)
                
                stat_str = f"{stat:.4f}" if isinstance(stat, (int, float)) else "N/A"
                pval_str = f"{pval:.4f}" if isinstance(pval, (int, float)) else "N/A"
                crit_str = f"{crit:.4f}" if isinstance(crit, (int, float)) else "N/A"
                
                report.append(f"| South | {stat_str} | {pval_str} | {crit_str} | {'Yes' if stationary else 'No'} |")
            
            report.append("")
            
            # Add unit root conclusion
            stationary = unit_root.get('north_stationary', False) and unit_root.get('south_stationary', False)
            report.append(f"**Conclusion:** Price series {'are' if stationary else 'are not'} stationary.")
            report.append("")
            
            # Add interpretation
            if 'unit_root' in interpretations:
                unit_root_interp = interpretations['unit_root']
                if isinstance(unit_root_interp, dict):
                    report.append("### Interpretation")
                    report.append("")
                    report.append(unit_root_interp.get('summary', ''))
                    report.append("")
                    
                    if 'implications' in unit_root_interp:
                        report.append("**Implications:**")
                        report.append("")
                        for implication in unit_root_interp['implications']:
                            report.append(f"- {implication}")
                        report.append("")
    
    # Add cointegration section
    if 'cointegration_results' in all_results:
        report.extend([
            "## Cointegration Analysis",
            ""
        ])
        
        cointegration = all_results['cointegration_results']
        if isinstance(cointegration, dict):
            # Add Johansen test results
            jo_result = cointegration.get('johansen_test', {})
            if isinstance(jo_result, dict):
                report.extend([
                    "### Johansen Test Results",
                    "",
                    "| Rank | Trace Statistic | Critical Value (5%) | p-value |",
                    "|------|----------------|---------------------|---------|"
                ])
                
                trace_stat = jo_result.get('trace_stat')
                crit_vals = jo_result.get('crit_vals')
                pvals = jo_result.get('pvals')
                
                if isinstance(trace_stat, list) and isinstance(crit_vals, list) and isinstance(pvals, list):
                    for i, (stat, crit, pval) in enumerate(zip(trace_stat, crit_vals, pvals)):
                        stat_str = f"{stat:.4f}" if isinstance(stat, (int, float)) else "N/A"
                        crit_str = f"{crit:.4f}" if isinstance(crit, (int, float)) else "N/A"
                        pval_str = f"{pval:.4f}" if isinstance(pval, (int, float)) else "N/A"
                        
                        report.append(f"| {i} | {stat_str} | {crit_str} | {pval_str} |")
                
                report.append("")
            
            # Add cointegration conclusion
            cointegrated = cointegration.get('cointegrated', False)
            report.append(f"**Conclusion:** Markets {'are' if cointegrated else 'are not'} cointegrated.")
            report.append("")
            
            # Add interpretation
            if 'cointegration' in interpretations:
                cointegration_interp = interpretations['cointegration']
                if isinstance(cointegration_interp, dict):
                    report.append("### Interpretation")
                    report.append("")
                    report.append(cointegration_interp.get('summary', ''))
                    report.append("")
                    
                    if 'implications' in cointegration_interp:
                        report.append("**Implications:**")
                        report.append("")
                        for implication in cointegration_interp['implications']:
                            report.append(f"- {implication}")
                        report.append("")
    
    # Join report lines into a single string
    return "\n".join(report)


def _generate_latex_report(
    all_results: Dict[str, Any],
    commodity: str,
    interpretations: Dict[str, Any]
) -> str:
    """Generate a LaTeX report."""
    # Start building the report
    report = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{graphicx}",
        "\\usepackage{amsmath}",
        "\\usepackage{hyperref}",
        "\\usepackage{float}",
        "\\usepackage[margin=1in]{geometry}",
        "",
        f"\\title{{Market Integration Analysis: {commodity}}}",
        "\\author{Yemen Market Integration Project}",
        f"\\date{{{datetime.now().strftime('%B %d, %Y')}}}",
        "",
        "\\begin{document}",
        "",
        "\\maketitle",
        "",
        "\\section{Executive Summary}",
        ""
    ]
    
    # Add interpretations
    if interpretations:
        report.append("\\begin{itemize}")
        for interp_type, interp_data in interpretations.items():
            if isinstance(interp_data, dict) and 'summary' in interp_data:
                report.append(f"\\item {interp_data['summary']}")
        report.append("\\end{itemize}")
        report.append("")
    
    # Add unit root section
    if 'unit_root_results' in all_results:
        report.extend([
            "\\section{Unit Root Analysis}",
            ""
        ])
        
        unit_root = all_results['unit_root_results']
        if isinstance(unit_root, dict):
            # Add unit root test results
            report.extend([
                "\\subsection{ADF Test Results}",
                "",
                "\\begin{table}[H]",
                "\\centering",
                "\\caption{Augmented Dickey-Fuller Test Results}",
                "\\begin{tabular}{lrrrc}",
                "\\toprule",
                "Market & Test Statistic & p-value & Critical Value (5\\%) & Stationary \\\\",
                "\\midrule"
            ])
            
            # North market
            north_adf = unit_root.get('north_adf', {})
            if isinstance(north_adf, dict):
                stat = north_adf.get('statistic')
                pval = north_adf.get('pvalue')
                crit = north_adf.get('critical_values', {}).get('5%')
                stationary = unit_root.get('north_stationary', False)
                
                stat_str = f"{stat:.4f}" if isinstance(stat, (int, float)) else "N/A"
                pval_str = f"{pval:.4f}" if isinstance(pval, (int, float)) else "N/A"
                crit_str = f"{crit:.4f}" if isinstance(crit, (int, float)) else "N/A"
                
                report.append(f"North & {stat_str} & {pval_str} & {crit_str} & {'Yes' if stationary else 'No'} \\\\")
            
            # South market
            south_adf = unit_root.get('south_adf', {})
            if isinstance(south_adf, dict):
                stat = south_adf.get('statistic')
                pval = south_adf.get('pvalue')
                crit = south_adf.get('critical_values', {}).get('5%')
                stationary = unit_root.get('south_stationary', False)
                
                stat_str = f"{stat:.4f}" if isinstance(stat, (int, float)) else "N/A"
                pval_str = f"{pval:.4f}" if isinstance(pval, (int, float)) else "N/A"
                crit_str = f"{crit:.4f}" if isinstance(crit, (int, float)) else "N/A"
                
                report.append(f"South & {stat_str} & {pval_str} & {crit_str} & {'Yes' if stationary else 'No'} \\\\")
            
            report.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                ""
            ])
    
    # End document
    report.extend([
        "\\end{document}"
    ])
    
    # Join report lines into a single string
    return "\n".join(report)


def _generate_html_report(
    all_results: Dict[str, Any],
    commodity: str,
    interpretations: Dict[str, Any]
) -> str:
    """Generate an HTML report."""
    # Start building the report
    report = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>Market Integration Analysis: {commodity}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #3498db; margin-top: 30px; }",
        "h3 { color: #2980b9; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
        "th { background-color: #f2f2f2; }",
        "tr:hover { background-color: #f5f5f5; }",
        ".conclusion { font-weight: bold; margin: 20px 0; }",
        ".implications { margin-left: 20px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Market Integration Analysis Report: {commodity}</h1>",
        f"<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>",
        "<h2>Executive Summary</h2>"
    ]
    
    # Add interpretations
    if interpretations:
        report.append("<ul>")
        for interp_type, interp_data in interpretations.items():
            if isinstance(interp_data, dict) and 'summary' in interp_data:
                report.append(f"<li>{interp_data['summary']}</li>")
        report.append("</ul>")
    
    # End document
    report.append("</body>")
    report.append("</html>")
    
    # Join report lines into a single string
    return "\n".join(report)


def _generate_markdown_summary(findings: Dict[str, Any]) -> str:
    """Generate a markdown executive summary."""
    summary = [
        f"# Executive Summary: {findings.get('commodity', 'Market Integration Analysis')}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
        findings.get('summary', ''),
        "",
        "## Key Findings",
        ""
    ]
    
    # Add key points
    key_points = findings.get('key_points', [])
    for point in key_points:
        summary.append(f"- {point}")
    
    # Join summary lines into a single string
    return "\n".join(summary)


def _generate_latex_summary(findings: Dict[str, Any]) -> str:
    """Generate a LaTeX executive summary."""
    summary = [
        "\\documentclass{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{hyperref}",
        "",
        f"\\title{{Executive Summary: {findings.get('commodity', 'Market Integration Analysis')}}}",
        "\\author{Yemen Market Integration Project}",
        f"\\date{{{datetime.now().strftime('%B %d, %Y')}}}",
        "",
        "\\begin{document}",
        "",
        "\\maketitle",
        "",
        findings.get('summary', ''),
        "",
        "\\section{Key Findings}",
        "",
        "\\begin{itemize}"
    ]
    
    # Add key points
    key_points = findings.get('key_points', [])
    for point in key_points:
        summary.append(f"\\item {point}")
    
    summary.append("\\end{itemize}")
    summary.append("\\end{document}")
    
    # Join summary lines into a single string
    return "\n".join(summary)


def _generate_html_summary(findings: Dict[str, Any]) -> str:
    """Generate an HTML executive summary."""
    summary = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>Executive Summary: {findings.get('commodity', 'Market Integration Analysis')}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #3498db; margin-top: 30px; }",
        "ul { margin-top: 20px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Executive Summary: {findings.get('commodity', 'Market Integration Analysis')}</h1>",
        f"<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>",
        f"<p>{findings.get('summary', '')}</p>",
        "<h2>Key Findings</h2>",
        "<ul>"
    ]
    
    # Add key points
    key_points = findings.get('key_points', [])
    for point in key_points:
        summary.append(f"<li>{point}</li>")
    
    summary.append("</ul>")
    summary.append("</body>")
    summary.append("</html>")
    
    # Join summary lines into a single string
    return "\n".join(summary)


def _export_latex_tables(results: Dict[str, Any], commodity: str, output_path: Path) -> None:
    """Export LaTeX tables for publication."""
    tables = []
    
    # Add header
    tables.extend([
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{caption}",
        "\\usepackage{float}",
        "",
        "\\begin{document}",
        ""
    ])
    
    # Add threshold model results table if available
    if 'model_results' in results:
        model_results = results['model_results']
        if isinstance(model_results, dict):
            tables.extend([
                "\\begin{table}[H]",
                "\\centering",
                f"\\caption{{Threshold Model Results for {commodity}}}",
                "\\begin{tabular}{lrr}",
                "\\toprule",
                "Parameter & Below Threshold & Above Threshold \\\\",
                "\\midrule"
            ])
            
            # Add adjustment parameters
            adjustment = model_results.get('adjustment', {})
            if isinstance(adjustment, dict):
                below = adjustment.get('below')
                above = adjustment.get('above')
                
                below_str = f"{below:.4f}" if isinstance(below, (int, float)) else "N/A"
                above_str = f"{above:.4f}" if isinstance(above, (int, float)) else "N/A"
                
                tables.append(f"Adjustment Speed & {below_str} & {above_str} \\\\")
            
            # Add half-lives
            half_lives = model_results.get('half_lives', {})
            if isinstance(half_lives, dict):
                below = half_lives.get('below')
                above = half_lives.get('above')
                
                below_str = f"{below:.2f}" if isinstance(below, (int, float)) and not np.isinf(below) else "$\\infty$"
                above_str = f"{above:.2f}" if isinstance(above, (int, float)) and not np.isinf(above) else "$\\infty$"
                
                tables.append(f"Half-Life (periods) & {below_str} & {above_str} \\\\")
            
            # Add threshold value
            threshold = results.get('threshold')
            if threshold is not None:
                tables.append("\\midrule")
                tables.append(f"Threshold Value & \\multicolumn{{2}}{{c}}{{ {threshold:.4f} }} \\\\")
            
            # Add model fit statistics
            aic = model_results.get('aic')
            bic = model_results.get('bic')
            llf = model_results.get('llf')
            
            if any(x is not None for x in [aic, bic, llf]):
                tables.append("\\midrule")
                
                if aic is not None:
                    tables.append(f"AIC & \\multicolumn{{2}}{{c}}{{ {aic:.4f} }} \\\\")
                
                if bic is not None:
                    tables.append(f"BIC & \\multicolumn{{2}}{{c}}{{ {bic:.4f} }} \\\\")
                
                if llf is not None:
                    tables.append(f"Log-Likelihood & \\multicolumn{{2}}{{c}}{{ {llf:.4f} }} \\\\")
            
            tables.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                ""
            ])
    
    # Add model comparison table if available
    if 'model_comparison' in results:
        model_comparison = results['model_comparison']
        if isinstance(model_comparison, dict) and 'information_criteria' in model_comparison:
            ic_comparison = model_comparison['information_criteria']
            if isinstance(ic_comparison, dict):
                tables.extend([
                    "\\begin{table}[H]",
                    "\\centering",
                    f"\\caption{{Model Comparison for {commodity}}}",
                    "\\begin{tabular}{lrrrr}",
                    "\\toprule",
                    "Model & AIC & BIC & Log-Likelihood & Threshold \\\\",
                    "\\midrule"
                ])
                
                # Add rows for each model
                for model_name, criteria in ic_comparison.items():
                    if isinstance(criteria, dict):
                        aic = criteria.get('aic')
                        bic = criteria.get('bic')
                        llf = criteria.get('llf')
                        threshold = criteria.get('threshold')
                        
                        aic_str = f"{aic:.4f}" if isinstance(aic, (int, float)) else "N/A"
                        bic_str = f"{bic:.4f}" if isinstance(bic, (int, float)) else "N/A"
                        llf_str = f"{llf:.4f}" if isinstance(llf, (int, float)) else "N/A"
                        threshold_str = f"{threshold:.4f}" if isinstance(threshold, (int, float)) else "N/A"
                        
                        tables.append(f"{model_name} & {aic_str} & {bic_str} & {llf_str} & {threshold_str} \\\\")
                
                # Add best model row
                best_model = model_comparison.get('best_model')
                if best_model:
                    tables.append("\\midrule")
                    tables.append(f"Best Model & \\multicolumn{{4}}{{c}}{{ {best_model} }} \\\\")
                
                tables.extend([
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\end{table}",
                    ""
                ])
    
    # End document
    tables.append("\\end{document}")
    
    # Write tables to file
    with open(output_path, 'w') as f:
        f.write("\n".join(tables))
    
    logger.info(f"LaTeX tables exported to {output_path}")


def _export_figures(results: Dict[str, Any], commodity: str, output_dir: Path, formats: List[str], dpi: int = 300) -> Dict[str, Path]:
    """Export figures for publication."""
    figure_paths = {}
    
    # Create figures directory if it doesn't exist
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    # Export adjustment speed plot if data is available
    if 'adjustment_speeds' in results:
        adjustment = results['adjustment_speeds']
        if isinstance(adjustment, dict):
            below = adjustment.get('below')
            above = adjustment.get('above')
            
            if below is not None and above is not None:
                # Create figure
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot adjustment speeds
                regimes = ['Below Threshold', 'Above Threshold']
                speeds = [below, above]
                
                ax.bar(regimes, speeds, color=['#3498db', '#e74c3c'])
                ax.set_ylabel('Adjustment Speed')
                ax.set_title(f'Adjustment Speed by Regime: {commodity}')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add values on top of bars
                for i, speed in enumerate(speeds):
                    ax.text(i, speed + 0.01, f'{speed:.4f}', ha='center')
                
                # Save figure in each requested format
                for fmt in formats:
                    if fmt in ['png', 'pdf', 'svg']:
                        fig_path = figures_dir / f"{commodity.replace(' ', '_')}_adjustment_speeds.{fmt}"
                        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                        figure_paths[f'adjustment_speeds_{fmt}'] = fig_path
                
                plt.close(fig)
    
    # Export half-life plot if data is available
    if 'half_lives' in results:
        half_lives = results['half_lives']
        if isinstance(half_lives, dict):
            below = half_lives.get('below')
            above = half_lives.get('above')
            
            if below is not None and above is not None:
                # Create figure
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot half-lives
                regimes = ['Below Threshold', 'Above Threshold']
                lives = [below if not np.isinf(below) else np.nan, 
                         above if not np.isinf(above) else np.nan]
                
                ax.bar(regimes, lives, color=['#3498db', '#e74c3c'])
                ax.set_ylabel('Half-Life (periods)')
                ax.set_title(f'Price Adjustment Half-Life by Regime: {commodity}')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add values on top of bars
                for i, life in enumerate(lives):
                    if not np.isnan(life):
                        ax.text(i, life + 0.1, f'{life:.2f}', ha='center')
                    else:
                        ax.text(i, 0.5, 'Infinite', ha='center')
                
                # Save figure in each requested format
                for fmt in formats:
                    if fmt in ['png', 'pdf', 'svg']:
                        fig_path = figures_dir / f"{commodity.replace(' ', '_')}_half_lives.{fmt}"
                        fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                        figure_paths[f'half_lives_{fmt}'] = fig_path
                
                plt.close(fig)
    
    # Export model comparison plot if data is available
    if 'model_comparison' in results:
        model_comparison = results['model_comparison']
        if isinstance(model_comparison, dict) and 'information_criteria' in model_comparison:
            ic_comparison = model_comparison['information_criteria']
            if isinstance(ic_comparison, dict):
                # Extract AIC values for each model
                models = []
                aic_values = []
                bic_values = []
                
                for model_name, criteria in ic_comparison.items():
                    if isinstance(criteria, dict):
                        aic = criteria.get('aic')
                        bic = criteria.get('bic')
                        
                        if aic is not None and bic is not None:
                            models.append(model_name)
                            aic_values.append(aic)
                            bic_values.append(bic)
                
                if models and aic_values and bic_values:
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Set width of bars
                    bar_width = 0.35
                    x = np.arange(len(models))
                    
                    # Plot bars
                    ax.bar(x - bar_width/2, aic_values, bar_width, label='AIC', color='#3498db')
                    ax.bar(x + bar_width/2, bic_values, bar_width, label='BIC', color='#e74c3c')
                    
                    # Add labels and title
                    ax.set_xlabel('Model')
                    ax.set_ylabel('Information Criteria')
                    ax.set_title(f'Model Comparison: {commodity}')
                    ax.set_xticks(x)
                    ax.set_xticklabels(models, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Highlight best model
                    best_model = model_comparison.get('best_model')
                    if best_model in models:
                        best_idx = models.index(best_model)
                        ax.get_xticklabels()[best_idx].set_weight('bold')
                        ax.get_xticklabels()[best_idx].set_color('green')
                    
                    plt.tight_layout()
                    
                    # Save figure in each requested format
                    for fmt in formats:
                        if fmt in ['png', 'pdf', 'svg']:
                            fig_path = figures_dir / f"{commodity.replace(' ', '_')}_model_comparison.{fmt}"
                            fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
                            figure_paths[f'model_comparison_{fmt}'] = fig_path
                    
                    plt.close(fig)
    
    logger.info(f"Exported {len(figure_paths)} figures to {figures_dir}")
    
    return figure_paths
