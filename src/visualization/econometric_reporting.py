"""
Econometric reporting module for generating publication-quality tables and visualizations.

This module provides functions for creating standardized econometric tables and 
visualizations following World Bank and academic publication standards.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, List, Union, Optional
import json

# Configure logger
logger = logging.getLogger(__name__)

def generate_econometric_report(
    model,
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = 'markdown',
    publication_quality: bool = True
) -> Path:
    """
    Generate a comprehensive econometric report with publication-quality tables.
    
    Parameters
    ----------
    model : object
        The fitted model object
    results : Dict[str, Any]
        Dictionary of model results
    output_path : Union[str, Path]
        Path to save the report
    format : str
        Output format ('markdown', 'latex', 'html', 'json')
    publication_quality : bool
        Whether to generate publication-quality output with enhanced formatting
        
    Returns
    -------
    Path
        Path to the generated report
    """
    # Ensure output_path is a Path object
    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Determine the appropriate format function
    if format == 'markdown':
        report_text = _generate_markdown_report(model, results, publication_quality)
    elif format == 'latex':
        report_text = _generate_latex_report(model, results, publication_quality)
    elif format == 'html':
        report_text = _generate_html_report(model, results, publication_quality)
    elif format == 'json':
        report_dict = _generate_json_report(model, results)
        # Write JSON directly and return
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        return output_path
    else:
        raise ValueError(f"Unsupported report format: {format}")
    
    # Write the report to the output path
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    # Generate supplementary visualizations if requested
    if publication_quality:
        _generate_supplementary_visualizations(model, results, output_path.parent)
    
    logger.info(f"Econometric report generated at {output_path}")
    return output_path

def _generate_markdown_report(model, results: Dict[str, Any], publication_quality: bool) -> str:
    """
    Generate a markdown-formatted econometric report.
    
    Parameters
    ----------
    model : object
        The fitted model object
    results : Dict[str, Any]
        Dictionary of model results
    publication_quality : bool
        Whether to generate publication-quality output
        
    Returns
    -------
    str
        Markdown-formatted report text
    """
    # Initialize the report
    report = []
    
    # Add title and introduction
    report.append("# Econometric Analysis Report\n")
    report.append(f"## Model: {type(model).__name__}\n")
    
    # Add executive summary
    report.append("## Executive Summary\n")
    if 'cointegration' in results and 'cointegrated' in results['cointegration']:
        cointegrated = results['cointegration']['cointegrated']
        report.append(f"* **Cointegration**: {'Evidence of cointegration found' if cointegrated else 'No evidence of cointegration'}\n")
    
    if hasattr(model, 'threshold'):
        report.append(f"* **Threshold**: {model.threshold:.4f}\n")
    
    # Model specification section
    report.append("## Model Specification\n")
    report.append("```\n")
    if hasattr(model, 'summary') and callable(model.summary):
        try:
            report.append(str(model.summary()))
        except:
            report.append(f"Model type: {type(model).__name__}")
    else:
        report.append(f"Model type: {type(model).__name__}")
    report.append("```\n")
    
    # Add parameter estimates table
    report.append("## Parameter Estimates\n")
    if 'parameters' in results:
        params = results['parameters']
        
        # Create a markdown table
        report.append("| Parameter | Estimate | Std. Error | t-value | p-value |\n")
        report.append("|-----------|----------|------------|---------|--------|\n")
        
        for param_name, param_value in params.items():
            if isinstance(param_value, dict) and 'estimate' in param_value:
                estimate = param_value.get('estimate', 'N/A')
                std_err = param_value.get('std_error', 'N/A')
                t_value = param_value.get('t_value', 'N/A')
                p_value = param_value.get('p_value', 'N/A')
                
                report.append(f"| {param_name} | {estimate:.4f} | {std_err:.4f} | {t_value:.4f} | {p_value:.4f} |\n")
            elif isinstance(param_value, (int, float)):
                report.append(f"| {param_name} | {param_value:.4f} | N/A | N/A | N/A |\n")
    
    # Add model diagnostics
    report.append("## Model Diagnostics\n")
    
    if 'diagnostics' in results:
        diag = results['diagnostics']
        
        # Create a markdown table for diagnostics
        report.append("| Statistic | Value | p-value | Interpretation |\n")
        report.append("|-----------|-------|---------|----------------|\n")
        
        for diag_name, diag_value in diag.items():
            if isinstance(diag_value, dict):
                value = diag_value.get('value', 'N/A')
                p_value = diag_value.get('p_value', 'N/A')
                interp = diag_value.get('interpretation', 'N/A')
                
                report.append(f"| {diag_name} | {value:.4f} | {p_value:.4f} | {interp} |\n")
            elif isinstance(diag_value, (int, float)):
                report.append(f"| {diag_name} | {diag_value:.4f} | N/A | N/A |\n")
    
    # Add information criteria
    report.append("## Information Criteria\n")
    
    criteria = {}
    
    # Extract criteria from results or model
    if 'information_criteria' in results:
        criteria = results['information_criteria']
    else:
        # Try to extract from model attributes
        for criterion in ['aic', 'bic', 'hqic']:
            if hasattr(model, criterion):
                criteria[criterion] = getattr(model, criterion)
    
    if criteria:
        # Create a markdown table for information criteria
        report.append("| Criterion | Value |\n")
        report.append("|-----------|-------|\n")
        
        for criterion_name, criterion_value in criteria.items():
            if isinstance(criterion_value, (int, float)):
                report.append(f"| {criterion_name.upper()} | {criterion_value:.4f} |\n")
    
    # Add interpretation section
    report.append("## Interpretation and Implications\n")
    
    if 'interpretation' in results:
        interp = results['interpretation']
        
        for key, value in interp.items():
            report.append(f"### {key.capitalize()}\n")
            
            if isinstance(value, str):
                report.append(f"{value}\n")
            elif isinstance(value, list):
                for item in value:
                    report.append(f"* {item}\n")
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    report.append(f"#### {subkey.capitalize()}\n")
                    
                    if isinstance(subvalue, str):
                        report.append(f"{subvalue}\n")
                    elif isinstance(subvalue, list):
                        for item in subvalue:
                            report.append(f"* {item}\n")
    
    # Additional notes on methodology
    report.append("## Methodology Notes\n")
    report.append("This analysis employed standard econometric techniques for time series analysis, including:\n")
    report.append("* Unit root testing to assess stationarity\n")
    report.append("* Cointegration analysis to detect long-run equilibrium relationships\n")
    report.append("* Threshold modeling to account for nonlinearities and transaction costs\n")
    report.append("* Robust standard errors to address heteroskedasticity\n")
    
    return "\n".join(report)

def _generate_latex_report(model, results: Dict[str, Any], publication_quality: bool) -> str:
    """
    Generate a LaTeX-formatted econometric report.
    
    Parameters
    ----------
    model : object
        The fitted model object
    results : Dict[str, Any]
        Dictionary of model results
    publication_quality : bool
        Whether to generate publication-quality output
        
    Returns
    -------
    str
        LaTeX-formatted report text
    """
    # Initialize the report
    report = []
    
    # Add preamble for standalone LaTeX document
    if publication_quality:
        report.append("\\documentclass{article}")
        report.append("\\usepackage{booktabs}")
        report.append("\\usepackage{amsmath}")
        report.append("\\usepackage{graphicx}")
        report.append("\\usepackage{float}")
        report.append("\\usepackage{caption}")
        report.append("\\usepackage{subcaption}")
        report.append("\\usepackage{threeparttable}")
        report.append("\\usepackage{longtable}")
        report.append("\\usepackage{hyperref}")
        report.append("\\usepackage{fancyhdr}")
        report.append("\\usepackage{microtype}")
        report.append("")
        report.append("\\begin{document}")
    
    # Add title and introduction
    report.append("\\section{Econometric Analysis Report}")
    report.append(f"\\subsection{{{type(model).__name__} Model}}")
    
    # Add executive summary
    report.append("\\subsection{Executive Summary}")
    report.append("\\begin{itemize}")
    if 'cointegration' in results and 'cointegrated' in results['cointegration']:
        cointegrated = results['cointegration']['cointegrated']
        report.append(f"\\item \\textbf{{Cointegration}}: {'Evidence of cointegration found' if cointegrated else 'No evidence of cointegration'}")
    
    if hasattr(model, 'threshold'):
        report.append(f"\\item \\textbf{{Threshold}}: {model.threshold:.4f}")
    report.append("\\end{itemize}")
    
    # Model specification section
    report.append("\\subsection{Model Specification}")
    report.append("\\begin{verbatim}")
    if hasattr(model, 'summary') and callable(model.summary):
        try:
            report.append(str(model.summary()))
        except:
            report.append(f"Model type: {type(model).__name__}")
    else:
        report.append(f"Model type: {type(model).__name__}")
    report.append("\\end{verbatim}")
    
    # Add parameter estimates table
    report.append("\\subsection{Parameter Estimates}")
    if 'parameters' in results:
        params = results['parameters']
        
        # Create a LaTeX table
        report.append("\\begin{table}[htbp]")
        report.append("\\centering")
        report.append("\\caption{Parameter Estimates}")
        report.append("\\begin{tabular}{lrrrr}")
        report.append("\\toprule")
        report.append("Parameter & Estimate & Std. Error & t-value & p-value \\\\")
        report.append("\\midrule")
        
        for param_name, param_value in params.items():
            if isinstance(param_value, dict) and 'estimate' in param_value:
                estimate = param_value.get('estimate', 'N/A')
                std_err = param_value.get('std_error', 'N/A')
                t_value = param_value.get('t_value', 'N/A')
                p_value = param_value.get('p_value', 'N/A')
                
                report.append(f"{param_name} & {estimate:.4f} & {std_err:.4f} & {t_value:.4f} & {p_value:.4f} \\\\")
            elif isinstance(param_value, (int, float)):
                report.append(f"{param_name} & {param_value:.4f} & N/A & N/A & N/A \\\\")
        
        report.append("\\bottomrule")
        report.append("\\end{tabular}")
        report.append("\\end{table}")
    
    # Add model diagnostics
    report.append("\\subsection{Model Diagnostics}")
    
    if 'diagnostics' in results:
        diag = results['diagnostics']
        
        # Create a LaTeX table for diagnostics
        report.append("\\begin{table}[htbp]")
        report.append("\\centering")
        report.append("\\caption{Diagnostic Tests}")
        report.append("\\begin{tabular}{lrrp{5cm}}")
        report.append("\\toprule")
        report.append("Statistic & Value & p-value & Interpretation \\\\")
        report.append("\\midrule")
        
        for diag_name, diag_value in diag.items():
            if isinstance(diag_value, dict):
                value = diag_value.get('value', 'N/A')
                p_value = diag_value.get('p_value', 'N/A')
                interp = diag_value.get('interpretation', 'N/A')
                
                report.append(f"{diag_name} & {value:.4f} & {p_value:.4f} & {interp} \\\\")
            elif isinstance(diag_value, (int, float)):
                report.append(f"{diag_name} & {diag_value:.4f} & N/A & N/A \\\\")
        
        report.append("\\bottomrule")
        report.append("\\end{tabular}")
        report.append("\\end{table}")
    
    # Add information criteria
    report.append("\\subsection{Information Criteria}")
    
    criteria = {}
    
    # Extract criteria from results or model
    if 'information_criteria' in results:
        criteria = results['information_criteria']
    else:
        # Try to extract from model attributes
        for criterion in ['aic', 'bic', 'hqic']:
            if hasattr(model, criterion):
                criteria[criterion] = getattr(model, criterion)
    
    if criteria:
        # Create a LaTeX table for information criteria
        report.append("\\begin{table}[htbp]")
        report.append("\\centering")
        report.append("\\caption{Information Criteria}")
        report.append("\\begin{tabular}{lr}")
        report.append("\\toprule")
        report.append("Criterion & Value \\\\")
        report.append("\\midrule")
        
        for criterion_name, criterion_value in criteria.items():
            if isinstance(criterion_value, (int, float)):
                report.append(f"{criterion_name.upper()} & {criterion_value:.4f} \\\\")
        
        report.append("\\bottomrule")
        report.append("\\end{tabular}")
        report.append("\\end{table}")
    
    # Add interpretation section
    report.append("\\subsection{Interpretation and Implications}")
    
    if 'interpretation' in results:
        interp = results['interpretation']
        
        for key, value in interp.items():
            report.append(f"\\subsubsection{{{key.capitalize()}}}")
            
            if isinstance(value, str):
                report.append(f"{value}")
            elif isinstance(value, list):
                report.append("\\begin{itemize}")
                for item in value:
                    report.append(f"\\item {item}")
                report.append("\\end{itemize}")
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    report.append(f"\\paragraph{{{subkey.capitalize()}}}")
                    
                    if isinstance(subvalue, str):
                        report.append(f"{subvalue}")
                    elif isinstance(subvalue, list):
                        report.append("\\begin{itemize}")
                        for item in subvalue:
                            report.append(f"\\item {item}")
                        report.append("\\end{itemize}")
    
    # Additional notes on methodology
    report.append("\\subsection{Methodology Notes}")
    report.append("This analysis employed standard econometric techniques for time series analysis, including:")
    report.append("\\begin{itemize}")
    report.append("\\item Unit root testing to assess stationarity")
    report.append("\\item Cointegration analysis to detect long-run equilibrium relationships")
    report.append("\\item Threshold modeling to account for nonlinearities and transaction costs")
    report.append("\\item Robust standard errors to address heteroskedasticity")
    report.append("\\end{itemize}")
    
    # Close the document if it's a standalone LaTeX file
    if publication_quality:
        report.append("\\end{document}")
    
    return "\n".join(report)

def _generate_html_report(model, results: Dict[str, Any], publication_quality: bool) -> str:
    """
    Generate an HTML-formatted econometric report.
    
    Parameters
    ----------
    model : object
        The fitted model object
    results : Dict[str, Any]
        Dictionary of model results
    publication_quality : bool
        Whether to generate publication-quality output
        
    Returns
    -------
    str
        HTML-formatted report text
    """
    # Initialize the report
    report = []
    
    # Add HTML preamble
    report.append("<!DOCTYPE html>")
    report.append("<html>")
    report.append("<head>")
    report.append("    <title>Econometric Analysis Report</title>")
    if publication_quality:
        # Add CSS for publication-quality styling
        report.append("    <style>")
        report.append("        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }")
        report.append("        h1, h2, h3, h4 { color: #333366; }")
        report.append("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        report.append("        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }")
        report.append("        th { background-color: #f2f2f2; color: #333366; }")
        report.append("        tr:hover { background-color: #f5f5f5; }")
        report.append("        .summary { background-color: #f9f9f9; padding: 15px; border-left: 5px solid #333366; }")
        report.append("        .code { font-family: monospace; background-color: #f8f8f8; padding: 10px; overflow: auto; }")
        report.append("        .footnote { font-size: 0.8em; color: #666; }")
        report.append("    </style>")
    report.append("</head>")
    report.append("<body>")
    
    # Add title and introduction
    report.append("<h1>Econometric Analysis Report</h1>")
    report.append(f"<h2>{type(model).__name__} Model</h2>")
    
    # Add executive summary
    report.append("<h2>Executive Summary</h2>")
    report.append("<div class='summary'>")
    report.append("<ul>")
    if 'cointegration' in results and 'cointegrated' in results['cointegration']:
        cointegrated = results['cointegration']['cointegrated']
        report.append(f"<li><strong>Cointegration</strong>: {'Evidence of cointegration found' if cointegrated else 'No evidence of cointegration'}</li>")
    
    if hasattr(model, 'threshold'):
        report.append(f"<li><strong>Threshold</strong>: {model.threshold:.4f}</li>")
    report.append("</ul>")
    report.append("</div>")
    
    # Model specification section
    report.append("<h2>Model Specification</h2>")
    report.append("<pre class='code'>")
    if hasattr(model, 'summary') and callable(model.summary):
        try:
            report.append(str(model.summary()))
        except:
            report.append(f"Model type: {type(model).__name__}")
    else:
        report.append(f"Model type: {type(model).__name__}")
    report.append("</pre>")
    
    # Add parameter estimates table
    report.append("<h2>Parameter Estimates</h2>")
    if 'parameters' in results:
        params = results['parameters']
        
        # Create an HTML table
        report.append("<table>")
        report.append("<tr><th>Parameter</th><th>Estimate</th><th>Std. Error</th><th>t-value</th><th>p-value</th></tr>")
        
        for param_name, param_value in params.items():
            if isinstance(param_value, dict) and 'estimate' in param_value:
                estimate = param_value.get('estimate', 'N/A')
                std_err = param_value.get('std_error', 'N/A')
                t_value = param_value.get('t_value', 'N/A')
                p_value = param_value.get('p_value', 'N/A')
                
                report.append(f"<tr><td>{param_name}</td><td>{estimate:.4f}</td><td>{std_err:.4f}</td><td>{t_value:.4f}</td><td>{p_value:.4f}</td></tr>")
            elif isinstance(param_value, (int, float)):
                report.append(f"<tr><td>{param_name}</td><td>{param_value:.4f}</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>")
        
        report.append("</table>")
    
    # Add model diagnostics
    report.append("<h2>Model Diagnostics</h2>")
    
    if 'diagnostics' in results:
        diag = results['diagnostics']
        
        # Create an HTML table for diagnostics
        report.append("<table>")
        report.append("<tr><th>Statistic</th><th>Value</th><th>p-value</th><th>Interpretation</th></tr>")
        
        for diag_name, diag_value in diag.items():
            if isinstance(diag_value, dict):
                value = diag_value.get('value', 'N/A')
                p_value = diag_value.get('p_value', 'N/A')
                interp = diag_value.get('interpretation', 'N/A')
                
                report.append(f"<tr><td>{diag_name}</td><td>{value:.4f}</td><td>{p_value:.4f}</td><td>{interp}</td></tr>")
            elif isinstance(diag_value, (int, float)):
                report.append(f"<tr><td>{diag_name}</td><td>{diag_value:.4f}</td><td>N/A</td><td>N/A</td></tr>")
        
        report.append("</table>")
    
    # Add information criteria
    report.append("<h2>Information Criteria</h2>")
    
    criteria = {}
    
    # Extract criteria from results or model
    if 'information_criteria' in results:
        criteria = results['information_criteria']
    else:
        # Try to extract from model attributes
        for criterion in ['aic', 'bic', 'hqic']:
            if hasattr(model, criterion):
                criteria[criterion] = getattr(model, criterion)
    
    if criteria:
        # Create an HTML table for information criteria
        report.append("<table>")
        report.append("<tr><th>Criterion</th><th>Value</th></tr>")
        
        for criterion_name, criterion_value in criteria.items():
            if isinstance(criterion_value, (int, float)):
                report.append(f"<tr><td>{criterion_name.upper()}</td><td>{criterion_value:.4f}</td></tr>")
        
        report.append("</table>")
    
    # Add interpretation section
    report.append("<h2>Interpretation and Implications</h2>")
    
    if 'interpretation' in results:
        interp = results['interpretation']
        
        for key, value in interp.items():
            report.append(f"<h3>{key.capitalize()}</h3>")
            
            if isinstance(value, str):
                report.append(f"<p>{value}</p>")
            elif isinstance(value, list):
                report.append("<ul>")
                for item in value:
                    report.append(f"<li>{item}</li>")
                report.append("</ul>")
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    report.append(f"<h4>{subkey.capitalize()}</h4>")
                    
                    if isinstance(subvalue, str):
                        report.append(f"<p>{subvalue}</p>")
                    elif isinstance(subvalue, list):
                        report.append("<ul>")
                        for item in subvalue:
                            report.append(f"<li>{item}</li>")
                        report.append("</ul>")
    
    # Additional notes on methodology
    report.append("<h2>Methodology Notes</h2>")
    report.append("<p>This analysis employed standard econometric techniques for time series analysis, including:</p>")
    report.append("<ul>")
    report.append("<li>Unit root testing to assess stationarity</li>")
    report.append("<li>Cointegration analysis to detect long-run equilibrium relationships</li>")
    report.append("<li>Threshold modeling to account for nonlinearities and transaction costs</li>")
    report.append("<li>Robust standard errors to address heteroskedasticity</li>")
    report.append("</ul>")
    
    # Add footer
    if publication_quality:
        report.append("<hr>")
        report.append("<p class='footnote'>Generated using econometric_reporting.py</p>")
    
    # Close HTML tags
    report.append("</body>")
    report.append("</html>")
    
    return "\n".join(report)

def _generate_json_report(model, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a JSON-formatted econometric report.
    
    Parameters
    ----------
    model : object
        The fitted model object
    results : Dict[str, Any]
        Dictionary of model results
        
    Returns
    -------
    Dict[str, Any]
        JSON-serializable report dictionary
    """
    # Initialize the report dictionary
    report = {
        "model_type": type(model).__name__,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Add executive summary
    summary = {}
    if 'cointegration' in results and 'cointegrated' in results['cointegration']:
        summary["cointegration"] = results['cointegration']['cointegrated']
    
    if hasattr(model, 'threshold'):
        summary["threshold"] = float(model.threshold)
    
    report["executive_summary"] = summary
    
    # Add model specification
    if hasattr(model, 'summary') and callable(model.summary):
        try:
            # Try to convert summary to string
            report["model_specification"] = str(model.summary())
        except:
            report["model_specification"] = f"Model type: {type(model).__name__}"
    else:
        report["model_specification"] = f"Model type: {type(model).__name__}"
    
    # Add parameter estimates
    if 'parameters' in results:
        # Convert to serializable format
        serializable_params = {}
        for param_name, param_value in results['parameters'].items():
            if isinstance(param_value, dict):
                serializable_params[param_name] = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in param_value.items()
                }
            elif isinstance(param_value, (int, float, np.number)):
                serializable_params[param_name] = float(param_value)
            else:
                serializable_params[param_name] = str(param_value)
        
        report["parameter_estimates"] = serializable_params
    
    # Add model diagnostics
    if 'diagnostics' in results:
        # Convert to serializable format
        serializable_diag = {}
        for diag_name, diag_value in results['diagnostics'].items():
            if isinstance(diag_value, dict):
                serializable_diag[diag_name] = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in diag_value.items()
                }
            elif isinstance(diag_value, (int, float, np.number)):
                serializable_diag[diag_name] = float(diag_value)
            else:
                serializable_diag[diag_name] = str(diag_value)
        
        report["model_diagnostics"] = serializable_diag
    
    # Add information criteria
    criteria = {}
    
    # Extract criteria from results or model
    if 'information_criteria' in results:
        criteria = results['information_criteria']
    else:
        # Try to extract from model attributes
        for criterion in ['aic', 'bic', 'hqic']:
            if hasattr(model, criterion):
                criteria[criterion] = getattr(model, criterion)
    
    if criteria:
        # Convert to serializable format
        report["information_criteria"] = {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in criteria.items()
        }
    
    # Add interpretation
    if 'interpretation' in results:
        report["interpretation"] = results['interpretation']
    
    return report

def _generate_supplementary_visualizations(model, results: Dict[str, Any], output_dir: Path) -> None:
    """
    Generate supplementary visualizations for the econometric report.
    
    Parameters
    ----------
    model : object
        The fitted model object
    results : Dict[str, Any]
        Dictionary of model results
    output_dir : Path
        Directory to save the visualizations
    """
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set matplotlib style for publication quality
    plt.style.use('seaborn-whitegrid')
    
    # Generate visualizations based on model type and available data
    
    # 1. Residual diagnostic plots
    if hasattr(model, 'resid') or 'residuals' in results:
        residuals = getattr(model, 'resid', results.get('residuals', None))
        if residuals is not None:
            # Convert to numpy array if needed
            if not isinstance(residuals, np.ndarray):
                residuals = np.array(residuals)
            
            # Create residual diagnostics plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Residual time series
            axes[0, 0].plot(residuals)
            axes[0, 0].set_title('Residuals Over Time')
            axes[0, 0].set_xlabel('Observation')
            axes[0, 0].set_ylabel('Residual')
            
            # Residual histogram
            axes[0, 1].hist(residuals, bins=20, edgecolor='black')
            axes[0, 1].set_title('Residual Distribution')
            axes[0, 1].set_xlabel('Residual')
            axes[0, 1].set_ylabel('Frequency')
            
            # Residual QQ plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            
            # Residual autocorrelation
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals, lags=20, ax=axes[1, 1])
            axes[1, 1].set_title('Autocorrelation Function')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'residual_diagnostics.png', dpi=300)
            plt.close()
    
    # 2. If it's a threshold model, plot regime classification
    if hasattr(model, 'threshold') and hasattr(model, 'regime_data'):
        regime_data = model.regime_data
        
        if isinstance(regime_data, dict) and 'dates' in regime_data and 'regimes' in regime_data:
            dates = regime_data['dates']
            regimes = regime_data['regimes']
            
            # Create regime plot
            plt.figure(figsize=(12, 6))
            plt.plot(dates, regimes, linestyle='-', marker='o', markersize=4)
            plt.axhline(y=model.threshold, color='r', linestyle='--', label=f'Threshold: {model.threshold:.4f}')
            plt.title('Regime Classification')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'regime_classification.png', dpi=300)
            plt.close()
    
    # 3. Model comparison plot if available
    if 'model_comparison' in results:
        comparison = results['model_comparison']
        
        if isinstance(comparison, dict) and 'models' in comparison and 'criteria' in comparison:
            models = comparison['models']
            criteria = comparison['criteria']
            
            # Create model comparison plot
            plt.figure(figsize=(10, 6))
            
            x = np.arange(len(models))
            width = 0.8 / len(criteria)
            
            for i, (criterion, values) in enumerate(criteria.items()):
                plt.bar(x + i * width - 0.4 + width/2, values, width, label=criterion.upper())
            
            plt.xlabel('Model')
            plt.ylabel('Value')
            plt.title('Model Comparison')
            plt.xticks(x, models)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'model_comparison.png', dpi=300)
            plt.close()
    
    # 4. Parameter stability plot if available
    if 'parameter_stability' in results:
        stability = results['parameter_stability']
        
        if isinstance(stability, dict) and len(stability) > 0:
            # Create parameter stability plot
            plt.figure(figsize=(12, 6))
            
            for param, values in stability.items():
                if isinstance(values, dict) and 'dates' in values and 'estimates' in values:
                    plt.plot(values['dates'], values['estimates'], label=param)
            
            plt.title('Parameter Stability Over Time')
            plt.xlabel('Date')
            plt.ylabel('Parameter Estimate')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'parameter_stability.png', dpi=300)
            plt.close()