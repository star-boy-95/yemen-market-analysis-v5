"""
Enhanced Econometric Reporting Module for Yemen Market Analysis.

This module provides advanced reporting capabilities for econometric analysis results,
following World Bank standards and best practices in econometrics. It generates
publication-quality tables, visualizations, and comparative model analysis reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, List, Union, Optional, Tuple
import json
import os
from datetime import datetime
import statsmodels.api as sm
from scipy import stats
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Configure logger
logger = logging.getLogger(__name__)

class EconometricReporter:
    """
    Comprehensive econometric reporter for market integration analysis.
    
    This class provides methods to generate standardized econometric reports,
    tables, and visualizations following World Bank standards. It supports
    cross-commodity comparisons, model selection reporting, and publication-quality
    outputs in multiple formats.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        publication_quality: bool = True,
        style: str = 'world_bank',
        format: str = 'markdown'
    ):
        """
        Initialize the econometric reporter.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory to save report outputs
        publication_quality : bool, optional
            Whether to generate publication-quality outputs
        style : str, optional
            Visual style for reports ('world_bank', 'academic', 'policy')
        format : str, optional
            Default output format ('markdown', 'latex', 'html', 'json')
        """
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.publication_quality = publication_quality
        self.style = style
        self.format = format
        # Set matplotlib style based on selected style
        if style == 'world_bank':
            plt.style.use('seaborn-v0_8-whitegrid')  # Updated style name for newer matplotlib
            self.colors = ['#0071bc', '#d55e00', '#009e73', '#cc79a7', '#f0e442']
            self.fig_size = (10, 6)
            self.dpi = 300
        elif style == 'academic':
            plt.style.use('seaborn-v0_8')  # Updated style name for newer matplotlib
            self.colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00']
            self.fig_size = (8, 5)
            self.dpi = 300
        elif style == 'policy':
            plt.style.use('seaborn-v0_8-talk')  # Updated style name for newer matplotlib
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            self.fig_size = (12, 8)
            self.dpi = 200
        else:
            # Default style
            plt.style.use('seaborn-v0_8-whitegrid')  # Updated style name for newer matplotlib
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            self.fig_size = (10, 6)
            self.dpi = 300
        
        logger.info(f"Initialized EconometricReporter with style='{style}', format='{format}'")
        
    def generate_model_comparison_report(
        self,
        model_results: Dict[str, Any],
        commodity: str,
        output_file: Optional[str] = None,
        format: Optional[str] = None
    ) -> Path:
        """
        Generate a comprehensive model comparison report.
        
        Parameters
        ----------
        model_results : dict
            Dictionary containing model comparison results
        commodity : str
            Commodity name
        output_file : str, optional
            Output file name (without extension)
        format : str, optional
            Output format (overrides default format)
            
        Returns
        -------
        Path
            Path to the generated report
        """
        format = format or self.format
        output_file = output_file or f"{commodity.replace(' ', '_')}_model_comparison"
        
        # Ensure output file has the correct extension
        if not output_file.endswith(f".{format}"):
            output_file = f"{output_file}.{format}"
        
        output_path = self.output_dir / output_file
        
        # Extract information criteria comparison
        ic_comparison = model_results.get('information_criteria', {})
        best_model = model_results.get('best_model', 'Unknown')
        
        # Create report content based on format
        if format == 'markdown':
            content = self._generate_markdown_comparison_report(model_results, commodity, best_model)
        elif format == 'latex':
            content = self._generate_latex_comparison_report(model_results, commodity, best_model)
        elif format == 'html':
            content = self._generate_html_comparison_report(model_results, commodity, best_model)
        elif format == 'json':
            # For JSON, we just save the model results directly
            with open(output_path, 'w') as f:
                json.dump(model_results, f, indent=2, default=str)
            return output_path
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write(content)
        
        # Generate visualizations
        self._generate_model_comparison_visualizations(model_results, commodity)
        
        return output_path
        
    def _generate_markdown_comparison_report(
        self,
        model_results: Dict[str, Any],
        commodity: str,
        best_model: str
    ) -> str:
        """Generate a markdown report for model comparison."""
        ic_comparison = model_results.get('information_criteria', {})
        performance_metrics = model_results.get('performance_metrics', {})
        
        # Start building the report
        report = [
            f"# Model Comparison Report for {commodity}",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## Executive Summary",
            "",
            f"This report compares different threshold model specifications for {commodity} price integration analysis.",
            f"The best model based on AIC is **{best_model}**.",
            "",
            "## Information Criteria Comparison",
            "",
            "| Model | AIC | BIC | HQIC | Log-Likelihood | Parameters |",
            "|-------|-----|-----|------|----------------|------------|"
        ]
        
        # Add rows for each model
        for model_name, criteria in ic_comparison.items():
            if isinstance(criteria, dict):
                aic = criteria.get('aic', 'N/A')
                bic = criteria.get('bic', 'N/A')
                hqic = criteria.get('hqic', 'N/A')
                llf = criteria.get('llf', 'N/A')
                n_params = criteria.get('n_params', 'N/A')
                
                # Format numbers
                aic = f"{aic:.4f}" if isinstance(aic, (int, float)) else aic
                bic = f"{bic:.4f}" if isinstance(bic, (int, float)) else bic
                hqic = f"{hqic:.4f}" if isinstance(hqic, (int, float)) else hqic
                llf = f"{llf:.4f}" if isinstance(llf, (int, float)) else llf
                
                # Add row
                report.append(f"| {model_name} | {aic} | {bic} | {hqic} | {llf} | {n_params} |")
        
        # Add performance metrics section if available
        if performance_metrics:
            report.extend([
                "",
                "## Performance Metrics",
                "",
                "| Model | RMSE | MAE | R² | Adjusted R² |",
                "|-------|------|-----|----|-----------:|"
            ])
            
            # Add rows for each model
            for model_name, metrics in performance_metrics.items():
                if isinstance(metrics, dict):
                    rmse = metrics.get('rmse', 'N/A')
                    mae = metrics.get('mae', 'N/A')
                    r2 = metrics.get('r2', 'N/A')
                    adj_r2 = metrics.get('adj_r2', 'N/A')
                    
                    # Format numbers
                    rmse = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse
                    mae = f"{mae:.4f}" if isinstance(mae, (int, float)) else mae
                    r2 = f"{r2:.4f}" if isinstance(r2, (int, float)) else r2
                    adj_r2 = f"{adj_r2:.4f}" if isinstance(adj_r2, (int, float)) else adj_r2
                    
                    # Add row
                    report.append(f"| {model_name} | {rmse} | {mae} | {r2} | {adj_r2} |")
        
        # Add model parameters section
        report.extend([
            "",
            "## Model Parameters",
            "",
            "### Threshold Values",
            "",
            "| Model | Threshold | Lower CI | Upper CI | p-value |",
            "|-------|-----------|----------|----------|---------|"
        ])
        
        # Add conclusion
        report.extend([
            "",
            "## Conclusion",
            "",
            f"Based on the information criteria and performance metrics, the **{best_model}** "
            f"provides the best fit for the {commodity} price data. This suggests that "
            f"this specification most accurately captures the market integration dynamics for {commodity}.",
            "",
            "## Appendix: Diagnostic Tests",
            "",
            "### Residual Diagnostics",
            "",
            "| Model | Normality (JB) | Serial Correlation (LM) | Heteroskedasticity (White) |",
            "|-------|----------------|-------------------------|----------------------------|"
        ])
        
        return "\n".join(report)
        
    def _generate_latex_comparison_report(
        self,
        model_results: Dict[str, Any],
        commodity: str,
        best_model: str
    ) -> str:
        """Generate a LaTeX report for model comparison."""
        ic_comparison = model_results.get('information_criteria', {})
        performance_metrics = model_results.get('performance_metrics', {})
        
        # Start building the report
        report = [
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage{caption}",
            "\\usepackage{graphicx}",
            "\\usepackage{amsmath}",
            "\\usepackage{geometry}",
            "\\usepackage{siunitx}",
            "\\usepackage{xcolor}",
            "\\usepackage{hyperref}",
            "",
            "\\geometry{margin=1in}",
            "\\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}",
            "",
            "\\begin{document}",
            "",
            f"\\title{{Model Comparison Report for {commodity}}}",
            f"\\date{{{datetime.now().strftime('%Y-%m-%d')}}}",
            "\\maketitle",
            "",
            "\\section{Executive Summary}",
            "",
            f"This report compares different threshold model specifications for {commodity} price integration analysis. ",
            f"The best model based on AIC is \\textbf{{{best_model}}}.",
            "",
            "\\section{Information Criteria Comparison}",
            "",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Information Criteria Comparison}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Model & AIC & BIC & HQIC & Log-Likelihood & Parameters \\\\",
            "\\midrule"
        ]
        
        # Add rows for each model
        for model_name, criteria in ic_comparison.items():
            if isinstance(criteria, dict):
                aic = criteria.get('aic', 'N/A')
                bic = criteria.get('bic', 'N/A')
                hqic = criteria.get('hqic', 'N/A')
                llf = criteria.get('llf', 'N/A')
                n_params = criteria.get('n_params', 'N/A')
                
                # Format numbers
                aic = f"{aic:.4f}" if isinstance(aic, (int, float)) else aic
                bic = f"{bic:.4f}" if isinstance(bic, (int, float)) else bic
                hqic = f"{hqic:.4f}" if isinstance(hqic, (int, float)) else hqic
                llf = f"{llf:.4f}" if isinstance(llf, (int, float)) else llf
                
                # Add row
                report.append(f"{model_name} & {aic} & {bic} & {hqic} & {llf} & {n_params} \\\\")
        
        # Close the table
        report.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            ""
        ])
        
        # Add performance metrics section if available
        if performance_metrics:
            report.extend([
                "\\section{Performance Metrics}",
                "",
                "\\begin{table}[h]",
                "\\centering",
                "\\caption{Performance Metrics Comparison}",
                "\\begin{tabular}{lrrrr}",
                "\\toprule",
                "Model & RMSE & MAE & $R^2$ & Adjusted $R^2$ \\\\",
                "\\midrule"
            ])
            
            # Add rows for each model
            for model_name, metrics in performance_metrics.items():
                if isinstance(metrics, dict):
                    rmse = metrics.get('rmse', 'N/A')
                    mae = metrics.get('mae', 'N/A')
                    r2 = metrics.get('r2', 'N/A')
                    adj_r2 = metrics.get('adj_r2', 'N/A')
                    
                    # Format numbers
                    rmse = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse
                    mae = f"{mae:.4f}" if isinstance(mae, (int, float)) else mae
                    r2 = f"{r2:.4f}" if isinstance(r2, (int, float)) else r2
                    adj_r2 = f"{adj_r2:.4f}" if isinstance(adj_r2, (int, float)) else adj_r2
                    
                    # Add row
                    report.append(f"{model_name} & {rmse} & {mae} & {r2} & {adj_r2} \\\\")
            
            # Close the table
            report.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                ""
            ])
        
        # Add model parameters section
        report.extend([
            "\\section{Model Parameters}",
            "",
            "\\subsection{Threshold Values}",
            "",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Threshold Values and Confidence Intervals}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Model & Threshold & Lower CI & Upper CI & p-value \\\\",
            "\\midrule",
            "% Placeholder for threshold values",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\section{Conclusion}",
            "",
            f"Based on the information criteria and performance metrics, the \\textbf{{{best_model}}} ",
            f"provides the best fit for the {commodity} price data. This suggests that ",
            f"this specification most accurately captures the market integration dynamics for {commodity}.",
            "",
            "\\section{Appendix: Diagnostic Tests}",
            "",
            "\\subsection{Residual Diagnostics}",
            "",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Diagnostic Test Results}",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Model & Normality (JB) & Serial Correlation (LM) & Heteroskedasticity (White) \\\\",
            "\\midrule",
            "% Placeholder for diagnostic test results",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\end{document}"
        ])
        
        return "\n".join(report)
        
    def _generate_html_comparison_report(
        self,
        model_results: Dict[str, Any],
        commodity: str,
        best_model: str
    ) -> str:
        """Generate an HTML report for model comparison."""
        ic_comparison = model_results.get('information_criteria', {})
        performance_metrics = model_results.get('performance_metrics', {})
        
        # Start building the report
        report = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Model Comparison Report for {commodity}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }",
            "h1 { color: #333366; }",
            "h2 { color: #333366; margin-top: 30px; }",
            "h3 { color: #333366; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
            "th { background-color: #f2f2f2; }",
            "tr:hover { background-color: #f5f5f5; }",
            ".best-model { font-weight: bold; color: #009900; }",
            ".executive-summary { background-color: #f9f9f9; padding: 15px; border-left: 5px solid #333366; }",
            ".conclusion { background-color: #f9f9f9; padding: 15px; border-left: 5px solid #009900; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Model Comparison Report for {commodity}</h1>",
            f"<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>",
            "",
            "<h2>Executive Summary</h2>",
            "<div class='executive-summary'>",
            f"<p>This report compares different threshold model specifications for {commodity} price integration analysis. ",
            f"The best model based on AIC is <strong>{best_model}</strong>.</p>",
            "</div>",
            "",
            "<h2>Information Criteria Comparison</h2>",
            "<table>",
            "<tr><th>Model</th><th>AIC</th><th>BIC</th><th>HQIC</th><th>Log-Likelihood</th><th>Parameters</th></tr>"
        ]
        
        # Add rows for each model
        for model_name, criteria in ic_comparison.items():
            if isinstance(criteria, dict):
                aic = criteria.get('aic', 'N/A')
                bic = criteria.get('bic', 'N/A')
                hqic = criteria.get('hqic', 'N/A')
                llf = criteria.get('llf', 'N/A')
                n_params = criteria.get('n_params', 'N/A')
                
                # Format numbers
                aic = f"{aic:.4f}" if isinstance(aic, (int, float)) else aic
                bic = f"{bic:.4f}" if isinstance(bic, (int, float)) else bic
                hqic = f"{hqic:.4f}" if isinstance(hqic, (int, float)) else hqic
                llf = f"{llf:.4f}" if isinstance(llf, (int, float)) else llf
                
                # Add class for best model
                class_attr = " class='best-model'" if model_name == best_model else ""
                
                # Add row
                report.append(f"<tr{class_attr}><td>{model_name}</td><td>{aic}</td><td>{bic}</td><td>{hqic}</td><td>{llf}</td><td>{n_params}</td></tr>")
        
        # Close the table
        report.append("</table>")
        
        # Add performance metrics section if available
        if performance_metrics:
            report.extend([
                "<h2>Performance Metrics</h2>",
                "<table>",
                "<tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Adjusted R²</th></tr>"
            ])
            
            # Add rows for each model
            for model_name, metrics in performance_metrics.items():
                if isinstance(metrics, dict):
                    rmse = metrics.get('rmse', 'N/A')
                    mae = metrics.get('mae', 'N/A')
                    r2 = metrics.get('r2', 'N/A')
                    adj_r2 = metrics.get('adj_r2', 'N/A')
                    
                    # Format numbers
                    rmse = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else rmse
                    mae = f"{mae:.4f}" if isinstance(mae, (int, float)) else mae
                    r2 = f"{r2:.4f}" if isinstance(r2, (int, float)) else r2
                    adj_r2 = f"{adj_r2:.4f}" if isinstance(adj_r2, (int, float)) else adj_r2
                    
                    # Add class for best model
                    class_attr = " class='best-model'" if model_name == best_model else ""
                    
                    # Add row
                    report.append(f"<tr{class_attr}><td>{model_name}</td><td>{rmse}</td><td>{mae}</td><td>{r2}</td><td>{adj_r2}</td></tr>")
            
            # Close the table
            report.append("</table>")
        
        # Add model parameters section
        report.extend([
            "<h2>Model Parameters</h2>",
            "<h3>Threshold Values</h3>",
            "<table>",
            "<tr><th>Model</th><th>Threshold</th><th>Lower CI</th><th>Upper CI</th><th>p-value</th></tr>",
            "<!-- Placeholder for threshold values -->",
            "</table>",
            "",
            "<h2>Conclusion</h2>",
            "<div class='conclusion'>",
            f"<p>Based on the information criteria and performance metrics, the <strong>{best_model}</strong> ",
            f"provides the best fit for the {commodity} price data. This suggests that ",
            f"this specification most accurately captures the market integration dynamics for {commodity}.</p>",
            "</div>",
            "",
            "<h2>Appendix: Diagnostic Tests</h2>",
            "<h3>Residual Diagnostics</h3>",
            "<table>",
            "<tr><th>Model</th><th>Normality (JB)</th><th>Serial Correlation (LM)</th><th>Heteroskedasticity (White)</th></tr>",
            "<!-- Placeholder for diagnostic test results -->",
            "</table>",
            "",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(report)
        
    def _generate_model_comparison_visualizations(
        self,
        model_results: Dict[str, Any],
        commodity: str
    ):
        """Generate visualizations for model comparison."""
        ic_comparison = model_results.get('information_criteria', {})
        performance_metrics = model_results.get('performance_metrics', {})
        
        if not ic_comparison:
            logger.warning("No information criteria data available for visualization")
            return
        
        # Create directory for visualizations
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Information Criteria Comparison
        self._plot_information_criteria(ic_comparison, commodity, viz_dir)
        
        # 2. Performance Metrics Comparison
        if performance_metrics:
            self._plot_performance_metrics(performance_metrics, commodity, viz_dir)
    
    def _plot_information_criteria(
        self,
        ic_comparison: Dict[str, Dict[str, float]],
        commodity: str,
        output_dir: Path
    ):
        """Plot information criteria comparison."""
        # Extract data
        models = []
        aic_values = []
        bic_values = []
        hqic_values = []
        
        for model_name, criteria in ic_comparison.items():
            if isinstance(criteria, dict):
                aic = criteria.get('aic')
                bic = criteria.get('bic')
                hqic = criteria.get('hqic')
                
                if aic is not None and bic is not None:
                    models.append(model_name)
                    aic_values.append(float(aic) if isinstance(aic, (int, float)) else np.nan)
                    bic_values.append(float(bic) if isinstance(bic, (int, float)) else np.nan)
                    hqic_values.append(float(hqic) if isinstance(hqic, (int, float)) else np.nan)
        
        if not models:
            logger.warning("No valid information criteria data for plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Set width of bars
        bar_width = 0.25
        
        # Set positions of bars on x-axis
        r1 = np.arange(len(models))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        ax.bar(r1, aic_values, width=bar_width, label='AIC', color=self.colors[0])
        ax.bar(r2, bic_values, width=bar_width, label='BIC', color=self.colors[1])
        ax.bar(r3, hqic_values, width=bar_width, label='HQIC', color=self.colors[2])
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Value')
        ax.set_title(f'Information Criteria Comparison for {commodity}')
        ax.set_xticks([r + bar_width for r in range(len(models))])
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"{commodity.replace(' ', '_')}_information_criteria.png"
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Information criteria plot saved to {output_path}")
    
    def _plot_performance_metrics(
        self,
        performance_metrics: Dict[str, Dict[str, float]],
        commodity: str,
        output_dir: Path
    ):
        """Plot performance metrics comparison."""
        # Extract data
        models = []
        rmse_values = []
        mae_values = []
        r2_values = []
        
        for model_name, metrics in performance_metrics.items():
            if isinstance(metrics, dict):
                rmse = metrics.get('rmse')
                mae = metrics.get('mae')
                r2 = metrics.get('r2')
                
                if rmse is not None and mae is not None and r2 is not None:
                    models.append(model_name)
                    rmse_values.append(float(rmse) if isinstance(rmse, (int, float)) else np.nan)
                    mae_values.append(float(mae) if isinstance(mae, (int, float)) else np.nan)
                    r2_values.append(float(r2) if isinstance(r2, (int, float)) else np.nan)
        
        if not models:
            logger.warning("No valid performance metrics data for plotting")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_size[0] * 1.5, self.fig_size[1]))
        
        # Plot RMSE and MAE in the first subplot
        bar_width = 0.35
        r1 = np.arange(len(models))
        r2 = [x + bar_width for x in r1]
        
        ax1.bar(r1, rmse_values, width=bar_width, label='RMSE', color=self.colors[0])
        ax1.bar(r2, mae_values, width=bar_width, label='MAE', color=self.colors[1])
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Error')
        ax1.set_title('Error Metrics')
        ax1.set_xticks([r + bar_width/2 for r in range(len(models))])
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot R² in the second subplot
        ax2.bar(models, r2_values, color=self.colors[2])
        ax2.set_xlabel('Model')
        ax2.set_ylabel('R²')
        ax2.set_title('R² Values')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Format y-axis as percentage for R²
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"{commodity.replace(' ', '_')}_performance_metrics.png"
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
        logger.info(f"Performance metrics plot saved to {output_path}")


def generate_enhanced_report(
    all_results: Dict[str, Any],
    commodity: str,
    output_dir: Union[str, Path],
    format: str = 'markdown',
    publication_quality: bool = True,
    style: str = 'world_bank'
) -> Path:
    """
    Generate an enhanced econometric report with publication-quality tables and visualizations.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all analysis results
    commodity : str
        Commodity name
    output_dir : str or Path
        Directory to save the report
    format : str, optional
        Report format ('markdown', 'latex', 'html', 'json')
    publication_quality : bool, optional
        Whether to generate publication-quality outputs
    style : str, optional
        Visual style for reports ('world_bank', 'academic', 'policy')
        
    Returns
    -------
    Path
        Path to the generated report
    """
    # Initialize reporter
    reporter = EconometricReporter(
        output_dir=output_dir,
        publication_quality=publication_quality,
        style=style,
        format=format
    )
    
    # Generate model comparison report if available
    if 'threshold_analysis' in all_results and 'model_comparison' in all_results['threshold_analysis']:
        report_path = reporter.generate_model_comparison_report(
            model_results=all_results['threshold_analysis']['model_comparison'],
            commodity=commodity
        )
    else:
        # Generate basic report
        report_path = Path(output_dir) / f"{commodity.replace(' ', '_')}_report.{format}"
        with open(report_path, 'w') as f:
            f.write(f"# Analysis Report for {commodity}\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
            f.write("No model comparison results available.\n")
    
    return report_path


def generate_cross_commodity_comparison(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path],
    format: str = 'markdown',
    publication_quality: bool = True
) -> Path:
    """
    Generate a cross-commodity comparison report.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing results for all commodities
    output_dir : str or Path
        Directory to save the report
    format : str, optional
        Report format ('markdown', 'latex', 'html', 'json')
    publication_quality : bool, optional
        Whether to generate publication-quality outputs
        
    Returns
    -------
    Path
        Path to the generated report
    """
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / f"cross_commodity_comparison.{format}"
    
    # Extract best models and their metrics for each commodity
    commodities = []
    best_models = []
    aic_values = []
    r2_values = []
    thresholds = []
    
    for commodity, results in all_results.items():
        if 'threshold_analysis' in results and 'model_comparison' in results['threshold_analysis']:
            model_comparison = results['threshold_analysis']['model_comparison']
            best_model = model_comparison.get('best_model', 'Unknown')
            
            # Get AIC for the best model
            aic = None
            if 'information_criteria' in model_comparison:
                ic_data = model_comparison['information_criteria'].get(best_model, {})
                if isinstance(ic_data, dict):
                    aic = ic_data.get('aic')
            
            # Get R² for the best model
            r2 = None
            if 'performance_metrics' in model_comparison:
                perf_data = model_comparison['performance_metrics'].get(best_model, {})
                if isinstance(perf_data, dict):
                    r2 = perf_data.get('r2')
            
            # Get threshold value
            threshold = None
            if 'best_model_results' in results['threshold_analysis']:
                best_results = results['threshold_analysis']['best_model_results']
                if isinstance(best_results, dict):
                    threshold = best_results.get('threshold')
            
            commodities.append(commodity)
            best_models.append(best_model)
            aic_values.append(aic)
            r2_values.append(r2)
            thresholds.append(threshold)
    
    # Create report based on format
    if format == 'markdown':
        content = _generate_markdown_cross_commodity_report(
            commodities, best_models, aic_values, r2_values, thresholds
        )
    elif format == 'latex':
        content = _generate_latex_cross_commodity_report(
            commodities, best_models, aic_values, r2_values, thresholds
        )
    elif format == 'html':
        content = _generate_html_cross_commodity_report(
            commodities, best_models, aic_values, r2_values, thresholds
        )
    elif format == 'json':
        # For JSON, create a structured dictionary
        data = {
            'commodities': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for i, commodity in enumerate(commodities):
            data['commodities'].append({
                'name': commodity,
                'best_model': best_models[i],
                'aic': aic_values[i],
                'r2': r2_values[i],
                'threshold': thresholds[i]
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_path
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write(content)
    
    # Generate cross-commodity visualization
    if publication_quality and len(commodities) > 1:
        _generate_cross_commodity_visualization(
            commodities, best_models, aic_values, r2_values, thresholds, output_dir
        )
    
    return output_path


def _generate_markdown_cross_commodity_report(
    commodities: List[str],
    best_models: List[str],
    aic_values: List[Optional[float]],
    r2_values: List[Optional[float]],
    thresholds: List[Optional[float]]
) -> str:
    """Generate a markdown report for cross-commodity comparison."""
    # Start building the report
    report = [
        "# Cross-Commodity Comparison Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## Executive Summary",
        "",
        f"This report compares market integration results across {len(commodities)} commodities.",
        "",
        "## Best Model Comparison",
        "",
        "| Commodity | Best Model | AIC | R² | Threshold |",
        "|-----------|------------|-----|----|-----------:|"
    ]
    
    # Add rows for each commodity
    for i, commodity in enumerate(commodities):
        aic = aic_values[i]
        r2 = r2_values[i]
        threshold = thresholds[i]
        
        # Format numbers
        aic_str = f"{aic:.4f}" if isinstance(aic, (int, float)) else "N/A"
        r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else "N/A"
        threshold_str = f"{threshold:.4f}" if isinstance(threshold, (int, float)) else "N/A"
        
        # Add row
        report.append(f"| {commodity} | {best_models[i]} | {aic_str} | {r2_str} | {threshold_str} |")
    
    # Add conclusion
    report.extend([
        "",
        "## Conclusion",
        "",
        "The table above shows the best-fitting model for each commodity based on AIC. "
        "The threshold values represent the estimated price differential at which market adjustment mechanisms change."
    ])
    
    return "\n".join(report)


def _generate_latex_cross_commodity_report(
    commodities: List[str],
    best_models: List[str],
    aic_values: List[Optional[float]],
    r2_values: List[Optional[float]],
    thresholds: List[Optional[float]]
) -> str:
    """Generate a LaTeX report for cross-commodity comparison."""
    # Start building the report
    report = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{caption}",
        "\\usepackage{graphicx}",
        "\\usepackage{amsmath}",
        "\\usepackage{geometry}",
        "\\usepackage{siunitx}",
        "\\usepackage{xcolor}",
        "\\usepackage{hyperref}",
        "",
        "\\geometry{margin=1in}",
        "\\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}",
        "",
        "\\begin{document}",
        "",
        "\\title{Cross-Commodity Comparison Report}",
        f"\\date{{{datetime.now().strftime('%Y-%m-%d')}}}",
        "\\maketitle",
        "",
        "\\section{Executive Summary}",
        "",
        f"This report compares market integration results across {len(commodities)} commodities.",
        "",
        "\\section{Best Model Comparison}",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Best Model Comparison Across Commodities}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Commodity & Best Model & AIC & $R^2$ & Threshold \\\\",
        "\\midrule"
    ]
    
    # Add rows for each commodity
    for i, commodity in enumerate(commodities):
        aic = aic_values[i]
        r2 = r2_values[i]
        threshold = thresholds[i]
        
        # Format numbers
        aic_str = f"{aic:.4f}" if isinstance(aic, (int, float)) else "N/A"
        r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else "N/A"
        threshold_str = f"{threshold:.4f}" if isinstance(threshold, (int, float)) else "N/A"
        
        # Add row
        report.append(f"{commodity} & {best_models[i]} & {aic_str} & {r2_str} & {threshold_str} \\\\")
    
    # Close the table
    report.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\section{Conclusion}",
        "",
        "The table above shows the best-fitting model for each commodity based on AIC. ",
        "The threshold values represent the estimated price differential at which market adjustment mechanisms change.",
        "",
        "\\end{document}"
    ])
    
    return "\n".join(report)


def _generate_html_cross_commodity_report(
    commodities: List[str],
    best_models: List[str],
    aic_values: List[Optional[float]],
    r2_values: List[Optional[float]],
    thresholds: List[Optional[float]]
) -> str:
    """Generate an HTML report for cross-commodity comparison."""
    # Start building the report
    report = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<title>Cross-Commodity Comparison Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }",
        "h1 { color: #333366; }",
        "h2 { color: #333366; margin-top: 30px; }",
        "h3 { color: #333366; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
        "th { background-color: #f2f2f2; }",
        "tr:hover { background-color: #f5f5f5; }",
        ".executive-summary { background-color: #f9f9f9; padding: 15px; border-left: 5px solid #333366; }",
        ".conclusion { background-color: #f9f9f9; padding: 15px; border-left: 5px solid #009900; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Cross-Commodity Comparison Report</h1>",
        f"<p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>",
        "",
        "<h2>Executive Summary</h2>",
        "<div class='executive-summary'>",
        f"<p>This report compares market integration results across {len(commodities)} commodities.</p>",
        "</div>",
        "",
        "<h2>Best Model Comparison</h2>",
        "<table>",
        "<tr><th>Commodity</th><th>Best Model</th><th>AIC</th><th>R²</th><th>Threshold</th></tr>"
    ]
    
    # Add rows for each commodity
    for i, commodity in enumerate(commodities):
        aic = aic_values[i]
        r2 = r2_values[i]
        threshold = thresholds[i]
        
        # Format numbers
        aic_str = f"{aic:.4f}" if isinstance(aic, (int, float)) else "N/A"
        r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else "N/A"
        threshold_str = f"{threshold:.4f}" if isinstance(threshold, (int, float)) else "N/A"
        
        # Add row
        report.append(f"<tr><td>{commodity}</td><td>{best_models[i]}</td><td>{aic_str}</td><td>{r2_str}</td><td>{threshold_str}</td></tr>")
    
    # Close the table
    report.append("</table>")
    
    # Add conclusion
    report.extend([
        "<h2>Conclusion</h2>",
        "<div class='conclusion'>",
        "<p>The table above shows the best-fitting model for each commodity based on AIC. ",
        "The threshold values represent the estimated price differential at which market adjustment mechanisms change.</p>",
        "</div>",
        "",
        "</body>",
        "</html>"
    ])
    
    return "\n".join(report)


def _generate_cross_commodity_visualization(
    commodities: List[str],
    best_models: List[str],
    aic_values: List[Optional[float]],
    r2_values: List[Optional[float]],
    thresholds: List[Optional[float]],
    output_dir: Path
):
    """Generate visualizations for cross-commodity comparison."""
    # Create directory for visualizations
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Filter out None values
    valid_indices = [i for i, val in enumerate(thresholds) if isinstance(val, (int, float))]
    valid_commodities = [commodities[i] for i in valid_indices]
    valid_thresholds = [thresholds[i] for i in valid_indices]
    
    if not valid_commodities:
        logger.warning("No valid threshold data for cross-commodity visualization")
        return
    
    # Create figure for threshold comparison
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(valid_commodities))
    plt.barh(y_pos, valid_thresholds, align='center', color='#0071bc')
    plt.yticks(y_pos, valid_commodities)
    
    # Add labels and title
    plt.xlabel('Threshold Value')
    plt.title('Threshold Values Across Commodities')
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = viz_dir / "cross_commodity_thresholds.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Cross-commodity threshold comparison saved to {output_path}")
