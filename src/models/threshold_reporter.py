"""
Standardized academic reporting for threshold models.

This module provides a consistent reporting mechanism for all threshold
model modes, supporting multiple output formats with academic publication standards.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from yemen_market_integration.utils.error_handler import handle_errors
from yemen_market_integration.utils.plotting_utils import (
    set_plotting_style, save_plot, add_annotations, format_date_axis
)
from yemen_market_integration.utils.config import config

# Initialize module logger
logger = logging.getLogger(__name__)


class ThresholdReporter:
    """
    Standardized reporting for threshold models.
    
    This class provides a consistent reporting mechanism for all threshold
    model modes, supporting multiple output formats.
    
    Parameters
    ----------
    model : ThresholdModel
        Threshold model instance
    format : str, optional
        Report format ('markdown', 'json', or 'latex')
    output_path : str, optional
        Path to save the report
    """
    
    def __init__(
        self, 
        model, 
        format: str = "markdown", 
        output_path: Optional[str] = None
    ):
        """Initialize the reporter."""
        self.model = model
        self.format = format
        self.output_path = output_path
        
        # Validate format
        valid_formats = ["markdown", "json", "latex"]
        if format not in valid_formats:
            raise ValueError(f"Invalid format: {format}. Must be one of {valid_formats}")
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report.
        
        Returns
        -------
        dict
            Report content and metadata
        """
        # Ensure model has results
        if self.model.threshold is None:
            logger.info("Running full analysis first")
            self.model.run_full_analysis()
        
        # Generate tables, visualizations, and interpretation
        tables = self.generate_tables()
        visualizations = self.generate_visualizations()
        interpretation = self.generate_interpretation()
        
        # Combine into report
        report = {
            'tables': tables,
            'visualizations': visualizations,
            'interpretation': interpretation,
            'metadata': {
                'model_type': self.model.__class__.__name__,
                'model_mode': self.model.mode,
                'market1': self.model.market1_name,
                'market2': self.model.market2_name,
                'timestamp': datetime.now().isoformat(),
                'format': self.format
            }
        }
        
        # Export if path provided
        if self.output_path:
            self.export_report(report)
        
        import matplotlib.pyplot as plt
        plt.close('all')
        
        return report
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def generate_tables(self) -> Dict[str, Any]:
        """
        Generate statistical tables.
        
        Returns
        -------
        dict
            Statistical tables
        """
        tables = {}
        
        # Cointegration table
        if hasattr(self.model, 'beta0') and hasattr(self.model, 'beta1'):
            tables['cointegration'] = {
                'title': 'Cointegration Analysis',
                'headers': ['Parameter', 'Value', 'Description'],
                'rows': [
                    ['Intercept (β₀)', f"{self.model.beta0:.4f}", "Long-run equilibrium intercept"],
                    ['Slope (β₁)', f"{self.model.beta1:.4f}", "Long-run price transmission elasticity"],
                    ['Cointegration Equation', f"{self.model.market1_name} = {self.model.beta0:.4f} + {self.model.beta1:.4f} × {self.model.market2_name}", "Long-run equilibrium relationship"]
                ]
            }
        
        # Threshold table
        if hasattr(self.model, 'threshold'):
            below_prop = np.mean(self.model.eq_errors <= self.model.threshold)
            above_prop = 1 - below_prop
            
            tables['threshold'] = {
                'title': 'Threshold Estimation',
                'headers': ['Parameter', 'Value', 'Description'],
                'rows': [
                    ['Threshold', f"{self.model.threshold:.4f}" if self.model.threshold is not None else "N/A", "Estimated transaction cost threshold"],
                    ['Below Threshold', f"{below_prop:.1%}", "Proportion of observations below threshold"],
                    ['Above Threshold', f"{above_prop:.1%}", "Proportion of observations above threshold"]
                ]
            }
        
        # Mode-specific tables
        if self.model.mode == "standard" or self.model.mode == "fixed":
            # Add TVECM results if available
            if hasattr(self.model, 'results') and self.model.results is not None:
                if 'adjustment_below_1' in self.model.results and 'adjustment_above_1' in self.model.results:
                    tables['adjustment_speeds'] = {
                        'title': 'Adjustment Speed Analysis',
                        'headers': ['Regime', 'Adjustment Speed', 'Half-Life (periods)', 'Description'],
                        'rows': [
                            ['Below Threshold', f"{self.model.results['adjustment_below_1']:.4f}", 
                             self._format_half_life(self.model.results.get('asymmetric_adjustment', {}).get('half_life_below_1', np.nan)),
                             "Adjustment when price differential is within transaction costs"],
                            ['Above Threshold', f"{self.model.results['adjustment_above_1']:.4f}", 
                             self._format_half_life(self.model.results.get('asymmetric_adjustment', {}).get('half_life_above_1', np.nan)),
                             "Adjustment when price differential exceeds transaction costs"],
                            ['Asymmetry', f"{self.model.results.get('asymmetric_adjustment', {}).get('asymmetry_1', 0):.4f}", 
                             "N/A", "Difference in adjustment speeds (above - below)"]
                        ]
                    }
        
        elif self.model.mode == "vecm":
            # Add VECM-specific tables
            if hasattr(self.model, 'below_model') and hasattr(self.model, 'above_model'):
                tables['vecm_results'] = {
                    'title': 'Threshold VECM Results',
                    'headers': ['Parameter', 'Below Threshold', 'Above Threshold', 'Difference'],
                    'rows': [
                        ['Alpha (Market 1)',
                         f"{self.model.below_model.alpha[0, 0]:.4f}" if hasattr(self.model.below_model, 'alpha') else f"{self.model.below_model.params.iloc[1]:.4f}",
                         f"{self.model.above_model.alpha[0, 0]:.4f}" if hasattr(self.model.above_model, 'alpha') else f"{self.model.above_model.params.iloc[1]:.4f}",
                         "N/A"],
                        ['Alpha (Market 2)',
                         f"{self.model.below_model.alpha[1, 0]:.4f}" if hasattr(self.model.below_model, 'alpha') else f"{self.model.below_model.params.iloc[1]:.4f}",
                         f"{self.model.above_model.alpha[1, 0]:.4f}" if hasattr(self.model.above_model, 'alpha') else f"{self.model.above_model.params.iloc[1]:.4f}",
                         "N/A"],
                        ['Log-Likelihood',
                         f"{self.model.below_model.llf:.2f}" if hasattr(self.model.below_model, 'llf') else "N/A",
                         f"{self.model.above_model.llf:.2f}" if hasattr(self.model.above_model, 'llf') else "N/A",
                         f"{self.model.below_model.llf + self.model.above_model.llf - self.model.linear_model.llf:.2f}" if (hasattr(self.model.below_model, 'llf') and hasattr(self.model.above_model, 'llf') and hasattr(self.model.linear_model, 'llf')) else "N/A"],
                        ['AIC',
                         f"{self.model.below_model.aic:.2f}" if hasattr(self.model.below_model, 'aic') else "N/A",
                         f"{self.model.above_model.aic:.2f}" if hasattr(self.model.above_model, 'aic') else "N/A",
                         "N/A"]
                    ]
                }
        
        elif self.model.mode == "mtar":
            # Add M-TAR specific tables
            if hasattr(self.model, 'mtar_results'):
                tables['mtar_results'] = {
                    'title': 'Momentum Threshold Results',
                    'headers': ['Parameter', 'Positive Momentum', 'Negative Momentum', 'Difference'],
                    'rows': [
                        ['Adjustment Speed', f"{self.model.mtar_results['adjustment_positive']:.4f}", 
                         f"{self.model.mtar_results['adjustment_negative']:.4f}", 
                         f"{self.model.mtar_results['adjustment_positive'] - self.model.mtar_results['adjustment_negative']:.4f}"],
                        ['Half-Life (periods)', 
                         self._format_half_life(self.model.mtar_results['half_life_positive']), 
                         self._format_half_life(self.model.mtar_results['half_life_negative']), 
                         "N/A"],
                        ['Proportion', f"{self.model.mtar_results['n_positive'] / (self.model.mtar_results['n_positive'] + self.model.mtar_results['n_negative']):.1%}", 
                         f"{self.model.mtar_results['n_negative'] / (self.model.mtar_results['n_positive'] + self.model.mtar_results['n_negative']):.1%}", 
                         "N/A"],
                        ['Asymmetry Test p-value', f"{self.model.mtar_results['p_value']:.4f}", 
                         "N/A", 
                         "N/A"]
                    ]
                }
        
        return tables
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def generate_visualizations(self) -> Dict[str, Any]:
        """
        Generate visualizations.
        
        Returns
        -------
        dict
            Visualization specifications
        """
        visualizations = {}
        
        # Create regime dynamics plot
        try:
            regime_dynamics_plot = self._create_regime_dynamics_plot()
            visualizations['regime_dynamics'] = {
                'title': 'Threshold Regime Dynamics',
                'description': 'Visualization of price adjustment dynamics in different regimes',
                'plot': regime_dynamics_plot
            }
        except Exception as e:
            logger.warning(f"Failed to create regime dynamics plot: {str(e)}")
        
        # Create time series plot
        try:
            time_series_plot = self._create_time_series_plot()
            visualizations['time_series'] = {
                'title': 'Market Price Time Series',
                'description': 'Time series of prices in both markets',
                'plot': time_series_plot
            }
        except Exception as e:
            logger.warning(f"Failed to create time series plot: {str(e)}")
        
        # Create equilibrium error plot
        try:
            eq_error_plot = self._create_equilibrium_error_plot()
            visualizations['equilibrium_error'] = {
                'title': 'Equilibrium Error Dynamics',
                'description': 'Deviations from long-run equilibrium with threshold',
                'plot': eq_error_plot
            }
        except Exception as e:
            logger.warning(f"Failed to create equilibrium error plot: {str(e)}")
        
        return visualizations
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def generate_interpretation(self) -> Dict[str, str]:
        """
        Generate text interpretation.
        
        Returns
        -------
        dict
            Text interpretations
        """
        interpretation = {}
        
        # Cointegration interpretation
        if hasattr(self.model, 'beta0') and hasattr(self.model, 'beta1'):
            interpretation['cointegration'] = self._interpret_cointegration()
        
        # Threshold interpretation
        if hasattr(self.model, 'threshold'):
            interpretation['threshold'] = self._interpret_threshold()
        
        # Mode-specific interpretation
        if self.model.mode == "standard" or self.model.mode == "fixed":
            if hasattr(self.model, 'results') and self.model.results is not None:
                interpretation['adjustment'] = self._interpret_adjustment_speeds()
        
        elif self.model.mode == "vecm":
            if hasattr(self.model, 'below_model') and hasattr(self.model, 'above_model'):
                interpretation['vecm'] = self._interpret_vecm_results()
        
        elif self.model.mode == "mtar":
            if hasattr(self.model, 'mtar_results'):
                interpretation['mtar'] = self._interpret_mtar_results()
        
        # Overall market integration assessment
        interpretation['market_integration'] = self._assess_market_integration()
        
        return interpretation
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def export_report(self, report: Dict[str, Any]) -> str:
        """
        Export report to file.
        
        Parameters
        ----------
        report : dict
            Report content
            
        Returns
        -------
        str
            Path to exported report
        """
        if self.output_path is None:
            # Create default output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"threshold_report_{self.model.mode}_{timestamp}"
            
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            if self.format == "markdown":
                self.output_path = f"results/{filename}.md"
            elif self.format == "json":
                self.output_path = f"results/{filename}.json"
            elif self.format == "latex":
                self.output_path = f"results/{filename}.tex"
        
        # Export based on format
        if self.format == "markdown":
            self._export_markdown(report)
        elif self.format == "json":
            self._export_json(report)
        elif self.format == "latex":
            self._export_latex(report)
        
        logger.info(f"Exported report to {self.output_path}")
        
        return self.output_path
    
    def _export_markdown(self, report: Dict[str, Any]) -> None:
        """Export report in Markdown format."""
        with open(self.output_path, 'w') as f:
            # Title
            f.write(f"# Threshold Model Analysis Report\n\n")
            
            # Metadata
            f.write("## Metadata\n\n")
            f.write(f"- **Model Type:** {report['metadata']['model_type']}\n")
            f.write(f"- **Model Mode:** {report['metadata']['model_mode']}\n")
            f.write(f"- **Market 1:** {report['metadata']['market1']}\n")
            f.write(f"- **Market 2:** {report['metadata']['market2']}\n")
            f.write(f"- **Generated:** {report['metadata']['timestamp']}\n\n")
            
            # Interpretation
            f.write("## Interpretation\n\n")
            for section, text in report['interpretation'].items():
                f.write(f"### {section.replace('_', ' ').title()}\n\n")
                f.write(f"{text}\n\n")
            
            # Tables
            f.write("## Statistical Results\n\n")
            for table_id, table in report['tables'].items():
                f.write(f"### {table['title']}\n\n")
                
                # Write table headers
                f.write("| " + " | ".join(table['headers']) + " |\n")
                f.write("| " + " | ".join(["---"] * len(table['headers'])) + " |\n")
                
                # Write table rows
                for row in table['rows']:
                    f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
                
                f.write("\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            for viz_id, viz in report['visualizations'].items():
                f.write(f"### {viz['title']}\n\n")
                f.write(f"{viz['description']}\n\n")
                
                # For markdown, we save the plots and reference them
                if 'plot' in viz:
                    plot_filename = f"{os.path.splitext(os.path.basename(self.output_path))[0]}_{viz_id}.png"
                    plot_path = os.path.join(os.path.dirname(self.output_path), plot_filename)
                    save_plot(viz['plot'], plot_path, dpi=300)
                    f.write(f"![{viz['title']}]({plot_filename})\n\n")
    
    def _export_json(self, report: Dict[str, Any]) -> None:
        """Export report in JSON format."""
        # Convert plots to filenames
        json_report = report.copy()
        
        # Save plots and update references
        for viz_id, viz in json_report['visualizations'].items():
            if 'plot' in viz:
                plot_filename = f"{os.path.splitext(os.path.basename(self.output_path))[0]}_{viz_id}.png"
                plot_path = os.path.join(os.path.dirname(self.output_path), plot_filename)
                save_plot(viz['plot'], plot_path, dpi=300)
                viz['plot'] = plot_filename
        
        # Write JSON file
        with open(self.output_path, 'w') as f:
            json.dump(json_report, f, indent=2)
    
    def _export_latex(self, report: Dict[str, Any]) -> None:
        """Export report in LaTeX format."""
        with open(self.output_path, 'w') as f:
            # Document preamble
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{caption}\n")
            f.write("\\usepackage{float}\n")
            f.write("\\usepackage{geometry}\n")
            f.write("\\geometry{margin=1in}\n")
            f.write("\\title{Threshold Model Analysis Report}\n")
            f.write(f"\\author{{{report['metadata']['market1']} and {report['metadata']['market2']}}}\n")
            f.write("\\date{\\today}\n\n")
            f.write("\\begin{document}\n\n")
            f.write("\\maketitle\n\n")
            
            # Metadata
            f.write("\\section{Metadata}\n\n")
            f.write("\\begin{itemize}\n")
            f.write(f"\\item \\textbf{{Model Type:}} {report['metadata']['model_type']}\n")
            f.write(f"\\item \\textbf{{Model Mode:}} {report['metadata']['model_mode']}\n")
            f.write(f"\\item \\textbf{{Market 1:}} {report['metadata']['market1']}\n")
            f.write(f"\\item \\textbf{{Market 2:}} {report['metadata']['market2']}\n")
            f.write(f"\\item \\textbf{{Generated:}} {report['metadata']['timestamp']}\n")
            f.write("\\end{itemize}\n\n")
            
            # Interpretation
            f.write("\\section{Interpretation}\n\n")
            for section, text in report['interpretation'].items():
                f.write(f"\\subsection{{{section.replace('_', ' ').title()}}}\n\n")
                f.write(f"{text}\n\n")
            
            # Tables
            f.write("\\section{Statistical Results}\n\n")
            for table_id, table in report['tables'].items():
                f.write(f"\\subsection{{{table['title']}}}\n\n")
                
                # Begin table
                f.write("\\begin{table}[H]\n")
                f.write("\\centering\n")
                f.write("\\caption{" + table['title'] + "}\n")
                f.write("\\begin{tabular}{" + "l" * len(table['headers']) + "}\n")
                f.write("\\toprule\n")
                
                # Write table headers
                f.write(" & ".join(table['headers']) + " \\\\\n")
                f.write("\\midrule\n")
                
                # Write table rows
                for row in table['rows']:
                    f.write(" & ".join(str(cell) for cell in row) + " \\\\\n")
                
                # End table
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Visualizations
            f.write("\\section{Visualizations}\n\n")
            for viz_id, viz in report['visualizations'].items():
                f.write(f"\\subsection{{{viz['title']}}}\n\n")
                f.write(f"{viz['description']}\n\n")
                
                # For LaTeX, we save the plots and include them
                if 'plot' in viz:
                    plot_filename = f"{os.path.splitext(os.path.basename(self.output_path))[0]}_{viz_id}.png"
                    plot_path = os.path.join(os.path.dirname(self.output_path), plot_filename)
                    save_plot(viz['plot'], plot_path, dpi=300)
                    
                    f.write("\\begin{figure}[H]\n")
                    f.write("\\centering\n")
                    f.write(f"\\includegraphics[width=0.8\\textwidth]{{{plot_filename}}}\n")
                    f.write(f"\\caption{{{viz['title']}}}\n")
                    f.write("\\end{figure}\n\n")
            
            # End document
            f.write("\\end{document}\n")
    
    def _create_regime_dynamics_plot(self) -> plt.Figure:
        """Create plot of regime dynamics."""
        # Set plotting style
        set_plotting_style()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f"Threshold Regime Dynamics\nThreshold = {self.model.threshold:.4f}", fontsize=14)
        
        # Plot 1: Error correction term over time with threshold
        ax = axes[0]
        if self.model.index is not None:
            # Time series plot
            ax.plot(self.model.index, self.model.eq_errors, color='blue', label='Equilibrium Error')
            format_date_axis(ax)
        else:
            # Index plot
            ax.plot(self.model.eq_errors, color='blue', label='Equilibrium Error')
            
        ax.axhline(y=self.model.threshold, color='r', linestyle='--', label=f'Threshold: {self.model.threshold:.4f}')
        ax.set_title("Equilibrium Error Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equilibrium Error")
        ax.legend()
        
        # Plot 2: Regime distribution
        ax = axes[1]
        ax.hist(self.model.eq_errors, bins=30, density=True, alpha=0.7)
        ax.axvline(x=self.model.threshold, color='r', linestyle='--')
        
        # Add annotation about regime proportions
        below_prop = np.mean(self.model.eq_errors <= self.model.threshold)
        above_prop = 1 - below_prop
        ax.annotate(
            f"Below: {below_prop:.1%}\nAbove: {above_prop:.1%}",
            xy=(self.model.threshold, 0.8 * ax.get_ylim()[1]),
            xytext=(10, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )
        
        ax.set_title("Distribution of Equilibrium Error")
        ax.set_xlabel("Equilibrium Error")
        ax.set_ylabel("Density")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig
    
    def _create_time_series_plot(self) -> plt.Figure:
        """Create plot of market price time series."""
        # Set plotting style
        set_plotting_style()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot time series
        if self.model.index is not None:
            # Time series plot with dates
            ax.plot(self.model.index, self.model.data1, label=self.model.market1_name)
            ax.plot(self.model.index, self.model.data2, label=self.model.market2_name)
            format_date_axis(ax)
        else:
            # Index plot
            ax.plot(self.model.data1, label=self.model.market1_name)
            ax.plot(self.model.data2, label=self.model.market2_name)
        
        ax.set_title(f"Price Series: {self.model.market1_name} and {self.model.market2_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        
        plt.tight_layout()
        
        return fig
    
    def _create_equilibrium_error_plot(self) -> plt.Figure:
        """Create plot of equilibrium error with threshold."""
        # Set plotting style
        set_plotting_style()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot equilibrium error
        if self.model.index is not None:
            # Time series plot with dates
            below_mask = self.model.eq_errors <= self.model.threshold
            above_mask = ~below_mask
            
            # Plot points below threshold in blue
            ax.scatter(
                self.model.index[below_mask], 
                self.model.eq_errors[below_mask], 
                color='blue', 
                alpha=0.6, 
                label='Below Threshold'
            )
            
            # Plot points above threshold in red
            ax.scatter(
                self.model.index[above_mask], 
                self.model.eq_errors[above_mask], 
                color='red', 
                alpha=0.6, 
                label='Above Threshold'
            )
            
            format_date_axis(ax)
        else:
            # Index plot
            below_mask = self.model.eq_errors <= self.model.threshold
            above_mask = ~below_mask
            
            # Plot points below threshold in blue
            ax.scatter(
                np.arange(len(self.model.eq_errors))[below_mask], 
                self.model.eq_errors[below_mask], 
                color='blue', 
                alpha=0.6, 
                label='Below Threshold'
            )
            
            # Plot points above threshold in red
            ax.scatter(
                np.arange(len(self.model.eq_errors))[above_mask], 
                self.model.eq_errors[above_mask], 
                color='red', 
                alpha=0.6, 
                label='Above Threshold'
            )
        
        # Add threshold line
        ax.axhline(y=self.model.threshold, color='black', linestyle='--', label=f'Threshold: {self.model.threshold:.4f}')
        
        ax.set_title("Equilibrium Error with Threshold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equilibrium Error")
        ax.legend()
        
        plt.tight_layout()
        
        return fig
    
    def _interpret_cointegration(self) -> str:
        """Generate interpretation of cointegration results."""
        # Check if markets are cointegrated
        coint_results = self.model.estimate_cointegration()
        
        if coint_results['cointegrated']:
            # Interpret cointegration relationship
            if abs(self.model.beta1 - 1.0) < 0.1:
                elasticity_text = (
                    f"The long-run price transmission elasticity is approximately 1 "
                    f"({self.model.beta1:.2f}), indicating strong market integration with "
                    f"complete price transmission between {self.model.market1_name} and "
                    f"{self.model.market2_name} in the long run."
                )
            elif self.model.beta1 > 1.0:
                elasticity_text = (
                    f"The long-run price transmission elasticity is {self.model.beta1:.2f}, "
                    f"indicating that a 1% change in {self.model.market2_name} prices is associated "
                    f"with a {self.model.beta1:.2f}% change in {self.model.market1_name} prices. "
                    f"This suggests that {self.model.market1_name} prices are more volatile."
                )
            else:
                elasticity_text = (
                    f"The long-run price transmission elasticity is {self.model.beta1:.2f}, "
                    f"indicating that a 1% change in {self.model.market2_name} prices is associated "
                    f"with a {self.model.beta1:.2f}% change in {self.model.market1_name} prices. "
                    f"This suggests partial price transmission between markets."
                )
            
            return (
                f"The markets {self.model.market1_name} and {self.model.market2_name} are cointegrated, "
                f"indicating a stable long-run equilibrium relationship between their prices. "
                f"{elasticity_text} The cointegration relationship is statistically significant "
                f"(p-value: {coint_results['pvalue']:.4f})."
            )
        else:
            return (
                f"The markets {self.model.market1_name} and {self.model.market2_name} are not cointegrated "
                f"(p-value: {coint_results['pvalue']:.4f}). This suggests that these markets do not share "
                f"a stable long-run equilibrium relationship, possibly due to significant barriers to trade "
                f"such as conflict zones, political fragmentation, or transportation constraints."
            )
    
    def _interpret_threshold(self) -> str:
        """Generate interpretation of threshold results."""
        if self.model.threshold is None:
            return (
                f"No valid threshold could be estimated between {self.model.market1_name} and {self.model.market2_name}. "
                f"This may be due to insufficient data, lack of cointegration, or complex market dynamics "
                f"that cannot be captured by the current threshold model specification."
            )
            
        below_prop = np.mean(self.model.eq_errors <= self.model.threshold)
        above_prop = 1 - below_prop
        
        return (
            f"The estimated threshold is {self.model.threshold:.4f}, representing the transaction cost "
            f"barrier between {self.model.market1_name} and {self.model.market2_name}. "
            f"When price differentials are below this threshold ({below_prop:.1%} of observations), "
            f"arbitrage is not profitable due to transaction costs. When price differentials exceed "
            f"the threshold ({above_prop:.1%} of observations), arbitrage becomes profitable and "
            f"drives prices back toward equilibrium. In Yemen's conflict context, this threshold "
            f"captures barriers including security checkpoints, conflict zones, and dual exchange "
            f"rate effects."
        )
    
    def _interpret_adjustment_speeds(self) -> str:
        """Generate interpretation of adjustment speeds."""
        if not hasattr(self.model, 'results') or self.model.results is None:
            return "Adjustment speed analysis not available."
        
        if 'adjustment_below_1' not in self.model.results or 'adjustment_above_1' not in self.model.results:
            return "Adjustment speed analysis not available."
        
        adj_below = self.model.results['adjustment_below_1']
        adj_above = self.model.results['adjustment_above_1']
        
        # Get half-lives if available
        half_life_below = self.model.results.get('asymmetric_adjustment', {}).get('half_life_below_1', np.nan)
        half_life_above = self.model.results.get('asymmetric_adjustment', {}).get('half_life_above_1', np.nan)
        
        # Check if adjustment is faster above threshold
        if abs(adj_above) > 2 * abs(adj_below):
            speed_text = (
                f"Price adjustment is substantially faster when price differentials exceed the threshold "
                f"({abs(adj_above):.2f} vs. {abs(adj_below):.2f}). This indicates significant "
                f"transaction cost barriers that, once overcome, allow rapid price convergence."
            )
        elif abs(adj_above) > 1.2 * abs(adj_below):
            speed_text = (
                f"Price adjustment is moderately faster when price differentials exceed the threshold "
                f"({abs(adj_above):.2f} vs. {abs(adj_below):.2f}). This is consistent with the presence "
                f"of transaction costs that partially impede arbitrage at smaller price differentials."
            )
        elif abs(adj_below) > 1.2 * abs(adj_above):
            speed_text = (
                f"Unusually, price adjustment is faster below than above the threshold "
                f"({abs(adj_below):.2f} vs. {abs(adj_above):.2f}). This may indicate data quality issues, "
                f"structural breaks, or complex market dynamics requiring further investigation."
            )
        else:
            speed_text = (
                f"Price adjustment speeds are similar in both regimes ({abs(adj_below):.2f} vs. "
                f"{abs(adj_above):.2f}), suggesting either minimal threshold effects or uniform "
                f"barriers affecting all price levels."
            )
        
        # Add half-life interpretation if available
        if not np.isnan(half_life_below) and not np.isnan(half_life_above):
            if np.isinf(half_life_below) and not np.isinf(half_life_above):
                half_life_text = (
                    f"When price differentials are below the threshold, deviations from equilibrium "
                    f"persist indefinitely. Above the threshold, deviations have a half-life of "
                    f"{half_life_above:.1f} periods."
                )
            elif not np.isinf(half_life_below) and np.isinf(half_life_above):
                half_life_text = (
                    f"When price differentials are above the threshold, deviations from equilibrium "
                    f"persist indefinitely. Below the threshold, deviations have a half-life of "
                    f"{half_life_below:.1f} periods."
                )
            elif np.isinf(half_life_below) and np.isinf(half_life_above):
                half_life_text = (
                    f"Deviations from equilibrium persist indefinitely in both regimes, suggesting "
                    f"very weak or non-existent price adjustment mechanisms."
                )
            else:
                half_life_text = (
                    f"The half-life of deviations is {half_life_below:.1f} periods below the threshold "
                    f"and {half_life_above:.1f} periods above the threshold."
                )
        else:
            half_life_text = ""
        
        return f"{speed_text} {half_life_text}"
    
    def _interpret_vecm_results(self) -> str:
        """Generate interpretation of VECM results."""
        if not hasattr(self.model, 'below_model') or not hasattr(self.model, 'above_model'):
            return "VECM results not available."
        
        # Check if models have alpha attribute
        if not hasattr(self.model.below_model, 'alpha') or not hasattr(self.model.above_model, 'alpha'):
            return "VECM adjustment parameters not available for this model specification."
            
        below_alpha = self.model.below_model.alpha
        above_alpha = self.model.above_model.alpha
        
        # Calculate average absolute adjustment in each regime
        below_adj_avg = np.mean(np.abs(below_alpha))
        above_adj_avg = np.mean(np.abs(above_alpha))
        
        # Compare adjustment speeds
        if above_adj_avg > 2 * below_adj_avg:
            speed_text = (
                f"Price adjustment is substantially faster ({above_adj_avg/below_adj_avg:.1f}x) "
                f"when price differentials exceed the threshold. This indicates significant "
                f"transaction cost barriers that, once overcome, allow rapid price convergence."
            )
        elif above_adj_avg > 1.2 * below_adj_avg:
            speed_text = (
                f"Price adjustment is moderately faster ({above_adj_avg/below_adj_avg:.1f}x) "
                f"above the threshold, consistent with the presence of transaction costs "
                f"that partially impede arbitrage at smaller price differentials."
            )
        elif below_adj_avg > 1.2 * above_adj_avg:
            speed_text = (
                f"Unusually, price adjustment is faster below than above the threshold. "
                f"This may indicate data quality issues, structural breaks, or complex "
                f"market dynamics requiring further investigation."
            )
        else:
            speed_text = (
                f"Price adjustment speeds are similar in both regimes, suggesting either "
                f"minimal threshold effects or uniform barriers affecting all price levels."
            )
        
        # Assess market integration
        if below_adj_avg < 0.05 and above_adj_avg < 0.05:
            integration_text = (
                "Overall adjustment speeds are very low in both regimes, indicating weak "
                "market integration likely due to significant conflict barriers and political fragmentation."
            )
        elif below_adj_avg < 0.05 and above_adj_avg >= 0.05:
            integration_text = (
                "Markets show threshold-limited integration, with adjustment occurring only "
                "when price differentials exceed transaction costs. This pattern is consistent "
                "with significant but surmountable conflict-related barriers to trade."
            )
        elif above_adj_avg > 0.1:
            integration_text = (
                "Substantial price adjustment above the threshold indicates strong long-run "
                "market relationships despite the presence of threshold effects."
            )
        else:
            integration_text = (
                "Moderate price adjustment in both regimes suggests partial market integration "
                "with persistent barriers affecting price transmission."
            )
        
        return f"{speed_text} {integration_text}"
    
    def _interpret_mtar_results(self) -> str:
        """Generate interpretation of M-TAR results."""
        if not hasattr(self.model, 'mtar_results'):
            return "M-TAR results not available."
        
        mtar_results = self.model.mtar_results
        
        # Check if asymmetry is significant
        if mtar_results['asymmetric']:
            if abs(mtar_results['adjustment_positive']) > abs(mtar_results['adjustment_negative']):
                asymm_text = (
                    f"Price adjustment is significantly faster when prices are rising "
                    f"({abs(mtar_results['adjustment_positive']):.2f}) than when they are falling "
                    f"({abs(mtar_results['adjustment_negative']):.2f}). This indicates that "
                    f"price increases are transmitted more quickly than price decreases, "
                    f"suggesting potential market power or information asymmetries."
                )
            else:
                asymm_text = (
                    f"Price adjustment is significantly faster when prices are falling "
                    f"({abs(mtar_results['adjustment_negative']):.2f}) than when they are rising "
                    f"({abs(mtar_results['adjustment_positive']):.2f}). This indicates that "
                    f"price decreases are transmitted more quickly than price increases, "
                    f"which is unusual and may reflect specific market conditions or policies."
                )
        else:
            asymm_text = (
                f"There is no significant difference in price adjustment speeds between "
                f"rising and falling prices (p-value: {mtar_results['p_value']:.4f}). "
                f"This suggests symmetric price transmission regardless of price direction."
            )
        
        return asymm_text
    
    def _assess_market_integration(self) -> str:
        """Generate overall market integration assessment."""
        # Check if markets are cointegrated
        coint_results = self.model.estimate_cointegration()
        
        if not coint_results['cointegrated']:
            return (
                f"The markets {self.model.market1_name} and {self.model.market2_name} are not integrated. "
                f"The absence of cointegration indicates that prices do not share a long-run equilibrium "
                f"relationship, suggesting these markets operate independently. This is likely due to "
                f"significant barriers to trade such as conflict zones, political fragmentation, or "
                f"transportation constraints that prevent effective arbitrage."
            )
        
        # For cointegrated markets, assess based on mode
        if self.model.mode == "standard" or self.model.mode == "fixed":
            if hasattr(self.model, 'results') and self.model.results is not None:
                adj_below = self.model.results.get('adjustment_below_1', 0)
                adj_above = self.model.results.get('adjustment_above_1', 0)
                
                # Check adjustment speeds
                if abs(adj_below) < 0.05 and abs(adj_above) < 0.05:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show weak integration. "
                        f"While cointegrated, the slow adjustment speeds in both regimes indicate significant "
                        f"barriers to trade that impede price transmission. This pattern is consistent with "
                        f"conflict-affected markets where arbitrage is difficult due to security risks, "
                        f"checkpoints, and political fragmentation."
                    )
                elif abs(adj_below) < 0.05 and abs(adj_above) >= 0.05:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show threshold-dependent "
                        f"integration. Price adjustment occurs primarily when differentials exceed the transaction "
                        f"cost threshold, indicating that arbitrage becomes effective only when price gaps are large "
                        f"enough to overcome conflict-related barriers."
                    )
                elif abs(adj_above) > 0.1:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show strong threshold "
                        f"integration. Despite transaction cost barriers, price adjustment is rapid when differentials "
                        f"exceed the threshold, indicating effective arbitrage mechanisms once price gaps are large enough."
                    )
                else:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show moderate integration "
                        f"with threshold effects. Price adjustment occurs in both regimes but is constrained by "
                        f"transaction costs that create nonlinear dynamics in price transmission."
                    )
        
        elif self.model.mode == "vecm":
            if hasattr(self.model, 'below_model') and hasattr(self.model, 'above_model'):
                below_alpha = self.model.below_model.alpha
                above_alpha = self.model.above_model.alpha
                
                # Calculate average absolute adjustment in each regime
                below_adj_avg = np.mean(np.abs(below_alpha))
                above_adj_avg = np.mean(np.abs(above_alpha))
                
                if below_adj_avg < 0.05 and above_adj_avg < 0.05:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show weak integration "
                        f"in the VECM framework. The slow adjustment speeds in both regimes indicate significant "
                        f"barriers to trade that impede price transmission."
                    )
                elif below_adj_avg < 0.05 and above_adj_avg >= 0.05:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show threshold-dependent "
                        f"integration in the VECM framework. Price adjustment occurs primarily when differentials exceed "
                        f"the transaction cost threshold."
                    )
                elif above_adj_avg > 0.1:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show strong threshold "
                        f"integration in the VECM framework. Despite transaction cost barriers, price adjustment is rapid "
                        f"when differentials exceed the threshold."
                    )
                else:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show moderate integration "
                        f"with threshold effects in the VECM framework. Price adjustment occurs in both regimes but is "
                        f"constrained by transaction costs."
                    )
        
        elif self.model.mode == "mtar":
            if hasattr(self.model, 'mtar_results'):
                mtar_results = self.model.mtar_results
                
                if mtar_results['asymmetric']:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show asymmetric price "
                        f"transmission. The significant difference in adjustment speeds between rising and falling "
                        f"prices indicates market imperfections such as market power, information asymmetries, or "
                        f"policy interventions that create directional biases in price transmission."
                    )
                else:
                    return (
                        f"The markets {self.model.market1_name} and {self.model.market2_name} show symmetric price "
                        f"transmission. The absence of significant differences in adjustment speeds between rising "
                        f"and falling prices suggests relatively efficient market mechanisms without directional biases."
                    )
        
        # Default assessment if specific mode analysis not available
        return (
            f"The markets {self.model.market1_name} and {self.model.market2_name} are cointegrated, indicating "
            f"a long-run equilibrium relationship. However, the presence of a significant threshold suggests "
            f"that transaction costs create nonlinear dynamics in price transmission between these markets."
        )
    
    def _format_half_life(self, half_life: float) -> str:
        """Format half-life value for display."""
        if np.isnan(half_life):
            return "N/A"
        elif np.isinf(half_life):
            return "∞"
        else:
            return f"{half_life:.1f}"
            

class AcademicThresholdReporter(ThresholdReporter):
    """
    Academic threshold reporting with statistical rigor enhancements.
    
    This class extends the standard ThresholdReporter to provide enhanced
    statistical reporting suitable for academic publications, including:
    - Confidence intervals for all parameters
    - Formal hypothesis testing with significance indicators
    - Comprehensive diagnostic tests
    - Publication-quality tables and visualizations
    """
    
    def __init__(
        self, 
        model, 
        format: str = "markdown", 
        output_path: Optional[str] = None,
        confidence_level: float = 0.95,
        significance_indicators: bool = True,
        robust_standard_errors: bool = False,
        style: str = 'world_bank'
    ):
        """
        Initialize the academic reporter.
        
        Parameters
        ----------
        model : ThresholdModel
            Threshold model instance
        format : str, optional
            Report format ('markdown', 'json', or 'latex')
        output_path : str, optional
            Path to save the report
        confidence_level : float, optional
            Confidence level for intervals (0.90, 0.95, or 0.99)
        significance_indicators : bool, optional
            Whether to add significance indicators (*, **, ***)
        robust_standard_errors : bool, optional
            Whether to use robust standard errors (HC3)
        style : str, optional
            Visual style ('world_bank', 'academic', 'policy')
        """
        super().__init__(model, format, output_path)
        
        self.confidence_level = confidence_level
        self.significance_indicators = significance_indicators
        self.robust_standard_errors = robust_standard_errors
        self.style = style
        
        # Import necessary components from result_analysis
        try:
            from src.result_analysis.statistical_tests import (
                hypothesis_test, 
                threshold_significance_test,
                calculate_significance_indicators
            )
            from src.result_analysis.diagnostics import run_comprehensive_diagnostics
            
            self.hypothesis_test = hypothesis_test
            self.threshold_significance_test = threshold_significance_test
            self.calculate_significance_indicators = calculate_significance_indicators
            self.run_diagnostics = run_comprehensive_diagnostics
            self.has_enhanced_stats = True
        except ImportError:
            logger.warning("Could not import result_analysis modules. Using basic reporting capabilities.")
            self.has_enhanced_stats = False
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive academic report.
        
        Returns
        -------
        dict
            Report content and metadata
        """
        # Ensure model has results
        if self.model.threshold is None:
            logger.info("Running full analysis first")
            self.model.run_full_analysis()
        
        # Run statistical enhancements if available
        if self.has_enhanced_stats:
            # Calculate confidence intervals and significance
            self._enhance_parameter_statistics()
            
            # Run diagnostic tests
            self._run_model_diagnostics()
        
        # Generate tables, visualizations, and interpretation
        tables = self.generate_tables()
        visualizations = self.generate_visualizations()
        interpretation = self.generate_interpretation()
        
        # Add statistical test results
        if self.has_enhanced_stats:
            statistical_tests = self.generate_statistical_tests()
        else:
            statistical_tests = {}
        
        # Combine into report
        report = {
            'tables': tables,
            'visualizations': visualizations,
            'interpretation': interpretation,
            'statistical_tests': statistical_tests,
            'metadata': {
                'model_type': self.model.__class__.__name__,
                'model_mode': self.model.mode,
                'market1': self.model.market1_name,
                'market2': self.model.market2_name,
                'timestamp': datetime.now().isoformat(),
                'format': self.format,
                'confidence_level': self.confidence_level,
                'significance_indicators': self.significance_indicators,
                'robust_standard_errors': self.robust_standard_errors,
                'style': self.style
            }
        }
        
        # Export if path provided
        if self.output_path:
            self.export_report(report)
        
        # Close any open matplotlib figures
        plt.close('all')
        
        return report
    
    def _enhance_parameter_statistics(self) -> None:
        """
        Calculate enhanced parameter statistics.
        
        This method adds confidence intervals, p-values, and
        significance indicators for all model parameters.
        """
        if not hasattr(self.model, 'results'):
            logger.warning("Model results not available for statistical enhancement")
            return
        
        # Add statistics for threshold parameter
        if hasattr(self.model, 'threshold'):
            # For threshold, we need to use a specialized test
            threshold_test = self.threshold_significance_test(self.model)
            
            # Add results to model
            self.model.threshold_statistics = {
                'p_value': threshold_test.get('p_value', None),
                'significance': threshold_test.get('significance', ''),
                'reject_null': threshold_test.get('reject_null', None),
                'interpretation': threshold_test.get('interpretation', '')
            }
            
            # Calculate bootstrap confidence interval if available
            if 'bootstrap_results' in threshold_test and threshold_test['bootstrap_results']:
                bootstrap_stats = threshold_test['bootstrap_results'].get('bootstrap_stats', [])
                if len(bootstrap_stats) > 0:
                    # Calculate confidence interval from bootstrap distribution
                    alpha = 1 - self.confidence_level
                    ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
                    ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
                    
                    self.model.threshold_statistics['ci_lower'] = ci_lower
                    self.model.threshold_statistics['ci_upper'] = ci_upper
        
        # Add statistics for adjustment parameters
        if 'adjustment_below_1' in self.model.results and 'adjustment_above_1' in self.model.results:
            # Extract adjustment parameters
            adj_below = self.model.results['adjustment_below_1']
            adj_above = self.model.results['adjustment_above_1']
            
            # Get standard errors if available
            se_below = self.model.results.get('se_below_1', None)
            se_above = self.model.results.get('se_above_1', None)
            
            # Calculate hypothesis tests for adjustment parameters
            if se_below is not None:
                below_test = self.hypothesis_test(
                    theta=adj_below,
                    se=se_below,
                    null_value=0,
                    alternative='less'  # Adjustment should be negative
                )
                
                # Add results to model
                self.model.results['p_below_1'] = below_test['p_value']
                self.model.results['sig_below_1'] = below_test['significance']
                self.model.results['ci_below_1'] = below_test['confidence_interval']
            
            if se_above is not None:
                above_test = self.hypothesis_test(
                    theta=adj_above,
                    se=se_above,
                    null_value=0,
                    alternative='less'  # Adjustment should be negative
                )
                
                # Add results to model
                self.model.results['p_above_1'] = above_test['p_value']
                self.model.results['sig_above_1'] = above_test['significance']
                self.model.results['ci_above_1'] = above_test['confidence_interval']
            
            # Test for asymmetry (adjustment_above_1 ≠ adjustment_below_1)
            if se_below is not None and se_above is not None:
                # Calculate standard error of the difference using Delta method
                diff = adj_above - adj_below
                se_diff = np.sqrt(se_above**2 + se_below**2)
                
                asymmetry_test = self.hypothesis_test(
                    theta=diff,
                    se=se_diff,
                    null_value=0,
                    alternative='two-sided'
                )
                
                # Add results to model
                self.model.results['asymmetry_p_1'] = asymmetry_test['p_value']
                self.model.results['asymmetry_sig_1'] = asymmetry_test['significance']
                self.model.results['asymmetry_ci_1'] = asymmetry_test['confidence_interval']
    
    def _run_model_diagnostics(self) -> None:
        """
        Run comprehensive diagnostic tests on the model.
        """
        try:
            # Extract residuals model or create one if needed
            if hasattr(self.model, 'model'):
                # Model is already available
                diagnostics = self.run_diagnostics(self.model.model)
            elif hasattr(self.model, 'residuals') and hasattr(self.model, 'data1') and hasattr(self.model, 'data2'):
                # Create a simple model for diagnostics
                import statsmodels.api as sm
                
                # Use data1 as dependent and data2 as independent variable
                X = sm.add_constant(self.model.data2)
                model = sm.OLS(self.model.data1, X).fit()
                
                # Run diagnostics
                diagnostics = self.run_diagnostics(model)
            else:
                logger.warning("Could not create model for diagnostics")
                return
            
            # Add diagnostics to model
            self.model.diagnostics = diagnostics
            
        except Exception as e:
            logger.error(f"Error running diagnostics: {str(e)}")
    
    def generate_statistical_tests(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical test results.
        
        Returns
        -------
        dict
            Dictionary containing statistical test results
        """
        tests = {}
        
        # Add threshold significance test results
        if hasattr(self.model, 'threshold_statistics'):
            tests['threshold_significance'] = {
                'test_name': "Bootstrap threshold significance test",
                'p_value': self.model.threshold_statistics.get('p_value'),
                'significance': self.model.threshold_statistics.get('significance'),
                'reject_null': self.model.threshold_statistics.get('reject_null'),
                'confidence_interval': (
                    self.model.threshold_statistics.get('ci_lower'),
                    self.model.threshold_statistics.get('ci_upper')
                ),
                'interpretation': self.model.threshold_statistics.get('interpretation')
            }
        
        # Add adjustment parameter test results
        if hasattr(self.model, 'results'):
            tests['adjustment_parameters'] = {
                'below_threshold': {
                    'test_name': "Adjustment below threshold significance test",
                    'parameter': self.model.results.get('adjustment_below_1'),
                    'std_error': self.model.results.get('se_below_1'),
                    'p_value': self.model.results.get('p_below_1'),
                    'significance': self.model.results.get('sig_below_1'),
                    'confidence_interval': self.model.results.get('ci_below_1'),
                    'interpretation': (
                        "Adjustment below threshold is " + 
                        ("statistically significant" 
                         if self.model.results.get('p_below_1', 1) < 0.05 else 
                         "not statistically significant")
                    )
                },
                'above_threshold': {
                    'test_name': "Adjustment above threshold significance test",
                    'parameter': self.model.results.get('adjustment_above_1'),
                    'std_error': self.model.results.get('se_above_1'),
                    'p_value': self.model.results.get('p_above_1'),
                    'significance': self.model.results.get('sig_above_1'),
                    'confidence_interval': self.model.results.get('ci_above_1'),
                    'interpretation': (
                        "Adjustment above threshold is " + 
                        ("statistically significant" 
                         if self.model.results.get('p_above_1', 1) < 0.05 else 
                         "not statistically significant")
                    )
                },
                'asymmetry': {
                    'test_name': "Asymmetry test (above ≠ below)",
                    'parameter': (
                        self.model.results.get('adjustment_above_1', 0) - 
                        self.model.results.get('adjustment_below_1', 0)
                    ),
                    'p_value': self.model.results.get('asymmetry_p_1'),
                    'significance': self.model.results.get('asymmetry_sig_1'),
                    'confidence_interval': self.model.results.get('asymmetry_ci_1'),
                    'interpretation': (
                        "Asymmetry in adjustment speeds is " + 
                        ("statistically significant" 
                         if self.model.results.get('asymmetry_p_1', 1) < 0.05 else 
                         "not statistically significant")
                    )
                }
            }
        
        # Add diagnostic test results if available
        if hasattr(self.model, 'diagnostics'):
            tests['diagnostics'] = self.model.diagnostics
        
        return tests
    
    def _export_latex(self, report: Dict[str, Any]) -> None:
        """
        Export report in enhanced LaTeX format with academic standards.
        """
        # Try to import LaTeX formatter
        try:
            from src.result_analysis.academic_formatting import AcademicTableFormatter
            formatter = AcademicTableFormatter(
                journal_style=self.style,
                include_significance=self.significance_indicators,
                confidence_level=self.confidence_level
            )
            has_formatter = True
        except ImportError:
            logger.warning("AcademicTableFormatter not available, using basic LaTeX formatting")
            has_formatter = False
        
        # Start building the LaTeX report
        latex = []
        
        # Document preamble
        latex.extend([
            "\\documentclass{article}",
            "\\usepackage{booktabs}",
            "\\usepackage{caption}",
            "\\usepackage{graphicx}",
            "\\usepackage{amsmath}",
            "\\usepackage{geometry}",
            "\\usepackage{float}",
            "\\usepackage{siunitx}",
            "\\usepackage{xcolor}",
            "\\usepackage{hyperref}",
            "",
            "\\geometry{margin=1in}",
            "\\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}",
            "",
            f"\\title{{Threshold Model Analysis: {self.model.market1_name} and {self.model.market2_name}}}",
            f"\\author{{Yemen Market Integration Project}}",
            "\\date{\\today}",
            "",
            "\\begin{document}",
            "",
            "\\maketitle",
            ""
        ])
        
        # Add executive summary
        latex.extend([
            "\\section{Executive Summary}",
            "",
            "This report presents a threshold cointegration analysis of market integration between " +
            f"{self.model.market1_name} and {self.model.market2_name} using a {self.model.mode} threshold model.",
            ""
        ])
        
        # Add interpretation if available
        if 'interpretation' in report and 'market_integration' in report['interpretation']:
            latex.append(report['interpretation']['market_integration'])
            latex.append("")
        
        # Add model specifications
        latex.extend([
            "\\section{Model Specification}",
            "",
            "\\begin{align}",
            "\\Delta P_{1,t} &= \\begin{cases}",
            f"\\alpha_1^- \\hat{{\\varepsilon}}_{{{self.model.market1_name}, {self.model.market2_name}, t-1}} + \\sum_{{i=1}}^p \\delta_{{1i}} \\Delta P_{{{self.model.market1_name}, t-i}} + \\sum_{{j=1}}^p \\delta_{{2j}} \\Delta P_{{{self.model.market2_name}, t-j}} + u_{{1t}} & \\text{{if }} \\hat{{\\varepsilon}}_{{{self.model.market1_name}, {self.model.market2_name}, t-1}} \\leq \\tau \\\\",
            f"\\alpha_1^+ \\hat{{\\varepsilon}}_{{{self.model.market1_name}, {self.model.market2_name}, t-1}} + \\sum_{{i=1}}^p \\delta_{{1i}} \\Delta P_{{{self.model.market1_name}, t-i}} + \\sum_{{j=1}}^p \\delta_{{2j}} \\Delta P_{{{self.model.market2_name}, t-j}} + u_{{1t}} & \\text{{if }} \\hat{{\\varepsilon}}_{{{self.model.market1_name}, {self.model.market2_name}, t-1}} > \\tau \\\\",
            "\\end{cases}",
            "\\end{align}",
            "",
            "where:",
            "\\begin{itemize}",
            f"\\item $P_{{{self.model.market1_name}, t}}$ and $P_{{{self.model.market2_name}, t}}$ are prices in {self.model.market1_name} and {self.model.market2_name} at time $t$",
            f"\\item $\\hat{{\\varepsilon}}_{{{self.model.market1_name}, {self.model.market2_name}, t-1}}$ is the equilibrium error term",
            f"\\item $\\tau$ is the threshold parameter",
            f"\\item $\\alpha_1^-$ and $\\alpha_1^+$ are the adjustment parameters below and above the threshold",
            "\\end{itemize}",
            ""
        ])
        
        # Add threshold results table
        if has_formatter:
            # Extract threshold and adjustment results
            threshold = self.model.threshold
            threshold_p = getattr(self.model, 'threshold_statistics', {}).get('p_value', None)
            threshold_ci_lower = getattr(self.model, 'threshold_statistics', {}).get('ci_lower', None)
            threshold_ci_upper = getattr(self.model, 'threshold_statistics', {}).get('ci_upper', None)
            
            adjustment_below = self.model.results.get('adjustment_below_1', None)
            adjustment_above = self.model.results.get('adjustment_above_1', None)
            
            se_below = self.model.results.get('se_below_1', None)
            se_above = self.model.results.get('se_above_1', None)
            
            p_below = self.model.results.get('p_below_1', None)
            p_above = self.model.results.get('p_above_1', None)
            
            # Create table data
            threshold_data = {
                'threshold': threshold,
                'threshold_p_value': threshold_p,
                'threshold_ci_lower': threshold_ci_lower,
                'threshold_ci_upper': threshold_ci_upper,
                'adjustment_below': adjustment_below,
                'adjustment_above': adjustment_above,
                'se_below': se_below,
                'se_above': se_above,
                'p_value_below': p_below,
                'p_value_above': p_above
            }
            
            # Generate table using formatter
            threshold_table = formatter.format_threshold_table_latex(
                threshold_data,
                f"Threshold Model Results for {self.model.market1_name}-{self.model.market2_name}"
            )
            
            latex.append(threshold_table)
            latex.append("")
            
        else:
            # Basic threshold results without formatter
            latex.extend([
                "\\section{Threshold Results}",
                "",
                "\\begin{table}[H]",
                "\\centering",
                f"\\caption{{Threshold Model Results for {self.model.market1_name}-{self.model.market2_name}}}",
                "\\begin{tabular}{lcc}",
                "\\toprule",
                "Parameter & Below Threshold & Above Threshold \\\\",
                "\\midrule",
                f"Adjustment Speed & {self.model.results.get('adjustment_below_1', 'N/A')} & {self.model.results.get('adjustment_above_1', 'N/A')} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                ""
            ])
        
        # Add statistical test results if available
        if 'statistical_tests' in report and report['statistical_tests']:
            latex.extend([
                "\\section{Hypothesis Tests}",
                ""
            ])
            
            # Add threshold significance test
            if 'threshold_significance' in report['statistical_tests']:
                threshold_test = report['statistical_tests']['threshold_significance']
                
                latex.extend([
                    "\\subsection{Threshold Significance Test}",
                    "",
                    "\\begin{table}[H]",
                    "\\centering",
                    "\\caption{Threshold Significance Test Results}",
                    "\\begin{tabular}{lr}",
                    "\\toprule",
                    "Parameter & Value \\\\",
                    "\\midrule",
                    f"Threshold & {self.model.threshold:.4f} \\\\",
                    f"p-value & {threshold_test.get('p_value', 'N/A')} \\\\",
                ])
                
                # Add confidence interval if available
                if 'confidence_interval' in threshold_test and threshold_test['confidence_interval'][0] is not None:
                    ci_lower, ci_upper = threshold_test['confidence_interval']
                    latex.append(f"{self.confidence_level*100:.0f}\\% Confidence Interval & [{ci_lower:.4f}, {ci_upper:.4f}] \\\\")
                
                latex.extend([
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\end{table}",
                    "",
                    f"Interpretation: {threshold_test.get('interpretation', 'N/A')}",
                    ""
                ])
            
            # Add adjustment parameter tests
            if 'adjustment_parameters' in report['statistical_tests']:
                adjustment_tests = report['statistical_tests']['adjustment_parameters']
                
                latex.extend([
                    "\\subsection{Adjustment Parameter Tests}",
                    "",
                    "\\begin{table}[H]",
                    "\\centering",
                    "\\caption{Adjustment Parameter Significance Tests}",
                    "\\begin{tabular}{lrrrr}",
                    "\\toprule",
                    "Parameter & Estimate & Std. Error & p-value & Significance \\\\",
                    "\\midrule"
                ])
                
                # Below threshold
                below = adjustment_tests['below_threshold']
                param_below = below.get('parameter', 'N/A')
                se_below = below.get('std_error', 'N/A')
                p_below = below.get('p_value', 'N/A')
                sig_below = below.get('significance', '')
                
                latex.append(f"Below Threshold & {param_below:.4f} & {se_below:.4f} & {p_below:.4f} & {sig_below} \\\\")
                
                # Above threshold
                above = adjustment_tests['above_threshold']
                param_above = above.get('parameter', 'N/A')
                se_above = above.get('std_error', 'N/A')
                p_above = above.get('p_value', 'N/A')
                sig_above = above.get('significance', '')
                
                latex.append(f"Above Threshold & {param_above:.4f} & {se_above:.4f} & {p_above:.4f} & {sig_above} \\\\")
                
                # Asymmetry
                asymm = adjustment_tests['asymmetry']
                param_asymm = asymm.get('parameter', 'N/A')
                p_asymm = asymm.get('p_value', 'N/A')
                sig_asymm = asymm.get('significance', '')
                
                latex.append(f"Asymmetry (Above - Below) & {param_asymm:.4f} & -- & {p_asymm:.4f} & {sig_asymm} \\\\")
                
                latex.extend([
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\end{table}",
                    "",
                    "Interpretation:",
                    "\\begin{itemize}",
                    f"\\item Below Threshold: {below.get('interpretation', 'N/A')}",
                    f"\\item Above Threshold: {above.get('interpretation', 'N/A')}",
                    f"\\item Asymmetry: {asymm.get('interpretation', 'N/A')}",
                    "\\end{itemize}",
                    ""
                ])
        
        # Add diagnostic test results if available
        if 'statistical_tests' in report and 'diagnostics' in report['statistical_tests']:
            diagnostics = report['statistical_tests']['diagnostics']
            
            latex.extend([
                "\\section{Diagnostic Tests}",
                ""
            ])
            
            # Add heteroskedasticity test results
            if 'heteroskedasticity' in diagnostics:
                het_tests = diagnostics['heteroskedasticity']
                
                latex.extend([
                    "\\subsection{Heteroskedasticity Tests}",
                    "",
                    "\\begin{table}[H]",
                    "\\centering",
                    "\\caption{Heteroskedasticity Test Results}",
                    "\\begin{tabular}{lrrl}",
                    "\\toprule",
                    "Test & Statistic & p-value & Result \\\\",
                    "\\midrule"
                ])
                
                # White test
                if 'white' in het_tests and isinstance(het_tests['white'], dict):
                    white = het_tests['white']
                    
                    test_stat = white.get('test_statistic', 'N/A')
                    p_value = white.get('p_value', 'N/A')
                    reject = white.get('reject_null', False)
                    
                    test_stat_str = f"{test_stat:.4f}" if isinstance(test_stat, (int, float)) else "N/A"
                    p_value_str = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "N/A"
                    
                    result = "Heteroskedasticity present" if reject else "Homoskedasticity"
                    
                    latex.append(f"White & {test_stat_str} & {p_value_str} & {result} \\\\")
                
                # Close the table
                latex.extend([
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\end{table}",
                    ""
                ])
            
            # Add normality test results
            if 'normality' in diagnostics:
                normality = diagnostics['normality']
                
                latex.extend([
                    "\\subsection{Normality Tests}",
                    "",
                    "\\begin{table}[H]",
                    "\\centering",
                    "\\caption{Normality Test Results}",
                    "\\begin{tabular}{lrrl}",
                    "\\toprule",
                    "Test & Statistic & p-value & Result \\\\",
                    "\\midrule"
                ])
                
                # Jarque-Bera test
                if 'jarque_bera' in normality and isinstance(normality['jarque_bera'], dict):
                    jb = normality['jarque_bera']
                    
                    test_stat = jb.get('test_statistic', 'N/A')
                    p_value = jb.get('p_value', 'N/A')
                    reject = jb.get('reject_null', False)
                    
                    test_stat_str = f"{test_stat:.4f}" if isinstance(test_stat, (int, float)) else "N/A"
                    p_value_str = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "N/A"
                    
                    result = "Non-normal residuals" if reject else "Normal residuals"
                    
                    latex.append(f"Jarque-Bera & {test_stat_str} & {p_value_str} & {result} \\\\")
                
                # Close the table
                latex.extend([
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\end{table}",
                    ""
                ])
            
            # Add summary of diagnostic issues
            if 'summary' in diagnostics:
                summary = diagnostics['summary']
                
                latex.extend([
                    "\\subsection{Diagnostic Summary}",
                    ""
                ])
                
                if 'interpretation' in summary:
                    latex.append(summary['interpretation'])
                    latex.append("")
                
                if 'has_issues' in summary and summary['has_issues']:
                    if 'issues_detected' in summary:
                        latex.append("Issues Detected:")
                        latex.append("\\begin{itemize}")
                        
                        for issue in summary['issues_detected']:
                            latex.append(f"\\item {issue}")
                        
                        latex.append("\\end{itemize}")
                        latex.append("")
                    
                    if 'recommendations' in summary:
                        latex.append("Recommendations:")
                        latex.append("\\begin{itemize}")
                        
                        for rec in summary['recommendations']:
                            latex.append(f"\\item {rec}")
                        
                        latex.append("\\end{itemize}")
                        latex.append("")
        
        # Add significance legend if indicators are used
        if self.significance_indicators:
            latex.extend([
                "\\section*{Notes}",
                "\\begin{itemize}",
                "\\item[*] $p<0.1$",
                "\\item[**] $p<0.05$",
                "\\item[***] $p<0.01$",
                "\\end{itemize}",
                ""
            ])
        
        # Close document
        latex.append("\\end{document}")
        
        # Write to file
        with open(self.output_path, 'w') as f:
            f.write('\n'.join(latex))
        
        logger.info(f"Exported enhanced LaTeX report to {self.output_path}")