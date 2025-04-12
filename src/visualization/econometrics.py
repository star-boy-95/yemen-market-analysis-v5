"""
Econometrics visualization module for Yemen Market Analysis.

This module provides the EconometricsPlotter class for creating plots of econometric results.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

class EconometricsPlotter:
    """
    Econometrics plotter for Yemen Market Analysis.
    
    This class provides methods for creating plots of econometric results.
    
    Attributes:
        figsize (Tuple[int, int]): Figure size.
        dpi (int): Figure DPI.
        style (str): Plot style.
    """
    
    def __init__(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        dpi: Optional[int] = None,
        style: Optional[str] = None
    ):
        """
        Initialize the econometrics plotter.
        
        Args:
            figsize: Figure size.
            dpi: Figure DPI.
            style: Plot style.
        """
        self.figsize = figsize or config.get('visualization.figure_size', (10, 6))
        self.dpi = dpi or config.get('visualization.dpi', 300)
        self.style = style or config.get('visualization.style', 'seaborn-v0_8-whitegrid')
        
        # Set plot style
        plt.style.use(self.style)
    
    @handle_errors
    def plot_cointegration_results(
        self, results: Dict[str, Any], title: Optional[str] = None
    ) -> Figure:
        """
        Plot cointegration test results.
        
        Args:
            results: Cointegration test results.
            title: Title for the plot.
            
        Returns:
            Matplotlib figure with the cointegration results plot.
        """
        logger.info("Creating cointegration results plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract data
        p_values = results.get('p_values', [])
        test_stats = results.get('test_statistics', [])
        method = results.get('method', 'Unknown')
        
        # Create the plot
        ax.bar(range(len(test_stats)), test_stats, color='blue', alpha=0.7)
        
        # Add critical values if available
        critical_values = results.get('critical_values', {})
        if critical_values:
            for level, value in critical_values.items():
                ax.axhline(y=value, linestyle='--', color='red', 
                         label=f'Critical Value ({level})')
        
        # Set labels and title
        ax.set_xlabel('Test')
        ax.set_ylabel('Test Statistic')
        ax.set_title(title or f'Cointegration Test Results ({method})')
        
        # Add legend if we have critical values
        if critical_values:
            ax.legend()
        
        # Add p-values as text annotations
        for i, p in enumerate(p_values):
            ax.annotate(f'p={p:.3f}', (i, test_stats[i]), 
                       xytext=(0, 5), textcoords='offset points',
                       ha='center')
        
        return fig
    
    @handle_errors
    def plot_threshold_model(
        self, results: Dict[str, Any], title: Optional[str] = None
    ) -> Figure:
        """
        Plot threshold model results.
        
        Args:
            results: Threshold model results.
            title: Title for the plot.
            
        Returns:
            Matplotlib figure with the threshold model plot.
        """
        logger.info("Creating threshold model plot")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Extract data
        threshold = results.get('threshold', 0)
        regime1_coef = results.get('regime1_coefficients', {})
        regime2_coef = results.get('regime2_coefficients', {})
        
        # Plot coefficients for each regime
        params = list(set(list(regime1_coef.keys()) + list(regime2_coef.keys())))
        params = [p for p in params if p != 'const']  # Exclude constant
        
        x = np.arange(len(params))
        width = 0.35
        
        # Regime 1 bars
        r1_values = [regime1_coef.get(p, 0) for p in params]
        ax1.bar(x - width/2, r1_values, width, label='Regime 1', color='blue', alpha=0.7)
        
        # Regime 2 bars
        r2_values = [regime2_coef.get(p, 0) for p in params]
        ax1.bar(x + width/2, r2_values, width, label='Regime 2', color='green', alpha=0.7)
        
        # Set labels and title for coefficients plot
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Coefficient Value')
        ax1.set_title('Regime Coefficients')
        ax1.set_xticks(x)
        ax1.set_xticklabels(params)
        ax1.legend()
        
        # Plot adjustment speeds if available
        adjustment = results.get('adjustment_speeds', {})
        if adjustment:
            adj_labels = list(adjustment.keys())
            adj_values = list(adjustment.values())
            
            ax2.bar(range(len(adj_values)), adj_values, color='red', alpha=0.7)
            ax2.set_xlabel('Adjustment Type')
            ax2.set_ylabel('Adjustment Speed')
            ax2.set_title(f'Adjustment Speeds (Threshold = {threshold:.3f})')
            ax2.set_xticks(range(len(adj_labels)))
            ax2.set_xticklabels(adj_labels)
        
        # Set overall title
        fig.suptitle(title or 'Threshold Model Results')
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        
        return fig
    
    @handle_errors
    def save_plot(self, fig: Figure, output_path: str) -> None:
        """
        Save a plot to a file.
        
        Args:
            fig: Matplotlib figure to save.
            output_path: Path to save the figure to.
        """
        logger.info(f"Saving plot to {output_path}")
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
