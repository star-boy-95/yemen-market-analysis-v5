"""
Plot styling and management for Yemen Market Analysis.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from core.decorators import error_handler, performance_tracker
from core.exceptions import VisualizationError

logger = logging.getLogger(__name__)


class PlotManager:
    """Manager for consistent plot styling and configuration."""
    
    def __init__(
        self,
        style: str = 'seaborn-v0_8-whitegrid',
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
        font_scale: float = 1.0,
        output_dir: Optional[str] = None
    ):
        """Initialize the plot manager with styling options."""
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.font_scale = font_scale
        self.output_dir = output_dir
        
        # Set default styling
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Set up plot styling."""
        try:
            # Set style
            plt.style.use(self.style)
            
            # Set font sizes
            plt.rcParams['font.size'] = 10 * self.font_scale
            plt.rcParams['axes.titlesize'] = 12 * self.font_scale
            plt.rcParams['axes.labelsize'] = 10 * self.font_scale
            plt.rcParams['xtick.labelsize'] = 9 * self.font_scale
            plt.rcParams['ytick.labelsize'] = 9 * self.font_scale
            plt.rcParams['legend.fontsize'] = 9 * self.font_scale
            
            # Set figure parameters
            plt.rcParams['figure.figsize'] = self.figsize
            plt.rcParams['figure.dpi'] = self.dpi
            
            # Set line widths
            plt.rcParams['lines.linewidth'] = 1.5
            plt.rcParams['axes.linewidth'] = 1.0
            plt.rcParams['grid.linewidth'] = 0.8
            
            # Set colors
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ])
            
            # Ensure Directory exists
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
        
        except Exception as e:
            logger.error(f"Error setting up plot style: {str(e)}")
    
    @error_handler(fallback_value=(None, None))
    def create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[int, int]] = None,
        constrained_layout: bool = True
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create a new figure with consistent styling.
        
        Args:
            nrows: Number of rows in subplot grid
            ncols: Number of columns in subplot grid
            figsize: Optional figure size (defaults to class setting)
            constrained_layout: Whether to use constrained layout
            
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize or self.figsize,
            dpi=self.dpi,
            constrained_layout=constrained_layout
        )
        
        return fig, axes
    
    @error_handler(fallback_value=None)
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        subdirectory: Optional[str] = None,
        formats: List[str] = None,
        dpi: Optional[int] = None,
        transparent: bool = False
    ) -> str:
        """
        Save figure to file with consistent formatting.
        
        Args:
            fig: Figure to save
            filename: Base filename (without extension)
            subdirectory: Optional subdirectory within output directory
            formats: List of formats to save (e.g., ['png', 'pdf'])
            dpi: Optional override for DPI
            transparent: Whether to use transparent background
            
        Returns:
            Path to saved file
        """
        if not self.output_dir:
            raise VisualizationError("No output directory specified for saving figures")
        
        # Set default formats if not provided
        if formats is None:
            formats = ['png']
        
        # Create target directory
        target_dir = self.output_dir
        if subdirectory:
            target_dir = os.path.join(target_dir, subdirectory)
            os.makedirs(target_dir, exist_ok=True)
        
        # Ensure filename doesn't have an extension
        base_filename = os.path.splitext(filename)[0]
        
        # Save in each format
        paths = []
        for fmt in formats:
            output_path = os.path.join(target_dir, f"{base_filename}.{fmt}")
            fig.savefig(
                output_path,
                dpi=dpi or self.dpi,
                bbox_inches='tight',
                transparent=transparent
            )
            paths.append(output_path)
            logger.debug(f"Saved figure to {output_path}")
        
        return paths[0]  # Return path to the first format
    
    @staticmethod
    @error_handler(fallback_value=None)
    def set_axis_style(
        ax: plt.Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[Tuple[Any, Any]] = None,
        ylim: Optional[Tuple[Any, Any]] = None,
        grid: bool = True,
        legend: bool = False,
        legend_loc: str = 'best'
    ) -> plt.Axes:
        """
        Apply consistent styling to an axis.
        
        Args:
            ax: Axes to style
            title: Title text
            xlabel: X-axis label
            ylabel: Y-axis label
            xlim: X-axis limits
            ylim: Y-axis limits
            grid: Whether to show grid
            legend: Whether to show legend
            legend_loc: Legend location
            
        Returns:
            Styled axes
        """
        if title:
            ax.set_title(title)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_ylabel(ylabel)
        
        if xlim:
            ax.set_xlim(xlim)
        
        if ylim:
            ax.set_ylim(ylim)
        
        ax.grid(grid, linestyle='--', alpha=0.7)
        
        if legend:
            ax.legend(loc=legend_loc, frameon=True, framealpha=0.8)
        
        return ax


# Global plot manager instance
plot_manager = PlotManager()


def set_plot_manager(
    style: str = 'seaborn-v0_8-whitegrid',
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    font_scale: float = 1.0,
    output_dir: Optional[str] = None
) -> PlotManager:
    """Set the global plot manager with new settings."""
    global plot_manager
    plot_manager = PlotManager(
        style=style,
        figsize=figsize,
        dpi=dpi,
        font_scale=font_scale,
        output_dir=output_dir
    )
    return plot_manager


def get_plot_manager() -> PlotManager:
    """Get the global plot manager instance."""
    return plot_manager