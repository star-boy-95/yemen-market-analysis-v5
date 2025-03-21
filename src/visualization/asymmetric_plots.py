"""
Visualization module for asymmetric adjustment patterns in threshold cointegration models.

This module provides specialized visualizations for asymmetric price adjustment
patterns and regime-switching dynamics in threshold cointegration models.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List, Dict, Any, Union
import datetime

from yemen_market_integration.utils import (
    handle_errors,
    config,
    validate_dataframe,
    raise_if_invalid,
    set_plotting_style,
    format_date_axis,
    save_plot,
    VisualizationError
)

logger = logging.getLogger(__name__)

class AsymmetricAdjustmentVisualizer:
    """Specialized visualizations for asymmetric price adjustment patterns."""
    
    def __init__(self):
        """Initialize the visualizer with default styling."""
        set_plotting_style()
        
        # Get styling parameters from config
        self.fig_width = config.get('visualization.default_fig_width', 12)
        self.fig_height = config.get('visualization.default_fig_height', 8)
        self.dpi = config.get('visualization.figure_dpi', 300)
        self.date_format = config.get('visualization.date_format', '%Y-%m')
        self.save_dir = config.get('visualization.save_dir', 'results/plots')
        
        # Colors for different regimes
        self.above_color = config.get('visualization.above_threshold_color', '#d73027')
        self.below_color = config.get('visualization.below_threshold_color', '#4575b4')
        self.middle_color = config.get('visualization.middle_regime_color', '#ffffbf')
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError), reraise=True)
    def plot_regime_dynamics(
        self,
        price_diff: np.ndarray,
        dates: Union[np.ndarray, List],
        threshold: float,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize price differential series with regime highlighting.
        
        Parameters
        ----------
        price_diff : np.ndarray
            Time series of price differentials
        dates : np.ndarray or list
            Corresponding dates for price differentials
        threshold : float
            Threshold value for regime separation
        title : str, optional
            Plot title, default None
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        ax : plt.Axes
            The matplotlib axes
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot price differential
        ax.plot(dates, price_diff, color='k', linewidth=1.5, label='Price Differential')
        
        # Add threshold lines
        ax.axhline(y=threshold, color=self.above_color, linestyle='--', 
                  linewidth=1.5, label=f'Threshold: {threshold:.2f}')
        ax.axhline(y=-threshold, color=self.below_color, linestyle='--', 
                  linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Identify regimes
        above_thresh = price_diff > threshold
        below_thresh = price_diff < -threshold
        inside_band = ~(above_thresh | below_thresh)
        
        # Shade regimes with transparency
        for i in range(len(dates) - 1):
            start_date = dates[i]
            end_date = dates[i+1]
            
            if above_thresh[i]:
                ax.axvspan(start_date, end_date, color=self.above_color, alpha=0.2)
            elif below_thresh[i]:
                ax.axvspan(start_date, end_date, color=self.below_color, alpha=0.2)
            else:
                ax.axvspan(start_date, end_date, color=self.middle_color, alpha=0.2)
        
        # Calculate regime proportions for legend
        n_above = np.sum(above_thresh)
        n_below = np.sum(below_thresh)
        n_inside = np.sum(inside_band)
        total = len(price_diff)
        
        above_pct = n_above / total * 100
        below_pct = n_below / total * 100
        inside_pct = n_inside / total * 100
        
        # Create custom legend
        custom_lines = [
            mpatches.Patch(color=self.above_color, alpha=0.4),
            mpatches.Patch(color=self.middle_color, alpha=0.4),
            mpatches.Patch(color=self.below_color, alpha=0.4)
        ]
        
        ax.legend(
            custom_lines, 
            [f'Above Threshold ({above_pct:.1f}%)',
             f'Inside Band ({inside_pct:.1f}%)',
             f'Below Threshold ({below_pct:.1f}%)'],
            loc='upper right'
        )
        
        # Add labels and title
        ax.set_ylabel('Price Differential')
        ax.set_xlabel('Date')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Price Differential Regimes')
        
        # Format date axis
        format_date_axis(ax, date_format=self.date_format)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
        
        return fig, ax
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError), reraise=True)
    def plot_asymmetric_adjustment(
        self,
        threshold_model: Any,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Visualize asymmetric adjustment patterns from a threshold model.
        
        Parameters
        ----------
        threshold_model : Any
            Threshold model object with asymmetric adjustment results
        title : str, optional
            Plot title, default None
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        axs : List[plt.Axes]
            List of matplotlib axes
        """
        # Create figure with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(self.fig_width, self.fig_height * 1.5),
                               gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Extract data from threshold model
        price_diff = threshold_model.price_diff
        dates = threshold_model.dates
        threshold = threshold_model.threshold
        
        # Get adjustment parameters and half-lives
        adj_below = getattr(threshold_model, 'adjustment_below', None)
        adj_middle = getattr(threshold_model, 'adjustment_middle', None)
        adj_above = getattr(threshold_model, 'adjustment_above', None)
        
        # Look for different attribute patterns depending on the model implementation
        if adj_below is None:
            # Try alternative attribute names
            adj_below = getattr(threshold_model, 'adjustment_below_1', None)
            adj_above = getattr(threshold_model, 'adjustment_above_1', None)
            
        # Get half-lives if available
        half_life_below = getattr(threshold_model, 'half_life_below', None)
        half_life_above = getattr(threshold_model, 'half_life_above', None)
        
        if half_life_below is None:
            # Try alternative attribute patterns
            half_life_below = getattr(threshold_model, 'half_life_below_1', None)
            half_life_above = getattr(threshold_model, 'half_life_above_1', None)
            
            # If still not found, calculate from adjustment speeds
            if half_life_below is None and adj_below is not None:
                if adj_below != 0:
                    half_life_below = np.log(2) / abs(adj_below)
                    half_life_above = np.log(2) / abs(adj_above) if adj_above != 0 else np.inf
        
        # Top plot: Price differential with regime highlighting
        axs[0].plot(dates, price_diff, color='k', linewidth=1.5, label='Price Differential')
        axs[0].axhline(y=threshold, color=self.above_color, linestyle='--', linewidth=1.5)
        axs[0].axhline(y=-threshold, color=self.below_color, linestyle='--', linewidth=1.5)
        axs[0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Shade regimes
        above_thresh = price_diff > threshold
        below_thresh = price_diff < -threshold
        
        for i in range(len(dates) - 1):
            start_date = dates[i]
            end_date = dates[i+1]
            
            if above_thresh[i]:
                axs[0].axvspan(start_date, end_date, color=self.above_color, alpha=0.2)
            elif below_thresh[i]:
                axs[0].axvspan(start_date, end_date, color=self.below_color, alpha=0.2)
            else:
                axs[0].axvspan(start_date, end_date, color=self.middle_color, alpha=0.2)
        
        axs[0].set_ylabel('Price Differential')
        axs[0].set_title('Asymmetric Price Adjustment Analysis')
        format_date_axis(axs[0], date_format=self.date_format)
        axs[0].grid(True, alpha=0.3)
        
        # Middle plot: Adjustment speeds by regime
        labels = ['Below Threshold', 'Inside Band', 'Above Threshold']
        colors = [self.below_color, self.middle_color, self.above_color]
        
        if None not in [adj_below, adj_middle, adj_above]:
            adjustment_speeds = [abs(adj) if adj is not None else 0 for adj in [adj_below, adj_middle, adj_above]]
            
            bars = axs[1].bar(labels, adjustment_speeds, color=colors, alpha=0.7)
            
            # Add text labels
            for bar, speed in zip(bars, adjustment_speeds):
                height = bar.get_height()
                axs[1].text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f'{speed:.3f}',
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontsize=9
                )
            
            axs[1].set_ylabel('Adjustment Speed')
            axs[1].grid(True, alpha=0.3, axis='y')
        else:
            axs[1].text(0.5, 0.5, 'Adjustment speeds not available', 
                      ha='center', va='center', transform=axs[1].transAxes)
        
        # Bottom plot: Half-lives by regime
        if None not in [half_life_below, half_life_above]:
            half_lives = [
                half_life_below if half_life_below is not None else 0,
                np.nan,  # No half-life for middle regime
                half_life_above if half_life_above is not None else 0
            ]
            
            # Filter out middle regime for half-life plot
            regime_labels = [labels[0], labels[2]]
            regime_colors = [colors[0], colors[2]]
            regime_half_lives = [hl for hl in [half_lives[0], half_lives[2]] if not np.isnan(hl)]
            
            bars = axs[2].bar(regime_labels, regime_half_lives, color=regime_colors, alpha=0.7)
            
            # Add text labels
            for bar, hl in zip(bars, regime_half_lives):
                height = bar.get_height()
                axs[2].text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.5,
                    f'{hl:.1f}',
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontsize=9
                )
            
            # Calculate asymmetry measure
            if half_life_below != 0 and half_life_above != 0:
                asymmetry_ratio = half_life_below / half_life_above
                asymmetry_text = f'Asymmetry Ratio (Below/Above): {asymmetry_ratio:.2f}'
                
                # Add asymmetry text
                axs[2].text(
                    0.5, 0.9, 
                    asymmetry_text,
                    transform=axs[2].transAxes,
                    ha='center',
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
                )
            
            axs[2].set_ylabel('Half-Life (periods)')
            axs[2].grid(True, alpha=0.3, axis='y')
        else:
            axs[2].text(0.5, 0.5, 'Half-lives not available', 
                      ha='center', va='center', transform=axs[2].transAxes)
        
        # Set overall title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        
        # Adjust spacing
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
        
        return fig, axs
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError), reraise=True)
    def plot_regime_transitions(
        self,
        price_diff: np.ndarray,
        dates: Union[np.ndarray, List],
        threshold: float,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize regime transitions over time.
        
        Parameters
        ----------
        price_diff : np.ndarray
            Time series of price differentials
        dates : np.ndarray or list
            Corresponding dates for price differentials
        threshold : float
            Threshold value for regime separation
        title : str, optional
            Plot title, default None
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        axs : list of plt.Axes
            The matplotlib axes
        """
        # Create figure with two subplots
        fig, axs = plt.subplots(2, 1, figsize=(self.fig_width, self.fig_height),
                               gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Top plot: Price differential
        axs[0].plot(dates, price_diff, color='k', linewidth=1.5, label='Price Differential')
        axs[0].axhline(y=threshold, color=self.above_color, linestyle='--', 
                      linewidth=1.5, label=f'Threshold: {threshold:.2f}')
        axs[0].axhline(y=-threshold, color=self.below_color, linestyle='--', 
                      linewidth=1.5)
        axs[0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Identify regimes (1: above, 0: inside, -1: below)
        regimes = np.zeros_like(price_diff, dtype=int)
        regimes[price_diff > threshold] = 1
        regimes[price_diff < -threshold] = -1
        
        # Bottom plot: Regime indicator
        # Create custom colormap for regimes
        cmap = LinearSegmentedColormap.from_list(
            'regime_cmap', 
            [(0, self.below_color), (0.5, self.middle_color), (1, self.above_color)],
            N=3
        )
        
        # Plot regime as colored steps
        for i in range(len(dates)-1):
            regime = regimes[i]
            # Map regime to color: -1 -> 0, 0 -> 0.5, 1 -> 1
            color_idx = (regime + 1) / 2
            color = cmap(color_idx)
            
            axs[1].axvspan(dates[i], dates[i+1], color=color, alpha=0.9)
        
        # Add regime labels on y-axis
        axs[1].set_yticks([-1, 0, 1])
        axs[1].set_yticklabels(['Below', 'Inside', 'Above'])
        axs[1].set_ylabel('Regime')
        axs[1].set_ylim(-1.5, 1.5)
        
        # Calculate regime statistics
        regime_counts = {
            'Above': np.sum(regimes == 1),
            'Inside': np.sum(regimes == 0),
            'Below': np.sum(regimes == -1)
        }
        
        total_obs = len(regimes)
        regime_pcts = {k: count/total_obs*100 for k, count in regime_counts.items()}
        
        # Calculate regime transitions
        transitions = 0
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transitions += 1
        
        # Add transition info to plot
        transition_text = (
            f"Transitions: {transitions}\n"
            f"Above: {regime_pcts['Above']:.1f}%\n"
            f"Inside: {regime_pcts['Inside']:.1f}%\n"
            f"Below: {regime_pcts['Below']:.1f}%"
        )
        
        # Add text box in top right
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        axs[0].text(0.95, 0.95, transition_text, transform=axs[0].transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=props)
        
        # Add labels and titles
        axs[0].set_ylabel('Price Differential')
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle('Regime Transitions Analysis', fontsize=14)
        
        # Format date axis
        format_date_axis(axs[1], date_format=self.date_format)
        axs[1].set_xlabel('Date')
        
        # Add grid to top plot
        axs[0].grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
        
        return fig, axs
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError), reraise=True)
    def compare_adjustment_patterns(
        self,
        original_model: Any,
        simulated_model: Any,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Compare asymmetric adjustment patterns before and after policy intervention.
        
        Parameters
        ----------
        original_model : Any
            Original threshold model before policy intervention
        simulated_model : Any
            Simulated threshold model after policy intervention
        title : str, optional
            Plot title, default None
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        axs : List[plt.Axes]
            List of matplotlib axes
        """
        # Create figure with two rows and two columns
        fig, axs = plt.subplots(2, 2, figsize=(self.fig_width, self.fig_height),
                               gridspec_kw={'height_ratios': [2, 1]})
        
        # Extract data from both models
        models = [original_model, simulated_model]
        titles = ['Original', 'After Policy']
        
        for i, (model, title_prefix) in enumerate(zip(models, titles)):
            # Extract model data
            try:
                price_diff = model.price_diff
                dates = model.dates
                threshold = model.threshold
                
                # Get adjustment parameters
                adj_below = getattr(model, 'adjustment_below', None)
                adj_middle = getattr(model, 'adjustment_middle', None)
                adj_above = getattr(model, 'adjustment_above', None)
                
                if adj_below is None:
                    # Try alternative attribute patterns
                    adj_below = getattr(model, 'adjustment_below_1', None)
                    adj_above = getattr(model, 'adjustment_above_1', None)
                
                # Check if adjustment parameters are available
                if None not in [adj_below, adj_above]:
                    # Top row: Price differential with thresholds
                    axs[0, i].plot(dates, price_diff, color='k', linewidth=1.2)
                    axs[0, i].axhline(y=threshold, color=self.above_color, linestyle='--', linewidth=1.2)
                    axs[0, i].axhline(y=-threshold, color=self.below_color, linestyle='--', linewidth=1.2)
                    axs[0, i].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                    
                    # Shade regimes
                    above_thresh = price_diff > threshold
                    below_thresh = price_diff < -threshold
                    
                    for j in range(len(dates) - 1):
                        start_date = dates[j]
                        end_date = dates[j+1]
                        
                        if above_thresh[j]:
                            axs[0, i].axvspan(start_date, end_date, color=self.above_color, alpha=0.2)
                        elif below_thresh[j]:
                            axs[0, i].axvspan(start_date, end_date, color=self.below_color, alpha=0.2)
                    
                    axs[0, i].set_title(f'{title_prefix} (Threshold: {threshold:.3f})')
                    format_date_axis(axs[0, i], date_format=self.date_format)
                    axs[0, i].grid(True, alpha=0.3)
                    
                    # Bottom row: Adjustment speeds
                    labels = ['Below', 'Middle', 'Above']
                    speeds = [abs(adj) if adj is not None else 0 
                             for adj in [adj_below, adj_middle, adj_above]]
                    colors = [self.below_color, self.middle_color, self.above_color]
                    
                    bars = axs[1, i].bar(labels, speeds, color=colors, alpha=0.7)
                    
                    # Add text labels
                    for bar, speed in zip(bars, speeds):
                        height = bar.get_height()
                        if height > 0:
                            axs[1, i].text(
                                bar.get_x() + bar.get_width()/2.,
                                height + 0.01,
                                f'{speed:.3f}',
                                ha='center',
                                va='bottom',
                                fontsize=8
                            )
                    
                    axs[1, i].set_ylabel('Adjustment Speed')
                    axs[1, i].grid(True, alpha=0.3, axis='y')
                    
                    # Calculate absolute asymmetry
                    asymmetry = abs(abs(adj_below) - abs(adj_above))
                    axs[1, i].text(
                        0.5, 0.9,
                        f'Asymmetry: {asymmetry:.3f}',
                        transform=axs[1, i].transAxes,
                        ha='center',
                        va='top',
                        bbox=dict(facecolor='white', alpha=0.8)
                    )
                else:
                    # Handle missing data
                    axs[0, i].text(0.5, 0.5, 'Data not available', 
                                 ha='center', va='center', transform=axs[0, i].transAxes)
                    axs[1, i].text(0.5, 0.5, 'Data not available', 
                                 ha='center', va='center', transform=axs[1, i].transAxes)
            except AttributeError as e:
                logger.warning(f"Could not extract data from {title_prefix} model: {e}")
                axs[0, i].text(0.5, 0.5, f'Error extracting data from {title_prefix} model', 
                             ha='center', va='center', transform=axs[0, i].transAxes)
                axs[1, i].text(0.5, 0.5, 'Data not available', 
                             ha='center', va='center', transform=axs[1, i].transAxes)
        
        # Add y-label to first column only
        axs[0, 0].set_ylabel('Price Differential')
        
        # Set overall title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        else:
            fig.suptitle('Comparison of Asymmetric Adjustment Patterns', fontsize=14, y=0.98)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
        
        return fig, axs
