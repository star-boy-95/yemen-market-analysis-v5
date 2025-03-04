"""
Market analysis visualizations for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from core.decorators import error_handler, performance_tracker
from .plot_manager import get_plot_manager, PlotManager

logger = logging.getLogger(__name__)


@error_handler(fallback_value=None)
@performance_tracker()
def plot_price_series(
    north_prices: pd.Series,
    south_prices: pd.Series,
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False,
    price_type: str = 'USD'
) -> Optional[plt.Figure]:
    """
    Plot north and south price series.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        title: Optional plot title
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        price_type: Label for price type (USD, YER, etc.)
        
    Returns:
        Figure object if successful, None otherwise
    """
    if north_prices.empty or south_prices.empty:
        logger.warning("Empty price series")
        return None
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig, ax = plot_manager.create_figure()
    
    # Plot price series
    ax.plot(north_prices.index, north_prices, marker='o', markersize=3, 
           linestyle='-', color='blue', alpha=0.7, label='North')
    ax.plot(south_prices.index, south_prices, marker='s', markersize=3, 
           linestyle='-', color='red', alpha=0.7, label='South')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Price Comparison for {commodity or 'Commodity'}")
    
    # Set labels
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Price ({price_type})')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend(loc='best')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plot_manager.save_figure(fig, filename)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


@error_handler(fallback_value=None)
@performance_tracker()
def plot_arbitrage_opportunities(
    price_diff_df: pd.DataFrame,
    threshold: float,
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot arbitrage opportunities based on price differential.
    
    Args:
        price_diff_df: DataFrame with price differentials
        threshold: Threshold value
        title: Optional plot title
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    if price_diff_df is None or price_diff_df.empty:
        logger.warning("Empty price differential DataFrame")
        return None
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plot_manager.create_figure(nrows=2, figsize=(10, 8))
    
    # Identify arbitrage opportunities
    north_to_south = price_diff_df['diff_pct'] < -threshold
    south_to_north = price_diff_df['diff_pct'] > threshold
    
    # Calculate arbitrage frequencies
    n2s_freq = north_to_south.mean() * 100
    s2n_freq = south_to_north.mean() * 100
    
    # Plot price differential on top subplot
    dates = price_diff_df.index
    ax1.plot(dates, price_diff_df['diff_pct'], color='gray', alpha=0.7, label='Price differential (%)')
    
    # Plot threshold lines
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    ax1.axhline(y=-threshold, color='red', linestyle='--')
    
    # Highlight arbitrage opportunities
    ax1.fill_between(dates, price_diff_df['diff_pct'], threshold, 
                    where=south_to_north, color='salmon', alpha=0.3, 
                    label=f'S→N arbitrage ({s2n_freq:.1f}%)')
    ax1.fill_between(dates, price_diff_df['diff_pct'], -threshold, 
                    where=north_to_south, color='skyblue', alpha=0.3,
                    label=f'N→S arbitrage ({n2s_freq:.1f}%)')
    
    # Set top subplot labels and legend
    ax1.set_title(f"Price Differential and Arbitrage Opportunities for {commodity or 'Commodity'}")
    ax1.set_xlabel('')  # No x-label for top subplot
    ax1.set_ylabel('Price Differential (%)')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot profit potential on bottom subplot
    n2s_profit = (-price_diff_df['diff_pct'] - threshold)[north_to_south]
    s2n_profit = (price_diff_df['diff_pct'] - threshold)[south_to_north]
    
    # Calculate profit statistics
    n2s_profit_mean = n2s_profit.mean() * 100 if len(n2s_profit) > 0 else 0
    s2n_profit_mean = s2n_profit.mean() * 100 if len(s2n_profit) > 0 else 0
    
    n2s_dates = price_diff_df.index[north_to_south]
    s2n_dates = price_diff_df.index[south_to_north]
    
    # Plot profit potential
    if len(n2s_profit) > 0:
        ax2.scatter(n2s_dates, n2s_profit * 100, color='blue', alpha=0.7, 
                   label=f'N→S profit (avg: {n2s_profit_mean:.1f}%)')
    
    if len(s2n_profit) > 0:
        ax2.scatter(s2n_dates, s2n_profit * 100, color='red', alpha=0.7,
                   label=f'S→N profit (avg: {s2n_profit_mean:.1f}%)')
    
    # Set bottom subplot labels and legend
    ax2.set_title("Profit Potential from Arbitrage")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Profit Margin (%)')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axes
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plot_manager.save_figure(fig, filename)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


@error_handler(fallback_value=None)
@performance_tracker()
def plot_market_integration(
    integration_results: Dict[str, Any],
    model_results: Dict[str, Any],
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot market integration analysis.
    
    Args:
        integration_results: Results from market integration analysis
        model_results: Threshold model results
        title: Optional plot title
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig, ax = plot_manager.create_figure(figsize=(8, 6))
    
    # Extract integration components
    component_scores = integration_results.get('integration', {}).get('component_scores', {})
    if not component_scores:
        logger.warning("No component scores in integration results")
        return None
    
    # Extract values
    components = list(component_scores.keys())
    scores = [component_scores[c] for c in components]
    
    # Format component names for display
    component_labels = [c.replace('_', ' ').title() for c in components]
    
    # Create bar chart
    bars = ax.bar(component_labels, scores, color=['blue', 'green', 'orange'], alpha=0.7)
    
    # Add integration index value
    integration_index = integration_results.get('integration', {}).get('integration_index', 0)
    integration_level = integration_results.get('integration', {}).get('integration_level', 'Unknown')
    
    # Add threshold value
    threshold = model_results.get('threshold', 0)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Market Integration for {commodity or 'Commodity'}")
    
    # Add integration index and threshold information
    ax.text(0.5, 0.91, f"Integration Index: {integration_index:.2f} ({integration_level})", 
            horizontalalignment='center', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    ax.text(0.5, 0.82, f"Threshold: {threshold:.3f}", 
            horizontalalignment='center', transform=ax.transAxes,
            fontsize=11, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Set labels
    ax.set_xlabel('Component')
    ax.set_ylabel('Score (0-1)')
    
    # Set y-limits
    ax.set_ylim(0, 1.1)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plot_manager.save_figure(fig, filename)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


@error_handler(fallback_value=None)
@performance_tracker()
def plot_conflict_impact(
    conflict_results: Dict[str, Any],
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot impact of conflict on market integration.
    
    Args:
        conflict_results: Results from conflict impact analysis
        title: Optional plot title
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    # Extract high/low conflict metrics
    high_conflict = conflict_results.get('high_conflict', {})
    low_conflict = conflict_results.get('low_conflict', {})
    
    if not high_conflict or not low_conflict:
        logger.warning("Missing conflict metrics")
        return None
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plot_manager.create_figure(nrows=3, figsize=(10, 12))
    
    # Define metrics to plot
    metrics = [
        ('avg_price_diff_pct', 'Avg Price Differential (%)'),
        ('arbitrage_freq', 'Arbitrage Frequency (%)'),
        ('price_diff_volatility', 'Price Differential Volatility (%)')
    ]
    
    # Colors for high/low conflict
    colors = ['red', 'green']
    
    # Plot price differential comparison (bar chart)
    metric, label = metrics[0]
    values = [high_conflict.get(metric, 0), low_conflict.get(metric, 0)]
    bars1 = ax1.bar(['High Conflict', 'Low Conflict'], values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax1.set_title(f"Price Differential by Conflict Level")
    ax1.set_ylabel(label)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Calculate percentage increase
    pct_increase = ((values[0] / values[1]) - 1) * 100 if values[1] > 0 else 0
    ax1.text(0.5, 0.9, f"Increase in high conflict: {pct_increase:.1f}%", 
            horizontalalignment='center', transform=ax1.transAxes, 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot arbitrage frequency comparison (bar chart)
    metric, label = metrics[1]
    values = [high_conflict.get(metric, 0), low_conflict.get(metric, 0)]
    bars2 = ax2.bar(['High Conflict', 'Low Conflict'], values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax2.set_title(f"Arbitrage Frequency by Conflict Level")
    ax2.set_ylabel(label)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Calculate difference
    diff = values[0] - values[1]
    ax2.text(0.5, 0.9, f"Difference: {diff:.1f} percentage points", 
            horizontalalignment='center', transform=ax2.transAxes, 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot volatility comparison (bar chart)
    metric, label = metrics[2]
    values = [high_conflict.get(metric, 0), low_conflict.get(metric, 0)]
    bars3 = ax3.bar(['High Conflict', 'Low Conflict'], values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax3.set_title(f"Price Differential Volatility by Conflict Level")
    ax3.set_ylabel(label)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Calculate percentage increase
    pct_increase = ((values[0] / values[1]) - 1) * 100 if values[1] > 0 else 0
    ax3.text(0.5, 0.9, f"Increase in high conflict: {pct_increase:.1f}%", 
            horizontalalignment='center', transform=ax3.transAxes, 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"Conflict Impact on Market Integration for {commodity or 'Commodity'}", 
                    fontsize=14, fontweight='bold')
    plt.subplots_adjust(top=0.92)  # Adjust for suptitle
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plot_manager.save_figure(fig, filename)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig