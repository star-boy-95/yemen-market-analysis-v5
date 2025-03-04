"""
Threshold model visualizations for Yemen Market Analysis.
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
def plot_threshold_regimes(
    price_diff_df: pd.DataFrame,
    threshold: float,
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot price differential with threshold regimes.
    
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
    
    # Create figure
    fig, ax = plot_manager.create_figure()
    
    # Compute percentages in each regime
    above_mask = price_diff_df['diff_pct'] > threshold
    below_mask = price_diff_df['diff_pct'] < -threshold
    band_mask = ~(above_mask | below_mask)
    
    above_pct = above_mask.mean() * 100
    below_pct = below_mask.mean() * 100
    band_pct = band_mask.mean() * 100
    
    # Plot price differential
    dates = price_diff_df.index
    ax.plot(dates, price_diff_df['diff_pct'], color='gray', alpha=0.7, label='Price differential (%)')
    
    # Plot threshold bands
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    ax.axhline(y=-threshold, color='red', linestyle='--')
    
    # Fill regimes
    ax.fill_between(dates, price_diff_df['diff_pct'], threshold, 
                   where=price_diff_df['diff_pct'] > threshold, 
                   color='salmon', alpha=0.3, label=f'Upper regime ({above_pct:.1f}%)')
    ax.fill_between(dates, price_diff_df['diff_pct'], -threshold, 
                   where=price_diff_df['diff_pct'] < -threshold, 
                   color='skyblue', alpha=0.3, label=f'Lower regime ({below_pct:.1f}%)')
    ax.fill_between(dates, threshold, -threshold, 
                   where=band_mask, 
                   color='lightgreen', alpha=0.2, label=f'Neutral band ({band_pct:.1f}%)')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Threshold Regimes for {commodity or 'Commodity'}")
    
    # Set labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Price Differential (%)')
    
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
def plot_adjustment_speeds(
    model_results: Dict[str, Any],
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot adjustment speeds from threshold model results.
    
    Args:
        model_results: Threshold model results
        title: Optional plot title
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    # Extract adjustment parameters
    adjustment = model_results.get('adjustment_dynamics', {})
    if not adjustment:
        logger.warning("No adjustment dynamics in model results")
        return None
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig, ax = plot_manager.create_figure(figsize=(8, 6))
    
    # Extract parameters
    alpha_down = adjustment.get('alpha_down', 0)
    alpha_up = adjustment.get('alpha_up', 0)
    half_life_down = adjustment.get('half_life_down', float('inf'))
    half_life_up = adjustment.get('half_life_up', float('inf'))
    
    # Format half-life strings
    hl_down_str = f"{half_life_down:.1f}" if half_life_down < 100 else "∞"
    hl_up_str = f"{half_life_up:.1f}" if half_life_up < 100 else "∞"
    
    # Data for bar chart
    regimes = ['Lower Regime', 'Upper Regime']
    alphas = [abs(alpha_down), abs(alpha_up)]
    colors = ['skyblue', 'salmon']
    
    # Plot bars
    bars = ax.bar(regimes, alphas, color=colors, alpha=0.7)
    
    # Add threshold value
    threshold = model_results.get('threshold', 0)
    ax.text(0.5, 0.95, f"Threshold: {threshold:.3f}", 
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    # Add half-life annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        hl_text = hl_down_str if i == 0 else hl_up_str
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Half-life: {hl_text}',
                ha='center', va='bottom', fontsize=10)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Adjustment Speeds for {commodity or 'Commodity'}")
    
    # Set labels
    ax.set_xlabel('Regime')
    ax.set_ylabel('Adjustment Speed (|α|)')
    
    # Add threshold significance indicator
    is_significant = model_results.get('threshold_significant', False)
    sig_text = "Threshold is statistically significant" if is_significant else "Threshold is not statistically significant"
    p_value = model_results.get('p_value', None)
    if p_value is not None:
        sig_text += f" (p = {p_value:.3f})"
    
    ax.text(0.5, 0.01, sig_text,
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            color='green' if is_significant else 'red')
    
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
def plot_rolling_thresholds(
    rolling_results: Dict[str, Any],
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot rolling window threshold estimates.
    
    Args:
        rolling_results: Results from rolling window analysis
        title: Optional plot title
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    # Extract data
    data = rolling_results.get('data', [])
    if not data:
        logger.warning("No data in rolling results")
        return None
    
    # Convert to DataFrame if needed
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig, ax = plot_manager.create_figure()
    
    # Plot threshold values
    ax.plot(df['date'], df['threshold'], marker='o', markersize=4, 
           linestyle='-', color='blue', alpha=0.7, label='Threshold')
    
    # Add mean threshold line
    mean_threshold = df['threshold'].mean()
    ax.axhline(y=mean_threshold, color='red', linestyle='--', 
              label=f'Mean threshold ({mean_threshold:.3f})')
    
    # Add structural break markers if available
    breaks = rolling_results.get('structural_breaks', {}).get('dates', [])
    if breaks:
        for break_date in breaks:
            ax.axvline(x=break_date, color='green', linestyle='-', alpha=0.5)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Rolling Threshold Estimates for {commodity or 'Commodity'}")
    
    # Set labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Threshold Value')
    
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
def plot_threshold_history(
    threshold_history: List[Dict[str, Any]],
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot historical threshold values for a commodity.
    
    Args:
        threshold_history: List of threshold history records
        title: Optional plot title
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    if not threshold_history:
        logger.warning("Empty threshold history")
        return None
    
    # Create DataFrame from history
    data = []
    for record in threshold_history:
        try:
            # Parse date
            date = datetime.fromisoformat(record['date'])
            threshold = record['threshold']
            significant = record.get('significant', False)
            
            data.append({
                'date': date,
                'threshold': threshold,
                'significant': significant
            })
        except (KeyError, ValueError) as e:
            logger.warning(f"Error processing threshold history record: {str(e)}")
    
    if not data:
        logger.warning("No valid threshold history records")
        return None
    
    df = pd.DataFrame(data)
    df = df.sort_values('date')
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig, ax = plot_manager.create_figure()
    
    # Plot all threshold values with different markers for significance
    significant = df[df['significant']]
    not_significant = df[~df['significant']]
    
    ax.plot(not_significant['date'], not_significant['threshold'], 
           marker='o', markersize=5, linestyle='', color='blue', alpha=0.5,
           label='Not significant')
    
    ax.plot(significant['date'], significant['threshold'], 
           marker='*', markersize=10, linestyle='', color='red', alpha=0.7,
           label='Significant')
    
    # Add connecting line for all points
    ax.plot(df['date'], df['threshold'], linestyle='-', color='gray', alpha=0.3)
    
    # Add trend line using lowess or simple smoothing
    if len(df) >= 5:
        try:
            from scipy.signal import savgol_filter
            window_length = min(5, len(df) - (len(df) % 2 == 0))
            window_length = window_length if window_length % 2 == 1 else window_length + 1
            if window_length >= 3:
                smooth_threshold = savgol_filter(df['threshold'], window_length, 1)
                ax.plot(df['date'], smooth_threshold, linestyle='-', color='green', 
                       alpha=0.7, label='Trend')
        except Exception as e:
            logger.warning(f"Error creating trend line: {str(e)}")
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Threshold History for {commodity or 'Commodity'}")
    
    # Set labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Threshold Value')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
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