"""
Comparative heatmaps for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple, Any, Union

from core.decorators import error_handler, performance_tracker
from .plot_manager import get_plot_manager, PlotManager

logger = logging.getLogger(__name__)


@error_handler(fallback_value=None)
@performance_tracker()
def plot_integration_heatmap(
    results_by_commodity: Dict[str, Dict[str, Any]],
    sort_by: str = 'integration_index',
    reverse: bool = True,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot heatmap of market integration metrics across commodities.
    
    Args:
        results_by_commodity: Dictionary mapping commodities to results
        sort_by: Metric to sort by ('integration_index', 'threshold', etc.)
        reverse: Whether to sort in descending order
        title: Optional plot title
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    if not results_by_commodity:
        logger.warning("No results provided")
        return None
    
    # Extract metrics for each commodity
    commodities = []
    metrics = {
        'integration_index': [],
        'threshold': [],
        'price_diff': [],
        'arbitrage_freq': []
    }
    
    for commodity, results in results_by_commodity.items():
        # Skip commodities with missing data
        if not results:
            continue
            
        # Add commodity
        commodities.append(commodity)
        
        # Extract metrics
        metrics['integration_index'].append(
            results.get('integration', {}).get('integration_index', 0.0)
        )
        metrics['threshold'].append(
            results.get('threshold', 0.0)
        )
        metrics['price_diff'].append(
            results.get('avg_price_differential_pct', 0.0)
        )
        metrics['arbitrage_freq'].append(
            results.get('arbitrage_frequency', 0.0)
        )
    
    if not commodities:
        logger.warning("No valid commodities with results")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(metrics, index=commodities)
    
    # Sort by specified metric
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not reverse)
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig, ax = plot_manager.create_figure(figsize=(10, 8))
    
    # Define colormap ranges for each metric
    cmap_ranges = {
        'integration_index': (0, 1, 'RdYlGn', True),  # Higher is better
        'threshold': (0, 0.3, 'RdYlGn_r', True),     # Lower is better
        'price_diff': (0, 30, 'RdYlGn_r', True),     # Lower is better
        'arbitrage_freq': (0, 50, 'RdYlGn_r', True)  # Lower is better
    }
    
    # Create normalized DataFrames for each metric based on their ranges
    norm_df = pd.DataFrame(index=df.index)
    
    for metric, values in df.items():
        vmin, vmax, _, _ = cmap_ranges[metric]
        # Normalize to 0-1 range
        norm_values = np.clip((values - vmin) / (vmax - vmin) if vmax > vmin else 0.5, 0, 1)
        norm_df[metric] = norm_values
    
    # Plot heatmap
    im = ax.imshow(norm_df.values, cmap='RdYlGn', aspect='auto')
    
    # Add labels for each cell
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            metric = df.columns[j]
            value = df.iloc[i, j]
            
            # Format values appropriately
            if metric == 'integration_index':
                text = f"{value:.2f}"
            elif metric == 'threshold':
                text = f"{value:.3f}"
            else:
                text = f"{value:.1f}%"
            
            # Determine text color based on cell darkness
            color = 'black' if norm_df.iloc[i, j] > 0.5 else 'white'
            
            ax.text(j, i, text, ha="center", va="center", color=color)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Market Integration Metrics by Commodity")
    
    # Set axis labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    
    # Set tick labels
    metric_labels = {
        'integration_index': 'Integration\nIndex',
        'threshold': 'Threshold',
        'price_diff': 'Price\nDifferential',
        'arbitrage_freq': 'Arbitrage\nFrequency'
    }
    ax.set_xticklabels([metric_labels.get(col, col) for col in df.columns])
    ax.set_yticklabels(df.index)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar legend for reference
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Scale\n(Green = Better)", rotation=-90, va="bottom")
    
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
def plot_adjustment_heatmap(
    results_by_commodity: Dict[str, Dict[str, Any]],
    sort_by: str = 'adjustment',
    reverse: bool = True,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot heatmap of adjustment parameters across commodities.
    
    Args:
        results_by_commodity: Dictionary mapping commodities to results
        sort_by: Metric to sort by ('adjustment', 'half_life', etc.)
        reverse: Whether to sort in descending order
        title: Optional plot title
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    if not results_by_commodity:
        logger.warning("No results provided")
        return None
    
    # Extract metrics for each commodity
    commodities = []
    metrics = {
        'alpha_down': [],
        'alpha_up': [],
        'half_life_down': [],
        'half_life_up': []
    }
    
    for commodity, results in results_by_commodity.items():
        # Skip commodities with missing data
        adjustment = results.get('adjustment_dynamics', {})
        if not adjustment:
            continue
            
        # Add commodity
        commodities.append(commodity)
        
        # Extract metrics
        metrics['alpha_down'].append(adjustment.get('alpha_down', 0.0))
        metrics['alpha_up'].append(adjustment.get('alpha_up', 0.0))
        metrics['half_life_down'].append(adjustment.get('half_life_down', float('inf')))
        metrics['half_life_up'].append(adjustment.get('half_life_up', float('inf')))
    
    if not commodities:
        logger.warning("No valid commodities with adjustment parameters")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(metrics, index=commodities)
    
    # Cap half-lives for better visualization
    max_half_life = 12
    for col in ['half_life_down', 'half_life_up']:
        df[col] = df[col].apply(lambda x: min(x, max_half_life))
    
    # Sort by specified metric
    if sort_by == 'adjustment':
        # Sort by average adjustment speed
        avg_adjustment = abs(df['alpha_down'] + df['alpha_up']) / 2
        sorted_idx = avg_adjustment.sort_values(ascending=not reverse).index
        df = df.loc[sorted_idx]
    elif sort_by == 'half_life':
        # Sort by average half-life
        avg_half_life = (df['half_life_down'] + df['half_life_up']) / 2
        sorted_idx = avg_half_life.sort_values(ascending=reverse).index
        df = df.loc[sorted_idx]
    elif sort_by == 'asymmetry':
        # Sort by adjustment asymmetry
        asymmetry = abs(abs(df['alpha_down']) - abs(df['alpha_up']))
        sorted_idx = asymmetry.sort_values(ascending=not reverse).index
        df = df.loc[sorted_idx]
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig, ax = plot_manager.create_figure(figsize=(10, 8))
    
    # Define colormap ranges for each metric
    cmap_ranges = {
        'alpha_down': (-0.4, 0, 'RdYlGn', True),     # More negative is better
        'alpha_up': (-0.4, 0, 'RdYlGn', True),       # More negative is better
        'half_life_down': (0, max_half_life, 'RdYlGn_r', False),  # Lower is better
        'half_life_up': (0, max_half_life, 'RdYlGn_r', False)     # Lower is better
    }
    
    # Create normalized DataFrames for each metric based on their ranges
    norm_df = pd.DataFrame(index=df.index)
    
    for metric, values in df.items():
        vmin, vmax, cmap_name, invert = cmap_ranges[metric]
        if metric.startswith('alpha'):
            # For alpha parameters, more negative is better
            norm_values = np.clip((values - vmin) / (vmax - vmin) if vmax > vmin else 0.5, 0, 1)
            if invert:
                norm_values = 1 - norm_values
        else:
            # For half-lives, smaller is better
            norm_values = np.clip((values - vmin) / (vmax - vmin) if vmax > vmin else 0.5, 0, 1)
            if invert:
                norm_values = 1 - norm_values
        
        norm_df[metric] = norm_values
    
    # Plot heatmap
    im = ax.imshow(norm_df.values, cmap='RdYlGn', aspect='auto')
    
    # Add labels for each cell
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            metric = df.columns[j]
            value = df.iloc[i, j]
            
            # Format values appropriately
            if metric.startswith('alpha'):
                text = f"{value:.3f}"
            else:
                if value >= max_half_life:
                    text = ">12"
                else:
                    text = f"{value:.1f}"
            
            # Determine text color based on cell darkness
            color = 'black' if norm_df.iloc[i, j] > 0.5 else 'white'
            
            ax.text(j, i, text, ha="center", va="center", color=color)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Price Adjustment Parameters by Commodity")
    
    # Set axis labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    
    # Set tick labels
    metric_labels = {
        'alpha_down': 'Lower\nAdjustment',
        'alpha_up': 'Upper\nAdjustment',
        'half_life_down': 'Lower\nHalf-life',
        'half_life_up': 'Upper\nHalf-life'
    }
    ax.set_xticklabels([metric_labels.get(col, col) for col in df.columns])
    ax.set_yticklabels(df.index)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar legend for reference
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Scale\n(Green = Better)", rotation=-90, va="bottom")
    
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
def plot_conflict_sensitivity_heatmap(
    conflict_results_by_commodity: Dict[str, Dict[str, Any]],
    sort_by: str = 'impact',
    reverse: bool = True,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot heatmap of conflict sensitivity across commodities.
    
    Args:
        conflict_results_by_commodity: Dictionary mapping commodities to conflict analysis results
        sort_by: Metric to sort by ('impact', 'correlation', etc.)
        reverse: Whether to sort in descending order
        title: Optional plot title
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    if not conflict_results_by_commodity:
        logger.warning("No conflict results provided")
        return None
    
    # Extract metrics for each commodity
    commodities = []
    metrics = {
        'diff_increase': [],
        'arbitrage_increase': [],
        'correlation': [],
        'high_volatility': [],
        'low_volatility': []
    }
    
    for commodity, results in conflict_results_by_commodity.items():
        # Skip commodities with missing data
        if not results:
            continue
            
        # Extract impact metrics
        impact = results.get('impact', {})
        if not impact:
            continue
            
        # Add commodity
        commodities.append(commodity)
        
        # Extract metrics
        metrics['diff_increase'].append(
            impact.get('diff_increase_pct', 0.0)
        )
        metrics['arbitrage_increase'].append(
            impact.get('arbitrage_increase_pct', 0.0)
        )
        
        # Extract correlation
        stats = results.get('statistics', {})
        metrics['correlation'].append(
            stats.get('correlation', 0.0)
        )
        
        # Extract volatility metrics
        high_conflict = results.get('high_conflict', {})
        low_conflict = results.get('low_conflict', {})
        metrics['high_volatility'].append(
            high_conflict.get('price_diff_volatility', 0.0)
        )
        metrics['low_volatility'].append(
            low_conflict.get('price_diff_volatility', 0.0)
        )
    
    if not commodities:
        logger.warning("No valid commodities with conflict sensitivity data")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(metrics, index=commodities)
    
    # Sort by specified metric
    if sort_by == 'impact':
        # Sort by price differential increase
        sorted_idx = df['diff_increase'].sort_values(ascending=not reverse).index
        df = df.loc[sorted_idx]
    elif sort_by == 'correlation':
        # Sort by correlation with conflict
        sorted_idx = df['correlation'].abs().sort_values(ascending=not reverse).index
        df = df.loc[sorted_idx]
    elif sort_by == 'volatility_ratio':
        # Sort by volatility ratio
        volatility_ratio = df['high_volatility'] / df['low_volatility']
        sorted_idx = volatility_ratio.sort_values(ascending=not reverse).index
        df = df.loc[sorted_idx]
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig, ax = plot_manager.create_figure(figsize=(10, 8))
    
    # Define colormap ranges for each metric
    cmap_ranges = {
        'diff_increase': (0, 100, 'RdYlGn_r', False),         # Lower is better
        'arbitrage_increase': (0, 30, 'RdYlGn_r', False),     # Lower is better
        'correlation': (0, 1, 'RdYlGn_r', False),             # Lower is better
        'high_volatility': (0, 30, 'RdYlGn_r', False),        # Lower is better
        'low_volatility': (0, 20, 'RdYlGn_r', False)          # Lower is better
    }
    
    # Create normalized DataFrames for each metric based on their ranges
    norm_df = pd.DataFrame(index=df.index)
    
    for metric, values in df.items():
        vmin, vmax, cmap_name, invert = cmap_ranges[metric]
        norm_values = np.clip((values - vmin) / (vmax - vmin) if vmax > vmin else 0.5, 0, 1)
        if invert:
            norm_values = 1 - norm_values
        norm_df[metric] = norm_values
    
    # Calculate normalized volatility ratio for coloring
    norm_df['volatility_ratio'] = norm_df['high_volatility'] / norm_df['low_volatility']
    
    # Plot heatmap
    display_cols = ['diff_increase', 'arbitrage_increase', 'correlation', 'high_volatility', 'low_volatility']
    im = ax.imshow(norm_df[display_cols].values, cmap='RdYlGn_r', aspect='auto')
    
    # Add labels for each cell
    for i in range(len(df.index)):
        for j in range(len(display_cols)):
            metric = display_cols[j]
            value = df.iloc[i, df.columns.get_loc(metric)]
            
            # Format values appropriately
            if metric == 'correlation':
                text = f"{value:.2f}"
            elif metric in ['high_volatility', 'low_volatility']:
                text = f"{value:.1f}%"
            else:
                text = f"{value:.1f}%"
            
            # Determine text color based on cell darkness
            color = 'black' if norm_df[display_cols].iloc[i, j] < 0.5 else 'white'
            
            ax.text(j, i, text, ha="center", va="center", color=color)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Conflict Sensitivity by Commodity")
    
    # Set axis labels
    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_yticks(np.arange(len(df.index)))
    
    # Set tick labels
    metric_labels = {
        'diff_increase': 'Price Diff\nIncrease',
        'arbitrage_increase': 'Arbitrage\nIncrease',
        'correlation': 'Conflict\nCorrelation',
        'high_volatility': 'High Conflict\nVolatility',
        'low_volatility': 'Low Conflict\nVolatility'
    }
    ax.set_xticklabels([metric_labels.get(col, col) for col in display_cols])
    ax.set_yticklabels(df.index)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar legend
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Conflict Sensitivity\n(Red = More Sensitive)", rotation=-90, va="bottom")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plot_manager.save_figure(fig, filename)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig