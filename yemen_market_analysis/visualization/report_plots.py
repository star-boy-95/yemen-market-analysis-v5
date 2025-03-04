"""
Report-specific visualizations for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from core.decorators import error_handler, performance_tracker
from .plot_manager import get_plot_manager, PlotManager

logger = logging.getLogger(__name__)


@error_handler(fallback_value=None)
@performance_tracker()
def plot_commodity_dashboard(
    model_results: Dict[str, Any],
    price_diff_df: pd.DataFrame,
    north_prices: pd.Series,
    south_prices: pd.Series,
    conflict_series: Optional[pd.Series] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Create comprehensive dashboard for a commodity.
    
    Args:
        model_results: Threshold model results
        price_diff_df: DataFrame with price differentials
        north_prices: North market price series
        south_prices: South market price series
        conflict_series: Optional conflict intensity series
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    if not model_results or price_diff_df.empty or north_prices.empty or south_prices.empty:
        logger.warning("Missing required data for dashboard")
        return None
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Extract key parameters
    threshold = model_results.get('threshold', 0.0)
    adjustment = model_results.get('adjustment_dynamics', {})
    alpha_down = adjustment.get('alpha_down', 0.0)
    alpha_up = adjustment.get('alpha_up', 0.0)
    half_life_down = adjustment.get('half_life_down', float('inf'))
    half_life_up = adjustment.get('half_life_up', float('inf'))
    
    # Format half-life strings
    hl_down_str = f"{half_life_down:.1f}" if half_life_down < 100 else "∞"
    hl_up_str = f"{half_life_up:.1f}" if half_life_up < 100 else "∞"
    
    # Calculate regime statistics
    above_mask = price_diff_df['diff_pct'] > threshold
    below_mask = price_diff_df['diff_pct'] < -threshold
    band_mask = ~(above_mask | below_mask)
    
    above_pct = above_mask.mean() * 100
    below_pct = below_mask.mean() * 100
    band_pct = band_mask.mean() * 100
    
    # Create dashboard figure
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 3)
    
    # 1. Price Series Plot (top left)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(north_prices.index, north_prices, marker='', linestyle='-', color='blue', alpha=0.8, label='North')
    ax1.plot(south_prices.index, south_prices, marker='', linestyle='-', color='red', alpha=0.8, label='South')
    
    # Format plot
    ax1.set_title(f"Price Comparison for {commodity}")
    ax1.set_xlabel('')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # 2. Price Differential and Regimes (middle left)
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.plot(price_diff_df.index, price_diff_df['diff_pct'], color='gray', alpha=0.7)
    
    # Plot threshold bands
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    ax2.axhline(y=-threshold, color='red', linestyle='--')
    
    # Fill regimes
    ax2.fill_between(price_diff_df.index, price_diff_df['diff_pct'], threshold, 
                     where=price_diff_df['diff_pct'] > threshold, 
                     color='salmon', alpha=0.3)
    ax2.fill_between(price_diff_df.index, price_diff_df['diff_pct'], -threshold, 
                     where=price_diff_df['diff_pct'] < -threshold, 
                     color='skyblue', alpha=0.3)
    
    # Format plot
    ax2.set_title("Price Differential and Threshold Regimes")
    ax2.set_xlabel('')
    ax2.set_ylabel('Price Differential (%)')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # 3. Adjustment Parameters Bar Chart (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(['Lower\nRegime', 'Upper\nRegime'], 
                  [abs(alpha_down), abs(alpha_up)], 
                  color=['skyblue', 'salmon'], alpha=0.7)
    
    # Add half-life annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        hl_text = hl_down_str if i == 0 else hl_up_str
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'Half-life: {hl_text}',
                ha='center', va='bottom', fontsize=9)
    
    # Format plot
    ax3.set_title("Adjustment Speed by Regime")
    ax3.set_ylabel('Adjustment Speed (|α|)')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 4. Regime Distribution Pie Chart (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    wedges, texts, autotexts = ax4.pie([below_pct, band_pct, above_pct], 
                                     labels=['Lower', 'Band', 'Upper'],
                                     autopct='%1.1f%%',
                                     colors=['skyblue', 'lightgreen', 'salmon'],
                                     startangle=90)
    
    # Format pie chart
    ax4.set_title("Regime Distribution")
    plt.setp(autotexts, size=9, weight='bold')
    
    # 5. Conflict Impact (if available)
    if conflict_series is not None and not conflict_series.empty:
        ax5 = fig.add_subplot(gs[2, 0:2])
        
        # Align series
        aligned_index = price_diff_df.index.intersection(conflict_series.index)
        if len(aligned_index) > 0:
            aligned_diff = price_diff_df.loc[aligned_index, 'diff_pct'].abs()
            aligned_conflict = conflict_series.loc[aligned_index]
            
            # Plot absolute price differential
            ax5.plot(aligned_index, aligned_diff, color='blue', alpha=0.7, label='|Price Diff %|')
            
            # Plot conflict on secondary y-axis
            ax5_2 = ax5.twinx()
            ax5_2.plot(aligned_index, aligned_conflict, color='red', alpha=0.7, label='Conflict')
            ax5_2.set_ylabel('Conflict Intensity', color='red')
            ax5_2.tick_params(axis='y', labelcolor='red')
            
            # Format plot
            ax5.set_title("Price Differential vs. Conflict Intensity")
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Absolute Price Differential (%)', color='blue')
            ax5.tick_params(axis='y', labelcolor='blue')
            ax5.grid(True, linestyle='--', alpha=0.7)
            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax5.get_xticklabels(), rotation=45)
            
            # Add correlation text
            correlation = aligned_diff.corr(aligned_conflict)
            ax5.text(0.05, 0.95, f"Correlation: {correlation:.2f}", 
                    transform=ax5.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # Combine legends
            lines1, labels1 = ax5.get_legend_handles_labels()
            lines2, labels2 = ax5_2.get_legend_handles_labels()
            ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 6. Summary statistics (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')  # No axis for text box
    
    # Create text summary
    summary_text = f"""
    SUMMARY: {commodity.upper()}
    
    Threshold: {threshold:.3f}
    
    Adjustment Parameters:
      • Lower regime (α): {alpha_down:.3f}
      • Upper regime (α): {alpha_up:.3f}
      • Lower half-life: {hl_down_str}
      • Upper half-life: {hl_up_str}
    
    Regime Distribution:
      • Lower regime: {below_pct:.1f}%
      • Neutral band: {band_pct:.1f}%
      • Upper regime: {above_pct:.1f}%
    
    Threshold Significance:
      • p-value: {model_results.get('p_value', 0.0):.3f}
      • Significant: {"Yes" if model_results.get('threshold_significant', False) else "No"}
    """
    
    ax6.text(0, 1, summary_text, fontsize=10, verticalalignment='top', 
            fontfamily='monospace', bbox=dict(facecolor='lightgray', alpha=0.5))
    
    # Add overall title
    fig.suptitle(f"Market Analysis Dashboard: {commodity}", fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save figure if filename provided
    if filename:
        plot_manager.save_figure(fig, filename)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


@error_handler(fallback_value=None)
@performance_tracker()
def plot_policy_dashboard(
    policy_results: Dict[str, Any],
    model_results: Dict[str, Any],
    welfare_results: Dict[str, Any],
    title: Optional[str] = None,
    commodity: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Create policy implications dashboard.
    
    Args:
        policy_results: Results from policy implications analysis
        model_results: Threshold model results
        welfare_results: Results from welfare analysis
        title: Optional plot title
        commodity: Commodity name
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    if not policy_results:
        logger.warning("No policy results provided")
        return None
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Policy Priority Gauge (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get policy priority
    priority = policy_results.get('policy_priority', 'Medium')
    priority_values = {
        'Low': 0.25,
        'Medium': 0.5,
        'Medium-High': 0.75,
        'High': 1.0
    }
    
    # Create gauge visualization
    gauge_value = priority_values.get(priority, 0.5)
    
    # Draw gauge
    theta = np.linspace(0.05 * np.pi, 0.95 * np.pi, 100)
    r = 0.8
    
    # Draw gauge background
    cmap = plt.cm.RdYlGn_r
    colors = cmap(np.linspace(0, 1, 100))
    
    for i in range(99):
        ax1.add_patch(plt.Rectangle((0.05 + 0.9*i/100, 0.05), 0.9/100, 0.15, color=colors[i]))
    
    # Draw gauge pointer
    pointer_angle = 0.05 * np.pi + gauge_value * 0.9 * np.pi
    ax1.arrow(0.5, 0.3, 0.4 * np.cos(pointer_angle), 0.4 * np.sin(pointer_angle),
             width=0.03, head_width=0.07, head_length=0.1, fc='black', ec='black')
    
    # Add gauge labels
    ax1.text(0.1, 0.05, 'Low', fontsize=10, horizontalalignment='center')
    ax1.text(0.5, 0.05, 'Medium', fontsize=10, horizontalalignment='center')
    ax1.text(0.9, 0.05, 'High', fontsize=10, horizontalalignment='center')
    
    # Show priority
    ax1.text(0.5, 0.7, f"Policy Priority: {priority}", fontsize=14, fontweight='bold',
            horizontalalignment='center')
    
    # Add explanation
    explanation = policy_results.get('explanation', '')
    ax1.text(0.5, 0.6, explanation, fontsize=10, horizontalalignment='center',
            verticalalignment='top', wrap=True)
    
    # Remove axis
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 2. Policy Recommendations (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')  # No axis for text box
    
    # Get recommendations
    recommendations = policy_results.get('recommendations', [])
    
    # Create recommendations text
    recommendations_text = "POLICY RECOMMENDATIONS:\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        recommendations_text += f"{i}. {rec}\n\n"
    
    ax2.text(0, 1, recommendations_text, fontsize=11, verticalalignment='top',
            bbox=dict(facecolor='lightblue', alpha=0.3))
    
    # 3. Welfare Impact (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Extract welfare metrics
    dwl_percent = welfare_results.get('dwl_percent_of_market', 0.0)
    arbitrage_freq = welfare_results.get('arbitrage_frequency', 0.0)
    north_higher = welfare_results.get('north_higher_pct', 0.0)
    south_higher = welfare_results.get('south_higher_pct', 0.0)
    
    # Create bar chart
    metrics = ['Deadweight\nLoss', 'Arbitrage\nFrequency', 'North Price\nHigher', 'South Price\nHigher']
    values = [dwl_percent, arbitrage_freq, north_higher, south_higher]
    colors = ['red', 'orange', 'blue', 'green']
    
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Format plot
    ax3.set_title("Welfare Impact Metrics")
    ax3.set_ylabel('Percentage (%)')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax3.set_ylim(0, max(values) * 1.2)
    
    # 4. Market Integration Metrics (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Extract key metrics
    key_metrics = policy_results.get('key_metrics', {})
    integration_level = key_metrics.get('integration_level', 'Unknown')
    integration_index = key_metrics.get('integration_index', 0.0)
    adjustment_asymmetry = key_metrics.get('adjustment_asymmetry', 1.0)
    faster_regime = key_metrics.get('faster_regime', 'unknown')
    threshold_value = key_metrics.get('threshold_value', 0.0)
    
    # Create text for key metrics
    metrics_text = f"""
    MARKET INTEGRATION METRICS:
    
    Integration Level: {integration_level}
    Integration Index: {integration_index:.2f}
    
    Threshold Value: {threshold_value:.3f}
    
    Adjustment Asymmetry: {adjustment_asymmetry:.2f}x
    (Faster regime: {faster_regime})
    
    Welfare Impact:
      • Deadweight Loss: {dwl_percent:.2f}% of market value
      • Surplus Direction: {welfare_results.get('surplus_direction', '')}
      • Net Impact: {welfare_results.get('net_welfare_impact', '')}
    """
    
    ax4.text(0, 1, metrics_text, fontsize=10, verticalalignment='top',
            fontfamily='monospace', bbox=dict(facecolor='lightgray', alpha=0.5))
    
    # Remove axis
    ax4.axis('off')
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"Policy Dashboard: {commodity}", fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save figure if filename provided
    if filename:
        plot_manager.save_figure(fig, filename)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


@error_handler(fallback_value=None)
@performance_tracker()
def plot_summary_overview(
    summary_df: pd.DataFrame,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Create summary overview of key metrics across commodities.
    
    Args:
        summary_df: DataFrame with summary statistics per commodity
        title: Optional plot title
        filename: Optional filename for saving
        show: Whether to show the plot
        
    Returns:
        Figure object if successful, None otherwise
    """
    if summary_df.empty:
        logger.warning("Empty summary DataFrame")
        return None
    
    # Get plot manager
    plot_manager = get_plot_manager()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Threshold comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Sort by threshold
    threshold_df = summary_df.sort_values('threshold', ascending=True)
    
    # Plot threshold bars
    bars1 = ax1.barh(threshold_df['commodity'], threshold_df['threshold'], 
                    color='blue', alpha=0.7)
    
    # Add value labels
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', va='center', fontsize=9)
    
    # Format plot
    ax1.set_title("Threshold Values by Commodity")
    ax1.set_xlabel('Threshold')
    ax1.set_xlim(0, max(threshold_df['threshold']) * 1.2)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 2. Integration index comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Sort by integration index
    integration_df = summary_df.sort_values('integration_index', ascending=False)
    
    # Create color map for integration level
    cmap = plt.cm.RdYlGn
    colors = []
    for idx, row in integration_df.iterrows():
        level = row['integration_level']
        if level == 'High':
            colors.append(cmap(0.8))
        elif level == 'Moderate':
            colors.append(cmap(0.5))
        else:
            colors.append(cmap(0.2))
    
    # Plot integration index bars
    bars2 = ax2.barh(integration_df['commodity'], integration_df['integration_index'], 
                    color=colors, alpha=0.7)
    
    # Add value labels
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', fontsize=9)
    
    # Format plot
    ax2.set_title("Market Integration Index by Commodity")
    ax2.set_xlabel('Integration Index')
    ax2.set_xlim(0, 1.0)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add legend for integration levels
    high_patch = mpatches.Patch(color=cmap(0.8), label='High')
    mod_patch = mpatches.Patch(color=cmap(0.5), label='Moderate')
    low_patch = mpatches.Patch(color=cmap(0.2), label='Low')
    ax2.legend(handles=[high_patch, mod_patch, low_patch], loc='lower right')
    
    # 3. Adjustment speeds comparison (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Sort by average adjustment speed
    summary_df['avg_adjustment'] = (abs(summary_df['alpha_down']) + abs(summary_df['alpha_up'])) / 2
    adjustment_df = summary_df.sort_values('avg_adjustment', ascending=False)
    
    # Prepare data for grouped bar chart
    commodities = adjustment_df['commodity']
    alpha_down = adjustment_df['alpha_down'].abs()
    alpha_up = adjustment_df['alpha_up'].abs()
    
    # Set positions and width
    pos = np.arange(len(commodities))
    width = 0.35
    
    # Create grouped bar chart
    ax3.barh(pos - width/2, alpha_down, width, color='skyblue', alpha=0.7, label='Lower Regime')
    ax3.barh(pos + width/2, alpha_up, width, color='salmon', alpha=0.7, label='Upper Regime')
    
    # Format plot
    ax3.set_title("Adjustment Speeds by Commodity")
    ax3.set_xlabel('Adjustment Speed (|α|)')
    ax3.set_yticks(pos)
    ax3.set_yticklabels(commodities)
    ax3.legend()
    ax3.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 4. Half-lives comparison (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Sort by average half-life
    summary_df['avg_half_life'] = (summary_df['half_life_down'] + summary_df['half_life_up']) / 2
    halflife_df = summary_df.sort_values('avg_half_life', ascending=True)
    
    # Cap half-lives for better visualization
    max_display = 12
    hl_down = halflife_df['half_life_down'].apply(lambda x: min(x, max_display))
    hl_up = halflife_df['half_life_up'].apply(lambda x: min(x, max_display))
    
    # Set positions and width
    pos = np.arange(len(halflife_df))
    width = 0.35
    
    # Create grouped bar chart
    ax4.barh(pos - width/2, hl_down, width, color='skyblue', alpha=0.7, label='Lower Regime')
    ax4.barh(pos + width/2, hl_up, width, color='salmon', alpha=0.7, label='Upper Regime')
    
    # Format plot
    ax4.set_title("Half-Lives by Commodity")
    ax4.set_xlabel('Half-Life (periods)')
    ax4.set_yticks(pos)
    ax4.set_yticklabels(halflife_df['commodity'])
    ax4.legend()
    ax4.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add note for capped values
    ax4.text(0.5, -0.1, f"Note: Half-lives larger than {max_display} are truncated", 
            ha='center', transform=ax4.transAxes, fontsize=8, fontstyle='italic')
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle("Market Integration Summary", fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    
    # Save figure if filename provided
    if filename:
        plot_manager.save_figure(fig, filename)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig