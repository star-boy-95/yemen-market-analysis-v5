"""
Result table generators for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union

from core.decorators import error_handler, performance_tracker
from core.exceptions import ReportingError

logger = logging.getLogger(__name__)


@error_handler(fallback_value=pd.DataFrame())
@performance_tracker()
def create_summary_table(
    results_by_commodity: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create summary table from results for all commodities.
    
    Args:
        results_by_commodity: Dictionary with results keyed by commodity
        
    Returns:
        DataFrame with summary statistics
    """
    if not results_by_commodity:
        logger.warning("No results provided for summary table")
        return pd.DataFrame()
    
    # Initialize data for DataFrame
    data = []
    
    for commodity, results in results_by_commodity.items():
        if not results:
            continue
        
        # Extract key metrics
        threshold = results.get('threshold', np.nan)
        p_value = results.get('p_value', np.nan)
        threshold_significant = results.get('threshold_significant', False)
        
        # Extract adjustment parameters
        adjustment = results.get('adjustment_dynamics', {})
        alpha_down = adjustment.get('alpha_down', np.nan)
        alpha_up = adjustment.get('alpha_up', np.nan)
        half_life_down = adjustment.get('half_life_down', np.nan)
        half_life_up = adjustment.get('half_life_up', np.nan)
        asymmetry_significant = adjustment.get('asymmetry_significant', False)
        
        # Extract integration metrics
        integration = results.get('integration', {})
        integration_index = integration.get('integration_index', np.nan)
        integration_level = integration.get('integration_level', 'Unknown')
        
        # Create row
        row = {
            'commodity': commodity,
            'threshold': threshold,
            'p_value': p_value,
            'threshold_significant': threshold_significant,
            'alpha_down': alpha_down,
            'alpha_up': alpha_up,
            'half_life_down': half_life_down,
            'half_life_up': half_life_up,
            'asymmetry_significant': asymmetry_significant,
            'integration_index': integration_index,
            'integration_level': integration_level
        }
        
        data.append(row)
    
    # Create DataFrame
    if not data:
        logger.warning("No valid data for summary table")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Set commodity as index
    df = df.set_index('commodity')
    
    return df


@error_handler(fallback_value=pd.DataFrame())
@performance_tracker()
def create_conflict_summary_table(
    conflict_results_by_commodity: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create summary table of conflict impact results.
    
    Args:
        conflict_results_by_commodity: Dictionary with conflict results by commodity
        
    Returns:
        DataFrame with conflict impact summary
    """
    if not conflict_results_by_commodity:
        logger.warning("No conflict results provided for summary table")
        return pd.DataFrame()
    
    # Initialize data for DataFrame
    data = []
    
    for commodity, results in conflict_results_by_commodity.items():
        if not results:
            continue
        
        # Extract high conflict metrics
        high_conflict = results.get('high_conflict', {})
        high_price_diff = high_conflict.get('avg_price_diff_pct', np.nan)
        high_arbitrage = high_conflict.get('arbitrage_freq', np.nan)
        high_volatility = high_conflict.get('price_diff_volatility', np.nan)
        
        # Extract low conflict metrics
        low_conflict = results.get('low_conflict', {})
        low_price_diff = low_conflict.get('avg_price_diff_pct', np.nan)
        low_arbitrage = low_conflict.get('arbitrage_freq', np.nan)
        low_volatility = low_conflict.get('price_diff_volatility', np.nan)
        
        # Extract impact metrics
        impact = results.get('impact', {})
        diff_increase = impact.get('diff_increase_pct', np.nan)
        arbitrage_increase = impact.get('arbitrage_increase_pct', np.nan)
        magnitude = impact.get('magnitude', 'Unknown')
        
        # Extract correlation
        statistics = results.get('statistics', {})
        correlation = statistics.get('correlation', np.nan)
        diff_significant = statistics.get('diff_significant', False)
        
        # Create row
        row = {
            'commodity': commodity,
            'high_conflict_price_diff': high_price_diff,
            'low_conflict_price_diff': low_price_diff,
            'diff_increase_pct': diff_increase,
            'high_conflict_arbitrage': high_arbitrage,
            'low_conflict_arbitrage': low_arbitrage,
            'arbitrage_increase_pct': arbitrage_increase,
            'high_conflict_volatility': high_volatility,
            'low_conflict_volatility': low_volatility,
            'volatility_ratio': high_volatility / low_volatility if low_volatility > 0 else np.nan,
            'conflict_correlation': correlation,
            'diff_significant': diff_significant,
            'impact_magnitude': magnitude
        }
        
        data.append(row)
    
    # Create DataFrame
    if not data:
        logger.warning("No valid data for conflict summary table")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Set commodity as index
    df = df.set_index('commodity')
    
    return df


@error_handler(fallback_value=pd.DataFrame())
@performance_tracker()
def create_welfare_summary_table(
    welfare_results_by_commodity: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create summary table of welfare analysis results.
    
    Args:
        welfare_results_by_commodity: Dictionary with welfare results by commodity
        
    Returns:
        DataFrame with welfare analysis summary
    """
    if not welfare_results_by_commodity:
        logger.warning("No welfare results provided for summary table")
        return pd.DataFrame()
    
    # Initialize data for DataFrame
    data = []
    
    for commodity, results in welfare_results_by_commodity.items():
        if not results:
            continue
        
        # Extract welfare metrics
        dwl_total = results.get('total_deadweight_loss', np.nan)
        dwl_percent = results.get('dwl_percent_of_market', np.nan)
        arbitrage_freq = results.get('arbitrage_frequency', np.nan)
        arbitrage_days = results.get('arbitrage_days', np.nan)
        avg_diff = results.get('average_price_differential', np.nan)
        max_diff = results.get('max_price_differential', np.nan)
        
        # Extract direction metrics
        north_higher = results.get('north_higher_pct', np.nan)
        south_higher = results.get('south_higher_pct', np.nan)
        surplus_direction = results.get('surplus_direction', 'Unknown')
        net_impact = results.get('net_welfare_impact', 'Unknown')
        
        # Create row
        row = {
            'commodity': commodity,
            'deadweight_loss_pct': dwl_percent,
            'arbitrage_frequency': arbitrage_freq,
            'arbitrage_days': arbitrage_days,
            'avg_price_differential': avg_diff,
            'max_price_differential': max_diff,
            'north_higher_pct': north_higher,
            'south_higher_pct': south_higher,
            'price_direction': 'North' if north_higher > south_higher else 'South',
            'surplus_direction': surplus_direction,
            'net_welfare_impact': net_impact
        }
        
        data.append(row)
    
    # Create DataFrame
    if not data:
        logger.warning("No valid data for welfare summary table")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Set commodity as index
    df = df.set_index('commodity')
    
    return df


@error_handler(fallback_value='')
def format_summary_table_html(
    df: pd.DataFrame,
    title: str = "Market Integration Summary",
    description: str = ""
) -> str:
    """
    Format summary table as HTML with styling.
    
    Args:
        df: DataFrame with summary data
        title: Table title
        description: Table description
        
    Returns:
        HTML formatted table
    """
    if df.empty:
        return "<p>No data available</p>"
    
    # Create styler
    styler = df.style
    
    # Format columns
    format_dict = {
        'threshold': '{:.3f}',
        'p_value': '{:.3f}',
        'alpha_down': '{:.3f}',
        'alpha_up': '{:.3f}',
        'half_life_down': '{:.1f}',
        'half_life_up': '{:.1f}',
        'integration_index': '{:.2f}',
        'high_conflict_price_diff': '{:.1f}',
        'low_conflict_price_diff': '{:.1f}',
        'diff_increase_pct': '{:.1f}%',
        'arbitrage_increase_pct': '{:.1f}%',
        'conflict_correlation': '{:.2f}',
        'deadweight_loss_pct': '{:.2f}%',
        'arbitrage_frequency': '{:.1f}%',
        'avg_price_differential': '{:.1f}%',
        'max_price_differential': '{:.1f}%',
        'north_higher_pct': '{:.1f}%',
        'south_higher_pct': '{:.1f}%'
    }
    
    # Apply formatting where columns exist
    format_dict = {col: fmt for col, fmt in format_dict.items() if col in df.columns}
    styler = styler.format(format_dict)
    
    # Add styles for boolean columns
    if 'threshold_significant' in df.columns:
        styler = styler.apply(
            lambda x: ['background-color: #d4f1d4' if x else 'background-color: #f1d4d4' 
                      for x in df['threshold_significant']],
            subset=['threshold_significant']
        )
    
    if 'asymmetry_significant' in df.columns:
        styler = styler.apply(
            lambda x: ['background-color: #d4f1d4' if x else 'background-color: #f1d4d4' 
                      for x in df['asymmetry_significant']],
            subset=['asymmetry_significant']
        )
    
    # Add colormap for integration index
    if 'integration_index' in df.columns:
        styler = styler.background_gradient(
            cmap='RdYlGn', 
            subset=['integration_index'],
            vmin=0, 
            vmax=1
        )
    
    # Add caption
    styler = styler.set_caption(title)
    
    # Generate HTML
    html = f"""
    <div class="summary-table">
        <h2>{title}</h2>
        <p>{description}</p>
        {styler.to_html()}
    </div>
    """
    
    return html