"""
Welfare impact analysis for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

from core.decorators import error_handler, performance_tracker
from core.exceptions import AnalysisError
from models.threshold import calculate_welfare_effects

logger = logging.getLogger(__name__)


@error_handler(fallback_value={})
@performance_tracker()
def calculate_welfare_impact(
    north_prices: pd.Series,
    south_prices: pd.Series,
    threshold: float,
    commodity: str,
    demand_elasticity: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate welfare impact of market fragmentation.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        threshold: Estimated threshold value
        commodity: Commodity name
        demand_elasticity: Price elasticity of demand (commodity-specific if provided)
        
    Returns:
        Dictionary with welfare impact analysis
    """
    # Set commodity-specific elasticities if not provided
    if demand_elasticity is None:
        # Default elasticities based on commodity type
        elasticities = {
            'fuel (diesel)': -0.3,
            'wheat': -0.6,
            'rice': -0.5,
            'beans (kidney red)': -0.7,
            'beans (white)': -0.7,
            'flour': -0.6,
            'sugar': -0.4,
            'oil (vegetable)': -0.5,
            'exchange rate': -1.5
        }
        # Use default if commodity not found
        demand_elasticity = elasticities.get(commodity.lower(), -0.7)
    
    # Calculate welfare effects
    total_dwl, welfare_results = calculate_welfare_effects(
        north_prices, south_prices, threshold, demand_elasticity
    )
    
    # Analyze welfare distribution
    north_higher_pct = (north_prices > south_prices).mean() * 100
    south_higher_pct = (south_prices > north_prices).mean() * 100
    
    # Calculate directional price effects
    if north_higher_pct > south_higher_pct:
        price_direction = "north_higher"
        price_difference = "North prices are higher than South prices"
    elif south_higher_pct > north_higher_pct:
        price_direction = "south_higher"
        price_difference = "South prices are higher than North prices"
    else:
        price_direction = "balanced"
        price_difference = "No consistent price difference between regions"
    
    # Determine market power dynamics
    if price_direction == "north_higher":
        market_power = "South market consumers gain, North producers gain"
        recommendation = "Focus on reducing barriers to south-to-north trade"
    elif price_direction == "south_higher":
        market_power = "North market consumers gain, South producers gain"
        recommendation = "Focus on reducing barriers to north-to-south trade"
    else:
        market_power = "No consistent welfare transfer direction"
        recommendation = "Focus on reducing overall transaction costs"
    
    # Calculate welfare losses as percentage of GDP
    # Assuming a rough estimate of market size as proportion of GDP
    market_size_pct = {
        'fuel (diesel)': 0.12,
        'wheat': 0.04,
        'rice': 0.03,
        'beans (kidney red)': 0.01,
        'beans (white)': 0.01,
        'flour': 0.03,
        'sugar': 0.02,
        'oil (vegetable)': 0.02,
        'exchange rate': 0.20
    }
    
    default_market_size = 0.02  # 2% of GDP
    market_pct = market_size_pct.get(commodity.lower(), default_market_size)
    gdp_loss_pct = welfare_results.get("dwl_percent_of_market", 0.0) * market_pct
    
    # Analyze welfare impact severity
    if gdp_loss_pct > 0.1:
        severity = "High"
    elif gdp_loss_pct > 0.05:
        severity = "Moderate"
    elif gdp_loss_pct > 0.01:
        severity = "Low"
    else:
        severity = "Negligible"
    
    # Return comprehensive analysis
    return {
        "commodity": commodity,
        "total_deadweight_loss": float(total_dwl),
        "dwl_percent_of_market": float(welfare_results.get("dwl_percent_of_market", 0.0)),
        "estimated_gdp_impact_pct": float(gdp_loss_pct),
        "arbitrage_frequency": float(welfare_results.get("arbitrage_frequency", 0.0)),
        "arbitrage_days": int(welfare_results.get("arbitrage_days", 0)),
        "average_price_differential": float(welfare_results.get("average_price_differential", 0.0)),
        "max_price_differential": float(welfare_results.get("max_price_differential", 0.0)),
        "north_higher_pct": float(north_higher_pct),
        "south_higher_pct": float(south_higher_pct),
        "price_direction": price_direction,
        "price_difference": price_difference,
        "market_power_dynamics": market_power,
        "welfare_impact_severity": severity,
        "policy_recommendation": recommendation,
        "demand_elasticity": float(demand_elasticity)
    }


@error_handler(fallback_value={})
@performance_tracker()
def analyze_welfare_distribution(
    df: pd.DataFrame,
    commodity: str,
    threshold: float,
    price_col: str = 'usdprice'
) -> Dict[str, Any]:
    """
    Analyze the distribution of welfare impacts across different regions.
    
    Args:
        df: Input DataFrame with market data
        commodity: Commodity name
        threshold: Threshold value from model
        price_col: Price column name
        
    Returns:
        Dictionary with welfare distribution analysis
    """
    # Filter for specific commodity
    commodity_df = df[df['commodity'] == commodity].copy()
    
    if commodity_df.empty:
        return {"error": f"No data for {commodity}"}
    
    # Calculate regional price averages
    north_df = commodity_df[commodity_df['exchange_rate_regime'] == 'north']
    south_df = commodity_df[commodity_df['exchange_rate_regime'] == 'south']
    
    # Group by admin1 region to get regional average prices
    north_regions = north_df.groupby('admin1')[price_col].mean().to_dict()
    south_regions = south_df.groupby('admin1')[price_col].mean().to_dict()
    
    # Calculate regional price differentials
    regional_diffs = {}
    for region in set(north_regions.keys()) & set(south_regions.keys()):
        price_diff = (north_regions[region] - south_regions[region]) / min(north_regions[region], south_regions[region])
        regional_diffs[region] = price_diff * 100  # as percentage
    
    # Identify regions with arbitrage opportunities
    arbitrage_regions = {region: diff for region, diff in regional_diffs.items() if abs(diff) > threshold * 100}
    
    # Sort regions by price differential magnitude
    sorted_regions = sorted(regional_diffs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Calculate regional welfare impacts
    regional_welfare = {}
    for region, diff in sorted_regions:
        if abs(diff) <= threshold * 100:
            continue
            
        # Simplified welfare impact calculation
        # Assume welfare loss is proportional to (diff - threshold)^2
        welfare_loss = (abs(diff) - threshold * 100) ** 2 / 100  # Normalized measure
        direction = "north_to_south" if diff > 0 else "south_to_north"
        
        regional_welfare[region] = {
            "price_diff_pct": float(diff),
            "welfare_loss_index": float(welfare_loss),
            "arbitrage_direction": direction,
            "exceeds_threshold": abs(diff) > threshold * 100
        }
    
    # Identify priority regions for intervention
    if regional_welfare:
        priority_regions = sorted(
            regional_welfare.items(), 
            key=lambda x: x[1]['welfare_loss_index'], 
            reverse=True
        )[:3]
    else:
        priority_regions = []
    
    return {
        "commodity": commodity,
        "threshold": float(threshold * 100),  # as percentage
        "regional_price_differentials": regional_diffs,
        "arbitrage_regions_count": len(arbitrage_regions),
        "regions_with_welfare_impacts": len(regional_welfare),
        "regional_welfare": regional_welfare,
        "priority_regions": [r[0] for r in priority_regions],
        "max_regional_diff": float(max(abs(d) for d in regional_diffs.values())) if regional_diffs else 0.0
    }


@error_handler(fallback_value={})
@performance_tracker()
def simulate_policy_impacts(
    north_prices: pd.Series,
    south_prices: pd.Series,
    threshold: float,
    threshold_reduction: float = 0.5,  # 50% reduction in threshold
    commodity: str = "unknown"
) -> Dict[str, Any]:
    """
    Simulate welfare impacts of policy interventions that reduce thresholds.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        threshold: Current threshold value
        threshold_reduction: Percentage reduction in threshold from policy
        commodity: Commodity name
        
    Returns:
        Dictionary with policy simulation results
    """
    if north_prices.empty or south_prices.empty:
        return {"error": "Empty price series"}
    
    # Calculate current welfare impact
    current_dwl, current_results = calculate_welfare_effects(
        north_prices, south_prices, threshold
    )
    
    # Calculate welfare impact with reduced threshold
    reduced_threshold = threshold * (1 - threshold_reduction)
    new_dwl, new_results = calculate_welfare_effects(
        north_prices, south_prices, reduced_threshold
    )
    
    # Calculate welfare gains
    welfare_gain = current_dwl - new_dwl
    welfare_gain_pct = (welfare_gain / current_dwl) * 100 if current_dwl > 0 else 0.0
    
    # Calculate change in arbitrage frequency
    current_arbitrage = current_results.get("arbitrage_frequency", 0.0)
    new_arbitrage = new_results.get("arbitrage_frequency", 0.0)
    arbitrage_increase = new_arbitrage - current_arbitrage
    
    # Simulate trade volume increase (simplified model)
    # Assume trade volume increases proportionally to arbitrage frequency
    trade_volume_increase = arbitrage_increase * 2  # Simplified multiplier
    
    return {
        "commodity": commodity,
        "current_threshold": float(threshold),
        "reduced_threshold": float(reduced_threshold),
        "threshold_reduction_pct": float(threshold_reduction * 100),
        "current_deadweight_loss": float(current_dwl),
        "new_deadweight_loss": float(new_dwl),
        "welfare_gain": float(welfare_gain),
        "welfare_gain_pct": float(welfare_gain_pct),
        "current_arbitrage_frequency": float(current_arbitrage),
        "new_arbitrage_frequency": float(new_arbitrage),
        "arbitrage_increase": float(arbitrage_increase),
        "estimated_trade_volume_increase_pct": float(trade_volume_increase),
        "policy_effectiveness": "High" if welfare_gain_pct > 30 else 
                               "Medium" if welfare_gain_pct > 10 else 
                               "Low"
    }