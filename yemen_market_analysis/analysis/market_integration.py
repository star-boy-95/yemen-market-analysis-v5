"""
Market integration analysis for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

from core.decorators import error_handler, performance_tracker
from core.exceptions import AnalysisError
from models.threshold import (
    calculate_adjustment_half_life, test_threshold_symmetry,
    calculate_market_integration_index, generate_policy_implications,
    calculate_welfare_effects
)
from models.statistics import (
    run_bidirectional_granger_causality, run_all_nonlinearity_tests
)

logger = logging.getLogger(__name__)


@error_handler(fallback_value={})
@performance_tracker()
def analyze_market_integration(
    model_results: Dict[str, Any],
    price_diff_df: pd.DataFrame,
    commodity: str
) -> Dict[str, Any]:
    """
    Analyze market integration based on threshold model results.
    
    Args:
        model_results: Threshold model estimation results
        price_diff_df: DataFrame with price differentials
        commodity: Commodity name
        
    Returns:
        Dictionary with market integration analysis
    """
    # Extract key parameters
    threshold = model_results.get('threshold', 0.0)
    alpha_down = model_results.get('adjustment_dynamics', {}).get('alpha_down', 0.0)
    alpha_up = model_results.get('adjustment_dynamics', {}).get('alpha_up', 0.0)
    
    # Calculate price differential metrics
    if price_diff_df is not None and not price_diff_df.empty:
        # Average price differential (percentage)
        avg_diff_pct = price_diff_df['diff_pct'].abs().mean()
        
        # Calculate arbitrage frequency (percentage price diff > threshold)
        arbitrage_freq = (price_diff_df['diff_pct'].abs() > threshold).mean()
        
        # Calculate volatility of price differential
        diff_volatility = price_diff_df['diff_pct'].std()
    else:
        # Default values if no price differential data
        avg_diff_pct = 0.0
        arbitrage_freq = 0.0
        diff_volatility = 0.0
    
    # Calculate market integration index
    integration_results = calculate_market_integration_index(
        price_differential=avg_diff_pct,
        adjustment_speed=min(alpha_down, alpha_up),
        arbitrage_frequency=arbitrage_freq
    )
    
    # Generate policy implications
    policy_results = generate_policy_implications(
        threshold_value=threshold,
        diagnostics=integration_results,
        model_results=model_results
    )
    
    # Calculate half-lives
    half_life_down = calculate_adjustment_half_life(alpha_down)
    half_life_up = calculate_adjustment_half_life(alpha_up)
    
    # Perform nonlinearity tests on price differential
    if price_diff_df is not None and not price_diff_df.empty:
        nonlinearity_results = run_all_nonlinearity_tests(
            price_diff_df['diff'].values, ar_order=2
        )
    else:
        nonlinearity_results = {"nonlinearity": False, "evidence": "No data available"}
    
    # Combine results
    results = {
        "commodity": commodity,
        "threshold": float(threshold),
        "avg_price_differential_pct": float(avg_diff_pct),
        "arbitrage_frequency": float(arbitrage_freq),
        "price_diff_volatility": float(diff_volatility),
        "adjustment_speed": {
            "alpha_down": float(alpha_down),
            "alpha_up": float(alpha_up),
            "half_life_down": float(half_life_down),
            "half_life_up": float(half_life_up),
            "faster_regime": "lower" if abs(alpha_down) > abs(alpha_up) else "upper"
        },
        "integration": integration_results,
        "policy": policy_results,
        "nonlinearity": nonlinearity_results
    }
    
    return results


@error_handler(fallback_value={})
@performance_tracker()
def analyze_price_leadership(
    north_prices: pd.Series,
    south_prices: pd.Series,
    commodity: str,
    max_lags: int = 4
) -> Dict[str, Any]:
    """
    Analyze price leadership direction using Granger causality.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        commodity: Commodity name
        max_lags: Maximum lag order for causality tests
        
    Returns:
        Dictionary with price leadership analysis
    """
    # Run bidirectional Granger causality tests
    causality_results = run_bidirectional_granger_causality(
        north_prices, south_prices, max_lags=max_lags
    )
    
    # Determine price leadership
    dominant_direction = causality_results.get('dominant_direction', 'none')
    
    if dominant_direction == 'north_to_south' or dominant_direction == 'north_to_south_stronger':
        leader = "north"
        follower = "south"
        strength = "strong" if dominant_direction == 'north_to_south' else "moderate"
    elif dominant_direction == 'south_to_north' or dominant_direction == 'south_to_north_stronger':
        leader = "south"
        follower = "north"
        strength = "strong" if dominant_direction == 'south_to_north' else "moderate"
    else:
        leader = "neither"
        follower = "neither"
        strength = "none"
    
    # Calculate correlations at different lags
    corr_lags = {}
    for lag in range(1, max_lags + 1):
        # North leading (south lagged behind north)
        north_leading = north_prices.iloc[:-lag].reset_index(drop=True)
        south_lagged = south_prices.iloc[lag:].reset_index(drop=True)
        
        if len(north_leading) > 0 and len(south_lagged) > 0:
            corr_north_leading = north_leading.corr(south_lagged)
        else:
            corr_north_leading = np.nan
        
        # South leading (north lagged behind south)
        south_leading = south_prices.iloc[:-lag].reset_index(drop=True)
        north_lagged = north_prices.iloc[lag:].reset_index(drop=True)
        
        if len(south_leading) > 0 and len(north_lagged) > 0:
            corr_south_leading = south_leading.corr(north_lagged)
        else:
            corr_south_leading = np.nan
        
        corr_lags[lag] = {
            "north_leading": float(corr_north_leading),
            "south_leading": float(corr_south_leading)
        }
    
    # Create leadership summary
    leadership_summary = {
        "commodity": commodity,
        "leader": leader,
        "follower": follower,
        "strength": strength,
        "causality": causality_results,
        "correlation_lags": corr_lags
    }
    
    return leadership_summary


@error_handler(fallback_value=(None, {}))
@performance_tracker()
def calculate_market_fragmentation_index(
    north_prices: pd.Series,
    south_prices: pd.Series,
    threshold: float,
    integration_index: float
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate market fragmentation index.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        threshold: Estimated threshold value
        integration_index: Market integration index
        
    Returns:
        Tuple of (fragmentation_index, detailed_results)
    """
    # Calculate average price levels
    north_avg = north_prices.mean()
    south_avg = south_prices.mean()
    
    # Calculate relative price differential
    if min(north_avg, south_avg) > 0:
        price_diff_pct = abs(north_avg - south_avg) / min(north_avg, south_avg) * 100
    else:
        price_diff_pct = 0.0
    
    # Calculate price coefficient of variation in each market
    north_cv = north_prices.std() / north_prices.mean() if north_prices.mean() > 0 else 0.0
    south_cv = south_prices.std() / south_prices.mean() if south_prices.mean() > 0 else 0.0
    
    # Calculate price correlation
    price_correlation = north_prices.corr(south_prices)
    
    # Calculate percentage of periods with arbitrage opportunities
    price_diff_series = (north_prices - south_prices).abs() / pd.concat([north_prices, south_prices], axis=1).min(axis=1)
    arbitrage_pct = (price_diff_series > threshold).mean() * 100
    
    # Calculate fragmentation index (inverse of integration)
    fragmentation_index = 1.0 - integration_index
    
    # Detailed results
    results = {
        "fragmentation_index": float(fragmentation_index),
        "price_differential_pct": float(price_diff_pct),
        "price_correlation": float(price_correlation),
        "north_price_cv": float(north_cv),
        "south_price_cv": float(south_cv),
        "arbitrage_opportunity_pct": float(arbitrage_pct),
        "integration_index": float(integration_index)
    }
    
    return fragmentation_index, results


@error_handler(fallback_value={})
@performance_tracker()
def analyze_welfare_impact(
    model_results: Dict[str, Any],
    north_prices: pd.Series,
    south_prices: pd.Series,
    commodity: str,
    demand_elasticity: float = -0.7
) -> Dict[str, Any]:
    """
    Analyze welfare impact of market fragmentation.
    
    Args:
        model_results: Threshold model estimation results
        north_prices: North market price series
        south_prices: South market price series
        commodity: Commodity name
        demand_elasticity: Price elasticity of demand
        
    Returns:
        Dictionary with welfare impact analysis
    """
    # Extract threshold
    threshold = model_results.get('threshold', 0.0)
    
    # Calculate welfare effects
    total_dwl, welfare_results = calculate_welfare_effects(
        north_prices, south_prices, threshold, demand_elasticity
    )
    
    # Analyze welfare distribution
    north_higher_pct = (north_prices > south_prices).mean() * 100
    south_higher_pct = (south_prices > north_prices).mean() * 100
    
    # Calculate consumer/producer surplus changes
    avg_price_diff = welfare_results.get("average_price_differential", 0.0)
    arbitrage_freq = welfare_results.get("arbitrage_frequency", 0.0)
    
    # Analyze highest/lowest prices to determine surplus direction
    if north_higher_pct > south_higher_pct:
        # North prices are generally higher
        surplus_direction = "North market consumers lose surplus, producers gain"
        net_welfare_impact = "Producer surplus gains in North, consumer surplus gains in South"
    elif south_higher_pct > north_higher_pct:
        # South prices are generally higher
        surplus_direction = "South market consumers lose surplus, producers gain"
        net_welfare_impact = "Producer surplus gains in South, consumer surplus gains in North"
    else:
        # Mixed or balanced
        surplus_direction = "Mixed welfare impacts across markets"
        net_welfare_impact = "No consistent directional welfare impacts"
    
    # Combine results
    results = {
        "commodity": commodity,
        "total_deadweight_loss": float(total_dwl),
        "dwl_percent_of_market": float(welfare_results.get("dwl_percent_of_market", 0.0)),
        "arbitrage_frequency": float(welfare_results.get("arbitrage_frequency", 0.0)),
        "arbitrage_days": int(welfare_results.get("arbitrage_days", 0)),
        "average_price_differential": float(welfare_results.get("average_price_differential", 0.0)),
        "max_price_differential": float(welfare_results.get("max_price_differential", 0.0)),
        "north_higher_pct": float(north_higher_pct),
        "south_higher_pct": float(south_higher_pct),
        "surplus_direction": surplus_direction,
        "net_welfare_impact": net_welfare_impact,
        "demand_elasticity": float(demand_elasticity)
    }
    
    return results