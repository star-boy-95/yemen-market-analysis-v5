"""
Common threshold utilities for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

from core.decorators import error_handler, performance_tracker
from core.exceptions import ThresholdModelError

logger = logging.getLogger(__name__)


@error_handler(fallback_value=float('inf'))
def calculate_adjustment_half_life(alpha: float) -> float:
    """Calculate half-life of deviation based on adjustment parameter."""
    if alpha is None or alpha >= 0:
        return float('inf')
    
    # Calculate half-life using log formula
    return np.log(0.5) / np.log(1 + alpha)


@error_handler(fallback_value=(False, 1.0))
def test_threshold_symmetry(
    alpha_up: float,
    alpha_down: float,
    alpha_up_se: float,
    alpha_down_se: float,
    significance_level: float = 0.05
) -> Tuple[bool, float]:
    """Test for symmetry in threshold adjustment parameters."""
    # Calculate absolute difference
    diff = abs(alpha_up - alpha_down)
    
    # Calculate standard error of difference
    diff_se = np.sqrt(alpha_up_se**2 + alpha_down_se**2)
    
    # Calculate t-statistic
    t_stat = diff / diff_se if diff_se > 0 else 0
    
    # Calculate p-value (two-sided test)
    from scipy import stats
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), np.inf))
    
    # Test significance
    asymmetry_significant = p_value < significance_level
    
    return asymmetry_significant, p_value


@error_handler(fallback_value={})
def calculate_market_integration_index(
    price_differential: float,
    adjustment_speed: float,
    arbitrage_frequency: float,
    transport_costs: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate market integration index based on key metrics.
    
    Args:
        price_differential: Average price difference as percentage
        adjustment_speed: Adjustment parameter (negative for mean reversion)
        arbitrage_frequency: Frequency of arbitrage opportunities
        transport_costs: Estimated transport costs as percentage
        
    Returns:
        Dictionary with integration index and component scores
    """
    # Normalize components to 0-1 scale
    price_diff_norm = max(0, 1 - (price_differential / (transport_costs * 2)))
    
    if adjustment_speed < 0:
        half_life = calculate_adjustment_half_life(adjustment_speed)
        adj_speed_norm = max(0, 1 - (half_life - 1) / 11) if half_life < 12 else 0
    else:
        adj_speed_norm = 0
        
    arb_freq_norm = 1 - min(1, arbitrage_frequency)
    
    # Apply weights
    weights = {"price_differential": 0.4, "adjustment_speed": 0.4, "arbitrage_frequency": 0.2}
    
    component_scores = {
        "price_differential": price_diff_norm,
        "adjustment_speed": adj_speed_norm,
        "arbitrage_frequency": arb_freq_norm
    }
    
    # Calculate integration index (0-1 scale)
    integration_index = sum(weights[k] * component_scores[k] for k in weights)
    
    # Categorize integration level
    if integration_index >= 0.7:
        integration_level = "High"
    elif integration_index >= 0.4:
        integration_level = "Moderate"
    else:
        integration_level = "Low"
    
    return {
        "integration_index": float(integration_index),
        "integration_level": integration_level,
        "component_scores": component_scores
    }


@error_handler(fallback_value={})
def generate_policy_implications(
    threshold_value: float,
    diagnostics: Dict[str, Any],
    model_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate policy implications based on threshold model diagnostics.
    
    Args:
        threshold_value: Estimated threshold value
        diagnostics: Model diagnostics
        model_results: Model estimation results
        
    Returns:
        Dictionary with policy implications and recommendations
    """
    # Extract key metrics
    integration_index = diagnostics.get("integration_index", 0.0)
    integration_level = diagnostics.get("integration_level", "Unknown")
    
    alpha_down = model_results.get("adjustment_dynamics", {}).get("alpha_down", 0)
    alpha_up = model_results.get("adjustment_dynamics", {}).get("alpha_up", 0)
    
    alpha_ratio = (abs(alpha_up) / abs(alpha_down)) if abs(alpha_down) > 0 else float('inf')
    if alpha_ratio < 1:
        alpha_ratio = 1 / alpha_ratio
        faster = "lower"
    else:
        faster = "upper"
    
    # Determine policy priority
    if integration_level == "Low":
        priority = "High"
        explanation = "Markets show weak integration requiring immediate policy attention."
    elif integration_level == "Moderate" and alpha_ratio > 2:
        priority = "Medium-High"
        explanation = "Markets show moderate integration but significant asymmetry in adjustment speeds."
    elif integration_level == "Moderate":
        priority = "Medium"
        explanation = "Markets show moderate integration with relatively symmetric adjustment."
    else:
        priority = "Low"
        explanation = "Markets show high integration with efficient price transmission."
    
    # Generate recommendations
    recommendations = []
    
    if integration_level in ["Low", "Moderate"]:
        recommendations.append("Invest in transportation infrastructure to reduce trade barriers.")
    
    if alpha_ratio > 2:
        slower = "lower" if faster == "upper" else "upper"
        recommendations.append(
            f"Improve market information systems in the {slower} regime to accelerate price adjustment."
        )
    
    if integration_level == "Low":
        recommendations.append("Support mobile banking and digital payments to reduce transaction costs.")
        recommendations.append("Facilitate trader financing to enable arbitrage activity.")
    
    if threshold_value > 0.2:
        recommendations.append("Consider strategic reserves in deficit areas to reduce price spikes.")
    
    return {
        "policy_priority": priority,
        "explanation": explanation,
        "recommendations": recommendations,
        "key_metrics": {
            "integration_level": integration_level,
            "integration_index": float(integration_index),
            "adjustment_asymmetry": float(alpha_ratio),
            "faster_regime": faster,
            "threshold_value": float(threshold_value)
        }
    }


@error_handler(fallback_value=(None, None))
def calculate_welfare_effects(
    north_prices: pd.Series, 
    south_prices: pd.Series, 
    threshold: float,
    demand_elasticity: float = -0.7
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate welfare effects based on price differentials and arbitrage opportunities.
    
    Args:
        north_prices: Price series for northern markets
        south_prices: Price series for southern markets
        threshold: Estimated threshold value
        demand_elasticity: Price elasticity of demand
        
    Returns:
        Tuple of (total_deadweight_loss, detailed_results)
    """
    if north_prices.empty or south_prices.empty:
        return 0.0, {"error": "Empty price series"}
    
    # Align series
    aligned = pd.DataFrame({
        'north': north_prices,
        'south': south_prices
    }).dropna()
    
    if aligned.empty:
        return 0.0, {"error": "No common dates in price series"}
    
    # Calculate price differential as percentage
    price_diff = aligned['north'] - aligned['south']
    price_diff_pct = price_diff / aligned[['north', 'south']].min(axis=1)
    
    # Identify periods with arbitrage opportunities
    arbitrage_periods = abs(price_diff_pct) > threshold
    arbitrage_freq = arbitrage_periods.mean()
    
    # Calculate welfare losses using simplified Harberger triangle approach
    deadweight_losses = []
    
    for idx in aligned[arbitrage_periods].index:
        north_price = aligned.loc[idx, 'north']
        south_price = aligned.loc[idx, 'south']
        diff = abs(north_price - south_price)
        
        # Effective price difference after accounting for threshold
        min_price = min(north_price, south_price)
        effective_diff = diff - (min_price * threshold)
        
        if effective_diff <= 0:
            continue
            
        # Calculate approximate quantity impact using elasticity
        avg_price = (north_price + south_price) / 2
        q_impact = effective_diff / avg_price * abs(demand_elasticity)
        
        # Calculate deadweight loss (1/2 × price diff × quantity effect)
        dwl = 0.5 * effective_diff * q_impact * avg_price
        deadweight_losses.append(dwl)
    
    # Calculate total deadweight loss
    total_dwl = sum(deadweight_losses)
    
    # Calculate as percentage of market value
    total_market_value = aligned[['north', 'south']].mean(axis=1).sum()
    dwl_pct = (total_dwl / total_market_value) * 100 if total_market_value > 0 else 0
    
    results = {
        "total_deadweight_loss": float(total_dwl),
        "dwl_percent_of_market": float(dwl_pct),
        "arbitrage_frequency": float(arbitrage_freq),
        "arbitrage_days": int(arbitrage_periods.sum()),
        "average_price_differential": float(abs(price_diff_pct).mean() * 100),
        "max_price_differential": float(abs(price_diff_pct).max() * 100)
    }
    
    return total_dwl, results


@error_handler(fallback_value=None)
def adaptive_threshold_selection(
    residuals: np.ndarray,
    test_function: Callable[[float], float],
    min_regime_size: float = 0.1,
    grid_points: int = 50,
    threshold_range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """
    Select threshold adaptively based on test statistic.
    
    Args:
        residuals: Cointegrating residuals
        test_function: Function that evaluates a threshold value and returns test statistic
        min_regime_size: Minimum proportion of observations in each regime
        grid_points: Number of grid points to evaluate
        threshold_range: Optional explicit range for threshold search
        
    Returns:
        Tuple of (optimal_threshold, test_statistic)
    """
    # Clean residuals
    clean_res = residuals[~np.isnan(residuals)]
    
    if len(clean_res) == 0:
        return 0.0, 0.0
    
    # Set up threshold range
    if threshold_range is None:
        # Sort residuals for percentile-based grid
        z_sorted = np.sort(clean_res)
        n = len(z_sorted)
        
        # Trim to ensure minimum regime size
        lower_idx = int(n * min_regime_size)
        upper_idx = int(n * (1 - min_regime_size))
        
        # Create grid of threshold values
        grid_values = np.linspace(z_sorted[lower_idx], z_sorted[upper_idx], grid_points)
    else:
        # Use provided threshold range
        grid_values = np.linspace(threshold_range[0], threshold_range[1], grid_points)
    
    # Initialize results
    best_test_stat = -np.inf
    best_threshold = None
    
    # Run grid search
    for gamma in grid_values:
        # Check regime balance
        regime = (clean_res > gamma).astype(int)
        reg0_pct = np.mean(regime == 0)
        reg1_pct = np.mean(regime == 1)
        
        if reg0_pct < min_regime_size or reg1_pct < min_regime_size:
            continue
        
        # Evaluate threshold
        try:
            test_stat = test_function(gamma)
            
            # Update best threshold
            if test_stat > best_test_stat:
                best_test_stat = test_stat
                best_threshold = gamma
        except:
            continue
    
    # If no valid threshold found, use median
    if best_threshold is None:
        best_threshold = np.median(clean_res)
        best_test_stat = 0.0
        
    return best_threshold, best_test_stat