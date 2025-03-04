"""
Economic interpretations for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

from core.decorators import error_handler, performance_tracker
from core.exceptions import AnalysisError

logger = logging.getLogger(__name__)


@error_handler(fallback_value={})
@performance_tracker()
def interpret_threshold_economics(
    threshold: float,
    commodity: str,
    model_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Provide economic interpretation of threshold value.
    
    Args:
        threshold: Estimated threshold value
        commodity: Commodity name
        model_results: Model estimation results
        
    Returns:
        Dictionary with economic interpretation
    """
    # Define commodity-specific information
    commodity_info = {
        "wheat": {
            "expected_transport_cost": 0.05,  # 5% of price
            "tradability": "high",
            "storage_capacity": "medium",
            "value_to_weight": "medium"
        },
        "rice": {
            "expected_transport_cost": 0.05,  # 5% of price
            "tradability": "high",
            "storage_capacity": "high",
            "value_to_weight": "medium"
        },
        "beans (kidney red)": {
            "expected_transport_cost": 0.07,  # 7% of price
            "tradability": "medium",
            "storage_capacity": "high", 
            "value_to_weight": "medium"
        },
        "beans (white)": {
            "expected_transport_cost": 0.07,  # 7% of price
            "tradability": "medium",
            "storage_capacity": "high",
            "value_to_weight": "medium"
        },
        "flour": {
            "expected_transport_cost": 0.06,  # 6% of price
            "tradability": "high",
            "storage_capacity": "medium",
            "value_to_weight": "low"
        },
        "sugar": {
            "expected_transport_cost": 0.04,  # 4% of price
            "tradability": "high",
            "storage_capacity": "high",
            "value_to_weight": "medium"
        },
        "oil (vegetable)": {
            "expected_transport_cost": 0.03,  # 3% of price
            "tradability": "high",
            "storage_capacity": "high",
            "value_to_weight": "high"
        },
        "fuel (diesel)": {
            "expected_transport_cost": 0.02,  # 2% of price
            "tradability": "high",
            "storage_capacity": "medium",
            "value_to_weight": "high"
        },
        "exchange rate": {
            "expected_transport_cost": 0.005,  # 0.5% of price
            "tradability": "very high",
            "storage_capacity": "very high",
            "value_to_weight": "very high"
        }
    }
    
    # Use default values if commodity not found
    if commodity.lower() not in commodity_info:
        commodity_info[commodity.lower()] = {
            "expected_transport_cost": 0.05,  # 5% of price
            "tradability": "medium",
            "storage_capacity": "medium",
            "value_to_weight": "medium"
        }
    
    # Get commodity-specific information
    info = commodity_info[commodity.lower()]
    expected_threshold = info["expected_transport_cost"]
    
    # Calculate threshold assessment metrics
    threshold_ratio = threshold / expected_threshold
    threshold_deviation = threshold - expected_threshold
    
    # Determine threshold interpretation
    if threshold_ratio < 0.8:
        threshold_assessment = "Lower than expected transport costs"
        barrier_interpretation = "Minimal trade barriers, highly efficient market"
    elif threshold_ratio < 1.2:
        threshold_assessment = "Consistent with transport costs"
        barrier_interpretation = "Normal trade barriers, efficient market"
    elif threshold_ratio < 2.0:
        threshold_assessment = "Higher than expected transport costs"
        barrier_interpretation = "Moderate trade barriers or market inefficiencies"
    elif threshold_ratio < 3.0:
        threshold_assessment = "Substantially higher than transport costs"
        barrier_interpretation = "Significant trade barriers or market segmentation"
    else:
        threshold_assessment = "Extremely high relative to transport costs"
        barrier_interpretation = "Severe market fragmentation, possible trade restrictions"
    
    # Calculate half-lives
    half_life_down = model_results.get("adjustment_dynamics", {}).get("half_life_down", float('inf'))
    half_life_up = model_results.get("adjustment_dynamics", {}).get("half_life_up", float('inf'))
    
    # Interpret adjustment speeds
    if min(half_life_down, half_life_up) < 2:
        adjustment_interpretation = "Very rapid price adjustment"
        market_efficiency = "High market efficiency"
    elif min(half_life_down, half_life_up) < 4:
        adjustment_interpretation = "Rapid price adjustment"
        market_efficiency = "Good market efficiency"
    elif min(half_life_down, half_life_up) < 6:
        adjustment_interpretation = "Moderate price adjustment"
        market_efficiency = "Moderate market efficiency"
    elif min(half_life_down, half_life_up) < 10:
        adjustment_interpretation = "Slow price adjustment"
        market_efficiency = "Low market efficiency"
    else:
        adjustment_interpretation = "Very slow or no price adjustment"
        market_efficiency = "Very low market efficiency"
    
    # Calculate regime balance
    regime_balance = model_results.get("regime_balance", {})
    lower_pct = regime_balance.get("lower", 0.5) * 100
    upper_pct = regime_balance.get("upper", 0.5) * 100
    
    if abs(lower_pct - upper_pct) < 10:
        balance_interpretation = "Balanced regimes, symmetric price patterns"
    elif lower_pct > upper_pct:
        balance_interpretation = f"Lower regime dominant ({lower_pct:.1f}%), prices often below equilibrium"
    else:
        balance_interpretation = f"Upper regime dominant ({upper_pct:.1f}%), prices often above equilibrium"
    
    # Combine results
    results = {
        "commodity": commodity,
        "threshold": float(threshold),
        "expected_transport_cost": float(expected_threshold),
        "threshold_ratio": float(threshold_ratio),
        "threshold_deviation": float(threshold_deviation),
        "threshold_assessment": threshold_assessment,
        "barrier_interpretation": barrier_interpretation,
        "adjustment": {
            "half_life_down": float(half_life_down),
            "half_life_up": float(half_life_up),
            "interpretation": adjustment_interpretation,
            "market_efficiency": market_efficiency
        },
        "regime_balance": {
            "lower_percentage": float(lower_pct),
            "upper_percentage": float(upper_pct),
            "interpretation": balance_interpretation
        },
        "commodity_info": info
    }
    
    return results


@error_handler(fallback_value={})
@performance_tracker()
def calculate_price_transmission_elasticity(
    model_results: Dict[str, Any],
    north_prices: pd.Series,
    south_prices: pd.Series
) -> Dict[str, Any]:
    """
    Calculate price transmission elasticity between markets.
    
    Args:
        model_results: Model estimation results
        north_prices: North market price series
        south_prices: South market price series
        
    Returns:
        Dictionary with price transmission analysis
    """
    # Extract key parameters
    commodity = model_results.get("commodity")
    threshold = model_results.get("threshold", 0.0)
    beta = model_results.get("beta", 1.0)
    
    # Create log-transformed prices
    if not north_prices.empty and not south_prices.empty:
        log_north = np.log(north_prices)
        log_south = np.log(south_prices)
        
        # Calculate first differences
        dlog_north = log_north.diff().dropna()
        dlog_south = log_south.diff().dropna()
        
        # Align series
        common_index = dlog_north.index.intersection(dlog_south.index)
        dlog_north = dlog_north.loc[common_index]
        dlog_south = dlog_south.loc[common_index]
        
        # Calculate price differential
        price_diff = (north_prices.loc[common_index] - south_prices.loc[common_index]) / south_prices.loc[common_index]
        
        # Create regime indicators
        upper_regime = price_diff > threshold
        lower_regime = price_diff < -threshold
        band_regime = ~(upper_regime | lower_regime)
        
        # Estimate elasticities for each regime
        if len(common_index) > 0:
            # Full sample elasticity
            full_elasticity, full_rsquared = _estimate_elasticity(dlog_south, dlog_north)
            
            # Upper regime elasticity
            if upper_regime.sum() > 3:
                upper_elasticity, upper_rsquared = _estimate_elasticity(
                    dlog_south[upper_regime], dlog_north[upper_regime]
                )
            else:
                upper_elasticity, upper_rsquared = np.nan, np.nan
            
            # Lower regime elasticity
            if lower_regime.sum() > 3:
                lower_elasticity, lower_rsquared = _estimate_elasticity(
                    dlog_south[lower_regime], dlog_north[lower_regime]
                )
            else:
                lower_elasticity, lower_rsquared = np.nan, np.nan
            
            # Band regime elasticity
            if band_regime.sum() > 3:
                band_elasticity, band_rsquared = _estimate_elasticity(
                    dlog_south[band_regime], dlog_north[band_regime]
                )
            else:
                band_elasticity, band_rsquared = np.nan, np.nan
        else:
            full_elasticity = upper_elasticity = lower_elasticity = band_elasticity = np.nan
            full_rsquared = upper_rsquared = lower_rsquared = band_rsquared = np.nan
    else:
        full_elasticity = upper_elasticity = lower_elasticity = band_elasticity = np.nan
        full_rsquared = upper_rsquared = lower_rsquared = band_rsquared = np.nan
    
    # Interpret elasticities
    if np.isnan(full_elasticity):
        full_interpretation = "Insufficient data"
    elif full_elasticity > 0.8:
        full_interpretation = "High price transmission"
    elif full_elasticity > 0.5:
        full_interpretation = "Moderate price transmission"
    elif full_elasticity > 0.2:
        full_interpretation = "Low price transmission"
    else:
        full_interpretation = "Very low or no price transmission"
    
    # Combine results
    results = {
        "commodity": commodity,
        "elasticities": {
            "full_sample": float(full_elasticity),
            "upper_regime": float(upper_elasticity),
            "lower_regime": float(lower_elasticity),
            "band_regime": float(band_elasticity)
        },
        "r_squared": {
            "full_sample": float(full_rsquared),
            "upper_regime": float(upper_rsquared),
            "lower_regime": float(lower_rsquared),
            "band_regime": float(band_rsquared)
        },
        "regimes_count": {
            "upper_regime": int(upper_regime.sum()) if hasattr(upper_regime, 'sum') else 0,
            "lower_regime": int(lower_regime.sum()) if hasattr(lower_regime, 'sum') else 0,
            "band_regime": int(band_regime.sum()) if hasattr(band_regime, 'sum') else 0
        },
        "interpretation": {
            "full_sample": full_interpretation
        }
    }
    
    return results


@error_handler(fallback_value=(0.0, 0.0))
def _estimate_elasticity(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """
    Estimate elasticity between two series.
    
    Args:
        x: Independent variable
        y: Dependent variable
        
    Returns:
        Tuple of (elasticity, r_squared)
    """
    if len(x) < 4 or len(y) < 4:
        return 0.0, 0.0
    
    try:
        import statsmodels.api as sm
        
        # Add constant
        X = sm.add_constant(x)
        
        # Estimate model
        model = sm.OLS(y, X).fit()
        
        # Extract coefficients
        elasticity = model.params[1]
        r_squared = model.rsquared
        
        return elasticity, r_squared
    except Exception as e:
        logger.warning(f"Error estimating elasticity: {str(e)}")
        return 0.0, 0.0


@error_handler(fallback_value={})
@performance_tracker()
def analyze_arbitrage_profitability(
    north_prices: pd.Series,
    south_prices: pd.Series,
    threshold: float,
    commodity: str,
    transport_cost_pct: float = 0.05
) -> Dict[str, Any]:
    """
    Analyze profitability of arbitrage operations.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        threshold: Estimated threshold value
        commodity: Commodity name
        transport_cost_pct: Transport cost as percentage of price
        
    Returns:
        Dictionary with arbitrage profitability analysis
    """
    if north_prices.empty or south_prices.empty:
        return {"error": "Empty price series"}
    
    # Create aligned price series
    common_index = north_prices.index.intersection(south_prices.index)
    
    if len(common_index) == 0:
        return {"error": "No common dates between price series"}
    
    north = north_prices.loc[common_index]
    south = south_prices.loc[common_index]
    
    # Calculate price differential
    price_diff = north - south
    price_diff_pct = price_diff / pd.concat([north, south], axis=1).min(axis=1)
    
    # Define arbitrage opportunities
    north_to_south = price_diff_pct < -threshold  # South price higher
    south_to_north = price_diff_pct > threshold   # North price higher
    
    # Calculate potential profit after transport costs
    north_to_south_profit = (-price_diff_pct - transport_cost_pct)[north_to_south]
    south_to_north_profit = (price_diff_pct - transport_cost_pct)[south_to_north]
    
    # Calculate arbitrage frequency
    n2s_freq = north_to_south.mean() * 100
    s2n_freq = south_to_north.mean() * 100
    
    # Calculate profit statistics
    if len(north_to_south_profit) > 0:
        n2s_profit_mean = north_to_south_profit.mean() * 100
        n2s_profit_max = north_to_south_profit.max() * 100
    else:
        n2s_profit_mean = n2s_profit_max = 0.0
    
    if len(south_to_north_profit) > 0:
        s2n_profit_mean = south_to_north_profit.mean() * 100
        s2n_profit_max = south_to_north_profit.max() * 100
    else:
        s2n_profit_mean = s2n_profit_max = 0.0
    
    # Determine dominant direction
    if n2s_freq > s2n_freq:
        dominant_direction = "north_to_south"
        dominant_freq = n2s_freq
        dominant_profit = n2s_profit_mean
    elif s2n_freq > n2s_freq:
        dominant_direction = "south_to_north"
        dominant_freq = s2n_freq
        dominant_profit = s2n_profit_mean
    else:
        dominant_direction = "balanced"
        dominant_freq = (n2s_freq + s2n_freq) / 2
        dominant_profit = (n2s_profit_mean + s2n_profit_mean) / 2
    
    # Calculate persistence
    n2s_persistence = _calculate_persistence(north_to_south)
    s2n_persistence = _calculate_persistence(south_to_north)
    
    # Combine results
    results = {
        "commodity": commodity,
        "threshold": float(threshold),
        "transport_cost_pct": float(transport_cost_pct),
        "north_to_south": {
            "frequency": float(n2s_freq),
            "avg_profit_pct": float(n2s_profit_mean),
            "max_profit_pct": float(n2s_profit_max),
            "persistence": float(n2s_persistence)
        },
        "south_to_north": {
            "frequency": float(s2n_freq),
            "avg_profit_pct": float(s2n_profit_mean),
            "max_profit_pct": float(s2n_profit_max),
            "persistence": float(s2n_persistence)
        },
        "dominant": {
            "direction": dominant_direction,
            "frequency": float(dominant_freq),
            "avg_profit_pct": float(dominant_profit)
        }
    }
    
    return results


@error_handler(fallback_value=1.0)
def _calculate_persistence(series: pd.Series) -> float:
    """
    Calculate the average persistence of True values in a boolean series.
    
    Args:
        series: Boolean series
        
    Returns:
        Average persistence in periods
    """
    if not isinstance(series, pd.Series) or len(series) == 0:
        return 1.0
    
    # Create runs
    runs = (series != series.shift()).cumsum()
    
    # Calculate run lengths
    run_lengths = series.groupby(runs).transform('size')
    
    # Filter to only include True runs
    true_runs = run_lengths[series]
    
    if len(true_runs) == 0:
        return 1.0
    
    return true_runs.mean()