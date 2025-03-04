"""
Conflict impact analysis for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats

from core.decorators import error_handler, performance_tracker
from core.exceptions import AnalysisError
from data.transformers import identify_arbitrage_opportunities

logger = logging.getLogger(__name__)


@error_handler(fallback_value={})
@performance_tracker()
def analyze_conflict_impact(
    model_results: Dict[str, Any],
    price_diff_df: pd.DataFrame,
    conflict_series: pd.Series,
    commodity: str
) -> Dict[str, Any]:
    """
    Analyze the impact of conflict on market integration.
    
    Args:
        model_results: Threshold model estimation results
        price_diff_df: DataFrame with price differentials
        conflict_series: Series with conflict intensity data
        commodity: Commodity name
        
    Returns:
        Dictionary with conflict impact analysis
    """
    if price_diff_df is None or price_diff_df.empty:
        return {"error": "No price differential data available"}
    
    if conflict_series is None or conflict_series.empty:
        return {"error": "No conflict data available"}
    
    # Extract threshold
    threshold = model_results.get('threshold', 0.0)
    
    # Ensure conflict series aligns with price differentials
    common_index = price_diff_df.index.intersection(conflict_series.index)
    
    if len(common_index) == 0:
        return {"error": "No common dates between price data and conflict data"}
    
    # Create aligned data
    conflict_aligned = conflict_series.loc[common_index]
    diff_df_aligned = price_diff_df.loc[common_index].copy()
    
    # Add conflict data to price differentials
    diff_df_aligned['conflict_intensity'] = conflict_aligned.values
    
    # Create high/low conflict periods
    median_conflict = diff_df_aligned['conflict_intensity'].median()
    diff_df_aligned['high_conflict'] = diff_df_aligned['conflict_intensity'] > median_conflict
    
    # Calculate arbitrage opportunities
    diff_df_aligned['arbitrage'] = diff_df_aligned['diff_pct'].abs() > threshold
    
    # Analyze price differentials during high vs low conflict
    high_conflict_df = diff_df_aligned[diff_df_aligned['high_conflict']]
    low_conflict_df = diff_df_aligned[~diff_df_aligned['high_conflict']]
    
    # Calculate key metrics for each period
    high_conflict_metrics = {
        "avg_price_diff_pct": high_conflict_df['diff_pct'].abs().mean() * 100,
        "arbitrage_freq": high_conflict_df['arbitrage'].mean() * 100,
        "price_diff_volatility": high_conflict_df['diff_pct'].std() * 100,
        "observations": len(high_conflict_df)
    }
    
    low_conflict_metrics = {
        "avg_price_diff_pct": low_conflict_df['diff_pct'].abs().mean() * 100,
        "arbitrage_freq": low_conflict_df['arbitrage'].mean() * 100,
        "price_diff_volatility": low_conflict_df['diff_pct'].std() * 100,
        "observations": len(low_conflict_df)
    }
    
    # Statistical tests for differences
    # T-test for price differentials
    if len(high_conflict_df) > 0 and len(low_conflict_df) > 0:
        t_stat, p_value = stats.ttest_ind(
            high_conflict_df['diff_pct'].abs(),
            low_conflict_df['diff_pct'].abs(),
            equal_var=False
        )
        
        diff_significant = p_value < 0.05
        
        # Calculate correlation between conflict and price differentials
        correlation = diff_df_aligned['conflict_intensity'].corr(diff_df_aligned['diff_pct'].abs())
    else:
        t_stat = p_value = correlation = 0.0
        diff_significant = False
    
    # Calculate impact metrics
    if high_conflict_metrics["avg_price_diff_pct"] > 0 and low_conflict_metrics["avg_price_diff_pct"] > 0:
        diff_increase_pct = ((high_conflict_metrics["avg_price_diff_pct"] / 
                             low_conflict_metrics["avg_price_diff_pct"]) - 1) * 100
    else:
        diff_increase_pct = 0.0
    
    arbitrage_increase_pct = (high_conflict_metrics["arbitrage_freq"] - 
                             low_conflict_metrics["arbitrage_freq"])
    
    # Determine impact magnitude
    if diff_significant and diff_increase_pct > 50:
        impact_magnitude = "Strong"
    elif diff_significant and diff_increase_pct > 20:
        impact_magnitude = "Moderate"
    elif diff_significant:
        impact_magnitude = "Weak"
    else:
        impact_magnitude = "Negligible"
    
    # Combine results
    results = {
        "commodity": commodity,
        "high_conflict": high_conflict_metrics,
        "low_conflict": low_conflict_metrics,
        "statistics": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "diff_significant": diff_significant,
            "correlation": float(correlation)
        },
        "impact": {
            "diff_increase_pct": float(diff_increase_pct),
            "arbitrage_increase_pct": float(arbitrage_increase_pct),
            "magnitude": impact_magnitude
        }
    }
    
    return results


@error_handler(fallback_value={})
@performance_tracker()
def analyze_conflict_regime_effects(
    north_prices: pd.Series,
    south_prices: pd.Series,
    conflict_series: pd.Series,
    model_results: Dict[str, Any],
    commodity: str
) -> Dict[str, Any]:
    """
    Analyze how conflict affects different market regimes.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        conflict_series: Series with conflict intensity data
        model_results: Threshold model estimation results
        commodity: Commodity name
        
    Returns:
        Dictionary with conflict regime analysis
    """
    # Extract model parameters
    threshold = model_results.get('threshold', 0.0)
    
    # Create price differential series
    price_diff = (north_prices - south_prices)
    price_diff_pct = price_diff / pd.concat([north_prices, south_prices], axis=1).min(axis=1)
    
    # Create regime indicators
    above_threshold = price_diff_pct > threshold
    below_threshold = price_diff_pct < -threshold
    band_regime = ~(above_threshold | below_threshold)
    
    # Align conflict series with price data
    common_index = price_diff.index.intersection(conflict_series.index)
    
    if len(common_index) == 0:
        return {"error": "No common dates between price data and conflict data"}
    
    # Create DataFrame with aligned data
    regime_df = pd.DataFrame({
        'price_diff': price_diff.loc[common_index],
        'price_diff_pct': price_diff_pct.loc[common_index],
        'conflict': conflict_series.loc[common_index],
        'above_regime': above_threshold.loc[common_index],
        'below_regime': below_threshold.loc[common_index],
        'band_regime': band_regime.loc[common_index]
    })
    
    # Calculate conflict intensity by regime
    regime_conflict = {
        'above': regime_df[regime_df['above_regime']]['conflict'].mean(),
        'below': regime_df[regime_df['below_regime']]['conflict'].mean(),
        'band': regime_df[regime_df['band_regime']]['conflict'].mean()
    }
    
    # Calculate regime frequencies by conflict level
    median_conflict = regime_df['conflict'].median()
    high_conflict = regime_df['conflict'] > median_conflict
    low_conflict = ~high_conflict
    
    # High conflict regime distribution
    high_conflict_regimes = {
        'above': regime_df[high_conflict]['above_regime'].mean() * 100,
        'below': regime_df[high_conflict]['below_regime'].mean() * 100,
        'band': regime_df[high_conflict]['band_regime'].mean() * 100
    }
    
    # Low conflict regime distribution
    low_conflict_regimes = {
        'above': regime_df[low_conflict]['above_regime'].mean() * 100,
        'below': regime_df[low_conflict]['below_regime'].mean() * 100,
        'band': regime_df[low_conflict]['band_regime'].mean() * 100
    }
    
    # Calculate persistence in each regime
    def calculate_persistence(series):
        runs = (series != series.shift()).cumsum()
        run_lengths = series.groupby(runs).transform('count')
        return run_lengths.mean()
    
    above_persistence = {
        'high_conflict': calculate_persistence(regime_df[high_conflict]['above_regime']),
        'low_conflict': calculate_persistence(regime_df[low_conflict]['above_regime'])
    }
    
    below_persistence = {
        'high_conflict': calculate_persistence(regime_df[high_conflict]['below_regime']),
        'low_conflict': calculate_persistence(regime_df[low_conflict]['below_regime'])
    }
    
    band_persistence = {
        'high_conflict': calculate_persistence(regime_df[high_conflict]['band_regime']),
        'low_conflict': calculate_persistence(regime_df[low_conflict]['band_regime'])
    }
    
    # Determine if conflict changes regime distribution
    if abs(high_conflict_regimes['band'] - low_conflict_regimes['band']) > 10:
        distribution_impact = "Strong"
    elif abs(high_conflict_regimes['band'] - low_conflict_regimes['band']) > 5:
        distribution_impact = "Moderate"
    else:
        distribution_impact = "Weak"
    
    # Combine results
    results = {
        "commodity": commodity,
        "threshold": float(threshold),
        "regime_conflict_intensity": {
            "above": float(regime_conflict['above']),
            "below": float(regime_conflict['below']),
            "band": float(regime_conflict['band'])
        },
        "high_conflict_regimes": {
            "above": float(high_conflict_regimes['above']),
            "below": float(high_conflict_regimes['below']),
            "band": float(high_conflict_regimes['band'])
        },
        "low_conflict_regimes": {
            "above": float(low_conflict_regimes['above']),
            "below": float(low_conflict_regimes['below']),
            "band": float(low_conflict_regimes['band'])
        },
        "regime_persistence": {
            "above": above_persistence,
            "below": below_persistence,
            "band": band_persistence
        },
        "impact": {
            "magnitude": distribution_impact,
            "band_change": float(high_conflict_regimes['band'] - low_conflict_regimes['band']),
            "above_change": float(high_conflict_regimes['above'] - low_conflict_regimes['above']),
            "below_change": float(high_conflict_regimes['below'] - low_conflict_regimes['below'])
        }
    }
    
    return results


@error_handler(fallback_value={})
def analyze_conflict_intensity_thresholds(
    price_diff_df: pd.DataFrame,
    conflict_series: pd.Series,
    commodity: str,
    n_thresholds: int = 4
) -> Dict[str, Any]:
    """
    Analyze how different conflict intensity thresholds affect price differentials.
    
    Args:
        price_diff_df: DataFrame with price differentials
        conflict_series: Series with conflict intensity data
        commodity: Commodity name
        n_thresholds: Number of conflict intensity thresholds to analyze
        
    Returns:
        Dictionary with conflict threshold analysis
    """
    if price_diff_df is None or price_diff_df.empty:
        return {"error": "No price differential data available"}
    
    if conflict_series is None or conflict_series.empty:
        return {"error": "No conflict data available"}
    
    # Align data
    common_index = price_diff_df.index.intersection(conflict_series.index)
    
    if len(common_index) == 0:
        return {"error": "No common dates between price data and conflict data"}
    
    # Create aligned DataFrame
    aligned_df = pd.DataFrame({
        'price_diff': price_diff_df.loc[common_index, 'diff'],
        'price_diff_pct': price_diff_df.loc[common_index, 'diff_pct'],
        'conflict': conflict_series.loc[common_index]
    })
    
    # Calculate conflict intensity thresholds
    conflict_quantiles = np.linspace(0, 1, n_thresholds + 1)[1:]
    conflict_thresholds = [aligned_df['conflict'].quantile(q) for q in conflict_quantiles]
    
    # Analyze price differentials at each threshold
    threshold_results = []
    
    for i, threshold in enumerate(conflict_thresholds):
        # Create indicator for conflict above threshold
        above_threshold = aligned_df['conflict'] > threshold
        
        # Calculate metrics for periods above/below threshold
        above_metrics = {
            "avg_price_diff_pct": aligned_df.loc[above_threshold, 'price_diff_pct'].abs().mean() * 100,
            "price_diff_volatility": aligned_df.loc[above_threshold, 'price_diff_pct'].std() * 100,
            "observations": above_threshold.sum(),
            "quantile": float(conflict_quantiles[i]),
            "conflict_threshold": float(threshold)
        }
        
        below_metrics = {
            "avg_price_diff_pct": aligned_df.loc[~above_threshold, 'price_diff_pct'].abs().mean() * 100,
            "price_diff_volatility": aligned_df.loc[~above_threshold, 'price_diff_pct'].std() * 100,
            "observations": (~above_threshold).sum(),
            "quantile": float(conflict_quantiles[i]),
            "conflict_threshold": float(threshold)
        }
        
        # Calculate relative increase
        if below_metrics["avg_price_diff_pct"] > 0:
            diff_increase_pct = ((above_metrics["avg_price_diff_pct"] / 
                                 below_metrics["avg_price_diff_pct"]) - 1) * 100
        else:
            diff_increase_pct = 0.0
        
        threshold_results.append({
            "conflict_threshold": float(threshold),
            "conflict_quantile": float(conflict_quantiles[i]),
            "above_threshold": above_metrics,
            "below_threshold": below_metrics,
            "diff_increase_pct": float(diff_increase_pct)
        })
    
    # Calculate correlations at different lags
    lag_correlations = {}
    for lag in range(0, 4):
        if lag == 0:
            corr = aligned_df['conflict'].corr(aligned_df['price_diff_pct'].abs())
        else:
            # Correlation between lagged conflict and current price diff
            corr = aligned_df['conflict'].shift(lag).corr(aligned_df['price_diff_pct'].abs())
        
        lag_correlations[lag] = float(corr)
    
    # Combine results
    results = {
        "commodity": commodity,
        "threshold_analysis": threshold_results,
        "lag_correlations": lag_correlations,
        "conflict_increase_impact": sorted(threshold_results, key=lambda x: x["diff_increase_pct"], reverse=True)[0]
    }
    
    return results