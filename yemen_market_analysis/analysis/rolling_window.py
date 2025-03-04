"""
Rolling window analysis for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

from core.decorators import error_handler, performance_tracker
from core.exceptions import AnalysisError
from models.threshold import (
    HansenSeoModel, EndersSiklosModel, ThresholdHistoryTracker
)
from models.statistics import (
    test_cointegration, run_bidirectional_granger_causality
)

logger = logging.getLogger(__name__)


@error_handler(fallback_value={})
@performance_tracker()
def run_rolling_window_analysis(
    north_prices: pd.Series,
    south_prices: pd.Series,
    window_size: int = 24,
    step_size: int = 3,
    model_type: str = 'hansen_seo',
    model_config: Optional[Dict[str, Any]] = None,
    commodity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run rolling window threshold analysis to capture time-varying relationships.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        window_size: Size of rolling window
        step_size: Step size between windows
        model_type: Type of threshold model
        model_config: Configuration for threshold model
        commodity: Commodity name
        
    Returns:
        Dictionary with rolling window analysis results
    """
    if north_prices.empty or south_prices.empty:
        return {"error": "Empty price series"}
    
    # Check for sufficient observations
    if len(north_prices) < window_size:
        return {"error": f"Insufficient observations ({len(north_prices)}) for window size {window_size}"}
    
    # Create default model configuration if not provided
    if model_config is None:
        model_config = {
            'grid_points': 20,
            'nboot': 100,
            'min_regime_size': 0.15,
            'max_lags': 4
        }
    
    # Initialize threshold model
    if model_type == 'hansen_seo':
        model = HansenSeoModel(model_config)
    elif model_type == 'enders_siklos':
        model = EndersSiklosModel(model_config)
    else:
        return {"error": f"Unknown model type: {model_type}"}
    
    # Initialize results
    windows = []
    dates = []
    thresholds = []
    p_values = []
    alphas_down = []
    alphas_up = []
    half_lives_down = []
    half_lives_up = []
    cointegration_flags = []
    
    # Run rolling window analysis
    for start_idx in range(0, len(north_prices) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Get window data
        north_window = north_prices.iloc[start_idx:end_idx]
        south_window = south_prices.iloc[start_idx:end_idx]
        window_date = north_prices.index[start_idx + window_size // 2]  # Middle of window
        
        try:
            # Test for cointegration
            cointegrated, _ = test_cointegration(
                north_window.values, south_window.values, trend='c'
            )
            
            # Estimate threshold model
            results = model.fit(
                north_window.values, south_window.values, commodity=commodity
            )
            
            # Extract key parameters
            threshold = results.get('threshold', np.nan)
            p_value = results.get('p_value', np.nan)
            alpha_down = results.get('adjustment_dynamics', {}).get('alpha_down', np.nan)
            alpha_up = results.get('adjustment_dynamics', {}).get('alpha_up', np.nan)
            half_life_down = results.get('adjustment_dynamics', {}).get('half_life_down', np.nan)
            half_life_up = results.get('adjustment_dynamics', {}).get('half_life_up', np.nan)
            
            # Store results
            windows.append((start_idx, end_idx))
            dates.append(window_date)
            thresholds.append(threshold)
            p_values.append(p_value)
            alphas_down.append(alpha_down)
            alphas_up.append(alpha_up)
            half_lives_down.append(half_life_down)
            half_lives_up.append(half_life_up)
            cointegration_flags.append(cointegrated)
        
        except Exception as e:
            logger.warning(f"Error in window {start_idx}-{end_idx}: {str(e)}")
            # Add placeholder values
            windows.append((start_idx, end_idx))
            dates.append(window_date)
            thresholds.append(np.nan)
            p_values.append(np.nan)
            alphas_down.append(np.nan)
            alphas_up.append(np.nan)
            half_lives_down.append(np.nan)
            half_lives_up.append(np.nan)
            cointegration_flags.append(False)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'date': dates,
        'threshold': thresholds,
        'p_value': p_values,
        'alpha_down': alphas_down,
        'alpha_up': alphas_up,
        'half_life_down': half_lives_down,
        'half_life_up': half_lives_up,
        'cointegrated': cointegration_flags
    })
    
    # Calculate summary statistics
    valid_thresholds = [t for t in thresholds if not np.isnan(t)]
    
    summary_stats = {
        'mean_threshold': np.mean(valid_thresholds) if valid_thresholds else np.nan,
        'std_threshold': np.std(valid_thresholds) if valid_thresholds else np.nan,
        'min_threshold': np.min(valid_thresholds) if valid_thresholds else np.nan,
        'max_threshold': np.max(valid_thresholds) if valid_thresholds else np.nan,
        'cointegration_pct': np.mean(cointegration_flags) * 100 if cointegration_flags else np.nan
    }
    
    # Identify structural breaks
    break_indices = _identify_structural_breaks(
        np.array(thresholds), significance=0.05, min_distance=3
    )
    
    break_dates = [dates[idx] for idx in break_indices]
    
    # Create combined results
    results = {
        "commodity": commodity,
        "model_type": model_type,
        "window_size": window_size,
        "step_size": step_size,
        "n_windows": len(windows),
        "summary": summary_stats,
        "structural_breaks": {
            "indices": break_indices,
            "dates": break_dates,
            "count": len(break_indices)
        },
        "data": results_df.to_dict(orient='records')
    }
    
    return results


@error_handler(fallback_value=[])
def _identify_structural_breaks(
    series: np.ndarray,
    window: int = 5,
    significance: float = 0.05,
    min_distance: int = 3
) -> List[int]:
    """
    Identify structural breaks in a time series.
    
    Args:
        series: Array of parameter values
        window: Window size for break detection
        significance: Significance level for break detection
        min_distance: Minimum distance between breaks
        
    Returns:
        List of indices where structural breaks occur
    """
    # Remove NaN values
    valid_mask = ~np.isnan(series)
    clean_series = series[valid_mask]
    
    if len(clean_series) < 2 * window:
        return []
    
    # Calculate rolling mean and standard deviation
    means = np.zeros(len(clean_series) - window)
    stds = np.zeros(len(clean_series) - window)
    
    for i in range(len(clean_series) - window):
        means[i] = np.mean(clean_series[i:i+window])
        stds[i] = np.std(clean_series[i:i+window])
    
    # Calculate z-scores for break detection
    z_scores = np.zeros(len(clean_series) - 2 * window)
    
    for i in range(len(z_scores)):
        left_mean = means[i]
        right_mean = means[i+window]
        pooled_std = np.sqrt((stds[i]**2 + stds[i+window]**2) / 2)
        
        if pooled_std > 0:
            z_scores[i] = abs(left_mean - right_mean) / pooled_std
        else:
            z_scores[i] = 0
    
    # Identify breaks using critical value
    from scipy import stats
    critical_value = stats.norm.ppf(1 - significance / 2)
    
    break_candidates = np.where(z_scores > critical_value)[0]
    
    # Filter breaks to ensure minimum distance
    if len(break_candidates) == 0:
        return []
    
    filtered_breaks = [break_candidates[0] + window]
    
    for brk in break_candidates[1:]:
        if brk - filtered_breaks[-1] >= min_distance:
            filtered_breaks.append(brk + window)
    
    # Map breaks back to original series with NaNs
    if not valid_mask.all():
        final_breaks = []
        offset = 0
        
        for i, valid in enumerate(valid_mask):
            if not valid:
                offset += 1
            elif i - offset in filtered_breaks:
                final_breaks.append(i)
        
        return final_breaks
    
    return filtered_breaks


@error_handler(fallback_value={})
@performance_tracker()
def run_rolling_cointegration_analysis(
    north_prices: pd.Series,
    south_prices: pd.Series,
    window_size: int = 24,
    step_size: int = 3,
    commodity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run rolling window cointegration analysis.
    
    Args:
        north_prices: North market price series
        south_prices: South market price series
        window_size: Size of rolling window
        step_size: Step size between windows
        commodity: Commodity name
        
    Returns:
        Dictionary with rolling cointegration analysis results
    """
    if north_prices.empty or south_prices.empty:
        return {"error": "Empty price series"}
    
    # Check for sufficient observations
    if len(north_prices) < window_size:
        return {"error": f"Insufficient observations ({len(north_prices)}) for window size {window_size}"}
    
    # Initialize results
    windows = []
    dates = []
    cointegrated_flags = []
    p_values = []
    
    # Run rolling window analysis
    for start_idx in range(0, len(north_prices) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        
        # Get window data
        north_window = north_prices.iloc[start_idx:end_idx]
        south_window = south_prices.iloc[start_idx:end_idx]
        window_date = north_prices.index[start_idx + window_size // 2]  # Middle of window
        
        try:
            # Test for cointegration
            cointegrated, coint_results = test_cointegration(
                north_window.values, south_window.values, trend='c', return_diagnostics=True
            )
            
            # Store results
            windows.append((start_idx, end_idx))
            dates.append(window_date)
            cointegrated_flags.append(cointegrated)
            p_values.append(coint_results.get('p_value', np.nan))
        
        except Exception as e:
            logger.warning(f"Error in window {start_idx}-{end_idx}: {str(e)}")
            # Add placeholder values
            windows.append((start_idx, end_idx))
            dates.append(window_date)
            cointegrated_flags.append(False)
            p_values.append(np.nan)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'date': dates,
        'cointegrated': cointegrated_flags,
        'p_value': p_values
    })
    
    # Calculate summary statistics
    cointegration_pct = np.mean(cointegrated_flags) * 100 if cointegrated_flags else np.nan
    
    # Create combined results
    results = {
        "commodity": commodity,
        "window_size": window_size,
        "step_size": step_size,
        "n_windows": len(windows),
        "cointegration_percentage": float(cointegration_pct),
        "data": results_df.to_dict(orient='records')
    }
    
    return results


@error_handler(fallback_value={})
@performance_tracker()
def analyze_parameter_stability(
    rolling_results: Dict[str, Any],
    conflict_series: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Analyze stability of threshold model parameters.
    
    Args:
        rolling_results: Results from rolling window analysis
        conflict_series: Optional series with conflict intensity data
        
    Returns:
        Dictionary with parameter stability analysis
    """
    if "data" not in rolling_results or not rolling_results["data"]:
        return {"error": "No rolling window data available"}
    
    # Convert results to DataFrame if needed
    if isinstance(rolling_results["data"], list):
        results_df = pd.DataFrame(rolling_results["data"])
    else:
        results_df = rolling_results["data"]
    
    # Calculate stability metrics
    threshold_stability = {
        "mean": float(results_df['threshold'].mean()),
        "std": float(results_df['threshold'].std()),
        "cv": float(results_df['threshold'].std() / results_df['threshold'].mean()) 
              if results_df['threshold'].mean() != 0 else float('inf'),
        "range": float(results_df['threshold'].max() - results_df['threshold'].min()),
        "iqr": float(results_df['threshold'].quantile(0.75) - results_df['threshold'].quantile(0.25))
    }
    
    alpha_down_stability = {
        "mean": float(results_df['alpha_down'].mean()),
        "std": float(results_df['alpha_down'].std()),
        "cv": float(abs(results_df['alpha_down'].std() / results_df['alpha_down'].mean())) 
              if results_df['alpha_down'].mean() != 0 else float('inf'),
        "sign_changes": int(np.sum(np.diff(np.sign(results_df['alpha_down'])) != 0))
    }
    
    alpha_up_stability = {
        "mean": float(results_df['alpha_up'].mean()),
        "std": float(results_df['alpha_up'].std()),
        "cv": float(abs(results_df['alpha_up'].std() / results_df['alpha_up'].mean())) 
              if results_df['alpha_up'].mean() != 0 else float('inf'),
        "sign_changes": int(np.sum(np.diff(np.sign(results_df['alpha_up'])) != 0))
    }
    
    # Analyze relation to conflict if data available
    conflict_relation = {}
    
    if conflict_series is not None and not conflict_series.empty:
        # Merge conflict data with rolling results
        merged_df = results_df.copy()
        merged_df['conflict'] = np.nan
        
        for i, date in enumerate(merged_df['date']):
            if date in conflict_series.index:
                merged_df.loc[i, 'conflict'] = conflict_series.loc[date]
        
        # Calculate correlations
        conflict_relation = {
            "threshold_correlation": float(merged_df['threshold'].corr(merged_df['conflict'])),
            "alpha_down_correlation": float(merged_df['alpha_down'].corr(merged_df['conflict'])),
            "alpha_up_correlation": float(merged_df['alpha_up'].corr(merged_df['conflict']))
        }
    
    # Evaluate stability
    if threshold_stability["cv"] < 0.2:
        threshold_stability_assessment = "High stability"
    elif threshold_stability["cv"] < 0.5:
        threshold_stability_assessment = "Moderate stability"
    else:
        threshold_stability_assessment = "Low stability"
    
    # Combine results
    stability_analysis = {
        "commodity": rolling_results.get("commodity"),
        "threshold": threshold_stability,
        "alpha_down": alpha_down_stability,
        "alpha_up": alpha_up_stability,
        "structural_breaks": rolling_results.get("structural_breaks", {}),
        "conflict_relation": conflict_relation,
        "overall_assessment": {
            "threshold_stability": threshold_stability_assessment,
            "parameter_consistency": "High" if (alpha_down_stability["sign_changes"] + alpha_up_stability["sign_changes"]) == 0 else 
                                    "Moderate" if (alpha_down_stability["sign_changes"] + alpha_up_stability["sign_changes"]) <= 2 else
                                    "Low"
        }
    }
    
    return stability_analysis