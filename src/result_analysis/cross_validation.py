"""
Cross-validation module for Yemen Market Integration analysis.

This module provides utilities for model validation, including threshold
parameter stability testing and predictive performance evaluation.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any, Union, Optional, List, Tuple, Callable
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

def cross_validate_threshold(
    y: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Cross-validate threshold estimation using k-fold cross-validation.
    
    Parameters
    ----------
    y : ndarray
        Dependent variable
    X : ndarray
        Independent variables
    n_folds : int, optional
        Number of cross-validation folds
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    dict
        Cross-validation results
    """
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create folds
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    
    # Initialize results
    thresholds = []
    errors = []
    
    # Perform cross-validation
    for i in range(n_folds):
        # Define test indices
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n
        test_indices = indices[test_start:test_end]
        
        # Define training indices
        train_indices = np.setdiff1d(indices, test_indices)
        
        # Split data
        y_train, X_train = y[train_indices], X[train_indices]
        y_test, X_test = y[test_indices], X[test_indices]
        
        # Estimate cointegrating relationship
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        linear_model = sm.OLS(y_train, X_train_const).fit()
        
        # Calculate residuals
        train_residuals = linear_model.resid
        
        # Use a simplified grid search for threshold
        grid = np.sort(train_residuals)
        n_grid = len(grid)
        grid = grid[int(n_grid*0.15):int(n_grid*0.85)]  # Trim 15% from each tail
        
        # For each threshold, compute SSR
        best_thresh = None
        best_ssr = float('inf')
        
        for thresh in grid:
            below = train_residuals <= thresh
            above = ~below
            
            if sum(below) > 10 and sum(above) > 10:
                model_below = sm.OLS(y_train[below], X_train_const[below]).fit()
                model_above = sm.OLS(y_train[above], X_train_const[above]).fit()
                
                ssr = model_below.ssr + model_above.ssr
                
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_thresh = thresh
        
        # If a threshold was found, evaluate on test set
        if best_thresh is not None:
            # Make predictions on test set
            test_residuals = y_test - linear_model.predict(X_test_const)
            below = test_residuals <= best_thresh
            above = ~below
            
            # Calculate prediction error
            mse = 0
            n_test = len(y_test)
            
            if sum(below) > 0:
                model_below = sm.OLS(y_train[train_residuals <= best_thresh], 
                                    X_train_const[train_residuals <= best_thresh]).fit()
                y_pred_below = model_below.predict(X_test_const[below])
                mse += np.sum((y_test[below] - y_pred_below) ** 2) / n_test
            
            if sum(above) > 0:
                model_above = sm.OLS(y_train[train_residuals > best_thresh], 
                                    X_train_const[train_residuals > best_thresh]).fit()
                y_pred_above = model_above.predict(X_test_const[above])
                mse += np.sum((y_test[above] - y_pred_above) ** 2) / n_test
            
            # Store results
            thresholds.append(best_thresh)
            errors.append(mse)
    
    # Calculate summary statistics
    thresholds = np.array(thresholds)
    errors = np.array(errors)
    
    return {
        'thresholds': thresholds,
        'errors': errors,
        'mean_threshold': np.mean(thresholds),
        'std_threshold': np.std(thresholds),
        'cv_error': np.mean(errors),
        'cv_std_error': np.std(errors)
    }


def calculate_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate prediction metrics for model evaluation.
    
    Parameters
    ----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
        
    Returns
    -------
    dict
        Prediction metrics
    """
    # Calculate various metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R-squared
    r2 = r2_score(y_true, y_pred)
    
    # Calculate adjusted R-squared
    n = len(y_true)
    p = 1  # Assuming one predictor (simple model)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Calculate normalized RMSE
    y_range = np.max(y_true) - np.min(y_true)
    if y_range > 0:
        nrmse = rmse / y_range
    else:
        nrmse = np.nan
    
    # Calculate mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None))) * 100
    
    # Calculate symmetric mean absolute percentage error
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    # Calculate Theil's U statistic
    # Theil's U1 (forecast accuracy)
    u1_num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    u1_den = np.sqrt(np.mean(y_true ** 2)) + np.sqrt(np.mean(y_pred ** 2))
    if u1_den > 0:
        theil_u1 = u1_num / u1_den
    else:
        theil_u1 = np.nan
    
    # Theil's U2 (relative forecast accuracy)
    # Compare to naive forecast (y_t = y_{t-1})
    naive_pred = np.roll(y_true, 1)
    naive_pred[0] = naive_pred[1]  # Replace first value
    
    u2_num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    u2_den = np.sqrt(np.mean((y_true - naive_pred) ** 2))
    if u2_den > 0:
        theil_u2 = u2_num / u2_den
    else:
        theil_u2 = np.nan
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2,
        'nrmse': nrmse,
        'mape': mape,
        'smape': smape,
        'theil_u1': theil_u1,
        'theil_u2': theil_u2
    }


def out_of_sample_forecast(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Perform out-of-sample forecasting and calculate performance metrics.
    
    Parameters
    ----------
    model : Any
        Fitted model object with predict method
    X_test : ndarray
        Test features
    y_test : ndarray
        Test target
        
    Returns
    -------
    dict
        Forecast results and performance metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_prediction_metrics(y_test, y_pred)
        
        return {
            'y_pred': y_pred,
            'metrics': metrics
        }
    
    except Exception as e:
        logger.error(f"Error in out-of-sample forecast: {str(e)}")
        return {
            'error': str(e)
        }


def stability_test_expanding_window(
    y: np.ndarray,
    X: np.ndarray,
    min_window: int = 30,
    step: int = 1,
    parameter_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Test parameter stability using expanding window estimation.
    
    Parameters
    ----------
    y : ndarray
        Dependent variable
    X : ndarray
        Independent variables
    min_window : int, optional
        Minimum window size
    step : int, optional
        Step size for expanding window
    parameter_names : list of str, optional
        Names of parameters (for labeling)
        
    Returns
    -------
    dict
        Stability test results
    """
    n = len(y)
    
    if min_window >= n:
        logger.warning(f"min_window ({min_window}) >= n ({n})")
        return {
            'error': "min_window too large"
        }
    
    # Add constant to X if needed
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if np.all(X[:, 0] != 1):
        X = sm.add_constant(X)
    
    # Number of parameters
    k = X.shape[1]
    
    # Set parameter names if not provided
    if parameter_names is None:
        if k == 2:
            parameter_names = ['const', 'beta']
        else:
            parameter_names = ['const'] + [f'beta{i}' for i in range(1, k)]
    
    # Initialize storage for parameter estimates
    params = np.zeros((n - min_window + 1, k))
    window_ends = np.arange(min_window, n+1)
    
    # Estimate model for each window
    for i, end in enumerate(window_ends):
        window_y = y[:end]
        window_X = X[:end]
        
        model = sm.OLS(window_y, window_X).fit()
        params[i] = model.params
    
    # Create DataFrame with results
    results_df = pd.DataFrame(params, columns=parameter_names)
    results_df['window_end'] = window_ends
    results_df['window_size'] = window_ends
    
    # Calculate parameter stability statistics
    stability_stats = {}
    for param in parameter_names:
        param_values = results_df[param]
        
        # Calculate various stability metrics
        stability_stats[param] = {
            'mean': np.mean(param_values),
            'std': np.std(param_values),
            'min': np.min(param_values),
            'max': np.max(param_values),
            'cv': np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else np.nan
        }
    
    return {
        'parameter_estimates': results_df,
        'stability_stats': stability_stats
    }


def parameter_evolution_plot(
    stability_results: Dict[str, Any],
    parameter: str = 'beta',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a plot showing the evolution of parameter estimates.
    
    Parameters
    ----------
    stability_results : dict
        Results from stability_test_expanding_window
    parameter : str, optional
        Parameter to plot
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure showing parameter evolution
    """
    # Extract parameter estimates
    results_df = stability_results.get('parameter_estimates')
    
    if results_df is None or parameter not in results_df.columns:
        logger.warning(f"Parameter '{parameter}' not found in stability results")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot parameter evolution
    ax.plot(results_df['window_size'], results_df[parameter], marker='o', markersize=3)
    
    # Add horizontal line at final estimate
    final_estimate = results_df[parameter].iloc[-1]
    ax.axhline(y=final_estimate, color='r', linestyle='--', alpha=0.7,
              label=f'Final estimate: {final_estimate:.4f}')
    
    # Calculate confidence bands (±2*SE)
    param_std = stability_results.get('stability_stats', {}).get(parameter, {}).get('std', 0)
    ax.fill_between(
        results_df['window_size'],
        final_estimate - 2*param_std,
        final_estimate + 2*param_std,
        alpha=0.2,
        color='r',
        label='±2 SE band'
    )
    
    # Add labels and title
    ax.set_xlabel('Window Size')
    ax.set_ylabel(f'Estimated {parameter}')
    ax.set_title(f'Evolution of {parameter} Estimate with Expanding Window')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def recursive_residuals(
    y: np.ndarray,
    X: np.ndarray,
    min_window: int = 30
) -> Dict[str, Any]:
    """
    Calculate recursive residuals for stability testing.
    
    Parameters
    ----------
    y : ndarray
        Dependent variable
    X : ndarray
        Independent variables
    min_window : int, optional
        Minimum window size
        
    Returns
    -------
    dict
        Recursive residuals and CUSUM test results
    """
    n = len(y)
    
    if min_window >= n:
        logger.warning(f"min_window ({min_window}) >= n ({n})")
        return {
            'error': "min_window too large"
        }
    
    # Add constant to X if needed
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if np.all(X[:, 0] != 1):
        X = sm.add_constant(X)
    
    # Number of parameters
    k = X.shape[1]
    
    # Initialize storage for recursive residuals
    rec_resid = np.zeros(n - min_window)
    
    # Calculate recursive residuals
    for t in range(min_window, n):
        # Estimate model using observations up to t-1
        X_t1 = X[:t]
        y_t1 = y[:t]
        
        model = sm.OLS(y_t1, X_t1).fit()
        
        # Calculate prediction for observation t
        x_t = X[t:t+1]
        y_pred = model.predict(x_t)[0]
        
        # Calculate forecast error
        error = y[t] - y_pred
        
        # Calculate forecast variance
        forecast_var = 1 + x_t @ np.linalg.inv(X_t1.T @ X_t1) @ x_t.T
        
        # Calculate recursive residual
        rec_resid[t - min_window] = error / np.sqrt(forecast_var)
    
    # Calculate CUSUM and CUSUM-sq statistics
    cusum = np.cumsum(rec_resid)
    cusum_sq = np.cumsum(rec_resid ** 2)
    
    # Standardize CUSUM
    std_cusum = cusum / np.std(rec_resid)
    
    # Calculate critical values for CUSUM test
    # Using 5% significance level
    alpha = 0.05
    n_resid = len(rec_resid)
    cv = 0.948  # Approximate critical value for 5% significance
    critical_values = cv * np.sqrt(n_resid) * (1 + 2 * np.arange(n_resid) / n_resid)
    
    # Perform CUSUM test
    cusum_violation = np.any(np.abs(std_cusum) > critical_values)
    
    # Calculate CUSUM-sq test statistics
    s = np.cumsum(rec_resid ** 2) / np.sum(rec_resid ** 2)
    
    # Calculate critical lines for CUSUM-sq test
    # Using 5% significance level
    lower = np.arange(n_resid) / n_resid - cv * np.sqrt(n_resid)
    upper = np.arange(n_resid) / n_resid + cv * np.sqrt(n_resid)
    
    # Perform CUSUM-sq test
    cusumq_violation = np.any((s < lower) | (s > upper))
    
    return {
        'recursive_residuals': rec_resid,
        'cusum': std_cusum,
        'cusum_sq': s,
        'critical_values': critical_values,
        'cusum_violation': cusum_violation,
        'cusumq_violation': cusumq_violation,
        'stability_p_value': alpha if cusum_violation or cusumq_violation else 1.0
    }