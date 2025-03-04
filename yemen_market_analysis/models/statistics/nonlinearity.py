"""
Nonlinearity testing for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import statsmodels.api as sm
from scipy import stats

from core.decorators import error_handler, performance_tracker
from core.exceptions import StatisticalTestError

logger = logging.getLogger(__name__)


@error_handler(fallback_value=(False, {"error": "Tsay test failed"}))
@performance_tracker()
def tsay_test(
    series: Union[pd.Series, np.ndarray],
    order: int = 2,
    lags: int = 2,
    significance: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:
    """
    Tsay (1989) test for threshold nonlinearity.
    
    Args:
        series: Time series data
        order: AR order for the model
        lags: Number of lags for the test
        significance: Significance level
        
    Returns:
        Tuple of (nonlinearity_detected, test_results)
    """
    # Convert to numpy array if needed
    if isinstance(series, pd.Series):
        data = series.values
    else:
        data = series
    
    # Calculate differences if needed
    if np.std(np.diff(data)) > 0:
        y = np.diff(data)
    else:
        y = data
    
    # Create lagged matrix
    n = len(y)
    X = np.zeros((n - order, order))
    
    for i in range(order):
        X[:, i] = y[i:n-order+i]
    
    # Recursive estimation
    recursive_residuals = np.zeros(n - order - lags)
    
    for t in range(lags, n - order):
        # Estimate model using data up to t
        X_t = X[:t, :]
        y_t = y[:t]
        
        try:
            # Calculate OLS estimates
            beta = np.linalg.inv(X_t.T @ X_t) @ X_t.T @ y_t
            
            # Calculate residual for next observation
            yhat = X[t, :] @ beta
            recursive_residuals[t-lags] = y[t] - yhat
        except:
            recursive_residuals[t-lags] = 0
    
    # Arrange regression based on predictive residuals
    X_arranged = X[lags:, :]
    
    # Exclude zeros from recursive residuals
    valid = recursive_residuals != 0
    
    if np.sum(valid) <= order + 1:
        return False, {"error": "Insufficient valid observations for Tsay test"}
    
    # Arranged regression
    try:
        test_model = sm.OLS(recursive_residuals[valid], X_arranged[valid]).fit()
        
        # Calculate F-statistic
        f_stat = test_model.fvalue
        p_value = test_model.f_pvalue
        
        # Determine if nonlinearity is detected
        nonlinearity = p_value < significance
        
        return nonlinearity, {
            'nonlinearity': nonlinearity,
            'p_value': float(p_value),
            'f_statistic': float(f_stat),
            'order': order,
            'lags': lags,
            'n_obs': int(np.sum(valid)),
            'method': 'Tsay'
        }
    except:
        return False, {"error": "Failed to compute Tsay test"}


@error_handler(fallback_value=(False, {"error": "Reset test failed"}))
def reset_test(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    power: int = 3,
    significance: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:
    """
    Ramsey's RESET test for nonlinearity.
    
    Args:
        y: Dependent variable
        X: Independent variables (should include constant)
        power: Maximum power of fitted values to include
        significance: Significance level
        
    Returns:
        Tuple of (nonlinearity_detected, test_results)
    """
    # Convert to numpy arrays if needed
    if isinstance(y, pd.Series):
        y_data = y.values
    else:
        y_data = y
        
    if isinstance(X, pd.DataFrame):
        X_data = X.values
    else:
        X_data = X
    
    # Step 1: Estimate linear model
    try:
        linear_model = sm.OLS(y_data, X_data).fit()
        yhat = linear_model.fittedvalues
    except:
        return False, {"error": "Failed to estimate linear model"}
    
    # Step 2: Create augmented model with powers of fitted values
    X_augmented = X_data.copy()
    
    for p in range(2, power + 1):
        X_augmented = np.column_stack([X_augmented, yhat**p])
    
    # Step 3: Estimate augmented model
    try:
        augmented_model = sm.OLS(y_data, X_augmented).fit()
        
        # Step 4: Perform F-test for nonlinearity
        f_stat = ((linear_model.ssr - augmented_model.ssr) / (power - 1)) / (augmented_model.ssr / (len(y_data) - X_augmented.shape[1]))
        p_value = 1 - stats.f.cdf(f_stat, power - 1, len(y_data) - X_augmented.shape[1])
        
        # Determine if nonlinearity is detected
        nonlinearity = p_value < significance
        
        return nonlinearity, {
            'nonlinearity': nonlinearity,
            'p_value': float(p_value),
            'f_statistic': float(f_stat),
            'power': power,
            'n_obs': len(y_data),
            'method': 'RESET'
        }
    except:
        return False, {"error": "Failed to compute RESET test"}


@error_handler(fallback_value=(False, {"error": "Keenan test failed"}))
def keenan_test(
    series: Union[pd.Series, np.ndarray],
    order: int = 2,
    significance: float = 0.05
) -> Tuple[bool, Dict[str, Any]]:
    """
    Keenan test for nonlinearity.
    
    Args:
        series: Time series data
        order: AR order for the model
        significance: Significance level
        
    Returns:
        Tuple of (nonlinearity_detected, test_results)
    """
    # Convert to numpy array if needed
    if isinstance(series, pd.Series):
        data = series.values
    else:
        data = series
    
    # Step 1: Fit AR(p) model
    n = len(data)
    y = data[order:]
    
    # Create lagged X matrix
    X = np.ones((n - order, order + 1))
    for i in range(order):
        X[:, i + 1] = data[order-i-1:n-i-1]
    
    # Fit AR model
    try:
        ar_model = sm.OLS(y, X).fit()
        yhat = ar_model.fittedvalues
    except:
        return False, {"error": "Failed to estimate AR model"}
    
    # Step 2: Regress squared fitted values on X
    try:
        yhat2 = yhat**2
        yhat2_model = sm.OLS(yhat2, X).fit()
        v = yhat2 - yhat2_model.fittedvalues  # Residuals
    except:
        return False, {"error": "Failed in second stage regression"}
    
    # Step 3: Regress y on X and v
    try:
        final_X = np.column_stack([X, v])
        final_model = sm.OLS(y, final_X).fit()
        
        # Extract t-statistic for the v coefficient
        t_stat = final_model.tvalues[-1]
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - order - 2))
        
        # Determine if nonlinearity is detected
        nonlinearity = p_value < significance
        
        return nonlinearity, {
            'nonlinearity': nonlinearity,
            'p_value': float(p_value),
            't_statistic': float(t_stat),
            'order': order,
            'n_obs': n - order,
            'method': 'Keenan'
        }
    except:
        return False, {"error": "Failed in final stage of Keenan test"}


@error_handler(fallback_value={"error": "Nonlinearity tests failed"})
def run_all_nonlinearity_tests(
    series: Union[pd.Series, np.ndarray],
    ar_order: int = 2,
    significance: float = 0.05
) -> Dict[str, Any]:
    """
    Run multiple nonlinearity tests and return combined results.
    
    Args:
        series: Time series data
        ar_order: AR order for tests
        significance: Significance level
        
    Returns:
        Dictionary with results from all tests
    """
    # Run Tsay test
    tsay_nonlinearity, tsay_results = tsay_test(
        series, order=ar_order, lags=ar_order, significance=significance
    )
    
    # Prepare data for RESET test
    n = len(series)
    y = series[ar_order:]
    
    # Create lagged X matrix
    X = np.ones((n - ar_order, ar_order + 1))
    for i in range(ar_order):
        X[:, i + 1] = series[ar_order-i-1:n-i-1]
    
    # Run RESET test
    reset_nonlinearity, reset_results = reset_test(
        y, X, power=3, significance=significance
    )
    
    # Run Keenan test
    keenan_nonlinearity, keenan_results = keenan_test(
        series, order=ar_order, significance=significance
    )
    
    # Determine overall nonlinearity
    nonlinear_tests = [tsay_nonlinearity, reset_nonlinearity, keenan_nonlinearity]
    nonlinear_count = sum(nonlinear_tests)
    
    if nonlinear_count >= 2:
        nonlinearity = True
        evidence = "Strong evidence of nonlinearity"
    elif nonlinear_count == 1:
        nonlinearity = True
        evidence = "Moderate evidence of nonlinearity"
    else:
        nonlinearity = False
        evidence = "No strong evidence of nonlinearity"
    
    return {
        'nonlinearity': nonlinearity,
        'evidence': evidence,
        'nonlinear_tests_count': nonlinear_count,
        'tsay_test': tsay_results,
        'reset_test': reset_results,
        'keenan_test': keenan_results,
        'ar_order': ar_order,
        'significance': significance
    }