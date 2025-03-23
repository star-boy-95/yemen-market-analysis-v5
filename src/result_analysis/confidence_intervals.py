"""
Confidence interval calculation module for Yemen Market Integration analysis.

This module provides functions for calculating confidence intervals for
various model parameters, including specialized methods for threshold models.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Any, Union, Optional, List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

def calculate_confidence_interval(
    theta: float,
    se: float,
    confidence_level: float = 0.95,
    df: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate confidence interval for parameter estimate.
    
    Parameters
    ----------
    theta : float
        Parameter estimate
    se : float
        Standard error of the estimate
    confidence_level : float, optional
        Confidence level (0.90, 0.95, 0.99)
    df : int, optional
        Degrees of freedom (if None, use normal distribution)
        
    Returns
    -------
    tuple
        (lower bound, upper bound) of confidence interval
    """
    alpha = 1 - confidence_level
    
    if df is None:
        # Use normal distribution
        z = stats.norm.ppf(1 - alpha/2)
        lower = theta - z * se
        upper = theta + z * se
    else:
        # Use t-distribution
        t = stats.t.ppf(1 - alpha/2, df=df)
        lower = theta - t * se
        upper = theta + t * se
    
    return (lower, upper)


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: Callable,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    method: str = 'percentile'
) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : ndarray
        Data to resample
    statistic_func : callable
        Function to calculate statistic on each bootstrap sample
    confidence_level : float, optional
        Confidence level (0.90, 0.95, 0.99)
    n_resamples : int, optional
        Number of bootstrap resamples
    method : str, optional
        Bootstrap method: 'percentile', 'basic', or 'bca'
        
    Returns
    -------
    dict
        Bootstrap results including:
        - 'point_estimate': Original estimate
        - 'confidence_interval': (lower, upper) bounds
        - 'standard_error': Bootstrap standard error
        - 'bootstrap_values': All bootstrap statistics
    """
    alpha = 1 - confidence_level
    n = len(data)
    
    # Calculate original statistic
    theta_hat = statistic_func(data)
    
    # Generate bootstrap replicates
    bootstrap_values = []
    for i in range(n_resamples):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample = data[indices]
        
        # Calculate statistic on bootstrap sample
        theta_boot = statistic_func(bootstrap_sample)
        bootstrap_values.append(theta_boot)
    
    # Calculate bootstrap standard error
    bootstrap_values = np.array(bootstrap_values)
    bootstrap_se = np.std(bootstrap_values, ddof=1)
    
    # Calculate confidence interval based on method
    if method == 'percentile':
        # Percentile method
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower = np.percentile(bootstrap_values, lower_percentile)
        upper = np.percentile(bootstrap_values, upper_percentile)
    
    elif method == 'basic':
        # Basic bootstrap method
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower = 2 * theta_hat - np.percentile(bootstrap_values, upper_percentile)
        upper = 2 * theta_hat - np.percentile(bootstrap_values, lower_percentile)
    
    elif method == 'bca':
        # Bias-corrected and accelerated bootstrap
        # Compute bias correction factor
        z0 = stats.norm.ppf(np.mean(bootstrap_values < theta_hat))
        
        # Compute acceleration factor
        theta_jackknife = []
        for i in range(n):
            # Remove ith observation
            jackknife_sample = np.delete(data, i, axis=0)
            theta_jack = statistic_func(jackknife_sample)
            theta_jackknife.append(theta_jack)
        
        theta_jackknife = np.array(theta_jackknife)
        theta_dot = np.mean(theta_jackknife)
        num = np.sum((theta_dot - theta_jackknife) ** 3)
        den = 6 * np.sum((theta_dot - theta_jackknife) ** 2) ** (3/2)
        
        # Avoid division by zero
        if den != 0:
            a = num / den
        else:
            logger.warning("Denominator is zero in acceleration factor calculation")
            a = 0
        
        # Compute BCa confidence limits
        z_alpha1 = stats.norm.ppf(alpha / 2)
        z_alpha2 = stats.norm.ppf(1 - alpha / 2)
        
        p1 = stats.norm.cdf(z0 + (z0 + z_alpha1) / (1 - a * (z0 + z_alpha1)))
        p2 = stats.norm.cdf(z0 + (z0 + z_alpha2) / (1 - a * (z0 + z_alpha2)))
        
        lower = np.percentile(bootstrap_values, p1 * 100)
        upper = np.percentile(bootstrap_values, p2 * 100)
    
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")
    
    return {
        'point_estimate': theta_hat,
        'confidence_interval': (lower, upper),
        'standard_error': bootstrap_se,
        'bootstrap_values': bootstrap_values
    }


def threshold_confidence_interval(
    model: Any,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    method: str = 'grid_search'
) -> Dict[str, Any]:
    """
    Calculate confidence interval for threshold parameter.
    
    Parameters
    ----------
    model : Any
        Fitted threshold model object
    confidence_level : float, optional
        Confidence level (0.90, 0.95, 0.99)
    n_resamples : int, optional
        Number of bootstrap resamples (for bootstrap method)
    method : str, optional
        Method: 'grid_search', 'bootstrap', or 'profile_likelihood'
        
    Returns
    -------
    dict
        Threshold confidence interval results
    """
    # Extract threshold parameter
    threshold = getattr(model, 'threshold', None)
    
    if threshold is None:
        logger.warning("Threshold parameter not found in model")
        return {
            'point_estimate': None,
            'confidence_interval': (None, None),
            'method': method,
            'error': "Threshold parameter not found"
        }
    
    try:
        if method == 'grid_search':
            # Grid search method (Hansen, 2000)
            # This is a simplified version
            
            # Extract data from model
            if hasattr(model, 'data1') and hasattr(model, 'data2'):
                y = model.data1
                X = model.data2
                
                # Create grid of threshold values
                if hasattr(model, 'eq_errors') and model.eq_errors is not None:
                    # Use equilibrium errors for grid
                    eq_errors = model.eq_errors
                    grid = np.sort(eq_errors)
                    
                    # Trim grid to exclude extreme percentiles
                    n = len(grid)
                    trim = int(n * 0.15)  # Trim 15% from each tail
                    grid = grid[trim:-trim]
                    
                    # Calculate likelihood ratio for each threshold
                    ssr_min = float('inf')
                    lr_stats = []
                    
                    for thresh in grid:
                        # Split data based on threshold
                        below = eq_errors <= thresh
                        above = ~below
                        
                        if sum(below) > 0 and sum(above) > 0:
                            # Estimate model for each regime
                            X_const = sm.add_constant(X)
                            
                            model_below = sm.OLS(y[below], X_const[below]).fit()
                            model_above = sm.OLS(y[above], X_const[above]).fit()
                            
                            # Calculate sum of squared residuals
                            ssr = model_below.ssr + model_above.ssr
                            ssr_min = min(ssr, ssr_min)
                            
                            # Calculate likelihood ratio statistic
                            lr_stats.append((thresh, ssr))
                    
                    # Convert to numpy array
                    lr_array = np.array(lr_stats)
                    
                    # Calculate likelihood ratio statistic
                    lr_array[:, 1] = (lr_array[:, 1] - ssr_min) / ssr_min
                    
                    # Find confidence region
                    alpha = 1 - confidence_level
                    critical_value = -2 * np.log(1 - np.sqrt(1 - alpha))
                    
                    # Find all threshold values within confidence region
                    confidence_region = lr_array[lr_array[:, 1] <= critical_value]
                    
                    # Get confidence interval
                    lower = confidence_region[0, 0] if len(confidence_region) > 0 else None
                    upper = confidence_region[-1, 0] if len(confidence_region) > 0 else None
                    
                    return {
                        'point_estimate': threshold,
                        'confidence_interval': (lower, upper),
                        'method': 'grid_search',
                        'likelihood_ratio_stats': lr_array,
                        'critical_value': critical_value
                    }
                
                else:
                    logger.warning("Equilibrium errors not found in model")
                    return {
                        'point_estimate': threshold,
                        'confidence_interval': (None, None),
                        'method': 'grid_search',
                        'error': "Equilibrium errors not found"
                    }
            
            else:
                logger.warning("Data not found in model")
                return {
                    'point_estimate': threshold,
                    'confidence_interval': (None, None),
                    'method': 'grid_search',
                    'error': "Data not found"
                }
        
        elif method == 'bootstrap':
            # Bootstrap method
            
            # Extract data from model
            if hasattr(model, 'data1') and hasattr(model, 'data2'):
                y = model.data1
                X = model.data2
                
                # Define function to calculate threshold on bootstrap sample
                def calculate_threshold(bootstrap_indices):
                    # Extract bootstrap sample
                    y_boot = y[bootstrap_indices]
                    X_boot = X[bootstrap_indices]
                    
                    # Estimate cointegrating relationship
                    X_const = sm.add_constant(X_boot)
                    linear_model = sm.OLS(y_boot, X_const).fit()
                    residuals = linear_model.resid
                    
                    # Use a simplified grid search for threshold
                    grid = np.sort(residuals)
                    n = len(grid)
                    grid = grid[int(n*0.15):int(n*0.85)]  # Trim 15% from each tail
                    
                    # For each threshold, compute SSR
                    best_thresh = None
                    best_ssr = float('inf')
                    
                    for thresh in grid:
                        below = residuals <= thresh
                        above = ~below
                        
                        if sum(below) > 10 and sum(above) > 10:
                            model_below = sm.OLS(y_boot[below], X_const[below]).fit()
                            model_above = sm.OLS(y_boot[above], X_const[above]).fit()
                            
                            ssr = model_below.ssr + model_above.ssr
                            
                            if ssr < best_ssr:
                                best_ssr = ssr
                                best_thresh = thresh
                    
                    return best_thresh
                
                # Generate bootstrap samples
                n = len(y)
                bootstrap_thresholds = []
                
                for i in range(n_resamples):
                    # Generate bootstrap indices
                    indices = np.random.choice(n, size=n, replace=True)
                    
                    # Calculate threshold for this sample
                    threshold_boot = calculate_threshold(indices)
                    
                    if threshold_boot is not None:
                        bootstrap_thresholds.append(threshold_boot)
                
                # Calculate confidence interval
                bootstrap_thresholds = np.array(bootstrap_thresholds)
                alpha = 1 - confidence_level
                lower = np.percentile(bootstrap_thresholds, alpha/2 * 100)
                upper = np.percentile(bootstrap_thresholds, (1 - alpha/2) * 100)
                
                return {
                    'point_estimate': threshold,
                    'confidence_interval': (lower, upper),
                    'method': 'bootstrap',
                    'bootstrap_values': bootstrap_thresholds
                }
            
            else:
                logger.warning("Data not found in model")
                return {
                    'point_estimate': threshold,
                    'confidence_interval': (None, None),
                    'method': 'bootstrap',
                    'error': "Data not found"
                }
        
        elif method == 'profile_likelihood':
            # Profile likelihood method
            # This is a placeholder for the actual implementation
            
            logger.warning("Profile likelihood method not fully implemented")
            return {
                'point_estimate': threshold,
                'confidence_interval': (threshold - 0.1, threshold + 0.1),  # Placeholder
                'method': 'profile_likelihood',
                'warning': "Method not fully implemented"
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    except Exception as e:
        logger.error(f"Error calculating threshold confidence interval: {str(e)}")
        return {
            'point_estimate': threshold,
            'confidence_interval': (None, None),
            'method': method,
            'error': str(e)
        }


def vector_parameter_confidence_interval(
    theta: np.ndarray,
    vcov: np.ndarray,
    confidence_level: float = 0.95,
    joint: bool = False
) -> Dict[str, Any]:
    """
    Calculate confidence intervals for vector of parameters.
    
    Parameters
    ----------
    theta : ndarray
        Vector of parameter estimates
    vcov : ndarray
        Covariance matrix of parameter estimates
    confidence_level : float, optional
        Confidence level (0.90, 0.95, 0.99)
    joint : bool, optional
        Whether to calculate joint confidence region
        
    Returns
    -------
    dict
        Confidence interval results
    """
    k = len(theta)
    alpha = 1 - confidence_level
    
    # Calculate individual confidence intervals
    individual_cis = []
    for i in range(k):
        se = np.sqrt(vcov[i, i])
        ci = calculate_confidence_interval(theta[i], se, confidence_level)
        individual_cis.append(ci)
    
    results = {
        'point_estimates': theta,
        'individual_confidence_intervals': individual_cis
    }
    
    # Calculate joint confidence region if requested
    if joint:
        # For joint confidence region, we need to calculate the critical value
        # from the chi-squared distribution
        critical_value = stats.chi2.ppf(confidence_level, df=k)
        
        results['joint_confidence_region'] = {
            'critical_value': critical_value,
            'covariance_matrix': vcov
        }
    
    return results


def robust_confidence_interval(
    model: Any,
    param_index: int,
    confidence_level: float = 0.95,
    method: str = 'hc3'
) -> Dict[str, Any]:
    """
    Calculate robust confidence interval for model parameter.
    
    Parameters
    ----------
    model : Any
        Fitted model object
    param_index : int
        Index of parameter to calculate CI for
    confidence_level : float, optional
        Confidence level (0.90, 0.95, 0.99)
    method : str, optional
        Robust covariance method: 'hc0', 'hc1', 'hc2', 'hc3', or 'cluster'
        
    Returns
    -------
    dict
        Robust confidence interval results
    """
    try:
        # Extract parameter estimate
        theta = model.params[param_index]
        
        # Calculate robust covariance matrix
        if hasattr(model, 'get_robustcov_results'):
            robust_model = model.get_robustcov_results(cov_type=method)
            robust_se = np.sqrt(robust_model.cov_params().iloc[param_index, param_index])
        else:
            # Manual calculation for models without get_robustcov_results
            X = model.model.exog
            n, k = X.shape
            
            # Residuals
            resid = model.resid
            
            # Calculate residual variance matrix based on method
            if method == 'hc0':
                # Basic heteroskedasticity-consistent
                omega = np.diag(resid ** 2)
            elif method == 'hc1':
                # HC1 (scale by n/(n-k))
                scale = n / (n - k)
                omega = np.diag(scale * (resid ** 2))
            elif method == 'hc2':
                # HC2 (scale by 1/(1-h_ii))
                h = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
                omega = np.diag((resid ** 2) / (1 - h))
            elif method == 'hc3':
                # HC3 (scale by 1/(1-h_ii)^2)
                h = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
                omega = np.diag((resid ** 2) / ((1 - h) ** 2))
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Calculate robust covariance matrix
            bread = np.linalg.inv(X.T @ X)
            robust_vcov = bread @ X.T @ omega @ X @ bread
            
            # Extract standard error for parameter
            robust_se = np.sqrt(robust_vcov[param_index, param_index])
        
        # Calculate degrees of freedom
        df = getattr(model, 'df_resid', None)
        
        # Calculate confidence interval
        ci = calculate_confidence_interval(theta, robust_se, confidence_level, df)
        
        # Calculate non-robust standard error for comparison
        regular_se = np.sqrt(model.cov_params().iloc[param_index, param_index])
        regular_ci = calculate_confidence_interval(theta, regular_se, confidence_level, df)
        
        return {
            'parameter': theta,
            'regular_se': regular_se,
            'robust_se': robust_se,
            'regular_ci': regular_ci,
            'robust_ci': ci,
            'method': method,
            'confidence_level': confidence_level
        }
    
    except Exception as e:
        logger.error(f"Error calculating robust confidence interval: {str(e)}")
        return {
            'error': str(e),
            'parameter': None,
            'robust_ci': (None, None)
        }