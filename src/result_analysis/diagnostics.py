"""
Diagnostic testing module for Yemen Market Integration analysis.

This module provides comprehensive diagnostic tests for econometric models,
including tests for heteroskedasticity, serial correlation, normality,
and model specification.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic as diag
from scipy import stats
from typing import Dict, Any, Union, Optional, List, Tuple
import logging

from .statistical_tests import calculate_significance_indicators

logger = logging.getLogger(__name__)

def heteroskedasticity_test(
    model: Any,
    test_type: str = 'white'
) -> Dict[str, Any]:
    """
    Perform heteroskedasticity test on model residuals.
    
    Parameters
    ----------
    model : Any
        Fitted model object with residuals
    test_type : str, optional
        Type of test: 'white', 'breusch_pagan', or 'arch'
        
    Returns
    -------
    dict
        Test results
    """
    # Extract residuals and exog data
    resid = model.resid
    exog = model.model.exog
    
    try:
        if test_type == 'white':
            # White's test
            test_stat, p_value, f_stat, f_p_value = diag.het_white(resid, exog)
            test_name = "White's test"
            distribution = "Chi-squared"
            df = exog.shape[1]
        
        elif test_type == 'breusch_pagan':
            # Breusch-Pagan test
            test_stat, p_value, f_stat, f_p_value = diag.het_breuschpagan(resid, exog)
            test_name = "Breusch-Pagan test"
            distribution = "Chi-squared"
            df = exog.shape[1] - 1
        
        elif test_type == 'arch':
            # ARCH test
            lags = min(4, len(resid) // 5)  # Use reasonable number of lags
            test_stat, p_value, f_stat, f_p_value = diag.het_arch(resid, nlags=lags)
            test_name = "ARCH test"
            distribution = "Chi-squared"
            df = lags
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Determine result
        reject_null = p_value < 0.05
        
        return {
            'test_name': test_name,
            'test_statistic': test_stat,
            'p_value': p_value,
            'f_statistic': f_stat,
            'f_p_value': f_p_value,
            'distribution': distribution,
            'df': df,
            'reject_null': reject_null,
            'significance': calculate_significance_indicators(p_value),
            'interpretation': (
                f"{test_name} indicates " +
                ("heteroskedasticity is present" if reject_null else "homoskedasticity cannot be rejected") +
                f" (p-value: {p_value:.4f})"
            )
        }
    
    except Exception as e:
        logger.error(f"Error in heteroskedasticity test: {str(e)}")
        return {
            'test_name': f"{test_type} test",
            'error': str(e),
            'p_value': None,
            'reject_null': None
        }


def serial_correlation_test(
    model: Any,
    test_type: str = 'breusch_godfrey',
    lags: int = 4
) -> Dict[str, Any]:
    """
    Perform serial correlation test on model residuals.
    
    Parameters
    ----------
    model : Any
        Fitted model object with residuals
    test_type : str, optional
        Type of test: 'breusch_godfrey', 'durbin_watson', or 'ljung_box'
    lags : int, optional
        Number of lags to include
        
    Returns
    -------
    dict
        Test results
    """
    # Extract residuals
    resid = model.resid
    
    try:
        if test_type == 'breusch_godfrey':
            # Perform Breusch-Godfrey test
            bg_test = diag.acorr_breusch_godfrey(model, nlags=lags)
            test_stat, p_value, f_stat, f_p_value = bg_test
            
            # Determine result
            reject_null = p_value < 0.05
            
            return {
                'test_name': "Breusch-Godfrey test",
                'test_statistic': test_stat,
                'p_value': p_value,
                'f_statistic': f_stat,
                'f_p_value': f_p_value,
                'lags': lags,
                'reject_null': reject_null,
                'significance': calculate_significance_indicators(p_value),
                'interpretation': (
                    "Breusch-Godfrey test indicates " +
                    ("serial correlation is present" if reject_null else "no serial correlation") +
                    f" at {lags} lags (p-value: {p_value:.4f})"
                )
            }
            
        elif test_type == 'durbin_watson':
            # Durbin-Watson test (for AR(1) errors)
            dw_stat = diag.durbin_watson(resid)
            
            # Interpret Durbin-Watson statistic
            if dw_stat < 1.5:
                result = "positive autocorrelation"
                significance = "**"  # Arbitrary significance for interpretation
            elif dw_stat > 2.5:
                result = "negative autocorrelation"
                significance = "**"  # Arbitrary significance for interpretation
            else:
                result = "no autocorrelation"
                significance = ""
            
            return {
                'test_name': "Durbin-Watson test",
                'test_statistic': dw_stat,
                'significance': significance,
                'interpretation': f"Durbin-Watson statistic ({dw_stat:.4f}) suggests {result}"
            }
            
        elif test_type == 'ljung_box':
            # Ljung-Box test
            lb_stat, lb_p_value = sm.stats.acorr_ljungbox(resid, lags=[lags])
            test_stat = lb_stat[0]
            p_value = lb_p_value[0]
            
            # Determine result
            reject_null = p_value < 0.05
            
            return {
                'test_name': "Ljung-Box test",
                'test_statistic': test_stat,
                'p_value': p_value,
                'lags': lags,
                'reject_null': reject_null,
                'significance': calculate_significance_indicators(p_value),
                'interpretation': (
                    "Ljung-Box test indicates " +
                    ("serial correlation is present" if reject_null else "no serial correlation") +
                    f" at {lags} lags (p-value: {p_value:.4f})"
                )
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    except Exception as e:
        logger.error(f"Error in serial correlation test: {str(e)}")
        return {
            'test_name': f"{test_type} test",
            'error': str(e),
            'p_value': None,
            'reject_null': None
        }


def normality_test(
    model: Any,
    test_type: str = 'jarque_bera'
) -> Dict[str, Any]:
    """
    Perform normality test on model residuals.
    
    Parameters
    ----------
    model : Any
        Fitted model object with residuals
    test_type : str, optional
        Type of test: 'jarque_bera' or 'shapiro_wilk'
        
    Returns
    -------
    dict
        Test results
    """
    # Extract residuals
    resid = model.resid
    
    try:
        if test_type == 'jarque_bera':
            # Jarque-Bera test
            jb_stat, jb_p_value, skew, kurtosis = diag.jarque_bera(resid)
            
            # Determine result
            reject_null = jb_p_value < 0.05
            
            return {
                'test_name': "Jarque-Bera test",
                'test_statistic': jb_stat,
                'p_value': jb_p_value,
                'skewness': skew,
                'kurtosis': kurtosis,
                'reject_null': reject_null,
                'significance': calculate_significance_indicators(jb_p_value),
                'interpretation': (
                    "Jarque-Bera test " +
                    ("rejects" if reject_null else "does not reject") +
                    f" normality (p-value: {jb_p_value:.4f})"
                )
            }
            
        elif test_type == 'shapiro_wilk':
            # Shapiro-Wilk test (limited to 5000 observations)
            if len(resid) <= 5000:
                sw_stat, sw_p_value = stats.shapiro(resid)
                
                # Determine result
                reject_null = sw_p_value < 0.05
                
                return {
                    'test_name': "Shapiro-Wilk test",
                    'test_statistic': sw_stat,
                    'p_value': sw_p_value,
                    'reject_null': reject_null,
                    'significance': calculate_significance_indicators(sw_p_value),
                    'interpretation': (
                        "Shapiro-Wilk test " +
                        ("rejects" if reject_null else "does not reject") +
                        f" normality (p-value: {sw_p_value:.4f})"
                    )
                }
            else:
                return {
                    'test_name': "Shapiro-Wilk test",
                    'error': "Sample size exceeds 5000 observations",
                    'note': "Use Jarque-Bera test for large samples"
                }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    except Exception as e:
        logger.error(f"Error in normality test: {str(e)}")
        return {
            'test_name': f"{test_type} test",
            'error': str(e),
            'p_value': None,
            'reject_null': None
        }


def model_specification_test(
    model: Any,
    test_type: str = 'reset',
    power: int = 2
) -> Dict[str, Any]:
    """
    Perform model specification test.
    
    Parameters
    ----------
    model : Any
        Fitted model object
    test_type : str, optional
        Type of test: 'reset' or 'rainbow'
    power : int, optional
        Maximum power of fitted values to include (for RESET test)
        
    Returns
    -------
    dict
        Test results
    """
    try:
        if test_type == 'reset':
            # Perform Ramsey RESET test
            reset_test = diag.linear_reset(model, power=power, test_type='fitted', use_f=True)
            test_stat, p_value, powers = reset_test
            
            # Determine result
            reject_null = p_value < 0.05
            
            return {
                'test_name': "Ramsey RESET test",
                'test_statistic': test_stat,
                'p_value': p_value,
                'power': powers,
                'reject_null': reject_null,
                'significance': calculate_significance_indicators(p_value),
                'interpretation': (
                    "Ramsey RESET test " +
                    ("indicates model misspecification" if reject_null else "does not indicate misspecification") +
                    f" (p-value: {p_value:.4f})"
                )
            }
            
        elif test_type == 'rainbow':
            # Perform Rainbow test
            rainbow_test = diag.linear_rainbow(model)
            test_stat, p_value = rainbow_test
            
            # Determine result
            reject_null = p_value < 0.05
            
            return {
                'test_name': "Rainbow test",
                'test_statistic': test_stat,
                'p_value': p_value,
                'reject_null': reject_null,
                'significance': calculate_significance_indicators(p_value),
                'interpretation': (
                    "Rainbow test " +
                    ("indicates parameter instability" if reject_null else "does not indicate parameter instability") +
                    f" (p-value: {p_value:.4f})"
                )
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    except Exception as e:
        logger.error(f"Error in model specification test: {str(e)}")
        return {
            'test_name': f"{test_type} test",
            'error': str(e),
            'p_value': None,
            'reject_null': None
        }


def stability_test(
    model: Any,
    test_type: str = 'cusum'
) -> Dict[str, Any]:
    """
    Perform stability test on model residuals.
    
    Parameters
    ----------
    model : Any
        Fitted model object with residuals
    test_type : str, optional
        Type of test: 'cusum' or 'cusumq'
        
    Returns
    -------
    dict
        Test results
    """
    # Extract residuals
    resid = model.resid
    nobs = len(resid)
    
    try:
        if test_type == 'cusum':
            # CUSUM test
            from statsmodels.stats.diagnostic import breaks_cusumolsresid
            
            # Calculate recursive residuals
            cusum, pval = breaks_cusumolsresid(resid)
            
            # Determine result
            reject_null = pval < 0.05
            
            return {
                'test_name': "CUSUM test",
                'test_statistic': cusum.max(),
                'p_value': pval,
                'reject_null': reject_null,
                'significance': calculate_significance_indicators(pval),
                'interpretation': (
                    "CUSUM test " +
                    ("indicates parameter instability" if reject_null else "does not indicate parameter instability") +
                    f" (p-value: {pval:.4f})"
                )
            }
            
        elif test_type == 'cusumq':
            # CUSUM-sq test
            from statsmodels.stats.diagnostic import breaks_cusumolsresid
            
            # Calculate recursive residuals
            cusumq, pval = breaks_cusumolsresid(resid ** 2)
            
            # Determine result
            reject_null = pval < 0.05
            
            return {
                'test_name': "CUSUM-sq test",
                'test_statistic': cusumq.max(),
                'p_value': pval,
                'reject_null': reject_null,
                'significance': calculate_significance_indicators(pval),
                'interpretation': (
                    "CUSUM-sq test " +
                    ("indicates variance instability" if reject_null else "does not indicate variance instability") +
                    f" (p-value: {pval:.4f})"
                )
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    except Exception as e:
        logger.error(f"Error in stability test: {str(e)}")
        return {
            'test_name': f"{test_type} test",
            'error': str(e),
            'p_value': None,
            'reject_null': None
        }


def run_comprehensive_diagnostics(
    model: Any,
    level: str = 'standard'
) -> Dict[str, Any]:
    """
    Run comprehensive diagnostic tests on a model.
    
    Parameters
    ----------
    model : Any
        Fitted model object
    level : str, optional
        Level of diagnostics: 'basic', 'standard', or 'comprehensive'
        
    Returns
    -------
    dict
        Comprehensive diagnostic results
    """
    results = {}
    
    # Define tests based on level
    if level == 'basic':
        # Basic diagnostics
        het_tests = ['white']
        sc_tests = ['durbin_watson']
        norm_tests = ['jarque_bera']
        spec_tests = []
        stability_tests = []
    
    elif level == 'standard':
        # Standard diagnostics
        het_tests = ['white', 'breusch_pagan']
        sc_tests = ['breusch_godfrey', 'durbin_watson']
        norm_tests = ['jarque_bera']
        spec_tests = ['reset']
        stability_tests = []
    
    elif level == 'comprehensive':
        # Comprehensive diagnostics
        het_tests = ['white', 'breusch_pagan', 'arch']
        sc_tests = ['breusch_godfrey', 'durbin_watson', 'ljung_box']
        norm_tests = ['jarque_bera', 'shapiro_wilk']
        spec_tests = ['reset', 'rainbow']
        stability_tests = ['cusum', 'cusumq']
    
    else:
        raise ValueError(f"Unknown diagnostic level: {level}")
    
    # Run heteroskedasticity tests
    results['heteroskedasticity'] = {}
    for test_type in het_tests:
        results['heteroskedasticity'][test_type] = heteroskedasticity_test(model, test_type=test_type)
    
    # Run serial correlation tests
    results['serial_correlation'] = {}
    for test_type in sc_tests:
        results['serial_correlation'][test_type] = serial_correlation_test(model, test_type=test_type)
    
    # Run normality tests
    results['normality'] = {}
    for test_type in norm_tests:
        results['normality'][test_type] = normality_test(model, test_type=test_type)
    
    # Run specification tests
    results['specification'] = {}
    for test_type in spec_tests:
        results['specification'][test_type] = model_specification_test(model, test_type=test_type)
    
    # Run stability tests
    results['stability'] = {}
    for test_type in stability_tests:
        results['stability'][test_type] = stability_test(model, test_type=test_type)
    
    # Create summary of issues
    het_problem = any(test.get('reject_null', False) 
                     for test_type, test in results['heteroskedasticity'].items() 
                     if isinstance(test, dict) and 'reject_null' in test)
    
    sc_problem = any(test.get('reject_null', False)
                    for test_type, test in results['serial_correlation'].items()
                    if isinstance(test, dict) and 'reject_null' in test)
    
    norm_problem = any(test.get('reject_null', False)
                      for test_type, test in results['normality'].items()
                      if isinstance(test, dict) and 'reject_null' in test)
    
    spec_problem = any(test.get('reject_null', False)
                      for test_type, test in results.get('specification', {}).items()
                      if isinstance(test, dict) and 'reject_null' in test)
    
    stability_problem = any(test.get('reject_null', False)
                           for test_type, test in results.get('stability', {}).items()
                           if isinstance(test, dict) and 'reject_null' in test)
    
    # Create summary of issues
    issues = []
    if het_problem:
        issues.append("heteroskedasticity")
    if sc_problem:
        issues.append("serial correlation")
    if norm_problem:
        issues.append("non-normal residuals")
    if spec_problem:
        issues.append("model misspecification")
    if stability_problem:
        issues.append("parameter instability")
    
    if issues:
        recommendations = []
        if het_problem:
            recommendations.append("use robust standard errors")
        if sc_problem:
            recommendations.append("consider different lag structure or dynamic specification")
        if norm_problem:
            recommendations.append("consider transformations or robust inference methods")
        if spec_problem:
            recommendations.append("revise model specification")
        if stability_problem:
            recommendations.append("check for structural breaks")
        
        results['summary'] = {
            'has_issues': True,
            'issues_detected': issues,
            'recommendations': recommendations,
            'interpretation': (
                f"Diagnostic tests suggest {', '.join(issues)}. "
                f"Recommended actions: {', '.join(recommendations)}."
            )
        }
    else:
        results['summary'] = {
            'has_issues': False,
            'interpretation': "Model passes all diagnostic tests. No issues detected."
        }
    
    return results