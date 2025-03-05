"""
Statistical utilities for econometric analysis in the Yemen Market Integration Project.
Provides optimized functions for statistical tests and modeling.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import scipy.stats as stats
from typing import Union, List, Dict, Any, Optional, Tuple, Callable
import logging
import warnings
from arch.unitroot import DFGLS, PhillipsPerron, ZivotAndrews

from src.utils.error_handler import handle_errors, ModelError
from src.utils.decorators import timer, m1_optimized

logger = logging.getLogger(__name__)

@handle_errors(logger=logger)
def test_stationarity(
    series: Union[pd.Series, np.ndarray],
    test: str = 'adf',
    regression: str = 'c',
    lags: Optional[int] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test time series for stationarity
    
    Parameters
    ----------
    series : array_like
        Time series data
    test : str, optional
        Test type ('adf', 'kpss', 'pp', 'dfgls', 'za')
    regression : str, optional
        Regression type ('c', 'ct', 'ctt', 'n')
    lags : int, optional
        Number of lags
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    # Convert to numpy array if pandas Series
    if isinstance(series, pd.Series):
        series = series.values
    
    # Check for missing values
    if np.any(np.isnan(series)):
        logger.warning("Series contains missing values, removing them")
        series = series[~np.isnan(series)]
    
    # Check minimum length
    if len(series) < 20:
        logger.warning(f"Series is very short (length {len(series)}), test results may be unreliable")
    
    # Initialize result dictionary
    result = {
        'test': test,
        'statistic': None,
        'pvalue': None,
        'critical_values': None,
        'lags': None,
        'stationary': None,
        'parameters': {
            'regression': regression,
            'lags': lags,
            'alpha': alpha
        }
    }
    
    # Perform the selected test
    if test.lower() == 'adf':
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series, regression=regression, maxlag=lags)
        result['statistic'] = adf_result[0]
        result['pvalue'] = adf_result[1]
        result['lags'] = adf_result[2]
        result['nobs'] = adf_result[3]
        result['critical_values'] = adf_result[4]
        result['stationary'] = result['pvalue'] < alpha
        
    elif test.lower() == 'kpss':
        # KPSS test (null hypothesis: stationary)
        # Note the reverse interpretation compared to ADF
        kpss_result = kpss(series, regression=regression, nlags=lags)
        result['statistic'] = kpss_result[0]
        result['pvalue'] = kpss_result[1]
        result['lags'] = lags
        result['critical_values'] = kpss_result[3]
        result['stationary'] = result['pvalue'] >= alpha
        
    elif test.lower() == 'pp':
        # Phillips-Perron test
        pp_test = PhillipsPerron(series, trend=regression, lags=lags)
        result['statistic'] = pp_test.stat
        result['pvalue'] = pp_test.pvalue
        result['lags'] = pp_test.lags
        result['critical_values'] = pp_test.critical_values
        result['stationary'] = result['pvalue'] < alpha
        
    elif test.lower() == 'dfgls':
        # ADF-GLS test
        dfgls_test = DFGLS(series, trend=regression, lags=lags)
        result['statistic'] = dfgls_test.stat
        result['pvalue'] = dfgls_test.pvalue
        result['lags'] = dfgls_test.lags
        result['critical_values'] = dfgls_test.critical_values
        result['stationary'] = result['pvalue'] < alpha
        
    elif test.lower() == 'za':
        # Zivot-Andrews test for unit root with structural break
        za_test = ZivotAndrews(series, trend=regression, lags=lags)
        result['statistic'] = za_test.stat
        result['pvalue'] = za_test.pvalue
        result['lags'] = za_test.lags
        result['critical_values'] = za_test.critical_values
        result['stationary'] = result['pvalue'] < alpha
        result['breakpoint'] = za_test.breakpoint
        
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return result

@handle_errors(logger=logger)
def test_cointegration(
    y: Union[pd.Series, np.ndarray],
    x: Union[pd.Series, np.ndarray, List[Union[pd.Series, np.ndarray]]],
    method: str = 'engle-granger',
    trend: str = 'c',
    lags: Optional[int] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for cointegration between time series
    
    Parameters
    ----------
    y : array_like
        Dependent variable
    x : array_like or list of array_like
        Independent variable(s)
    method : str, optional
        Cointegration test method ('engle-granger', 'johansen')
    trend : str, optional
        Trend specification ('n', 'c', 'ct', 'ctt')
    lags : int, optional
        Number of lags
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    # Convert to numpy arrays if pandas Series
    if isinstance(y, pd.Series):
        y = y.values
    
    # Handle x input: can be a single array or a list of arrays
    if isinstance(x, (list, tuple)):
        x_arrays = []
        for xi in x:
            if isinstance(xi, pd.Series):
                x_arrays.append(xi.values)
            else:
                x_arrays.append(xi)
    else:
        if isinstance(x, pd.Series):
            x_arrays = [x.values]
        else:
            x_arrays = [x]
    
    # Check for missing values
    if np.any(np.isnan(y)):
        logger.warning("y contains missing values, removing corresponding observations")
        mask = ~np.isnan(y)
        y = y[mask]
        x_arrays = [xi[mask] for xi in x_arrays]
    
    for i, xi in enumerate(x_arrays):
        if np.any(np.isnan(xi)):
            logger.warning(f"x{i} contains missing values, removing corresponding observations")
            mask = ~np.isnan(xi)
            y = y[mask]
            x_arrays = [xj[mask] for xj in x_arrays]
    
    # Initialize result dictionary
    result = {
        'method': method,
        'statistic': None,
        'pvalue': None,
        'critical_values': None,
        'lags': lags,
        'cointegrated': None,
        'parameters': {
            'trend': trend,
            'lags': lags,
            'alpha': alpha
        }
    }
    
    # Perform the selected cointegration test
    if method.lower() == 'engle-granger':
        # Engle-Granger two-step procedure
        if len(x_arrays) > 1:
            logger.warning("Engle-Granger test using the first independent variable only")
        
        coint_result = coint(y, x_arrays[0], trend=trend, maxlag=lags, autolag='AIC')
        result['statistic'] = coint_result[0]
        result['pvalue'] = coint_result[1]
        result['critical_values'] = coint_result[2]
        result['cointegrated'] = result['pvalue'] < alpha
        
        # Estimate cointegrating relationship
        X = sm.add_constant(x_arrays[0])
        model = sm.OLS(y, X).fit()
        result['beta'] = model.params
        result['residuals'] = model.resid
        
    elif method.lower() == 'johansen':
        # Johansen test
        # Stack variables
        data = np.column_stack([y] + x_arrays)
        
        # Map trend argument
        if trend == 'n':
            det_order = 0  # No deterministic terms
        elif trend == 'c':
            det_order = 1  # Constant
        elif trend in ['ct', 'ctt']:
            det_order = 2  # Constant and trend
        else:
            raise ValueError(f"Invalid trend specification: {trend}")
        
        # Set default lag order if not provided
        if lags is None:
            # Rule of thumb: lag order = cube root of sample size
            lags = int(np.cbrt(len(y)))
        
        # Perform Johansen test
        johansen_result = coint_johansen(data, det_order=det_order, k_ar_diff=lags)
        
        # Extract trace and eigenvalue statistics
        trace_stats = johansen_result.lr1
        eigen_stats = johansen_result.lr2
        
        # Get critical values (at selected significance level)
        if alpha == 0.01:
            crit_idx = 2
        elif alpha == 0.05:
            crit_idx = 1
        elif alpha == 0.1:
            crit_idx = 0
        else:
            # Default to 5%
            logger.warning(f"Alpha {alpha} not available, using 0.05")
            crit_idx = 1
        
        trace_crit = johansen_result.cvt[:, crit_idx]
        eigen_crit = johansen_result.cvm[:, crit_idx]
        
        # Determine cointegration rank
        rank_trace = sum(trace_stats > trace_crit)
        rank_eigen = sum(eigen_stats > eigen_crit)
        
        # Store results
        result['trace_statistics'] = trace_stats
        result['eigen_statistics'] = eigen_stats
        result['trace_critical_values'] = trace_crit
        result['eigen_critical_values'] = eigen_crit
        result['rank_trace'] = rank_trace
        result['rank_eigen'] = rank_eigen
        result['eigenvectors'] = johansen_result.evec
        result['eigenvalues'] = johansen_result.eig
        
        # Series are cointegrated if rank > 0
        result['cointegrated'] = rank_trace > 0
        
    else:
        raise ValueError(f"Unknown cointegration test method: {method}")
    
    return result

@handle_errors(logger=logger)
def test_granger_causality(
    y: Union[pd.Series, np.ndarray],
    x: Union[pd.Series, np.ndarray],
    max_lags: int = 5,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for Granger causality between time series
    
    Parameters
    ----------
    y : array_like
        Time series that might be caused
    x : array_like
        Time series that might cause y
    max_lags : int, optional
        Maximum number of lags to test
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    # Convert to pandas Series if numpy arrays
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    
    # Create a DataFrame with the two series
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Remove missing values
    data = data.dropna()
    
    # Perform Granger causality test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gc_test = grangercausalitytests(data, maxlag=max_lags, verbose=False)
    
    # Initialize result dictionary
    result = {
        'causality': False,
        'optimal_lag': None,
        'min_pvalue': 1.0,
        'lags': {},
        'parameters': {
            'max_lags': max_lags,
            'alpha': alpha
        }
    }
    
    # Extract results for each lag
    for lag in range(1, max_lags + 1):
        # Use the 'ssr_chi2test' results (default F-test can be accessed with 'ssr_ftest')
        p_value = gc_test[lag][0]['ssr_chi2test'][1]
        result['lags'][lag] = {
            'p_value': p_value,
            'significant': p_value < alpha
        }
        
        # Update minimum p-value and corresponding lag
        if p_value < result['min_pvalue']:
            result['min_pvalue'] = p_value
            result['optimal_lag'] = lag
    
    # Determine if there's evidence of Granger causality
    result['causality'] = result['min_pvalue'] < alpha
    
    return result

@handle_errors(logger=logger)
@m1_optimized(use_numba=True)
def fit_var_model(
    data: Union[pd.DataFrame, np.ndarray],
    lags: int = 1,
    trend: str = 'c',
    season: int = 0
) -> Dict[str, Any]:
    """
    Fit a Vector Autoregression (VAR) model
    
    Parameters
    ----------
    data : array_like
        Multivariate time series data
    lags : int, optional
        Number of lags
    trend : str, optional
        Trend specification ('n', 'c', 'ct', 'ctt')
    season : int, optional
        Number of seasonal periods
        
    Returns
    -------
    dict
        Model results
    """
    from statsmodels.tsa.api import VAR
    
    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # Drop missing values
    data = data.dropna()
    
    # Fit VAR model
    model = VAR(data)
    results = model.fit(lags, trend=trend, season=season)
    
    # Create dictionary with model results
    var_results = {
        'coefficients': results.params,
        'stderr': results.stderr,
        'pvalues': results.pvalues,
        'aic': results.aic,
        'bic': results.bic,
        'fpe': results.fpe,
        'hqic': results.hqic,
        'fitted_values': results.fittedvalues,
        'residuals': results.resid,
        'information_criteria': {
            'aic': results.aic,
            'bic': results.bic,
            'hqic': results.hqic,
            'fpe': results.fpe
        },
        'summary_text': str(results.summary()),
        'parameters': {
            'lags': lags,
            'trend': trend,
            'season': season
        }
    }
    
    return var_results

@handle_errors(logger=logger)
@m1_optimized(use_numba=True)
def fit_vecm_model(
    data: Union[pd.DataFrame, np.ndarray],
    k_ar_diff: int = 1,
    coint_rank: int = 1,
    deterministic: str = 'ci'
) -> Dict[str, Any]:
    """
    Fit a Vector Error Correction Model (VECM)
    
    Parameters
    ----------
    data : array_like
        Multivariate time series data
    k_ar_diff : int, optional
        Number of lagged differences
    coint_rank : int, optional
        Cointegration rank
    deterministic : str, optional
        Deterministic term specification
        
    Returns
    -------
    dict
        Model results
    """
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # Drop missing values
    data = data.dropna()
    
    # Fit VECM model
    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic)
    results = model.fit()
    
    # Create dictionary with model results
    vecm_results = {
        'alpha': results.alpha,
        'beta': results.beta,
        'gamma': results.gamma,
        'pi': results.pi,
        'llf': results.llf,
        'aic': results.aic,
        'bic': results.bic,
        'hqic': results.hqic,
        'fitted_values': results.fittedvalues,
        'residuals': results.resid,
        'parameters': {
            'k_ar_diff': k_ar_diff,
            'coint_rank': coint_rank,
            'deterministic': deterministic
        }
    }
    
    return vecm_results

@handle_errors(logger=logger)
def test_autocorrelation(
    series: Union[pd.Series, np.ndarray], 
    lags: int = 10,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for autocorrelation in a time series
    
    Parameters
    ----------
    series : array_like
        Time series data
    lags : int, optional
        Number of lags to test
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    # Convert to numpy array if pandas Series
    if isinstance(series, pd.Series):
        series = series.values
    
    # Check for missing values
    if np.any(np.isnan(series)):
        logger.warning("Series contains missing values, removing them")
        series = series[~np.isnan(series)]
    
    # Perform Ljung-Box test
    lb_test = acorr_ljungbox(series, lags=lags, return_df=True)
    
    # Initialize result dictionary
    result = {
        'test': 'ljung-box',
        'statistics': lb_test['lb_stat'].values,
        'pvalues': lb_test['lb_pvalue'].values,
        'lags': range(1, lags + 1),
        'significant_lags': [],
        'has_autocorrelation': False,
        'parameters': {
            'lags': lags,
            'alpha': alpha
        }
    }
    
    # Identify significant lags
    for i, p_value in enumerate(result['pvalues']):
        lag = i + 1
        if p_value < alpha:
            result['significant_lags'].append(lag)
    
    # Determine if there's evidence of autocorrelation
    result['has_autocorrelation'] = len(result['significant_lags']) > 0
    
    return result

@handle_errors(logger=logger)
@timer
def test_white_noise(
    series: Union[pd.Series, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test if a time series is white noise
    
    Parameters
    ----------
    series : array_like
        Time series data
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    # Convert to numpy array if pandas Series
    if isinstance(series, pd.Series):
        series = series.values
    
    # Check for missing values
    if np.any(np.isnan(series)):
        logger.warning("Series contains missing values, removing them")
        series = series[~np.isnan(series)]
    
    # Calculate basic statistics
    mean = np.mean(series)
    variance = np.var(series)
    
    # Run autocorrelation test
    acorr_result = test_autocorrelation(series, lags=10, alpha=alpha)
    
    # Run normality test (Jarque-Bera)
    jb_stat, jb_pvalue, skew, kurtosis = stats.jarque_bera(series)
    
    # Initialize result dictionary
    result = {
        'is_white_noise': False,
        'mean': mean,
        'variance': variance,
        'mean_test': {
            'statistic': mean / (np.std(series) / np.sqrt(len(series))),
            'pvalue': 2 * (1 - stats.norm.cdf(abs(mean) / (np.std(series) / np.sqrt(len(series))))),
            'mean_zero': False
        },
        'autocorrelation_test': acorr_result,
        'normality_test': {
            'method': 'jarque-bera',
            'statistic': jb_stat,
            'pvalue': jb_pvalue,
            'skewness': skew,
            'kurtosis': kurtosis,
            'normal': jb_pvalue >= alpha
        }
    }
    
    # Test if mean is significantly different from zero
    result['mean_test']['mean_zero'] = result['mean_test']['pvalue'] >= alpha
    
    # Series is white noise if: mean is zero, no autocorrelation, normally distributed
    result['is_white_noise'] = (
        result['mean_test']['mean_zero'] and
        not acorr_result['has_autocorrelation'] and
        result['normality_test']['normal']
    )
    
    return result

@handle_errors(logger=logger)
def test_covariate_significance(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test significance of covariates in a linear regression
    
    Parameters
    ----------
    y : array_like
        Dependent variable
    X : array_like
        Independent variables
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    # Convert to numpy arrays
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(X, pd.DataFrame):
        X_names = X.columns.tolist()
        X = X.values
    else:
        X_names = [f'x{i}' for i in range(X.shape[1])]
    
    # Check for missing values
    if np.any(np.isnan(y)) or np.any(np.isnan(X)):
        logger.warning("Data contains missing values, removing corresponding observations")
        mask = ~np.isnan(y)
        for i in range(X.shape[1]):
            mask = mask & ~np.isnan(X[:, i])
        
        y = y[mask]
        X = X[mask]
    
    # Add constant term
    X_with_const = sm.add_constant(X)
    X_names = ['const'] + X_names
    
    # Fit the model
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Initialize result dictionary
    result = {
        'coefficients': results.params,
        'std_errors': results.bse,
        'tvalues': results.tvalues,
        'pvalues': results.pvalues,
        'conf_int': results.conf_int(alpha=alpha),
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'f_statistic': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'significant_vars': [],
        'parameters': {
            'alpha': alpha
        }
    }
    
    # Identify significant variables
    for i, name in enumerate(X_names):
        if results.pvalues[i] < alpha:
            result['significant_vars'].append({
                'name': name,
                'coefficient': results.params[i],
                'pvalue': results.pvalues[i]
            })
    
    return result

@handle_errors(logger=logger)
def compute_rolling_correlation(
    series1: Union[pd.Series, np.ndarray],
    series2: Union[pd.Series, np.ndarray],
    window: int = 20
) -> Union[pd.Series, np.ndarray]:
    """
    Compute rolling correlation between two time series
    
    Parameters
    ----------
    series1 : array_like
        First time series
    series2 : array_like
        Second time series
    window : int, optional
        Rolling window size
        
    Returns
    -------
    pandas.Series or numpy.ndarray
        Rolling correlation
    """
    # Convert to pandas Series
    if isinstance(series1, np.ndarray):
        series1 = pd.Series(series1)
    if isinstance(series2, np.ndarray):
        series2 = pd.Series(series2)
    
    # Check for missing values
    if series1.isna().any() or series2.isna().any():
        logger.warning("Series contain missing values, results may be affected")
    
    # Compute rolling correlation
    rolling_corr = series1.rolling(window=window).corr(series2)
    
    return rolling_corr

@handle_errors(logger=logger)
@m1_optimized(use_numba=True)
def estimate_threshold_tar(
    y: Union[pd.Series, np.ndarray],
    threshold_var: Optional[Union[pd.Series, np.ndarray]] = None,
    lags: int = 1,
    n_regimes: int = 2,
    threshold_range: Optional[Tuple[float, float]] = None,
    trim: float = 0.15
) -> Dict[str, Any]:
    """
    Estimate a Threshold Autoregressive (TAR) model
    
    Parameters
    ----------
    y : array_like
        Time series data
    threshold_var : array_like, optional
        Threshold variable (defaults to lagged y)
    lags : int, optional
        Number of lags
    n_regimes : int, optional
        Number of regimes
    threshold_range : tuple, optional
        Range for threshold search
    trim : float, optional
        Trimming percentage
        
    Returns
    -------
    dict
        Model results
    """
    # Convert to numpy arrays
    if isinstance(y, pd.Series):
        y = y.values
    if threshold_var is not None and isinstance(threshold_var, pd.Series):
        threshold_var = threshold_var.values
    
    # Check for missing values
    if np.any(np.isnan(y)):
        logger.warning("y contains missing values, removing them")
        y = y[~np.isnan(y)]
    
    if threshold_var is not None and np.any(np.isnan(threshold_var)):
        logger.warning("threshold_var contains missing values, removing corresponding observations")
        mask = ~np.isnan(threshold_var)
        y = y[mask]
        threshold_var = threshold_var[mask]
    
    # Use lagged y as threshold variable if not provided
    if threshold_var is None:
        threshold_var = np.roll(y, 1)
        # Avoid using the first observation (due to lag)
        threshold_var = threshold_var[1:]
        y = y[1:]
    
    # Check lengths
    if len(y) != len(threshold_var):
        raise ValueError("y and threshold_var must have the same length")
    
    # Create lagged variables for the AR part
    X = np.zeros((len(y) - lags, lags))
    for i in range(lags):
        X[:, i] = y[lags - i - 1:-i - 1] if i < lags - 1 else y[lags - i - 1:]
    
    # Adjust dependent variable and threshold variable
    y_adjusted = y[lags:]
    threshold_var_adjusted = threshold_var[lags:]
    
    # Set threshold range
    if threshold_range is None:
        # Use trimmed range of threshold variable
        sorted_thresh = np.sort(threshold_var_adjusted)
        min_idx = int(len(sorted_thresh) * trim)
        max_idx = int(len(sorted_thresh) * (1 - trim))
        threshold_range = (sorted_thresh[min_idx], sorted_thresh[max_idx])
    
    # Grid search for optimal threshold
    if n_regimes == 2:
        # Single threshold
        thresholds = np.linspace(threshold_range[0], threshold_range[1], 100)
        best_ssr = float('inf')
        best_threshold = None
        best_ar_params = None
        
        for threshold in thresholds:
            # Split data based on threshold
            below_mask = threshold_var_adjusted <= threshold
            above_mask = ~below_mask
            
            # Add constant
            X_below = sm.add_constant(X[below_mask])
            X_above = sm.add_constant(X[above_mask])
            
            # Estimate AR parameters for each regime
            if np.sum(below_mask) > lags + 1 and np.sum(above_mask) > lags + 1:
                model_below = sm.OLS(y_adjusted[below_mask], X_below)
                model_above = sm.OLS(y_adjusted[above_mask], X_above)
                
                try:
                    results_below = model_below.fit()
                    results_above = model_above.fit()
                    
                    # Calculate overall SSR
                    ssr = results_below.ssr + results_above.ssr
                    
                    if ssr < best_ssr:
                        best_ssr = ssr
                        best_threshold = threshold
                        best_ar_params = {
                            'below': results_below.params,
                            'above': results_above.params
                        }
                except:
                    continue
        
        if best_threshold is None:
            raise ValueError("Failed to find optimal threshold")
        
        # Final model using best threshold
        below_mask = threshold_var_adjusted <= best_threshold
        above_mask = ~below_mask
        
        X_below = sm.add_constant(X[below_mask])
        X_above = sm.add_constant(X[above_mask])
        
        model_below = sm.OLS(y_adjusted[below_mask], X_below)
        model_above = sm.OLS(y_adjusted[above_mask], X_above)
        
        results_below = model_below.fit()
        results_above = model_above.fit()
        
        # Save results
        result = {
            'thresholds': [best_threshold],
            'n_regimes': 2,
            'lags': lags,
            'ssr': best_ssr,
            'regimes': {
                'below': {
                    'params': results_below.params,
                    'bse': results_below.bse,
                    'tvalues': results_below.tvalues,
                    'pvalues': results_below.pvalues,
                    'nobs': results_below.nobs
                },
                'above': {
                    'params': results_above.params,
                    'bse': results_above.bse,
                    'tvalues': results_above.tvalues,
                    'pvalues': results_above.pvalues,
                    'nobs': results_above.nobs
                }
            }
        }
        
    elif n_regimes == 3:
        # Two thresholds (three regimes)
        raise NotImplementedError("Three-regime TAR model not implemented")
        
    else:
        raise ValueError("n_regimes must be 2 or 3")
    
    return result

@handle_errors(logger=logger)
def calculate_threshold_ci(
    y: Union[pd.Series, np.ndarray],
    threshold_var: Union[pd.Series, np.ndarray],
    estimated_threshold: float,
    lags: int = 1,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Calculate confidence interval for TAR model threshold
    
    Parameters
    ----------
    y : array_like
        Time series data
    threshold_var : array_like
        Threshold variable
    estimated_threshold : float
        Estimated threshold value
    lags : int, optional
        Number of lags
    confidence : float, optional
        Confidence level
    n_bootstrap : int, optional
        Number of bootstrap replications
        
    Returns
    -------
    dict
        Confidence interval results
    """
    # Convert to numpy arrays
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(threshold_var, pd.Series):
        threshold_var = threshold_var.values
    
    # Check lengths
    if len(y) != len(threshold_var):
        raise ValueError("y and threshold_var must have the same length")
    
    # Create lagged variables for the AR part
    X = np.zeros((len(y) - lags, lags))
    for i in range(lags):
        X[:, i] = y[lags - i - 1:-i - 1] if i < lags - 1 else y[lags - i - 1:]
    
    # Adjust dependent variable and threshold variable
    y_adjusted = y[lags:]
    threshold_var_adjusted = threshold_var[lags:]
    
    # Split data based on estimated threshold
    below_mask = threshold_var_adjusted <= estimated_threshold
    above_mask = ~below_mask
    
    # Add constant
    X_below = sm.add_constant(X[below_mask])
    X_above = sm.add_constant(X[above_mask])
    
    # Estimate AR parameters for each regime
    model_below = sm.OLS(y_adjusted[below_mask], X_below)
    model_above = sm.OLS(y_adjusted[above_mask], X_above)
    
    results_below = model_below.fit()
    results_above = model_above.fit()
    
    # Get residuals
    resid_below = results_below.resid
    resid_above = results_above.resid
    
    # Create residuals vector corresponding to the original data
    residuals = np.zeros_like(y_adjusted)
    residuals[below_mask] = resid_below
    residuals[above_mask] = resid_above
    
    # Fitted values
    fitted_below = results_below.predict(X_below)
    fitted_above = results_above.predict(X_above)
    
    fitted = np.zeros_like(y_adjusted)
    fitted[below_mask] = fitted_below
    fitted[above_mask] = fitted_above
    
    # Bootstrap thresholds
    bootstrap_thresholds = []
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample
        bootstrap_resid = residuals[np.random.randint(0, len(residuals), len(residuals))]
        bootstrap_y = fitted + bootstrap_resid
        
        # Grid search for optimal threshold
        thresholds = np.linspace(
            np.percentile(threshold_var_adjusted, 10),
            np.percentile(threshold_var_adjusted, 90),
            50
        )
        best_ssr = float('inf')
        best_threshold = None
        
        for threshold in thresholds:
            # Split data based on threshold
            below_mask_boot = threshold_var_adjusted <= threshold
            above_mask_boot = ~below_mask_boot
            
            # Add constant
            X_below_boot = sm.add_constant(X[below_mask_boot])
            X_above_boot = sm.add_constant(X[above_mask_boot])
            
            # Estimate AR parameters for each regime
            if np.sum(below_mask_boot) > lags + 1 and np.sum(above_mask_boot) > lags + 1:
                model_below_boot = sm.OLS(bootstrap_y[below_mask_boot], X_below_boot)
                model_above_boot = sm.OLS(bootstrap_y[above_mask_boot], X_above_boot)
                
                try:
                    results_below_boot = model_below_boot.fit()
                    results_above_boot = model_above_boot.fit()
                    
                    # Calculate overall SSR
                    ssr = results_below_boot.ssr + results_above_boot.ssr
                    
                    if ssr < best_ssr:
                        best_ssr = ssr
                        best_threshold = threshold
                except:
                    continue
        
        if best_threshold is not None:
            bootstrap_thresholds.append(best_threshold)
    
    # Calculate confidence interval
    bootstrap_thresholds = np.array(bootstrap_thresholds)
    lower_bound = np.percentile(bootstrap_thresholds, (1 - confidence) * 100 / 2)
    upper_bound = np.percentile(bootstrap_thresholds, 100 - (1 - confidence) * 100 / 2)
    
    # Store results
    result = {
        'estimated_threshold': estimated_threshold,
        'confidence_level': confidence,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'bootstrap_samples': n_bootstrap,
        'bootstrap_thresholds': bootstrap_thresholds
    }
    
    return result

@handle_errors(logger=logger)
@timer
def test_linearity(
    y: Union[pd.Series, np.ndarray],
    threshold_var: Optional[Union[pd.Series, np.ndarray]] = None,
    lags: int = 1,
    method: str = 'hansen'
) -> Dict[str, Any]:
    """
    Test linearity against threshold model alternative
    
    Parameters
    ----------
    y : array_like
        Time series data
    threshold_var : array_like, optional
        Threshold variable (defaults to lagged y)
    lags : int, optional
        Number of lags
    method : str, optional
        Test method ('hansen', 'tsay')
        
    Returns
    -------
    dict
        Test results
    """
    # Convert to numpy arrays
    if isinstance(y, pd.Series):
        y = y.values
    if threshold_var is not None and isinstance(threshold_var, pd.Series):
        threshold_var = threshold_var.values
    
    # Check for missing values
    if np.any(np.isnan(y)):
        logger.warning("y contains missing values, removing them")
        y = y[~np.isnan(y)]
    
    if threshold_var is not None and np.any(np.isnan(threshold_var)):
        logger.warning("threshold_var contains missing values, removing corresponding observations")
        mask = ~np.isnan(threshold_var)
        y = y[mask]
        threshold_var = threshold_var[mask]
    
    # Use lagged y as threshold variable if not provided
    if threshold_var is None:
        threshold_var = np.roll(y, 1)
        # Avoid using the first observation (due to lag)
        threshold_var = threshold_var[1:]
        y = y[1:]
    
    # Check lengths
    if len(y) != len(threshold_var):
        raise ValueError("y and threshold_var must have the same length")
    
    # Create lagged variables for the AR part
    X = np.zeros((len(y) - lags, lags))
    for i in range(lags):
        X[:, i] = y[lags - i - 1:-i - 1] if i < lags - 1 else y[lags - i - 1:]
    
    # Adjust dependent variable and threshold variable
    y_adjusted = y[lags:]
    threshold_var_adjusted = threshold_var[lags:]
    
    # Initialize result dictionary
    result = {
        'method': method,
        'statistic': None,
        'pvalue': None,
        'threshold': None,
        'linearity_rejected': None,
        'parameters': {
            'lags': lags
        }
    }
    
    if method.lower() == 'hansen':
        # Hansen (1999) test
        # First estimate linear AR model
        X_with_const = sm.add_constant(X)
        linear_model = sm.OLS(y_adjusted, X_with_const)
        linear_results = linear_model.fit()
        ssr_linear = linear_results.ssr
        
        # Grid search for optimal threshold
        thresholds = np.linspace(
            np.percentile(threshold_var_adjusted, 15),
            np.percentile(threshold_var_adjusted, 85),
            100
        )
        best_ssr = float('inf')
        best_threshold = None
        
        for threshold in thresholds:
            # Split data based on threshold
            below_mask = threshold_var_adjusted <= threshold
            above_mask = ~below_mask
            
            # Add constant
            X_below = sm.add_constant(X[below_mask])
            X_above = sm.add_constant(X[above_mask])
            
            # Estimate AR parameters for each regime
            if np.sum(below_mask) > lags + 1 and np.sum(above_mask) > lags + 1:
                model_below = sm.OLS(y_adjusted[below_mask], X_below)
                model_above = sm.OLS(y_adjusted[above_mask], X_above)
                
                try:
                    results_below = model_below.fit()
                    results_above = model_above.fit()
                    
                    # Calculate overall SSR
                    ssr = results_below.ssr + results_above.ssr
                    
                    if ssr < best_ssr:
                        best_ssr = ssr
                        best_threshold = threshold
                except:
                    continue
        
        if best_threshold is None:
            raise ValueError("Failed to find optimal threshold")
        
        # Calculate F-statistic
        n = len(y_adjusted)
        k = lags + 1  # Number of parameters per regime (including constant)
        f_stat = ((ssr_linear - best_ssr) / k) / (best_ssr / (n - 2 * k))
        
        # Bootstrap p-value
        n_bootstrap = 1000
        bootstrap_f_stats = []
        
        for _ in range(n_bootstrap):
            # Generate bootstrap sample
            bootstrap_resid = np.random.normal(0, np.std(linear_results.resid), n)
            bootstrap_y = linear_results.predict() + bootstrap_resid
            
            # Estimate linear model
            linear_model_boot = sm.OLS(bootstrap_y, X_with_const)
            linear_results_boot = linear_model_boot.fit()
            ssr_linear_boot = linear_results_boot.ssr
            
            # Grid search for optimal threshold
            best_ssr_boot = float('inf')
            
            for threshold in thresholds:
                # Split data based on threshold
                below_mask = threshold_var_adjusted <= threshold
                above_mask = ~below_mask
                
                # Add constant
                X_below = sm.add_constant(X[below_mask])
                X_above = sm.add_constant(X[above_mask])
                
                # Estimate AR parameters for each regime
                if np.sum(below_mask) > lags + 1 and np.sum(above_mask) > lags + 1:
                    model_below = sm.OLS(bootstrap_y[below_mask], X_below)
                    model_above = sm.OLS(bootstrap_y[above_mask], X_above)
                    
                    try:
                        results_below = model_below.fit()
                        results_above = model_above.fit()
                        
                        # Calculate overall SSR
                        ssr = results_below.ssr + results_above.ssr
                        
                        if ssr < best_ssr_boot:
                            best_ssr_boot = ssr
                    except:
                        continue
            
            # Calculate F-statistic
            f_stat_boot = ((ssr_linear_boot - best_ssr_boot) / k) / (best_ssr_boot / (n - 2 * k))
            bootstrap_f_stats.append(f_stat_boot)
        
        # Calculate p-value
        p_value = np.mean(np.array(bootstrap_f_stats) > f_stat)
        
        # Store results
        result['statistic'] = f_stat
        result['pvalue'] = p_value
        result['threshold'] = best_threshold
        result['linearity_rejected'] = p_value < 0.05
        
    elif method.lower() == 'tsay':
        # Tsay (1989) test
        # Sort data based on threshold variable
        sort_idx = np.argsort(threshold_var_adjusted)
        
        y_sorted = y_adjusted[sort_idx]
        X_sorted = X[sort_idx]
        threshold_var_sorted = threshold_var_adjusted[sort_idx]
        
        # Recursive residuals
        k = lags + 1  # Number of parameters including constant
        n = len(y_sorted)
        m = k + 20  # Starting point for recursive estimation
        
        if n < m + 10:  # Need enough observations
            raise ValueError("Not enough observations for Tsay test")
        
        # First m observations for initial OLS
        X_init = sm.add_constant(X_sorted[:m])
        y_init = y_sorted[:m]
        
        model_init = sm.OLS(y_init, X_init)
        results_init = model_init.fit()
        
        # Initialize arrays for recursive residuals and predictors
        arranged_predictors = np.zeros((n - m, k))
        recursive_residuals = np.zeros(n - m)
        
        # Calculate recursive residuals
        for i in range(m, n):
            # Predict using previous observations
            X_i = sm.add_constant(X_sorted[i:i+1])
            y_pred = results_init.predict(X_i)[0]
            
            # Calculate recursive residual
            y_actual = y_sorted[i]
            recursive_residuals[i - m] = y_actual - y_pred
            
            # Predictors for the final regression
            arranged_predictors[i - m, 0] = 1  # Constant
            arranged_predictors[i - m, 1:] = X_sorted[i]
            
            # Update model with new observation
            X_update = sm.add_constant(X_sorted[:i+1])
            y_update = y_sorted[:i+1]
            
            model_update = sm.OLS(y_update, X_update)
            results_init = model_update.fit()
        
        # Final regression of recursive residuals on predictors
        model_final = sm.OLS(recursive_residuals, arranged_predictors)
        results_final = model_final.fit()
        
        # F-statistic for joint significance
        f_stat = results_final.fvalue
        p_value = results_final.f_pvalue
        
        # Store results
        result['statistic'] = f_stat
        result['pvalue'] = p_value
        result['linearity_rejected'] = p_value < 0.05
        
    else:
        raise ValueError(f"Unknown linearity test method: {method}")
    
    return result

@handle_errors(logger=logger)
def fit_threshold_vecm(
    data: Union[pd.DataFrame, np.ndarray],
    k_ar_diff: int = 1,
    coint_rank: int = 1,
    threshold_variable: Optional[Union[pd.Series, np.ndarray]] = None,
    deterministic: str = 'ci'
) -> Dict[str, Any]:
    """
    Fit a Threshold Vector Error Correction Model (TVECM)
    
    Parameters
    ----------
    data : array_like
        Multivariate time series data
    k_ar_diff : int, optional
        Number of lagged differences
    coint_rank : int, optional
        Cointegration rank
    threshold_variable : array_like, optional
        Threshold variable (defaults to error correction term)
    deterministic : str, optional
        Deterministic term specification
        
    Returns
    -------
    dict
        Model results
    """
    from statsmodels.tsa.vector_ar.vecm import VECM
    
    # Convert to DataFrame if numpy array
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # Drop missing values
    data = data.dropna()
    
    # First fit standard VECM to get cointegration relation
    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic)
    results = model.fit()
    
    # Get error correction term
    if threshold_variable is None:
        # Use first cointegration relation as threshold variable
        beta = results.beta
        if deterministic == 'ci':
            z = np.column_stack([np.ones(len(data)), data.values])[:, :-1]
        else:
            z = data.values
        
        threshold_variable = z @ beta
    
    # Convert threshold variable to numpy array
    if isinstance(threshold_variable, pd.Series):
        threshold_variable = threshold_variable.values
    
    # Grid search for optimal threshold
    # Trimming values
    trim = 0.15
    sorted_thresh = np.sort(threshold_variable)
    min_idx = int(len(sorted_thresh) * trim)
    max_idx = int(len(sorted_thresh) * (1 - trim))
    threshold_range = sorted_thresh[min_idx:max_idx]
    
    # Grid search
    thresholds = np.linspace(threshold_range[0], threshold_range[-1], 100)
    best_llf = -np.inf
    best_threshold = None
    
    for threshold in thresholds:
        # Split data based on threshold
        below_mask = threshold_variable <= threshold
        above_mask = ~below_mask
        
        # Check if enough observations in each regime
        if np.sum(below_mask) < k_ar_diff + coint_rank + 5 or np.sum(above_mask) < k_ar_diff + coint_rank + 5:
            continue
        
        # Fit VECM for each regime
        try:
            model_below = VECM(
                data.iloc[below_mask], 
                k_ar_diff=k_ar_diff, 
                coint_rank=coint_rank, 
                deterministic=deterministic
            )
            model_above = VECM(
                data.iloc[above_mask], 
                k_ar_diff=k_ar_diff, 
                coint_rank=coint_rank, 
                deterministic=deterministic
            )
            
            results_below = model_below.fit(method='ml')
            results_above = model_above.fit(method='ml')
            
            # Calculate overall log-likelihood
            llf = results_below.llf + results_above.llf
            
            if llf > best_llf:
                best_llf = llf
                best_threshold = threshold
        except:
            continue
    
    if best_threshold is None:
        raise ValueError("Failed to find optimal threshold")
    
    # Final model using best threshold
    below_mask = threshold_variable <= best_threshold
    above_mask = ~below_mask
    
    model_below = VECM(
        data.iloc[below_mask], 
        k_ar_diff=k_ar_diff, 
        coint_rank=coint_rank, 
        deterministic=deterministic
    )
    model_above = VECM(
        data.iloc[above_mask], 
        k_ar_diff=k_ar_diff, 
        coint_rank=coint_rank, 
        deterministic=deterministic
    )
    
    results_below = model_below.fit(method='ml')
    results_above = model_above.fit(method='ml')
    
    # Store results
    result = {
        'threshold': best_threshold,
        'llf': best_llf,
        'below_regime': {
            'alpha': results_below.alpha,
            'beta': results_below.beta,
            'gamma': results_below.gamma,
            'pi': results_below.pi,
            'llf': results_below.llf,
            'aic': results_below.aic,
            'bic': results_below.bic,
            'hqic': results_below.hqic,
            'nobs': results_below.nobs
        },
        'above_regime': {
            'alpha': results_above.alpha,
            'beta': results_above.beta,
            'gamma': results_above.gamma,
            'pi': results_above.pi,
            'llf': results_above.llf,
            'aic': results_above.aic,
            'bic': results_above.bic,
            'hqic': results_above.hqic,
            'nobs': results_above.nobs
        },
        'parameters': {
            'k_ar_diff': k_ar_diff,
            'coint_rank': coint_rank,
            'deterministic': deterministic
        }
    }
    
    return result

@handle_errors(logger=logger)
@m1_optimized(use_numba=True)
def calculate_half_life(
    ar_coefficient: float,
    regime: str = 'linear'
) -> float:
    """
    Calculate half-life of shock from AR coefficient
    
    Parameters
    ----------
    ar_coefficient : float
        AR(1) coefficient
    regime : str, optional
        Model regime ('linear', 'threshold')
        
    Returns
    -------
    float
        Half-life in periods
    """
    if regime == 'linear':
        # Linear AR(1) model
        if ar_coefficient >= 1.0:
            return float('inf')  # Non-stationary, no convergence
        elif ar_coefficient <= 0.0:
            return 0.0  # Immediate convergence or oscillation
        else:
            # Half-life formula for AR(1): log(0.5) / log(|ar_coefficient|)
            return np.log(0.5) / np.log(abs(ar_coefficient))
        
    elif regime == 'threshold':
        # Threshold model - same formula but interpret with caution
        if ar_coefficient >= 1.0:
            return float('inf')
        elif ar_coefficient <= 0.0:
            return 0.0
        else:
            return np.log(0.5) / np.log(abs(ar_coefficient))
        
    else:
        raise ValueError(f"Unknown regime: {regime}")

@handle_errors(logger=logger)
def bootstrap_confidence_interval(
    data: Union[pd.Series, np.ndarray],
    statistic_func: Callable,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    method: str = 'percentile',
    **kwargs
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence interval for a statistic
    
    Parameters
    ----------
    data : array_like
        Input data
    statistic_func : callable
        Function to compute statistic
    alpha : float, optional
        Significance level
    n_bootstrap : int, optional
        Number of bootstrap replications
    method : str, optional
        Bootstrap method ('percentile', 'basic', 'bca')
    **kwargs
        Additional arguments for statistic_func
        
    Returns
    -------
    dict
        Confidence interval results
    """
    # Convert to numpy array
    if isinstance(data, pd.Series):
        data = data.values
    
    # Compute statistic on original data
    original_stat = statistic_func(data, **kwargs)
    
    # Generate bootstrap samples
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Generate bootstrap sample
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        
        # Compute statistic
        stat = statistic_func(bootstrap_sample, **kwargs)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute confidence interval
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    if method == 'percentile':
        # Percentile method
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
    elif method == 'basic':
        # Basic bootstrap method
        lower_bound = 2 * original_stat - np.percentile(bootstrap_stats, upper_percentile)
        upper_bound = 2 * original_stat - np.percentile(bootstrap_stats, lower_percentile)
        
    elif method == 'bca':
        # BCa method (bias-corrected and accelerated)
        # This is more complex and would require more code
        raise NotImplementedError("BCa method not implemented")
        
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")
    
    # Store results
    result = {
        'statistic': original_stat,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': 1 - alpha,
        'bootstrap_samples': n_bootstrap,
        'method': method,
        'bootstrap_stats': bootstrap_stats
    }
    
    return result

@handle_errors(logger=logger)
def compute_variance_ratio(
    series: Union[pd.Series, np.ndarray],
    periods: List[int] = [2, 5, 10],
    overlapping: bool = True
) -> Dict[str, Any]:
    """
    Compute variance ratio test for random walk hypothesis
    
    Parameters
    ----------
    series : array_like
        Time series data
    periods : list, optional
        List of periods to compute ratios
    overlapping : bool, optional
        Whether to use overlapping windows
        
    Returns
    -------
    dict
        Variance ratio test results
    """
    # Convert to numpy array
    if isinstance(series, pd.Series):
        series = series.values
    
    # Compute log returns
    log_prices = np.log(series)
    log_returns = np.diff(log_prices)
    
    # Compute variance ratio for each period
    results = {}
    
    for period in periods:
        if period >= len(log_returns):
            logger.warning(f"Period {period} is too large for the given series")
            continue
        
        # Compute k-period returns
        if overlapping:
            # Overlapping windows
            k_returns = np.zeros(len(log_returns) - period + 1)
            for i in range(len(k_returns)):
                k_returns[i] = np.sum(log_returns[i:i+period])
        else:
            # Non-overlapping windows
            n_windows = len(log_returns) // period
            k_returns = np.zeros(n_windows)
            for i in range(n_windows):
                k_returns[i] = np.sum(log_returns[i*period:(i+1)*period])
        
        # Compute variances
        var1 = np.var(log_returns)
        vark = np.var(k_returns) / period
        
        # Compute variance ratio
        vr = vark / var1
        
        # Compute standard error (Lo-MacKinlay)
        n = len(log_returns)
        if overlapping:
            phi = 0
            for j in range(1, period):
                phi += (2 * (period - j) / period) ** 2
            std_err = np.sqrt((2 * (2 * period - 1) * (period - 1)) / (3 * period * n))
        else:
            phi = 0
            std_err = np.sqrt((2 * (period - 1)) / (period * n_windows))
        
        # Compute z-statistic and p-value
        z_stat = (vr - 1) / std_err
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Store results for this period
        results[period] = {
            'variance_ratio': vr,
            'z_statistic': z_stat,
            'p_value': p_value,
            'std_error': std_err,
            'random_walk': p_value >= 0.05
        }
    
    # Store overall results
    result = {
        'period_results': results,
        'random_walk_rejected': any(not res['random_walk'] for res in results.values())
    }
    
    return result

@handle_errors(logger=logger)
def test_causality_granger(
    y: Union[pd.Series, np.ndarray],
    x: Union[pd.Series, np.ndarray],
    maxlag: int = 5,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for Granger causality from x to y
    
    Parameters
    ----------
    y : array_like
        Time series that might be caused
    x : array_like
        Time series that might cause y
    maxlag : int, optional
        Maximum number of lags to test
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    # Convert to pandas Series if numpy arrays
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    
    # Create a DataFrame with the two series
    data = pd.DataFrame({'y': y, 'x': x})
    
    # Remove missing values
    data = data.dropna()
    
    # Perform Granger causality test
    test_results = {}
    
    for lag in range(1, maxlag + 1):
        # Create lagged variables for restricted model (y ~ y lags)
        X_restricted = np.zeros((len(data) - lag, lag))
        for i in range(lag):
            X_restricted[:, i] = data['y'].iloc[lag - i - 1:-i - 1].values
        
        # Create lagged variables for unrestricted model (y ~ y lags + x lags)
        X_unrestricted = np.zeros((len(data) - lag, 2 * lag))
        for i in range(lag):
            X_unrestricted[:, i] = data['y'].iloc[lag - i - 1:-i - 1].values
            X_unrestricted[:, lag + i] = data['x'].iloc[lag - i - 1:-i - 1].values
        
        # Add constant
        X_restricted = sm.add_constant(X_restricted)
        X_unrestricted = sm.add_constant(X_unrestricted)
        
        # Get dependent variable
        y_lag = data['y'].iloc[lag:].values
        
        # Fit restricted model
        model_restricted = sm.OLS(y_lag, X_restricted)
        results_restricted = model_restricted.fit()
        
        # Fit unrestricted model
        model_unrestricted = sm.OLS(y_lag, X_unrestricted)
        results_unrestricted = model_unrestricted.fit()
        
        # Compute F-statistic
        ssr_restricted = results_restricted.ssr
        ssr_unrestricted = results_unrestricted.ssr
        
        df1 = lag  # Number of restrictions
        df2 = len(y_lag) - 2 * lag - 1  # Degrees of freedom in the unrestricted model
        
        f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        # Store results for this lag
        test_results[lag] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'df1': df1,
            'df2': df2,
            'restricted_ssr': ssr_restricted,
            'unrestricted_ssr': ssr_unrestricted,
            'significant': p_value < alpha
        }
    
    # Select optimal lag based on minimal p-value
    optimal_lag = min(test_results.keys(), key=lambda k: test_results[k]['p_value'])
    
    # Store overall results
    result = {
        'lag_results': test_results,
        'optimal_lag': optimal_lag,
        'min_pvalue': test_results[optimal_lag]['p_value'],
        'causality_detected': any(res['significant'] for res in test_results.values()),
        'parameters': {
            'maxlag': maxlag,
            'alpha': alpha
        }
    }
    
    return result

@handle_errors(logger=logger)
def test_structural_break(
    y: Union[pd.Series, np.ndarray],
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    method: str = 'chow',
    break_date: Optional[int] = None,
    trim: float = 0.15,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for structural break in a time series or regression model
    
    Parameters
    ----------
    y : array_like
        Dependent variable
    X : array_like, optional
        Independent variables (if None, tests for a break in y)
    method : str, optional
        Test method ('chow', 'quandt', 'andrews')
    break_date : int, optional
        Index of the break date (required for Chow test)
    trim : float, optional
        Trimming percentage for Quandt and Andrews tests
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    # Convert to numpy arrays
    if isinstance(y, pd.Series):
        y = y.values
    if X is not None and isinstance(X, pd.DataFrame):
        X_names = X.columns.tolist()
        X = X.values
    elif X is None:
        # For time series only, create a time trend
        t = np.arange(len(y))
        X = sm.add_constant(t.reshape(-1, 1))
        X_names = ['const', 'trend']
    else:
        X_names = [f'x{i}' for i in range(X.shape[1])]
        X = sm.add_constant(X)
        X_names = ['const'] + X_names
    
    # Trim for Quandt and Andrews tests
    n = len(y)
    min_idx = int(n * trim)
    max_idx = int(n * (1 - trim))
    
    # Initialize result dictionary
    result = {
        'method': method,
        'statistic': None,
        'pvalue': None,
        'break_date': break_date,
        'significant': None,
        'parameters': {
            'trim': trim,
            'alpha': alpha
        }
    }
    
    if method.lower() == 'chow':
        # Chow test for a known break date
        if break_date is None:
            raise ValueError("break_date must be provided for Chow test")
        
        # Fit full model
        model_full = sm.OLS(y, X)
        results_full = model_full.fit()
        ssr_full = results_full.ssr
        
        # Fit models for each subsample
        X1 = X[:break_date]
        y1 = y[:break_date]
        X2 = X[break_date:]
        y2 = y[break_date:]
        
        model1 = sm.OLS(y1, X1)
        model2 = sm.OLS(y2, X2)
        
        results1 = model1.fit()
        results2 = model2.fit()
        
        ssr1 = results1.ssr
        ssr2 = results2.ssr
        
        # Calculate Chow F-statistic
        k = X.shape[1]  # Number of parameters
        df1 = k
        df2 = n - 2 * k
        
        f_stat = ((ssr_full - (ssr1 + ssr2)) / df1) / ((ssr1 + ssr2) / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        # Store results
        result['statistic'] = f_stat
        result['pvalue'] = p_value
        result['significant'] = p_value < alpha
        result['parameters']['df1'] = df1
        result['parameters']['df2'] = df2
        
    elif method.lower() == 'quandt':
        # Quandt likelihood ratio test (sup-F test)
        f_stats = []
        
        # Fit full model
        model_full = sm.OLS(y, X)
        results_full = model_full.fit()
        ssr_full = results_full.ssr
        
        # Calculate F-statistics for each possible break date
        for break_idx in range(min_idx, max_idx):
            X1 = X[:break_idx]
            y1 = y[:break_idx]
            X2 = X[break_idx:]
            y2 = y[break_idx:]
            
            model1 = sm.OLS(y1, X1)
            model2 = sm.OLS(y2, X2)
            
            results1 = model1.fit()
            results2 = model2.fit()
            
            ssr1 = results1.ssr
            ssr2 = results2.ssr
            
            # Calculate F-statistic
            k = X.shape[1]  # Number of parameters
            df1 = k
            df2 = n - 2 * k
            
            f_stat = ((ssr_full - (ssr1 + ssr2)) / df1) / ((ssr1 + ssr2) / df2)
            f_stats.append((break_idx, f_stat))
        
        # Find break date with maximum F-statistic
        break_idx, max_f = max(f_stats, key=lambda x: x[1])
        
        # Store results
        result['statistic'] = max_f
        result['break_date'] = break_idx
        result['f_stats'] = f_stats
        
        # Calculate p-value based on Andrews (1993) critical values
        # This is an approximation as the exact distribution is complex
        # The implementer should update this with more accurate critical values
        critical_values = {
            0.01: 12.90,
            0.05: 8.58,
            0.10: 7.04
        }
        
        result['significant'] = max_f > critical_values.get(alpha, 8.58)
        result['critical_value'] = critical_values.get(alpha, 8.58)
        
    elif method.lower() == 'andrews':
        # Andrews-Ploberger test (exp-F test)
        f_stats = []
        
        # Fit full model
        model_full = sm.OLS(y, X)
        results_full = model_full.fit()
        ssr_full = results_full.ssr
        
        # Calculate F-statistics for each possible break date
        for break_idx in range(min_idx, max_idx):
            X1 = X[:break_idx]
            y1 = y[:break_idx]
            X2 = X[break_idx:]
            y2 = y[break_idx:]
            
            model1 = sm.OLS(y1, X1)
            model2 = sm.OLS(y2, X2)
            
            results1 = model1.fit()
            results2 = model2.fit()
            
            ssr1 = results1.ssr
            ssr2 = results2.ssr
            
            # Calculate F-statistic
            k = X.shape[1]  # Number of parameters
            df1 = k
            df2 = n - 2 * k
            
            f_stat = ((ssr_full - (ssr1 + ssr2)) / df1) / ((ssr1 + ssr2) / df2)
            f_stats.append(f_stat)
        
        # Calculate Andrews-Ploberger statistic
        exp_f = np.mean(np.exp(0.5 * np.array(f_stats)))
        
        # Store results
        result['statistic'] = exp_f
        result['f_stats'] = f_stats
        
        # Calculate p-value based on Andrews and Ploberger (1994) critical values
        # This is an approximation as the exact distribution is complex
        critical_values = {
            0.01: 6.19,
            0.05: 4.22,
            0.10: 3.37
        }
        
        result['significant'] = exp_f > critical_values.get(alpha, 4.22)
        result['critical_value'] = critical_values.get(alpha, 4.22)
        
    else:
        raise ValueError(f"Unknown structural break test method: {method}")
    
    return result