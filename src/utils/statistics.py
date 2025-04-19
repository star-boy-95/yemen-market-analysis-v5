"""Statistical utility functions for Yemen Market Analysis.

This module provides statistical utility functions for the Yemen Market Analysis
package. It includes functions for descriptive statistics, statistical tests,
distribution analysis, correlation analysis, and outlier detection.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.gofplots import qqplot

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

# Implement a simple version of the Granger causality test
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def grangercausalitytest(data, maxlag=5):
    """
    Simple implementation of the Granger causality test.

    Args:
        data: DataFrame with columns ['y', 'x']
        maxlag: Maximum number of lags to test

    Returns:
        Dictionary with test results
    """
    results = {}

    for lag in range(1, maxlag + 1):
        # Create lagged variables
        data_lagged = data.copy()

        # Create lags of y
        for i in range(1, lag + 1):
            data_lagged[f'y_lag{i}'] = data['y'].shift(i)
            data_lagged[f'x_lag{i}'] = data['x'].shift(i)

        # Drop missing values
        data_lagged = data_lagged.dropna()

        # Restricted model (y explained by its own lags)
        y_lags = [f'y_lag{i}' for i in range(1, lag + 1)]
        X_restricted = sm.add_constant(data_lagged[y_lags])
        model_restricted = OLS(data_lagged['y'], X_restricted)
        results_restricted = model_restricted.fit()
        ssr_restricted = sum(results_restricted.resid**2)

        # Unrestricted model (y explained by its own lags and x lags)
        x_lags = [f'x_lag{i}' for i in range(1, lag + 1)]
        X_unrestricted = sm.add_constant(data_lagged[y_lags + x_lags])
        model_unrestricted = OLS(data_lagged['y'], X_unrestricted)
        results_unrestricted = model_unrestricted.fit()
        ssr_unrestricted = sum(results_unrestricted.resid**2)

        # Calculate F-statistic
        n = len(data_lagged)
        df_restricted = n - len(y_lags) - 1
        df_unrestricted = n - len(y_lags) - len(x_lags) - 1
        df1 = len(x_lags)
        df2 = df_unrestricted

        f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        # Store results
        results[lag] = [{"ssr_ftest": (f_stat, p_value)}]

    return results


@handle_errors
def descriptive_statistics(
    data: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for the given data.

    Args:
        data: DataFrame containing the data.
        columns: Columns to calculate statistics for. If None, uses all numeric columns.

    Returns:
        DataFrame containing descriptive statistics.

    Raises:
        YemenAnalysisError: If the data is invalid or no numeric columns are found.
    """
    logger.info("Calculating descriptive statistics")

    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        logger.error("Data is not a pandas DataFrame")
        raise YemenAnalysisError("Data is not a pandas DataFrame")

    # Get numeric columns if columns is None
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns.tolist()

    # Check if any numeric columns were found
    if not columns:
        logger.error("No numeric columns found in data")
        raise YemenAnalysisError("No numeric columns found in data")

    try:
        # Calculate descriptive statistics
        stats_df = data[columns].describe()

        # Add additional statistics
        stats_df.loc['skew'] = data[columns].skew()
        stats_df.loc['kurtosis'] = data[columns].kurtosis()
        stats_df.loc['median'] = data[columns].median()
        stats_df.loc['var'] = data[columns].var()
        stats_df.loc['se'] = data[columns].sem()
        stats_df.loc['cv'] = data[columns].std() / data[columns].mean()  # Coefficient of variation

        logger.info("Descriptive statistics calculated successfully")
        return stats_df
    except Exception as e:
        logger.error(f"Error calculating descriptive statistics: {e}")
        raise YemenAnalysisError(f"Error calculating descriptive statistics: {e}")


@handle_errors
def correlation_analysis(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix for the given data.

    Args:
        data: DataFrame containing the data.
        columns: Columns to calculate correlation for. If None, uses all numeric columns.
        method: Correlation method. Options are 'pearson', 'kendall', and 'spearman'.

    Returns:
        DataFrame containing the correlation matrix.

    Raises:
        YemenAnalysisError: If the data is invalid or no numeric columns are found.
    """
    logger.info(f"Calculating {method} correlation matrix")

    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        logger.error("Data is not a pandas DataFrame")
        raise YemenAnalysisError("Data is not a pandas DataFrame")

    # Get numeric columns if columns is None
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns.tolist()

    # Check if any numeric columns were found
    if not columns:
        logger.error("No numeric columns found in data")
        raise YemenAnalysisError("No numeric columns found in data")

    # Validate method
    valid_methods = ['pearson', 'kendall', 'spearman']
    if method not in valid_methods:
        logger.error(f"Invalid correlation method: {method}")
        raise YemenAnalysisError(f"Invalid correlation method: {method}. Valid options are {valid_methods}")

    try:
        # Calculate correlation matrix
        corr_matrix = data[columns].corr(method=method)
        logger.info("Correlation matrix calculated successfully")
        return corr_matrix
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        raise YemenAnalysisError(f"Error calculating correlation matrix: {e}")


@handle_errors
def detect_outliers(
    data: pd.Series,
    method: str = 'zscore',
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in the given data.

    Args:
        data: Series containing the data.
        method: Method for outlier detection. Options are 'zscore', 'iqr', and 'modified_zscore'.
        threshold: Threshold for outlier detection. For 'zscore' and 'modified_zscore',
                  values with absolute z-scores greater than the threshold are considered
                  outliers. For 'iqr', values outside the range [Q1 - threshold * IQR,
                  Q3 + threshold * IQR] are considered outliers.

    Returns:
        Boolean Series indicating outliers.

    Raises:
        YemenAnalysisError: If the data is invalid or the method is invalid.
    """
    logger.info(f"Detecting outliers using {method} method")

    # Check if data is a Series
    if not isinstance(data, pd.Series):
        logger.error("Data is not a pandas Series")
        raise YemenAnalysisError("Data is not a pandas Series")

    # Validate method
    valid_methods = ['zscore', 'iqr', 'modified_zscore']
    if method not in valid_methods:
        logger.error(f"Invalid outlier detection method: {method}")
        raise YemenAnalysisError(f"Invalid outlier detection method: {method}. Valid options are {valid_methods}")

    try:
        if method == 'zscore':
            # Z-score method
            z_scores = (data - data.mean()) / data.std()
            outliers = abs(z_scores) > threshold
        elif method == 'iqr':
            # IQR method
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (data < lower_bound) | (data > upper_bound)
        elif method == 'modified_zscore':
            # Modified Z-score method
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = abs(modified_z_scores) > threshold

        logger.info(f"Detected {outliers.sum()} outliers out of {len(data)} observations")
        return outliers
    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        raise YemenAnalysisError(f"Error detecting outliers: {e}")


@handle_errors
def normality_test(
    data: pd.Series,
    test: str = 'shapiro'
) -> Dict[str, Any]:
    """
    Test for normality of the given data.

    Args:
        data: Series containing the data.
        test: Normality test to use. Options are 'shapiro', 'ks', 'jarque_bera',
              and 'anderson'.

    Returns:
        Dictionary containing the test results.

    Raises:
        YemenAnalysisError: If the data is invalid or the test is invalid.
    """
    logger.info(f"Testing normality using {test} test")

    # Check if data is a Series
    if not isinstance(data, pd.Series):
        logger.error("Data is not a pandas Series")
        raise YemenAnalysisError("Data is not a pandas Series")

    # Validate test
    valid_tests = ['shapiro', 'ks', 'jarque_bera', 'anderson']
    if test not in valid_tests:
        logger.error(f"Invalid normality test: {test}")
        raise YemenAnalysisError(f"Invalid normality test: {test}. Valid options are {valid_tests}")

    try:
        if test == 'shapiro':
            # Shapiro-Wilk test
            stat, p_value = stats.shapiro(data.dropna())
            result = {
                'test': 'Shapiro-Wilk',
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
        elif test == 'ks':
            # Kolmogorov-Smirnov test
            stat, p_value = stats.kstest(data.dropna(), 'norm')
            result = {
                'test': 'Kolmogorov-Smirnov',
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
        elif test == 'jarque_bera':
            # Jarque-Bera test
            jb_stat, p_value, skew, kurtosis = jarque_bera(data.dropna())
            result = {
                'test': 'Jarque-Bera',
                'statistic': jb_stat,
                'p_value': p_value,
                'skew': skew,
                'kurtosis': kurtosis,
                'normal': p_value > 0.05
            }
        elif test == 'anderson':
            # Anderson-Darling test
            result = stats.anderson(data.dropna(), dist='norm')
            critical_values = result.critical_values
            significance_levels = [15, 10, 5, 2.5, 1]

            # Find the highest significance level where the test statistic is greater than the critical value
            for i, (cv, sl) in enumerate(zip(critical_values, significance_levels)):
                if result.statistic > cv:
                    break
            else:
                i = len(significance_levels)

            result = {
                'test': 'Anderson-Darling',
                'statistic': result.statistic,
                'critical_values': dict(zip(significance_levels, critical_values)),
                'significance_level': significance_levels[i-1] if i > 0 else None,
                'normal': i == 0  # Normal if the statistic is less than all critical values
            }

        logger.info(f"Normality test result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error testing normality: {e}")
        raise YemenAnalysisError(f"Error testing normality: {e}")


@handle_errors
def heteroskedasticity_test(
    residuals: pd.Series,
    exog: pd.DataFrame
) -> Dict[str, Any]:
    """
    Test for heteroskedasticity in the residuals.

    Args:
        residuals: Series containing the residuals.
        exog: DataFrame containing the exogenous variables.

    Returns:
        Dictionary containing the test results.

    Raises:
        YemenAnalysisError: If the data is invalid.
    """
    logger.info("Testing for heteroskedasticity")

    # Check if residuals is a Series
    if not isinstance(residuals, pd.Series):
        logger.error("Residuals is not a pandas Series")
        raise YemenAnalysisError("Residuals is not a pandas Series")

    # Check if exog is a DataFrame
    if not isinstance(exog, pd.DataFrame):
        logger.error("Exog is not a pandas DataFrame")
        raise YemenAnalysisError("Exog is not a pandas DataFrame")

    try:
        # Breusch-Pagan test
        bp_stat, bp_p_value, _, _ = het_breuschpagan(residuals, exog)

        result = {
            'test': 'Breusch-Pagan',
            'statistic': bp_stat,
            'p_value': bp_p_value,
            'homoskedastic': bp_p_value > 0.05
        }

        logger.info(f"Heteroskedasticity test result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error testing for heteroskedasticity: {e}")
        raise YemenAnalysisError(f"Error testing for heteroskedasticity: {e}")


@handle_errors
def autocorrelation_test(
    residuals: pd.Series,
    lags: int = 10
) -> Dict[str, Any]:
    """
    Test for autocorrelation in the residuals.

    Args:
        residuals: Series containing the residuals.
        lags: Number of lags to test.

    Returns:
        Dictionary containing the test results.

    Raises:
        YemenAnalysisError: If the data is invalid.
    """
    logger.info(f"Testing for autocorrelation with {lags} lags")

    # Check if residuals is a Series
    if not isinstance(residuals, pd.Series):
        logger.error("Residuals is not a pandas Series")
        raise YemenAnalysisError("Residuals is not a pandas Series")

    try:
        # Ljung-Box test
        lb_stat, lb_p_value = acorr_ljungbox(residuals, lags=[lags])

        # Durbin-Watson test
        dw_stat = durbin_watson(residuals)

        result = {
            'ljung_box': {
                'test': 'Ljung-Box',
                'statistic': lb_stat[0],
                'p_value': lb_p_value[0],
                'no_autocorrelation': lb_p_value[0] > 0.05
            },
            'durbin_watson': {
                'test': 'Durbin-Watson',
                'statistic': dw_stat,
                'no_autocorrelation': 1.5 < dw_stat < 2.5
            }
        }

        logger.info(f"Autocorrelation test result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error testing for autocorrelation: {e}")
        raise YemenAnalysisError(f"Error testing for autocorrelation: {e}")


@handle_errors
def granger_causality(
    y: pd.Series,
    x: pd.Series,
    maxlag: int = 5
) -> Dict[str, Any]:
    """
    Test for Granger causality between two time series.

    Args:
        y: Series containing the dependent variable.
        x: Series containing the independent variable.
        maxlag: Maximum number of lags to test.

    Returns:
        Dictionary containing the test results.

    Raises:
        YemenAnalysisError: If the data is invalid.
    """
    logger.info(f"Testing for Granger causality with {maxlag} lags")

    # Check if y and x are Series
    if not isinstance(y, pd.Series):
        logger.error("y is not a pandas Series")
        raise YemenAnalysisError("y is not a pandas Series")

    if not isinstance(x, pd.Series):
        logger.error("x is not a pandas Series")
        raise YemenAnalysisError("x is not a pandas Series")

    try:
        # Create DataFrame with both series
        data = pd.concat([y, x], axis=1)
        data.columns = ['y', 'x']

        # Drop missing values
        data = data.dropna()

        # Test for Granger causality
        gc_result = grangercausalitytest(data, maxlag=maxlag)

        # Extract results
        result = {
            'test': 'Granger Causality',
            'lags': {}
        }

        for lag in range(1, maxlag + 1):
            # Extract F-statistic and p-value
            f_stat = gc_result[lag][0]['ssr_ftest'][0]
            p_value = gc_result[lag][0]['ssr_ftest'][1]

            result['lags'][lag] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'causality': p_value < 0.05
            }

        logger.info(f"Granger causality test result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error testing for Granger causality: {e}")
        raise YemenAnalysisError(f"Error testing for Granger causality: {e}")


@handle_errors
def stationarity_test(
    data: pd.Series,
    test: str = 'adf',
    regression: str = 'c',
    lags: Optional[int] = None
) -> Dict[str, Any]:
    """
    Test for stationarity of a time series.

    Args:
        data: Series containing the time series.
        test: Stationarity test to use. Options are 'adf' and 'kpss'.
        regression: Regression type for ADF test. Options are 'c' (constant),
                   'ct' (constant and trend), 'ctt' (constant, trend, and quadratic trend),
                   and 'n' (no constant or trend).
        lags: Number of lags to use. If None, uses automatic lag selection.

    Returns:
        Dictionary containing the test results.

    Raises:
        YemenAnalysisError: If the data is invalid or the test is invalid.
    """
    logger.info(f"Testing for stationarity using {test} test")

    # Check if data is a Series
    if not isinstance(data, pd.Series):
        logger.error("Data is not a pandas Series")
        raise YemenAnalysisError("Data is not a pandas Series")

    # Validate test
    valid_tests = ['adf', 'kpss']
    if test not in valid_tests:
        logger.error(f"Invalid stationarity test: {test}")
        raise YemenAnalysisError(f"Invalid stationarity test: {test}. Valid options are {valid_tests}")

    try:
        if test == 'adf':
            # Augmented Dickey-Fuller test
            adf_result = adfuller(data.dropna(), regression=regression, maxlag=lags)

            result = {
                'test': 'Augmented Dickey-Fuller',
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'lags': adf_result[2],
                'n_obs': adf_result[3],
                'critical_values': adf_result[4],
                'stationary': adf_result[1] < 0.05
            }
        elif test == 'kpss':
            # KPSS test
            kpss_result = kpss(data.dropna(), regression=regression, nlags=lags)

            result = {
                'test': 'KPSS',
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'lags': kpss_result[2],
                'critical_values': kpss_result[3],
                'stationary': kpss_result[1] > 0.05  # Note: KPSS null hypothesis is stationarity
            }

        logger.info(f"Stationarity test result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error testing for stationarity: {e}")
        raise YemenAnalysisError(f"Error testing for stationarity: {e}")


@handle_errors
def bootstrap_statistic(
    data: pd.Series,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for a statistic.

    Args:
        data: Series containing the data.
        statistic: Function that calculates the statistic.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level for the intervals.

    Returns:
        Dictionary containing the bootstrap results.

    Raises:
        YemenAnalysisError: If the data is invalid.
    """
    logger.info(f"Calculating bootstrap confidence intervals with {n_bootstrap} samples")

    # Check if data is a Series
    if not isinstance(data, pd.Series):
        logger.error("Data is not a pandas Series")
        raise YemenAnalysisError("Data is not a pandas Series")

    try:
        # Calculate the statistic on the original data
        original_stat = statistic(data.dropna().values)

        # Generate bootstrap samples
        bootstrap_stats = []
        data_values = data.dropna().values
        n = len(data_values)

        for _ in range(n_bootstrap):
            # Sample with replacement
            sample = np.random.choice(data_values, size=n, replace=True)

            # Calculate statistic on the sample
            sample_stat = statistic(sample)
            bootstrap_stats.append(sample_stat)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)

        result = {
            'statistic': original_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'confidence_level': confidence_level,
            'confidence_interval': (lower_bound, upper_bound)
        }

        logger.info(f"Bootstrap results: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calculating bootstrap confidence intervals: {e}")
        raise YemenAnalysisError(f"Error calculating bootstrap confidence intervals: {e}")