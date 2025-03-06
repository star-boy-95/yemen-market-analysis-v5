"""
Unit root testing module for time series analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List
from statsmodels.tsa.stattools import adfuller, kpss
import arch.unitroot as unitroot

from src.utils import (
    # Error handling
    handle_errors, ModelError,
    
    # Validation
    validate_time_series, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, memoize,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values following recommended pattern
DEFAULT_ALPHA = config.get('analysis.cointegration.alpha', 0.05)
MAX_LAGS = config.get('analysis.cointegration.max_lags', 4)
TREND = config.get('analysis.cointegration.trend', 'c')


class UnitRootTester:
    """Perform unit root tests on time series data."""
    
    def __init__(self):
        """Initialize the unit root tester."""
        pass
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_adf(
        self, 
        series: Union[pd.Series, np.ndarray], 
        regression: str = 'c', 
        lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        regression : str, optional
            Constant and trend order to include in regression
            'c' : constant only (default)
            'ct' : constant and trend
            'ctt' : constant, and linear and quadratic trend
            'nc' : no constant, no trend
        lags : int, optional
            Number of lags to use in the ADF regression
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        # Validate input with custom validators
        def validate_series_values(s):
            """Check that series has valid numeric values."""
            array = s.values if isinstance(s, pd.Series) else s
            return not (np.isnan(array).any() or np.isinf(array).any())
            
        valid, errors = validate_time_series(
            series, 
            min_length=10,
            max_nulls=0,
            check_constant=True,
            custom_validators=[validate_series_values]
        )
        raise_if_invalid(valid, errors, "Invalid time series for ADF test")
        
        # Convert to numpy array if pandas Series
        if isinstance(series, pd.Series):
            series = series.values
        
        # Run ADF test
        result = adfuller(series, regression=regression, maxlag=lags)
        
        # Format result
        adf_result = {
            'statistic': result[0],
            'pvalue': result[1],
            'usedlag': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'icbest': result[5],
            'stationary': result[1] < DEFAULT_ALPHA
        }
        
        logger.info(f"ADF test: statistic={adf_result['statistic']:.4f}, p-value={adf_result['pvalue']:.4f}")
        return adf_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_adf_gls(
        self, 
        series: Union[pd.Series, np.ndarray], 
        lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform ADF-GLS test (Elliot, Rothenberg, Stock).
        
        Parameters
        ----------
        series : array_like
            The time series to test
        lags : int, optional
            Number of lags to use in the ADF regression
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        # Validate input
        valid, errors = validate_time_series(series, min_length=10)
        raise_if_invalid(valid, errors, "Invalid time series for ADF-GLS test")
        
        # Convert to numpy array if pandas Series
        if isinstance(series, pd.Series):
            series = series.values
        
        # Run ADF-GLS test
        result = unitroot.DFGLS(series, lags=lags)
        
        # Format result
        adfgls_result = {
            'statistic': result.stat,
            'pvalue': result.pvalue,
            'critical_values': result.critical_values,
            'lags': result.lags,
            'stationary': result.pvalue < DEFAULT_ALPHA
        }
        
        logger.info(f"ADF-GLS test: statistic={adfgls_result['statistic']:.4f}, p-value={adfgls_result['pvalue']:.4f}")
        return adfgls_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_kpss(
        self, 
        series: Union[pd.Series, np.ndarray], 
        regression: str = 'c', 
        lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform KPSS test for stationarity.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        regression : str, optional
            'c' : constant only (default)
            'ct' : constant and trend
        lags : int, optional
            Number of lags to use in the KPSS regression
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        # Validate input
        valid, errors = validate_time_series(series, min_length=10)
        raise_if_invalid(valid, errors, "Invalid time series for KPSS test")
        
        # Convert to numpy array if pandas Series
        if isinstance(series, pd.Series):
            series = series.values
        
        # Run KPSS test
        result = kpss(series, regression=regression, nlags=lags)
        
        # Format result (note: KPSS has opposite null hypothesis from ADF)
        kpss_result = {
            'statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[3],
            'stationary': result[1] > DEFAULT_ALPHA  # Note: opposite from ADF
        }
        
        logger.info(f"KPSS test: statistic={kpss_result['statistic']:.4f}, p-value={kpss_result['pvalue']:.4f}")
        return kpss_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_zivot_andrews(
        self, 
        series: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform Zivot-Andrews test for unit root with structural break.
        
        Parameters
        ----------
        series : array_like
            The time series to test
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        # Validate input
        valid, errors = validate_time_series(series, min_length=20)
        raise_if_invalid(valid, errors, "Invalid time series for Zivot-Andrews test")
        
        # Convert to numpy array if pandas Series
        if isinstance(series, pd.Series):
            series = series.values
        
        # Run Zivot-Andrews test
        result = unitroot.ZivotAndrews(series)
        
        # Format result
        za_result = {
            'statistic': result.stat,
            'pvalue': result.pvalue,
            'critical_values': result.critical_values,
            'stationary': result.pvalue < DEFAULT_ALPHA,
            'breakpoint': result.breakpoint
        }
        
        logger.info(
            f"Zivot-Andrews test: statistic={za_result['statistic']:.4f}, "
            f"p-value={za_result['pvalue']:.4f}, breakpoint={za_result['breakpoint']}"
        )
        return za_result
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def run_all_tests(
        self, 
        series: Union[pd.Series, np.ndarray], 
        lags: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all unit root tests.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        lags : int, optional
            Number of lags
            
        Returns
        -------
        dict
            Dictionary with results of all tests
        """
        # Validate input
        valid, errors = validate_time_series(series, min_length=20)
        raise_if_invalid(valid, errors, "Invalid time series for unit root tests")
        
        # Run all tests
        return {
            'adf': self.test_adf(series, lags=lags),
            'adf_gls': self.test_adf_gls(series, lags=lags),
            'kpss': self.test_kpss(series, lags=lags),
            'zivot_andrews': self.test_zivot_andrews(series)
        }


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def determine_integration_order(
    series: Union[pd.Series, np.ndarray], 
    max_order: int = 2, 
    test: str = 'adf'
) -> int:
    """
    Determine order of integration for a time series.
    
    Parameters
    ----------
    series : array_like
        The time series to test
    max_order : int, optional
        Maximum integration order to test
    test : str, optional
        Unit root test to use ('adf', 'adf_gls', 'kpss')
        
    Returns
    -------
    int
        Order of integration
    """
    # Initialize tester
    tester = UnitRootTester()
    
    # Make a copy of the series
    if isinstance(series, pd.Series):
        test_series = series.copy()
    else:
        test_series = np.copy(series)
    
    # Test for unit root and difference if necessary
    for d in range(max_order + 1):
        # Log current integration order being tested
        logger.info(f"Testing integration order {d}")
        
        # Test stationarity
        if test == 'adf':
            result = tester.test_adf(test_series)
        elif test == 'adf_gls':
            result = tester.test_adf_gls(test_series)
        elif test == 'kpss':
            result = tester.test_kpss(test_series)
        else:
            raise ValueError(f"Unknown test: {test}")
        
        # If stationary, return current order
        if result['stationary']:
            logger.info(f"Series is I({d}) - stationary after {d} differences")
            return d
        
        # Difference the series and continue testing
        if d < max_order:
            if isinstance(test_series, pd.Series):
                test_series = test_series.diff().dropna()
            else:
                test_series = np.diff(test_series)
    
    # If we reach here, series has higher order of integration than max_order
    logger.warning(f"Series has integration order > {max_order}")
    return max_order + 1