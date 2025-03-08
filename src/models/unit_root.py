"""
Unit root testing module for time series analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import gc
import psutil
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.stattools import adfuller, kpss
import arch.unitroot as unitroot
import ruptures as rpt

from src.utils import (
    # Error handling
    handle_errors, ModelError,
    
    # Validation
    validate_time_series, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, memoize, disk_cache,
    configure_system_for_performance, optimize_dataframe, parallelize_dataframe,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values following recommended pattern
DEFAULT_ALPHA = config.get('analysis.cointegration.alpha', 0.05)
MAX_LAGS = config.get('analysis.cointegration.max_lags', 4)
TREND = config.get('analysis.cointegration.trend', 'c')

# Configure system for optimal performance
configure_system_for_performance()

class UnitRootTester:
    """Perform unit root tests on time series data."""
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def __init__(self):
        """Initialize the unit root tester."""
        # Get number of available workers based on CPU count
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"Initialized UnitRootTester. Memory usage: {memory_usage:.2f} MB")
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
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
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
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
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"ADF test: statistic={adf_result['statistic']:.4f}, p-value={adf_result['pvalue']:.4f}. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return adf_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
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
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
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
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"ADF-GLS test: statistic={adfgls_result['statistic']:.4f}, p-value={adfgls_result['pvalue']:.4f}. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return adfgls_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
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
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
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
            'lags': result[2],
            'critical_values': result[3],
            'stationary': result[1] > DEFAULT_ALPHA  # Note: opposite from ADF
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"KPSS test: statistic={kpss_result['statistic']:.4f}, p-value={kpss_result['pvalue']:.4f}. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return kpss_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
    def test_phillips_perron(
        self, 
        series: Union[pd.Series, np.ndarray],
        trend: str = 'c',
        lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Phillips-Perron test for unit root.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        trend : str, optional
            'c' : constant only (default)
            'ct' : constant and trend
            'nc' : no constant, no trend
        lags : int, optional
            Number of lags to use in the PP regression
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate input
        valid, errors = validate_time_series(series, min_length=10)
        raise_if_invalid(valid, errors, "Invalid time series for Phillips-Perron test")
        
        # Convert to numpy array if pandas Series
        if isinstance(series, pd.Series):
            series = series.values
        
        # Run Phillips-Perron test
        result = unitroot.PhillipsPerron(series, trend=trend, lags=lags)
        
        # Format result
        pp_result = {
            'statistic': result.stat,
            'pvalue': result.pvalue,
            'critical_values': result.critical_values,
            'lags': result.lags,
            'stationary': result.pvalue < DEFAULT_ALPHA
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Phillips-Perron test: statistic={pp_result['statistic']:.4f}, p-value={pp_result['pvalue']:.4f}. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return pp_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
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
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate input
        valid, errors = validate_time_series(series, min_length=20)
        raise_if_invalid(valid, errors, "Invalid time series for Zivot-Andrews test")
        
        # Store original index if Series with DatetimeIndex
        has_datetime_index = isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex)
        original_index = series.index if has_datetime_index else None
        
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
        
        # Add breakpoint_date if datetime index is available
        if has_datetime_index and original_index is not None:
            try:
                breakpoint_idx = result.breakpoint
                if 0 <= breakpoint_idx < len(original_index):
                    za_result['breakpoint_date'] = original_index[breakpoint_idx]
            except (IndexError, TypeError) as e:
                logger.warning(f"Could not determine breakpoint date: {e}")
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(
            f"Zivot-Andrews test: statistic={za_result['statistic']:.4f}, "
            f"p-value={za_result['pvalue']:.4f}, breakpoint={za_result['breakpoint']}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection
        gc.collect()
        
        return za_result
    
    @timer
    @m1_optimized(parallel=True)
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
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate input
        valid, errors = validate_time_series(series, min_length=20)
        raise_if_invalid(valid, errors, "Invalid time series for unit root tests")
        
        # Run tests in parallel for better performance
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit test tasks
            futures = {}
            futures['adf'] = executor.submit(self.test_adf, series, lags=lags)
            futures['adf_gls'] = executor.submit(self.test_adf_gls, series, lags=lags)
            futures['kpss'] = executor.submit(self.test_kpss, series, lags=lags)
            futures['phillips_perron'] = executor.submit(self.test_phillips_perron, series, lags=lags)
            
            # Zivot-Andrews test is more computationally intensive, so run it separately
            futures['zivot_andrews'] = executor.submit(self.test_zivot_andrews, series)
            
            # Collect results as they complete
            for test_name, future in futures.items():
                try:
                    results[test_name] = future.result()
                except Exception as e:
                    logger.warning(f"Error in {test_name} test: {e}")
                    results[test_name] = {'error': str(e)}
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"All unit root tests complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return results
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def determine_integration_order(
        self, 
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
            Unit root test to use ('adf', 'adf_gls', 'kpss', 'phillips_perron')
            
        Returns
        -------
        int
            Order of integration
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Make a copy of the series to avoid modifying the original
        if isinstance(series, pd.Series):
            test_series = series.copy()
        else:
            test_series = np.copy(series)
        
        # Test for unit root and difference if necessary
        for d in range(max_order + 1):
            # Log current integration order being tested
            logger.info(f"Testing integration order {d}")
            
            # Test stationarity using the specified test method
            if test == 'adf':
                result = self.test_adf(test_series)
            elif test == 'adf_gls':
                result = self.test_adf_gls(test_series)
            elif test == 'kpss':
                result = self.test_kpss(test_series)
            elif test == 'phillips_perron':
                result = self.test_phillips_perron(test_series)
            else:
                raise ValueError(f"Unknown test: {test}")
            
            # If stationary, return current order
            if result['stationary']:
                logger.info(f"Series is I({d}) - stationary after {d} differences")
                
                # Track memory after processing
                end_mem = process.memory_info().rss / (1024 * 1024)  # MB
                memory_diff = end_mem - start_mem
                logger.debug(f"Integration order determination complete. Memory usage: {memory_diff:.2f} MB")
                
                return d
            
            # Difference the series and continue testing
            if d < max_order:
                if isinstance(test_series, pd.Series):
                    test_series = test_series.diff().dropna()
                else:
                    test_series = np.diff(test_series)
        
        # If we reach here, series has higher order of integration than max_order
        logger.warning(f"Series has integration order > {max_order}")
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        logger.debug(f"Integration order determination complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return max_order + 1


class StructuralBreakTester:
    """Detect structural breaks in time series data."""
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def __init__(self):
        """Initialize the structural break tester."""
        # Get number of available workers based on CPU count
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"Initialized StructuralBreakTester. Memory usage: {memory_usage:.2f} MB")
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
    def test_bai_perron(
        self,
        series: Union[pd.Series, np.ndarray],
        min_size: int = 5,
        n_breaks: int = 3,
        method: str = 'dynp',
        pen: float = 3.0
    ) -> Dict[str, Any]:
        """
        Detect multiple structural breaks using Bai-Perron method via ruptures.
        
        Parameters
        ----------
        series : array_like
            The time series to analyze
        min_size : int, optional
            Minimum segment length
        n_breaks : int, optional
            Maximum number of breaks to detect
        method : str, optional
            Detection method: 'dynp' (dynamic programming) or 'binseg' (binary segmentation)
        pen : float, optional
            Penalty for adding a breakpoint
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate input
        valid, errors = validate_time_series(series, min_length=2*min_size)
        raise_if_invalid(valid, errors, "Invalid time series for Bai-Perron test")
        
        # Store original index if Series with DatetimeIndex
        has_datetime_index = isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex)
        original_index = series.index if has_datetime_index else None
        
        # Convert to numpy array if pandas Series
        if isinstance(series, pd.Series):
            array = series.values
        else:
            array = np.array(series)
        
        array = array.reshape(-1, 1)  # Ensure 2D for ruptures
        
        # Select algorithm
        if method == 'dynp':
            algo = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(array)
        elif method == 'binseg':
            algo = rpt.Binseg(model="l2", min_size=min_size).fit(array)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        # Get optimal breakpoints
        breakpoints = algo.predict(n_bkps=n_breaks, pen=pen)
        
        # Remove the last breakpoint if it's just the series length
        if breakpoints and breakpoints[-1] == len(array):
            breakpoints = breakpoints[:-1]
        
        # Create breakpoint_dates list if datetime index available
        breakpoint_dates = None
        if has_datetime_index and original_index is not None:
            try:
                breakpoint_dates = [original_index[bp-1] for bp in breakpoints]
            except (IndexError, TypeError) as e:
                logger.warning(f"Could not determine breakpoint dates: {e}")
        
        # Format result
        bp_result = {
            'breakpoints': breakpoints,
            'n_breakpoints': len(breakpoints),
            'method': method,
            'pen': pen
        }
        
        # Add dates if available
        if breakpoint_dates:
            bp_result['breakpoint_dates'] = breakpoint_dates
        
        # Create segments (for potential plotting)
        segments = []
        start = 0
        for bp in breakpoints:
            segments.append((start, bp))
            start = bp
        
        bp_result['segments'] = segments
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Bai-Perron test: detected {bp_result['n_breakpoints']} breakpoints. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return bp_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
    def test_gregory_hansen(
        self,
        y: Union[pd.Series, np.ndarray],
        x: Union[pd.Series, pd.DataFrame, np.ndarray],
        model: str = 'cc'
    ) -> Dict[str, Any]:
        """
        Gregory-Hansen test for cointegration with regime shifts.
        
        Parameters
        ----------
        y : array_like
            Dependent variable time series
        x : array_like
            Independent variable(s) time series
        model : str, optional
            Type of structural change model:
            'cc' : level shift (default)
            'ct' : level shift with trend
            'ctt' : regime shift (intercept and slope)
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate input
        valid_y, errors_y = validate_time_series(y, min_length=30)
        raise_if_invalid(valid_y, errors_y, "Invalid dependent variable for Gregory-Hansen test")
        
        # Store original index if Series with DatetimeIndex
        has_datetime_index = isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex)
        original_index = y.index if has_datetime_index else None
        
        # Prepare data
        y_array = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Handle x as DataFrame, Series, or ndarray
        if isinstance(x, pd.DataFrame):
            x_array = x.values
        elif isinstance(x, pd.Series):
            x_array = x.values.reshape(-1, 1)
        else:
            x_array = np.array(x)
            if x_array.ndim == 1:
                x_array = x_array.reshape(-1, 1)
        
        # Ensure y and x have the same length
        if len(y_array) != len(x_array):
            raise ValueError("Dependent and independent variables must have the same length")
        
        # For large datasets, process in chunks to reduce memory usage
        if len(y_array) > 5000:
            return self._process_gregory_hansen_parallel(y_array, x_array, model, original_index)
        
        # Use unitroot_adf test with rolling windows to detect structural breaks
        n = len(y_array)
        trim = int(0.15 * n)  # Trim 15% from each end
        test_range = range(trim, n - trim)
        
        min_adf = np.inf
        breakpoint = None
        results = []
        
        # Setup unit root tester
        ur_tester = UnitRootTester()
        
        # Test each possible breakpoint
        for i in test_range:
            # Create dummy variable for break
            dummy = np.zeros(n)
            dummy[i:] = 1
            
            # Create regressor matrix based on model type
            if model == 'cc':
                # Level shift only
                X = np.column_stack((np.ones(n), dummy, x_array))
            elif model == 'ct':
                # Level shift with trend
                trend = np.arange(n)
                X = np.column_stack((np.ones(n), dummy, trend, x_array))
            elif model == 'ctt':
                # Regime shift (intercept and slope)
                X_with_dummy = x_array * dummy.reshape(-1, 1)
                X = np.column_stack((np.ones(n), dummy, x_array, X_with_dummy))
            else:
                raise ValueError(f"Unknown model type: {model}")
            
            # OLS regression
            try:
                from statsmodels.regression.linear_model import OLS
                model_fit = OLS(y_array, X).fit()
                residuals = model_fit.resid
                
                # Test for cointegration (stationarity of residuals)
                adf_result = ur_tester.test_adf(residuals, regression='nc')
                
                # Store results
                result = {
                    'breakpoint': i,
                    'adf_stat': adf_result['statistic'],
                    'pvalue': adf_result['pvalue']
                }
                results.append(result)
                
                # Update minimum ADF statistic
                if adf_result['statistic'] < min_adf:
                    min_adf = adf_result['statistic'] 
                    breakpoint = i
                
            except Exception as e:
                logger.warning(f"Error during Gregory-Hansen testing at breakpoint {i}: {e}")
                continue
        
        # Format final result
        critical_values = {
            'cc': {'1%': -5.13, '5%': -4.61, '10%': -4.34},
            'ct': {'1%': -5.45, '5%': -4.99, '10%': -4.72},
            'ctt': {'1%': -5.47, '5%': -4.95, '10%': -4.68}
        }.get(model, {'1%': -5.13, '5%': -4.61, '10%': -4.34})
        
        # Determine if cointegrated (stationary residuals)
        cointegrated = min_adf < critical_values['5%']
        
        gh_result = {
            'adf_stat': min_adf,
            'breakpoint': breakpoint,
            'critical_values': critical_values,
            'cointegrated': cointegrated,
            'model': model,
            'all_tests': results
        }
        
        # Add breakpoint_date if datetime index is available
        if has_datetime_index and original_index is not None and breakpoint is not None:
            try:
                gh_result['breakpoint_date'] = original_index[breakpoint]
            except (IndexError, TypeError) as e:
                logger.warning(f"Could not determine breakpoint date: {e}")
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(
            f"Gregory-Hansen test: ADF stat={gh_result['adf_stat']:.4f}, "
            f"breakpoint={gh_result['breakpoint']}, cointegrated={gh_result['cointegrated']}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection
        gc.collect()
        
        return gh_result
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def _process_gregory_hansen_parallel(
        self,
        y_array: np.ndarray,
        x_array: np.ndarray,
        model: str,
        original_index: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        Process Gregory-Hansen test in parallel for large datasets.
        
        Parameters
        ----------
        y_array : np.ndarray
            Dependent variable array
        x_array : np.ndarray
            Independent variable(s) array
        model : str
            Model type
        original_index : pd.DatetimeIndex, optional
            Original datetime index
            
        Returns
        -------
        Dict[str, Any]
            Test results
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Get dimensions
        n = len(y_array)
        trim = int(0.15 * n)  # Trim 15% from each end
        test_range = list(range(trim, n - trim))
        
        # Split test range into batches for parallel processing
        batch_size = max(100, len(test_range) // (self.n_workers * 2))  # Ensure enough tasks
        batches = [test_range[i:i + batch_size] for i in range(0, len(test_range), batch_size)]
        
        logger.info(f"Processing Gregory-Hansen test in {len(batches)} batches")
        
        # Initialize UnitRootTester outside loop to avoid repeated initialization
        ur_tester = UnitRootTester()
        
        # Process batches in parallel
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit batch tasks
            futures = []
            for i, batch in enumerate(batches):
                futures.append(executor.submit(
                    self._process_gh_batch,
                    y_array=y_array,
                    x_array=x_array,
                    model=model,
                    breakpoints=batch,
                    batch_idx=i,
                    ur_tester=ur_tester
                ))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    if batch_results:
                        all_results.extend(batch_results)
                except Exception as e:
                    logger.warning(f"Error in Gregory-Hansen batch: {e}")
        
        # Find minimum ADF statistic
        min_result = min(all_results, key=lambda x: x['adf_stat']) if all_results else None
        
        if min_result is None:
            logger.warning("No valid Gregory-Hansen results found")
            return {
                'adf_stat': None,
                'breakpoint': None,
                'cointegrated': False,
                'model': model,
                'all_tests': []
            }
        
        min_adf = min_result['adf_stat']
        breakpoint = min_result['breakpoint']
        
        # Format final result
        critical_values = {
            'cc': {'1%': -5.13, '5%': -4.61, '10%': -4.34},
            'ct': {'1%': -5.45, '5%': -4.99, '10%': -4.72},
            'ctt': {'1%': -5.47, '5%': -4.95, '10%': -4.68}
        }.get(model, {'1%': -5.13, '5%': -4.61, '10%': -4.34})
        
        # Determine if cointegrated (stationary residuals)
        cointegrated = min_adf < critical_values['5%']
        
        gh_result = {
            'adf_stat': min_adf,
            'breakpoint': breakpoint,
            'critical_values': critical_values,
            'cointegrated': cointegrated,
            'model': model,
            'all_tests': all_results
        }
        
        # Add breakpoint_date if datetime index is available
        if original_index is not None and breakpoint is not None:
            try:
                gh_result['breakpoint_date'] = original_index[breakpoint]
            except (IndexError, TypeError) as e:
                logger.warning(f"Could not determine breakpoint date: {e}")
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(
            f"Gregory-Hansen test (parallel): ADF stat={gh_result['adf_stat']:.4f}, "
            f"breakpoint={gh_result['breakpoint']}, cointegrated={gh_result['cointegrated']}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection
        gc.collect()
        
        return gh_result
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def _process_gh_batch(
        self,
        y_array: np.ndarray,
        x_array: np.ndarray,
        model: str,
        breakpoints: List[int],
        batch_idx: int,
        ur_tester: Optional[UnitRootTester] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of Gregory-Hansen tests.
        
        Parameters
        ----------
        y_array : np.ndarray
            Dependent variable array
        x_array : np.ndarray
            Independent variable(s) array
        model : str
            Model type
        breakpoints : List[int]
            Breakpoints to test in this batch
        batch_idx : int
            Batch index for logging
        ur_tester : UnitRootTester, optional
            Unit root tester instance
            
        Returns
        -------
        List[Dict[str, Any]]
            Batch test results
        """
        # Create UnitRootTester if not provided
        if ur_tester is None:
            ur_tester = UnitRootTester()
        
        n = len(y_array)
        batch_results = []
        
        # Test each possible breakpoint in this batch
        for i, breakpoint in enumerate(breakpoints):
            if i % 50 == 0:
                logger.debug(f"Processing breakpoint {i}/{len(breakpoints)} in batch {batch_idx}")
            
            # Create dummy variable for break
            dummy = np.zeros(n)
            dummy[breakpoint:] = 1
            
            # Create regressor matrix based on model type
            try:
                if model == 'cc':
                    # Level shift only
                    X = np.column_stack((np.ones(n), dummy, x_array))
                elif model == 'ct':
                    # Level shift with trend
                    trend = np.arange(n)
                    X = np.column_stack((np.ones(n), dummy, trend, x_array))
                elif model == 'ctt':
                    # Regime shift (intercept and slope)
                    X_with_dummy = x_array * dummy.reshape(-1, 1)
                    X = np.column_stack((np.ones(n), dummy, x_array, X_with_dummy))
                else:
                    raise ValueError(f"Unknown model type: {model}")
                
                # OLS regression
                from statsmodels.regression.linear_model import OLS
                model_fit = OLS(y_array, X).fit()
                residuals = model_fit.resid
                
                # Test for cointegration (stationarity of residuals)
                adf_result = ur_tester.test_adf(residuals, regression='nc')
                
                # Store results
                batch_results.append({
                    'breakpoint': breakpoint,
                    'adf_stat': adf_result['statistic'],
                    'pvalue': adf_result['pvalue']
                })
                
            except Exception as e:
                logger.debug(f"Error testing breakpoint {breakpoint} in batch {batch_idx}: {e}")
                continue
        
        logger.debug(f"Completed batch {batch_idx}: processed {len(batch_results)}/{len(breakpoints)} breakpoints")
        return batch_results