"""
Unit root testing module for time series analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from statsmodels.tsa.stattools import adfuller, kpss
import arch.unitroot as unitroot
import ruptures as rpt

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
        
        logger.info(f"Phillips-Perron test: statistic={pp_result['statistic']:.4f}, p-value={pp_result['pvalue']:.4f}")
        return pp_result
    
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
            'phillips_perron': self.test_phillips_perron(series, lags=lags),
            'zivot_andrews': self.test_zivot_andrews(series)
        }
    
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


class StructuralBreakTester:
    """Detect structural breaks in time series data."""
    
    def __init__(self):
        """Initialize the structural break tester."""
        pass
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
            'method': method
        }
        
        # Add dates if available
        if breakpoint_dates:
            bp_result['breakpoint_dates'] = breakpoint_dates
        
        logger.info(f"Bai-Perron test: detected {bp_result['n_breakpoints']} breakpoints")
        return bp_result
    
    @memoize
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
        
        logger.info(
            f"Gregory-Hansen test: ADF stat={gh_result['adf_stat']:.4f}, "
            f"breakpoint={gh_result['breakpoint']}, cointegrated={gh_result['cointegrated']}"
        )
        return gh_result