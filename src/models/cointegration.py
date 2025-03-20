"""
Cointegration testing module for time series analysis.
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
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from scipy import stats

from utils import (
    # Error handling
    handle_errors, ModelError,
    
    # Validation
    validate_time_series, raise_if_invalid, validate_dataframe,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, disk_cache, parallelize_dataframe,
    configure_system_for_performance, optimize_dataframe,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get default configuration
DEFAULT_ALPHA = config.get('analysis.cointegration.alpha', 0.05)
DEFAULT_TREND = config.get('analysis.cointegration.trend', 'c')
DEFAULT_MAX_LAGS = config.get('analysis.cointegration.max_lags', 4)

# Configure system for optimal performance
configure_system_for_performance()

class CointegrationTester:
    """Perform cointegration tests on time series data."""
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def __init__(self):
        """Initialize the cointegration tester."""
        # Get number of available workers based on CPU count
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"Initialized CointegrationTester. Memory usage: {memory_usage:.2f} MB")
    
    @disk_cache(cache_dir='.cache/cointegration')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
    def test_engle_granger(
        self, 
        y: Union[pd.Series, np.ndarray], 
        x: Union[pd.Series, np.ndarray], 
        trend: str = DEFAULT_TREND, 
        maxlag: Optional[int] = None, 
        autolag: str = 'AIC'
    ) -> Dict[str, Any]:
        """
        Perform Engle-Granger two-step cointegration test.
        
        Parameters
        ----------
        y : array_like
            First time series
        x : array_like
            Second time series
        trend : str, optional
            'c' : constant (default)
            'ct' : constant and trend
            'ctt' : constant, linear and quadratic trend
            'nc' : no constant, no trend
        maxlag : int, optional
            Maximum lag to be used
        autolag : str, optional
            Method to use for automatic lag selection
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate inputs with custom validators
        def validate_equal_length(series1, series2):
            """Check that both series have the same length."""
            len1 = len(series1.values if isinstance(series1, pd.Series) else series1)
            len2 = len(series2.values if isinstance(series2, pd.Series) else series2)
            return len1 == len2
        
        def validate_no_missing(s):
            """Check for missing values."""
            array = s.values if isinstance(s, pd.Series) else s
            return not np.isnan(array).any()
        
        # Validate y
        valid1, errors1 = validate_time_series(
            y, 
            min_length=20,
            max_nulls=0,
            check_constant=True
        )
        if not valid1:
            raise_if_invalid(valid1, errors1, "Invalid y time series for Engle-Granger test")
            
        # Validate x
        valid2, errors2 = validate_time_series(
            x, 
            min_length=20,
            max_nulls=0,
            check_constant=True
        )
        if not valid2:
            raise_if_invalid(valid2, errors2, "Invalid x time series for Engle-Granger test")
            
        # Additional validation for both series together
        if not validate_equal_length(y, x):
            raise ValueError(f"y and x must have the same length, got {len(y)} and {len(x)}")
        
        # Convert to numpy arrays if pandas Series
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.Series):
            x = x.values
        
        # Check lengths
        if len(y) != len(x):
            raise ValueError(f"y and x must have the same length, got {len(y)} and {len(x)}")
        
        # Run Engle-Granger test
        result = coint(y, x, trend=trend, maxlag=maxlag, autolag=autolag)
        
        # Estimate cointegrating relationship
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        
        # Format result
        eg_result = {
            'statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[2],
            'cointegrated': result[1] < DEFAULT_ALPHA,
            'beta': model.params,  # Cointegrating vector
            'residuals': model.resid,  # Equilibrium errors
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(
            f"Engle-Granger test: statistic={eg_result['statistic']:.4f}, "
            f"p-value={eg_result['pvalue']:.4f}, cointegrated={eg_result['cointegrated']}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection to free memory
        gc.collect()
        
        return eg_result
    
    @disk_cache(cache_dir='.cache/cointegration')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @m1_optimized(parallel=True)
    @timer
    def test_johansen(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        det_order: int = 1, 
        k_ar_diff: int = DEFAULT_MAX_LAGS
    ) -> Dict[str, Any]:
        """
        Perform Johansen cointegration test.
        
        Parameters
        ----------
        data : array_like
            Matrix of time series
        det_order : int, optional
            0 : no deterministic terms
            1 : constant term
            2 : constant and trend
        k_ar_diff : int, optional
            Number of lagged differences in the VAR model
            
        Returns
        -------
        dict
            Dictionary with test results including:
            - trace_statistics: Trace statistics for each cointegration rank
            - trace_critical_values: Critical values for trace statistics
            - max_statistics: Maximum eigenvalue statistics
            - max_critical_values: Critical values for maximum eigenvalue statistics
            - p_values_trace: P-values calculated using chi-squared approximation (approximate)
            - rank_trace: Cointegration rank based on trace statistic
            - rank_max: Cointegration rank based on maximum eigenvalue statistic
            - cointegration_vectors: Estimated cointegration vectors
            - eigenvalues: Eigenvalues from decomposition
            - cointegrated: Boolean indicating if cointegration is present
            
        Notes
        -----
        The p-values are calculated using a chi-squared approximation to the 
        asymptotic distribution of the test statistics. These are approximate 
        p-values and should be interpreted with caution, especially in small samples.
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Ensure data is a DataFrame or numpy array
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError(f"data must be a DataFrame or numpy array, got {type(data)}")
        
        # Convert to numpy array if DataFrame
        if isinstance(data, pd.DataFrame):
            # Optimize memory usage for large DataFrames
            if data.shape[0] > 1000 or data.shape[1] > 20:
                data = optimize_dataframe(data)
            data_values = data.values
        else:
            data_values = data
        
        # Check matrix dimensions
        if data_values.ndim != 2:
            raise ValueError(f"data must be a 2D array, got {data_values.ndim}D")
        
        n_obs, n_vars = data_values.shape
        
        # Validate
        if n_obs < 20:
            raise ValueError(f"Too few observations for reliable Johansen test: {n_obs}")
        
        if n_vars < 2:
            raise ValueError(f"Need at least 2 variables for cointegration test, got {n_vars}")
        
        # For large datasets, process in chunks to reduce memory usage
        chunk_size = 5000
        if n_obs > chunk_size:
            # Split data into chunks for testing
            n_chunks = (n_obs + chunk_size - 1) // chunk_size
            chunks = [data_values[i * chunk_size:min((i + 1) * chunk_size, n_obs)] for i in range(n_chunks)]
            
            logger.info(f"Large dataset detected. Processing Johansen test in {n_chunks} chunks.")
            
            # Process each chunk in parallel
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit test tasks
                futures = []
                for i, chunk in enumerate(chunks):
                    if len(chunk) >= 20:  # Minimum required for Johansen test
                        futures.append(executor.submit(
                            self._process_johansen_chunk,
                            chunk, det_order, k_ar_diff, i
                        ))
                
                # Collect results
                chunk_results = []
                for future in as_completed(futures):
                    try:
                        chunk_result = future.result()
                        if chunk_result is not None:
                            chunk_results.append(chunk_result)
                    except Exception as e:
                        logger.warning(f"Error processing Johansen test chunk: {e}")
                
                # Combine results
                combined_result = self._combine_johansen_results(chunk_results, n_vars)
                
                # Track memory after processing
                end_mem = process.memory_info().rss / (1024 * 1024)  # MB
                memory_diff = end_mem - start_mem
                
                logger.info(
                    f"Johansen test complete (chunked): cointegration rank (trace)={combined_result['rank_trace']}, "
                    f"rank (max)={combined_result['rank_max']}, cointegrated={combined_result['cointegrated']}. "
                    f"Memory usage: {memory_diff:.2f} MB"
                )
                
                # Force garbage collection
                gc.collect()
                
                return combined_result
        
        # For smaller datasets or testing individual chunks
        # Run Johansen test
        result = coint_johansen(data_values, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Extract trace and max eigenvalue statistics
        trace_stat = result.lr1
        trace_crit = result.cvt
        max_stat = result.lr2
        max_crit = result.cvm
        
        # Calculate p-values using chi-squared approximation
        # Degrees of freedom for trace test is (n_vars - r) for each r
        df_trace = np.array([(n_vars - r) for r in range(n_vars)])
        p_values_trace = [1 - stats.chi2.cdf(trace_stat[i], df_trace[i]) for i in range(len(trace_stat))]
        
        # Determine the cointegration rank using 5% significance level
        rank_trace = sum(trace_stat > trace_crit[:, 1])  # 5% significance level
        rank_max = sum(max_stat > max_crit[:, 1])  # 5% significance level
        
        johansen_result = {
            'trace_statistics': trace_stat,
            'trace_critical_values': trace_crit,
            'max_statistics': max_stat,
            'max_critical_values': max_crit,
            'p_values_trace': np.array(p_values_trace),
            'rank_trace': rank_trace,
            'rank_max': rank_max,
            'cointegration_vectors': result.evec,
            'eigenvalues': result.eig,
            'cointegrated': rank_trace > 0
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(
            f"Johansen test complete: cointegration rank (trace)={rank_trace}, "
            f"rank (max)={rank_max}, cointegrated={johansen_result['cointegrated']}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection
        gc.collect()
        
        return johansen_result
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def _process_johansen_chunk(
        self, 
        chunk: np.ndarray,
        det_order: int,
        k_ar_diff: int,
        chunk_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a chunk of data for Johansen test.
        
        Parameters
        ----------
        chunk : np.ndarray
            Data chunk to process
        det_order : int
            Deterministic term specification
        k_ar_diff : int
            Number of lagged differences
        chunk_idx : int
            Chunk index for logging
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Johansen test results for this chunk or None if error
        """
        try:
            # Run Johansen test on chunk
            result = coint_johansen(chunk, det_order=det_order, k_ar_diff=k_ar_diff)
            
            # Extract statistics
            n_vars = chunk.shape[1]
            trace_stat = result.lr1
            trace_crit = result.cvt
            max_stat = result.lr2
            max_crit = result.cvm
            
            # Calculate p-values
            df_trace = np.array([(n_vars - r) for r in range(n_vars)])
            p_values_trace = [1 - stats.chi2.cdf(trace_stat[i], df_trace[i]) for i in range(len(trace_stat))]
            
            # Determine ranks
            rank_trace = sum(trace_stat > trace_crit[:, 1])
            rank_max = sum(max_stat > max_crit[:, 1])
            
            chunk_result = {
                'trace_statistics': trace_stat,
                'trace_critical_values': trace_crit,
                'max_statistics': max_stat,
                'max_critical_values': max_crit,
                'p_values_trace': np.array(p_values_trace),
                'rank_trace': rank_trace,
                'rank_max': rank_max,
                'cointegration_vectors': result.evec,
                'eigenvalues': result.eig,
                'cointegrated': rank_trace > 0,
                'chunk_size': len(chunk)
            }
            
            logger.debug(f"Processed Johansen test chunk {chunk_idx}: rank_trace={rank_trace}, rank_max={rank_max}")
            return chunk_result
            
        except Exception as e:
            logger.warning(f"Error in Johansen test for chunk {chunk_idx}: {e}")
            return None
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def _combine_johansen_results(
        self, 
        chunk_results: List[Dict[str, Any]],
        n_vars: int
    ) -> Dict[str, Any]:
        """
        Combine Johansen test results from multiple chunks.
        
        Parameters
        ----------
        chunk_results : List[Dict[str, Any]]
            Results from individual chunks
        n_vars : int
            Number of variables
            
        Returns
        -------
        Dict[str, Any]
            Combined test results
        """
        if not chunk_results:
            raise ValueError("No valid chunk results to combine")
        
        # Calculate weighted averages based on chunk sizes
        total_size = sum(result['chunk_size'] for result in chunk_results)
        
        # Combine trace statistics (weighted average)
        trace_stats = np.zeros(n_vars)
        for result in chunk_results:
            weight = result['chunk_size'] / total_size
            trace_stats += result['trace_statistics'] * weight
        
        # Use critical values from first chunk (they should be the same for all chunks)
        trace_crit = chunk_results[0]['trace_critical_values']
        
        # Combine max statistics (weighted average)
        max_stats = np.zeros(n_vars)
        for result in chunk_results:
            weight = result['chunk_size'] / total_size
            max_stats += result['max_statistics'] * weight
        
        max_crit = chunk_results[0]['max_critical_values']
        
        # Recalculate p-values
        df_trace = np.array([(n_vars - r) for r in range(n_vars)])
        p_values_trace = [1 - stats.chi2.cdf(trace_stats[i], df_trace[i]) for i in range(len(trace_stats))]
        
        # Recalculate ranks
        rank_trace = sum(trace_stats > trace_crit[:, 1])
        rank_max = sum(max_stats > max_crit[:, 1])
        
        # For cointegration vectors and eigenvalues, use the result from the largest chunk
        largest_chunk_idx = max(range(len(chunk_results)), key=lambda i: chunk_results[i]['chunk_size'])
        
        combined_result = {
            'trace_statistics': trace_stats,
            'trace_critical_values': trace_crit,
            'max_statistics': max_stats,
            'max_critical_values': max_crit,
            'p_values_trace': np.array(p_values_trace),
            'rank_trace': rank_trace,
            'rank_max': rank_max,
            'cointegration_vectors': chunk_results[largest_chunk_idx]['cointegration_vectors'],
            'eigenvalues': chunk_results[largest_chunk_idx]['eigenvalues'],
            'cointegrated': rank_trace > 0,
            'method': 'chunked'
        }
        
        return combined_result
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_combined(
        self, 
        y: Union[pd.Series, np.ndarray], 
        x: Union[pd.Series, np.ndarray], 
        trend: str = DEFAULT_TREND
    ) -> Dict[str, Any]:
        """
        Perform both Engle-Granger and Johansen tests.
        
        Parameters
        ----------
        y : array_like
            First time series
        x : array_like
            Second time series
        trend : str, optional
            Trend specification
            
        Returns
        -------
        dict
            Dictionary with results of both tests and additional metrics:
            - engle_granger: Results from Engle-Granger test
            - johansen: Results from Johansen test
            - cointegrated: Boolean indicating if cointegration is present in either test
            - half_life: Time (in periods) for deviations to revert halfway to equilibrium
                         (only calculated if Engle-Granger shows cointegration)
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Convert inputs to numpy arrays
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        if isinstance(x, pd.Series):
            x_values = x.values
        else:
            x_values = x
        
        # Create matrix for Johansen test
        data = np.column_stack([y_values, x_values])
        
        # Map trend to det_order for Johansen
        det_order = 1  # default: constant
        if trend == 'ct':
            det_order = 2  # constant and trend
        elif trend == 'nc':
            det_order = 0  # no deterministic terms
        
        # Run tests in parallel for better performance
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit test tasks
            future_eg = executor.submit(self.test_engle_granger, y_values, x_values, trend=trend)
            future_jo = executor.submit(self.test_johansen, data, det_order=det_order)
            
            # Collect results
            eg_result = future_eg.result()
            jo_result = future_jo.result()
        
        # Calculate half-life if cointegrated according to Engle-Granger test
        half_life = None
        if eg_result['cointegrated']:
            try:
                half_life = calculate_half_life(eg_result['residuals'])
                logger.info(f"Half-life of deviations: {half_life:.2f} periods")
            except Exception as e:
                logger.warning(f"Could not calculate half-life: {e}")
        
        # Combine results
        combined = {
            'engle_granger': eg_result,
            'johansen': jo_result,
            'cointegrated': eg_result['cointegrated'] or jo_result['cointegrated'],
            'half_life': half_life
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Combined cointegration tests complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return combined

    @disk_cache(cache_dir='.cache/cointegration')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @m1_optimized(use_numba=True)
    @timer
    def test_gregory_hansen(
        self, 
        y: Union[pd.Series, np.ndarray], 
        x: Union[pd.Series, np.ndarray], 
        trend: str = DEFAULT_TREND,
        model: str = "regime_shift",
        trim: float = 0.15
    ) -> Dict[str, Any]:
        """
        Perform Gregory-Hansen cointegration test with structural breaks.
        
        Parameters
        ----------
        y : array_like
            Dependent variable time series
        x : array_like
            Independent variable time series
        trend : str, optional
            Trend specification ('c', 'ct', 'ctt', 'nc')
        model : str, optional
            Structural break model:
            - 'level_shift': Change in intercept only (C)
            - 'trend_shift': Change in intercept and trend (C/T)
            - 'regime_shift': Change in intercept and slope (full break) (C/S)
        trim : float, optional
            Trimming percentage for structural break search
            
        Returns
        -------
        dict
            Dictionary with test results including:
            - test_statistics: ADF statistics for each potential break point
            - min_statistic: Minimum ADF statistic (most significant)
            - break_point: Index of the break point
            - break_date: Date of the break (if y or x is a Series with DatetimeIndex)
            - critical_values: Critical values for the test
            - p_value: Approximate p-value
            - cointegrated: Boolean indicating if series are cointegrated
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate inputs
        valid1, errors1 = validate_time_series(y, min_length=30)
        valid2, errors2 = validate_time_series(x, min_length=30)
        
        if not valid1:
            raise_if_invalid(valid1, errors1, "Invalid y time series for Gregory-Hansen test")
        if not valid2:
            raise_if_invalid(valid2, errors2, "Invalid x time series for Gregory-Hansen test")
        
        # Convert to numpy arrays if pandas Series
        date_index = None
        if isinstance(y, pd.Series):
            if isinstance(y.index, pd.DatetimeIndex):
                date_index = y.index
            y = y.values
        if isinstance(x, pd.Series):
            if isinstance(x.index, pd.DatetimeIndex) and date_index is None:
                date_index = x.index
            x = x.values
        
        # Check lengths
        if len(y) != len(x):
            raise ValueError(f"y and x must have the same length, got {len(y)} and {len(x)}")
        
        # Get sample size and calculate trim points
        n = len(y)
        start_idx = int(n * trim)
        end_idx = int(n * (1 - trim))
        
        # Create arrays to store results
        test_statistics = np.zeros(end_idx - start_idx)
        
        # Get critical values based on model
        # These are from Gregory and Hansen (1996) Table 1
        critical_values = {
            'level_shift': {'1%': -5.13, '5%': -4.61, '10%': -4.34},
            'trend_shift': {'1%': -5.45, '5%': -4.99, '10%': -4.72},
            'regime_shift': {'1%': -5.47, '5%': -4.95, '10%': -4.68}
        }
        
        if model not in critical_values:
            raise ValueError(f"Invalid model: {model}. Must be one of {list(critical_values.keys())}")
        
        # Loop over possible break points
        for i, break_idx in enumerate(range(start_idx, end_idx)):
            # Create dummy variables based on model
            if model == 'level_shift':
                # Level shift: change in intercept
                dummy = np.zeros(n)
                dummy[break_idx:] = 1
                X = sm.add_constant(np.column_stack([x, dummy]))
            
            elif model == 'trend_shift':
                # Trend shift: change in intercept and trend
                dummy = np.zeros(n)
                dummy[break_idx:] = 1
                time_trend = np.arange(n)
                X = sm.add_constant(np.column_stack([x, time_trend, dummy]))
            
            elif model == 'regime_shift':
                # Regime shift: change in intercept and slope
                dummy = np.zeros(n)
                dummy[break_idx:] = 1
                X = sm.add_constant(np.column_stack([x, dummy, x * dummy]))
            
            # Run regression
            model_fit = sm.OLS(y, X).fit()
            residuals = model_fit.resid
            
            # Perform ADF test on residuals
            adf_result = sm.tsa.stattools.adfuller(residuals, regression=trend)
            test_statistics[i] = adf_result[0]
        
        # Find break point with minimum test statistic
        min_idx = np.argmin(test_statistics)
        min_statistic = test_statistics[min_idx]
        break_point = start_idx + min_idx
        
        # Get break date if date index is available
        break_date = None
        if date_index is not None:
            break_date = date_index[break_point]
        
        # Determine if cointegrated based on critical value (5% level)
        cv_5pct = critical_values[model]['5%']
        cointegrated = min_statistic < cv_5pct
        
        # Approximate p-value (this is approximate since exact p-values require simulation)
        # We'll interpolate between critical values
        crit_values_array = np.array([
            critical_values[model]['10%'],
            critical_values[model]['5%'],
            critical_values[model]['1%']
        ])
        p_levels = np.array([0.1, 0.05, 0.01])
        
        if min_statistic <= crit_values_array[-1]:
            p_value = 0.01  # Significant at 1% level
        elif min_statistic >= crit_values_array[0]:
            p_value = 0.1  # Not significant at 10% level
        else:
            # Interpolate
            p_value = np.interp(min_statistic, crit_values_array[::-1], p_levels[::-1])
        
        # Create result dictionary
        result = {
            'test_statistics': test_statistics,
            'min_statistic': min_statistic,
            'break_point': break_point,
            'break_fraction': break_point / n,
            'break_date': break_date,
            'critical_values': critical_values[model],
            'p_value': p_value,
            'cointegrated': cointegrated,
            'model': model
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(
            f"Gregory-Hansen test: statistic={min_statistic:.4f}, "
            f"break point={break_point}, cointegrated={cointegrated}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection
        gc.collect()
        
        return result

    @disk_cache(cache_dir='.cache/cointegration')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @m1_optimized(parallel=True)
    @timer
    def test_threshold_cointegration(
        self, 
        y: Union[pd.Series, np.ndarray], 
        x: Union[pd.Series, np.ndarray], 
        lag: int = 1,
        trim: float = 0.15,
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform Hansen-Seo threshold cointegration test.
        
        Parameters
        ----------
        y : array_like
            Dependent variable time series
        x : array_like
            Independent variable time series
        lag : int, optional
            Lag order for the VECM
        trim : float, optional
            Trimming percentage for threshold search
        n_bootstrap : int, optional
            Number of bootstrap replications for p-value
            
        Returns
        -------
        dict
            Dictionary with test results including:
            - test_statistic: SupLM test statistic
            - threshold: Estimated threshold value
            - threshold_effect: Boolean indicating if threshold effect is significant
            - p_value: p-value from bootstrap
            - beta: Cointegrating vector
            - model_statistics: Statistics for the threshold model
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate inputs
        valid1, errors1 = validate_time_series(y, min_length=30)
        valid2, errors2 = validate_time_series(x, min_length=30)
        
        if not valid1:
            raise_if_invalid(valid1, errors1, "Invalid y time series for threshold cointegration test")
        if not valid2:
            raise_if_invalid(valid2, errors2, "Invalid x time series for threshold cointegration test")
        
        # Convert to numpy arrays if pandas Series
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.Series):
            x = x.values
        
        # Step 1: Estimate cointegrating vector using linear model
        beta, residuals = estimate_cointegration_vector(y, x)
        
        # Step 2: Create error correction term (ECT)
        ect = residuals
        
        # Step 3: Set up VECM for linear model
        # Create lagged differences
        y_diff = np.diff(y)
        x_diff = np.diff(x)
        
        # Lag the ECT
        ect_lagged = ect[:-1]
        
        # Set up data for linear VECM
        X_linear = sm.add_constant(np.column_stack([ect_lagged, y_diff[:-1], x_diff[:-1]]))
        y_eq = y_diff[1:]
        x_eq = x_diff[1:]
        
        # Estimate linear VECM
        model_y = sm.OLS(y_eq, X_linear).fit()
        model_x = sm.OLS(x_eq, X_linear).fit()
        
        # Get sum of squared residuals for linear model
        ssr_linear = np.sum(model_y.resid**2) + np.sum(model_x.resid**2)
        
        # Step 4: Grid search for optimal threshold
        ect_sorted = np.sort(ect_lagged)
        min_idx = int(len(ect_sorted) * trim)
        max_idx = int(len(ect_sorted) * (1 - trim))
        threshold_grid = ect_sorted[min_idx:max_idx]
        
        # Initialize variables for grid search
        min_ssr = float('inf')
        best_threshold = None
        
        for threshold in threshold_grid:
            # Split data based on threshold
            below_mask = ect_lagged <= threshold
            above_mask = ect_lagged > threshold
            
            # Ensure minimum observations in each regime
            if np.sum(below_mask) < 5 or np.sum(above_mask) < 5:
                continue
            
            # Estimate threshold VECM
            X_below = sm.add_constant(np.column_stack([ect_lagged[below_mask], 
                                                      y_diff[:-1][below_mask], 
                                                      x_diff[:-1][below_mask]]))
            X_above = sm.add_constant(np.column_stack([ect_lagged[above_mask], 
                                                      y_diff[:-1][above_mask], 
                                                      x_diff[:-1][above_mask]]))
            
            # Estimate equations for each regime
            try:
                model_y_below = sm.OLS(y_eq[below_mask], X_below).fit()
                model_y_above = sm.OLS(y_eq[above_mask], X_above).fit()
                model_x_below = sm.OLS(x_eq[below_mask], X_below).fit()
                model_x_above = sm.OLS(x_eq[above_mask], X_above).fit()
                
                # Calculate SSR for threshold model
                ssr_threshold = (np.sum(model_y_below.resid**2) + np.sum(model_y_above.resid**2) + 
                                np.sum(model_x_below.resid**2) + np.sum(model_x_above.resid**2))
                
                if ssr_threshold < min_ssr:
                    min_ssr = ssr_threshold
                    best_threshold = threshold
            except:
                continue
        
        # Step 5: Calculate test statistic
        if best_threshold is None:
            raise ValueError("Could not find valid threshold")
        
        # LM-like test statistic
        n = len(y_diff)
        test_statistic = n * (ssr_linear - min_ssr) / ssr_linear
        
        # Step 6: Bootstrap p-value
        # This is computationally intensive, so we'll use parallel processing
        bootstrap_stats = []
        
        # Extract fixed cointegrating vector
        beta_fixed = beta[1]  # Coefficient on x
        
        # Define bootstrap worker
        def bootstrap_worker(seed):
            # Set seed for reproducibility
            np.random.seed(seed)
            
            # Resample residuals
            indices = np.random.randint(0, len(model_y.resid), len(model_y.resid))
            boot_resid_y = model_y.resid[indices]
            boot_resid_x = model_x.resid[indices]
            
            # Simulate data under null (linear VECM)
            boot_y_diff = X_linear @ model_y.params + boot_resid_y
            boot_x_diff = X_linear @ model_x.params + boot_resid_x
            
            # Reconstruct levels by cumulative sum
            boot_y = np.cumsum(np.concatenate([[y[0]], boot_y_diff]))
            boot_x = np.cumsum(np.concatenate([[x[0]], boot_x_diff]))
            
            # Calculate bootstrap ECT (with fixed cointegrating vector)
            boot_ect = boot_y - beta[0] - beta_fixed * boot_x
            boot_ect_lagged = boot_ect[:-1]
            
            # Set up data for bootstrap linear VECM
            boot_X_linear = sm.add_constant(np.column_stack([boot_ect_lagged, 
                                                            np.diff(boot_y)[:-1], 
                                                            np.diff(boot_x)[:-1]]))
            boot_y_eq = np.diff(boot_y)[1:]
            boot_x_eq = np.diff(boot_x)[1:]
            
            # Estimate bootstrap linear VECM
            try:
                boot_model_y = sm.OLS(boot_y_eq, boot_X_linear).fit()
                boot_model_x = sm.OLS(boot_x_eq, boot_X_linear).fit()
                
                # Get SSR for bootstrap linear model
                boot_ssr_linear = np.sum(boot_model_y.resid**2) + np.sum(boot_model_x.resid**2)
                
                # Grid search for optimal threshold in bootstrap
                boot_min_ssr = float('inf')
                
                for threshold in threshold_grid:
                    # Split data based on threshold
                    below_mask = boot_ect_lagged <= threshold
                    above_mask = boot_ect_lagged > threshold
                    
                    # Ensure minimum observations in each regime
                    if np.sum(below_mask) < 5 or np.sum(above_mask) < 5:
                        continue
                    
                    # Estimate threshold VECM for bootstrap
                    boot_X_below = sm.add_constant(np.column_stack([boot_ect_lagged[below_mask], 
                                                                   np.diff(boot_y)[:-1][below_mask], 
                                                                   np.diff(boot_x)[:-1][below_mask]]))
                    boot_X_above = sm.add_constant(np.column_stack([boot_ect_lagged[above_mask], 
                                                                   np.diff(boot_y)[:-1][above_mask], 
                                                                   np.diff(boot_x)[:-1][above_mask]]))
                    
                    # Estimate equations for each regime
                    try:
                        boot_model_y_below = sm.OLS(boot_y_eq[below_mask], boot_X_below).fit()
                        boot_model_y_above = sm.OLS(boot_y_eq[above_mask], boot_X_above).fit()
                        boot_model_x_below = sm.OLS(boot_x_eq[below_mask], boot_X_below).fit()
                        boot_model_x_above = sm.OLS(boot_x_eq[above_mask], boot_X_above).fit()
                        
                        # Calculate SSR for bootstrap threshold model
                        boot_ssr_threshold = (np.sum(boot_model_y_below.resid**2) + 
                                             np.sum(boot_model_y_above.resid**2) + 
                                             np.sum(boot_model_x_below.resid**2) + 
                                             np.sum(boot_model_x_above.resid**2))
                        
                        if boot_ssr_threshold < boot_min_ssr:
                            boot_min_ssr = boot_ssr_threshold
                    except:
                        continue
                
                # Calculate bootstrap test statistic
                boot_stat = n * (boot_ssr_linear - boot_min_ssr) / boot_ssr_linear
                return boot_stat
            except:
                return None
        
        # Use parallel processing for bootstrap
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Create a range of seeds for reproducibility
            seeds = np.random.randint(0, 10000, n_bootstrap)
            futures = [executor.submit(bootstrap_worker, seed) for seed in seeds]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        bootstrap_stats.append(result)
                except Exception as e:
                    logger.warning(f"Error in bootstrap: {e}")
        
        # Convert to numpy array and remove None values
        bootstrap_stats = np.array([stat for stat in bootstrap_stats if stat is not None])
        
        # Calculate p-value
        p_value = np.mean(bootstrap_stats > test_statistic)
        
        # Get adjustment coefficients for threshold model
        below_mask = ect_lagged <= best_threshold
        above_mask = ect_lagged > best_threshold
        
        X_below = sm.add_constant(np.column_stack([ect_lagged[below_mask], 
                                                  y_diff[:-1][below_mask], 
                                                  x_diff[:-1][below_mask]]))
        X_above = sm.add_constant(np.column_stack([ect_lagged[above_mask], 
                                                  y_diff[:-1][above_mask], 
                                                  x_diff[:-1][above_mask]]))
        
        model_y_below = sm.OLS(y_eq[below_mask], X_below).fit()
        model_y_above = sm.OLS(y_eq[above_mask], X_above).fit()
        model_x_below = sm.OLS(x_eq[below_mask], X_below).fit()
        model_x_above = sm.OLS(x_eq[above_mask], X_above).fit()
        
        # Create result dictionary
        result = {
            'test_statistic': test_statistic,
            'p_value': p_value,
            'threshold': best_threshold,
            'threshold_effect': p_value < DEFAULT_ALPHA,
            'beta': beta,
            'coint_test': test_statistic > 0,  # Simple test for cointegration
            'adjustment_coefficients': {
                'below': {
                    'y_equation': model_y_below.params[1],  # Coefficient on ECT
                    'x_equation': model_x_below.params[1]   # Coefficient on ECT
                },
                'above': {
                    'y_equation': model_y_above.params[1],  # Coefficient on ECT
                    'x_equation': model_x_above.params[1]   # Coefficient on ECT
                }
            },
            'model_stats': {
                'ssr_linear': ssr_linear,
                'ssr_threshold': min_ssr,
                'n_below': np.sum(below_mask),
                'n_above': np.sum(above_mask)
            },
            'bootstrap_results': {
                'n_bootstrap': len(bootstrap_stats),
                'bootstrap_mean': np.mean(bootstrap_stats),
                'bootstrap_std': np.std(bootstrap_stats)
            }
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(
            f"Threshold cointegration test: statistic={test_statistic:.4f}, "
            f"threshold={best_threshold:.4f}, p-value={p_value:.4f}, "
            f"threshold effect={result['threshold_effect']}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection
        gc.collect()
        
        return result

    @disk_cache(cache_dir='.cache/cointegration')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @timer
    def calculate_asymmetric_adjustment(
        self, 
        y: Union[pd.Series, np.ndarray], 
        x: Union[pd.Series, np.ndarray], 
        lag: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate asymmetric price adjustment speeds.
        
        Parameters
        ----------
        y : array_like
            Dependent variable time series
        x : array_like
            Independent variable time series
        lag : int, optional
            Lag order for the error correction model
            
        Returns
        -------
        dict
            Dictionary with results including:
            - cointegration_result: Results of cointegration test
            - adjustment_positive: Adjustment coefficient for positive deviations
            - adjustment_negative: Adjustment coefficient for negative deviations
            - asymmetry: Measure of asymmetry (difference in adjustment speeds)
            - asymmetry_significant: Boolean indicating if asymmetry is statistically significant
            - model_results: Full model statistics
            - half_life_positive: Half-life for positive deviations
            - half_life_negative: Half-life for negative deviations
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate inputs
        valid1, errors1 = validate_time_series(y, min_length=30)
        valid2, errors2 = validate_time_series(x, min_length=30)
        
        if not valid1:
            raise_if_invalid(valid1, errors1, "Invalid y time series for asymmetric adjustment calculation")
        if not valid2:
            raise_if_invalid(valid2, errors2, "Invalid x time series for asymmetric adjustment calculation")
        
        # Convert to numpy arrays if pandas Series
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.Series):
            x = x.values
        
        # First test for cointegration
        coint_result = self.test_engle_granger(y, x)
        
        if not coint_result['cointegrated']:
            logger.warning("Series are not cointegrated, asymmetric adjustment may not be meaningful")
        
        # Get cointegration residuals (deviations from equilibrium)
        residuals = coint_result['residuals']
        
        # Create lagged residuals
        residuals_lagged = residuals[:-1]
        
        # Split residuals into positive and negative components
        residuals_pos = np.maximum(residuals_lagged, 0)
        residuals_neg = np.minimum(residuals_lagged, 0)
        
        # Create first difference of y
        y_diff = np.diff(y)
        
        # Create lagged differences for y and x
        y_diff_lags = []
        x_diff_lags = []
        
        for i in range(1, lag + 1):
            if i < len(y_diff):
                y_diff_lags.append(y_diff[:-i])
                x_diff_lags.append(np.diff(x)[:-i])
        
# Combine lagged differences into predictor matrix
        X_lags = np.column_stack([arr for arr in y_diff_lags + x_diff_lags])
        
        # Add positive and negative residuals to predictor matrix
        if len(X_lags) > 0:
            X = sm.add_constant(np.column_stack([residuals_pos, residuals_neg, X_lags]))
        else:
            X = sm.add_constant(np.column_stack([residuals_pos, residuals_neg]))
        
        # Dependent variable (current difference of y)
        y_eq = y_diff[lag:]
        
        # Ensure X and y_eq have the same length
        if len(X) > len(y_eq):
            X = X[-len(y_eq):]
        elif len(X) < len(y_eq):
            y_eq = y_eq[-len(X):]
        
        # Fit the asymmetric adjustment model
        model = sm.OLS(y_eq, X).fit()
        
        # Extract adjustment coefficients
        adj_pos = model.params[1]  # Coefficient on positive residuals
        adj_neg = model.params[2]  # Coefficient on negative residuals
        
        # Calculate standard errors and p-values
        se_pos = model.bse[1]
        se_neg = model.bse[2]
        p_pos = model.pvalues[1]
        p_neg = model.pvalues[2]
        
        # Test for asymmetry (null hypothesis: adj_pos = adj_neg)
        asymmetry = adj_pos - adj_neg
        # Calculate standard error of the difference
        se_diff = np.sqrt(se_pos**2 + se_neg**2)
        t_stat = asymmetry / se_diff
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), model.df_resid))
        
        # Calculate half-lives
        half_life_pos = None
        half_life_neg = None
        
        if adj_pos < 0:  # Adjustment coefficient should be negative for convergence
            half_life_pos = np.log(0.5) / np.log(1 + adj_pos)
        
        if adj_neg < 0:  # Adjustment coefficient should be negative for convergence
            half_life_neg = np.log(0.5) / np.log(1 + adj_neg)
        
        # Create result dictionary
        result = {
            'cointegration_result': coint_result,
            'adjustment_positive': adj_pos,
            'adjustment_negative': adj_neg,
            'se_positive': se_pos,
            'se_negative': se_neg,
            'p_value_positive': p_pos,
            'p_value_negative': p_neg,
            'asymmetry': asymmetry,
            'asymmetry_t_statistic': t_stat,
            'asymmetry_p_value': p_value,
            'asymmetry_significant': p_value < DEFAULT_ALPHA,
            'half_life_positive': half_life_pos,
            'half_life_negative': half_life_neg,
            'model_results': {
                'params': model.params,
                'rsquared': model.rsquared,
                'rsquared_adj': model.rsquared_adj,
                'aic': model.aic,
                'bic': model.bic,
                'fvalue': model.fvalue,
                'f_pvalue': model.f_pvalue
            }
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        # Log main results
        logger.info(
            f"Asymmetric adjustment: positive={adj_pos:.4f}, negative={adj_neg:.4f}, "
            f"asymmetry={asymmetry:.4f}, p-value={p_value:.4f}, significant={result['asymmetry_significant']}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection
        gc.collect()
        
        return result

    @disk_cache(cache_dir='.cache/cointegration')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    @m1_optimized(use_numba=True)
    @timer
    def test_for_nonlinearity(
        self, 
        y: Union[pd.Series, np.ndarray], 
        x: Union[pd.Series, np.ndarray], 
        test_type: str = 'both',
        lag: int = 1,
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Test if the price adjustment process is nonlinear.
        
        Parameters
        ----------
        y : array_like
            Dependent variable time series
        x : array_like
            Independent variable time series
        test_type : str, optional
            Type of nonlinearity test ('asymmetry', 'threshold', 'both')
        lag : int, optional
            Lag order for the error correction model
        n_bootstrap : int, optional
            Number of bootstrap replications for threshold test
            
        Returns
        -------
        dict
            Dictionary with test results including:
            - asymmetry_test: Results of asymmetry test
            - threshold_test: Results of threshold test
            - nonlinearity: Boolean indicating if any nonlinearity is detected
            - recommended_model: Recommended model type based on test results
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate inputs
        valid1, errors1 = validate_time_series(y, min_length=30)
        valid2, errors2 = validate_time_series(x, min_length=30)
        
        if not valid1:
            raise_if_invalid(valid1, errors1, "Invalid y time series for nonlinearity test")
        if not valid2:
            raise_if_invalid(valid2, errors2, "Invalid x time series for nonlinearity test")
        
        # Convert to numpy arrays if pandas Series
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.Series):
            x = x.values
        
        # Initialize results
        results = {}
        
        # Run asymmetry test if requested
        if test_type in ['asymmetry', 'both']:
            asymmetry_result = self.calculate_asymmetric_adjustment(y, x, lag=lag)
            results['asymmetry_test'] = {
                'asymmetry': asymmetry_result['asymmetry'],
                'p_value': asymmetry_result['asymmetry_p_value'],
                'significant': asymmetry_result['asymmetry_significant']
            }
        
        # Run threshold test if requested
        if test_type in ['threshold', 'both']:
            threshold_result = self.test_threshold_cointegration(y, x, lag=lag, n_bootstrap=n_bootstrap)
            results['threshold_test'] = {
                'threshold': threshold_result['threshold'],
                'test_statistic': threshold_result['test_statistic'],
                'p_value': threshold_result['p_value'],
                'significant': threshold_result['threshold_effect']
            }
        
        # Determine if any nonlinearity is detected
        nonlinearity = False
        recommended_model = 'linear'
        
        if test_type in ['asymmetry', 'both'] and results['asymmetry_test']['significant']:
            nonlinearity = True
            recommended_model = 'asymmetric'
        
        if test_type in ['threshold', 'both'] and results['threshold_test']['significant']:
            nonlinearity = True
            # If both tests are significant, choose the one with the lowest p-value
            if test_type == 'both' and results['asymmetry_test']['significant']:
                if results['threshold_test']['p_value'] < results['asymmetry_test']['p_value']:
                    recommended_model = 'threshold'
            else:
                recommended_model = 'threshold'
        
        # Add summary results
        results['nonlinearity_detected'] = nonlinearity
        results['recommended_model'] = recommended_model
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        # Log main results
        logger.info(
            f"Nonlinearity test: nonlinearity detected={nonlinearity}, "
            f"recommended model={recommended_model}. "
            f"Memory usage: {memory_diff:.2f} MB"
        )
        
        # Force garbage collection
        gc.collect()
        
        return results


@m1_optimized(use_numba=True)
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
@timer
def estimate_cointegration_vector(
    y: Union[pd.Series, np.ndarray],
    x: Union[pd.Series, np.ndarray, List[Union[pd.Series, np.ndarray]]],
    method: str = 'ols',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate cointegration vector for cointegrated series.
    
    Parameters
    ----------
    y : array_like
        Dependent variable
    x : array_like or list of array_like
        Independent variable(s)
    method : str, optional
        Estimation method ('ols', 'dols', 'fmols')
    **kwargs : dict
        Additional parameters:
        - leads_lags : int, default=1
            Number of leads and lags for DOLS
        - kernel : str, default='bartlett'
            Kernel for long-run covariance estimation in FMOLS
            Options: 'bartlett', 'parzen', 'quadratic-spectral'
        - bandwidth : int, optional
            Bandwidth parameter for kernel in FMOLS
            If None, uses Andrews (1991) automatic bandwidth
        
    Returns
    -------
    tuple
        (beta, residuals)
    """
    # Track memory usage
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Handle x as a single series or a list of series
    if not isinstance(x, (list, tuple)):
        x_list = [x]
    else:
        x_list = x
    
    # Convert all series to numpy arrays
    if isinstance(y, pd.Series):
        y = y.values
        
    x_arrays = []
    for xi in x_list:
        if isinstance(xi, pd.Series):
            x_arrays.append(xi.values)
        else:
            x_arrays.append(xi)
    
    # Combine x arrays
    if len(x_arrays) == 1:
        X = x_arrays[0].reshape(-1, 1)
    else:
        X = np.column_stack(x_arrays)
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Estimate cointegration vector using requested method
    if method == 'ols':
        model = sm.OLS(y, X_with_const)
        results = model.fit()
        beta = results.params
        residuals = results.resid
    
    elif method == 'dols':
        # Extract DOLS parameters
        leads_lags = kwargs.get('leads_lags', 1)
        
        if leads_lags < 1:
            raise ValueError("leads_lags must be at least 1")
        
        # Get effective sample size after accounting for leads/lags
        n_obs = len(y)
        n_vars = X.shape[1]
        effective_obs = n_obs - 2 * leads_lags
        
        if effective_obs < n_vars + 5:
            raise ValueError(f"Too few observations after accounting for leads/lags: {effective_obs}")
        
        logger.debug(f"DOLS estimation with {leads_lags} leads and lags, {effective_obs} effective observations")
        
        try:
            # Calculate first differences of X variables
            dX = np.diff(X, axis=0)
            
            # Pad first observation since diff reduces length by 1
            dX = np.vstack([np.zeros((1, n_vars)), dX])
            
            # Create matrix to store all lead/lag terms
            lead_lag_terms = np.zeros((n_obs, 2 * leads_lags * n_vars))
            
            # Fill in lead and lag terms
            col_idx = 0
            for i in range(n_vars):
                # Add lags
                for lag in range(1, leads_lags + 1):
                    lead_lag_terms[lag:, col_idx] = dX[:-lag, i]
                    col_idx += 1
                
                # Add leads
                for lead in range(1, leads_lags + 1):
                    if lead < n_obs:
                        lead_lag_terms[:-lead, col_idx] = dX[lead:, i]
                    col_idx += 1
            
            # Create augmented matrix
            X_augmented = np.column_stack([X_with_const, lead_lag_terms])
            
            # Trim sample to account for leads/lags
            X_augmented_trimmed = X_augmented[leads_lags:-leads_lags]
            y_trimmed = y[leads_lags:-leads_lags]
            
            # Estimate DOLS model
            model = sm.OLS(y_trimmed, X_augmented_trimmed)
            results = model.fit()
            
            # Extract only the cointegrating vector (first n_vars+1 coefficients)
            beta = results.params[:n_vars+1]
            
            # Calculate residuals using full sample and only cointegrating vector
            residuals = y - X_with_const @ beta
            
            logger.info(f"DOLS estimation completed with {leads_lags} leads/lags")
            
        except Exception as e:
            logger.error(f"Error in DOLS estimation: {str(e)}")
            raise
    
    elif method == 'fmols':
        # Extract FMOLS parameters
        kernel = kwargs.get('kernel', 'bartlett')
        bandwidth = kwargs.get('bandwidth', None)
        
        valid_kernels = ['bartlett', 'parzen', 'quadratic-spectral']
        if kernel not in valid_kernels:
            raise ValueError(f"Invalid kernel: {kernel}. Must be one of {valid_kernels}")
        
        try:
            # Step 1: Initial OLS estimation
            ols_model = sm.OLS(y, X_with_const)
            ols_results = ols_model.fit()
            residuals = ols_results.resid
            
            # Step 2: Calculate differenced variables
            dy = np.diff(y)
            dX = np.diff(X, axis=0)
            
            # Step 3: Estimate long-run covariance matrices
            # Create combined matrix of residuals and differenced X
            u_dX = np.column_stack([residuals[1:], dX])
            
            # Estimate long-run covariance matrix
            n_obs = len(residuals) - 1
            
            # Automatic bandwidth selection if not provided
            if bandwidth is None:
                # Andrews (1991) automatic bandwidth selection
                # Using simplified formula for demonstration
                rho1 = np.array([np.corrcoef(u_dX[1:, i], u_dX[:-1, i])[0, 1] for i in range(u_dX.shape[1])])
                ar1_factor = 4 * (rho1 / (1 - rho1**2))**2
                bandwidth = int(0.75 * n_obs**(1/3) * np.mean(ar1_factor)**(1/3))
                
                logger.debug(f"Automatic bandwidth selection: {bandwidth}")
            
            # Compute long-run covariance matrix using specified kernel
            Omega = np.zeros((u_dX.shape[1], u_dX.shape[1]))
            
            # Auto-covariance at lag 0
            Gamma_0 = u_dX.T @ u_dX / n_obs
            Omega += Gamma_0
            
            # Add weighted auto-covariances at other lags
            for j in range(1, bandwidth + 1):
                # Calculate weight based on kernel
                if kernel == 'bartlett':
                    weight = 1 - j / (bandwidth + 1)
                elif kernel == 'parzen':
                    q = j / (bandwidth + 1)
                    if q <= 0.5:
                        weight = 1 - 6 * q**2 + 6 * q**3
                    else:
                        weight = 2 * (1 - q)**3
                else:  # quadratic-spectral
                    q = 6 * np.pi * j / (5 * (bandwidth + 1))
                    if q < 0.001:
                        weight = 1
                    else:
                        weight = 3 * (np.sin(q) / q - np.cos(q)) / q**2
                
                # Auto-covariance at lag j
                Gamma_j = u_dX[j:].T @ u_dX[:-j] / n_obs
                
                # Add to long-run covariance (both Gamma_j and its transpose)
                Omega += weight * (Gamma_j + Gamma_j.T)
            
            # Normalize
            Omega /= 2
            
            # Extract components
            Omega_ue = Omega[0, 1:]  # Covariance between residuals and differenced X
            Omega_ee = Omega[1:, 1:]  # Covariance among differenced X variables
            Omega_uu = Omega[0, 0]    # Variance of residuals
            
            # Calculate Delta_ue+ (one-sided long-run covariance)
            Delta_ue_plus = np.zeros(dX.shape[1])
            
            # Auto-covariance at lag 0
            Delta_0 = u_dX[:-1, 0] @ u_dX[1:, 1:] / n_obs
            Delta_ue_plus += Delta_0
            
            # Add weighted auto-covariances at other lags
            for j in range(1, bandwidth + 1):
                if j < n_obs - 1:
                    # Calculate weight based on kernel
                    if kernel == 'bartlett':
                        weight = 1 - j / (bandwidth + 1)
                    elif kernel == 'parzen':
                        q = j / (bandwidth + 1)
                        if q <= 0.5:
                            weight = 1 - 6 * q**2 + 6 * q**3
                        else:
                            weight = 2 * (1 - q)**3
                    else:  # quadratic-spectral
                        q = 6 * np.pi * j / (5 * (bandwidth + 1))
                        if q < 0.001:
                            weight = 1
                        else:
                            weight = 3 * (np.sin(q) / q - np.cos(q)) / q**2
                    
                    # Calculate covariance at lag j
                    if j + 1 < n_obs:
                        Gamma_j = u_dX[:-j-1, 0] @ u_dX[j+1:, 1:] / (n_obs - j - 1)
                        Delta_ue_plus += weight * Gamma_j
            
            # Step 4: Apply endogeneity correction to dependent variable
            y_plus = y - Omega_ue @ np.linalg.inv(Omega_ee) @ dX.T
            
            # Step 5: Apply serial correlation correction
            # Estimate FMOLS model
            model = sm.OLS(y_plus, X_with_const)
            results = model.fit()
            
            # Get coefficients
            beta_ols = results.params
            
            # Apply serial correlation correction
            n_obs = len(y)
            correction_term = np.zeros(len(beta_ols))
            correction_term[1:] = (Delta_ue_plus @ np.linalg.inv(Omega_ee)).T
            
            # Final FMOLS coefficients
            beta = beta_ols - correction_term / n_obs
            
            # Calculate residuals
            residuals = y - X_with_const @ beta
            
            logger.info(f"FMOLS estimation completed with {kernel} kernel, bandwidth {bandwidth}")
            
        except Exception as e:
            logger.error(f"Error in FMOLS estimation: {str(e)}")
            raise
            
    else:
        raise ValueError(f"Unsupported estimation method: {method}")
    
    # Track memory after processing
    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
    memory_diff = end_mem - start_mem
    
    logger.debug(f"Estimated cointegration vector using {method}. Memory usage: {memory_diff:.2f} MB")
    
    # Force garbage collection to free memory 
    gc.collect()
    
    return beta, residuals 

@m1_optimized()
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
@timer
def calculate_half_life(residuals: np.ndarray) -> float:
    """
    Calculate half-life of deviations from cointegration equilibrium.
    
    Parameters
    ----------
    residuals : array_like
        Residuals from cointegration regression
        
    Returns
    -------
    float
        Half-life in periods
    """
    # Create lagged residuals
    y = residuals[1:]
    x = sm.add_constant(residuals[:-1])
    
    # Estimate AR(1) coefficient
    model = sm.OLS(y, x).fit()
    ar_coef = model.params[1]
    
    # Calculate half-life
    if ar_coef >= 1.0:
        return float('inf')  # Non-stationary, no convergence
    elif ar_coef <= 0.0:
        return 0.0  # Immediate convergence or oscillation
    else:
        # Half-life formula for AR(1): log(0.5) / log(|ar_coefficient|)
        return np.log(0.5) / np.log(abs(ar_coef))