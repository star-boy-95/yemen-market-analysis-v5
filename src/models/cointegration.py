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

from src.utils import (
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


@m1_optimized(use_numba=True)
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
@timer
def estimate_cointegration_vector(
    y: Union[pd.Series, np.ndarray],
    x: Union[pd.Series, np.ndarray, List[Union[pd.Series, np.ndarray]]],
    method: str = 'ols'
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
        # Implement Dynamic OLS (Stock-Watson)
        # This is a placeholder implementation - would need more complex implementation
        # for a proper DOLS with leads and lags
        raise NotImplementedError("DOLS method not yet implemented")
    elif method == 'fmols':
        # Implement Fully Modified OLS (Phillips-Hansen)
        # This would require a more complex implementation to account for
        # endogeneity and serial correlation
        raise NotImplementedError("FMOLS method not yet implemented")
    else:
        raise ValueError(f"Unsupported estimation method: {method}")
    
    # Track memory after processing
    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
    memory_diff = end_mem - start_mem
    
    logger.debug(f"Estimated cointegration vector using {method}. Memory usage: {memory_diff:.2f} MB")
    
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