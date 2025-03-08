"""
Comprehensive diagnostics module for econometric models.

Provides diagnostic tools for:
- Standard model residual tests (normality, autocorrelation, heteroskedasticity)
- Threshold model diagnostics (Hansen & Seo sup-LM test)
- Asymmetric adjustment tests for M-TAR models
- Spatial diagnostics (Moran's I, spatial autocorrelation)
- Structural break tests (Bai-Perron, Zivot-Andrews)
- Parameter stability testing
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white
import matplotlib.pyplot as plt
import logging
import os
import time
import gc
import psutil
from typing import Dict, Any, Tuple, Union, Optional, List, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import ruptures as rpt
from scipy import stats

from src.utils import (
    # Error handling
    handle_errors, ModelError,
    
    # Validation
    validate_time_series, raise_if_invalid, validate_dataframe,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, memoize, disk_cache,
    parallelize_dataframe, configure_system_for_performance, optimize_dataframe,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get default configuration
DEFAULT_ALPHA = config.get('analysis.diagnostics.alpha', 0.05)
DEFAULT_LAGS = config.get('analysis.diagnostics.lags', 12)
PLOT_DIR = config.get('paths.plots', 'plots')

# Configure system for optimal performance
configure_system_for_performance()

class ModelDiagnostics:
    """
    Comprehensive diagnostics for econometric models.
    
    This class provides methods for testing statistical properties of model residuals,
    creating diagnostic plots, and evaluating model validity. Includes specialized
    diagnostics for threshold models, spatial models, and cointegration analysis.
    """
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def __init__(self, residuals=None, model_name=None, original_data=None):
        """
        Initialize the diagnostics.
        
        Parameters
        ----------
        residuals : array_like, optional
            Model residuals for testing
        model_name : str, optional
            Name of the model for logging and plots
        original_data : array_like, optional
            Original data used for model estimation
        """
        self.residuals = residuals
        self.model_name = model_name or "Model"
        self.original_data = original_data
        
        # Get number of available workers based on CPU count
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"Initializing ModelDiagnostics for {self.model_name}. Memory usage: {memory_usage:.2f} MB")
        
        # Ensure plot directory exists
        if not os.path.exists(PLOT_DIR):
            try:
                os.makedirs(PLOT_DIR)
                logger.info(f"Created plot directory: {PLOT_DIR}")
            except Exception as e:
                logger.warning(f"Could not create plot directory: {str(e)}")
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, np.linalg.LinAlgError, RuntimeError))
    @timer
    def test_normality(self, residuals=None) -> Dict[str, Any]:
        """
        Test normality of residuals using Jarque-Bera test.
        
        Parameters
        ----------
        residuals : array_like, optional
            Residuals to test, uses self.residuals if not provided
            
        Returns
        -------
        dict
            Normality test results
        """
        residuals = self._get_residuals(residuals)
        
        # Convert to numpy array if pandas Series
        if isinstance(residuals, pd.Series):
            residuals_arr = residuals.values
        else:
            residuals_arr = np.asarray(residuals)
        
        # Run Jarque-Bera test
        jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals_arr)
        
        # Determine result using configured alpha
        normal = jb_pval > DEFAULT_ALPHA
        
        result = {
            'statistic': jb_stat,
            'p_value': jb_pval,
            'skewness': skew,
            'kurtosis': kurtosis,
            'normal': normal
        }
        
        logger.info(f"Normality test (Jarque-Bera): stat={jb_stat:.4f}, p-value={jb_pval:.4f}, normal={normal}")
        return result
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, np.linalg.LinAlgError, RuntimeError))
    @timer
    def test_autocorrelation(self, residuals=None, lags=None) -> Dict[str, Any]:
        """
        Test for autocorrelation using Breusch-Godfrey test.
        
        Parameters
        ----------
        residuals : array_like, optional
            Residuals to test, uses self.residuals if not provided
        lags : int, optional
            Number of lags to test, uses DEFAULT_LAGS if not provided
            
        Returns
        -------
        dict
            Autocorrelation test results
        """
        residuals = self._get_residuals(residuals)
        if lags is None:
            lags = DEFAULT_LAGS
        
        # Convert to numpy array if pandas Series
        if isinstance(residuals, pd.Series):
            residuals_arr = residuals.values
        else:
            residuals_arr = np.asarray(residuals)
        
        # Create design matrix with constant
        X = np.ones((len(residuals_arr), 1))
        
        try:
            # Run Breusch-Godfrey test
            lm_stat, lm_pval, fstat, fpval = acorr_breusch_godfrey(residuals_arr, X, nlags=lags)
            
            # Determine result using configured alpha
            no_autocorr = lm_pval > DEFAULT_ALPHA
            
            result = {
                'lm_statistic': lm_stat,
                'p_value': lm_pval,
                'f_statistic': fstat,
                'f_p_value': fpval,
                'no_autocorrelation': no_autocorr,
                'lags': lags
            }
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Error in Breusch-Godfrey test: {str(e)}. Using simplified approach.")
            
            # Fallback to simple autocorrelation calculation
            from statsmodels.stats.stattools import acf
            acf_values = acf(residuals_arr, nlags=lags)[1:]  # Exclude lag 0
            
            # Check if any autocorrelation exceeds threshold
            significant_lags = sum(abs(val) > (1.96 / np.sqrt(len(residuals_arr))) for val in acf_values)
            no_autocorr = significant_lags == 0
            
            result = {
                'acf_values': acf_values,
                'significant_lags': significant_lags,
                'no_autocorrelation': no_autocorr,
                'lags': lags,
                'note': "Fallback method used due to LinAlgError"
            }
        
        logger.info(f"Autocorrelation test: no_autocorrelation={result['no_autocorrelation']}")
        return result
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, np.linalg.LinAlgError, RuntimeError))
    @timer
    def test_heteroskedasticity(self, residuals=None) -> Dict[str, Any]:
        """
        Test for heteroskedasticity using White test.
        
        Parameters
        ----------
        residuals : array_like, optional
            Residuals to test, uses self.residuals if not provided
            
        Returns
        -------
        dict
            Heteroskedasticity test results
        """
        residuals = self._get_residuals(residuals)
        
        # Convert to numpy array if pandas Series
        if isinstance(residuals, pd.Series):
            residuals_arr = residuals.values
        else:
            residuals_arr = np.asarray(residuals)
        
        # Create design matrix with constant
        X = np.ones((len(residuals_arr), 1))
        
        try:
            # Run White test
            white_stat, white_pval, f_stat, f_pval = het_white(residuals_arr, X)
            
            # Determine result using configured alpha
            homoskedastic = white_pval > DEFAULT_ALPHA
            
            result = {
                'statistic': white_stat,
                'p_value': white_pval,
                'f_statistic': f_stat,
                'f_p_value': f_pval,
                'homoskedastic': homoskedastic
            }
            
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Error in White test: {str(e)}. Using simplified approach.")
            
            # Fallback to Bartlett's test for equal variances
            # Split the sample in half and test variance equality
            half = len(residuals_arr) // 2
            first_half = residuals_arr[:half]
            second_half = residuals_arr[half:]
            
            # Use Bartlett's test
            stat, pval = stats.bartlett(first_half, second_half)
            homoskedastic = pval > DEFAULT_ALPHA
            
            result = {
                'statistic': stat,
                'p_value': pval,
                'homoskedastic': homoskedastic,
                'note': "Fallback to Bartlett's test due to error in White test"
            }
        
        logger.info(f"Heteroskedasticity test: homoskedastic={result['homoskedastic']}")
        return result
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, np.linalg.LinAlgError, RuntimeError))
    @timer
    def residual_tests(self, residuals=None, lags=None) -> Dict[str, Any]:
        """
        Run comprehensive tests on model residuals.
        
        Parameters
        ----------
        residuals : array_like, optional
            Model residuals, uses self.residuals if not provided
        lags : int, optional
            Number of lags for autocorrelation test
            
        Returns
        -------
        dict
            All test results and overall assessment
        """
        residuals = self._get_residuals(residuals)
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run individual tests in parallel for better performance
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit test tasks
            future_normality = executor.submit(self.test_normality, residuals)
            future_autocorr = executor.submit(self.test_autocorrelation, residuals, lags)
            future_hetero = executor.submit(self.test_heteroskedasticity, residuals)
            
            # Collect results
            normality_results = future_normality.result()
            autocorr_results = future_autocorr.result()
            hetero_results = future_hetero.result()
        
        # Calculate basic statistics
        if isinstance(residuals, pd.Series):
            residuals_arr = residuals.values
        else:
            residuals_arr = np.asarray(residuals)
            
        statistics = {
            'mean': np.mean(residuals_arr),
            'std': np.std(residuals_arr),
            'min': np.min(residuals_arr),
            'max': np.max(residuals_arr)
        }
        
        # Overall assessment
        issues = []
        
        if not normality_results.get('normal', True):
            msg = f"Non-normal residuals (JB p-value: {normality_results['p_value']:.4f})"
            issues.append(msg)
            logger.warning(msg)
            
        if not autocorr_results.get('no_autocorrelation', True):
            msg = f"Autocorrelation detected (p-value: {autocorr_results.get('p_value', 0):.4f})"
            issues.append(msg)
            logger.warning(msg)
            
        if not hetero_results.get('homoskedastic', True):
            msg = f"Heteroskedasticity detected (p-value: {hetero_results['p_value']:.4f})"
            issues.append(msg)
            logger.warning(msg)
        
        # Determine if all tests pass
        valid_tests = (
            normality_results.get('normal', False) and
            autocorr_results.get('no_autocorrelation', False) and
            hetero_results.get('homoskedastic', False)
        )
        
        results = {
            'normality': normality_results,
            'autocorrelation': autocorr_results,
            'heteroskedasticity': hetero_results,
            'statistics': statistics,
            'overall': {
                'valid': valid_tests,
                'issues': issues
            }
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Residual diagnostics complete: valid={valid_tests}. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return results
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def plot_diagnostics(self, residuals=None, title=None, save_path=None, 
                        fig_size=(12, 10), dpi=300, plot_acf=True, 
                        plot_dist=True, plot_qq=True, plot_ts=True) -> Dict[str, plt.Figure]:
        """
        Create diagnostic plots for model residuals.
        
        Parameters
        ----------
        residuals : array_like, optional
            Model residuals, uses self.residuals if not provided
        title : str, optional
            Plot title, uses self.model_name if not provided
        save_path : str, optional
            Path to save the plot, if None, uses configured PLOT_DIR
        fig_size : tuple, optional
            Figure size as (width, height) in inches
        dpi : int, optional
            Resolution for saved figure
        plot_acf : bool, optional
            Whether to plot autocorrelation function
        plot_dist : bool, optional
            Whether to plot distribution
        plot_qq : bool, optional
            Whether to plot QQ plot
        plot_ts : bool, optional
            Whether to plot time series
            
        Returns
        -------
        dict
            Dictionary of created figures
        """
        residuals = self._get_residuals(residuals)
        if title is None:
            title = f"{self.model_name} Diagnostics"
        
        # Convert to pandas Series if not already
        if isinstance(residuals, pd.Series):
            residuals_series = residuals
        else:
            residuals_series = pd.Series(residuals)
        
        figures = {}
        
        # Create directory if saving plots
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Combined plot with multiple diagnostics
        if plot_ts or plot_dist or plot_qq or plot_acf:
            fig, axes = plt.subplots(2, 2, figsize=fig_size)
            fig.suptitle(title, fontsize=14)
            
            # Time series plot
            if plot_ts:
                ax = axes[0, 0]
                x = residuals_series.index if hasattr(residuals_series, 'index') else np.arange(len(residuals_series))
                ax.plot(x, residuals_series)
                ax.axhline(y=0, color='r', linestyle='-')
                ax.set_title("Residuals")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
            
            # Distribution plot
            if plot_dist:
                ax = axes[0, 1]
                ax.hist(residuals_series, bins=30, density=True, alpha=0.7)
                
                # Add normal curve
                x = np.linspace(min(residuals_series), max(residuals_series), 100)
                mu, std = stats.norm.fit(residuals_series)
                p = stats.norm.pdf(x, mu, std)
                ax.plot(x, p, 'k', linewidth=2)
                ax.set_title("Histogram of Residuals")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
            
            # QQ plot
            if plot_qq:
                ax = axes[1, 0]
                from statsmodels.graphics.gofplots import qqplot
                qqplot(residuals_series, line='s', ax=ax)
                ax.set_title("QQ Plot")
            
            # ACF plot
            if plot_acf:
                ax = axes[1, 1]
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(residuals_series, lags=min(20, len(residuals_series) // 4), ax=ax)
                ax.set_title("Autocorrelation")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            figures['combined'] = fig
            
            # Save plot if requested
            if save_path is None and PLOT_DIR:
                # Generate filename from title
                filename = f"{title.replace(' ', '_').lower()}_diagnostics.png"
                save_path = os.path.join(PLOT_DIR, filename)
            
            if save_path:
                try:
                    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Saved diagnostic plot to {save_path}")
                except Exception as e:
                    logger.warning(f"Failed to save plot: {str(e)}")
        
        return figures
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def test_model_stability(self, window_size=None, step_size=10, 
                           stability_threshold=0.5, model_func=None, 
                           data=None) -> Dict[str, Any]:
        """
        Test parameter stability using rolling estimation.
        
        Parameters
        ----------
        window_size : int, optional
            Size of the rolling window (default: 20% of sample)
        step_size : int, optional
            Step size for rolling window
        stability_threshold : float, optional
            Threshold for coefficient of variation to determine stability
        model_func : callable, optional
            Function to estimate model parameters
        data : array_like, optional
            Time series data, uses self.original_data if not provided
            
        Returns
        -------
        dict
            Stability test results
        """
        # Get data for stability testing
        if data is None:
            data = self.original_data
            if data is None:
                raise ModelError("No data provided for stability test")
        
        # Set default window size if not provided
        if window_size is None:
            window_size = max(20, int(len(data) * 0.2))
        
        # Validate inputs
        if not isinstance(window_size, int) or window_size <= 0:
            raise ModelError(f"window_size must be a positive integer, got {window_size}")
            
        if not isinstance(step_size, int) or step_size <= 0:
            raise ModelError(f"step_size must be a positive integer, got {step_size}")
        
        # Validate data length
        if len(data) < window_size:
            raise ModelError(f"Data length ({len(data)}) must be at least window_size ({window_size})")
        
        # If model_func not provided, create a simple OLS estimator
        if model_func is None:
            def model_func(data_window):
                # Use OLS with a simple AR(1) specification
                y = data_window[1:]
                X = sm.add_constant(data_window[:-1])
                model = sm.OLS(y, X).fit()
                return model.params
        
        logger.info(f"Testing model stability with window_size={window_size}, step_size={step_size}")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Get total sample size
        n = len(data)
        
        # Calculate number of windows
        n_windows = (n - window_size) // step_size + 1
        if n_windows <= 1:
            raise ModelError(f"Not enough data for multiple windows. Try smaller window_size or step_size.")
        
        # Create array to store parameters
        params_history = []
        dates = []
        
        # Process windows in parallel for better performance
        windows = [(i * step_size, i * step_size + window_size) for i in range(n_windows)]
        
        # For very large datasets, use chunking to reduce memory usage
        max_chunk_size = config.get('data.max_chunk_size', 10000)
        
        if n > max_chunk_size:
            # Split the data into manageable chunks
            data_chunks = []
            for i in range(0, n, max_chunk_size):
                end_idx = min(i + max_chunk_size, n)
                data_chunks.append(data[i:end_idx])
            
            logger.info(f"Data split into {len(data_chunks)} chunks for memory efficiency")
            
            # Process each chunk
            all_results = []
            for chunk_idx, chunk in enumerate(data_chunks):
                chunk_windows = [w for w in windows if w[0] >= chunk_idx * max_chunk_size and 
                                w[1] <= (chunk_idx + 1) * max_chunk_size or chunk_idx == len(data_chunks) - 1]
                
                if chunk_windows:
                    chunk_results = self._process_stability_windows_parallel(
                        chunk, chunk_windows, model_func
                    )
                    all_results.extend(chunk_results)
            
            # Sort results by window index
            all_results.sort(key=lambda x: x[0])
            
            # Extract params and dates
            for window_idx, params, date in all_results:
                params_history.append(params)
                dates.append(date)
                
        else:
            # Process all windows at once for smaller datasets
            window_results = self._process_stability_windows_parallel(
                data, windows, model_func
            )
            
            # Extract params and dates
            for window_idx, params, date in window_results:
                params_history.append(params)
                dates.append(date)
        
        # Check if we have any successful estimations
        if len(params_history) == 0:
            raise ModelError("No successful model estimations. Check model function.")
        
        # Convert to numpy array
        params_history = np.array(params_history)
        
        # Calculate statistics
        mean_params = np.mean(params_history, axis=0)
        std_params = np.std(params_history, axis=0)
        
        # Calculate coefficient of variation
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_params = std_params / np.abs(mean_params)
            # Replace NaN and infinity with a large number
            cv_params = np.where(np.isfinite(cv_params), cv_params, 10)
        
        # Determine if parameters are stable
        is_stable = (cv_params < stability_threshold).all()
        
        unstable_params = []
        if not is_stable:
            # Identify which parameters are unstable
            for i, cv in enumerate(cv_params):
                if cv >= stability_threshold:
                    unstable_params.append(f"Parameter {i+1}: CV = {cv:.2f}")
        
        results = {
            'params_history': params_history,
            'dates': dates,
            'mean_params': mean_params,
            'std_params': std_params,
            'cv_params': cv_params,
            'stable': is_stable,
            'unstable_params': unstable_params
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Model stability test complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        if is_stable:
            logger.info("Stability test result: Model parameters are stable")
        else:
            logger.warning(f"Model parameters are UNSTABLE. {', '.join(unstable_params)}")
        
        return results
    
    @handle_errors(logger=logger)
    def _process_stability_windows_parallel(
        self, 
        data: np.ndarray, 
        windows: List[Tuple[int, int]], 
        model_func: Callable
    ) -> List[Tuple[int, np.ndarray, Any]]:
        """
        Process stability windows in parallel.
        
        Parameters
        ----------
        data : np.ndarray
            Data to analyze
        windows : List[Tuple[int, int]]
            List of window start and end indices
        model_func : Callable
            Function to estimate model parameters
            
        Returns
        -------
        List[Tuple[int, np.ndarray, Any]]
            List of (window_index, parameters, date) tuples
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit window processing tasks
            futures = {}
            for i, (start_idx, end_idx) in enumerate(windows):
                future = executor.submit(
                    self._process_stability_window,
                    data, start_idx, end_idx, model_func, i
                )
                futures[future] = i
            
            # Collect results as they complete
            for future in as_completed(futures):
                window_idx = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append((window_idx, *result))
                except Exception as e:
                    logger.warning(f"Error processing window {window_idx}: {e}")
        
        return results
    
    @handle_errors(logger=logger)
    def _process_stability_window(
        self, 
        data: np.ndarray, 
        start_idx: int, 
        end_idx: int, 
        model_func: Callable, 
        window_idx: int
    ) -> Optional[Tuple[np.ndarray, Any]]:
        """
        Process a single stability window.
        
        Parameters
        ----------
        data : np.ndarray
            Data to analyze
        start_idx : int
            Start index of window
        end_idx : int
            End index of window
        model_func : Callable
            Function to estimate model parameters
        window_idx : int
            Window index for logging
            
        Returns
        -------
        Optional[Tuple[np.ndarray, Any]]
            Tuple of (parameters, date) or None if estimation fails
        """
        try:
            # Get window data
            window_data = data[start_idx:end_idx]
            
            # Estimate model on window
            params = model_func(window_data)
            
            # Ensure params is a numpy array
            if not isinstance(params, np.ndarray):
                params = np.array(params)
            
            # Store corresponding date if available
            if hasattr(data, 'index') and isinstance(data.index, pd.DatetimeIndex):
                date = data.index[end_idx - 1]
            else:
                date = end_idx - 1
                
            return params, date
            
        except Exception as e:
            logger.warning(f"Error estimating model for window {window_idx}: {str(e)}. Skipping window.")
            return None
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer  
    def test_structural_breaks(self, data=None, min_size=5, method='dynp', 
                              max_breaks=3) -> Dict[str, Any]:
        """
        Test for structural breaks using Bai-Perron or similar methods.
        
        Parameters
        ----------
        data : array_like, optional
            Time series to test, uses self.original_data if not provided
        min_size : int, optional
            Minimum segment length
        method : str, optional
            Detection method: 'dynp' (dynamic programming) or 'binseg' (binary segmentation)
        max_breaks : int, optional
            Maximum number of breaks to detect
            
        Returns
        -------
        dict
            Structural break test results
        """
        # Get data for testing
        if data is None:
            data = self.original_data
            if data is None:
                raise ModelError("No data provided for structural break test")
        
        # Validate inputs
        if not isinstance(min_size, int) or min_size <= 1:
            raise ModelError(f"min_size must be at least 2, got {min_size}")
        
        if method not in ['dynp', 'binseg']:
            raise ModelError(f"method must be 'dynp' or 'binseg', got {method}")
        
        # Store original index for later
        has_datetime_index = False
        original_index = None
        
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
            original_index = data.index if has_datetime_index else None
            array = data.values
        else:
            array = np.asarray(data)
        
        array = array.reshape(-1, 1)  # Ensure 2D for ruptures
        
        logger.info(f"Testing for structural breaks using {method} method, max_breaks={max_breaks}")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Select algorithm
        if method == 'dynp':
            algo = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(array)
        elif method == 'binseg':
            algo = rpt.Binseg(model="l2", min_size=min_size).fit(array)
        
        # Get optimal breakpoints
        breakpoints = algo.predict(n_bkps=max_breaks)
        
        # Remove the last breakpoint if it's just the series length
        if breakpoints and breakpoints[-1] == len(array):
            breakpoints = breakpoints[:-1]
        
        # Create breakpoint_dates list if datetime index available
        breakpoint_dates = None
        if has_datetime_index and original_index is not None:
            try:
                breakpoint_dates = [original_index[bp-1] for bp in breakpoints]
            except (IndexError, TypeError) as e:
                logger.warning(f"Could not determine breakpoint dates: {str(e)}")
        
        # Format result
        result = {
            'breakpoints': breakpoints,
            'n_breakpoints': len(breakpoints),
            'method': method
        }
        
        # Add dates if available
        if breakpoint_dates:
            result['breakpoint_dates'] = breakpoint_dates
        
        # Create segments (for potential plotting)
        segments = []
        start = 0
        for bp in breakpoints:
            segments.append((start, bp))
            start = bp
        
        result['segments'] = segments
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Structural break test: detected {result['n_breakpoints']} breakpoints. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return result
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @m1_optimized(parallel=True)
    @timer  
    def test_threshold_validity(self, tvecm_result, data=None, bootstrap_reps=100) -> Dict[str, Any]:
        """
        Test validity of threshold model using Hansen & Seo sup-LM test.
        
        Parameters
        ----------
        tvecm_result : dict
            Results from threshold_vecm or threshold cointegration estimation
        data : array_like, optional
            Original data, uses self.original_data if not provided
        bootstrap_reps : int, optional
            Number of bootstrap replications
            
        Returns
        -------
        dict
            Test results
        """
        # Import specialized functions from threshold_vecm module
        try:
            from src.models.threshold_vecm import ThresholdVECM
        except ImportError:
            logger.warning("Could not import ThresholdVECM. Using simplified approach.")
            # Fallback to manual threshold validity check
            
            # Extract key information from tvecm_result
            has_threshold = 'threshold' in tvecm_result
            threshold_value = tvecm_result.get('threshold', None)
            
            # Check for evidence of threshold effect in adjustment speeds
            adjustment_below = tvecm_result.get('adjustment_below_1', None)
            adjustment_above = tvecm_result.get('adjustment_above_1', None)
            
            if adjustment_below is not None and adjustment_above is not None:
                # Calculate difference in adjustment speeds
                adjustment_diff = abs(adjustment_above) - abs(adjustment_below)
                significant_diff = abs(adjustment_diff) > 0.1  # Arbitrary threshold
                
                result = {
                    'threshold_model_valid': significant_diff,
                    'threshold_value': threshold_value,
                    'adjustment_difference': adjustment_diff,
                    'note': "Simplified threshold validity check (ThresholdVECM not available)"
                }
            else:
                result = {
                    'threshold_model_valid': None,
                    'note': "Cannot determine threshold validity (insufficient data in tvecm_result)"
                }
            
            return result
        
        # Get data for testing
        if data is None:
            data = self.original_data
            if data is None:
                raise ModelError("No data provided for threshold validity test")
        
        # Extract info from tvecm_result
        threshold = tvecm_result.get('threshold')
        if threshold is None:
            raise ModelError("No threshold value found in tvecm_result")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Initialize ThresholdVECM
        model = ThresholdVECM(data)
        
        # Process bootstrap iterations in parallel
        results = []
        
        # Create a pool of workers
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit bootstrap tasks
            futures = []
            for i in range(bootstrap_reps):
                futures.append(executor.submit(
                    self._run_bootstrap_iteration, model, i, bootstrap_reps
                ))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Error in bootstrap iteration: {e}")
        
        # Calculate bootstrap results
        if results:
            bootstrap_stats = [stat for stat in results if stat is not None]
            original_lr = results[0]['original_lr'] if results else None
            
            # Calculate p-value
            p_value = sum(lr > original_lr for lr in bootstrap_stats) / len(bootstrap_stats) if bootstrap_stats else np.nan
            
            # Calculate critical values
            critical_values = {
                '10%': np.percentile(bootstrap_stats, 90),
                '5%': np.percentile(bootstrap_stats, 95),
                '1%': np.percentile(bootstrap_stats, 99)
            }
            
            test_result = {
                'lr_statistic': original_lr,
                'p_value': p_value,
                'significant': p_value < DEFAULT_ALPHA,
                'critical_values': critical_values,
                'bootstrap_distribution': bootstrap_stats,
                'n_bootstrap': len(bootstrap_stats)
            }
        else:
            test_result = {
                'lr_statistic': None,
                'p_value': None,
                'significant': None,
                'note': "Failed to perform bootstrap threshold test"
            }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Threshold validity test complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return test_result
    
    @handle_errors(logger=logger)
    def _run_bootstrap_iteration(
        self, 
        model: Any, 
        iteration: int, 
        total_iterations: int
    ) -> Optional[float]:
        """
        Run a single bootstrap iteration for threshold validity test.
        
        Parameters
        ----------
        model : Any
            ThresholdVECM model instance
        iteration : int
            Current iteration number
        total_iterations : int
            Total number of iterations
            
        Returns
        -------
        Optional[float]
            Likelihood ratio statistic or None if error occurs
        """
        try:
            # If first iteration, calculate original LR statistic
            if iteration == 0:
                # Estimate linear VECM
                linear_results = model.estimate_linear_vecm()
                
                # Estimate TVECM
                model.grid_search_threshold()
                tvecm_results = model.estimate_tvecm()
                
                # Calculate LR statistic
                original_lr = 2 * (
                    (tvecm_results['below_regime']['llf'] + tvecm_results['above_regime']['llf']) - 
                    linear_results['llf']
                )
                
                return {
                    'original_lr': original_lr,
                    'bootstrap_lr': original_lr
                }
            
            # Generate bootstrap sample
            bootstrap_data = model._generate_bootstrap_sample()
            
            # Create new model with bootstrap data
            bootstrap_model = type(model)(
                data=bootstrap_data,
                k_ar_diff=model.k_ar_diff,
                deterministic=model.deterministic,
                coint_rank=model.coint_rank
            )
            
            # Estimate linear VECM
            bootstrap_linear = bootstrap_model.estimate_linear_vecm()
            
            # Estimate threshold
            bootstrap_model.grid_search_threshold()
            
            # Estimate TVECM
            bootstrap_tvecm = bootstrap_model.estimate_tvecm()
            
            # Calculate LR statistic
            bootstrap_lr = 2 * (
                (bootstrap_tvecm['below_regime']['llf'] + bootstrap_tvecm['above_regime']['llf']) - 
                bootstrap_linear['llf']
            )
            
            # Log progress for long-running bootstrap
            if total_iterations > 50 and iteration % 10 == 0:
                logger.debug(f"Completed bootstrap iteration {iteration}/{total_iterations}")
            
            return bootstrap_lr
            
        except Exception as e:
            logger.debug(f"Error in bootstrap iteration {iteration}: {str(e)}")
            return None
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def test_asymmetric_adjustment(self, residuals, threshold=0) -> Dict[str, Any]:
        """
        Test for asymmetric adjustment speeds in error correction models.
        
        Parameters
        ----------
        residuals : array_like
            Equilibrium errors (residuals from cointegration equation)
        threshold : float, optional
            Threshold value for asymmetry
            
        Returns
        -------
        dict
            Test results
        """
        # Convert to numpy array if pandas Series
        if isinstance(residuals, pd.Series):
            residuals_arr = residuals.values
        else:
            residuals_arr = np.asarray(residuals)
        
        # Create lagged residuals
        y = residuals_arr[1:]
        x = residuals_arr[:-1]
        
        # Create indicator for positive and negative deviations
        pos_indicator = x > threshold
        neg_indicator = x <= threshold
        
        # Create design matrix for asymmetric adjustment
        X_pos = x * pos_indicator
        X_neg = x * neg_indicator
        
        # Add constant
        X = sm.add_constant(np.column_stack((X_pos, X_neg)))
        
        # Estimate model
        model = sm.OLS(y, X).fit()
        
        # Extract coefficients
        const = model.params[0]
        rho_pos = model.params[1]  # Adjustment speed for positive deviations
        rho_neg = model.params[2]  # Adjustment speed for negative deviations
        
        # Calculate t-statistic for hypothesis rho_pos = rho_neg
        coef_diff = rho_pos - rho_neg
        cov_matrix = model.cov_params()
        # Calculate variance of difference using covariance matrix
        var_diff = cov_matrix[1,1] + cov_matrix[2,2] - 2*cov_matrix[1,2]
        t_stat = coef_diff / np.sqrt(var_diff)
        
        # Get p-value (two-sided test)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 3))
        
        # Determine if asymmetry is significant
        asymmetry_significant = p_value < DEFAULT_ALPHA
        
        result = {
            'rho_positive': rho_pos,
            'rho_negative': rho_neg,
            'difference': coef_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'asymmetry_significant': asymmetry_significant,
            'faster_adjustment': 'positive' if abs(rho_pos) > abs(rho_neg) else 'negative'
        }
        
        logger.info(f"Asymmetric adjustment test: significant={asymmetry_significant}, "
                   f"positive={rho_pos:.4f}, negative={rho_neg:.4f}, p-value={p_value:.4f}")
        
        return result
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def test_spatial_autocorrelation(self, data, weights_matrix) -> Dict[str, Any]:
        """
        Test for spatial autocorrelation using Moran's I.
        
        Parameters
        ----------
        data : array_like
            Spatial data to test
        weights_matrix : libpysal.weights or similar
            Spatial weights matrix
            
        Returns
        -------
        dict
            Test results
        """
        # Try to import specialized functions from spatial module
        try:
            from src.models.spatial import SpatialEconometrics
            
            # Initialize SpatialEconometrics with data and weights
            spatial_model = SpatialEconometrics(data)
            
            # If weights_matrix is not None, set it
            if weights_matrix is not None:
                spatial_model.weights = weights_matrix
            
            # Run Moran's I test
            if isinstance(data, pd.DataFrame) and 'price' in data.columns:
                moran_result = spatial_model.moran_i_test('price')
            else:
                # Try to find a suitable numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    moran_result = spatial_model.moran_i_test(numeric_cols[0])
                else:
                    raise ModelError("No suitable numeric column found for Moran's I test")
            
            return moran_result
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Error importing SpatialEconometrics: {str(e)}. Using esda directly.")
            
            # Fallback to direct use of esda
            try:
                from esda.moran import Moran
                
                # Extract values to test
                if isinstance(data, pd.DataFrame):
                    # Try to find a suitable column
                    if 'price' in data.columns:
                        values = data['price'].values
                    else:
                        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            values = data[numeric_cols[0]].values
                        else:
                            raise ModelError("No suitable numeric column found")
                else:
                    values = np.asarray(data)
                
                # Run Moran's I test
                moran = Moran(values, weights_matrix)
                
                result = {
                    'I': moran.I,
                    'expected_I': moran.EI,
                    'p_norm': moran.p_norm,
                    'z_norm': moran.z_norm,
                    'significant': moran.p_norm < DEFAULT_ALPHA,
                    'positive_autocorrelation': moran.I > moran.EI and moran.p_norm < DEFAULT_ALPHA
                }
                
                logger.info(f"Moran's I test: I={result['I']:.4f}, p={result['p_norm']:.4f}, "
                           f"significant={result['significant']}")
                return result
                
            except Exception as nested_e:
                logger.error(f"Error running Moran's I test: {str(nested_e)}")
                return {
                    'error': str(nested_e),
                    'note': "Could not run spatial autocorrelation test"
                }
    
    def _get_residuals(self, residuals=None):
        """Helper method to get residuals, either from input or stored attribute."""
        if residuals is not None:
            return residuals
        elif self.residuals is not None:
            return self.residuals
        else:
            raise ModelError("No residuals provided or stored in object")
    
    @m1_optimized(parallel=True)
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def run_all_diagnostics(self, residuals=None, data=None, model_type=None,
                          tvecm_result=None, weights_matrix=None, 
                          plot=True, save_plots=True) -> Dict[str, Any]:
        """
        Run comprehensive model diagnostics based on model type.
        
        Parameters
        ----------
        residuals : array_like, optional
            Model residuals
        data : array_like, optional
            Original data
        model_type : str, optional
            Type of model ('vecm', 'threshold', 'spatial', or 'general')
        tvecm_result : dict, optional
            Results from threshold_vecm estimation (required for threshold diagnostics)
        weights_matrix : object, optional
            Spatial weights matrix (required for spatial diagnostics)
        plot : bool, optional
            Whether to create diagnostic plots
        save_plots : bool, optional
            Whether to save plots to disk
            
        Returns
        -------
        dict
            Comprehensive diagnostic results
        """
        # Set instance variables if provided
        if residuals is not None:
            self.residuals = residuals
        if data is not None:
            self.original_data = optimize_dataframe(data) if isinstance(data, pd.DataFrame) else data
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Initialize results dictionary
        results = {}
        
        # Run diagnostics in parallel when possible
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit diagnostic tasks
            futures = {}
            
            # Basic residual diagnostics for all model types
            future_residuals = executor.submit(self.residual_tests, self.residuals)
            futures['residuals'] = future_residuals
            
            # Model-specific diagnostics
            if model_type == 'threshold' or model_type == 'tvecm':
                # Threshold validity test
                if tvecm_result is not None:
                    future_threshold = executor.submit(
                        self.test_threshold_validity, tvecm_result, self.original_data
                    )
                    futures['threshold_validity'] = future_threshold
                
                # Asymmetric adjustment test
                if self.residuals is not None:
                    future_asymmetry = executor.submit(
                        self.test_asymmetric_adjustment, self.residuals
                    )
                    futures['asymmetric_adjustment'] = future_asymmetry
                    
            elif model_type == 'spatial':
                # Spatial autocorrelation test
                if weights_matrix is not None and self.original_data is not None:
                    future_spatial = executor.submit(
                        self.test_spatial_autocorrelation, self.original_data, weights_matrix
                    )
                    futures['spatial_autocorrelation'] = future_spatial
            
            # Common additional diagnostics for all model types
            if self.original_data is not None:
                # Structural break test
                future_breaks = executor.submit(
                    self.test_structural_breaks, self.original_data
                )
                futures['structural_breaks'] = future_breaks
                
                # Parameter stability test - not easily parallelizable due to model_func
                # Will run separately
            
            # Collect results as they complete
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.warning(f"Error in {name} diagnostics: {e}")
                    results[name] = {'error': str(e)}
        
        # Run parameter stability test if original data is available
        if self.original_data is not None:
            try:
                results['parameter_stability'] = self.test_model_stability(data=self.original_data)
            except Exception as e:
                logger.warning(f"Error in parameter stability test: {e}")
                results['parameter_stability'] = {'error': str(e)}
        
        # Create plots if requested
        if plot and self.residuals is not None:
            try:
                plots = self.plot_diagnostics(
                    self.residuals, 
                    title=f"{self.model_name} Diagnostics",
                    save_path=os.path.join(PLOT_DIR, f"{self.model_name.lower()}_diagnostics.png") if save_plots else None
                )
                results['plots'] = plots
            except Exception as e:
                logger.warning(f"Error creating diagnostic plots: {e}")
                results['plots'] = {'error': str(e)}
        
        # Overall assessment
        valid_model = True
        issues = []
        
        # Check residual diagnostics
        if 'residuals' in results and 'overall' in results['residuals']:
            residual_valid = results['residuals']['overall'].get('valid', True)
            residual_issues = results['residuals']['overall'].get('issues', [])
            
            valid_model = valid_model and residual_valid
            issues.extend(residual_issues)
        
        # Check threshold validity for threshold models
        if model_type == 'threshold' and 'threshold_validity' in results:
            threshold_valid = results['threshold_validity'].get('threshold_model_valid', True)
            
            if threshold_valid is False:
                valid_model = False
                issues.append("Threshold effect not significant")
        
        # Check spatial autocorrelation for spatial models
        if model_type == 'spatial' and 'spatial_autocorrelation' in results:
            spatial_valid = results['spatial_autocorrelation'].get('significant', True)
            
            if spatial_valid is False:
                valid_model = False
                issues.append("No significant spatial autocorrelation")
        
        # Check parameter stability
        if 'parameter_stability' in results:
            stability_valid = results['parameter_stability'].get('stable', True)
            
            if stability_valid is False:
                valid_model = False
                issues.append("Model parameters not stable")
        
        results['overall'] = {
            'valid_model': valid_model,
            'issues': issues
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Model diagnostics complete: valid_model={valid_model}. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return results


@m1_optimized(parallel=True)
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
@timer
def calculate_fit_statistics(
    observed: Union[pd.Series, np.ndarray], 
    predicted: Union[pd.Series, np.ndarray],
    n_params: int = 1
) -> Dict[str, float]:
    """
    Calculate comprehensive model fit statistics.
    
    Parameters
    ----------
    observed : array_like
        Observed values
    predicted : array_like
        Predicted values
    n_params : int, optional
        Number of parameters in the model
        
    Returns
    -------
    dict
        Dictionary of fit statistics
    """
    # Validate inputs
    valid1, errors1 = validate_time_series(observed, min_length=3, max_nulls=0)
    valid2, errors2 = validate_time_series(predicted, min_length=3, max_nulls=0)
    
    if not valid1:
        raise_if_invalid(valid1, errors1, "Invalid observed values")
    if not valid2:
        raise_if_invalid(valid2, errors2, "Invalid predicted values")
    
    # Convert to numpy arrays
    if isinstance(observed, pd.Series):
        observed = observed.values
    if isinstance(predicted, pd.Series):
        predicted = predicted.values
    
    # Check lengths match
    if len(observed) != len(predicted):
        raise ModelError(f"Length of observed ({len(observed)}) must match predicted ({len(predicted)})")
    
    # For large arrays, process in chunks to reduce memory usage
    chunk_size = 10000
    n = len(observed)
    
    if n > chunk_size:
        # Process in chunks and combine results
        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
        
        # Process chunks in parallel
        with ProcessPoolExecutor() as executor:
            futures = {}
            for i, (start, end) in enumerate(chunks):
                future = executor.submit(
                    _calculate_fit_statistics_chunk,
                    observed[start:end],
                    predicted[start:end],
                    n_params
                )
                futures[future] = i
            
            # Collect partial statistics
            chunk_results = []
            for future in as_completed(futures):
                chunk_results.append(future.result())
        
        # Combine chunk statistics
        return _combine_fit_statistics_chunks(chunk_results, n, n_params)
        
    else:
        # Process all data at once for smaller arrays
        return _calculate_fit_statistics_chunk(observed, predicted, n_params)


@handle_errors(logger=logger)
def _calculate_fit_statistics_chunk(
    observed: np.ndarray, 
    predicted: np.ndarray, 
    n_params: int
) -> Dict[str, float]:
    """
    Calculate fit statistics for a data chunk.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values for this chunk
    predicted : np.ndarray
        Predicted values for this chunk
    n_params : int
        Number of parameters in the model
        
    Returns
    -------
    Dict[str, float]
        Partial fit statistics for this chunk
    """
    # Calculate residuals
    residuals = observed - predicted
    
    # Calculate statistics
    n_chunk = len(observed)
    y_mean_chunk = np.mean(observed)
    
    # Total sum of squares
    ss_total_chunk = np.sum((observed - y_mean_chunk) ** 2)
    
    # Residual sum of squares
    ss_residual_chunk = np.sum(residuals ** 2)
    
    # RMSE and MAE
    rmse_chunk = np.sqrt(ss_residual_chunk / n_chunk)
    mae_chunk = np.mean(np.abs(residuals))
    
    # Mean Absolute Percentage Error (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_percent_errors = np.abs(residuals / observed) * 100
        mape_chunk = np.mean(abs_percent_errors[np.isfinite(abs_percent_errors)])
    
    # Log likelihood (assuming normal errors)
    sigma2_chunk = ss_residual_chunk / n_chunk
    loglikelihood_chunk = -n_chunk/2 * (1 + np.log(2 * np.pi) + np.log(sigma2_chunk))
    
    return {
        'n': n_chunk,
        'y_mean': y_mean_chunk,
        'ss_total': ss_total_chunk,
        'ss_residual': ss_residual_chunk,
        'rmse': rmse_chunk,
        'mae': mae_chunk,
        'mape': mape_chunk,
        'loglikelihood': loglikelihood_chunk
    }


@handle_errors(logger=logger)
def _combine_fit_statistics_chunks(
    chunk_results: List[Dict[str, float]], 
    n: int, 
    n_params: int
) -> Dict[str, float]:
    """
    Combine fit statistics from multiple chunks.
    
    Parameters
    ----------
    chunk_results : List[Dict[str, float]]
        Statistics calculated for each chunk
    n : int
        Total number of observations
    n_params : int
        Number of parameters in the model
        
    Returns
    -------
    Dict[str, float]
        Combined fit statistics
    """
    # Sum values across chunks
    ss_total = sum(chunk['ss_total'] for chunk in chunk_results)
    ss_residual = sum(chunk['ss_residual'] for chunk in chunk_results)
    
    # Calculate combined metrics
    r_squared = 1 - (ss_residual / ss_total)
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_params - 1))
    
    # RMSE (recalculate from total ss_residual)
    rmse = np.sqrt(ss_residual / n)
    
    # MAE (weighted average)
    mae = sum(chunk['mae'] * chunk['n'] for chunk in chunk_results) / n
    
    # MAPE (weighted average)
    mape = sum(chunk['mape'] * chunk['n'] for chunk in chunk_results) / n
    
    # Log likelihood (sum across chunks)
    loglikelihood = sum(chunk['loglikelihood'] for chunk in chunk_results)
    
    # AIC and BIC
    aic = -2 * loglikelihood + 2 * n_params
    bic = -2 * loglikelihood + n_params * np.log(n)
    
    return {
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'aic': aic,
        'bic': bic,
        'loglikelihood': loglikelihood,
        'n_obs': n,
        'n_params': n_params
    }


@timer
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
def bootstrap_confidence_intervals(
    data: Union[pd.Series, np.ndarray],
    statistic_func: Callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    method: str = 'percentile'
) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for a statistic.
    
    Parameters
    ----------
    data : array_like
        Data to bootstrap from
    statistic_func : callable
        Function to compute the statistic
    n_bootstrap : int, optional
        Number of bootstrap samples
    alpha : float, optional
        Significance level (e.g., 0.05 for 95% confidence)
    method : str, optional
        Method for computing intervals ('percentile' or 'bca')
        
    Returns
    -------
    dict
        Bootstrap results
    """
    # Convert to numpy array if pandas Series
    if isinstance(data, pd.Series):
        data_arr = data.values
    else:
        data_arr = np.asarray(data)
    
    n = len(data_arr)
    
    # Calculate the observed statistic
    observed_stat = statistic_func(data_arr)
    
    # Get number of available workers
    n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
    
    # Track memory usage
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Generate bootstrap samples in parallel
    bootstrap_stats = []
    
    # Distribute bootstrap iterations across workers
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit bootstrap tasks
        futures = []
        batch_size = max(10, n_bootstrap // (n_workers * 2))  # Ensure enough tasks for parallelism
        
        for i in range(0, n_bootstrap, batch_size):
            n_samples = min(batch_size, n_bootstrap - i)
            futures.append(executor.submit(
                _run_bootstrap_batch,
                data_arr, n, statistic_func, n_samples, i
            ))
        
        # Collect results
        for future in as_completed(futures):
            try:
                batch_stats = future.result()
                bootstrap_stats.extend(batch_stats)
            except Exception as e:
                logger.warning(f"Error in bootstrap batch: {e}")
    
    # Calculate confidence intervals
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    if method == 'percentile':
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    elif method == 'bca':
        # BCa method (bias-corrected and accelerated) - simplified implementation
        # This is not a full BCa implementation but a simplified version
        z0 = stats.norm.ppf(np.mean(np.array(bootstrap_stats) < observed_stat))
        
        # Calculate acceleration factor
        jackknife_stats = []
        
        # Process jackknife estimates in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit jackknife tasks in batches
            futures = []
            batch_size = max(10, n // (n_workers * 2))
            
            for i in range(0, n, batch_size):
                end_idx = min(i + batch_size, n)
                futures.append(executor.submit(
                    _run_jackknife_batch,
                    data_arr, statistic_func, list(range(i, end_idx))
                ))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_stats = future.result()
                    jackknife_stats.extend(batch_stats)
                except Exception as e:
                    logger.warning(f"Error in jackknife batch: {e}")
        
        jackknife_mean = np.mean(jackknife_stats)
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        
        # Avoid division by zero
        if denominator != 0:
            a = numerator / denominator
        else:
            a = 0
        
        # Adjusted percentiles
        z_alpha = stats.norm.ppf(alpha / 2)
        z_1_alpha = stats.norm.ppf(1 - alpha / 2)
        
        # BCa percentiles
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
        
        lower_bound = np.percentile(bootstrap_stats, p_lower * 100)
        upper_bound = np.percentile(bootstrap_stats, p_upper * 100)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Track memory after processing
    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
    memory_diff = end_mem - start_mem
    
    logger.info(f"Bootstrap confidence intervals calculated. Memory usage: {memory_diff:.2f} MB")
    
    # Force garbage collection
    gc.collect()
    
    return {
        'statistic': observed_stat,
        'bootstrap_stats': bootstrap_stats,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha,
        'method': method
    }


@handle_errors(logger=logger)
def _run_bootstrap_batch(
    data: np.ndarray, 
    n: int, 
    statistic_func: Callable, 
    n_samples: int,
    batch_idx: int
) -> List[float]:
    """
    Run a batch of bootstrap samples.
    
    Parameters
    ----------
    data : np.ndarray
        Original data array
    n : int
        Length of data
    statistic_func : Callable
        Function to compute statistic
    n_samples : int
        Number of bootstrap samples in this batch
    batch_idx : int
        Batch index for logging
        
    Returns
    -------
    List[float]
        Bootstrap statistics for this batch
    """
    np.random.seed(batch_idx)  # Ensure reproducibility with different seeds
    
    batch_stats = []
    for i in range(n_samples):
        # Draw random sample with replacement
        indices = np.random.randint(0, n, size=n)
        sample = data[indices]
        
        # Calculate statistic
        try:
            stat = statistic_func(sample)
            batch_stats.append(stat)
        except Exception as e:
            logger.debug(f"Error in bootstrap iteration {batch_idx * n_samples + i}: {e}")
    
    return batch_stats


@handle_errors(logger=logger)
def _run_jackknife_batch(
    data: np.ndarray, 
    statistic_func: Callable, 
    indices_to_remove: List[int]
) -> List[float]:
    """
    Run a batch of jackknife samples.
    
    Parameters
    ----------
    data : np.ndarray
        Original data array
    statistic_func : Callable
        Function to compute statistic
    indices_to_remove : List[int]
        Indices to remove for jackknife samples
        
    Returns
    -------
    List[float]
        Jackknife statistics for this batch
    """
    batch_stats = []
    for i in indices_to_remove:
        # Create leave-one-out sample
        sample = np.delete(data, i)
        
        # Calculate statistic
        try:
            stat = statistic_func(sample)
            batch_stats.append(stat)
        except Exception as e:
            logger.debug(f"Error in jackknife sample {i}: {e}")
    
    return batch_stats


@timer
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
def compute_prediction_intervals(
    model_func: Callable,
    data: Union[pd.DataFrame, np.ndarray],
    pred_x: Union[pd.DataFrame, np.ndarray],
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute prediction intervals using bootstrap.
    
    Parameters
    ----------
    model_func : callable
        Function that takes data and returns model
    data : array_like
        Original data for model estimation
    pred_x : array_like
        Predictor values for which to compute intervals
    n_bootstrap : int, optional
        Number of bootstrap samples
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Prediction intervals
    """
    # Convert to numpy arrays if pandas objects
    if isinstance(data, pd.DataFrame):
        data_arr = data.values
    else:
        data_arr = np.asarray(data)
    
    if isinstance(pred_x, pd.DataFrame):
        pred_x_arr = pred_x.values
    else:
        pred_x_arr = np.asarray(pred_x)
    
    n = len(data_arr)
    
    # Fit model to original data and get predictions
    model = model_func(data_arr)
    predictions = model.predict(pred_x_arr)
    
    # Get number of available workers
    n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
    
    # Track memory usage
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Generate bootstrap predictions in parallel
    bootstrap_predictions = []
    
    # Distribute bootstrap iterations across workers
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit bootstrap tasks
        futures = []
        batch_size = max(10, n_bootstrap // (n_workers * 2))  # Ensure enough tasks for parallelism
        
        for i in range(0, n_bootstrap, batch_size):
            n_samples = min(batch_size, n_bootstrap - i)
            futures.append(executor.submit(
                _run_prediction_bootstrap_batch,
                data_arr, pred_x_arr, model_func, n, n_samples, i
            ))
        
        # Collect results
        for future in as_completed(futures):
            try:
                batch_preds = future.result()
                bootstrap_predictions.extend(batch_preds)
            except Exception as e:
                logger.warning(f"Error in bootstrap prediction batch: {e}")
    
    # Convert to numpy array
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Calculate intervals
    lower_bound = np.percentile(bootstrap_predictions, alpha/2 * 100, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, (1-alpha/2) * 100, axis=0)
    
    # Track memory after processing
    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
    memory_diff = end_mem - start_mem
    
    logger.info(f"Prediction intervals calculated. Memory usage: {memory_diff:.2f} MB")
    
    # Force garbage collection
    gc.collect()
    
    return {
        'predictions': predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha
    }


@handle_errors(logger=logger)
def _run_prediction_bootstrap_batch(
    data: np.ndarray, 
    pred_x: np.ndarray, 
    model_func: Callable, 
    n: int, 
    n_samples: int,
    batch_idx: int
) -> List[np.ndarray]:
    """
    Run a batch of bootstrap prediction samples.
    
    Parameters
    ----------
    data : np.ndarray
        Original data array
    pred_x : np.ndarray
        Predictor values for predictions
    model_func : Callable
        Function to fit model
    n : int
        Length of data
    n_samples : int
        Number of bootstrap samples in this batch
    batch_idx : int
        Batch index for logging
        
    Returns
    -------
    List[np.ndarray]
        Bootstrap predictions for this batch
    """
    np.random.seed(batch_idx)  # Ensure reproducibility with different seeds
    
    batch_preds = []
    for i in range(n_samples):
        try:
            # Draw random sample with replacement
            indices = np.random.randint(0, n, size=n)
            sample = data[indices]
            
            # Fit model to bootstrap sample
            bootstrap_model = model_func(sample)
            
            # Get predictions
            bootstrap_pred = bootstrap_model.predict(pred_x)
            batch_preds.append(bootstrap_pred)
            
        except Exception as e:
            logger.debug(f"Error in prediction bootstrap iteration {batch_idx * n_samples + i}: {e}")
    
    return batch_preds