"""
Comprehensive diagnostics module for econometric models.

Provides diagnostic tools for:
- Standard model residual tests (normality, autocorrelation, heteroskedasticity)
- Threshold model diagnostics (Hansen & Seo sup-LM test)
- Asymmetric adjustment tests for M-TAR models
- Spatial diagnostics (Moran's I, Geary's C, spatial autocorrelation)
- Structural break tests (Bai-Perron, Zivot-Andrews)
- Parameter stability testing (CUSUM, recursive estimation)
- Prediction intervals (analytical and bootstrap methods)

The module is designed for optimal performance on the M1 Mac architecture with
memory optimization techniques for handling large datasets.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white, recursive_olsresiduals
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import logging
import os
import time
import gc
import psutil
from typing import Dict, Any, Tuple, Union, Optional, List, Callable, Generator, TypeVar
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing as mp
import ruptures as rpt
from scipy import stats, sparse
from scipy.linalg import toeplitz
from pathlib import Path
import warnings

from src.utils import (
    # Error handling
    handle_errors, ModelError,
    
    # Validation
    validate_time_series, raise_if_invalid, validate_dataframe, validate_geodataframe,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, memoize, disk_cache,
    parallelize_dataframe, configure_system_for_performance, optimize_dataframe,
    
    # Spatial utilities
    create_spatial_weight_matrix, reproject_gdf,
    
    # Statistics utilities
    bootstrap_confidence_interval, test_structural_break,
    test_white_noise, test_autocorrelation, test_stationarity,
    
    # Plotting utilities
    set_plotting_style, create_figure, format_date_axis, plot_time_series,
    add_annotations, save_plot
)

# Type variable for generic functions
T = TypeVar('T')

# Initialize module logger
logger = logging.getLogger(__name__)

# Get default configuration
DEFAULT_ALPHA = config.get('analysis.diagnostics.alpha', 0.05)
DEFAULT_LAGS = config.get('analysis.diagnostics.lags', 12)
PLOT_DIR = config.get('paths.plots', 'plots')
DEFAULT_BOOTSTRAP_SAMPLES = config.get('analysis.diagnostics.bootstrap_samples', 1000)
DEFAULT_BATCH_SIZE = config.get('analysis.diagnostics.batch_size', 100)

# Configure system for optimal performance
configure_system_for_performance()

# Create plot directory if it doesn't exist
Path(PLOT_DIR).mkdir(exist_ok=True, parents=True)


class ResidualsAnalysis:
    """
    Specialized component for analyzing model residuals.
    
    Provides methods for testing normality, autocorrelation, heteroskedasticity,
    and other statistical properties of model residuals.
    """
    
    def __init__(self, residuals=None):
        """
        Initialize the residuals analyzer.
        
        Parameters
        ----------
        residuals : array_like, optional
            Model residuals for testing
        """
        self.residuals = residuals
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
    
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
    def run_all_tests(self, residuals=None, lags=None) -> Dict[str, Any]:
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

    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def create_advanced_diagnostic_plots(
        self, 
        residuals: Union[pd.Series, np.ndarray], 
        save_path: Optional[str] = None,
        include_bootstrap: bool = True,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Create advanced diagnostic plots for model residuals.
        
        This method generates a comprehensive set of visualizations for
        analyzing residuals, including time series plots with confidence bands,
        ACF/PACF plots, QQ plots, histograms, rolling statistics, and CUSUM
        stability tests.
        
        Parameters
        ----------
        residuals : array_like
            Model residuals for analysis
        save_path : str, optional
            Base path to save the plots. If provided, creates a directory
            for the plots if it doesn't exist.
        include_bootstrap : bool, optional
            Whether to calculate bootstrap confidence bands
        n_bootstrap : int, optional
            Number of bootstrap replications if bootstrap bands are requested
        confidence_level : float, optional
            Confidence level for intervals (0-1)
            
        Returns
        -------
        dict
            Dictionary containing figure objects for each plot type:
            - 'time_series': Time series plot with confidence bands
            - 'acf_pacf': Autocorrelation and partial autocorrelation plots
            - 'qq_plot': QQ plot with confidence bands
            - 'histogram': Histogram with KDE and normal overlay
            - 'rolling_stats': Rolling statistics plots
            - 'cusum': CUSUM stability test plot
            
        Notes
        -----
        This method applies performance optimizations for M1 Mac hardware
        when generating bootstrap confidence bands. Bootstrap confidence bands
        provide robust uncertainty quantification especially for non-normal residuals.
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Set plotting style
        set_plotting_style()
        
        # Convert input to numpy array if pandas Series
        if isinstance(residuals, pd.Series):
            residuals_series = residuals
            residuals_array = residuals.values
            has_index = True
            index = residuals.index
        else:
            residuals_array = np.asarray(residuals)
            residuals_series = pd.Series(residuals_array)
            has_index = False
            index = np.arange(len(residuals_array))
        
        # Create output directory if saving plots
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory for diagnostic plots: {save_dir}")
        
        # Initialize results dictionary
        figures = {}
        
        # 1. Time series plot with confidence bands
        figures['time_series'] = self._create_time_series_plot(
            residuals_array, index, has_index, 
            include_bootstrap, n_bootstrap, confidence_level
        )
        
        # 2. ACF/PACF plots
        figures['acf_pacf'] = self._create_acf_pacf_plot(residuals_array)
        
        # 3. QQ plot with confidence bands
        figures['qq_plot'] = self._create_qq_plot(residuals_array, confidence_level)
        
        # 4. Histogram with KDE and normal overlay
        figures['histogram'] = self._create_histogram_plot(residuals_array)
        
        # 5. Rolling statistics
        figures['rolling_stats'] = self._create_rolling_stats_plot(
            residuals_array, index, has_index
        )
        
        # 6. CUSUM stability test
        figures['cusum'] = self._create_cusum_plot(residuals_array, index, has_index)
        
        # Save figures if save_path is provided
        if save_path:
            for plot_name, fig in figures.items():
                if fig is not None:
                    plot_path = Path(save_path) / f"{plot_name}.png"
                    save_plot(fig, plot_path, dpi=300)
                    logger.info(f"Saved {plot_name} plot to {plot_path}")
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        logger.info(f"Advanced diagnostic plots created. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return figures

    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _create_time_series_plot(
        self, 
        residuals: np.ndarray, 
        index: Union[pd.Index, np.ndarray],
        has_index: bool,
        include_bootstrap: bool, 
        n_bootstrap: int, 
        confidence_level: float
    ) -> plt.Figure:
        """
        Create time series plot with confidence bands.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residual values
        index : pandas.Index or numpy.ndarray
            Index values for x-axis
        has_index : bool
            Whether a meaningful index is available
        include_bootstrap : bool
            Whether to use bootstrap for confidence bands
        n_bootstrap : int
            Number of bootstrap replications
        confidence_level : float
            Confidence level for intervals
            
        Returns
        -------
        matplotlib.figure.Figure
            Time series plot figure
        """
        fig, ax = create_figure(width=12, height=6)
        
        # Plot residuals time series
        if has_index:
            ax.plot(index, residuals, marker='.', linestyle='-', alpha=0.6, 
                    label='Residuals')
        else:
            ax.plot(residuals, marker='.', linestyle='-', alpha=0.6, 
                    label='Residuals')
        
        # Add mean line
        mean_value = np.mean(residuals)
        ax.axhline(y=mean_value, color='r', linestyle='-', label=f'Mean: {mean_value:.4f}')
        
        # Add confidence bands
        if include_bootstrap:
            # Calculate bootstrap confidence bands
            bands = self._calculate_bootstrap_bands(
                residuals, n_bootstrap, confidence_level
            )
            
            # Add confidence bands to plot
            if has_index:
                ax.fill_between(index, bands['lower'], bands['upper'], 
                               color='blue', alpha=0.2, 
                               label=f'{confidence_level*100:.0f}% Confidence Band')
            else:
                ax.fill_between(range(len(residuals)), bands['lower'], bands['upper'], 
                               color='blue', alpha=0.2, 
                               label=f'{confidence_level*100:.0f}% Confidence Band')
        else:
            # Simple analytical bands (±1.96 standard errors for approximately 95% CI)
            std_dev = np.std(residuals)
            z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            upper = mean_value + z_value * std_dev
            lower = mean_value - z_value * std_dev
            
            ax.axhline(y=upper, color='g', linestyle='--', alpha=0.7,
                      label=f'Upper Bound: {upper:.4f}')
            ax.axhline(y=lower, color='g', linestyle='--', alpha=0.7,
                      label=f'Lower Bound: {lower:.4f}')
        
        # Add formatting
        ax.set_title("Residual Time Series with Confidence Bands")
        ax.set_xlabel("Time" if not has_index else "")
        ax.set_ylabel("Residual Value")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format date axis if available
        if has_index and pd.api.types.is_datetime64_any_dtype(index):
            format_date_axis(ax)
        
        # Add runs test to check randomness
        runs_test_result = self._perform_runs_test(residuals)
        
        # Add annotations
        add_annotations(ax, {
            (0.02, 0.02): f"Runs Test p-value: {runs_test_result['p_value']:.4f}\n"
                          f"Random: {runs_test_result['random']}",
        }, offset_x=0, offset_y=0, ha='left', 
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        return fig

    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _calculate_bootstrap_bands(
        self, 
        residuals: np.ndarray, 
        n_bootstrap: int, 
        confidence_level: float
    ) -> Dict[str, np.ndarray]:
        """
        Calculate bootstrap confidence bands for residuals.
        
        Uses parallel processing for efficient bootstrap computation on M1 hardware.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residual values
        n_bootstrap : int
            Number of bootstrap replications
        confidence_level : float
            Confidence level for intervals
            
        Returns
        -------
        dict
            Dictionary with lower and upper confidence bands
        """
        # Use bootstrap_confidence_interval utility
        def mean_func(x):
            return np.mean(x)
        
        # Calculate bootstrap confidence interval for mean
        bootstrap_result = bootstrap_confidence_interval(
            data=residuals,
            statistic_func=mean_func,
            alpha=1-confidence_level,
            n_bootstrap=n_bootstrap,
            method='percentile'
        )
        
        # Get bounds and replicate for time series
        n = len(residuals)
        lower = np.ones(n) * bootstrap_result['lower_bound']
        upper = np.ones(n) * bootstrap_result['upper_bound']
        
        return {'lower': lower, 'upper': upper}

    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _perform_runs_test(self, residuals: np.ndarray) -> Dict[str, Any]:
        """
        Perform runs test for randomness in residuals.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residual values
            
        Returns
        -------
        dict
            Runs test results including p-value and randomness assessment
        """
        # Count runs above and below mean
        mean = np.mean(residuals)
        above_mean = residuals > mean
        
        # Count runs
        runs = np.sum(np.diff(above_mean) != 0) + 1
        
        # Calculate expected runs and standard deviation
        n1 = np.sum(above_mean)
        n2 = len(residuals) - n1
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                           ((n1 + n2) ** 2 * (n1 + n2 - 1)))
        
        # Calculate z-statistic and p-value
        z = (runs - expected_runs) / std_runs
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # Two-sided test
        
        return {
            'runs': runs,
            'expected_runs': expected_runs,
            'z_statistic': z,
            'p_value': p_value,
            'random': p_value >= 0.05
        }

    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _create_acf_pacf_plot(self, residuals: np.ndarray) -> plt.Figure:
        """
        Create ACF and PACF plots with significance testing.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residual values
            
        Returns
        -------
        matplotlib.figure.Figure
            ACF/PACF plot figure
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Calculate and plot ACF
        plot_acf(residuals, ax=ax1, lags=min(40, len(residuals) // 2),
                 alpha=0.05, title='Autocorrelation Function')
        
        # Calculate and plot PACF
        plot_pacf(residuals, ax=ax2, lags=min(40, len(residuals) // 2),
                  alpha=0.05, title='Partial Autocorrelation Function')
        
        # Use project's test_autocorrelation utility
        autocorr_result = test_autocorrelation(
            residuals, lags=min(30, len(residuals) // 4)
        )
        
        # Add test results as annotation
        add_annotations(ax1, {
            (0.5, 0.02): f"Autocorrelation Test (H0: No Autocorrelation)\n"
                         f"Significant Lags: {len(autocorr_result.get('significant_lags', []))}\n"
                         f"Has Autocorrelation: {autocorr_result.get('has_autocorrelation', False)}"
        }, offset_x=0, offset_y=0, ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        return fig

    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _create_qq_plot(self, residuals: np.ndarray, confidence_level: float) -> plt.Figure:
        """
        Create QQ plot with theoretical confidence bands.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residual values
        confidence_level : float
            Confidence level for intervals
            
        Returns
        -------
        matplotlib.figure.Figure
            QQ plot figure
        """
        from statsmodels.graphics.gofplots import qqplot
        
        fig, ax = create_figure(width=10, height=6)
        
        # Create QQ plot
        qq = qqplot(residuals, line='s', ax=ax, fit=True)
        
        # Calculate theoretical confidence bands
        n = len(residuals)
        alpha = 1 - confidence_level
        
        # Generate uniform quantiles
        p = np.linspace(alpha/2, 1-alpha/2, 100)
        
        # Transform to normal quantiles
        theoretical_quantiles = stats.norm.ppf(p)
        
        # Calculate confidence bands
        se = (1 / stats.norm.pdf(theoretical_quantiles)) * np.sqrt(p * (1 - p) / n)
        upper_band = theoretical_quantiles + se * stats.norm.ppf(1 - alpha/2)
        lower_band = theoretical_quantiles - se * stats.norm.ppf(1 - alpha/2)
        
        # Add confidence bands to plot
        ax.plot(theoretical_quantiles, upper_band, 'r--', alpha=0.5,
                label=f'{confidence_level*100:.0f}% Confidence Band')
        ax.plot(theoretical_quantiles, lower_band, 'r--', alpha=0.5)
        
        # Use project's test_normality utility
        normality_result = self.test_normality(residuals)
        
        # Add test results as annotation
        add_annotations(ax, {
            (0.02, 0.95): f"Normality Test Results:\n"
                         f"Statistic: {normality_result.get('statistic', 0):.4f}\n"
                         f"p-value: {normality_result.get('p_value', 0):.4f}\n"
                         f"Normal: {normality_result.get('normal', False)}"
        }, offset_x=0, offset_y=0, va='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax.set_title("QQ Plot with Confidence Bands")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _create_histogram_plot(self, residuals: np.ndarray) -> plt.Figure:
        """
        Create histogram with KDE and normal overlay.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residual values
            
        Returns
        -------
        matplotlib.figure.Figure
            Histogram plot figure
        """
        fig, ax = create_figure(width=10, height=6)
        
        # Calculate statistics
        mean = np.mean(residuals)
        std = np.std(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        # Create histogram
        n, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.6,
                                  label='Histogram')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(residuals)
        x = np.linspace(min(residuals), max(residuals), 1000)
        ax.plot(x, kde(x), 'r-', label='KDE')
        
        # Add normal distribution overlay
        ax.plot(x, stats.norm.pdf(x, mean, std), 'k--', linewidth=2,
               label='Normal Distribution')
        
        # Add vertical lines for mean and +/- 1 and 2 standard deviations
        ax.axvline(x=mean, color='g', linestyle='-', alpha=0.7, label=f'Mean: {mean:.4f}')
        ax.axvline(x=mean+std, color='g', linestyle='--', alpha=0.5, label=f'+1 SD: {mean+std:.4f}')
        ax.axvline(x=mean-std, color='g', linestyle='--', alpha=0.5, label=f'-1 SD: {mean-std:.4f}')
        ax.axvline(x=mean+2*std, color='g', linestyle=':', alpha=0.5, label=f'+2 SD: {mean+2*std:.4f}')
        ax.axvline(x=mean-2*std, color='g', linestyle=':', alpha=0.5, label=f'-2 SD: {mean-2*std:.4f}')
        
        # Use project's test_white_noise utility for comprehensive normality testing
        white_noise_result = test_white_noise(residuals)
        
        # Add test results and statistics as annotation
        add_annotations(ax, {
            (0.98, 0.95): f"Distribution Statistics:\n"
                         f"Mean: {mean:.4f}\n"
                         f"Std Dev: {std:.4f}\n"
                         f"Skewness: {skewness:.4f}\n"
                         f"Kurtosis: {kurtosis:.4f}\n\n"
                         f"White Noise Test:\n"
                         f"Is White Noise: {white_noise_result.get('is_white_noise', False)}\n"
                         f"Normal: {white_noise_result.get('normality_test', {}).get('normal', False)}"
        }, offset_x=0, offset_y=0, va='top', ha='right',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax.set_title("Residual Distribution with Normal Overlay")
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Density")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _create_rolling_stats_plot(
        self, 
        residuals: np.ndarray, 
        index: Union[pd.Index, np.ndarray],
        has_index: bool
    ) -> plt.Figure:
        """
        Create rolling statistics plots for residuals.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residual values
        index : pandas.Index or numpy.ndarray
            Index values for x-axis
        has_index : bool
            Whether a meaningful index is available
            
        Returns
        -------
        matplotlib.figure.Figure
            Rolling statistics plot figure
        """
        # Create DataFrame for rolling calculations
        if has_index:
            df = pd.Series(residuals, index=index)
        else:
            df = pd.Series(residuals)
        
        # Calculate rolling statistics (20% of data points window, min 10 points)
        window = max(10, int(len(residuals) * 0.2))
        rolling_mean = df.rolling(window=window).mean()
        rolling_std = df.rolling(window=window).std()
        rolling_acf = df.rolling(window=window).apply(
            lambda x: pd.Series(x).autocorr(1) if len(x) > 5 else np.nan
        )
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot rolling mean
        if has_index:
            ax1.plot(index, rolling_mean, 'r-', label=f'Rolling Mean (window={window})')
            ax1.axhline(y=np.mean(residuals), color='k', linestyle='--', 
                       label=f'Overall Mean: {np.mean(residuals):.4f}')
        else:
            ax1.plot(rolling_mean, 'r-', label=f'Rolling Mean (window={window})')
            ax1.axhline(y=np.mean(residuals), color='k', linestyle='--', 
                       label=f'Overall Mean: {np.mean(residuals):.4f}')
        
        ax1.set_title("Rolling Mean")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot rolling standard deviation
        if has_index:
            ax2.plot(index, rolling_std, 'g-', label=f'Rolling Std Dev (window={window})')
            ax2.axhline(y=np.std(residuals), color='k', linestyle='--', 
                       label=f'Overall Std Dev: {np.std(residuals):.4f}')
        else:
            ax2.plot(rolling_std, 'g-', label=f'Rolling Std Dev (window={window})')
            ax2.axhline(y=np.std(residuals), color='k', linestyle='--', 
                       label=f'Overall Std Dev: {np.std(residuals):.4f}')
        
        ax2.set_title("Rolling Standard Deviation")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot rolling autocorrelation
        if has_index:
            ax3.plot(index, rolling_acf, 'b-', label=f'Rolling Autocorrelation (window={window})')
            ax3.axhline(y=pd.Series(residuals).autocorr(1), color='k', linestyle='--', 
                       label=f'Overall Autocorr: {pd.Series(residuals).autocorr(1):.4f}')
        else:
            ax3.plot(rolling_acf, 'b-', label=f'Rolling Autocorrelation (window={window})')
            ax3.axhline(y=pd.Series(residuals).autocorr(1), color='k', linestyle='--', 
                       label=f'Overall Autocorr: {pd.Series(residuals).autocorr(1):.4f}')
        
        ax3.set_title("Rolling Autocorrelation (Lag 1)")
        ax3.set_xlabel("Time" if not has_index else "")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Format date axis if available
        if has_index and pd.api.types.is_datetime64_any_dtype(index):
            for ax in [ax1, ax2, ax3]:
                format_date_axis(ax)
        
        # Use project's test_structural_break utility
        struct_break = test_structural_break(
            y=rolling_mean.dropna().values,
            method='quandt'
        )
        
        # Add stability test summary
        add_annotations(ax1, {
            (0.02, 0.05): f"Structural Break Test:\n"
                         f"Break Detected: {struct_break.get('significant', False)}\n"
                         f"Break Point: {struct_break.get('break_date', 'None')}"
        }, offset_x=0, offset_y=0,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        return fig

    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _create_cusum_plot(
        self, 
        residuals: np.ndarray, 
        index: Union[pd.Index, np.ndarray],
        has_index: bool
    ) -> plt.Figure:
        """
        Create CUSUM stability test plot.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residual values
        index : pandas.Index or numpy.ndarray
            Index values for x-axis
        has_index : bool
            Whether a meaningful index is available
            
        Returns
        -------
        matplotlib.figure.Figure
            CUSUM plot figure
        """
        # Calculate CUSUM statistics
        cusum = np.cumsum(residuals - np.mean(residuals)) / np.std(residuals)
        
        # Calculate boundary lines
        n = len(residuals)
        alpha = 0.05  # 5% significance level
        
        # Critical values based on alpha
        if alpha == 0.01:
            critical_value = 1.63  # 99% confidence
        elif alpha == 0.05:
            critical_value = 1.36  # 95% confidence
        elif alpha == 0.10:
            critical_value = 1.22  # 90% confidence
        else:
            critical_value = 1.36  # Default to 95% confidence
        
        # Calculate boundary lines
        t = np.arange(1, n+1)
        bound = critical_value * np.sqrt(t)
        
        # Create plot
        fig, ax = create_figure(width=12, height=6)
        
        # Plot CUSUM
        if has_index:
            ax.plot(index, cusum, 'b-', label='CUSUM')
            ax.plot(index, bound, 'r--', label='Boundary (95% Confidence)')
            ax.plot(index, -bound, 'r--')
        else:
            ax.plot(cusum, 'b-', label='CUSUM')
            ax.plot(bound, 'r--', label='Boundary (95% Confidence)')
            ax.plot(-bound, 'r--')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Check if CUSUM crosses boundaries
        crosses_boundary = np.any(np.abs(cusum) > bound)
        
        # Add annotation with result
        if crosses_boundary:
            crossing_point = np.argmax(np.abs(cusum) > bound)
            
            add_annotations(ax, {
                (0.5, 0.02): "Boundary Crossed: Parameter Instability Detected"
            }, offset_x=0, offset_y=0, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="salmon", ec="red", alpha=0.8))
            
            # Mark crossing point
            if has_index:
                crossing_x = index[crossing_point]
            else:
                crossing_x = crossing_point
                
            crossing_y = cusum[crossing_point]
            ax.plot(crossing_x, crossing_y, 'ro', markersize=10)
            
            # Add annotation at crossing point
            if has_index:
                ax.annotate(
                    f"First Boundary Crossing",
                    xy=(crossing_x, crossing_y), 
                    xytext=(30, 30),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
                )
        else:
            add_annotations(ax, {
                (0.5, 0.02): "No Boundary Crossing: Parameters Stable"
            }, offset_x=0, offset_y=0, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.8))
        
        ax.set_title("CUSUM Stability Test")
        ax.set_xlabel("Time" if not has_index else "")
        ax.set_ylabel("CUSUM Statistic")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format date axis if available
        if has_index and pd.api.types.is_datetime64_any_dtype(index):
            format_date_axis(ax)
        
        plt.tight_layout()
        return fig
        
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
            Plot title
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
            title = "Residual Diagnostics"
        
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
    
    def _get_residuals(self, residuals=None):
        """Helper method to get residuals, either from input or stored attribute."""
        if residuals is not None:
            return residuals
        elif self.residuals is not None:
            return self.residuals
        else:
            raise ModelError("No residuals provided or stored in object")


class StabilityTesting:
    """
    Specialized component for testing parameter stability.
    
    Provides methods for testing parameter stability using rolling windows,
    recursive estimation, and CUSUM tests.
    """
    
    def __init__(self, data=None):
        """
        Initialize the stability tester.
        
        Parameters
        ----------
        data : array_like, optional
            Original data for stability testing
        """
        self.data = data
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def test_parameter_stability(
        self, data=None, window_size=None, step_size=10, 
        stability_threshold=0.5, model_func=None, 
        method="rolling", 
        visualization=True,
        max_chunk_size=10000
    ) -> Dict[str, Any]:
        """
        Test parameter stability using rolling estimation or recursive methods.
        
        Parameters
        ----------
        data : array_like, optional
            Time series data, uses self.data if not provided
        window_size : int, optional
            Size of the rolling window (default: 20% of sample)
        step_size : int, optional
            Step size for rolling window
        stability_threshold : float, optional
            Threshold for coefficient of variation to determine stability
        model_func : callable, optional
            Function to estimate model parameters
        method : str, optional
            Method for stability testing ('rolling', 'recursive', 'cusum')
        visualization : bool, optional
            Whether to create visualizations
        max_chunk_size : int, optional
            Maximum chunk size for memory optimization
            
        Returns
        -------
        dict
            Stability test results
        """
        # Get data for stability testing
        if data is None:
            data = self.data
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
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def test_cusum(self, data=None, model_func=None, significance=0.05, 
                 plot=True, save_path=None) -> Dict[str, Any]:
        """
        Perform CUSUM test for parameter stability.
        
        Parameters
        ----------
        data : array_like, optional
            Time series data, uses self.data if not provided
        model_func : callable, optional
            Function to estimate model parameters
        significance : float, optional
            Significance level for boundary calculation
        plot : bool, optional
            Whether to create diagnostic plot
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        dict
            CUSUM test results
        """
        # Get data for testing
        if data is None:
            data = self.data
            if data is None:
                raise ModelError("No data provided for CUSUM test")
        
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
            original_index = data.index if has_datetime_index else None
            data_arr = data.values
        else:
            data_arr = np.asarray(data)
            has_datetime_index = False
            original_index = None
        
        # If model_func not provided, create a simple OLS estimator
        if model_func is None:
            def model_func(data_window):
                # Use OLS with a simple AR(1) specification
                y = data_window[1:]
                X = sm.add_constant(data_window[:-1])
                return sm.OLS(y, X)
        
        # Fit the model to the full sample
        model = model_func(data_arr)
        
        # Calculate recursive residuals
        try:
            # Try to use statsmodels recursive_olsresiduals if it's an OLS model
            if hasattr(model, 'model') and isinstance(model.model, sm.regression.linear_model.OLS):
                rresid, rparams, rstderr = recursive_olsresiduals(model.model, return_params=True)
            else:
                # Manual calculation of recursive residuals
                rresid, rparams = self._calculate_recursive_residuals(data_arr, model)
        except Exception as e:
            logger.warning(f"Error calculating recursive residuals: {e}. Using manual calculation.")
            # Fallback to manual calculation
            rresid, rparams = self._calculate_recursive_residuals(data_arr, model)
        
        # Calculate CUSUM statistics
        cusum = np.cumsum(rresid) / np.std(rresid)
        
        # Calculate boundaries
        n = len(cusum)
        k = rparams.shape[1] if hasattr(rparams, 'shape') and len(rparams.shape) > 1 else 1
        
        # Critical value based on significance level
        if significance == 0.01:
            critical_value = 1.63  # 99% confidence
        elif significance == 0.05:
            critical_value = 1.36  # 95% confidence
        elif significance == 0.10:
            critical_value = 1.22  # 90% confidence
        else:
            critical_value = 1.36  # Default to 95%
        
        # Calculate boundary lines
        t = np.arange(k + 1, n + 1)
        bound = critical_value * np.sqrt(t)
        
        # Check if CUSUM crosses boundaries
        crosses_boundary = np.any(np.abs(cusum) > bound)
        
        # Create plot if requested
        fig = None
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot CUSUM
            if has_datetime_index and original_index is not None:
                # Use datetime index for x-axis
                plot_index = original_index[k:]
                ax.plot(plot_index, cusum, label='CUSUM')
                ax.plot(plot_index, bound, 'r--', label='Boundary')
                ax.plot(plot_index, -bound, 'r--')
            else:
                # Use numeric index
                ax.plot(t, cusum, label='CUSUM')
                ax.plot(t, bound, 'r--', label='Boundary')
                ax.plot(t, -bound, 'r--')
            
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_title('CUSUM Test for Parameter Stability')
            ax.set_xlabel('Time')
            ax.set_ylabel('CUSUM')
            ax.legend()
            
            # Save plot if requested
            if save_path is None and PLOT_DIR:
                save_path = os.path.join(PLOT_DIR, "cusum_test.png")
            
            if save_path:
                try:
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved CUSUM plot to {save_path}")
                except Exception as e:
                    logger.warning(f"Failed to save plot: {str(e)}")
        
        # Prepare results
        result = {
            'cusum': cusum,
            'bounds': bound,
            'times': t,
            'crosses_boundary': crosses_boundary,
            'max_cusum': np.max(np.abs(cusum)),
            'break_point': np.argmax(np.abs(cusum)) + k if crosses_boundary else None,
            'break_percentage': np.argmax(np.abs(cusum)) / len(cusum) if crosses_boundary else None,
            'first_crossing': np.argmax(np.abs(cusum) > bound) + k if crosses_boundary else None,
            'boundary_margin': np.max(np.abs(cusum) / bound) if len(bound) > 0 else None,
            'critical_value': critical_value,
            'max_bound': np.max(bound) if len(bound) > 0 else None,
            'n_obs': n,
            'k_params': k,
            'cross_details': {
                'up_cross': np.where(cusum > bound)[0].tolist() if crosses_boundary else [],
                'down_cross': np.where(cusum < -bound)[0].tolist() if crosses_boundary else [],
            },
            'interpretation': (
                "Parameter instability detected with CUSUM test. "
                f"The CUSUM statistic crossed the {100*(1-significance)}% confidence bounds, "
                f"suggesting a structural break around observation {np.argmax(np.abs(cusum) > bound) + k}."
            ) if crosses_boundary else (
                "No parameter instability detected with CUSUM test. "
                f"The CUSUM statistic remained within the {100*(1-significance)}% confidence bounds."
            ),
            'stable': not crosses_boundary,
            'significance': significance,
            'plot': fig
        }
        
        logger.info(f"CUSUM test: stable={not crosses_boundary}, significance={significance}")
        
        return result
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def detect_structural_breaks(
        self, 
        data: Union[pd.Series, np.ndarray], 
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_breaks: int = 5,
        min_size: int = 10,
        model: str = 'rbf'
    ) -> Dict[str, Any]:
        """
        Detect structural breaks in time series data.
        
        Parameters
        ----------
        data : array_like
            Time series data
        X : array_like, optional
            Independent variables if using regression model
        max_breaks : int, optional
            Maximum number of breaks to detect
        min_size : int, optional
            Minimum number of observations between breaks
        model : str, optional
            Model type ('rbf', 'linear', 'l1', 'l2', 'normal')
            
        Returns
        -------
        dict
            Structural break results
        """
        # Convert to numpy arrays
        if isinstance(data, pd.Series):
            has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
            index = data.index if has_datetime_index else None
            data_arr = data.values
        else:
            data_arr = np.asarray(data)
            has_datetime_index = False
            index = None
        
        # Detect breaks using ruptures
        algo = rpt.Pelt(model=model, min_size=min_size).fit(data_arr.reshape(-1, 1))
        break_indices = algo.predict(pen=2)
        
        # Prepare results
        results = {
            'break_indices': break_indices,
            'break_dates': [index[i] for i in break_indices] if has_datetime_index and index is not None else break_indices,
            'n_breaks': len(break_indices),
            'segments': len(break_indices) + 1
        }
        
        return results
    
    @handle_errors(logger=logger)
    def _calculate_recursive_residuals(self, data, model):
        """
        Calculate recursive residuals manually.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        model : object
            Model object with fit method
            
        Returns
        -------
        tuple
            (recursive_residuals, recursive_parameters)
        """
        # Extract model data
        if hasattr(model, 'model'):
            # Statsmodels model
            y = model.model.endog
            X = model.model.exog
        else:
            # Simple AR(1) model
            y = data[1:]
            X = sm.add_constant(data[:-1])
        
        n, k = X.shape
        
        # Initialize arrays for recursive residuals and parameters
        rresid = np.zeros(n - k)
        rparams = np.zeros((n - k, k))
        
        # Initial estimation with minimum sample
        params = np.linalg.lstsq(X[:k], y[:k], rcond=None)[0]
        
        # Recursive estimation
        for i in range(k, n):
            # Prediction for the next observation
            x_i = X[i]
            pred_i = np.dot(x_i, params)
            
            # Calculate recursive residual
            sigma2 = 1 + np.dot(x_i, np.linalg.lstsq(X[:i], x_i, rcond=None)[0])
            rresid[i - k] = (y[i] - pred_i) / np.sqrt(sigma2)
            
            # Update parameters
            rparams[i - k] = params
            
            # Re-estimate with one more observation
            params = np.linalg.lstsq(X[:i+1], y[:i+1], rcond=None)[0]
        
        return rresid, rparams
    
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


class SpatialDiagnostics:
    """
    Specialized component for spatial econometric diagnostics.
    
    Provides methods for testing spatial autocorrelation, spatial heterogeneity,
    and other spatial properties of data.
    """
    
    def __init__(self):
        """Initialize the spatial diagnostics component."""
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
        # Store weights matrix for reuse
        self.weights_matrix = None
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    @m1_optimized(parallel=True)
    def test_spatial_autocorrelation(
        self, 
        data, 
        weights_matrix=None, 
        variables=None, 
        method='both', 
        permutations=999, 
        significance_level=0.05,
        sparse_threshold=1000,
        lisa=False
    ) -> Dict[str, Any]:
        """
        Test for spatial autocorrelation using multiple statistics.
        
        This method performs spatial autocorrelation analysis using both Moran's I
        and Geary's C statistics. For large datasets, it uses sparse matrix
        optimizations to improve computational efficiency. Statistical significance
        is determined through permutation tests.
        
        Parameters
        ----------
        data : array_like
            Spatial data to test (GeoDataFrame or DataFrame with geometry)
        weights_matrix : libpysal.weights or similar, optional
            Spatial weights matrix (will be created if not provided)
        variables : list, optional
            Variables to test if data is a DataFrame
        method : str, optional
            Test method ('moran_i', 'geary_c', 'both')
        permutations : int, optional
            Number of permutations for statistical inference
        significance_level : float, optional
            Significance level for hypothesis testing (default: 0.05)
        sparse_threshold : int, optional
            Threshold for using sparse matrix optimization (default: 1000)
        lisa : bool, optional
            Whether to compute Local Indicators of Spatial Association (default: False)
            
        Returns
        -------
        dict
            Spatial autocorrelation test results with test statistics,
            p-values, and interpretations for each variable tested
        """
        # Import required libraries
        try:
            from libpysal.weights import W
            from esda.moran import Moran, Moran_Local
            from esda.geary import Geary
        except ImportError as e:
            logger.error(f"Could not import spatial analysis libraries: {e}")
            return {
                'error': f"Missing required libraries: {e}",
                'note': "Install libpysal and esda for spatial analysis"
            }
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Validate inputs
        if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError("data must be a DataFrame, Series, or ndarray")
        
        # Set up weights matrix
        if weights_matrix is None:
            try:
                if self.weights_matrix is not None:
                    # Reuse cached weights matrix if available
                    weights_matrix = self.weights_matrix
                    logger.info("Using cached spatial weights matrix")
                elif isinstance(data, pd.DataFrame) and hasattr(data, 'geometry'):
                    # GeoDataFrame - create weights matrix
                    logger.info("Creating spatial weights matrix from GeoDataFrame")
                    weights_matrix = create_spatial_weight_matrix(data, method='knn', k=5)
                    self.weights_matrix = weights_matrix  # Cache for reuse
                else:
                    raise ValueError("Cannot create weights matrix from non-spatial data. Please provide a weights_matrix.")
            except Exception as e:
                logger.warning(f"Could not create weights matrix: {e}")
                return {
                    'error': f"Could not create weights matrix: {e}",
                    'note': "Provide a weights matrix or use a GeoDataFrame with geometries"
                }
        else:
            # Store provided weights matrix for future use
            self.weights_matrix = weights_matrix
        
        # Determine variables to test
        if variables is None:
            if isinstance(data, pd.DataFrame):
                # Try to find suitable numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if 'price' in numeric_cols:
                    variables = ['price']  # Prioritize price variable if available
                elif any(col for col in numeric_cols if 'price' in col.lower()):
                    # Find columns with 'price' in their name
                    price_cols = [col for col in numeric_cols if 'price' in col.lower()]
                    variables = price_cols[:3]  # Take up to 3 price-related columns
                elif numeric_cols:
                    # Use first few numeric columns (up to 3)
                    variables = numeric_cols[:3]
                else:
                    raise ValueError("No numeric columns found in DataFrame")
            else:
                # For Series or ndarray, use the data directly
                variables = ['value']
                if isinstance(data, pd.Series):
                    data = pd.DataFrame({'value': data})
                else:
                    data = pd.DataFrame({'value': data.flatten()})
        
        # Initialize results dictionary
        results = {
            'variables': variables,
            'method': method,
            'permutations': permutations,
            'significance_level': significance_level,
            'moran_i': {},       # Will hold Moran's I results
            'geary_c': {},       # Will hold Geary's C results
            'lisa': {},          # Will hold LISA results if requested
            'summary': {}        # Overall summary
        }
        
        # Process each variable
        for var in variables:
            # Extract values
            if var in data.columns:
                values = data[var].values
            else:
                raise ValueError(f"Variable {var} not found in data")
            
            # Check for missing values
            if np.any(np.isnan(values)):
                logger.warning(f"Variable {var} contains missing values, removing them")
                valid_mask = ~np.isnan(values)
                values = values[valid_mask]
                # Adjust weights matrix if needed
                if hasattr(weights_matrix, 'subset'):
                    weights_matrix = weights_matrix.subset(valid_mask)
            
            # Determine if we should use sparse matrices based on data size
            use_sparse = len(values) > sparse_threshold
            if use_sparse:
                logger.info(f"Using sparse matrix optimization for {var} (n={len(values)})")
            
            # Run Moran's I test
            if method.lower() in ['moran_i', 'both']:
                try:
                    # For large datasets, use sparse matrix optimization
                    if use_sparse:
                        # Ensure weights matrix is in sparse format
                        if not hasattr(weights_matrix, 'sparse') or not isinstance(weights_matrix.sparse, sparse.spmatrix):
                            weights_matrix.transform = 'r'  # Row-standardize
                            w_sparse = weights_matrix.sparse
                        else:
                            w_sparse = weights_matrix.sparse
                        
                        # Efficient Moran's I calculation with sparse matrices
                        z = values - values.mean()
                        z_std = z / (z.std() or 1.0)  # Avoid division by zero
                        
                        # Calculate numerator: z'Wz
                        numerator = float(z_std @ (w_sparse @ z_std))
                        
                        # Calculate denominator: z'z
                        denominator = float(z_std @ z_std)
                        
                        # Calculate Moran's I statistic
                        I = numerator / (denominator or 1.0)  # Avoid division by zero
                        
                        # Expected value
                        EI = -1.0 / (len(values) - 1)
                        
                        # Variance calculation for normal approximation
                        s0 = w_sparse.sum()
                        s1 = float(((w_sparse + w_sparse.T) ** 2).sum() / 2)
                        
                        # Calculate row and column sums
                        row_sums = w_sparse.sum(axis=1).A1
                        col_sums = w_sparse.sum(axis=0).A1
                        s2 = float((row_sums + col_sums).T @ (row_sums + col_sums))
                        
                        # Calculate variance
                        n = len(values)
                        var_norm = (n * s1 - n * s2 + 3 * s0**2) / (s0**2 * (n**2 - 1))
                        
                        # Z-score and p-value
                        z_norm = (I - EI) / np.sqrt(var_norm)
                        p_norm = 2 * (1 - stats.norm.cdf(abs(z_norm)))
                        
                        # Permutation test if requested
                        p_sim = None
                        if permutations > 0:
                            # Perform permutation test in batches for memory efficiency
                            sim_vals = []
                            batch_size = min(100, permutations)
                            remaining = permutations
                            
                            while remaining > 0:
                                # Process in batches
                                batch = min(batch_size, remaining)
                                sim_batch = []
                                
                                for _ in range(batch):
                                    # Permute values
                                    perm_values = np.random.permutation(values)
                                    perm_z = perm_values - perm_values.mean()
                                    perm_z_std = perm_z / (perm_z.std() or 1.0)
                                    
                                    # Calculate Moran's I for permuted values
                                    perm_num = float(perm_z_std @ (w_sparse @ perm_z_std))
                                    perm_denom = float(perm_z_std @ perm_z_std) or 1.0
                                    sim_batch.append(perm_num / perm_denom)
                                
                                sim_vals.extend(sim_batch)
                                remaining -= batch
                            
                            # Calculate p-value from permutations
                            larger = sum(1 for sim_val in sim_vals if sim_val >= I)
                            p_sim = (larger + 1) / (permutations + 1)
                        
                        # Store comprehensive results
                        moran_result = {
                            'I': I,
                            'expected_I': EI,
                            'z_norm': z_norm,
                            'p_norm': p_norm,
                            'p_sim': p_sim,
                            'var_norm': var_norm,
                            'significant': p_norm < significance_level,
                            'positive_autocorrelation': I > EI and p_norm < significance_level,
                            'method': 'sparse_matrix',
                            'interpretation': self._interpret_moran_i(I, p_norm, EI)
                        }
                        
                    else:
                        # For smaller datasets, use standard implementation
                        moran = Moran(values, weights_matrix, permutations=permutations)
                        
                        moran_result = {
                            'I': moran.I,
                            'expected_I': moran.EI,
                            'p_norm': moran.p_norm,
                            'p_sim': moran.p_sim if permutations > 0 else None,
                            'z_norm': moran.z_norm,
                            'significant': moran.p_norm < significance_level,
                            'positive_autocorrelation': moran.I > moran.EI and moran.p_norm < significance_level,
                            'method': 'standard',
                            'interpretation': self._interpret_moran_i(moran.I, moran.p_norm, moran.EI)
                        }
                    
                    # Add results to the dictionary
                    results['moran_i'][var] = moran_result
                    logger.info(f"Moran's I for {var}: I={moran_result['I']:.4f}, p={moran_result['p_norm']:.4f}, significant={moran_result['significant']}")
                    
                except Exception as e:
                    logger.warning(f"Error calculating Moran's I for {var}: {str(e)}")
                    results['moran_i'][var] = {'error': str(e)}
            
            # Run Geary's C test
            if method.lower() in ['geary_c', 'both']:
                try:
                    # For large datasets, use sparse matrix optimization
                    if use_sparse:
                        # Ensure weights matrix is in sparse format
                        if not hasattr(weights_matrix, 'sparse') or not isinstance(weights_matrix.sparse, sparse.spmatrix):
                            weights_matrix.transform = 'r'  # Row-standardize
                            w_sparse = weights_matrix.sparse
                        else:
                            w_sparse = weights_matrix.sparse
                        
                        # Efficient Geary's C calculation with sparse matrices
                        n = len(values)
                        y_mean = values.mean()
                        s0 = w_sparse.sum()
                        
                        # This is the most computationally intensive part, using a vectorized approach
                        y_diff_sq = np.zeros(w_sparse.nnz)
                        
                        # Get indices and data from sparse matrix
                        w_i, w_j = w_sparse.nonzero()
                        w_data = np.array(w_sparse[w_i, w_j]).flatten()
                        
                        # Calculate squared differences weighted by the spatial weights
                        y_diff_sq = (values[w_i] - values[w_j])**2 * w_data
                        
                        # Sum to get numerator
                        numerator = y_diff_sq.sum()
                        
                        # Calculate denominator
                        denominator = 2 * s0 * ((values - y_mean)**2).sum() or 1.0  # Avoid division by zero
                        
                        # Calculate Geary's C statistic
                        C = (n - 1) * numerator / denominator
                        
                        # Expected value
                        EC = 1.0
                        
                        # Calculate variance for normal approximation
                        var_norm = (1 / (2 * (n + 1) * s0**2)) * ((n - 1) * s0 * (n**2 - 3*n + 3 - (n-1) * C)) or 1.0
                        
                        # Z-score and p-value
                        z_norm = (C - EC) / np.sqrt(var_norm)
                        p_norm = 2 * (1 - stats.norm.cdf(abs(z_norm)))
                        
                        # Permutation test if requested
                        p_sim = None
                        if permutations > 0:
                            # Perform permutation test in batches for memory efficiency
                            sim_vals = []
                            batch_size = min(100, permutations)
                            remaining = permutations
                            
                            while remaining > 0:
                                # Process in batches
                                batch = min(batch_size, remaining)
                                sim_batch = []
                                
                                for _ in range(batch):
                                    # Permute values
                                    perm_values = np.random.permutation(values)
                                    perm_mean = perm_values.mean()
                                    
                                    # Calculate squared differences for permuted values
                                    perm_diff_sq = (perm_values[w_i] - perm_values[w_j])**2 * w_data
                                    perm_num = perm_diff_sq.sum()
                                    perm_denom = 2 * s0 * ((perm_values - perm_mean)**2).sum() or 1.0
                                    sim_batch.append((n - 1) * perm_num / perm_denom)
                                
                                sim_vals.extend(sim_batch)
                                remaining -= batch
                            
                            # Calculate p-value from permutations
                            larger = sum(1 for sim_val in sim_vals if sim_val <= C)  # Note: for Geary's C, lower values indicate clustering
                            p_sim = (larger + 1) / (permutations + 1)
                        
                        # Store comprehensive results
                        geary_result = {
                            'C': C,
                            'expected_C': EC,
                            'z_norm': z_norm,
                            'p_norm': p_norm,
                            'p_sim': p_sim,
                            'var_norm': var_norm,
                            'significant': p_norm < significance_level,
                            'positive_autocorrelation': C < EC and p_norm < significance_level,
                            'method': 'sparse_matrix',
                            'interpretation': self._interpret_geary_c(C, p_norm, EC)
                        }
                        
                    else:
                        # For smaller datasets, use standard implementation
                        geary = Geary(values, weights_matrix, permutations=permutations)
                        
                        geary_result = {
                            'C': geary.C,
                            'expected_C': geary.EC,
                            'p_norm': geary.p_norm,
                            'p_sim': geary.p_sim if permutations > 0 else None,
                            'z_norm': geary.z_norm,
                            'significant': geary.p_norm < significance_level,
                            'positive_autocorrelation': geary.C < geary.EC and geary.p_norm < significance_level,
                            'method': 'standard',
                            'interpretation': self._interpret_geary_c(geary.C, geary.p_norm, geary.EC)
                        }
                    
                    # Add results to the dictionary
                    results['geary_c'][var] = geary_result
                    logger.info(f"Geary's C for {var}: C={geary_result['C']:.4f}, p={geary_result['p_norm']:.4f}, significant={geary_result['significant']}")
                    
                except Exception as e:
                    logger.warning(f"Error calculating Geary's C for {var}: {str(e)}")
                    results['geary_c'][var] = {'error': str(e)}
            
            # Run LISA (Local Indicators of Spatial Association) analysis if requested
            if lisa:
                try:
                    # Local Moran's I
                    local_moran = Moran_Local(values, weights_matrix, permutations=permutations)
                    
                    # Classify clusters
                    sig = local_moran.p_sim < significance_level
                    hotspot = sig & (local_moran.q == 1)
                    coldspot = sig & (local_moran.q == 3)
                    doughnut = sig & (local_moran.q == 2)
                    diamond = sig & (local_moran.q == 4)
                    
                    # Store LISA results
                    lisa_result = {
                        'Is': local_moran.Is,
                        'p_sim': local_moran.p_sim,
                        'significant': sig,
                        'hotspots': hotspot,
                        'coldspots': coldspot,
                        'doughnuts': doughnut,
                        'diamonds': diamond,
                        'clusters': local_moran.q,
                        'cluster_summary': {
                            'hotspots_count': np.sum(hotspot),
                            'coldspots_count': np.sum(coldspot),
                            'doughnuts_count': np.sum(doughnut),
                            'diamonds_count': np.sum(diamond)
                        }
                    }
                    
                    # Add results to the dictionary
                    results['lisa'][var] = lisa_result
                    logger.info(f"LISA analysis for {var}: significant clusters={np.sum(sig)}")
                    
                except Exception as e:
                    logger.warning(f"Error calculating LISA for {var}: {str(e)}")
                    results['lisa'][var] = {'error': str(e)}
        
        # Create overall assessment
        has_spatial_autocorrelation = False
        autocorrelation_vars = []
        
        # Analyze results from both methods
        if method.lower() in ['moran_i', 'both']:
            for var, result in results['moran_i'].items():
                if isinstance(result, dict) and result.get('significant', False) and 'error' not in result:
                    has_spatial_autocorrelation = True
                    autocorrelation_vars.append({
                        'variable': var,
                        'statistic': 'Moran\'s I',
                        'value': result.get('I'),
                        'p_value': result.get('p_norm'),
                        'interpretation': result.get('interpretation')
                    })
        
        if method.lower() in ['geary_c', 'both']:
            for var, result in results['geary_c'].items():
                if isinstance(result, dict) and result.get('significant', False) and 'error' not in result:
                    has_spatial_autocorrelation = True
                    autocorrelation_vars.append({
                        'variable': var,
                        'statistic': 'Geary\'s C',
                        'value': result.get('C'),
                        'p_value': result.get('p_norm'),
                        'interpretation': result.get('interpretation')
                    })
        
        # Add summary information to results
        results['has_spatial_autocorrelation'] = has_spatial_autocorrelation
        results['summary'] = {
            'has_spatial_autocorrelation': has_spatial_autocorrelation,
            'significant_variables': autocorrelation_vars,
            'total_variables_tested': len(variables),
            'spatial_clusters_detected': lisa and any('lisa' in results and 
                                                     any(result.get('significant', np.array([])).any() 
                                                         for result in results['lisa'].values() 
                                                         if isinstance(result, dict) and 'error' not in result))
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        logger.info(f"Spatial autocorrelation analysis complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return results
    
    def _interpret_moran_i(self, I: float, p_value: float, expected_I: float = -1.0) -> str:
        """
        Generate interpretation of Moran's I statistic.
        
        Parameters
        ----------
        I : float
            Moran's I value
        p_value : float
            P-value for significance test
        expected_I : float
            Expected value of I under null hypothesis
            
        Returns
        -------
        str
            Interpretation of the results
        """
        if p_value >= 0.05:
            return "No significant spatial autocorrelation detected (random spatial pattern)"
        
        if I > expected_I:
            strength = "strong" if I > 0.5 else "moderate" if I > 0.3 else "weak"
            return f"Significant positive spatial autocorrelation ({strength}). Similar values tend to cluster together spatially."
        else:
            strength = "strong" if I < -0.5 else "moderate" if I < -0.3 else "weak"
            return f"Significant negative spatial autocorrelation ({strength}). Dissimilar values tend to be near each other (checkerboard pattern)."
    
    def _interpret_geary_c(self, C: float, p_value: float, expected_C: float = 1.0) -> str:
        """
        Generate interpretation of Geary's C statistic.
        
        Parameters
        ----------
        C : float
            Geary's C value
        p_value : float
            P-value for significance test
        expected_C : float
            Expected value of C under null hypothesis
            
        Returns
        -------
        str
            Interpretation of the results
        """
        if p_value >= 0.05:
            return "No significant spatial autocorrelation detected (random spatial pattern)"
        
        if C < expected_C:
            strength = "strong" if C < 0.5 else "moderate" if C < 0.7 else "weak"
            return f"Significant positive spatial autocorrelation ({strength}). Similar values tend to cluster together spatially."
        else:
            strength = "strong" if C > 1.5 else "moderate" if C > 1.3 else "weak"
            return f"Significant negative spatial autocorrelation ({strength}). Dissimilar values tend to be near each other (checkerboard pattern)."


class PredictionAnalysis:
    """
    Specialized component for prediction analysis and intervals.
    
    Provides methods for computing prediction intervals, forecast evaluation,
    and other prediction-related diagnostics.
    """
    
    def __init__(self):
        """Initialize the prediction analysis component."""
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
        
        # Set default parameters from config
        self.default_bootstrap_samples = config.get('analysis.diagnostics.bootstrap_samples', 1000)
        self.default_batch_size = config.get('analysis.diagnostics.batch_size', 100)
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    @m1_optimized(parallel=True)
    def compute_prediction_intervals(
        self, 
        model_func: Callable, 
        data: Union[pd.DataFrame, np.ndarray],
        pred_x: Union[pd.DataFrame, np.ndarray],
        method: str = 'bootstrap',
        conf_level: float = 0.95,
        n_bootstrap: int = 1000,
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        convergence_threshold: Optional[float] = 0.001,
        check_convergence_interval: int = 50
    ) -> Dict[str, Any]:
        """
        Compute prediction intervals using analytical or bootstrap methods.
        
        Parameters
        ----------
        model_func : callable
            Function to fit model on bootstrap samples
        data : array_like
            Original data for model estimation
        pred_x : array_like
            Predictor values for predictions
        method : str, optional
            Method for interval calculation ('analytical', 'bootstrap')
        conf_level : float, optional
            Confidence level (0.9, 0.95, 0.99)
        n_bootstrap : int, optional
            Number of bootstrap samples
        batch_size : int, optional
            Batch size for parallel processing
        max_workers : int, optional
            Number of worker processes
        convergence_threshold : float, optional
            Threshold for early stopping when convergence is detected
        check_convergence_interval : int, optional
            Interval for checking convergence
            
        Returns
        -------
        dict
            Prediction intervals and metadata
        """
        # Set default max_workers if not provided
        if max_workers is None:
            max_workers = self.n_workers
        
        # Convert to numpy arrays if pandas objects
        if isinstance(data, pd.DataFrame):
            data_arr = data.values
        else:
            data_arr = np.asarray(data)
        
        if isinstance(pred_x, pd.DataFrame):
            pred_x_arr = pred_x.values
        else:
            pred_x_arr = np.asarray(pred_x)
        
        # Calculate alpha from confidence level
        alpha = 1 - conf_level
        
        # Fit model to original data and get predictions
        model = model_func(data_arr)
        predictions = model.predict(pred_x_arr)
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Choose method for interval calculation
        result = None
        
        if method == 'analytical':
            # Analytical method (if model supports it)
            try:
                # Check if model has get_prediction method (statsmodels style)
                if hasattr(model, 'get_prediction'):
                    pred = model.get_prediction(pred_x_arr)
                    pred_int = pred.conf_int(alpha=alpha)
                    
                    # Extract lower and upper bounds
                    lower_bound = pred_int[:, 0]
                    upper_bound = pred_int[:, 1]
                    
                    # Store additional information if available
                    if hasattr(pred, 'summary_frame'):
                        summary = pred.summary_frame()
                        std_err = summary['mean_se'].values if 'mean_se' in summary else None
                    else:
                        std_err = None
                    
                    result = {
                        'predictions': predictions,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'std_err': std_err,
                        'conf_level': conf_level,
                        'method': 'analytical'
                    }
                    
                # Check if model has predict_interval method (sklearn style)
                elif hasattr(model, 'predict_interval'):
                    lower_bound, upper_bound = model.predict_interval(pred_x_arr, alpha=alpha)
                    
                    result = {
                        'predictions': predictions,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'conf_level': conf_level,
                        'method': 'analytical'
                    }
                    
                # Check if model has conf_int method (custom style)
                elif hasattr(model, 'conf_int'):
                    pred_int = model.conf_int(pred_x_arr, alpha=alpha)
                    
                    # Extract lower and upper bounds
                    lower_bound = pred_int[:, 0]
                    upper_bound = pred_int[:, 1]
                    
                    result = {
                        'predictions': predictions,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'conf_level': conf_level,
                        'method': 'analytical'
                    }
                    
                else:
                    # Fallback to bootstrap method
                    logger.warning("Model does not support analytical intervals, falling back to bootstrap")
                    method = 'bootstrap'
                    
            except Exception as e:
                logger.warning(f"Error calculating analytical intervals: {e}. Falling back to bootstrap")
                method = 'bootstrap'
        
        # Bootstrap method
        if method == 'bootstrap':
            # Initialize arrays to store bootstrap results
            bootstrap_predictions = []
            previous_bounds = None
            converged = False
            
            # Prepare batches
            total_batches = (n_bootstrap + batch_size - 1) // batch_size
            batch_indices = [(i, min(i + batch_size, n_bootstrap)) for i in range(0, n_bootstrap, batch_size)]
            
            # Track memory and progress
            process = psutil.Process(os.getpid())
            start_time = time.time()
            
            # Process bootstrap in parallel batches
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                logger.info(f"Starting bootstrap with {total_batches} batches (batch_size={batch_size}, n_bootstrap={n_bootstrap})")
                
                # Submit initial batches
                running_futures = {}
                submitted_count = 0
                
                # Submit first set of batches
                initial_batch_count = min(max_workers * 2, len(batch_indices))
                for batch_idx in range(initial_batch_count):
                    start_idx, end_idx = batch_indices[batch_idx]
                    n_samples = end_idx - start_idx
                    
                    future = executor.submit(
                        self._run_prediction_bootstrap_batch,
                        data_arr, pred_x_arr, model_func, len(data_arr), n_samples, batch_idx
                    )
                    running_futures[future] = batch_idx
                    submitted_count += 1
                
                # Process results as they complete and submit new batches
                completed_count = 0
                completed_samples = 0
                
                while running_futures:
                    # Wait for a batch to complete
                    done, running_futures = wait(running_futures, return_when=FIRST_COMPLETED)
                    
                    # Process completed batches
                    for future in done:
                        batch_idx = running_futures.pop(future)
                        
                        try:
                            # Get batch results
                            batch_preds = future.result()
                            bootstrap_predictions.extend(batch_preds)
                            completed_samples += len(batch_preds)
                            
                            # Log progress periodically
                            completed_count += 1
                            if completed_count % 5 == 0 or completed_count == total_batches:
                                elapsed = time.time() - start_time
                                memory_usage = process.memory_info().rss / (1024 * 1024)
                                logger.info(f"Bootstrap progress: {completed_count}/{total_batches} batches "
                                          f"({completed_samples}/{n_bootstrap} samples), "
                                          f"elapsed: {elapsed:.1f}s, memory: {memory_usage:.1f} MB")
                            
                            # Check for convergence periodically
                            if (convergence_threshold is not None and 
                                len(bootstrap_predictions) % check_convergence_interval == 0 and
                                len(bootstrap_predictions) >= check_convergence_interval * 2):
                                
                                # Calculate current bounds
                                current_array = np.array(bootstrap_predictions)
                                current_lower = np.percentile(current_array, (1 - conf_level) / 2 * 100, axis=0)
                                current_upper = np.percentile(current_array, (1 + conf_level) / 2 * 100, axis=0)
                                
                                # Check convergence if we have previous bounds
                                if previous_bounds is not None:
                                    prev_lower, prev_upper = previous_bounds
                                    
                                    # Calculate relative change in bounds
                                    lower_change = np.mean(np.abs(current_lower - prev_lower) / (np.abs(prev_lower) + 1e-10))
                                    upper_change = np.mean(np.abs(current_upper - prev_upper) / (np.abs(prev_upper) + 1e-10))
                                    max_change = max(lower_change, upper_change)
                                    
                                    logger.debug(f"Bootstrap convergence check: change={max_change:.6f} (threshold={convergence_threshold})")
                                    
                                    if max_change < convergence_threshold:
                                        logger.info(f"Bootstrap converged after {len(bootstrap_predictions)} samples with change={max_change:.6f}")
                                        converged = True
                                        break
                                
                                # Store current bounds for next check
                                previous_bounds = (current_lower, current_upper)
                        
                        except Exception as e:
                            logger.warning(f"Error in bootstrap batch {batch_idx}: {str(e)}")
                        
                        # Submit a new batch if we haven't reached the end and not converged
                        if submitted_count < len(batch_indices) and not converged:
                            start_idx, end_idx = batch_indices[submitted_count]
                            n_samples = end_idx - start_idx
                            
                            future = executor.submit(
                                self._run_prediction_bootstrap_batch,
                                data_arr, pred_x_arr, model_func, len(data_arr), n_samples, submitted_count
                            )
                            running_futures[future] = submitted_count
                            submitted_count += 1
                    
                    # Stop if converged
                    if converged:
                        # Cancel any remaining futures
                        for future in running_futures:
                            future.cancel()
                        break
            
            # Check if we have enough bootstrap samples
            if len(bootstrap_predictions) < n_bootstrap * 0.5 and not converged:
                logger.warning(f"Only {len(bootstrap_predictions)} of {n_bootstrap} bootstrap samples succeeded")
            
            # Calculate confidence intervals from bootstrap samples
            bootstrap_predictions_array = np.array(bootstrap_predictions)
            
            # Calculate bounds using memory-efficient approach for large arrays
            alpha = 1 - conf_level
            lower_bound = np.percentile(bootstrap_predictions_array, alpha/2 * 100, axis=0)
            upper_bound = np.percentile(bootstrap_predictions_array, (1-alpha/2) * 100, axis=0)
            
            result = {
                'predictions': predictions,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': upper_bound - lower_bound,
                'n_bootstrap_samples': len(bootstrap_predictions),
                'requested_samples': n_bootstrap,
                'converged': converged,
                'alpha': alpha,
                'conf_level': conf_level,
                'method': 'bootstrap'
            }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        logger.info(f"Prediction intervals calculated using {method} method. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()

        return result
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    @m1_optimized(parallel=True)
    def bootstrap_confidence_intervals(
        self,
        data: Union[pd.Series, np.ndarray],
        statistic_func: Callable,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        method: str = 'percentile',
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        convergence_threshold: Optional[float] = 0.001,
        check_interval: int = 50,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals with optimizations.
        
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
            Method for computing intervals ('percentile', 'bca')
        batch_size : int, optional
            Size of batches for parallel processing
        max_workers : int, optional
            Number of worker processes (defaults to CPU count - 1)
        convergence_threshold : float, optional
            Early stopping threshold for bootstrap convergence
        check_interval : int, optional
            Interval for checking convergence
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Bootstrap results with confidence intervals
        """
        # Set default max_workers if not provided
        if max_workers is None:
            max_workers = self.n_workers
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            data_arr = data.values
        else:
            data_arr = np.asarray(data)
        
        n = len(data_arr)
        
        # Calculate the observed statistic
        observed_stat = statistic_func(data_arr)
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Generate bootstrap samples in parallel
        bootstrap_stats = []
        
        # Prepare batches
        total_batches = (n_bootstrap + batch_size - 1) // batch_size
        batch_indices = [(i, min(i + batch_size, n_bootstrap)) for i in range(0, n_bootstrap, batch_size)]
        
        # Process bootstrap in parallel batches
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit bootstrap tasks
            futures = [
                executor.submit(
                    self._run_bootstrap_batch,
                    data_arr, n, statistic_func, batch_size, batch_idx
                )
                for batch_idx, (_, _) in enumerate(batch_indices)
            ]
            
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
            # Calculate bias correction factor
            z0 = stats.norm.ppf(np.mean(np.array(bootstrap_stats) < observed_stat))
            
            # Calculate acceleration factor using jackknife
            jackknife_stats = []
            
            # Process jackknife in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Split indices into chunks for parallel processing
                chunk_size = max(10, n // max_workers)
                chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
                
                # Submit jackknife tasks
                futures = [
                    executor.submit(
                        self._run_jackknife_batch,
                        data_arr, statistic_func, list(range(start, end))
                    )
                    for start, end in chunks
                ]
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        batch_stats = future.result()
                        jackknife_stats.extend(batch_stats)
                    except Exception as e:
                        logger.warning(f"Error in jackknife batch: {e}")
            
            # Calculate acceleration factor
            jackknife_mean = np.mean(jackknife_stats)
            numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
            denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
            
            # Avoid division by zero
            if denominator != 0:
                a = numerator / denominator
            else:
                a = 0
            
            # Calculate adjusted percentiles
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
            'bootstrap_statistics': bootstrap_stats,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'conf_interval': (lower_bound, upper_bound),
            'conf_level': 1 - alpha,
            'n_bootstrap': len(bootstrap_stats),
            'method': method
        }
    
    @handle_errors(logger=logger)
    def _run_prediction_bootstrap_batch(
        self,
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
    
    @handle_errors(logger=logger)
    def _run_bootstrap_batch(
        self,
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
            try:
                # Draw random sample with replacement
                indices = np.random.randint(0, n, size=n)
                sample = data[indices]
                
                # Calculate statistic
                stat = statistic_func(sample)
                batch_stats.append(stat)
            except Exception as e:
                logger.debug(f"Error in bootstrap iteration {batch_idx * n_samples + i}: {str(e)}")
        
        return batch_stats
    
    @handle_errors(logger=logger)
    def _run_jackknife_batch(
        self,
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
            try:
                # Create leave-one-out sample
                sample = np.delete(data, i)
                
                # Calculate statistic
                stat = statistic_func(sample)
                batch_stats.append(stat)
            except Exception as e:
                logger.debug(f"Error in jackknife sample {i}: {str(e)}")
        
        return batch_stats


class ModelDiagnostics:
    """
    Comprehensive model diagnostics class for econometric analysis.
    
    This class serves as a facade for specialized diagnostic components:
    - ResidualsAnalysis: For analyzing model residuals
    - StabilityTesting: For parameter stability testing
    - SpatialDiagnostics: For spatial autocorrelation and clustering
    - PredictionAnalysis: For prediction intervals and forecasting
    
    It provides a unified interface for conducting diagnostic tests
    on econometric models with performance optimizations for handling
    large spatial and time series datasets.
    """
    
    def __init__(self, residuals=None, data=None, model=None, model_name="Model"):
        """
        Initialize the model diagnostics suite.
        
        Parameters
        ----------
        residuals : array_like, optional
            Model residuals for testing
        data : array_like, optional
            Original data for model estimation
        model : object, optional
            Fitted model object
        model_name : str, optional
            Name of the model for logging and plot titles
        """
        self.residuals = residuals
        self.data = data
        self.model = model
        self.model_name = model_name
        
        # Initialize specialized components
        self.residuals_analyzer = ResidualsAnalysis(residuals)
        self.stability_tester = StabilityTesting(data)
        self.spatial_diagnostics = SpatialDiagnostics()
        self.prediction_analyzer = PredictionAnalysis()
        
        # Configure performance settings
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
    
    # Factory methods for specialized diagnostics
    @classmethod
    def create_residuals_analyzer(cls, residuals):
        """Create a specialized residuals analyzer."""
        return ResidualsAnalysis(residuals)
    
    @classmethod
    def create_stability_tester(cls, data):
        """Create a specialized stability tester."""
        return StabilityTesting(data)
    
    @classmethod
    def create_spatial_diagnostics(cls):
        """Create a specialized spatial diagnostics component."""
        return SpatialDiagnostics()
    
    @classmethod
    def create_prediction_analyzer(cls):
        """Create a specialized prediction analyzer."""
        return PredictionAnalysis()
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def detect_structural_breaks(
        self, 
        data: Union[pd.Series, np.ndarray] = None, 
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        max_breaks: int = 5,
        min_size: int = 10,
        model: str = 'rbf'
    ) -> Dict[str, Any]:
        """
        Detect structural breaks in time series data.
        
        Parameters
        ----------
        data : array_like, optional
            Time series data, uses self.data if not provided
        X : array_like, optional
            Independent variables if using regression model
        max_breaks : int, optional
            Maximum number of breaks to detect
        min_size : int, optional
            Minimum number of observations between breaks
        model : str, optional
            Model type ('rbf', 'linear', 'l1', 'l2', 'normal')
            
        Returns
        -------
        dict
            Structural break results
        """
        if data is None:
            data = self.data
            if data is None:
                raise ModelError("No data provided for structural break detection")
        
        return self.stability_tester.detect_structural_breaks(
            data=data,
            X=X,
            max_breaks=max_breaks,
            min_size=min_size,
            model=model
        )
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def test_spatial_autocorrelation(
        self, 
        data=None, 
        weights_matrix=None, 
        variables=None, 
        method='both', 
        permutations=999, 
        significance_level=0.05,
        sparse_threshold=1000
    ) -> Dict[str, Any]:
        """
        Test for spatial autocorrelation using multiple statistics.
        
        Parameters
        ----------
        data : array_like, optional
            Spatial data to test, uses self.data if not provided
        weights_matrix : libpysal.weights or similar, optional
            Spatial weights matrix (will be created if not provided)
        variables : list, optional
            Variables to test if data is a DataFrame
        method : str, optional
            Test method ('moran_i', 'geary_c', 'both')
        permutations : int, optional
            Number of permutations for statistical inference
        significance_level : float, optional
            Significance level for hypothesis testing
        sparse_threshold : int, optional
            Threshold for using sparse matrix optimization
            
        Returns
        -------
        dict
            Spatial autocorrelation test results
        """
        if data is None:
            data = self.data
            if data is None:
                raise ModelError("No data provided for spatial autocorrelation test")
        
        return self.spatial_diagnostics.test_spatial_autocorrelation(
            data=data,
            weights_matrix=weights_matrix,
            variables=variables,
            method=method,
            permutations=permutations,
            significance_level=significance_level,
            sparse_threshold=sparse_threshold
        )
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def compute_prediction_intervals(
        self, 
        model_func: Callable, 
        data: Union[pd.DataFrame, np.ndarray] = None,
        pred_x: Union[pd.DataFrame, np.ndarray] = None,
        method: str = 'bootstrap',
        conf_level: float = 0.95,
        n_bootstrap: int = 1000,
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        convergence_threshold: Optional[float] = 0.001,
        check_convergence_interval: int = 50
    ) -> Dict[str, Any]:
        """
        Compute prediction intervals using analytical or bootstrap methods.
        
        Parameters
        ----------
        model_func : callable
            Function to fit model on bootstrap samples
        data : array_like, optional
            Original data for model estimation, uses self.data if not provided
        pred_x : array_like, optional
            Predictor values for predictions
        method : str, optional
            Method for interval calculation ('analytical', 'bootstrap')
        conf_level : float, optional
            Confidence level (0.9, 0.95, 0.99)
        n_bootstrap : int, optional
            Number of bootstrap samples
        batch_size : int, optional
            Batch size for parallel processing
        max_workers : int, optional
            Number of worker processes
        convergence_threshold : float, optional
            Threshold for early stopping
        check_convergence_interval : int, optional
            Interval for checking convergence
            
        Returns
        -------
        dict
            Prediction intervals and metadata
        """
        if data is None:
            data = self.data
            if data is None:
                raise ModelError("No data provided for prediction intervals")
        
        return self.prediction_analyzer.compute_prediction_intervals(
            model_func=model_func,
            data=data,
            pred_x=pred_x,
            method=method,
            conf_level=conf_level,
            n_bootstrap=n_bootstrap,
            batch_size=batch_size,
            max_workers=max_workers,
            convergence_threshold=convergence_threshold,
            check_convergence_interval=check_convergence_interval
        )
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def bootstrap_confidence_intervals(
        self,
        data: Union[pd.Series, np.ndarray] = None,
        statistic_func: Callable = None,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        method: str = 'percentile',
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        convergence_threshold: Optional[float] = 0.001,
        check_interval: int = 50,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals with optimizations.
        
        Parameters
        ----------
        data : array_like, optional
            Data to bootstrap from, uses self.data if not provided
        statistic_func : callable
            Function to compute the statistic
        n_bootstrap : int, optional
            Number of bootstrap samples
        alpha : float, optional
            Significance level (e.g., 0.05 for 95% confidence)
        method : str, optional
            Method for computing intervals ('percentile', 'bca')
        batch_size : int, optional
            Size of batches for parallel processing
        max_workers : int, optional
            Number of worker processes (defaults to CPU count - 1)
        convergence_threshold : float, optional
            Early stopping threshold for bootstrap convergence
        check_interval : int, optional
            Interval for checking convergence
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Bootstrap results with confidence intervals
        """
        if data is None:
            data = self.data
            if data is None:
                raise ModelError("No data provided for bootstrap confidence intervals")
        
        if statistic_func is None:
            raise ModelError("statistic_func must be provided for bootstrap confidence intervals")
        
        return self.prediction_analyzer.bootstrap_confidence_intervals(
            data=data,
            statistic_func=statistic_func,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            method=method,
            batch_size=batch_size,
            max_workers=max_workers,
            convergence_threshold=convergence_threshold,
            check_interval=check_interval,
            random_seed=random_seed
        )
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def test_parameter_stability(
        self, 
        data=None, 
        window_size=None, 
        step_size=10, 
        stability_threshold=0.5, 
        model_func=None,
        method='rolling',
        visualization=True,
        max_chunk_size=10000
    ) -> Dict[str, Any]:
        """
        Test parameter stability using rolling estimation or recursive methods.
        
        Parameters
        ----------
        data : array_like, optional
            Time series data, uses self.data if not provided
        window_size : int, optional
            Size of the rolling window (default: 20% of sample)
        step_size : int, optional
            Step size for rolling window
        stability_threshold : float, optional
            Threshold for coefficient of variation to determine stability
        model_func : callable, optional
            Function to estimate model parameters
        method : str, optional
            Method for stability testing ('rolling', 'recursive', 'cusum')
        visualization : bool, optional
            Whether to create visualizations
        max_chunk_size : int, optional
            Maximum chunk size for memory optimization
            
        Returns
        -------
        dict
            Stability test results
        """
        if data is None:
            data = self.data
            if data is None:
                raise ModelError("No data provided for parameter stability test")
        
        return self.stability_tester.test_parameter_stability(
            data=data,
            window_size=window_size,
            step_size=step_size,
            stability_threshold=stability_threshold,
            model_func=model_func,
            method=method,
            visualization=visualization,
            max_chunk_size=max_chunk_size
        )
    
    def test_normality(self, residuals=None):
        """Test normality of residuals."""
        if residuals is None:
            residuals = self.residuals
        
        return self.residuals_analyzer.test_normality(residuals)
    
    def test_autocorrelation(self, residuals=None, lags=None):
        """Test for autocorrelation in residuals."""
        if residuals is None:
            residuals = self.residuals
        
        return self.residuals_analyzer.test_autocorrelation(residuals, lags=lags)
    
    def test_heteroskedasticity(self, residuals=None):
        """Test for heteroskedasticity in residuals."""
        if residuals is None:
            residuals = self.residuals
        
        return self.residuals_analyzer.test_heteroskedasticity(residuals)
    
    def test_cusum(self, data=None, model_func=None, significance=0.05, plot=True, save_path=None):
        """Perform CUSUM test for parameter stability."""
        if data is None:
            data = self.data
        
        return self.stability_tester.test_cusum(
            data=data,
            model_func=model_func,
            significance=significance,
            plot=plot,
            save_path=save_path
        )
    
    def run_all_residual_tests(self, residuals=None, lags=None):
        """Run comprehensive tests on model residuals."""
        if residuals is None:
            residuals = self.residuals
        
        return self.residuals_analyzer.run_all_tests(residuals, lags=lags)
    
    def plot_diagnostics(self, residuals=None, title=None, save_path=None, **kwargs):
        """Create diagnostic plots for model residuals."""
        if residuals is None:
            residuals = self.residuals
        
        if title is None:
            title = f"{self.model_name} Diagnostics"
        
        return self.residuals_analyzer.plot_diagnostics(
            residuals=residuals,
            title=title,
            save_path=save_path,
            **kwargs
        )
    
    def create_advanced_diagnostic_plots(
        self, 
        residuals=None, 
        save_path=None,
        include_bootstrap=True,
        n_bootstrap=1000,
        confidence_level=0.95
    ):
        """Create advanced diagnostic plots for model residuals."""
        if residuals is None:
            residuals = self.residuals
        
        return self.residuals_analyzer.create_advanced_diagnostic_plots(
            residuals=residuals,
            save_path=save_path,
            include_bootstrap=include_bootstrap,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @timer
    def run_all_diagnostics(
        self, 
        residuals=None, 
        data=None, 
        model_type=None,
        weights_matrix=None,
        plot=True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics based on model type.
        
        Parameters
        ----------
        residuals : array_like, optional
            Model residuals
        data : array_like, optional
            Original data
        model_type : str, optional
            Type of model ('linear', 'threshold', 'spatial', etc.)
        weights_matrix : object, optional
            Spatial weights matrix for spatial models
        plot : bool, optional
            Whether to create diagnostic plots
        **kwargs : dict, optional
            Additional parameters for specific tests
            
        Returns
        -------
        dict
            Comprehensive diagnostic results
        """
        # Set instance variables if provided
        if residuals is not None:
            self.residuals = residuals
        if data is not None:
            self.data = data
        
        # Initialize results dictionary
        results = {}
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run diagnostics based on model type
        if model_type is None or model_type == 'linear':
            # Basic tests for all model types
            results['residuals'] = self.residuals_analyzer.run_all_tests(
                self.residuals, lags=kwargs.get('lags')
            )
            
        elif model_type == 'spatial':
            # Spatial model diagnostics
            if weights_matrix is not None and self.data is not None:
                results['spatial_autocorrelation'] = self.test_spatial_autocorrelation(
                    data=self.data,
                    weights_matrix=weights_matrix,
                    method='both',
                    permutations=kwargs.get('permutations', 999)
                )
            
            # Also run basic residual tests
            results['residuals'] = self.residuals_analyzer.run_all_tests(
                self.residuals, lags=kwargs.get('lags')
            )
            
        elif model_type in ['threshold', 'tvecm']:
            # Threshold model diagnostics
            results['parameter_stability'] = self.test_parameter_stability(
                data=self.data,
                method=kwargs.get('stability_method', 'recursive'),
                visualization=plot
            )
            
            # Also run basic residual tests
            results['residuals'] = self.residuals_analyzer.run_all_tests(
                self.residuals, lags=kwargs.get('lags')
            )
        
        # Common additional diagnostics
        if self.data is not None and 'parameter_stability' not in results:
            results['parameter_stability'] = self.stability_tester.test_parameter_stability(
                data=self.data,
                method=kwargs.get('stability_method', 'rolling'),
                visualization=plot
            )
        
        # Create plots if requested
        if plot and self.residuals is not None:
            if kwargs.get('advanced_plots', False):
                results['plots'] = self.residuals_analyzer.create_advanced_diagnostic_plots(
                    self.residuals,
                    save_path=kwargs.get('save_path')
                )
            else:
                results['plots'] = self.residuals_analyzer.plot_diagnostics(
                    self.residuals,
                    title=kwargs.get('plot_title', f"{self.model_name} Diagnostics"),
                    save_path=kwargs.get('save_path')
                )
        
        # Calculate memory usage
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        logger.info(f"Comprehensive diagnostics complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return results


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
        Dictionary of fit statistics including R-squared, RMSE, MAE, etc.
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
    
    # Calculate residuals
    residuals = observed - predicted
    
    # Calculate statistics
    n = len(observed)
    y_mean = np.mean(observed)
    
    # Total sum of squares
    ss_total = np.sum((observed - y_mean) ** 2)
    
    # Residual sum of squares
    ss_residual = np.sum(residuals ** 2)
    
    # R-squared and adjusted R-squared
    r_squared = 1 - (ss_residual / ss_total)
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - n_params - 1))
    
    # Root Mean Squared Error
    rmse = np.sqrt(ss_residual / n)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(residuals))
    
    # Mean Absolute Percentage Error (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_percent_errors = np.abs(residuals / observed) * 100
        mape = np.mean(abs_percent_errors[np.isfinite(abs_percent_errors)])
    
    # Log likelihood (assuming normal errors)
    sigma2 = ss_residual / n
    loglikelihood = -n/2 * (1 + np.log(2 * np.pi) + np.log(sigma2))
    
    # Information criteria
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
        'n_params': n_params,
        'ss_total': ss_total,
        'ss_residual': ss_residual,
        'sigma2': sigma2
    }