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
from typing import Dict, Any, Tuple, Union, Optional, List, Callable
import ruptures as rpt
from scipy import stats

from src.utils import (
    # Error handling
    handle_errors, ModelError,
    
    # Validation
    validate_time_series, raise_if_invalid, validate_dataframe,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, memoize, disk_cache,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get default configuration
DEFAULT_ALPHA = config.get('analysis.diagnostics.alpha', 0.05)
DEFAULT_LAGS = config.get('analysis.diagnostics.lags', 12)
PLOT_DIR = config.get('paths.plots', 'plots')


class ModelDiagnostics:
    """
    Comprehensive diagnostics for econometric models.
    
    This class provides methods for testing statistical properties of model residuals,
    creating diagnostic plots, and evaluating model validity. Includes specialized
    diagnostics for threshold models, spatial models, and cointegration analysis.
    """
    
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
        
        logger.info(f"Initializing ModelDiagnostics for {self.model_name}")
        
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
        
        # Run individual tests
        normality_results = self.test_normality(residuals)
        autocorr_results = self.test_autocorrelation(residuals, lags)
        hetero_results = self.test_heteroskedasticity(residuals)
        
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
        
        logger.info(f"Residual diagnostics complete: valid={valid_tests}")
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
        
        # Get total sample size
        n = len(data)
        
        # Calculate number of windows
        n_windows = (n - window_size) // step_size + 1
        if n_windows <= 1:
            raise ModelError(f"Not enough data for multiple windows. Try smaller window_size or step_size.")
        
        # Create array to store parameters
        params_history = []
        dates = []
        
        # Rolling estimation
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            # Get window data
            window_data = data[start_idx:end_idx]
            
            try:
                # Estimate model on window
                params = model_func(window_data)
                
                # Ensure params is a numpy array
                if not isinstance(params, np.ndarray):
                    params = np.array(params)
                
                params_history.append(params)
                
                # Store corresponding date if available
                if hasattr(data, 'index') and isinstance(data.index, pd.DatetimeIndex):
                    dates.append(data.index[end_idx - 1])
                else:
                    dates.append(end_idx - 1)
                    
            except Exception as e:
                logger.warning(f"Error estimating model for window {i+1}: {str(e)}. Skipping window.")
        
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
        
        if is_stable:
            logger.info("Stability test result: Model parameters are stable")
        else:
            logger.warning(f"Model parameters are UNSTABLE. {', '.join(unstable_params)}")
        
        return results
    
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
        
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            # Store original index for later
            has_datetime_index = isinstance(data.index, pd.DatetimeIndex)
            original_index = data.index if has_datetime_index else None
            array = data.values
        else:
            has_datetime_index = False
            original_index = None
            array = np.asarray(data)
        
        array = array.reshape(-1, 1)  # Ensure 2D for ruptures
        
        logger.info(f"Testing for structural breaks using {method} method, max_breaks={max_breaks}")
        
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
        
        if breakpoint_dates:
            result['breakpoint_dates'] = breakpoint_dates
        
        # Create segments (for potential plotting)
        segments = []
        start = 0
        for bp in breakpoints:
            segments.append((start, bp))
            start = bp
        
        result['segments'] = segments
        
        logger.info(f"Structural break test: detected {result['n_breakpoints']} breakpoints")
        return result
    
    @disk_cache(cache_dir='.cache/diagnostics')
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
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
        
        # Initialize ThresholdVECM
        model = ThresholdVECM(data)
        
        # Run test for threshold effect
        try:
            test_result = model.test_threshold_significance(n_bootstrap=bootstrap_reps)
            return test_result
        except Exception as e:
            logger.warning(f"Error in threshold significance test: {str(e)}. Using simplified approach.")
            
            # Fallback to simpler approach
            coint_results = model.estimate_linear_vecm()
            tvecm_results = model.estimate_tvecm()
            
            # Compare likelihoods
            llf_vecm = coint_results.llf
            llf_tvecm = tvecm_results.get('llf')
            
            if llf_vecm is not None and llf_tvecm is not None:
                lr_stat = 2 * (llf_tvecm - llf_vecm)
                # Simple rule: LR statistic > 5 suggests threshold effect
                threshold_valid = lr_stat > 5
                
                result = {
                    'lr_statistic': lr_stat,
                    'threshold_model_valid': threshold_valid,
                    'note': "Simplified likelihood ratio test used due to error in threshold_significance test"
                }
            else:
                result = {
                    'threshold_model_valid': None,
                    'note': "Cannot determine threshold validity"
                }
            
            return result
    
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
    
    @m1_optimized()
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
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
            self.original_data = data
        
        # Initialize results dictionary
        results = {}
        
        # Basic residual diagnostics for all model types
        try:
            results['residuals'] = self.residual_tests(self.residuals)
        except Exception as e:
            logger.warning(f"Error in residual tests: {str(e)}")
            results['residuals'] = {'error': str(e)}
        
        # Create plots if requested
        if plot:
            try:
                plots = self.plot_diagnostics(
                    self.residuals, 
                    title=f"{self.model_name} Diagnostics",
                    save_path=os.path.join(PLOT_DIR, f"{self.model_name.lower()}_diagnostics.png") if save_plots else None
                )
                results['plots'] = plots
            except Exception as e:
                logger.warning(f"Error creating diagnostic plots: {str(e)}")
                results['plots'] = {'error': str(e)}
        
        # Model-specific diagnostics
        if model_type == 'threshold' or model_type == 'tvecm':
            # Threshold validity test
            if tvecm_result is not None:
                try:
                    results['threshold_validity'] = self.test_threshold_validity(tvecm_result, self.original_data)
                except Exception as e:
                    logger.warning(f"Error in threshold validity test: {str(e)}")
                    results['threshold_validity'] = {'error': str(e)}
            
            # Asymmetric adjustment test
            try:
                # Use equilibrium errors from tvecm_result if available
                eq_errors = tvecm_result.get('equilibrium_errors', self.residuals)
                results['asymmetric_adjustment'] = self.test_asymmetric_adjustment(eq_errors)
            except Exception as e:
                logger.warning(f"Error in asymmetric adjustment test: {str(e)}")
                results['asymmetric_adjustment'] = {'error': str(e)}
                
        elif model_type == 'spatial':
            # Spatial autocorrelation test
            if weights_matrix is not None and self.original_data is not None:
                try:
                    results['spatial_autocorrelation'] = self.test_spatial_autocorrelation(
                        self.original_data, weights_matrix
                    )
                except Exception as e:
                    logger.warning(f"Error in spatial autocorrelation test: {str(e)}")
                    results['spatial_autocorrelation'] = {'error': str(e)}
        
        # Common additional diagnostics for all model types
        if self.original_data is not None:
            # Structural break test
            try:
                results['structural_breaks'] = self.test_structural_breaks(self.original_data)
            except Exception as e:
                logger.warning(f"Error in structural break test: {str(e)}")
                results['structural_breaks'] = {'error': str(e)}
            
            # Parameter stability test
            try:
                results['parameter_stability'] = self.test_model_stability(data=self.original_data)
            except Exception as e:
                logger.warning(f"Error in parameter stability test: {str(e)}")
                results['parameter_stability'] = {'error': str(e)}
        
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
        
        logger.info(f"Model diagnostics complete: valid_model={valid_model}")
        return results


@m1_optimized()
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
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
    
    # Calculate residuals
    residuals = observed - predicted
    
    # Calculate fit statistics
    n = len(observed)
    p = n_params
    
    # Mean of observed values
    y_mean = np.mean(observed)
    
    # Total sum of squares
    ss_total = np.sum((observed - y_mean) ** 2)
    
    # Residual sum of squares
    ss_residual = np.sum(residuals ** 2)
    
    # R-squared
    r_squared = 1 - (ss_residual / ss_total)
    
    # Adjusted R-squared
    adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    
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
    
    # AIC and BIC
    aic = -2 * loglikelihood + 2 * p
    bic = -2 * loglikelihood + p * np.log(n)
    
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
        'n_params': p
    }


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
    
    # Generate bootstrap samples
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Draw random sample with replacement
        indices = np.random.randint(0, n, size=n)
        sample = data_arr[indices]
        
        # Calculate statistic
        stat = statistic_func(sample)
        bootstrap_stats.append(stat)
    
    # Calculate confidence intervals
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    if method == 'percentile':
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    elif method == 'bca':
        # BCa method (bias-corrected and accelerated) - simplified implementation
        # This is not a full BCa implementation but a simplified version
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < observed_stat))
        
        # Calculate acceleration factor
        jackknife_stats = []
        for i in range(n):
            # Leave one out
            sample = np.delete(data_arr, i)
            jackknife_stats.append(statistic_func(sample))
        
        jackknife_mean = np.mean(jackknife_stats)
        a = np.sum((jackknife_mean - jackknife_stats) ** 3) / (6 * np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        
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
    
    return {
        'statistic': observed_stat,
        'bootstrap_stats': bootstrap_stats,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha,
        'method': method
    }


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
    
    # Generate bootstrap predictions
    bootstrap_predictions = []
    
    for _ in range(n_bootstrap):
        # Draw random sample with replacement
        indices = np.random.randint(0, n, size=n)
        sample = data_arr[indices]
        
        # Fit model to bootstrap sample
        bootstrap_model = model_func(sample)
        
        # Get predictions
        bootstrap_pred = bootstrap_model.predict(pred_x_arr)
        bootstrap_predictions.append(bootstrap_pred)
    
    # Convert to numpy array
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Calculate intervals
    lower_bound = np.percentile(bootstrap_predictions, alpha/2 * 100, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, (1-alpha/2) * 100, axis=0)
    
    return {
        'predictions': predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_bootstrap': n_bootstrap,
        'alpha': alpha
    }