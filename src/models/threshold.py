"""
Threshold cointegration module for market integration analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import matplotlib.pyplot as plt

from yemen_market_integration.utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_time_series, validate_model_inputs, validate_dataframe, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, disk_cache, parallelize_dataframe,
    optimize_dataframe, configure_system_for_performance,
    
    # Data processing
    fill_missing_values, create_lag_features, normalize_columns, detect_outliers,
    
    # Statistical utilities
    test_stationarity, test_cointegration, bootstrap_confidence_interval,
    test_linearity, calculate_half_life, compute_variance_ratio, 
    test_structural_break, test_white_noise,
    
    # Plotting utilities
    set_plotting_style, format_date_axis, plot_time_series, 
    plot_dual_axis, save_plot, add_annotations,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values
DEFAULT_ALPHA = config.get('analysis.threshold.alpha', 0.05)
DEFAULT_TRIM = config.get('analysis.threshold.trim', 0.15)
DEFAULT_MAX_LAGS = config.get('analysis.threshold.max_lags', 4)
DEFAULT_N_BOOTSTRAP = config.get('analysis.threshold.n_bootstrap', 1000)
DEFAULT_MTAR_THRESHOLD = config.get('analysis.threshold.mtar_default_threshold', 0.0)


class ThresholdCointegration:
    """
    Threshold cointegration model implementation.
    
    This class implements a two-regime threshold cointegration model to analyze 
    price transmission between two markets. In the Yemen context, thresholds represent
    transaction costs and barriers to trade that impede arbitrage.
    """
    
    def __init__(
        self, 
        data1: Union[pd.Series, np.ndarray], 
        data2: Union[pd.Series, np.ndarray], 
        max_lags: int = DEFAULT_MAX_LAGS,
        market1_name: str = "Market 1",
        market2_name: str = "Market 2"
    ):
        """
        Initialize the threshold cointegration model.
        
        Parameters
        ----------
        data1 : array_like
            First time series (typically price series from first market)
        data2 : array_like
            Second time series (typically price series from second market)
        max_lags : int, optional
            Maximum number of lags to consider
        market1_name : str, optional
            Name of the first market (for plotting and reporting)
        market2_name : str, optional
            Name of the second market (for plotting and reporting)
        """
        # Optimize system for performance
        configure_system_for_performance()
        
        # Validate input time series
        self._validate_inputs(data1, data2)
        
        # Store input time series
        if isinstance(data1, pd.Series):
            self.data1 = data1.copy()
            self.index = data1.index
        else:
            self.data1 = np.asarray(data1)
            self.index = None
        
        if isinstance(data2, pd.Series):
            self.data2 = data2.copy()
            if self.index is None:
                self.index = data2.index
        else:
            self.data2 = np.asarray(data2)
        
        # Store market names
        self.market1_name = market1_name
        self.market2_name = market2_name
        
        # Validate max_lags
        valid, errors = validate_model_inputs(
            model_name="threshold",
            params={"max_lags": max_lags},
            required_params={"max_lags"},
            param_validators={
                "max_lags": lambda x: isinstance(x, int) and x > 0
            }
        )
        raise_if_invalid(valid, errors, "Invalid max_lags parameter")
        
        self.max_lags = max_lags
        self.results = None
        self.beta0 = None
        self.beta1 = None
        self.eq_errors = None
        self.threshold = None
        self.ssr = None
        
        logger.info(f"Initialized ThresholdCointegration with {len(self.data1)} observations")
    
    def _validate_inputs(self, data1, data2):
        """Validate input time series."""
        for i, data in enumerate([data1, data2], 1):
            valid, errors = validate_time_series(
                data,
                min_length=30,
                max_nulls=0,
                check_constant=True
            )
            raise_if_invalid(valid, errors, f"Invalid time series for data{i}")
        
        # Check equal lengths
        if len(data1) != len(data2):
            raise ValidationError(
                f"Time series must have equal length, got {len(data1)} and {len(data2)}"
            )
    
    @disk_cache(cache_dir=".cache/yemen_market_integration/threshold")
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def estimate_cointegration(self) -> Dict[str, Any]:
        """
        Estimate the cointegration relationship.
        
        Tests for long-run equilibrium relationship between two market prices
        using the Engle-Granger procedure.
        
        Returns
        -------
        dict
            Cointegration test results including:
            - statistic: Test statistic value
            - pvalue: P-value of the test
            - critical_values: Critical values at different significance levels
            - cointegrated: Boolean indicating if markets are cointegrated
            - beta0: Intercept of cointegrating relationship
            - beta1: Slope coefficient
            - equilibrium_errors: Residuals representing deviations from equilibrium
            
        Notes
        -----
        In the Yemen market context, cointegration indicates price linkages between markets.
        The beta1 coefficient shows the long-run price transmission elasticity,
        while equilibrium errors represent deviations due to transaction costs and barriers.
        """
        # Use project's test_cointegration utility
        result = test_cointegration(
            self.data1, 
            self.data2, 
            method='engle-granger', 
            trend='c', 
            lags=self.max_lags,
            alpha=DEFAULT_ALPHA
        )
        
        # Store the cointegration vector from the results
        self.beta0 = result['beta'][0]  # Intercept
        self.beta1 = result['beta'][1]  # Slope
        
        # Calculate the equilibrium errors using the cointegration vector
        self.eq_errors = self.data1 - (self.beta0 + self.beta1 * self.data2)
        
        logger.info(
            f"Cointegration test: statistic={result['statistic']:.4f}, "
            f"p-value={result['pvalue']:.4f}, "
            f"beta0={self.beta0:.4f}, beta1={self.beta1:.4f}"
        )
        
        # Format result
        return {
            'statistic': result['statistic'],
            'pvalue': result['pvalue'],
            'critical_values': result['critical_values'],
            'cointegrated': result['cointegrated'],
            'beta0': self.beta0,
            'beta1': self.beta1,
            'equilibrium_errors': self.eq_errors,
            'long_run_relationship': f"{self.market1_name} = {self.beta0:.4f} + {self.beta1:.4f} × {self.market2_name}",
            'residuals': self.eq_errors
        }
    
    @timer
    @memory_usage_decorator
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def estimate_threshold(
        self, 
        n_grid: int = 300, 
        trim: float = DEFAULT_TRIM
    ) -> Dict[str, Any]:
        """
        Estimate the threshold parameter using grid search.
        
        The threshold represents transaction costs between markets. When price
        differentials exceed this threshold, arbitrage becomes profitable and
        prices adjust more quickly.
        
        Parameters
        ----------
        n_grid : int, optional
            Number of grid points to evaluate
        trim : float, optional
            Trimming percentage for excluding extreme values
            
        Returns
        -------
        dict
            Threshold estimation results including:
            - threshold: Optimal threshold value
            - ssr: Sum of squared residuals at optimal threshold
            - all_thresholds: All evaluated thresholds
            - all_ssrs: SSRs for all thresholds
            
        Notes
        -----
        In Yemen's fragmented markets, the threshold estimates transaction costs that
        include conflict-related barriers (checkpoints, security risks), transportation
        costs, and dual exchange rate effects.
        """
        # Make sure we have cointegration results
        if self.eq_errors is None:
            logger.info("Running cointegration estimation first")
            self.estimate_cointegration()
        
        # Save trim for later use
        self.trim = trim
        
        # Validate parameters
        valid, errors = validate_model_inputs(
            model_name="threshold",
            params={"n_grid": n_grid, "trim": trim},
            required_params={"n_grid", "trim"},
            param_validators={
                "n_grid": lambda x: isinstance(x, int) and x > 0,
                "trim": lambda x: 0.0 < x < 0.5
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold estimation parameters")
        
        # Make sure we have cointegration results
        if self.eq_errors is None:
            logger.info("Running cointegration estimation first")
            self.estimate_cointegration()
        
        # Save trim for later use
        self.trim = trim
        
        # Validate parameters
        valid, errors = validate_model_inputs(
            model_name="threshold",
            params={"n_grid": n_grid, "trim": trim},
            required_params={"n_grid", "trim"},
            param_validators={
                "n_grid": lambda x: isinstance(x, int) and x > 0,
                "trim": lambda x: 0.0 < x < 0.5
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold estimation parameters")
        
        # Get threshold candidates using helper function
        candidates = self._get_threshold_candidates(trim, n_grid)
        
        # Grid search with parallelization for better performance
        logger.info(f"Starting grid search with {len(candidates)} threshold candidates")
        
        def process_threshold(threshold_candidate):
            return (threshold_candidate, self._compute_ssr_for_threshold(threshold_candidate))
        
        df_candidates = pd.DataFrame({'threshold': candidates})
        results_df = parallelize_dataframe(df_candidates, lambda df: df.apply(
            lambda row: process_threshold(row['threshold']), axis=1))
        
        # Process results using helper function
        best_threshold, best_ssr, thresholds, ssrs = self._process_threshold_results(results_df)
        
        self.threshold = best_threshold
        self.ssr = best_ssr
        
        logger.info(f"Threshold estimation complete: threshold={best_threshold:.4f}, ssr={best_ssr:.4f}")
        
        return {
            'threshold': best_threshold,
            'ssr': best_ssr,
            'all_thresholds': thresholds,
            'all_ssrs': ssrs,
            'proportion_below': np.mean(self.eq_errors <= best_threshold),
            'proportion_above': np.mean(self.eq_errors > best_threshold)
        }
    
    @timer
    @memory_usage_decorator
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _compute_ssr_for_threshold(self, threshold: float) -> float:
        """
        Compute sum of squared residuals for a given threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value to evaluate
            
        Returns
        -------
        float
            Sum of squared residuals
        """
        import statsmodels.api as sm
        
        # Indicator function
        below = self.eq_errors <= threshold
        above = ~below
        
        # Prepare data for regression
        y = np.diff(self.data1)
        X1 = np.column_stack([
            self.eq_errors[:-1] * below[:-1],
            self.eq_errors[:-1] * above[:-1]
        ])
        
        # Add lagged differences using project utilities
        df_diffs = pd.DataFrame({
            'd1': np.diff(self.data1),
            'd2': np.diff(self.data2)
        })
        
        # Create lag features using utility function
        df_with_lags = create_lag_features(
            df_diffs, 
            columns=['d1', 'd2'], 
            lags=list(range(1, min(self.max_lags + 1, len(y))))
        )
        
        # Fill missing values using utility function
        df_with_lags = fill_missing_values(df_with_lags, numeric_strategy='median')
        
        # Extract lagged values
        lag_diffs = df_with_lags.iloc[self.max_lags:].values
        
        # Combine features and add constant
        X1 = np.column_stack([X1, lag_diffs])
        X1 = sm.add_constant(X1)
        
        # Fit the model and return SSR
        return sm.OLS(y[self.max_lags:], X1).fit().ssr

    def _get_threshold_candidates(self, trim: float, n_grid: int) -> np.ndarray:
        """Get candidate threshold values for grid search."""
        sorted_errors = np.sort(self.eq_errors)
        lower_idx = int(len(sorted_errors) * trim)
        upper_idx = int(len(sorted_errors) * (1 - trim))
        candidates = sorted_errors[lower_idx:upper_idx]
        if len(candidates) > n_grid:
            step = len(candidates) // n_grid
            candidates = candidates[::step]
        return candidates

    def _process_threshold_results(self, results_df) -> Tuple[float, float, List[float], List[float]]:
        """Process threshold grid search results."""
        best_ssr = np.inf
        best_threshold = None
        thresholds = []
        ssrs = []
        for i, row in results_df.iterrows():
            threshold, ssr = row[0]
            thresholds.append(threshold)
            ssrs.append(ssr)
            if ssr < best_ssr:
                best_ssr = ssr
                best_threshold = threshold
        return best_threshold, best_ssr, thresholds, ssrs
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def estimate_tvecm(self, run_diagnostics: bool = False) -> Dict[str, Any]:
        """
        Estimate the Threshold Vector Error Correction Model.
        
        Implements a two-regime VECM where adjustment speeds differ 
        based on whether price differentials exceed the threshold (transaction costs).
        
        Parameters
        ----------
        run_diagnostics : bool, optional
            Whether to run diagnostic tests on the model
            
        Returns
        -------
        dict
            TVECM estimation results including:
            - equation1: OLS results for first market equation
            - equation2: OLS results for second market equation
            - adjustment_below_1: Adjustment speed for market 1 below threshold
            - adjustment_above_1: Adjustment speed for market 1 above threshold
            - adjustment_below_2: Adjustment speed for market 2 below threshold
            - adjustment_above_2: Adjustment speed for market 2 above threshold
            - threshold: Estimated threshold value
            - cointegration_beta: Long-run coefficient
            - equilibrium_error: Residuals from cointegration equation
            - diagnostic_hooks: Integration points for diagnostic module
            
        Notes
        -----
        In Yemen's context, the adjustment coefficients represent the speed of
        price transmission between markets. Coefficients should be:
        - Larger (in absolute value) above threshold when arbitrage is profitable
        - Smaller below threshold when price differentials are within transaction costs
        """
        # Make sure we have a threshold
        if self.threshold is None:
            logger.info("Estimating threshold first")
            self.estimate_threshold()
        
        # Estimate models for each regime
        results = self._estimate_regime_models()
        
        # Extract and format results
        self.results = self._format_tvecm_results(results)
        
        # Add equilibrium error to results for convenience
        self.results['equilibrium_error'] = self.eq_errors
        
        # Calculate half lives using utility function
        half_lives = calculate_half_life(self.eq_errors)
        
        # Add diagnostic hooks
        self.results['diagnostic_hooks'] = {
            'residuals1': self.results['equation1'].resid,
            'residuals2': self.results['equation2'].resid,
            'residuals_below': self._extract_regime_residuals('below'),
            'residuals_above': self._extract_regime_residuals('above'),
            'threshold': self.threshold,
            'equilibrium_error': self.eq_errors,
            'half_lives': {
                'overall': half_lives['overall'],
                'below': calculate_half_life(self._extract_regime_residuals('below'))['overall'],
                'above': calculate_half_life(self._extract_regime_residuals('above'))['overall']
            },
            'model_type': 'threshold_cointegration'
        }
        
        # Calculate asymmetric adjustment
        asymm_adj = calculate_asymmetric_adjustment(self.results)
        self.results['asymmetric_adjustment'] = asymm_adj
        
        # Add summary stats for easier reference
        self.results['summary'] = {
            'threshold': self.threshold,
            'half_life_below_1': asymm_adj['half_life_below_1'],
            'half_life_above_1': asymm_adj['half_life_above_1'],
            'asymmetry_1': asymm_adj['asymmetry_1'],
            'cointegration_beta': self.beta1,
            'interpretation': _interpret_adjustment_speeds(
                self.results['adjustment_below_1'], 
                self.results['adjustment_above_1'],
                self.market1_name, self.market2_name
            )
        }
        
        logger.info(
            f"TVECM estimation complete: threshold={self.threshold:.4f}, "
            f"adjustment_below_1={self.results['adjustment_below_1']:.4f}, "
            f"adjustment_above_1={self.results['adjustment_above_1']:.4f}"
        )
        
        # Run diagnostics if requested
        if run_diagnostics:
            from models.diagnostics import ModelDiagnostics
            diagnostics = ModelDiagnostics(
                residuals=self.results['equation1'].resid,
                model_name="TVECM"
            )
            self.results['diagnostics'] = diagnostics.residual_tests()
        
        return self.results
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def estimate_mtar(self, run_diagnostics: bool = False) -> Dict[str, Any]:
        """
        Estimate the Momentum-Threshold model for asymmetric price adjustment.
        
        This method implements the Enders & Siklos approach by focusing on the
        momentum (change) in equilibrium errors rather than just their levels.
        
        Parameters
        ----------
        run_diagnostics : bool, optional
            Whether to run diagnostic tests on the model
            
        Returns
        -------
        dict
            M-TAR model estimation results
            
        Notes
        -----
        In Yemen's fragmented markets, momentum-based thresholds may detect
        different asymmetries than level-based thresholds, particularly when:
        - Exchange rate fluctuations cause rapid price changes
        - Conflict escalations create sudden transaction cost spikes
        - Political barriers affect the direction of price transmission
        """
        # Make sure we have cointegration results
        if self.eq_errors is None:
            logger.info("Running cointegration estimation first")
            self.estimate_cointegration()
        
        # Test for M-TAR adjustment
        mtar_results = test_mtar_adjustment(self.eq_errors, threshold=self.threshold)
        
        # Calculate additional metrics
        mtar_results['eq_errors_momentum'] = np.diff(self.eq_errors)
        
        # Add model type for reference
        mtar_results['model_type'] = 'mtar'
        
        # Store results for later use
        self.mtar_results = mtar_results
        
        # Run diagnostics if requested
        if run_diagnostics:
            # Add diagnostics specific to M-TAR model using project utilities
            try:
                # Test for white noise using project utility
                mtar_residuals = np.diff(self.eq_errors)
                white_noise_test = test_white_noise(mtar_residuals, alpha=DEFAULT_ALPHA)
                
                self.mtar_results['diagnostics'] = white_noise_test
            except Exception as e:
                logger.warning(f"Could not run M-TAR diagnostics: {e}")
        
        logger.info(
            f"M-TAR estimation complete: asymmetric={mtar_results['asymmetric']}, "
            f"p-value={mtar_results['p_value']:.4f}, "
            f"adj_positive={mtar_results['adjustment_positive']:.4f}, "
            f"adj_negative={mtar_results['adjustment_negative']:.4f}"
        )
        
        return mtar_results
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostic tests on the TVECM model.
        
        Performs comprehensive diagnostics including:
        - Normality tests on residuals
        - Autocorrelation tests
        - Heteroskedasticity tests
        - Parameter stability tests
        
        Returns
        -------
        dict
            Results of diagnostic tests
        """
        # Make sure we have TVECM results
        if self.results is None:
            logger.info("Estimating TVECM first")
            self.estimate_tvecm()
        
        logger.info("Running diagnostic tests on TVECM model")
        
        # Use project's ModelDiagnostics
        from models.diagnostics import ModelDiagnostics
        diagnostics = ModelDiagnostics(
            residuals=self.results['equation1'].resid,
            model_name="TVECM"
        )
        
        # Get residuals from both equations
        residuals1 = self.results['equation1'].resid
        residuals2 = self.results['equation2'].resid
        
        # Run comprehensive tests
        resid_tests1 = diagnostics.residual_tests(residuals1)
        resid_tests2 = diagnostics.residual_tests(residuals2)
        
        # Test for asymmetric adjustment
        asymm_test = test_asymmetric_adjustment(self.eq_errors, self.threshold)
        
        # Create diagnostic plots
        try:
            plot_results = diagnostics.plot_diagnostics(
                residuals=residuals1, 
                title=f"Model Diagnostics: {self.market1_name}"
            )
        except Exception as e:
            logger.warning(f"Failed to create diagnostic plots: {str(e)}")
            plot_results = {}
        
        # Test for structural breaks
        try:
            struct_break = test_structural_break(
                y=np.diff(self.data1)[self.max_lags:],
                X=None,  # Will create time trend automatically
                method='quandt'
            )
        except Exception as e:
            logger.warning(f"Failed to test for structural breaks: {str(e)}")
            struct_break = {}
        
        diagnostic_results = {
            'residual_tests': {
                'eq1': resid_tests1,
                'eq2': resid_tests2
            },
            'asymmetric_adjustment': asymm_test,
            'structural_breaks': struct_break,
            'plots': plot_results,
            'summary': {
                'normal_residuals': resid_tests1.get('normality', {}).get('normal', False) and 
                                   resid_tests2.get('normality', {}).get('normal', False),
                'no_autocorrelation': resid_tests1.get('autocorrelation', {}).get('no_autocorrelation', False) and 
                                     resid_tests2.get('autocorrelation', {}).get('no_autocorrelation', False),
                'homoskedastic': resid_tests1.get('heteroskedasticity', {}).get('homoskedastic', False) and 
                                resid_tests2.get('heteroskedasticity', {}).get('homoskedastic', False),
                'asymmetric_adjustment': asymm_test.get('asymmetric', False),
                'structural_breaks': struct_break.get('significant', False)
            }
        }
        
        logger.info(
            f"Diagnostics complete: normal_residuals={diagnostic_results['summary']['normal_residuals']}, "
            f"no_autocorrelation={diagnostic_results['summary']['no_autocorrelation']}, "
            f"homoskedastic={diagnostic_results['summary']['homoskedastic']}, "
            f"asymmetric_adjustment={diagnostic_results['summary']['asymmetric_adjustment']}"
        )
        
        return diagnostic_results
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete threshold cointegration analysis workflow.
        
        Performs the full analytical sequence:
        1. Test for cointegration between markets
        2. Estimate transaction cost threshold
        3. Estimate threshold VECM with regime-specific adjustment speeds
        4. Run comprehensive diagnostics
        5. Calculate asymmetric adjustment metrics
        6. Run M-TAR analysis and compare with TAR model
        
        Returns
        -------
        dict
            Complete analysis results
        """
        logger.info("Starting complete threshold cointegration analysis")
        
        # Run cointegration test
        coint_results = self.estimate_cointegration()
        
        # Check if series are cointegrated
        if not coint_results['cointegrated']:
            logger.warning("Series are not cointegrated, threshold estimation may not be valid")
        
        # Estimate threshold and TVECM
        threshold_results = self.estimate_threshold()
        tvecm_results = self.estimate_tvecm(run_diagnostics=True)
        
        # Run M-TAR analysis
        mtar_results = self.estimate_mtar(run_diagnostics=True)
        
        # Test threshold significance
        threshold_significance = self.test_threshold_significance()
        
        # Calculate threshold confidence intervals
        threshold_ci = self.calculate_threshold_confidence_intervals()
        
        # Test for linearity using project utilities
        linearity_test = test_linearity(y=np.diff(self.data1), threshold_var=self.eq_errors[:-1])
        
        # Compare TAR and M-TAR models
        model_comparison = {
            'tar_ssr': tvecm_results['equation1'].ssr + tvecm_results['equation2'].ssr,
            'mtar_ssr': mtar_results.get('ssr', float('inf')),
            'tar_asymmetry_pvalue': tvecm_results.get('asymmetric_adjustment', {}).get('p_value', 1.0),
            'mtar_asymmetry_pvalue': mtar_results['p_value'],
            'preferred_model': 'M-TAR' if mtar_results['p_value'] < tvecm_results.get('asymmetric_adjustment', {}).get('p_value', 1.0) else 'TAR',
            'linearity_test': linearity_test
        }
        
        # Compile all results
        full_results = {
            'cointegration': coint_results,
            'threshold': threshold_results,
            'tvecm': tvecm_results,
            'mtar': mtar_results,
            'threshold_significance': threshold_significance,
            'threshold_confidence_intervals': threshold_ci,
            'model_comparison': model_comparison,
            'market_integration_assessment': _assess_market_integration(
                tvecm_results['asymmetric_adjustment'],
                coint_results['cointegrated']
            ),
            'summary': {
                'cointegrated': coint_results['cointegrated'],
                'threshold': self.threshold,
                'asymmetric_adjustment_tar': tvecm_results['asymmetric_adjustment'].get('asymmetry_1', 0) != 0,
                'asymmetric_adjustment_mtar': mtar_results['asymmetric'],
                'preferred_model': model_comparison['preferred_model'],
                'linearity_rejected': linearity_test.get('linearity_rejected', False)
            }
        }
        
        logger.info(
            f"Full analysis complete: cointegrated={coint_results['cointegrated']}, "
            f"threshold={self.threshold:.4f}, "
            f"asymmetric_tar={tvecm_results['asymmetric_adjustment'].get('asymmetry_1', 0) != 0}, "
            f"asymmetric_mtar={mtar_results['asymmetric']}, "
            f"preferred_model={model_comparison['preferred_model']}"
        )
        
        return full_results
    
    @m1_optimized()
    def _estimate_regime_models(self) -> Dict[str, Any]:
        """Estimate OLS models for each regime."""
        import statsmodels.api as sm
        
        # Indicator function
        below = self.eq_errors <= self.threshold
        above = ~below
        
        # Prepare data
        y1 = np.diff(self.data1)
        y2 = np.diff(self.data2)
        
        # Create design matrices using project utilities
        X = self._create_regime_design_matrix(below, above)
        
        # Fit models
        model1 = sm.OLS(y1[self.max_lags:], X)
        model2 = sm.OLS(y2[self.max_lags:], X)
        
        return {
            'results1': model1.fit(),
            'results2': model2.fit()
        }
    
    def _create_regime_design_matrix(self, below, above) -> np.ndarray:
        """Create design matrix for regime-specific models."""
        import statsmodels.api as sm
        
        # Create regime-specific terms
        regime_terms = np.column_stack([
            self.eq_errors[:-1] * below[:-1],
            self.eq_errors[:-1] * above[:-1]
        ])
        
        # Create DataFrame for lag creation
        df_diffs = pd.DataFrame({
            'd1': np.diff(self.data1),
            'd2': np.diff(self.data2)
        })
        
        # Create lag features using utility function
        df_with_lags = create_lag_features(
            df_diffs, 
            columns=['d1', 'd2'], 
            lags=list(range(1, min(self.max_lags + 1, len(self.data1) - 1)))
        )
        
        # Fill missing values using utility function
        df_with_lags = fill_missing_values(df_with_lags, numeric_strategy='median')
        
        # Get values for design matrix
        lag_diffs = df_with_lags.iloc[self.max_lags:].values
        
        # Combine and add constant
        X = np.column_stack([regime_terms[self.max_lags:], lag_diffs])
        return sm.add_constant(X)
    
    def _format_tvecm_results(self, model_results) -> Dict[str, Any]:
        """Extract and format TVECM results."""
        results1 = model_results['results1']
        results2 = model_results['results2']
        
        # Extract adjustment speeds
        adj_below_1 = results1.params[1]
        adj_above_1 = results1.params[2]
        adj_below_2 = results2.params[1]
        adj_above_2 = results2.params[2]
        
        return {
            'equation1': results1,
            'equation2': results2,
            'adjustment_below_1': adj_below_1,
            'adjustment_above_1': adj_above_1,
            'adjustment_below_2': adj_below_2,
            'adjustment_above_2': adj_above_2,
            'threshold': self.threshold,
            'cointegration_beta': self.beta1
        }
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def export_equilibrium_errors(self) -> pd.DataFrame:
        """
        Export equilibrium errors (residuals) with regime information.
        
        Returns a DataFrame containing the equilibrium error term (ECT) from 
        the cointegration model, along with regime information based on the threshold.
        This facilitates further diagnostics and visualization of regime-specific behavior.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            - 'date': Time index if available
            - 'equilibrium_error': Cointegration residuals
            - 'regime': 'below' or 'above' based on threshold
            - 'threshold': Threshold value
            - 'lag_equilibrium_error': Lagged residuals for TVECM analysis
            
        Notes
        -----
        Equilibrium errors represent deviations from long-run equilibrium relationship.
        In Yemen's market context:
        - Persistent positive errors indicate north prices above equilibrium
        - Persistent negative errors indicate south prices above equilibrium
        - Errors exceeding threshold trigger faster adjustment (profitable arbitrage)
        - Conflict shocks may temporarily push errors beyond threshold
        """
        # Ensure we have results
        if self.eq_errors is None:
            self.estimate_cointegration()
        
        # Get equilibrium errors
        eq_errors = self.eq_errors
        
        # Convert to DataFrame
        if self.index is not None:
            df = pd.DataFrame({'equilibrium_error': eq_errors}, index=self.index)
            df.index.name = 'date'
            df = df.reset_index()
        else:
            df = pd.DataFrame({'equilibrium_error': eq_errors})
        
        # Add threshold and regime information
        df['threshold'] = self.threshold
        df['regime'] = np.where(df['equilibrium_error'] <= self.threshold, 'below', 'above')
        
        # Add lagged equilibrium error for TVECM analysis
        df['lag_equilibrium_error'] = df['equilibrium_error'].shift(1)
        
        # Optimize DataFrame memory usage using utility function
        df = optimize_dataframe(df)
        
        return df
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _extract_regime_residuals(self, regime: str) -> pd.Series:
        """
        Extract equilibrium errors for a specific regime.
        
        Parameters
        ----------
        regime : str
            Either 'below' or 'above' to specify which regime to extract
            
        Returns
        -------
        pd.Series
            Equilibrium errors for the specified regime
        """
        if self.eq_errors is None:
            self.estimate_cointegration()
        
        if self.threshold is None:
            self.estimate_threshold()
            
        if regime not in ['below', 'above']:
            raise ValueError("regime must be either 'below' or 'above'")
        
        # Filter errors by regime
        if regime == 'below':
            mask = self.eq_errors <= self.threshold
        else:  # above
            mask = self.eq_errors > self.threshold
            
        # Create Series with proper index if available
        if self.index is not None:
            return pd.Series(self.eq_errors[mask], index=self.index[mask])
        else:
            return pd.Series(self.eq_errors[mask])
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def prepare_simulation_data(self) -> Dict[str, Any]:
        """
        Prepare model results for use in simulation module.
        
        Creates a standardized data structure for simulation of policy counterfactuals,
        such as exchange rate unification or reduced conflict barriers.
        
        Returns
        -------
        dict
            Simulation-ready data structure containing:
            - threshold: Transaction cost parameter
            - cointegration_vector: Long-run price relationship
            - adjustment_speeds: Regime-specific price adjustment rates
            - half_lives: Time required for shocks to dissipate
            - equilibrium_errors: Deviations from long-run equilibrium
            - model_type: Model identifier for simulation module
            
        Notes
        -----
        This standardized output format facilitates integration with the
        simulation module for policy counterfactual analysis.
        """
        # Ensure model is estimated
        if self.results is None:
            self.estimate_tvecm()
        
        # Extract needed information
        asymm_adj = self.results['asymmetric_adjustment']
        
        # Create simulation-ready data structure
        simulation_data = {
            'threshold': self.threshold,
            'cointegration': {
                'beta0': self.beta0,
                'beta1': self.beta1,
                'long_run_relationship': f"{self.market1_name} = {self.beta0:.4f} + {self.beta1:.4f} × {self.market2_name}"
            },
            'adjustment_speeds': {
                'below': {
                    'market1': self.results['adjustment_below_1'],
                    'market2': self.results['adjustment_below_2']
                },
                'above': {
                    'market1': self.results['adjustment_above_1'],
                    'market2': self.results['adjustment_above_2']
                }
            },
            'half_lives': {
                'below': asymm_adj['half_life_below_1'],
                'above': asymm_adj['half_life_above_1'],
                'asymmetry': asymm_adj['asymmetry_1']
            },
            'regime_proportions': {
                'below': np.mean(self.eq_errors <= self.threshold),
                'above': np.mean(self.eq_errors > self.threshold)
            },
            'equilibrium_errors': self.eq_errors,
            'market_names': {
                'market1': self.market1_name,
                'market2': self.market2_name
            },
            'model_type': 'threshold_cointegration',
            'integration_assessment': _assess_market_integration(
                asymm_adj, 
                True  # Assuming cointegration for simulation preparation
            )
        }
        
        return simulation_data
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def plot_regime_dynamics(self, 
                           save_path: Optional[str] = None,
                           fig_size: Tuple[int, int] = (12, 10),
                           dpi: int = 300) -> plt.Figure:
        """
        Plot regime dynamics and adjustment speeds.
        
        Creates a visualization of price transmission dynamics between markets,
        including threshold effects, adjustment speeds, and equilibrium errors.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot
        fig_size : tuple, optional
            Figure size as (width, height)
        dpi : int, optional
            Resolution for saved figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with regime dynamics
        """
        # Ensure model is estimated
        if self.results is None:
            self.estimate_tvecm()
        
        # Set plotting style using utility
        set_plotting_style()
        
        # Prepare data
        eq_errors = self.eq_errors
        threshold = self.threshold
        adj_below_1 = self.results['adjustment_below_1']
        adj_above_1 = self.results['adjustment_above_1']
        adj_below_2 = self.results['adjustment_below_2']
        adj_above_2 = self.results['adjustment_above_2']
        
        # Calculate regime proportions
        prop_below = np.mean(eq_errors <= threshold)
        prop_above = np.mean(eq_errors > threshold)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle(f"Threshold Cointegration: {self.market1_name} - {self.market2_name}\nThreshold = {threshold:.4f}", fontsize=16)
        
        # Plot 1: Equilibrium errors over time with threshold
        ax = axes[0, 0]
        if self.index is not None:
            # Use plot_time_series utility
            plot_time_series(
                pd.DataFrame({'date': self.index, 'error': eq_errors}),
                x='date',
                y='error',
                ax=ax,
                color='blue'
            )
        else:
            ax.plot(eq_errors)
            
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
        ax.set_title("Equilibrium Errors Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Error")
        ax.legend()
        
        # Plot 2: Density of equilibrium errors with threshold
        ax = axes[0, 1]
        ax.hist(eq_errors, bins=30, density=True, alpha=0.7)
        ax.axvline(x=threshold, color='r', linestyle='--', 
                 label=f'Threshold: {threshold:.4f}\nBelow: {prop_below:.1%}, Above: {prop_above:.1%}')
        ax.set_title("Distribution of Equilibrium Errors")
        ax.set_xlabel("Error")
        ax.set_ylabel("Density")
        ax.legend()
        
        # Plot 3: Adjustment speeds comparison for Market 1
        ax = axes[1, 0]
        
        bars1 = ax.bar([0], [abs(adj_below_1)], width=0.4, label='Below Threshold', color='skyblue')
        bars2 = ax.bar([0.5], [abs(adj_above_1)], width=0.4, label='Above Threshold', color='salmon')
        
        ax.set_title(f"Adjustment Speeds: {self.market1_name}")
        ax.set_ylabel("Absolute Adjustment Speed")
        ax.set_xticks([0.25])
        ax.set_xticklabels([self.market1_name])
        ax.legend()
        
        # Add value labels using annotations utility
        annotations = {}
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            annotations[(bar.get_x() + bar.get_width()/2., height)] = f'{height:.3f}'
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            annotations[(bar.get_x() + bar.get_width()/2., height)] = f'{height:.3f}'
            
        add_annotations(ax, annotations=annotations)
        
        # Plot 4: Adjustment speeds comparison for Market 2
        ax = axes[1, 1]
        
        bars1 = ax.bar([0], [abs(adj_below_2)], width=0.4, label='Below Threshold', color='skyblue')
        bars2 = ax.bar([0.5], [abs(adj_above_2)], width=0.4, label='Above Threshold', color='salmon')
        
        ax.set_title(f"Adjustment Speeds: {self.market2_name}")
        ax.set_ylabel("Absolute Adjustment Speed")
        ax.set_xticks([0.25])
        ax.set_xticklabels([self.market2_name])
        ax.legend()
        
        # Add value labels using annotations utility
        annotations = {}
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            annotations[(bar.get_x() + bar.get_width()/2., height)] = f'{height:.3f}'
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            annotations[(bar.get_x() + bar.get_width()/2., height)] = f'{height:.3f}'
            
        add_annotations(ax, annotations=annotations)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save if requested using utility function
        if save_path:
            try:
                save_plot(fig, save_path, dpi=dpi)
                logger.info(f"Saved regime dynamics plot to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save plot: {str(e)}")
        
        return fig
    
    @disk_cache(cache_dir=".cache/yemen_market_integration/threshold")
    @memory_usage_decorator
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def test_threshold_significance(self, n_bootstrap: int = DEFAULT_N_BOOTSTRAP) -> Dict[str, Any]:
        """
        Test for significance of threshold effect using bootstrap.
        
        Tests the null hypothesis of linear cointegration (no threshold)
        against the alternative of threshold cointegration (nonlinear adjustment).
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap replications
            
        Returns
        -------
        dict
            Test results including:
            - test_statistic: F-statistic for threshold significance
            - p_value: Bootstrap p-value
            - significant: Boolean indicating if threshold effect is significant
            - critical_values: Critical values at different significance levels
            - interpretation: Text summary of results
            
        Notes
        -----
        In Yemen's fragmented markets, a significant threshold effect indicates
        that transaction costs create meaningful barriers to arbitrage, resulting
        in nonlinear price transmission between markets.
        """
        # Ensure model is estimated
        if self.results is None:
            self.estimate_tvecm()
        
        # Use the project's test_linearity utility for this test
        linearity_test = test_linearity(
            y=np.diff(self.data1)[self.max_lags:],
            threshold_var=self.eq_errors[:-1][self.max_lags:],
            method='hansen',
            trim=self.trim
        )
        
        # Add custom interpretation
        if linearity_test['linearity_rejected']:
            interpretation = (
                "Significant threshold effect detected. Price adjustment exhibits "
                "nonlinear behavior depending on whether price differentials exceed "
                f"transaction costs (threshold = {self.threshold:.4f}). "
                "This indicates market segmentation due to barriers between markets."
            )
        else:
            interpretation = (
                "No significant threshold effect detected. Price adjustment appears "
                "to be linear regardless of price differential magnitudes. "
                "This suggests either efficient market integration or uniformly "
                "high transaction costs across all price levels."
            )
        
        # Format results
        result = {
            'test_statistic': linearity_test['test_statistic'],
            'p_value': linearity_test['p_value'],
            'significant': linearity_test['linearity_rejected'],
            'critical_values': linearity_test.get('critical_values', {}),
            'n_bootstrap': linearity_test.get('n_bootstrap', n_bootstrap),
            'interpretation': interpretation
        }
        
        logger.info(f"Threshold significance test: p-value={result['p_value']:.4f}, significant={result['significant']}")
        
        return result
    
    @disk_cache(cache_dir=".cache/yemen_market_integration/threshold")
    @memory_usage_decorator
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def calculate_threshold_confidence_intervals(
        self, 
        confidence_level: float = 0.95,
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    ) -> Dict[str, Any]:
        """
        Calculate confidence intervals for threshold parameter.
        
        Implements a bootstrap procedure to quantify uncertainty
        around the estimated transaction cost threshold.
        
        Parameters
        ----------
        confidence_level : float, optional
            Confidence level (default: 0.95)
        n_bootstrap : int, optional
            Number of bootstrap replications
            
        Returns
        -------
        dict
            Confidence interval results including:
            - threshold: Point estimate
            - lower_bound: Lower confidence bound
            - upper_bound: Upper confidence bound
            - confidence_level: Specified confidence level
            - bootstrap_thresholds: Distribution of bootstrap estimates
            
        Notes
        -----
        Wider confidence intervals indicate uncertainty in transaction cost 
        estimation, potentially due to conflict volatility, exchange rate 
        fluctuations, or data quality issues in Yemen's market monitoring.
        """
        # Ensure model is estimated
        if self.results is None:
            self.estimate_tvecm()
        
        # Use the project's bootstrap_confidence_interval utility
        def threshold_estimator(data):
            """Function to estimate threshold from bootstrap sample"""
            # Create a temporary ThresholdCointegration instance
            bootstrap_model = ThresholdCointegration(
                data1=self.data1, 
                data2=self.data2,
                max_lags=self.max_lags,
                market1_name=self.market1_name,
                market2_name=self.market2_name
            )
            # Set cointegration parameters to original estimates
            bootstrap_model.beta0 = self.beta0
            bootstrap_model.beta1 = self.beta1
            
            # Generate bootstrap errors
            bootstrap_model.eq_errors = bootstrap_model.data1 - (
                bootstrap_model.beta0 + bootstrap_model.beta1 * bootstrap_model.data2
            )
            
            # Estimate threshold
            result = bootstrap_model.estimate_threshold(trim=self.trim)
            return result['threshold']
        
        # Create bootstrap samples of the equilibrium errors
        bootstrap_result = bootstrap_confidence_interval(
            data=self.eq_errors,
            statistic_func=threshold_estimator,
            alpha=1-confidence_level,
            n_bootstrap=n_bootstrap,
            method='percentile'
        )
        
        # Format results
        result = {
            'threshold': self.threshold,
            'lower_bound': bootstrap_result['lower_bound'],
            'upper_bound': bootstrap_result['upper_bound'],
            'confidence_level': confidence_level,
            'bootstrap_thresholds': bootstrap_result['bootstrap_stats'],
            'interval_width': bootstrap_result['upper_bound'] - bootstrap_result['lower_bound'],
            'relative_width': (bootstrap_result['upper_bound'] - bootstrap_result['lower_bound']) / 
                             abs(self.threshold) if self.threshold != 0 else np.inf
        }
        
        logger.info(
            f"Threshold {confidence_level*100:.0f}% confidence interval: "
            f"[{result['lower_bound']:.4f}, {result['upper_bound']:.4f}]"
        )
        
        return result


@timer
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def calculate_asymmetric_adjustment(
    model_results: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate asymmetric adjustment metrics from TVECM results.
    
    Computes various metrics to quantify asymmetric price transmission:
    - Half-lives for deviations to return to equilibrium
    - Differences in adjustment speeds between regimes
    - Relative adjustment burden between markets
    
    Parameters
    ----------
    model_results : dict
        Results from threshold_cointegration estimation
        
    Returns
    -------
    dict
        Asymmetric adjustment metrics
        
    Notes
    -----
    In Yemen's fragmented markets, asymmetric adjustment may indicate:
    - Political barriers affecting price transmission in one direction
    - Security risks making arbitrage feasible only in certain directions
    - Market power differences between regions
    """
    # Extract adjustment parameters
    adj_below_1 = model_results['adjustment_below_1']
    adj_above_1 = model_results['adjustment_above_1']
    adj_below_2 = model_results['adjustment_below_2']
    adj_above_2 = model_results['adjustment_above_2']
    
    # Calculate half-lives using project utility if appropriate
    # Here we calculate manually for specific adjustment coefficients
    if adj_below_1 < 0:
        half_life_below_1 = np.log(0.5) / np.log(1 + adj_below_1)
    else:
        half_life_below_1 = float('inf')
        
    if adj_above_1 < 0:
        half_life_above_1 = np.log(0.5) / np.log(1 + adj_above_1)
    else:
        half_life_above_1 = float('inf')
    
    # Calculate asymmetry measures
    asymmetry_1 = abs(adj_above_1) - abs(adj_below_1)
    asymmetry_2 = abs(adj_above_2) - abs(adj_below_2)
    
    # Calculate relative adjustment speeds
    if abs(adj_below_1) + abs(adj_below_2) > 0:
        rel_adj_below = abs(adj_below_1) / (abs(adj_below_1) + abs(adj_below_2))
    else:
        rel_adj_below = float('nan')
    
    if abs(adj_above_1) + abs(adj_above_2) > 0:
        rel_adj_above = abs(adj_above_1) / (abs(adj_above_1) + abs(adj_above_2))
    else:
        rel_adj_above = float('nan')
    
    # Calculate adjustment ratios (above/below)
    adj_ratio_1 = abs(adj_above_1) / abs(adj_below_1) if adj_below_1 != 0 else float('inf')
    adj_ratio_2 = abs(adj_above_2) / abs(adj_below_2) if adj_below_2 != 0 else float('inf')
    
    # Determine if adjustment is economically significant
    sig_adj_below_1 = abs(adj_below_1) > 0.05
    sig_adj_above_1 = abs(adj_above_1) > 0.05
    sig_adj_below_2 = abs(adj_below_2) > 0.05
    sig_adj_above_2 = abs(adj_above_2) > 0.05
    
    return {
        'half_life_below_1': half_life_below_1,
        'half_life_above_1': half_life_above_1,
        'asymmetry_1': asymmetry_1,
        'asymmetry_2': asymmetry_2,
        'adjustment_ratio_1': adj_ratio_1,
        'adjustment_ratio_2': adj_ratio_2,
        'relative_adjustment_below': rel_adj_below,
        'relative_adjustment_above': rel_adj_above,
        'significant_adjustment_below_1': sig_adj_below_1,
        'significant_adjustment_above_1': sig_adj_above_1,
        'significant_adjustment_below_2': sig_adj_below_2,
        'significant_adjustment_above_2': sig_adj_above_2
    }


@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def test_asymmetric_adjustment(
    residuals: Union[pd.DataFrame, pd.Series, np.ndarray], 
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Test for asymmetric price adjustment speeds above and below threshold.
    
    In Yemen's fragmented markets, adjustment speeds often differ between regimes due to:
    1. Political barriers affecting north vs. south exchange rate regions
    2. Conflict intensity affecting transportation costs and risk premiums
    3. Distance and infrastructure quality creating natural transaction costs
    
    Parameters
    ----------
    residuals : Union[pd.DataFrame, pd.Series, np.ndarray]
        Equilibrium error term (residuals) from cointegration model
    threshold : float, optional
        Threshold value separating regimes. If None, uses 0 as threshold.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing asymmetry test results:
        - 'asymmetric': bool indicating if adjustment is asymmetric
        - 'p_value': p-value for asymmetry test
        - 'adjustment_below': adjustment speed below threshold
        - 'adjustment_above': adjustment speed above threshold
        - 'half_life_below': half-life of deviations below threshold
        - 'half_life_above': half-life of deviations above threshold
        - 'test_statistic': F-statistic for equality of adjustment speeds
    """
    import statsmodels.api as sm
    
    # Set default threshold if not provided
    if threshold is None:
        threshold = 0.0
        logger.info("No threshold provided, using 0.0 as default")
    
    # Convert to numpy array for easier manipulation
    if isinstance(residuals, pd.Series) or isinstance(residuals, pd.DataFrame):
        residuals = residuals.values.flatten()
    
    # Prepare data for the model
    y = np.diff(residuals)
    X_below = np.zeros_like(residuals[:-1])
    X_above = np.zeros_like(residuals[:-1])
    
    # Split observations by threshold
    below_mask = residuals[:-1] <= threshold
    above_mask = residuals[:-1] > threshold
    
    X_below[below_mask] = residuals[:-1][below_mask]
    X_above[above_mask] = residuals[:-1][above_mask]
    
    # Create design matrix
    X = np.column_stack([X_below, X_above])
    
    # Add constant (intercept)
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Extract adjustment speeds (correcting for the constant term index)
    adjustment_below = model.params[1]
    adjustment_above = model.params[2]
    
    # Test for equality of parameters
    restriction = np.zeros((1, model.params.shape[0]))
    restriction[0, 1] = 1
    restriction[0, 2] = -1
    r_matrix = restriction
    q_value = 0
    
    wald_test = model.wald_test((r_matrix, q_value), scalar=True)
    
    # Calculate half-lives using project utility function for specific regime residuals
    half_life_below = float('inf')
    half_life_above = float('inf')
    
    if adjustment_below < 0:
        half_life_below = np.log(0.5) / np.log(1 + adjustment_below)
    
    if adjustment_above < 0:
        half_life_above = np.log(0.5) / np.log(1 + adjustment_above)
    
    # Calculate counts
    n_below = np.sum(below_mask)
    n_above = np.sum(above_mask)
    
    # Prepare results
    results = {
        'asymmetric': wald_test.pvalue < DEFAULT_ALPHA,
        'p_value': wald_test.pvalue,
        'adjustment_below': adjustment_below,
        'adjustment_above': adjustment_above,
        'half_life_below': half_life_below,
        'half_life_above': half_life_above,
        'test_statistic': wald_test.statistic,
        'n_below': n_below,
        'n_above': n_above,
        'threshold': threshold,
        'interpretation': _interpret_asymmetric_test(
            wald_test.pvalue < DEFAULT_ALPHA, adjustment_below, adjustment_above
        )
    }
    
    return results


@m1_optimized()
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def test_mtar_adjustment(
    residuals: Union[pd.DataFrame, pd.Series, np.ndarray], 
    threshold: Optional[float] = DEFAULT_MTAR_THRESHOLD
) -> Dict[str, Any]:
    """
    Test for momentum-threshold asymmetric adjustment using the M-TAR model.
    
    The M-TAR model (Enders & Siklos) uses changes in residuals rather than levels
    to test for asymmetric price transmission based on price momentum.
    
    Parameters
    ----------
    residuals : Union[pd.DataFrame, pd.Series, np.ndarray]
        Equilibrium error term (residuals) from cointegration model
    threshold : float, optional
        Threshold value separating regimes. If None, uses 0 as threshold.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing M-TAR test results:
        - 'asymmetric': bool indicating if adjustment is asymmetric
        - 'p_value': p-value for asymmetry test
        - 'adjustment_positive': adjustment speed for positive momentum
        - 'adjustment_negative': adjustment speed for negative momentum
        - 'half_life_positive': half-life for positive momentum deviations
        - 'half_life_negative': half-life for negative momentum deviations
        - 'test_statistic': F-statistic for equality of adjustment speeds
        - 'interpretation': Text interpretation of results
    """
    import statsmodels.api as sm
    
    # Set default threshold if not provided
    if threshold is None:
        threshold = DEFAULT_MTAR_THRESHOLD
        logger.info(f"No threshold provided, using {threshold} as default")
    
    # Convert to numpy array for easier manipulation
    if isinstance(residuals, pd.Series) or isinstance(residuals, pd.DataFrame):
        residuals = residuals.values.flatten()
    
    # Calculate changes in residuals (momentum)
    d_residuals = np.diff(residuals)
    
    # Prepare data for the model
    y = np.diff(residuals[1:])  # Second difference for dependent variable
    X_positive = np.zeros_like(d_residuals[:-1])
    X_negative = np.zeros_like(d_residuals[:-1])
    
    # Split observations by momentum direction
    positive_mask = d_residuals[:-1] > threshold
    negative_mask = d_residuals[:-1] <= threshold
    
    X_positive[positive_mask] = residuals[1:-1][positive_mask]  # Level of residuals for regime classification
    X_negative[negative_mask] = residuals[1:-1][negative_mask]
    
    # Create design matrix
    X = np.column_stack([X_positive, X_negative])
    
    # Add constant (intercept)
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Extract adjustment speeds
    adjustment_positive = model.params[1]
    adjustment_negative = model.params[2]
    
    # Test for equality of parameters (H₀: ρ₁ = ρ₂)
    restriction = np.zeros((1, model.params.shape[0]))
    restriction[0, 1] = 1
    restriction[0, 2] = -1
    r_matrix = restriction
    q_value = 0
    
    wald_test = model.wald_test((r_matrix, q_value), scalar=True)
    
    # Calculate half-lives
    if adjustment_positive >= 0:
        half_life_positive = np.inf
    else:
        half_life_positive = np.log(0.5) / np.log(1 + adjustment_positive)
        
    if adjustment_negative >= 0:
        half_life_negative = np.inf
    else:
        half_life_negative = np.log(0.5) / np.log(1 + adjustment_negative)
    
    # Calculate counts
    n_positive = np.sum(positive_mask)
    n_negative = np.sum(negative_mask)
    
    # Prepare interpretation
    interpretation = _interpret_mtar_test(
        wald_test.pvalue < DEFAULT_ALPHA, 
        adjustment_positive, 
        adjustment_negative
    )
    
    # Prepare results
    results = {
        'asymmetric': wald_test.pvalue < DEFAULT_ALPHA,
        'p_value': wald_test.pvalue,
        'adjustment_positive': adjustment_positive,
        'adjustment_negative': adjustment_negative,
        'half_life_positive': half_life_positive,
        'half_life_negative': half_life_negative,
        'test_statistic': wald_test.statistic,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'threshold': threshold,
        'interpretation': interpretation
    }
    
    return results




@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def _interpret_adjustment_speeds(adj_below: float, adj_above: float, market1_name: str, market2_name: str) -> str:
    """
    Interpret adjustment speed coefficients in Yemen market context.
    
    Parameters
    ----------
    adj_below : float
        Adjustment coefficient below threshold
    adj_above : float
        Adjustment coefficient above threshold
    market1_name : str
        Name of first market
    market2_name : str
        Name of second market
        
    Returns
    -------
    str
        Interpretation of adjustment dynamics
    """
    # Correct sign for interpretation (negative coefficients mean adjustment)
    adj_below_correct = adj_below < 0
    adj_above_correct = adj_above < 0
    
    # Calculate magnitude comparison
    if adj_below_correct and adj_above_correct:
        faster_regime = 'above' if abs(adj_above) > abs(adj_below) else 'below'
        speed_ratio = abs(adj_above) / abs(adj_below) if abs(adj_below) > 0 else float('inf')
    else:
        faster_regime = 'above' if adj_above_correct else 'below' if adj_below_correct else None
        speed_ratio = np.nan
    
    # Generate interpretation
    if not adj_below_correct and not adj_above_correct:
        return f"No price adjustment detected between {market1_name} and {market2_name}. Both markets appear to be isolated from each other."
    
    elif not adj_below_correct and adj_above_correct:
        return (f"Asymmetric adjustment detected: {market1_name} adjusts to price differentials only when they exceed "
                f"the threshold, suggesting significant transaction costs prevent adjustment for small differentials.")
    
    elif adj_below_correct and not adj_above_correct:
        return (f"Unusual adjustment pattern: {market1_name} adjusts to small price differentials but not to large ones. "
                f"This could indicate data issues or structural breaks.")
    
    else:  # Both coefficients correct sign
        if speed_ratio > 1.5:
            return (f"Strong threshold effect: {market1_name} adjusts {speed_ratio:.1f}x faster when price differentials "
                    f"exceed the threshold, indicating profitable arbitrage above transaction costs.")
        elif speed_ratio > 1.1:
            return (f"Moderate threshold effect: {market1_name} adjusts {speed_ratio:.1f}x faster when price differentials "
                    f"exceed the threshold, suggesting some transaction cost barriers.")
        else:
            return (f"Weak threshold effect: {market1_name} adjusts at similar speeds regardless of price differential "
                    f"magnitude, suggesting relatively efficient market integration.")


@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def _interpret_asymmetric_test(is_asymmetric: bool, adj_below: float, adj_above: float) -> str:
    """
    Interpret asymmetric adjustment test results.
    
    Parameters
    ----------
    is_asymmetric : bool
        Whether adjustment is significantly asymmetric
    adj_below : float
        Adjustment coefficient below threshold
    adj_above : float
        Adjustment coefficient above threshold
        
    Returns
    -------
    str
        Interpretation of test results
    """
    if not is_asymmetric:
        return ("No significant asymmetry detected. Price adjustment appears to be symmetric "
                "regardless of whether differentials are above or below the threshold.")
    
    # Calculate magnitude comparison
    faster_regime = 'above' if abs(adj_above) > abs(adj_below) else 'below'
    
    if faster_regime == 'above':
        return ("Significant asymmetric adjustment detected. Prices adjust faster when "
                "differentials exceed the threshold, consistent with profitable arbitrage "
                "above transaction costs. This suggests effective but constrained market integration.")
    else:
        return ("Significant asymmetric adjustment detected, but with faster adjustment below "
                "the threshold. This unusual pattern may indicate measurement issues, structural "
                "breaks, or complex market dynamics requiring further investigation.")


@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def _interpret_mtar_test(is_asymmetric: bool, adj_positive: float, adj_negative: float) -> str:
    """
    Interpret M-TAR test results in Yemen market context.
    
    Parameters
    ----------
    is_asymmetric : bool
        Whether adjustment is significantly asymmetric
    adj_positive : float
        Adjustment coefficient for positive momentum
    adj_negative : float
        Adjustment coefficient for negative momentum
        
    Returns
    -------
    str
        Interpretation of test results
    """
    if not is_asymmetric:
        return ("No significant momentum-based asymmetry detected. Prices adjust at similar rates "
                "regardless of whether they are increasing or decreasing.")
    
    # Check adjustment direction (should be negative for error correction)
    adj_positive_correct = adj_positive < 0
    adj_negative_correct = adj_negative < 0
    
    # Calculate magnitude comparison
    if adj_positive_correct and adj_negative_correct:
        faster_regime = 'increasing' if abs(adj_positive) > abs(adj_negative) else 'decreasing'
        ratio = abs(adj_positive) / abs(adj_negative) if abs(adj_negative) > 0 else float('inf')
        
        if faster_regime == 'increasing':
            return (f"Significant momentum-based asymmetry detected. Price adjustment is {ratio:.2f}x "
                    f"faster when prices are rising than when they are falling. This could indicate "
                    f"retailers quickly pass cost increases to consumers but delay passing savings.")
        else:
            return (f"Significant momentum-based asymmetry detected. Price adjustment is {ratio:.2f}x "
                    f"faster when prices are falling than when they are rising. This could indicate "
                    f"strong competition or government intervention encouraging price reductions.")
    
    elif adj_positive_correct and not adj_negative_correct:
        return ("Unusual asymmetric pattern: Prices adjust only when rising but not when falling. "
                "This could indicate markets where price increases trigger corrective action, "
                "but decreases persist without intervention.")
    
    elif not adj_positive_correct and adj_negative_correct:
        return ("Unusual asymmetric pattern: Prices adjust only when falling but not when rising. "
                "This could indicate downward price pressure or oversupply conditions.")
    
    else:
        return ("No effective price adjustment mechanism detected. Neither rising nor falling "
                "prices trigger correction, suggesting severely fragmented markets.")

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _interpret_three_regime_tar(result: Dict[str, Any]) -> str:
    """
    Generate interpretation of three-regime TAR model results.
    
    Parameters
    ----------
    result : dict
        Three-regime TAR model results
        
    Returns
    -------
    str
        Text interpretation of results
    """
    thresholds = result['thresholds']
    half_lives = result['half_lives']
    adjustment_speeds = result['adjustment_speeds']
    threshold_effect = result['threshold_effect']
    
    # Initialize interpretation
    parts = []
    
    # Interpret threshold effect
    if threshold_effect:
        parts.append(
            f"The three-regime threshold model is statistically significant (p={result['p_value']:.4f}), "
            f"indicating nonlinear price adjustment dynamics with thresholds at {thresholds[0]:.4f} and {thresholds[1]:.4f}."
        )
    else:
        parts.append(
            f"The three-regime threshold model does not show a statistically significant improvement "
            f"over a linear model (p={result['p_value']:.4f}), suggesting that nonlinear adjustment "
            f"dynamics are not strongly present."
        )
    
    # Interpret regime observations
    n_obs = result['n_obs']
    parts.append(
        f"The data is distributed across regimes as follows: {n_obs['lower']} observations ({n_obs['lower']/n_obs['total']:.1%}) "
        f"in the lower regime, {n_obs['middle']} observations ({n_obs['middle']/n_obs['total']:.1%}) in the middle regime, "
        f"and {n_obs['upper']} observations ({n_obs['upper']/n_obs['total']:.1%}) in the upper regime."
    )
    
    # Interpret adjustment speeds
    if threshold_effect:
        # Compare adjustment speeds across regimes
        if adjustment_speeds['upper'] > adjustment_speeds['middle'] > adjustment_speeds['lower']:
            parts.append(
                f"Adjustment speeds increase monotonically across regimes (lower: {adjustment_speeds['lower']:.3f}, "
                f"middle: {adjustment_speeds['middle']:.3f}, upper: {adjustment_speeds['upper']:.3f}), indicating "
                f"that price convergence accelerates as the threshold variable increases. This pattern is consistent "
                f"with market arbitrage becoming more active as price differentials exceed transaction cost barriers."
            )
        elif adjustment_speeds['upper'] < adjustment_speeds['middle'] < adjustment_speeds['lower']:
            parts.append(
                f"Adjustment speeds decrease monotonically across regimes (lower: {adjustment_speeds['lower']:.3f}, "
                f"middle: {adjustment_speeds['middle']:.3f}, upper: {adjustment_speeds['upper']:.3f}), showing "
                f"an unusual pattern where price convergence slows as the threshold variable increases. This may "
                f"indicate structural impediments to arbitrage in high-threshold situations, possibly due to severe "
                f"conflict barriers."
            )
        else:
            # Non-monotonic pattern
            fastest = max(adjustment_speeds.items(), key=lambda x: x[1])[0]
            slowest = min(adjustment_speeds.items(), key=lambda x: x[1])[0]
            
            parts.append(
                f"Adjustment speeds show a non-monotonic pattern across regimes (lower: {adjustment_speeds['lower']:.3f}, "
                f"middle: {adjustment_speeds['middle']:.3f}, upper: {adjustment_speeds['upper']:.3f}), with "
                f"the fastest adjustment in the {fastest} regime and slowest in the {slowest} regime. This suggests "
                f"complex market dynamics where different factors affect price transmission in each regime."
            )
    
    # Interpret half-lives
    parts.append(
        f"The estimated half-lives for shocks are {half_lives['lower']:.1f} periods in the lower regime, "
        f"{half_lives['middle']:.1f} periods in the middle regime, and {half_lives['upper']:.1f} periods "
        f"in the upper regime, providing a quantitative measure of market integration strength in each regime."
    )
    
    # Add Yemen-specific context
    parts.append(
        "In Yemen's conflict context, these regimes likely reflect how different intensities of barriers "
        "(checkpoints, conflict zones, exchange rate differentials) affect market integration. The thresholds "
        "represent critical levels where price transmission behavior changes significantly."
    )
    
    # Combine all parts
    interpretation = " ".join(parts)
    
    return interpretation

@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def _assess_market_integration(asymm_adj: Dict[str, Any], cointegrated: bool) -> str:
    """
    Assess market integration based on cointegration and adjustment speeds.
    
    Provides a qualitative assessment of market integration status based
    on threshold cointegration results, interpreted in Yemen's context.
    
    Parameters
    ----------
    asymm_adj : dict
        Results from asymmetric adjustment analysis
    cointegrated : bool
        Whether markets are cointegrated
        
    Returns
    -------
    str
        Market integration assessment
    """
    if not cointegrated:
        return "No Market Integration: Markets lack a long-run equilibrium relationship, indicating complete market isolation."
    
    # Extract key metrics
    half_life_below = asymm_adj.get('half_life_below_1', np.inf)
    half_life_above = asymm_adj.get('half_life_above_1', np.inf)
    adj_ratio = asymm_adj.get('adjustment_ratio_1', np.nan)
    
    # Check if adjustment is significant
    sig_adj_below = asymm_adj.get('significant_adjustment_below_1', False)
    sig_adj_above = asymm_adj.get('significant_adjustment_above_1', False)
    
    if not sig_adj_below and not sig_adj_above:
        return "Fragmented Markets: Despite cointegration, no significant price adjustment detected, suggesting severe barriers to trade."
    
    elif not sig_adj_below and sig_adj_above:
        if half_life_above < 3:
            return "Threshold-Limited Integration: Price adjustment occurs only for large differentials, but is rapid once triggered. Indicates significant but surmountable transaction costs."
        else:
            return "Weak Threshold Integration: Price adjustment occurs only for large differentials and is slow. Suggests high conflict-induced barriers."
    
    elif sig_adj_below and not sig_adj_above:
        return "Anomalous Integration Pattern: Adjustment only for small price differentials, suggesting possible data issues or structural breaks."
    
    else:  # Both significant
        if adj_ratio > 2:
            if half_life_above < 3:
                return "Strong Threshold Integration: Rapid adjustment when differentials exceed transaction costs, with much slower adjustment below threshold."
            else:
                return "Moderate Threshold Integration: Faster adjustment above threshold, but still relatively slow, indicating persistent barriers."
        else:
            if half_life_below < 6 and half_life_above < 6:
                return "Strong Market Integration: Rapid adjustment regardless of price differential magnitude, indicating low transaction costs."
            else:
                return "Moderate Market Integration: Similar but generally slow adjustment speeds, suggesting persistent barriers affecting all price levels."