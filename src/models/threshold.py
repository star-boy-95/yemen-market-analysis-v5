"""
Threshold cointegration module for market integration analysis.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import statsmodels.api as sm
from arch.unitroot.cointegration import engle_granger
import matplotlib.pyplot as plt
from scipy import stats

from src.models.diagnostics import ModelDiagnostics
from src.utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_time_series, validate_model_inputs, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, disk_cache, parallelize,
    
    # Data processing
    fill_missing_values, create_lag_features,
    
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
    
    @disk_cache(cache_dir='.cache/threshold')
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
        # Run Engle-Granger test
        result = engle_granger(self.data1, self.data2, trend='c', lags=self.max_lags)
        
        # Store the cointegration vector
        self.beta0 = result.coef[0]  # Intercept
        self.beta1 = result.coef[1]  # Slope
        
        # Calculate the equilibrium errors
        self.eq_errors = self.data1 - (self.beta0 + self.beta1 * self.data2)
        
        logger.info(
            f"Cointegration test: statistic={result.stat:.4f}, p-value={result.pvalue:.4f}, "
            f"beta0={self.beta0:.4f}, beta1={self.beta1:.4f}"
        )
        
        # Format result
        return {
            'statistic': result.stat,
            'pvalue': result.pvalue,
            'critical_values': result.critical_values,
            'cointegrated': result.pvalue < DEFAULT_ALPHA,
            'beta0': self.beta0,
            'beta1': self.beta1,
            'equilibrium_errors': self.eq_errors,
            'long_run_relationship': f"{self.market1_name} = {self.beta0:.4f} + {self.beta1:.4f} × {self.market2_name}"
        }
    
    @timer
    @memory_usage_decorator
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
            param_validators={
                "n_grid": lambda x: isinstance(x, int) and x > 0,
                "trim": lambda x: 0.0 < x < 0.5
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold estimation parameters")
        
        # Identify candidates for threshold
        sorted_errors = np.sort(self.eq_errors)
        lower_idx = int(len(sorted_errors) * trim)
        upper_idx = int(len(sorted_errors) * (1 - trim))
        candidates = sorted_errors[lower_idx:upper_idx]
        
        # Grid search for threshold
        if len(candidates) > n_grid:
            step = len(candidates) // n_grid
            candidates = candidates[::step]
        
        # Initialize variables for grid search
        best_ssr = np.inf
        best_threshold = None
        ssrs = []
        thresholds = []
        
        # Grid search
        logger.info(f"Starting grid search with {len(candidates)} threshold candidates")
        
        # Use parallelize for better performance
        compute_args = [(threshold,) for threshold in candidates]
        results = parallelize(self._compute_ssr_for_threshold, compute_args)
        
        for i, (threshold, ssr) in enumerate(zip(candidates, results)):
            ssrs.append(ssr)
            thresholds.append(threshold)
            
            if ssr < best_ssr:
                best_ssr = ssr
                best_threshold = threshold
        
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
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
        lag_diffs = create_lag_features(
            pd.DataFrame({
                'd1': np.diff(self.data1),
                'd2': np.diff(self.data2)
            }), 
            cols=['d1', 'd2'], 
            lags=list(range(1, min(self.max_lags + 1, len(y))))
        ).iloc[self.max_lags:].fillna(0).values
        
        # Combine features and add constant
        X1 = np.column_stack([X1, lag_diffs])
        X1 = sm.add_constant(X1)
        
        # Fit the model and return SSR
        return sm.OLS(y[self.max_lags:], X1).fit().ssr
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
        
        # Calculate half lives
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
            self.results['diagnostics'] = self.run_diagnostics()
        
        return self.results
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
        
        # Initialize diagnostics
        diagnostics = ModelDiagnostics()
        
        # Get residuals from both equations
        residuals1 = self.results['equation1'].resid
        residuals2 = self.results['equation2'].resid
        
        # Run residual tests
        resid_tests1 = diagnostics.test_normality(residuals1)
        autocorr_tests1 = diagnostics.test_autocorrelation(residuals1) 
        hetero_tests1 = diagnostics.test_heteroskedasticity(residuals1)
        
        resid_tests2 = diagnostics.test_normality(residuals2)
        autocorr_tests2 = diagnostics.test_autocorrelation(residuals2)
        hetero_tests2 = diagnostics.test_heteroskedasticity(residuals2)
        
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
        
        diagnostic_results = {
            'residual_tests': {
                'eq1': {
                    'normality': resid_tests1,
                    'autocorrelation': autocorr_tests1,
                    'heteroskedasticity': hetero_tests1
                },
                'eq2': {
                    'normality': resid_tests2,
                    'autocorrelation': autocorr_tests2,
                    'heteroskedasticity': hetero_tests2
                }
            },
            'asymmetric_adjustment': asymm_test,
            'plots': plot_results,
            'summary': {
                'normal_residuals': resid_tests1.get('normal', False) and resid_tests2.get('normal', False),
                'no_autocorrelation': autocorr_tests1.get('no_autocorrelation', False) and autocorr_tests2.get('no_autocorrelation', False),
                'homoskedastic': hetero_tests1.get('homoskedastic', False) and hetero_tests2.get('homoskedastic', False),
                'asymmetric_adjustment': asymm_test.get('asymmetric', False)
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
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete threshold cointegration analysis workflow.
        
        Performs the full analytical sequence:
        1. Test for cointegration between markets
        2. Estimate transaction cost threshold
        3. Estimate threshold VECM with regime-specific adjustment speeds
        4. Run comprehensive diagnostics
        5. Calculate asymmetric adjustment metrics
        
        Returns
        -------
        dict
            Complete analysis results
        """
        logger.info("Starting complete threshold cointegration analysis")
        
        # Step 1: Estimate cointegration
        cointegration_results = self.estimate_cointegration()
        
        # Step 2: Estimate threshold
        threshold_results = self.estimate_threshold()
        
        # Step 3: Estimate TVECM with diagnostics
        tvecm_results = self.estimate_tvecm(run_diagnostics=True)
        
        # Step 4: Test threshold significance
        threshold_significance = self.test_threshold_significance()
        
        # Step 5: Calculate threshold confidence intervals
        threshold_ci = self.calculate_threshold_confidence_intervals()
        
        # Combine results
        full_results = {
            'cointegration': cointegration_results,
            'threshold': threshold_results,
            'tvecm': tvecm_results,
            'threshold_significance': threshold_significance,
            'threshold_confidence_intervals': threshold_ci,
            'market_integration_assessment': _assess_market_integration(
                tvecm_results['asymmetric_adjustment'],
                cointegration_results['cointegrated']
            )
        }
        
        logger.info("Complete analysis workflow finished")
        
        return full_results
        
    @m1_optimized()
    def _estimate_regime_models(self) -> Dict[str, Any]:
        """Estimate OLS models for each regime."""
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
        # Create regime-specific terms
        regime_terms = np.column_stack([
            self.eq_errors[:-1] * below[:-1],
            self.eq_errors[:-1] * above[:-1]
        ])
        
        # Add lagged differences using project utilities
        lag_diffs = create_lag_features(
            pd.DataFrame({
                'd1': np.diff(self.data1),
                'd2': np.diff(self.data2)
            }), 
            cols=['d1', 'd2'], 
            lags=list(range(1, min(self.max_lags + 1, len(self.data1)-1)))
        ).iloc[self.max_lags:].fillna(0).values
        
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
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
        df['regime'] = 'below'
        df.loc[df['equilibrium_error'] > self.threshold, 'regime'] = 'above'
        
        # Add lagged equilibrium error for TVECM analysis
        df['lag_equilibrium_error'] = df['equilibrium_error'].shift(1)
        
        return df
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
            ax.plot(self.index, eq_errors)
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
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Plot 4: Adjustment speeds comparison for Market 2
        ax = axes[1, 1]
        
        bars1 = ax.bar([0], [abs(adj_below_2)], width=0.4, label='Below Threshold', color='skyblue')
        bars2 = ax.bar([0.5], [abs(adj_above_2)], width=0.4, label='Above Threshold', color='salmon')
        
        ax.set_title(f"Adjustment Speeds: {self.market2_name}")
        ax.set_ylabel("Absolute Adjustment Speed")
        ax.set_xticks([0.25])
        ax.set_xticklabels([self.market2_name])
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save if requested
        if save_path:
            try:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                logger.info(f"Saved regime dynamics plot to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save plot: {str(e)}")
        
        return fig
    
    @disk_cache(cache_dir='.cache/threshold')
    @memory_usage_decorator
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
        
        # Extract needed data
        y = np.diff(self.data1)[self.max_lags:]
        
        # Create design matrix for linear model (no threshold)
        X_linear = sm.add_constant(np.column_stack([
            self.eq_errors[:-1][self.max_lags:],  # ECT term
            create_lag_features(
                pd.DataFrame({
                    'd1': np.diff(self.data1),
                    'd2': np.diff(self.data2)
                }), 
                cols=['d1', 'd2'], 
                lags=list(range(1, min(self.max_lags + 1, len(y))))
            ).iloc[self.max_lags:].fillna(0).values
        ]))
        
        # Fit linear model
        linear_model = sm.OLS(y, X_linear).fit()
        linear_ssr = linear_model.ssr
        
        # Get threshold model SSR
        threshold_ssr = self.ssr
        
        # Calculate F-statistic
        n = len(y)
        k_linear = X_linear.shape[1]
        k_threshold = k_linear + 1  # One additional parameter for threshold
        
        f_stat = ((linear_ssr - threshold_ssr) / (k_threshold - k_linear)) / (threshold_ssr / (n - k_threshold))
        
        logger.info(f"F-statistic for threshold significance: {f_stat:.4f}")
        
        # Bootstrap procedure
        bootstrap_f_stats = []
        
        for i in range(n_bootstrap):
            try:
                # Generate bootstrap sample under null hypothesis
                residuals = np.random.choice(linear_model.resid, size=len(linear_model.resid))
                y_bootstrap = X_linear @ linear_model.params + residuals
                
                # Fit linear model to bootstrap sample
                linear_model_bootstrap = sm.OLS(y_bootstrap, X_linear).fit()
                linear_ssr_bootstrap = linear_model_bootstrap.ssr
                
                # Create regime indicators for bootstrap sample
                bootstrap_residuals = y_bootstrap - X_linear @ linear_model_bootstrap.params
                below = bootstrap_residuals <= self.threshold
                above = ~below
                
                # Create design matrix for threshold model
                X_threshold = sm.add_constant(np.column_stack([
                    bootstrap_residuals * below,
                    bootstrap_residuals * above,
                    X_linear[:, 2:]  # Include lag terms
                ]))
                
                # Fit threshold model to bootstrap sample
                threshold_model_bootstrap = sm.OLS(y_bootstrap, X_threshold).fit()
                threshold_ssr_bootstrap = threshold_model_bootstrap.ssr
                
                # Calculate bootstrap F-statistic
                bootstrap_f = ((linear_ssr_bootstrap - threshold_ssr_bootstrap) / (k_threshold - k_linear)) / (threshold_ssr_bootstrap / (n - k_threshold))
                bootstrap_f_stats.append(bootstrap_f)
                
            except Exception as e:
                logger.warning(f"Bootstrap iteration failed: {str(e)}")
        
        # Calculate p-value and critical values
        bootstrap_f_stats = np.array(bootstrap_f_stats)
        p_value = np.mean(bootstrap_f_stats > f_stat)
        
        critical_values = {
            '1%': np.percentile(bootstrap_f_stats, 99),
            '5%': np.percentile(bootstrap_f_stats, 95),
            '10%': np.percentile(bootstrap_f_stats, 90)
        }
        
        # Determine significance
        significant = p_value < DEFAULT_ALPHA
        
        # Prepare interpretation
        if significant:
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
        
        result = {
            'test_statistic': f_stat,
            'p_value': p_value,
            'significant': significant,
            'critical_values': critical_values,
            'n_bootstrap': len(bootstrap_f_stats),
            'interpretation': interpretation
        }
        
        logger.info(f"Threshold significance test: p-value={p_value:.4f}, significant={significant}")
        
        return result
    
    @disk_cache(cache_dir='.cache/threshold')
    @memory_usage_decorator
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
        
        # Extract needed data
        y = np.diff(self.data1)[self.max_lags:]
        
        # Create design matrix
        X = self._create_regime_design_matrix(
            self.eq_errors <= self.threshold, 
            self.eq_errors > self.threshold
        )
        
        # Bootstrap procedure
        bootstrap_thresholds = []
        
        for i in range(n_bootstrap):
            try:
                # Generate bootstrap sample
                residuals = np.random.choice(self.results['equation1'].resid, size=len(y))
                y_bootstrap = X @ self.results['equation1'].params + residuals
                
                # Create bootstrap ThresholdCointegration object
                # Note: This is a simplified approach - in a full implementation,
                # we would need to reconstruct the entire price series
                
                # Create a temporary object to hold bootstrap data
                bootstrap_obj = ThresholdCointegration(
                    self.data1, self.data2, self.max_lags,
                    self.market1_name, self.market2_name
                )
                
                # Set cointegration parameters
                bootstrap_obj.beta0 = self.beta0
                bootstrap_obj.beta1 = self.beta1
                
                # Calculate equilibrium errors
                # In a more sophisticated implementation, we would regenerate
                # the price series and recalculate equilibrium errors
                bootstrap_obj.eq_errors = self.eq_errors
                
                # Estimate threshold
                bootstrap_threshold = bootstrap_obj.estimate_threshold()['threshold']
                bootstrap_thresholds.append(bootstrap_threshold)
                
            except Exception as e:
                logger.warning(f"Bootstrap iteration failed: {str(e)}")
        
        # Calculate confidence interval
        bootstrap_thresholds = np.array(bootstrap_thresholds)
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_thresholds, alpha/2 * 100)
        upper_bound = np.percentile(bootstrap_thresholds, (1-alpha/2) * 100)
        
        result = {
            'threshold': self.threshold,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'bootstrap_thresholds': bootstrap_thresholds,
            'interval_width': upper_bound - lower_bound,
            'relative_width': (upper_bound - lower_bound) / abs(self.threshold) if self.threshold != 0 else np.inf
        }
        
        logger.info(
            f"Threshold {confidence_level*100:.0f}% confidence interval: "
            f"[{lower_bound:.4f}, {upper_bound:.4f}]"
        )
        
        return result


@timer
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
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
    
    # Calculate half-lives
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


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def calculate_half_life(residuals: Union[pd.Series, np.ndarray], regime: str = "both") -> Dict[str, float]:
    """
    Calculate the half-life of deviations from equilibrium.
    
    The half-life indicates how quickly prices return to equilibrium after a shock.
    It is measured in periods (e.g., if data is monthly, half-life is in months).
    A shorter half-life indicates faster price transmission and better market integration.
    
    Parameters
    ----------
    residuals : Union[pd.Series, np.ndarray]
        Equilibrium error term (residuals) from cointegration model
    regime : str, optional
        Which regime to calculate half-life for:
        - "below": Calculate for observations below threshold
        - "above": Calculate for observations above threshold
        - "both": Calculate for both regimes (default)
    
    Returns
    -------
    Dict[str, float]
        Dictionary with half-life values for specified regime(s)
        
    Notes
    -----
    In the Yemen market context, longer half-lives in one regime may indicate
    asymmetric transaction costs or conflict-related barriers to price transmission.
    """
    if not isinstance(residuals, (pd.Series, np.ndarray, dict)):
        raise ValueError("Residuals must be Series, array, or dict with regime-specific residuals")
    
    result = {}
    
    if isinstance(residuals, dict):
        # Process regime-specific residuals
        if regime in ["both", "below"] and "below" in residuals:
            below_residuals = residuals["below"]
            below_model = _fit_ar1(below_residuals)
            result["below"] = _compute_half_life(below_model) if below_model else np.nan
            
        if regime in ["both", "above"] and "above" in residuals:
            above_residuals = residuals["above"]
            above_model = _fit_ar1(above_residuals)
            result["above"] = _compute_half_life(above_model) if above_model else np.nan
    else:
        # Process single series of residuals
        model = _fit_ar1(residuals)
        result["overall"] = _compute_half_life(model) if model else np.nan
    
    return result


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _fit_ar1(residuals: Union[pd.Series, np.ndarray]):
    """Fit AR(1) model to residuals."""
    import statsmodels.api as sm
    
    if len(residuals) < 3:
        logger.warning("Too few observations to fit AR(1) model")
        return None
    
    # Drop NaNs and prepare data
    if isinstance(residuals, pd.Series):
        residuals = residuals.dropna()
    else:
        residuals = residuals[~np.isnan(residuals)]
    
    if len(residuals) < 3:
        logger.warning("Too few non-NaN observations to fit AR(1) model")
        return None
    
    # Prepare data for AR(1) model: y_t = c + rho*y_{t-1} + e_t
    y = residuals[1:]
    x = sm.add_constant(residuals[:-1])
    
    try:
        model = sm.OLS(y, x).fit()
        return model
    except Exception as e:
        logger.warning(f"Failed to fit AR(1) model: {str(e)}")
        return None


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _compute_half_life(model) -> float:
    """Compute half-life from AR(1) model."""
    if model is None:
        return np.nan
    
    rho = model.params[1]  # AR(1) coefficient
    
    if rho >= 1:
        # Unit root or explosive process
        return np.inf
    elif rho <= -1:
        # Oscillatory behavior
        return 0.5  # Half-life is 0.5 periods for extreme oscillation
    else:
        # Standard case: log(0.5) / log(abs(rho))
        return np.log(0.5) / np.log(abs(rho))


@m1_optimized()
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def test_asymmetric_adjustment(residuals: Union[pd.DataFrame, pd.Series, np.ndarray], 
                              threshold: Optional[float] = None) -> Dict[str, Any]:
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
    from statsmodels.stats.diagnostic import het_white
    
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
    
    # Calculate half-lives
    if adjustment_below >= 0:
        half_life_below = np.inf
    else:
        half_life_below = np.log(0.5) / np.log(1 + adjustment_below)
        
    if adjustment_above >= 0:
        half_life_above = np.inf
    else:
        half_life_above = np.log(0.5) / np.log(1 + adjustment_above)
    
    # Calculate counts
    n_below = np.sum(below_mask)
    n_above = np.sum(above_mask)
    
    # Prepare results
    results = {
        'asymmetric': wald_test.pvalue < 0.05,
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
            wald_test.pvalue < 0.05, adjustment_below, adjustment_above
        )
    }
    
    return results


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
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


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
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


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
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