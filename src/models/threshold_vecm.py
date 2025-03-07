"""
Threshold Vector Error Correction Model implementation.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM
from scipy import stats
import matplotlib.pyplot as plt

from src.models.diagnostics import ModelDiagnostics
from src.utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_dataframe, validate_time_series, validate_model_inputs, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, disk_cache, parallelize,
    
    # Data processing
    fill_missing_values, create_lag_features, normalize_columns,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values
DEFAULT_ALPHA = config.get('analysis.threshold_vecm.alpha', 0.05)
DEFAULT_TRIM = config.get('analysis.threshold_vecm.trim', 0.15)
DEFAULT_GRID = config.get('analysis.threshold_vecm.n_grid', 300)
DEFAULT_K_AR = config.get('analysis.threshold_vecm.k_ar_diff', 2)
DEFAULT_N_BOOTSTRAP = config.get('analysis.threshold_vecm.bootstrap_reps', 1000)


class ThresholdVECM:
    """
    Threshold Vector Error Correction Model (TVECM) implementation.
    
    This class implements a two-regime threshold VECM following
    Hansen & Seo (2002) methodology. It models nonlinear price adjustment
    dynamics where adjustment speeds differ depending on whether the
    equilibrium error (deviation from long-run equilibrium) is above
    or below a threshold value.
    
    In the Yemen market integration context, this threshold represents
    transaction costs between markets. Higher thresholds indicate greater
    barriers to trade due to conflict, exchange rate differentials, or
    other factors.
    """
    
    def __init__(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        k_ar_diff: int = DEFAULT_K_AR, 
        deterministic: str = "ci"
    ):
        """
        Initialize the TVECM model.
        
        Parameters
        ----------
        data : array_like or pandas DataFrame
            The endogenous variables (typically price series from different markets)
        k_ar_diff : int, optional
            Number of lagged differences in the model
        deterministic : str, optional
            "n" - no deterministic terms
            "co" - constant outside the cointegration relation
            "ci" - constant inside the cointegration relation (default)
            "lo" - linear trend outside the cointegration relation
            "li" - linear trend inside the cointegration relation
        """
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        else:
            self.data = data
        
        # Validate data
        self._validate_data()
        
        # Validate model parameters
        valid, errors = validate_model_inputs(
            model_name="tvecm",
            params={
                "k_ar_diff": k_ar_diff,
                "deterministic": deterministic
            },
            param_validators={
                "k_ar_diff": lambda x: isinstance(x, int) and x > 0,
                "deterministic": lambda x: x in ["n", "co", "ci", "lo", "li"]
            }
        )
        raise_if_invalid(valid, errors, "Invalid TVECM model parameters")
        
        self.k_ar_diff = k_ar_diff
        self.deterministic = deterministic
        self.results = None
        self.linear_model = None
        self.linear_results = None
        self.threshold = None
        self.llf = None
        self.eq_errors = None
        self.trim = DEFAULT_TRIM
        
        logger.info(
            f"Initialized ThresholdVECM with {self.data.shape[0]} observations, "
            f"{self.data.shape[1]} variables"
        )
    
    def _validate_data(self):
        """Validate input data."""
        # Check if DataFrame
        if not isinstance(self.data, pd.DataFrame):
            raise ValidationError("data must be a pandas DataFrame")
        
        # Check dimensions
        if self.data.shape[1] < 2:
            raise ValidationError(f"data must have at least 2 variables, got {self.data.shape[1]}")
        
        # Check for missing values
        if self.data.isnull().any().any():
            raise ValidationError("data contains missing values")
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def estimate_linear_vecm(self) -> Any:
        """
        Estimate the linear VECM model (no threshold).
        
        Returns
        -------
        statsmodels.tsa.vector_ar.vecm.VECMResults
            Linear VECM estimation results
        """
        logger.info(f"Estimating linear VECM with k_ar_diff={self.k_ar_diff}")
        
        # Initialize linear model
        self.linear_model = VECM(
            self.data, 
            k_ar_diff=self.k_ar_diff, 
            deterministic=self.deterministic
        )
        
        # Fit model and store results
        self.linear_results = self.linear_model.fit()
        
        logger.info(
            f"Linear VECM estimation complete: AIC={self.linear_results.aic:.4f}, "
            f"Log-likelihood={self.linear_results.llf:.4f}"
        )
        
        return self.linear_results
    
    @timer
    @memory_usage_decorator
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def grid_search_threshold(
        self, 
        trim: float = DEFAULT_TRIM, 
        n_grid: int = DEFAULT_GRID, 
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Perform grid search to find the optimal threshold.
        
        Parameters
        ----------
        trim : float, optional
            Trimming percentage (default: 0.15)
            Controls the range of threshold values to consider
        n_grid : int, optional
            Number of grid points (default: 300)
            Higher values give more precise threshold estimates
        verbose : bool, optional
            Whether to print progress
            
        Returns
        -------
        dict
            Threshold estimation results including:
            - threshold: Optimal threshold value
            - llf: Log-likelihood at optimal threshold
            - all_thresholds: All evaluated thresholds
            - all_llfs: Log-likelihoods for all thresholds
            
        Notes
        -----
        The threshold represents transaction costs in market integration context.
        Higher values indicate greater barriers to trade between markets.
        """
        # Ensure linear VECM is estimated
        if not hasattr(self, 'linear_results') or self.linear_results is None:
            logger.info("Estimating linear VECM first")
            self.estimate_linear_vecm()
        
        # Store trim value for later use
        self.trim = trim
        
        # Validate parameters
        self._validate_grid_search_params(trim, n_grid)
        
        # Get cointegration relation and calculate equilibrium errors
        self.eq_errors = self._calculate_equilibrium_errors()
        
        # Generate threshold candidates
        candidates = self._generate_threshold_candidates(self.eq_errors, trim, n_grid)
        
        # Perform grid search
        return self._perform_grid_search(self.eq_errors, candidates, verbose)
    
    def _validate_grid_search_params(self, trim: float, n_grid: int) -> None:
        """Validate grid search parameters."""
        valid, errors = validate_model_inputs(
            model_name="tvecm",
            params={"trim": trim, "n_grid": n_grid},
            param_validators={
                "trim": lambda x: 0.0 < x < 0.5,
                "n_grid": lambda x: isinstance(x, int) and x > 0
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold grid search parameters")
    
    def _calculate_equilibrium_errors(self) -> np.ndarray:
        """
        Calculate equilibrium errors from cointegration relation.
        
        Returns
        -------
        numpy.ndarray
            Equilibrium errors (deviations from long-run equilibrium)
            
        Notes
        -----
        These errors represent price deviations between markets.
        In market integration context, they indicate arbitrage opportunities
        when exceeding transaction costs (the threshold).
        """
        beta = self.linear_results.beta
        y = self.data.values
        
        if self.deterministic == "ci":
            z = np.column_stack([np.ones(len(y)), y])[:, :-1]
        else:
            z = y
        
        return z @ beta
    
    def _generate_threshold_candidates(
        self, 
        eq_errors: np.ndarray, 
        trim: float, 
        n_grid: int
    ) -> np.ndarray:
        """Generate threshold candidates within trim range."""
        sorted_errors = np.sort(eq_errors.flatten())
        lower_idx = int(len(sorted_errors) * trim)
        upper_idx = int(len(sorted_errors) * (1 - trim))
        candidates = sorted_errors[lower_idx:upper_idx]
        
        if len(candidates) > n_grid:
            step = len(candidates) // n_grid
            candidates = candidates[::step]
        
        return candidates
    
    def _perform_grid_search(
        self, 
        eq_errors: np.ndarray, 
        candidates: np.ndarray, 
        verbose: bool
    ) -> Dict[str, Any]:
        """Perform grid search across threshold candidates."""
        # Initialize variables for grid search
        best_llf = -np.inf
        best_threshold = None
        llfs = []
        thresholds = []
        
        # Grid search
        logger.info(f"Starting grid search with {len(candidates)} threshold candidates")
        
        # Use parallelize for better performance
        compute_args = [(threshold, eq_errors) for threshold in candidates]
        results = parallelize(self._compute_llf_for_threshold, compute_args, progress_bar=verbose)
        
        for i, (threshold, llf) in enumerate(zip(candidates, results)):
            llfs.append(llf)
            thresholds.append(threshold)
            
            if llf > best_llf:
                best_llf = llf
                best_threshold = threshold
        
        self.threshold = best_threshold
        self.llf = best_llf
        
        logger.info(f"Threshold grid search complete: threshold={best_threshold:.4f}, llf={best_llf:.4f}")
        
        return {
            'threshold': best_threshold,
            'llf': best_llf,
            'all_thresholds': thresholds,
            'all_llfs': llfs
        }
    
    @m1_optimized()
    def _compute_llf_for_threshold(
        self, 
        args: Tuple[float, np.ndarray]
    ) -> float:
        """
        Compute log-likelihood for a given threshold.
        
        Parameters
        ----------
        args : tuple
            (threshold, eq_errors) tuple
            
        Returns
        -------
        float
            Log-likelihood
        """
        threshold, eq_errors = args
        
        # Indicator function for regimes
        below = eq_errors <= threshold
        above = ~below
        
        # Prepare data
        y = np.diff(self.data.values, axis=0)
        
        # Create design matrix with project utilities
        X = self._create_regime_design_matrix(below, above, eq_errors)
        
        # Fit model and return likelihood
        return sm.OLS(y, X).fit().llf
    
    def _create_regime_design_matrix(
        self, 
        below: np.ndarray, 
        above: np.ndarray,
        eq_errors: np.ndarray
    ) -> np.ndarray:
        """Create design matrix for regime-specific estimation."""
        # Create regime-specific terms
        X_below = np.column_stack([
            np.ones(len(below)-1) * below[:-1],
            eq_errors[:-1] * below[:-1]
        ])
        
        X_above = np.column_stack([
            np.ones(len(above)-1) * above[:-1],
            eq_errors[:-1] * above[:-1]
        ])
        
        # Get lagged differences using project utilities
        y_diff = np.diff(self.data.values, axis=0)
        lag_df = pd.DataFrame(y_diff)
        
        lag_diffs = create_lag_features(
            lag_df,
            cols=lag_df.columns.tolist(),
            lags=list(range(1, min(self.k_ar_diff + 1, len(y_diff))))
        ).iloc[self.k_ar_diff:].fillna(0)
        
        # Apply regime indicators to lagged diffs
        lag_below = lag_diffs.values * below[:-1, np.newaxis][self.k_ar_diff:]
        lag_above = lag_diffs.values * above[:-1, np.newaxis][self.k_ar_diff:]
        
        # Combine matrices
        X = np.column_stack([
            X_below[self.k_ar_diff:], 
            X_above[self.k_ar_diff:],
            lag_below,
            lag_above
        ])
        
        return X
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def estimate_tvecm(self, run_diagnostics: bool = False) -> Dict[str, Any]:
        """
        Estimate the Threshold VECM.
        
        Parameters
        ----------
        run_diagnostics : bool, optional
            Whether to run model diagnostics after estimation
        
        Returns
        -------
        dict
            TVECM estimation results including:
            - threshold: Optimal threshold value
            - alpha_below: Adjustment coefficients below threshold
            - alpha_above: Adjustment coefficients above threshold
            - beta: Cointegrating vector
            - llf: Log-likelihood
            - model: Fitted model object
            - diagnostic_hooks: Integration points for diagnostics
            
        Notes
        -----
        In Yemen market integration, adjustment coefficients represent
        the speed at which prices converge after shocks. Significant
        differences between regimes indicate nonlinear price transmission
        due to conflict, exchange rate fragmentation, or other barriers.
        """
        # Ensure threshold is estimated
        if self.threshold is None:
            logger.info("Estimating threshold first")
            self.grid_search_threshold()
        
        # Calculate equilibrium errors if not already done
        if self.eq_errors is None:
            self.eq_errors = self._calculate_equilibrium_errors()
        
        # Estimate threshold model
        regime_model = self._estimate_regime_model(self.eq_errors)
        
        # Extract and format results
        self.results = self._format_model_results(regime_model, self.eq_errors)
        
        # Add diagnostic hooks for integration with diagnostic module
        self.results['diagnostic_hooks'] = {
            'eq_errors': self.eq_errors,
            'threshold': self.threshold,
            'model_type': 'tvecm',
            'n_regimes': 2,
            'regime_indicators': {
                'below': self.eq_errors <= self.threshold,
                'above': self.eq_errors > self.threshold
            }
        }
        
        logger.info(
            f"TVECM estimation complete: threshold={self.threshold:.4f}, "
            f"llf={self.results['llf']:.4f}"
        )
        
        # Run diagnostics if requested
        if run_diagnostics:
            diagnostic_results = self.run_diagnostics()
            self.results['diagnostics'] = diagnostic_results
        
        return self.results
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def _estimate_regime_model(self, eq_errors: np.ndarray) -> Any:
        """
        Estimate regime-specific models.
        
        Parameters
        ----------
        eq_errors : numpy.ndarray
            Equilibrium errors from cointegration relation
            
        Returns
        -------
        object
            Fitted model
        """
        # Indicator function for regimes
        below = eq_errors <= self.threshold
        above = ~below
        
        # Prepare data
        y = np.diff(self.data.values, axis=0)
        
        # Create design matrix
        X = self._create_regime_design_matrix(below, above, eq_errors)
        
        # Fit model
        model = sm.OLS(y, X)
        results = model.fit()
        
        return results
    
    def _format_model_results(self, model_results: Any, eq_errors: np.ndarray) -> Dict[str, Any]:
        """
        Format model results for easier interpretation.
        
        Parameters
        ----------
        model_results : object
            Fitted model results
        eq_errors : numpy.ndarray
            Equilibrium errors
            
        Returns
        -------
        dict
            Formatted TVECM results
        """
        # Extract parameters
        params = model_results.params
        
        # First parameters are adjustment speeds (alpha) for each regime
        n_vars = self.data.shape[1]
        
        # Extract adjustment coefficients for each regime
        alpha_below = np.zeros(n_vars)
        alpha_above = np.zeros(n_vars)
        
        # Format depends on how many variables we have
        for i in range(n_vars):
            # Constant term is at index 0, error correction term at index 1
            # For each variable, we have different location in the params vector
            alpha_below[i] = params[1 + i]
            alpha_above[i] = params[1 + n_vars + i]
        
        # Prepare results structure for half-life calculation
        half_life_input = {
            'alpha_below': alpha_below,
            'alpha_above': alpha_above
        }
        
        # Calculate half-lives with the dedicated function
        half_lives = calculate_half_lives(half_life_input)
        
        # For backward compatibility, also calculate with the old method
        rich_half_lives = calculate_half_lives(
            {'below_regime': {'alpha': alpha_below}, 
             'above_regime': {'alpha': alpha_above}}, 
            rich_output=True
        )
        
        # Extract cointegrating vector
        beta = self.linear_results.beta
        
        # Calculate proportion of observations in each regime
        prop_below = np.mean(eq_errors <= self.threshold)
        prop_above = 1 - prop_below
        
        # Calculate regime transition matrix
        transition_matrix = calculate_regime_transition_matrix(eq_errors, self.threshold)
        
        # Format results structure for simulation integration
        below_regime = {'alpha': alpha_below}
        above_regime = {'alpha': alpha_above}
        
        # Return formatted results
        return {
            'threshold': self.threshold,
            'alpha_below': alpha_below,
            'alpha_above': alpha_above,
            'beta': beta,
            'half_lives': half_lives,
            'rich_half_lives': rich_half_lives,
            'proportion_below': prop_below,
            'proportion_above': prop_above,
            'transition_matrix': transition_matrix,
            'llf': model_results.llf,
            'aic': model_results.aic,
            'bic': model_results.bic,
            'params': params,
            'model': model_results,
            'below_regime': below_regime,
            'above_regime': above_regime,
            'integration_assessment': _assess_market_integration(half_lives)
        }
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def run_diagnostics(self, 
                        plot: bool = True, 
                        save_plots: bool = False,
                        plot_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run diagnostic tests on the estimated TVECM model.
        
        Parameters
        ----------
        plot : bool, optional
            Whether to create diagnostic plots
        save_plots : bool, optional
            Whether to save diagnostic plots
        plot_dir : str, optional
            Directory to save plots if save_plots is True
            
        Returns
        -------
        dict
            Diagnostic test results
        """
        logger.info("Running diagnostics on TVECM model")
        
        # Ensure model is estimated
        if self.results is None:
            logger.warning("Model not estimated. Estimating TVECM first.")
            self.estimate_tvecm()
        
        # Extract residuals
        model_results = self.results['model']
        residuals = pd.DataFrame(model_results.resid, 
                                index=self.data.index[1:],
                                columns=[f'eq{i+1}' for i in range(self.data.shape[1])])
        
        # Create diagnostics instance
        diagnostics = ModelDiagnostics(
            residuals=residuals, 
            model_name="TVECM",
            original_data=self.data
        )
        
        # Run tests
        normality_results = diagnostics.test_normality()
        autocorr_results = diagnostics.test_autocorrelation(lags=min(10, len(residuals) // 5))
        hetero_results = diagnostics.test_heteroskedasticity()
        
        # Calculate appropriate window size for stability test
        window_size = max(20, int(len(self.data) * 0.2))
        stability_results = diagnostics.test_model_stability(window_size=window_size)
        
        # Create plots if requested
        plot_results = {}
        if plot:
            plot_results = diagnostics.plot_diagnostics(
                save=save_plots, 
                save_dir=plot_dir,
                plot_acf=True,
                plot_dist=True,
                plot_qq=True,
                plot_ts=True
            )
        
        # Compile results
        diagnostic_results = {
            'normality': normality_results,
            'autocorrelation': autocorr_results,
            'heteroskedasticity': hetero_results,
            'stability': stability_results,
            'plots': plot_results,
            'summary': {
                'residuals_mean': residuals.mean().to_dict(),
                'residuals_std': residuals.std().to_dict(),
                'residuals_jarque_bera': normality_results.get('p_value'),
                'residuals_autocorr': autocorr_results.get('p_value'),
                'residuals_hetero': hetero_results.get('p_value'),
                'model_stable': stability_results.get('stable', False)
            }
        }
        
        logger.info(f"TVECM diagnostics complete. JB p-value: {diagnostic_results['summary']['residuals_jarque_bera']:.4f}, "
                   f"ACF p-value: {diagnostic_results['summary']['residuals_autocorr']:.4f}")
        
        return diagnostic_results
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def run_full_analysis(self, 
                          trim: float = DEFAULT_TRIM, 
                          n_grid: int = DEFAULT_GRID,
                          verbose: bool = False,
                          run_diagnostics: bool = True,
                          plot: bool = True,
                          save_plots: bool = False,
                          plot_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete TVECM workflow: linear VECM, threshold search, TVECM, and diagnostics.
        
        Parameters
        ----------
        trim : float, optional
            Trimming percentage for threshold search
        n_grid : int, optional
            Number of grid points for threshold search
        verbose : bool, optional
            Whether to print progress
        run_diagnostics : bool, optional
            Whether to run model diagnostics
        plot : bool, optional
            Whether to create diagnostic plots
        save_plots : bool, optional
            Whether to save diagnostic plots
        plot_dir : str, optional
            Directory to save plots if save_plots is True
            
        Returns
        -------
        dict
            Complete analysis results
        """
        logger.info("Starting complete TVECM analysis workflow")
        
        # Step 1: Estimate linear VECM
        linear_results = self.estimate_linear_vecm()
        
        # Step 2: Search for optimal threshold
        threshold_results = self.grid_search_threshold(
            trim=trim, 
            n_grid=n_grid, 
            verbose=verbose
        )
        
        # Step 3: Estimate TVECM and run diagnostics if requested
        tvecm_results = self.estimate_tvecm(run_diagnostics=run_diagnostics)
        
        # Run diagnostics separately if not already done
        if run_diagnostics and 'diagnostics' not in tvecm_results:
            diagnostic_results = self.run_diagnostics(
                plot=plot,
                save_plots=save_plots,
                plot_dir=plot_dir
            )
            tvecm_results['diagnostics'] = diagnostic_results
        
        # Compile complete results
        full_results = {
            'linear_vecm': {
                'model': self.linear_model,
                'results': linear_results,
                'llf': linear_results.llf,
                'aic': linear_results.aic,
                'bic': linear_results.bic
            },
            'threshold_search': threshold_results,
            'tvecm': tvecm_results
        }
        
        logger.info("Complete TVECM analysis workflow finished successfully")
        
        return full_results
    
    @disk_cache(cache_dir='.cache/threshold_vecm')
    @memory_usage_decorator
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_threshold_significance(
        self, 
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP, 
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test for threshold effect using bootstrap methodology.
        
        Tests the null hypothesis of no threshold effect (linear VECM)
        against the alternative of a threshold effect (TVECM).
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap replications (default: 1000)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Test results including:
            - test_statistic: 2 * (llf_tvecm - llf_vecm)
            - p_value: Bootstrap p-value
            - significant: Whether threshold effect is significant
            - bootstrap_statistics: Distribution of test statistics
            - critical_values: Critical values at different significance levels
            
        Notes
        -----
        A significant threshold effect indicates that transaction costs
        create meaningful market fragmentation that prevents arbitrage
        when price differentials are below the threshold.
        """
        # Ensure both models are estimated
        if self.linear_results is None:
            self.estimate_linear_vecm()
        
        if self.results is None:
            self.estimate_tvecm()
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate test statistic: 2 * (llf_tvecm - llf_vecm)
        llf_vecm = self.linear_results.llf
        llf_tvecm = self.results['llf']
        test_statistic = 2 * (llf_tvecm - llf_vecm)
        
        logger.info(f"Original test statistic: {test_statistic:.4f}")
        
        # Bootstrap procedure
        bootstrap_stats = []
        converged = 0
        
        # Extract residuals from linear VECM
        vecm_residuals = self.linear_results.resid
        
        # Get original data
        y = self.data.values
        T, n = y.shape
        
        # Parameters from linear VECM
        beta = self.linear_results.beta  # Cointegrating vector
        alpha = self.linear_results.alpha  # Adjustment speeds
        gamma = self.linear_results.gamma  # Short-run dynamics
        
        for i in range(n_bootstrap):
            try:
                # Generate bootstrap sample
                bootstrap_sample = self._generate_bootstrap_sample(
                    vecm_residuals, beta, alpha, gamma
                )
                
                # Create DataFrame
                bootstrap_df = pd.DataFrame(
                    bootstrap_sample, 
                    index=self.data.index,
                    columns=self.data.columns
                )
                
                # Estimate linear VECM
                bootstrap_vecm = VECM(
                    bootstrap_df, 
                    k_ar_diff=self.k_ar_diff, 
                    deterministic=self.deterministic
                ).fit()
                
                # Estimate threshold VECM
                bootstrap_tvecm = ThresholdVECM(
                    bootstrap_df,
                    k_ar_diff=self.k_ar_diff,
                    deterministic=self.deterministic
                )
                bootstrap_tvecm.estimate_linear_vecm()
                threshold_results = bootstrap_tvecm.grid_search_threshold()
                tvecm_results = bootstrap_tvecm.estimate_tvecm()
                
                # Calculate test statistic
                bs_llf_vecm = bootstrap_vecm.llf
                bs_llf_tvecm = tvecm_results['llf']
                bs_test_stat = 2 * (bs_llf_tvecm - bs_llf_vecm)
                
                bootstrap_stats.append(bs_test_stat)
                converged += 1
                
                if i % 100 == 0 and i > 0:
                    logger.info(f"Completed {i} bootstrap replications")
                
            except Exception as e:
                logger.warning(f"Bootstrap replication {i} failed: {str(e)}")
                continue
        
        # Calculate p-value: proportion of bootstrap stats > original stat
        bootstrap_stats = np.array(bootstrap_stats)
        p_value = np.mean(bootstrap_stats > test_statistic)
        
        # Calculate critical values
        critical_values = {
            "1%": np.percentile(bootstrap_stats, 99),
            "5%": np.percentile(bootstrap_stats, 95),
            "10%": np.percentile(bootstrap_stats, 90)
        }
        
        # Determine significance
        significant = p_value < DEFAULT_ALPHA
        
        logger.info(
            f"Threshold significance test complete: test_statistic={test_statistic:.4f}, "
            f"p_value={p_value:.4f}, significant={significant}"
        )
        
        return {
            'test_statistic': test_statistic,
            'p_value': p_value,
            'significant': significant,
            'bootstrap_statistics': bootstrap_stats,
            'critical_values': critical_values,
            'n_bootstrap': converged,
            'convergence_rate': converged / n_bootstrap if n_bootstrap > 0 else 0,
            'interpretation': _interpret_threshold_test(p_value, DEFAULT_ALPHA)
        }
    
    @m1_optimized()
    def _generate_bootstrap_sample(
        self, 
        residuals: np.ndarray, 
        beta: np.ndarray, 
        alpha: np.ndarray, 
        gamma: np.ndarray
    ) -> np.ndarray:
        """
        Generate bootstrap sample for threshold significance test.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residuals from linear VECM
        beta : numpy.ndarray
            Cointegrating vector
        alpha : numpy.ndarray
            Adjustment speeds
        gamma : numpy.ndarray
            Short-run dynamics
            
        Returns
        -------
        numpy.ndarray
            Bootstrap sample
        """
        # Get dimensions
        T, n = residuals.shape
        
        # Draw random residuals with replacement
        bootstrap_indices = np.random.randint(0, T, size=T)
        bootstrap_residuals = residuals[bootstrap_indices]
        
        # Initialize bootstrap sample with actual first k_ar_diff+1 observations
        bootstrap_sample = np.zeros((T, n))
        bootstrap_sample[:2] = self.data.values[:2]  # Use original initial values
        
        # Generate bootstrap sample recursively
        for t in range(2, T):
            # Previous period value
            y_prev = bootstrap_sample[t-1]
            
            # First difference
            dy = np.zeros(n)
            
            # Add equilibrium correction
            if self.deterministic == 'ci':
                Z = np.concatenate([[1], y_prev]) @ beta
            else:
                Z = y_prev @ beta
            dy += alpha * Z
            
            # Add lagged differences
            for i in range(1, min(gamma.shape[0] + 1, t)):
                if t-i >= 0:
                    dy_lag = bootstrap_sample[t-i] - bootstrap_sample[t-i-1]
                    gamma_slice = gamma[i-1] if gamma.ndim == 2 else gamma
                    dy += np.dot(gamma_slice.reshape(n, n), dy_lag)
            
            # Add residual
            dy += bootstrap_residuals[t-2]
            
            # Update bootstrap sample
            bootstrap_sample[t] = bootstrap_sample[t-1] + dy
        
        return bootstrap_sample

    @disk_cache(cache_dir='.cache/threshold_vecm')
    @memory_usage_decorator
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_asymmetric_adjustment(self) -> Dict[str, Any]:
        """
        Test for asymmetric adjustment speeds between regimes.
        
        Tests the null hypothesis of symmetric adjustment:
        H0: alpha_below = alpha_above
        against the alternative of asymmetric adjustment:
        H1: alpha_below ≠ alpha_above
        
        Returns
        -------
        dict
            Test results including:
            - test_statistics: Test statistics for each variable
            - p_values: P-values for each test
            - significant: Whether asymmetry is significant for each variable
            - faster_regime: Which regime has faster adjustment for each variable
            
        Notes
        -----
        Asymmetric adjustment in Yemen markets may indicate:
        - Faster response to price increases than decreases
        - Different barriers to trade in different directions
        - Political or economic factors affecting arbitrage
        """
        # Ensure model is estimated
        if self.results is None:
            self.estimate_tvecm()
        
        # Extract adjustment speeds
        alpha_below = self.results['alpha_below']
        alpha_above = self.results['alpha_above']
        
        # Calculate differences
        differences = alpha_below - alpha_above
        
        # Get number of variables
        n_vars = len(alpha_below)
        
        # Calculate standard errors
        # Note: This is a simplification; ideally we'd extract from covariance matrix
        # but this requires knowing the position in the params vector
        model = self.results['model']
        cov_matrix = model.cov_params()
        params = model.params
        
        # Initialize results
        test_statistics = np.zeros(n_vars)
        p_values = np.zeros(n_vars)
        significant = np.zeros(n_vars, dtype=bool)
        faster_regime = [""] * n_vars
        
        # Calculate test statistics and p-values for each variable
        for i in range(n_vars):
            # Positions in the params vector (may need adjustment based on model structure)
            pos_below = 1 + i  # Position of alpha_below in params
            pos_above = 1 + n_vars + i  # Position of alpha_above in params
            
            # Calculate standard error of difference
            if cov_matrix.shape[0] > max(pos_below, pos_above):
                # Get variance of difference
                var_diff = (
                    cov_matrix[pos_below, pos_below] + 
                    cov_matrix[pos_above, pos_above] - 
                    2 * cov_matrix[pos_below, pos_above]
                )
                se_diff = np.sqrt(var_diff)
                
                # Calculate t-statistic
                test_statistics[i] = differences[i] / se_diff
                
                # Calculate p-value (two-sided test)
                p_values[i] = 2 * (1 - stats.t.cdf(abs(test_statistics[i]), model.df_resid))
                
                # Determine significance
                significant[i] = p_values[i] < DEFAULT_ALPHA
                
                # Determine which regime has faster adjustment
                faster_regime[i] = "below" if abs(alpha_below[i]) > abs(alpha_above[i]) else "above"
            else:
                logger.warning(f"Could not calculate test statistic for variable {i}: index out of bounds")
                test_statistics[i] = np.nan
                p_values[i] = np.nan
                significant[i] = False
                faster_regime[i] = "unknown"
        
        # Compile results
        results = {
            'test_statistics': test_statistics,
            'p_values': p_values,
            'significant': significant,
            'faster_regime': faster_regime,
            'alpha_below': alpha_below,
            'alpha_above': alpha_above,
            'differences': differences
        }
        
        # Add summary for any significant asymmetry
        results['any_significant'] = np.any(significant)
        
        # Log results
        logger.info(
            f"Asymmetric adjustment test: any significant asymmetry={results['any_significant']}, "
            f"significant variables={np.sum(significant)}/{n_vars}"
        )
        
        return results
    
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def export_equilibrium_errors(self) -> pd.Series:
        """
        Export equilibrium errors for further analysis.
        
        Returns
        -------
        pandas.Series
            Equilibrium errors with proper time index
            
        Notes
        -----
        These errors can be used for:
        - Asymmetric adjustment testing
        - Diagnostic plotting
        - Integration with external diagnostic tools
        - Analyzing arbitrage opportunities in market integration
        """
        # Ensure model is estimated
        if self.eq_errors is None:
            if self.linear_results is None:
                self.estimate_linear_vecm()
            self.eq_errors = self._calculate_equilibrium_errors()
        
        # Create Series with proper time index
        eq_errors_series = pd.Series(
            self.eq_errors.flatten(),
            index=self.data.index
        )
        
        return eq_errors_series
    
    @disk_cache(cache_dir='.cache/threshold_vecm')
    @memory_usage_decorator
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def calculate_threshold_confidence_intervals(
        self, 
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        confidence_level: float = 0.95,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate confidence intervals for threshold parameter.
        
        Uses bootstrap method to estimate confidence intervals.
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap replications
        confidence_level : float, optional
            Confidence level (e.g., 0.95 for 95% confidence)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Confidence interval results including:
            - threshold: Original threshold estimate
            - lower_bound: Lower confidence bound
            - upper_bound: Upper confidence bound
            - bootstrap_thresholds: All bootstrap threshold estimates
            - confidence_level: Specified confidence level
            
        Notes
        -----
        Wide confidence intervals may indicate uncertainty in
        transaction cost estimates, potentially due to market
        volatility or data quality issues.
        """
        # Ensure model is estimated
        if self.results is None:
            self.estimate_tvecm()
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Extract original threshold
        threshold = self.threshold
        
        # Extract residuals from TVECM
        residuals = self.results['model'].resid
        
        # Extract model parameters
        beta = self.results['beta']
        alpha_below = self.results['alpha_below']
        alpha_above = self.results['alpha_above']
        
        # Get original data
        y = self.data.values
        T, n = y.shape
        
        # Bootstrap procedure
        bootstrap_thresholds = []
        converged = 0
        
        for i in range(n_bootstrap):
            try:
                # Generate bootstrap sample
                bootstrap_indices = np.random.randint(0, len(residuals), size=len(residuals))
                bootstrap_residuals = residuals[bootstrap_indices]
                
                # Create bootstrap data
                bootstrap_data = self._generate_bootstrap_sample_with_threshold(
                    bootstrap_residuals, beta, alpha_below, alpha_above
                )
                
                # Create DataFrame
                bootstrap_df = pd.DataFrame(
                    bootstrap_data,
                    index=self.data.index,
                    columns=self.data.columns
                )
                
                # Estimate threshold model
                bootstrap_model = ThresholdVECM(
                    bootstrap_df,
                    k_ar_diff=self.k_ar_diff,
                    deterministic=self.deterministic
                )
                bootstrap_model.estimate_linear_vecm()
                threshold_results = bootstrap_model.grid_search_threshold()
                
                # Store bootstrap threshold
                bootstrap_thresholds.append(threshold_results['threshold'])
                converged += 1
                
                if i % 100 == 0 and i > 0:
                    logger.info(f"Completed {i} bootstrap replications for CI")
                
            except Exception as e:
                logger.warning(f"Bootstrap replication {i} failed: {str(e)}")
                continue
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_thresholds, alpha/2 * 100)
        upper_bound = np.percentile(bootstrap_thresholds, (1-alpha/2) * 100)
        
        logger.info(
            f"Threshold confidence interval: {lower_bound:.4f} - {upper_bound:.4f} "
            f"(level={confidence_level:.2f})"
        )
        
        return {
            'threshold': threshold,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'bootstrap_thresholds': bootstrap_thresholds,
            'confidence_level': confidence_level,
            'n_bootstrap': converged,
            'interval_width': upper_bound - lower_bound,
            'relative_width': (upper_bound - lower_bound) / abs(threshold) if threshold != 0 else np.inf
        }
    
    @m1_optimized()
    def _generate_bootstrap_sample_with_threshold(
        self, 
        residuals: np.ndarray, 
        beta: np.ndarray, 
        alpha_below: np.ndarray, 
        alpha_above: np.ndarray
    ) -> np.ndarray:
        """
        Generate bootstrap sample for threshold confidence interval.
        
        Parameters
        ----------
        residuals : numpy.ndarray
            Residuals from TVECM
        beta : numpy.ndarray
            Cointegrating vector
        alpha_below : numpy.ndarray
            Adjustment speeds below threshold
        alpha_above : numpy.ndarray
            Adjustment speeds above threshold
            
        Returns
        -------
        numpy.ndarray
            Bootstrap sample
        """
        # Get dimensions
        T, n = self.data.shape
        
        # Initialize bootstrap sample with actual first k_ar_diff+1 observations
        bootstrap_sample = np.copy(self.data.values)
        
        # Generate bootstrap sample recursively
        for t in range(self.k_ar_diff + 1, T):
            # Previous period value
            y_prev = bootstrap_sample[t-1]
            
            # First difference
            dy = np.zeros(n)
            
            # Calculate equilibrium error
            z_prev = np.concatenate([np.ones(1), y_prev])  # Add constant if needed
            eq_error = np.dot(z_prev[:-1], beta)
            
            # Add equilibrium correction based on regime
            if eq_error <= self.threshold:
                dy += alpha_below * eq_error
            else:
                dy += alpha_above * eq_error
            
            # Add residual
            if t-1 < len(residuals):
                dy += residuals[t-1]
            
            # Update bootstrap sample
            bootstrap_sample[t] = bootstrap_sample[t-1] + dy
        
        return bootstrap_sample
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def plot_regime_dynamics(self, 
                            save_path: Optional[str] = None,
                            fig_size: Tuple[int, int] = (12, 10),
                            dpi: int = 300) -> plt.Figure:
        """
        Plot regime dynamics and adjustment speeds.
        
        Creates a visualization of the two regimes, threshold,
        and adjustment dynamics.
        
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
        
        # Extract key information
        eq_errors = self.export_equilibrium_errors()
        threshold = self.threshold
        alpha_below = self.results['alpha_below']
        alpha_above = self.results['alpha_above']
        half_lives = self.results['half_lives']
        prop_below = self.results['proportion_below']
        prop_above = self.results['proportion_above']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle(f"TVECM Regime Dynamics: Threshold = {threshold:.4f}", fontsize=16)
        
        # Plot 1: Equilibrium errors over time with threshold
        ax = axes[0, 0]
        ax.plot(eq_errors.index, eq_errors.values)
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
        ax.set_title("Equilibrium Errors Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Error")
        ax.legend()
        
        # Plot 2: Density of equilibrium errors with threshold
        ax = axes[0, 1]
        ax.hist(eq_errors.values, bins=30, density=True, alpha=0.7)
        ax.axvline(x=threshold, color='r', linestyle='--', 
                 label=f'Threshold: {threshold:.4f}\nBelow: {prop_below:.1%}, Above: {prop_above:.1%}')
        ax.set_title("Distribution of Equilibrium Errors")
        ax.set_xlabel("Error")
        ax.set_ylabel("Density")
        ax.legend()
        
        # Plot 3: Adjustment speeds comparison
        ax = axes[1, 0]
        variables = [f'Var{i+1}' for i in range(len(alpha_below))]
        x = np.arange(len(variables))
        width = 0.35
        
        ax.bar(x - width/2, np.abs(alpha_below), width, label='Below Threshold')
        ax.bar(x + width/2, np.abs(alpha_above), width, label='Above Threshold')
        
        ax.set_title("Adjustment Speeds by Regime")
        ax.set_xlabel("Variable")
        ax.set_ylabel("Absolute Adjustment Speed")
        ax.set_xticks(x)
        ax.set_xticklabels(variables)
        ax.legend()
        
        # Plot 4: Half-lives comparison
        ax = axes[1, 1]
        half_below = half_lives['below_regime']
        half_above = half_lives['above_regime']
        
        # Filter out infinite half-lives
        valid_indices = []
        for i in range(len(half_below)):
            if np.isfinite(half_below[i]) and np.isfinite(half_above[i]):
                valid_indices.append(i)
        
        if valid_indices:
            valid_vars = [variables[i] for i in valid_indices]
            valid_below = [half_below[i] for i in valid_indices]
            valid_above = [half_above[i] for i in valid_indices]
            
            x = np.arange(len(valid_vars))
            ax.bar(x - width/2, valid_below, width, label='Below Threshold')
            ax.bar(x + width/2, valid_above, width, label='Above Threshold')
            
            ax.set_title("Half-Lives by Regime (Lower is Faster)")
            ax.set_xlabel("Variable")
            ax.set_ylabel("Half-Life (Periods)")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_vars)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No valid half-lives to display", ha='center', va='center')
            ax.set_title("Half-Lives by Regime")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save if requested
        if save_path:
            try:
                plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                logger.info(f"Saved regime dynamics plot to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save plot: {str(e)}")
        
        return fig
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def _compute_threshold_lr_statistic(self, threshold: float) -> float:
        """
        Compute likelihood ratio test statistic for a given threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value to test
            
        Returns
        -------
        float
            Likelihood ratio test statistic
        """
        # Get the log-likelihood at the optimal threshold
        llf_optimal = self.llf
        
        # Calculate equilibrium errors if not available
        if self.eq_errors is None:
            self.eq_errors = self._calculate_equilibrium_errors()
        
        # Compute log-likelihood at the given threshold
        llf_test = self._compute_llf_for_threshold((threshold, self.eq_errors))
        
        # Compute LR statistic: 2 * (llf_optimal - llf_test)
        lr_stat = 2 * (llf_optimal - llf_test)
        
        return lr_stat
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def prepare_simulation_data(self) -> Dict[str, Any]:
        """
        Prepare TVECM results for use in simulation module.
        
        Returns
        -------
        dict
            Simulation-ready data structure
        """
        # Ensure model is estimated
        if self.results is None:
            self.estimate_tvecm()
        
        # Extract relevant information
        simulation_data = {
            'threshold': self.threshold,
            'cointegration_vector': self.results['beta'],
            'alpha_below': self.results['alpha_below'],
            'alpha_above': self.results['alpha_above'],
            'half_lives': self.results['half_lives'],
            'adjustment_speeds': {
                'below': -self.results['alpha_below'],  # Convert to positive for interpretability
                'above': -self.results['alpha_above']
            },
            'integration_assessment': self.results['integration_assessment'],
            'equilibrium_errors': self.eq_errors,
            'regime_proportions': {
                'below': self.results['proportion_below'],
                'above': self.results['proportion_above']
            },
            'transition_matrix': self.results['transition_matrix'],
            'model_type': 'tvecm'
        }
        
        return simulation_data


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def calculate_regime_transition_matrix(
    eq_errors: np.ndarray, 
    threshold: float
) -> pd.DataFrame:
    """
    Calculate regime transition matrix.
    
    Calculates probabilities of staying in or switching between regimes.
    
    Parameters
    ----------
    eq_errors : array_like
        Equilibrium errors
    threshold : float
        Threshold value
        
    Returns
    -------
    pandas.DataFrame
        Transition matrix with probabilities
        
    Notes
    -----
    The transition matrix shows the persistence of regimes,
    which is important for understanding market integration
    dynamics and the duration of arbitrage opportunities.
    """
    # Create regime indicator (0 for below, 1 for above)
    regimes = (eq_errors > threshold).astype(int)
    
    # Count transitions
    transitions = np.zeros((2, 2))
    
    for t in range(1, len(regimes)):
        from_regime = regimes[t-1]
        to_regime = regimes[t]
        transitions[from_regime, to_regime] += 1
    
    # Convert to probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_probs = np.where(row_sums > 0, transitions / row_sums, 0)
    
    # Convert to DataFrame
    result = pd.DataFrame(
        transition_probs,
        index=['Below', 'Above'],
        columns=['Below', 'Above']
    )
    
    return result


@m1_optimized()
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def calculate_half_lives(
    tvecm_results: Dict[str, Any],
    rich_output: bool = False
) -> Dict[str, List[float]]:
    """
    Calculate half-lives for each variable in each regime.
    
    Half-lives represent the time required for a deviation to
    reduce by half. Lower values indicate faster adjustment.
    
    Parameters
    ----------
    tvecm_results : dict
        TVECM estimation results with alpha coefficients
    rich_output : bool, optional
        Whether to include additional diagnostics in output
        
    Returns
    -------
    dict
        Half-lives for each variable in each regime
        
    Notes
    -----
    In Yemen market integration, half-lives indicate how quickly
    prices adjust after shocks. Differences between regimes show
    how transaction costs affect adjustment speeds.
    
    In general, we expect:
    - Faster adjustment (lower half-life) above threshold when
      arbitrage is profitable
    - Slower adjustment (higher half-life) below threshold when
      arbitrage is not profitable due to transaction costs
    """
    # Extract adjustment speeds
    try:
        # Try first format (direct alpha parameters)
        if 'alpha_below' in tvecm_results and 'alpha_above' in tvecm_results:
            alpha_below = tvecm_results['alpha_below']
            alpha_above = tvecm_results['alpha_above']
        # Try second format (nested structure)
        elif 'below_regime' in tvecm_results and 'above_regime' in tvecm_results:
            alpha_below = tvecm_results['below_regime']['alpha']
            alpha_above = tvecm_results['above_regime']['alpha']
        else:
            raise ValidationError("Invalid TVECM results format - missing alpha parameters")
    except (KeyError, TypeError) as e:
        raise ValidationError(f"Missing adjustment parameters in TVECM results: {str(e)}")
    
    # Calculate half-lives for each variable in each regime
    half_lives_below = [_calculate_single_half_life(a) for a in alpha_below]
    half_lives_above = [_calculate_single_half_life(a) for a in alpha_above]
    
    # Calculate average half-life (excluding infinite values)
    avg_below = np.mean([h for h in half_lives_below if np.isfinite(h)]) if any(np.isfinite(h) for h in half_lives_below) else np.inf
    avg_above = np.mean([h for h in half_lives_above if np.isfinite(h)]) if any(np.isfinite(h) for h in half_lives_above) else np.inf
    
    # Create basic results
    results = {
        'below_regime': half_lives_below,
        'above_regime': half_lives_above,
        'average_below': avg_below,
        'average_above': avg_above,
        'faster_regime': 'above' if avg_above < avg_below else 'below'
    }
    
    # Add rich diagnostic information if requested
    if rich_output:
        results['simulation_ready'] = _prepare_simulation_data(results, alpha_below, alpha_above)
        results['diagnostics'] = {
            'below_adjustment_speeds': [-a for a in alpha_below],
            'above_adjustment_speeds': [-a for a in alpha_above],
            'below_adjustment_significant': [_is_significant_adjustment(a) for a in alpha_below],
            'above_adjustment_significant': [_is_significant_adjustment(a) for a in alpha_above],
            'market_integration_assessment': _assess_market_integration(results)
        }
    
    return results


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _calculate_single_half_life(alpha: float) -> float:
    """
    Calculate half-life for a single adjustment parameter.
    
    Parameters
    ----------
    alpha : float
        Adjustment parameter from VECM
        
    Returns
    -------
    float
        Half-life in periods or np.inf for no adjustment
    """
    # Alpha should be negative for stable adjustment, but we use abs in formula
    # We need the parameter in the form (1+alpha) for half-life calculation
    
    # Case 1: alpha >= 0 (no adjustment or explosive)
    if alpha >= 0:
        return np.inf
        
    # Case 2: -2 < alpha < 0 (stable adjustment)
    elif alpha > -2:
        return np.log(0.5) / np.log(1 + alpha)
        
    # Case 3: alpha <= -2 (oscillatory behavior)
    else:
        return 0.5  # Half-life is 0.5 periods for extreme adjustment


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _is_significant_adjustment(alpha: float, threshold: float = -0.05) -> bool:
    """
    Check if adjustment coefficient is economically significant.
    
    Parameters
    ----------
    alpha : float
        Adjustment coefficient
    threshold : float, optional
        Significance threshold (default: -0.05)
        
    Returns
    -------
    bool
        Whether adjustment is economically significant
    """
    # For meaningful adjustment in monthly data, alpha should be below -0.05
    # This represents correction of at least 5% of disequilibrium per period
    return alpha < threshold


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _assess_market_integration(half_lives: Dict[str, Any]) -> str:
    """
    Assess the degree of market integration based on half-lives.
    
    Parameters
    ----------
    half_lives : dict
        Dictionary with half-life information
        
    Returns
    -------
    str
        Market integration assessment
    """
    # Get average half-lives
    below_mean = half_lives.get('average_below', np.inf)
    above_mean = half_lives.get('average_above', np.inf)
    
    # Handle infinite values
    if np.isinf(below_mean) and np.isinf(above_mean):
        return "No market integration (no adjustment)"
    
    # Strong integration: fast adjustment in both regimes
    if below_mean < 6 and above_mean < 6:  # Less than 6 periods
        return "Strong market integration"
        
    # Partial integration: fast adjustment only above threshold
    if np.isinf(below_mean) or below_mean > 12:  # More than 12 periods or no adjustment below
        if above_mean < 6:  # Fast adjustment above threshold
            return "Threshold-driven market integration (transaction cost barriers)"
        
    # Weak integration: slow adjustment in both regimes
    if (below_mean > 12 or np.isinf(below_mean)) and (above_mean > 12 or np.isinf(above_mean)):
        return "Weak market integration"
        
    # Default case
    return "Moderate market integration"


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _prepare_simulation_data(results: Dict, below_alphas: List[float], above_alphas: List[float]) -> Dict:
    """
    Prepare data structure for use with simulation module.
    
    Parameters
    ----------
    results : dict
        Half-life calculation results
    below_alphas : list
        Adjustment coefficients for below-threshold regime
    above_alphas : list
        Adjustment coefficients for above-threshold regime
        
    Returns
    -------
    dict
        Simulation-ready data structure
    """
    return {
        'half_lives': {
            'below': results['below_regime'],
            'above': results['above_regime'],
        },
        'adjustment_speeds': {
            'below': [-a for a in below_alphas],
            'above': [-a for a in above_alphas]
        },
        'integration_assessment': _assess_market_integration(results)
    }


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _interpret_threshold_test(p_value: float, alpha: float = DEFAULT_ALPHA) -> str:
    """
    Interpret threshold significance test results in Yemen market context.
    
    Parameters
    ----------
    p_value : float
        P-value from threshold test
    alpha : float, optional
        Significance level
        
    Returns
    -------
    str
        Interpretation of results
    """
    if p_value < alpha:
        return (
            "Significant threshold effect detected. This indicates the presence of "
            "transaction costs or other barriers creating two distinct price transmission "
            "regimes. In Yemen's context, this likely reflects conflict-related "
            "barriers to trade or administrative boundaries affecting arbitrage."
        )
    else:
        return (
            "No significant threshold effect detected. Price transmission appears to "
            "follow a linear pattern without distinct regimes. This suggests either "
            "efficient market integration or uniformly high barriers throughout the "
            "entire price range."
        )


@timer
@memory_usage_decorator
@m1_optimized(parallel=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def test_threshold_significance(
    tvecm_model, 
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP, 
    alpha: float = DEFAULT_ALPHA,
    show_progress: bool = False
) -> Dict[str, Any]:
    """
    Test significance of threshold effect using bootstrap.
    
    Implements Hansen & Seo (2002) methodology to test the null hypothesis
    of linearity against the alternative of threshold cointegration.
    
    Parameters
    ----------
    tvecm_model : object
        Fitted ThresholdVECM instance
    n_bootstrap : int, optional
        Number of bootstrap replications
    alpha : float, optional
        Significance level
    show_progress : bool, optional
        Whether to show progress bar during bootstrap
        
    Returns
    -------
    Dict[str, Any]
        Bootstrap test results including p-value and diagnostics
        
    Notes
    -----
    In Yemen's context, significant threshold effect indicates presence of
    transaction costs that create two distinct price transmission regimes.
    """
    # Forward to object method if tvecm_model is ThresholdVECM instance
    if isinstance(tvecm_model, ThresholdVECM):
        return tvecm_model.test_threshold_significance(n_bootstrap=n_bootstrap)
    
    # Validate inputs
    valid, errors = validate_model_inputs(
        model_name="threshold_bootstrap",
        params={"n_bootstrap": n_bootstrap, "alpha": alpha},
        param_validators={
            "n_bootstrap": lambda x: isinstance(x, int) and x > 0,
            "alpha": lambda x: 0 < x < 1
        }
    )
    raise_if_invalid(valid, errors, "Invalid bootstrap parameters")
    
    # Extract data and results needed for bootstrap
    data = tvecm_model.data
    linear_vecm = tvecm_model.linear_model
    linear_results = tvecm_model.linear_results
    tvecm_results = tvecm_model.results
    
    # Calculate test statistic: LR = 2(log L_tvecm - log L_vecm)
    observed_statistic = 2 * (tvecm_results['llf'] - linear_results.llf)
    
    logger.info(f"Starting threshold significance bootstrap with {n_bootstrap} replications")
    logger.info(f"Observed LR statistic: {observed_statistic:.4f}")
    
    # Initialize bootstrap distribution
    bootstrap_statistics = []
    converged_replications = 0
    
    # Define bootstrap function
    def single_bootstrap(seed):
        try:
            # Set seed for reproducibility
            np.random.seed(seed)
            
            # Generate bootstrap sample under null hypothesis (linear VECM)
            bootstrap_data = _generate_bootstrap_sample(data, linear_results)
            
            # Step 1: Fit linear VECM on bootstrap data
            bootstrap_linear_model = VECM(
                bootstrap_data, 
                k_ar_diff=tvecm_model.k_ar_diff, 
                deterministic=tvecm_model.deterministic
            )
            bootstrap_linear_results = bootstrap_linear_model.fit()
            
            # Step 2: Fit threshold VECM on bootstrap data
            bootstrap_tvecm = ThresholdVECM(
                bootstrap_data,
                k_ar_diff=tvecm_model.k_ar_diff,
                deterministic=tvecm_model.deterministic
            )
            bootstrap_tvecm.linear_model = bootstrap_linear_model
            bootstrap_tvecm.linear_results = bootstrap_linear_results
            
            # Search for threshold in bootstrap data
            threshold_results = bootstrap_tvecm.grid_search_threshold(
                trim=DEFAULT_TRIM,
                n_grid=100,  # Use fewer grid points for speed
                verbose=False
            )
            
            # Estimate TVECM with found threshold
            bootstrap_tvecm_results = bootstrap_tvecm.estimate_tvecm(run_diagnostics=False)
            
            # Calculate bootstrap LR statistic
            bootstrap_lr = 2 * (
                bootstrap_tvecm_results['llf'] - bootstrap_linear_results.llf
            )
            
            return {"statistic": bootstrap_lr, "converged": True}
        except Exception as e:
            logger.warning(f"Bootstrap replication failed: {str(e)}")
            return {"statistic": np.nan, "converged": False}
    
    # Run bootstrap in parallel
    seeds = np.random.randint(0, 10000, size=n_bootstrap)
    bootstrap_results = parallelize(
        single_bootstrap, 
        seeds, 
        n_workers=min(8, n_bootstrap),
        progress_bar=show_progress
    )
    
    # Process results
    for result in bootstrap_results:
        if result["converged"]:
            bootstrap_statistics.append(result["statistic"])
            converged_replications += 1
    
    # Check if enough replications converged
    if converged_replications < 0.9 * n_bootstrap:
        logger.warning(f"Only {converged_replications}/{n_bootstrap} bootstrap replications converged")
    
    # Calculate p-value
    bootstrap_statistics = np.array(bootstrap_statistics)
    bootstrap_statistics = bootstrap_statistics[~np.isnan(bootstrap_statistics)]
    p_value = np.mean(bootstrap_statistics > observed_statistic)
    
    # Format results
    results = {
        "observed_statistic": observed_statistic,
        "bootstrap_mean": np.mean(bootstrap_statistics),
        "bootstrap_std": np.std(bootstrap_statistics),
        "p_value": p_value,
        "significant": p_value < alpha,
        "n_bootstrap": n_bootstrap,
        "n_converged": converged_replications,
        "convergence_rate": converged_replications / n_bootstrap,
        "bootstrap_quantiles": {
            "90%": np.percentile(bootstrap_statistics, 90),
            "95%": np.percentile(bootstrap_statistics, 95),
            "99%": np.percentile(bootstrap_statistics, 99)
        },
        "interpretation": _interpret_threshold_test(p_value, alpha)
    }
    
    logger.info(
        f"Threshold significance test complete: p-value={p_value:.4f}, "
        f"significant={results['significant']}, "
        f"convergence_rate={results['convergence_rate']:.2f}"
    )
    
    return results


@m1_optimized(parallel=False)
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _generate_bootstrap_sample(data: pd.DataFrame, vecm_results: Any) -> pd.DataFrame:
    """
    Generate bootstrap sample using fitted VECM parameters.
    
    Parameters
    ----------
    data : DataFrame
        Original data
    vecm_results : VECMResults
        Fitted linear VECM results
        
    Returns
    -------
    DataFrame
        Bootstrapped data sample
    """
    # Extract parameters from fitted model
    alpha = vecm_results.alpha
    beta = vecm_results.beta
    gamma = vecm_results.gamma
    deterministic = vecm_results.deterministic
    
    # Extract residuals and center them
    residuals = vecm_results.resid
    centered_resids = residuals - residuals.mean(axis=0)
    
    # Create bootstrap residuals by sampling with replacement
    n_obs = len(residuals)
    bootstrap_indices = np.random.choice(n_obs, size=n_obs)
    bootstrap_residuals = centered_resids[bootstrap_indices]
    
    # Create bootstrap sample following VECM dynamics
    bootstrap_data = np.zeros((n_obs, data.shape[1]))
    bootstrap_data[:2] = data.values[:2]  # Use original initial values
    
    for t in range(2, n_obs):
        # Calculate cointegration term: Z_{t-1} = X_{t-1} @ beta
        if deterministic == 'ci':
            Z = np.concatenate([[1], bootstrap_data[t-1]]) @ beta
        else:
            Z = bootstrap_data[t-1] @ beta
        
        # Calculate adjustment: alpha @ Z
        adjustment = alpha * Z
        
        # Calculate lagged difference effects
        lagged_effect = np.zeros(data.shape[1])
        for i in range(1, min(gamma.shape[0] + 1, t)):
            if t-i >= 0:
                dy_lag = bootstrap_data[t-i] - bootstrap_data[t-i-1]
                gamma_slice = gamma[i-1] if gamma.ndim == 2 else gamma
                lagged_effect += np.dot(gamma_slice.reshape(data.shape[1], data.shape[1]), dy_lag)
        
        # Update bootstrap data
        bootstrap_data[t] = bootstrap_data[t-1] + adjustment + lagged_effect + bootstrap_residuals[t-2]
    
    # Convert to DataFrame with original index and columns
    if isinstance(data, pd.DataFrame):
        bootstrap_df = pd.DataFrame(
            bootstrap_data,
            index=data.index,
            columns=data.columns
        )
    else:
        bootstrap_df = pd.DataFrame(bootstrap_data)
    
    return bootstrap_df


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def combine_tvecm_results(
    model1_results: Dict[str, Any],
    model2_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine results from two TVECM models for comparison.
    
    Useful for comparing models with different specifications
    or data periods.
    
    Parameters
    ----------
    model1_results : dict
        Results from first TVECM model
    model2_results : dict
        Results from second TVECM model
        
    Returns
    -------
    dict
        Combined results for comparison
        
    Notes
    -----
    This function is particularly useful for comparing:
    - Pre-conflict vs. post-conflict market integration
    - Different exchange rate regimes
    - Different commodity markets
    """
    # Extract key metrics
    model1_threshold = model1_results.get('threshold')
    model2_threshold = model2_results.get('threshold')
    
    model1_half_lives = model1_results.get('half_lives', {})
    model2_half_lives = model2_results.get('half_lives', {})
    
    model1_llf = model1_results.get('llf')
    model2_llf = model2_results.get('llf')
    
    # Compile comparison
    comparison = {
        'thresholds': {
            'model1': model1_threshold,
            'model2': model2_threshold,
            'difference': model1_threshold - model2_threshold if (model1_threshold is not None and model2_threshold is not None) else None,
            'percent_change': ((model2_threshold - model1_threshold) / model1_threshold * 100) if (model1_threshold is not None and model2_threshold is not None and model1_threshold != 0) else None
        },
        'half_lives': {
            'model1': model1_half_lives,
            'model2': model2_half_lives
        },
        'fit': {
            'model1_llf': model1_llf,
            'model2_llf': model2_llf,
            'better_fit': 'model1' if (model1_llf is not None and model2_llf is not None and model1_llf > model2_llf) else 'model2'
        }
    }
    
    return comparison