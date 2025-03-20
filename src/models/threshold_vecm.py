"""
Threshold Vector Error Correction Model (TVECM) for market integration analysis.

This module implements the Hansen & Seo (2002) approach for multivariate
threshold cointegration analysis of price transmission between markets
in conflict-affected Yemen.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import matplotlib.pyplot as plt

from utils import (
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
    test_linearity, calculate_half_life, fit_vecm_model, 
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
DEFAULT_ALPHA = config.get('analysis.threshold_vecm.alpha', 0.05)
DEFAULT_TRIM = config.get('analysis.threshold_vecm.trim', 0.15)
DEFAULT_N_GRID = config.get('analysis.threshold_vecm.n_grid', 300)
DEFAULT_K_AR_DIFF = config.get('analysis.threshold_vecm.k_ar_diff', 2)
DEFAULT_BOOTSTRAP_REPS = config.get('analysis.threshold_vecm.bootstrap_reps', 1000)


class ThresholdVECM:
    """
    Enhanced Threshold Vector Error Correction Model implementation.
    
    This class implements a multivariate threshold cointegration model
    to analyze price transmission networks between multiple markets.
    In Yemen's context, thresholds represent transaction costs that
    impede arbitrage due to conflict, political fragmentation,
    and dual exchange rate regimes.
    """
    
    def __init__(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        k_ar_diff: int = DEFAULT_K_AR_DIFF,
        deterministic: str = "ci",
        coint_rank: int = 1,
        market_names: Optional[List[str]] = None
    ):
        """
        Initialize the threshold VECM model.
        
        Parameters
        ----------
        data : array_like
            Multivariate time series data (markets as columns)
        k_ar_diff : int, optional
            Number of lagged differences
        deterministic : str, optional
            Deterministic term specification ('n', 'c', 'ct', 'ci', 'cit')
            - 'n': no deterministic terms
            - 'c': constant
            - 'ct': constant and trend
            - 'ci': constant inside cointegration
            - 'cit': constant and trend inside cointegration
        coint_rank : int, optional
            Cointegration rank
        market_names : list of str, optional
            Names of markets for each column
        """
        # Optimize system for performance
        configure_system_for_performance()
        
        # Validate input data
        self._validate_input(data)
        
        # Store input data
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
            self.index = data.index
            
            # Use column names as market names if not provided
            if market_names is None:
                self.market_names = data.columns.tolist()
            else:
                self.market_names = market_names
        else:
            self.data = np.asarray(data)
            self.index = None
            
            # Generate default market names
            if market_names is None:
                self.market_names = [f"Market {i+1}" for i in range(data.shape[1])]
            else:
                self.market_names = market_names
        
        # Validate model parameters
        valid, errors = validate_model_inputs(
            model_name="threshold_vecm",
            params={
                "k_ar_diff": k_ar_diff,
                "deterministic": deterministic,
                "coint_rank": coint_rank
            },
            required_params={"k_ar_diff", "deterministic", "coint_rank"},
            param_validators={
                "k_ar_diff": lambda x: isinstance(x, int) and x > 0,
                "deterministic": lambda x: x in ["n", "c", "ct", "ci", "cit"],
                "coint_rank": lambda x: isinstance(x, int) and x > 0
            }
        )
        raise_if_invalid(valid, errors, "Invalid model parameters")
        
        # Store model parameters
        self.k_ar_diff = k_ar_diff
        self.deterministic = deterministic
        self.coint_rank = coint_rank
        
        # Initialize attributes for results
        self.linear_model = None
        self.threshold = None
        self.below_model = None
        self.above_model = None
        self.ec_term = None
        
        logger.info(
            f"Initialized ThresholdVECM with {self.data.shape} observations, "
            f"k_ar_diff={k_ar_diff}, deterministic='{deterministic}', "
            f"coint_rank={coint_rank}"
        )
    
    def _validate_input(self, data):
        """Validate input data for TVECM estimation."""
        # Check if data is a DataFrame or ndarray
        if isinstance(data, pd.DataFrame):
            # Check for missing values
            if data.isnull().any().any():
                logger.warning("Data contains missing values, these will be handled during modeling")
            
            # Check if at least 2 columns
            if data.shape[1] < 2:
                raise ValidationError("Data must have at least 2 columns for TVECM estimation")
            
            # Check sufficient observations
            if data.shape[0] < 30:
                logger.warning("Less than 30 observations may lead to unreliable estimates")
                
            # Validate each column as a time series
            for col in data.columns:
                valid, errors = validate_time_series(
                    data[col], min_length=30, check_constant=True
                )
                if not valid:
                    logger.warning(f"Validation issues for column {col}: {errors}")
        
        elif isinstance(data, np.ndarray):
            # Check for missing values
            if np.isnan(data).any():
                logger.warning("Data contains missing values, these will be handled during modeling")
            
            # Check dimensions
            if data.ndim != 2:
                raise ValidationError("Data must be 2-dimensional for TVECM estimation")
            
            # Check if at least 2 columns
            if data.shape[1] < 2:
                raise ValidationError("Data must have at least 2 columns for TVECM estimation")
            
            # Check sufficient observations
            if data.shape[0] < 30:
                logger.warning("Less than 30 observations may lead to unreliable estimates")
        
        else:
            raise ValidationError("Data must be a pandas DataFrame or numpy ndarray")
    
    @disk_cache(cache_dir='.cache/threshold_vecm')
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def estimate_linear_vecm(self) -> Dict[str, Any]:
        """
        Estimate linear VECM as a baseline model.
        
        This provides a standard VECM without threshold effects
        for comparison with the threshold model.
        
        Returns
        -------
        dict
            Linear VECM estimation results including:
            - alpha: Loading matrix (adjustment speeds)
            - beta: Cointegrating vectors
            - gamma: Short-run coefficient matrices
            - pi: Long-run impact matrix (alpha * beta')
            - llf: Log-likelihood value
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion
            - hqic: Hannan-Quinn Information Criterion
            - resid: Model residuals
            - fitted_values: Fitted values
            
        Notes
        -----
        In Yemen's fragmented markets, the linear VECM provides a baseline
        that assumes uniform price transmission regardless of price differential
        magnitude. This uniform approach may mask threshold effects due to
        conflict barriers.
        """
        # Use project utilities for VECM estimation
        linear_results = fit_vecm_model(
            data=self.data,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic
        )
        
        # Store for later use
        self.linear_model = linear_results
        
        # Extract error correction term
        if isinstance(self.data, pd.DataFrame):
            if self.deterministic in ['ci', 'cit']:
                # Add constant/trend to data for EC term
                if self.deterministic == 'ci':
                    z = np.column_stack([np.ones(len(self.data)), self.data.values])
                else:  # 'cit'
                    trend = np.arange(len(self.data))
                    z = np.column_stack([np.ones(len(self.data)), trend, self.data.values])
                
                # Remove last column which corresponds to dependent variable
                z = z[:, :-1]
            else:
                z = self.data.values
            
            # Calculate EC term using beta
            self.ec_term = np.dot(z, linear_results['beta'])
        
        logger.info(
            f"Linear VECM estimation complete: "
            f"log-likelihood={linear_results['llf']:.4f}, AIC={linear_results['aic']:.4f}"
        )
        
        return linear_results
    
    @timer
    @memory_usage_decorator
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def grid_search_threshold(
        self, 
        ec_term: Optional[np.ndarray] = None,
        trim: float = DEFAULT_TRIM,
        n_grid: int = DEFAULT_N_GRID
    ) -> Dict[str, Any]:
        """
        Find optimal threshold via grid search.
        
        Uses a grid search to find the threshold value that maximizes
        the likelihood of the threshold VECM model.
        
        Parameters
        ----------
        ec_term : array_like, optional
            Error correction term (will be calculated if not provided)
        trim : float, optional
            Trimming percentage for threshold search
        n_grid : int, optional
            Number of grid points
            
        Returns
        -------
        dict
            Threshold search results including:
            - threshold: Optimal threshold value
            - likelihood: Maximum likelihood value
            - all_thresholds: All evaluated thresholds
            - all_likelihoods: Likelihoods for all thresholds
            
        Notes
        -----
        The threshold represents transaction costs in Yemen's market context.
        These include:
        - Conflict barriers (checkpoints, security risks)
        - Transportation costs affected by fuel prices
        - Exchange rate conversion costs between regions
        """
        # Ensure we have linear VECM results
        if self.linear_model is None:
            logger.info("Estimating linear VECM first")
            self.estimate_linear_vecm()
        
        # Use provided EC term or the one from linear model
        if ec_term is not None:
            ec_term_values = ec_term
        else:
            ec_term_values = self.ec_term
        
        # Validate parameters
        valid, errors = validate_model_inputs(
            model_name="threshold_vecm",
            params={"trim": trim, "n_grid": n_grid},
            required_params={"trim", "n_grid"},
            param_validators={
                "trim": lambda x: 0.0 < x < 0.5,
                "n_grid": lambda x: isinstance(x, int) and x > 10
            }
        )
        raise_if_invalid(valid, errors, "Invalid threshold grid search parameters")
        
        # Get threshold candidates
        candidates = self._get_threshold_candidates(ec_term_values, trim, n_grid)
        
        # Grid search with parallelization
        logger.info(f"Starting grid search with {len(candidates)} threshold candidates")
        
        def process_threshold(threshold_candidate):
            return (threshold_candidate, self._compute_likelihood_for_threshold(threshold_candidate))
        
        # Parallelize the grid search for better performance
        df_candidates = pd.DataFrame({'threshold': candidates})
        results_df = parallelize_dataframe(df_candidates, lambda df: df.apply(
            lambda row: process_threshold(row['threshold']), axis=1))
        
        # Process results
        best_threshold, best_llf, thresholds, llfs = self._process_threshold_results(results_df)
        
        # Store the optimal threshold
        self.threshold = best_threshold
        
        # Calculate proportion of observations in each regime
        below_prop = np.mean(ec_term_values <= best_threshold)
        above_prop = 1 - below_prop
        
        # Create result dictionary
        result = {
            'threshold': best_threshold,
            'likelihood': best_llf,
            'all_thresholds': thresholds,
            'all_likelihoods': llfs,
            'proportion_below': below_prop,
            'proportion_above': above_prop
        }
        
        logger.info(
            f"Threshold search complete: threshold={best_threshold:.4f}, "
            f"log-likelihood={best_llf:.4f}, "
            f"below_prop={below_prop:.1%}, above_prop={above_prop:.1%}"
        )
        
        return result
    
    def _get_threshold_candidates(self, ec_term: np.ndarray, trim: float, n_grid: int) -> np.ndarray:
        """Get candidate threshold values for grid search."""
        sorted_ec = np.sort(ec_term)
        lower_idx = int(len(sorted_ec) * trim)
        upper_idx = int(len(sorted_ec) * (1 - trim))
        candidates = sorted_ec[lower_idx:upper_idx]
        
        if len(candidates) > n_grid:
            # Use a subset for efficiency
            step = len(candidates) // n_grid
            candidates = candidates[::step]
            
        return candidates
    
    @m1_optimized()
    def _compute_likelihood_for_threshold(self, threshold: float) -> float:
        """Compute log-likelihood for a given threshold."""
        try:
            # Split data by threshold
            ec_values = self.ec_term
            below_mask = ec_values <= threshold
            above_mask = ~below_mask
            
            # Check if enough observations in each regime
            min_obs = self.k_ar_diff + self.coint_rank + 2  # Minimum observations needed
            if np.sum(below_mask) < min_obs or np.sum(above_mask) < min_obs:
                return -np.inf  # Invalid threshold with insufficient observations
            
            # Get data for each regime
            if isinstance(self.data, pd.DataFrame):
                below_data = self.data.iloc[below_mask]
                above_data = self.data.iloc[above_mask]
            else:
                below_data = self.data[below_mask]
                above_data = self.data[above_mask]
            
            # Estimate VECM for each regime
            try:
                below_model = fit_vecm_model(
                    data=below_data,
                    k_ar_diff=self.k_ar_diff,
                    coint_rank=self.coint_rank,
                    deterministic=self.deterministic
                )
                
                above_model = fit_vecm_model(
                    data=above_data,
                    k_ar_diff=self.k_ar_diff,
                    coint_rank=self.coint_rank,
                    deterministic=self.deterministic
                )
                
                # Combined log-likelihood
                combined_llf = below_model['llf'] + above_model['llf']
                return combined_llf
                
            except Exception as e:
                logger.debug(f"Error estimating VECM for threshold {threshold}: {str(e)}")
                return -np.inf
                
        except Exception as e:
            logger.debug(f"Error in threshold computation {threshold}: {str(e)}")
            return -np.inf
    
    def _process_threshold_results(self, results_df) -> Tuple[float, float, List[float], List[float]]:
        """Process threshold grid search results."""
        best_llf = -np.inf
        best_threshold = None
        thresholds = []
        llfs = []
        
        for i, row in results_df.iterrows():
            threshold, llf = row[0]
            thresholds.append(threshold)
            llfs.append(llf)
            
            if llf > best_llf:
                best_llf = llf
                best_threshold = threshold
                
        return best_threshold, best_llf, thresholds, llfs
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def estimate_tvecm(self, run_diagnostics: bool = False) -> Dict[str, Any]:
        """
        Estimate the Threshold Vector Error Correction Model.
        
        Fits a two-regime VECM where each regime has its own
        adjustment speeds and dynamics, separated by a threshold.
        
        Parameters
        ----------
        run_diagnostics : bool, optional
            Whether to run diagnostic tests on the model
            
        Returns
        -------
        dict
            TVECM estimation results including:
            - threshold: Estimated threshold value
            - below_regime: Results for below-threshold regime
            - above_regime: Results for above-threshold regime
            - proportion_below: Proportion of observations below threshold
            - proportion_above: Proportion of observations above threshold
            - likelihood_ratio: LR test of threshold vs. linear model
            - diagnostics: Diagnostic test results (if requested)
            
        Notes
        -----
        The TVECM estimates different adjustment speeds in each regime, capturing
        how Yemen's market integration changes when price differentials exceed
        transaction costs imposed by conflict, political barriers, and dual
        exchange rates.
        """
        # Ensure we have a threshold
        if self.threshold is None:
            logger.info("Running threshold search first")
            self.grid_search_threshold()
        
        # Split data by threshold
        ec_values = self.ec_term
        below_mask = ec_values <= self.threshold
        above_mask = ~below_mask
        
        # Get data for each regime
        if isinstance(self.data, pd.DataFrame):
            below_data = self.data.iloc[below_mask]
            above_data = self.data.iloc[above_mask]
        else:
            below_data = self.data[below_mask]
            above_data = self.data[above_mask]
        
        # Estimate VECM for each regime
        logger.info("Estimating below-threshold regime model")
        below_model = fit_vecm_model(
            data=below_data,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic
        )
        
        logger.info("Estimating above-threshold regime model")
        above_model = fit_vecm_model(
            data=above_data,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic
        )
        
        # Store models for later use
        self.below_model = below_model
        self.above_model = above_model
        
        # Calculate likelihood ratio test statistic
        # LR = 2 * (log-likelihood of threshold model - log-likelihood of linear model)
        lr_stat = 2 * ((below_model['llf'] + above_model['llf']) - self.linear_model['llf'])
        
        # Compile results
        tvecm_results = {
            'threshold': self.threshold,
            'below_regime': below_model,
            'above_regime': above_model,
            'proportion_below': np.mean(below_mask),
            'proportion_above': np.mean(above_mask),
            'likelihood_ratio': {
                'statistic': lr_stat,
                'threshold_llf': below_model['llf'] + above_model['llf'],
                'linear_llf': self.linear_model['llf']
            },
            'summary': {
                'threshold': self.threshold,
                'below_obs': np.sum(below_mask),
                'above_obs': np.sum(above_mask),
                'below_alpha': below_model['alpha'],
                'above_alpha': above_model['alpha'],
                'adjustment_difference': np.abs(above_model['alpha'] - below_model['alpha'])
            }
        }
        
        # Add interpretation
        tvecm_results['interpretation'] = self._interpret_tvecm_results(tvecm_results)
        
        # Run diagnostics if requested
        if run_diagnostics:
            tvecm_results['diagnostics'] = self.run_diagnostics()
        
        logger.info(
            f"TVECM estimation complete: threshold={self.threshold:.4f}, "
            f"below_obs={np.sum(below_mask)}, above_obs={np.sum(above_mask)}, "
            f"LR={lr_stat:.4f}"
        )
        
        return tvecm_results
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def _interpret_tvecm_results(self, results: Dict[str, Any]) -> str:
        """
        Generate interpretation of TVECM results.
        
        Parameters
        ----------
        results : dict
            TVECM estimation results
            
        Returns
        -------
        str
            Text interpretation of results
        """
        below_alpha = results['below_regime']['alpha']
        above_alpha = results['above_regime']['alpha']
        threshold = results['threshold']
        adj_diff = results['summary']['adjustment_difference']
        
        # Calculate average absolute adjustment in each regime
        below_adj_avg = np.mean(np.abs(below_alpha))
        above_adj_avg = np.mean(np.abs(above_alpha))
        
        # Compare adjustment speeds
        if above_adj_avg > 2 * below_adj_avg:
            speed_text = (
                f"Price adjustment is substantially faster ({above_adj_avg/below_adj_avg:.1f}x) "
                f"when price differentials exceed the threshold. This indicates significant "
                f"transaction cost barriers that, once overcome, allow rapid price convergence."
            )
        elif above_adj_avg > 1.2 * below_adj_avg:
            speed_text = (
                f"Price adjustment is moderately faster ({above_adj_avg/below_adj_avg:.1f}x) "
                f"above the threshold, consistent with the presence of transaction costs "
                f"that partially impede arbitrage at smaller price differentials."
            )
        elif below_adj_avg > 1.2 * above_adj_avg:
            speed_text = (
                f"Unusually, price adjustment is faster below than above the threshold. "
                f"This may indicate data quality issues, structural breaks, or complex "
                f"market dynamics requiring further investigation."
            )
        else:
            speed_text = (
                f"Price adjustment speeds are similar in both regimes, suggesting either "
                f"minimal threshold effects or uniform barriers affecting all price levels."
            )
        
        # Assess market integration
        if below_adj_avg < 0.05 and above_adj_avg < 0.05:
            integration_text = (
                "Overall adjustment speeds are very low in both regimes, indicating weak "
                "market integration likely due to significant conflict barriers and political fragmentation."
            )
        elif below_adj_avg < 0.05 and above_adj_avg >= 0.05:
            integration_text = (
                "Markets show threshold-limited integration, with adjustment occurring only "
                "when price differentials exceed transaction costs. This pattern is consistent "
                "with significant but surmountable conflict-related barriers to trade."
            )
        elif above_adj_avg > 0.1:
            integration_text = (
                "Substantial price adjustment above the threshold indicates strong long-run "
                "market relationships despite the presence of threshold effects."
            )
        else:
            integration_text = (
                "Moderate price adjustment in both regimes suggests partial market integration "
                "with persistent barriers affecting price transmission."
            )
        
        # Complete interpretation
        interpretation = (
            f"Threshold VECM analysis detected a threshold of {threshold:.4f}, which represents "
            f"the transaction cost barrier between markets. {speed_text} {integration_text} "
            f"These results suggest that conflict-related barriers affect price transmission "
            f"through creating significant transaction costs between markets."
        )
        
        return interpretation
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostic tests on the TVECM.
        
        Performs tests for:
        - Residual normality
        - Serial correlation
        - Heteroskedasticity
        - Parameter stability
        
        Returns
        -------
        dict
            Diagnostic test results
        """
        # Make sure we have model results
        if self.below_model is None or self.above_model is None:
            logger.info("Estimating TVECM first")
            self.estimate_tvecm()
        
        logger.info("Running diagnostic tests on TVECM")
        
        # Get residuals
        below_resid = self.below_model.get('resid')
        above_resid = self.above_model.get('resid')
        
        # Initialize results
        diagnostics = {
            'below_regime': {},
            'above_regime': {},
            'summary': {}
        }
        
        # Test for white noise residuals
        try:
            below_white_noise = {}
            above_white_noise = {}
            
            if below_resid is not None:
                for i in range(below_resid.shape[1]):
                    col_name = self.market_names[i] if i < len(self.market_names) else f"Market {i+1}"
                    below_white_noise[col_name] = test_white_noise(below_resid[:, i])
            
            if above_resid is not None:
                for i in range(above_resid.shape[1]):
                    col_name = self.market_names[i] if i < len(self.market_names) else f"Market {i+1}"
                    above_white_noise[col_name] = test_white_noise(above_resid[:, i])
            
            diagnostics['below_regime']['white_noise'] = below_white_noise
            diagnostics['above_regime']['white_noise'] = above_white_noise
            
            # Summarize results
            below_residuals_valid = all(
                test.get('is_white_noise', False) 
                for test in below_white_noise.values()
            )
            
            above_residuals_valid = all(
                test.get('is_white_noise', False) 
                for test in above_white_noise.values()
            )
            
            diagnostics['summary']['below_residuals_valid'] = below_residuals_valid
            diagnostics['summary']['above_residuals_valid'] = above_residuals_valid
            diagnostics['summary']['all_residuals_valid'] = below_residuals_valid and above_residuals_valid
            
        except Exception as e:
            logger.warning(f"Error in white noise tests: {str(e)}")
        
        # Test for structural breaks
        try:
            # Use a subset of data for structural break test
            if isinstance(self.data, pd.DataFrame):
                first_var = self.data.iloc[:, 0]
            else:
                first_var = self.data[:, 0]
                
            struct_break = test_structural_break(
                y=first_var,
                X=None,
                method='quandt'
            )
            
            diagnostics['structural_breaks'] = struct_break
            diagnostics['summary']['has_structural_breaks'] = struct_break.get('significant', False)
            
        except Exception as e:
            logger.warning(f"Error in structural break test: {str(e)}")
        
        logger.info(
            f"Diagnostics complete: "
            f"below_residuals_valid={diagnostics['summary'].get('below_residuals_valid', False)}, "
            f"above_residuals_valid={diagnostics['summary'].get('above_residuals_valid', False)}, "
            f"structural_breaks={diagnostics['summary'].get('has_structural_breaks', False)}"
        )
        
        return diagnostics
    
    @disk_cache(cache_dir='.cache/threshold_vecm')
    @m1_optimized()
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_threshold_significance(
        self, 
        n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS
    ) -> Dict[str, Any]:
        """
        Test significance of threshold effect using bootstrap.
        
        Tests whether the threshold effect is statistically significant
        compared to a linear VECM without thresholds.
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap replications
            
        Returns
        -------
        dict
            Test results including:
            - lr_statistic: Likelihood ratio statistic
            - p_value: Bootstrap p-value
            - significant: Boolean indicating if threshold effect is significant
            - critical_values: Critical values at different significance levels
            
        Notes
        -----
        In Yemen's fragmented market context, a significant threshold effect confirms
        that transaction costs create nonlinear price adjustment patterns, with different
        dynamics depending on whether price differentials exceed conflict-related barriers.
        """
        # Make sure we have model results
        if self.below_model is None or self.above_model is None:
            logger.info("Estimating TVECM first")
            self.estimate_tvecm()
        
        # Calculate likelihood ratio
        lr_stat = 2 * ((self.below_model['llf'] + self.above_model['llf']) - self.linear_model['llf'])
        
        # Run bootstrap to obtain p-value
        bootstrap_lr = []
        
        logger.info(f"Starting bootstrap with {n_bootstrap} replications")
        
        # Generate data from linear model
        for i in range(n_bootstrap):
            if i % 50 == 0:
                logger.debug(f"Bootstrap replication {i}")
                
            try:
                # Generate bootstrap sample from linear model
                bootstrap_data = self._generate_bootstrap_sample()
                
                # Initialize new model with bootstrap data
                bootstrap_model = ThresholdVECM(
                    data=bootstrap_data,
                    k_ar_diff=self.k_ar_diff,
                    deterministic=self.deterministic,
                    coint_rank=self.coint_rank
                )
                
                # Estimate linear VECM
                bootstrap_linear = bootstrap_model.estimate_linear_vecm()
                
                # Estimate threshold
                bootstrap_model.grid_search_threshold()
                
                # Estimate TVECM
                bootstrap_tvecm = bootstrap_model.estimate_tvecm()
                
                # Calculate LR statistic
                bootstrap_lr_stat = 2 * (
                    (bootstrap_tvecm['below_regime']['llf'] + bootstrap_tvecm['above_regime']['llf']) - 
                    bootstrap_linear['llf']
                )
                
                bootstrap_lr.append(bootstrap_lr_stat)
                
            except Exception as e:
                logger.debug(f"Error in bootstrap replication {i}: {str(e)}")
                continue
        
        # Calculate p-value
        if bootstrap_lr:
            p_value = np.mean(np.array(bootstrap_lr) > lr_stat)
            
            # Calculate critical values
            critical_values = {
                '10%': np.percentile(bootstrap_lr, 90),
                '5%': np.percentile(bootstrap_lr, 95),
                '1%': np.percentile(bootstrap_lr, 99)
            }
        else:
            logger.warning("Bootstrap failed, cannot calculate p-value")
            p_value = np.nan
            critical_values = {}
        
        # Prepare results
        results = {
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'significant': p_value < DEFAULT_ALPHA if not np.isnan(p_value) else None,
            'critical_values': critical_values,
            'n_bootstrap': len(bootstrap_lr),
            'bootstrap_distribution': bootstrap_lr
        }
        
        logger.info(
            f"Threshold significance test: LR={lr_stat:.4f}, p-value={p_value:.4f}, "
            f"significant={results['significant']}"
        )
        
        return results
    
    @m1_optimized()
    def _generate_bootstrap_sample(self) -> np.ndarray:
        """Generate bootstrap sample from linear model."""
        # Get residuals from linear model
        if self.linear_model.get('resid') is None:
            raise ValueError("Linear model does not have residuals")
            
        resid = self.linear_model['resid']
        
        # Get fitted values
        if self.linear_model.get('fitted_values') is None:
            raise ValueError("Linear model does not have fitted values")
            
        fitted = self.linear_model['fitted_values']
        
        # Generate bootstrap sample
        n_obs = resid.shape[0]
        bootstrap_indices = np.random.randint(0, n_obs, size=n_obs)
        bootstrap_resid = resid[bootstrap_indices]
        
        # Create bootstrap sample
        bootstrap_data = fitted + bootstrap_resid
        
        return bootstrap_data
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def calculate_regime_transition_matrix(self) -> Dict[str, Any]:
        """
        Calculate transition matrix between regimes.
        
        Computes the probability of transitioning between regimes,
        showing how frequently markets move across the transaction
        cost threshold.
        
        Returns
        -------
        dict
            Transition analysis including:
            - transition_matrix: Regime transition probabilities
            - persistence: Regime persistence metrics
            - regime_counts: Number of observations in each regime
            
        Notes
        -----
        In Yemen's conflict context, high persistence in the below-threshold
        regime may indicate market segmentation, where price differentials
        remain too small to trigger arbitrage due to political separation
        between markets.
        """
        # Make sure we have the EC term
        if self.ec_term is None:
            logger.info("Estimating linear VECM first")
            self.estimate_linear_vecm()
        
        # Make sure we have a threshold
        if self.threshold is None:
            logger.info("Running threshold search first")
            self.grid_search_threshold()
        
        # Create regime indicator (0 for below, 1 for above)
        regimes = (self.ec_term > self.threshold).astype(int)
        
        # Calculate transitions
        transitions = np.zeros((2, 2))
        
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            transitions[from_regime, to_regime] += 1
        
        # Convert to probabilities
        transition_matrix = transitions / transitions.sum(axis=1, keepdims=True)
        
        # Calculate regime persistence
        persistence = {
            'below_persistence': transition_matrix[0, 0],
            'above_persistence': transition_matrix[1, 1],
            'expected_duration_below': 1 / (1 - transition_matrix[0, 0]) if transition_matrix[0, 0] < 1 else float('inf'),
            'expected_duration_above': 1 / (1 - transition_matrix[1, 1]) if transition_matrix[1, 1] < 1 else float('inf')
        }
        
        # Count regimes
        regime_counts = {
            'below': np.sum(regimes == 0),
            'above': np.sum(regimes == 1)
        }
        
        result = {
            'transition_matrix': transition_matrix,
            'persistence': persistence,
            'regime_counts': regime_counts
        }
        
        logger.info(
            f"Regime transition analysis: "
            f"below_persistence={persistence['below_persistence']:.2f}, "
            f"above_persistence={persistence['above_persistence']:.2f}, "
            f"below_duration={persistence['expected_duration_below']:.1f} periods, "
            f"above_duration={persistence['expected_duration_above']:.1f} periods"
        )
        
        return result
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def calculate_half_lives(self) -> Dict[str, Any]:
        """
        Calculate half-lives of price adjustments in each regime.
        
        Computes how quickly price shocks dissipate in each regime,
        measured by the time for a shock to decay to half its initial size.
        
        Returns
        -------
        dict
            Half-life calculations for each market in each regime
            
        Notes
        -----
        In Yemen's fragmented markets, half-lives are typically:
        - Longer below the threshold, as price adjustment is slower
          when differentials are within transaction costs
        - Shorter above the threshold, as arbitrage drives faster
          price convergence once differentials exceed conflict-related
          transaction costs
        """
        # Make sure we have model results
        if self.below_model is None or self.above_model is None:
            logger.info("Estimating TVECM first")
            self.estimate_tvecm()
        
        # Calculate half-lives for each market in each regime
        half_lives = {}
        
        # Below regime
        half_lives['below'] = {}
        for i, market in enumerate(self.market_names):
            if i < self.below_model['alpha'].shape[0]:
                alpha_i = self.below_model['alpha'][i, 0]  # Use first cointegration relation
                half_lives['below'][market] = calculate_half_life(alpha_i)
        
        # Above regime
        half_lives['above'] = {}
        for i, market in enumerate(self.market_names):
            if i < self.above_model['alpha'].shape[0]:
                alpha_i = self.above_model['alpha'][i, 0]  # Use first cointegration relation
                half_lives['above'][market] = calculate_half_life(alpha_i)
        
        # Calculate average half-lives
        below_values = [h for h in half_lives['below'].values() if not np.isinf(h)]
        above_values = [h for h in half_lives['above'].values() if not np.isinf(h)]
        
        half_lives['average'] = {
            'below': np.mean(below_values) if below_values else float('inf'),
            'above': np.mean(above_values) if above_values else float('inf')
        }
        
        # Calculate half-life ratio (above/below)
        if half_lives['average']['below'] > 0 and not np.isinf(half_lives['average']['below']):
            half_lives['ratio'] = half_lives['average']['above'] / half_lives['average']['below']
        else:
            half_lives['ratio'] = float('inf') if half_lives['average']['above'] > 0 else 0
        
        logger.info(
            f"Half-life calculation: "
            f"below={half_lives['average']['below']:.1f} periods, "
            f"above={half_lives['average']['above']:.1f} periods, "
            f"ratio={half_lives['ratio']:.2f}"
        )
        
        return half_lives
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def plot_regime_dynamics(
        self, 
        save_path: Optional[str] = None,
        fig_size: Tuple[int, int] = (12, 10),
        dpi: int = 300
    ) -> plt.Figure:
        """
        Plot regime dynamics and adjustment speeds.
        
        Creates a visualization of the threshold dynamics including:
        - Error correction term over time with threshold
        - Adjustment speeds in each regime
        - Regime distribution
        - Half-lives comparison
        
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
        # Make sure we have model results
        if self.below_model is None or self.above_model is None:
            logger.info("Estimating TVECM first")
            self.estimate_tvecm()
        
        # Calculate half-lives
        half_lives = self.calculate_half_lives()
        
        # Set plotting style using utility
        set_plotting_style()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle(f"Threshold VECM Dynamics\nThreshold = {self.threshold:.4f}", fontsize=16)
        
        # Plot 1: Error correction term over time with threshold
        ax = axes[0, 0]
        if self.index is not None and isinstance(self.data, pd.DataFrame):
            # Use plot_time_series utility
            ec_df = pd.DataFrame({'date': self.index, 'ec_term': self.ec_term})
            plot_time_series(
                ec_df,
                x='date',
                y='ec_term',
                ax=ax,
                color='blue'
            )
        else:
            ax.plot(self.ec_term, color='blue')
            
        ax.axhline(y=self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold:.4f}')
        ax.set_title("Error Correction Term Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("EC Term")
        ax.legend()
        
        # Plot 2: Regime distribution
        ax = axes[0, 1]
        ax.hist(self.ec_term, bins=30, density=True, alpha=0.7)
        ax.axvline(x=self.threshold, color='r', linestyle='--')
        
        # Add annotation about regime proportions
        below_prop = np.mean(self.ec_term <= self.threshold)
        above_prop = 1 - below_prop
        ax.annotate(
            f"Below: {below_prop:.1%}\nAbove: {above_prop:.1%}",
            xy=(self.threshold, 0.8 * ax.get_ylim()[1]),
            xytext=(10, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )
        
        ax.set_title("Distribution of Error Correction Term")
        ax.set_xlabel("EC Term")
        ax.set_ylabel("Density")
        
        # Plot 3: Adjustment speeds comparison (alpha)
        ax = axes[1, 0]
        
        # Get alpha values for first EC term
        below_alphas = self.below_model['alpha'][:, 0]
        above_alphas = self.above_model['alpha'][:, 0]
        
        # Limit to max 5 markets for clarity
        n_markets = min(5, len(self.market_names))
        
        x = np.arange(n_markets)
        width = 0.35
        
        # Plot bars for alpha values
        bars1 = ax.bar(x - width/2, np.abs(below_alphas[:n_markets]), width, label='Below Threshold', color='skyblue')
        bars2 = ax.bar(x + width/2, np.abs(above_alphas[:n_markets]), width, label='Above Threshold', color='salmon')
        
        ax.set_title("Adjustment Speeds by Market")
        ax.set_ylabel("Absolute Alpha")
        ax.set_xticks(x)
        ax.set_xticklabels([name[:10] for name in self.market_names[:n_markets]])
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        # Plot 4: Half-life comparison
        ax = axes[1, 1]
        
        # Prepare half-life data
        below_hl = [half_lives['below'].get(market, np.nan) for market in self.market_names[:n_markets]]
        above_hl = [half_lives['above'].get(market, np.nan) for market in self.market_names[:n_markets]]
        
        # Replace inf values with a large number for plotting
        below_hl = [min(val, 50) if not np.isnan(val) else 0 for val in below_hl]
        above_hl = [min(val, 50) if not np.isnan(val) else 0 for val in above_hl]
        
        # Plot bars
        bars1 = ax.bar(x - width/2, below_hl, width, label='Below Threshold', color='skyblue')
        bars2 = ax.bar(x + width/2, above_hl, width, label='Above Threshold', color='salmon')
        
        ax.set_title("Half-Lives by Market")
        ax.set_ylabel("Half-Life (periods)")
        ax.set_xticks(x)
        ax.set_xticklabels([name[:10] for name in self.market_names[:n_markets]])
        ax.legend()
        
        # Add value labels
        for bars, values in zip([bars1, bars2], [below_hl, above_hl]):
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height >= 50:
                    label = "∞"
                elif height == 0:
                    label = "N/A"
                else:
                    label = f'{height:.1f}'
                
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=dpi)
            logger.info(f"Saved regime dynamics plot to {save_path}")
        
        return fig
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete TVECM analysis workflow.
        
        Performs all analytical steps in sequence:
        1. Estimate linear VECM
        2. Search for optimal threshold
        3. Estimate TVECM with regime-specific dynamics
        4. Calculate regime transitions and half-lives
        5. Test significance of threshold effect
        6. Run diagnostics
        
        Returns
        -------
        dict
            Complete analysis results
            
        Notes
        -----
        This comprehensive analysis provides a full assessment of
        market integration patterns in Yemen, revealing how conflict-induced
        transaction costs create threshold effects in price transmission
        between politically and geographically separated markets.
        """
        logger.info("Running full TVECM analysis")
        
        # Step 1: Estimate linear VECM
        linear_results = self.estimate_linear_vecm()
        
        # Step 2: Search for optimal threshold
        threshold_results = self.grid_search_threshold()
        
        # Step 3: Estimate TVECM
        tvecm_results = self.estimate_tvecm()
        
        # Step 4: Calculate regime transitions and half-lives
        transitions = self.calculate_regime_transition_matrix()
        half_lives = self.calculate_half_lives()
        
        # Step 5: Test significance of threshold effect
        significance_test = self.test_threshold_significance()
        
        # Step 6: Run diagnostics
        diagnostics = self.run_diagnostics()
        
        # Compile all results
        full_results = {
            'linear_vecm': linear_results,
            'threshold_search': threshold_results,
            'tvecm': tvecm_results,
            'regime_transitions': transitions,
            'half_lives': half_lives,
            'threshold_significance': significance_test,
            'diagnostics': diagnostics,
            'summary': {
                'threshold': self.threshold,
                'threshold_significant': significance_test.get('significant', False),
                'below_persistence': transitions['persistence']['below_persistence'],
                'above_persistence': transitions['persistence']['above_persistence'],
                'below_half_life': half_lives['average']['below'],
                'above_half_life': half_lives['average']['above'],
                'half_life_ratio': half_lives['ratio'],
                'residuals_valid': diagnostics['summary'].get('all_residuals_valid', False),
                'structural_breaks': diagnostics['summary'].get('has_structural_breaks', False)
            }
        }
        
        # Add market integration assessment
        full_results['market_integration_assessment'] = self._assess_market_integration(full_results)
        
        logger.info(
            f"Full analysis complete: threshold={self.threshold:.4f}, "
            f"significant={significance_test.get('significant', False)}, "
            f"half_life_ratio={half_lives['ratio']:.2f}"
        )
        
        return full_results
    
    def _assess_market_integration(self, results: Dict[str, Any]) -> str:
        """
        Assess market integration based on TVECM results.
        
        Provides a qualitative summary of market integration status.
        
        Parameters
        ----------
        results : dict
            Full analysis results
            
        Returns
        -------
        str
            Market integration assessment
        """
        # Extract key metrics
        threshold_significant = results['summary']['threshold_significant']
        below_half_life = results['summary']['below_half_life']
        above_half_life = results['summary']['above_half_life']
        half_life_ratio = results['summary']['half_life_ratio']
        below_persistence = results['summary']['below_persistence'] 
        
        # Case 1: No significant threshold effect
        if not threshold_significant:
            if below_half_life < 5:
                return (
                    "Strong Linear Integration: Markets exhibit consistent price transmission "
                    "without significant threshold effects. This suggests low transaction costs "
                    "relative to price differentials, potentially due to established trade routes "
                    "despite conflict conditions."
                )
            else:
                return (
                    "Weak Linear Integration: Markets show consistent but slow price transmission "
                    "without threshold effects. This indicates persistent barriers affecting "
                    "all price levels similarly."
                )
        
        # Case 2: Significant threshold with faster adjustment above (normal)
        elif half_life_ratio < 0.7:  # Above adjustment faster
            if above_half_life < 3:
                return (
                    "Strong Threshold Integration: Markets show rapid adjustment when price "
                    "differentials exceed transaction costs, with much slower adjustment below "
                    "threshold. This indicates significant but surmountable conflict-related barriers."
                )
            else:
                return (
                    "Moderate Threshold Integration: Markets show faster but still relatively "
                    "slow adjustment above threshold. This suggests substantial barriers with "
                    "partial arbitrage once price differentials become large enough."
                )
        
        # Case 3: Significant threshold with similar adjustment speeds
        elif 0.7 <= half_life_ratio <= 1.3:
            return (
                "Symmetric Threshold Integration: Markets show similar adjustment speeds in both "
                "regimes despite significant threshold effects. This unusual pattern may indicate "
                "complex market dynamics where the nature rather than speed of adjustment differs "
                "across regimes."
            )
        
        # Case 4: Significant threshold with faster adjustment below (unusual)
        elif half_life_ratio > 1.3:
            return (
                "Anomalous Threshold Integration: Markets show faster adjustment below threshold "
                "than above. This unexpected pattern may indicate data quality issues, omitted "
                "variables, or complex conflict dynamics requiring further investigation."
            )
        
        # Case 5: High regime persistence
        if below_persistence > 0.9:
            return (
                "Persistent Segmentation: Markets rarely transition above the threshold, suggesting "
                "severe and persistent barriers to trade that maintain price differentials below "
                "the threshold that would trigger arbitrage."
            )
            
        # Default case
        return (
            "Moderate Market Integration: Markets show evidence of price transmission with "
            "threshold effects consistent with conflict-related transaction costs."
        )


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def combine_tvecm_results(
    models: Dict[str, ThresholdVECM],
    market_pairs: Dict[str, Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Combine results from multiple TVECM models for market comparison.
    
    Synthesizes results across multiple market pairs to identify patterns
    in threshold effects and adjustment speeds.
    
    Parameters
    ----------
    models : dict
        Dictionary of ThresholdVECM model instances
    market_pairs : dict
        Dictionary mapping model keys to market pair names
        
    Returns
    -------
    dict
        Combined analysis including:
        - thresholds: Comparison of threshold values
        - adjustment_speeds: Comparison of adjustment speeds
        - half_lives: Comparison of half-lives
        - market_ranking: Markets ranked by integration strength
        - regional_patterns: Integration patterns by region
        
    Notes
    -----
    This cross-market comparison reveals how transaction costs and market
    integration vary across Yemen's fragmented geography and conflict zones,
    highlighting where barriers are strongest.
    """
    # Check inputs
    if not models:
        raise ValueError("No models provided")
    
    if not all(isinstance(model, ThresholdVECM) for model in models.values()):
        raise ValueError("All models must be ThresholdVECM instances")
    
    # Initialize result structure
    result = {
        'thresholds': {},
        'adjustment_speeds': {
            'below': {},
            'above': {}
        },
        'half_lives': {
            'below': {},
            'above': {},
            'ratio': {}
        },
        'market_pairs': market_pairs
    }
    
    # Extract thresholds
    for key, model in models.items():
        if model.threshold is not None:
            result['thresholds'][key] = model.threshold
    
    # Extract adjustment speeds (alpha)
    for key, model in models.items():
        if model.below_model is not None and model.above_model is not None:
            # Use first loading coefficient (alpha) for first cointegration relation
            result['adjustment_speeds']['below'][key] = model.below_model['alpha'][0, 0]
            result['adjustment_speeds']['above'][key] = model.above_model['alpha'][0, 0]
    
    # Extract half-lives
    for key, model in models.items():
        half_lives = model.calculate_half_lives()
        result['half_lives']['below'][key] = half_lives['average']['below']
        result['half_lives']['above'][key] = half_lives['average']['above']
        result['half_lives']['ratio'][key] = half_lives['ratio']
    
    # Market ranking by integration strength
    # Lower half-lives indicate better integration
    sorted_markets = sorted(
        result['half_lives']['above'].items(),
        key=lambda x: x[1]
    )
    
    result['market_ranking'] = [
        {'market_pair': key, 'half_life': half_life}
        for key, half_life in sorted_markets
    ]
    
    # Analyze regional patterns if market names are available
    if all('north' in pair[0].lower() or 'north' in pair[1].lower() or
           'south' in pair[0].lower() or 'south' in pair[1].lower() 
           for pair in market_pairs.values()):
        
        # Categorize market pairs
        north_north = []
        north_south = []
        south_south = []
        
        for key, pair in market_pairs.items():
            market1, market2 = pair
            
            if ('north' in market1.lower() and 'north' in market2.lower()):
                north_north.append(key)
            elif ('south' in market1.lower() and 'south' in market2.lower()):
                south_south.append(key)
            else:
                north_south.append(key)
        
        # Calculate average metrics by region
        result['regional_patterns'] = {
            'north_north': {
                'pairs': north_north,
                'avg_threshold': np.mean([result['thresholds'][key] for key in north_north if key in result['thresholds']]) if north_north else np.nan,
                'avg_half_life_above': np.mean([result['half_lives']['above'][key] for key in north_north if key in result['half_lives']['above']]) if north_north else np.nan
            },
            'north_south': {
                'pairs': north_south,
                'avg_threshold': np.mean([result['thresholds'][key] for key in north_south if key in result['thresholds']]) if north_south else np.nan,
                'avg_half_life_above': np.mean([result['half_lives']['above'][key] for key in north_south if key in result['half_lives']['above']]) if north_south else np.nan
            },
            'south_south': {
                'pairs': south_south,
                'avg_threshold': np.mean([result['thresholds'][key] for key in south_south if key in result['thresholds']]) if south_south else np.nan,
                'avg_half_life_above': np.mean([result['half_lives']['above'][key] for key in south_south if key in result['half_lives']['above']]) if south_south else np.nan
            }
        }
    
    # Add summary statistics
    result['summary'] = {
        'avg_threshold': np.mean(list(result['thresholds'].values())) if result['thresholds'] else np.nan,
        'min_threshold': np.min(list(result['thresholds'].values())) if result['thresholds'] else np.nan,
        'max_threshold': np.max(list(result['thresholds'].values())) if result['thresholds'] else np.nan,
        'avg_half_life_below': np.mean([x for x in result['half_lives']['below'].values() if not np.isinf(x)]) if result['half_lives']['below'] else np.nan,
        'avg_half_life_above': np.mean([x for x in result['half_lives']['above'].values() if not np.isinf(x)]) if result['half_lives']['above'] else np.nan,
        'best_integrated_pair': sorted_markets[0][0] if sorted_markets else None,
        'worst_integrated_pair': sorted_markets[-1][0] if sorted_markets else None
    }
    
    # Generate an interpretation
    result['interpretation'] = _interpret_combined_results(result)
    
    return result


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def _interpret_combined_results(result: Dict[str, Any]) -> str:
    """
    Generate interpretation of combined TVECM results.
    
    Provides insights into market integration patterns across
    different regions and market pairs in Yemen.
    
    Parameters
    ----------
    result : dict
        Combined analysis result
        
    Returns
    -------
    str
        Text interpretation of combined results
    """
    # Extract summary metrics
    avg_threshold = result['summary'].get('avg_threshold', np.nan)
    min_threshold = result['summary'].get('min_threshold', np.nan)
    max_threshold = result['summary'].get('max_threshold', np.nan)
    avg_hl_below = result['summary'].get('avg_half_life_below', np.nan)
    avg_hl_above = result['summary'].get('avg_half_life_above', np.nan)
    best_pair = result['summary'].get('best_integrated_pair', None)
    worst_pair = result['summary'].get('worst_integrated_pair', None)
    
    # Initialize interpretation text
    parts = []
    
    # Interpret threshold variation
    if not np.isnan(min_threshold) and not np.isnan(max_threshold):
        threshold_range = max_threshold - min_threshold
        if threshold_range > 0.5 * avg_threshold:
            parts.append(
                f"Transaction costs (thresholds) vary substantially across market pairs "
                f"(range: {min_threshold:.2f}-{max_threshold:.2f}), indicating highly "
                f"heterogeneous market conditions across Yemen's conflict-affected landscape."
            )
        else:
            parts.append(
                f"Transaction costs (thresholds) are relatively consistent across market pairs "
                f"(range: {min_threshold:.2f}-{max_threshold:.2f}), suggesting similar arbitrage "
                f"constraints despite varied conflict conditions."
            )
    
    # Interpret adjustment speed differences
    if not np.isnan(avg_hl_below) and not np.isnan(avg_hl_above):
        hl_ratio = avg_hl_below / avg_hl_above if avg_hl_above > 0 else float('inf')
        
        if hl_ratio > 3:
            parts.append(
                f"Price adjustment is substantially faster ({hl_ratio:.1f}x) above the threshold "
                f"than below, indicating strong threshold effects where price convergence accelerates "
                f"once differentials exceed conflict-related transaction costs."
            )
        elif hl_ratio > 1.5:
            parts.append(
                f"Price adjustment is moderately faster ({hl_ratio:.1f}x) above the threshold "
                f"than below, consistent with threshold effects from conflict-related barriers."
            )
        else:
            parts.append(
                f"Price adjustment speeds show limited difference between regimes (ratio: {hl_ratio:.1f}), "
                f"suggesting that while thresholds exist, they don't dramatically alter adjustment dynamics."
            )
    
    # Interpret regional patterns if available
    if 'regional_patterns' in result:
        patterns = result['regional_patterns']
        
        # Compare within-region and cross-region integration
        nn_hl = patterns['north_north'].get('avg_half_life_above', np.nan)
        ss_hl = patterns['south_south'].get('avg_half_life_above', np.nan)
        ns_hl = patterns['north_south'].get('avg_half_life_above', np.nan)
        
        if not np.isnan(nn_hl) and not np.isnan(ss_hl) and not np.isnan(ns_hl):
            within_hl = (nn_hl + ss_hl) / 2
            
            if ns_hl > 2 * within_hl:
                parts.append(
                    f"Cross-regime market integration (between north and south) is substantially "
                    f"weaker than within-regime integration, with half-lives {ns_hl/within_hl:.1f}x longer. "
                    f"This indicates significant barriers between exchange rate regimes, likely "
                    f"due to the dual exchange rate system and political fragmentation."
                )
            elif ns_hl > 1.3 * within_hl:
                parts.append(
                    f"Cross-regime market integration is moderately weaker than within-regime integration, "
                    f"with half-lives {ns_hl/within_hl:.1f}x longer. This suggests barriers between "
                    f"exchange rate regimes affect market integration but don't completely segment markets."
                )
            else:
                parts.append(
                    f"Cross-regime and within-regime market integration show similar strength, "
                    f"suggesting the dual exchange rate system has limited impact on price transmission "
                    f"between north and south markets once adjusting for transaction costs."
                )
    
    # Highlight strongest and weakest market pairs
    if best_pair and worst_pair and best_pair in result['market_pairs'] and worst_pair in result['market_pairs']:
        parts.append(
            f"The strongest market integration is between {result['market_pairs'][best_pair][0]} and "
            f"{result['market_pairs'][best_pair][1]}, while the weakest integration is between "
            f"{result['market_pairs'][worst_pair][0]} and {result['market_pairs'][worst_pair][1]}. "
            f"This pattern highlights how geographic proximity, conflict intensity, and "
            f"political fragmentation create variable market integration conditions."
        )
    
    # Combine all parts
    interpretation = " ".join(parts)
    
    return interpretation