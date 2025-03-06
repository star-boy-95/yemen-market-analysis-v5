"""
Threshold Vector Error Correction Model implementation.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM

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


class ThresholdVECM:
    """
    Threshold Vector Error Correction Model (TVECM) implementation.
    
    This class implements a two-regime threshold VECM following
    Hansen & Seo (2002) methodology.
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
            The endogenous variables
        k_ar_diff : int, optional
            Number of lagged differences in the model
        deterministic : str, optional
            "n" - no deterministic terms
            "co" - constant outside the cointegration relation
            "ci" - constant inside the cointegration relation
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
            Trimming percentage
        n_grid : int, optional
            Number of grid points
        verbose : bool, optional
            Whether to print progress
            
        Returns
        -------
        dict
            Threshold estimation results
        """
        # Ensure linear VECM is estimated
        if not hasattr(self, 'linear_results') or self.linear_results is None:
            logger.info("Estimating linear VECM first")
            self.estimate_linear_vecm()
        
        # Validate parameters
        self._validate_grid_search_params(trim, n_grid)
        
        # Get cointegration relation and calculate equilibrium errors
        eq_errors = self._calculate_equilibrium_errors()
        
        # Generate threshold candidates
        candidates = self._generate_threshold_candidates(eq_errors, trim, n_grid)
        
        # Perform grid search
        return self._perform_grid_search(eq_errors, candidates, verbose)
    
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
        """Calculate equilibrium errors from cointegration relation."""
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
    def estimate_tvecm(self) -> Dict[str, Any]:
        """
        Estimate the Threshold VECM.
        
        Returns
        -------
        dict
            TVECM estimation results
        """
        # Ensure threshold is estimated
        if self.threshold is None:
            logger.info("Estimating threshold first")
            self.grid_search_threshold()
        
        # Calculate equilibrium errors
        eq_errors = self._calculate_equilibrium_errors()
        
        # Estimate threshold model
        regime_model = self._estimate_regime_model(eq_errors)
        
        # Extract and format results
        self.results = self._format_model_results(regime_model, eq_errors)
        
        logger.info(
            f"TVECM estimation complete: threshold={self.threshold:.4f}, "
            f"llf={self.results['llf']:.4f}"
        )
        
        return self.results
    
    def _estimate_regime_model(self, eq_errors: np.ndarray) -> Dict[str, Any]:
        """Estimate the TVECM model with threshold regimes."""
        # Create indicator functions
        below = eq_errors <= self.threshold
        above = ~below
        
        # Prepare data
        y_diff = np.diff(self.data.values, axis=0)
        
        # Create design matrix
        X = self._create_regime_design_matrix(below, above, eq_errors)
        
        # Fit model
        model = sm.OLS(y_diff, X)
        return model.fit()
    
    def _format_model_results(
        self, 
        results: Any, 
        eq_errors: np.ndarray
    ) -> Dict[str, Any]:
        """Extract and format results from the estimated model."""
        n_vars = self.data.shape[1]
        n_params_per_regime = 2 + self.k_ar_diff * n_vars
        
        # Extract regime-specific parameters
        params_below = results.params[:n_params_per_regime]
        params_above = results.params[n_params_per_regime:]
        
        # Extract adjustment speeds
        alpha_below = params_below[1:1+n_vars]
        alpha_above = params_above[1:1+n_vars]
        
        return {
            'model': results,
            'threshold': self.threshold,
            'cointegration_beta': self.linear_results.beta,
            'equilibrium_errors': eq_errors,
            'alpha_below': alpha_below,
            'alpha_above': alpha_above,
            'params_below': params_below,
            'params_above': params_above,
            'below_regime': {
                'alpha': alpha_below,
                'constant': params_below[0],
                'short_run': params_below[1+n_vars:]
            },
            'above_regime': {
                'alpha': alpha_above,
                'constant': params_above[0],
                'short_run': params_above[1+n_vars:]
            },
            'llf': results.llf,
            'aic': results.aic,
            'bic': results.bic
        }
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def test_threshold_significance(
        self, 
        n_bootstrap: int = 1000, 
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test for threshold effect using bootstrap.
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap replications
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Test results
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
        llf_diff = 2 * (self.results['llf'] - self.linear_results.llf)
        
        # Bootstrap procedure - implementation simplified for conciseness
        # In practice, this would require resampling residuals and re-estimating both models
        logger.warning("Threshold significance test using simplified approach")
        
        # For now, return placeholder result
        return {
            'test_statistic': llf_diff,
            'p_value': None,  # Would be calculated from bootstrap distribution
            'significant': None,
            'bootstrap_statistics': None
        }


@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def calculate_regime_transition_matrix(
    eq_errors: np.ndarray, 
    threshold: float
) -> pd.DataFrame:
    """
    Calculate regime transition matrix.
    
    Parameters
    ----------
    eq_errors : array_like
        Equilibrium errors
    threshold : float
        Threshold value
        
    Returns
    -------
    pandas.DataFrame
        Transition matrix
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
    transition_probs = transitions / row_sums
    
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
    tvecm_results: Dict[str, Any]
) -> Dict[str, List[float]]:
    """
    Calculate half-lives for each variable in each regime.
    
    Parameters
    ----------
    tvecm_results : dict
        TVECM estimation results
        
    Returns
    -------
    dict
        Half-lives for each variable in each regime
    """
    # Extract adjustment speeds
    alpha_below = tvecm_results['alpha_below']
    alpha_above = tvecm_results['alpha_above']
    
    # Helper function for half-life calculation
    def calc_half_life(alpha: float) -> float:
        """Calculate half-life for single adjustment parameter."""
        if alpha < 0:
            return np.log(0.5) / np.log(1 + alpha)
        else:
            return float('inf')
    
    # Calculate half-lives using vectorized operations
    half_lives_below = [calc_half_life(a) for a in alpha_below]
    half_lives_above = [calc_half_life(a) for a in alpha_above]
    
    return {
        'below_regime': half_lives_below,
        'above_regime': half_lives_above
    }