"""
Enders & Siklos (2001) threshold cointegration model implementation.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import statsmodels.api as sm
from scipy import stats

from core.decorators import error_handler, performance_tracker
from core.exceptions import ModelError, ThresholdModelError
from models.base import ThresholdModel

logger = logging.getLogger(__name__)


class EndersSiklosModel(ThresholdModel):
    """Enders & Siklos (2001) threshold autoregression model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Enders-Siklos model."""
        super().__init__(config)
        self.model_type = "enders_siklos"
        self.subtype = "TAR"  # Default to TAR, can be changed to MTAR
        self.beta = None
        self.intercept = None
        self.residuals = None
    
    @error_handler(fallback_value={})
    @performance_tracker("EndersSiklos.fit")
    def fit(
        self, 
        y: Union[np.ndarray, pd.Series], 
        x: Union[np.ndarray, pd.Series],
        commodity: Optional[str] = None,
        threshold_range: Optional[Tuple[float, float]] = None,
        model_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Fit Enders & Siklos threshold cointegration model.
        
        Args:
            y: Dependent variable (north prices)
            x: Independent variable (south prices)
            commodity: Commodity name
            threshold_range: Optional (lower, upper) bounds for threshold
            model_type: 'TAR', 'MTAR', or 'auto' (select best)
            
        Returns:
            Dictionary of results
        """
        # Convert to numpy arrays
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.Series):
            x = x.values
        
        # Validate inputs
        if not self.validate_inputs(y, x):
            raise ThresholdModelError("Invalid input data")
        
        # Step 1: Estimate linear cointegrating relationship
        X = sm.add_constant(x)
        linear_model = sm.OLS(y, X).fit()
        self.beta = linear_model.params[1]
        self.intercept = linear_model.params[0]
        
        # Calculate cointegrating residuals
        self.residuals = y - self.intercept - self.beta * x
        
        # Determine model type (TAR or MTAR)
        if model_type.lower() == 'auto':
            # Run both models and select the best one
            tar_results = self._fit_specific_model(
                self.residuals, 'TAR', threshold_range, commodity)
            mtar_results = self._fit_specific_model(
                self.residuals, 'MTAR', threshold_range, commodity)
            
            # Compare information criteria (lower is better)
            if tar_results.get('model_fit', {}).get('aic', float('inf')) <= \
               mtar_results.get('model_fit', {}).get('aic', float('inf')):
                self.results = tar_results
                self.subtype = 'TAR'
            else:
                self.results = mtar_results
                self.subtype = 'MTAR'
                
        else:
            # Use specified model type
            self.subtype = model_type.upper()
            self.results = self._fit_specific_model(
                self.residuals, self.subtype, threshold_range, commodity)
        
        self.is_fitted = True
        return self.results
    
    @error_handler(fallback_value={})
    def _fit_specific_model(
        self,
        residuals: np.ndarray,
        model_type: str,
        threshold_range: Optional[Tuple[float, float]] = None,
        commodity: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fit specific model type (TAR or MTAR)."""
        # Set up model parameters
        maxlags = self.config.get('max_lags', 4)
        grid_points = self.config.get('grid_points', 50)
        min_regime_size = self.config.get('min_regime_size', 0.1)
        
        # Step 2: Estimate optimal threshold
        self.threshold, self.test_statistic = self._run_grid_search(
            residuals,
            maxlags,
            model_type,
            grid_points=grid_points,
            min_regime_size=min_regime_size,
            threshold_range=threshold_range
        )
        
        # Step 3: Test threshold significance
        nboot = self.config.get('nboot', 500)
        block_size = self.config.get('block_size', 5)
        
        self.p_value, self.threshold_significant = self._bootstrap_test(
            residuals,
            maxlags,
            model_type,
            self.threshold,
            self.test_statistic,
            nboot=nboot,
            block_size=block_size
        )
        
        # Step 4: Estimate regime dynamics
        regime_dynamics = self._estimate_regime_dynamics(
            residuals,
            maxlags,
            model_type,
            self.threshold
        )
        
        # Step 5: Compile results
        results = {
            "model_type": f"enders_siklos_{model_type.lower()}",
            "commodity": commodity,
            "threshold": float(self.threshold),
            "test_statistic": float(self.test_statistic),
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "threshold_significant": self.threshold_significant,
            "beta": float(self.beta),
            "intercept": float(self.intercept),
            "regime_balance": self.calculate_regime_balance(residuals, self.threshold),
            **regime_dynamics
        }
        
        return results
    
    @error_handler(fallback_value=0.0)
    def estimate_threshold(
        self,
        residuals: np.ndarray,
        maxlags: int,
        model_type: str = 'TAR',
        grid_points: int = 50,
        min_regime_size: float = 0.1,
        threshold_range: Optional[Tuple[float, float]] = None
    ) -> float:
        """Estimate optimal threshold value."""
        threshold, _ = self._run_grid_search(
            residuals, maxlags, model_type, grid_points, 
            min_regime_size, threshold_range
        )
        return threshold
    
    @error_handler(fallback_value=(0.0, False))
    def test_threshold_significance(
        self,
        residuals: np.ndarray,
        maxlags: int,
        model_type: str,
        threshold: float,
        test_statistic: float,
        nboot: int = 500,
        block_size: int = 5
    ) -> Tuple[float, bool]:
        """Test significance of threshold effect."""
        return self._bootstrap_test(
            residuals, maxlags, model_type, threshold, test_statistic, 
            nboot=nboot, block_size=block_size
        )
    
    @error_handler(fallback_value={})
    def estimate_regime_dynamics(
        self,
        residuals: np.ndarray,
        maxlags: int,
        model_type: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Estimate regime-specific dynamics."""
        return self._estimate_regime_dynamics(
            residuals, maxlags, model_type, threshold
        )
    
    @error_handler(fallback_value=(0.0, 0.0))
    def _run_grid_search(
        self,
        residuals: np.ndarray,
        maxlags: int,
        model_type: str,
        grid_points: int = 50,
        min_regime_size: float = 0.1,
        threshold_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """Run grid search for optimal threshold value."""
        n = len(residuals)
        
        # Determine variable to use for threshold determination
        if model_type == 'TAR':
            # Use levels
            indicator_var = residuals[:-1]
        else:  # MTAR
            # Use differences
            indicator_var = np.diff(residuals)
        
        # Create lagged matrix for adjustment process
        u_lag = residuals[:-1]
        du = np.diff(residuals)
        
        # Create lag matrices
        if maxlags > 0:
            X_lags = np.zeros((n - maxlags - 1, maxlags))
            for i in range(maxlags):
                X_lags[:, i] = du[maxlags-i-1:n-i-2]
            
            # Trim to match dimensions
            u_lag = u_lag[maxlags:]
            indicator_var = indicator_var[maxlags:] if len(indicator_var) > maxlags else indicator_var
        else:
            X_lags = np.zeros((n - 1, 0))
        
        # Set up grid
        if threshold_range is None:
            # Sort indicator variable for percentile-based grid
            z_sorted = np.sort(indicator_var)
            n_ind = len(z_sorted)
            
            # Trim to ensure minimum regime size
            lower_idx = int(n_ind * min_regime_size)
            upper_idx = int(n_ind * (1 - min_regime_size))
            
            # Create grid of threshold values
            grid_values = np.linspace(z_sorted[lower_idx], z_sorted[upper_idx], grid_points)
        else:
            # Use provided threshold range
            grid_values = np.linspace(threshold_range[0], threshold_range[1], grid_points)
        
        # Initialize results
        best_ssr = np.inf
        best_threshold = None
        best_test_stat = 0.0
        
        # Run grid search
        for gamma in grid_values:
            # Create regime indicators
            if model_type == 'TAR':
                I_regime = (u_lag >= gamma).astype(float)
            else:  # MTAR
                I_regime = (indicator_var >= gamma).astype(float)
                
            I_comp = 1 - I_regime
            
            # Create design matrix
            if maxlags > 0:
                X = np.column_stack([
                    np.ones(n - maxlags - 1),       # Constant
                    I_regime * u_lag,               # Upper regime
                    I_comp * u_lag,                 # Lower regime
                    X_lags                          # Lagged differences
                ])
            else:
                X = np.column_stack([
                    np.ones(n - 1),                 # Constant
                    I_regime * u_lag,               # Upper regime
                    I_comp * u_lag,                 # Lower regime
                ])
            
            # Dependent variable
            y = du[maxlags:] if maxlags > 0 else du
            
            # Check regime balance
            below_pct = np.mean(I_comp)
            above_pct = np.mean(I_regime)
            
            if below_pct < min_regime_size or above_pct < min_regime_size:
                continue
            
            # Fit model
            try:
                model = sm.OLS(y, X).fit()
                
                # Calculate SSR
                ssr = np.sum(model.resid ** 2)
                
                # Calculate Wald test for threshold effect
                r_matrix = np.zeros((1, X.shape[1]))
                r_matrix[0, 1:3] = [1, -1]  # Test equality of regime coefficients
                
                wald_test = model.wald_test(r_matrix)
                test_stat = wald_test.statistic[0, 0]
                
                # Update best threshold
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_threshold = gamma
                    best_test_stat = test_stat
            except:
                continue
        
        if best_threshold is None:
            # If no valid threshold found, use median
            best_threshold = np.median(grid_values)
            best_test_stat = 0.0
            
        return best_threshold, best_test_stat
    
    @error_handler(fallback_value=(1.0, False))
    def _bootstrap_test(
        self,
        residuals: np.ndarray,
        maxlags: int,
        model_type: str,
        threshold: float,
        test_statistic: float,
        nboot: int = 500,
        block_size: int = 5
    ) -> Tuple[float, bool]:
        """Perform bootstrap inference for threshold significance."""
        n = len(residuals)
        
        if n <= maxlags + block_size:
            return 1.0, False
        
        # Calculate differenced residuals
        du = np.diff(residuals)
        
        # Estimate linear model (no threshold)
        u_lag = residuals[:-1]
        
        # Create lagged matrix for adjustment process
        if maxlags > 0:
            X_lags = np.zeros((n - maxlags - 1, maxlags))
            for i in range(maxlags):
                X_lags[:, i] = du[maxlags-i-1:n-i-2]
                
            X_linear = np.column_stack([
                np.ones(n - maxlags - 1),  # Constant
                u_lag[maxlags:],           # z_t-1
                X_lags                     # Lagged differences
            ])
            
            y = du[maxlags:]
        else:
            X_linear = np.column_stack([
                np.ones(n - 1),            # Constant
                u_lag                      # z_t-1
            ])
            
            y = du
        
        # Fit linear model
        linear_model = sm.OLS(y, X_linear).fit()
        
        # Get residuals from linear model
        linear_resid = linear_model.resid
        
        # Bootstrap distribution of test statistics
        boot_stats = np.zeros(nboot)
        
        for b in range(nboot):
            # Generate bootstrap sample with block bootstrap
            if block_size > 1:
                # Block bootstrap
                n_blocks = int(np.ceil(len(linear_resid) / block_size))
                block_indices = np.random.choice(len(linear_resid) - block_size + 1, n_blocks, replace=True)
                boot_indices = np.concatenate([np.arange(i, i + block_size) for i in block_indices])
                boot_indices = boot_indices[:len(linear_resid)]
                boot_resid = linear_resid[boot_indices]
            else:
                # Simple bootstrap
                boot_indices = np.random.choice(len(linear_resid), len(linear_resid), replace=True)
                boot_resid = linear_resid[boot_indices]
            
            # Generate bootstrap series
            boot_du = X_linear @ linear_model.params + boot_resid
            
            # Reconstruct levels
            boot_u = np.zeros(n)
            boot_u[0] = residuals[0]
            
            for i in range(len(boot_du)):
                boot_u[i+1] = boot_u[i] + boot_du[i]
            
            # Run grid search on bootstrap sample
            grid_values = np.linspace(np.percentile(boot_u, 15), np.percentile(boot_u, 85), 20)
            
            _, boot_test_stat = self._run_grid_search(
                boot_u,
                maxlags,
                model_type,
                grid_points=20,
                min_regime_size=self.config.get('min_regime_size', 0.1)
            )
            
            boot_stats[b] = boot_test_stat
        
        # Calculate p-value
        p_value = np.mean(boot_stats >= test_statistic)
        
        # Test significance
        threshold_significant = p_value < self.config.get('significance_level', 0.05)
        
        return p_value, threshold_significant
    
    @error_handler(fallback_value={})
    def _estimate_regime_dynamics(
        self,
        residuals: np.ndarray,
        maxlags: int,
        model_type: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Estimate regime-specific adjustment dynamics."""
        n = len(residuals)
        
        # Calculate differenced residuals
        du = np.diff(residuals)
        u_lag = residuals[:-1]
        
        # Determine variable for threshold determination
        if model_type == 'TAR':
            indicator_var = u_lag
        else:  # MTAR
            indicator_var = du
        
        # Create lagged matrix for adjustment process
        if maxlags > 0:
            X_lags = np.zeros((n - maxlags - 1, maxlags))
            for i in range(maxlags):
                X_lags[:, i] = du[maxlags-i-1:n-i-2]
            
            # Trim to match dimensions
            u_lag = u_lag[maxlags:]
            indicator_var = indicator_var[maxlags:] if len(indicator_var) >= maxlags + 1 else indicator_var[:-(maxlags + 1)]
        else:
            X_lags = np.zeros((n - 1, 0))
        
        # Create regime indicators
        if model_type == 'TAR':
            I_regime = (u_lag >= threshold).astype(float)
        else:  # MTAR
            I_regime = (indicator_var >= threshold).astype(float)
            
        I_comp = 1 - I_regime
        
        # Create design matrix
        if maxlags > 0:
            X = np.column_stack([
                np.ones(n - maxlags - 1),       # Constant
                I_regime * u_lag,               # Upper regime
                I_comp * u_lag,                 # Lower regime
                X_lags                          # Lagged differences
            ])
            
            y = du[maxlags:]
        else:
            X = np.column_stack([
                np.ones(n - 1),                 # Constant
                I_regime * u_lag,               # Upper regime
                I_comp * u_lag,                 # Lower regime
            ])
            
            y = du
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Extract adjustment parameters
        alpha_up = model.params[1]    # Adjustment when above threshold
        alpha_down = model.params[2]  # Adjustment when below threshold
        
        alpha_up_se = model.bse[1]
        alpha_down_se = model.bse[2]
        
        # Calculate half-lives
        half_life_up = self.calculate_half_life(alpha_up)
        half_life_down = self.calculate_half_life(alpha_down)
        
        # Test asymmetry
        asymmetry_significant, asymmetry_pvalue = self.test_asymmetry(
            alpha_up, alpha_down, alpha_up_se, alpha_down_se
        )
        
        return {
            "adjustment_dynamics": {
                "alpha_up": float(alpha_up),
                "alpha_down": float(alpha_down),
                "alpha_up_se": float(alpha_up_se),
                "alpha_down_se": float(alpha_down_se),
                "half_life_up": float(half_life_up),
                "half_life_down": float(half_life_down),
                "asymmetry_significant": asymmetry_significant,
                "asymmetry_pvalue": float(asymmetry_pvalue)
            },
            "model_fit": {
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "log_likelihood": float(model.llf)
            }
        }