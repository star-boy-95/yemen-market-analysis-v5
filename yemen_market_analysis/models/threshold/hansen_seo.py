"""
Hansen & Seo threshold cointegration model implementation.
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


class HansenSeoModel(ThresholdModel):
    """Hansen & Seo (2002) threshold cointegration model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Hansen-Seo model."""
        super().__init__(config)
        self.model_type = "hansen_seo"
        self.beta = None
        self.intercept = None
        self.optimal_lag = None
        self.residuals = None
    
    @error_handler(fallback_value={})
    @performance_tracker("HansenSeo.fit")
    def fit(
        self, 
        y: Union[np.ndarray, pd.Series], 
        x: Union[np.ndarray, pd.Series],
        commodity: Optional[str] = None,
        threshold_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Fit Hansen & Seo threshold cointegration model.
        
        Args:
            y: Dependent variable (north prices)
            x: Independent variable (south prices)
            commodity: Commodity name
            threshold_range: Optional (lower, upper) bounds for threshold
            
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
        
        # Step 2: Find optimal lag length
        max_lags = self.config.get('max_lags', 8)
        ic_type = self.config.get('ic_type', 'aic')
        self.optimal_lag = self._select_optimal_lag(self.residuals, max_lags, ic_type)
        
        # Step 3: Estimate threshold
        grid_points = self.config.get('grid_points', 50)
        min_regime_size = self.config.get('min_regime_size', 0.1)
        use_hac = self.config.get('use_hac', True)
        hac_maxlags = self.config.get('hac_maxlags', None)
        
        if threshold_range is None:
            # Sort residuals for percentile-based grid
            z_sorted = np.sort(self.residuals)
            n = len(z_sorted)
            
            # Trim to ensure minimum regime size
            lower_idx = int(n * min_regime_size)
            upper_idx = int(n * (1 - min_regime_size))
            
            # Create grid of threshold values
            grid_values = np.linspace(z_sorted[lower_idx], z_sorted[upper_idx], grid_points)
        else:
            # Use provided threshold range
            grid_values = np.linspace(threshold_range[0], threshold_range[1], grid_points)
        
        # Step 4: Run grid search for optimal threshold
        self.threshold, self.test_statistic = self._run_grid_search(
            self.residuals, 
            self.optimal_lag, 
            grid_values,
            min_regime_size=min_regime_size,
            use_hac=use_hac,
            hac_maxlags=hac_maxlags
        )
        
        # Step 5: Test threshold significance
        nboot = self.config.get('nboot', 500)
        block_size = self.config.get('block_size', 5)
        
        self.p_value, self.threshold_significant = self._bootstrap_test(
            self.residuals,
            self.optimal_lag,
            self.threshold,
            self.test_statistic,
            nboot=nboot,
            block_size=block_size,
            use_hac=use_hac,
            hac_maxlags=hac_maxlags
        )
        
        # Step 6: Estimate regime dynamics
        regime_dynamics = self._estimate_regime_dynamics(
            self.residuals,
            self.optimal_lag,
            self.threshold,
            use_hac=use_hac,
            hac_maxlags=hac_maxlags
        )
        
        # Step 7: Compile results
        self.results = {
            "model_type": self.model_type,
            "commodity": commodity,
            "threshold": float(self.threshold),
            "test_statistic": float(self.test_statistic),
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "threshold_significant": self.threshold_significant,
            "beta": float(self.beta),
            "intercept": float(self.intercept),
            "optimal_lag": int(self.optimal_lag),
            "regime_balance": self.calculate_regime_balance(self.residuals, self.threshold),
            **regime_dynamics
        }
        
        self.is_fitted = True
        return self.results
    
    @error_handler(fallback_value=0.0)
    def estimate_threshold(
        self,
        residuals: np.ndarray,
        lag: int,
        grid_values: np.ndarray,
        min_regime_size: float = 0.1
    ) -> float:
        """Estimate optimal threshold value."""
        threshold, _ = self._run_grid_search(residuals, lag, grid_values, min_regime_size)
        return threshold
    
    @error_handler(fallback_value=(0.0, False))
    def test_threshold_significance(
        self,
        residuals: np.ndarray,
        lag: int,
        threshold: float,
        test_statistic: float,
        nboot: int = 500,
        block_size: int = 5
    ) -> Tuple[float, bool]:
        """Test significance of threshold effect."""
        return self._bootstrap_test(
            residuals, lag, threshold, test_statistic, 
            nboot=nboot, block_size=block_size
        )
    
    @error_handler(fallback_value={})
    def estimate_regime_dynamics(
        self,
        residuals: np.ndarray,
        lag: int,
        threshold: float,
        use_hac: bool = True,
        hac_maxlags: Optional[int] = None
    ) -> Dict[str, Any]:
        """Estimate regime-specific dynamics."""
        return self._estimate_regime_dynamics(
            residuals, lag, threshold, use_hac=use_hac, hac_maxlags=hac_maxlags
        )
    
    @staticmethod
    def _select_optimal_lag(series: np.ndarray, max_lags: int, ic_type: str = 'aic') -> int:
        """Select optimal lag length using information criteria."""
        n = len(series)
        
        if max_lags >= n // 2:
            max_lags = n // 2 - 1
            
        if max_lags < 1:
            return 1
            
        # Calculate differenced series
        d_series = np.diff(series)
        
        # Set up data for each lag
        ic_values = np.zeros(max_lags)
        
        for lag in range(1, max_lags + 1):
            # Create lag matrix
            y = d_series[lag:]
            X = np.zeros((len(y), lag + 1))
            X[:, 0] = 1  # Constant
            
            for i in range(1, lag + 1):
                X[:, i] = series[lag-i:-i]
            
            # Fit model
            try:
                model = sm.OLS(y, X).fit()
                
                if ic_type.lower() == 'aic':
                    ic_values[lag-1] = model.aic
                elif ic_type.lower() == 'bic':
                    ic_values[lag-1] = model.bic
                else:
                    ic_values[lag-1] = model.aic
            except:
                ic_values[lag-1] = np.inf
        
        # Find lag with minimum information criterion
        optimal_lag = np.argmin(ic_values) + 1
        
        return optimal_lag
    
    @error_handler(fallback_value=(0.0, 0.0))
    def _run_grid_search(
        self,
        residuals: np.ndarray,
        lag: int,
        grid_values: np.ndarray,
        min_regime_size: float = 0.1,
        use_hac: bool = True,
        hac_maxlags: Optional[int] = None
    ) -> Tuple[float, float]:
        """Run grid search for optimal threshold value."""
        n = len(residuals)
        
        # Create lagged residuals
        z_lag = residuals[:-1]
        dz = np.diff(residuals)
        
        # Create lagged matrix for differenced residuals
        if lag > 0:
            X_lags = np.zeros((n - lag - 1, lag))
            for i in range(lag):
                X_lags[:, i] = dz[lag-i-1:n-i-2]
        else:
            X_lags = np.zeros((n - 1, 1))
        
        # Initialize results
        best_test_stat = -np.inf
        best_threshold = None
        
        # Run grid search
        for gamma in grid_values:
            # Create regime indicators
            below_threshold = (z_lag[lag:] <= gamma)
            above_threshold = ~below_threshold
            
            # Check regime balance
            below_pct = below_threshold.mean()
            above_pct = 1 - below_pct
            
            if below_pct < min_regime_size or above_pct < min_regime_size:
                continue
            
            # Create design matrix
            if lag > 0:
                X = np.column_stack([
                    np.ones(n - lag - 1),                  # Constant
                    below_threshold * z_lag[lag:],         # z_t-1 * I(z_t-1 <= gamma)
                    above_threshold * z_lag[lag:],         # z_t-1 * I(z_t-1 > gamma)
                    X_lags                                 # Lagged differences
                ])
            else:
                X = np.column_stack([
                    np.ones(n - 1),                        # Constant
                    below_threshold * z_lag,               # z_t-1 * I(z_t-1 <= gamma)
                    above_threshold * z_lag                # z_t-1 * I(z_t-1 > gamma)
                ])
            
            # Dependent variable
            y = dz[lag:]
            
            # Fit model
            try:
                if use_hac:
                    cov_type = 'HAC'
                    cov_kwds = {'maxlags': hac_maxlags or int(n**0.25)}
                else:
                    cov_type = 'nonrobust'
                    cov_kwds = {}
                    
                model = sm.OLS(y, X).fit(cov_type=cov_type, cov_kwds=cov_kwds)
                
                # Calculate Wald test statistic for threshold effect
                restriction = np.zeros(model.params.shape)
                restriction[1] = 1
                restriction[2] = -1
                
                r_matrix = np.zeros((1, len(model.params)))
                r_matrix[0, 1:3] = [1, -1]  # Test alpha_down = alpha_up
                
                wald_test = model.wald_test(r_matrix)
                test_stat = wald_test.statistic[0, 0]
                
                # Update best threshold
                if test_stat > best_test_stat:
                    best_test_stat = test_stat
                    best_threshold = gamma
            except:
                continue
        
        if best_threshold is None:
            # If no valid threshold found, use middle of range
            best_threshold = np.median(grid_values)
            best_test_stat = 0.0
            
        return best_threshold, best_test_stat
    
    @error_handler(fallback_value=(1.0, False))
    def _bootstrap_test(
        self,
        residuals: np.ndarray,
        lag: int,
        threshold: float,
        test_statistic: float,
        nboot: int = 500,
        block_size: int = 5,
        use_hac: bool = True,
        hac_maxlags: Optional[int] = None
    ) -> Tuple[float, bool]:
        """Perform bootstrap inference for threshold significance."""
        n = len(residuals)
        
        if n <= lag + block_size:
            return 1.0, False
        
        # Calculate differenced residuals
        dz = np.diff(residuals)
        
        # Estimate linear model (no threshold)
        z_lag = residuals[:-1]
        
        # Create lagged matrix for differenced residuals
        if lag > 0:
            X_lags = np.zeros((n - lag - 1, lag))
            for i in range(lag):
                X_lags[:, i] = dz[lag-i-1:n-i-2]
                
            X_linear = np.column_stack([
                np.ones(n - lag - 1),  # Constant
                z_lag[lag:],           # z_t-1
                X_lags                 # Lagged differences
            ])
        else:
            X_linear = np.column_stack([
                np.ones(n - 1),        # Constant
                z_lag                  # z_t-1
            ])
        
        # Dependent variable
        y = dz[lag:]
        
        # Fit linear model
        if use_hac:
            cov_type = 'HAC'
            cov_kwds = {'maxlags': hac_maxlags or int(n**0.25)}
        else:
            cov_type = 'nonrobust'
            cov_kwds = {}
            
        linear_model = sm.OLS(y, X_linear).fit(cov_type=cov_type, cov_kwds=cov_kwds)
        
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
            boot_dz = X_linear @ linear_model.params + boot_resid
            
            # Reconstruct levels
            boot_z = np.zeros(n)
            boot_z[0] = residuals[0]
            boot_z[1:lag+1] = residuals[1:lag+1]  # Use original values for initial lags
            
            for i in range(lag, n-1):
                boot_z[i+1] = boot_z[i] + boot_dz[i-lag]
            
            # Run grid search on bootstrap sample
            boot_z_lag = boot_z[:-1]
            grid_values = np.linspace(np.percentile(boot_z, 15), np.percentile(boot_z, 85), 20)
            
            _, boot_test_stat = self._run_grid_search(
                boot_z,
                lag,
                grid_values,
                min_regime_size=self.config.get('min_regime_size', 0.1),
                use_hac=use_hac,
                hac_maxlags=hac_maxlags
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
        lag: int,
        threshold: float,
        use_hac: bool = True,
        hac_maxlags: Optional[int] = None
    ) -> Dict[str, Any]:
        """Estimate regime-specific adjustment dynamics."""
        n = len(residuals)
        
        # Calculate differenced residuals
        dz = np.diff(residuals)
        z_lag = residuals[:-1]
        
        # Create lagged matrix for differenced residuals
        if lag > 0:
            X_lags = np.zeros((n - lag - 1, lag))
            for i in range(lag):
                X_lags[:, i] = dz[lag-i-1:n-i-2]
        else:
            X_lags = np.zeros((n - 1, 0))
        
        # Create regime indicators
        below_threshold = (z_lag[lag:] <= threshold) if lag > 0 else (z_lag <= threshold)
        above_threshold = ~below_threshold
        
        # Create design matrix
        if lag > 0:
            X = np.column_stack([
                np.ones(n - lag - 1),                  # Constant
                below_threshold * z_lag[lag:],         # z_t-1 * I(z_t-1 <= gamma)
                above_threshold * z_lag[lag:],         # z_t-1 * I(z_t-1 > gamma)
                X_lags                                 # Lagged differences
            ])
            y = dz[lag:]
        else:
            X = np.column_stack([
                np.ones(n - 1),                        # Constant
                below_threshold * z_lag,               # z_t-1 * I(z_t-1 <= gamma)
                above_threshold * z_lag,               # z_t-1 * I(z_t-1 > gamma)
            ])
            y = dz
        
        # Fit model
        if use_hac:
            cov_type = 'HAC'
            cov_kwds = {'maxlags': hac_maxlags or int(n**0.25)}
        else:
            cov_type = 'nonrobust'
            cov_kwds = {}
            
        model = sm.OLS(y, X).fit(cov_type=cov_type, cov_kwds=cov_kwds)
        
        # Extract adjustment parameters
        alpha_down = model.params[1]   # Adjustment when below threshold
        alpha_up = model.params[2]     # Adjustment when above threshold
        
        alpha_down_se = model.bse[1]
        alpha_up_se = model.bse[2]
        
        # Calculate half-lives
        half_life_down = self.calculate_half_life(alpha_down)
        half_life_up = self.calculate_half_life(alpha_up)
        
        # Test asymmetry
        asymmetry_significant, asymmetry_pvalue = self.test_asymmetry(
            alpha_up, alpha_down, alpha_up_se, alpha_down_se
        )
        
        return {
            "adjustment_dynamics": {
                "alpha_down": float(alpha_down),
                "alpha_up": float(alpha_up),
                "alpha_down_se": float(alpha_down_se),
                "alpha_up_se": float(alpha_up_se),
                "half_life_down": float(half_life_down),
                "half_life_up": float(half_life_up),
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