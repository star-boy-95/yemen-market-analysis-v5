"""
Base model classes and interfaces for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple

from core.exceptions import ModelError
from core.decorators import error_handler
from .schemas import ModelConfig

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model with optional configuration."""
        self.config = config or {}
        self.is_fitted = False
        self.results = {}
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Dict[str, Any]:
        """Fit the model to data."""
        pass
    
    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input data."""
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """Get model results."""
        if not self.is_fitted:
            raise ModelError("Model has not been fitted yet")
        return self.results
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update model configuration."""
        self.config.update(config)


class ThresholdModel(BaseModel):
    """Base class for threshold models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize threshold model."""
        super().__init__(config)
        self.threshold = None
        self.residuals = None
        self.test_statistic = None
        self.p_value = None
        self.threshold_significant = False
    
    @abstractmethod
    def estimate_threshold(self, *args, **kwargs) -> float:
        """Estimate the threshold parameter."""
        pass
    
    @abstractmethod
    def test_threshold_significance(self, *args, **kwargs) -> Tuple[float, bool]:
        """Test significance of threshold effect."""
        pass
    
    @abstractmethod
    def estimate_regime_dynamics(self, *args, **kwargs) -> Dict[str, Any]:
        """Estimate regime-specific dynamics."""
        pass
    
    @error_handler(fallback_value=False)
    def validate_inputs(self, y: np.ndarray, x: np.ndarray) -> bool:
        """Validate input data for threshold models."""
        # Check for None or empty arrays
        if y is None or x is None or len(y) == 0 or len(x) == 0:
            logger.error("Empty or None input arrays")
            return False
        
        # Check matching lengths
        if len(y) != len(x):
            logger.error(f"Input arrays have different lengths: {len(y)} vs {len(x)}")
            return False
        
        # Check for NaN or infinite values
        if np.isnan(y).any() or np.isnan(x).any():
            logger.error("Input arrays contain NaN values")
            return False
        
        if np.isinf(y).any() or np.isinf(x).any():
            logger.error("Input arrays contain infinite values")
            return False
        
        # Check for sufficient observations
        min_obs = self.config.get('min_observations', 20)
        if len(y) < min_obs:
            logger.error(f"Insufficient observations: {len(y)} < {min_obs}")
            return False
        
        # Check for constant values
        if np.std(y) == 0 or np.std(x) == 0:
            logger.error("Input arrays contain constant values")
            return False
        
        return True
    
    def calculate_regime_balance(self, values: np.ndarray, threshold: float) -> Dict[str, float]:
        """Calculate balance between regimes."""
        if values is None or len(values) == 0:
            return {"lower": 0, "upper": 0}
        
        lower_regime = (values <= threshold).mean()
        upper_regime = 1 - lower_regime
        
        return {
            "lower": float(lower_regime),
            "upper": float(upper_regime)
        }
    
    def check_regime_balance(
        self, 
        values: np.ndarray, 
        threshold: float, 
        min_regime_size: float = 0.1
    ) -> bool:
        """Check if regimes are sufficiently balanced."""
        regime_balance = self.calculate_regime_balance(values, threshold)
        
        if regime_balance["lower"] < min_regime_size or regime_balance["upper"] < min_regime_size:
            return False
        
        return True
    
    def calculate_half_life(self, alpha: float) -> float:
        """Calculate half-life of adjustment."""
        if alpha is None or alpha >= 0:
            return float('inf')
        
        return np.log(0.5) / np.log(1 + alpha)
    
    def test_asymmetry(
        self, 
        alpha_up: float, 
        alpha_down: float, 
        alpha_up_se: float, 
        alpha_down_se: float
    ) -> Tuple[bool, float]:
        """Test for asymmetric adjustment."""
        # Calculate absolute difference
        diff = abs(alpha_up - alpha_down)
        
        # Calculate standard error of difference
        diff_se = np.sqrt(alpha_up_se**2 + alpha_down_se**2)
        
        # Calculate t-statistic
        t_stat = diff / diff_se if diff_se > 0 else 0
        
        # Calculate p-value (two-sided test)
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), np.inf))
        
        # Test significance
        significance = self.config.get('significance_level', 0.05)
        asymmetry_significant = p_value < significance
        
        return asymmetry_significant, p_value


def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> ThresholdModel:
    """Factory function to create threshold model instances."""
    from .threshold.hansen_seo import HansenSeoModel
    from .threshold.enders_siklos import EndersSiklosModel
    
    if model_type == 'hansen_seo':
        return HansenSeoModel(config)
    elif model_type == 'enders_siklos':
        return EndersSiklosModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")