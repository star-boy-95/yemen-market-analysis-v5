"""
Numerical stability utilities for Yemen Market Analysis.
"""
import logging
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any

from core.decorators import error_handler
from core.exceptions import ComputationError

logger = logging.getLogger(__name__)


@error_handler(fallback_value=None)
def stabilize_matrix(X: np.ndarray, ridge_alpha: float = 1e-8) -> np.ndarray:
    """
    Apply regularization to improve matrix stability for inversion.
    
    Args:
        X: Input matrix
        ridge_alpha: Ridge regularization parameter
        
    Returns:
        Stabilized matrix
    """
    if X.shape[0] < X.shape[1]:
        # Underdetermined system
        logger.warning(f"Matrix is underdetermined: {X.shape[0]} < {X.shape[1]}")
    
    # Check condition number
    try:
        svd_values = np.linalg.svd(X, compute_uv=False)
        cond = svd_values[0] / svd_values[-1]
        
        if cond > 1e10:
            logger.warning(f"Matrix is ill-conditioned: condition number = {cond:.1e}")
            
            # Add regularization
            if len(X.shape) == 2:
                X_stable = X.copy()
                X_stable = X_stable + ridge_alpha * np.eye(X_stable.shape[0], X_stable.shape[1])
                return X_stable
            else:
                logger.error("Cannot stabilize non-2D matrix")
                return X
        else:
            return X
    except Exception as e:
        logger.error(f"Error in matrix stabilization: {str(e)}")
        return X


@error_handler(fallback_value=None)
def robust_inversion(X: np.ndarray, ridge_alpha: float = 1e-8) -> np.ndarray:
    """
    Compute robust matrix inverse with fallbacks for stability.
    
    Args:
        X: Input matrix to invert
        ridge_alpha: Ridge regularization parameter
        
    Returns:
        Inverted matrix
    """
    if X.shape[0] != X.shape[1]:
        logger.warning(f"Non-square matrix: {X.shape}")
        try:
            # Use pseudo-inverse for non-square matrices
            return np.linalg.pinv(X)
        except Exception as e:
            logger.error(f"Pseudo-inverse failed: {str(e)}")
            raise ComputationError(f"Matrix inversion failed: {str(e)}")
    
    try:
        # Try standard matrix inversion
        return np.linalg.inv(X)
    except np.linalg.LinAlgError:
        # If it fails, try SVD-based pseudo-inverse
        logger.warning("Standard inversion failed, using SVD-based approach")
        try:
            # Add small ridge term
            X_stable = X + ridge_alpha * np.eye(X.shape[0])
            return np.linalg.inv(X_stable)
        except Exception as e:
            logger.error(f"SVD-based inversion failed: {str(e)}")
            
            # Last resort: pseudo-inverse
            return np.linalg.pinv(X)


@error_handler(fallback_value=(None, None))
def robust_ols(
    X: np.ndarray, 
    y: np.ndarray, 
    ridge_alpha: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute OLS with robust fallbacks for stability.
    
    Args:
        X: Design matrix
        y: Dependent variable
        ridge_alpha: Ridge regularization parameter
        
    Returns:
        Tuple of (coefficients, residuals)
    """
    # Check dimensions
    if len(X.shape) != 2:
        raise ValueError(f"Design matrix must be 2D, got shape {X.shape}")
    
    if len(y.shape) not in (1, 2):
        raise ValueError(f"Dependent variable must be 1D or 2D, got shape {y.shape}")
    
    # Reshape y if needed
    y_flat = y.reshape(-1) if len(y.shape) == 2 and y.shape[1] == 1 else y
    
    try:
        # Try SVD for better numerical stability
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        
        # Add small ridge term for stability
        s_inv = 1.0 / (s + ridge_alpha)
        
        # Calculate coefficients
        beta = vh.T @ (s_inv.reshape(-1, 1) * (u.T @ y_flat.reshape(-1, 1)))
        beta = beta.flatten()
        
        # Calculate residuals
        residuals = y_flat - X @ beta
        
        return beta, residuals
    except Exception as e:
        logger.warning(f"SVD-based OLS failed: {str(e)}")
        
        # Fall back to normal equations with regularization
        try:
            # Calculate X'X with regularization
            XtX = X.T @ X + ridge_alpha * np.eye(X.shape[1])
            
            # Calculate X'y
            Xty = X.T @ y_flat
            
            # Solve for coefficients
            beta = robust_inversion(XtX) @ Xty
            
            # Calculate residuals
            residuals = y_flat - X @ beta
            
            return beta, residuals
        except Exception as e:
            logger.error(f"Normal equations OLS failed: {str(e)}")
            raise ComputationError(f"OLS estimation failed: {str(e)}")


@error_handler(fallback_value=None)
def robust_correlation(
    x: np.ndarray, 
    y: np.ndarray, 
    method: str = 'pearson'
) -> float:
    """
    Calculate correlation coefficient with error handling.
    
    Args:
        x: First variable
        y: Second variable
        method: 'pearson', 'spearman', or 'kendall'
        
    Returns:
        Correlation coefficient
    """
    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        logger.warning("Insufficient non-missing observations for correlation")
        return 0.0
    
    # Check for constant values
    if np.std(x_clean) == 0 or np.std(y_clean) == 0:
        logger.warning("Constant values detected, correlation undefined")
        return 0.0
    
    try:
        if method == 'pearson':
            return np.corrcoef(x_clean, y_clean)[0, 1]
        elif method == 'spearman':
            from scipy.stats import spearmanr
            return spearmanr(x_clean, y_clean)[0]
        elif method == 'kendall':
            from scipy.stats import kendalltau
            return kendalltau(x_clean, y_clean)[0]
        else:
            logger.error(f"Unknown correlation method: {method}")
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating correlation: {str(e)}")
        return 0.0


@error_handler
def scale_matrix(
    X: np.ndarray,
    method: str = 'standardize'
) -> np.ndarray:
    """
    Scale matrix for numerical stability.
    
    Args:
        X: Input matrix
        method: 'standardize', 'normalize', or 'robust'
        
    Returns:
        Scaled matrix
    """
    if method == 'standardize':
        # Z-score normalization
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        return (X - mean) / std
    
    elif method == 'normalize':
        # Min-max scaling
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # Avoid division by zero
        return (X - min_vals) / range_vals
    
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1.0  # Avoid division by zero
        return (X - median) / iqr
    
    else:
        logger.error(f"Unknown scaling method: {method}")
        return X


class NumericalStability:
    """Class with utility methods for numerical stability."""
    
    @staticmethod
    def has_collinearity(X: np.ndarray, threshold: float = 0.9) -> bool:
        """Check for multicollinearity in design matrix."""
        if X.shape[1] < 2:
            return False
            
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X, rowvar=False)
            
            # Get absolute values of off-diagonal elements
            off_diag = np.abs(corr_matrix - np.eye(corr_matrix.shape[0]))
            
            # Check if any are above threshold
            return np.any(off_diag > threshold)
        except Exception as e:
            logger.error(f"Error checking collinearity: {str(e)}")
            return False
    
    @staticmethod
    def vif_scores(X: np.ndarray) -> List[float]:
        """Calculate Variance Inflation Factors for multicollinearity detection."""
        n_features = X.shape[1]
        vif_scores = []
        
        for i in range(n_features):
            # Separate target feature and use others as predictors
            X_i = X[:, i]
            X_others = np.delete(X, i, axis=1)
            
            try:
                # Add constant
                X_others = np.column_stack([np.ones(X_others.shape[0]), X_others])
                
                # Fit linear model
                beta, _ = robust_ols(X_others, X_i)
                
                # Calculate R-squared
                y_pred = X_others @ beta
                y_mean = np.mean(X_i)
                tss = np.sum((X_i - y_mean) ** 2)
                rss = np.sum((X_i - y_pred) ** 2)
                r_squared = 1 - (rss / tss)
                
                # Calculate VIF
                vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                vif_scores.append(vif)
            except Exception as e:
                logger.error(f"Error calculating VIF for feature {i}: {str(e)}")
                vif_scores.append(float('nan'))
        
        return vif_scores