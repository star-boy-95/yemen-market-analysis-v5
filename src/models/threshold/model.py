"""
Main threshold model module for Yemen Market Analysis.

This module provides the main ThresholdModel class that integrates
all the individual threshold model implementations.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.models.threshold.tar import ThresholdAutoregressive
from src.models.threshold.mtar import MomentumThresholdAutoregressive
from src.models.threshold.tvecm import ThresholdVECM

# Initialize logger
logger = logging.getLogger(__name__)

class ThresholdModel:
    """
    Main threshold model for Yemen Market Analysis.
    
    This class integrates all the individual threshold model implementations and provides
    a unified interface for estimating threshold models.
    
    Attributes:
        y (pd.DataFrame): DataFrame containing the dependent variable.
        x (pd.DataFrame): DataFrame containing the independent variable.
        y_col (str): Column name for the dependent variable.
        x_col (str): Column name for the independent variable.
        mode (str): Mode for the threshold model. Options are 'standard' (TAR),
                   'fixed' (TAR with fixed threshold), 'mtar' (M-TAR), and 'tvecm' (TVECM).
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        tar (ThresholdAutoregressive): TAR model implementation.
        mtar (MomentumThresholdAutoregressive): M-TAR model implementation.
        tvecm (ThresholdVECM): TVECM model implementation.
    """
    
    def __init__(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        mode: str = 'standard', alpha: float = None, max_lags: int = None
    ):
        """
        Initialize the threshold model.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            mode: Mode for the threshold model. Options are 'standard' (TAR),
                 'fixed' (TAR with fixed threshold), 'mtar' (M-TAR), and 'tvecm' (TVECM).
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.y = y
        self.x = x
        self.y_col = y_col
        self.x_col = x_col
        self.mode = mode
        self.alpha = alpha if alpha is not None else config.get('analysis.threshold.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.threshold.max_lags', 4)
        
        # Initialize individual models
        self.tar = ThresholdAutoregressive(alpha=self.alpha, max_lags=self.max_lags)
        self.mtar = MomentumThresholdAutoregressive(alpha=self.alpha, max_lags=self.max_lags)
        self.tvecm = ThresholdVECM(alpha=self.alpha, max_lags=self.max_lags)
    
    @handle_errors
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run a full threshold model analysis.
        
        This method runs the appropriate threshold model based on the mode specified
        during initialization.
        
        Returns:
            Dictionary containing the analysis results.
            
        Raises:
            YemenAnalysisError: If the analysis fails.
        """
        logger.info(f"Running full threshold model analysis with mode={self.mode}")
        
        try:
            if self.mode == 'standard':
                # Run standard TAR model
                results = self.tar.estimate(self.y, self.x, self.y_col, self.x_col)
            elif self.mode == 'fixed':
                # Run TAR model with fixed threshold (0)
                threshold = config.get('analysis.threshold.mtar_default_threshold', 0.0)
                results = self.tar.estimate(self.y, self.x, self.y_col, self.x_col, fixed_threshold=threshold)
            elif self.mode == 'mtar':
                # Run M-TAR model
                results = self.mtar.estimate(self.y, self.x, self.y_col, self.x_col)
            elif self.mode == 'tvecm':
                # Run TVECM model
                k_ar_diff = config.get('analysis.threshold_vecm.k_ar_diff', 2)
                deterministic = config.get('analysis.threshold_vecm.deterministic', 'ci')
                coint_rank = config.get('analysis.threshold_vecm.coint_rank', 1)
                
                results = self.tvecm.estimate(
                    self.y, self.x, self.y_col, self.x_col,
                    k_ar_diff=k_ar_diff, deterministic=deterministic, coint_rank=coint_rank
                )
            else:
                logger.error(f"Invalid threshold model mode: {self.mode}")
                raise YemenAnalysisError(f"Invalid threshold model mode: {self.mode}")
            
            logger.info(f"Threshold model analysis completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error running threshold model analysis: {e}")
            raise YemenAnalysisError(f"Error running threshold model analysis: {e}")
    
    @handle_errors
    def run(self) -> Dict[str, Any]:
        """
        Run the threshold model analysis.
        
        This is an alias for run_full_analysis() to maintain a consistent interface.
        
        Returns:
            Dictionary containing the analysis results.
            
        Raises:
            YemenAnalysisError: If the analysis fails.
        """
        return self.run_full_analysis()
    
    @handle_errors
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare different threshold model specifications.
        
        This method estimates all available threshold models and compares them
        based on information criteria.
        
        Returns:
            Dictionary containing the comparison results.
            
        Raises:
            YemenAnalysisError: If the comparison fails.
        """
        logger.info("Comparing different threshold model specifications")
        
        try:
            # Estimate all models
            tar_results = self.tar.estimate(self.y, self.x, self.y_col, self.x_col)
            
            threshold = config.get('analysis.threshold.mtar_default_threshold', 0.0)
            tar_fixed_results = self.tar.estimate(self.y, self.x, self.y_col, self.x_col, fixed_threshold=threshold)
            
            mtar_results = self.mtar.estimate(self.y, self.x, self.y_col, self.x_col)
            
            k_ar_diff = config.get('analysis.threshold_vecm.k_ar_diff', 2)
            deterministic = config.get('analysis.threshold_vecm.deterministic', 'ci')
            coint_rank = config.get('analysis.threshold_vecm.coint_rank', 1)
            
            tvecm_results = self.tvecm.estimate(
                self.y, self.x, self.y_col, self.x_col,
                k_ar_diff=k_ar_diff, deterministic=deterministic, coint_rank=coint_rank
            )
            
            # Extract information criteria
            models = {
                'TAR': tar_results,
                'TAR (fixed)': tar_fixed_results,
                'M-TAR': mtar_results,
                'TVECM': tvecm_results,
            }
            
            aic_values = {name: results['aic'] for name, results in models.items()}
            bic_values = {name: results['bic'] for name, results in models.items()}
            
            # Find best model
            best_aic = min(aic_values, key=aic_values.get)
            best_bic = min(bic_values, key=bic_values.get)
            
            # Create comparison results
            comparison_results = {
                'models': models,
                'aic_values': aic_values,
                'bic_values': bic_values,
                'best_aic': best_aic,
                'best_bic': best_bic,
                'recommendation': best_bic,  # Use BIC as default recommendation
            }
            
            logger.info(f"Model comparison completed. Best model (AIC): {best_aic}, Best model (BIC): {best_bic}")
            return comparison_results
        except Exception as e:
            logger.error(f"Error comparing threshold models: {e}")
            raise YemenAnalysisError(f"Error comparing threshold models: {e}")
