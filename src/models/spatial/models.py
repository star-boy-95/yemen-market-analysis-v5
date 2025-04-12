"""
Base spatial models module for Yemen Market Analysis.

This module provides the base SpatialModel class for spatial econometric analysis.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import libpysal.weights as weights

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class SpatialModel:
    """
    Base spatial model for Yemen Market Analysis.
    
    This class provides the base functionality for spatial econometric models.
    
    Attributes:
        data (gpd.GeoDataFrame): GeoDataFrame containing spatial data.
        w (weights.W): Spatial weight matrix.
        y (pd.Series): Dependent variable.
        X (pd.DataFrame): Independent variables.
        model_type (str): Type of spatial model.
        results (Dict[str, Any]): Model results.
    """
    
    def __init__(
        self, data: Optional[gpd.GeoDataFrame] = None,
        w: Optional[weights.W] = None
    ):
        """
        Initialize the spatial model.
        
        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
        """
        self.data = data
        self.w = w
        self.y = None
        self.X = None
        self.model_type = 'base'
        self.results = {}
    
    @handle_errors
    def set_data(
        self, data: gpd.GeoDataFrame, y_col: str,
        x_cols: List[str], w: Optional[weights.W] = None
    ) -> None:
        """
        Set the data for the model.
        
        Args:
            data: GeoDataFrame containing spatial data.
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            w: Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data is invalid.
        """
        logger.info(f"Setting data with y_col={y_col}, x_cols={x_cols}")
        
        # Validate data
        validate_data(data, 'spatial')
        
        # Check if columns exist
        if y_col not in data.columns:
            logger.error(f"Dependent variable column {y_col} not found in data")
            raise YemenAnalysisError(f"Dependent variable column {y_col} not found in data")
        
        for col in x_cols:
            if col not in data.columns:
                logger.error(f"Independent variable column {col} not found in data")
                raise YemenAnalysisError(f"Independent variable column {col} not found in data")
        
        # Set data
        self.data = data
        self.y = data[y_col]
        self.X = data[x_cols]
        
        # Add constant to X if not already present
        if 'const' not in self.X.columns:
            self.X = sm.add_constant(self.X)
        
        # Set spatial weight matrix
        if w is not None:
            self.w = w
        
        logger.info(f"Set data with {len(self.data)} observations, {len(x_cols)} independent variables")
    
    @handle_errors
    def estimate_ols(self) -> Dict[str, Any]:
        """
        Estimate an OLS model.
        
        Returns:
            Dictionary containing the model results.
            
        Raises:
            YemenAnalysisError: If the data has not been set.
        """
        logger.info("Estimating OLS model")
        
        # Check if data has been set
        if self.y is None or self.X is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        try:
            # Estimate OLS model
            model = OLS(self.y, self.X)
            results = model.fit()
            
            # Store results
            self.results = {
                'model_type': 'OLS',
                'coefficients': results.params.to_dict(),
                'std_errors': results.bse.to_dict(),
                'p_values': results.pvalues.to_dict(),
                't_values': results.tvalues.to_dict(),
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'aic': results.aic,
                'bic': results.bic,
                'f_statistic': results.fvalue,
                'f_p_value': results.f_pvalue,
                'residuals': results.resid.to_dict(),
                'fitted_values': results.fittedvalues.to_dict(),
                'n_obs': results.nobs,
                'df_model': results.df_model,
                'df_resid': results.df_resid,
            }
            
            logger.info(f"Estimated OLS model with R-squared={results.rsquared:.4f}")
            return self.results
        except Exception as e:
            logger.error(f"Error estimating OLS model: {e}")
            raise YemenAnalysisError(f"Error estimating OLS model: {e}")
    
    @handle_errors
    def test_spatial_dependence(self) -> Dict[str, Any]:
        """
        Test for spatial dependence in the OLS residuals.
        
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the OLS model has not been estimated or the
                               spatial weight matrix has not been set.
        """
        logger.info("Testing for spatial dependence")
        
        # Check if OLS model has been estimated
        if not self.results or self.results.get('model_type') != 'OLS':
            logger.error("OLS model has not been estimated")
            raise YemenAnalysisError("OLS model has not been estimated")
        
        # Check if spatial weight matrix has been set
        if self.w is None:
            logger.error("Spatial weight matrix has not been set")
            raise YemenAnalysisError("Spatial weight matrix has not been set")
        
        try:
            # Import diagnostic tests from PySAL
            from spreg import diagnostics
            
            # Get OLS residuals
            residuals = pd.Series(self.results['residuals'])
            
            # Calculate Moran's I for residuals
            moran_result = diagnostics.morans(residuals, self.w)
            
            # Calculate Lagrange Multiplier tests
            lm_error = diagnostics.lm_error(residuals, self.w, self.X)
            lm_lag = diagnostics.lm_lag(residuals, self.w, self.X, self.y)
            rlm_error = diagnostics.rlm_error(residuals, self.w, self.X, self.y)
            rlm_lag = diagnostics.rlm_lag(residuals, self.w, self.X)
            
            # Store test results
            test_results = {
                'moran_i': {
                    'statistic': moran_result[0],
                    'p_value': moran_result[1],
                    'is_significant': moran_result[1] < 0.05,
                },
                'lm_error': {
                    'statistic': lm_error[0],
                    'p_value': lm_error[1],
                    'is_significant': lm_error[1] < 0.05,
                },
                'lm_lag': {
                    'statistic': lm_lag[0],
                    'p_value': lm_lag[1],
                    'is_significant': lm_lag[1] < 0.05,
                },
                'rlm_error': {
                    'statistic': rlm_error[0],
                    'p_value': rlm_error[1],
                    'is_significant': rlm_error[1] < 0.05,
                },
                'rlm_lag': {
                    'statistic': rlm_lag[0],
                    'p_value': rlm_lag[1],
                    'is_significant': rlm_lag[1] < 0.05,
                },
            }
            
            # Determine recommended model
            if lm_error[1] < 0.05 and lm_lag[1] < 0.05:
                if rlm_error[1] < 0.05 and rlm_lag[1] >= 0.05:
                    recommended_model = 'spatial_error'
                elif rlm_error[1] >= 0.05 and rlm_lag[1] < 0.05:
                    recommended_model = 'spatial_lag'
                elif rlm_error[1] < 0.05 and rlm_lag[1] < 0.05:
                    recommended_model = 'spatial_durbin'
                else:
                    recommended_model = 'ols'
            elif lm_error[1] < 0.05:
                recommended_model = 'spatial_error'
            elif lm_lag[1] < 0.05:
                recommended_model = 'spatial_lag'
            else:
                recommended_model = 'ols'
            
            test_results['recommended_model'] = recommended_model
            
            logger.info(f"Spatial dependence tests: Moran's I p-value={moran_result[1]:.4f}, recommended model={recommended_model}")
            return test_results
        except Exception as e:
            logger.error(f"Error testing for spatial dependence: {e}")
            raise YemenAnalysisError(f"Error testing for spatial dependence: {e}")
    
    @handle_errors
    def get_summary(self) -> str:
        """
        Get a summary of the model results.
        
        Returns:
            String containing the model summary.
            
        Raises:
            YemenAnalysisError: If the model has not been estimated.
        """
        logger.info("Getting model summary")
        
        # Check if model has been estimated
        if not self.results:
            logger.error("Model has not been estimated")
            raise YemenAnalysisError("Model has not been estimated")
        
        try:
            # Create summary
            summary = f"Model: {self.results['model_type']}\n"
            summary += f"Number of observations: {self.results['n_obs']}\n"
            summary += f"R-squared: {self.results['r_squared']:.4f}\n"
            summary += f"Adjusted R-squared: {self.results['adj_r_squared']:.4f}\n"
            summary += f"AIC: {self.results['aic']:.4f}\n"
            summary += f"BIC: {self.results['bic']:.4f}\n"
            
            if 'f_statistic' in self.results:
                summary += f"F-statistic: {self.results['f_statistic']:.4f} (p-value: {self.results['f_p_value']:.4f})\n"
            
            summary += "\nCoefficients:\n"
            for var, coef in self.results['coefficients'].items():
                std_err = self.results['std_errors'][var]
                p_value = self.results['p_values'][var]
                t_value = self.results['t_values'][var]
                
                summary += f"{var}: {coef:.4f} (std err: {std_err:.4f}, t-value: {t_value:.4f}, p-value: {p_value:.4f})\n"
            
            logger.info("Generated model summary")
            return summary
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            raise YemenAnalysisError(f"Error getting model summary: {e}")
