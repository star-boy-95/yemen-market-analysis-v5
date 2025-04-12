"""
Spatial tester module for Yemen Market Analysis.

This module provides the SpatialTester class that integrates all the spatial
analysis components.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal.weights as weights

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data
from src.models.spatial.weights import SpatialWeightMatrix
from src.models.spatial.models import SpatialModel
from src.models.spatial.error_model import SpatialErrorModel
from src.models.spatial.lag_model import SpatialLagModel
from src.models.spatial.conflict import ConflictIntegration

# Initialize logger
logger = logging.getLogger(__name__)

class SpatialTester:
    """
    Spatial tester for Yemen Market Analysis.
    
    This class integrates all the spatial analysis components and provides a unified
    interface for spatial analysis.
    
    Attributes:
        data (gpd.GeoDataFrame): GeoDataFrame containing spatial data.
        w (weights.W): Spatial weight matrix.
        weight_matrix (SpatialWeightMatrix): Spatial weight matrix manager.
        spatial_model (SpatialModel): Spatial model.
        error_model (SpatialErrorModel): Spatial error model.
        lag_model (SpatialLagModel): Spatial lag model.
        conflict_integration (ConflictIntegration): Conflict integration analysis.
        results (Dict[str, Any]): Analysis results.
    """
    
    def __init__(
        self, data: Optional[gpd.GeoDataFrame] = None,
        w: Optional[weights.W] = None
    ):
        """
        Initialize the spatial tester.
        
        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
        """
        self.data = data
        self.w = w
        self.weight_matrix = SpatialWeightMatrix(data)
        self.spatial_model = SpatialModel(data, w)
        self.error_model = SpatialErrorModel(data, w)
        self.lag_model = SpatialLagModel(data, w)
        self.conflict_integration = ConflictIntegration(data, w)
        self.results = {}
    
    @handle_errors
    def set_data(
        self, data: gpd.GeoDataFrame, w: Optional[weights.W] = None
    ) -> None:
        """
        Set the data for the spatial tester.
        
        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data is invalid.
        """
        logger.info("Setting data for spatial tester")
        
        # Validate data
        validate_data(data, 'spatial')
        
        # Set data
        self.data = data
        
        # Set spatial weight matrix
        if w is not None:
            self.w = w
        
        # Update components
        self.weight_matrix = SpatialWeightMatrix(data)
        self.spatial_model = SpatialModel(data, self.w)
        self.error_model = SpatialErrorModel(data, self.w)
        self.lag_model = SpatialLagModel(data, self.w)
        self.conflict_integration = ConflictIntegration(data, self.w)
        
        logger.info(f"Set data with {len(self.data)} observations")
    
    @handle_errors
    def create_weights(
        self, weight_type: str = 'queen', **kwargs
    ) -> weights.W:
        """
        Create a spatial weight matrix.
        
        Args:
            weight_type: Type of spatial weight matrix. Options are 'queen', 'rook',
                        'distance', 'kernel', and 'conflict'.
            **kwargs: Additional arguments for the weight matrix creation.
            
        Returns:
            Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the weight type is invalid.
        """
        logger.info(f"Creating spatial weights with type={weight_type}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        try:
            # Create weight matrix
            if weight_type == 'queen':
                self.w = self.weight_matrix.create_contiguity_weights(queen=True)
            elif weight_type == 'rook':
                self.w = self.weight_matrix.create_contiguity_weights(queen=False)
            elif weight_type == 'distance':
                self.w = self.weight_matrix.create_distance_weights(**kwargs)
            elif weight_type == 'kernel':
                self.w = self.weight_matrix.create_kernel_weights(**kwargs)
            elif weight_type == 'conflict':
                self.w = self.weight_matrix.create_conflict_weights(**kwargs)
            else:
                logger.error(f"Invalid weight type: {weight_type}")
                raise YemenAnalysisError(f"Invalid weight type: {weight_type}")
            
            # Update components
            self.spatial_model.w = self.w
            self.error_model.w = self.w
            self.lag_model.w = self.w
            self.conflict_integration.w = self.w
            
            logger.info(f"Created {weight_type} weights with {len(self.w.neighbors)} units")
            return self.w
        except Exception as e:
            logger.error(f"Error creating weights: {e}")
            raise YemenAnalysisError(f"Error creating weights: {e}")
    
    @handle_errors
    def run_spatial_diagnostics(
        self, y_col: str, x_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Run spatial diagnostics.
        
        Args:
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            
        Returns:
            Dictionary containing the diagnostic results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the diagnostics fail.
        """
        logger.info(f"Running spatial diagnostics with y_col={y_col}, x_cols={x_cols}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Check if spatial weight matrix has been set
        if self.w is None:
            logger.error("Spatial weight matrix has not been set")
            raise YemenAnalysisError("Spatial weight matrix has not been set")
        
        try:
            # Set data for spatial model
            self.spatial_model.set_data(self.data, y_col, x_cols, self.w)
            
            # Estimate OLS model
            ols_results = self.spatial_model.estimate_ols()
            
            # Test for spatial dependence
            test_results = self.spatial_model.test_spatial_dependence()
            
            # Store results
            self.results['diagnostics'] = {
                'ols_results': ols_results,
                'test_results': test_results,
            }
            
            logger.info(f"Ran spatial diagnostics: recommended model={test_results['recommended_model']}")
            return self.results['diagnostics']
        except Exception as e:
            logger.error(f"Error running spatial diagnostics: {e}")
            raise YemenAnalysisError(f"Error running spatial diagnostics: {e}")
    
    @handle_errors
    def estimate_spatial_model(
        self, y_col: str, x_cols: List[str], model_type: Optional[str] = None,
        method: str = 'ml', **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate a spatial model.
        
        Args:
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            model_type: Type of spatial model. Options are 'ols', 'spatial_error',
                       'spatial_lag', and None. If None, uses the recommended model
                       from the diagnostics.
            method: Estimation method. Options depend on the model type.
            **kwargs: Additional arguments for the model estimation.
            
        Returns:
            Dictionary containing the model results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the model cannot be estimated.
        """
        logger.info(f"Estimating spatial model with y_col={y_col}, x_cols={x_cols}, model_type={model_type}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Check if spatial weight matrix has been set
        if self.w is None:
            logger.error("Spatial weight matrix has not been set")
            raise YemenAnalysisError("Spatial weight matrix has not been set")
        
        try:
            # Determine model type
            if model_type is None:
                # Run diagnostics if not already run
                if 'diagnostics' not in self.results:
                    self.run_spatial_diagnostics(y_col, x_cols)
                
                model_type = self.results['diagnostics']['test_results']['recommended_model']
                logger.info(f"Using recommended model type: {model_type}")
            
            # Set data for the appropriate model
            if model_type == 'ols':
                self.spatial_model.set_data(self.data, y_col, x_cols, self.w)
                model_results = self.spatial_model.estimate_ols()
            elif model_type == 'spatial_error':
                self.error_model.set_data(self.data, y_col, x_cols, self.w)
                model_results = self.error_model.estimate(method=method, **kwargs)
            elif model_type == 'spatial_lag':
                self.lag_model.set_data(self.data, y_col, x_cols, self.w)
                model_results = self.lag_model.estimate(method=method, **kwargs)
            else:
                logger.error(f"Invalid model type: {model_type}")
                raise YemenAnalysisError(f"Invalid model type: {model_type}")
            
            # Store results
            self.results['model'] = {
                'type': model_type,
                'results': model_results,
            }
            
            logger.info(f"Estimated {model_type} model with R-squared={model_results['r_squared']:.4f}")
            return self.results['model']
        except Exception as e:
            logger.error(f"Error estimating spatial model: {e}")
            raise YemenAnalysisError(f"Error estimating spatial model: {e}")
    
    @handle_errors
    def analyze_conflict_integration(
        self, conflict_column: str = 'conflict_intensity',
        price_column: str = 'price', **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze the impact of conflict on market integration.
        
        Args:
            conflict_column: Column containing conflict intensity.
            price_column: Column containing prices.
            **kwargs: Additional arguments for the analysis.
            
        Returns:
            Dictionary containing the analysis results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the analysis fails.
        """
        logger.info(f"Analyzing conflict integration with conflict_column={conflict_column}, price_column={price_column}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        try:
            # Set data for conflict integration
            self.conflict_integration.set_data(
                self.data, self.w, conflict_column, price_column
            )
            
            # Create conflict weights if not already created
            if self.w is None or kwargs.get('create_weights', False):
                self.w = self.conflict_integration.create_conflict_weights(**kwargs)
                
                # Update components
                self.spatial_model.w = self.w
                self.error_model.w = self.w
                self.lag_model.w = self.w
            
            # Analyze price dispersion
            price_dispersion_results = self.conflict_integration.analyze_price_dispersion()
            
            # Analyze market integration
            market_integration_results = self.conflict_integration.analyze_market_integration(**kwargs)
            
            # Analyze price transmission
            if 'reference_market' in kwargs:
                price_transmission_results = self.conflict_integration.analyze_price_transmission(**kwargs)
            else:
                price_transmission_results = None
            
            # Store results
            self.results['conflict_integration'] = {
                'price_dispersion': price_dispersion_results,
                'market_integration': market_integration_results,
                'price_transmission': price_transmission_results,
            }
            
            logger.info("Analyzed conflict integration")
            return self.results['conflict_integration']
        except Exception as e:
            logger.error(f"Error analyzing conflict integration: {e}")
            raise YemenAnalysisError(f"Error analyzing conflict integration: {e}")
    
    @handle_errors
    def get_summary(self) -> str:
        """
        Get a summary of the analysis results.
        
        Returns:
            String containing the analysis summary.
            
        Raises:
            YemenAnalysisError: If no analyses have been performed.
        """
        logger.info("Getting analysis summary")
        
        # Check if any analyses have been performed
        if not self.results:
            logger.error("No analyses have been performed")
            raise YemenAnalysisError("No analyses have been performed")
        
        try:
            # Create summary
            summary = "Spatial Analysis Summary\n"
            summary += "======================\n\n"
            
            # Add diagnostics results
            if 'diagnostics' in self.results:
                summary += "Spatial Diagnostics\n"
                summary += "------------------\n"
                summary += f"OLS R-squared: {self.results['diagnostics']['ols_results']['r_squared']:.4f}\n"
                
                test_results = self.results['diagnostics']['test_results']
                summary += f"Moran's I: {test_results['moran_i']['statistic']:.4f} (p-value: {test_results['moran_i']['p_value']:.4f})\n"
                summary += f"LM Error: {test_results['lm_error']['statistic']:.4f} (p-value: {test_results['lm_error']['p_value']:.4f})\n"
                summary += f"LM Lag: {test_results['lm_lag']['statistic']:.4f} (p-value: {test_results['lm_lag']['p_value']:.4f})\n"
                summary += f"Robust LM Error: {test_results['rlm_error']['statistic']:.4f} (p-value: {test_results['rlm_error']['p_value']:.4f})\n"
                summary += f"Robust LM Lag: {test_results['rlm_lag']['statistic']:.4f} (p-value: {test_results['rlm_lag']['p_value']:.4f})\n"
                summary += f"Recommended model: {test_results['recommended_model']}\n\n"
            
            # Add model results
            if 'model' in self.results:
                summary += "Spatial Model\n"
                summary += "------------\n"
                summary += f"Model type: {self.results['model']['type']}\n"
                summary += f"R-squared: {self.results['model']['results']['r_squared']:.4f}\n"
                
                if self.results['model']['type'] == 'spatial_error':
                    summary += f"Lambda: {self.results['model']['results']['lambda']:.4f} (p-value: {self.results['model']['results']['lambda_p_value']:.4f})\n"
                elif self.results['model']['type'] == 'spatial_lag':
                    summary += f"Rho: {self.results['model']['results']['rho']:.4f} (p-value: {self.results['model']['results']['rho_p_value']:.4f})\n"
                
                summary += "\nCoefficients:\n"
                for var, coef in self.results['model']['results']['coefficients'].items():
                    p_value = self.results['model']['results']['p_values'][var]
                    summary += f"{var}: {coef:.4f} (p-value: {p_value:.4f})\n"
                
                summary += "\n"
            
            # Add conflict integration results
            if 'conflict_integration' in self.results:
                summary += "Conflict Integration Analysis\n"
                summary += "----------------------------\n"
                
                if self.results['conflict_integration']['price_dispersion'] is not None:
                    pd_results = self.results['conflict_integration']['price_dispersion']
                    summary += "Price Dispersion:\n"
                    summary += f"Effect of conflict: {pd_results['coefficients']['conflict_intensity']:.4f} (p-value: {pd_results['p_values']['conflict_intensity']:.4f})\n"
                    summary += f"R-squared: {pd_results['r_squared']:.4f}\n\n"
                
                if self.results['conflict_integration']['market_integration'] is not None:
                    mi_results = self.results['conflict_integration']['market_integration']
                    summary += "Market Integration:\n"
                    summary += f"Method: {mi_results['method']}\n"
                    summary += f"Effect of conflict: {mi_results['coefficients']['conflict_intensity']:.4f} (p-value: {mi_results['p_values']['conflict_intensity']:.4f})\n"
                    summary += f"R-squared: {mi_results['r_squared']:.4f}\n\n"
                
                if self.results['conflict_integration']['price_transmission'] is not None:
                    pt_results = self.results['conflict_integration']['price_transmission']
                    summary += "Price Transmission:\n"
                    
                    for market, results in pt_results.items():
                        summary += f"Market: {market}\n"
                        summary += f"Price transmission: {results['coefficients']['reference_price']:.4f} (p-value: {results['p_values']['reference_price']:.4f})\n"
                        summary += f"Effect of conflict: {results['coefficients']['interaction']:.4f} (p-value: {results['p_values']['interaction']:.4f})\n"
                        summary += f"R-squared: {results['r_squared']:.4f}\n\n"
            
            logger.info("Generated analysis summary")
            return summary
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            raise YemenAnalysisError(f"Error getting analysis summary: {e}")
