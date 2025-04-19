"""
Spatial econometrics module for Yemen Market Analysis.

This module provides functions for spatial econometric analysis, including
spatial weight matrices, spatial lag models, and spatial error models.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import libpysal.weights as weights
from spreg import ML_Lag, ML_Error, GM_Lag, GM_Error
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)


class SpatialModel:
    """
    Spatial model for Yemen Market Analysis.
    
    This class provides methods for spatial econometric analysis.
    
    Attributes:
        data (gpd.GeoDataFrame): GeoDataFrame containing spatial data.
        w (weights.W): Spatial weight matrix.
        model_type (str): Type of spatial model.
        results (Dict[str, Any]): Model results.
    """
    
    def __init__(
        self, data: Optional[gpd.GeoDataFrame] = None,
        w: Optional[weights.W] = None,
        model_type: str = 'error'
    ):
        """
        Initialize the spatial model.
        
        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
            model_type: Type of spatial model. Options are 'error', 'lag',
                      'durbin', and 'ols'.
        """
        self.data = data
        self.w = w
        self.model_type = model_type
        self.results = {}
    
    @handle_errors
    def set_data(
        self, data: gpd.GeoDataFrame, w: Optional[weights.W] = None
    ) -> None:
        """
        Set the data for the model.
        
        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data is invalid.
        """
        logger.info("Setting data for spatial model")
        
        # Validate data
        validate_data(data, 'spatial')
        
        # Set data
        self.data = data
        
        # Set spatial weight matrix
        if w is not None:
            self.w = w
        
        logger.info(f"Set data with {len(self.data)} observations")
    
    @handle_errors
    def create_weight_matrix(
        self, type: str = 'queen', k: int = 5, bandwidth: Optional[float] = None,
        conflict_column: Optional[str] = None, conflict_weight: float = 0.5
    ) -> weights.W:
        """
        Create a spatial weight matrix.
        
        Args:
            type: Type of spatial weight matrix. Options are 'queen', 'rook',
                'knn', 'distance', 'kernel', and 'conflict'.
            k: Number of nearest neighbors. Only used for 'knn' type.
            bandwidth: Bandwidth for kernel weights. Only used for 'kernel' type.
            conflict_column: Column containing conflict intensity. Only used for
                           'conflict' type.
            conflict_weight: Weight to apply to conflict intensity. Only used for
                           'conflict' type.
            
        Returns:
            Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the weight matrix
                               cannot be created.
        """
        logger.info(f"Creating {type} weight matrix")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        try:
            # Create weight matrix based on type
            if type == 'queen':
                self.w = weights.Queen.from_dataframe(self.data)
            elif type == 'rook':
                self.w = weights.Rook.from_dataframe(self.data)
            elif type == 'knn':
                self.w = weights.KNN.from_dataframe(self.data, k=k)
            elif type == 'distance':
                # Calculate a distance band weight matrix
                min_dist = self.data.geometry.distance(self.data.shift().geometry).min()
                max_dist = self.data.geometry.distance(self.data.shift().geometry).max()
                threshold = min_dist + (max_dist - min_dist) * 0.25  # 25% of the distance range
                self.w = weights.DistanceBand.from_dataframe(self.data, threshold=threshold, alpha=-1.0)
            elif type == 'kernel':
                # Calculate a kernel weight matrix
                if bandwidth is None:
                    # Estimate bandwidth using nearest neighbor distance
                    knn = weights.KNN.from_dataframe(self.data, k=k)
                    bandwidth = knn.neighbors_mean_dist
                self.w = weights.Kernel.from_dataframe(self.data, bandwidth=bandwidth)
            elif type == 'conflict':
                # Create conflict-adjusted weight matrix
                if conflict_column is None:
                    conflict_column = config.get('analysis.spatial.conflict_column', 'conflict_intensity')
                
                # Check if conflict column exists
                if conflict_column not in self.data.columns:
                    logger.error(f"Conflict column {conflict_column} not found in data")
                    raise YemenAnalysisError(f"Conflict column {conflict_column} not found in data")
                
                # Create base weight matrix
                base_w = weights.Queen.from_dataframe(self.data)
                
                # Get conflict intensity
                conflict = self.data[conflict_column]
                
                # Normalize conflict to [0, 1]
                if conflict.min() < 0 or conflict.max() > 1:
                    conflict = (conflict - conflict.min()) / (conflict.max() - conflict.min())
                
                # Create conflict-adjusted weights
                w_data = {}
                for i, neighbors in base_w.neighbors.items():
                    w_data[i] = {}
                    for j in neighbors:
                        # Adjust weight by conflict intensity
                        # Higher conflict reduces the weight (connectivity)
                        w_data[i][j] = 1 - (conflict.iloc[j] * conflict_weight)
                
                # Create weight matrix
                self.w = weights.W(w_data)
                
                # Row-standardize
                self.w.transform = 'r'
            else:
                logger.error(f"Invalid weight matrix type: {type}")
                raise YemenAnalysisError(f"Invalid weight matrix type: {type}")
            
            logger.info(f"Created {type} weight matrix with {len(self.w.neighbors)} units")
            return self.w
        except Exception as e:
            logger.error(f"Error creating weight matrix: {e}")
            raise YemenAnalysisError(f"Error creating weight matrix: {e}")
    
    @handle_errors
    def moran_test(
        self, variable: str, w: Optional[weights.W] = None,
        permutations: int = 999
    ) -> Dict[str, Any]:
        """
        Perform Moran's I test for spatial autocorrelation.
        
        Args:
            variable: Variable to test.
            w: Spatial weight matrix. If None, uses the weight matrix from the class.
            permutations: Number of permutations for the test.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the test fails.
        """
        logger.info(f"Performing Moran's I test for {variable}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Check if weight matrix has been set
        if w is None:
            w = self.w
        
        if w is None:
            logger.error("Weight matrix has not been set")
            raise YemenAnalysisError("Weight matrix has not been set")
        
        # Check if variable exists
        if variable not in self.data.columns:
            logger.error(f"Variable {variable} not found in data")
            raise YemenAnalysisError(f"Variable {variable} not found in data")
        
        try:
            # Perform Moran's I test
            moran = Moran(self.data[variable], w, permutations=permutations)
            
            # Create results dictionary
            results = {
                'test': 'Moran I',
                'variable': variable,
                'I': moran.I,
                'E[I]': moran.EI,
                'p_value': moran.p_sim,
                'z_score': moran.z_sim,
                'is_significant': moran.p_sim < self.alpha,
                'permutations': permutations,
                'alpha': self.alpha,
            }
            
            logger.info(f"Moran's I test results: I={moran.I:.4f}, p_value={moran.p_sim:.4f}")
            return results
        except Exception as e:
            logger.error(f"Error performing Moran's I test: {e}")
            raise YemenAnalysisError(f"Error performing Moran's I test: {e}")
    
    @handle_errors
    def local_moran_test(
        self, variable: str, w: Optional[weights.W] = None,
        permutations: int = 999
    ) -> Dict[str, Any]:
        """
        Perform Local Moran's I test for spatial clustering.
        
        Args:
            variable: Variable to test.
            w: Spatial weight matrix. If None, uses the weight matrix from the class.
            permutations: Number of permutations for the test.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the test fails.
        """
        logger.info(f"Performing Local Moran's I test for {variable}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Check if weight matrix has been set
        if w is None:
            w = self.w
        
        if w is None:
            logger.error("Weight matrix has not been set")
            raise YemenAnalysisError("Weight matrix has not been set")
        
        # Check if variable exists
        if variable not in self.data.columns:
            logger.error(f"Variable {variable} not found in data")
            raise YemenAnalysisError(f"Variable {variable} not found in data")
        
        try:
            # Perform Local Moran's I test
            local_moran = Moran_Local(self.data[variable], w, permutations=permutations)
            
            # Create results dictionary
            results = {
                'test': 'Local Moran I',
                'variable': variable,
                'Is': local_moran.Is,
                'p_values': local_moran.p_sim,
                'significant': local_moran.p_sim < self.alpha,
                'clusters': local_moran.q,
                'permutations': permutations,
                'alpha': self.alpha,
            }
            
            logger.info(f"Local Moran's I test completed, found {sum(results['significant'])} significant clusters")
            return results
        except Exception as e:
            logger.error(f"Error performing Local Moran's I test: {e}")
            raise YemenAnalysisError(f"Error performing Local Moran's I test: {e}")
    
    @handle_errors
    def getis_ord_test(
        self, variable: str, w: Optional[weights.W] = None,
        permutations: int = 999
    ) -> Dict[str, Any]:
        """
        Perform Getis-Ord G* test for hot spot analysis.
        
        Args:
            variable: Variable to test.
            w: Spatial weight matrix. If None, uses the weight matrix from the class.
            permutations: Number of permutations for the test.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the test fails.
        """
        logger.info(f"Performing Getis-Ord G* test for {variable}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Check if weight matrix has been set
        if w is None:
            w = self.w
        
        if w is None:
            logger.error("Weight matrix has not been set")
            raise YemenAnalysisError("Weight matrix has not been set")
        
        # Check if variable exists
        if variable not in self.data.columns:
            logger.error(f"Variable {variable} not found in data")
            raise YemenAnalysisError(f"Variable {variable} not found in data")
        
        try:
            # Perform Getis-Ord G* test
            g_local = G_Local(self.data[variable], w, permutations=permutations)
            
            # Create results dictionary
            results = {
                'test': 'Getis-Ord G*',
                'variable': variable,
                'Gs': g_local.Gs,
                'EGs': g_local.EGs,
                'p_values': g_local.p_sim,
                'z_values': g_local.z_sim,
                'significant': g_local.p_sim < self.alpha,
                'permutations': permutations,
                'alpha': self.alpha,
            }
            
            logger.info(f"Getis-Ord G* test completed, found {sum(results['significant'])} significant hot/cold spots")
            return results
        except Exception as e:
            logger.error(f"Error performing Getis-Ord G* test: {e}")
            raise YemenAnalysisError(f"Error performing Getis-Ord G* test: {e}")
    
    @handle_errors
    def spatial_lag_model(
        self, y_variable: str, x_variables: List[str],
        w: Optional[weights.W] = None, method: str = 'ml'
    ) -> Dict[str, Any]:
        """
        Estimate a spatial lag model.
        
        Args:
            y_variable: Dependent variable.
            x_variables: Independent variables.
            w: Spatial weight matrix. If None, uses the weight matrix from the class.
            method: Estimation method. Options are 'ml' (maximum likelihood) and
                  'gm' (generalized moments).
            
        Returns:
            Dictionary containing the model results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the model cannot be estimated.
        """
        logger.info(f"Estimating spatial lag model for {y_variable}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Check if weight matrix has been set
        if w is None:
            w = self.w
        
        if w is None:
            logger.error("Weight matrix has not been set")
            raise YemenAnalysisError("Weight matrix has not been set")
        
        # Check if variables exist
        if y_variable not in self.data.columns:
            logger.error(f"Variable {y_variable} not found in data")
            raise YemenAnalysisError(f"Variable {y_variable} not found in data")
        
        for x_var in x_variables:
            if x_var not in self.data.columns:
                logger.error(f"Variable {x_var} not found in data")
                raise YemenAnalysisError(f"Variable {x_var} not found in data")
        
        try:
            # Get data for the model
            y = self.data[y_variable]
            X = self.data[x_variables]
            
            # Add constant to X
            X = sm.add_constant(X)
            
            # Estimate spatial lag model
            if method == 'ml':
                model = ML_Lag(y, X, w)
            elif method == 'gm':
                model = GM_Lag(y, X, w)
            else:
                logger.error(f"Invalid estimation method: {method}")
                raise YemenAnalysisError(f"Invalid estimation method: {method}")
            
            # Create results dictionary
            results = {
                'model': 'Spatial Lag',
                'method': method,
                'variables': {
                    'y': y_variable,
                    'x': x_variables,
                },
                'coefficients': model.betas,
                'std_errors': model.std_err,
                'z_values': model.z_stat[:, 0],
                'p_values': model.z_stat[:, 1],
                'rho': model.rho,  # Spatial lag parameter
                'rho_std_err': model.std_err_rho,
                'rho_z_value': model.z_stat_rho[0],
                'rho_p_value': model.z_stat_rho[1],
                'log_likelihood': model.logll,
                'r_squared': model.pr2,  # Pseudo R-squared
                'aic': model.aic,
                'bic': model.schwarz,
                'n_observations': model.n,
                'n_variables': model.k,
            }
            
            # Save results
            self.results = results
            self.model_type = 'lag'
            
            logger.info(f"Spatial lag model results: rho={model.rho:.4f} (p-value={model.z_stat_rho[1]:.4f}), r_squared={model.pr2:.4f}")
            return results
        except Exception as e:
            logger.error(f"Error estimating spatial lag model: {e}")
            raise YemenAnalysisError(f"Error estimating spatial lag model: {e}")
    
    @handle_errors
    def spatial_error_model(
        self, y_variable: str, x_variables: List[str],
        w: Optional[weights.W] = None, method: str = 'ml'
    ) -> Dict[str, Any]:
        """
        Estimate a spatial error model.
        
        Args:
            y_variable: Dependent variable.
            x_variables: Independent variables.
            w: Spatial weight matrix. If None, uses the weight matrix from the class.
            method: Estimation method. Options are 'ml' (maximum likelihood) and
                  'gm' (generalized moments).
            
        Returns:
            Dictionary containing the model results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the model cannot be estimated.
        """
        logger.info(f"Estimating spatial error model for {y_variable}")
        
        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Check if weight matrix has been set
        if w is None:
            w = self.w
        
        if w is None:
            logger.error("Weight matrix has not been set")
            raise YemenAnalysisError("Weight matrix has not been set")
        
        # Check if variables exist
        if y_variable not in self.data.columns:
            logger.error(f"Variable {y_variable} not found in data")
            raise YemenAnalysisError(f"Variable {y_variable} not found in data")
        
        for x_var in x_variables:
            if x_var not in self.data.columns:
                logger.error(f"Variable {x_var} not found in data")
                raise YemenAnalysisError(f"Variable {x_var} not found in data")
        
        try:
            # Get data for the model
            y = self.data[y_variable]
            X = self.data[x_variables]
            
            # Add constant to X
            X = sm.add_constant(X)
            
            # Estimate spatial error model
            if method == 'ml':
                model = ML_Error(y, X, w)
            elif method == 'gm':
                model = GM_Error(y, X, w)
            else:
                logger.error(f"Invalid estimation method: {method}")
                raise YemenAnalysisError(f"Invalid estimation method: {method}")
            
            # Create results dictionary
            results = {
                'model': 'Spatial Error',
                'method': method,
                'variables': {
                    'y': y_variable,
                    'x': x_variables,
                },
                'coefficients': model.betas,
                'std_errors': model.std_err,
                'z_values': model.z_stat[:, 0],
                'p_values': model.z_stat[:, 1],
                'lambda': model.lam,  # Spatial error parameter
                'lambda_std_err': model.std_err_lam,
                'lambda_z_value': model.z_stat_lam[0],
                'lambda_p_value': model.z_stat_lam[1],
                'log_likelihood': model.logll,
                'r_squared': model.pr2,  # Pseudo R-squared
                'aic': model.aic,
                'bic': model.schwarz,
                'n_observations': model.n,
                'n_variables': model.k,
            }
            
            # Save results
            self.results = results
            self.model_type = 'error'
            
            logger.info(f"Spatial error model results: lambda={model.lam:.4f} (p-value={model.z_stat_lam[1]:.4f}), r_squared={model.pr2:.4f}")
            return results
        except Exception as e:
            logger.error(f"Error estimating spatial error model: {e}")
            raise YemenAnalysisError(f"Error estimating spatial error model: {e}")
    
    @handle_errors
    def predict(
        self, data: Optional[gpd.GeoDataFrame] = None,
        w: Optional[weights.W] = None
    ) -> np.ndarray:
        """
        Make predictions using the estimated model.
        
        Args:
            data: GeoDataFrame to make predictions for. If None, uses the data
                 used for estimation.
            w: Spatial weight matrix for new data. If None, uses the weight matrix
               used for estimation.
            
        Returns:
            Array of predicted values.
            
        Raises:
            YemenAnalysisError: If the model has not been estimated or the data is invalid.
        """
        logger.info("Making predictions with spatial model")
        
        # Check if model has been estimated
        if not self.results:
            logger.error("Model has not been estimated")
            raise YemenAnalysisError("Model has not been estimated")
        
        # Use data and weight matrix from estimation if not provided
        if data is None:
            data = self.data
        
        if w is None:
            w = self.w
        
        # Check if data and weight matrix are valid
        if data is None:
            logger.error("Data not provided")
            raise YemenAnalysisError("Data not provided")
        
        if w is None and self.model_type in ['lag', 'error']:
            logger.error("Weight matrix not provided for spatial model")
            raise YemenAnalysisError("Weight matrix not provided for spatial model")
        
        try:
            # Get variables
            y_var = self.results['variables']['y']
            x_vars = self.results['variables']['x']
            
            # Check if variables exist in data
            for x_var in x_vars:
                if x_var not in data.columns:
                    logger.error(f"Variable {x_var} not found in data")
                    raise YemenAnalysisError(f"Variable {x_var} not found in data")
            
            # Get data for the model
            X = data[x_vars]
            
            # Add constant to X
            X = sm.add_constant(X)
            
            # Make predictions based on model type
            if self.model_type == 'lag':
                # Get coefficients and rho
                betas = self.results['coefficients']
                rho = self.results['rho']
                
                # Calculate Wy
                if y_var in data.columns:
                    Wy = weights.lag_spatial(w, data[y_var])
                else:
                    # Initial prediction without spatial lag
                    initial_pred = np.dot(X, betas)
                    Wy = weights.lag_spatial(w, initial_pred)
                
                # Calculate predictions with spatial lag
                predictions = np.dot(X, betas) + rho * Wy
            elif self.model_type == 'error':
                # Get coefficients
                betas = self.results['coefficients']
                
                # Calculate predictions (without spatial error component)
                predictions = np.dot(X, betas)
            else:  # OLS
                # Get coefficients
                betas = self.results['coefficients']
                
                # Calculate predictions
                predictions = np.dot(X, betas)
            
            logger.info(f"Made {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise YemenAnalysisError(f"Error making predictions: {e}")
    
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
            # Extract results
            model_type = self.results['model']
            method = self.results['method']
            y_var = self.results['variables']['y']
            x_vars = self.results['variables']['x']
            coefs = self.results['coefficients']
            std_errs = self.results['std_errors']
            p_values = self.results['p_values']
            r_squared = self.results['r_squared']
            
            # Create summary
            summary = f"{model_type} Model Summary (Method: {method})\n"
            summary += "=" * 60 + "\n\n"
            
            summary += f"Dependent Variable: {y_var}\n"
            summary += f"Independent Variables: {', '.join(x_vars)}\n\n"
            
            summary += f"R-squared: {r_squared:.4f}\n"
            summary += f"AIC: {self.results['aic']:.4f}\n"
            summary += f"BIC: {self.results['bic']:.4f}\n"
            summary += f"Log-Likelihood: {self.results['log_likelihood']:.4f}\n"
            summary += f"Number of Observations: {self.results['n_observations']}\n\n"
            
            # Add spatial parameter
            if self.model_type == 'lag':
                summary += f"Spatial Lag (rho): {self.results['rho']:.4f} "
                summary += f"(p-value: {self.results['rho_p_value']:.4f})\n\n"
            elif self.model_type == 'error':
                summary += f"Spatial Error (lambda): {self.results['lambda']:.4f} "
                summary += f"(p-value: {self.results['lambda_p_value']:.4f})\n\n"
            
            # Add coefficients table
            summary += "Coefficients:\n"
            summary += f"{'Variable':<15} {'Coefficient':<15} {'Std Error':<15} {'p-value':<15}\n"
            summary += "-" * 60 + "\n"
            
            # Add constant
            summary += f"{'Constant':<15} {coefs[0]:<15.4f} {std_errs[0]:<15.4f} {p_values[0]:<15.4f}\n"
            
            # Add other variables
            for i, var in enumerate(x_vars):
                idx = i + 1  # Add 1 to skip constant
                summary += f"{var:<15} {coefs[idx]:<15.4f} {std_errs[idx]:<15.4f} {p_values[idx]:<15.4f}\n"
            
            logger.info("Generated model summary")
            return summary
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            raise YemenAnalysisError(f"Error getting model summary: {e}")


class SpatialTester:
    """
    Spatial tester for Yemen Market Analysis.
    
    This class provides methods for testing spatial autocorrelation and estimating
    spatial models.
    
    Attributes:
        data (gpd.GeoDataFrame): GeoDataFrame containing spatial data.
        w (weights.W): Spatial weight matrix.
        alpha (float): Significance level for hypothesis tests.
        spatial_model (SpatialModel): Spatial model instance.
    """
    
    def __init__(
        self, data: Optional[gpd.GeoDataFrame] = None,
        w: Optional[weights.W] = None,
        alpha: float = 0.05
    ):
        """
        Initialize the spatial tester.
        
        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
            alpha: Significance level for hypothesis tests.
        """
        self.data = data
        self.w = w
        self.alpha = alpha
        self.spatial_model = SpatialModel(data, w)
    
    @handle_errors
    def run_full_analysis(
        self, data: gpd.GeoDataFrame, variable: str,
        w_type: str = 'queen', x_variables: Optional[List[str]] = None,
        lagrange_multiplier_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Run a full spatial analysis.
        
        Args:
            data: GeoDataFrame containing spatial data.
            variable: Variable to analyze.
            w_type: Type of spatial weight matrix.
            x_variables: Independent variables for spatial regression models.
            lagrange_multiplier_tests: Whether to perform Lagrange Multiplier tests.
            
        Returns:
            Dictionary containing the analysis results.
            
        Raises:
            YemenAnalysisError: If the data is invalid or the analysis fails.
        """
        logger.info(f"Running full spatial analysis for {variable}")
        
        # Set data and create weight matrix
        self.data = data
        self.spatial_model.set_data(data)
        self.w = self.spatial_model.create_weight_matrix(type=w_type)
        
        # Set x_variables if not provided
        if x_variables is None:
            # Use all numeric columns except the variable as independent variables
            x_variables = [col for col in data.select_dtypes(include=['number']).columns 
                         if col != variable and 'geometry' not in col]
        
        # Initialize results dictionary
        results = {}
        
        try:
            # Perform Moran's I test
            moran_results = self.spatial_model.moran_test(variable, self.w)
            results['moran'] = moran_results
            
            # Perform Local Moran's I test
            local_moran_results = self.spatial_model.local_moran_test(variable, self.w)
            results['local_moran'] = local_moran_results
            
            # Perform Getis-Ord G* test
            getis_ord_results = self.spatial_model.getis_ord_test(variable, self.w)
            results['getis_ord'] = getis_ord_results
            
            # Perform Lagrange Multiplier tests if requested
            if lagrange_multiplier_tests:
                # Estimate OLS model
                y = data[variable]
                X = data[x_variables]
                X = sm.add_constant(X)
                ols_model = sm.OLS(y, X)
                ols_results = ols_model.fit()
                
                # Get residuals
                residuals = ols_results.resid
                
                # Import diagnostics from spreg
                from spreg import diagnostics
                
                # Perform LM tests
                lm_error = diagnostics.lm_error(residuals, self.w, X)
                lm_lag = diagnostics.lm_lag(residuals, self.w, X, y)
                rlm_error = diagnostics.rlm_error(residuals, self.w, X, y)
                rlm_lag = diagnostics.rlm_lag(residuals, self.w, X)
                
                # Create results dictionary
                lm_results = {
                    'lm_error': {
                        'statistic': lm_error[0],
                        'p_value': lm_error[1],
                        'significant': lm_error[1] < self.alpha,
                    },
                    'lm_lag': {
                        'statistic': lm_lag[0],
                        'p_value': lm_lag[1],
                        'significant': lm_lag[1] < self.alpha,
                    },
                    'rlm_error': {
                        'statistic': rlm_error[0],
                        'p_value': rlm_error[1],
                        'significant': rlm_error[1] < self.alpha,
                    },
                    'rlm_lag': {
                        'statistic': rlm_lag[0],
                        'p_value': rlm_lag[1],
                        'significant': rlm_lag[1] < self.alpha,
                    },
                }
                
                # Determine recommended model
                if (lm_error['significant'] and not lm_lag['significant']) or \
                   (lm_error['significant'] and lm_lag['significant'] and \
                    rlm_error['significant'] and not rlm_lag['significant']):
                    recommended_model = 'error'
                elif (not lm_error['significant'] and lm_lag['significant']) or \
                     (lm_error['significant'] and lm_lag['significant'] and \
                      not rlm_error['significant'] and rlm_lag['significant']):
                    recommended_model = 'lag'
                elif lm_error['significant'] and lm_lag['significant'] and \
                     rlm_error['significant'] and rlm_lag['significant']:
                    recommended_model = 'durbin'
                else:
                    recommended_model = 'ols'
                
                lm_results['recommended_model'] = recommended_model
                results['lm_tests'] = lm_results
                
                # Estimate recommended model
                if recommended_model == 'error':
                    model_results = self.spatial_model.spatial_error_model(variable, x_variables, self.w)
                elif recommended_model == 'lag':
                    model_results = self.spatial_model.spatial_lag_model(variable, x_variables, self.w)
                elif recommended_model == 'durbin':
                    # Spatial Durbin model not implemented, use lag model
                    model_results = self.spatial_model.spatial_lag_model(variable, x_variables, self.w)
                else:
                    # OLS model - convert statsmodels results to dictionary
                    model_results = {
                        'model': 'OLS',
                        'method': 'OLS',
                        'variables': {
                            'y': variable,
                            'x': x_variables,
                        },
                        'coefficients': ols_results.params.values,
                        'std_errors': ols_results.bse.values,
                        't_values': ols_results.tvalues.values,
                        'p_values': ols_results.pvalues.values,
                        'r_squared': ols_results.rsquared,
                        'adj_r_squared': ols_results.rsquared_adj,
                        'aic': ols_results.aic,
                        'bic': ols_results.bic,
                        'log_likelihood': ols_results.llf,
                        'n_observations': ols_results.nobs,
                        'n_variables': ols_results.df_model + 1,  # Add 1 for constant
                    }
                
                results['model'] = model_results
            
            logger.info(f"Full spatial analysis completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error running full spatial analysis: {e}")
            raise YemenAnalysisError(f"Error running full spatial analysis: {e}")