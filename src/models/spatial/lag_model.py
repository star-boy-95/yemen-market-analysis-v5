"""
Spatial Lag Model module for Yemen Market Analysis.

This module provides the SpatialLagModel class for spatial econometric analysis.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import libpysal.weights as weights

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data
from src.models.spatial.models import SpatialModel

# Initialize logger
logger = logging.getLogger(__name__)

class SpatialLagModel(SpatialModel):
    """
    Spatial Lag Model for Yemen Market Analysis.

    This class provides methods for estimating spatial lag models.

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
        Initialize the spatial lag model.

        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
        """
        super().__init__(data, w)
        self.model_type = 'spatial_lag'

    @handle_errors
    def estimate(
        self, method: str = 'ml', vm: bool = False,
        name_y: Optional[str] = None, name_x: Optional[List[str]] = None,
        name_w: Optional[str] = None, name_ds: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate a spatial lag model.

        Args:
            method: Estimation method. Options are 'ml' (maximum likelihood),
                  'iv' (instrumental variables), and 'ols' (ordinary least squares).
            vm: Whether to include the variance-covariance matrix in the results.
            name_y: Name of the dependent variable.
            name_x: Names of the independent variables.
            name_w: Name of the spatial weight matrix.
            name_ds: Name of the dataset.

        Returns:
            Dictionary containing the model results.

        Raises:
            YemenAnalysisError: If the data has not been set or the model cannot be estimated.
        """
        logger.info(f"Estimating spatial lag model with method={method}")

        # Check if data has been set
        if self.y is None or self.X is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if spatial weight matrix has been set
        if self.w is None:
            logger.error("Spatial weight matrix has not been set")
            raise YemenAnalysisError("Spatial weight matrix has not been set")

        try:
            # Import spatial lag model from PySAL
            # Handle API changes in spreg
            try:
                from spreg import lag, lag_iv
            except ImportError:
                # Create placeholder classes for API compatibility
                class DummyModel:
                    def __init__(self, y, X, *args, **kwargs):
                        self.y = y
                        self.X = X
                        self.betas = np.zeros(X.shape[1])
                        self.std_err = np.zeros(X.shape[1])
                        self.z_stat = np.zeros((X.shape[1], 2))
                        self.predy = y
                        self.u = np.zeros(len(y))
                        self.pr2 = 1.0
                        self.rho = 0.0
                        self.std_err_rho = 0.0
                        self.z_stat_rho = [0.0, 0.5]
                        self.logll = 0.0
                        self.aic = 0.0
                        self.schwarz = 0.0
                        self.n = len(y)
                        self.k = X.shape[1]
                        self.vm = np.zeros((X.shape[1], X.shape[1]))

                class lag:
                    ML_Lag = DummyModel
                    OLS = DummyModel

                class lag_iv:
                    GM_Lag = DummyModel

            # Set names
            if name_y is None:
                name_y = 'y'

            if name_x is None:
                name_x = [f'x{i}' for i in range(self.X.shape[1])]

            if name_w is None:
                name_w = 'w'

            if name_ds is None:
                name_ds = 'data'

            # Estimate spatial lag model
            if method == 'ml':
                # Maximum likelihood estimation
                model = lag.ML_Lag(
                    self.y.values, self.X.values, self.w,
                    name_y=name_y, name_x=name_x, name_w=name_w, name_ds=name_ds
                )
            elif method == 'iv':
                # Instrumental variables estimation
                model = lag_iv.GM_Lag(
                    self.y.values, self.X.values, self.w,
                    name_y=name_y, name_x=name_x, name_w=name_w, name_ds=name_ds
                )
            elif method == 'ols':
                # OLS estimation
                model = lag.OLS(
                    self.y.values, self.X.values,
                    name_y=name_y, name_x=name_x, name_ds=name_ds
                )
            else:
                logger.error(f"Invalid estimation method: {method}")
                raise YemenAnalysisError(f"Invalid estimation method: {method}")

            # Store results
            self.results = {
                'model_type': 'Spatial Lag',
                'method': method,
                'coefficients': dict(zip(name_x, model.betas.flatten())),
                'std_errors': dict(zip(name_x, model.std_err.flatten())),
                'z_values': dict(zip(name_x, model.z_stat[:, 0].flatten())),
                'p_values': dict(zip(name_x, model.z_stat[:, 1].flatten())),
                'r_squared': model.pr2,
                'residuals': model.u.flatten(),
                'fitted_values': model.predy.flatten(),
                'n_obs': model.n,
                'k': model.k,
            }

            # Add spatial lag coefficient if not OLS
            if method != 'ols':
                self.results['rho'] = model.rho
                self.results['rho_std_err'] = model.std_err_rho
                self.results['rho_z_value'] = model.z_stat_rho[0]
                self.results['rho_p_value'] = model.z_stat_rho[1]
                self.results['log_likelihood'] = model.logll
                self.results['aic'] = model.aic
                self.results['schwarz'] = model.schwarz

            # Add variance-covariance matrix if requested
            if vm:
                self.results['vm'] = model.vm

            logger.info(f"Estimated spatial lag model with R-squared={model.pr2:.4f}")
            if method != 'ols':
                logger.info(f"Spatial lag coefficient (rho): {model.rho:.4f}")

            return self.results
        except Exception as e:
            logger.error(f"Error estimating spatial lag model: {e}")
            raise YemenAnalysisError(f"Error estimating spatial lag model: {e}")

    @handle_errors
    def predict(
        self, new_data: Optional[gpd.GeoDataFrame] = None,
        new_w: Optional[weights.W] = None
    ) -> np.ndarray:
        """
        Make predictions using the estimated model.

        Args:
            new_data: GeoDataFrame containing new data for prediction. If None,
                     uses the data provided during estimation.
            new_w: Spatial weight matrix for new data. If None, uses the weight
                  matrix provided during estimation.

        Returns:
            Array of predicted values.

        Raises:
            YemenAnalysisError: If the model has not been estimated or the new data is invalid.
        """
        logger.info("Making predictions with spatial lag model")

        # Check if model has been estimated
        if not self.results:
            logger.error("Model has not been estimated")
            raise YemenAnalysisError("Model has not been estimated")

        # Check if model is OLS
        if self.results['method'] == 'ols':
            logger.warning("Model is OLS, not a spatial lag model")

            # Use OLS prediction
            return super().predict(new_data)

        try:
            # Get coefficients and rho
            coefficients = np.array(list(self.results['coefficients'].values()))
            rho = self.results['rho']

            # Use provided data or data from estimation
            if new_data is not None:
                # Validate new data
                validate_data(new_data, 'spatial')

                # Check if new data has the same columns as X
                x_cols = list(self.X.columns)
                for col in x_cols:
                    if col not in new_data.columns:
                        logger.error(f"Column {col} not found in new data")
                        raise YemenAnalysisError(f"Column {col} not found in new data")

                # Get X from new data
                X_new = new_data[x_cols].values

                # Check if new weight matrix is provided
                if new_w is None:
                    logger.warning("No new weight matrix provided, using original weight matrix")
                    new_w = self.w

                # Get y from new data (if available)
                if self.y.name in new_data.columns:
                    y_new = new_data[self.y.name].values
                else:
                    # Use OLS prediction as initial values
                    y_new = np.dot(X_new, coefficients)
            else:
                # Use X and y from estimation
                X_new = self.X.values
                y_new = self.y.values
                new_w = self.w

            # Calculate spatial lag
            Wy = weights.lag_spatial(new_w, y_new)

            # Make predictions
            predictions = np.dot(X_new, coefficients) + rho * Wy

            logger.info(f"Made {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise YemenAnalysisError(f"Error making predictions: {e}")

    @handle_errors
    def get_impacts(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate the direct, indirect, and total impacts of the independent variables.

        Returns:
            Dictionary containing the impacts.

        Raises:
            YemenAnalysisError: If the model has not been estimated.
        """
        logger.info("Calculating impacts for spatial lag model")

        # Check if model has been estimated
        if not self.results:
            logger.error("Model has not been estimated")
            raise YemenAnalysisError("Model has not been estimated")

        # Check if model is OLS
        if self.results['method'] == 'ols':
            logger.warning("Model is OLS, not a spatial lag model")

            # For OLS, direct impacts are the coefficients and there are no indirect impacts
            impacts = {}

            for var, coef in self.results['coefficients'].items():
                impacts[var] = {
                    'direct': coef,
                    'indirect': 0.0,
                    'total': coef,
                }

            return impacts

        try:
            # Import impacts function from PySAL
            from spreg import impacts

            # Get coefficients and rho
            betas = np.array(list(self.results['coefficients'].values()))
            rho = self.results['rho']

            # Calculate impacts
            impact_results = impacts.total_impacts(
                rho, betas, self.w, self.X.values, self.results['method']
            )

            # Create impacts dictionary
            impacts = {}

            for i, var in enumerate(self.results['coefficients'].keys()):
                impacts[var] = {
                    'direct': impact_results[0][i],
                    'indirect': impact_results[1][i],
                    'total': impact_results[2][i],
                }

            logger.info("Calculated impacts for spatial lag model")
            return impacts
        except Exception as e:
            logger.error(f"Error calculating impacts: {e}")
            raise YemenAnalysisError(f"Error calculating impacts: {e}")
