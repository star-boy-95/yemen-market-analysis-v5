"""
Market integration simulation module for Yemen Market Analysis.

This module provides classes and functions for simulating market integration scenarios,
including exchange rate unification and improved spatial connectivity.
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
from src.models.threshold import ThresholdModel
from src.models.threshold.tvecm import ThresholdVECM
from src.models.spatial.weights import SpatialWeightMatrix
from src.models.spatial.lag_model import SpatialLagModel

# Initialize logger
logger = logging.getLogger(__name__)

class MarketIntegrationSimulation:
    """
    Market integration simulation for Yemen Market Analysis.

    This class provides methods for simulating market integration scenarios,
    including exchange rate unification and improved spatial connectivity.

    Attributes:
        data (pd.DataFrame): DataFrame containing market data.
        exchange_rate_data (pd.DataFrame): DataFrame containing exchange rate data.
        threshold_model (ThresholdModel): Threshold model for simulating price adjustments.
        spatial_model (SpatialLagModel): Spatial model for simulating spatial connectivity.
        results (Dict[str, Any]): Simulation results.
    """

    def __init__(
        self, data: pd.DataFrame,
        exchange_rate_data: Optional[pd.DataFrame] = None,
        threshold_model: Optional[ThresholdModel] = None,
        spatial_model: Optional[SpatialLagModel] = None
    ):
        """
        Initialize the market integration simulation.

        Args:
            data: DataFrame containing market data.
            exchange_rate_data: DataFrame containing exchange rate data.
            threshold_model: Threshold model for simulating price adjustments.
            spatial_model: Spatial model for simulating spatial connectivity.
        """
        self.data = data
        self.exchange_rate_data = exchange_rate_data
        self.threshold_model = threshold_model
        self.spatial_model = spatial_model
        self.results = {}

    @handle_errors
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set the data for the simulation.

        Args:
            data: DataFrame containing market data.

        Raises:
            YemenAnalysisError: If the data is invalid.
        """
        logger.info("Setting data for market integration simulation")

        # Validate data
        validate_data(data, 'time_series')

        # Set data
        self.data = data

        logger.info(f"Set data with {len(self.data)} observations")

    @handle_errors
    def set_exchange_rate_data(self, exchange_rate_data: pd.DataFrame) -> None:
        """
        Set the exchange rate data for the simulation.

        Args:
            exchange_rate_data: DataFrame containing exchange rate data.

        Raises:
            YemenAnalysisError: If the data is invalid.
        """
        logger.info("Setting exchange rate data for market integration simulation")

        # Validate data
        validate_data(exchange_rate_data, 'time_series')

        # Set data
        self.exchange_rate_data = exchange_rate_data

        logger.info(f"Set exchange rate data with {len(self.exchange_rate_data)} observations")

    @handle_errors
    def simulate_exchange_rate_unification(
        self, target_rate: str = 'official',
        method: str = 'tvecm',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate exchange rate unification.

        This method simulates the impact of exchange rate unification on market integration
        by re-estimating threshold models with a unified exchange rate.

        Args:
            target_rate: Target exchange rate for unification. Options are 'official',
                        'parallel', and 'average'.
            method: Method for simulating exchange rate unification. Options are 'tvecm'
                  (Threshold VECM) and 'tar' (Threshold Autoregressive).
            **kwargs: Additional arguments for the threshold model estimation.

        Returns:
            Dictionary containing the simulation results.

        Raises:
            YemenAnalysisError: If the simulation fails.
        """
        logger.info(f"Simulating exchange rate unification with target_rate={target_rate}, method={method}")

        try:
            # Check if exchange rate data is available
            if self.exchange_rate_data is None:
                logger.error("Exchange rate data is not available")
                raise YemenAnalysisError("Exchange rate data is not available")

            # Create a copy of the exchange rate data for simulation
            sim_exchange_rate_data = self.exchange_rate_data.copy()

            # Get the exchange rates for Sana'a and Aden
            sanaa_rate_col = config.get('data.exchange_rate.sanaa_column', 'sanaa_rate')
            aden_rate_col = config.get('data.exchange_rate.aden_column', 'aden_rate')

            if sanaa_rate_col not in sim_exchange_rate_data.columns:
                logger.error(f"Sana'a exchange rate column {sanaa_rate_col} not found in data")
                raise YemenAnalysisError(f"Sana'a exchange rate column {sanaa_rate_col} not found in data")

            if aden_rate_col not in sim_exchange_rate_data.columns:
                logger.error(f"Aden exchange rate column {aden_rate_col} not found in data")
                raise YemenAnalysisError(f"Aden exchange rate column {aden_rate_col} not found in data")

            # Calculate the unified exchange rate based on the target rate
            if target_rate == 'official':
                # Use the official exchange rate (Aden)
                unified_rate = sim_exchange_rate_data[aden_rate_col]
            elif target_rate == 'parallel':
                # Use the parallel exchange rate (Sana'a)
                unified_rate = sim_exchange_rate_data[sanaa_rate_col]
            elif target_rate == 'average':
                # Use the average of the two rates
                unified_rate = (sim_exchange_rate_data[sanaa_rate_col] + sim_exchange_rate_data[aden_rate_col]) / 2
            else:
                logger.error(f"Invalid target rate: {target_rate}")
                raise YemenAnalysisError(f"Invalid target rate: {target_rate}")

            # Set both exchange rates to the unified rate
            sim_exchange_rate_data[sanaa_rate_col] = unified_rate
            sim_exchange_rate_data[aden_rate_col] = unified_rate

            # Calculate the exchange rate differential (should be zero)
            sim_exchange_rate_data['exchange_rate_diff'] = sim_exchange_rate_data[sanaa_rate_col] - sim_exchange_rate_data[aden_rate_col]

            # Merge the simulated exchange rate data with the market data
            date_col = config.get('data.date_column', 'date')
            sim_data = self.data.merge(sim_exchange_rate_data, on=date_col, how='left')

            # Run the threshold model with the simulated data
            if method == 'tvecm':
                # Use Threshold VECM
                if self.threshold_model is None or not isinstance(self.threshold_model, ThresholdVECM):
                    logger.info("Creating new ThresholdVECM model for simulation")
                    self.threshold_model = ThresholdVECM(alpha=kwargs.get('alpha', 0.05), max_lags=kwargs.get('max_lags', 4))

                # Get the dependent and independent variables
                y_col = kwargs.get('y_col', config.get('analysis.threshold.y_col', 'price_y'))
                x_col = kwargs.get('x_col', config.get('analysis.threshold.x_col', 'price_x'))

                # Estimate the threshold model with the simulated data
                sim_results = self.threshold_model.estimate(
                    sim_data, sim_data, y_col, x_col,
                    k_ar_diff=kwargs.get('k_ar_diff', 2),
                    deterministic=kwargs.get('deterministic', 'ci'),
                    coint_rank=kwargs.get('coint_rank', 1),
                    trim=kwargs.get('trim', 0.15)
                )

                # Store the simulation results
                self.results['exchange_rate_unification'] = {
                    'target_rate': target_rate,
                    'method': method,
                    'original_data': self.data,
                    'simulated_data': sim_data,
                    'original_exchange_rate_data': self.exchange_rate_data,
                    'simulated_exchange_rate_data': sim_exchange_rate_data,
                    'threshold_model_results': sim_results,
                }

                # Calculate the impact of exchange rate unification
                if 'original_threshold' in kwargs and 'original_adjustment_speed' in kwargs:
                    original_threshold = kwargs['original_threshold']
                    original_adjustment_speed = kwargs['original_adjustment_speed']
                    
                    # Calculate the percentage change in threshold parameter
                    threshold_change = (sim_results['threshold'] - original_threshold) / original_threshold * 100
                    
                    # Calculate the percentage change in adjustment speed
                    adjustment_speed_change = (sim_results['adjustment_speed'] - original_adjustment_speed) / original_adjustment_speed * 100
                    
                    # Store the impact results
                    self.results['exchange_rate_unification']['impact'] = {
                        'threshold_change_pct': threshold_change,
                        'adjustment_speed_change_pct': adjustment_speed_change,
                    }

                logger.info(f"Simulated exchange rate unification with threshold={sim_results['threshold']:.4f}")
                return self.results['exchange_rate_unification']
            
            elif method == 'tar':
                # Use Threshold Autoregressive model
                if self.threshold_model is None:
                    logger.info("Creating new ThresholdModel for simulation")
                    self.threshold_model = ThresholdModel(
                        None, None, None, None, 
                        mode=kwargs.get('mode', 'standard'),
                        alpha=kwargs.get('alpha', 0.05),
                        max_lags=kwargs.get('max_lags', 4)
                    )

                # Get the dependent and independent variables
                y_col = kwargs.get('y_col', config.get('analysis.threshold.y_col', 'price_y'))
                x_col = kwargs.get('x_col', config.get('analysis.threshold.x_col', 'price_x'))

                # Set the data for the threshold model
                self.threshold_model.y = sim_data
                self.threshold_model.x = sim_data
                self.threshold_model.y_col = y_col
                self.threshold_model.x_col = x_col

                # Run the threshold model with the simulated data
                sim_results = self.threshold_model.run_full_analysis()

                # Store the simulation results
                self.results['exchange_rate_unification'] = {
                    'target_rate': target_rate,
                    'method': method,
                    'original_data': self.data,
                    'simulated_data': sim_data,
                    'original_exchange_rate_data': self.exchange_rate_data,
                    'simulated_exchange_rate_data': sim_exchange_rate_data,
                    'threshold_model_results': sim_results,
                }

                # Calculate the impact of exchange rate unification
                if 'original_results' in kwargs:
                    original_results = kwargs['original_results']
                    
                    # Calculate the percentage change in threshold parameter
                    if 'threshold' in original_results and 'threshold' in sim_results:
                        threshold_change = (sim_results['threshold'] - original_results['threshold']) / original_results['threshold'] * 100
                        self.results['exchange_rate_unification']['impact'] = {
                            'threshold_change_pct': threshold_change,
                        }
                    
                    # Calculate the percentage change in adjustment speeds
                    if 'rho1' in original_results and 'rho1' in sim_results:
                        rho1_change = (sim_results['rho1'] - original_results['rho1']) / original_results['rho1'] * 100
                        self.results['exchange_rate_unification']['impact']['rho1_change_pct'] = rho1_change
                    
                    if 'rho2' in original_results and 'rho2' in sim_results:
                        rho2_change = (sim_results['rho2'] - original_results['rho2']) / original_results['rho2'] * 100
                        self.results['exchange_rate_unification']['impact']['rho2_change_pct'] = rho2_change

                logger.info(f"Simulated exchange rate unification with method={method}")
                return self.results['exchange_rate_unification']
            
            else:
                logger.error(f"Invalid method: {method}")
                raise YemenAnalysisError(f"Invalid method: {method}")
        
        except Exception as e:
            logger.error(f"Error simulating exchange rate unification: {e}")
            raise YemenAnalysisError(f"Error simulating exchange rate unification: {e}")

    @handle_errors
    def simulate_spatial_connectivity(
        self, data: gpd.GeoDataFrame,
        connectivity_improvement: float = 0.5,
        weight_type: str = 'distance',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simulate improved spatial connectivity.

        This method simulates the impact of improved spatial connectivity on market integration
        by modifying the spatial weight matrix to reflect reduced effective distances between markets.

        Args:
            data: GeoDataFrame containing spatial data.
            connectivity_improvement: Percentage improvement in connectivity (0-1).
            weight_type: Type of spatial weight matrix. Options are 'distance', 'contiguity',
                        and 'kernel'.
            **kwargs: Additional arguments for the spatial model estimation.

        Returns:
            Dictionary containing the simulation results.

        Raises:
            YemenAnalysisError: If the simulation fails.
        """
        logger.info(f"Simulating spatial connectivity with improvement={connectivity_improvement}, weight_type={weight_type}")

        try:
            # Validate data
            validate_data(data, 'spatial')

            # Create a copy of the data for simulation
            sim_data = data.copy()

            # Create the original spatial weight matrix
            weight_matrix = SpatialWeightMatrix(data)
            if weight_type == 'distance':
                original_w = weight_matrix.create_distance_weights(**kwargs)
            elif weight_type == 'contiguity':
                original_w = weight_matrix.create_contiguity_weights(**kwargs)
            elif weight_type == 'kernel':
                original_w = weight_matrix.create_kernel_weights(**kwargs)
            else:
                logger.error(f"Invalid weight type: {weight_type}")
                raise YemenAnalysisError(f"Invalid weight type: {weight_type}")

            # Create the simulated spatial weight matrix with improved connectivity
            sim_weight_matrix = SpatialWeightMatrix(sim_data)
            
            if weight_type == 'distance':
                # For distance weights, reduce the effective distance by the connectivity improvement
                # This is done by modifying the distance matrix before creating the weights
                
                # Get the distance matrix
                from libpysal.weights.util import get_points_array
                points = get_points_array(sim_data.geometry)
                from scipy.spatial.distance import cdist
                dist_matrix = cdist(points, points)
                
                # Reduce the distances by the connectivity improvement factor
                sim_dist_matrix = dist_matrix * (1 - connectivity_improvement)
                
                # Create a custom distance-based weight matrix
                from libpysal.weights import WSP
                from libpysal.weights.util import fill_diagonal
                
                # Convert to binary weights based on a threshold
                threshold = kwargs.get('threshold', None)
                if threshold is None:
                    # Use k-nearest neighbors
                    k = kwargs.get('k', 4)
                    sim_w = weights.KNN.from_array(points, k=k)
                else:
                    # Use distance band with the simulated distances
                    binary = np.zeros(sim_dist_matrix.shape)
                    binary[sim_dist_matrix <= threshold] = 1
                    binary[sim_dist_matrix > threshold] = 0
                    fill_diagonal(binary, 0)  # No self-connections
                    
                    # Create the weights object
                    wsp = WSP(binary)
                    sim_w = wsp.to_W()
            
            elif weight_type == 'contiguity':
                # For contiguity weights, we can't directly simulate improved connectivity
                # Instead, we'll create a hybrid approach that adds connections based on distance
                
                # Create the original contiguity weights
                sim_w = sim_weight_matrix.create_contiguity_weights(**kwargs)
                
                # Add additional connections based on distance
                # Get the distance matrix
                from libpysal.weights.util import get_points_array
                points = get_points_array(sim_data.geometry)
                from scipy.spatial.distance import cdist
                dist_matrix = cdist(points, points)
                
                # Determine a distance threshold based on the connectivity improvement
                # The smaller the threshold, the more connections will be added
                max_dist = dist_matrix.max()
                threshold = max_dist * (1 - connectivity_improvement)
                
                # Add connections for markets within the threshold distance
                for i in range(len(sim_data)):
                    for j in range(len(sim_data)):
                        if i != j and dist_matrix[i, j] <= threshold and j not in sim_w.neighbors[i]:
                            # Add the connection
                            sim_w.neighbors[i].append(j)
                            sim_w.weights[i].append(1.0)  # Equal weight for simplicity
                
                # Normalize the weights
                sim_w.transform = 'r'  # Row-standardize
            
            elif weight_type == 'kernel':
                # For kernel weights, adjust the bandwidth to reflect improved connectivity
                bandwidth = kwargs.get('bandwidth', None)
                if bandwidth is None:
                    # Use adaptive bandwidth based on k nearest neighbors
                    k = kwargs.get('k', 4)
                    # Increase k to reflect improved connectivity
                    sim_k = int(k * (1 + connectivity_improvement))
                    sim_w = sim_weight_matrix.create_kernel_weights(k=sim_k, **kwargs)
                else:
                    # Increase the bandwidth to reflect improved connectivity
                    sim_bandwidth = bandwidth * (1 + connectivity_improvement)
                    sim_w = sim_weight_matrix.create_kernel_weights(bandwidth=sim_bandwidth, **kwargs)

            # Create or get the spatial lag model
            if self.spatial_model is None:
                logger.info("Creating new SpatialLagModel for simulation")
                self.spatial_model = SpatialLagModel(sim_data, sim_w)
            else:
                # Update the data and weights
                self.spatial_model.set_data(sim_data, kwargs.get('y_col', 'price'), kwargs.get('x_cols', ['distance', 'conflict']), sim_w)

            # Estimate the spatial model with the simulated weights
            sim_results = self.spatial_model.estimate(
                method=kwargs.get('method', 'ml'),
                vm=kwargs.get('vm', False),
                name_y=kwargs.get('name_y', None),
                name_x=kwargs.get('name_x', None),
                name_w=kwargs.get('name_w', None),
                name_ds=kwargs.get('name_ds', None)
            )

            # Store the simulation results
            self.results['spatial_connectivity'] = {
                'connectivity_improvement': connectivity_improvement,
                'weight_type': weight_type,
                'original_data': data,
                'simulated_data': sim_data,
                'original_weights': original_w,
                'simulated_weights': sim_w,
                'spatial_model_results': sim_results,
            }

            # Calculate the impact of improved connectivity
            if 'original_results' in kwargs:
                original_results = kwargs['original_results']
                
                # Calculate the percentage change in spatial dependence
                if 'rho' in original_results and 'rho' in sim_results:
                    rho_change = (sim_results['rho'] - original_results['rho']) / original_results['rho'] * 100
                    self.results['spatial_connectivity']['impact'] = {
                        'rho_change_pct': rho_change,
                    }
                
                # Calculate the percentage change in price dispersion
                if 'price_dispersion' in original_results and 'price_dispersion' in sim_results:
                    dispersion_change = (sim_results['price_dispersion'] - original_results['price_dispersion']) / original_results['price_dispersion'] * 100
                    self.results['spatial_connectivity']['impact']['price_dispersion_change_pct'] = dispersion_change

            logger.info(f"Simulated spatial connectivity with improvement={connectivity_improvement}")
            return self.results['spatial_connectivity']
        
        except Exception as e:
            logger.error(f"Error simulating spatial connectivity: {e}")
            raise YemenAnalysisError(f"Error simulating spatial connectivity: {e}")

    @handle_errors
    def run_full_simulation(
        self, data: pd.DataFrame,
        exchange_rate_data: pd.DataFrame,
        spatial_data: gpd.GeoDataFrame,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a full market integration simulation.

        This method runs both exchange rate unification and spatial connectivity simulations.

        Args:
            data: DataFrame containing market data.
            exchange_rate_data: DataFrame containing exchange rate data.
            spatial_data: GeoDataFrame containing spatial data.
            **kwargs: Additional arguments for the simulations.

        Returns:
            Dictionary containing all simulation results.

        Raises:
            YemenAnalysisError: If the simulation fails.
        """
        logger.info("Running full market integration simulation")

        try:
            # Set data
            self.set_data(data)
            self.set_exchange_rate_data(exchange_rate_data)

            # Run exchange rate unification simulation
            exchange_rate_results = self.simulate_exchange_rate_unification(
                target_rate=kwargs.get('target_rate', 'official'),
                method=kwargs.get('method', 'tvecm'),
                **kwargs
            )

            # Run spatial connectivity simulation
            spatial_results = self.simulate_spatial_connectivity(
                spatial_data,
                connectivity_improvement=kwargs.get('connectivity_improvement', 0.5),
                weight_type=kwargs.get('weight_type', 'distance'),
                **kwargs
            )

            # Combine results
            self.results['full_simulation'] = {
                'exchange_rate_unification': exchange_rate_results,
                'spatial_connectivity': spatial_results,
            }

            logger.info("Completed full market integration simulation")
            return self.results['full_simulation']
        
        except Exception as e:
            logger.error(f"Error running full market integration simulation: {e}")
            raise YemenAnalysisError(f"Error running full market integration simulation: {e}")
