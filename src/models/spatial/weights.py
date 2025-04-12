"""
Spatial weight matrix module for Yemen Market Analysis.

This module provides the SpatialWeightMatrix class for creating and manipulating
spatial weight matrices for spatial econometric analysis.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy import sparse
import libpysal.weights as weights

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class SpatialWeightMatrix:
    """
    Spatial weight matrix for Yemen Market Analysis.
    
    This class provides methods for creating and manipulating spatial weight matrices
    for spatial econometric analysis.
    
    Attributes:
        data (gpd.GeoDataFrame): GeoDataFrame containing spatial data.
        w (weights.W): Spatial weight matrix.
        type (str): Type of spatial weight matrix.
        params (Dict[str, Any]): Parameters used to create the weight matrix.
    """
    
    def __init__(self, data: gpd.GeoDataFrame = None):
        """
        Initialize the spatial weight matrix.
        
        Args:
            data: GeoDataFrame containing spatial data.
        """
        self.data = data
        self.w = None
        self.type = None
        self.params = {}
    
    @handle_errors
    def create_contiguity_weights(
        self, data: Optional[gpd.GeoDataFrame] = None, queen: bool = True
    ) -> weights.W:
        """
        Create contiguity-based spatial weights.
        
        Args:
            data: GeoDataFrame containing spatial data. If None, uses the data
                 provided during initialization.
            queen: Whether to use queen contiguity (True) or rook contiguity (False).
            
        Returns:
            Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data is invalid or the weights cannot be created.
        """
        logger.info(f"Creating {'queen' if queen else 'rook'} contiguity weights")
        
        # Use provided data or data from initialization
        if data is not None:
            self.data = data
        
        # Validate data
        if self.data is None:
            logger.error("No data provided")
            raise YemenAnalysisError("No data provided")
        
        validate_data(self.data, 'spatial')
        
        try:
            # Create contiguity weights
            if queen:
                self.w = weights.Queen.from_dataframe(self.data)
            else:
                self.w = weights.Rook.from_dataframe(self.data)
            
            # Set type and parameters
            self.type = 'queen' if queen else 'rook'
            self.params = {'queen': queen}
            
            logger.info(f"Created {self.type} contiguity weights with {len(self.w.neighbors)} units")
            return self.w
        except Exception as e:
            logger.error(f"Error creating contiguity weights: {e}")
            raise YemenAnalysisError(f"Error creating contiguity weights: {e}")
    
    @handle_errors
    def create_distance_weights(
        self, data: Optional[gpd.GeoDataFrame] = None, k: int = 4,
        p: float = 2.0, alpha: float = -1.0, threshold: Optional[float] = None
    ) -> weights.W:
        """
        Create distance-based spatial weights.
        
        Args:
            data: GeoDataFrame containing spatial data. If None, uses the data
                 provided during initialization.
            k: Number of nearest neighbors.
            p: Power of the Minkowski distance metric (1: Manhattan, 2: Euclidean).
            alpha: Distance decay parameter.
            threshold: Distance threshold for cut-off.
            
        Returns:
            Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data is invalid or the weights cannot be created.
        """
        logger.info(f"Creating distance-based weights with k={k}, p={p}, alpha={alpha}")
        
        # Use provided data or data from initialization
        if data is not None:
            self.data = data
        
        # Validate data
        if self.data is None:
            logger.error("No data provided")
            raise YemenAnalysisError("No data provided")
        
        validate_data(self.data, 'spatial')
        
        try:
            # Create distance weights
            if threshold is not None:
                # Distance band weights
                self.w = weights.DistanceBand.from_dataframe(
                    self.data, threshold=threshold, alpha=alpha, binary=False
                )
                self.type = 'distance_band'
                self.params = {'threshold': threshold, 'alpha': alpha}
            else:
                # K-nearest neighbors weights
                self.w = weights.KNN.from_dataframe(self.data, k=k, p=p)
                self.type = 'knn'
                self.params = {'k': k, 'p': p}
            
            logger.info(f"Created {self.type} distance weights with {len(self.w.neighbors)} units")
            return self.w
        except Exception as e:
            logger.error(f"Error creating distance weights: {e}")
            raise YemenAnalysisError(f"Error creating distance weights: {e}")
    
    @handle_errors
    def create_kernel_weights(
        self, data: Optional[gpd.GeoDataFrame] = None, bandwidth: Optional[float] = None,
        kernel: str = 'gaussian', fixed: bool = True, k: int = 4
    ) -> weights.W:
        """
        Create kernel-based spatial weights.
        
        Args:
            data: GeoDataFrame containing spatial data. If None, uses the data
                 provided during initialization.
            bandwidth: Bandwidth for the kernel. If None, uses the default bandwidth.
            kernel: Kernel function to use. Options are 'gaussian', 'bisquare',
                   'triangular', and 'uniform'.
            fixed: Whether to use fixed bandwidth (True) or adaptive bandwidth (False).
            k: Number of nearest neighbors for adaptive bandwidth.
            
        Returns:
            Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data is invalid or the weights cannot be created.
        """
        logger.info(f"Creating kernel weights with kernel={kernel}, fixed={fixed}")
        
        # Use provided data or data from initialization
        if data is not None:
            self.data = data
        
        # Validate data
        if self.data is None:
            logger.error("No data provided")
            raise YemenAnalysisError("No data provided")
        
        validate_data(self.data, 'spatial')
        
        try:
            # Create kernel weights
            self.w = weights.Kernel.from_dataframe(
                self.data, bandwidth=bandwidth, function=kernel, fixed=fixed, k=k
            )
            
            # Set type and parameters
            self.type = 'kernel'
            self.params = {
                'bandwidth': bandwidth,
                'kernel': kernel,
                'fixed': fixed,
                'k': k,
            }
            
            logger.info(f"Created kernel weights with {len(self.w.neighbors)} units")
            return self.w
        except Exception as e:
            logger.error(f"Error creating kernel weights: {e}")
            raise YemenAnalysisError(f"Error creating kernel weights: {e}")
    
    @handle_errors
    def create_conflict_weights(
        self, data: Optional[gpd.GeoDataFrame] = None, conflict_column: str = 'conflict_intensity',
        distance_column: Optional[str] = None, alpha: float = -1.0,
        normalize: bool = True
    ) -> weights.W:
        """
        Create conflict-adjusted spatial weights.
        
        This method creates spatial weights that are adjusted for conflict intensity.
        The weights are calculated as:
        w_ij = d_ij^alpha * (1 - c_ij)
        
        where d_ij is the distance between units i and j, c_ij is the conflict
        intensity between units i and j, and alpha is the distance decay parameter.
        
        Args:
            data: GeoDataFrame containing spatial data. If None, uses the data
                 provided during initialization.
            conflict_column: Column containing conflict intensity.
            distance_column: Column containing distances. If None, calculates
                            distances from geometries.
            alpha: Distance decay parameter.
            normalize: Whether to row-normalize the weights.
            
        Returns:
            Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the data is invalid or the weights cannot be created.
        """
        logger.info(f"Creating conflict-adjusted weights with alpha={alpha}")
        
        # Use provided data or data from initialization
        if data is not None:
            self.data = data
        
        # Validate data
        if self.data is None:
            logger.error("No data provided")
            raise YemenAnalysisError("No data provided")
        
        validate_data(self.data, 'spatial')
        
        # Check if conflict column exists
        if conflict_column not in self.data.columns:
            logger.error(f"Conflict column {conflict_column} not found in data")
            raise YemenAnalysisError(f"Conflict column {conflict_column} not found in data")
        
        try:
            # Get conflict intensity
            conflict = self.data[conflict_column]
            
            # Normalize conflict intensity to [0, 1]
            if conflict.min() < 0 or conflict.max() > 1:
                conflict = (conflict - conflict.min()) / (conflict.max() - conflict.min())
            
            # Calculate distances
            if distance_column is not None:
                # Use provided distances
                if distance_column not in self.data.columns:
                    logger.error(f"Distance column {distance_column} not found in data")
                    raise YemenAnalysisError(f"Distance column {distance_column} not found in data")
                
                distances = self.data[distance_column]
            else:
                # Calculate distances from geometries
                # Get centroids
                centroids = self.data.geometry.centroid
                
                # Calculate distance matrix
                n = len(self.data)
                distances = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            distances[i, j] = centroids.iloc[i].distance(centroids.iloc[j])
            
            # Calculate conflict-adjusted weights
            n = len(self.data)
            w_data = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Calculate weight
                        distance_effect = distances[i, j] ** alpha
                        conflict_effect = 1 - conflict.iloc[j]
                        w_data[i, j] = distance_effect * conflict_effect
            
            # Create weight matrix
            self.w = weights.util.full2W(w_data)
            
            # Row-normalize if requested
            if normalize:
                self.w.transform = 'r'
            
            # Set type and parameters
            self.type = 'conflict'
            self.params = {
                'conflict_column': conflict_column,
                'distance_column': distance_column,
                'alpha': alpha,
                'normalize': normalize,
            }
            
            logger.info(f"Created conflict-adjusted weights with {len(self.w.neighbors)} units")
            return self.w
        except Exception as e:
            logger.error(f"Error creating conflict-adjusted weights: {e}")
            raise YemenAnalysisError(f"Error creating conflict-adjusted weights: {e}")
    
    @handle_errors
    def row_normalize(self) -> weights.W:
        """
        Row-normalize the spatial weight matrix.
        
        Returns:
            Row-normalized spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created.
        """
        logger.info("Row-normalizing spatial weights")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Row-normalize the weight matrix
            self.w.transform = 'r'
            
            logger.info("Row-normalized spatial weights")
            return self.w
        except Exception as e:
            logger.error(f"Error row-normalizing spatial weights: {e}")
            raise YemenAnalysisError(f"Error row-normalizing spatial weights: {e}")
    
    @handle_errors
    def to_sparse_matrix(self) -> sparse.csr_matrix:
        """
        Convert the spatial weight matrix to a sparse matrix.
        
        Returns:
            Sparse matrix representation of the spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created.
        """
        logger.info("Converting spatial weights to sparse matrix")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Convert to sparse matrix
            sparse_matrix = self.w.sparse
            
            logger.info("Converted spatial weights to sparse matrix")
            return sparse_matrix
        except Exception as e:
            logger.error(f"Error converting spatial weights to sparse matrix: {e}")
            raise YemenAnalysisError(f"Error converting spatial weights to sparse matrix: {e}")
    
    @handle_errors
    def to_dense_matrix(self) -> np.ndarray:
        """
        Convert the spatial weight matrix to a dense matrix.
        
        Returns:
            Dense matrix representation of the spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created.
        """
        logger.info("Converting spatial weights to dense matrix")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Convert to dense matrix
            dense_matrix = self.w.full()[0]
            
            logger.info("Converted spatial weights to dense matrix")
            return dense_matrix
        except Exception as e:
            logger.error(f"Error converting spatial weights to dense matrix: {e}")
            raise YemenAnalysisError(f"Error converting spatial weights to dense matrix: {e}")
    
    @handle_errors
    def save(self, file_path: str) -> None:
        """
        Save the spatial weight matrix to a file.
        
        Args:
            file_path: Path to save the weight matrix.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created or cannot be saved.
        """
        logger.info(f"Saving spatial weights to {file_path}")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Save weight matrix
            self.w.save(file_path)
            
            logger.info(f"Saved spatial weights to {file_path}")
        except Exception as e:
            logger.error(f"Error saving spatial weights: {e}")
            raise YemenAnalysisError(f"Error saving spatial weights: {e}")
    
    @handle_errors
    def load(self, file_path: str) -> weights.W:
        """
        Load a spatial weight matrix from a file.
        
        Args:
            file_path: Path to load the weight matrix from.
            
        Returns:
            Loaded spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the weight matrix cannot be loaded.
        """
        logger.info(f"Loading spatial weights from {file_path}")
        
        try:
            # Load weight matrix
            self.w = weights.W.from_file(file_path)
            
            # Set type and parameters
            self.type = 'loaded'
            self.params = {'file_path': file_path}
            
            logger.info(f"Loaded spatial weights from {file_path}")
            return self.w
        except Exception as e:
            logger.error(f"Error loading spatial weights: {e}")
            raise YemenAnalysisError(f"Error loading spatial weights: {e}")
    
    @handle_errors
    def get_neighbors(self, unit_id: int) -> List[int]:
        """
        Get the neighbors of a unit.
        
        Args:
            unit_id: ID of the unit.
            
        Returns:
            List of neighbor IDs.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created or the unit ID is invalid.
        """
        logger.info(f"Getting neighbors of unit {unit_id}")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Get neighbors
            neighbors = self.w.neighbors[unit_id]
            
            logger.info(f"Unit {unit_id} has {len(neighbors)} neighbors")
            return neighbors
        except KeyError:
            logger.error(f"Invalid unit ID: {unit_id}")
            raise YemenAnalysisError(f"Invalid unit ID: {unit_id}")
        except Exception as e:
            logger.error(f"Error getting neighbors: {e}")
            raise YemenAnalysisError(f"Error getting neighbors: {e}")
    
    @handle_errors
    def get_weights(self, unit_id: int) -> Dict[int, float]:
        """
        Get the weights of a unit's neighbors.
        
        Args:
            unit_id: ID of the unit.
            
        Returns:
            Dictionary mapping neighbor IDs to weights.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created or the unit ID is invalid.
        """
        logger.info(f"Getting weights of unit {unit_id}")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Get neighbors and weights
            neighbors = self.w.neighbors[unit_id]
            weights = self.w.weights[unit_id]
            
            # Create dictionary mapping neighbor IDs to weights
            neighbor_weights = dict(zip(neighbors, weights))
            
            logger.info(f"Unit {unit_id} has {len(neighbor_weights)} neighbors with weights")
            return neighbor_weights
        except KeyError:
            logger.error(f"Invalid unit ID: {unit_id}")
            raise YemenAnalysisError(f"Invalid unit ID: {unit_id}")
        except Exception as e:
            logger.error(f"Error getting weights: {e}")
            raise YemenAnalysisError(f"Error getting weights: {e}")
    
    @handle_errors
    def get_spatial_lag(self, data: pd.Series) -> pd.Series:
        """
        Calculate the spatial lag of a variable.
        
        Args:
            data: Series containing the variable.
            
        Returns:
            Series containing the spatial lag.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created or the data is invalid.
        """
        logger.info("Calculating spatial lag")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Calculate spatial lag
            spatial_lag = weights.lag_spatial(self.w, data)
            
            logger.info("Calculated spatial lag")
            return pd.Series(spatial_lag, index=data.index)
        except Exception as e:
            logger.error(f"Error calculating spatial lag: {e}")
            raise YemenAnalysisError(f"Error calculating spatial lag: {e}")
    
    @handle_errors
    def get_moran(self, data: pd.Series) -> Tuple[float, float]:
        """
        Calculate Moran's I for a variable.
        
        Args:
            data: Series containing the variable.
            
        Returns:
            Tuple containing Moran's I and its p-value.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created or the data is invalid.
        """
        logger.info("Calculating Moran's I")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Import Moran from PySAL
            from esda.moran import Moran
            
            # Calculate Moran's I
            moran = Moran(data, self.w)
            
            logger.info(f"Calculated Moran's I: {moran.I:.4f} (p-value: {moran.p_sim:.4f})")
            return moran.I, moran.p_sim
        except Exception as e:
            logger.error(f"Error calculating Moran's I: {e}")
            raise YemenAnalysisError(f"Error calculating Moran's I: {e}")
    
    @handle_errors
    def get_geary(self, data: pd.Series) -> Tuple[float, float]:
        """
        Calculate Geary's C for a variable.
        
        Args:
            data: Series containing the variable.
            
        Returns:
            Tuple containing Geary's C and its p-value.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created or the data is invalid.
        """
        logger.info("Calculating Geary's C")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Import Geary from PySAL
            from esda.geary import Geary
            
            # Calculate Geary's C
            geary = Geary(data, self.w)
            
            logger.info(f"Calculated Geary's C: {geary.C:.4f} (p-value: {geary.p_sim:.4f})")
            return geary.C, geary.p_sim
        except Exception as e:
            logger.error(f"Error calculating Geary's C: {e}")
            raise YemenAnalysisError(f"Error calculating Geary's C: {e}")
    
    @handle_errors
    def get_getis_ord(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Getis-Ord G* for a variable.
        
        Args:
            data: Series containing the variable.
            
        Returns:
            Tuple containing G* statistics and their p-values.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created or the data is invalid.
        """
        logger.info("Calculating Getis-Ord G*")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Import G_Local from PySAL
            from esda.getisord import G_Local
            
            # Calculate Getis-Ord G*
            g_local = G_Local(data, self.w)
            
            logger.info(f"Calculated Getis-Ord G* for {len(g_local.Zs)} units")
            return g_local.Zs, g_local.p_sim
        except Exception as e:
            logger.error(f"Error calculating Getis-Ord G*: {e}")
            raise YemenAnalysisError(f"Error calculating Getis-Ord G*: {e}")
    
    @handle_errors
    def get_local_moran(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Local Moran's I for a variable.
        
        Args:
            data: Series containing the variable.
            
        Returns:
            Tuple containing Local Moran's I statistics and their p-values.
            
        Raises:
            YemenAnalysisError: If the weight matrix has not been created or the data is invalid.
        """
        logger.info("Calculating Local Moran's I")
        
        # Check if weight matrix exists
        if self.w is None:
            logger.error("Weight matrix has not been created")
            raise YemenAnalysisError("Weight matrix has not been created")
        
        try:
            # Import Moran_Local from PySAL
            from esda.moran import Moran_Local
            
            # Calculate Local Moran's I
            local_moran = Moran_Local(data, self.w)
            
            logger.info(f"Calculated Local Moran's I for {len(local_moran.Is)} units")
            return local_moran.Is, local_moran.p_sim
        except Exception as e:
            logger.error(f"Error calculating Local Moran's I: {e}")
            raise YemenAnalysisError(f"Error calculating Local Moran's I: {e}")
