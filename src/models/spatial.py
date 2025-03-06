"""
Spatial econometric models for market integration analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from libpysal.weights import KNN, Kernel, W
from esda.moran import Moran
from spreg import OLS, ML_Lag, ML_Error

from src.utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_geodataframe, validate_dataframe, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, parallelize,
    
    # Spatial utilities
    reproject_gdf, calculate_distances, create_buffer,
    
    # Data processing
    clean_column_names, normalize_columns, fill_missing_values,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values
DEFAULT_CONFLICT_WEIGHT = config.get('analysis.spatial.conflict_weight', 0.5)
DEFAULT_KNN = config.get('analysis.spatial.knn', 5)


class SpatialEconometrics:
    """Spatial econometric analysis for market integration."""
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        """
        Initialize with a GeoDataFrame.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data
        """
        # Validate input
        self._validate_input(gdf)
        
        self.gdf = gdf
        self.weights = None
        logger.info(f"Initialized SpatialEconometrics with {len(gdf)} observations")
    
    def _validate_input(self, gdf: gpd.GeoDataFrame) -> None:
        """Validate input GeoDataFrame."""
        # Check if GeoDataFrame
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValidationError("Input must be a GeoDataFrame")
        
        # Validate with utility function
        valid, errors = validate_geodataframe(
            gdf,
            required_columns=None,  # No specific requirements yet
            min_rows=1,
            check_crs=True
        )
        raise_if_invalid(valid, errors, "Invalid GeoDataFrame for spatial analysis")
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def create_weight_matrix(
        self, 
        k: int = DEFAULT_KNN, 
        conflict_adjusted: bool = True, 
        conflict_col: str = 'conflict_intensity_normalized',
        conflict_weight: float = DEFAULT_CONFLICT_WEIGHT
    ) -> W:
        """
        Create spatial weights matrix.
        
        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors
        conflict_adjusted : bool, optional
            If True, adjust weights by conflict intensity
        conflict_col : str, optional
            Column name for conflict intensity
        conflict_weight : float, optional
            Weight to apply to conflict adjustment (0-1)
            
        Returns
        -------
        libpysal.weights.W
            Spatial weights matrix
        """
        # Validate parameters
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        if conflict_adjusted and conflict_col not in self.gdf.columns:
            raise ValueError(f"Conflict column '{conflict_col}' not found in GeoDataFrame")
        
        if not 0 <= conflict_weight <= 1:
            raise ValueError(f"conflict_weight must be between 0 and 1, got {conflict_weight}")
        
        # Basic KNN weights
        knn = KNN.from_dataframe(self.gdf, k=k)
        
        # Apply conflict adjustment if requested
        if conflict_adjusted:
            self.weights = self._adjust_weights_by_conflict(
                knn, conflict_col, conflict_weight
            )
        else:
            self.weights = knn
        
        logger.info(f"Created weight matrix with k={k}, conflict_adjusted={conflict_adjusted}")
        return self.weights
    
    @m1_optimized()
    def _adjust_weights_by_conflict(
        self, 
        knn: W, 
        conflict_col: str, 
        conflict_weight: float
    ) -> W:
        """
        Adjust weights based on conflict intensity.
        
        Parameters
        ----------
        knn : libpysal.weights.W
            KNN weights matrix
        conflict_col : str
            Column name for conflict intensity
        conflict_weight : float
            Weight to apply to conflict adjustment (0-1)
            
        Returns
        -------
        libpysal.weights.W
            Adjusted weights matrix
        """
        # Adjust weights based on conflict intensity
        # Higher conflict = lower weight (more economic distance)
        adj_weights = {}
        
        for i, neighbors in knn.neighbors.items():
            weights = []
            for j in neighbors:
                # Base weight (inverse distance)
                base_weight = knn.weights[i][knn.neighbors[i].index(j)]
                
                # Get conflict intensity for both regions
                conflict_i = self.gdf.iloc[i][conflict_col]
                conflict_j = self.gdf.iloc[j][conflict_col]
                
                # Average conflict intensity along the path
                avg_conflict = (conflict_i + conflict_j) / 2
                
                # Adjust weight: higher conflict = lower weight
                # conflict_weight controls how much conflict affects the weight
                adjusted_weight = base_weight * (1 - (conflict_weight * avg_conflict))
                weights.append(adjusted_weight)
            
            adj_weights[i] = weights
        
        # Create new weight matrix with adjusted weights
        return W(knn.neighbors, adj_weights)
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def moran_i_test(self, variable: str) -> Dict[str, Any]:
        """
        Test for spatial autocorrelation using Moran's I.
        
        Parameters
        ----------
        variable : str
            Column name in GeoDataFrame to test
            
        Returns
        -------
        dict
            Moran's I test results
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Check if variable exists
        if variable not in self.gdf.columns:
            raise ValueError(f"Variable '{variable}' not found in GeoDataFrame")
        
        # Calculate Moran's I
        moran = Moran(self.gdf[variable], self.weights)
        
        result = {
            'I': moran.I,
            'expected_I': moran.EI,
            'p_norm': moran.p_norm,
            'p_sim': moran.p_sim,
            'z_norm': moran.z_norm,
            'significant': moran.p_norm < 0.05,
            'positive_autocorrelation': moran.I > moran.EI and moran.p_norm < 0.05
        }
        
        logger.info(
            f"Moran's I test for {variable}: I={result['I']:.4f}, "
            f"p={result['p_norm']:.4f}, significant={result['significant']}"
        )
        
        return result
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def spatial_lag_model(
        self, 
        y_col: str, 
        x_cols: List[str],
        name_y: str = None,
        name_x: List[str] = None
    ) -> Any:
        """
        Estimate a spatial lag model.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name
        x_cols : list
            List of independent variable column names
        name_y : str, optional
            Name for dependent variable in output
        name_x : list, optional
            Names for independent variables in output
            
        Returns
        -------
        spreg.ML_Lag
            Spatial lag model results
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Validate columns
        self._validate_model_columns([y_col] + x_cols)
        
        # Clean and prepare data
        model_data = self.gdf.copy()
        
        # Normalize variables for better numerical stability
        model_data = normalize_columns(
            model_data, 
            columns=[y_col] + [col for col in x_cols if model_data[col].dtype.kind in 'if'],
            method='zscore'
        )
        
        # Prepare data
        y = model_data[y_col].values
        X = model_data[x_cols].values
        
        # Set default names if not provided
        if name_y is None:
            name_y = y_col
        if name_x is None:
            name_x = x_cols
        
        # Estimate model
        model = ML_Lag(y, X, self.weights, name_y=name_y, name_x=name_x)
        
        logger.info(
            f"Spatial lag model estimated: AIC={model.aic:.4f}, "
            f"R2={model.pr2:.4f}, rho={model.rho:.4f}"
        )
        
        return model
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def spatial_error_model(
        self, 
        y_col: str, 
        x_cols: List[str],
        name_y: str = None,
        name_x: List[str] = None
    ) -> Any:
        """
        Estimate a spatial error model.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name
        x_cols : list
            List of independent variable column names
        name_y : str, optional
            Name for dependent variable in output
        name_x : list, optional
            Names for independent variables in output
            
        Returns
        -------
        spreg.ML_Error
            Spatial error model results
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Validate columns
        self._validate_model_columns([y_col] + x_cols)
        
        # Prepare data
        y = self.gdf[y_col].values
        X = self.gdf[x_cols].values
        
        # Set default names if not provided
        if name_y is None:
            name_y = y_col
        if name_x is None:
            name_x = x_cols
        
        # Estimate model
        model = ML_Error(y, X, self.weights, name_y=name_y, name_x=name_x)
        
        logger.info(
            f"Spatial error model estimated: AIC={model.aic:.4f}, "
            f"R2={model.pr2:.4f}, lambda={model.lam:.4f}"
        )
        
        return model
    
    def _validate_model_columns(self, columns: List[str]) -> None:
        """Validate that columns exist in the GeoDataFrame."""
        missing_cols = [col for col in columns if col not in self.gdf.columns]
        if missing_cols:
            raise ValueError(f"Column(s) not found in GeoDataFrame: {', '.join(missing_cols)}")


@timer
@memory_usage_decorator
@m1_optimized(parallel=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def calculate_market_accessibility(
    markets_gdf: gpd.GeoDataFrame,
    population_gdf: gpd.GeoDataFrame,
    max_distance: float = 50000,  # 50 km
    distance_decay: float = 2.0,
    weight_col: str = 'population'
) -> gpd.GeoDataFrame:
    """
    Calculate market accessibility index for each market.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Markets with locations
    population_gdf : geopandas.GeoDataFrame
        Population centers with locations
    max_distance : float, optional
        Maximum distance in meters to consider
    distance_decay : float, optional
        Distance decay exponent
    weight_col : str, optional
        Column in population_gdf to use as weight
        
    Returns
    -------
    geopandas.GeoDataFrame
        Market GeoDataFrame with accessibility index
    """
    # Validate inputs
    if not isinstance(markets_gdf, gpd.GeoDataFrame):
        raise ValidationError("markets_gdf must be a GeoDataFrame")
    if not isinstance(population_gdf, gpd.GeoDataFrame):
        raise ValidationError("population_gdf must be a GeoDataFrame")
    
    # Ensure both GDFs have same CRS
    if markets_gdf.crs != population_gdf.crs:
        population_gdf = population_gdf.to_crs(markets_gdf.crs)
    
    # Create buffers around markets
    buffer_gdf = create_buffer(markets_gdf, distance=max_distance)
    
    # Create a copy of markets_gdf to store results
    result_gdf = markets_gdf.copy()
    result_gdf['accessibility_index'] = 0.0
    
    # Process each market in parallel
    def process_market(idx_market):
        idx, market = idx_market
        
        # Find population centers within buffer
        pop_in_buffer = population_gdf[population_gdf.intersects(market.geometry)]
        
        if len(pop_in_buffer) == 0:
            return idx, 0.0
        
        # Calculate distances to all population centers in buffer
        distances = pop_in_buffer.geometry.distance(markets_gdf.loc[idx].geometry)
        
        # Apply distance decay function and weight by population
        accessibility = sum(
            pop_in_buffer[weight_col] / (distances ** distance_decay)
        )
        
        return idx, accessibility
    
    # Use parallelize for better performance
    process_args = [(i, market) for i, market in buffer_gdf.iterrows()]
    results = parallelize(process_market, process_args)
    
    # Update result GeoDataFrame
    for idx, accessibility in results:
        result_gdf.loc[idx, 'accessibility_index'] = accessibility
    
    logger.info(f"Calculated accessibility index for {len(markets_gdf)} markets")
    return result_gdf


@timer
@m1_optimized(parallel=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def calculate_market_isolation(
    markets_gdf: gpd.GeoDataFrame,
    transport_network_gdf: Optional[gpd.GeoDataFrame] = None,
    conflict_col: Optional[str] = None,
    max_distance: float = 50000  # 50 km
) -> gpd.GeoDataFrame:
    """
    Calculate market isolation index based on distance to other markets and conflict.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Markets with locations
    transport_network_gdf : geopandas.GeoDataFrame, optional
        Road or transport network
    conflict_col : str, optional
        Column with conflict intensity
    max_distance : float, optional
        Maximum distance in meters to consider
        
    Returns
    -------
    geopandas.GeoDataFrame
        Market GeoDataFrame with isolation index
    """
    # Validate inputs
    if not isinstance(markets_gdf, gpd.GeoDataFrame):
        raise ValidationError("markets_gdf must be a GeoDataFrame")
    
    # Create a copy of markets_gdf to store results
    result_gdf = markets_gdf.copy()
    
    # Calculate market-to-market distances
    distances = calculate_distances(
        markets_gdf, 
        markets_gdf, 
        id_col=markets_gdf.index.name or 'index'
    )
    
    # Calculate base isolation (inverse of connectivity)
    isolation_scores = []
    
    # For each market, calculate isolation score
    for i, market in markets_gdf.iterrows():
        # Get distances to other markets (excluding self)
        market_distances = distances[distances.origin_id != distances.dest_id]
        market_distances = market_distances[
            (market_distances.origin_id == i) & 
            (market_distances.distance <= max_distance)
        ]
        
        # Count nearby markets and calculate average distance
        n_nearby = len(market_distances)
        if n_nearby == 0:
            avg_distance = max_distance
        else:
            avg_distance = market_distances.distance.mean()
        
        # Base isolation score: higher when fewer nearby markets or greater distances
        base_isolation = 1 - (n_nearby / len(markets_gdf)) * (1 - avg_distance / max_distance)
        
        # Adjust for conflict if specified
        if conflict_col and conflict_col in markets_gdf.columns:
            conflict_factor = 1 + market[conflict_col]
            isolation = base_isolation * conflict_factor
        else:
            isolation = base_isolation
        
        isolation_scores.append(isolation)
    
    # Add isolation index to result
    result_gdf['isolation_index'] = isolation_scores
    
    logger.info(f"Calculated isolation index for {len(markets_gdf)} markets")
    return result_gdf