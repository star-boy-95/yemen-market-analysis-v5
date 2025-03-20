# Spatial Econometrics Module Implementation Outline

This document provides a detailed implementation outline for the `spatial.py` module, which is a critical component of the Yemen Market Integration project. This implementation will follow the mathematical specifications outlined in the econometric_methods.md document.

## Overview

The spatial econometrics module implements methods for analyzing geographic dependencies in market integration, with special focus on how conflict affects spatial market relationships. It provides tools for creating conflict-adjusted spatial weights, estimating spatial models, and calculating market accessibility indices.

## Class Structure

```python
class SpatialEconometrics:
    """
    Spatial econometric analysis tools for market integration.
    
    This class provides specialized tools for spatial analysis of market
    integration in conflict-affected settings, with particular emphasis
    on how geographic barriers and conflict intensity affect market relationships.
    """
    
    def __init__(self, gdf):
        """
        Initialize SpatialEconometrics with market GeoDataFrame.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing market locations with price and other attributes
        """
        # Implementation
        
    def create_weight_matrix(self, k=5, conflict_adjusted=True, conflict_col=None, **kwargs):
        """
        Create spatial weights matrix with optional conflict adjustment.
        
        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors for KNN weights
        conflict_adjusted : bool, optional
            Whether to adjust weights for conflict intensity
        conflict_col : str, optional
            Column name containing conflict intensity
        **kwargs : dict
            Additional parameters for weight matrix creation
            
        Returns
        -------
        W : libpysal.weights.W
            Spatial weights matrix
        """
        # Implementation
        
    def moran_i_test(self, col, weights=None, **kwargs):
        """
        Perform Moran's I test for spatial autocorrelation.
        
        Parameters
        ----------
        col : str
            Column name for variable to test
        weights : libpysal.weights.W, optional
            Spatial weights matrix (uses self.weights if None)
        **kwargs : dict
            Additional parameters for Moran's I test
            
        Returns
        -------
        dict
            Moran's I test results including:
            - I: Moran's I statistic
            - p_norm: p-value from normal approximation
            - p_rand: p-value from randomization
            - z_norm: z-score for normal approximation
            - autocorrelation: classification of autocorrelation type
        """
        # Implementation
        
    def local_moran_test(self, col, weights=None, **kwargs):
        """
        Perform Local Moran's I test for local spatial autocorrelation.
        
        Parameters
        ----------
        col : str
            Column name for variable to test
        weights : libpysal.weights.W, optional
            Spatial weights matrix (uses self.weights if None)
        **kwargs : dict
            Additional parameters for Local Moran's I test
            
        Returns
        -------
        geopandas.GeoDataFrame
            Original GeoDataFrame with additional columns:
            - local_i: Local Moran's I statistic
            - p_sim: p-value from simulation
            - quadrant: Classification of spatial association
            - cluster_type: Type of spatial cluster
        """
        # Implementation
        
    def spatial_lag_model(self, y_col, x_cols, weights=None, **kwargs):
        """
        Estimate Spatial Lag Model (SLM).
        
        Parameters
        ----------
        y_col : str
            Column name for dependent variable
        x_cols : list of str
            Column names for independent variables
        weights : libpysal.weights.W, optional
            Spatial weights matrix (uses self.weights if None)
        **kwargs : dict
            Additional parameters for model estimation
            
        Returns
        -------
        dict
            Model results including:
            - rho: spatial autoregressive parameter
            - betas: coefficient estimates
            - std_err: standard errors
            - t_stats: t-statistics
            - p_values: p-values
            - r2: R-squared
            - log_likelihood: log-likelihood
            - aic: Akaike Information Criterion
            - direct_effects: direct effects of variables
            - indirect_effects: indirect (spillover) effects
            - total_effects: total effects
        """
        # Implementation
        
    def spatial_error_model(self, y_col, x_cols, weights=None, **kwargs):
        """
        Estimate Spatial Error Model (SEM).
        
        Parameters
        ----------
        y_col : str
            Column name for dependent variable
        x_cols : list of str
            Column names for independent variables
        weights : libpysal.weights.W, optional
            Spatial weights matrix (uses self.weights if None)
        **kwargs : dict
            Additional parameters for model estimation
            
        Returns
        -------
        dict
            Model results including:
            - lambda: spatial error parameter
            - betas: coefficient estimates
            - std_err: standard errors
            - t_stats: t-statistics
            - p_values: p-values
            - r2: R-squared
            - log_likelihood: log-likelihood
            - aic: Akaike Information Criterion
        """
        # Implementation
        
    def spatial_durbin_model(self, y_col, x_cols, weights=None, **kwargs):
        """
        Estimate Spatial Durbin Model (SDM).
        
        Parameters
        ----------
        y_col : str
            Column name for dependent variable
        x_cols : list of str
            Column names for independent variables
        weights : libpysal.weights.W, optional
            Spatial weights matrix (uses self.weights if None)
        **kwargs : dict
            Additional parameters for model estimation
            
        Returns
        -------
        dict
            Model results including:
            - rho: spatial autoregressive parameter
            - betas: coefficient estimates
            - thetas: spatial lag coefficient estimates
            - std_err: standard errors
            - t_stats: t-statistics
            - p_values: p-values
            - r2: R-squared
            - log_likelihood: log-likelihood
            - aic: Akaike Information Criterion
            - direct_effects: direct effects of variables
            - indirect_effects: indirect (spillover) effects
            - total_effects: total effects
        """
        # Implementation
        
    def compute_accessibility_index(
        self, population_gdf, max_distance=None, distance_decay=2.0, 
        weight_col='population', **kwargs
    ):
        """
        Compute market accessibility index considering population and distance.
        
        Parameters
        ----------
        population_gdf : geopandas.GeoDataFrame
            GeoDataFrame with population points/polygons
        max_distance : float, optional
            Maximum distance to consider
        distance_decay : float, optional
            Distance decay parameter (power)
        weight_col : str, optional
            Column name in population_gdf for weights
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        geopandas.GeoDataFrame
            Original GeoDataFrame with additional column 'accessibility_index'
        """
        # Implementation
        
    def calculate_market_isolation(
        self, transport_network_gdf=None, conflict_col=None,
        conflict_weight=0.5, **kwargs
    ):
        """
        Calculate market isolation index considering conflict barriers.
        
        Parameters
        ----------
        transport_network_gdf : geopandas.GeoDataFrame, optional
            GeoDataFrame with transport network
        conflict_col : str, optional
            Column name for conflict intensity
        conflict_weight : float, optional
            Weight for conflict in isolation calculation
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        geopandas.GeoDataFrame
            Original GeoDataFrame with additional column 'isolation_index'
        """
        # Implementation
        
    def visualize_conflict_adjusted_weights(self, **kwargs):
        """
        Visualize conflict-adjusted spatial weights.
        
        Parameters
        ----------
        **kwargs : dict
            Visualization parameters
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with spatial weights visualization
        """
        # Implementation
```

## Function Definitions

### 1. Distance and Connectivity Functions

```python
def calculate_distances(origin_gdf, destination_gdf, id_col=None, crs=None):
    """
    Calculate distances between all pairs of points.
    
    Parameters
    ----------
    origin_gdf : geopandas.GeoDataFrame
        GeoDataFrame with origin points
    destination_gdf : geopandas.GeoDataFrame
        GeoDataFrame with destination points
    id_col : str, optional
        Column name for identifying points
    crs : str or int, optional
        Coordinate reference system for distance calculation
        
    Returns
    -------
    pandas.DataFrame
        Distance matrix with origin IDs as index and destination IDs as columns
    """
    # Implementation

def calculate_distance_matrix(gdf, id_col=None, method='euclidean', crs=None):
    """
    Calculate distance matrix for a set of points.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with points
    id_col : str, optional
        Column name for identifying points
    method : str, optional
        Distance calculation method ('euclidean', 'geodesic', 'network')
    crs : str or int, optional
        Coordinate reference system for distance calculation
        
    Returns
    -------
    numpy.ndarray
        Distance matrix
    """
    # Implementation

def find_nearest_points(gdf1, gdf2, target_col=None, max_distance=None):
    """
    Find nearest points in gdf2 for each point in gdf1.
    
    Parameters
    ----------
    gdf1 : geopandas.GeoDataFrame
        GeoDataFrame with origin points
    gdf2 : geopandas.GeoDataFrame
        GeoDataFrame with target points
    target_col : str, optional
        Column name in gdf2 to return for nearest points
    max_distance : float, optional
        Maximum distance to search
        
    Returns
    -------
    geopandas.GeoDataFrame
        Original GeoDataFrame with additional columns:
        - nearest_id: ID of nearest point
        - nearest_distance: Distance to nearest point
        - nearest_{target_col}: Value of target_col for nearest point
    """
    # Implementation
```

### 2. Conflict Adjustment Functions

```python
def create_conflict_adjusted_weights(
    gdf, k=5, conflict_col=None, conflict_weight=0.5,
    additional_cost_matrix=None, additional_weight=None
):
    """
    Create spatial weights adjusted for conflict intensity.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with points
    k : int, optional
        Number of nearest neighbors
    conflict_col : str, optional
        Column name for conflict intensity
    conflict_weight : float, optional
        Weight for conflict in distance adjustment
    additional_cost_matrix : numpy.ndarray, optional
        Additional cost matrix (e.g., from regime boundaries)
    additional_weight : float, optional
        Weight for additional cost matrix
        
    Returns
    -------
    libpysal.weights.W
        Conflict-adjusted spatial weights matrix
    """
    # Implementation

def calculate_conflict_adjusted_distance(
    gdf, origin_idx, destination_idx, conflict_col=None, 
    conflict_weight=0.5, base_distance=None
):
    """
    Calculate distance adjusted for conflict intensity.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with points
    origin_idx : int
        Index of origin point
    destination_idx : int
        Index of destination point
    conflict_col : str, optional
        Column name for conflict intensity
    conflict_weight : float, optional
        Weight for conflict in distance adjustment
    base_distance : float, optional
        Base distance (if already calculated)
        
    Returns
    -------
    float
        Conflict-adjusted distance
    """
    # Implementation
```

### 3. Market Integration and Boundary Functions

```python
def create_market_catchments(
    markets_gdf, population_gdf, market_id_col='market_id',
    population_weight_col='population', max_distance=None, distance_decay=2.0
):
    """
    Create market catchment areas based on accessibility.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        GeoDataFrame with market points
    population_gdf : geopandas.GeoDataFrame
        GeoDataFrame with population points/polygons
    market_id_col : str, optional
        Column name for market IDs
    population_weight_col : str, optional
        Column name for population weights
    max_distance : float, optional
        Maximum catchment distance
    distance_decay : float, optional
        Distance decay parameter
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with market catchment polygons
    """
    # Implementation

def create_exchange_regime_boundaries(
    admin_gdf, regime_col='exchange_rate_regime', 
    dissolve=True, simplify_tolerance=None
):
    """
    Create boundary between exchange rate regimes.
    
    Parameters
    ----------
    admin_gdf : geopandas.GeoDataFrame
        GeoDataFrame with administrative boundaries
    regime_col : str, optional
        Column name for exchange rate regime
    dissolve : bool, optional
        Whether to dissolve boundaries within regimes
    simplify_tolerance : float, optional
        Tolerance for geometry simplification
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with regime boundary geometries
    """
    # Implementation

def assign_exchange_rate_regime(
    points_gdf, regime_polygons_gdf, regime_col='exchange_rate_regime', 
    default_regime=None
):
    """
    Assign exchange rate regime to points based on spatial join.
    
    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        GeoDataFrame with points to assign regimes to
    regime_polygons_gdf : geopandas.GeoDataFrame
        GeoDataFrame with regime polygons
    regime_col : str, optional
        Column name for exchange rate regime
    default_regime : any, optional
        Default value for points outside all regimes
        
    Returns
    -------
    geopandas.GeoDataFrame
        Original GeoDataFrame with additional column for regime
    """
    # Implementation
```

### 4. Market Integration Analysis Functions

```python
def market_integration_index(
    gdf, weights, market_id_col='market_id', price_col='price', 
    time_col='date', **kwargs
):
    """
    Calculate time-varying market integration index.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with market data
    weights : libpysal.weights.W
        Spatial weights matrix
    market_id_col : str, optional
        Column name for market IDs
    price_col : str, optional
        Column name for prices
    time_col : str, optional
        Column name for time periods
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    pandas.DataFrame
        Integration indices by time period
    """
    # Implementation

def simulate_improved_connectivity(
    gdf, conflict_reduction=0.5, conflict_col='conflict_intensity_normalized',
    price_col='price', **kwargs
):
    """
    Simulate improved connectivity through conflict reduction.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with market data
    conflict_reduction : float, optional
        Proportion of conflict intensity to reduce
    conflict_col : str, optional
        Column name for conflict intensity
    price_col : str, optional
        Column name for prices
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    dict
        Simulation results including:
        - simulated_data: GeoDataFrame with simulated data
        - original_weights: Original spatial weights
        - simulated_weights: Simulated spatial weights
        - price_convergence: Price convergence metrics
    """
    # Implementation
```

## Implementation Dependencies

The implementation will rely on:

1. **Geospatial Libraries**: Uses GeoPandas, PySAL, Shapely for spatial data handling.
2. **Spatial Econometrics**: Uses PySAL's spreg module for spatial regression models.
3. **Validation Utilities**: Uses validation functions from `utils/validation.py` to ensure valid inputs.
4. **Performance Utilities**: Uses performance optimization techniques for large spatial datasets.
5. **Visualization**: Depends on matplotlib and visualization utilities for spatial visualization.

## Implementation Approaches

### Spatial Weight Matrix Implementation Strategy

1. Handle multiple weight types (KNN, distance-based, contiguity)
2. Implement conflict adjustment by modifying the distance matrix
3. Consider exchange rate regime boundaries as additional barriers
4. Ensure proper normalization and standardization of weights
5. Optimize for large spatial datasets with sparse matrix representation

### Spatial Regression Implementation Strategy

1. Implement different model types (SLM, SEM, SDM)
2. Support both maximum likelihood and GMM estimation
3. Calculate direct and indirect effects for proper interpretation
4. Provide comprehensive model diagnostics
5. Handle spatial heterogeneity with spatial regimes
6. Ensure proper handling of spatial data structures

### Market Accessibility Implementation Strategy

1. Calculate population-weighted accessibility considering distance decay
2. Implement conflict-adjusted distance calculations
3. Support different distance decay functions (power, exponential)
4. Optimize for large population datasets
5. Provide visualization capabilities for accessibility maps

## Error Handling and Performance Considerations

1. **Strict Input Validation**: Validate all spatial data for:
   - Valid geometry types
   - Consistent coordinate reference systems
   - Required attributes
   - No missing geometries

2. **Memory Efficiency**:
   - Use sparse matrices for spatial weights
   - Implement chunked processing for large datasets
   - Optimize distance calculations
   - Use spatial indexing for efficient queries

3. **Error Handling**:
   - Use the `handle_errors` decorator for consistent error management
   - Provide specialized error messages for common GIS issues
   - Implement fallback options when possible

4. **Performance**:
   - Cache intermediate results like distance matrices
   - Use parallel processing for computationally intensive operations
   - Implement spatial indexing for nearest neighbor searches
   - Optimize memory usage for large GeoJSON processing

## Testing and Validation

1. **Unit Tests**:
   - Test with small synthetic spatial datasets
   - Test with different weight specifications
   - Verify spatial regression results against known examples

2. **Integration Tests**:
   - Test interaction with real market and conflict data
   - Test integration with visualization components
   - Validate against established spatial econometric software

3. **Performance Tests**:
   - Benchmark performance on large spatial datasets
   - Test with different spatial optimization techniques
   - Measure memory usage for different spatial data structures

## Future Extensions

1. **Advanced Spatial Methods**:
   - Geographically Weighted Regression (GWR)
   - Spatial panel data models
   - Space-time models for dynamic analysis

2. **Network Analysis**:
   - Road network-based distance calculations
   - Network disruption analysis due to conflict
   - Optimal route analysis for market access

3. **Integration with Remote Sensing**:
   - Incorporate satellite imagery for conflict detection
   - Use nighttime lights data for economic activity
   - Integrate vegetation indices for agricultural production
