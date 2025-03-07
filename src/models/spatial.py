"""
Spatial econometric models for market integration analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from libpysal.weights import KNN, Kernel, W
from esda.moran import Moran, Moran_Local
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import networkx as nx
import seaborn as sns
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
    """
    Spatial econometric analysis for market integration.
    
    This class implements spatial econometric methods for analyzing geographic 
    relationships in market data, with specific adaptations for conflict-affected
    environments like Yemen where traditional distance metrics may be misleading.
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        """
        Initialize with a GeoDataFrame.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data with market locations
        """
        # Validate input
        self._validate_input(gdf)
        
        self.gdf = gdf
        self.weights = None
        self.diagnostic_hooks = {}
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
        conflict_weight: float = DEFAULT_CONFLICT_WEIGHT,
        store_diagnostic_info: bool = True
    ) -> W:
        """
        Create spatial weights matrix with optional conflict adjustment.
        
        In Yemen's context, conflict creates effective barriers between markets
        that aren't captured by simple geographic distance. This method adjusts
        weights based on conflict intensity along market connections.
        
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
            Higher values mean conflict has stronger impact on connectivity
        store_diagnostic_info : bool, optional
            Whether to store diagnostic information for visualization
            
        Returns
        -------
        libpysal.weights.W
            Spatial weights matrix
            
        Notes
        -----
        A conflict-adjusted weights matrix better captures the effective economic 
        distance between markets in fragmented economies like Yemen where physical
        distance alone is misleading.
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
        
        # Store original weights for comparison if requested
        if store_diagnostic_info:
            self.diagnostic_hooks['original_weights'] = knn
            self.diagnostic_hooks['k'] = k
        
        # Apply conflict adjustment if requested
        if conflict_adjusted:
            self.weights = self._adjust_weights_by_conflict(
                knn, conflict_col, conflict_weight, store_diagnostic_info
            )
            # Store parameters for reference
            if store_diagnostic_info:
                self.diagnostic_hooks['conflict_adjusted'] = True
                self.diagnostic_hooks['conflict_col'] = conflict_col
                self.diagnostic_hooks['conflict_weight'] = conflict_weight
        else:
            self.weights = knn
            if store_diagnostic_info:
                self.diagnostic_hooks['conflict_adjusted'] = False
        
        logger.info(f"Created weight matrix with k={k}, conflict_adjusted={conflict_adjusted}")
        return self.weights
    
    @m1_optimized()
    def _adjust_weights_by_conflict(
        self, 
        knn: W, 
        conflict_col: str, 
        conflict_weight: float,
        store_diagnostic_info: bool = False
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
        store_diagnostic_info : bool
            Whether to store diagnostic information
            
        Returns
        -------
        libpysal.weights.W
            Adjusted weights matrix
        """
        # Adjust weights based on conflict intensity
        # Higher conflict = lower weight (more economic distance)
        adj_weights = {}
        adjustment_factors = {}
        
        for i, neighbors in knn.neighbors.items():
            weights = []
            adjustments = []
            
            for j in neighbors:
                # Base weight (inverse distance)
                base_weight = knn.weights[i][knn.neighbors[i].index(j)]
                
                # Get conflict intensity for both regions
                conflict_i = self.gdf.iloc[i][conflict_col]
                conflict_j = self.gdf.iloc[j][conflict_col]
                
                # Average conflict intensity along the path
                avg_conflict = (conflict_i + conflict_j) / 2
                
                # Calculate adjustment factor
                adjustment = 1 - (conflict_weight * avg_conflict)
                
                # Adjust weight: higher conflict = lower weight
                adjusted_weight = base_weight * adjustment
                
                weights.append(adjusted_weight)
                if store_diagnostic_info:
                    adjustments.append(adjustment)
            
            adj_weights[i] = weights
            if store_diagnostic_info:
                adjustment_factors[i] = adjustments
        
        # Store adjustment factors for diagnostics if requested
        if store_diagnostic_info:
            self.diagnostic_hooks['adjustment_factors'] = adjustment_factors
        
        # Create new weight matrix with adjusted weights
        return W(knn.neighbors, adj_weights)
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def moran_i_test(self, variable: str) -> Dict[str, Any]:
        """
        Test for spatial autocorrelation using Moran's I.
        
        In Yemen's market context, Moran's I indicates whether prices
        in neighboring markets move together, suggesting integration.
        
        Parameters
        ----------
        variable : str
            Column name in GeoDataFrame to test
            
        Returns
        -------
        dict
            Moran's I test results including:
            - I: Moran's I statistic
            - expected_I: Expected value under null hypothesis
            - p_norm: P-value (normal approximation)
            - z_norm: Z-score (normal approximation)
            - significant: Whether spatial autocorrelation is significant
            - positive_autocorrelation: Whether autocorrelation is positive
            
        Notes
        -----
        Significant positive Moran's I suggests market integration
        despite conflict barriers, while insignificant values suggest
        fragmented markets.
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
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def local_moran_test(self, variable: str) -> gpd.GeoDataFrame:
        """
        Conduct Local Moran's I test to identify spatial clusters.
        
        Local Moran's I helps identify clusters of markets with
        similar price behavior, and outliers with distinct patterns.
        
        Parameters
        ----------
        variable : str
            Column name in GeoDataFrame to test
            
        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with original data plus Local Moran results:
            - moran_local_i: Local Moran's I statistic
            - moran_p_value: P-value for local statistic
            - moran_significant: Whether local statistic is significant
            - cluster_type: Classification as high-high, low-low, etc.
            
        Notes
        -----
        In Yemen's context, clusters can reveal regions with different
        price dynamics due to political fragmentation, conflict barriers,
        or exchange rate effects.
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Check if variable exists
        if variable not in self.gdf.columns:
            raise ValueError(f"Variable '{variable}' not found in GeoDataFrame")
        
        # Calculate Local Moran's I
        local_moran = Moran_Local(self.gdf[variable], self.weights)
        
        # Create copy of GeoDataFrame to add results
        result_gdf = self.gdf.copy()
        
        # Add results to GeoDataFrame
        result_gdf['moran_local_i'] = local_moran.Is
        result_gdf['moran_p_value'] = local_moran.p_sim
        result_gdf['moran_significant'] = local_moran.p_sim < 0.05
        
        # Create cluster classification
        significant = result_gdf['moran_p_value'] < 0.05
        high = result_gdf[variable] > result_gdf[variable].mean()
        high_neighbors = local_moran.lag > local_moran.lag.mean()
        
        # Classify clusters
        result_gdf['cluster_type'] = 'not_significant'
        result_gdf.loc[significant & high & high_neighbors, 'cluster_type'] = 'high-high'
        result_gdf.loc[significant & ~high & ~high_neighbors, 'cluster_type'] = 'low-low'
        result_gdf.loc[significant & high & ~high_neighbors, 'cluster_type'] = 'high-low'
        result_gdf.loc[significant & ~high & high_neighbors, 'cluster_type'] = 'low-high'
        
        logger.info(
            f"Local Moran's I test for {variable}: "
            f"Significant clusters: {result_gdf['moran_significant'].sum()}/{len(result_gdf)}"
        )
        
        return result_gdf
    
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
        
        In Yemen's context, this model captures how prices in one market
        are directly influenced by prices in neighboring markets, adjusted
        for conflict barriers.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name (typically price)
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
            
        Notes
        -----
        The spatial autoregressive parameter (rho) indicates the strength
        of price transmission between markets. Higher values suggest
        better market integration despite conflict barriers.
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
        
        This model captures how unobserved spatial factors affect prices,
        such as regional conflict patterns not directly measured.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name (typically price)
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
            
        Notes
        -----
        A significant lambda parameter suggests spatial structure in
        the unobserved factors affecting prices, such as unmeasured
        regional conflict dynamics or exchange rate effects.
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
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def visualize_conflict_adjusted_weights(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        title: str = "Conflict-Adjusted Market Connectivity",
        node_color_col: Optional[str] = None,
        base_node_size: float = 100,
        edge_scale: float = 5,
        labels: bool = False
    ) -> plt.Figure:
        """
        Visualize the conflict-adjusted spatial weights as a network.
        
        Creates a network visualization showing how conflict affects
        connectivity between markets, with edge weights reflecting
        the strength of market linkages.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
        figsize : tuple, optional
            Figure size as (width, height)
        title : str, optional
            Figure title
        node_color_col : str, optional
            Column to use for node colors
        base_node_size : float, optional
            Base size for nodes
        edge_scale : float, optional
            Scaling factor for edge widths
        labels : bool, optional
            Whether to show node labels
            
        Returns
        -------
        matplotlib.figure.Figure
            Network visualization figure
            
        Notes
        -----
        This visualization reveals how conflict creates effective barriers
        between markets that may be geographically close but economically
        disconnected due to conflict.
        """
        if 'original_weights' not in self.diagnostic_hooks or self.weights is None:
            raise ValueError("No weights available. Call create_weight_matrix with store_diagnostic_info=True first.")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Create networkx graph from original weights
        G_orig = nx.Graph()
        
        # Add nodes
        for i in range(len(self.gdf)):
            G_orig.add_node(i, pos=(self.gdf.iloc[i].geometry.x, self.gdf.iloc[i].geometry.y))
        
        # Add edges with weights
        orig_weights = self.diagnostic_hooks['original_weights']
        for i, neighbors in orig_weights.neighbors.items():
            for j, neighbor in enumerate(neighbors):
                weight = orig_weights.weights[i][j]
                G_orig.add_edge(i, neighbor, weight=weight)
        
        # Create networkx graph from conflict-adjusted weights
        G_adj = nx.Graph()
        
        # Add nodes
        for i in range(len(self.gdf)):
            G_adj.add_node(i, pos=(self.gdf.iloc[i].geometry.x, self.gdf.iloc[i].geometry.y))
        
        # Add edges with weights
        for i, neighbors in self.weights.neighbors.items():
            for j, neighbor in enumerate(neighbors):
                weight = self.weights.weights[i][j]
                G_adj.add_edge(i, neighbor, weight=weight)
        
        # Get node positions
        pos = nx.get_node_attributes(G_orig, 'pos')
        
        # Determine node colors
        if node_color_col and node_color_col in self.gdf.columns:
            node_colors = self.gdf[node_color_col].values
            vmin = min(node_colors)
            vmax = max(node_colors)
        else:
            node_colors = 'skyblue'
            vmin = None
            vmax = None
        
        # Draw original network
        ax1.set_title("Geographic Connectivity")
        
        # Draw edges with weights as width
        edge_weights_orig = [G_orig[u][v]['weight'] * edge_scale for u, v in G_orig.edges()]
        nx.draw_networkx_edges(G_orig, pos, width=edge_weights_orig, edge_color='gray', alpha=0.6, ax=ax1)
        
        # Draw nodes
        nx.draw_networkx_nodes(G_orig, pos, node_size=base_node_size, node_color=node_colors, 
                             cmap='viridis', vmin=vmin, vmax=vmax, ax=ax1)
        
        # Draw labels if requested
        if labels:
            nx.draw_networkx_labels(G_orig, pos, font_size=8, ax=ax1)
        
        # Draw conflict-adjusted network
        ax2.set_title("Conflict-Adjusted Connectivity")
        
        # Draw edges with weights as width
        edge_weights_adj = [G_adj[u][v]['weight'] * edge_scale for u, v in G_adj.edges()]
        nx.draw_networkx_edges(G_adj, pos, width=edge_weights_adj, edge_color='gray', alpha=0.6, ax=ax2)
        
        # Draw nodes
        nx.draw_networkx_nodes(G_adj, pos, node_size=base_node_size, node_color=node_colors, 
                             cmap='viridis', vmin=vmin, vmax=vmax, ax=ax2)
        
        # Draw labels if requested
        if labels:
            nx.draw_networkx_labels(G_adj, pos, font_size=8, ax=ax2)
        
        # Add colorbar if using node colors
        if node_color_col and node_color_col in self.gdf.columns:
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.05, aspect=40)
            cbar.set_label(node_color_col)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weights visualization to {save_path}")
        
        return fig
    
    def _validate_model_columns(self, columns: List[str]) -> None:
        """Validate that columns exist in the GeoDataFrame."""
        missing_cols = [col for col in columns if col not in self.gdf.columns]
        if missing_cols:
            raise ValueError(f"Column(s) not found in GeoDataFrame: {', '.join(missing_cols)}")
        
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def prepare_simulation_data(self) -> Dict[str, Any]:
        """
        Prepare model results for use in simulation module.
        
        Creates a standardized data structure for simulation of policy counterfactuals,
        such as conflict reduction or transportation network improvements.
        
        Returns
        -------
        dict
            Simulation-ready data structure containing:
            - weights_matrix: Spatial weights with conflict adjustment
            - original_weights: Weights without conflict adjustment
            - gdf: Market data with spatial information
            - diagnostic_hooks: Model diagnostics and parameters
            - model_type: Model identifier for simulation module
            
        Notes
        -----
        This standardized output format facilitates integration with the
        simulation module for policy counterfactual analysis.
        """
        if self.weights is None:
            raise ValueError("Weights matrix not created. Call create_weight_matrix first.")
        
        simulation_data = {
            'weights_matrix': self.weights,
            'original_weights': self.diagnostic_hooks.get('original_weights'),
            'gdf': self.gdf,
            'diagnostic_hooks': self.diagnostic_hooks,
            'model_type': 'spatial_econometrics'
        }
        
        return simulation_data
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def calculate_spatial_barriers(
        self,
        conflict_col: str,
        threshold: float = 0.5,
        return_gdf: bool = True
    ) -> Union[gpd.GeoDataFrame, Dict[str, Any]]:
        """
        Identify and quantify critical spatial barriers between markets.
        
        Parameters
        ----------
        conflict_col : str
            Column with conflict intensity data
        threshold : float, optional
            Threshold for identifying critical barriers
        return_gdf : bool, optional
            Whether to return a GeoDataFrame (True) or dict (False)
            
        Returns
        -------
        Union[geopandas.GeoDataFrame, Dict[str, Any]]
            Market data with barrier metrics or dictionary with barrier information
            
        Notes
        -----
        Critical barriers represent connections with high conflict intensity
        that significantly impede price transmission between markets.
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Check if conflict column exists
        if conflict_col not in self.gdf.columns:
            raise ValueError(f"Conflict column '{conflict_col}' not found in GeoDataFrame")
        
        # Create copy of GeoDataFrame to add results
        result_gdf = self.gdf.copy()
        
        # Identify critical barriers (connections with high conflict)
        barriers = []
        barrier_intensities = []
        
        # For each market, count connections above conflict threshold
        for i in range(len(result_gdf)):
            if i not in self.weights.neighbors:
                continue
                
            neighbors = self.weights.neighbors[i]
            barriers_count = 0
            total_barrier_intensity = 0
            
            for j in neighbors:
                # Get conflict intensity for both regions
                conflict_i = result_gdf.iloc[i][conflict_col]
                conflict_j = result_gdf.iloc[j][conflict_col]
                
                # Average conflict intensity along the path
                avg_conflict = (conflict_i + conflict_j) / 2
                
                # Count as barrier if above threshold
                if avg_conflict > threshold:
                    barriers_count += 1
                    total_barrier_intensity += avg_conflict
            
            barriers.append(barriers_count)
            if barriers_count > 0:
                barrier_intensities.append(total_barrier_intensity / barriers_count)
            else:
                barrier_intensities.append(0)
        
        # Add barrier metrics to GeoDataFrame
        result_gdf['barrier_count'] = barriers
        result_gdf['barrier_intensity'] = barrier_intensities
        result_gdf['barrier_isolation'] = result_gdf['barrier_count'] / result_gdf['barrier_count'].max() if result_gdf['barrier_count'].max() > 0 else 0
        
        # Calculate market-wide barrier metrics
        total_connections = sum(len(neighbors) for neighbors in self.weights.neighbors.values())
        total_barriers = sum(barriers)
        barrier_rate = total_barriers / total_connections if total_connections > 0 else 0
        
        barrier_metrics = {
            'total_barriers': total_barriers,
            'total_connections': total_connections,
            'barrier_rate': barrier_rate,
            'high_barrier_markets': sum(1 for count in barriers if count > 0),
            'barrier_threshold': threshold,
            'barrier_gdf': result_gdf
        }
        
        logger.info(
            f"Identified {total_barriers} barriers out of {total_connections} connections "
            f"({barrier_rate:.1%}) using threshold {threshold}"
        )
        
        if return_gdf:
            return result_gdf
        else:
            return barrier_metrics


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
    
    This index represents how easily markets can be reached by population centers,
    accounting for distance and population size.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Markets with locations
    population_gdf : geopandas.GeoDataFrame
        Population centers with locations
    max_distance : float, optional
        Maximum distance in meters to consider
    distance_decay : float, optional
        Distance decay exponent (higher values give more weight to nearby population)
    weight_col : str, optional
        Column in population_gdf to use as weight (e.g., 'population')
        
    Returns
    -------
    geopandas.GeoDataFrame
        Market GeoDataFrame with accessibility index
        
    Notes
    -----
    In Yemen's context, accessibility reflects the potential market
    demand accounting for population distribution. Higher values indicate
    markets with better access to population centers.
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
    
    # Add normalized accessibility for easier comparison
    max_access = result_gdf['accessibility_index'].max()
    if max_access > 0:
        result_gdf['accessibility_normalized'] = result_gdf['accessibility_index'] / max_access
    else:
        result_gdf['accessibility_normalized'] = 0
    
    logger.info(f"Calculated accessibility index for {len(markets_gdf)} markets")
    return result_gdf


@timer
@m1_optimized(parallel=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def calculate_market_isolation(
    markets_gdf: gpd.GeoDataFrame,
    transport_network_gdf: Optional[gpd.GeoDataFrame] = None,
    population_gdf: Optional[gpd.GeoDataFrame] = None,
    conflict_col: Optional[str] = None,
    max_distance: float = 50000,  # 50 km
    adjustment_method: str = 'linear'
) -> gpd.GeoDataFrame:
    """
    Calculate market isolation index based on multiple factors.
    
    This comprehensive isolation index combines distance to other markets,
    transport infrastructure, population centers, and conflict barriers.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market locations
    transport_network_gdf : geopandas.GeoDataFrame, optional
        Transportation network
    population_gdf : geopandas.GeoDataFrame, optional
        Population centers
    conflict_col : str, optional
        Column with conflict intensity
    max_distance : float, optional
        Maximum distance in meters to consider
    adjustment_method : str, optional
        Method for conflict adjustment ('linear', 'exponential', 'threshold')
        
    Returns
    -------
    geopandas.GeoDataFrame
        Markets with isolation index
        
    Notes
    -----
    In Yemen's context, market isolation reflects barriers to price transmission
    due to geographic distance, lack of infrastructure, and conflict. Higher 
    values indicate markets that are more isolated from the broader market system.
    """
    # Validate inputs
    valid, errors = validate_geodataframe(
        markets_gdf,
        required_columns=[col for col in [conflict_col] if col],
        min_rows=1,
        check_crs=True
    )
    raise_if_invalid(valid, errors, "Invalid market data for isolation analysis")
    
    # Make a copy of markets GeoDataFrame
    result_gdf = markets_gdf.copy()
    
    # Initialize isolation components
    components = []
    
    # 1. Distance to transportation network
    if transport_network_gdf is not None:
        # Ensure same CRS
        if markets_gdf.crs != transport_network_gdf.crs:
            transport_network_gdf = transport_network_gdf.to_crs(markets_gdf.crs)
        
        result_gdf['dist_to_transport'] = float('inf')
        
        for i, market in enumerate(result_gdf.geometry):
            min_distance = float('inf')
            
            for line in transport_network_gdf.geometry:
                distance = market.distance(line)
                min_distance = min(min_distance, distance)
                
            result_gdf.loc[i, 'dist_to_transport'] = min_distance
        
        # Normalize distances
        max_dist = result_gdf['dist_to_transport'].max()
        if max_dist > 0:
            result_gdf['transport_isolation'] = result_gdf['dist_to_transport'] / max_dist
        else:
            result_gdf['transport_isolation'] = 0
        
        components.append('transport_isolation')
    
    # 2. Distance to population centers
    if population_gdf is not None:
        # Ensure same CRS
        if markets_gdf.crs != population_gdf.crs:
            population_gdf = population_gdf.to_crs(markets_gdf.crs)
        
        # Find nearest population center
        nearest_pop = find_nearest_points(result_gdf, population_gdf)
        result_gdf['dist_to_population'] = nearest_pop['distance']
        
        # Normalize distances
        max_dist = result_gdf['dist_to_population'].max()
        if max_dist > 0:
            result_gdf['population_isolation'] = result_gdf['dist_to_population'] / max_dist
        else:
            result_gdf['population_isolation'] = 0
        
        components.append('population_isolation')
    
    # 3. Conflict barrier - enhanced with different adjustment methods
    if conflict_col and conflict_col in result_gdf.columns:
        # Apply different adjustment methods
        if adjustment_method == 'linear':
            # Normalize conflict intensity
            conflict = result_gdf[conflict_col].values
            if conflict.min() != conflict.max():
                result_gdf['conflict_isolation'] = (conflict - conflict.min()) / (conflict.max() - conflict.min())
            else:
                result_gdf['conflict_isolation'] = 0
        elif adjustment_method == 'exponential':
            # Exponential impact of conflict (higher sensitivity to conflict)
            conflict = result_gdf[conflict_col].values
            if conflict.min() != conflict.max():
                normalized = (conflict - conflict.min()) / (conflict.max() - conflict.min())
                result_gdf['conflict_isolation'] = 1 - np.exp(-3 * normalized)
            else:
                result_gdf['conflict_isolation'] = 0
        elif adjustment_method == 'threshold':
            # Threshold-based impact (binary impact above threshold)
            conflict = result_gdf[conflict_col].values
            threshold = 0.5 if isinstance(conflict.max(), (int, float)) and conflict.max() > 0 else 0
            result_gdf['conflict_isolation'] = (conflict > threshold).astype(float)
        else:
            # Default to linear
            conflict = result_gdf[conflict_col].values
            if conflict.min() != conflict.max():
                result_gdf['conflict_isolation'] = (conflict - conflict.min()) / (conflict.max() - conflict.min())
            else:
                result_gdf['conflict_isolation'] = 0
        
        components.append('conflict_isolation')
    
    # 4. Calculate distance to nearest markets
    # Calculate pairwise distances between markets
    market_distances = calculate_distances(
        result_gdf, 
        result_gdf, 
        id_col=result_gdf.index.name or 'index'
    )
    
    # For each market, calculate isolation from other markets
    for i, market in result_gdf.iterrows():
        # Get distances to other markets (excluding self)
        market_distances_subset = market_distances[
            (market_distances.origin_id != market_distances.dest_id) & 
            (market_distances.origin_id == i) & 
            (market_distances.distance <= max_distance)
        ]
        
        # Count nearby markets and calculate average distance
        n_nearby = len(market_distances_subset)
        if n_nearby == 0:
            avg_distance = max_distance
        else:
            avg_distance = market_distances_subset.distance.mean()
        
        # Base isolation score: higher when fewer nearby markets or greater distances
        base_isolation = 1 - (n_nearby / len(result_gdf)) * (1 - avg_distance / max_distance)
        
        # Store in result
        result_gdf.loc[i, 'market_distance_isolation'] = base_isolation
    
    components.append('market_distance_isolation')
    
    # Calculate overall isolation index as average of components
    if components:
        result_gdf['isolation_index'] = result_gdf[components].mean(axis=1)
    else:
        result_gdf['isolation_index'] = 0
    
    logger.info(f"Calculated isolation index for {len(markets_gdf)} markets")
    return result_gdf


@timer
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def find_nearest_points(
    origins_gdf: gpd.GeoDataFrame,
    destinations_gdf: gpd.GeoDataFrame,
    target_col: Optional[str] = None,
    max_distance: Optional[float] = None
) -> pd.DataFrame:
    """
    For each origin point, find the nearest destination point.
    
    Parameters
    ----------
    origins_gdf : geopandas.GeoDataFrame
        GeoDataFrame with origin points
    destinations_gdf : geopandas.GeoDataFrame
        GeoDataFrame with destination points
    target_col : str, optional
        Column from destinations_gdf to include in results
    max_distance : float, optional
        Maximum search distance
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with nearest point information
    """
    # Validate inputs
    valid, errors = validate_geodataframe(
        origins_gdf, 
        geometry_type="Point", 
        min_rows=1,
        check_crs=True
    )
    raise_if_invalid(valid, errors, "Invalid origins GeoDataFrame")
    
    valid, errors = validate_geodataframe(
        destinations_gdf,
        required_columns=[target_col] if target_col else None,
        geometry_type="Point", 
        min_rows=1,
        check_crs=True
    )
    raise_if_invalid(valid, errors, "Invalid destinations GeoDataFrame")
    
    # Ensure same CRS
    if origins_gdf.crs != destinations_gdf.crs:
        destinations_gdf = destinations_gdf.to_crs(origins_gdf.crs)
    
    results = []
    
    # Process each origin
    for idx, origin in origins_gdf.iterrows():
        origin_geom = origin.geometry
        
        # Calculate distances to all destinations
        distances = [origin_geom.distance(dest.geometry) for _, dest in destinations_gdf.iterrows()]
        
        # Find minimum distance
        min_distance = min(distances)
        min_idx = distances.index(min_distance)
        
        # Skip if beyond max_distance
        if max_distance is not None and min_distance > max_distance:
            continue
        
        # Get destination information
        dest = destinations_gdf.iloc[min_idx]
        
        # Create result record
        result = {
            'origin_id': idx,
            'dest_id': dest.name,
            'distance': min_distance
        }
        
        # Add target column if specified
        if target_col and target_col in dest:
            result[target_col] = dest[target_col]
        
        results.append(result)
    
    return pd.DataFrame(results)


@timer
@m1_optimized(parallel=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def simulate_improved_connectivity(
    markets_gdf: gpd.GeoDataFrame,
    conflict_reduction: float,
    conflict_col: str = 'conflict_intensity_normalized',
    price_col: str = 'price',
    spatial_model: Optional[SpatialEconometrics] = None,
    return_full_results: bool = False
) -> Dict[str, Any]:
    """
    Simulate improved market connectivity by reducing conflict barriers.
    
    This function creates a counterfactual scenario where conflict barriers
    are reduced, and estimates the impact on market integration and prices.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data with prices, locations, and conflict intensity
    conflict_reduction : float
        Percentage reduction in conflict (0-1)
    conflict_col : str, optional
        Column with conflict intensity
    price_col : str, optional
        Column with price data
    spatial_model : SpatialEconometrics, optional
        Pre-fitted spatial model
    return_full_results : bool, optional
        Whether to return full model objects
        
    Returns
    -------
    dict
        Simulation results including:
        - scenario: Description of simulation scenario
        - metrics: Changes in market integration metrics
        - models: Spatial model results before and after
        - visualizations: Optional visualization figures
        
    Notes
    -----
    This simulation enables policy analysis by quantifying how reduced
    conflict barriers would affect market integration. It models both
    the direct effects through spatial weights and the resulting changes
    in price transmission.
    """
    # Validate inputs
    if not isinstance(markets_gdf, gpd.GeoDataFrame):
        raise ValidationError("markets_gdf must be a GeoDataFrame")
    
    if not 0 <= conflict_reduction <= 1:
        raise ValidationError(f"conflict_reduction must be between 0 and 1, got {conflict_reduction}")
    
    if conflict_col not in markets_gdf.columns:
        raise ValidationError(f"Conflict column '{conflict_col}' not found in GeoDataFrame")
    
    if price_col not in markets_gdf.columns:
        raise ValidationError(f"Price column '{price_col}' not found in GeoDataFrame")
    
    # Initialize results dictionary
    results = {
        'scenario': f"{conflict_reduction*100:.0f}% Conflict Reduction",
        'reduction_factor': conflict_reduction,
        'metrics': {},
        'models': {}
    }
    
    # Create a copy of the data for the simulation
    sim_data = markets_gdf.copy()
    
    # Apply conflict reduction
    sim_data[f'{conflict_col}_reduced'] = sim_data[conflict_col] * (1 - conflict_reduction)
    
    # Create spatial model for original data if not provided
    if spatial_model is None:
        original_model = SpatialEconometrics(markets_gdf)
    else:
        original_model = spatial_model
    
    # Create spatial model for simulation data
    sim_model = SpatialEconometrics(sim_data)
    
    # Create weight matrices - original
    original_weights = original_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=conflict_col
    )
    
    # Create weight matrices - with reduced conflict
    sim_weights = sim_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=f'{conflict_col}_reduced'
    )
    
    # Test for spatial autocorrelation in prices - original
    original_moran = original_model.moran_i_test(price_col)
    
    # Test for spatial autocorrelation in prices - simulation
    sim_moran = sim_model.moran_i_test(price_col)  # Same prices, different weights
    
    # Store Moran's I results
    results['metrics']['moran_I_original'] = original_moran['I']
    results['metrics']['moran_I_simulated'] = sim_moran['I']
    results['metrics']['moran_I_change'] = sim_moran['I'] - original_moran['I']
    results['metrics']['moran_pvalue_original'] = original_moran['p_norm']
    results['metrics']['moran_pvalue_simulated'] = sim_moran['p_norm']
    
    # Check if market data contains exchange rate regimes
    if 'exchange_rate_regime' in markets_gdf.columns:
        # Create dummy for exchange rate regime
        if 'exchange_rate_regime' in sim_data.columns:
            sim_data['regime_north'] = (sim_data['exchange_rate_regime'] == 'north').astype(int)
        
        # Define regression variables
        y_col = price_col
        x_cols = [conflict_col, 'regime_north']
        
        # Estimate spatial models - original
        try:
            original_lag = original_model.spatial_lag_model(y_col=y_col, x_cols=x_cols)
            results['models']['original_lag'] = original_lag if return_full_results else {
                'rho': original_lag.rho,
                'betas': original_lag.betas.tolist(),
                'r2': original_lag.pr2,
                'aic': original_lag.aic
            }
        except Exception as e:
            logger.warning(f"Could not estimate original spatial lag model: {e}")
        
        # Estimate spatial models - simulated
        try:
            x_cols_sim = [f'{conflict_col}_reduced', 'regime_north']
            sim_lag = sim_model.spatial_lag_model(y_col=y_col, x_cols=x_cols_sim)
            results['models']['simulated_lag'] = sim_lag if return_full_results else {
                'rho': sim_lag.rho,
                'betas': sim_lag.betas.tolist(),
                'r2': sim_lag.pr2,
                'aic': sim_lag.aic
            }
        except Exception as e:
            logger.warning(f"Could not estimate simulated spatial lag model: {e}")
    
    # Calculate price dispersion metrics
    if 'exchange_rate_regime' in markets_gdf.columns:
        # Calculate price differentials between north and south
        north_original = markets_gdf[markets_gdf['exchange_rate_regime'] == 'north']
        south_original = markets_gdf[markets_gdf['exchange_rate_regime'] == 'south']
        
        if len(north_original) > 0 and len(south_original) > 0:
            north_price = north_original[price_col].mean()
            south_price = south_original[price_col].mean()
            price_diff_original = abs(north_price - south_price)
            price_diff_pct_original = price_diff_original / ((north_price + south_price) / 2) * 100
            
            # Calculate expected price differentials after conflict reduction
            # This is a simplified model - in reality would need a more sophisticated approach
            # based on estimated effect of conflict on price transmission
            if 'models' in results and 'original_lag' in results['models'] and 'simulated_lag' in results['models']:
                original_rho = results['models']['original_lag']['rho']
                sim_rho = results['models']['simulated_lag']['rho']
                rho_improvement = (sim_rho - original_rho) / original_rho if original_rho != 0 else 0
                
                # A higher rho means stronger spatial price transmission
                # Estimate reduced price differential using rho improvement
                price_diff_simulated = price_diff_original * (1 - rho_improvement * conflict_reduction)
                price_diff_pct_simulated = price_diff_pct_original * (1 - rho_improvement * conflict_reduction)
                
                results['metrics']['price_diff_original'] = price_diff_original
                results['metrics']['price_diff_simulated'] = price_diff_simulated
                results['metrics']['price_diff_pct_original'] = price_diff_pct_original
                results['metrics']['price_diff_pct_simulated'] = price_diff_pct_simulated
                results['metrics']['price_convergence_pct'] = (
                    (price_diff_original - price_diff_simulated) / price_diff_original * 100
                    if price_diff_original > 0 else 0
                )
    
    # Create visualizations for the simulation
    try:
        import matplotlib.pyplot as plt
        
        # Create visualization of conflict reduction impact
        figs = {}
        
        # Plot Moran's I comparison
        fig_moran, ax_moran = plt.subplots(figsize=(8, 6))
        moran_values = [results['metrics']['moran_I_original'], results['metrics']['moran_I_simulated']]
        ax_moran.bar(['Original', 'Reduced Conflict'], moran_values, color=['blue', 'green'])
        ax_moran.set_title("Impact of Conflict Reduction on Spatial Autocorrelation")
        ax_moran.set_ylabel("Moran's I")
        ax_moran.grid(True, alpha=0.3)
        figs['moran_comparison'] = fig_moran
        
        # If price differentials were calculated
        if 'price_diff_original' in results['metrics']:
            # Plot price differential comparison
            fig_price, ax_price = plt.subplots(figsize=(8, 6))
            price_diff_values = [
                results['metrics']['price_diff_original'],
                results['metrics']['price_diff_simulated']
            ]
            price_labels = ['Original', f"{conflict_reduction*100:.0f}% Conflict Reduction"]
            ax_price.bar(price_labels, price_diff_values, color=['red', 'green'])
            ax_price.set_title("Impact of Conflict Reduction on Price Differentials")
            ax_price.set_ylabel("Price Differential")
            ax_price.grid(True, alpha=0.3)
            figs['price_diff_comparison'] = fig_price
        
        results['visualizations'] = figs
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
    
    logger.info(
        f"Simulated {conflict_reduction*100:.0f}% conflict reduction scenario. "
        f"Moran's I change: {results['metrics']['moran_I_change']:.4f}, "
        f"Price convergence: {results['metrics'].get('price_convergence_pct', 'N/A')}"
    )
    
    return results


@timer
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, TypeError))
def market_integration_index(
    prices_df: pd.DataFrame,
    weights_matrix: W,
    market_id_col: str,
    price_col: str = 'price',
    time_col: str = 'date',
    windows: Optional[List[int]] = None,
    return_components: bool = False
) -> pd.DataFrame:
    """
    Calculate time-varying market integration metrics accounting for
    spatial autocorrelation in prices.
    
    Parameters
    ----------
    prices_df : pandas.DataFrame
        DataFrame with market prices over time
    weights_matrix : libpysal.weights.W
        Spatial weights matrix
    market_id_col : str
        Column identifying markets
    price_col : str, optional
        Column containing price data
    time_col : str, optional
        Column containing time data
    windows : list, optional
        List of window sizes for rolling calculations
        Default: [1, 3, 6, 12]
    return_components : bool, optional
        Whether to return component metrics
        
    Returns
    -------
    pandas.DataFrame
        Time series of integration indices
        
    Notes
    -----
    This index combines price dispersion and spatial autocorrelation
    to create a comprehensive measure of market integration over time.
    Higher values indicate stronger integration, with prices moving
    together across space and exhibiting lower dispersion.
    """
    # Validate inputs
    if not isinstance(prices_df, pd.DataFrame):
        raise ValidationError("prices_df must be a pandas DataFrame")
    
    if not isinstance(weights_matrix, W):
        raise ValidationError("weights_matrix must be a libpysal.weights.W object")
    
    valid, errors = validate_dataframe(
        prices_df, 
        required_columns=[market_id_col, price_col, time_col],
        min_rows=10
    )
    raise_if_invalid(valid, errors, "Invalid price data for integration analysis")
    
    # Set default windows if not provided
    if windows is None:
        windows = [1, 3, 6, 12]
    
    # Ensure time column is datetime
    prices_df[time_col] = pd.to_datetime(prices_df[time_col])
    
    # Get unique time periods and markets
    time_periods = sorted(prices_df[time_col].unique())
    markets = sorted(prices_df[market_id_col].unique())
    
    # Check if we have enough observations for time series analysis
    if len(time_periods) < max(windows) + 1:
        raise ValidationError(
            f"Not enough time periods ({len(time_periods)}) for window analysis with max window {max(windows)}"
        )
    
    # Initialize results
    results = []
    
    # For each time period, calculate integration metrics
    for t, period in enumerate(time_periods):
        # Skip periods that don't have enough history for all windows
        if t < max(windows):
            continue
        
        # Get data for current period
        period_data = prices_df[prices_df[time_col] == period]
        
        # Need data for all markets to calculate integration metrics
        if len(period_data) < len(markets):
            logger.warning(f"Skipping period {period} with incomplete market data")
            continue
        
        # Create a price vector for all markets
        price_vector = period_data.set_index(market_id_col)[price_col]
        
        # Calculate integration metrics for this period
        period_metrics = {
            time_col: period,
            'num_markets': len(period_data)
        }
        
        # Calculate coefficient of variation (price dispersion)
        period_metrics['price_cv'] = price_vector.std() / price_vector.mean() if price_vector.mean() > 0 else 0
        
        # Calculate Moran's I for spatial price autocorrelation
        try:
            moran = Moran(price_vector, weights_matrix)
            period_metrics['moran_I'] = moran.I
            period_metrics['moran_pvalue'] = moran.p_norm
            period_metrics['significant_autocorrelation'] = moran.p_norm < 0.05
        except Exception as e:
            logger.warning(f"Could not calculate Moran's I for period {period}: {e}")
            period_metrics['moran_I'] = None
            period_metrics['moran_pvalue'] = None
            period_metrics['significant_autocorrelation'] = False
        
        # Calculate market integration metrics for different window sizes
        for window in windows:
            # Get data for window periods ago
            past_period = time_periods[t - window]
            past_data = prices_df[prices_df[time_col] == past_period]
            
            # Skip if we don't have complete data
            if len(past_data) < len(markets):
                continue
            
            # Get price vector for past period
            past_price_vector = past_data.set_index(market_id_col)[price_col]
            
            # Ensure we have matched markets
            common_markets = set(price_vector.index) & set(past_price_vector.index)
            if len(common_markets) < 2:
                continue
                
            # Calculate price convergence metrics
            current_prices = price_vector.loc[common_markets]
            past_prices = past_price_vector.loc[common_markets]
            
            # Price convergence: reduction in coefficient of variation
            current_cv = current_prices.std() / current_prices.mean() if current_prices.mean() > 0 else 0
            past_cv = past_prices.std() / past_prices.mean() if past_prices.mean() > 0 else 0
            cv_change = (current_cv - past_cv) / past_cv if past_cv > 0 else 0
            
            # Calculate average price change
            price_changes = (current_prices - past_prices) / past_prices
            avg_price_change = price_changes.mean()
            price_change_cv = price_changes.std() / abs(avg_price_change) if avg_price_change != 0 else float('inf')
            
            # Store window-specific metrics
            period_metrics[f'price_cv_change_w{window}'] = cv_change
            period_metrics[f'avg_price_change_w{window}'] = avg_price_change
            period_metrics[f'price_change_cv_w{window}'] = price_change_cv
            
            # Calculate Market Integration Index for this window
            # Higher values = better integration (lower dispersion, more spatial correlation)
            # Components:
            # 1. Low coefficient of variation (normalized)
            # 2. High spatial autocorrelation (if positive)
            # 3. Low price change dispersion (normalized)
            
            # Normalize CV (lower is better, range 0-1)
            norm_cv = max(0, 1 - min(current_cv * 5, 1))  # Cap at 0-1 range
            
            # Normalize Moran's I (higher positive values are better)
            norm_moran = max(0, min(1, (period_metrics['moran_I'] + 1) / 2)) if period_metrics['moran_I'] is not None else 0.5
            
            # Normalize price change dispersion (lower is better)
            norm_price_change = max(0, 1 - min(price_change_cv * 0.2, 1))  # Cap at 0-1 range
            
            # Combined index (simple average of components)
            integration_index = (norm_cv + norm_moran + norm_price_change) / 3
            period_metrics[f'integration_index_w{window}'] = integration_index
            
            # Store components if requested
            if return_components:
                period_metrics[f'integration_cv_component_w{window}'] = norm_cv
                period_metrics[f'integration_moran_component_w{window}'] = norm_moran
                period_metrics[f'integration_price_change_component_w{window}'] = norm_price_change
        
        # Add to results
        results.append(period_metrics)
    
    # Convert to DataFrame and sort by time
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(time_col)
    
    logger.info(f"Calculated market integration index for {len(results_df)} time periods")
    return results_df