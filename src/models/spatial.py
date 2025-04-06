"""
Spatial econometric models for market integration analysis in Yemen.
Implements statistical methods for analyzing geographic market relationships
in conflict-affected environments.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple, Generator
from libpysal.weights import W
from esda.moran import Moran, Moran_Local
from spreg import ML_Lag, ML_Error
import matplotlib.pyplot as plt
import networkx as nx
import gc
import tempfile
from pathlib import Path
import weakref
import os

from yemen_market_integration.utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_geodataframe, validate_dataframe, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator,
    
    # Spatial utilities
    reproject_gdf, calculate_distances, create_buffer, find_nearest_points,
    create_spatial_weight_matrix, create_conflict_adjusted_weights,
    calculate_market_isolation,
    
    # Data processing
    clean_column_names, normalize_columns, fill_missing_values,
    
    # Configuration
    config,
    
    # Plotting
    plot_yemen_market_integration
)

# Import new memory optimization utilities
from src.utils.m3_utils import (
    m3_optimized, chunk_iterator, process_in_chunks, 
    optimize_array_computation, create_mmap_array,
    memory_profile, tiered_cache, chunk_iterator
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values
DEFAULT_CONFLICT_WEIGHT = config.get('analysis.spatial.conflict_weight', 0.5)
DEFAULT_KNN = config.get('analysis.spatial.knn', 5)
DEFAULT_CRS = config.get('analysis.spatial.crs', 32638)  # UTM Zone 38N for Yemen

# Directory for cached spatial data
SPATIAL_CACHE_DIR = Path(tempfile.gettempdir()) / "yemen_market_spatial_cache"
os.makedirs(SPATIAL_CACHE_DIR, exist_ok=True)


class SpatialEconometrics:
    """
    Spatial econometric analysis for market integration.
    
    Implements spatial econometric methods for analyzing geographic 
    relationships in Yemen's market data, with adaptations for conflict-affected
    environments where traditional distance metrics may be misleading.
    """
    
    def __init__(self, gdf):
        """
        Initialize with a GeoDataFrame.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data with market locations
        """
        # Validate input
        self._validate_input(gdf)
        
        # Convert data types to more memory-efficient ones
        self.gdf = self._optimize_gdf(gdf)
        self.weights = None
        self.diagnostic_hooks = {}
        self.lag_model = None
        self.error_model = None
        
        # Set up caches for memory-intensive operations
        self._setup_caches()
        logger.info(f"Initialized SpatialEconometrics with {len(gdf)} observations")
    
    def _optimize_gdf(self, gdf):
        """
        Optimize GeoDataFrame for memory efficiency.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
        
        Returns
        -------
        geopandas.GeoDataFrame
            Memory-optimized GeoDataFrame
        """
        # Create a copy to avoid modifying the original
        optimized_gdf = gdf.copy()
        
        # Downcast numeric columns to save memory
        for col in optimized_gdf.columns:
            # Skip geometry column
            if col == 'geometry':
                continue
                
            col_dtype = optimized_gdf[col].dtype
            
            # Handle integer columns
            if pd.api.types.is_integer_dtype(col_dtype):
                # Check value range and downcast if possible
                col_min = optimized_gdf[col].min()
                col_max = optimized_gdf[col].max()
                
                if col_min >= 0:  # Unsigned integers
                    if col_max <= 255:
                        optimized_gdf[col] = optimized_gdf[col].astype(np.uint8)
                    elif col_max <= 65535:
                        optimized_gdf[col] = optimized_gdf[col].astype(np.uint16)
                    elif col_max <= 4294967295:
                        optimized_gdf[col] = optimized_gdf[col].astype(np.uint32)
                else:  # Signed integers
                    if col_min >= -128 and col_max <= 127:
                        optimized_gdf[col] = optimized_gdf[col].astype(np.int8)
                    elif col_min >= -32768 and col_max <= 32767:
                        optimized_gdf[col] = optimized_gdf[col].astype(np.int16)
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        optimized_gdf[col] = optimized_gdf[col].astype(np.int32)
            
            # Handle float columns
            elif pd.api.types.is_float_dtype(col_dtype):
                # Check if float32 precision is sufficient
                # This can reduce memory usage by half for float columns
                optimized_gdf[col] = optimized_gdf[col].astype(np.float32)
        
        return optimized_gdf
    
    def _setup_caches(self):
        """Setup caches for memory-intensive operations."""
        # Create different caches for different operation types
        
        # Cache for weight matrix calculations
        self._weight_cache = tiered_cache(
            maxsize=10,
            disk_cache_dir=str(SPATIAL_CACHE_DIR / "weights"),
            memory_limit_mb=500
        )(lambda x: x)  # Simple identity function as base
        
        # Cache for Moran's I calculations
        self._moran_cache = tiered_cache(
            maxsize=20,
            disk_cache_dir=str(SPATIAL_CACHE_DIR / "moran"),
            memory_limit_mb=200
        )(lambda x: x)
    
    @handle_errors(logger=logger, error_type=(ValidationError, TypeError), reraise=True)
    def _validate_input(self, gdf):
        """Validate input GeoDataFrame."""
        # Check if GeoDataFrame
        if not isinstance(gdf, pd.DataFrame) or not hasattr(gdf, 'geometry'):
            raise ValidationError("Input must be a GeoDataFrame")
        
        # Validate with utility function
        valid, errors = validate_geodataframe(
            gdf,
            min_rows=1,
            check_crs=True
        )
        raise_if_invalid(valid, errors, "Invalid GeoDataFrame for spatial analysis")
    
    @timer
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def create_weight_matrix(
        self, 
        k=DEFAULT_KNN, 
        conflict_adjusted=True, 
        conflict_col='conflict_intensity_normalized',
        conflict_weight=DEFAULT_CONFLICT_WEIGHT
    ):
        """
        Create spatial weights matrix with optional conflict adjustment.
        
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
        
        # Generate cache key
        cache_key = f"weights_k{k}_ca{int(conflict_adjusted)}_cw{conflict_weight}"
        if conflict_adjusted:
            cache_key += f"_col{conflict_col}"
        
        # Check cache first
        cached_weights = self._weight_cache.cache.get(cache_key)
        if cached_weights is not None:
            logger.info(f"Using cached weight matrix with k={k}, conflict_adjusted={conflict_adjusted}")
            self.weights = cached_weights
            return self.weights
        
        # Use project utility to create weights
        if conflict_adjusted:
            # For large datasets, use chunked processing
            if len(self.gdf) > 1000:
                logger.info(f"Using chunked processing for conflict-adjusted weights (n={len(self.gdf)})")
                self.weights = self._create_conflict_adj_weights_chunked(
                    k=k,
                    conflict_col=conflict_col,
                    conflict_weight=conflict_weight
                )
            else:
                # For smaller datasets, use standard implementation
                self.weights = create_conflict_adjusted_weights(
                    self.gdf,
                    k=k,
                    conflict_col=conflict_col,
                    conflict_weight=conflict_weight
                )
            # Store diagnostic info
            self.diagnostic_hooks['conflict_adjusted'] = True
            self.diagnostic_hooks['conflict_col'] = conflict_col
            self.diagnostic_hooks['conflict_weight'] = conflict_weight
        else:
            self.weights = create_spatial_weight_matrix(
                self.gdf,
                method='knn',
                k=k
            )
            self.diagnostic_hooks['conflict_adjusted'] = False
        
        # Store original weights for comparison
        self.diagnostic_hooks['original_weights'] = create_spatial_weight_matrix(
            self.gdf, method='knn', k=k
        )
        self.diagnostic_hooks['k'] = k
        
        # Cache the weights for future use
        self._weight_cache.cache.set(cache_key, self.weights)
        
        logger.info(f"Created weight matrix with k={k}, conflict_adjusted={conflict_adjusted}")
        return self.weights
    
    def _create_conflict_adj_weights_chunked(self, k, conflict_col, conflict_weight):
        """
        Create conflict-adjusted weights using chunked processing for large datasets.
        
        Parameters
        ----------
        k : int
            Number of nearest neighbors
        conflict_col : str
            Column name for conflict intensity
        conflict_weight : float
            Weight to apply to conflict adjustment
            
        Returns
        -------
        libpysal.weights.W
            Conflict-adjusted weights matrix
        """
        # Create base weights matrix using KNN
        base_weights = create_spatial_weight_matrix(
            self.gdf,
            method='knn',
            k=k
        )
        
        # Process in chunks for memory efficiency
        n = len(self.gdf)
        chunk_size = min(100, max(10, n // 10))  # Adaptive chunk size
        
        # Create an empty weights dictionary
        neighbors = {}
        weights = {}
        
        # Get conflict values as array for faster access
        conflict_values = self.gdf[conflict_col].values
        
        # Process in chunks
        for i in range(0, n, chunk_size):
            end_idx = min(i + chunk_size, n)
            chunk_indices = list(range(i, end_idx))
            
            # Process each index in the chunk
            for idx in chunk_indices:
                # Get base neighbors
                idx_neighbors = base_weights.neighbors[idx]
                
                # Get conflict values
                idx_conflict = conflict_values[idx]
                neighbor_conflicts = conflict_values[idx_neighbors]
                
                # Calculate conflict-adjusted weights
                conflict_factors = 1.0 - (conflict_weight * 0.5 * (idx_conflict + neighbor_conflicts))
                conflict_factors = np.clip(conflict_factors, 0.1, 1.0)  # Ensure weights are positive
                
                # Normalize weights
                if len(conflict_factors) > 0:
                    normalized_weights = conflict_factors / conflict_factors.sum()
                else:
                    normalized_weights = conflict_factors
                
                # Store results
                neighbors[idx] = idx_neighbors
                weights[idx] = list(normalized_weights)
        
        # Create W object from processed chunks
        from libpysal.weights import W
        return W(neighbors, weights)
    
    @timer
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def moran_i_test(self, variable):
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
        
        # Generate cache key
        cache_key = f"moran_i_{variable}"
        
        # Check cache first
        cached_result = self._moran_cache.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached Moran's I result for {variable}")
            return cached_result
        
        # Calculate Moran's I
        # For large datasets, optimize the variable array
        if len(self.gdf) > 1000:
            logger.info(f"Optimizing large array for Moran's I test (n={len(self.gdf)})")
            var_array = self.gdf[variable].values
            var_array = optimize_array_computation(var_array, precision='float32')
            moran = Moran(var_array, self.weights)
        else:
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
        
        # Cache the result
        self._moran_cache.cache.set(cache_key, result)
        
        logger.info(
            f"Moran's I test for {variable}: I={result['I']:.4f}, "
            f"p={result['p_norm']:.4f}, significant={result['significant']}"
        )
        
        return result
    
    @timer
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def local_moran_test(self, variable):
        """
        Conduct Local Moran's I test to identify spatial clusters.
        
        Parameters
        ----------
        variable : str
            Column name in GeoDataFrame to test
            
        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with original data plus Local Moran results
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Check if variable exists
        if variable not in self.gdf.columns:
            raise ValueError(f"Variable '{variable}' not found in GeoDataFrame")
        
        # Generate cache key
        cache_key = f"local_moran_{variable}"
        
        # Check cache first
        cached_result = self._moran_cache.cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached Local Moran's I result for {variable}")
            return cached_result
        
        # Calculate Local Moran's I
        # For large datasets, optimize the variable array
        if len(self.gdf) > 1000:
            var_array = self.gdf[variable].values
            var_array = optimize_array_computation(var_array, precision='float32')
            local_moran = Moran_Local(var_array, self.weights)
        else:
            local_moran = Moran_Local(self.gdf[variable], self.weights)
        
        # Create copy of GeoDataFrame to add results
        result_gdf = self.gdf.copy()
        
        # Add results to GeoDataFrame
        result_gdf['moran_local_i'] = local_moran.Is
        result_gdf['moran_p_value'] = local_moran.p_sim
        result_gdf['moran_significant'] = local_moran.p_sim < 0.05
        
        # Create cluster classification
        self._classify_moran_clusters(result_gdf, variable, local_moran)
        
        # Cache the result
        self._moran_cache.cache.set(cache_key, result_gdf)
        
        logger.info(
            f"Local Moran's I test for {variable}: "
            f"Significant clusters: {result_gdf['moran_significant'].sum()}/{len(result_gdf)}"
        )
        
        return result_gdf
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _classify_moran_clusters(self, gdf, variable, local_moran):
        """
        Classify clusters based on Local Moran's I results.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame to update with cluster types
        variable : str
            Variable used for clustering
        local_moran : esda.moran.Moran_Local
            Local Moran's I results
        """
        # Use boolean masks for more memory-efficient classification
        significant = gdf['moran_p_value'] < 0.05
        high = gdf[variable] > gdf[variable].mean()
        
        # Get the spatially lagged variable
        y_lag = local_moran.y_lag
        high_neighbors = y_lag > y_lag.mean()
        
        # Classify clusters - use in-place operations
        gdf['cluster_type'] = 'not_significant'
        
        # Set values for each cluster type using boolean indexing
        gdf.loc[significant & high & high_neighbors, 'cluster_type'] = 'high-high'
        gdf.loc[significant & ~high & ~high_neighbors, 'cluster_type'] = 'low-low'
        gdf.loc[significant & high & ~high_neighbors, 'cluster_type'] = 'high-low'
        gdf.loc[significant & ~high & high_neighbors, 'cluster_type'] = 'low-high'
    
    @timer
    @memory_usage_decorator
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def spatial_lag_model(self, y_col, x_cols, name_y=None, name_x=None):
        """
        Estimate a spatial lag model.
        
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
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Validate columns
        self._validate_model_columns([y_col] + x_cols)
        
        # Prepare model data - use memory-efficient implementation
        model_data, y, X = self._prepare_model_data_efficient(y_col, x_cols)
        
        # Set default names if not provided
        if name_y is None:
            name_y = y_col
        if name_x is None:
            name_x = x_cols
        
        # Force garbage collection before model estimation
        gc.collect()
        
        # Estimate model
        logger.info(f"Estimating spatial lag model with {len(X)} observations and {X.shape[1]} variables")
        self.lag_model = ML_Lag(y, X, self.weights, name_y=name_y, name_x=name_x)
        
        logger.info(
            f"Spatial lag model estimated: AIC={self.lag_model.aic:.4f}, "
            f"R2={self.lag_model.pr2:.4f}, rho={self.lag_model.rho:.4f}"
        )
        
        return self.lag_model
    
    @timer
    @memory_usage_decorator
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def spatial_error_model(self, y_col, x_cols, name_y=None, name_x=None):
        """
        Estimate a spatial error model.
        
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
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Validate columns
        self._validate_model_columns([y_col] + x_cols)
        
        # Prepare model data - use memory-efficient implementation
        model_data, y, X = self._prepare_model_data_efficient(y_col, x_cols)
        
        # Set default names if not provided
        if name_y is None:
            name_y = y_col
        if name_x is None:
            name_x = x_cols
        
        # Force garbage collection before model estimation
        gc.collect()
        
        # Estimate model
        logger.info(f"Estimating spatial error model with {len(X)} observations and {X.shape[1]} variables")
        self.error_model = ML_Error(y, X, self.weights, name_y=name_y, name_x=name_x)
        
        logger.info(
            f"Spatial error model estimated: AIC={self.error_model.aic:.4f}, "
            f"R2={self.error_model.pr2:.4f}, lambda={self.error_model.lam:.4f}"
        )
        
        return self.error_model
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _prepare_model_data_efficient(self, y_col, x_cols):
        """
        Prepare data for spatial regression models with memory efficiency.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name
        x_cols : list
            Independent variable column names
            
        Returns
        -------
        tuple
            (model_data, y, X)
        """
        # Create a lightweight view/copy with only necessary columns
        cols_to_use = [y_col] + x_cols
        model_data = self.gdf[cols_to_use].copy()
        
        # Determine column types for appropriate normalization
        numeric_cols = [col for col in cols_to_use 
                       if model_data[col].dtype.kind in 'if']
        
        # Normalize numeric columns to float32 for better memory usage
        for col in numeric_cols:
            # Calculate z-scores directly without copying entire DataFrames
            if model_data[col].std() != 0:
                model_data[col] = ((model_data[col] - model_data[col].mean()) / model_data[col].std()).astype(np.float32)
            else:
                model_data[col] = (model_data[col] - model_data[col].mean()).astype(np.float32)
        
        # For very large datasets, consider using memory-mapped arrays
        if len(model_data) > 10000:
            # Extract data as memory-efficient arrays
            y = model_data[y_col].values.astype(np.float32)
            X = model_data[x_cols].values.astype(np.float32)
            
            # Use memory mapping for large arrays
            y, _ = create_mmap_array(y)
            X, _ = create_mmap_array(X)
        else:
            # For smaller datasets, just use numpy arrays
            y = model_data[y_col].values
            X = model_data[x_cols].values
        
        return model_data, y, X
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _prepare_model_data(self, y_col, x_cols):
        """
        Legacy method - redirects to efficient implementation.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name
        x_cols : list
            Independent variable column names
            
        Returns
        -------
        tuple
            (model_data, y, X)
        """
        return self._prepare_model_data_efficient(y_col, x_cols)
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def visualize_conflict_adjusted_weights(
        self,
        save_path=None,
        figsize=(12, 10),
        title="Conflict-Adjusted Market Connectivity",
        node_color_col=None
    ):
        """
        Visualize the conflict-adjusted spatial weights as a network.
        
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
            
        Returns
        -------
        matplotlib.figure.Figure
            Network visualization figure
        """
        if 'original_weights' not in self.diagnostic_hooks or self.weights is None:
            raise ValueError("No weights available. Call create_weight_matrix with diagnostic info first.")
        
        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Use memory-efficient network creation for large datasets
        if len(self.gdf) > 500:
            self._draw_network_comparison_chunked(ax1, ax2, node_color_col)
        else:
            # For smaller datasets, use standard implementation
            self._draw_network_comparison(ax1, ax2, node_color_col)
        
        # Adjust layout and save if requested
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weights visualization to {save_path}")
        
        return fig
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _draw_network_comparison_chunked(self, ax1, ax2, node_color_col, 
                               base_node_size=100, edge_scale=5, labels=False):
        """
        Memory-efficient implementation of network comparison drawing.
        
        Parameters
        ----------
        ax1 : matplotlib.axes.Axes
            Axes for original network
        ax2 : matplotlib.axes.Axes
            Axes for conflict-adjusted network
        node_color_col : str or None
            Column for node colors
        base_node_size : float
            Base size for nodes
        edge_scale : float
            Scaling factor for edge widths
        labels : bool
            Whether to show node labels
        """
        # Create networkx graphs more efficiently by processing in chunks
        G_orig = self._create_network_graph_chunked(self.diagnostic_hooks['original_weights'])
        G_adj = self._create_network_graph_chunked(self.weights)
        
        # Get node positions with minimal memory usage
        positions = {}
        for i in range(len(self.gdf)):
            positions[i] = (self.gdf.iloc[i].geometry.x, self.gdf.iloc[i].geometry.y)
        
        # Determine node colors efficiently
        if node_color_col and node_color_col in self.gdf.columns:
            node_colors = self.gdf[node_color_col].values
            vmin, vmax = np.min(node_colors), np.max(node_colors)
        else:
            node_colors = 'skyblue'
            vmin, vmax = None, None
        
        # Draw networks
        ax1.set_title("Geographic Connectivity")
        self._draw_network_efficient(G_orig, positions, node_colors, base_node_size, 
                              edge_scale, labels, ax1, vmin, vmax)
        
        ax2.set_title("Conflict-Adjusted Connectivity")
        self._draw_network_efficient(G_adj, positions, node_colors, base_node_size, 
                              edge_scale, labels, ax2, vmin, vmax)
        
        # Add colorbar if using node colors
        if node_color_col and node_color_col in self.gdf.columns:
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                     norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', 
                               pad=0.05, aspect=40)
            cbar.set_label(node_color_col)
        
        # Clean up memory
        del G_orig, G_adj, positions
        gc.collect()
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _create_network_graph_chunked(self, weights):
        """
        Create networkx graph from weights matrix using chunked processing.
        
        Parameters
        ----------
        weights : libpysal.weights.W
            Weights matrix
            
        Returns
        -------
        networkx.Graph
            Network graph
        """
        G = nx.Graph()
        
        # Process nodes in chunks
        chunk_size = 100  # Process 100 nodes at a time
        for i in range(0, len(self.gdf), chunk_size):
            end = min(i + chunk_size, len(self.gdf))
            
            # Add nodes for this chunk
            for j in range(i, end):
                G.add_node(j, pos=(self.gdf.iloc[j].geometry.x, self.gdf.iloc[j].geometry.y))
        
        # Process edges in chunks
        for i in range(0, len(self.gdf), chunk_size):
            end = min(i + chunk_size, len(self.gdf))
            
            # Add edges for each node in this chunk
            for j in range(i, end):
                if j in weights.neighbors:
                    for k, neighbor in enumerate(weights.neighbors[j]):
                        weight = weights.weights[j][k]
                        G.add_edge(j, neighbor, weight=weight)
        
        return G
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _draw_network_efficient(self, G, pos, node_colors, node_size, edge_scale, 
                      labels, ax, vmin=None, vmax=None):
        """
        Memory-efficient network drawing.
        
        Parameters
        ----------
        G : networkx.Graph
            Network graph
        pos : dict
            Node positions
        node_colors : array-like or str
            Node colors
        node_size : float
            Node size
        edge_scale : float
            Edge width scaling
        labels : bool
            Whether to show labels
        ax : matplotlib.axes.Axes
            Axes to draw on
        vmin : float or None
            Minimum value for colormap
        vmax : float or None
            Maximum value for colormap
        """
        # Process edges in chunks to avoid memory spikes
        chunks = []
        edge_weights = []
        
        # Get edges and weights in chunks
        chunk_size = min(1000, len(G.edges))
        edges = list(G.edges)
        
        for i in range(0, len(edges), chunk_size):
            chunk = edges[i:i+chunk_size]
            chunks.append(chunk)
            # Get weights
            weights = [G[u][v]['weight'] * edge_scale for u, v in chunk]
            edge_weights.append(weights)
        
        # Draw edges in chunks
        for chunk, weights in zip(chunks, edge_weights):
            nx.draw_networkx_edges(G, pos, edgelist=chunk, width=weights, 
                                  edge_color='gray', alpha=0.6, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, 
                             cmap='viridis', vmin=vmin, vmax=vmax, ax=ax)
        
        # Draw labels if requested
        if labels:
            # For large networks, only show a subset of labels
            if len(G) > 100:
                label_nodes = list(G.nodes)[:100]  # First 100 nodes
                label_dict = {n: str(n) for n in label_nodes}
            else:
                label_dict = {n: str(n) for n in G.nodes}
            
            nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8, ax=ax)
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _draw_network_comparison(self, ax1, ax2, node_color_col, 
                               base_node_size=100, edge_scale=5, labels=False):
        """
        Legacy method for small datasets - draw network comparison between original and conflict-adjusted weights.
        
        Parameters
        ----------
        ax1 : matplotlib.axes.Axes
            Axes for original network
        ax2 : matplotlib.axes.Axes
            Axes for conflict-adjusted network
        node_color_col : str or None
            Column for node colors
        base_node_size : float
            Base size for nodes
        edge_scale : float
            Scaling factor for edge widths
        labels : bool
            Whether to show node labels
        """
        # Create networkx graphs
        G_orig = self._create_network_graph(self.diagnostic_hooks['original_weights'])
        G_adj = self._create_network_graph(self.weights)
        
        # Get node positions
        pos = nx.get_node_attributes(G_orig, 'pos')
        
        # Determine node colors
        if node_color_col and node_color_col in self.gdf.columns:
            node_colors = self.gdf[node_color_col].values
            vmin, vmax = min(node_colors), max(node_colors)
        else:
            node_colors = 'skyblue'
            vmin, vmax = None, None
        
        # Draw original network
        ax1.set_title("Geographic Connectivity")
        self._draw_network(G_orig, pos, node_colors, base_node_size, 
                        edge_scale, labels, ax1, vmin, vmax)
        
        # Draw conflict-adjusted network
        ax2.set_title("Conflict-Adjusted Connectivity")
        self._draw_network(G_adj, pos, node_colors, base_node_size, 
                        edge_scale, labels, ax2, vmin, vmax)
        
        # Add colorbar if using node colors
        if node_color_col and node_color_col in self.gdf.columns:
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                      norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', 
                               pad=0.05, aspect=40)
            cbar.set_label(node_color_col)
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _create_network_graph(self, weights):
        """
        Legacy method - Create networkx graph from weights matrix.
        
        Parameters
        ----------
        weights : libpysal.weights.W
            Weights matrix
            
        Returns
        -------
        networkx.Graph
            Network graph
        """
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(self.gdf)):
            G.add_node(i, pos=(self.gdf.iloc[i].geometry.x, self.gdf.iloc[i].geometry.y))
        
        # Add edges with weights
        for i, neighbors in weights.neighbors.items():
            for j, neighbor in enumerate(neighbors):
                weight = weights.weights[i][j]
                G.add_edge(i, neighbor, weight=weight)
        
        return G
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _draw_network(self, G, pos, node_colors, node_size, edge_scale, 
                    labels, ax, vmin=None, vmax=None):
        """
        Legacy method - Draw network graph on axes.
        
        Parameters
        ----------
        G : networkx.Graph
            Network graph
        pos : dict
            Node positions
        node_colors : array-like or str
            Node colors
        node_size : float
            Node size
        edge_scale : float
            Edge width scaling
        labels : bool
            Whether to show labels
        ax : matplotlib.axes.Axes
            Axes to draw on
        vmin : float or None
            Minimum value for colormap
        vmax : float or None
            Maximum value for colormap
        """
        # Draw edges with weights as width
        edge_weights = [G[u][v]['weight'] * edge_scale for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', 
                               alpha=0.6, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, 
                              cmap='viridis', vmin=vmin, vmax=vmax, ax=ax)
        
        # Draw labels if requested
        if labels:
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    @timer
    @memory_usage_decorator
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def calculate_impacts(self, model_type='lag'):
        """
        Calculate direct, indirect, and total effects for spatial models.
        
        Parameters
        ----------
        model_type : str, optional
            Type of model to calculate impacts for ('lag' or 'error')
            
        Returns
        -------
        dict
            Dictionary of direct, indirect, and total effects
        """
        # Check if we have the specified model
        if model_type == 'lag' and hasattr(self, 'lag_model'):
            # Calculate impacts for lag model
            impacts = self._calculate_lag_impacts()
            return impacts
        elif model_type == 'error' and hasattr(self, 'error_model'):
            # Calculate impacts for error model
            impacts = self._calculate_error_impacts()
            return impacts
        else:
            raise ValueError(f"Model type '{model_type}' not available or not estimated")
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _calculate_lag_impacts(self):
        """
        Calculate impacts for spatial lag model with improved memory efficiency.
        
        Returns
        -------
        dict
            Dictionary of direct, indirect, and total effects
        """
        # Get parameters
        rho = self.lag_model.rho
        betas = self.lag_model.betas
        
        # Get variable names
        var_names = self.lag_model.name_x
        
        # Calculate impacts efficiently
        direct = {}
        indirect = {}
        total = {}
        
        # Use numpy arrays for calculations to avoid memory fragmentation
        betas_array = np.array(betas)
        direct_vals = betas_array / (1 - rho)
        indirect_vals = betas_array * rho / ((1 - rho) * (1 - rho))
        total_vals = direct_vals + indirect_vals
        
        # Convert to dictionary for API compatibility
        for i, var in enumerate(var_names):
            direct[var] = float(direct_vals[i])  # Convert to native Python type
            indirect[var] = float(indirect_vals[i])
            total[var] = float(total_vals[i])
        
        return {
            'direct': direct,
            'indirect': indirect,
            'total': total,
            'rho': float(rho),  # Convert to native Python type
            'model_type': 'lag'
        }
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _calculate_error_impacts(self):
        """
        Calculate impacts for spatial error model with improved memory efficiency.
        
        Returns
        -------
        dict
            Dictionary of direct, indirect, and total effects
        """
        # For error model, indirect effects are zero
        # Direct effects are just the coefficients
        
        # Get parameters
        betas = self.error_model.betas
        lambda_param = self.error_model.lam
        
        # Get variable names
        var_names = self.error_model.name_x
        
        # Calculate impacts efficiently
        direct = {}
        indirect = {}
        total = {}
        
        # Use numpy array for calculations
        betas_array = np.array(betas)
        
        # Convert to dictionary for API compatibility
        for i, var in enumerate(var_names):
            direct[var] = float(betas_array[i])  # Convert to native Python type
            indirect[var] = 0.0
            total[var] = float(betas_array[i])
        
        return {
            'direct': direct,
            'indirect': indirect,
            'total': total,
            'lambda': float(lambda_param),  # Convert to native Python type
            'model_type': 'error'
        }
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def prepare_simulation_data(self):
        """
        Prepare model results for use in simulation module.
        
        Returns
        -------
        dict
            Simulation-ready data structure
        """
        if self.weights is None:
            raise ValueError("Weights matrix not created. Call create_weight_matrix first.")
        
        # Create lightweight simulation data
        simulation_data = {
            'weights_matrix': self.weights,
            'original_weights': self.diagnostic_hooks.get('original_weights'),
            'model_type': 'spatial_econometrics'
        }
        
        # Create a lightweight copy of the GeoDataFrame with only essential columns
        essential_cols = ['geometry']
        for col in self.gdf.columns:
            if col.endswith('_normalized') or 'price' in col.lower():
                essential_cols.append(col)
        
        # Add downcast version of GeoDataFrame
        simulation_data['gdf'] = self.gdf[essential_cols].copy()
        
        # Add selected diagnostic info (not all to save memory)
        essential_hooks = {}
        for key in ['conflict_adjusted', 'conflict_col', 'conflict_weight', 'k']:
            if key in self.diagnostic_hooks:
                essential_hooks[key] = self.diagnostic_hooks[key]
        simulation_data['diagnostic_hooks'] = essential_hooks
        
        # Add model impacts if available
        if hasattr(self, 'lag_model') and self.lag_model is not None:
            try:
                lag_impacts = self.calculate_impacts(model_type='lag')
                simulation_data['lag_impacts'] = lag_impacts
            except Exception as e:
                logger.warning(f"Could not calculate lag model impacts: {e}")
                
        if hasattr(self, 'error_model') and self.error_model is not None:
            try:
                error_impacts = self.calculate_impacts(model_type='error')
                simulation_data['error_impacts'] = error_impacts
            except Exception as e:
                logger.warning(f"Could not calculate error model impacts: {e}")
        
        return simulation_data
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _validate_model_columns(self, columns):
        """
        Validate that columns exist in the GeoDataFrame.
        
        Parameters
        ----------
        columns : list
            Column names to validate
        """
        missing_cols = [col for col in columns if col not in self.gdf.columns]
        if missing_cols:
            raise ValueError(f"Column(s) not found in GeoDataFrame: {', '.join(missing_cols)}")
    
    @timer
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def calculate_spatial_barriers(self, conflict_col, threshold=0.5, return_gdf=True):
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
        """
        # Check if weight matrix has been created
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Check if conflict column exists
        if conflict_col not in self.gdf.columns:
            raise ValueError(f"Conflict column '{conflict_col}' not found in GeoDataFrame")
        
        # Calculate barriers using helper methods - use chunked processing for large datasets
        if len(self.gdf) > 1000:
            result_gdf = self._calculate_barrier_metrics_chunked(conflict_col, threshold)
        else:
            result_gdf = self._calculate_barrier_metrics(conflict_col, threshold)
            
        barrier_metrics = self._summarize_barrier_results(result_gdf, threshold)
        
        if return_gdf:
            return result_gdf
        else:
            return barrier_metrics
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _calculate_barrier_metrics_chunked(self, conflict_col, threshold):
        """
        Memory-efficient implementation for calculating barrier metrics.
        
        Parameters
        ----------
        conflict_col : str
            Column with conflict intensity
        threshold : float
            Threshold for barrier identification
            
        Returns
        -------
        geopandas.GeoDataFrame
            Markets with barrier metrics
        """
        result_gdf = self.gdf.copy()
        
        # Pre-allocate arrays for results
        barriers = np.zeros(len(result_gdf), dtype=np.int32)
        barrier_intensities = np.zeros(len(result_gdf), dtype=np.float32)
        
        # Get conflict values as array for faster access
        conflict_values = result_gdf[conflict_col].values
        
        # Process in chunks
        chunk_size = 100  # Process 100 markets at a time
        for chunk_start in range(0, len(result_gdf), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(result_gdf))
            indices = range(chunk_start, chunk_end)
            
            for idx in indices:
                if idx not in self.weights.neighbors:
                    continue
                    
                neighbors = self.weights.neighbors[idx]
                if not neighbors:
                    continue
                
                # Calculate metrics for this market efficiently
                idx_conflict = conflict_values[idx]
                neighbor_conflicts = conflict_values[neighbors]
                
                # Calculate average conflict levels along paths
                avg_conflicts = (idx_conflict + neighbor_conflicts) / 2
                
                # Count barriers (above threshold)
                barrier_mask = avg_conflicts > threshold
                barriers_count = np.sum(barrier_mask)
                
                # Calculate barrier intensity
                if barriers_count > 0:
                    total_intensity = np.sum(avg_conflicts[barrier_mask])
                    barrier_intensities[idx] = total_intensity / barriers_count
                
                barriers[idx] = barriers_count
        
        # Add barrier metrics to GeoDataFrame
        result_gdf['barrier_count'] = barriers
        result_gdf['barrier_intensity'] = barrier_intensities
        
        # Add normalized barrier isolation
        max_count = result_gdf['barrier_count'].max()
        if max_count > 0:
            result_gdf['barrier_isolation'] = result_gdf['barrier_count'] / max_count
        else:
            result_gdf['barrier_isolation'] = 0
            
        return result_gdf
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _calculate_barrier_metrics(self, conflict_col, threshold):
        """
        Legacy method - Calculate barrier metrics for each market.
        
        Parameters
        ----------
        conflict_col : str
            Column with conflict intensity
        threshold : float
            Threshold for barrier identification
            
        Returns
        -------
        geopandas.GeoDataFrame
            Markets with barrier metrics
        """
        result_gdf = self.gdf.copy()
        barriers = []
        barrier_intensities = []
        
        # For each market, count connections above conflict threshold
        for i in range(len(result_gdf)):
            if i not in self.weights.neighbors:
                barriers.append(0)
                barrier_intensities.append(0)
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
        
        # Add normalized barrier isolation
        max_count = result_gdf['barrier_count'].max()
        if max_count > 0:
            result_gdf['barrier_isolation'] = result_gdf['barrier_count'] / max_count
        else:
            result_gdf['barrier_isolation'] = 0
            
        return result_gdf
    
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _summarize_barrier_results(self, result_gdf, threshold):
        """
        Summarize barrier analysis results with improved memory efficiency.
        
        Parameters
        ----------
        result_gdf : geopandas.GeoDataFrame
            Markets with barrier metrics
        threshold : float
            Threshold used for barriers
            
        Returns
        -------
        dict
            Summary barrier metrics
        """
        # Calculate market-wide barrier metrics
        # Use memory-efficient approach for large weight matrices
        if len(self.weights.neighbors) > 1000:
            total_connections = 0
            for chunk_start in range(0, len(self.weights.neighbors), 100):
                chunk_end = min(chunk_start + 100, len(self.weights.neighbors))
                indices = list(self.weights.neighbors.keys())[chunk_start:chunk_end]
                for idx in indices:
                    total_connections += len(self.weights.neighbors[idx])
        else:
            total_connections = sum(len(neighbors) for neighbors in self.weights.neighbors.values())
        
        total_barriers = int(result_gdf['barrier_count'].sum())
        
        if total_connections > 0:
            barrier_rate = total_barriers / total_connections
        else:
            barrier_rate = 0
        
        barrier_metrics = {
            'total_barriers': total_barriers,
            'total_connections': total_connections,
            'barrier_rate': barrier_rate,
            'high_barrier_markets': int((result_gdf['barrier_count'] > 0).sum()),
            'barrier_threshold': threshold
        }
        
        # Only include reference to GeoDataFrame to conserve memory
        barrier_metrics['barrier_columns'] = ['barrier_count', 'barrier_intensity', 'barrier_isolation']
        
        logger.info(
            f"Identified {total_barriers} barriers out of {total_connections} connections "
            f"({barrier_rate:.1%}) using threshold {threshold}"
        )
        
        return barrier_metrics


@timer
@memory_usage_decorator
@m3_optimized(parallel=True, memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def calculate_market_accessibility(markets_gdf, population_gdf, max_distance=50000,
                                  distance_decay=2.0, weight_col='population'):
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
    _validate_accessibility_inputs(markets_gdf, population_gdf, weight_col)
    
    # For large datasets, optimize memory usage
    if len(markets_gdf) > 100 or len(population_gdf) > 1000:
        logger.info(f"Using memory-optimized accessibility calculation for {len(markets_gdf)} markets and {len(population_gdf)} population centers")
        return _calculate_accessibility_chunked(
            markets_gdf=markets_gdf,
            population_gdf=population_gdf,
            max_distance=max_distance,
            distance_decay=distance_decay,
            weight_col=weight_col
        )
    
    # For smaller datasets, use the project utility
    from yemen_market_integration.utils import compute_accessibility_index
    result_gdf = compute_accessibility_index(
        markets_gdf=markets_gdf,
        population_gdf=population_gdf,
        max_distance=max_distance,
        distance_decay=distance_decay,
        weight_col=weight_col
    )
    
    logger.info(f"Calculated accessibility index for {len(markets_gdf)} markets")
    return result_gdf


@m3_optimized(parallel=True, memory_intensive=True)
def _calculate_accessibility_chunked(markets_gdf, population_gdf, max_distance, 
                                   distance_decay, weight_col):
    """
    Memory-efficient implementation of accessibility calculation.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Markets with locations
    population_gdf : geopandas.GeoDataFrame
        Population centers with locations
    max_distance : float
        Maximum distance in meters to consider
    distance_decay : float
        Distance decay exponent
    weight_col : str
        Column in population_gdf to use as weight
        
    Returns
    -------
    geopandas.GeoDataFrame
        Market GeoDataFrame with accessibility index
    """
    result_gdf = markets_gdf.copy()
    
    # Extract weight values as numpy array for faster access
    population_weights = population_gdf[weight_col].values.astype(np.float32)
    
    # Pre-allocate array for accessibility values
    accessibility_values = np.zeros(len(result_gdf), dtype=np.float32)
    
    # Process in chunks
    market_chunk_size = min(50, max(10, len(markets_gdf) // 10))
    pop_chunk_size = min(500, max(50, len(population_gdf) // 10))
    
    # For each market chunk
    for i in range(0, len(markets_gdf), market_chunk_size):
        market_end = min(i + market_chunk_size, len(markets_gdf))
        market_chunk = markets_gdf.iloc[i:market_end]
        
        # Process each market against population chunks
        for j in range(0, len(population_gdf), pop_chunk_size):
            pop_end = min(j + pop_chunk_size, len(population_gdf))
            pop_chunk = population_gdf.iloc[j:pop_end]
            pop_weights_chunk = population_weights[j:pop_end]
            
            # Calculate distances between market chunk and population chunk
            # This is memory-intensive, so we do it in small chunks
            for m_idx, market in market_chunk.iterrows():
                market_idx = markets_gdf.index.get_loc(m_idx)
                
                # Calculate distances from this market to all population centers in chunk
                distances = np.array([market.geometry.distance(p.geometry) for _, p in pop_chunk.iterrows()], 
                                  dtype=np.float32)
                
                # Apply distance decay function
                valid_distances = distances <= max_distance
                if np.any(valid_distances):
                    valid_weights = pop_weights_chunk[valid_distances]
                    valid_dist = distances[valid_distances]
                    
                    # Calculate accessibility contribution
                    contributions = valid_weights / (valid_dist ** distance_decay)
                    accessibility_values[market_idx] += np.sum(contributions)
        
    # Normalize for better interpretability
    if np.max(accessibility_values) > 0:
        accessibility_values = accessibility_values / np.max(accessibility_values)
    
    # Add to result GeoDataFrame
    result_gdf['accessibility_index'] = accessibility_values
    
    return result_gdf


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _validate_accessibility_inputs(markets_gdf, population_gdf, weight_col):
    """
    Validate inputs for accessibility calculation.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Markets with locations
    population_gdf : geopandas.GeoDataFrame
        Population centers with locations
    weight_col : str
        Population weight column
    """
    # Validate markets data
    valid, errors = validate_geodataframe(
        markets_gdf,
        min_rows=1,
        geometry_type='Point',
        check_crs=True
    )
    raise_if_invalid(valid, errors, "Invalid market data for accessibility analysis")
    
    # Validate population data
    valid, errors = validate_geodataframe(
        population_gdf,
        required_columns=[weight_col],
        min_rows=1,
        geometry_type='Point',
        check_crs=True
    )
    raise_if_invalid(valid, errors, "Invalid population data for accessibility analysis")


@timer
@m3_optimized(parallel=True, memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def simulate_improved_connectivity(markets_gdf, conflict_reduction, conflict_col='conflict_intensity_normalized',
                                 price_col='price', spatial_model=None):
    """
    Simulate improved market connectivity by reducing conflict barriers.
    
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
        
    Returns
    -------
    dict
        Simulation results
    """
    # Validate inputs
    _validate_simulation_inputs(markets_gdf, conflict_reduction, conflict_col, price_col)
    
    # Initialize results dictionary
    results = {
        'scenario': f"{conflict_reduction*100:.0f}% Conflict Reduction",
        'reduction_factor': conflict_reduction,
        'metrics': {},
        'models': {}
    }
    
    # Prepare simulation data with memory optimization
    sim_data = _prepare_simulation_data(markets_gdf, conflict_reduction, conflict_col)
    
    # Run spatial analysis with conflict-reduced data
    if len(markets_gdf) > 1000:
        # For large datasets, use chunked processing
        logger.info(f"Using memory-optimized spatial analysis for large dataset (n={len(markets_gdf)})")
        spatial_results = _run_spatial_analysis_chunked(
            markets_gdf, sim_data, conflict_col, price_col, spatial_model
        )
    else:
        # For smaller datasets, use standard implementation
        spatial_results = _run_spatial_analysis(
            markets_gdf, sim_data, conflict_col, price_col, spatial_model
        )
    
    # Store results
    results['models'] = spatial_results
    
    # Add price differential metrics
    price_metrics = _add_price_differential_metrics(
        markets_gdf, conflict_reduction, price_col, results
    )
    results['metrics'].update(price_metrics)
    
    # Generate visualizations
    viz_data = _add_simulation_visualizations(results)
    results['visualizations'] = viz_data
    
    return results


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _validate_simulation_inputs(markets_gdf, conflict_reduction, conflict_col, price_col):
    """
    Validate inputs for connectivity simulation.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data
    conflict_reduction : float
        Conflict reduction factor
    conflict_col : str
        Conflict column name
    price_col : str
        Price column name
    """
    # Validate GeoDataFrame
    valid, errors = validate_geodataframe(
        markets_gdf,
        required_columns=[conflict_col, price_col],
        min_rows=5,
        check_crs=True
    )
    raise_if_invalid(valid, errors, "Invalid market data for connectivity simulation")
    
    # Validate conflict_reduction
    if not 0 <= conflict_reduction <= 1:
        raise ValueError(f"conflict_reduction must be between 0 and 1, got {conflict_reduction}")


@m3_optimized(memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _prepare_simulation_data(markets_gdf, conflict_reduction, conflict_col):
    """
    Prepare data for connectivity simulation with memory efficiency.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data
    conflict_reduction : float
        Conflict reduction factor
    conflict_col : str
        Conflict column name
        
    Returns
    -------
    dict
        Simulation data
    """
    # Create a copy with only necessary columns to save memory
    sim_gdf = markets_gdf.copy()
    
    # Calculate reduced conflict values
    original_conflict = sim_gdf[conflict_col].values.copy()
    reduced_conflict = original_conflict * (1 - conflict_reduction)
    
    # Add reduced conflict column
    reduced_col = f"{conflict_col}_reduced"
    sim_gdf[reduced_col] = reduced_conflict
    
    # Create simulation data
    sim_data = {
        'original_conflict': original_conflict,
        'reduced_conflict': reduced_conflict,
        'conflict_reduction': conflict_reduction,
        'conflict_col': conflict_col,
        'reduced_col': reduced_col
    }
    
    return sim_data


@m3_optimized(parallel=True, memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _run_spatial_analysis_chunked(markets_gdf, sim_data, conflict_col, price_col, 
                               existing_model=None):
    """
    Memory-efficient implementation of spatial analysis for connectivity simulation.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data
    sim_data : dict
        Simulation data
    conflict_col : str
        Conflict column name
    price_col : str
        Price column name
    existing_model : SpatialEconometrics, optional
        Pre-fitted spatial model
        
    Returns
    -------
    dict
        Spatial analysis results
    """
    # Create new spatial model with reduced conflict
    reduced_col = sim_data['reduced_col']
    
    # Use existing model if provided
    if existing_model is not None:
        spatial_model = existing_model
        
        # Update conflict column with reduced values
        spatial_model.gdf[reduced_col] = sim_data['reduced_conflict']
    else:
        # Create new model from scratch
        spatial_model = SpatialEconometrics(markets_gdf)
    
    # Create conflict-adjusted weights with reduced conflict
    weights = spatial_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=reduced_col,
        conflict_weight=0.8  # Higher weight for simulation
    )
    
    # Calculate Moran's I for price with reduced conflict weights
    moran_result = spatial_model.moran_i_test(price_col)
    
    # Create spatial lag model 
    # Use a limited set of predictors to save memory
    predictors = [reduced_col]
    
    # Add additional predictors if they exist
    for col in ['accessibility_index', 'exchange_rate', 'distance_to_port']:
        if col in markets_gdf.columns:
            predictors.append(col)
    
    # Fit model with chunked data processing
    lag_model = spatial_model.spatial_lag_model(price_col, predictors)
    
    # Calculate impacts
    impacts = spatial_model.calculate_impacts(model_type='lag')
    
    # Return results
    return {
        'spatial_model': spatial_model,
        'moran': moran_result,
        'lag_model': {
            'rho': float(lag_model.rho),
            'r2': float(lag_model.pr2),
            'aic': float(lag_model.aic),
            'impacts': impacts
        }
    }


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _run_spatial_analysis(markets_gdf, sim_data, conflict_col, price_col, 
                        existing_model=None):
    """
    Legacy implementation of spatial analysis for connectivity simulation.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data
    sim_data : dict
        Simulation data
    conflict_col : str
        Conflict column name
    price_col : str
        Price column name
    existing_model : SpatialEconometrics, optional
        Pre-fitted spatial model
        
    Returns
    -------
    dict
        Spatial analysis results
    """
    # Create new spatial model with reduced conflict
    reduced_col = sim_data['reduced_col']
    
    # Use existing model if provided
    if existing_model is not None:
        spatial_model = existing_model
        
        # Update conflict column with reduced values
        spatial_model.gdf[reduced_col] = sim_data['reduced_conflict']
    else:
        # Create new model from scratch
        spatial_model = SpatialEconometrics(markets_gdf)
    
    # Create conflict-adjusted weights with reduced conflict
    weights = spatial_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=reduced_col,
        conflict_weight=0.8  # Higher weight for simulation
    )
    
    # Calculate Moran's I for price with reduced conflict weights
    moran_result = spatial_model.moran_i_test(price_col)
    
    # Create spatial lag model
    predictors = [reduced_col]
    for col in ['accessibility_index', 'exchange_rate', 'distance_to_port']:
        if col in markets_gdf.columns:
            predictors.append(col)
    
    lag_model = spatial_model.spatial_lag_model(price_col, predictors)
    
    # Calculate impacts
    impacts = spatial_model.calculate_impacts(model_type='lag')
    
    # Return results
    return {
        'spatial_model': spatial_model,
        'moran': moran_result,
        'lag_model': {
            'rho': lag_model.rho,
            'r2': lag_model.pr2,
            'aic': lag_model.aic,
            'impacts': impacts
        }
    }


@m3_optimized(memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _add_price_differential_metrics(markets_gdf, conflict_reduction, price_col, results):
    """
    Calculate price differential metrics for reduced conflict scenario.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data
    conflict_reduction : float
        Conflict reduction factor
    price_col : str
        Price column name
    results : dict
        Simulation results
        
    Returns
    -------
    dict
        Price differential metrics
    """
    # Access the lag model parameters
    lag_model_results = results['models']['lag_model']
    rho = lag_model_results['rho']
    impacts = lag_model_results['impacts']
    
    # Calculate price effect based on reduced conflict
    conflict_impact = impacts['total'].get('conflict_intensity_normalized_reduced', 0)
    
    # For large datasets, use chunked processing to calculate price differentials
    if len(markets_gdf) > 1000:
        price_diff_pct = _calculate_price_diff_chunked(
            markets_gdf, conflict_reduction, conflict_impact
        )
    else:
        # Simple approach for smaller datasets
        # Calculate average price reduction
        conflict_col = results['models']['spatial_model'].diagnostic_hooks.get('conflict_col', 'conflict_intensity_normalized')
        avg_conflict = markets_gdf[conflict_col].mean()
        avg_conflict_change = avg_conflict * conflict_reduction
        price_diff_pct = avg_conflict_change * conflict_impact * 100
    
    # Add metrics
    metrics = {
        'avg_price_diff_pct': float(price_diff_pct),
        'spatial_dependence_original': float(results['models']['moran']['I']),
        'network_density_change_pct': float(conflict_reduction * 100 * 0.8)
    }
    
    return metrics


@m3_optimized(memory_intensive=True)
def _calculate_price_diff_chunked(markets_gdf, conflict_reduction, conflict_impact):
    """
    Calculate price differentials in chunks for memory efficiency.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data
    conflict_reduction : float
        Conflict reduction factor
    conflict_impact : float
        Impact of conflict on price
        
    Returns
    -------
    float
        Average price difference percentage
    """
    # Process in chunks
    chunk_size = 200  # Process 200 markets at a time
    conflict_col = 'conflict_intensity_normalized'
    
    total_diff = 0
    
    for i in range(0, len(markets_gdf), chunk_size):
        end_idx = min(i + chunk_size, len(markets_gdf))
        chunk = markets_gdf.iloc[i:end_idx]
        
        # Calculate conflict change for chunk
        conflict_values = chunk[conflict_col].values
        conflict_change = conflict_values * conflict_reduction
        
        # Calculate price impact for chunk
        chunk_price_diff = conflict_change * conflict_impact * 100
        
        # Add to total
        total_diff += chunk_price_diff.sum()
    
    # Calculate average
    avg_price_diff = total_diff / len(markets_gdf)
    
    return avg_price_diff


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _add_simulation_visualizations(results):
    """
    Add visualizations to simulation results.
    
    Parameters
    ----------
    results : dict
        Simulation results
        
    Returns
    -------
    dict
        Visualization data
    """
    # Return empty visualization data
    # Actual visualization generation is typically deferred to avoid memory overhead
    return {
        'types_available': ['network', 'choropleth', 'histogram'],
        'description': 'Visualizations can be generated on demand'
    }


@timer
@memory_usage_decorator
@m3_optimized(parallel=True, memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def market_integration_index(prices_df: pd.DataFrame, weights_matrix, market_id_col: str, 
                           price_col: str, time_col: str = 'date',
                           window: int = 12, cross_section: bool = True) -> pd.DataFrame:
    """
    Calculate market integration index between connected markets.
    
    This function has been optimized for memory efficiency with large price panels.
    
    Parameters
    ----------
    prices_df : pd.DataFrame
        Panel of market prices over time
    weights_matrix : libpysal.weights.W
        Spatial weights matrix defining market connections
    market_id_col : str
        Column with market identifiers
    price_col : str
        Column with price data
    time_col : str, optional
        Column with time periods
    window : int, optional
        Window size for rolling calculations
    cross_section : bool, optional
        Whether to calculate cross-sectional metrics
        
    Returns
    -------
    pd.DataFrame
        Integration metrics by time period
    """
    # Validate inputs
    _validate_integration_inputs(prices_df, weights_matrix, market_id_col, price_col, time_col)
    
    # For large datasets, use chunked processing
    if len(prices_df) > 10000:
        logger.info(f"Using chunked processing for large price panel (n={len(prices_df)})")
        return _process_large_price_panel(
            prices_df, weights_matrix, market_id_col, price_col, time_col, window, cross_section
        )
    
    # Convert time column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(prices_df[time_col]):
        prices_df[time_col] = pd.to_datetime(prices_df[time_col])
    
    # Sort by time
    prices_df = prices_df.sort_values(time_col)
    
    # Get unique time periods
    time_periods = prices_df[time_col].unique()
    
    # Initialize results DataFrame
    results = []
    
    # Calculate integration metrics for each time period
    for t, time_period in enumerate(time_periods):
        # Get current period data
        current_data = prices_df[prices_df[time_col] == time_period]
        
        # Create period metrics
        period_metrics = {
            time_col: time_period,
            'period_idx': t,
            'n_markets': len(current_data),
        }
        
        # Calculate window metrics if enough prior periods
        windows = [
            {'size': window, 'label': f'{window}_period'}
        ]
        
        # Add metrics for current period
        if len(current_data) >= 3:  # Need at least 3 markets for meaningful measures
            # Create price vector
            price_vector = current_data.set_index(market_id_col)[price_col]
            
            # Add current period metrics
            _add_current_period_metrics(period_metrics, price_vector, weights_matrix)
            
            # Add window-based metrics if we have enough history
            if t >= window:
                _add_window_metrics(period_metrics, prices_df, time_periods, t, windows,
                                   market_id_col, price_col, weights_matrix)
        
        # Add to results
        results.append(period_metrics)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Add 3-period rolling average for smoothing
    for col in results_df.columns:
        if col not in [time_col, 'period_idx', 'n_markets'] and not col.startswith('raw_'):
            results_df[f'{col}_smoothed'] = results_df[col].rolling(3, min_periods=1).mean()
    
    return results_df


@m3_optimized(parallel=True, memory_intensive=True)
def _process_large_price_panel(prices_df, weights_matrix, market_id_col, price_col, 
                            time_col, window, cross_section):
    """
    Memory-efficient processing of large price panels.
    
    Parameters
    ----------
    prices_df : pd.DataFrame
        Panel of market prices over time
    weights_matrix : libpysal.weights.W
        Spatial weights matrix defining market connections
    market_id_col : str
        Column with market identifiers
    price_col : str
        Column with price data
    time_col : str
        Column with time periods
    window : int
        Window size for rolling calculations
    cross_section : bool
        Whether to calculate cross-sectional metrics
        
    Returns
    -------
    pd.DataFrame
        Integration metrics by time period
    """
    # Convert time column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(prices_df[time_col]):
        prices_df[time_col] = pd.to_datetime(prices_df[time_col])
    
    # Sort by time
    prices_df = prices_df.sort_values(time_col)
    
    # Get unique time periods
    time_periods = prices_df[time_col].unique()
    
    # Process in chunks based on time periods
    chunk_size = min(12, max(3, len(time_periods) // 10))  # Adaptive chunk size
    
    # Use parallel processing for time period chunks
    return _process_periods_parallel(
        prices_df, weights_matrix, market_id_col, price_col, time_col,
        time_periods, window, chunk_size, cross_section
    )


@m3_optimized(parallel=True, memory_intensive=True)
def _process_periods_parallel(
    prices_df, weights_matrix, market_id_col, price_col, time_col,
    time_periods, window, chunk_size, cross_section
):
    """
    Process time periods in parallel with memory efficiency.
    
    Parameters
    ----------
    prices_df : pd.DataFrame
        Price panel data
    weights_matrix : libpysal.weights.W
        Spatial weights matrix
    market_id_col : str
        Market ID column
    price_col : str
        Price column
    time_col : str
        Time column
    time_periods : array-like
        Unique time periods
    window : int
        Window size
    chunk_size : int
        Size of time period chunks
    cross_section : bool
        Whether to calculate cross-sectional metrics
        
    Returns
    -------
    pd.DataFrame
        Integration metrics by time period
    """
    # Create chunks of time periods
    period_chunks = []
    for i in range(0, len(time_periods), chunk_size):
        end = min(i + chunk_size, len(time_periods))
        period_chunks.append(time_periods[i:end])
    
    # Create processing function
    def process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
        # Get time periods for this chunk
        chunk_periods = chunk_df[time_col].unique()
        
        # Initialize results
        chunk_results = []
        
        # Process each period
        for t, time_period in enumerate(chunk_periods):
            # Global period index
            global_t = np.where(time_periods == time_period)[0][0]
            
            # Get current period data
            current_data = chunk_df[chunk_df[time_col] == time_period]
            
            # Create period metrics
            period_metrics = {
                time_col: time_period,
                'period_idx': global_t,
                'n_markets': len(current_data),
            }
            
            # Define windows
            windows = [
                {'size': window, 'label': f'{window}_period'}
            ]
            
            # Add metrics for current period
            if len(current_data) >= 3:
                # Create price vector
                price_vector = current_data.set_index(market_id_col)[price_col]
                
                # Add current period metrics
                _add_current_period_metrics(period_metrics, price_vector, weights_matrix)
                
                # Add window-based metrics if we have enough history
                if global_t >= window:
                    # We need to get historical data
                    historical_periods = time_periods[global_t-window:global_t]
                    historical_filter = prices_df[time_col].isin(historical_periods)
                    historical_data = prices_df[historical_filter]
                    
                    _add_window_metrics(
                        period_metrics, historical_data, time_periods, 
                        global_t, windows, market_id_col, price_col, weights_matrix
                    )
            
            # Add to results
            chunk_results.append(period_metrics)
        
        return pd.DataFrame(chunk_results)
    
    # Create chunks based on time periods
    chunk_dfs = []
    for periods in period_chunks:
        period_filter = prices_df[time_col].isin(periods)
        chunk_dfs.append(prices_df[period_filter])
    
    # Process chunks - potentially in parallel for large datasets
    if len(chunk_dfs) > 4:
        try:
            import multiprocessing as mp
            with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
                chunk_results = pool.map(process_chunk, chunk_dfs)
        except Exception:
            # Fall back to sequential processing
            chunk_results = [process_chunk(chunk) for chunk in chunk_dfs]
    else:
        # Process sequentially for small number of chunks
        chunk_results = [process_chunk(chunk) for chunk in chunk_dfs]
    
    # Combine results
    combined_results = pd.concat(chunk_results, ignore_index=True)
    
    # Sort by time
    combined_results = combined_results.sort_values(time_col)
    
    # Add 3-period rolling average for smoothing
    for col in combined_results.columns:
        if col not in [time_col, 'period_idx', 'n_markets'] and not col.startswith('raw_'):
            combined_results[f'{col}_smoothed'] = combined_results[col].rolling(3, min_periods=1).mean()
    
    return combined_results


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _validate_integration_inputs(prices_df, weights_matrix, market_id_col, price_col, time_col):
    """
    Validate inputs for market integration calculation.
    
    Parameters
    ----------
    prices_df : pd.DataFrame
        Price panel data
    weights_matrix : libpysal.weights.W
        Spatial weights matrix
    market_id_col : str
        Market ID column
    price_col : str
        Price column
    time_col : str
        Time column
    """
    # Check DataFrame
    if not isinstance(prices_df, pd.DataFrame):
        raise ValueError("prices_df must be a pandas DataFrame")
    
    # Check columns
    for col in [market_id_col, price_col, time_col]:
        if col not in prices_df.columns:
            raise ValueError(f"Column '{col}' not found in prices_df")
    
    # Check weights matrix
    if not isinstance(weights_matrix, W):
        raise ValueError("weights_matrix must be a libpysal.weights.W object")
    
    # Check if market IDs in DataFrame match weights matrix
    markets_in_df = set(prices_df[market_id_col].unique())
    markets_in_weights = set(weights_matrix.neighbors.keys())
    
    if not markets_in_weights.intersection(markets_in_df):
        raise ValueError("No market IDs in DataFrame match weights matrix. Check market ID formats.")


@m3_optimized(memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _calculate_period_metrics(period_data, prices_df, time_periods, t, windows,
                           market_id_col, price_col, weights_matrix):
    """
    Calculate metrics for a time period.
    
    Parameters
    ----------
    period_data : pd.DataFrame
        Data for the current period
    prices_df : pd.DataFrame
        Full price panel data
    time_periods : array-like
        Unique time periods
    t : int
        Current period index
    windows : list
        Window specifications
    market_id_col : str
        Market ID column
    price_col : str
        Price column
    weights_matrix : libpysal.weights.W
        Spatial weights matrix
        
    Returns
    -------
    dict
        Period metrics
    """
    # Create price vector
    price_vector = period_data.set_index(market_id_col)[price_col]
    
    # Initialize metrics
    period_metrics = {
        'period_idx': t,
        'n_markets': len(period_data),
    }
    
    # Add current period metrics
    _add_current_period_metrics(period_metrics, price_vector, weights_matrix)
    
    # Add window-based metrics
    _add_window_metrics(
        period_metrics, prices_df, time_periods, t, windows,
        market_id_col, price_col, weights_matrix
    )
    
    return period_metrics


@m3_optimized(memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _add_current_period_metrics(period_metrics, price_vector, weights_matrix):
    """
    Add metrics for current period.
    
    Parameters
    ----------
    period_metrics : dict
        Metrics dictionary to update
    price_vector : pd.Series
        Price vector for current period
    weights_matrix : libpysal.weights.W
        Spatial weights matrix
    """
    # Check if we have prices for markets in the weights matrix
    common_markets = set(price_vector.index).intersection(set(weights_matrix.neighbors.keys()))
    
    if len(common_markets) < 3:
        # Not enough markets for spatial statistics
        period_metrics['moran_i'] = np.nan
        period_metrics['cv'] = np.nan
        period_metrics['integration_index'] = np.nan
        return
    
    # Calculate coefficient of variation
    cv = price_vector.std() / price_vector.mean() if price_vector.mean() > 0 else np.nan
    period_metrics['cv'] = cv
    
    # Filter price vector to common markets
    filtered_prices = price_vector.loc[list(common_markets)]
    
    # Calculate Moran's I (if we have at least 4 markets)
    if len(filtered_prices) >= 4:
        try:
            # Create a subset of the weights matrix for available markets
            w_subset = weights_matrix.sparse.toarray()[list(common_markets)][:, list(common_markets)]
            w_subset = W.from_array(w_subset)
            
            # Calculate Moran's I
            from esda.moran import Moran
            moran = Moran(filtered_prices, w_subset)
            period_metrics['moran_i'] = moran.I
            period_metrics['raw_p_value'] = moran.p_sim
        except Exception:
            period_metrics['moran_i'] = np.nan
            period_metrics['raw_p_value'] = np.nan
    else:
        period_metrics['moran_i'] = np.nan
        period_metrics['raw_p_value'] = np.nan
    
    # Calculate integration index
    period_metrics['integration_index'] = 1 - (cv / 0.5) if cv <= 0.5 else 0.0


@m3_optimized(memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _add_window_metrics(period_metrics, prices_df, time_periods, t, windows,
                      market_id_col, price_col, weights_matrix):
    """
    Add metrics for time windows.
    
    Parameters
    ----------
    period_metrics : dict
        Metrics dictionary to update
    prices_df : pd.DataFrame
        Full price panel data
    time_periods : array-like
        Unique time periods
    t : int
        Current period index
    windows : list
        Window specifications
    market_id_col : str
        Market ID column
    price_col : str
        Price column
    weights_matrix : libpysal.weights.W
        Spatial weights matrix
    """
    # Current period data
    current_period = time_periods[t]
    current_data = prices_df[prices_df[time_col] == current_period]
    current_prices = current_data.set_index(market_id_col)[price_col]
    
    # Calculate metrics for each window
    for window in windows:
        window_size = window['size']
        window_label = window['label']
        
        # Check if we have enough history
        if t < window_size:
            continue
        
        # Get previous period
        past_period_idx = t - window_size
        past_period = time_periods[past_period_idx]
        past_data = prices_df[prices_df[time_col] == past_period]
        past_prices = past_data.set_index(market_id_col)[price_col]
        
        # Calculate convergence metrics
        _calculate_convergence_metrics(
            period_metrics, current_prices, past_prices, window_label,
            weights_matrix
        )


@m3_optimized(memory_intensive=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _calculate_convergence_metrics(period_metrics, current_prices, past_prices, window,
                                weights_matrix):
    """
    Calculate market convergence metrics.
    
    Parameters
    ----------
    period_metrics : dict
        Metrics dictionary to update
    current_prices : pd.Series
        Current period prices
    past_prices : pd.Series
        Past period prices
    window : str
        Window label
    weights_matrix : libpysal.weights.W
        Spatial weights matrix
    """
    # Find common markets
    common_markets = set(current_prices.index).intersection(set(past_prices.index))
    
    if len(common_markets) < 3:
        # Not enough data for convergence metrics
        period_metrics[f'cv_change_{window}'] = np.nan
        period_metrics[f'moran_change_{window}'] = np.nan
        period_metrics[f'convergence_index_{window}'] = np.nan
        return
    
    # Filter to common markets
    c_prices = current_prices.loc[list(common_markets)]
    p_prices = past_prices.loc[list(common_markets)]
    
    # Calculate coefficient of variation for both periods
    c_cv = c_prices.std() / c_prices.mean() if c_prices.mean() > 0 else np.nan
    p_cv = p_prices.std() / p_prices.mean() if p_prices.mean() > 0 else np.nan
    
    # Calculate change in CV
    cv_change = c_cv - p_cv
    period_metrics[f'cv_change_{window}'] = cv_change
    
    # Calculate price change correlation with spatial lags
    try:
        # Get price changes
        price_changes = (c_prices - p_prices) / p_prices
        
        # Calculate spatial lag of price changes
        common_in_weights = common_markets.intersection(set(weights_matrix.neighbors.keys()))
        
        if len(common_in_weights) >= 4:
            # Create a subset of the weights matrix for available markets
            w_subset = weights_matrix.sparse.toarray()[list(common_in_weights)][:, list(common_in_weights)]
            w_subset = W.from_array(w_subset)
            
            # Calculate Moran's I of price changes
            from esda.moran import Moran
            moran = Moran(price_changes.loc[list(common_in_weights)], w_subset)
            
            period_metrics[f'moran_change_{window}'] = moran.I
        else:
            period_metrics[f'moran_change_{window}'] = np.nan
    except Exception:
        period_metrics[f'moran_change_{window}'] = np.nan
    
    # Calculate integration index change
    _calculate_integration_index(period_metrics, c_cv, p_cv, window,
                              period_metrics.get(f'moran_change_{window}', np.nan))


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _calculate_integration_index(period_metrics, current_cv, past_cv, window,
                              price_change_cv):
    """
    Calculate market integration index.
    
    Parameters
    ----------
    period_metrics : dict
        Metrics dictionary to update
    current_cv : float
        Current coefficient of variation
    past_cv : float
        Past coefficient of variation
    window : str
        Window label
    price_change_cv : float
        Coefficient of variation of price changes
    """
    # Calculate integration change components
    cv_component = 0.0
    
    # CV convergence component (0 to 1)
    if not np.isnan(current_cv) and not np.isnan(past_cv):
        # Positive if CV decreased (markets converged)
        if current_cv < past_cv:
            cv_component = (past_cv - current_cv) / past_cv
            cv_component = min(1.0, cv_component)
        else:
            cv_component = -min(1.0, (current_cv - past_cv) / past_cv)
    
    # Price change correlation component (-1 to 1)
    corr_component = 0.0
    if not np.isnan(price_change_cv):
        # Scale from -1,1 to 0,1
        corr_component = (price_change_cv + 1) / 2
    
    # Integration index: weighted average of components
    # Higher weight for CV component as it's more direct measure of integration
    integration_index = (0.7 * cv_component) + (0.3 * corr_component)
    
    # Constrain to -1 to 1 range
    integration_index = max(-1.0, min(1.0, integration_index))
    
    period_metrics[f'convergence_index_{window}'] = integration_index