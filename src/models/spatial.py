"""
Spatial econometric models for market integration analysis in Yemen.
Implements statistical methods for analyzing geographic market relationships
in conflict-affected environments.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from libpysal.weights import W
from esda.moran import Moran, Moran_Local
from spreg import ML_Lag, ML_Error
import matplotlib.pyplot as plt
import networkx as nx

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

# Initialize module logger
logger = logging.getLogger(__name__)

# Get configuration values
DEFAULT_CONFLICT_WEIGHT = config.get('analysis.spatial.conflict_weight', 0.5)
DEFAULT_KNN = config.get('analysis.spatial.knn', 5)
DEFAULT_CRS = config.get('analysis.spatial.crs', 32638)  # UTM Zone 38N for Yemen


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
        
        self.gdf = gdf
        self.weights = None
        self.diagnostic_hooks = {}
        self.lag_model = None
        self.error_model = None
        logger.info(f"Initialized SpatialEconometrics with {len(gdf)} observations")
    
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
        
        # Use project utility to create weights
        if conflict_adjusted:
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
        
        logger.info(f"Created weight matrix with k={k}, conflict_adjusted={conflict_adjusted}")
        return self.weights
    
    @timer
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
        
        # Calculate Local Moran's I
        local_moran = Moran_Local(self.gdf[variable], self.weights)
        
        # Create copy of GeoDataFrame to add results
        result_gdf = self.gdf.copy()
        
        # Add results to GeoDataFrame
        result_gdf['moran_local_i'] = local_moran.Is
        result_gdf['moran_p_value'] = local_moran.p_sim
        result_gdf['moran_significant'] = local_moran.p_sim < 0.05
        
        # Create cluster classification
        self._classify_moran_clusters(result_gdf, variable, local_moran)
        
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
        significant = gdf['moran_p_value'] < 0.05
        high = gdf[variable] > gdf[variable].mean()
        
        # Get the spatially lagged variable (y_lag is the correct attribute in Moran_Local)
        y_lag = local_moran.y_lag
        high_neighbors = y_lag > y_lag.mean()
        
        # Classify clusters
        gdf['cluster_type'] = 'not_significant'
        gdf.loc[significant & high & high_neighbors, 'cluster_type'] = 'high-high'
        gdf.loc[significant & ~high & ~high_neighbors, 'cluster_type'] = 'low-low'
        gdf.loc[significant & high & ~high_neighbors, 'cluster_type'] = 'high-low'
        gdf.loc[significant & ~high & high_neighbors, 'cluster_type'] = 'low-high'
    
    @timer
    @memory_usage_decorator
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
        
        # Prepare model data
        model_data, y, X = self._prepare_model_data(y_col, x_cols)
        
        # Set default names if not provided
        if name_y is None:
            name_y = y_col
        if name_x is None:
            name_x = x_cols
        
        # Estimate model
        self.lag_model = ML_Lag(y, X, self.weights, name_y=name_y, name_x=name_x)
        
        logger.info(
            f"Spatial lag model estimated: AIC={self.lag_model.aic:.4f}, "
            f"R2={self.lag_model.pr2:.4f}, rho={self.lag_model.rho:.4f}"
        )
        
        return self.lag_model
    
    @timer
    @memory_usage_decorator
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
        
        # Prepare model data
        model_data, y, X = self._prepare_model_data(y_col, x_cols)
        
        # Set default names if not provided
        if name_y is None:
            name_y = y_col
        if name_x is None:
            name_x = x_cols
        
        # Estimate model
        self.error_model = ML_Error(y, X, self.weights, name_y=name_y, name_x=name_x)
        
        logger.info(
            f"Spatial error model estimated: AIC={self.error_model.aic:.4f}, "
            f"R2={self.error_model.pr2:.4f}, lambda={self.error_model.lam:.4f}"
        )
        
        return self.error_model
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _prepare_model_data(self, y_col, x_cols):
        """
        Prepare data for spatial regression models.
        
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
        # Clean and prepare data
        model_data = self.gdf.copy()
        
        # Normalize variables for better numerical stability
        numeric_cols = [col for col in [y_col] + x_cols 
                        if model_data[col].dtype.kind in 'if']
        
        model_data = normalize_columns(
            model_data, 
            columns=numeric_cols,
            method='zscore'
        )
        
        # Extract arrays for regression
        y = model_data[y_col].values
        X = model_data[x_cols].values
        
        return model_data, y, X
    
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
        
        # Create and draw the networks
        self._draw_network_comparison(ax1, ax2, node_color_col)
        
        # Adjust layout and save if requested
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weights visualization to {save_path}")
        
        return fig
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _draw_network_comparison(self, ax1, ax2, node_color_col, 
                              base_node_size=100, edge_scale=5, labels=False):
        """
        Draw network comparison between original and conflict-adjusted weights.
        
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
        Create networkx graph from weights matrix.
        
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
        Draw network graph on axes.
        
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
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _calculate_lag_impacts(self):
        """
        Calculate impacts for spatial lag model.
        
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
        
        # Calculate impacts
        direct = {}
        indirect = {}
        total = {}
        
        # For each variable
        for i, var in enumerate(var_names):
            # Direct effect
            direct[var] = betas[i] / (1 - rho)
            
            # Indirect effect (spatial spillover)
            indirect[var] = betas[i] * rho / ((1 - rho) * (1 - rho))
            
            # Total effect
            total[var] = direct[var] + indirect[var]
        
        return {
            'direct': direct,
            'indirect': indirect,
            'total': total,
            'rho': rho,
            'model_type': 'lag'
        }
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _calculate_error_impacts(self):
        """
        Calculate impacts for spatial error model.
        
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
        
        # Calculate impacts
        direct = {}
        indirect = {}
        total = {}
        
        # For each variable
        for i, var in enumerate(var_names):
            # Direct effect is just the coefficient
            direct[var] = betas[i]
            
            # No indirect effects in spatial error model
            indirect[var] = 0
            
            # Total effect equals direct effect
            total[var] = betas[i]
        
        return {
            'direct': direct,
            'indirect': indirect,
            'total': total,
            'lambda': lambda_param,
            'model_type': 'error'
        }
    
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
        
        simulation_data = {
            'weights_matrix': self.weights,
            'original_weights': self.diagnostic_hooks.get('original_weights'),
            'gdf': self.gdf,
            'diagnostic_hooks': self.diagnostic_hooks,
            'model_type': 'spatial_econometrics'
        }
        
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
        
        # Calculate barriers using helper methods
        result_gdf = self._calculate_barrier_metrics(conflict_col, threshold)
        barrier_metrics = self._summarize_barrier_results(result_gdf, threshold)
        
        if return_gdf:
            return result_gdf
        else:
            return barrier_metrics
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _calculate_barrier_metrics(self, conflict_col, threshold):
        """
        Calculate barrier metrics for each market.
        
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
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _summarize_barrier_results(self, result_gdf, threshold):
        """
        Summarize barrier analysis results.
        
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
        total_connections = sum(len(neighbors) for neighbors in self.weights.neighbors.values())
        total_barriers = sum(result_gdf['barrier_count'])
        
        if total_connections > 0:
            barrier_rate = total_barriers / total_connections
        else:
            barrier_rate = 0
        
        barrier_metrics = {
            'total_barriers': total_barriers,
            'total_connections': total_connections,
            'barrier_rate': barrier_rate,
            'high_barrier_markets': (result_gdf['barrier_count'] > 0).sum(),
            'barrier_threshold': threshold,
            'barrier_gdf': result_gdf
        }
        
        logger.info(
            f"Identified {total_barriers} barriers out of {total_connections} connections "
            f"({barrier_rate:.1%}) using threshold {threshold}"
        )
        
        return barrier_metrics


@timer
@memory_usage_decorator
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
    
    # Use the project utility for calculating accessibility index
    result_gdf = compute_accessibility_index(
        markets_gdf=markets_gdf,
        population_gdf=population_gdf,
        max_distance=max_distance,
        distance_decay=distance_decay,
        weight_col=weight_col
    )
    
    logger.info(f"Calculated accessibility index for {len(markets_gdf)} markets")
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
@m1_optimized(parallel=True)
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
    
    # Create simulation data
    sim_data = _prepare_simulation_data(markets_gdf, conflict_reduction, conflict_col)
    
    # Run spatial analysis
    results = _run_spatial_analysis(
        markets_gdf, sim_data, conflict_col, price_col, 
        spatial_model, conflict_reduction, results
    )
    
    # Create visualizations
    _add_simulation_visualizations(results)
    
    logger.info(
        f"Simulated {conflict_reduction*100:.0f}% conflict reduction scenario. "
        f"Moran's I change: {results['metrics']['moran_I_change']:.4f}, "
        f"Price convergence: {results['metrics'].get('price_convergence_pct', 'N/A')}"
    )
    
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
        Reduction factor
    conflict_col : str
        Conflict column
    price_col : str
        Price column
    """
    valid, errors = validate_geodataframe(
        markets_gdf,
        required_columns=[conflict_col, price_col],
        min_rows=5,
        check_crs=True
    )
    raise_if_invalid(valid, errors, "Invalid market data for simulation")
    
    if not 0 <= conflict_reduction <= 1:
        raise ValidationError(f"conflict_reduction must be between 0 and 1, got {conflict_reduction}")


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _prepare_simulation_data(markets_gdf, conflict_reduction, conflict_col):
    """
    Prepare data for connectivity simulation.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data
    conflict_reduction : float
        Reduction factor
    conflict_col : str
        Conflict column
        
    Returns
    -------
    geopandas.GeoDataFrame
        Simulation data
    """
    # Create a copy of the data for the simulation
    sim_data = markets_gdf.copy()
    
    # Apply conflict reduction
    reduced_col = f'{conflict_col}_reduced'
    sim_data[reduced_col] = sim_data[conflict_col] * (1 - conflict_reduction)
    
    return sim_data


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _run_spatial_analysis(markets_gdf, sim_data, conflict_col, price_col, 
                       spatial_model, conflict_reduction, results):
    """
    Run spatial analysis for simulation.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Original market data
    sim_data : geopandas.GeoDataFrame
        Simulation data with reduced conflict
    conflict_col : str
        Conflict column
    price_col : str
        Price column
    spatial_model : SpatialEconometrics or None
        Existing spatial model
    conflict_reduction : float
        Conflict reduction factor
    results : dict
        Results dictionary to update
        
    Returns
    -------
    dict
        Updated results
    """
    # Create spatial model for original data if not provided
    if spatial_model is None:
        original_model = SpatialEconometrics(markets_gdf)
    else:
        original_model = spatial_model
    
    # Create spatial model for simulation data
    sim_model = SpatialEconometrics(sim_data)
    
    # Create weight matrices
    original_weights = original_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=conflict_col
    )
    
    # Create weight matrices with reduced conflict
    reduced_col = f'{conflict_col}_reduced'
    sim_weights = sim_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=reduced_col
    )
    
    # Test for spatial autocorrelation
    original_moran = original_model.moran_i_test(price_col)
    sim_moran = sim_model.moran_i_test(price_col)
    
    # Store Moran's I results
    results['metrics']['moran_I_original'] = original_moran['I']
    results['metrics']['moran_I_simulated'] = sim_moran['I']
    results['metrics']['moran_I_change'] = sim_moran['I'] - original_moran['I']
    results['metrics']['moran_pvalue_original'] = original_moran['p_norm']
    results['metrics']['moran_pvalue_simulated'] = sim_moran['p_norm']
    
    # Estimate spatial lag models if possible
    try:
        # Estimate models with common variables
        if 'distance' in markets_gdf.columns and 'population' in markets_gdf.columns:
            x_cols = ['distance', 'population']
            
            # Estimate original model
            original_lag = original_model.spatial_lag_model(price_col, x_cols)
            results['models']['original_lag'] = {
                'rho': original_lag.rho,
                'betas': original_lag.betas.tolist(),
                'aic': original_lag.aic
            }
            
            # Estimate simulation model
            sim_lag = sim_model.spatial_lag_model(price_col, x_cols)
            results['models']['simulated_lag'] = {
                'rho': sim_lag.rho,
                'betas': sim_lag.betas.tolist(),
                'aic': sim_lag.aic
            }
            
            # Add impacts if available
            try:
                original_impacts = original_model.calculate_impacts(model_type='lag')
                sim_impacts = sim_model.calculate_impacts(model_type='lag')
                
                results['models']['original_impacts'] = original_impacts
                results['models']['simulated_impacts'] = sim_impacts
                
                # Calculate impact changes
                impact_changes = {
                    'direct': {},
                    'indirect': {},
                    'total': {}
                }
                
                for var in original_impacts['direct'].keys():
                    for effect_type in ['direct', 'indirect', 'total']:
                        orig_val = original_impacts[effect_type][var]
                        sim_val = sim_impacts[effect_type][var]
                        
                        if orig_val != 0:
                            pct_change = (sim_val - orig_val) / abs(orig_val) * 100
                        else:
                            pct_change = 0
                            
                        impact_changes[effect_type][var] = {
                            'original': orig_val,
                            'simulated': sim_val,
                            'absolute_change': sim_val - orig_val,
                            'percent_change': pct_change
                        }
                
                results['models']['impact_changes'] = impact_changes
                
            except Exception as e:
                logger.warning(f"Could not calculate impact changes: {e}")
    except Exception as e:
        logger.warning(f"Could not estimate spatial models: {e}")
    
    # Add exchange rate regime analysis if available
    if 'exchange_rate_regime' in markets_gdf.columns:
        # Add price differential metrics
        _add_price_differential_metrics(
            markets_gdf, conflict_reduction, price_col, results
        )
    
    return results


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _add_price_differential_metrics(markets_gdf, conflict_reduction, price_col, results):
    """
    Add price differential metrics to simulation results.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market data
    conflict_reduction : float
        Conflict reduction factor
    price_col : str
        Price column
    results : dict
        Results dictionary to update
    """
    # Prepare data by exchange rate regime
    north_original = markets_gdf[markets_gdf['exchange_rate_regime'] == 'north']
    south_original = markets_gdf[markets_gdf['exchange_rate_regime'] == 'south']
    
    if len(north_original) == 0 or len(south_original) == 0:
        logger.warning("Cannot calculate price differentials: missing north or south data")
        return
    
    # Calculate original price differentials
    north_price = north_original[price_col].mean()
    south_price = south_original[price_col].mean()
    price_diff_original = abs(north_price - south_price)
    price_diff_pct_original = price_diff_original / ((north_price + south_price) / 2) * 100
    
    # Estimate model parameters if available
    model_data_available = (
        'models' in results and 
        'original_lag' in results['models'] and 
        'simulated_lag' in results['models']
    )
    
    if model_data_available:
        original_rho = results['models']['original_lag']['rho']
        sim_rho = results['models']['simulated_lag']['rho']
        
        if original_rho != 0:
            rho_improvement = (sim_rho - original_rho) / original_rho
        else:
            rho_improvement = 0
        
        # Estimate reduced price differential
        price_diff_simulated = price_diff_original * (1 - rho_improvement * conflict_reduction)
        price_diff_pct_simulated = price_diff_pct_original * (1 - rho_improvement * conflict_reduction)
        
        # Add metrics to results
        results['metrics']['price_diff_original'] = price_diff_original
        results['metrics']['price_diff_simulated'] = price_diff_simulated
        results['metrics']['price_diff_pct_original'] = price_diff_pct_original
        results['metrics']['price_diff_pct_simulated'] = price_diff_pct_simulated
        
        # Calculate price convergence percentage
        if price_diff_original > 0:
            results['metrics']['price_convergence_pct'] = (
                (price_diff_original - price_diff_simulated) / price_diff_original * 100
            )
        else:
            results['metrics']['price_convergence_pct'] = 0


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _add_simulation_visualizations(results):
    """
    Add visualizations to simulation results.
    
    Parameters
    ----------
    results : dict
        Simulation results to update with visualizations
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create visualization of conflict reduction impact
        figs = {}
        
        # Plot Moran's I comparison
        fig_moran, ax_moran = plt.subplots(figsize=(8, 6))
        moran_values = [
            results['metrics']['moran_I_original'], 
            results['metrics']['moran_I_simulated']
        ]
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
            reduction = results['reduction_factor'] * 100
            price_labels = ['Original', f"{reduction:.0f}% Conflict Reduction"]
            ax_price.bar(price_labels, price_diff_values, color=['red', 'green'])
            ax_price.set_title("Impact of Conflict Reduction on Price Differentials")
            ax_price.set_ylabel("Price Differential")
            ax_price.grid(True, alpha=0.3)
            figs['price_diff_comparison'] = fig_price
        
        results['visualizations'] = figs
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


@timer
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
def market_integration_index(prices_df: pd.DataFrame, weights_matrix, market_id_col: str, 
                          price_col: str = 'price', time_col: str = 'date', windows: Optional[List[int]] = None,
                          return_components: bool = False) -> pd.DataFrame:
    """
    Calculate time-varying market integration metrics.
    
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
    return_components : bool, optional
        Whether to return component metrics
        
    Returns
    -------
    pandas.DataFrame
        Time series of integration indices
    """
    # Configure system for optimal performance
    configure_system_for_performance()
    
    # Validate inputs
    _validate_integration_inputs(prices_df, weights_matrix, market_id_col, price_col, time_col)
    
    # Optimize memory usage of input DataFrame
    prices_df = optimize_dataframe(prices_df)
    
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
            f"Not enough time periods ({len(time_periods)}) for window analysis "
            f"with max window {max(windows)}"
        )
    
    # Use parallel processing if we have a significant number of periods
    start_time = time.time()
    if len(time_periods) > 10:
        logger.info(f"Using parallel processing for {len(time_periods)} time periods")
        results = _process_periods_parallel(
            prices_df, time_periods, windows, markets, 
            market_id_col, price_col, time_col, weights_matrix, return_components
        )
    else:
        # Initialize results for sequential processing
        results = []
        
        # Process each time period sequentially
        for t, period in enumerate(time_periods):
            # Skip periods that don't have enough history
            if t < max(windows):
                continue
            
            # Get data for current period
            period_data = prices_df[prices_df[time_col] == period]
            
            # Need data for all markets
            if len(period_data) < len(markets):
                logger.warning(f"Skipping period {period} with incomplete market data")
                continue
            
            # Calculate metrics for this period
            period_metrics = _calculate_period_metrics(
                period_data, prices_df, time_periods, t, windows,
                market_id_col, price_col, time_col, weights_matrix, period,
                return_components
            )
            
            # Add to results
            results.append(period_metrics)
    
    # Convert to DataFrame and optimize
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(time_col)
        results_df = optimize_dataframe(results_df)  # Optimize output dataframe
    
    processing_time = time.time() - start_time
    logger.info(f"Calculated market integration index for {len(results_df)} time periods in {processing_time:.2f} seconds")
    return results_df


@m1_optimized(parallel=True)
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _process_periods_parallel(
    prices_df: pd.DataFrame, time_periods: List, windows: List[int], markets: List,
    market_id_col: str, price_col: str, time_col: str, weights_matrix,
    return_components: bool
) -> List[Dict[str, Any]]:
    """
    Process time periods in parallel for market integration calculation.
    
    Parameters
    ----------
    prices_df : pandas.DataFrame
        DataFrame with market prices over time
    time_periods : list
        List of time periods to process
    windows : list
        Window sizes for analysis
    markets : list
        List of market IDs
    market_id_col : str
        Column identifying markets
    price_col : str
        Column containing price data
    time_col : str
        Column containing time data
    weights_matrix : libpysal.weights.W
        Spatial weights
    return_components : bool
        Whether to return component metrics
        
    Returns
    -------
    list
        List of period metrics dictionaries
    """
    # Create DataFrame with time periods that have enough history
    max_window = max(windows)
    valid_periods = []
    
    for t, period in enumerate(time_periods):
        if t >= max_window:
            valid_periods.append({
                'period': period, 
                'index': t,
                'time_col': time_col  # Include for identification in results
            })
            
    # Convert to DataFrame for parallelization
    periods_df = pd.DataFrame(valid_periods)
    
    # Define function to process each chunk
    def process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
        chunk_results = []
        
        for _, row in chunk_df.iterrows():
            period = row['period']
            t = row['index']
            
            # Get data for current period
            period_data = prices_df[prices_df[time_col] == period]
            
            # Skip if we don't have data for all markets
            if len(period_data) < len(markets):
                continue
                
            # Calculate metrics
            period_metrics = _calculate_period_metrics(
                period_data, prices_df, time_periods, t, windows,
                market_id_col, price_col, time_col, weights_matrix, period,
                return_components
            )
            
            chunk_results.append(period_metrics)
            
        return pd.DataFrame(chunk_results) if chunk_results else pd.DataFrame()
    
    # Process in parallel
    results_dfs = parallelize_dataframe(periods_df, process_chunk)
    
    # Convert results to list of dictionaries
    return results_dfs.to_dict('records') if not results_dfs.empty else []


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _validate_integration_inputs(prices_df, weights_matrix, market_id_col, price_col, time_col):
    """
    Validate inputs for market integration analysis.
    
    Parameters
    ----------
    prices_df : pandas.DataFrame
        Price data
    weights_matrix : libpysal.weights.W
        Weights matrix
    market_id_col : str
        Market ID column
    price_col : str
        Price column
    time_col : str
        Time column
    """
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


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _calculate_period_metrics(period_data, prices_df, time_periods, t, windows,
                          market_id_col, price_col, time_col, weights_matrix, period,
                          return_components=False):
    """
    Calculate market integration metrics for a specific time period.
    
    Parameters
    ----------
    period_data : pandas.DataFrame
        Data for current period
    prices_df : pandas.DataFrame
        Full price dataset
    time_periods : list
        Sorted list of time periods
    t : int
        Index of current period
    windows : list
        Window sizes for analysis
    market_id_col : str
        Market ID column
    price_col : str
        Price column
    time_col : str
        Time column
    weights_matrix : libpysal.weights.W
        Spatial weights
    period : datetime-like
        Current time period
    return_components : bool, optional
        Whether to return component metrics
        
    Returns
    -------
    dict
        Period metrics
    """
    # Create price vector
    price_vector = period_data.set_index(market_id_col)[price_col]
    
    # Initialize period metrics
    period_metrics = {
        time_col: period,
        'num_markets': len(period_data)
    }
    
    # Add current period metrics
    _add_current_period_metrics(period_metrics, price_vector, weights_matrix)
    
    # Add window-based metrics
    for window in windows:
        _add_window_metrics(
            period_metrics, prices_df, time_periods, t, window,
            market_id_col, price_col, price_vector, return_components
        )
    
    return period_metrics


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _add_current_period_metrics(period_metrics, price_vector, weights_matrix):
    """
    Add current period metrics to results.
    
    Parameters
    ----------
    period_metrics : dict
        Metrics to update
    price_vector : pandas.Series
        Prices for current period
    weights_matrix : libpysal.weights.W
        Spatial weights
    """
    # Calculate coefficient of variation (price dispersion)
    if price_vector.mean() > 0:
        period_metrics['price_cv'] = price_vector.std() / price_vector.mean()
    else:
        period_metrics['price_cv'] = 0
    
    # Calculate Moran's I for spatial price autocorrelation
    try:
        moran = Moran(price_vector, weights_matrix)
        period_metrics['moran_I'] = moran.I
        period_metrics['moran_pvalue'] = moran.p_norm
        period_metrics['significant_autocorrelation'] = moran.p_norm < 0.05
    except Exception as e:
        logger.warning(f"Could not calculate Moran's I: {e}")
        period_metrics['moran_I'] = None
        period_metrics['moran_pvalue'] = None
        period_metrics['significant_autocorrelation'] = False


@timer
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _add_window_metrics(period_metrics, prices_df, time_periods, t, window,
                     market_id_col, price_col, current_prices, return_components=False):
    """
    Add window-based metrics to results.
    
    Parameters
    ----------
    period_metrics : dict
        Metrics to update
    prices_df : pandas.DataFrame
        Full price dataset
    time_periods : list
        Sorted time periods
    t : int
        Current period index
    window : int
        Window size
    market_id_col : str
        Market ID column
    price_col : str
        Price column
    current_prices : pandas.Series
        Current period prices
    return_components : bool, optional
        Whether to return component metrics
    """
    # Log window processing start
    start_time = time.time()
    
    # Get data for window periods ago
    past_period = time_periods[t - window]
    past_data = prices_df[prices_df[time_col] == past_period]
    
    # Skip if we don't have complete data
    if len(past_data) < period_metrics['num_markets']:
        logger.debug(f"Skipping window {window} for period {period_metrics.get(time_col)}: incomplete data")
        return
    
    # Get price vector for past period
    past_prices = past_data.set_index(market_id_col)[price_col]
    
    # Ensure we have matched markets
    common_markets = set(current_prices.index) & set(past_prices.index)
    if len(common_markets) < 2:
        logger.debug(f"Skipping window {window} for period {period_metrics.get(time_col)}: fewer than 2 common markets")
        return
        
    # Extract comparable price vectors
    curr_prices = current_prices.loc[common_markets]
    past_prices = past_prices.loc[common_markets]
    
    # Calculate price convergence metrics
    _calculate_convergence_metrics(
        period_metrics, curr_prices, past_prices, window, return_components
    )
    
    # Log window processing time
    processing_time = time.time() - start_time
    logger.debug(f"Window {window} metrics calculated in {processing_time:.4f} seconds")


@timer
@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _calculate_convergence_metrics(period_metrics, current_prices, past_prices, window,
                               return_components=False):
    """
    Calculate price convergence metrics between two periods.
    
    Parameters
    ----------
    period_metrics : dict
        Metrics to update
    current_prices : pandas.Series
        Current period prices
    past_prices : pandas.Series
        Past period prices
    window : int
        Window size
    return_components : bool, optional
        Whether to return component metrics
    """
    # Log memory usage before calculation
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    
    # Price dispersion metrics
    if current_prices.mean() > 0:
        current_cv = current_prices.std() / current_prices.mean()
    else:
        current_cv = 0
        
    if past_prices.mean() > 0:
        past_cv = past_prices.std() / past_prices.mean()
    else:
        past_cv = 0
    
    # Calculate CV change
    if past_cv > 0:
        cv_change = (current_cv - past_cv) / past_cv
    else:
        cv_change = 0
    
    # Calculate price changes
    price_changes = (current_prices - past_prices) / past_prices
    avg_price_change = price_changes.mean()
    
    # Calculate price change dispersion
    if avg_price_change != 0:
        price_change_cv = price_changes.std() / abs(avg_price_change)
    else:
        price_change_cv = float('inf')
    
    # Store window-specific metrics
    period_metrics[f'price_cv_change_w{window}'] = cv_change
    period_metrics[f'avg_price_change_w{window}'] = avg_price_change
    period_metrics[f'price_change_cv_w{window}'] = price_change_cv
    
    # Calculate Market Integration Index
    _calculate_integration_index(
        period_metrics, current_cv, price_change_cv, window, return_components
    )
    
    # Log memory usage after calculation
    end_mem = process.memory_info().rss / (1024 * 1024)
    logger.debug(f"Convergence metrics for window {window} used {end_mem - start_mem:.2f} MB memory")


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _calculate_integration_index(period_metrics, current_cv, price_change_cv, window,
                             return_components=False):
    """
    Calculate market integration index from components.
    
    Parameters
    ----------
    period_metrics : dict
        Metrics to update
    current_cv : float
        Current coefficient of variation
    price_change_cv : float
        Price change coefficient of variation
    window : int
        Window size
    return_components : bool, optional
        Whether to return component metrics
    """
    # Normalize CV (lower is better, range 0-1)
    norm_cv = max(0, 1 - min(current_cv * 5, 1))
    
    # Normalize Moran's I (higher positive values are better)
    if period_metrics['moran_I'] is not None:
        norm_moran = max(0, min(1, (period_metrics['moran_I'] + 1) / 2))
    else:
        norm_moran = 0.5
    
    # Normalize price change dispersion (lower is better)
    norm_price_change = max(0, 1 - min(price_change_cv * 0.2, 1))
    
    # Combined index (simple average of components)
    integration_index = (norm_cv + norm_moran + norm_price_change) / 3
    period_metrics[f'integration_index_w{window}'] = integration_index
    
    # Store components if requested
    if return_components:
        period_metrics[f'integration_cv_component_w{window}'] = norm_cv
        period_metrics[f'integration_moran_component_w{window}'] = norm_moran
        period_metrics[f'integration_price_change_component_w{window}'] = norm_price_change