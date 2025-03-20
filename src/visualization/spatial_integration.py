"""
Spatial visualization module for market integration analysis.

This module provides specialized visualizations for spatial market integration patterns,
including conflict-adjusted market networks, choropleth maps for integration metrics,
and combined spatial-temporal visualizations.
"""
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import contextily as ctx
from typing import Optional, Tuple, List, Dict, Any, Union
import os

from src.utils import (
    handle_errors,
    config,
    validate_dataframe,
    validate_geodataframe,
    raise_if_invalid,
    set_plotting_style,
    save_plot,
    VisualizationError
)

logger = logging.getLogger(__name__)

class SpatialIntegrationVisualizer:
    """Specialized visualizations for spatial market integration patterns."""
    
    def __init__(self):
        """Initialize the visualizer with default styling."""
        set_plotting_style()
        
        # Get styling parameters from config
        self.fig_width = config.get('visualization.default_fig_width', 12)
        self.fig_height = config.get('visualization.default_fig_height', 8)
        self.dpi = config.get('visualization.figure_dpi', 300)
        self.save_dir = config.get('visualization.save_dir', 'results/plots')
        
        # Colors for different integration levels
        self.integrated_color = config.get('visualization.integrated_color', '#1a9641')
        self.partial_color = config.get('visualization.partial_color', '#fdae61')
        self.not_integrated_color = config.get('visualization.not_integrated_color', '#d7191c')
        self.north_color = config.get('visualization.north_color', '#1f77b4')
        self.south_color = config.get('visualization.south_color', '#ff7f0e')
        
        # Check if contextily is available for adding basemaps
        try:
            import contextily as ctx
            self.has_contextily = True
        except ImportError:
            logger.warning("Contextily not available. Basemaps will not be added to maps.")
            self.has_contextily = False
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError))
    def plot_market_network(
        self,
        market_gdf: gpd.GeoDataFrame,
        edges_gdf: Optional[gpd.GeoDataFrame] = None,
        market_id_col: str = 'market_id',
        market_name_col: Optional[str] = 'market_name',
        region_col: Optional[str] = 'exchange_rate_regime',
        edge_color_col: Optional[str] = 'integration_level_num',
        edge_width_col: Optional[str] = None,
        title: Optional[str] = None,
        basemap: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize market integration network.
        
        Parameters
        ----------
        market_gdf : gpd.GeoDataFrame
            GeoDataFrame containing market locations
        edges_gdf : gpd.GeoDataFrame, optional
            GeoDataFrame containing market pair connections
        market_id_col : str, optional
            Column with market IDs, default 'market_id'
        market_name_col : str, optional
            Column with market names, default 'market_name'
        region_col : str, optional
            Column with exchange rate regime, default 'exchange_rate_regime'
        edge_color_col : str, optional
            Column to color edges by, default 'integration_level_num'
        edge_width_col : str, optional
            Column to set edge width by, default None
        title : str, optional
            Plot title, default None
        basemap : bool, optional
            Whether to add a basemap, default True
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        ax : plt.Axes
            The matplotlib axes
        """
        # Validate input data
        valid, errors = validate_geodataframe(market_gdf, required_columns=[market_id_col])
        raise_if_invalid(valid, errors, "Invalid market geodataframe")
        
        if edges_gdf is not None:
            valid, errors = validate_geodataframe(edges_gdf)
            raise_if_invalid(valid, errors, "Invalid edges geodataframe")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Plot edges if provided
        if edges_gdf is not None:
            # Set edge colors based on integration level
            if edge_color_col in edges_gdf.columns:
                edge_cmap = colors.LinearSegmentedColormap.from_list(
                    'integration_cmap',
                    [self.not_integrated_color, self.partial_color, self.integrated_color],
                    N=3
                )
                
                # Normalize edge_color_col values to 0-1 for colormap
                if edges_gdf[edge_color_col].nunique() <= 3:
                    # Discrete values for integration levels
                    norm = colors.Normalize(vmin=-0.5, vmax=2.5)
                else:
                    # Continuous values
                    norm = colors.Normalize(
                        vmin=edges_gdf[edge_color_col].min(),
                        vmax=edges_gdf[edge_color_col].max()
                    )
                
                # Set edge widths
                edge_widths = 1.0
                if edge_width_col is not None and edge_width_col in edges_gdf.columns:
                    # Scale edge widths between 0.5 and 3.0
                    min_width = 0.5
                    max_width = 3.0
                    
                    if edges_gdf[edge_width_col].min() != edges_gdf[edge_width_col].max():
                        edge_widths = min_width + (
                            (edges_gdf[edge_width_col] - edges_gdf[edge_width_col].min()) / 
                            (edges_gdf[edge_width_col].max() - edges_gdf[edge_width_col].min())
                        ) * (max_width - min_width)
                    else:
                        edge_widths = (min_width + max_width) / 2
                
                # Plot edges with color and width
                edges_gdf.plot(
                    ax=ax,
                    column=edge_color_col,
                    cmap=edge_cmap,
                    norm=norm,
                    linewidth=edge_widths,
                    alpha=0.7
                )
                
                # Add colorbar for edges
                sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
                
                # Set colorbar label based on column name
                if edge_color_col == 'integration_level_num':
                    cbar.set_label('Integration Level')
                    
                    # Set custom colorbar ticks for integration levels
                    if edges_gdf[edge_color_col].nunique() <= 3:
                        cbar.set_ticks([0, 1, 2])
                        cbar.set_ticklabels(['Not Integrated', 'Symmetric', 'Asymmetric'])
                else:
                    cbar.set_label(edge_color_col.replace('_', ' ').title())
            else:
                # Plot edges without color mapping
                edges_gdf.plot(ax=ax, color='gray', linewidth=1.0, alpha=0.5)
        
        # Plot markets with different colors by region
        if region_col is not None and region_col in market_gdf.columns:
            # Get unique regions
            regions = market_gdf[region_col].unique()
            
            # Define colors for regions (default to north/south)
            region_colors = {}
            if 'north' in regions and 'south' in regions:
                region_colors = {
                    'north': self.north_color,
                    'south': self.south_color
                }
            
            # Add colors for any other regions
            for region in regions:
                if region not in region_colors:
                    region_colors[region] = plt.cm.tab10(len(region_colors) % 10)
            
            # Plot markets by region
            for region in regions:
                region_data = market_gdf[market_gdf[region_col] == region]
                region_data.plot(
                    ax=ax,
                    color=region_colors[region],
                    markersize=50,
                    marker='o',
                    label=region
                )
                
                # Add market labels if name column is provided
                if market_name_col is not None and market_name_col in region_data.columns:
                    for idx, row in region_data.iterrows():
                        ax.annotate(
                            row[market_name_col],
                            xy=(row.geometry.x, row.geometry.y),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            ha='left',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                        )
        else:
            # Plot all markets with same style
            market_gdf.plot(
                ax=ax,
                color='blue',
                markersize=50,
                marker='o'
            )
            
            # Add market labels if name column is provided
            if market_name_col is not None and market_name_col in market_gdf.columns:
                for idx, row in market_gdf.iterrows():
                    ax.annotate(
                        row[market_name_col],
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        ha='left',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                    )
        
        # Add basemap if enabled and contextily is available
        if basemap and self.has_contextily:
            try:
                # Reproject to Web Mercator for basemap
                if market_gdf.crs != 'EPSG:3857':
                    ax_crs = market_gdf.to_crs('EPSG:3857').plot(ax=ax, alpha=0)
                
                # Add basemap
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.Positron,
                    zoom=10
                )
            except Exception as e:
                logger.warning(f"Could not add basemap: {e}")
        
        # Add legend if regions were used
        if region_col is not None and region_col in market_gdf.columns:
            # Create legend handles for regions
            region_handles = [
                Patch(facecolor=region_colors[region], label=region.upper())
                for region in regions
            ]
            
            # Add integration level to legend if used
            if edges_gdf is not None and edge_color_col == 'integration_level_num':
                # Add integration level legend entries
                integration_handles = [
                    Line2D([0], [0], color=self.not_integrated_color, lw=2, label='Not Integrated'),
                    Line2D([0], [0], color=self.partial_color, lw=2, label='Symmetric Integration'),
                    Line2D([0], [0], color=self.integrated_color, lw=2, label='Asymmetric Integration')
                ]
                
                # Combine handles
                all_handles = region_handles + integration_handles
                ax.legend(handles=all_handles, loc='upper right')
            else:
                ax.legend(handles=region_handles, loc='upper right')
        
        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Market Integration Network')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
        
        return fig, ax
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError))
    def plot_integration_choropleth(
        self,
        market_gdf: gpd.GeoDataFrame,
        metric_col: str,
        market_id_col: str = 'market_id',
        region_col: Optional[str] = 'exchange_rate_regime',
        title: Optional[str] = None,
        cmap: Optional[str] = 'RdYlGn',
        basemap: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create choropleth map for integration metrics.
        
        Parameters
        ----------
        market_gdf : gpd.GeoDataFrame
            GeoDataFrame containing market locations and metrics
        metric_col : str
            Column with integration metric for coloring
        market_id_col : str, optional
            Column with market IDs, default 'market_id'
        region_col : str, optional
            Column with exchange rate regime, default 'exchange_rate_regime'
        title : str, optional
            Plot title, default None
        cmap : str, optional
            Colormap name, default 'RdYlGn'
        basemap : bool, optional
            Whether to add a basemap, default True
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        ax : plt.Axes
            The matplotlib axes
        """
        # Validate input data
        valid, errors = validate_geodataframe(
            market_gdf, 
            required_columns=[market_id_col, metric_col]
        )
        raise_if_invalid(valid, errors, "Invalid market geodataframe for choropleth")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Get market buffer regions for choropleth
        # This creates polygon areas around market points
        markets_buffer = market_gdf.copy()
        
        # Create buffers with sizes based on market importance or fixed size
        if 'market_size' in markets_buffer.columns:
            # Buffer size based on market size
            buffer_sizes = markets_buffer['market_size'] * 0.05
            markets_buffer['geometry'] = markets_buffer.geometry.buffer(buffer_sizes)
        else:
            # Fixed buffer size
            markets_buffer['geometry'] = markets_buffer.geometry.buffer(0.05)
        
        # Plot choropleth
        markets_buffer.plot(
            ax=ax,
            column=metric_col,
            cmap=cmap,
            legend=True,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add market labels
        if 'market_name' in market_gdf.columns:
            for idx, row in market_gdf.iterrows():
                ax.annotate(
                    row['market_name'],
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                )
        
        # Add region borders if region column is provided
        if region_col in market_gdf.columns:
            # Dissolve by region to get region polygons
            regions = markets_buffer.dissolve(by=region_col)
            
            # Plot region boundaries
            regions.boundary.plot(
                ax=ax,
                color='black',
                linewidth=1.5,
                linestyle='--'
            )
            
            # Add region labels at centroids
            for region_name, region_data in regions.iterrows():
                centroid = region_data.geometry.centroid
                ax.annotate(
                    region_name.upper(),
                    xy=(centroid.x, centroid.y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5)
                )
        
        # Add basemap if enabled and contextily is available
        if basemap and self.has_contextily:
            try:
                # Reproject to Web Mercator for basemap
                if market_gdf.crs != 'EPSG:3857':
                    ax_crs = market_gdf.to_crs('EPSG:3857').plot(ax=ax, alpha=0)
                
                # Add basemap
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.Positron,
                    zoom=10
                )
            except Exception as e:
                logger.warning(f"Could not add basemap: {e}")
        
        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Market Integration: {metric_col.replace("_", " ").title()}')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
        
        return fig, ax
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError))
    def plot_conflict_adjusted_network(
        self,
        market_gdf: gpd.GeoDataFrame,
        conflict_col: str = 'conflict_intensity_normalized',
        region_col: Optional[str] = 'exchange_rate_regime',
        market_id_col: str = 'market_id',
        title: Optional[str] = None,
        basemap: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize conflict-adjusted market network.
        
        Parameters
        ----------
        market_gdf : gpd.GeoDataFrame
            GeoDataFrame containing market locations and conflict data
        conflict_col : str, optional
            Column with conflict intensity, default 'conflict_intensity_normalized'
        region_col : str, optional
            Column with exchange rate regime, default 'exchange_rate_regime'
        market_id_col : str, optional
            Column with market IDs, default 'market_id'
        title : str, optional
            Plot title, default None
        basemap : bool, optional
            Whether to add a basemap, default True
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        ax : plt.Axes
            The matplotlib axes
        """
        # Validate input data
        valid, errors = validate_geodataframe(
            market_gdf, 
            required_columns=[market_id_col, conflict_col]
        )
        raise_if_invalid(valid, errors, "Invalid market geodataframe for conflict network")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Create a colormap for conflict intensity
        conflict_cmap = plt.cm.YlOrRd
        
        # Create connectivity edges between markets
        from scipy.spatial import Delaunay
        
        # Get market coordinates
        coords = np.array([(p.x, p.y) for p in market_gdf.geometry])
        
        # Check if we have enough points for triangulation
        if len(coords) >= 3:
            # Create Delaunay triangulation
            try:
                tri = Delaunay(coords)
                
                # Get connections from triangulation
                edges = set()
                for simplex in tri.simplices:
                    edges.add((simplex[0], simplex[1]))
                    edges.add((simplex[1], simplex[2]))
                    edges.add((simplex[0], simplex[2]))
                
                # Draw edges with conflict-adjusted colors and widths
                for i, j in edges:
                    # Get market ids
                    market_i = market_gdf.iloc[i]
                    market_j = market_gdf.iloc[j]
                    
                    # Calculate average conflict intensity for this link
                    conflict_ij = (market_i[conflict_col] + market_j[conflict_col]) / 2
                    
                    # Scale conflict intensity to 0-1
                    conflict_normalized = conflict_ij / market_gdf[conflict_col].max()
                    
                    # Get color based on conflict intensity
                    edge_color = conflict_cmap(conflict_normalized)
                    
                    # Get line width based on inverse of conflict intensity
                    # Higher conflict = thinner line (more barriers)
                    line_width = 3 * (1 - conflict_normalized)
                    
                    # Draw the edge
                    ax.plot(
                        [coords[i][0], coords[j][0]], 
                        [coords[i][1], coords[j][1]],
                        color=edge_color,
                        linewidth=line_width,
                        alpha=0.7,
                        zorder=1
                    )
            except Exception as e:
                logger.warning(f"Could not create Delaunay triangulation: {e}")
        
        # Plot markets with different colors by region
        if region_col is not None and region_col in market_gdf.columns:
            # Get unique regions
            regions = market_gdf[region_col].unique()
            
            # Define colors for regions (default to north/south)
            region_colors = {}
            if 'north' in regions and 'south' in regions:
                region_colors = {
                    'north': self.north_color,
                    'south': self.south_color
                }
            
            # Add colors for any other regions
            for region in regions:
                if region not in region_colors:
                    region_colors[region] = plt.cm.tab10(len(region_colors) % 10)
            
            # Plot markets by region
            for region in regions:
                region_data = market_gdf[market_gdf[region_col] == region]
                
                # Scale market size by inverse of conflict intensity
                if conflict_col in region_data.columns:
                    # Higher conflict = smaller market size
                    conflict_normalized = region_data[conflict_col] / market_gdf[conflict_col].max()
                    market_sizes = 100 * (1 - conflict_normalized)
                else:
                    market_sizes = 50
                
                # Plot markets
                region_data.plot(
                    ax=ax,
                    color=region_colors[region],
                    markersize=market_sizes,
                    marker='o',
                    label=region,
                    zorder=2
                )
        else:
            # Scale market size by inverse of conflict intensity
            if conflict_col in market_gdf.columns:
                # Higher conflict = smaller market size
                conflict_normalized = market_gdf[conflict_col] / market_gdf[conflict_col].max()
                market_sizes = 100 * (1 - conflict_normalized)
            else:
                market_sizes = 50
                
            # Plot all markets with same style but varying size
            market_gdf.plot(
                ax=ax,
                color='blue',
                markersize=market_sizes,
                marker='o',
                zorder=2
            )
        
        # Add market labels
        if 'market_name' in market_gdf.columns:
            for idx, row in market_gdf.iterrows():
                ax.annotate(
                    row['market_name'],
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    ha='left',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                    zorder=3
                )
        
        # Add basemap if enabled and contextily is available
        if basemap and self.has_contextily:
            try:
                # Reproject to Web Mercator for basemap
                if market_gdf.crs != 'EPSG:3857':
                    ax_crs = market_gdf.to_crs('EPSG:3857').plot(ax=ax, alpha=0)
                
                # Add basemap
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.Positron,
                    zoom=10
                )
            except Exception as e:
                logger.warning(f"Could not add basemap: {e}")
        
        # Add legend for regions if used
        if region_col is not None and region_col in market_gdf.columns:
            # Create legend handles for regions
            region_handles = [
                Patch(facecolor=region_colors[region], label=region.upper())
                for region in regions
            ]
            
            # Add conflict intensity to legend
            conflict_handles = [
                Line2D([0], [0], color=conflict_cmap(0.0), lw=3, label='Low Conflict'),
                Line2D([0], [0], color=conflict_cmap(0.5), lw=2, label='Medium Conflict'),
                Line2D([0], [0], color=conflict_cmap(1.0), lw=1, label='High Conflict')
            ]
            
            # Combine handles
            all_handles = region_handles + conflict_handles
            ax.legend(handles=all_handles, loc='upper right')
        else:
            # Just add conflict legend
            conflict_handles = [
                Line2D([0], [0], color=conflict_cmap(0.0), lw=3, label='Low Conflict'),
                Line2D([0], [0], color=conflict_cmap(0.5), lw=2, label='Medium Conflict'),
                Line2D([0], [0], color=conflict_cmap(1.0), lw=1, label='High Conflict')
            ]
            ax.legend(handles=conflict_handles, loc='upper right')
        
        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Conflict-Adjusted Market Network')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
        
        return fig, ax
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError))
    def plot_market_integration_comparison(
        self,
        original_gdf: gpd.GeoDataFrame,
        simulated_gdf: gpd.GeoDataFrame,
        metric_col: str,
        market_id_col: str = 'market_id',
        region_col: Optional[str] = 'exchange_rate_regime',
        title: Optional[str] = None,
        cmap: Optional[str] = 'RdYlGn',
        basemap: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Compare spatial integration patterns before and after policy intervention.
        
        Parameters
        ----------
        original_gdf : gpd.GeoDataFrame
            GeoDataFrame with original integration metrics
        simulated_gdf : gpd.GeoDataFrame
            GeoDataFrame with simulated integration metrics
        metric_col : str
            Column with integration metric for comparison
        market_id_col : str, optional
            Column with market IDs, default 'market_id'
        region_col : str, optional
            Column with exchange rate regime, default 'exchange_rate_regime'
        title : str, optional
            Plot title, default None
        cmap : str, optional
            Colormap name, default 'RdYlGn'
        basemap : bool, optional
            Whether to add a basemap, default True
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        axs : list of plt.Axes
            The matplotlib axes
        """
        # Validate input data
        for gdf, label in [(original_gdf, 'original'), (simulated_gdf, 'simulated')]:
            valid, errors = validate_geodataframe(
                gdf, 
                required_columns=[market_id_col, metric_col]
            )
            raise_if_invalid(valid, errors, f"Invalid {label} geodataframe for comparison")
        
        # Create figure with two subplots side by side
        fig, axs = plt.subplots(1, 2, figsize=(self.fig_width * 1.5, self.fig_height),
                               sharex=True, sharey=True)
        
        # Create buffer geometries for both GeoDataFrames
        markets_buffer_orig = original_gdf.copy()
        markets_buffer_sim = simulated_gdf.copy()
        
        # Create buffers with sizes based on market importance or fixed size
        if 'market_size' in markets_buffer_orig.columns:
            # Buffer size based on market size
            buffer_sizes_orig = markets_buffer_orig['market_size'] * 0.05
            buffer_sizes_sim = markets_buffer_sim['market_size'] * 0.05
            
            markets_buffer_orig['geometry'] = markets_buffer_orig.geometry.buffer(buffer_sizes_orig)
            markets_buffer_sim['geometry'] = markets_buffer_sim.geometry.buffer(buffer_sizes_sim)
        else:
            # Fixed buffer size
            markets_buffer_orig['geometry'] = markets_buffer_orig.geometry.buffer(0.05)
            markets_buffer_sim['geometry'] = markets_buffer_sim.geometry.buffer(0.05)
        
        # Get overall min and max for consistent color scale
        vmin = min(markets_buffer_orig[metric_col].min(), 
                  markets_buffer_sim[metric_col].min())
        vmax = max(markets_buffer_orig[metric_col].max(), 
                  markets_buffer_sim[metric_col].max())
        
        # Plot original data on left subplot
        markets_buffer_orig.plot(
            ax=axs[0],
            column=metric_col,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            legend=True,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Plot simulated data on right subplot
        markets_buffer_sim.plot(
            ax=axs[1],
            column=metric_col,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            legend=True,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add region borders if region column is provided
        if region_col in original_gdf.columns:
            # Create region boundaries for both plots
            for i, markets_buffer in enumerate([markets_buffer_orig, markets_buffer_sim]):
                # Dissolve by region to get region polygons
                regions = markets_buffer.dissolve(by=region_col)
                
                # Plot region boundaries
                regions.boundary.plot(
                    ax=axs[i],
                    color='black',
                    linewidth=1.5,
                    linestyle='--'
                )
        
        # Add basemap if enabled and contextily is available
        if basemap and self.has_contextily:
            try:
                for i, market_gdf in enumerate([original_gdf, simulated_gdf]):
                    # Reproject to Web Mercator for basemap
                    if market_gdf.crs != 'EPSG:3857':
                        ax_crs = market_gdf.to_crs('EPSG:3857').plot(ax=axs[i], alpha=0)
                    
                    # Add basemap
                    ctx.add_basemap(
                        axs[i],
                        source=ctx.providers.CartoDB.Positron,
                        zoom=10
                    )
            except Exception as e:
                logger.warning(f"Could not add basemap: {e}")
        
        # Add titles to subplots
        axs[0].set_title('Original')
        axs[1].set_title('After Policy Intervention')
        
        # Set labels
        axs[0].set_xlabel('Longitude')
        axs[0].set_ylabel('Latitude')
        axs[1].set_xlabel('Longitude')
        
        # Set main title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.95)
        else:
            fig.suptitle(f'Market Integration Comparison: {metric_col.replace("_", " ").title()}',
                       fontsize=14, y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
        
        return fig, axs
