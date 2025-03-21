# src/visualization/maps.py

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import folium
from folium.plugins import HeatMap
from typing import Optional, List, Dict, Union, Tuple, Any

from yemen_market_integration.utils import (
    handle_errors,
    config,
    validate_geodataframe,
    raise_if_invalid,
    reproject_gdf,
    save_plot,
    create_figure,
    create_buffer,
    find_nearest_points,
    calculate_market_isolation,
    create_exchange_regime_boundaries,
    calculate_exchange_rate_boundary,
    plot_yemen_market_integration,
    add_annotations,
    configure_axes_for_print,
    VisualizationError
)

logger = logging.getLogger(__name__)

class MarketMapVisualizer:
    """Enhanced spatial visualizations for market data."""
    
    def __init__(self):
        """Initialize the visualizer with default styling."""
        # Get styling parameters from config
        self.fig_width = config.get('visualization.default_fig_width', 12)
        self.fig_height = config.get('visualization.default_fig_height', 8)
        self.dpi = config.get('visualization.figure_dpi', 300)
        self.cmap = config.get('visualization.color_palette', 'viridis')
        self.north_color = config.get('visualization.north_color', '#1f77b4')
        self.south_color = config.get('visualization.south_color', '#ff7f0e')
        self.conflict_color = config.get('visualization.conflict_color', '#d62728')
        self.save_dir = config.get('visualization.save_dir', 'results/plots')
        
        # Set default CRS for maps
        self.web_mercator_crs = 3857  # Web Mercator for basemaps
        self.yemen_crs = config.get('analysis.spatial.crs', 32638)  # UTM Zone 38N for Yemen
        
        # Set buffer distance for spatial operations
        self.buffer_distance = config.get('analysis.spatial.distance_threshold', 50000)  # meters
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def plot_static_map(
        self, 
        gdf: gpd.GeoDataFrame, 
        column: Optional[str] = None, 
        cmap: Optional[str] = None,
        add_basemap: bool = True,
        title: Optional[str] = None,
        legend: bool = True,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create static map using matplotlib.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the spatial data
        column : str, optional
            Column name to use for coloring, default None
        cmap : str, optional
            Colormap to use, default from config
        add_basemap : bool, optional
            Whether to add a basemap, default True
        title : str, optional
            Plot title, default None
        legend : bool, optional
            Whether to show a legend for the column, default True
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        ax : plt.Axes
            The matplotlib axes
        """
        # Validate inputs
        valid, errors = validate_geodataframe(gdf, check_nulls=False)
        # Only validate required columns exist, not null values
        raise_if_invalid(valid, [e for e in errors if "null values" not in e],
                        "Invalid GeoDataFrame for static map")
        
        if column is not None and column not in gdf.columns:
            raise ValueError(f"Column '{column}' not found in GeoDataFrame")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Set colormap
        if cmap is None:
            cmap = self.cmap
        
        # Ensure GeoDataFrame has the right CRS for basemap (if needed)
        if add_basemap:
            plot_gdf = reproject_gdf(gdf.copy(), to_crs=self.web_mercator_crs)
        else:
            plot_gdf = gdf.copy()
        
        # Plot the GeoDataFrame
        if column is not None:
            plot_gdf.plot(
                column=column,
                cmap=cmap,
                legend=legend,
                ax=ax
            )
        else:
            plot_gdf.plot(ax=ax)
        
        # Add basemap if requested
        if add_basemap:
            try:
                ctx.add_basemap(
                    ax, 
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    zoom='auto'
                )
            except Exception as e:
                logger.warning(f"Failed to add basemap: {str(e)}")
        
        # Set title if provided
        if title:
            ax.set_title(title)
        
        # Remove axis labels
        ax.set_axis_off()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
            
        return fig, ax
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def create_interactive_map(
        self,
        gdf: gpd.GeoDataFrame,
        column: Optional[str] = None,
        popup_cols: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        key_on: str = 'feature.properties.market_id'
    ) -> folium.Map:
        """
        Create interactive map using folium.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the spatial data
        column : str, optional
            Column name to use for coloring markers, default None
        popup_cols : list of str, optional
            List of column names to include in popups, default None
        title : str, optional
            Map title, default None
        save_path : str, optional
            Path to save the HTML map, default None
        key_on : str, optional
            The key in GeoJSON to bind the data to, default 'feature.properties.market_id'
            
        Returns
        -------
        m : folium.Map
            The folium map
        """
        # Validate inputs
        valid, errors = validate_geodataframe(gdf, check_nulls=False)
        # Only validate required columns exist, not null values
        raise_if_invalid(valid, [e for e in errors if "null values" not in e],
                        "Invalid GeoDataFrame for interactive map")
        
        # Ensure GeoDataFrame has WGS84 CRS for folium
        plot_gdf = reproject_gdf(gdf.copy(), to_crs=4326)  # WGS84 for web mapping
        
        # Convert any Timestamp objects to strings to avoid JSON serialization issues
        for col in plot_gdf.columns:
            if pd.api.types.is_datetime64_any_dtype(plot_gdf[col]):
                plot_gdf[col] = plot_gdf[col].astype(str)
        
        # Determine center of map
        # First reproject to a projected CRS for accurate centroid calculation
        projected_gdf = plot_gdf.copy()
        if projected_gdf.crs.is_geographic:
            # Use a suitable projected CRS (UTM zone for Yemen)
            projected_gdf = projected_gdf.to_crs(self.yemen_crs)
        
        # Calculate centroids in the projected CRS
        centroids = projected_gdf.geometry.centroid
        
        # Convert back to WGS84 for folium
        if projected_gdf.crs != 4326:
            centroids = centroids.to_crs(4326)
        
        # Calculate the mean center
        center = [
            centroids.y.mean(),
            centroids.x.mean()
        ]
        
        # Create folium map
        m = folium.Map(
            location=center,
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Add title if provided
        if title:
            title_html = f'''
                <h3 align="center" style="font-size:16px">{title}</h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
        
        # Prepare for choropleth if column is specified
        if column is not None and column in plot_gdf.columns:
            # Normalize the column for color scale
            if np.issubdtype(plot_gdf[column].dtype, np.number):
                vmin = plot_gdf[column].min()
                vmax = plot_gdf[column].max()
                
                # Extract the key column from the key_on parameter
                key_parts = key_on.split('.')
                if len(key_parts) >= 3:
                    key_column = key_parts[-1]  # Last part of the key_on string
                else:
                    key_column = 'index'  # Default to index if key_on format is unexpected
                
                # Check if the key column exists in the dataframe
                if key_column not in plot_gdf.columns and key_column != 'index':
                    # Use index as fallback
                    logger.warning(f"Column '{key_column}' not found in GeoDataFrame, using index instead")
                    key_column = 'index'
                    # Add index as a column if needed
                    plot_gdf['index'] = plot_gdf.index
                
                # Add choropleth
                folium.Choropleth(
                    geo_data=plot_gdf.__geo_interface__,
                    name=column,
                    data=plot_gdf,
                    columns=[key_column, column],
                    key_on=key_on,
                    fill_color=self.cmap,
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name=column
                ).add_to(m)
        
        # Add markers for each point
        for idx, row in plot_gdf.iterrows():
            # Skip if not a point geometry
            if row.geometry.geom_type != 'Point':
                continue
                
            # Prepare popup content
            popup_content = ""
            if popup_cols is not None:
                for col in popup_cols:
                    if col in row:
                        popup_content += f"<b>{col}:</b> {row[col]}<br>"
            
            # Add marker
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=folium.Popup(popup_content, max_width=300) if popup_content else None
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save if requested
        if save_path:
            m.save(save_path)
            
        return m
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def plot_price_heatmap(
        self, 
        gdf: gpd.GeoDataFrame, 
        commodity: Optional[str] = None, 
        date: Optional[str] = None,
        price_col: str = 'price',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> folium.Map:
        """
        Create price heatmap for specific commodity and date.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the market data
        commodity : str, optional
            Commodity to filter by, default None
        date : str, optional
            Date to filter by, default None
        price_col : str, optional
            Name of the price column, default 'price'
        title : str, optional
            Map title, default None
        save_path : str, optional
            Path to save the HTML map, default None
            
        Returns
        -------
        m : folium.Map
            The folium map
        """
        # Validate inputs
        valid, errors = validate_geodataframe(gdf, required_columns=[price_col], check_nulls=False)
        # Only validate required columns exist, not null values
        raise_if_invalid(valid, [e for e in errors if "null values" not in e],
                        "Invalid GeoDataFrame for price heatmap")
        
        # Filter by commodity and date if provided
        plot_gdf = gdf.copy()
        
        if commodity is not None and 'commodity' in plot_gdf.columns:
            plot_gdf = plot_gdf[plot_gdf['commodity'] == commodity]
            
        if date is not None and 'date' in plot_gdf.columns:
            plot_gdf = plot_gdf[plot_gdf['date'] == date]
        
        # Ensure GeoDataFrame has WGS84 CRS for folium
        plot_gdf = reproject_gdf(plot_gdf, to_crs=4326)  # WGS84 for web mapping
        
        # Determine center of map
        # First reproject to a projected CRS for accurate centroid calculation
        projected_gdf = plot_gdf.copy()
        if projected_gdf.crs.is_geographic:
            # Use a suitable projected CRS (UTM zone for Yemen)
            projected_gdf = projected_gdf.to_crs(self.yemen_crs)
        
        # Calculate centroids in the projected CRS
        centroids = projected_gdf.geometry.centroid
        
        # Convert back to WGS84 for folium
        if projected_gdf.crs != 4326:
            centroids = centroids.to_crs(4326)
        
        # Calculate the mean center
        center = [
            centroids.y.mean(),
            centroids.x.mean()
        ]
        
        # Create folium map
        m = folium.Map(
            location=center,
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Add title if provided
        if title:
            title_html = f'''
                <h3 align="center" style="font-size:16px">{title}</h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
        else:
            # Create title from commodity and date if available
            parts = []
            if commodity is not None:
                parts.append(f"Commodity: {commodity}")
            if date is not None:
                parts.append(f"Date: {date}")
                
            if parts:
                title_html = f'''
                    <h3 align="center" style="font-size:16px">Price Heatmap - {' | '.join(parts)}</h3>
                '''
                m.get_root().html.add_child(folium.Element(title_html))
        
        # Create heat map data
        heat_data = []
        for idx, row in plot_gdf.iterrows():
            # Skip if no price or not a point geometry
            if np.isnan(row[price_col]) or row.geometry.geom_type != 'Point':
                continue
                
            # Add point with intensity based on price
            heat_data.append([row.geometry.y, row.geometry.x, row[price_col]])
        
        # Add heat map layer
        HeatMap(
            heat_data,
            radius=15,
            blur=10,
            max_zoom=13,
            name='Price Heatmap'
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save if requested
        if save_path:
            m.save(save_path)
            
        return m
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError), reraise=True)
    def create_market_integration_map(
        self, 
        gdf: gpd.GeoDataFrame, 
        isolation_col: str = 'isolation_index',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create market integration map showing isolation indices.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing the market data
        isolation_col : str, optional
            Column name for the isolation index, default 'isolation_index'
        title : str, optional
            Map title, default None
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        ax : plt.Axes
            The matplotlib axes
        """
        # Validate inputs
        valid, errors = validate_geodataframe(gdf, required_columns=[isolation_col], check_nulls=False)
        # Only validate required columns exist, not null values
        raise_if_invalid(valid, [e for e in errors if "null values" not in e],
                        "Invalid GeoDataFrame for market integration map")
        
        # If isolation_col doesn't exist but we have conflict data, calculate isolation index
        if isolation_col not in gdf.columns and 'conflict_intensity' in gdf.columns:
            logger.info("Calculating market isolation index from conflict data")
            gdf = calculate_market_isolation(
                gdf, 
                conflict_col='conflict_intensity',
                output_col=isolation_col
            )
        
        # Use the specialized Yemen market integration plot utility
        fig = plot_yemen_market_integration(
            gdf,
            integration_col=isolation_col,
            title=title or 'Market Isolation Index',
            figsize=(self.fig_width, self.fig_height)
        )
        
        # Get the axis from the figure
        ax = fig.get_axes()[0]
        
        # Configure axes for print
        configure_axes_for_print(ax)
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
            
        return fig, ax
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def plot_policy_impact_map(
        self, 
        original_gdf: gpd.GeoDataFrame, 
        simulated_gdf: gpd.GeoDataFrame, 
        metric_col: str = 'price',
        diff_method: str = 'percent',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create map showing policy impact.
        
        Parameters
        ----------
        original_gdf : gpd.GeoDataFrame
            GeoDataFrame containing the original data
        simulated_gdf : gpd.GeoDataFrame
            GeoDataFrame containing the simulated data after policy intervention
        metric_col : str, optional
            Column name to compare, default 'price'
        diff_method : str, optional
            Method to calculate difference: 'absolute' or 'percent', default 'percent'
        title : str, optional
            Map title, default None
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        ax : plt.Axes
            The matplotlib axes
        """
        # Validate inputs
        for gdf, name in [(original_gdf, 'original'), (simulated_gdf, 'simulated')]:
            valid, errors = validate_geodataframe(gdf, required_columns=[metric_col], check_nulls=False)
            # Only validate required columns exist, not null values
            raise_if_invalid(valid, [e for e in errors if "null values" not in e],
                            f"Invalid {name} GeoDataFrame")
        
        # Create a new GeoDataFrame with the difference
        diff_gdf = original_gdf.copy()
        
        # Calculate difference
        if diff_method == 'absolute':
            diff_gdf['diff'] = simulated_gdf[metric_col] - original_gdf[metric_col]
            legend_label = f'Absolute Change in {metric_col}'
        else:  # percent
            diff_gdf['diff'] = (simulated_gdf[metric_col] - original_gdf[metric_col]) / original_gdf[metric_col] * 100
            legend_label = f'Percent Change in {metric_col} (%)'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Ensure GeoDataFrame has the right CRS for basemap
        plot_gdf = reproject_gdf(diff_gdf.copy(), to_crs=self.web_mercator_crs)
        
        # Use a diverging colormap centered at 0
        vmin, vmax = plot_gdf['diff'].min(), plot_gdf['diff'].max()
        abs_max = max(abs(vmin), abs(vmax))
        
        # Plot the difference
        plot_gdf.plot(
            column='diff',
            cmap='RdBu_r',  # Red-Blue diverging, blue for positive change, red for negative
            legend=True,
            ax=ax,
            vmin=-abs_max,
            vmax=abs_max,
            legend_kwds={'label': legend_label}
        )
        
        # Add market names if available
        if 'market' in plot_gdf.columns:
            for idx, row in plot_gdf.iterrows():
                ax.annotate(
                    row['market'],
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(3, 3),
                    textcoords='offset points',
                    fontsize=8
                )
        
        # Add basemap
        try:
            ctx.add_basemap(
                ax, 
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom='auto'
            )
        except Exception as e:
            logger.warning(f"Failed to add basemap: {str(e)}")
        
        # Set title if provided
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Impact of Policy Intervention on {metric_col}')
        
        # Remove axis labels
        ax.set_axis_off()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
            
        return fig, ax