"""Maps visualization module for Yemen Market Analysis.

This module provides functionality for creating and visualizing spatial maps of market data.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Patch

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

class MapPlotter:
    """
    Map plotter for Yemen Market Analysis.

    This class provides methods for creating and visualizing spatial maps of market data.

    Attributes:
        style (str): Style for maps.
        dpi (int): DPI for map images.
        figsize (tuple): Figure size for maps.
    """

    def __init__(self, style: Optional[str] = None, dpi: Optional[int] = None, figsize: Optional[Tuple[int, int]] = None):
        """
        Initialize the map plotter.

        Args:
            style: Style for maps.
            dpi: DPI for map images.
            figsize: Figure size for maps.
        """
        self.style = style or config.get('visualization.style', 'seaborn-v0_8-whitegrid')
        self.dpi = dpi or config.get('visualization.dpi', 300)
        self.figsize = figsize or config.get('visualization.figure_size', (10, 6))

        # Set plot style
        plt.style.use(self.style)

    @handle_errors
    def plot_choropleth(self, gdf: gpd.GeoDataFrame, column: str, title: str,
                      cmap: str = 'viridis', **kwargs) -> plt.Figure:
        """
        Create a choropleth map from GeoDataFrame.

        Args:
            gdf: GeoDataFrame to plot.
            column: Column to use for the choropleth map.
            title: Title for the map.
            cmap: Colormap for the choropleth map.
            **kwargs: Additional arguments to pass to the plot method.

        Returns:
            Matplotlib figure with the choropleth map.
        """
        logger.info(f"Creating choropleth map for {column}")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot the GeoDataFrame
        gdf.plot(column=column, cmap=cmap, ax=ax, legend=True, **kwargs)

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    @handle_errors
    def plot_market_network(self, gdf: gpd.GeoDataFrame, edges: List[Tuple[str, str]],
                          title: str, color_by: Optional[str] = None) -> plt.Figure:
        """
        Create a market network map.

        Args:
            gdf: GeoDataFrame with market locations.
            edges: List of edges between markets (pairs of market IDs).
            title: Title for the map.
            color_by: Column to use for coloring edges.

        Returns:
            Matplotlib figure with the market network map.
        """
        logger.info(f"Creating market network map with {len(edges)} edges")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot markets
        gdf.plot(ax=ax, color='blue', markersize=50, alpha=0.7)

        # Plot network edges
        for edge in edges:
            market1 = gdf[gdf['market_id'] == edge[0]].iloc[0]
            market2 = gdf[gdf['market_id'] == edge[1]].iloc[0]

            x = [market1.geometry.x, market2.geometry.x]
            y = [market1.geometry.y, market2.geometry.y]

            ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.5)

        # Add market labels
        for idx, row in gdf.iterrows():
            ax.annotate(row['market'], (row.geometry.x, row.geometry.y),
                       xytext=(3, 3), textcoords="offset points")

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        return fig

    @handle_errors
    def save_map(self, fig: plt.Figure, output_path: str) -> None:
        """
        Save a map figure to a file.

        Args:
            fig: Matplotlib figure to save.
            output_path: Path to save the figure to.
        """
        logger.info(f"Saving map to {output_path}")
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    @handle_errors
    def generate_all_visualizations(self, spatial_results: Dict[str, Any],
                                  publication_quality: bool = True,
                                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate all spatial visualizations.

        Args:
            spatial_results: Results from spatial analysis.
            publication_quality: Whether to generate publication-quality visualizations.
            output_dir: Directory to save visualizations.

        Returns:
            Dictionary containing the visualization results.
        """
        logger.info("Generating all spatial visualizations")

        # Set DPI based on publication quality
        original_dpi = self.dpi
        if publication_quality:
            self.dpi = 300

        results = {}

        try:
            # Check if we have the necessary data
            if not spatial_results or 'data' not in spatial_results:
                logger.warning("No spatial data available for visualization")
                return {}

            # Extract data
            data = spatial_results.get('data')

            # Create choropleth maps for various metrics
            if 'moran' in spatial_results:
                # Create Moran's I map
                moran_fig = self.plot_choropleth(
                    data,
                    'price',
                    'Price Distribution with Moran\'s I',
                    cmap='viridis'
                )

                if output_dir:
                    self.save_map(moran_fig, f"{output_dir}/moran_i_map.png")

                results['moran_i'] = moran_fig

            # Create market integration network if available
            if 'market_integration' in spatial_results:
                integration_data = spatial_results.get('market_integration', {})

                if 'integrated_markets' in integration_data:
                    integrated_markets = integration_data['integrated_markets']

                    # Create network map
                    network_fig = self.plot_market_network(
                        data,
                        integrated_markets,
                        'Market Integration Network'
                    )

                    if output_dir:
                        self.save_map(network_fig, f"{output_dir}/market_integration_network.png")

                    results['market_network'] = network_fig

            # Create conflict integration map if available
            if 'conflict_integration' in spatial_results:
                conflict_data = spatial_results.get('conflict_integration', {})

                if 'conflict_intensity' in data.columns:
                    conflict_fig = self.plot_choropleth(
                        data,
                        'conflict_intensity',
                        'Conflict Intensity Distribution',
                        cmap='Reds'
                    )

                    if output_dir:
                        self.save_map(conflict_fig, f"{output_dir}/conflict_intensity_map.png")

                    results['conflict_intensity'] = conflict_fig

            return results

        finally:
            # Restore original DPI
            self.dpi = original_dpi