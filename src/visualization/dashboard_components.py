"""
Dashboard components module for creating comprehensive interactive visualizations
of market integration analysis results.

This module provides a high-level API for creating dashboards that combine
spatial, temporal, and statistical visualizations into cohesive displays
for analysis and reporting.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Check if plotly is available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Project-specific imports
from src.utils.error_handler import handle_errors
from src.utils.decorators import timer, memory_usage_decorator
from src.utils.plotting_utils import (
    plot_interactive_dashboard, 
    _create_matplotlib_dashboard,
    format_date_axis
)

from src.visualization.time_series import TimeSeriesVisualizer
from src.visualization.maps import MarketMapVisualizer

# Optional model imports - wrap in try/except to avoid hard dependencies
try:
    from src.models.vecm import ThresholdVECM
    from src.models.simulation import MarketIntegrationSimulation
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False

# Initialize logger
logger = logging.getLogger(__name__)


class DashboardCreator:
    """
    Creates interactive dashboards for market integration analysis.
    
    This class provides methods to build comprehensive dashboards combining
    maps, time series, and model diagnostics into single visualization outputs.
    Dashboards can be created with both matplotlib and plotly backends.
    """

    def __init__(
        self, 
        output_dir: Optional[Union[str, Path]] = None,
        backend: str = 'plotly',
        style: str = 'default'
    ):
        """
        Initialize dashboard creator with configuration settings.
        
        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save dashboard outputs
        backend : str, optional
            Visualization backend ('plotly' or 'matplotlib')
        style : str, optional
            Visual style for the dashboards
        """
        self.backend = backend if HAS_PLOTLY else 'matplotlib'
        self.style = style
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
        
        # Initialize visualization helpers
        self.ts_viz = TimeSeriesVisualizer(backend=self.backend)
        self.map_viz = MarketMapVisualizer(backend=self.backend)
        
        logger.info(f"Initialized DashboardCreator with backend='{self.backend}', style='{self.style}'")
    
    @handle_errors(logger=logger)
    @memory_usage_decorator
    @timer
    def create_market_integration_dashboard(
        self,
        spatial_data: pd.DataFrame,
        time_series_data: pd.DataFrame,
        analysis_results: Optional[Dict[str, Any]] = None,
        market_pairs: Optional[List[Tuple[str, str]]] = None,
        title: str = "Yemen Market Integration Analysis Dashboard",
        date_col: str = 'date',
        price_col: str = 'price',
        market_col: str = 'market',
        location_col: str = 'geometry',
        commodity_filter: Optional[str] = None,
        region_filter: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        height: int = 1000,
        width: int = 1200,
        output_file: Optional[str] = None
    ) -> Any:
        """
        Create a comprehensive dashboard for market integration analysis.
        
        Combines spatial visualizations of market locations and price distributions
        with time series visualizations of price trends and cointegration analysis.
        
        Parameters
        ----------
        spatial_data : pd.DataFrame
            GeoDataFrame with market locations and attributes
        time_series_data : pd.DataFrame
            DataFrame with time series price data
        analysis_results : dict, optional
            Dictionary containing analysis results from model runs
        market_pairs : list of tuples, optional
            List of market pairs to analyze for cointegration
        title : str, optional
            Dashboard title
        date_col : str, optional
            Column name for dates
        price_col : str, optional
            Column name for price data
        market_col : str, optional
            Column name for market identifiers
        location_col : str, optional
            Column name for geometries
        commodity_filter : str, optional
            Filter data for specific commodity
        region_filter : str, optional
            Filter data for specific region
        start_date : str, optional
            Start date for time range filter
        end_date : str, optional
            End date for time range filter
        height : int, optional
            Dashboard height in pixels
        width : int, optional
            Dashboard width in pixels
        output_file : str, optional
            Path to save the dashboard output file
            
        Returns
        -------
        dashboard : Dashboard object or dict
            The created dashboard object (plotly.Figure or dict of matplotlib figures)
        """
        logger.info("Creating market integration dashboard")
        
        # Filter data if needed
        if commodity_filter:
            if 'commodity' in time_series_data.columns:
                time_series_data = time_series_data[time_series_data['commodity'] == commodity_filter]
            if 'commodity' in spatial_data.columns:
                spatial_data = spatial_data[spatial_data['commodity'] == commodity_filter]
        
        if region_filter:
            if 'admin1' in time_series_data.columns:
                time_series_data = time_series_data[time_series_data['admin1'] == region_filter]
            if 'admin1' in spatial_data.columns:
                spatial_data = spatial_data[spatial_data['admin1'] == region_filter]
        
        if start_date and date_col in time_series_data.columns:
            time_series_data = time_series_data[time_series_data[date_col] >= start_date]
            
        if end_date and date_col in time_series_data.columns:
            time_series_data = time_series_data[time_series_data[date_col] <= end_date]
        
        # Prepare components
        dashboard_components = {}
        
        # Market map component
        dashboard_components['market_map'] = {
            'type': 'map',
            'data': spatial_data,
            'params': {
                'title': 'Market Locations and Prices',
                'column': price_col if price_col in spatial_data.columns else None,
                'hover_data': [market_col, price_col, 'admin1'] if all(col in spatial_data.columns for col in [market_col, 'admin1']) else None,
                'mapbox_style': 'carto-positron'
            }
        }
        
        # Time series component
        dashboard_components['price_trends'] = {
            'type': 'timeseries',
            'data': time_series_data,
            'params': {
                'title': 'Price Trends Over Time',
                'x': date_col,
                'y_columns': [price_col] if price_col in time_series_data.columns else [],
                'group_col': market_col if market_col in time_series_data.columns else None,
                'ylabel': 'Price'
            }
        }
        
        # Add market integration analysis if available
        if analysis_results and market_pairs:
            # Add cointegration heatmap if available
            if 'cointegration_matrix' in analysis_results:
                dashboard_components['cointegration_heatmap'] = {
                    'type': 'heatmap',
                    'data': analysis_results['cointegration_matrix'],
                    'params': {
                        'title': 'Market Cointegration Analysis',
                        'color_scale': 'RdBu_r',
                        'colorbar_title': 'Cointegration Strength',
                        'zmin': -1,
                        'zmax': 1
                    }
                }
            
            # Add adjustment speed analysis if available
            if 'adjustment_speeds' in analysis_results:
                dashboard_components['adjustment_speeds'] = {
                    'type': 'timeseries',
                    'data': analysis_results['adjustment_speeds'],
                    'params': {
                        'title': 'Price Adjustment Speeds',
                        'x': 'market_pair',
                        'y_columns': ['adjustment_speed'],
                        'ylabel': 'Half-life (days)'
                    }
                }
        
        # Create dashboard layout
        layout = [
            ['market_map', 'price_trends'],
            ['cointegration_heatmap', 'adjustment_speeds'] if all(k in dashboard_components for k in ['cointegration_heatmap', 'adjustment_speeds']) else None
        ]
        
        # Filter out None rows
        layout = [row for row in layout if row]
        
        # Create the dashboard
        dashboard = plot_interactive_dashboard(
            data_dict=dashboard_components,
            layout=layout,
            title=title,
            height=height,
            width=width
        )
        
        # Save if output file is specified
        if output_file:
            self._save_dashboard(dashboard, output_file)
            
        return dashboard

    @handle_errors(logger=logger)
    @memory_usage_decorator
    @timer
    def create_simulation_results_dashboard(
        self,
        baseline_data: pd.DataFrame,
        simulation_data: pd.DataFrame,
        simulation_results: Optional[Dict[str, Any]] = None,
        spatial_data: Optional[pd.DataFrame] = None,
        title: str = "Yemen Market Integration Simulation Results",
        date_col: str = 'date',
        price_col: str = 'price',
        market_col: str = 'market',
        region_col: str = 'admin1',
        welfare_col: str = 'welfare_effect',
        difference_col: str = 'price_difference',
        height: int = 1000,
        width: int = 1200,
        output_file: Optional[str] = None
    ) -> Any:
        """
        Create a dashboard for comparing baseline and simulation results.
        
        Visualizes price differences, welfare effects, and other outcomes from
        market intervention simulations.
        
        Parameters
        ----------
        baseline_data : pd.DataFrame
            DataFrame with baseline time series data
        simulation_data : pd.DataFrame
            DataFrame with simulation time series data
        simulation_results : dict, optional
            Dictionary with additional simulation result metrics
        spatial_data : pd.DataFrame, optional
            GeoDataFrame with spatial data for mapping results
        title : str, optional
            Dashboard title
        date_col : str, optional
            Column name for dates
        price_col : str, optional
            Column name for price data
        market_col : str, optional
            Column name for market identifiers
        region_col : str, optional
            Column name for region identifiers
        welfare_col : str, optional
            Column name for welfare effect metrics
        difference_col : str, optional
            Column name for price differences
        height : int, optional
            Dashboard height in pixels
        width : int, optional
            Dashboard width in pixels
        output_file : str, optional
            Path to save the dashboard output file
            
        Returns
        -------
        dashboard : Dashboard object or dict
            The created dashboard object (plotly.Figure or dict of matplotlib figures)
        """
        logger.info("Creating simulation results dashboard")
        
        # Prepare components
        dashboard_components = {}
        
        # Time series comparison component
        # Combine baseline and simulation data with an identifier column
        baseline_marked = baseline_data.copy()
        baseline_marked['data_type'] = 'Baseline'
        
        simulation_marked = simulation_data.copy()
        simulation_marked['data_type'] = 'Simulation'
        
        combined_data = pd.concat([baseline_marked, simulation_marked])
        
        dashboard_components['price_comparison'] = {
            'type': 'timeseries',
            'data': combined_data,
            'params': {
                'title': 'Baseline vs Simulation Price Comparison',
                'x': date_col,
                'y_columns': [price_col] if price_col in combined_data.columns else [],
                'group_col': 'data_type',
                'ylabel': 'Price'
            }
        }
        
        # Regional comparison if region column is available
        if region_col in combined_data.columns:
            dashboard_components['regional_comparison'] = {
                'type': 'timeseries',
                'data': combined_data,
                'params': {
                    'title': 'Regional Price Changes',
                    'x': date_col,
                    'y_columns': [price_col] if price_col in combined_data.columns else [],
                    'group_col': region_col,
                    'color_by': 'data_type',
                    'ylabel': 'Price'
                }
            }
            
        # Welfare effects map if spatial data is available
        if spatial_data is not None and welfare_col in spatial_data.columns:
            dashboard_components['welfare_map'] = {
                'type': 'map',
                'data': spatial_data,
                'params': {
                    'title': 'Welfare Effects by Location',
                    'column': welfare_col,
                    'color_scale': 'RdYlGn',  # Red (negative) to green (positive) welfare effects
                    'mapbox_style': 'carto-positron',
                    'hover_data': [market_col, welfare_col, region_col] if all(col in spatial_data.columns for col in [market_col, region_col]) else None
                }
            }
            
        # Price differences histogram
        if simulation_results and 'price_differences' in simulation_results:
            dashboard_components['price_difference_hist'] = {
                'type': 'histogram',
                'data': simulation_results['price_differences'],
                'params': {
                    'title': 'Distribution of Price Changes',
                    'x': difference_col,
                    'color': region_col if region_col in simulation_results['price_differences'].columns else None,
                    'nbins': 20,
                    'xlabel': 'Price Change',
                    'ylabel': 'Frequency'
                }
            }
            
        # Welfare summary if available
        if simulation_results and 'welfare_summary' in simulation_results:
            dashboard_components['welfare_summary'] = {
                'type': 'bar',
                'data': simulation_results['welfare_summary'],
                'params': {
                    'title': 'Welfare Effects by Region',
                    'x': region_col,
                    'y': welfare_col,
                    'color': region_col,
                    'xlabel': 'Region',
                    'ylabel': 'Welfare Effect'
                }
            }
            
        # Create dashboard layout based on available components
        layout = [
            ['price_comparison', 'regional_comparison'] if 'regional_comparison' in dashboard_components else ['price_comparison', 'price_comparison'],
            ['welfare_map', 'price_difference_hist'] if all(k in dashboard_components for k in ['welfare_map', 'price_difference_hist']) else None,
            ['welfare_summary'] if 'welfare_summary' in dashboard_components else None
        ]
        
        # Filter out None rows
        layout = [row for row in layout if row]
        
        # Create the dashboard
        dashboard = plot_interactive_dashboard(
            data_dict=dashboard_components,
            layout=layout,
            title=title,
            height=height,
            width=width
        )
        
        # Save if output file is specified
        if output_file:
            self._save_dashboard(dashboard, output_file)
            
        return dashboard

    @handle_errors(logger=logger)
    @memory_usage_decorator
    @timer
    def create_threshold_analysis_dashboard(
        self,
        threshold_model,
        market1_data: pd.DataFrame,
        market2_data: pd.DataFrame,
        title: str = "Threshold Cointegration Analysis Dashboard",
        date_col: str = 'date',
        price_col: str = 'price',
        height: int = 1000,
        width: int = 1200,
        output_file: Optional[str] = None
    ) -> Any:
        """
        Create a dashboard for threshold cointegration analysis.
        
        Visualizes threshold estimation, regime-specific dynamics, adjustment speeds,
        and diagnostic statistics from threshold vector error correction models.
        
        Parameters
        ----------
        threshold_model : ThresholdVECM or dict
            Fitted threshold model or dictionary with results
        market1_data : pd.DataFrame
            Time series data for first market
        market2_data : pd.DataFrame
            Time series data for second market
        title : str, optional
            Dashboard title
        date_col : str, optional
            Column name for dates
        price_col : str, optional
            Column name for price data
        height : int, optional
            Dashboard height in pixels
        width : int, optional
            Dashboard width in pixels
        output_file : str, optional
            Path to save the dashboard output file
            
        Returns
        -------
        dashboard : Dashboard object or dict
            The created dashboard object (plotly.Figure or dict of matplotlib figures)
        """
        logger.info("Creating threshold analysis dashboard")
        
        if not HAS_MODELS:
            logger.warning("Models module not available, using dict interface for threshold model")
        
        # Extract model results
        if hasattr(threshold_model, 'threshold_result'):
            threshold_result = threshold_model.threshold_result
            coint_result = getattr(threshold_model, 'coint_result', {})
            tvecm_result = getattr(threshold_model, 'tvecm_result', {})
            diagnostic_stats = getattr(threshold_model, 'diagnostic_stats', {})
        elif isinstance(threshold_model, dict):
            threshold_result = threshold_model.get('threshold_result', {})
            coint_result = threshold_model.get('coint_result', {})
            tvecm_result = threshold_model.get('tvecm_result', {})
            diagnostic_stats = threshold_model.get('diagnostic_stats', {})
        else:
            logger.error("Invalid threshold model format")
            raise ValueError("Threshold model must be a ThresholdVECM instance or a dictionary with results")
        
        # Prepare components
        dashboard_components = {}
        
        # Price time series component
        combined_prices = pd.merge(
            market1_data[[date_col, price_col]].rename(columns={price_col: 'market1_price'}),
            market2_data[[date_col, price_col]].rename(columns={price_col: 'market2_price'}),
            on=date_col,
            how='outer'
        )
        
        dashboard_components['price_series'] = {
            'type': 'timeseries',
            'data': combined_prices,
            'params': {
                'title': 'Price Series for Both Markets',
                'x': date_col,
                'y_columns': ['market1_price', 'market2_price'],
                'ylabel': 'Price'
            }
        }
        
        # Scatter plot for cointegration relationship
        scatter_data = pd.DataFrame({
            'market1_price': combined_prices['market1_price'],
            'market2_price': combined_prices['market2_price']
        }).dropna()
        
        dashboard_components['cointegration_scatter'] = {
            'type': 'scatter',
            'data': scatter_data,
            'params': {
                'title': 'Cointegration Relationship',
                'x': 'market1_price',
                'y': 'market2_price',
                'xlabel': 'Market 1 Price',
                'ylabel': 'Market 2 Price',
                'fit_line': True
            }
        }
        
        # Threshold estimation component
        if 'thresholds' in threshold_result and 'ssrs' in threshold_result:
            threshold_data = pd.DataFrame({
                'threshold': threshold_result['thresholds'],
                'ssr': threshold_result['ssrs']
            })
            
            dashboard_components['threshold_estimation'] = {
                'type': 'scatter',
                'data': threshold_data,
                'params': {
                    'title': 'Threshold Estimation',
                    'x': 'threshold',
                    'y': 'ssr',
                    'mode': 'lines+markers',
                    'xlabel': 'Threshold Value',
                    'ylabel': 'Sum of Squared Residuals',
                    'highlight_min': True
                }
            }
            
        # Regime-specific adjustment speeds
        if tvecm_result:
            regimes = ['Below Threshold', 'Above Threshold']
            markets = ['Market 1', 'Market 2']
            
            # Extract adjustment speeds
            adj_speeds = {
                'regime': regimes + regimes,
                'market': markets * 2,
                'adjustment_speed': [
                    tvecm_result.get('adjustment_below_1', 0),
                    tvecm_result.get('adjustment_below_2', 0),
                    tvecm_result.get('adjustment_above_1', 0),
                    tvecm_result.get('adjustment_above_2', 0)
                ]
            }
            
            adj_df = pd.DataFrame(adj_speeds)
            
            dashboard_components['adjustment_speeds'] = {
                'type': 'bar',
                'data': adj_df,
                'params': {
                    'title': 'Adjustment Speeds by Regime',
                    'x': 'regime',
                    'y': 'adjustment_speed',
                    'color': 'market',
                    'barmode': 'group',
                    'xlabel': 'Regime',
                    'ylabel': 'Adjustment Speed'
                }
            }
            
        # Residual diagnostics
        if 'residuals' in tvecm_result:
            residuals = pd.DataFrame({
                'index': range(len(tvecm_result['residuals'])),
                'residual': tvecm_result['residuals']
            })
            
            dashboard_components['residuals'] = {
                'type': 'timeseries',
                'data': residuals,
                'params': {
                    'title': 'Model Residuals',
                    'x': 'index',
                    'y_columns': ['residual'],
                    'mode': 'lines',
                    'xlabel': 'Observation',
                    'ylabel': 'Residual'
                }
            }
            
        # Create dashboard layout
        layout = [
            ['price_series', 'cointegration_scatter'],
            ['threshold_estimation', 'adjustment_speeds'] if all(k in dashboard_components for k in ['threshold_estimation', 'adjustment_speeds']) else None,
            ['residuals'] if 'residuals' in dashboard_components else None
        ]
        
        # Filter out None rows
        layout = [row for row in layout if row]
        
        # Create the dashboard
        dashboard = plot_interactive_dashboard(
            data_dict=dashboard_components,
            layout=layout,
            title=title,
            height=height,
            width=width
        )
        
        # Save if output file is specified
        if output_file:
            self._save_dashboard(dashboard, output_file)
            
        return dashboard

    @handle_errors(logger=logger)
    def create_market_map_component(
        self,
        spatial_data: pd.DataFrame,
        price_col: str = 'price',
        title: str = 'Market Prices Map',
        color_scheme: str = 'quantiles',
        n_classes: int = 5,
        cmap: str = 'viridis',
        add_basemap: bool = True
    ) -> Dict[str, Any]:
        """
        Create a market map component for dashboards.
        
        Parameters
        ----------
        spatial_data : pd.DataFrame
            GeoDataFrame with spatial market data
        price_col : str, optional
            Column name for price data
        title : str, optional
            Component title
        color_scheme : str, optional
            Classification scheme for coloring
        n_classes : int, optional
            Number of classes for classification
        cmap : str, optional
            Colormap name
        add_basemap : bool, optional
            Whether to add a basemap
            
        Returns
        -------
        dict
            Dashboard component specification
        """
        return {
            'type': 'map',
            'data': spatial_data,
            'params': {
                'title': title,
                'column': price_col,
                'scheme': color_scheme,
                'k': n_classes,
                'cmap': cmap,
                'add_basemap': add_basemap
            }
        }

    @handle_errors(logger=logger)
    def create_time_series_component(
        self,
        time_series_data: pd.DataFrame,
        date_col: str = 'date',
        price_col: str = 'price',
        group_col: Optional[str] = None,
        title: str = 'Price Time Series',
        ylabel: str = 'Price'
    ) -> Dict[str, Any]:
        """
        Create a time series component for dashboards.
        
        Parameters
        ----------
        time_series_data : pd.DataFrame
            DataFrame with time series data
        date_col : str, optional
            Column name for dates
        price_col : str, optional
            Column name for price data
        group_col : str, optional
            Column name for grouping
        title : str, optional
            Component title
        ylabel : str, optional
            Y-axis label
            
        Returns
        -------
        dict
            Dashboard component specification
        """
        return {
            'type': 'timeseries',
            'data': time_series_data,
            'params': {
                'title': title,
                'x': date_col,
                'y_columns': [price_col],
                'group_col': group_col,
                'ylabel': ylabel
            }
        }

    @handle_errors(logger=logger)
    def create_model_diagnostics_component(
        self,
        model_results: Dict[str, Any],
        component_type: str = 'heatmap',
        title: str = 'Model Diagnostics'
    ) -> Dict[str, Any]:
        """
        Create a model diagnostics component for dashboards.
        
        Parameters
        ----------
        model_results : dict
            Dictionary with model results
        component_type : str, optional
            Type of component ('heatmap', 'bar', etc.)
        title : str, optional
            Component title
            
        Returns
        -------
        dict
            Dashboard component specification
        """
        params = {
            'title': title
        }
        
        if component_type == 'heatmap':
            params.update({
                'color_scale': 'RdBu_r',
                'colorbar_title': 'Value'
            })
        elif component_type == 'bar':
            params.update({
                'x': 'metric',
                'y': 'value',
                'color': 'category'
            })
        
        return {
            'type': component_type,
            'data': model_results,
            'params': params
        }

    @handle_errors(logger=logger)
    def _save_dashboard(self, dashboard, output_file: str) -> None:
        """
        Save dashboard to file.
        
        Parameters
        ----------
        dashboard : Dashboard object
            The dashboard to save
        output_file : str
            Output file path
        """
        output_path = self.output_dir / output_file if self.output_dir else Path(output_file)
        
        if output_path.suffix == '':
            output_path = output_path.with_suffix('.html')
            
        try:
            if hasattr(dashboard, 'write_html'):
                # Save plotly figure
                dashboard.write_html(
                    output_path,
                    include_plotlyjs='cdn',
                    full_html=True
                )
                logger.info(f"Saved interactive dashboard to {output_path}")
            elif isinstance(dashboard, dict):
                # Save collection of matplotlib figures
                for name, fig in dashboard.items():
                    fig_path = output_path.parent / f"{output_path.stem}_{name}{output_path.suffix}"
                    fig.savefig(fig_path)
                logger.info(f"Saved {len(dashboard)} matplotlib figures to {output_path.parent}")
            else:
                # Single matplotlib figure
                dashboard.savefig(output_path)
                logger.info(f"Saved figure to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save dashboard: {str(e)}")
            raise
