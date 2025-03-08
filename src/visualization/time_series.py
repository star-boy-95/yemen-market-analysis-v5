# src/visualization/time_series.py

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from typing import Optional, List, Dict, Union, Tuple, Any

from src.utils import (
    handle_errors, 
    config, 
    validate_dataframe, 
    raise_if_invalid,
    set_plotting_style, 
    format_date_axis, 
    format_currency_axis,
    save_plot,
    create_figure,
    plot_time_series,
    plot_multiple_time_series,
    plot_time_series_by_group,
    plot_dual_axis,
    add_annotations,
    configure_axes_for_print,
    VisualizationError
)

logger = logging.getLogger(__name__)

class TimeSeriesVisualizer:
    """Enhanced time series visualizations for market data."""
    
    def __init__(self):
        """Initialize the visualizer with default styling."""
        set_plotting_style()
        
        # Get styling parameters from config
        self.fig_width = config.get('visualization.default_fig_width', 12)
        self.fig_height = config.get('visualization.default_fig_height', 8)
        self.dpi = config.get('visualization.figure_dpi', 300)
        self.date_format = config.get('visualization.date_format', '%Y-%m')
        self.north_color = config.get('visualization.north_color', '#1f77b4')
        self.south_color = config.get('visualization.south_color', '#ff7f0e')
        self.save_dir = config.get('visualization.save_dir', 'results/plots')
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, VisualizationError))
    def plot_price_series(
        self, 
        df: pd.DataFrame, 
        price_col: str = 'price', 
        date_col: str = 'date', 
        group_col: Optional[str] = None,
        title: Optional[str] = None,
        ylabel: str = 'Price',
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot price time series data, optionally grouped by a category.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the price data
        price_col : str, optional
            Name of the price column, default 'price'
        date_col : str, optional
            Name of the date column, default 'date'
        group_col : str, optional
            Column to group by (e.g., 'exchange_rate_regime'), default None
        title : str, optional
            Plot title, default None
        ylabel : str, optional
            Y-axis label, default 'Price'
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
        required_cols = [date_col, price_col]
        if group_col is not None:
            required_cols.append(group_col)
            
        valid, errors = validate_dataframe(df, required_columns=required_cols)
        raise_if_invalid(valid, errors, "Invalid data for price series plot")
        
        if group_col is not None:
            # Use the utility function for grouped time series
            fig, ax = plot_time_series_by_group(
                df=df,
                x=date_col,
                y=price_col,
                group=group_col,
                title=title,
                ylabel=ylabel,
                figsize=(self.fig_width, self.fig_height)
            )
        else:
            # Use the utility function for single time series
            fig, ax = plot_time_series(
                df=df,
                x=date_col,
                y=price_col,
                title=title,
                ylabel=ylabel,
                figsize=(self.fig_width, self.fig_height)
            )
        
        # Configure axes for print if needed
        configure_axes_for_print(ax)
            
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
            
        return fig, ax

    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def plot_price_differentials(
        self, 
        df: pd.DataFrame, 
        date_col: str = 'date', 
        north_col: str = 'north_price', 
        south_col: str = 'south_price', 
        diff_col: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot price differentials between north and south markets.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the price data
        date_col : str, optional
            Name of the date column, default 'date'
        north_col : str, optional
            Name of the northern prices column, default 'north_price'
        south_col : str, optional
            Name of the southern prices column, default 'south_price'
        diff_col : str, optional
            Name of the price differential column, default None
            If None, calculates the differential as north_col - south_col
        title : str, optional
            Plot title, default None
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
        required_cols = [date_col]
        if diff_col is None:
            required_cols.extend([north_col, south_col])
        else:
            required_cols.append(diff_col)
            
        valid, errors = validate_dataframe(df, required_columns=required_cols)
        raise_if_invalid(valid, errors, "Invalid data for price differentials plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        # Calculate differential if not provided
        if diff_col is None:
            differential = df[north_col] - df[south_col]
            diff_col = 'Price Differential'
        else:
            differential = df[diff_col]
            
        # Plot differential
        ax.plot(
            df[date_col], 
            differential, 
            color='k',
            linewidth=config.get('visualization.linewidth', 1.5)
        )
        
        # Add zero line
        ax.axhline(
            y=0, 
            color='r', 
            linestyle='--', 
            alpha=0.5,
            linewidth=config.get('visualization.linewidth', 1.5) * 0.8
        )
        
        # Set labels and title
        ax.set_ylabel('Price Differential')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Price Differential (North - South)')
        
        # Format date axis
        format_date_axis(
            ax, 
            date_format=self.date_format,
            interval=config.get('visualization.date_interval', 'month')
        )
        
        # Add grid if configured
        if config.get('visualization.grid', True):
            ax.grid(True, alpha=0.3)
            
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
            
        return fig, ax
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, ImportError))
    def plot_interactive_time_series(
        self, 
        df: pd.DataFrame, 
        price_col: str = 'price', 
        date_col: str = 'date',
        group_col: Optional[str] = None,
        title: Optional[str] = None,
        ylabel: str = 'Price',
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive time series plot using Plotly.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the price data
        price_col : str, optional
            Name of the price column, default 'price'
        date_col : str, optional
            Name of the date column, default 'date'
        group_col : str, optional
            Column to group by (e.g., 'exchange_rate_regime'), default None
        title : str, optional
            Plot title, default None
        ylabel : str, optional
            Y-axis label, default 'Price'
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : go.Figure
            The plotly figure
        """
        # Validate inputs
        required_cols = [date_col, price_col]
        if group_col is not None:
            required_cols.append(group_col)
            
        valid, errors = validate_dataframe(df, required_columns=required_cols)
        raise_if_invalid(valid, errors, "Invalid data for interactive time series plot")
        
        # Get figure dimensions
        width = config.get('visualization.interactive_width', 800)
        height = config.get('visualization.interactive_height', 600)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add traces
        if group_col is not None:
            # Group data and plot each group with a different color
            groups = df[group_col].unique()
            for group in groups:
                group_data = df[df[group_col] == group]
                fig.add_trace(
                    go.Scatter(
                        x=group_data[date_col],
                        y=group_data[price_col],
                        mode='lines',
                        name=str(group)
                    )
                )
        else:
            # Plot a single line
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[price_col],
                    mode='lines'
                )
            )
        
        # Set layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=ylabel,
            width=width,
            height=height,
            hovermode='x unified'
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, AttributeError))
    def plot_threshold_analysis(
        self, 
        threshold_model: Any, 
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize threshold model results.
        
        Parameters
        ----------
        threshold_model : object
            Threshold model object with results
        title : str, optional
            Plot title, default None
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        axs : list of plt.Axes
            The matplotlib axes
        """
        # Create figure with subplots
        fig, axs = plt.subplots(
            2, 1, 
            figsize=(self.fig_width, self.fig_height),
            gridspec_kw={'height_ratios': [2, 1]}
        )
        
        # Top plot: Price series and threshold
        axs[0].plot(
            threshold_model.dates, 
            threshold_model.price_diff, 
            color='k',
            label='Price Differential'
        )
        
        # Add threshold lines
        threshold = threshold_model.threshold
        axs[0].axhline(
            y=threshold, 
            color='r', 
            linestyle='--', 
            label=f'Threshold: {threshold:.2f}'
        )
        axs[0].axhline(
            y=-threshold, 
            color='r', 
            linestyle='--'
        )
        
        # Shade regime areas
        above_threshold = threshold_model.price_diff > threshold
        below_neg_threshold = threshold_model.price_diff < -threshold
        
        # Shade regions by regime
        for i in range(len(threshold_model.dates) - 1):
            if above_threshold[i]:
                axs[0].axvspan(
                    threshold_model.dates[i],
                    threshold_model.dates[i+1],
                    alpha=0.2,
                    color='red'
                )
            elif below_neg_threshold[i]:
                axs[0].axvspan(
                    threshold_model.dates[i],
                    threshold_model.dates[i+1],
                    alpha=0.2,
                    color='blue'
                )
                
        axs[0].set_ylabel('Price Differential')
        axs[0].legend()
        
        # Bottom plot: Adjustment speeds by regime
        regimes = ['Below Threshold', 'Middle Regime', 'Above Threshold']
        adjustments = [
            abs(threshold_model.adjustment_below), 
            abs(threshold_model.adjustment_middle), 
            abs(threshold_model.adjustment_above)
        ]
        
        axs[1].bar(
            regimes,
            adjustments,
            color=['blue', 'gray', 'red']
        )
        axs[1].set_ylabel('Speed of Adjustment')
        
        # Format axes
        format_date_axis(
            axs[0], 
            date_format=self.date_format,
            interval=config.get('visualization.date_interval', 'month')
        )
        
        # Set title if provided
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle('Threshold Cointegration Analysis')
            
        # Adjust layout
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
            
        return fig, axs
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def plot_simulation_comparison(
        self, 
        original_data: pd.DataFrame, 
        simulated_data: pd.DataFrame,
        date_col: str = 'date',
        price_cols: List[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Compare original and simulated price series.
        
        Parameters
        ----------
        original_data : pd.DataFrame
            DataFrame containing the original price data
        simulated_data : pd.DataFrame
            DataFrame containing the simulated price data
        date_col : str, optional
            Name of the date column, default 'date'
        price_cols : list of str, optional
            Names of price columns to compare, default None
            If None, uses all numeric columns except date_col
        title : str, optional
            Plot title, default None
        save_path : str, optional
            Path to save the figure, default None
            
        Returns
        -------
        fig : plt.Figure
            The matplotlib figure
        axs : list of plt.Axes
            The matplotlib axes
        """
        # Validate inputs
        for df in [original_data, simulated_data]:
            valid, errors = validate_dataframe(df, required_columns=[date_col])
            raise_if_invalid(valid, errors, "Invalid data for simulation comparison plot")
        
        # Determine price columns if not specified
        if price_cols is None:
            price_cols = [col for col in original_data.columns 
                         if col != date_col and 
                         np.issubdtype(original_data[col].dtype, np.number)]
        
        # Create figure with subplots
        n_cols = len(price_cols)
        fig, axs = plt.subplots(
            n_cols, 1, 
            figsize=(self.fig_width, self.fig_height * n_cols / 2),
            sharex=True
        )
        
        # If only one price column, wrap the axes in a list
        if n_cols == 1:
            axs = [axs]
        
        # Plot each series
        for i, col in enumerate(price_cols):
            ax = axs[i]
            
            # Plot original data
            ax.plot(
                original_data[date_col], 
                original_data[col],
                color='k',
                label='Actual'
            )
            
            # Plot simulated data
            ax.plot(
                simulated_data[date_col], 
                simulated_data[col],
                color='r',
                linestyle='--',
                label='Simulated'
            )
            
            ax.set_ylabel(col)
            ax.legend()
            
            # Add grid if configured
            if config.get('visualization.grid', True):
                ax.grid(True, alpha=0.3)
        
        # Format date axis on bottom subplot
        format_date_axis(
            axs[-1], 
            date_format=self.date_format,
            interval=config.get('visualization.date_interval', 'month')
        )
        
        # Set title if provided
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle('Actual vs. Simulated Prices')
            
        # Adjust layout
        fig.tight_layout()
        
        # Save if requested
        if save_path:
            save_plot(fig, save_path, dpi=self.dpi)
            
        return fig, axs