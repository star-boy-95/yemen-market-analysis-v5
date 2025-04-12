"""
Time series visualization module for Yemen Market Analysis.

This module provides the TimeSeriesPlotter class for creating time series plots.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

# Set default style
sns.set_style('whitegrid')

class TimeSeriesPlotter:
    """
    Time series plotter for Yemen Market Analysis.

    This class provides methods for creating time series plots.

    Attributes:
        data (pd.DataFrame): DataFrame containing time series data.
        date_column (str): Column containing dates.
        figsize (Tuple[int, int]): Figure size.
        dpi (int): Figure DPI.
        style (str): Plot style.
    """

    def __init__(
        self, data: Optional[pd.DataFrame] = None,
        date_column: str = 'date',
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 100,
        style: str = 'whitegrid'
    ):
        """
        Initialize the time series plotter.

        Args:
            data: DataFrame containing time series data.
            date_column: Column containing dates.
            figsize: Figure size.
            dpi: Figure DPI.
            style: Plot style.
        """
        self.data = data
        self.date_column = date_column
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        # Set style
        sns.set_style(style)

    @handle_errors
    def set_data(
        self, data: pd.DataFrame, date_column: Optional[str] = None
    ) -> None:
        """
        Set the data for the plotter.

        Args:
            data: DataFrame containing time series data.
            date_column: Column containing dates.

        Raises:
            YemenAnalysisError: If the data is invalid.
        """
        logger.info("Setting data for time series plotter")

        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            logger.error("Data is not a pandas DataFrame")
            raise YemenAnalysisError("Data is not a pandas DataFrame")

        # Set date column
        if date_column is not None:
            self.date_column = date_column

        # Check if date column exists
        if self.date_column not in data.columns:
            logger.error(f"Date column {self.date_column} not found in data")
            raise YemenAnalysisError(f"Date column {self.date_column} not found in data")

        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[self.date_column]):
            logger.info(f"Converting {self.date_column} to datetime")
            data[self.date_column] = pd.to_datetime(data[self.date_column])

        # Set data
        self.data = data

        logger.info(f"Set data with {len(self.data)} observations")

    @handle_errors
    def plot_time_series(
        self, y_column: str, group_column: Optional[str] = None,
        title: Optional[str] = None, ylabel: Optional[str] = None,
        xlabel: Optional[str] = None, color: Optional[str] = None,
        palette: Optional[str] = None, legend_title: Optional[str] = None,
        ax: Optional[Axes] = None, **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Create a time series plot.

        Args:
            y_column: Column to plot on the y-axis.
            group_column: Column to group by. If provided, creates a separate line
                         for each group.
            title: Plot title.
            ylabel: Y-axis label.
            xlabel: X-axis label.
            color: Line color. Only used if group_column is None.
            palette: Color palette. Only used if group_column is not None.
            legend_title: Legend title. Only used if group_column is not None.
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            Tuple containing the figure and axes.

        Raises:
            YemenAnalysisError: If the data has not been set or the columns are invalid.
        """
        logger.info(f"Creating time series plot for {y_column}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if y_column exists
        if y_column not in self.data.columns:
            logger.error(f"Column {y_column} not found in data")
            raise YemenAnalysisError(f"Column {y_column} not found in data")

        # Check if group_column exists
        if group_column is not None and group_column not in self.data.columns:
            logger.error(f"Column {group_column} not found in data")
            raise YemenAnalysisError(f"Column {group_column} not found in data")

        try:
            # Create figure if ax is None
            if ax is None:
                fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            else:
                fig = ax.figure

            # Set default title and labels
            if title is None:
                title = f"Time Series of {y_column}"

            if ylabel is None:
                ylabel = y_column

            if xlabel is None:
                xlabel = self.date_column

            # Plot data
            if group_column is None:
                # Plot a single line
                ax.plot(
                    self.data[self.date_column], self.data[y_column],
                    color=color, **kwargs
                )
            else:
                # Plot a line for each group
                for group, group_data in self.data.groupby(group_column):
                    ax.plot(
                        group_data[self.date_column], group_data[y_column],
                        label=group, **kwargs
                    )

                # Add legend
                ax.legend(title=legend_title)

            # Set title and labels
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Format x-axis
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

            # Adjust layout
            plt.tight_layout()

            logger.info(f"Created time series plot for {y_column}")
            return fig, ax
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            raise YemenAnalysisError(f"Error creating time series plot: {e}")

    @handle_errors
    def plot_multiple_time_series(
        self, y_columns: List[str], title: Optional[str] = None,
        ylabel: Optional[str] = None, xlabel: Optional[str] = None,
        colors: Optional[List[str]] = None, ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Create a plot with multiple time series.

        Args:
            y_columns: Columns to plot on the y-axis.
            title: Plot title.
            ylabel: Y-axis label.
            xlabel: X-axis label.
            colors: Line colors. If None, uses default colors.
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            Tuple containing the figure and axes.

        Raises:
            YemenAnalysisError: If the data has not been set or the columns are invalid.
        """
        logger.info(f"Creating multiple time series plot for {y_columns}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if y_columns exist
        for column in y_columns:
            if column not in self.data.columns:
                logger.error(f"Column {column} not found in data")
                raise YemenAnalysisError(f"Column {column} not found in data")

        try:
            # Create figure if ax is None
            if ax is None:
                fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            else:
                fig = ax.figure

            # Set default title and labels
            if title is None:
                title = "Multiple Time Series"

            if ylabel is None:
                ylabel = "Value"

            if xlabel is None:
                xlabel = self.date_column

            # Plot data
            for i, column in enumerate(y_columns):
                color = colors[i] if colors is not None and i < len(colors) else None
                ax.plot(
                    self.data[self.date_column], self.data[column],
                    label=column, color=color, **kwargs
                )

            # Add legend
            ax.legend()

            # Set title and labels
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Format x-axis
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

            # Adjust layout
            plt.tight_layout()

            logger.info(f"Created multiple time series plot for {y_columns}")
            return fig, ax
        except Exception as e:
            logger.error(f"Error creating multiple time series plot: {e}")
            raise YemenAnalysisError(f"Error creating multiple time series plot: {e}")

    @handle_errors
    def plot_seasonal_decomposition(
        self, y_column: str, period: int = 12, model: str = 'additive',
        title: Optional[str] = None, figsize: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Figure:
        """
        Create a seasonal decomposition plot.

        Args:
            y_column: Column to decompose.
            period: Seasonal period.
            model: Decomposition model. Options are 'additive' and 'multiplicative'.
            title: Plot title.
            figsize: Figure size. If None, uses a larger version of the default figsize.
            **kwargs: Additional arguments to pass to the seasonal_decompose function.

        Returns:
            Matplotlib figure.

        Raises:
            YemenAnalysisError: If the data has not been set or the column is invalid.
        """
        logger.info(f"Creating seasonal decomposition plot for {y_column}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if y_column exists
        if y_column not in self.data.columns:
            logger.error(f"Column {y_column} not found in data")
            raise YemenAnalysisError(f"Column {y_column} not found in data")

        try:
            # Import seasonal_decompose
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Set default figsize
            if figsize is None:
                figsize = (self.figsize[0], self.figsize[1] * 2)

            # Set default title
            if title is None:
                title = f"Seasonal Decomposition of {y_column}"

            # Set index to date column
            data = self.data.set_index(self.date_column)

            # Perform seasonal decomposition
            result = seasonal_decompose(
                data[y_column], model=model, period=period, **kwargs
            )

            # Create plot
            fig = result.plot()
            fig.set_size_inches(figsize)
            fig.set_dpi(self.dpi)
            fig.suptitle(title, fontsize=14)

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            logger.info(f"Created seasonal decomposition plot for {y_column}")
            return fig
        except Exception as e:
            logger.error(f"Error creating seasonal decomposition plot: {e}")
            raise YemenAnalysisError(f"Error creating seasonal decomposition plot: {e}")

    @handle_errors
    def plot_acf_pacf(
        self, y_column: str, lags: int = 40, alpha: float = 0.05,
        title: Optional[str] = None, figsize: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Figure:
        """
        Create ACF and PACF plots.

        Args:
            y_column: Column to analyze.
            lags: Number of lags to include.
            alpha: Significance level for confidence intervals.
            title: Plot title.
            figsize: Figure size. If None, uses the default figsize.
            **kwargs: Additional arguments to pass to the plot_acf and plot_pacf functions.

        Returns:
            Matplotlib figure.

        Raises:
            YemenAnalysisError: If the data has not been set or the column is invalid.
        """
        logger.info(f"Creating ACF and PACF plots for {y_column}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if y_column exists
        if y_column not in self.data.columns:
            logger.error(f"Column {y_column} not found in data")
            raise YemenAnalysisError(f"Column {y_column} not found in data")

        try:
            # Import plot_acf and plot_pacf
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

            # Set default figsize
            if figsize is None:
                figsize = self.figsize

            # Set default title
            if title is None:
                title = f"ACF and PACF of {y_column}"

            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=self.dpi)

            # Plot ACF
            plot_acf(self.data[y_column], lags=lags, alpha=alpha, ax=ax1, **kwargs)
            ax1.set_title(f"Autocorrelation Function (ACF) of {y_column}")

            # Plot PACF
            plot_pacf(self.data[y_column], lags=lags, alpha=alpha, ax=ax2, **kwargs)
            ax2.set_title(f"Partial Autocorrelation Function (PACF) of {y_column}")

            # Set overall title
            fig.suptitle(title, fontsize=14)

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            logger.info(f"Created ACF and PACF plots for {y_column}")
            return fig
        except Exception as e:
            logger.error(f"Error creating ACF and PACF plots: {e}")
            raise YemenAnalysisError(f"Error creating ACF and PACF plots: {e}")

    @handle_errors
    def plot_price_dispersion(
        self, price_column: str, group_column: str,
        title: Optional[str] = None, ylabel: Optional[str] = None,
        xlabel: Optional[str] = None, ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Create a price dispersion plot.

        Args:
            price_column: Column containing prices.
            group_column: Column to group by (e.g., market, region).
            title: Plot title.
            ylabel: Y-axis label.
            xlabel: X-axis label.
            ax: Matplotlib axes to plot on. If None, creates a new figure.
            **kwargs: Additional arguments to pass to the plot function.

        Returns:
            Tuple containing the figure and axes.

        Raises:
            YemenAnalysisError: If the data has not been set or the columns are invalid.
        """
        logger.info(f"Creating price dispersion plot for {price_column} by {group_column}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if columns exist
        if price_column not in self.data.columns:
            logger.error(f"Column {price_column} not found in data")
            raise YemenAnalysisError(f"Column {price_column} not found in data")

        if group_column not in self.data.columns:
            logger.error(f"Column {group_column} not found in data")
            raise YemenAnalysisError(f"Column {group_column} not found in data")

        try:
            # Create figure if ax is None
            if ax is None:
                fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            else:
                fig = ax.figure

            # Set default title and labels
            if title is None:
                title = f"Price Dispersion of {price_column} by {group_column}"

            if ylabel is None:
                ylabel = "Coefficient of Variation"

            if xlabel is None:
                xlabel = self.date_column

            # Calculate price dispersion
            # Group by date and calculate coefficient of variation
            dispersion = self.data.groupby(self.date_column)[price_column].agg(['mean', 'std'])
            dispersion['cv'] = dispersion['std'] / dispersion['mean']

            # Plot data
            ax.plot(dispersion.index, dispersion['cv'], **kwargs)

            # Set title and labels
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # Format x-axis
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)

            # Adjust layout
            plt.tight_layout()

            logger.info(f"Created price dispersion plot for {price_column} by {group_column}")
            return fig, ax
        except Exception as e:
            logger.error(f"Error creating price dispersion plot: {e}")
            raise YemenAnalysisError(f"Error creating price dispersion plot: {e}")

    @handle_errors
    def plot_price_correlation(
        self, price_column: str, group_column: str,
        title: Optional[str] = None, figsize: Optional[Tuple[int, int]] = None,
        cmap: str = 'coolwarm', **kwargs
    ) -> Figure:
        """
        Create a price correlation heatmap.

        Args:
            price_column: Column containing prices.
            group_column: Column to group by (e.g., market, region).
            title: Plot title.
            figsize: Figure size. If None, uses a square version of the default figsize.
            cmap: Colormap for the heatmap.
            **kwargs: Additional arguments to pass to the heatmap function.

        Returns:
            Matplotlib figure.

        Raises:
            YemenAnalysisError: If the data has not been set or the columns are invalid.
        """
        logger.info(f"Creating price correlation heatmap for {price_column} by {group_column}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if columns exist
        if price_column not in self.data.columns:
            logger.error(f"Column {price_column} not found in data")
            raise YemenAnalysisError(f"Column {price_column} not found in data")

        if group_column not in self.data.columns:
            logger.error(f"Column {group_column} not found in data")
            raise YemenAnalysisError(f"Column {group_column} not found in data")

        try:
            # Set default figsize
            if figsize is None:
                size = max(self.figsize)
                figsize = (size, size)

            # Set default title
            if title is None:
                title = f"Price Correlation of {price_column} by {group_column}"

            # Pivot data to wide format
            pivot_data = self.data.pivot_table(
                index=self.date_column, columns=group_column, values=price_column
            )

            # Calculate correlation matrix
            corr_matrix = pivot_data.corr()

            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)

            # Create heatmap
            sns.heatmap(
                corr_matrix, annot=True, cmap=cmap, ax=ax,
                vmin=-1, vmax=1, center=0, **kwargs
            )

            # Set title
            ax.set_title(title)

            # Adjust layout
            plt.tight_layout()

            logger.info(f"Created price correlation heatmap for {price_column} by {group_column}")
            return fig
        except Exception as e:
            logger.error(f"Error creating price correlation heatmap: {e}")
            raise YemenAnalysisError(f"Error creating price correlation heatmap: {e}")

    @handle_errors
    def save_plot(
        self, fig: Figure, file_path: str, dpi: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Save a plot to a file.

        Args:
            fig: Matplotlib figure to save.
            file_path: Path to save the figure to.
            dpi: DPI for the saved figure. If None, uses the figure's DPI.
            **kwargs: Additional arguments to pass to the savefig function.

        Raises:
            YemenAnalysisError: If the figure cannot be saved.
        """
        logger.info(f"Saving plot to {file_path}")

        try:
            # Set default DPI
            if dpi is None:
                dpi = self.dpi

            # Save figure
            fig.savefig(file_path, dpi=dpi, **kwargs)

            logger.info(f"Saved plot to {file_path}")
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            raise YemenAnalysisError(f"Error saving plot: {e}")