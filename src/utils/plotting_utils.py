"""
Plotting utilities for the Yemen Market Integration Project.
Common functions for creating and customizing plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Union, List, Tuple, Dict, Any, Optional, Callable
import logging
from pathlib import Path
import contextlib
import os
import gc
import psutil
import time

from src.utils.error_handler import handle_errors, VisualizationError
from src.utils.decorators import timer, m1_optimized, memory_usage_decorator
from src.utils.performance_utils import configure_system_for_performance, parallelize_dataframe, optimize_dataframe

logger = logging.getLogger(__name__)

# Configure default plot style
def set_plotting_style():
    """Set default plotting style for consistent visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Optimize for M1 Mac
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300

# Set style on import
set_plotting_style()

# Color palettes for different contexts
COLOR_PALETTES = {
    'default': sns.color_palette('deep'),
    'categorical': sns.color_palette('Set2'),
    'sequential': sns.color_palette('Blues'),
    'diverging': sns.color_palette('RdBu_r'),
    'north_south': ['#1f77b4', '#ff7f0e']  # Blue for north, orange for south
}

@contextlib.contextmanager
def plotting_context(style: str = 'whitegrid', context: str = 'notebook', 
                    palette: str = 'deep', font_scale: float = 1.0):
    """
    Context manager for temporary plot styling
    
    Parameters
    ----------
    style : str, optional
        Seaborn style name
    context : str, optional
        Seaborn context name
    palette : str, optional
        Seaborn color palette name
    font_scale : float, optional
        Font scale factor
    """
    # Save current configuration
    original_style = plt.rcParams.copy()
    
    # Apply new style
    with sns.plotting_context(context, font_scale=font_scale):
        with sns.axes_style(style):
            sns.set_palette(palette)
            yield
    
    # Restore original style
    plt.rcParams.update(original_style)

@handle_errors(logger=logger)
def create_figure(
    width: float = 10.0, 
    height: float = 6.0, 
    constrained_layout: bool = True
) -> Tuple[Figure, Axes]:
    """
    Create a matplotlib figure with custom dimensions
    
    Parameters
    ----------
    width : float, optional
        Figure width in inches
    height : float, optional
        Figure height in inches
    constrained_layout : bool, optional
        Whether to use constrained layout
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=constrained_layout)
    return fig, ax

@handle_errors(logger=logger)
def format_date_axis(
    ax: Axes, 
    date_format: str = '%Y-%m', 
    rotation: float = 45,
    interval: str = 'month'
) -> None:
    """
    Format the x-axis for dates
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    date_format : str, optional
        Date format string
    rotation : float, optional
        Rotation angle for tick labels
    interval : str, optional
        Tick interval ('day', 'week', 'month', 'quarter', 'year')
    """
    # Set formatter
    date_formatter = mdates.DateFormatter(date_format)
    ax.xaxis.set_major_formatter(date_formatter)
    
    # Set locator based on interval
    if interval == 'day':
        locator = mdates.DayLocator()
    elif interval == 'week':
        locator = mdates.WeekdayLocator(byweekday=0)  # Monday
    elif interval == 'month':
        locator = mdates.MonthLocator()
    elif interval == 'quarter':
        locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    elif interval == 'year':
        locator = mdates.YearLocator()
    else:
        raise ValueError(f"Invalid interval: {interval}")
    
    ax.xaxis.set_major_locator(locator)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')

@handle_errors(logger=logger)
def format_currency_axis(ax: Axes, axis: str = 'y', symbol: str = 'YER') -> None:
    """
    Format axis to display currency values
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format
    axis : str, optional
        Axis to format ('x' or 'y')
    symbol : str, optional
        Currency symbol
    """
    # Create a formatter function
    def currency_formatter(x, pos):
        if x >= 1e6:
            return f"{x*1e-6:.1f}M {symbol}"
        elif x >= 1e3:
            return f"{x*1e-3:.1f}K {symbol}"
        else:
            return f"{x:.0f} {symbol}"
    
    # Apply formatter to the specified axis
    formatter = mticker.FuncFormatter(currency_formatter)
    if axis.lower() == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)

@handle_errors(logger=logger)
def plot_time_series(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax: Optional[Axes] = None,
    color: Optional[str] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    date_format: str = '%Y-%m',
    interval: str = 'month',
    marker: Optional[str] = None,
    linestyle: str = '-',
    alpha: float = 1.0,
    grid: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot a time series
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    x : str
        Column name for x-axis (date)
    y : str
        Column name for y-axis (value)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    color : str, optional
        Line color
    label : str, optional
        Line label
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    date_format : str, optional
        Date format string
    interval : str, optional
        Tick interval
    marker : str, optional
        Marker style
    linestyle : str, optional
        Line style
    alpha : float, optional
        Alpha transparency
    grid : bool, optional
        Whether to show grid
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Plot the time series
    ax.plot(df[x], df[y], color=color, label=label, marker=marker, 
           linestyle=linestyle, alpha=alpha)
    
    # Format date axis
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        format_date_axis(ax, date_format=date_format, interval=interval)
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add legend if label is provided
    if label:
        ax.legend()
    
    # Add grid
    ax.grid(grid)
    
    return fig, ax

@handle_errors(logger=logger)
def plot_multiple_time_series(
    df: pd.DataFrame,
    x: str,
    y_columns: List[str],
    ax: Optional[Axes] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    date_format: str = '%Y-%m',
    interval: str = 'month',
    marker: Optional[str] = None,
    linestyle: str = '-',
    alpha: float = 1.0,
    grid: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot multiple time series on the same axes
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    x : str
        Column name for x-axis (date)
    y_columns : list of str
        Column names for y-axis (values)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    colors : list of str, optional
        Line colors
    labels : list of str, optional
        Line labels
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    date_format : str, optional
        Date format string
    interval : str, optional
        Tick interval
    marker : str, optional
        Marker style
    linestyle : str, optional
        Line style
    alpha : float, optional
        Alpha transparency
    grid : bool, optional
        Whether to show grid
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Use default colors if not provided
    if colors is None:
        colors = COLOR_PALETTES['default']
    
    # Use column names as labels if not provided
    if labels is None:
        labels = y_columns
    
    # Plot each time series
    for i, y_col in enumerate(y_columns):
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else y_col
        ax.plot(df[x], df[y_col], color=color, label=label, marker=marker,
               linestyle=linestyle, alpha=alpha)
    
    # Format date axis
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        format_date_axis(ax, date_format=date_format, interval=interval)
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(grid)
    
    return fig, ax

@handle_errors(logger=logger)
def plot_time_series_by_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    ax: Optional[Axes] = None,
    palette: Optional[Union[str, List[str]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    date_format: str = '%Y-%m',
    interval: str = 'month',
    marker: Optional[str] = None,
    linestyle: str = '-',
    alpha: float = 1.0,
    grid: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot time series grouped by a categorical variable
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    x : str
        Column name for x-axis (date)
    y : str
        Column name for y-axis (value)
    group : str
        Column name for grouping
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    palette : str or list, optional
        Color palette or list of colors
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    date_format : str, optional
        Date format string
    interval : str, optional
        Tick interval
    marker : str, optional
        Marker style
    linestyle : str, optional
        Line style
    alpha : float, optional
        Alpha transparency
    grid : bool, optional
        Whether to show grid
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Get unique groups
    groups = df[group].unique()
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Use default palette if not provided
    if palette is None:
        if len(groups) == 2 and 'north' in groups and 'south' in groups:
            # Special case for north/south
            palette = COLOR_PALETTES['north_south']
        else:
            # Otherwise use categorical palette
            palette = COLOR_PALETTES['categorical']
    
    # Convert palette name to actual colors
    if isinstance(palette, str):
        palette = sns.color_palette(palette, n_colors=len(groups))
    
    # Plot each group
    for i, group_val in enumerate(groups):
        group_data = df[df[group] == group_val]
        color = palette[i % len(palette)]
        ax.plot(group_data[x], group_data[y], color=color, label=group_val,
               marker=marker, linestyle=linestyle, alpha=alpha)
    
    # Format date axis
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        format_date_axis(ax, date_format=date_format, interval=interval)
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(grid)
    
    return fig, ax

@handle_errors(logger=logger)
def plot_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax: Optional[Axes] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    horizontal: bool = False,
    sort_values: bool = False,
    sort_ascending: bool = False,
    grid: bool = True,
    rotation: float = 0
) -> Tuple[Figure, Axes]:
    """
    Plot a bar chart
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    x : str
        Column name for categories
    y : str
        Column name for values
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    color : str, optional
        Bar color
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    horizontal : bool, optional
        Whether to plot horizontal bars
    sort_values : bool, optional
        Whether to sort by values
    sort_ascending : bool, optional
        Sort order if sorting by values
    grid : bool, optional
        Whether to show grid
    rotation : float, optional
        Rotation angle for tick labels
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Sort data if requested
    if sort_values:
        df = df.sort_values(by=y, ascending=sort_ascending)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Plot horizontal or vertical bars
    if horizontal:
        ax.barh(df[x], df[y], color=color)
    else:
        ax.bar(df[x], df[y], color=color)
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Rotate tick labels
    if horizontal:
        plt.setp(ax.get_yticklabels(), rotation=rotation)
    else:
        plt.setp(ax.get_xticklabels(), rotation=rotation)
    
    # Add grid
    ax.grid(grid, axis='y' if horizontal else 'x')
    
    return fig, ax

@handle_errors(logger=logger)
def plot_stacked_bar(
    df: pd.DataFrame,
    x: str,
    y_columns: List[str],
    ax: Optional[Axes] = None,
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    horizontal: bool = False,
    grid: bool = True,
    rotation: float = 0,
    legend_title: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a stacked bar chart
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    x : str
        Column name for categories
    y_columns : list of str
        Column names for stacked values
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    colors : list of str, optional
        Bar colors
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    horizontal : bool, optional
        Whether to plot horizontal bars
    grid : bool, optional
        Whether to show grid
    rotation : float, optional
        Rotation angle for tick labels
    legend_title : str, optional
        Title for the legend
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Use default colors if not provided
    if colors is None:
        colors = COLOR_PALETTES['categorical']
    
    # Plot stacked bars
    if horizontal:
        bottom = np.zeros(len(df))
        for i, col in enumerate(y_columns):
            ax.barh(df[x], df[col], left=bottom, color=colors[i % len(colors)], label=col)
            bottom += df[col].values
    else:
        bottom = np.zeros(len(df))
        for i, col in enumerate(y_columns):
            ax.bar(df[x], df[col], bottom=bottom, color=colors[i % len(colors)], label=col)
            bottom += df[col].values
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Rotate tick labels
    if horizontal:
        plt.setp(ax.get_yticklabels(), rotation=rotation)
    else:
        plt.setp(ax.get_xticklabels(), rotation=rotation)
    
    # Add legend
    legend = ax.legend(title=legend_title)
    
    # Add grid
    ax.grid(grid, axis='x' if horizontal else 'y')
    
    return fig, ax

@handle_errors(logger=logger)
def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax: Optional[Axes] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    alpha: float = 0.7,
    grid: bool = True,
    fit_line: bool = False,
    fit_order: int = 1,
    size: Optional[Union[float, str]] = None,
    size_scale: float = 100,
    annotate: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a scatter plot
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    x : str
        Column name for x-axis
    y : str
        Column name for y-axis
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    color : str, optional
        Point color
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    alpha : float, optional
        Alpha transparency
    grid : bool, optional
        Whether to show grid
    fit_line : bool, optional
        Whether to add a regression line
    fit_order : int, optional
        Order of polynomial fit
    size : float or str, optional
        Point size or column name for sizes
    size_scale : float, optional
        Scaling factor for sizes if using a column
    annotate : str, optional
        Column name for point annotations
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Determine point sizes
    if size is None:
        s = 50  # Default size
    elif isinstance(size, (int, float)):
        s = size
    else:
        # Use a column for point sizes
        s = df[size] * size_scale
    
    # Create scatter plot
    scatter = ax.scatter(df[x], df[y], c=color, alpha=alpha, s=s)
    
    # Add regression line if requested
    if fit_line:
        x_values = df[x].values
        y_values = df[y].values
        
        # Remove NaN values
        mask = ~(np.isnan(x_values) | np.isnan(y_values))
        x_values = x_values[mask]
        y_values = y_values[mask]
        
        if len(x_values) > fit_order:
            # Fit polynomial
            coeffs = np.polyfit(x_values, y_values, fit_order)
            poly = np.poly1d(coeffs)
            
            # Generate x values for smooth curve
            x_range = np.linspace(min(x_values), max(x_values), 100)
            
            # Plot the line
            ax.plot(x_range, poly(x_range), '--', color='red', 
                   alpha=0.7, label=f'Fit (order {fit_order})')
            ax.legend()
    
    # Add annotations if requested
    if annotate and annotate in df.columns:
        for i, txt in enumerate(df[annotate]):
            ax.annotate(txt, (df[x].iloc[i], df[y].iloc[i]), 
                       textcoords="offset points", xytext=(0, 5), 
                       ha='center', fontsize=8)
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add grid
    ax.grid(grid)
    
    return fig, ax

@handle_errors(logger=logger)
def plot_heatmap(
    df: pd.DataFrame,
    ax: Optional[Axes] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    annot: bool = True,
    fmt: str = '.2f',
    linewidths: float = 0.5,
    cbar: bool = True,
    cbar_label: Optional[str] = None,
    mask_upper: bool = False,
    center: Optional[float] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a heatmap
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot (should be a matrix)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    cmap : str, optional
        Colormap name
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    annot : bool, optional
        Whether to annotate cells
    fmt : str, optional
        Format string for annotations
    linewidths : float, optional
        Width of cell borders
    cbar : bool, optional
        Whether to show colorbar
    cbar_label : str, optional
        Label for colorbar
    mask_upper : bool, optional
        Whether to mask the upper triangle
    center : float, optional
        Center value for diverging colormaps
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)
    
    # Create heatmap
    heatmap = sns.heatmap(df, ax=ax, cmap=cmap, annot=annot, fmt=fmt,
                         linewidths=linewidths, cbar=cbar, mask=mask,
                         center=center)
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add colorbar label
    if cbar and cbar_label:
        cbar = heatmap.collections[0].colorbar
        cbar.set_label(cbar_label)
    
    return fig, ax

@handle_errors(logger=logger)
def plot_histogram(
    df: pd.DataFrame,
    column: str,
    ax: Optional[Axes] = None,
    bins: int = 20,
    color: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = 'Frequency',
    grid: bool = True,
    kde: bool = False,
    vertical_line: Optional[float] = None,
    vertical_line_color: str = 'red',
    vertical_line_label: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a histogram
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    column : str
        Column name to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    bins : int, optional
        Number of bins
    color : str, optional
        Bar color
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    grid : bool, optional
        Whether to show grid
    kde : bool, optional
        Whether to add a kernel density estimate
    vertical_line : float, optional
        Value to mark with a vertical line
    vertical_line_color : str, optional
        Color for vertical line
    vertical_line_label : str, optional
        Label for vertical line
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Create histogram
    sns.histplot(df[column], bins=bins, ax=ax, color=color, kde=kde)
    
    # Add vertical line if requested
    if vertical_line is not None:
        ax.axvline(x=vertical_line, color=vertical_line_color, 
                  linestyle='--', label=vertical_line_label)
        if vertical_line_label:
            ax.legend()
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(column)
    
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    # Add grid
    ax.grid(grid)
    
    return fig, ax

@handle_errors(logger=logger)
def save_plot(
    fig: Figure,
    filename: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight',
    pad_inches: float = 0.1,
    transparent: bool = False,
    facecolor: Optional[str] = None
) -> Path:
    """
    Save a plot to a file
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str or Path
        Output filename
    dpi : int, optional
        Resolution in dots per inch
    bbox_inches : str, optional
        Bounding box setting
    pad_inches : float, optional
        Padding around figure
    transparent : bool, optional
        Whether to use a transparent background
    facecolor : str, optional
        Figure background color
        
    Returns
    -------
    Path
        Path to the saved file
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        filename,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        transparent=transparent,
        facecolor=facecolor
    )
    
    logger.info(f"Saved plot to {filename}")
    return filename

@handle_errors(logger=logger)
def add_annotations(
    ax: Axes,
    annotations: Dict[Tuple[float, float], str],
    offset_x: float = 0,
    offset_y: float = 5,
    fontsize: int = 9,
    alpha: float = 0.8,
    ha: str = 'center',
    rotation: float = 0,
    bbox: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add text annotations to a plot
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate
    annotations : dict
        Dictionary mapping (x,y) coordinates to text
    offset_x : float, optional
        Horizontal offset in points
    offset_y : float, optional
        Vertical offset in points
    fontsize : int, optional
        Font size
    alpha : float, optional
        Text transparency
    ha : str, optional
        Horizontal alignment
    rotation : float, optional
        Text rotation
    bbox : dict, optional
        Properties for text box
    """
    for (x, y), text in annotations.items():
        ax.annotate(
            text,
            (x, y),
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            fontsize=fontsize,
            alpha=alpha,
            ha=ha,
            rotation=rotation,
            bbox=bbox
        )

@handle_errors(logger=logger)
def configure_axes_for_print(
    ax: Axes,
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    fontsize_ticks: int = 10,
    linewidth: float = 1.0
) -> None:
    """
    Configure axes for print-ready graphics
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to configure
    fontsize_title : int, optional
        Font size for title
    fontsize_labels : int, optional
        Font size for axis labels
    fontsize_ticks : int, optional
        Font size for tick labels
    linewidth : float, optional
        Line width for spines
    """
    # Set font sizes
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=fontsize_title)
    
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize_labels)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize_labels)
    
    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    
    # Set linewidth for spines
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)

@handle_errors(logger=logger)
def plot_dual_axis(
    df: pd.DataFrame,
    x: str,
    y1: str,
    y2: str,
    color1: str = 'blue',
    color2: str = 'red',
    label1: Optional[str] = None,
    label2: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel1: Optional[str] = None,
    ylabel2: Optional[str] = None
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Create a plot with dual y-axes
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    x : str
        Column name for x-axis
    y1 : str
        Column name for primary y-axis
    y2 : str
        Column name for secondary y-axis
    color1 : str, optional
        Color for primary series
    color2 : str, optional
        Color for secondary series
    label1 : str, optional
        Label for primary series
    label2 : str, optional
        Label for secondary series
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel1 : str, optional
        Primary y-axis label
    ylabel2 : str, optional
        Secondary y-axis label
        
    Returns
    -------
    tuple
        (figure, (ax1, ax2))
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # Primary y-axis
    ax1.plot(df[x], df[y1], color=color1, label=label1 or y1)
    if ylabel1:
        ax1.set_ylabel(ylabel1, color=color1)
    else:
        ax1.set_ylabel(y1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df[x], df[y2], color=color2, label=label2 or y2)
    if ylabel2:
        ax2.set_ylabel(ylabel2, color=color2)
    else:
        ax2.set_ylabel(y2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Format date axis if needed
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        format_date_axis(ax1)
    
    # Add labels and title
    if xlabel:
        ax1.set_xlabel(xlabel)
    if title:
        ax1.set_title(title)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    return fig, (ax1, ax2)

@handle_errors(logger=logger)
def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    by: Optional[str] = None,
    ax: Optional[Axes] = None,
    vert: bool = True,
    palette: Optional[Union[str, List[str]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = True,
    showfliers: bool = True,
    rotation: float = 0
) -> Tuple[Figure, Axes]:
    """
    Create a box plot
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    column : str
        Column name for values
    by : str, optional
        Column name for grouping
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, creates new figure if None
    vert : bool, optional
        Whether to plot vertical boxes
    palette : str or list, optional
        Color palette or list of colors
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    grid : bool, optional
        Whether to show grid
    showfliers : bool, optional
        Whether to show outliers
    rotation : float, optional
        Rotation angle for tick labels
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.figure
    
    # Create box plot
    if by is None:
        sns.boxplot(x=df[column], ax=ax, vert=vert, palette=palette, showfliers=showfliers)
    else:
        if vert:
            sns.boxplot(x=by, y=column, data=df, ax=ax, palette=palette, showfliers=showfliers)
        else:
            sns.boxplot(y=by, x=column, data=df, ax=ax, palette=palette, showfliers=showfliers)
    
    # Add labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Rotate tick labels
    if by is not None:
        if vert:
            plt.setp(ax.get_xticklabels(), rotation=rotation)
        else:
            plt.setp(ax.get_yticklabels(), rotation=rotation)
    
    # Add grid
    ax.grid(grid, axis='y' if vert else 'x')
    
    return fig, ax

@handle_errors(logger=logger)
def plot_price_deviation_by_conflict(
    df: pd.DataFrame,
    price_col: str = 'price',
    conflict_col: str = 'conflict_intensity_normalized',
    regime_col: str = 'exchange_rate_regime',
    title: str = 'Price Deviations vs Conflict Intensity',
    figsize: Tuple[float, float] = (10, 6)
) -> Tuple[Figure, Axes]:
    """
    Plot price deviations against conflict intensity with regime breakdown.
    
    Visualizes how conflict affects price transmission in Yemen's markets.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    price_col : str, optional
        Column name for price data
    conflict_col : str, optional
        Column name for conflict intensity
    regime_col : str, optional
        Column name for exchange rate regime
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure dimensions
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    # Validate inputs
    for col in [price_col, conflict_col, regime_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique regimes
    regimes = df[regime_col].unique()
    
    # Use north/south specific colors if applicable
    if set(regimes) <= set(['north', 'south']):
        colors = COLOR_PALETTES['north_south']
        regime_to_color = dict(zip(['north', 'south'], colors))
    else:
        # Use categorical palette
        colors = COLOR_PALETTES['categorical']
        regime_to_color = dict(zip(regimes, colors[:len(regimes)]))
    
    # Plot each regime
    for regime in regimes:
        regime_data = df[df[regime_col] == regime]
        ax.scatter(
            regime_data[conflict_col],
            regime_data[price_col],
            label=regime,
            color=regime_to_color.get(regime),
            alpha=0.7,
            edgecolor='w',
            linewidth=0.5
        )
        
        # Add trend line
        if len(regime_data) >= 2:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                regime_data[conflict_col],
                regime_data[price_col]
            )
            
            x = np.array([regime_data[conflict_col].min(), regime_data[conflict_col].max()])
            y = intercept + slope * x
            
            ax.plot(
                x, y,
                color=regime_to_color.get(regime),
                linestyle='--',
                linewidth=1.5,
                alpha=0.8
            )
            
            # Add R² annotation
            r_squared = r_value**2
            ax.annotate(
                f"R² = {r_squared:.3f}",
                xy=(x.mean(), intercept + slope * x.mean()),
                xytext=(5, 5),
                textcoords='offset points',
                color=regime_to_color.get(regime),
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
            )
    
    # Add labels and title
    ax.set_xlabel('Conflict Intensity')
    ax.set_ylabel(price_col)
    ax.set_title(title)
    
    # Add legend
    ax.legend(title=regime_col)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig, ax

@handle_errors(logger=logger)
def plot_yemen_market_integration(
    markets_gdf,
    integration_col: str,
    title: str = 'Market Integration in Yemen',
    cmap: str = 'RdYlGn',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> Figure:
    """
    Create a map of Yemen showing market integration levels.
    
    Specialized for Yemen market analysis with appropriate basemap and borders.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        GeoDataFrame with market locations and integration metric
    integration_col : str
        Column name for integration metric
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name
    figsize : tuple, optional
        Figure dimensions
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Validate inputs
    if not isinstance(markets_gdf, pd.DataFrame) or not hasattr(markets_gdf, 'geometry'):
        raise ValueError("markets_gdf must be a GeoDataFrame")
    
    if integration_col not in markets_gdf.columns:
        raise ValueError(f"Column {integration_col} not found in GeoDataFrame")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot base map of Yemen
    try:
        # Try to get Yemen boundaries if we have them
        import geopandas as gpd
        
        # Try to load Yemen boundaries from natural earth
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        yemen = world[world['name'] == 'Yemen']
        
        if not yemen.empty:
            yemen.plot(
                ax=ax,
                color='lightgray',
                edgecolor='gray',
                linewidth=0.5
            )
        else:
            logger.warning("Yemen boundaries not found in natural earth dataset")
    except Exception as e:
        logger.warning(f"Could not plot Yemen boundaries: {str(e)}")
    
    # Plot markets by integration level
    scatter = markets_gdf.plot(
        column=integration_col,
        ax=ax,
        cmap=cmap,
        legend=True,
        markersize=50,
        alpha=0.8,
        edgecolor='k',
        linewidth=0.5
    )
    
    # If we have an exchange rate regime column, add boundary
    if 'exchange_rate_regime' in markets_gdf.columns:
        try:
            from src.utils import calculate_exchange_rate_boundary
            
            # Calculate boundary
            boundary = calculate_exchange_rate_boundary(markets_gdf, 'exchange_rate_regime')
            
            # Plot boundary if exists
            if not boundary.empty:
                boundary.plot(
                    ax=ax,
                    color='red',
                    linewidth=2,
                    linestyle='--',
                    label='Exchange Rate Boundary'
                )
                ax.legend()
        except Exception as e:
            logger.warning(f"Could not plot exchange rate boundary: {str(e)}")
    
    # Add basemap if contextily is available
    try:
        import contextily as ctx
        ctx.add_basemap(
            ax,
            crs=markets_gdf.crs.to_string(),
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.5
        )
    except:
        logger.warning("Contextily not available, skipping basemap")
    
    # Add title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add legend title
    legend = ax.get_legend()
    if legend:
        legend.set_title(integration_col)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved map to {save_path}")
    
    return fig

# Interactive plotting utilities
def has_plotly():
    """
    Check if plotly is available for interactive plotting.
    
    Returns
    -------
    bool
        Whether plotly is available
    """
    try:
        import plotly.express as px
        return True
    except ImportError:
        return False


@handle_errors(logger=logger, error_type=(ValueError, ImportError, AttributeError))
def plot_interactive_map(
    gdf,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 900,
    color_scale: str = 'viridis',
    fallback_cmap: str = 'viridis',
    center: Optional[Dict[str, float]] = None,
    zoom: Optional[float] = None,
    opacity: float = 0.8,
    mapbox_style: str = 'open-street-map'
) -> Any:
    """
    Create an interactive map visualization using plotly.express.
    Falls back to matplotlib if plotly is not available.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometry column
    color_col : str, optional
        Column name for point colors
    size_col : str, optional
        Column name for point sizes
    hover_data : list, optional
        List of columns to show on hover
    title : str, optional
        Map title
    height : int, optional
        Map height in pixels
    width : int, optional
        Map width in pixels
    color_scale : str, optional
        Plotly color scale name
    fallback_cmap : str, optional
        Matplotlib colormap name for fallback
    center : dict, optional
        Map center as {'lat': float, 'lon': float}
    zoom : float, optional
        Initial zoom level
    opacity : float, optional
        Marker opacity (0-1)
    mapbox_style : str, optional
        Mapbox style for background map
    
    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Interactive map or static fallback
    """
    # Validate input
    if not hasattr(gdf, 'geometry'):
        raise ValueError("Input must be a GeoDataFrame with geometry column")
    
    # Check for plotly
    if has_plotly():
        return _create_plotly_map(
            gdf, color_col, size_col, hover_data, title, height, width,
            color_scale, center, zoom, opacity, mapbox_style
        )
    else:
        logger.info("Plotly not available, falling back to matplotlib")
        return _create_matplotlib_map(
            gdf, color_col, size_col, title, fallback_cmap
        )


@handle_errors(logger=logger, error_type=(ValueError, ImportError, AttributeError))
def _create_plotly_map(
    gdf, color_col, size_col, hover_data, title, height, width,
    color_scale, center, zoom, opacity, mapbox_style
):
    """
    Create an interactive map with plotly.express.
    
    Parameters
    ----------
    See plot_interactive_map for parameter descriptions
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive map
    """
    import plotly.express as px
    
    # Prepare GeoDataFrame for plotly
    gdf_copy = gdf.copy()
    
    # Convert geometry to lat/lon
    if not all(geom.geom_type == 'Point' for geom in gdf_copy.geometry):
        # Convert non-point geometries to points (centroids)
        gdf_copy['geometry'] = gdf_copy.geometry.centroid
    
    # Extract lat/lon
    gdf_copy['lon'] = gdf_copy.geometry.x
    gdf_copy['lat'] = gdf_copy.geometry.y
    
    # Set hover data
    if hover_data is None:
        # Select a few default columns for hover (exclude geometry and lat/lon)
        hover_data = [col for col in gdf_copy.columns 
                     if col not in ['geometry', 'lat', 'lon', color_col, size_col]][:5]
    
    # Create plot
    fig = px.scatter_mapbox(
        gdf_copy,
        lat='lat',
        lon='lon',
        color=color_col,
        size=size_col,
        hover_name=hover_data[0] if hover_data else None,
        hover_data=hover_data,
        color_continuous_scale=color_scale,
        opacity=opacity,
        title=title,
        height=height,
        width=width,
        mapbox_style=mapbox_style
    )
    
    # Set center and zoom
    if center is None and not gdf_copy.empty:
        # Use centroid of all points
        center = {
            'lat': gdf_copy['lat'].mean(),
            'lon': gdf_copy['lon'].mean()
        }
    
    if zoom is None:
        # Estimate zoom based on data extent
        lat_range = gdf_copy['lat'].max() - gdf_copy['lat'].min()
        lon_range = gdf_copy['lon'].max() - gdf_copy['lon'].min()
        max_range = max(lat_range, lon_range)
        if max_range < 0.1:
            zoom = 12  # City level
        elif max_range < 1:
            zoom = 10  # Metropolitan area
        elif max_range < 5:
            zoom = 8   # Region
        else:
            zoom = 6   # Country
    
    fig.update_layout(
        mapbox=dict(
            center=center,
            zoom=zoom
        )
    )
    
    return fig


@handle_errors(logger=logger)
def _create_matplotlib_map(gdf, color_col, size_col, title, cmap):
    """
    Create a static map with matplotlib as fallback.
    
    Parameters
    ----------
    See plot_interactive_map for parameter descriptions
    
    Returns
    -------
    matplotlib.figure.Figure
        Static map
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot with color if specified
    if color_col and color_col in gdf.columns:
        gdf.plot(
            column=color_col,
            cmap=cmap,
            legend=True,
            ax=ax,
            markersize=50 if size_col is None else None,
            alpha=0.8
        )
    else:
        gdf.plot(ax=ax, markersize=50 if size_col is None else None, alpha=0.8)
    
    # Adjust point size if specified
    if size_col and size_col in gdf.columns:
        # Normalize sizes between 20 and 150
        sizes = gdf[size_col]
        if sizes.min() < sizes.max():  # Avoid division by zero
            norm_sizes = 20 + 130 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            norm_sizes = 50  # Default size if all values are the same
            
        for idx, row in gdf.iterrows():
            ax.scatter(
                row.geometry.x, row.geometry.y,
                s=norm_sizes.loc[idx],
                color='none' if color_col else 'blue',
                edgecolor='k', linewidth=0.5,
                alpha=0.8
            )
    
    if title:
        ax.set_title(title)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    
    return fig


@timer
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, ImportError, AttributeError))
def plot_interactive_market_integration(
    markets_gdf,
    integration_col: str,
    conflict_col: Optional[str] = None,
    title: str = "Market Integration in Yemen",
    height: int = 700,
    width: int = 1000,
    color_scale: str = 'RdYlGn',
    fallback_cmap: str = 'RdYlGn',
    hover_data: Optional[List[str]] = None,
    show_exchange_boundary: bool = True,
    exchange_regime_col: str = 'exchange_rate_regime',
    mapbox_style: str = 'open-street-map'
) -> Any:
    """
    Create an interactive map of Yemen showing market integration levels.
    
    Specialized for Yemen market analysis with appropriate basemap and additional
    controls for conflict and exchange rate regime visualization.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        GeoDataFrame with market locations and integration metric
    integration_col : str
        Column name for integration metric
    conflict_col : str, optional
        Column name for conflict intensity
    title : str, optional
        Map title
    height : int, optional
        Map height in pixels
    width : int, optional
        Map width in pixels
    color_scale : str, optional
        Plotly color scale name
    fallback_cmap : str, optional
        Matplotlib colormap name for fallback
    hover_data : list, optional
        List of columns to show on hover
    show_exchange_boundary : bool, optional
        Whether to show exchange rate regime boundary
    exchange_regime_col : str, optional
        Column name for exchange rate regime
    mapbox_style : str, optional
        Mapbox style for background map
    
    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Interactive map or static fallback
    """
    # Validate inputs
    if not isinstance(markets_gdf, pd.DataFrame) or not hasattr(markets_gdf, 'geometry'):
        raise ValueError("markets_gdf must be a GeoDataFrame")
    
    if integration_col not in markets_gdf.columns:
        raise ValueError(f"Column {integration_col} not found in GeoDataFrame")
    
    # Set default hover data if not provided
    if hover_data is None:
        hover_data = [
            integration_col,
            exchange_regime_col if exchange_regime_col in markets_gdf.columns else None,
            conflict_col if conflict_col else None
        ]
        hover_data = [col for col in hover_data if col]
    
    # Check for plotly
    if has_plotly():
        # Use plotly for interactive visualization
        fig = _create_interactive_market_map(
            markets_gdf, integration_col, conflict_col, title, height, width,
            color_scale, hover_data, show_exchange_boundary, exchange_regime_col,
            mapbox_style
        )
        return fig
    else:
        # Fall back to matplotlib
        logger.info("Plotly not available, falling back to matplotlib")
        return plot_yemen_market_integration(
            markets_gdf,
            integration_col=integration_col,
            title=title,
            cmap=fallback_cmap,
            figsize=(width/100, height/100)  # Convert pixels to inches (approximate)
        )


@handle_errors(logger=logger)
def _create_interactive_market_map(
    markets_gdf, integration_col, conflict_col, title, height, width,
    color_scale, hover_data, show_exchange_boundary, exchange_regime_col,
    mapbox_style
):
    """
    Create an interactive market map with plotly.
    
    Parameters
    ----------
    See plot_interactive_market_integration for parameter descriptions
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive map
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Prepare GeoDataFrame
    gdf_copy = markets_gdf.copy()
    
    # Convert geometry to lat/lon
    if not all(geom.geom_type == 'Point' for geom in gdf_copy.geometry):
        gdf_copy['geometry'] = gdf_copy.geometry.centroid
    
    gdf_copy['lon'] = gdf_copy.geometry.x
    gdf_copy['lat'] = gdf_copy.geometry.y
    
    # Create base map
    fig = px.scatter_mapbox(
        gdf_copy,
        lat='lat',
        lon='lon',
        color=integration_col,
        size=conflict_col if conflict_col else None,
        hover_name=hover_data[0] if hover_data else None,
        hover_data=hover_data,
        color_continuous_scale=color_scale,
        opacity=0.8,
        title=title,
        height=height,
        width=width,
        mapbox_style=mapbox_style
    )
    
    # Add exchange rate boundary if requested
    if show_exchange_boundary and exchange_regime_col in gdf_copy.columns:
        try:
            from src.utils import calculate_exchange_rate_boundary
            
            # Calculate boundary
            boundary = calculate_exchange_rate_boundary(gdf_copy, exchange_regime_col)
            
            # Plot boundary if exists
            if not boundary.empty:
                for idx, row in boundary.iterrows():
                    # Convert linestring to lat/lon pairs
                    if hasattr(row.geometry, 'coords'):
                        coords = list(row.geometry.coords)
                        lons = [coord[0] for coord in coords]
                        lats = [coord[1] for coord in coords]
                        
                        # Add line to the map
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=lats,
                                lon=lons,
                                mode='lines',
                                line=dict(width=3, color='red'),
                                name='Exchange Rate Boundary'
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not plot exchange rate boundary: {str(e)}")
    
    # Set center and zoom
    center = {'lat': gdf_copy['lat'].mean(), 'lon': gdf_copy['lon'].mean()}
    
    # Estimate zoom based on data extent
    lat_range = gdf_copy['lat'].max() - gdf_copy['lat'].min()
    lon_range = gdf_copy['lon'].max() - gdf_copy['lon'].min()
    max_range = max(lat_range, lon_range)
    
    if max_range < 0.1:
        zoom = 12  # City level
    elif max_range < 1:
        zoom = 10  # Metropolitan area
    elif max_range < 5:
        zoom = 8   # Region
    else:
        zoom = 6   # Country
    
    fig.update_layout(
        mapbox=dict(
            center=center,
            zoom=zoom
        )
    )
    
    return fig


@timer
@handle_errors(logger=logger, error_type=(ValueError, ImportError, AttributeError))
def plot_interactive_time_series(
    df: pd.DataFrame,
    x: str,
    y_columns: Union[str, List[str]],
    group_col: Optional[str] = None,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 900,
    color_sequence: Optional[List[str]] = None,
    hover_data: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    range_slider: bool = True
) -> Any:
    """
    Create an interactive time series plot.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot
    x : str
        Column name for x-axis (typically dates)
    y_columns : str or list
        Column name(s) for y-axis values
    group_col : str, optional
        Column to group by (for multiple series)
    title : str, optional
        Plot title
    height : int, optional
        Plot height in pixels
    width : int, optional
        Plot width in pixels
    color_sequence : list, optional
        List of colors for series
    hover_data : list, optional
        Additional columns to show on hover
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    legend_title : str, optional
        Title for legend
    range_slider : bool, optional
        Whether to include range slider
    
    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Interactive time series or static fallback
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if x not in df.columns:
        raise ValueError(f"Column {x} not found in DataFrame")
    
    # Convert single y column to list
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    
    # Validate y columns
    missing_cols = [col for col in y_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) not found in DataFrame: {', '.join(missing_cols)}")
    
    # Check for plotly
    if has_plotly():
        return _create_plotly_time_series(
            df, x, y_columns, group_col, title, height, width,
            color_sequence, hover_data, xlabel, ylabel, legend_title, range_slider
        )
    else:
        logger.info("Plotly not available, falling back to matplotlib")
        return _create_matplotlib_time_series(
            df, x, y_columns, group_col, title, 
            xlabel, ylabel, legend_title
        )


@handle_errors(logger=logger)
def _create_plotly_time_series(
    df, x, y_columns, group_col, title, height, width,
    color_sequence, hover_data, xlabel, ylabel, legend_title, range_slider
):
    """
    Create an interactive time series with plotly.
    
    Parameters
    ----------
    See plot_interactive_time_series for parameter descriptions
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive time series
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Set up hover data
    if hover_data is None:
        hover_data = []
    
    # Use a different approach depending on whether we're grouping
    if group_col:
        # Long-form data with grouping
        # First convert to long form if needed
        if len(y_columns) > 1:
            plot_df = df.melt(
                id_vars=[x, group_col] + hover_data, 
                value_vars=y_columns,
                var_name='variable',
                value_name='value'
            )
            fig = px.line(
                plot_df,
                x=x,
                y='value',
                color=group_col,
                line_dash='variable' if len(y_columns) > 1 else None,
                labels={'value': ylabel or 'Value'},
                title=title,
                height=height,
                width=width,
                color_discrete_sequence=color_sequence,
                hover_data=hover_data
            )
        else:
            # Single y column with grouping
            fig = px.line(
                df,
                x=x,
                y=y_columns[0],
                color=group_col,
                title=title,
                height=height,
                width=width,
                color_discrete_sequence=color_sequence,
                hover_data=hover_data
            )
    else:
        # No grouping, plot multiple y columns
        fig = go.Figure()
        
        # Default to COLOR_PALETTES if no sequence provided
        if color_sequence is None:
            color_sequence = COLOR_PALETTES['default']
        
        # Add each y column as a separate trace
        for i, col in enumerate(y_columns):
            color = color_sequence[i % len(color_sequence)]
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(color=color),
                    hovertemplate=f"{col}: %{{y}}<br>{x}: %{{x}}"
                )
            )
        
        fig.update_layout(
            title=title,
            height=height,
            width=width
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=xlabel or x,
        yaxis_title=ylabel or (y_columns[0] if len(y_columns) == 1 else 'Value'),
        legend_title=legend_title or group_col or 'Series',
        hovermode='closest'
    )
    
    # Add range slider if requested
    if range_slider:
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date' if pd.api.types.is_datetime64_any_dtype(df[x]) else '-'
            )
        )
    
    return fig


@handle_errors(logger=logger)
def _create_matplotlib_time_series(
    df, x, y_columns, group_col, title, xlabel, ylabel, legend_title
):
    """
    Create a static time series with matplotlib as fallback.
    
    Parameters
    ----------
    See plot_interactive_time_series for parameter descriptions
    
    Returns
    -------
    matplotlib.figure.Figure
        Static time series
    """
    # Use existing plotting functions
    if group_col:
        # Use plot_time_series_by_group for grouped data
        if len(y_columns) > 1:
            logger.warning("Multiple y columns with grouping not fully supported in matplotlib fallback")
        
        fig, ax = plot_time_series_by_group(
            df,
            x=x,
            y=y_columns[0],
            group=group_col,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel
        )
    else:
        # Use plot_multiple_time_series for multiple columns
        fig, ax = plot_multiple_time_series(
            df,
            x=x,
            y_columns=y_columns,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel
        )
    
    return fig


@timer
@m1_optimized(parallel=True)
@handle_errors(logger=logger, error_type=(ValueError, ImportError, AttributeError))
def plot_interactive_heatmap(
    df: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    values: Optional[str] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 900,
    color_scale: str = 'RdBu_r',
    fallback_cmap: str = 'RdBu_r',
    hover_data: Optional[List[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    annotation_format: str = ".1f",
    show_colorbar: bool = True,
    colorbar_title: Optional[str] = None
) -> Any:
    """
    Create an interactive heatmap with plotly.
    Falls back to matplotlib if plotly is not available.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to plot (can be pivoted or unpivoted)
    x : str, optional
        Column name for x-axis (required for unpivoted data)
    y : str, optional
        Column name for y-axis (required for unpivoted data)
    values : str, optional
        Column name for values (required for unpivoted data)
    title : str, optional
        Heatmap title
    height : int, optional
        Plot height in pixels
    width : int, optional
        Plot width in pixels
    color_scale : str, optional
        Plotly color scale name
    fallback_cmap : str, optional
        Matplotlib colormap name for fallback
    hover_data : list, optional
        Additional columns to show on hover (for unpivoted data)
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    zmin : float, optional
        Minimum value for color scale
    zmax : float, optional
        Maximum value for color scale
    annotation_format : str, optional
        Format string for cell annotations
    show_colorbar : bool, optional
        Whether to show colorbar
    colorbar_title : str, optional
        Title for colorbar
    
    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Interactive heatmap or static fallback
    """
    # Check input format
    pivoted = x is None and y is None and values is None
    
    # For pivoted data, we use the dataframe directly
    # For unpivoted data, we need to pivot it first
    if not pivoted:
        # Validate unpivoted inputs
        if x is None or y is None or values is None:
            raise ValueError("For unpivoted data, x, y, and values must be provided")
        
        if x not in df.columns or y not in df.columns or values not in df.columns:
            missing = []
            if x not in df.columns:
                missing.append(f"x: {x}")
            if y not in df.columns:
                missing.append(f"y: {y}")
            if values not in df.columns:
                missing.append(f"values: {values}")
            raise ValueError(f"Column(s) not found in DataFrame: {', '.join(missing)}")
        
        # Pivot the data
        plot_df = df.pivot(index=y, columns=x, values=values)
    else:
        # Use the data as-is
        plot_df = df.copy()
    
    # Check for plotly
    if has_plotly():
        return _create_plotly_heatmap(
            plot_df, title, height, width, color_scale,
            hover_data, xlabel, ylabel, zmin, zmax, 
            annotation_format, show_colorbar, colorbar_title
        )
    else:
        logger.info("Plotly not available, falling back to matplotlib")
        return _create_matplotlib_heatmap(
            plot_df, title, fallback_cmap, xlabel, ylabel,
            zmin, zmax, annotation_format
        )


@handle_errors(logger=logger)
def _create_plotly_heatmap(
    df, title, height, width, color_scale,
    hover_data, xlabel, ylabel, zmin, zmax, 
    annotation_format, show_colorbar, colorbar_title
):
    """
    Create an interactive heatmap with plotly.
    
    Parameters
    ----------
    See plot_interactive_heatmap for parameter descriptions
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive heatmap
    """
    import plotly.express as px
    
    # Create the heatmap
    fig = px.imshow(
        df,
        color_continuous_scale=color_scale,
        title=title,
        height=height,
        width=width,
        zmin=zmin,
        zmax=zmax,
        labels=dict(color=colorbar_title or "Value"),
        x=df.columns.tolist(),
        y=df.index.tolist()
    )
    
    # Add annotations if the data isn't too large
    if df.size <= 400:  # Reasonable limit for annotations
        for y in range(len(df.index)):
            for x in range(len(df.columns)):
                value = df.iloc[y, x]
                if pd.notna(value):
                    fig.add_annotation(
                        x=df.columns[x],
                        y=df.index[y],
                        text=f"{value:{annotation_format}}",
                        showarrow=False,
                        font=dict(color="white" if abs(value) > (df.values.max() - df.values.min()) / 2 else "black")
                    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=xlabel or "X",
        yaxis_title=ylabel or "Y",
        coloraxis_showscale=show_colorbar
    )
    
    return fig


@handle_errors(logger=logger)
def _create_matplotlib_heatmap(
    df, title, cmap, xlabel, ylabel, vmin, vmax, annotation_format
):
    """
    Create a static heatmap with matplotlib as fallback.
    
    Parameters
    ----------
    See plot_interactive_heatmap for parameter descriptions
    
    Returns
    -------
    matplotlib.figure.Figure
        Static heatmap
    """
    # Use existing plot_heatmap function
    fig, ax = plot_heatmap(
        df,
        cmap=cmap,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        annot=True,
        fmt=annotation_format,
        center=None,
        cbar=True
    )
    
    # Set color limits if provided
    if vmin is not None or vmax is not None:
        im = ax.collections[0]
        im.set_clim(vmin, vmax)
    
    return fig


@timer
@handle_errors(logger=logger, error_type=(ValueError, ImportError, AttributeError))
def plot_interactive_dashboard(
    data_dict: Dict[str, Dict[str, Any]],
    layout: Optional[List[List[str]]] = None,
    title: str = "Yemen Market Integration Dashboard",
    height: int = 1000,
    width: int = 1200
) -> Any:
    """
    Create an interactive dashboard with multiple plots.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of data and plot specifications
        Format: {
            'plot1': {
                'type': 'map|timeseries|heatmap',
                'data': DataFrame or GeoDataFrame,
                'params': dict of plot-specific parameters
            },
            'plot2': {...},
            ...
        }
    layout : list, optional
        List of lists specifying grid layout
        e.g. [['plot1', 'plot2'], ['plot3', 'plot3']]
    title : str, optional
        Dashboard title
    height : int, optional
        Dashboard height in pixels
    width : int, optional
        Dashboard width in pixels
    
    Returns
    -------
    plotly.graph_objects.Figure or dict
        Interactive dashboard or dictionary of matplotlib figures
    """
    # Check for plotly
    if not has_plotly():
        logger.info("Plotly not available, creating separate matplotlib figures")
        return _create_matplotlib_dashboard(data_dict, title)
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Determine layout if not provided
    if layout is None:
        num_plots = len(data_dict)
        if num_plots == 1:
            layout = [[list(data_dict.keys())[0]]]
        elif num_plots == 2:
            layout = [[list(data_dict.keys())[0]], [list(data_dict.keys())[1]]]
        elif num_plots == 3:
            layout = [
                [list(data_dict.keys())[0], list(data_dict.keys())[1]],
                [list(data_dict.keys())[2], list(data_dict.keys())[2]]
            ]
        elif num_plots == 4:
            layout = [
                [list(data_dict.keys())[0], list(data_dict.keys())[1]],
                [list(data_dict.keys())[2], list(data_dict.keys())[3]]
            ]
        else:
            # For more plots, use a more complex layout
            layout = []
            plots_per_row = 2
            keys = list(data_dict.keys())
            for i in range(0, num_plots, plots_per_row):
                row = []
                for j in range(plots_per_row):
                    if i + j < num_plots:
                        row.append(keys[i + j])
                    else:
                        row.append(None)  # Empty cell
                layout.append(row)
    
    # Create subplot grid
    specs = []
    subplot_titles = []
    
    for row in layout:
        row_specs = []
        for plot_id in row:
            if plot_id is None:
                row_specs.append(None)
            else:
                plot_type = data_dict[plot_id].get('type', 'timeseries')
                row_specs.append({'type': 'mapbox' if plot_type == 'map' else None})
                subplot_titles.append(data_dict[plot_id]['params'].get('title', plot_id))
        specs.append(row_specs)
    
    fig = make_subplots(
        rows=len(layout),
        cols=max(len(row) for row in layout),
        subplot_titles=subplot_titles,
        specs=specs
    )
    
    # Add plots to the dashboard
    for i, row in enumerate(layout):
        for j, plot_id in enumerate(row):
            if plot_id is None or plot_id not in data_dict:
                continue
                
            plot_spec = data_dict[plot_id]
            plot_type = plot_spec.get('type', 'timeseries')
            data = plot_spec.get('data')
            params = plot_spec.get('params', {})
            
            # Create individual plot
            if plot_type == 'map':
                _add_map_to_dashboard(fig, data, params, i+1, j+1)
            elif plot_type == 'timeseries':
                _add_timeseries_to_dashboard(fig, data, params, i+1, j+1)
            elif plot_type == 'heatmap':
                _add_heatmap_to_dashboard(fig, data, params, i+1, j+1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


@handle_errors(logger=logger)
def _add_map_to_dashboard(fig, data, params, row, col):
    """
    Add a map to the dashboard.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Dashboard figure
    data : geopandas.GeoDataFrame
        Data for the map
    params : dict
        Plot parameters
    row : int
        Row index (1-based)
    col : int
        Column index (1-based)
    """
    import plotly.express as px
    
    # Extract lat/lon
    gdf_copy = data.copy()
    
    # Convert geometry to lat/lon
    if not all(geom.geom_type == 'Point' for geom in gdf_copy.geometry):
        gdf_copy['geometry'] = gdf_copy.geometry.centroid
    
    gdf_copy['lon'] = gdf_copy.geometry.x
    gdf_copy['lat'] = gdf_copy.geometry.y
    
    # Get parameters
    color_col = params.get('color_col')
    size_col = params.get('size_col')
    hover_data = params.get('hover_data', [])
    
    # Create temporary figure
    temp_fig = px.scatter_mapbox(
        gdf_copy,
        lat='lat',
        lon='lon',
        color=color_col,
        size=size_col,
        hover_name=hover_data[0] if hover_data else None,
        hover_data=hover_data,
        color_continuous_scale=params.get('color_scale', 'viridis'),
        opacity=params.get('opacity', 0.8),
        mapbox_style=params.get('mapbox_style', 'open-street-map')
    )
    
    # Transfer traces to main figure
    for trace in temp_fig.data:
        fig.add_trace(trace, row=row, col=col)
    
    # Update mapbox settings
    center = {'lat': gdf_copy['lat'].mean(), 'lon': gdf_copy['lon'].mean()}
    zoom = params.get('zoom', 6)
    
    fig.update_mapboxes(
        center=center,
        zoom=zoom,
        style=params.get('mapbox_style', 'open-street-map'),
        row=row,
        col=col
    )


@handle_errors(logger=logger)
def _add_timeseries_to_dashboard(fig, data, params, row, col):
    """
    Add a time series to the dashboard.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Dashboard figure
    data : pandas.DataFrame
        Data for the time series
    params : dict
        Plot parameters
    row : int
        Row index (1-based)
    col : int
        Column index (1-based)
    """
    import plotly.graph_objects as go
    
    # Get parameters
    x = params.get('x')
    y_columns = params.get('y_columns', [])
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    
    group_col = params.get('group_col')
    color_sequence = params.get('color_sequence', COLOR_PALETTES['default'])
    
    if group_col:
        # Group data
        for group_val in data[group_col].unique():
            group_data = data[data[group_col] == group_val]
            for i, col_name in enumerate(y_columns):
                color_idx = list(data[group_col].unique()).index(group_val)
                color = color_sequence[color_idx % len(color_sequence)]
                
                fig.add_trace(
                    go.Scatter(
                        x=group_data[x],
                        y=group_data[col_name],
                        mode='lines',
                        name=f"{group_val} - {col_name}",
                        line=dict(color=color),
                        legendgroup=str(group_val)
                    ),
                    row=row, col=col
                )
    else:
        # No grouping
        for i, col_name in enumerate(y_columns):
            color = color_sequence[i % len(color_sequence)]
            
            fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[col_name],
                    mode='lines',
                    name=col_name,
                    line=dict(color=color)
                ),
                row=row, col=col
            )
    
    # Update axes
    fig.update_xaxes(
        title_text=params.get('xlabel', x),
        row=row, col=col
    )
    
    fig.update_yaxes(
        title_text=params.get('ylabel', y_columns[0] if y_columns else ''),
        row=row, col=col
    )


@handle_errors(logger=logger)
def _add_heatmap_to_dashboard(fig, data, params, row, col):
    """
    Add a heatmap to the dashboard.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Dashboard figure
    data : pandas.DataFrame
        Data for the heatmap
    params : dict
        Plot parameters
    row : int
        Row index (1-based)
    col : int
        Column index (1-based)
    """
    import plotly.graph_objects as go
    
    # Get parameters
    x = params.get('x')
    y = params.get('y')
    values = params.get('values')
    
    # Check if data needs pivoting
    if x is not None and y is not None and values is not None:
        # Pivot the data
        data = data.pivot(index=y, columns=x, values=values)
    
    # Create heatmap trace
    fig.add_trace(
        go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=params.get('color_scale', 'RdBu_r'),
            zmin=params.get('zmin'),
            zmax=params.get('zmax'),
            colorbar=dict(
                title=params.get('colorbar_title', ''),
                len=0.5,
                y=0.5
            ),
            showscale=params.get('show_colorbar', True)
        ),
        row=row, col=col
    )
    
    # Update axes
    fig.update_xaxes(
        title_text=params.get('xlabel', ''),
        row=row, col=col
    )
    
    fig.update_yaxes(
        title_text=params.get('ylabel', ''),
        row=row, col=col
    )


@handle_errors(logger=logger)
def _create_matplotlib_dashboard(data_dict, title):
    """
    Create a collection of matplotlib figures as fallback.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary of data and plot specifications
    title : str
        Dashboard title
    
    Returns
    -------
    dict
        Dictionary of matplotlib figures
    """
    figures = {}
    
    for plot_id, plot_spec in data_dict.items():
        plot_type = plot_spec.get('type', 'timeseries')
        data = plot_spec.get('data')
        params = plot_spec.get('params', {})
        
        if plot_type == 'map':
            figures[plot_id] = _create_matplotlib_map(
                data,
                params.get('color_col'),
                params.get('size_col'),
                params.get('title', plot_id),
                params.get('cmap', 'viridis')
            )
        elif plot_type == 'timeseries':
            if 'group_col' in params:
                figures[plot_id] = _create_matplotlib_time_series(
                    data,
                    params.get('x'),
                    params.get('y_columns'),
                    params.get('group_col'),
                    params.get('title', plot_id),
                    params.get('xlabel'),
                    params.get('ylabel'),
                    params.get('legend_title')
                )
            else:
                figures[plot_id] = _create_matplotlib_time_series(
                    data,
                    params.get('x'),
                    params.get('y_columns'),
                    None,
                    params.get('title', plot_id),
                    params.get('xlabel'),
                    params.get('ylabel'),
                    params.get('legend_title')
                )
        elif plot_type == 'heatmap':
            # Handle pivoting if needed
            if all(k in params for k in ['x', 'y', 'values']):
                plot_data = data.pivot(
                    index=params['y'],
                    columns=params['x'],
                    values=params['values']
                )
            else:
                plot_data = data
                
            figures[plot_id] = _create_matplotlib_heatmap(
                plot_data,
                params.get('title', plot_id),
                params.get('cmap', 'RdBu_r'),
                params.get('xlabel'),
                params.get('ylabel'),
                params.get('vmin'),
                params.get('vmax'),
                params.get('annotation_format', '.1f')
            )
    
    return figures