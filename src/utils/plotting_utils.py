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

from src.utils.error_handler import handle_errors, VisualizationError

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