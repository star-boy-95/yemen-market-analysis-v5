# tests/visualization/test_time_series.py

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.visualization import TimeSeriesVisualizer

@pytest.fixture
def sample_time_series():
    """Create sample time series data for testing."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    north_prices = 100 + np.random.normal(0, 5, 100).cumsum()
    south_prices = 90 + np.random.normal(0, 5, 100).cumsum()
    
    df = pd.DataFrame({
        'date': dates,
        'north_price': north_prices,
        'south_price': south_prices,
        'price_diff': north_prices - south_prices,
        'exchange_rate_regime': ['north'] * 50 + ['south'] * 50
    })
    
    return df

@pytest.fixture
def visualizer():
    """Create a TimeSeriesVisualizer instance for testing."""
    return TimeSeriesVisualizer()

def test_init(visualizer):
    """Test initializing the visualizer."""
    assert visualizer is not None
    assert hasattr(visualizer, 'fig_width')
    assert hasattr(visualizer, 'fig_height')
    assert hasattr(visualizer, 'dpi')
    assert hasattr(visualizer, 'date_format')

def test_plot_price_series(visualizer, sample_time_series):
    """Test plotting a price series."""
    fig, ax = visualizer.plot_price_series(
        sample_time_series,
        price_col='north_price',
        date_col='date',
        title='Test Price Series'
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Test Price Series'
    plt.close(fig)

def test_plot_price_series_with_groups(visualizer, sample_time_series):
    """Test plotting a price series with groups."""
    fig, ax = visualizer.plot_price_series(
        sample_time_series,
        price_col='north_price',
        date_col='date',
        group_col='exchange_rate_regime',
        title='Test Price Series by Regime'
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Test Price Series by Regime'
    assert len(ax.get_legend().get_texts()) == 2  # Two regimes
    plt.close(fig)

def test_plot_price_differentials(visualizer, sample_time_series):
    """Test plotting price differentials."""
    fig, ax = visualizer.plot_price_differentials(
        sample_time_series,
        date_col='date',
        north_col='north_price',
        south_col='south_price',
        title='Test Price Differentials'
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Test Price Differentials'
    plt.close(fig)

def test_plot_price_differentials_with_diff_col(visualizer, sample_time_series):
    """Test plotting price differentials with provided diff column."""
    fig, ax = visualizer.plot_price_differentials(
        sample_time_series,
        date_col='date',
        diff_col='price_diff',
        title='Test Price Differentials'
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Test Price Differentials'
    plt.close(fig)

def test_interactive_time_series(visualizer, sample_time_series):
    """Test creating an interactive time series plot."""
    try:
        fig = visualizer.plot_interactive_time_series(
            sample_time_series,
            price_col='north_price',
            date_col='date',
            title='Test Interactive Series'
        )
        assert fig is not None
    except ImportError:
        # Skip if plotly is not available
        pytest.skip("Plotly not available for testing")

def test_invalid_dataframe(visualizer):
    """Test handling invalid DataFrames."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(Exception):
        visualizer.plot_price_series(empty_df)
        
    with pytest.raises(Exception):
        visualizer.plot_price_differentials(empty_df)