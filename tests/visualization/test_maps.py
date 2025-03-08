# tests/visualization/test_maps.py

import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from src.visualization import MarketMapVisualizer

@pytest.fixture
def sample_market_data():
    """Create sample market GeoDataFrame for testing."""
    # Create sample data for Yemen markets
    markets = [
        {'market_id': 1, 'market': 'Sana\'a', 'price': 150, 'conflict_intensity': 0.8, 
         'exchange_rate_regime': 'north', 'isolation_index': 0.7, 'x': 44.2, 'y': 15.35},
        {'market_id': 2, 'market': 'Aden', 'price': 130, 'conflict_intensity': 0.4, 
         'exchange_rate_regime': 'south', 'isolation_index': 0.3, 'x': 45.0, 'y': 12.77},
        {'market_id': 3, 'market': 'Taiz', 'price': 140, 'conflict_intensity': 0.9, 
         'exchange_rate_regime': 'north', 'isolation_index': 0.8, 'x': 44.0, 'y': 13.58},
        {'market_id': 4, 'market': 'Hudaydah', 'price': 145, 'conflict_intensity': 0.6, 
         'exchange_rate_regime': 'north', 'isolation_index': 0.5, 'x': 42.95, 'y': 14.8},
        {'market_id': 5, 'market': 'Mukalla', 'price': 125, 'conflict_intensity': 0.2, 
         'exchange_rate_regime': 'south', 'isolation_index': 0.2, 'x': 49.12, 'y': 14.53}
    ]
    
    # Convert to GeoDataFrame with Point geometry
    df = pd.DataFrame(markets)
    geometry = [Point(x, y) for x, y in zip(df['x'], df['y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Add a simulated version with different prices
    simulated = gdf.copy()
    simulated['price'] = simulated['price'] * 0.9  # 10% lower prices
    
    return {'original': gdf, 'simulated': simulated}

@pytest.fixture
def visualizer():
    """Create a MarketMapVisualizer instance for testing."""
    return MarketMapVisualizer()

def test_init(visualizer):
    """Test initializing the visualizer."""
    assert visualizer is not None
    assert hasattr(visualizer, 'fig_width')
    assert hasattr(visualizer, 'fig_height')
    assert hasattr(visualizer, 'dpi')
    assert hasattr(visualizer, 'cmap')
    assert hasattr(visualizer, 'north_color')
    assert hasattr(visualizer, 'south_color')
    assert hasattr(visualizer, 'yemen_crs')

def test_plot_static_map(visualizer, sample_market_data):
    """Test plotting a static map."""
    gdf = sample_market_data['original']
    
    fig, ax = visualizer.plot_static_map(
        gdf,
        column='price',
        title='Test Static Map',
        add_basemap=False  # Skip basemap for testing
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Test Static Map'
    plt.close(fig)

def test_plot_static_map_with_regime(visualizer, sample_market_data):
    """Test plotting a static map with exchange rate regime."""
    gdf = sample_market_data['original']
    
    fig, ax = visualizer.plot_static_map(
        gdf,
        column='exchange_rate_regime',
        title='Test Regime Map',
        add_basemap=False  # Skip basemap for testing
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Test Regime Map'
    plt.close(fig)

def test_create_interactive_map(visualizer, sample_market_data):
    """Test creating an interactive map."""
    try:
        gdf = sample_market_data['original']
        
        m = visualizer.create_interactive_map(
            gdf,
            column='price',
            popup_cols=['market', 'price', 'exchange_rate_regime'],
            title='Test Interactive Map'
        )
        
        assert m is not None
    except ImportError:
        # Skip if folium is not available
        pytest.skip("Folium not available for testing")

def test_create_market_integration_map(visualizer, sample_market_data):
    """Test creating a market integration map."""
    gdf = sample_market_data['original']
    
    fig, ax = visualizer.create_market_integration_map(
        gdf,
        isolation_col='isolation_index',
        title='Test Integration Map'
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Test Integration Map'
    plt.close(fig)

def test_plot_policy_impact_map(visualizer, sample_market_data):
    """Test plotting a policy impact map."""
    original = sample_market_data['original']
    simulated = sample_market_data['simulated']
    
    fig, ax = visualizer.plot_policy_impact_map(
        original,
        simulated,
        metric_col='price',
        title='Test Policy Impact'
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == 'Test Policy Impact'
    plt.close(fig)

def test_invalid_geodataframe(visualizer):
    """Test handling invalid GeoDataFrames."""
    empty_gdf = gpd.GeoDataFrame()
    
    with pytest.raises(Exception):
        visualizer.plot_static_map(empty_gdf)
        
    with pytest.raises(Exception):
        visualizer.create_market_integration_map(empty_gdf)