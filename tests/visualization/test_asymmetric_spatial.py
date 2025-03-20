"""
Test module for the specialized visualization components.

This module tests the asymmetric adjustment and spatial integration
visualization components implemented for the Yemen Market Integration project.
"""
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import tempfile
import os
import logging
from unittest.mock import patch, MagicMock

# Prevent logging during tests
logging.disable(logging.CRITICAL)

from src.visualization.asymmetric_plots import AsymmetricAdjustmentVisualizer
from src.visualization.spatial_integration import SpatialIntegrationVisualizer

class TestAsymmetricAdjustmentVisualizer(unittest.TestCase):
    """Test cases for the AsymmetricAdjustmentVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures, creating sample time series data."""
        np.random.seed(42)  # For reproducibility
        
        # Generate dates
        self.dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # Generate price differential with threshold behavior
        threshold = 20.0
        above_threshold_indices = np.random.choice(range(100), size=30, replace=False)
        below_threshold_indices = np.random.choice(
            list(set(range(100)) - set(above_threshold_indices)), 
            size=30, 
            replace=False
        )
        
        # Create price differential array
        self.price_diff = np.random.normal(0, 10, 100)
        
        # Set values above and below threshold
        self.price_diff[above_threshold_indices] = threshold + np.random.normal(10, 5, len(above_threshold_indices))
        self.price_diff[below_threshold_indices] = -threshold - np.random.normal(10, 5, len(below_threshold_indices))
        
        # Create mock threshold model
        self.mock_threshold_model = self._create_mock_threshold_model()
        
        # Create visualizer
        self.visualizer = AsymmetricAdjustmentVisualizer()
        
    def _create_mock_threshold_model(self):
        """Create a mock threshold model for testing."""
        mock_model = MagicMock()
        
        # Set required attributes
        mock_model.price_diff = self.price_diff
        mock_model.dates = self.dates
        mock_model.threshold = 20.0
        
        # Set adjustment speeds
        mock_model.adjustment_below = -0.3
        mock_model.adjustment_middle = -0.1
        mock_model.adjustment_above = -0.2
        
        # Set half-lives
        mock_model.half_life_below = np.log(2) / 0.3
        mock_model.half_life_above = np.log(2) / 0.2
        
        return mock_model
    
    def test_init(self):
        """Test initialization of the visualizer."""
        visualizer = AsymmetricAdjustmentVisualizer()
        
        # Check default attributes
        self.assertIsNotNone(visualizer.fig_width)
        self.assertIsNotNone(visualizer.fig_height)
        self.assertIsNotNone(visualizer.above_color)
        self.assertIsNotNone(visualizer.below_color)
        self.assertIsNotNone(visualizer.middle_color)
    
    def test_plot_regime_dynamics(self):
        """Test plotting of regime dynamics."""
        # Plot regime dynamics
        fig, ax = self.visualizer.plot_regime_dynamics(
            self.price_diff,
            self.dates,
            threshold=20.0,
            title="Test Regime Dynamics"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check title
        self.assertEqual(ax.get_title(), "Test Regime Dynamics")
        
        # Clean up
        plt.close(fig)
    
    def test_plot_asymmetric_adjustment(self):
        """Test plotting of asymmetric adjustment patterns."""
        # Plot asymmetric adjustment
        fig, axs = self.visualizer.plot_asymmetric_adjustment(
            self.mock_threshold_model,
            title="Test Asymmetric Adjustment"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(axs, list)
        self.assertEqual(len(axs), 3)  # Three subplots
        
        # Clean up
        plt.close(fig)
    
    def test_plot_regime_transitions(self):
        """Test plotting of regime transitions."""
        # Plot regime transitions
        fig, axs = self.visualizer.plot_regime_transitions(
            self.price_diff,
            self.dates,
            threshold=20.0,
            title="Test Regime Transitions"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(axs, list)
        self.assertEqual(len(axs), 2)  # Two subplots
        
        # Check title
        self.assertEqual(fig._suptitle.get_text(), "Test Regime Transitions")
        
        # Clean up
        plt.close(fig)
    
    def test_compare_adjustment_patterns(self):
        """Test comparison of adjustment patterns."""
        # Create a second mock model with different adjustment speeds
        second_model = self._create_mock_threshold_model()
        second_model.adjustment_below = -0.2
        second_model.adjustment_above = -0.4
        second_model.half_life_below = np.log(2) / 0.2
        second_model.half_life_above = np.log(2) / 0.4
        
        # Plot comparison
        fig, axs = self.visualizer.compare_adjustment_patterns(
            self.mock_threshold_model,
            second_model,
            title="Test Comparison"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(axs, np.ndarray)
        self.assertEqual(axs.shape, (2, 2))  # 2x2 subplot grid
        
        # Clean up
        plt.close(fig)


class TestSpatialIntegrationVisualizer(unittest.TestCase):
    """Test cases for the SpatialIntegrationVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures, creating sample geospatial data."""
        np.random.seed(42)  # For reproducibility
        
        # Create sample market data
        n_markets = 10
        market_ids = [f'M{i}' for i in range(n_markets)]
        market_names = [f'Market {i}' for i in range(n_markets)]
        
        # Regions (5 north, 5 south)
        regions = ['north'] * 5 + ['south'] * 5
        
        # Coordinates (north markets at top, south at bottom)
        lats = np.linspace(15, 13, 5).tolist() + np.linspace(12, 10, 5).tolist()
        lons = np.linspace(44, 45, n_markets)
        
        # Integration metrics (random values)
        integration_metrics = np.random.uniform(0, 1, n_markets)
        
        # Conflict intensity (higher in some areas)
        conflict_raw = np.abs(np.random.normal(0, 1, n_markets))
        conflict_normalized = conflict_raw / conflict_raw.max()
        
        # Create data
        data = []
        for i in range(n_markets):
            data.append({
                'market_id': market_ids[i],
                'market_name': market_names[i],
                'exchange_rate_regime': regions[i],
                'integration_metric': integration_metrics[i],
                'conflict_intensity_normalized': conflict_normalized[i],
                'geometry': Point(lons[i], lats[i])
            })
        
        # Convert to GeoDataFrame
        self.market_gdf = gpd.GeoDataFrame(data, geometry='geometry')
        
        # Create sample edges GeoDataFrame
        edges = []
        for i in range(n_markets):
            for j in range(i+1, n_markets):
                # Create a random integration level (0, 1, or 2)
                integration_level = np.random.choice([0, 1, 2])
                
                # Create a LineString connecting the markets
                from shapely.geometry import LineString
                line = LineString([(lons[i], lats[i]), (lons[j], lats[j])])
                
                edges.append({
                    'market1_id': market_ids[i],
                    'market2_id': market_ids[j],
                    'integration_level_num': integration_level,
                    'geometry': line
                })
        
        # Convert to GeoDataFrame
        self.edges_gdf = gpd.GeoDataFrame(edges, geometry='geometry')
        
        # Create visualizer
        self.visualizer = SpatialIntegrationVisualizer()
        
        # Mock contextily for basemap tests
        self.has_contextily_backup = self.visualizer.has_contextily
        self.visualizer.has_contextily = False  # Disable for tests
        
    def tearDown(self):
        """Clean up after tests."""
        # Restore original contextily setting
        self.visualizer.has_contextily = self.has_contextily_backup
    
    def test_init(self):
        """Test initialization of the visualizer."""
        visualizer = SpatialIntegrationVisualizer()
        
        # Check default attributes
        self.assertIsNotNone(visualizer.fig_width)
        self.assertIsNotNone(visualizer.fig_height)
        self.assertIsNotNone(visualizer.integrated_color)
        self.assertIsNotNone(visualizer.partial_color)
        self.assertIsNotNone(visualizer.not_integrated_color)
    
    def test_plot_market_network(self):
        """Test plotting of market network."""
        # Plot market network
        fig, ax = self.visualizer.plot_market_network(
            self.market_gdf,
            edges_gdf=self.edges_gdf,
            market_id_col='market_id',
            market_name_col='market_name',
            region_col='exchange_rate_regime',
            edge_color_col='integration_level_num',
            title="Test Market Network"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check title
        self.assertEqual(ax.get_title(), "Test Market Network")
        
        # Clean up
        plt.close(fig)
    
    def test_plot_integration_choropleth(self):
        """Test plotting of integration choropleth map."""
        # Plot integration choropleth
        fig, ax = self.visualizer.plot_integration_choropleth(
            self.market_gdf,
            metric_col='integration_metric',
            market_id_col='market_id',
            region_col='exchange_rate_regime',
            title="Test Integration Choropleth"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check title
        self.assertEqual(ax.get_title(), "Test Integration Choropleth")
        
        # Clean up
        plt.close(fig)
    
    def test_plot_conflict_adjusted_network(self):
        """Test plotting of conflict-adjusted network."""
        # Plot conflict-adjusted network
        fig, ax = self.visualizer.plot_conflict_adjusted_network(
            self.market_gdf,
            conflict_col='conflict_intensity_normalized',
            region_col='exchange_rate_regime',
            market_id_col='market_id',
            title="Test Conflict-Adjusted Network"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Check title
        self.assertEqual(ax.get_title(), "Test Conflict-Adjusted Network")
        
        # Clean up
        plt.close(fig)
    
    def test_plot_market_integration_comparison(self):
        """Test plotting of market integration comparison."""
        # Create a modified version of the original data for simulated results
        simulated_gdf = self.market_gdf.copy()
        simulated_gdf['integration_metric'] = simulated_gdf['integration_metric'] * 1.2  # 20% improvement
        
        # Plot market integration comparison
        fig, axs = self.visualizer.plot_market_integration_comparison(
            self.market_gdf,
            simulated_gdf,
            metric_col='integration_metric',
            market_id_col='market_id',
            region_col='exchange_rate_regime',
            title="Test Integration Comparison"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(axs, np.ndarray)
        self.assertEqual(len(axs), 2)  # Two subplots
        
        # Check title
        self.assertTrue(hasattr(fig, '_suptitle'))
        self.assertEqual(fig._suptitle.get_text(), "Test Integration Comparison")
        
        # Clean up
        plt.close(fig)
    
    def test_basemap_usage(self):
        """Test behavior when basemap is requested but not available."""
        # Temporarily set has_contextily to False to simulate missing contextily
        self.visualizer.has_contextily = False
        
        # Plot should still work without errors
        fig, ax = self.visualizer.plot_market_network(
            self.market_gdf,
            basemap=True,  # Request basemap even though not available
            title="Test without Basemap"
        )
        
        # Check that plot was created
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        
        # Clean up
        plt.close(fig)
        
        # Simulate having contextily but with an error in basemap addition
        with patch('contextily.add_basemap', side_effect=Exception("Simulated basemap error")):
            self.visualizer.has_contextily = True
            
            # Plot should still work without errors
            fig, ax = self.visualizer.plot_market_network(
                self.market_gdf,
                basemap=True,
                title="Test with Basemap Error"
            )
            
            # Check that plot was created despite basemap error
            self.assertIsInstance(fig, plt.Figure)
            self.assertIsInstance(ax, plt.Axes)
            
            # Clean up
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
