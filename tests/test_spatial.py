"""
Unit tests for spatial econometrics module.
"""
import unittest
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.spatial import (
    SpatialEconometrics, 
    calculate_market_accessibility,
    calculate_market_isolation
)
from src.utils import ValidationError, ModelError


class TestSpatialEconometrics(unittest.TestCase):
    """Tests for the SpatialEconometrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test GeoDataFrame
        np.random.seed(42)
        n = 20
        
        # Create coordinates
        x = np.random.uniform(30, 50, n)
        y = np.random.uniform(10, 20, n)
        
        # Create data
        data = {
            'price': np.random.uniform(100, 200, n),
            'quantity': np.random.uniform(10, 30, n),
            'admin1': np.random.choice(['abyan', 'aden', 'sanaa', 'taiz'], n),
            'conflict_intensity_normalized': np.random.uniform(0, 1, n),
            'exchange_rate_regime': np.random.choice(['north', 'south'], n),
            'geometry': [Point(x[i], y[i]) for i in range(n)]
        }
        
        # Create GeoDataFrame
        self.gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        
        # Create a dummy population GeoDataFrame
        population_data = {
            'population': np.random.uniform(1000, 10000, n),
            'geometry': [Point(x[i] + np.random.uniform(-0.1, 0.1), 
                               y[i] + np.random.uniform(-0.1, 0.1)) 
                         for i in range(n)]
        }
        self.population_gdf = gpd.GeoDataFrame(population_data, crs="EPSG:4326")
    
    def test_initialization(self):
        """Test initialization of spatial econometrics model."""
        # Valid initialization
        model = SpatialEconometrics(self.gdf)
        self.assertEqual(len(model.gdf), len(self.gdf))
        
        # Test invalid inputs
        with self.assertRaises(ValidationError):
            # Not a GeoDataFrame
            SpatialEconometrics(pd.DataFrame({'x': [1, 2, 3]}))
    
    def test_weight_matrix(self):
        """Test creation of weight matrix."""
        model = SpatialEconometrics(self.gdf)
        
        # Test without conflict adjustment
        weights = model.create_weight_matrix(k=3, conflict_adjusted=False)
        self.assertEqual(weights.n, len(self.gdf))
        
        # Each location should have exactly 3 neighbors
        for neighbors in weights.neighbors.values():
            self.assertEqual(len(neighbors), 3)
        
        # Test with conflict adjustment
        conflict_weights = model.create_weight_matrix(
            k=3, 
            conflict_adjusted=True,
            conflict_col='conflict_intensity_normalized'
        )
        
        # Should still have same number of neighbors
        for neighbors in conflict_weights.neighbors.values():
            self.assertEqual(len(neighbors), 3)
        
        # Average weight should be lower with conflict adjustment
        avg_weight_regular = np.mean([np.mean(w) for w in weights.weights.values()])
        avg_weight_conflict = np.mean([np.mean(w) for w in conflict_weights.weights.values()])
        
        # Conflict adjustment should reduce weights
        self.assertLess(avg_weight_conflict, avg_weight_regular)
    
    def test_moran_i(self):
        """Test Moran's I calculation."""
        model = SpatialEconometrics(self.gdf)
        model.create_weight_matrix(k=3)
        
        # Test for price variable
        moran_result = model.moran_i_test('price')
        
        # Check result structure
        self.assertIn('I', moran_result)
        self.assertIn('p_norm', moran_result)
        self.assertIn('z_norm', moran_result)
        self.assertIn('significant', moran_result)
        
        # Moran's I should be between -1 and 1
        self.assertTrue(-1 <= moran_result['I'] <= 1)
        
        # Test with weight matrix not created
        model_no_weights = SpatialEconometrics(self.gdf)
        with self.assertRaises(ValueError):
            model_no_weights.moran_i_test('price')
        
        # Test with invalid column
        with self.assertRaises(ValueError):
            model.moran_i_test('nonexistent_column')
    
    def test_spatial_lag_model(self):
        """Test spatial lag model estimation."""
        model = SpatialEconometrics(self.gdf)
        model.create_weight_matrix(k=3)
        
        # Create dummy categorical variable for exchange_rate_regime
        self.gdf['north'] = (self.gdf['exchange_rate_regime'] == 'north').astype(int)
        
        # Estimate model
        result = model.spatial_lag_model(
            y_col='price',
            x_cols=['quantity', 'conflict_intensity_normalized', 'north']
        )
        
        # Check model attributes
        self.assertTrue(hasattr(result, 'rho'))  # Spatial autoregressive parameter
        self.assertTrue(hasattr(result, 'betas'))  # Coefficient estimates
        self.assertTrue(hasattr(result, 'std_err'))  # Standard errors
        
        # rho should be between -1 and 1
        self.assertTrue(-1 <= result.rho <= 1)
    
    def test_spatial_error_model(self):
        """Test spatial error model estimation."""
        model = SpatialEconometrics(self.gdf)
        model.create_weight_matrix(k=3)
        
        # Create dummy categorical variable for exchange_rate_regime
        self.gdf['north'] = (self.gdf['exchange_rate_regime'] == 'north').astype(int)
        
        # Estimate model
        result = model.spatial_error_model(
            y_col='price',
            x_cols=['quantity', 'conflict_intensity_normalized', 'north']
        )
        
        # Check model attributes
        self.assertTrue(hasattr(result, 'lam'))  # Spatial error parameter
        self.assertTrue(hasattr(result, 'betas'))  # Coefficient estimates
        self.assertTrue(hasattr(result, 'std_err'))  # Standard errors
        
        # lambda should be between -1 and 1
        self.assertTrue(-1 <= result.lam <= 1)
    
    def test_market_accessibility(self):
        """Test market accessibility calculation."""
        # Calculate accessibility
        result_gdf = calculate_market_accessibility(
            self.gdf,
            self.population_gdf,
            max_distance=1.0,  # Using degrees for test
            weight_col='population'
        )
        
        # Check result
        self.assertEqual(len(result_gdf), len(self.gdf))
        self.assertIn('accessibility_index', result_gdf.columns)
        
        # Accessibility should be non-negative
        self.assertTrue((result_gdf['accessibility_index'] >= 0).all())
    
    def test_market_isolation(self):
        """Test market isolation calculation."""
        # Calculate isolation
        result_gdf = calculate_market_isolation(
            self.gdf,
            conflict_col='conflict_intensity_normalized',
            max_distance=1.0  # Using degrees for test
        )
        
        # Check result
        self.assertEqual(len(result_gdf), len(self.gdf))
        self.assertIn('isolation_index', result_gdf.columns)
        
        # Isolation should be between 0 and 2
        # (base isolation 0-1, conflict multiplier ~1-2)
        self.assertTrue((result_gdf['isolation_index'] >= 0).all())
        self.assertTrue((result_gdf['isolation_index'] <= 2).all())


if __name__ == '__main__':
    unittest.main()