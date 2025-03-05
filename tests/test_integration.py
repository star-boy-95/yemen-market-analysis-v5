"""Unit tests for data integration module."""
import unittest
from unittest.mock import patch
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.integration import DataIntegrator


class TestDataIntegrator(unittest.TestCase):
    """Tests for the DataIntegrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integrator = DataIntegrator('./data')
        
        # Create test market data
        self.market_data = gpd.GeoDataFrame(
            {
                'admin1': ['abyan', 'abyan', 'aden', 'aden'],
                'commodity': ['beans', 'beans', 'beans', 'rice'],
                'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-01-01', '2020-02-01']),
                'price': [100, 110, 120, 130],
                'usdprice': [1.0, 1.1, 1.2, 1.3],
                'exchange_rate_regime': ['north', 'north', 'south', 'south'],
                'geometry': [
                    Point(45.0, 13.0),
                    Point(45.0, 13.0),
                    Point(46.0, 12.0),
                    Point(46.0, 12.0)
                ]
            },
            crs="EPSG:4326"
        )
        
        # Create conflict data
        self.conflict_data = gpd.GeoDataFrame(
            {
                'admin1': ['abyan', 'abyan', 'aden', 'aden'],
                'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-01-01', '2020-02-01']),
                'events': [10, 15, 5, 8],
                'fatalities': [5, 8, 2, 3],
                'geometry': [
                    Point(45.0, 13.0),
                    Point(45.0, 13.0),
                    Point(46.0, 12.0),
                    Point(46.0, 12.0)
                ]
            },
            crs="EPSG:4326"
        )

    @patch('src.utils.read_geojson')
    @patch('src.utils.validate_geodataframe')
    def test_integrate_conflict_data(self, mock_validate, mock_read_geojson):
        """Test integrating conflict data."""
        # Configure mocks
        mock_read_geojson.return_value = self.conflict_data
        mock_validate.return_value = (True, [])
        
        # Call the method
        result = self.integrator.integrate_conflict_data(self.market_data, 'conflict.geojson')
        
        # Verify result has conflict columns
        self.assertIn('events', result.columns)
        self.assertIn('fatalities', result.columns)
    
    @patch('src.utils.read_geojson')
    @patch('src.utils.validate_geodataframe')
    def test_get_spatial_boundaries(self, mock_validate, mock_read_geojson):
        """Test loading administrative boundaries."""
        # Create test boundaries data
        boundaries_data = gpd.GeoDataFrame(
            {
                'admin1': ['abyan', 'aden'],
                'geometry': [
                    Point(45.0, 13.0).buffer(1),
                    Point(46.0, 12.0).buffer(1)
                ]
            },
            crs="EPSG:4326"
        )
        
        # Configure mocks
        mock_read_geojson.return_value = boundaries_data
        mock_validate.return_value = (True, [])
        
        # Call the method
        result = self.integrator.get_spatial_boundaries('boundaries.geojson')
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertListEqual(list(result['admin1']), ['abyan', 'aden'])


if __name__ == '__main__':
    unittest.main()