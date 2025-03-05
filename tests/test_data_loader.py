"""
Unit tests for data loader module.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader
from src.utils import DataError


class TestDataLoader(unittest.TestCase):
    """Tests for the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader('./data')
        
        # Create a temporary test GeoDataFrame
        self.test_data = gpd.GeoDataFrame(
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

    @patch('src.utils.read_geojson')
    @patch('src.utils.validate_geodataframe')
    def test_load_geojson(self, mock_validate, mock_read_geojson):
        """Test loading GeoJSON file."""
        # Configure mocks
        mock_read_geojson.return_value = self.test_data
        mock_validate.return_value = (True, [])
        
        # Call the method
        result = self.loader.load_geojson('test.geojson')
        
        # Verify mocks were called correctly
        mock_read_geojson.assert_called_once()
        mock_validate.assert_called_once()
        
        # Verify result is correct
        self.assertEqual(len(result), 4)
        self.assertListEqual(list(result['admin1']), ['abyan', 'abyan', 'aden', 'aden'])
    
    @patch('src.utils.read_geojson')
    @patch('src.utils.validate_geodataframe')
    @patch('pathlib.Path.exists')
    def test_load_geojson_file_not_found(self, mock_exists, mock_validate, mock_read_geojson):
        """Test loading non-existent GeoJSON file."""
        # Configure mocks
        mock_exists.return_value = False
        
        # Verify that the correct exception is raised
        with self.assertRaises(DataError):
            self.loader.load_geojson('nonexistent.geojson')
        
        # Verify mocks were called correctly
        mock_read_geojson.assert_not_called()
        mock_validate.assert_not_called()
    
    @patch('src.utils.read_geojson')
    @patch('src.utils.validate_geodataframe')
    @patch('pathlib.Path.exists')
    def test_load_geojson_invalid_data(self, mock_exists, mock_validate, mock_read_geojson):
        """Test loading invalid GeoJSON file."""
        # Configure mocks
        mock_exists.return_value = True
        mock_read_geojson.return_value = self.test_data
        mock_validate.return_value = (False, ["Missing required column 'admin1'"])
        
        # Verify that the correct exception is raised
        with self.assertRaises(ValueError):
            self.loader.load_geojson('invalid.geojson')
    
    @patch('src.utils.write_geojson')
    def test_save_processed_data(self, mock_write_geojson):
        """Test saving processed data."""
        # Call the method
        self.loader.save_processed_data(self.test_data, 'processed.geojson')
        
        # Verify mock was called correctly
        mock_write_geojson.assert_called_once()
    
    def test_save_processed_data_invalid_input(self):
        """Test saving processed data with invalid input."""
        # Create an invalid input (not a GeoDataFrame)
        invalid_data = pd.DataFrame({
            'admin1': ['abyan', 'aden'],
            'commodity': ['beans', 'rice']
        })
        
        # Verify that the correct exception is raised
        with self.assertRaises(DataError):
            self.loader.save_processed_data(invalid_data, 'processed.geojson')
    
    def test_split_by_exchange_regime(self):
        """Test splitting by exchange rate regime."""
        north, south = self.loader.split_by_exchange_regime(self.test_data)
        
        self.assertEqual(len(north), 2)
        self.assertEqual(len(south), 2)
        self.assertTrue(all(north['exchange_rate_regime'] == 'north'))
        self.assertTrue(all(south['exchange_rate_regime'] == 'south'))
    
    def test_split_by_exchange_regime_missing_column(self):
        """Test splitting by exchange rate regime with missing column."""
        # Create data without exchange_rate_regime column
        invalid_data = self.test_data.drop(columns=['exchange_rate_regime'])
        
        # Verify that the correct exception is raised
        with self.assertRaises(ValueError):
            self.loader.split_by_exchange_regime(invalid_data)
    
    def test_get_time_series(self):
        """Test getting time series for a specific region and commodity."""
        ts = self.loader.get_time_series(self.test_data, 'abyan', 'beans')
        
        self.assertEqual(len(ts), 2)
        self.assertTrue(all(ts['admin1'] == 'abyan'))
        self.assertTrue(all(ts['commodity'] == 'beans'))
        self.assertTrue(ts['date'].is_monotonic_increasing)
    
    def test_get_time_series_missing_columns(self):
        """Test getting time series with missing columns."""
        # Create data without required columns
        invalid_data = self.test_data.drop(columns=['admin1'])
        
        # Verify that the correct exception is raised
        with self.assertRaises(ValueError):
            self.loader.get_time_series(invalid_data, 'abyan', 'beans')
    
    def test_get_time_series_no_data(self):
        """Test getting time series with no matching data."""
        ts = self.loader.get_time_series(self.test_data, 'nonexistent', 'beans')
        
        self.assertEqual(len(ts), 0)
    
    def test_get_commodity_list(self):
        """Test getting list of commodities."""
        commodities = self.loader.get_commodity_list(self.test_data)
        
        self.assertEqual(len(commodities), 2)
        self.assertListEqual(commodities, ['beans', 'rice'])
    
    def test_get_commodity_list_missing_column(self):
        """Test getting commodity list with missing column."""
        # Create data without commodity column
        invalid_data = self.test_data.drop(columns=['commodity'])
        
        # Verify that the correct exception is raised
        with self.assertRaises(ValueError):
            self.loader.get_commodity_list(invalid_data)
    
    def test_get_region_list(self):
        """Test getting list of regions."""
        regions = self.loader.get_region_list(self.test_data)
        
        self.assertEqual(len(regions), 2)
        self.assertListEqual(regions, ['abyan', 'aden'])
    
    def test_get_region_list_missing_column(self):
        """Test getting region list with missing column."""
        # Create data without admin1 column
        invalid_data = self.test_data.drop(columns=['admin1'])
        
        # Verify that the correct exception is raised
        with self.assertRaises(ValueError):
            self.loader.get_region_list(invalid_data)


if __name__ == '__main__':
    unittest.main()