"""Unit tests for data preprocessor module."""
import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessor import DataPreprocessor
from src.utils import ValidationError


class TestDataPreprocessor(unittest.TestCase):
    """Tests for the DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create a temporary test GeoDataFrame
        self.test_data = gpd.GeoDataFrame(
            {
                'admin1': ['abyan', 'abyan', 'aden', 'aden'],
                'commodity': ['beans', 'beans', 'beans', 'rice'],
                'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-01-01', '2020-02-01']),
                'price': [100, 110, 120, 130],
                'usdprice': [1.0, 1.1, 1.2, 1.3],
                'exchange_rate_regime': ['north', 'north', 'south', 'south'],
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
        
        # Create test data with missing values
        self.test_data_with_nulls = self.test_data.copy()
        self.test_data_with_nulls.loc[1, 'price'] = np.nan
        self.test_data_with_nulls.loc[2, 'events'] = np.nan

    def test_preprocess_geojson(self):
        """Test preprocessing GeoJSON data."""
        processed = self.preprocessor.preprocess_geojson(self.test_data)
        
        # Check that new columns were created
        self.assertIn('year', processed.columns)
        self.assertIn('month', processed.columns)
        self.assertIn('price_log', processed.columns)
        
        # Check values
        self.assertEqual(processed['year'].iloc[0], 2020)
        self.assertEqual(processed['month'].iloc[0], 1)
        self.assertAlmostEqual(processed['price_log'].iloc[0], np.log(100), places=5)
    
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        processed = self.preprocessor._handle_missing_values(self.test_data_with_nulls)
        
        # Check that nulls were handled
        self.assertFalse(processed['price'].isna().any())
        self.assertFalse(processed['events'].isna().any())
    
    def test_calculate_price_differentials(self):
        """Test calculation of price differentials."""
        differentials = self.preprocessor.calculate_price_differentials(self.test_data)
        
        # Check that the DataFrame has the expected columns
        expected_columns = ['commodity', 'date', 'north_price', 'south_price', 'price_diff', 'price_diff_pct']
        for col in expected_columns:
            self.assertIn(col, differentials.columns)
        
        # Check calculations
        beans_diff = differentials[differentials['commodity'] == 'beans']
        self.assertEqual(len(beans_diff), 2)  # 2 dates
        
        # Verify price difference calculation
        first_row = beans_diff.iloc[0]
        self.assertEqual(first_row['price_diff'], first_row['north_price'] - first_row['south_price'])


if __name__ == '__main__':
    unittest.main()