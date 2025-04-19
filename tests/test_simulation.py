"""
Tests for the market integration simulation module.
"""
import unittest
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from unittest.mock import patch, MagicMock

from src.models.simulation import MarketIntegrationSimulation
from src.utils.error_handling import YemenAnalysisError

class TestMarketIntegrationSimulation(unittest.TestCase):
    """Test cases for the MarketIntegrationSimulation class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.data = pd.DataFrame({
            'date': dates,
            'price': np.random.normal(100, 10, 100)
        })
        self.data.set_index('date', inplace=True)
        
        # Create sample exchange rate data
        self.exchange_rate_data = pd.DataFrame({
            'date': dates,
            'sanaa_rate': [600 + i * 0.5 for i in range(100)],
            'aden_rate': [500 + i * 0.3 for i in range(100)]
        })
        self.exchange_rate_data.set_index('date', inplace=True)
        
        # Create sample spatial data
        geometry = [Point(x, y) for x, y in zip(np.random.rand(10), np.random.rand(10))]
        self.spatial_data = gpd.GeoDataFrame({
            'market': [f'Market_{i}' for i in range(10)],
            'price': np.random.normal(100, 10, 10),
            'distance': np.random.rand(10) * 100,
            'conflict': np.random.rand(10),
            'geometry': geometry
        })
        
        # Create the simulation object
        self.simulator = MarketIntegrationSimulation(
            data=self.data,
            exchange_rate_data=self.exchange_rate_data
        )

    def test_initialization(self):
        """Test initialization of the MarketIntegrationSimulation class."""
        self.assertIsNotNone(self.simulator)
        self.assertEqual(self.simulator.data.equals(self.data), True)
        self.assertEqual(self.simulator.exchange_rate_data.equals(self.exchange_rate_data), True)
        self.assertIsNone(self.simulator.threshold_model)
        self.assertIsNone(self.simulator.spatial_model)
        self.assertEqual(self.simulator.results, {})

    def test_set_data(self):
        """Test setting data for the simulation."""
        new_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=50, freq='D'),
            'price': np.random.normal(100, 10, 50)
        })
        new_data.set_index('date', inplace=True)
        
        self.simulator.set_data(new_data)
        self.assertEqual(self.simulator.data.equals(new_data), True)

    def test_set_exchange_rate_data(self):
        """Test setting exchange rate data for the simulation."""
        new_exchange_rate_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=50, freq='D'),
            'sanaa_rate': [600 + i * 0.5 for i in range(50)],
            'aden_rate': [500 + i * 0.3 for i in range(50)]
        })
        new_exchange_rate_data.set_index('date', inplace=True)
        
        self.simulator.set_exchange_rate_data(new_exchange_rate_data)
        self.assertEqual(self.simulator.exchange_rate_data.equals(new_exchange_rate_data), True)

    @patch('src.models.threshold.tvecm.ThresholdVECM')
    def test_simulate_exchange_rate_unification(self, mock_tvecm):
        """Test simulating exchange rate unification."""
        # Mock the threshold model
        mock_tvecm_instance = MagicMock()
        mock_tvecm_instance.estimate.return_value = {
            'threshold': 0.05,
            'adjustment_speed': 0.2,
            'rho1': -0.1,
            'rho2': -0.3
        }
        mock_tvecm.return_value = mock_tvecm_instance
        
        # Run the simulation
        results = self.simulator.simulate_exchange_rate_unification(
            target_rate='official',
            method='tvecm',
            original_threshold=0.1,
            original_adjustment_speed=0.1,
            y_col='price',
            x_col='price'
        )
        
        # Check the results
        self.assertIn('target_rate', results)
        self.assertEqual(results['target_rate'], 'official')
        self.assertIn('method', results)
        self.assertEqual(results['method'], 'tvecm')
        self.assertIn('threshold_model_results', results)
        self.assertIn('impact', results)
        
        # Check that the exchange rates are unified
        self.assertTrue((results['simulated_exchange_rate_data']['sanaa_rate'] == 
                         results['simulated_exchange_rate_data']['aden_rate']).all())

    @patch('src.models.spatial.lag_model.SpatialLagModel')
    @patch('src.models.spatial.weights.SpatialWeightMatrix')
    def test_simulate_spatial_connectivity(self, mock_weight_matrix, mock_lag_model):
        """Test simulating spatial connectivity."""
        # Mock the spatial weight matrix
        mock_weight_matrix_instance = MagicMock()
        mock_weight_matrix_instance.create_distance_weights.return_value = MagicMock()
        mock_weight_matrix_instance.create_contiguity_weights.return_value = MagicMock()
        mock_weight_matrix_instance.create_kernel_weights.return_value = MagicMock()
        mock_weight_matrix.return_value = mock_weight_matrix_instance
        
        # Mock the spatial lag model
        mock_lag_model_instance = MagicMock()
        mock_lag_model_instance.estimate.return_value = {
            'rho': 0.5,
            'r_squared': 0.7,
            'price_dispersion': 0.2
        }
        mock_lag_model.return_value = mock_lag_model_instance
        
        # Run the simulation
        results = self.simulator.simulate_spatial_connectivity(
            data=self.spatial_data,
            connectivity_improvement=0.5,
            weight_type='distance',
            original_results={
                'rho': 0.3,
                'r_squared': 0.5,
                'price_dispersion': 0.4
            }
        )
        
        # Check the results
        self.assertIn('connectivity_improvement', results)
        self.assertEqual(results['connectivity_improvement'], 0.5)
        self.assertIn('weight_type', results)
        self.assertEqual(results['weight_type'], 'distance')
        self.assertIn('spatial_model_results', results)
        self.assertIn('impact', results)

    def test_run_full_simulation(self):
        """Test running a full simulation."""
        # Mock the exchange rate unification and spatial connectivity methods
        self.simulator.simulate_exchange_rate_unification = MagicMock(return_value={'test': 'exchange_rate'})
        self.simulator.simulate_spatial_connectivity = MagicMock(return_value={'test': 'spatial'})
        
        # Run the full simulation
        results = self.simulator.run_full_simulation(
            data=self.data,
            exchange_rate_data=self.exchange_rate_data,
            spatial_data=self.spatial_data
        )
        
        # Check the results
        self.assertIn('exchange_rate_unification', results)
        self.assertEqual(results['exchange_rate_unification'], {'test': 'exchange_rate'})
        self.assertIn('spatial_connectivity', results)
        self.assertEqual(results['spatial_connectivity'], {'test': 'spatial'})
        
        # Check that the methods were called
        self.simulator.simulate_exchange_rate_unification.assert_called_once()
        self.simulator.simulate_spatial_connectivity.assert_called_once()

if __name__ == '__main__':
    unittest.main()
