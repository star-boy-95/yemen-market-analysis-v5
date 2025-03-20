"""
Test script for the integrated analysis workflow.

This script tests the functionality of the integrated analysis modules:
- Spatiotemporal integration
- Interpretation
- Reporting
"""
import unittest
import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.spatiotemporal import integrate_time_series_spatial_results
from src.models.interpretation import (
    interpret_unit_root_results,
    interpret_cointegration_results,
    interpret_threshold_results,
    interpret_spatial_results,
    interpret_simulation_results
)
from src.models.reporting import (
    generate_comprehensive_report,
    create_executive_summary,
    export_results_for_publication
)


class TestIntegratedAnalysis(unittest.TestCase):
    """Test cases for the integrated analysis workflow."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary output directory
        self.output_dir = Path('tests/output')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create mock data for unit root results
        self.unit_root_results = {
            'north': {
                'adf': {'statistic': -2.5, 'p_value': 0.1, 'stationary': False},
                'kpss': {'statistic': 0.5, 'p_value': 0.05, 'stationary': False},
                'zivot_andrews': {'statistic': -4.5, 'p_value': 0.01, 'stationary': True, 'breakpoint': 50},
                'integration_order': 1
            },
            'south': {
                'adf': {'statistic': -2.2, 'p_value': 0.2, 'stationary': False},
                'kpss': {'statistic': 0.6, 'p_value': 0.03, 'stationary': False},
                'zivot_andrews': {'statistic': -4.2, 'p_value': 0.02, 'stationary': True, 'breakpoint': 60},
                'integration_order': 1
            },
            'merged_data': pd.DataFrame({
                'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
                'price_north': np.random.normal(100, 10, 100),
                'price_south': np.random.normal(120, 15, 100)
            })
        }
        
        # Create mock data for cointegration results
        self.cointegration_results = {
            'engle_granger': {
                'statistic': -3.5, 
                'critical_value': -3.0, 
                'p_value': 0.03, 
                'cointegrated': True,
                'beta': [10.0, 1.1]
            },
            'johansen': {
                'trace_stat': [25.0, 5.0],
                'trace_crit': [20.0, 9.0],
                'trace_pval': [0.01, 0.3],
                'max_stat': [20.0, 5.0],
                'max_crit': [18.0, 9.0],
                'max_pval': [0.02, 0.3],
                'rank_trace': 1,
                'rank_max': 1
            },
            'gregory_hansen': {
                'statistic': -4.5,
                'critical_value': -4.0,
                'p_value': 0.02,
                'cointegrated': True,
                'breakpoint': 55
            },
            'merged_data': self.unit_root_results['merged_data']
        }
        
        # Create mock data for threshold results
        self.threshold_results = {
            'cointegrated': True,
            'cointegration': {
                'cointegrated': True,
                'beta': [10.0, 1.1]
            },
            'threshold': {
                'threshold': 15.0,
                'search_range': [5.0, 25.0],
                'significant': True
            },
            'tvecm': {
                'threshold': 15.0,
                'adjustment_below_1': -0.05,
                'adjustment_above_1': -0.2,
                'adjustment_below_2': 0.02,
                'adjustment_above_2': 0.1
            },
            'mtar': {
                'asymmetric': True,
                'adjustment_positive': -0.1,
                'adjustment_negative': -0.3
            }
        }
        
        # Create mock data for spatial results
        self.spatial_results = {
            'global_moran': {
                'I': 0.3,
                'p': 0.02
            },
            'local_moran': None,  # Simplified for testing
            'lag_model': type('MockLagModel', (), {
                'rho': 0.4,
                'r2': 0.65,
                'name_x': ['usdprice', 'conflict_intensity_normalized', 'distance_to_port'],
                'betas': [0.8, -0.3, -0.2]
            }),
            'error_model': type('MockErrorModel', (), {
                'lambda_': 0.35,
                'r2': 0.62,
                'name_x': ['usdprice', 'conflict_intensity_normalized', 'distance_to_port'],
                'betas': [0.75, -0.25, -0.18]
            }),
            'spillover_effects': {
                'direct': {'usdprice': 0.85, 'conflict_intensity_normalized': -0.32, 'distance_to_port': -0.22},
                'indirect': {'usdprice': 0.35, 'conflict_intensity_normalized': -0.12, 'distance_to_port': -0.08},
                'total': {'usdprice': 1.2, 'conflict_intensity_normalized': -0.44, 'distance_to_port': -0.3}
            }
        }
        
        # Create mock data for simulation results
        self.simulation_results = {
            'exchange_unification': {
                'avg_price_north': 95.0,
                'avg_price_south': 125.0,
                'price_differential': 30.0,
                'price_volatility': 0.08,
                'integration_index': 0.6,
                'welfare_gain': 15.0
            },
            'conflict_reduction': {
                'avg_price_north': 98.0,
                'avg_price_south': 122.0,
                'price_differential': 24.0,
                'price_volatility': 0.07,
                'integration_index': 0.65,
                'welfare_gain': 18.0
            },
            'combined_policies': {
                'avg_price_north': 97.0,
                'avg_price_south': 118.0,
                'price_differential': 21.0,
                'price_volatility': 0.06,
                'integration_index': 0.7,
                'welfare_gain': 25.0
            },
            'welfare_effects': {
                'exchange_rate': {
                    'official': {
                        'beans (kidney red)': {
                            'total_welfare': 15.0,
                            'price_convergence': {'relative_convergence': 12.5},
                            'distributional': {
                                'gini_change': -0.02,
                                'bottom_quintile_impact': -8.0,
                                'food_security_improvement': 5.0
                            }
                        }
                    }
                },
                'connectivity': {
                    'reduction_50': {
                        'beans (kidney red)': {
                            'total_welfare': 18.0,
                            'price_convergence': {'relative_convergence': 15.0},
                            'distributional': {
                                'gini_change': -0.025,
                                'bottom_quintile_impact': -10.0,
                                'food_security_improvement': 6.0
                            }
                        }
                    }
                },
                'combined': {
                    'official_reduction_50': {
                        'beans (kidney red)': {
                            'total_welfare': 25.0,
                            'price_convergence': {'relative_convergence': 20.0},
                            'distributional': {
                                'gini_change': -0.03,
                                'bottom_quintile_impact': -15.0,
                                'food_security_improvement': 8.5
                            }
                        }
                    }
                }
            }
        }
        
        # Compile all results
        self.all_results = {
            'unit_root_results': self.unit_root_results,
            'cointegration_results': self.cointegration_results,
            'threshold_results': self.threshold_results,
            'spatial_results': self.spatial_results,
            'simulation_results': self.simulation_results
        }
        
        # Set up logger
        import logging
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            self.logger.addHandler(handler)

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up created files
        for file in self.output_dir.glob('*'):
            try:
                file.unlink()
            except:
                pass
        
        try:
            self.output_dir.rmdir()
        except:
            pass

    def test_spatiotemporal_integration(self):
        """Test spatiotemporal integration module."""
        time_series_results = {
            'unit_root': self.unit_root_results,
            'cointegration': self.cointegration_results,
            'tvecm': self.threshold_results['tvecm']
        }
        
        integrated_results = integrate_time_series_spatial_results(
            time_series_results=time_series_results,
            spatial_results=self.spatial_results,
            commodity='beans (kidney red)'
        )
        
        # Check that the integration was successful
        self.assertIsNotNone(integrated_results)
        self.assertIn('integration_index', integrated_results)
        self.assertIn('spatial_time_correlation', integrated_results)
        self.assertIn('regime_boundary_effect', integrated_results)

    def test_interpretation_modules(self):
        """Test interpretation modules."""
        # Test unit root interpretation
        unit_root_interp = interpret_unit_root_results(
            self.unit_root_results, 
            'beans (kidney red)'
        )
        self.assertIn('summary', unit_root_interp)
        self.assertIn('details', unit_root_interp)
        self.assertIn('implications', unit_root_interp)
        
        # Test cointegration interpretation
        coint_interp = interpret_cointegration_results(
            self.cointegration_results, 
            'beans (kidney red)'
        )
        self.assertIn('summary', coint_interp)
        self.assertIn('details', coint_interp)
        self.assertIn('implications', coint_interp)
        
        # Test threshold interpretation
        threshold_interp = interpret_threshold_results(
            self.threshold_results, 
            'beans (kidney red)'
        )
        self.assertIn('summary', threshold_interp)
        self.assertIn('details', threshold_interp)
        self.assertIn('implications', threshold_interp)
        
        # Test spatial interpretation
        spatial_interp = interpret_spatial_results(
            self.spatial_results, 
            'beans (kidney red)'
        )
        self.assertIn('summary', spatial_interp)
        self.assertIn('details', spatial_interp)
        self.assertIn('implications', spatial_interp)
        
        # Test simulation interpretation
        sim_interp = interpret_simulation_results(
            self.simulation_results, 
            'beans (kidney red)'
        )
        self.assertIn('summary', sim_interp)
        self.assertIn('policy_recommendations', sim_interp)
        self.assertIn('welfare_effects', sim_interp)
        self.assertIn('implementation_considerations', sim_interp)

    def test_reporting_modules(self):
        """Test reporting modules."""
        # Test comprehensive report generation
        report_path = generate_comprehensive_report(
            all_results=self.all_results,
            commodity='beans (kidney red)',
            output_path=self.output_dir,
            logger=self.logger
        )
        self.assertTrue(report_path.exists())
        
        # Test executive summary creation
        summary_path = create_executive_summary(
            all_results=self.all_results,
            commodity='beans (kidney red)',
            output_path=self.output_dir,
            logger=self.logger
        )
        self.assertTrue(summary_path.exists())
        
        # Test publication export
        publication_path = export_results_for_publication(
            all_results=self.all_results,
            commodity='beans (kidney red)',
            output_path=self.output_dir,
            logger=self.logger
        )
        self.assertTrue(publication_path.exists())


if __name__ == '__main__':
    unittest.main()