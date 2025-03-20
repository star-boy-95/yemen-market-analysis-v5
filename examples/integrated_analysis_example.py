#!/usr/bin/env python
"""
Example script demonstrating how to use the integrated analysis modules.

This script shows how to:
1. Run the full integrated analysis
2. Generate reports and visualizations
3. Interpret results

Usage:
    python examples/integrated_analysis_example.py
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester
from src.models.threshold import ThresholdCointegration
from src.models.spatial import SpatialEconometrics
from src.models.simulation import MarketIntegrationSimulation

# Import new integration modules
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


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def main():
    """Run the example analysis."""
    logger = setup_logging()
    logger.info("Starting integrated analysis example")
    
    # Create output directory
    output_path = Path('examples/output')
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output will be saved to: {output_path}")
    
    # Define parameters
    data_file = 'data/raw/unified_data.geojson'
    commodity = 'beans (kidney red)'
    max_lags = 4
    k_neighbors = 5
    conflict_weight = 1.0
    
    # Load and preprocess data
    logger.info(f"Loading data from: {data_file}")
    try:
        loader = DataLoader("./data")
        gdf = loader.load_geojson(os.path.basename(data_file))
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        processed_gdf = preprocessor.preprocess_geojson(gdf)
        
        # Step 1: Unit Root Analysis
        logger.info("Step 1: Running unit root analysis")
        unit_root_tester = UnitRootTester()
        
        # Get data for north and south
        north_data = processed_gdf[
            (processed_gdf['commodity'] == commodity) & 
            (processed_gdf['exchange_rate_regime'] == 'north')
        ]
        south_data = processed_gdf[
            (processed_gdf['commodity'] == commodity) & 
            (processed_gdf['exchange_rate_regime'] == 'south')
        ]
        
        # Interpret unit root results
        logger.info("Interpreting unit root results")
        unit_root_results = {
            'north': {
                'adf': unit_root_tester.test_adf(north_data['price']),
                'kpss': unit_root_tester.test_kpss(north_data['price']),
                'zivot_andrews': unit_root_tester.test_zivot_andrews(north_data['price']),
                'integration_order': unit_root_tester.determine_integration_order(north_data['price'])
            },
            'south': {
                'adf': unit_root_tester.test_adf(south_data['price']),
                'kpss': unit_root_tester.test_kpss(south_data['price']),
                'zivot_andrews': unit_root_tester.test_zivot_andrews(south_data['price']),
                'integration_order': unit_root_tester.determine_integration_order(south_data['price'])
            }
        }
        
        interpretation = interpret_unit_root_results(unit_root_results, commodity)
        logger.info(f"Unit Root Interpretation: {interpretation['summary']}")
        
        # Step 2: Cointegration Analysis
        logger.info("Step 2: Running cointegration analysis")
        cointegration_tester = CointegrationTester()
        
        # Run Engle-Granger test
        eg_result = cointegration_tester.test_engle_granger(
            north_data['price'], south_data['price']
        )
        
        # Interpret cointegration results
        cointegration_results = {
            'engle_granger': eg_result
        }
        
        interpretation = interpret_cointegration_results(cointegration_results, commodity)
        logger.info(f"Cointegration Interpretation: {interpretation['summary']}")
        
        # Step 3: Threshold Analysis
        logger.info("Step 3: Running threshold analysis")
        threshold_model = ThresholdCointegration(
            north_data['price'], south_data['price'],
            max_lags=max_lags,
            market1_name="North",
            market2_name="South"
        )
        
        # Estimate threshold
        threshold_result = threshold_model.estimate_threshold()
        
        # Interpret threshold results
        threshold_results = {
            'cointegrated': True,
            'threshold': threshold_result,
            'tvecm': threshold_model.estimate_tvecm()
        }
        
        interpretation = interpret_threshold_results(threshold_results, commodity)
        logger.info(f"Threshold Interpretation: {interpretation['summary']}")
        
        # Step 4: Spatial Analysis
        logger.info("Step 4: Running spatial analysis")
        latest_date = processed_gdf['date'].max()
        latest_data = processed_gdf[
            (processed_gdf['commodity'] == commodity) & 
            (processed_gdf['date'] == latest_date)
        ]
        
        spatial_model = SpatialEconometrics(latest_data)
        
        # Create spatial weights matrix
        spatial_model.create_weight_matrix(
            k=k_neighbors,
            conflict_adjusted=True,
            conflict_col='conflict_intensity_normalized',
            conflict_weight=conflict_weight
        )
        
        # Run global spatial autocorrelation test
        global_moran = spatial_model.moran_i_test(variable='price')
        
        # Interpret spatial results
        spatial_results = {
            'global_moran': global_moran,
            'lag_model': spatial_model.spatial_lag_model(
                y_col='price',
                x_cols=['usdprice', 'conflict_intensity_normalized']
            )
        }
        
        interpretation = interpret_spatial_results(spatial_results, commodity)
        logger.info(f"Spatial Interpretation: {interpretation['summary']}")
        
        # Step 5: Simulation Analysis
        logger.info("Step 5: Running simulation analysis")
        simulation_model = MarketIntegrationSimulation(
            data=processed_gdf[processed_gdf['commodity'] == commodity]
        )
        
        # Run exchange rate unification simulation
        exchange_unification = simulation_model.simulate_exchange_rate_unification()
        
        # Run conflict reduction simulation
        conflict_reduction = simulation_model.simulate_improved_connectivity(reduction_factor=0.5)
        
        # Interpret simulation results
        simulation_results = {
            'exchange_unification': exchange_unification,
            'conflict_reduction': conflict_reduction,
            'combined_policies': simulation_model.simulate_combined_policy()
        }
        
        interpretation = interpret_simulation_results(simulation_results, commodity)
        logger.info(f"Simulation Interpretation: {interpretation['summary']}")
        
        # Step 6: Integrate Time Series and Spatial Results
        logger.info("Step 6: Integrating time series and spatial results")
        time_series_results = {
            'unit_root': unit_root_results,
            'cointegration': cointegration_results,
            'tvecm': threshold_results['tvecm']
        }
        
        integrated_results = integrate_time_series_spatial_results(
            time_series_results=time_series_results,
            spatial_results=spatial_results,
            commodity=commodity
        )
        
        logger.info(f"Integration Index: {integrated_results.get('integration_index', 'N/A')}")
        
        # Step 7: Generate Reports
        logger.info("Step 7: Generating reports")
        
        # Compile all results
        all_results = {
            'unit_root_results': unit_root_results,
            'cointegration_results': cointegration_results,
            'threshold_results': threshold_results,
            'spatial_results': spatial_results,
            'simulation_results': simulation_results,
            'integrated_results': integrated_results
        }
        
        # Generate comprehensive report
        report_path = generate_comprehensive_report(
            all_results=all_results,
            commodity=commodity,
            output_path=output_path,
            logger=logger
        )
        
        # Create executive summary
        summary_path = create_executive_summary(
            all_results=all_results,
            commodity=commodity,
            output_path=output_path,
            logger=logger
        )
        
        # Export results for publication
        publication_path = export_results_for_publication(
            all_results=all_results,
            commodity=commodity,
            output_path=output_path,
            logger=logger
        )
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        logger.info(f"Executive summary saved to: {summary_path}")
        logger.info(f"Publication-ready results saved to: {publication_path}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        logger.exception("Detailed traceback:")
        return 1
    
    logger.info("Integrated analysis example completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())