"""
Main module for Yemen Market Analysis.

This module provides the main entry point for running the Yemen Market Analysis
package. It orchestrates the data loading, preprocessing, analysis, and reporting
steps.
"""
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

# Suppress urllib3 NotOpenSSLWarning
warnings.filterwarnings("ignore", category=Warning, message="urllib3 v2 only supports OpenSSL 1.1.1+")

from src.config import config
from src.data.loader import DataLoader
from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester
from src.models.threshold import ThresholdModel
from src.models.spatial import SpatialTester
from src.models.simulation import MarketIntegrationSimulation
from src.models.reporting import ResultsReporter
from src.visualization.time_series import TimeSeriesPlotter as TimeSeriesVisualizer
from src.visualization.maps import MapPlotter as SpatialVisualizer
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.performance import MemoryManager, ParallelProcessor

# Initialize logger
logger = logging.getLogger(__name__)

# Helper functions for parallel processing - needs to be at module level for pickling
def _process_market_unit_root(args):
    market_name, market_data, unit_root_tester = args
    logger.info(f"Running unit root tests for {market_name}")
    return market_name, unit_root_tester.run_all_tests(market_data)

def _process_cointegration_pair(args):
    market1_name, market2_name, market1_data, market2_data, cointegration_tester = args
    pair_name = f"{market1_name}_{market2_name}"
    logger.info(f"Running cointegration tests for {pair_name}")
    return pair_name, cointegration_tester.run_all_tests(market1_data, market2_data)

def _process_threshold_pair_mode(args):
    market1, market2, mode, data1, data2 = args
    pair_name = f"{market1}_{market2}_{mode}"
    logger.info(f"Running threshold model ({mode}) for {market1} and {market2}")

    # Initialize threshold model for this pair
    model = ThresholdModel(data1, data2, mode=mode)
    return pair_name, model.run()

class YemenMarketAnalysis:
    """
    Main class for Yemen Market Analysis.

    This class orchestrates the data loading, preprocessing, analysis, and reporting
    steps for the Yemen Market Analysis package.

    Attributes:
        data_loader (DataLoader): Data loader instance.
        unit_root_tester (UnitRootTester): Unit root tester instance.
        cointegration_tester (CointegrationTester): Cointegration tester instance.
        threshold_model (ThresholdModel): Threshold model instance.
        spatial_analyzer (SpatialAnalyzer): Spatial analyzer instance.
        market_integration_simulator (MarketIntegrationSimulation): Market integration simulator instance.
        results_reporter (ResultsReporter): Results reporter instance.
        time_series_visualizer (TimeSeriesVisualizer): Time series visualizer instance.
        spatial_visualizer (SpatialVisualizer): Spatial visualizer instance.
        memory_manager (MemoryManager): Memory manager instance.
        parallel_processor (ParallelProcessor): Parallel processor instance.
    """

    def __init__(self):
        """Initialize the Yemen Market Analysis."""
        self.data_loader = DataLoader()
        self.unit_root_tester = UnitRootTester()
        self.cointegration_tester = CointegrationTester()
        self.threshold_model = None  # Will be initialized with data
        self.spatial_analyzer = SpatialTester()
        self.market_integration_simulator = None  # Will be initialized with data
        self.results_reporter = ResultsReporter()
        self.time_series_visualizer = TimeSeriesVisualizer()
        self.spatial_visualizer = SpatialVisualizer()
        self.memory_manager = MemoryManager()
        self.parallel_processor = ParallelProcessor()

        # Initialize logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('yemen_analysis.log')
            ]
        )

    @handle_errors
    def run_analysis(
        self,
        data_path: Optional[Union[str, Path]] = None,
        commodity: Optional[str] = None,
        markets: Optional[List[str]] = None,
        threshold_modes: Optional[List[str]] = None,
        include_spatial: bool = True,
        include_simulation: bool = True,
        publication_quality: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete analysis for Yemen market integration.

        Args:
            data_path: Path to the data file. If None, uses the default path from config.
            commodity: Commodity to analyze. If None, analyzes all commodities.
            markets: List of markets to analyze. If None, analyzes all markets.
            threshold_modes: List of threshold modes to use. If None, uses all modes.
            include_spatial: Whether to include spatial analysis.
            include_simulation: Whether to include market integration simulation.
            publication_quality: Whether to generate publication-quality outputs.
            output_dir: Directory to save outputs. If None, uses the current directory.

        Returns:
            Dictionary containing the analysis results.
        """
        start_time = time.time()
        logger.info("Starting Yemen Market Analysis")

        # Set default values
        if data_path is None:
            data_path = config.get('data.raw_path')
        if threshold_modes is None:
            threshold_modes = ['standard', 'fixed', 'mtar']
        if output_dir is None:
            output_dir = Path('results')
        else:
            output_dir = Path(output_dir)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        logger.info(f"Loading data from {data_path}")
        data = self.data_loader.load_geojson(data_path)

        # Filter data if needed
        if commodity is not None:
            data = self.data_loader.filter_by_commodity(data, commodity)
        if markets is not None:
            data = self.data_loader.filter_by_markets(data, markets)

        # Preprocess data
        logger.info("Preprocessing data")
        preprocessed_data = self.data_loader.preprocess_data(data)

        # Run unit root tests
        logger.info("Running unit root tests")
        unit_root_results = self._run_unit_root_tests(preprocessed_data)

        # Run cointegration tests
        logger.info("Running cointegration tests")
        cointegration_results = self._run_cointegration_tests(preprocessed_data)

        # Run threshold models
        logger.info("Running threshold models")
        threshold_results = self._run_threshold_models(
            preprocessed_data, threshold_modes
        )

        # Run spatial analysis if requested
        spatial_results = None
        if include_spatial:
            logger.info("Running spatial analysis")
            spatial_results = self._run_spatial_analysis(preprocessed_data)

        # Run market integration simulation if requested
        simulation_results = None
        if include_simulation:
            logger.info("Running market integration simulation")
            simulation_results = self._run_market_integration_simulation(
                preprocessed_data, threshold_results, spatial_results
            )

        # Generate visualizations
        logger.info("Generating visualizations")
        visualization_results = self._generate_visualizations(
            preprocessed_data,
            unit_root_results,
            cointegration_results,
            threshold_results,
            spatial_results,
            publication_quality,
            output_dir
        )

        # Generate report
        logger.info("Generating report")
        # Create a results dictionary for the report
        results_dict = {
            'data': preprocessed_data,
            'unit_root': unit_root_results,
            'cointegration': cointegration_results,
            'threshold': threshold_results,
            'spatial': spatial_results,
            'simulation': simulation_results,
            'visualizations': visualization_results
        }

        # Generate the report
        report_path = output_dir / 'report.md'
        report = self.results_reporter.generate_report(
            results_dict,
            output_path=str(report_path),
            publication_quality=publication_quality,
            output_dir=str(output_dir)
        )

        # Compile results
        results = {
            'data': preprocessed_data,
            'unit_root': unit_root_results,
            'cointegration': cointegration_results,
            'threshold': threshold_results,
            'spatial': spatial_results,
            'simulation': simulation_results,
            'visualizations': visualization_results,
            'report': report,
        }

        end_time = time.time()
        logger.info(f"Yemen Market Analysis completed in {end_time - start_time:.2f} seconds")

        return results

    @handle_errors
    def _run_unit_root_tests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Run unit root tests on the data.

        Args:
            data: Dictionary mapping market names to DataFrames.

        Returns:
            Dictionary containing the unit root test results.
        """
        results = {}

        # Run tests in parallel
        market_items = list(data.items())
        processed_results = self.parallel_processor.process(
            _process_market_unit_root,
            [(name, df, self.unit_root_tester) for name, df in market_items]
        )

        # Collect results
        for market_name, market_results in processed_results:
            results[market_name] = market_results

        return results

    def _run_cointegration_tests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Run cointegration tests on the data.

        Args:
            data: Dictionary mapping market names to DataFrames.

        Returns:
            Dictionary containing the cointegration test results.
        """
        results = {}

        # Generate all pairs of markets
        market_names = list(data.keys())
        market_pairs = []

        for i in range(len(market_names)):
            for j in range(i + 1, len(market_names)):
                market_pairs.append((market_names[i], market_names[j]))

        # Run tests in parallel using the module-level helper function
        processed_results = self.parallel_processor.process(
            _process_cointegration_pair,
            [(m1, m2, data[m1], data[m2], self.cointegration_tester) for m1, m2 in market_pairs]
        )

        # Collect results
        for pair_name, pair_results in processed_results:
            results[pair_name] = pair_results

        return results

    @handle_errors
    def _run_threshold_models(
        self,
        data: Dict[str, pd.DataFrame],
        threshold_modes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run threshold models on the data.

        Args:
            data: Dictionary mapping market names to DataFrames.
            threshold_modes: List of threshold modes to use.

        Returns:
            Dictionary containing the threshold model results.
        """
        results = {}

        # Generate all pairs of markets
        market_names = list(data.keys())
        market_pairs = []

        for i in range(len(market_names)):
            for j in range(i + 1, len(market_names)):
                market_pairs.append((market_names[i], market_names[j]))

        # Generate all combinations of pairs and modes
        tasks = []
        for m1, m2 in market_pairs:
            for mode in threshold_modes:
                tasks.append((m1, m2, mode, data[m1], data[m2]))

        # Run models in parallel using the module-level helper function
        try:
            processed_results = self.parallel_processor.process(
                _process_threshold_pair_mode,
                tasks
            )

            # Collect results
            for pair_name, pair_results in processed_results:
                results[pair_name] = pair_results
        except Exception as e:
            logger.warning(f"Error running threshold models: {e}. Continuing with analysis.")
            # Return empty results if threshold models fail
            return {}

        return results

    def _run_spatial_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run spatial analysis on the data.

        Args:
            data: Dictionary mapping market names to DataFrames.

        Returns:
            Dictionary containing the spatial analysis results.
        """
        # Combine data from all markets
        combined_data = self.data_loader.combine_market_data(data)

        # Run spatial analysis
        return self.spatial_analyzer.run_full_analysis(combined_data)

    @handle_errors
    def _run_market_integration_simulation(
        self,
        data: Dict[str, pd.DataFrame],
        threshold_results: Dict[str, Dict[str, Any]],
        spatial_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run market integration simulation on the data.

        Args:
            data: Dictionary mapping market names to DataFrames.
            threshold_results: Threshold model results.
            spatial_results: Spatial analysis results.

        Returns:
            Dictionary containing the simulation results.
        """
        # Initialize the market integration simulator if not already initialized
        if self.market_integration_simulator is None:
            logger.info("Initializing market integration simulator")
            # Get exchange rate data from the data loader
            try:
                exchange_rate_data = self.data_loader.load_exchange_rate_data()
                self.market_integration_simulator = MarketIntegrationSimulation(
                    data=next(iter(data.values())),  # Use the first market's data
                    exchange_rate_data=exchange_rate_data
                )
            except Exception as e:
                logger.warning(f"Could not load exchange rate data: {e}. Using dummy data.")
                # Create dummy exchange rate data for testing
                dates = next(iter(data.values())).index
                exchange_rate_data = pd.DataFrame({
                    'date': dates,
                    'sanaa_rate': [600 + i * 0.5 for i in range(len(dates))],
                    'aden_rate': [500 + i * 0.3 for i in range(len(dates))]
                })
                exchange_rate_data.set_index('date', inplace=True)
                self.market_integration_simulator = MarketIntegrationSimulation(
                    data=next(iter(data.values())),
                    exchange_rate_data=exchange_rate_data
                )

        # Run exchange rate unification simulation
        logger.info("Running exchange rate unification simulation")
        try:
            # Get the first threshold model result to use as a baseline
            if threshold_results and len(threshold_results) > 0:
                first_pair = next(iter(threshold_results.items()))
                _, pair_results = first_pair

                # Extract original threshold and adjustment speed parameters
                original_threshold = pair_results.get('threshold', 0.0)
                original_adjustment_speed = pair_results.get('adjustment_speed', 0.0)

                # Run the simulation with the original parameters
                exchange_rate_results = self.market_integration_simulator.simulate_exchange_rate_unification(
                    target_rate='official',
                    method='tvecm',
                    original_threshold=original_threshold,
                    original_adjustment_speed=original_adjustment_speed,
                    y_col='price',
                    x_col='price'
                )
            else:
                logger.warning("No threshold results available for simulation. Using default parameters.")
                exchange_rate_results = self.market_integration_simulator.simulate_exchange_rate_unification(
                    target_rate='official',
                    method='tvecm',
                    y_col='price',
                    x_col='price'
                )
        except Exception as e:
            logger.warning(f"Error running exchange rate unification simulation: {e}. Using empty results.")
            exchange_rate_results = {}

        # Run spatial connectivity simulation if spatial results are available
        spatial_connectivity_results = {}
        if spatial_results is not None:
            logger.info("Running spatial connectivity simulation")
            try:
                # Extract the spatial data from the spatial results
                if 'data' in spatial_results:
                    spatial_data = spatial_results['data']

                    # Extract original spatial model parameters
                    original_results = {}
                    if 'model' in spatial_results and 'results' in spatial_results['model']:
                        original_results = spatial_results['model']['results']

                    # Run the simulation with the original parameters
                    spatial_connectivity_results = self.market_integration_simulator.simulate_spatial_connectivity(
                        data=spatial_data,
                        connectivity_improvement=0.5,
                        weight_type='distance',
                        original_results=original_results
                    )
                else:
                    logger.warning("No spatial data available for simulation. Skipping spatial connectivity simulation.")
            except Exception as e:
                logger.warning(f"Error running spatial connectivity simulation: {e}. Using empty results.")

        # Combine results
        simulation_results = {
            'exchange_rate_unification': exchange_rate_results,
            'spatial_connectivity': spatial_connectivity_results
        }

        return simulation_results

    def _generate_visualizations(
        self,
        data: Dict[str, pd.DataFrame],
        unit_root_results: Dict[str, Dict[str, Any]],
        cointegration_results: Dict[str, Dict[str, Any]],
        threshold_results: Dict[str, Dict[str, Any]],
        spatial_results: Optional[Dict[str, Any]],
        publication_quality: bool,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Generate visualizations for the analysis results.

        Args:
            data: Dictionary mapping market names to DataFrames.
            unit_root_results: Unit root test results.
            cointegration_results: Cointegration test results.
            threshold_results: Threshold model results.
            spatial_results: Spatial analysis results.
            publication_quality: Whether to generate publication-quality visualizations.
            output_dir: Directory to save visualizations.

        Returns:
            Dictionary containing the visualization results.
        """
        visualization_results = {}

        # Create visualization directory
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        # Generate time series visualizations
        ts_results = self.time_series_visualizer.generate_all_visualizations(
            data,
            unit_root_results,
            cointegration_results,
            threshold_results,
            publication_quality=publication_quality,
            output_dir=viz_dir
        )
        visualization_results['time_series'] = ts_results

        # Generate spatial visualizations if spatial results are available
        if spatial_results is not None:
            spatial_viz_results = self.spatial_visualizer.generate_all_visualizations(
                spatial_results,
                publication_quality=publication_quality,
                output_dir=viz_dir
            )
            visualization_results['spatial'] = spatial_viz_results

        return visualization_results


def run_analysis(
    data_path: Optional[Union[str, Path]] = None,
    commodity: Optional[str] = None,
    markets: Optional[List[str]] = None,
    threshold_modes: Optional[List[str]] = None,
    include_spatial: bool = True,
    include_simulation: bool = True,
    publication_quality: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Run a complete analysis for Yemen market integration.

    This is a convenience function that creates a YemenMarketAnalysis instance
    and runs the analysis.

    Args:
        data_path: Path to the data file. If None, uses the default path from config.
        commodity: Commodity to analyze. If None, analyzes all commodities.
        markets: List of markets to analyze. If None, analyzes all markets.
        threshold_modes: List of threshold modes to use. If None, uses all modes.
        include_spatial: Whether to include spatial analysis.
        include_simulation: Whether to include market integration simulation.
        publication_quality: Whether to generate publication-quality outputs.
        output_dir: Directory to save outputs. If None, uses the current directory.

    Returns:
        Dictionary containing the analysis results.
    """
    analyzer = YemenMarketAnalysis()
    return analyzer.run_analysis(
        data_path=data_path,
        commodity=commodity,
        markets=markets,
        threshold_modes=threshold_modes,
        include_spatial=include_spatial,
        include_simulation=include_simulation,
        publication_quality=publication_quality,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Yemen Market Analysis")
    parser.add_argument("--data-path", type=str, help="Path to the data file")
    parser.add_argument("--commodity", type=str, help="Commodity to analyze")
    parser.add_argument("--markets", type=str, nargs="+", help="Markets to analyze")
    parser.add_argument("--threshold-modes", type=str, nargs="+", help="Threshold modes to use")
    parser.add_argument("--no-spatial", action="store_true", help="Disable spatial analysis")
    parser.add_argument("--no-simulation", action="store_true", help="Disable market integration simulation")
    parser.add_argument("--no-publication-quality", action="store_true", help="Disable publication-quality outputs")
    parser.add_argument("--output-dir", type=str, help="Directory to save outputs")

    args = parser.parse_args()

    run_analysis(
        data_path=args.data_path,
        commodity=args.commodity,
        markets=args.markets,
        threshold_modes=args.threshold_modes,
        include_spatial=not args.no_spatial,
        include_simulation=not args.no_simulation,
        publication_quality=not args.no_publication_quality,
        output_dir=args.output_dir
    )
