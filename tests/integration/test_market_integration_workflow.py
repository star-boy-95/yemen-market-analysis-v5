"""
Integration tests for complete market integration analysis workflows.

This module tests the complete workflow from data loading through
modeling to visualization and simulation, ensuring that all
components work together correctly.
"""
import os
import unittest
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Point

from src.data.loader import load_market_data, load_spatial_data
from src.data.preprocessor import preprocess_market_data
from src.models.threshold import ThresholdCointegration
from src.models.spatial import SpatialEconometrics
from src.models.simulation import MarketIntegrationSimulation
from src.models.diagnostics import ModelDiagnostics
from src.visualization.time_series import TimeSeriesVisualizer
from src.visualization.asymmetric_plots import AsymmetricAdjustmentVisualizer
from src.visualization.spatial_integration import SpatialIntegrationVisualizer
from src.utils.validation import validate_dataframe


class TestMarketIntegrationWorkflow(unittest.TestCase):
    """Test cases for complete market integration analysis workflows."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Create temporary data directory
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Generate synthetic data
        cls.market_data, cls.spatial_data = cls._generate_test_data()
        
        # Save test data to temporary files
        cls.market_data_path = cls.temp_dir / "market_data.csv"
        cls.spatial_data_path = cls.temp_dir / "spatial_data.geojson"
        
        cls.market_data.to_csv(cls.market_data_path, index=False)
        cls.spatial_data.to_file(cls.spatial_data_path, driver="GeoJSON")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after all tests."""
        # Remove temporary files
        if cls.market_data_path.exists():
            cls.market_data_path.unlink()
            
        if cls.spatial_data_path.exists():
            cls.spatial_data_path.unlink()
            
        # Remove temporary directory
        cls.temp_dir.rmdir()
    
    @classmethod
    def _generate_test_data(cls):
        """
        Generate synthetic data for testing.
        
        Returns
        -------
        market_data : pandas.DataFrame
            Synthetic market price data
        spatial_data : geopandas.GeoDataFrame
            Synthetic spatial data for markets
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate dates
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(100)]
        
        # Generate markets
        markets = []
        for i in range(10):
            # Determine region (5 north, 5 south)
            if i < 5:
                region = "north"
                exchange_rate = 600
                lat = 15 - i * 0.3  # North markets at top
            else:
                region = "south"
                exchange_rate = 800
                lat = 12 - (i - 5) * 0.3  # South markets at bottom
            
            # Generate coordinates
            lon = 44 + i * 0.3
            
            # Add market to list
            markets.append({
                "market_id": f"M{i}",
                "market_name": f"Market {i}",
                "exchange_rate_regime": region,
                "exchange_rate": exchange_rate,
                "latitude": lat,
                "longitude": lon,
                "geometry": Point(lon, lat)
            })
        
        # Create spatial data
        spatial_data = gpd.GeoDataFrame(markets, geometry="geometry")
        
        # Generate market prices
        market_data_rows = []
        
        # Set base prices (north lower than south)
        north_base_price = 100
        south_base_price = 150
        
        # Generate prices for each market, date, and commodity
        commodities = ["Wheat", "Rice", "Flour"]
        
        for market in markets:
            # Set base price based on region
            if market["exchange_rate_regime"] == "north":
                base_price = north_base_price + np.random.normal(0, 5)
            else:
                base_price = south_base_price + np.random.normal(0, 5)
            
            # Add conflict intensity (random)
            conflict_intensity = np.abs(np.random.normal(0, 0.5)) 
            
            for date_idx, date in enumerate(dates):
                for commodity in commodities:
                    # Adjust price by commodity
                    if commodity == "Wheat":
                        commodity_price = base_price
                    elif commodity == "Rice":
                        commodity_price = base_price * 1.2
                    else:  # Flour
                        commodity_price = base_price * 0.8
                    
                    # Add time component
                    time_trend = date_idx * 0.3  # Increasing trend
                    season = 5 * np.sin(date_idx / 10)  # Seasonal component
                    noise = np.random.normal(0, 3)  # Random noise
                    
                    # Generate price
                    price = commodity_price + time_trend + season + noise
                    
                    # Add row to market data
                    market_data_rows.append({
                        "market_id": market["market_id"],
                        "market_name": market["market_name"],
                        "exchange_rate_regime": market["exchange_rate_regime"],
                        "exchange_rate": market["exchange_rate"],
                        "date": date,
                        "commodity": commodity,
                        "price": price,
                        "conflict_intensity_normalized": conflict_intensity
                    })
        
        # Create market data DataFrame
        market_data = pd.DataFrame(market_data_rows)
        
        return market_data, spatial_data
    
    def test_data_loading(self):
        """Test data loading from files."""
        # Load market data
        market_data = load_market_data(self.market_data_path)
        spatial_data = load_spatial_data(self.spatial_data_path)
        
        # Validate loaded data
        self.assertIsInstance(market_data, pd.DataFrame)
        self.assertIsInstance(spatial_data, gpd.GeoDataFrame)
        
        # Check basic properties
        self.assertEqual(len(market_data), len(self.market_data))
        self.assertEqual(len(spatial_data), len(self.spatial_data))
        
        # Check required columns
        required_market_cols = ["market_id", "date", "commodity", "price", 
                               "exchange_rate_regime"]
        required_spatial_cols = ["market_id", "geometry"]
        
        for col in required_market_cols:
            self.assertIn(col, market_data.columns)
            
        for col in required_spatial_cols:
            self.assertIn(col, spatial_data.columns)
    
    def test_time_series_workflow(self):
        """Test time series analysis workflow."""
        # Load data for a specific commodity
        wheat_data = self.market_data[self.market_data["commodity"] == "Wheat"].copy()
        
        # Calculate average prices by date and regime
        wheat_avg = wheat_data.groupby(["date", "exchange_rate_regime"])["price"].mean().unstack()
        wheat_avg = wheat_avg.rename(columns={"north": "north_price", "south": "south_price"}).reset_index()
        
        # Create time series visualizer
        ts_viz = TimeSeriesVisualizer()
        
        # Verify that plots can be created
        fig, ax = ts_viz.plot_price_series(
            wheat_data,
            price_col="price",
            date_col="date",
            group_col="exchange_rate_regime"
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Test price differential plot
        fig, ax = ts_viz.plot_price_differentials(
            wheat_avg,
            date_col="date",
            north_col="north_price",
            south_col="south_price"
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Clean up plots
        plt.close("all")
    
    def test_threshold_cointegration_workflow(self):
        """Test threshold cointegration analysis workflow."""
        # Load data for a specific commodity
        wheat_data = self.market_data[self.market_data["commodity"] == "Wheat"].copy()
        
        # Calculate average prices by date and regime
        wheat_avg = wheat_data.groupby(["date", "exchange_rate_regime"])["price"].mean().unstack()
        wheat_avg = wheat_avg.rename(columns={"north": "north_price", "south": "south_price"}).reset_index()
        
        # Extract price series
        north_prices = wheat_avg["north_price"].values
        south_prices = wheat_avg["south_price"].values
        
        # Create threshold cointegration model
        model = ThresholdCointegration(north_prices, south_prices, 
                                      market1_name="North", market2_name="South")
        
        # Test cointegration estimation
        cointegration_result = model.estimate_cointegration()
        self.assertIsNotNone(cointegration_result)
        self.assertIn("cointegrated", cointegration_result)
        self.assertIn("test_statistic", cointegration_result)
        
        # Only proceed with threshold estimation if cointegrated
        if cointegration_result["cointegrated"]:
            # Test threshold estimation
            threshold_result = model.estimate_threshold()
            self.assertIsNotNone(threshold_result)
            self.assertIn("threshold", threshold_result)
            
            # Store dates and price differentials for visualization
            model.dates = wheat_avg["date"].values
            model.price_diff = north_prices - south_prices
            
            # Test M-TAR model estimation
            mtar_result = model.estimate_mtar()
            self.assertIsNotNone(mtar_result)
            self.assertIn("asymmetric", mtar_result)
            self.assertIn("adjustment_negative", mtar_result)
            self.assertIn("adjustment_positive", mtar_result)
            
            # Test asymmetric visualization
            asym_viz = AsymmetricAdjustmentVisualizer()
            fig, ax = asym_viz.plot_regime_dynamics(
                model.price_diff,
                model.dates,
                model.threshold
            )
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            
            # Clean up plots
            plt.close("all")
    
    def test_spatial_analysis_workflow(self):
        """Test spatial analysis workflow."""
        # Load the latest data for each market
        latest_date = self.market_data["date"].max()
        latest_wheat = self.market_data[
            (self.market_data["date"] == latest_date) & 
            (self.market_data["commodity"] == "Wheat")
        ].copy()
        
        # Merge with spatial data
        market_data_geo = latest_wheat.merge(
            self.spatial_data[["market_id", "geometry"]], 
            on="market_id"
        )
        market_data_geo = gpd.GeoDataFrame(market_data_geo, geometry="geometry")
        
        # Calculate a simple integration metric
        market_metrics = self.market_data[
            self.market_data["commodity"] == "Wheat"
        ].groupby("market_id")["price"].std().reset_index()
        
        market_metrics["integration_metric"] = 1 / (
            market_metrics["price"] / market_metrics["price"].max()
        )
        market_metrics["integration_metric"] = market_metrics["integration_metric"] / market_metrics["integration_metric"].max()
        
        # Merge metrics with latest wheat data
        latest_wheat = latest_wheat.merge(
            market_metrics[["market_id", "integration_metric"]], 
            on="market_id"
        )
        
        # Create spatial integration visualizer
        spatial_viz = SpatialIntegrationVisualizer()
        
        # Convert to GeoDataFrame
        latest_wheat_geo = gpd.GeoDataFrame(
            latest_wheat.merge(self.spatial_data[["market_id", "geometry"]], on="market_id"),
            geometry="geometry"
        )
        
        # Plot integration choropleth
        fig, ax = spatial_viz.plot_integration_choropleth(
            latest_wheat_geo,
            metric_col="integration_metric",
            market_id_col="market_id",
            region_col="exchange_rate_regime"
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Clean up plots
        plt.close("all")
    
    def test_policy_simulation_workflow(self):
        """Test policy simulation workflow."""
        # Get latest data for wheat
        latest_date = self.market_data["date"].max()
        latest_wheat = self.market_data[
            (self.market_data["date"] == latest_date) & 
            (self.market_data["commodity"] == "Wheat")
        ].copy()
        
        # Merge with spatial data
        latest_wheat_geo = latest_wheat.merge(
            self.spatial_data[["market_id", "geometry"]], 
            on="market_id"
        )
        latest_wheat_geo = gpd.GeoDataFrame(latest_wheat_geo, geometry="geometry")
        
        # Calculate price series for north and south
        wheat_data = self.market_data[self.market_data["commodity"] == "Wheat"].copy()
        wheat_avg = wheat_data.groupby(["date", "exchange_rate_regime"])["price"].mean().unstack()
        wheat_avg = wheat_avg.rename(columns={"north": "north_price", "south": "south_price"}).reset_index()
        
        # Create threshold model
        north_prices = wheat_avg["north_price"].values
        south_prices = wheat_avg["south_price"].values
        threshold_model = ThresholdCointegration(
            north_prices, south_prices, 
            market1_name="North", market2_name="South"
        )
        threshold_model.estimate_cointegration()
        
        # Create simulation with market data and threshold model
        simulation = MarketIntegrationSimulation(
            latest_wheat_geo,
            threshold_model=threshold_model
        )
        
        # Test exchange rate unification simulation
        exchange_rate_results = simulation.simulate_exchange_rate_unification(
            target_rate="official"
        )
        self.assertIsNotNone(exchange_rate_results)
        self.assertIn("unified_rate", exchange_rate_results)
        self.assertIn("price_changes", exchange_rate_results)
        
        # Test improved connectivity simulation
        connectivity_results = simulation.simulate_improved_connectivity(
            reduction_factor=0.5
        )
        self.assertIsNotNone(connectivity_results)
        self.assertIn("simulated_data", connectivity_results)
        
        # Test combined policy simulation
        combined_results = simulation.simulate_combined_policy(
            exchange_rate_target="official",
            conflict_reduction=0.5
        )
        self.assertIsNotNone(combined_results)
        self.assertIn("simulated_data", combined_results)
        
        # Test welfare effects calculation
        welfare = simulation.calculate_welfare_effects("combined_policy")
        self.assertIsNotNone(welfare)
        self.assertIn("price_convergence", welfare)
    
    def test_model_diagnostics_workflow(self):
        """Test model diagnostics workflow."""
        # Load data for a specific commodity
        wheat_data = self.market_data[self.market_data["commodity"] == "Wheat"].copy()
        
        # Calculate average prices by date and regime
        wheat_avg = wheat_data.groupby(["date", "exchange_rate_regime"])["price"].mean().unstack()
        wheat_avg = wheat_avg.rename(columns={"north": "north_price", "south": "south_price"}).reset_index()
        
        # Extract price series
        north_prices = wheat_avg["north_price"].values
        south_prices = wheat_avg["south_price"].values
        
        # Create threshold cointegration model
        model = ThresholdCointegration(north_prices, south_prices)
        
        # Estimate cointegration
        model.estimate_cointegration()
        
        # Use residuals for diagnostics
        residuals = model.residuals
        
        # Create model diagnostics
        diagnostics = ModelDiagnostics(residuals=residuals)
        
        # Test residual diagnostics
        residual_tests = diagnostics.residual_tests()
        self.assertIsNotNone(residual_tests)
        self.assertIn("normality", residual_tests)
        self.assertIn("autocorrelation", residual_tests)
        
        # Calculate model selection criteria
        criteria = diagnostics.model_selection_criteria(
            observed=south_prices,
            predicted=north_prices * model.beta,
            n_params=2
        )
        self.assertIsNotNone(criteria)
        self.assertIn("aic", criteria)
        self.assertIn("bic", criteria)
        self.assertIn("r_squared", criteria)


if __name__ == "__main__":
    unittest.main()
