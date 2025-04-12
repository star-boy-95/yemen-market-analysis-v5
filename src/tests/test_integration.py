"""
Integration tests for the Yemen Market Analysis package.

This test suite runs key components of the package to verify they work together properly.
"""
import unittest
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add paths to both implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class IntegrationTest(unittest.TestCase):
    """Test the integration of various components of the Yemen Market Analysis package."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and configurations once for all tests."""
        # Sample data path - update to your test data location
        cls.data_path = Path("../../data/raw/sample_data.geojson")
        cls.commodity = "wheat"
        cls.output_dir_new = Path("./test_output/new")
        
        # Create output directory
        cls.output_dir_new.mkdir(parents=True, exist_ok=True)
    
    def test_unit_root_results(self):
        """Test that unit root analysis works correctly."""
        from src.data.loader import DataLoader
        from src.models.unit_root import UnitRootTester
        
        loader = DataLoader()
        data = loader.load_data(self.data_path)
        tester = UnitRootTester()
        results = tester.run_adf_test(data["north"])
        
        # Verify key result properties exist
        self.assertIn("adf_statistic", results)
        self.assertIn("p_value", results)
        self.assertIn("critical_values", results)
        
        # Verify result is a valid ADF test output
        if results["p_value"] < 0.05:
            self.assertLess(results["adf_statistic"], results["critical_values"]["5%"])
        else:
            self.assertGreaterEqual(results["adf_statistic"], results["critical_values"]["5%"])
    
    def test_cointegration_results(self):
        """Test that cointegration results are equivalent."""
        # Similar structure to the unit root test
        # Implement comparison of cointegration results
        pass
    
    def test_threshold_model_results(self):
        """Test that threshold model results are equivalent."""
        # Implement comparison of threshold model results
        pass
    
    def test_end_to_end_analysis(self):
        """Run complete analysis with the main entry point."""
        from src.main import run_analysis
        
        results = run_analysis(
            data_path=self.data_path,
            commodity=self.commodity,
            threshold_modes=["standard"],
            include_spatial=True,
            output_dir=self.output_dir_new
        )
        
        # Verify output files were generated
        output_files = list(self.output_dir_new.glob("*.csv"))
        
        # Check that we have output files
        self.assertGreater(
            len(output_files), 0,
            "No output files were generated"
        )
        
        # Basic checks on output files
        for file in output_files:
            # Verify file exists and is not empty
            self.assertTrue(file.exists())
            self.assertGreater(file.stat().st_size, 0)
            
            # Read in CSV and check it has data
            try:
                df = pd.read_csv(file)
                self.assertGreater(len(df), 0, f"Empty data in {file.name}")
                self.assertGreater(len(df.columns), 0, f"No columns in {file.name}")
            except Exception as e:
                self.fail(f"Failed to read output file {file.name}: {str(e)}")


if __name__ == "__main__":
    unittest.main()
