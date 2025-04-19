"""
Example script demonstrating how to handle insufficient data exceptions in a user interface.
"""
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the modules
from src.utils.error_handling import YemenAnalysisError
from src.models.unit_root import UnitRootTester
from src.models.cointegration.engle_granger import EngleGrangerTester

def analyze_market_data(data, column='price'):
    """
    Analyze market data and handle insufficient data exceptions.
    
    Args:
        data: DataFrame containing market data.
        column: Column to analyze.
        
    Returns:
        Dictionary containing analysis results or None if analysis failed.
    """
    results = {}
    
    # Try to run unit root test
    try:
        logger.info(f"Running ADF test on {column}...")
        tester = UnitRootTester()
        adf_results = tester.test_adf(data, column=column)
        results['adf'] = adf_results
        logger.info(f"ADF test results: {adf_results['is_stationary']}")
    except YemenAnalysisError as e:
        logger.warning(f"Could not run ADF test: {e}")
        results['adf'] = None
    
    # Try to run KPSS test
    try:
        logger.info(f"Running KPSS test on {column}...")
        tester = UnitRootTester()
        kpss_results = tester.test_kpss(data, column=column)
        results['kpss'] = kpss_results
        logger.info(f"KPSS test results: {kpss_results['is_stationary']}")
    except YemenAnalysisError as e:
        logger.warning(f"Could not run KPSS test: {e}")
        results['kpss'] = None
    
    return results

def analyze_market_pair(y_data, x_data):
    """
    Analyze a pair of markets and handle insufficient data exceptions.
    
    Args:
        y_data: DataFrame containing data for the first market.
        x_data: DataFrame containing data for the second market.
        
    Returns:
        Dictionary containing analysis results or None if analysis failed.
    """
    results = {}
    
    # Try to run Engle-Granger cointegration test
    try:
        logger.info("Running Engle-Granger cointegration test...")
        tester = EngleGrangerTester()
        eg_results = tester.test(y_data, x_data)
        results['engle_granger'] = eg_results
        logger.info(f"Engle-Granger test results: {eg_results['is_cointegrated']}")
    except YemenAnalysisError as e:
        logger.warning(f"Could not run Engle-Granger test: {e}")
        results['engle_granger'] = None
    
    return results

def main():
    """Main function."""
    logger.info("Starting market analysis...")
    
    # Create a small dataset with only 3 observations (insufficient data)
    small_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=3),
        'price': [10.0, 10.5, 11.0]
    })
    
    # Create a larger dataset with 20 observations (sufficient data)
    large_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=20),
        'price': np.random.normal(10, 1, 20)
    })
    
    # Analyze small dataset (should handle exceptions)
    logger.info("Analyzing small dataset...")
    small_results = analyze_market_data(small_data)
    
    # Analyze large dataset (should work)
    logger.info("Analyzing large dataset...")
    large_results = analyze_market_data(large_data)
    
    # Compare results
    logger.info("Analysis complete.")
    logger.info(f"Small dataset results: {small_results}")
    logger.info(f"Large dataset results available: ADF={large_results['adf'] is not None}, KPSS={large_results['kpss'] is not None}")
    
    # Provide user-friendly summary
    print("\nAnalysis Summary:")
    print("-----------------")
    print("Small dataset (3 observations):")
    if all(v is None for v in small_results.values()):
        print("  - All tests failed due to insufficient data")
    else:
        for test, result in small_results.items():
            print(f"  - {test.upper()}: {'Success' if result is not None else 'Failed - insufficient data'}")
    
    print("\nLarge dataset (20 observations):")
    if all(v is not None for v in large_results.values()):
        print("  - All tests completed successfully")
    else:
        for test, result in large_results.items():
            print(f"  - {test.upper()}: {'Success' if result is not None else 'Failed - insufficient data'}")

if __name__ == "__main__":
    main()
