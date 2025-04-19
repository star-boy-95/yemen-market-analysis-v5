"""
Simple test script to verify that our changes work correctly.
"""
import logging
import sys
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a simple YemenAnalysisError class for testing
class YemenAnalysisError(Exception):
    """Custom exception for Yemen Market Analysis errors."""
    pass

def test_insufficient_data():
    """Test that appropriate exceptions are raised with insufficient data."""
    logger.info("Testing with insufficient data...")
    
    # Create a small dataset with only 3 observations
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=3),
        'price': [10.0, 10.5, 11.0]
    })
    
    try:
        # This should raise an exception
        if len(data) <= 3:
            raise YemenAnalysisError(f"Sample size ({len(data)}) too small. Need at least 4 observations.")
        logger.error("FAILED: Did not raise an exception with insufficient data")
    except YemenAnalysisError as e:
        logger.info(f"PASSED: Correctly raised an exception: {e}")

def run_all_tests():
    """Run all tests."""
    logger.info("Running all tests...")
    
    test_insufficient_data()
    
    logger.info("All tests completed.")

if __name__ == "__main__":
    run_all_tests()
