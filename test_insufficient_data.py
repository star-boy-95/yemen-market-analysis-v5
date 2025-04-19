"""
Test script to verify that appropriate exceptions are raised when data is insufficient.
"""
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the modules to test
from src.models.unit_root import UnitRootTester
from src.models.cointegration.engle_granger import EngleGrangerTester
from src.models.cointegration.gregory_hansen import GregoryHansenTester
from src.models.cointegration.johansen import JohansenTester
from src.models.threshold.tar import ThresholdAutoregressive
from src.models.threshold.mtar import MomentumThresholdAutoregressive
from src.models.threshold.tvecm import ThresholdVECM
from src.utils.error_handling import YemenAnalysisError

def test_unit_root_insufficient_data():
    """Test that UnitRootTester raises an exception with insufficient data."""
    logger.info("Testing UnitRootTester with insufficient data...")
    
    # Create a small dataset with only 3 observations
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=3),
        'price': [10.0, 10.5, 11.0]
    })
    
    tester = UnitRootTester()
    
    try:
        # This should raise an exception
        tester.test_adf(data, column='price')
        logger.error("FAILED: UnitRootTester did not raise an exception with insufficient data")
    except YemenAnalysisError as e:
        logger.info(f"PASSED: UnitRootTester correctly raised an exception: {e}")

def test_engle_granger_insufficient_data():
    """Test that EngleGrangerTester raises an exception with insufficient data."""
    logger.info("Testing EngleGrangerTester with insufficient data...")
    
    # Create a small dataset with only 5 observations
    y_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'price': [10.0, 10.5, 11.0, 11.5, 12.0]
    })
    
    x_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5),
        'price': [20.0, 20.5, 21.0, 21.5, 22.0]
    })
    
    tester = EngleGrangerTester()
    
    try:
        # This should raise an exception
        tester.test(y_data, x_data)
        logger.error("FAILED: EngleGrangerTester did not raise an exception with insufficient data")
    except YemenAnalysisError as e:
        logger.info(f"PASSED: EngleGrangerTester correctly raised an exception: {e}")

def test_gregory_hansen_insufficient_data():
    """Test that GregoryHansenTester raises an exception with insufficient data."""
    logger.info("Testing GregoryHansenTester with insufficient data...")
    
    # Create a small dataset with only 7 observations
    y_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'price': [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
    })
    
    x_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'price': [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0]
    })
    
    tester = GregoryHansenTester()
    
    try:
        # This should raise an exception
        tester.test(y_data, x_data)
        logger.error("FAILED: GregoryHansenTester did not raise an exception with insufficient data")
    except YemenAnalysisError as e:
        logger.info(f"PASSED: GregoryHansenTester correctly raised an exception: {e}")

def test_johansen_insufficient_data():
    """Test that JohansenTester raises an exception with insufficient data."""
    logger.info("Testing JohansenTester with insufficient data...")
    
    # Create a small dataset with only 7 observations
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'y': [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
        'x': [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0]
    })
    
    tester = JohansenTester()
    
    try:
        # This should raise an exception
        tester.test(data, columns=['y', 'x'])
        logger.error("FAILED: JohansenTester did not raise an exception with insufficient data")
    except YemenAnalysisError as e:
        logger.info(f"PASSED: JohansenTester correctly raised an exception: {e}")

def test_tar_insufficient_data():
    """Test that ThresholdAutoregressive raises an exception with insufficient data."""
    logger.info("Testing ThresholdAutoregressive with insufficient data...")
    
    # Create a small dataset with only 7 observations
    y_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'price': [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
    })
    
    x_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'price': [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0]
    })
    
    model = ThresholdAutoregressive()
    
    try:
        # This should raise an exception
        model.estimate(y_data, x_data)
        logger.error("FAILED: ThresholdAutoregressive did not raise an exception with insufficient data")
    except YemenAnalysisError as e:
        logger.info(f"PASSED: ThresholdAutoregressive correctly raised an exception: {e}")

def test_mtar_insufficient_data():
    """Test that MomentumThresholdAutoregressive raises an exception with insufficient data."""
    logger.info("Testing MomentumThresholdAutoregressive with insufficient data...")
    
    # Create a small dataset with only 7 observations
    y_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'price': [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
    })
    
    x_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'price': [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0]
    })
    
    model = MomentumThresholdAutoregressive()
    
    try:
        # This should raise an exception
        model.estimate(y_data, x_data)
        logger.error("FAILED: MomentumThresholdAutoregressive did not raise an exception with insufficient data")
    except YemenAnalysisError as e:
        logger.info(f"PASSED: MomentumThresholdAutoregressive correctly raised an exception: {e}")

def test_tvecm_insufficient_data():
    """Test that ThresholdVECM raises an exception with insufficient data."""
    logger.info("Testing ThresholdVECM with insufficient data...")
    
    # Create a small dataset with only 7 observations
    y_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'price': [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
    })
    
    x_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=7),
        'price': [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0]
    })
    
    model = ThresholdVECM()
    
    try:
        # This should raise an exception
        model.estimate(y_data, x_data)
        logger.error("FAILED: ThresholdVECM did not raise an exception with insufficient data")
    except YemenAnalysisError as e:
        logger.info(f"PASSED: ThresholdVECM correctly raised an exception: {e}")

def run_all_tests():
    """Run all tests."""
    logger.info("Running all tests...")
    
    test_unit_root_insufficient_data()
    test_engle_granger_insufficient_data()
    test_gregory_hansen_insufficient_data()
    test_johansen_insufficient_data()
    test_tar_insufficient_data()
    test_mtar_insufficient_data()
    test_tvecm_insufficient_data()
    
    logger.info("All tests completed.")

if __name__ == "__main__":
    run_all_tests()
