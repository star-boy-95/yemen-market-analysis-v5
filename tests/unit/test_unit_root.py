import pytest
import pandas as pd
import numpy as np
from src.models.unit_root import UnitRootTester

def test_adf_runs_without_errors():
    # Create a sample time series
    data = np.random.randn(100)
    series = pd.Series(data)

    # Initialize the UnitRootTester
    tester = UnitRootTester()

    # Run the test_adf method
    try:
        tester.test_adf(series)
    except Exception as e:
        pytest.fail(f"test_adf raised an exception: {e}")