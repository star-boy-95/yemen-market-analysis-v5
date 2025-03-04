"""
Stationarity testing for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from statsmodels.tsa.stattools import adfuller, kpss

from core.decorators import error_handler, performance_tracker
from core.exceptions import StatisticalTestError

logger = logging.getLogger(__name__)


class StationarityTest:
    """Class for comprehensive stationarity testing of time series data."""
    
    def __init__(
        self, 
        series: Union[pd.Series, np.ndarray],
        test_methods: List[str] = None
    ):
        """Initialize stationarity test with data series."""
        self.series = series if isinstance(series, np.ndarray) else series.values
        self.test_methods = test_methods or ['adf', 'kpss', 'pp']
        self.results = {}
    
    @error_handler(fallback_value={"error": "ADF test failed"})
    def adf_test(
        self, 
        regression: str = 'c', 
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run Augmented Dickey-Fuller test."""
        result = adfuller(
            self.series, 
            regression=regression,
            maxlag=max_lags,
            autolag='AIC'
        )
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'stationary': result[1] < 0.05,
            'method': 'ADF',
            'regression': regression
        }
    
    @error_handler(fallback_value={"error": "KPSS test failed"})
    def kpss_test(
        self, 
        regression: str = 'c', 
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run KPSS test."""
        result = kpss(
            self.series, 
            regression=regression,
            nlags=max_lags
        )
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags': result[2],
            'critical_values': result[3],
            'stationary': result[1] > 0.05,  # Note: opposite null from ADF
            'method': 'KPSS',
            'regression': regression
        }
    
    @error_handler(fallback_value={"error": "Phillips-Perron test failed"})
    def pp_test(
        self, 
        regression: str = 'c', 
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run Phillips-Perron test."""
        from statsmodels.tsa.stattools import phillips_perron
        
        result = phillips_perron(
            self.series, 
            trend=regression,
            lags=max_lags
        )
        
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[2],
            'stationary': result[1] < 0.05,
            'method': 'Phillips-Perron',
            'regression': regression
        }
    
    def _run_test_by_name(
        self, 
        test_name: str, 
        regression: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a specific test by name."""
        if test_name.lower() == 'adf':
            return self.adf_test(regression, max_lags)
        elif test_name.lower() == 'kpss':
            return self.kpss_test(regression, max_lags)
        elif test_name.lower() == 'pp':
            return self.pp_test(regression, max_lags)
        else:
            return {"error": f"Unknown test method: {test_name}"}
    
    def _calculate_consensus(self) -> Dict[str, Any]:
        """Calculate consensus across multiple test results."""
        # Count number of tests that indicate stationarity
        stationary_count = sum(
            1 for method in self.results
            if self.results[method].get('stationary', False)
        )
        
        total_tests = len(self.results)
        
        if total_tests == 0:
            return {"error": "No valid test results"}
        
        # Calculate consensus
        consensus_pct = (stationary_count / total_tests) * 100
        
        if consensus_pct >= 75:
            consensus = "Strong evidence of stationarity"
            stationary = True
        elif consensus_pct >= 50:
            consensus = "Moderate evidence of stationarity"
            stationary = True
        elif consensus_pct >= 25:
            consensus = "Weak evidence of stationarity"
            stationary = False
        else:
            consensus = "Strong evidence of non-stationarity"
            stationary = False
        
        return {
            'consensus': consensus,
            'stationary': stationary,
            'stationary_percentage': consensus_pct,
            'stationary_count': stationary_count,
            'total_tests': total_tests
        }
    
    @error_handler(fallback_value={"error": "Stationarity testing failed"})
    @performance_tracker()
    def run_all_tests(
        self, 
        regression: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run all specified stationarity tests and compute consensus."""
        # Execute individual tests
        for method in self.test_methods:
            self.results[method.upper()] = self._run_test_by_name(method, regression, max_lags)

        # Compute consensus and statistics
        consensus = self._calculate_consensus()
        
        return {
            'individual_tests': self.results,
            'consensus': consensus,
            'regression': regression
        }


@error_handler(fallback_value=(False, {"error": "Stationarity test failed"}))
def test_stationarity(
    series: Union[pd.Series, np.ndarray],
    method: str = 'adf',
    regression: str = 'c',
    max_lags: Optional[int] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test for stationarity using specified method.
    
    Args:
        series: Time series data
        method: Test method ('adf', 'kpss', 'pp', or 'all')
        regression: Type of regression ('c', 'ct', 'n', etc.)
        max_lags: Maximum number of lags
        
    Returns:
        Tuple of (is_stationary, test_results)
    """
    # Convert to numpy array if needed
    data = series.values if isinstance(series, pd.Series) else series
    
    # Check for sufficient observations
    if len(data) < 20:
        return False, {"error": "Insufficient observations for stationarity test"}
    
    # Check for constant values
    if np.std(data) == 0:
        return False, {"error": "Constant series cannot be stationary"}
    
    # Run comprehensive test if 'all' specified
    if method.lower() == 'all':
        tester = StationarityTest(data, test_methods=['adf', 'kpss', 'pp'])
        results = tester.run_all_tests(regression, max_lags)
        return results['consensus']['stationary'], results
    
    # Run single test
    tester = StationarityTest(data, test_methods=[method])
    result = tester._run_test_by_name(method, regression, max_lags)
    
    return result.get('stationary', False), result


@error_handler(fallback_value=(False, {"error": "Stationarity test failed"}))
def test_for_unit_root(
    series: Union[pd.Series, np.ndarray],
    significance_level: float = 0.05,
    regression: str = 'c',
    max_lags: Optional[int] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test for unit root (non-stationarity).
    
    Args:
        series: Time series data
        significance_level: Significance level for tests
        regression: Type of regression ('c', 'ct', 'n', etc.)
        max_lags: Maximum number of lags
        
    Returns:
        Tuple of (has_unit_root, test_results)
    """
    stationary, results = test_stationarity(
        series, method='adf', regression=regression, max_lags=max_lags
    )
    
    # ADF null hypothesis is unit root, so non-rejection means has_unit_root=True
    has_unit_root = not stationary
    
    return has_unit_root, results