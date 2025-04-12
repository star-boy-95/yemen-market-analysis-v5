"""
Unit root testing module for Yemen Market Analysis.

This module provides functions for testing for unit roots in time series data.
It includes implementations of various unit root tests, including ADF, KPSS,
Phillips-Perron, and Zivot-Andrews tests.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from arch.unitroot import PhillipsPerron, ZivotAndrews, DFGLS

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class UnitRootTester:
    """
    Unit root tester for Yemen Market Analysis.
    
    This class provides methods for testing for unit roots in time series data.
    It includes implementations of various unit root tests, including ADF, KPSS,
    Phillips-Perron, and Zivot-Andrews tests.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
    """
    
    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the unit root tester.
        
        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.unit_root.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.unit_root.max_lags', 4)
    
    @handle_errors
    def test_adf(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test for unit root.
        
        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant),
                  'ct' (constant and trend), 'ctt' (constant, linear and quadratic trend),
                  and 'n' (no trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the column is not found or the test fails.
        """
        logger.info(f"Performing ADF test on {column} with trend={trend}")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        col_data = data[column].dropna()
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
            
        # Handle small sample sizes
        n_obs = len(col_data)
        n_trend = 0
        if trend == 'c':
            n_trend = 1
        elif trend == 'ct':
            n_trend = 2
        elif trend == 'ctt':
            n_trend = 3
            
        # Calculate maximum possible lags for this sample
        max_possible_lags = int((n_obs / 2) - 1 - n_trend)
        
        # If max_lags is too large, reduce it
        if max_lags >= max_possible_lags:
            logger.warning(f"Reducing max_lags from {max_lags} to {max(1, max_possible_lags - 1)} due to small sample size")
            max_lags = max(1, max_possible_lags - 1)  # Ensure at least 1 lag
        
        try:
            # Perform ADF test
            if n_obs <= 3:  # Handle extremely small samples with mock results
                logger.warning(f"Sample size ({n_obs}) too small for ADF test. Returning mock results.")
                return self._mock_adf_results(column, trend, n_obs)
                
            adf_result = adfuller(col_data, maxlag=max_lags, regression=trend)
            
            # Extract results
            test_statistic = adf_result[0]
            p_value = adf_result[1]
            critical_values = adf_result[4]
            n_lags = adf_result[2]
            n_obs = adf_result[3]
            
            # Determine if the series is stationary
            is_stationary = p_value < self.alpha
            
            # Create results dictionary
            results = {
                'test': 'ADF',
                'column': column,
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'n_lags': n_lags,
                'n_obs': n_obs,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
            }
            
            logger.info(f"ADF test results: test_statistic={test_statistic:.4f}, p_value={p_value:.4f}, is_stationary={is_stationary}")
            return results
        except Exception as e:
            logger.error(f"Error performing ADF test: {e}")
            raise YemenAnalysisError(f"Error performing ADF test: {e}")
    
    def _mock_adf_results(self, column: str, trend: str, n_obs: int) -> Dict[str, Any]:
        """
        Create mock ADF test results for small sample sizes.
        
        Args:
            column: Column name that was tested.
            trend: Trend that was used in the test.
            n_obs: Number of observations.
            
        Returns:
            Dictionary containing mock ADF test results.
        """
        return {
            'test': 'ADF',
            'column': column,
            'trend': trend,
            'test_statistic': -2.0,  # Mock value suggesting non-stationarity
            'p_value': 0.2,  # Mock p-value > alpha suggesting non-stationarity
            'critical_values': {'1%': -3.5, '5%': -2.9, '10%': -2.6},  # Typical critical values
            'n_lags': 1,
            'n_obs': n_obs,
            'is_stationary': False,  # Assuming non-stationarity for safety
            'alpha': self.alpha,
            'mock_result': True  # Flag to indicate this is a mock result
        }
    
    @handle_errors
    def test_kpss(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform KPSS test for stationarity.
        
        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant) and
                  'ct' (constant and trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the column is not found or the test fails.
        """
        logger.info(f"Performing KPSS test on {column} with trend={trend}")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        col_data = data[column].dropna()
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
            
        # Handle small sample sizes
        n_obs = len(col_data)
        
        # For KPSS, lags must be < number of observations
        if max_lags >= n_obs:
            logger.warning(f"Reducing max_lags from {max_lags} to {max(1, n_obs - 2)} due to small sample size")
            max_lags = max(1, n_obs - 2)  # Ensure at least 1 lag
        
        try:
            # Perform KPSS test
            if n_obs <= 3:  # Handle extremely small samples with mock results
                logger.warning(f"Sample size ({n_obs}) too small for KPSS test. Returning mock results.")
                return self._mock_kpss_results(column, trend, n_obs)
                
            kpss_result = kpss(col_data, regression=trend, nlags=max_lags)
            
            # Extract results
            test_statistic = kpss_result[0]
            p_value = kpss_result[1]
            critical_values = kpss_result[3]
            n_lags = kpss_result[2]
            
            # Determine if the series is stationary (note: KPSS null hypothesis is stationarity)
            is_stationary = p_value > self.alpha
            
            # Create results dictionary
            results = {
                'test': 'KPSS',
                'column': column,
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'n_lags': n_lags,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
            }
            
            logger.info(f"KPSS test results: test_statistic={test_statistic:.4f}, p_value={p_value:.4f}, is_stationary={is_stationary}")
            return results
        except Exception as e:
            logger.error(f"Error performing KPSS test: {e}")
            raise YemenAnalysisError(f"Error performing KPSS test: {e}")
    
    def _mock_kpss_results(self, column: str, trend: str, n_obs: int) -> Dict[str, Any]:
        """
        Create mock KPSS test results for small sample sizes.
        
        Args:
            column: Column name that was tested.
            trend: Trend that was used in the test.
            n_obs: Number of observations.
            
        Returns:
            Dictionary containing mock KPSS test results.
        """
        return {
            'test': 'KPSS',
            'column': column,
            'trend': trend,
            'test_statistic': 0.2,  # Mock value suggesting stationarity
            'p_value': 0.1,  # KPSS doesn't provide p-values directly
            'critical_values': {'1%': 0.739, '5%': 0.463, '10%': 0.347},  # Typical critical values
            'n_lags': 1,
            'n_obs': n_obs,
            'is_stationary': True,  # Assuming stationarity for safety
            'alpha': self.alpha,
            'mock_result': True  # Flag to indicate this is a mock result
        }
    
    @handle_errors
    def test_pp(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Phillips-Perron test for unit root.
        
        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant),
                  'ct' (constant and trend), and 'n' (no trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the column is not found or the test fails.
        """
        logger.info(f"Performing Phillips-Perron test on {column} with trend={trend}")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        col_data = data[column].dropna()
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
            
        # Handle small sample sizes
        n_obs = len(col_data)
        
        try:
            # Handle small sample sizes with mock results
            if n_obs < 4:  # PP test requires at least 4 observations
                logger.warning(f"Sample size ({n_obs}) too small for Phillips-Perron test. Returning mock results.")
                return self._mock_pp_results(column, trend, n_obs)
                
            # Perform Phillips-Perron test
            pp = PhillipsPerron(col_data, trend=trend, lags=max_lags)
            pp_result = pp.run()
            
            # Extract results
            test_statistic = pp_result.stat
            p_value = pp_result.pvalue
            critical_values = {
                '1%': pp_result.critical_values['1%'],
                '5%': pp_result.critical_values['5%'],
                '10%': pp_result.critical_values['10%']
            }
            n_lags = pp.lags
            
            # Determine if the series is stationary
            is_stationary = p_value < self.alpha
            
            # Create results dictionary
            results = {
                'test': 'Phillips-Perron',
                'column': column,
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'n_lags': n_lags,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
            }
            
            logger.info(f"Phillips-Perron test results: test_statistic={test_statistic:.4f}, p_value={p_value:.4f}, is_stationary={is_stationary}")
            return results
        except Exception as e:
            logger.error(f"Error performing Phillips-Perron test: {e}")
            raise YemenAnalysisError(f"Error performing Phillips-Perron test: {e}")
    
    def _mock_pp_results(self, column: str, trend: str, n_obs: int) -> Dict[str, Any]:
        """
        Create mock Phillips-Perron test results for small sample sizes.
        
        Args:
            column: Column name that was tested.
            trend: Trend that was used in the test.
            n_obs: Number of observations.
            
        Returns:
            Dictionary containing mock Phillips-Perron test results.
        """
        return {
            'test': 'Phillips-Perron',
            'column': column,
            'trend': trend,
            'test_statistic': -2.0,  # Mock value suggesting non-stationarity
            'p_value': 0.2,  # Mock p-value > alpha suggesting non-stationarity
            'critical_values': {'1%': -3.5, '5%': -2.9, '10%': -2.6},  # Typical critical values
            'n_lags': 1,
            'n_obs': n_obs,
            'is_stationary': False,  # Assuming non-stationarity for safety
            'alpha': self.alpha,
            'mock_result': True  # Flag to indicate this is a mock result
        }
    
    @handle_errors
    def test_za(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Zivot-Andrews test for unit root with a structural break.
        
        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant),
                  't' (trend), and 'ct' (constant and trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the column is not found or the test fails.
        """
        logger.info(f"Performing Zivot-Andrews test on {column} with trend={trend}")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        col_data = data[column].dropna()
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
            
        # Handle small sample sizes
        n_obs = len(col_data)
        
        try:
            # Handle small sample sizes with mock results
            if n_obs <= 5:  # ZA test requires a minimum sample size
                logger.warning(f"Sample size ({n_obs}) too small for Zivot-Andrews test. Returning mock results.")
                return self._mock_za_results(column, trend, n_obs)
                
            # Perform Zivot-Andrews test
            za = ZivotAndrews(col_data, model=trend, lags=max_lags)
            za_result = za.run()
            
            # Extract results
            test_statistic = za_result.stat
            p_value = za_result.pvalue
            critical_values = {
                '1%': za_result.critical_values['1%'],
                '5%': za_result.critical_values['5%'],
                '10%': za_result.critical_values['10%']
            }
            n_lags = za_result.lags
            break_date = za_result.zacd
            
            # Determine if the series is stationary
            is_stationary = p_value < self.alpha
            
            # Create results dictionary
            results = {
                'test': 'Zivot-Andrews',
                'column': column,
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'n_lags': n_lags,
                'break_date': break_date,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
            }
            
            logger.info(f"Zivot-Andrews test results: test_statistic={test_statistic:.4f}, p_value={p_value:.4f}, is_stationary={is_stationary}, break_date={break_date}")
            return results
        except Exception as e:
            logger.error(f"Error performing Zivot-Andrews test: {e}")
            raise YemenAnalysisError(f"Error performing Zivot-Andrews test: {e}")
    
    def _mock_za_results(self, column: str, trend: str, n_obs: int) -> Dict[str, Any]:
        """
        Create mock Zivot-Andrews test results for small sample sizes.
        
        Args:
            column: Column name that was tested.
            trend: Trend that was used in the test.
            n_obs: Number of observations.
            
        Returns:
            Dictionary containing mock Zivot-Andrews test results.
        """
        return {
            'test': 'Zivot-Andrews',
            'column': column,
            'trend': trend,
            'test_statistic': -3.0,  # Mock value suggesting non-stationarity with break
            'p_value': 0.15,  # Mock p-value > alpha suggesting non-stationarity
            'critical_values': {'1%': -5.34, '5%': -4.80, '10%': -4.58},  # Typical critical values
            'n_lags': 1,
            'n_obs': n_obs,
            'is_stationary': False,  # Assuming non-stationarity for safety
            'alpha': self.alpha,
            'break_date': None,  # No break date for mock result
            'mock_result': True  # Flag to indicate this is a mock result
        }
    
    @handle_errors
    def test_dfgls(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform DF-GLS test for unit root.
        
        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant) and
                  'ct' (constant and trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the column is not found or the test fails.
        """
        logger.info(f"Performing DF-GLS test on {column} with trend={trend}")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        col_data = data[column].dropna()
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
            
        # Handle small sample sizes
        n_obs = len(col_data)
        
        try:
            # Handle small sample sizes with mock results
            if n_obs < 8:  # DF-GLS requires at least 8 observations
                logger.warning(f"Sample size ({n_obs}) too small for DF-GLS test. Returning mock results.")
                return self._mock_dfgls_results(column, trend, n_obs)
                
            # Perform DF-GLS test
            dfgls = DFGLS(col_data, trend=trend, lags=max_lags)
            dfgls_result = dfgls.run()
            
            # Extract results
            test_statistic = dfgls_result.stat
            p_value = dfgls_result.pvalue
            critical_values = {
                '1%': dfgls_result.critical_values['1%'],
                '5%': dfgls_result.critical_values['5%'],
                '10%': dfgls_result.critical_values['10%']
            }
            n_lags = dfgls.lags
            
            # Determine if the series is stationary
            is_stationary = p_value < self.alpha
            
            # Create results dictionary
            results = {
                'test': 'DF-GLS',
                'column': column,
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': critical_values,
                'n_lags': n_lags,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
            }
            
            logger.info(f"DF-GLS test results: test_statistic={test_statistic:.4f}, p_value={p_value:.4f}, is_stationary={is_stationary}")
            return results
        except Exception as e:
            logger.error(f"Error performing DF-GLS test: {e}")
            raise YemenAnalysisError(f"Error performing DF-GLS test: {e}")
    
    def _mock_dfgls_results(self, column: str, trend: str, n_obs: int) -> Dict[str, Any]:
        """
        Create mock DF-GLS test results for small sample sizes.
        
        Args:
            column: Column name that was tested.
            trend: Trend that was used in the test.
            n_obs: Number of observations.
            
        Returns:
            Dictionary containing mock DF-GLS test results.
        """
        return {
            'test': 'DF-GLS',
            'column': column,
            'trend': trend,
            'test_statistic': -1.8,  # Mock value suggesting non-stationarity
            'p_value': 0.2,  # Mock p-value > alpha suggesting non-stationarity
            'critical_values': {'1%': -3.5, '5%': -2.9, '10%': -2.6},  # Typical critical values
            'n_lags': 1,
            'n_obs': n_obs,
            'is_stationary': False,  # Assuming non-stationarity for safety
            'alpha': self.alpha,
            'mock_result': True  # Flag to indicate this is a mock result
        }
    
    @handle_errors
    def run_all_tests(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None, include_dfgls: bool = True,
        include_za: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all unit root tests.
        
        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the tests.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            include_dfgls: Whether to include the DF-GLS test.
            include_za: Whether to include the Zivot-Andrews test.
            
        Returns:
            Dictionary mapping test names to test results.
            
        Raises:
            YemenAnalysisError: If the column is not found or any of the tests fail.
        """
        logger.info(f"Running all unit root tests on {column}")
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
        
        # Run tests
        results = {}
        
        # ADF test
        adf_results = self.test_adf(data, column, trend, max_lags)
        results['ADF'] = adf_results
        
        # KPSS test
        kpss_results = self.test_kpss(data, column, trend, max_lags)
        results['KPSS'] = kpss_results
        
        # Phillips-Perron test
        pp_results = self.test_pp(data, column, trend, max_lags)
        results['PP'] = pp_results
        
        # DF-GLS test
        if include_dfgls:
            dfgls_results = self.test_dfgls(data, column, trend, max_lags)
            results['DFGLS'] = dfgls_results
        
        # Zivot-Andrews test
        if include_za:
            za_results = self.test_za(data, column, trend, max_lags)
            results['ZA'] = za_results
        
        # Determine overall stationarity
        is_stationary = (
            results['ADF']['is_stationary'] and
            results['KPSS']['is_stationary'] and
            results['PP']['is_stationary']
        )
        
        if include_dfgls:
            is_stationary = is_stationary and results['DFGLS']['is_stationary']
        
        if include_za:
            is_stationary = is_stationary and results['ZA']['is_stationary']
        
        # Add overall result
        results['overall'] = {
            'is_stationary': is_stationary,
            'alpha': self.alpha,
        }
        
        logger.info(f"Overall stationarity result: {is_stationary}")
        return results
    
    @handle_errors
    def test_order_of_integration(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None, max_diff: int = 2
    ) -> Dict[str, Any]:
        """
        Determine the order of integration of a time series.
        
        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the tests.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            max_diff: Maximum number of differences to consider.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the column is not found or the test fails.
        """
        logger.info(f"Determining order of integration for {column}")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
        
        # Make a copy of the data
        data_copy = data.copy()
        
        # Test the original series
        level_results = self.run_all_tests(
            data_copy, column, trend, max_lags, include_dfgls=False, include_za=False
        )
        
        if level_results['overall']['is_stationary']:
            logger.info(f"{column} is stationary at level (I(0))")
            return {
                'order': 0,
                'level_results': level_results,
                'diff_results': None,
            }
        
        # Test first difference
        data_copy[f'{column}_diff1'] = data_copy[column].diff().dropna()
        diff1_results = self.run_all_tests(
            data_copy, f'{column}_diff1', trend, max_lags, include_dfgls=False, include_za=False
        )
        
        if diff1_results['overall']['is_stationary']:
            logger.info(f"{column} is integrated of order 1 (I(1))")
            return {
                'order': 1,
                'level_results': level_results,
                'diff_results': diff1_results,
            }
        
        # Test second difference if needed
        if max_diff >= 2:
            data_copy[f'{column}_diff2'] = data_copy[f'{column}_diff1'].diff().dropna()
            diff2_results = self.run_all_tests(
                data_copy, f'{column}_diff2', trend, max_lags, include_dfgls=False, include_za=False
            )
            
            if diff2_results['overall']['is_stationary']:
                logger.info(f"{column} is integrated of order 2 (I(2))")
                return {
                    'order': 2,
                    'level_results': level_results,
                    'diff_results': {
                        'diff1': diff1_results,
                        'diff2': diff2_results,
                    },
                }
        
        # If we get here, the series is not stationary after max_diff differences
        logger.warning(f"{column} is not stationary after {max_diff} differences")
        return {
            'order': max_diff + 1,  # Indicates higher order than max_diff
            'level_results': level_results,
            'diff_results': {
                'diff1': diff1_results,
                'diff2': diff2_results if max_diff >= 2 else None,
            },
        }
