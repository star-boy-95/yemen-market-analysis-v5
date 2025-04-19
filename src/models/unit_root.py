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
            # Check if sample size is sufficient
            if n_obs <= 3:  # ADF test requires more observations
                logger.warning(f"Sample size ({n_obs}) too small for ADF test.")
                raise YemenAnalysisError(f"Sample size ({n_obs}) too small for ADF test. Need at least 4 observations.")

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



    @handle_errors
    def test_kpss(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None, robust: bool = True
    ) -> Dict[str, Any]:
        """
        Perform KPSS test for stationarity with robust variance estimation.

        The KPSS test complements the ADF and PP tests by testing the null hypothesis
        of stationarity (rather than non-stationarity). This implementation includes
        robust variance estimation which is particularly important for conflict-affected
        data that may exhibit heteroskedasticity and structural breaks.

        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant) and
                  'ct' (constant and trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            robust: Whether to use robust variance estimation for conflict-affected data.
                   Default is True.

        Returns:
            Dictionary containing the test results, including test statistic,
            p-value, critical values, and stationarity determination.

        Raises:
            YemenAnalysisError: If the column is not found or the test fails.
        """
        logger.info(f"Performing KPSS test on {column} with trend={trend}, robust={robust}")

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
            # Check if sample size is sufficient
            if n_obs <= 3:  # KPSS test requires more observations
                logger.warning(f"Sample size ({n_obs}) too small for KPSS test.")
                raise YemenAnalysisError(f"Sample size ({n_obs}) too small for KPSS test. Need at least 4 observations.")

            # Determine optimal lag length if not specified
            # For conflict-affected data, we use a more conservative approach
            if robust:
                # Use Newey-West automatic bandwidth selection with Bartlett kernel
                # This is more robust to heteroskedasticity and autocorrelation
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                # Calculate optimal lag using Schwert's rule which is more robust
                # for conflict-affected data with potential structural breaks
                optimal_lag = int(12 * (n_obs / 100)**(1/4))
                max_lags = min(optimal_lag, max_lags) if max_lags is not None else optimal_lag
                
                logger.info(f"Using robust lag selection for KPSS test: {max_lags}")
            
            # Perform KPSS test with specified parameters
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
                'robust': robust,
            }

            logger.info(f"KPSS test results: test_statistic={test_statistic:.4f}, p_value={p_value:.4f}, is_stationary={is_stationary}")
            return results
        except Exception as e:
            logger.error(f"Error performing KPSS test: {e}")
            raise YemenAnalysisError(f"Error performing KPSS test: {e}")



    @handle_errors
    def test_pp(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Phillips-Perron test for unit root.

        The Phillips-Perron test is more robust to heteroskedasticity and serial correlation
        in the error terms compared to the ADF test, making it particularly suitable for
        conflict-affected market data which often exhibits these characteristics.

        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant),
                  'ct' (constant and trend), and 'n' (no trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.

        Returns:
            Dictionary containing the test results, including test statistic,
            p-value, critical values, and stationarity determination.

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
            # Check if sample size is sufficient
            if n_obs < 4:  # PP test requires at least 4 observations
                logger.warning(f"Sample size ({n_obs}) too small for Phillips-Perron test.")
                raise YemenAnalysisError(f"Sample size ({n_obs}) too small for Phillips-Perron test. Need at least 4 observations.")

            # Map trend parameter to arch.unitroot format
            # arch uses 'c' for constant, 'ct' for constant and trend, 'n' for no trend
            if trend not in ['c', 'ct', 'n']:
                logger.warning(f"Trend '{trend}' not recognized for Phillips-Perron test. Using 'c' instead.")
                trend = 'c'

            # Use arch.unitroot.PhillipsPerron implementation
            pp_test = PhillipsPerron(col_data, trend=trend, lags=max_lags)
            result = pp_test.summary()
            
            # Extract results
            test_statistic = pp_test.stat
            p_value = pp_test.pvalue
            critical_values = pp_test.critical_values
            n_lags = pp_test.lags
            
            # Convert critical values to the expected format
            critical_values_dict = {
                '1%': critical_values['1%'],
                '5%': critical_values['5%'],
                '10%': critical_values['10%']
            }

            # Determine if the series is stationary
            is_stationary = p_value < self.alpha

            # Create results dictionary
            results = {
                'test': 'Phillips-Perron',
                'column': column,
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': critical_values_dict,
                'n_lags': n_lags,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
            }

            logger.info(f"Phillips-Perron test results: test_statistic={test_statistic:.4f}, p_value={p_value:.4f}, is_stationary={is_stationary}")
            return results
        except Exception as e:
            logger.error(f"Error performing Phillips-Perron test: {e}")
            raise YemenAnalysisError(f"Error performing Phillips-Perron test: {e}")


    @handle_errors
    def test_za(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Zivot-Andrews test for unit root with a structural break.
        
        This test is particularly valuable for conflict-affected market data where
        structural breaks are common due to conflict events, policy changes, or
        market disruptions. The test endogenously determines the most likely break point
        and tests for a unit root in the presence of this break.

        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant),
                  't' (trend), and 'ct' (constant and trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.

        Returns:
            Dictionary containing the test results, including test statistic,
            p-value, critical values, break date, and stationarity determination.

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
            # Check if sample size is sufficient
            # ZA test requires more observations than standard tests due to the break detection
            if n_obs <= 8:  # ZA test requires a minimum sample size
                logger.warning(f"Sample size ({n_obs}) too small for Zivot-Andrews test.")
                raise YemenAnalysisError(f"Sample size ({n_obs}) too small for Zivot-Andrews test. Need at least 9 observations.")

            # Map trend parameter to arch.unitroot format
            # ZivotAndrews uses 'c' for constant, 't' for trend, 'ct' for both
            if trend not in ['c', 't', 'ct']:
                logger.warning(f"Trend '{trend}' not recognized for Zivot-Andrews test. Using 'c' instead.")
                trend = 'c'

            # Use arch.unitroot.ZivotAndrews implementation
            za_test = ZivotAndrews(col_data, trend=trend, lags=max_lags)
            result = za_test.summary()
            
            # Extract results
            test_statistic = za_test.stat
            p_value = za_test.pvalue if hasattr(za_test, 'pvalue') else None
            critical_values = za_test.critical_values
            n_lags = za_test.lags
            
            # Get the break date - convert from index to string if it's a date
            # Handle different attribute names in different versions of the library
            if hasattr(za_test, 'break_idx'):
                break_idx = za_test.break_idx
            elif hasattr(za_test, 'break_index'):
                break_idx = za_test.break_index
            else:
                # If we can't find the break index, use the index of the minimum statistic
                # This is a reasonable approximation
                break_idx = int(np.argmin(za_test.stat_array) if hasattr(za_test, 'stat_array') else 0)
                logger.warning("Could not find break_idx or break_index attribute in ZivotAndrews object. Using approximation.")
            
            if data.index.dtype.kind == 'M':  # Check if index is datetime
                # If the data has a datetime index, get the actual date
                try:
                    # Handle case where break_idx might be out of bounds
                    if 0 <= break_idx < len(data.index):
                        break_date = data.index[break_idx].strftime('%Y-%m-%d')
                    else:
                        break_date = None
                except Exception:
                    break_date = str(break_idx)
            else:
                break_date = str(break_idx)
            
            # Convert critical values to the expected format
            critical_values_dict = {
                '1%': critical_values['1%'],
                '5%': critical_values['5%'],
                '10%': critical_values['10%']
            }

            # Determine if the series is stationary
            # For ZA test, we compare the test statistic with critical values
            # since p-values might not be directly available
            is_stationary = test_statistic < critical_values_dict['5%']
            
            # If p_value is available, use it instead
            if p_value is not None:
                is_stationary = p_value < self.alpha

            # Create results dictionary
            results = {
                'test': 'Zivot-Andrews',
                'column': column,
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': critical_values_dict,
                'n_lags': n_lags,
                'break_date': break_date,
                'break_idx': break_idx,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
            }

            logger.info(f"Zivot-Andrews test results: test_statistic={test_statistic:.4f}, is_stationary={is_stationary}, break_date={break_date}")
            return results
        except Exception as e:
            logger.error(f"Error performing Zivot-Andrews test: {e}")
            raise YemenAnalysisError(f"Error performing Zivot-Andrews test: {e}")


    @handle_errors
    def test_dfgls(
        self, data: pd.DataFrame, column: str = 'price', trend: str = 'c',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform DF-GLS test for unit root.
        
        The DF-GLS (Dickey-Fuller Generalized Least Squares) test has better power
        properties than the standard ADF test, especially in smaller samples and when
        the series has a mean or trend that is unknown. This makes it particularly
        valuable for conflict-affected market data where sample sizes may be limited.

        Args:
            data: DataFrame containing the data.
            column: Column to test.
            trend: Trend to include in the test. Options are 'c' (constant) and
                  'ct' (constant and trend).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.

        Returns:
            Dictionary containing the test results, including test statistic,
            critical values, and stationarity determination.

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
            # Check if sample size is sufficient
            if n_obs < 8:  # DF-GLS requires at least 8 observations
                logger.warning(f"Sample size ({n_obs}) too small for DF-GLS test.")
                raise YemenAnalysisError(f"Sample size ({n_obs}) too small for DF-GLS test. Need at least 8 observations.")

            # Map trend parameter to arch.unitroot format
            # DFGLS uses 'c' for constant, 'ct' for constant and trend
            if trend not in ['c', 'ct']:
                logger.warning(f"Trend '{trend}' not recognized for DF-GLS test. Using 'c' instead.")
                trend = 'c'

            # Use arch.unitroot.DFGLS implementation
            dfgls_test = DFGLS(col_data, trend=trend, lags=max_lags)
            result = dfgls_test.summary()
            
            # Extract results
            test_statistic = dfgls_test.stat
            p_value = dfgls_test.pvalue if hasattr(dfgls_test, 'pvalue') else None
            critical_values = dfgls_test.critical_values
            n_lags = dfgls_test.lags
            
            # Convert critical values to the expected format
            critical_values_dict = {
                '1%': critical_values['1%'],
                '5%': critical_values['5%'],
                '10%': critical_values['10%']
            }

            # Determine if the series is stationary
            # For DF-GLS test, we compare the test statistic with critical values
            # since p-values might not be directly available
            is_stationary = test_statistic < critical_values_dict['5%']
            
            # If p_value is available, use it instead
            if p_value is not None:
                is_stationary = p_value < self.alpha

            # Create results dictionary
            results = {
                'test': 'DF-GLS',
                'column': column,
                'trend': trend,
                'test_statistic': test_statistic,
                'p_value': p_value,
                'critical_values': critical_values_dict,
                'n_lags': n_lags,
                'is_stationary': is_stationary,
                'alpha': self.alpha,
            }

            logger.info(f"DF-GLS test results: test_statistic={test_statistic:.4f}, is_stationary={is_stationary}")
            return results
        except Exception as e:
            logger.error(f"Error performing DF-GLS test: {e}")
            raise YemenAnalysisError(f"Error performing DF-GLS test: {e}")



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
