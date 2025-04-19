"""
Gregory-Hansen cointegration test module for Yemen Market Analysis.

This module provides the GregoryHansenTester class for testing cointegration
with structural breaks using the Gregory-Hansen procedure.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class GregoryHansenTester:
    """
    Gregory-Hansen cointegration tester for Yemen Market Analysis.

    This class provides methods for testing cointegration with structural breaks
    using the Gregory-Hansen procedure.

    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
    """

    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the Gregory-Hansen tester.

        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.cointegration.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.cointegration.max_lags', 4)

        # Critical values for Gregory-Hansen test
        self.critical_values = {
            'c': {
                '1%': -5.13,
                '5%': -4.61,
                '10%': -4.34
            },
            'ct': {
                '1%': -5.45,
                '5%': -4.99,
                '10%': -4.72
            },
            'cshift': {
                '1%': -5.47,
                '5%': -4.95,
                '10%': -4.68
            }
        }

    @handle_errors
    def test(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        model: str = 'c', max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Gregory-Hansen test for cointegration with structural breaks.

        The Gregory-Hansen test extends the Engle-Granger test to allow for a structural
        break in the cointegrating relationship.

        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            model: Model type. Options are 'c' (level shift), 'ct' (level shift with trend),
                  and 'cshift' (regime shift).
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.

        Returns:
            Dictionary containing the test results.

        Raises:
            YemenAnalysisError: If the columns are not found or the test fails.
        """
        logger.info(f"Performing Gregory-Hansen test with model={model}")

        # Check if columns exist
        if y_col not in y.columns:
            logger.error(f"Column {y_col} not found in y data")
            raise YemenAnalysisError(f"Column {y_col} not found in y data")

        if x_col not in x.columns:
            logger.error(f"Column {x_col} not found in x data")
            raise YemenAnalysisError(f"Column {x_col} not found in x data")

        # Get column data
        y_data = y[y_col].dropna()
        x_data = x[x_col].dropna()

        # Ensure the series have the same length
        common_index = y_data.index.intersection(x_data.index)
        y_data = y_data.loc[common_index]
        x_data = x_data.loc[common_index]

        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags

        # Handle small sample sizes
        n_obs = len(common_index)
        n_trend = 1 # minimum for level shift model
        if model == 'ct':
            n_trend = 2  # level shift with trend
        elif model == 'cshift':
            n_trend = 2  # regime shift (two separate models)

        # For the Gregory-Hansen test, we need a minimum number of observations
        # The minimum should be enough to have a few observations before and after a break
        min_required_obs = 2 * (max_lags + 1 + n_trend) + 3  # +3 because we need to define break points

        if n_obs < min_required_obs:
            logger.warning(f"Sample size ({n_obs}) too small for Gregory-Hansen test. Needs at least {min_required_obs}.")
            raise YemenAnalysisError(f"Sample size ({n_obs}) too small for Gregory-Hansen test. Need at least {min_required_obs} observations.")

        try:
            # Implement Gregory-Hansen test
            # This is a simplified implementation that searches for a break point
            # and performs the Engle-Granger test at each potential break point

            n = len(y_data)
            min_size = int(0.15 * n)  # Trimming parameter

            # Initialize variables to store results
            min_adf = float('inf')
            break_idx = None
            break_date = None
            best_reg = None

            # Get early stopping threshold from config
            early_stop_threshold = config.get('analysis.gh.early_stop_threshold', -10.0)

            # Loop through potential break points
            for i in range(min_size, n - min_size):
                # Create dummy variable for break
                dummy = np.zeros(n)
                dummy[i:] = 1

                # Create interaction term for regime shift model
                if model == 'cshift':
                    x_with_break = pd.DataFrame({
                        'x': x_data,
                        'dummy': dummy,
                        'x_dummy': x_data * dummy
                    })
                    # Estimate model with regime shift
                    reg = OLS(y_data, sm.add_constant(x_with_break)).fit()
                elif model == 'ct':
                    # Create trend variable
                    trend = np.arange(n)
                    x_with_break = pd.DataFrame({
                        'x': x_data,
                        'trend': trend,
                        'dummy': dummy
                    })
                    # Estimate model with level shift and trend
                    reg = OLS(y_data, sm.add_constant(x_with_break)).fit()
                else:  # model == 'c'
                    x_with_break = pd.DataFrame({
                        'x': x_data,
                        'dummy': dummy
                    })
                    # Estimate model with level shift
                    reg = OLS(y_data, sm.add_constant(x_with_break)).fit()

                # Get residuals and perform ADF test
                resid = reg.resid
                adf_result = adfuller(resid, maxlag=max_lags, regression='c', autolag=None)

                # Update minimum ADF statistic
                if adf_result[0] < min_adf:
                    min_adf = adf_result[0]
                    break_idx = i
                    break_date = y_data.index[i]
                    best_reg = reg

                # Early stopping if we find a very significant result
                if min_adf < early_stop_threshold:
                    logger.info(f"Early stopping GH test at break point {i} with ADF statistic {min_adf:.4f}")
                    break

            # Determine if the series are cointegrated
            critical_value_5pct = self.critical_values[model]['5%']
            is_cointegrated = min_adf < critical_value_5pct

            # Create results dictionary
            results = {
                'test': 'Gregory-Hansen',
                'model': model,
                'test_statistic': min_adf,
                'critical_values': self.critical_values[model],
                'is_cointegrated': is_cointegrated,
                'break_index': break_idx,
                'break_date': break_date,
                'alpha': self.alpha,
                'cointegrating_vector': dict(zip(best_reg.params.index, best_reg.params)),
                'residuals': best_reg.resid,
                'n_obs': n,
            }

            logger.info(f"Gregory-Hansen test results: test_statistic={min_adf:.4f}, is_cointegrated={is_cointegrated}, break_date={break_date}")
            return results
        except Exception as e:
            logger.error(f"Error performing Gregory-Hansen test: {e}")
            raise YemenAnalysisError(f"Error performing Gregory-Hansen test: {e}")


