"""
Data transformation module for Yemen Market Analysis.

This module provides functions for transforming data to improve model performance,
including log transformations, differencing, and handling of outliers.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    Data transformer for Yemen Market Analysis.
    
    This class provides methods for transforming data to improve model performance,
    including log transformations, differencing, and handling of outliers.
    
    Attributes:
        transformations (Dict[str, Dict[str, Any]]): Dictionary of applied transformations.
    """
    
    def __init__(self):
        """Initialize the data transformer."""
        self.transformations = {}
    
    @handle_errors
    def log_transform(
        self, data: pd.DataFrame, columns: List[str], add_constant: float = 0.0
    ) -> pd.DataFrame:
        """
        Apply log transformation to specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to transform.
            add_constant: Constant to add before taking log (for zero or negative values).
            
        Returns:
            DataFrame with log-transformed data.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the transformation fails.
        """
        logger.info(f"Applying log transformation to columns: {columns}")
        
        # Check if columns exist
        for col in columns:
            if col not in data.columns:
                logger.error(f"Column {col} not found in data")
                raise YemenAnalysisError(f"Column {col} not found in data")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        # Apply log transformation
        for col in columns:
            # Check for non-positive values
            min_val = data_copy[col].min()
            if min_val <= 0:
                logger.warning(f"Column {col} contains non-positive values. Adding constant {add_constant + abs(min_val) + 1}")
                constant = add_constant + abs(min_val) + 1
            else:
                constant = add_constant
            
            # Apply transformation
            data_copy[f"{col}_log"] = np.log(data_copy[col] + constant)
            
            # Record transformation
            self.transformations[col] = {
                'type': 'log',
                'constant': constant
            }
        
        logger.info(f"Applied log transformation to {len(columns)} columns")
        return data_copy
    
    @handle_errors
    def difference(
        self, data: pd.DataFrame, columns: List[str], periods: int = 1,
        seasonal: bool = False, seasonal_periods: int = 12
    ) -> pd.DataFrame:
        """
        Apply differencing to specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to transform.
            periods: Number of periods to difference.
            seasonal: Whether to apply seasonal differencing.
            seasonal_periods: Number of periods in a season.
            
        Returns:
            DataFrame with differenced data.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the transformation fails.
        """
        logger.info(f"Applying differencing to columns: {columns}")
        
        # Check if columns exist
        for col in columns:
            if col not in data.columns:
                logger.error(f"Column {col} not found in data")
                raise YemenAnalysisError(f"Column {col} not found in data")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        # Apply differencing
        for col in columns:
            # Apply regular differencing
            data_copy[f"{col}_diff{periods}"] = data_copy[col].diff(periods=periods)
            
            # Apply seasonal differencing if requested
            if seasonal:
                data_copy[f"{col}_sdiff{seasonal_periods}"] = data_copy[col].diff(periods=seasonal_periods)
                
                # Apply both regular and seasonal differencing
                data_copy[f"{col}_diff{periods}_sdiff{seasonal_periods}"] = data_copy[f"{col}_sdiff{seasonal_periods}"].diff(periods=periods)
            
            # Record transformation
            self.transformations[col] = {
                'type': 'difference',
                'periods': periods,
                'seasonal': seasonal,
                'seasonal_periods': seasonal_periods if seasonal else None
            }
        
        logger.info(f"Applied differencing to {len(columns)} columns")
        return data_copy
    
    @handle_errors
    def remove_outliers(
        self, data: pd.DataFrame, columns: List[str], method: str = 'zscore',
        threshold: float = 3.0, replace_with: str = 'median'
    ) -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to transform.
            method: Method to use for outlier detection. Options are 'zscore', 'iqr',
                   and 'modified_zscore'.
            threshold: Threshold for outlier detection.
            replace_with: Method to use for replacing outliers. Options are 'median',
                         'mean', 'mode', 'nearest', and 'interpolate'.
            
        Returns:
            DataFrame with outliers removed.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the transformation fails.
        """
        logger.info(f"Removing outliers from columns: {columns}")
        
        # Check if columns exist
        for col in columns:
            if col not in data.columns:
                logger.error(f"Column {col} not found in data")
                raise YemenAnalysisError(f"Column {col} not found in data")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        # Remove outliers
        for col in columns:
            # Detect outliers
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(data_copy[col].dropna()))
                outliers = (z_scores > threshold)
                outlier_indices = data_copy[col].dropna().index[outliers]
            elif method == 'iqr':
                # IQR method
                q1 = data_copy[col].quantile(0.25)
                q3 = data_copy[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = (data_copy[col] < lower_bound) | (data_copy[col] > upper_bound)
                outlier_indices = data_copy.index[outliers]
            elif method == 'modified_zscore':
                # Modified Z-score method
                median = data_copy[col].median()
                mad = np.median(np.abs(data_copy[col] - median))
                modified_z_scores = 0.6745 * np.abs(data_copy[col] - median) / mad
                outliers = (modified_z_scores > threshold)
                outlier_indices = data_copy.index[outliers]
            else:
                logger.error(f"Invalid outlier detection method: {method}")
                raise YemenAnalysisError(f"Invalid outlier detection method: {method}")
            
            # Replace outliers
            if replace_with == 'median':
                # Replace with median
                data_copy.loc[outlier_indices, col] = data_copy[col].median()
            elif replace_with == 'mean':
                # Replace with mean
                data_copy.loc[outlier_indices, col] = data_copy[col].mean()
            elif replace_with == 'mode':
                # Replace with mode
                data_copy.loc[outlier_indices, col] = data_copy[col].mode()[0]
            elif replace_with == 'nearest':
                # Replace with nearest non-outlier value
                for idx in outlier_indices:
                    # Find nearest non-outlier value
                    non_outlier_indices = data_copy.index[~outliers]
                    nearest_idx = non_outlier_indices[np.abs(non_outlier_indices - idx).argmin()]
                    data_copy.loc[idx, col] = data_copy.loc[nearest_idx, col]
            elif replace_with == 'interpolate':
                # Replace with interpolated value
                data_copy.loc[outlier_indices, col] = np.nan
                data_copy[col] = data_copy[col].interpolate(method='linear')
            else:
                logger.error(f"Invalid outlier replacement method: {replace_with}")
                raise YemenAnalysisError(f"Invalid outlier replacement method: {replace_with}")
            
            # Record transformation
            self.transformations[col] = {
                'type': 'remove_outliers',
                'method': method,
                'threshold': threshold,
                'replace_with': replace_with,
                'n_outliers': len(outlier_indices)
            }
        
        logger.info(f"Removed outliers from {len(columns)} columns")
        return data_copy
    
    @handle_errors
    def normalize(
        self, data: pd.DataFrame, columns: List[str], method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Normalize specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to transform.
            method: Method to use for normalization. Options are 'standard', 'minmax',
                   and 'robust'.
            
        Returns:
            DataFrame with normalized data.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the transformation fails.
        """
        logger.info(f"Normalizing columns: {columns}")
        
        # Check if columns exist
        for col in columns:
            if col not in data.columns:
                logger.error(f"Column {col} not found in data")
                raise YemenAnalysisError(f"Column {col} not found in data")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        # Normalize data
        for col in columns:
            if method == 'standard':
                # Standardization (z-score normalization)
                mean = data_copy[col].mean()
                std = data_copy[col].std()
                data_copy[f"{col}_norm"] = (data_copy[col] - mean) / std
                
                # Record transformation
                self.transformations[col] = {
                    'type': 'normalize',
                    'method': 'standard',
                    'mean': mean,
                    'std': std
                }
            elif method == 'minmax':
                # Min-max normalization
                min_val = data_copy[col].min()
                max_val = data_copy[col].max()
                data_copy[f"{col}_norm"] = (data_copy[col] - min_val) / (max_val - min_val)
                
                # Record transformation
                self.transformations[col] = {
                    'type': 'normalize',
                    'method': 'minmax',
                    'min': min_val,
                    'max': max_val
                }
            elif method == 'robust':
                # Robust normalization
                median = data_copy[col].median()
                q1 = data_copy[col].quantile(0.25)
                q3 = data_copy[col].quantile(0.75)
                iqr = q3 - q1
                data_copy[f"{col}_norm"] = (data_copy[col] - median) / iqr
                
                # Record transformation
                self.transformations[col] = {
                    'type': 'normalize',
                    'method': 'robust',
                    'median': median,
                    'iqr': iqr
                }
            else:
                logger.error(f"Invalid normalization method: {method}")
                raise YemenAnalysisError(f"Invalid normalization method: {method}")
        
        logger.info(f"Normalized {len(columns)} columns")
        return data_copy
    
    @handle_errors
    def create_price_ratios(
        self, data: pd.DataFrame, price_col: str = 'price',
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create price ratios between markets.
        
        Args:
            data: DataFrame containing the data.
            price_col: Column containing price data.
            group_by: Column to group by (e.g., 'commodity').
            
        Returns:
            DataFrame with price ratios.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the transformation fails.
        """
        logger.info(f"Creating price ratios for {price_col}")
        
        # Check if columns exist
        if price_col not in data.columns:
            logger.error(f"Column {price_col} not found in data")
            raise YemenAnalysisError(f"Column {price_col} not found in data")
        
        if group_by is not None and group_by not in data.columns:
            logger.error(f"Column {group_by} not found in data")
            raise YemenAnalysisError(f"Column {group_by} not found in data")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        try:
            # Get unique markets
            if 'market' in data_copy.columns:
                markets = data_copy['market'].unique()
            elif 'market_name' in data_copy.columns:
                markets = data_copy['market_name'].unique()
            else:
                logger.error("No market column found in data")
                raise YemenAnalysisError("No market column found in data")
            
            # Create price ratios
            for i, market_i in enumerate(markets):
                for j, market_j in enumerate(markets):
                    if i < j:  # Only create ratios for unique pairs
                        # Filter data for each market
                        if 'market' in data_copy.columns:
                            market_i_data = data_copy[data_copy['market'] == market_i]
                            market_j_data = data_copy[data_copy['market'] == market_j]
                        else:
                            market_i_data = data_copy[data_copy['market_name'] == market_i]
                            market_j_data = data_copy[data_copy['market_name'] == market_j]
                        
                        # Group by commodity if requested
                        if group_by is not None:
                            for group in data_copy[group_by].unique():
                                market_i_group = market_i_data[market_i_data[group_by] == group]
                                market_j_group = market_j_data[market_j_data[group_by] == group]
                                
                                # Create ratio
                                ratio_name = f"{market_i}_{market_j}_{group}_ratio"
                                # TODO: Implement ratio calculation for grouped data
                        else:
                            # Create ratio
                            ratio_name = f"{market_i}_{market_j}_ratio"
                            # TODO: Implement ratio calculation
            
            logger.info(f"Created price ratios for {len(markets)} markets")
            return data_copy
        except Exception as e:
            logger.error(f"Error creating price ratios: {e}")
            raise YemenAnalysisError(f"Error creating price ratios: {e}")
    
    @handle_errors
    def create_exchange_rate_adjusted_prices(
        self, data: pd.DataFrame, price_col: str = 'price',
        exchange_rate_col: str = 'exchange_rate'
    ) -> pd.DataFrame:
        """
        Create exchange rate adjusted prices.
        
        Args:
            data: DataFrame containing the data.
            price_col: Column containing price data.
            exchange_rate_col: Column containing exchange rate data.
            
        Returns:
            DataFrame with exchange rate adjusted prices.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the transformation fails.
        """
        logger.info(f"Creating exchange rate adjusted prices for {price_col}")
        
        # Check if columns exist
        if price_col not in data.columns:
            logger.error(f"Column {price_col} not found in data")
            raise YemenAnalysisError(f"Column {price_col} not found in data")
        
        if exchange_rate_col not in data.columns:
            logger.error(f"Column {exchange_rate_col} not found in data")
            raise YemenAnalysisError(f"Column {exchange_rate_col} not found in data")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        # Create exchange rate adjusted prices
        data_copy[f"{price_col}_adjusted"] = data_copy[price_col] / data_copy[exchange_rate_col]
        
        # Record transformation
        self.transformations[price_col] = {
            'type': 'exchange_rate_adjusted',
            'exchange_rate_col': exchange_rate_col
        }
        
        logger.info(f"Created exchange rate adjusted prices for {price_col}")
        return data_copy
    
    @handle_errors
    def create_conflict_adjusted_prices(
        self, data: pd.DataFrame, price_col: str = 'price',
        conflict_col: str = 'conflict_intensity_normalized'
    ) -> pd.DataFrame:
        """
        Create conflict adjusted prices.
        
        Args:
            data: DataFrame containing the data.
            price_col: Column containing price data.
            conflict_col: Column containing conflict intensity data.
            
        Returns:
            DataFrame with conflict adjusted prices.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the transformation fails.
        """
        logger.info(f"Creating conflict adjusted prices for {price_col}")
        
        # Check if columns exist
        if price_col not in data.columns:
            logger.error(f"Column {price_col} not found in data")
            raise YemenAnalysisError(f"Column {price_col} not found in data")
        
        if conflict_col not in data.columns:
            logger.error(f"Column {conflict_col} not found in data")
            raise YemenAnalysisError(f"Column {conflict_col} not found in data")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        # Create conflict adjusted prices
        # Higher conflict intensity should increase the effective price
        data_copy[f"{price_col}_conflict_adj"] = data_copy[price_col] * (1 + data_copy[conflict_col])
        
        # Record transformation
        self.transformations[price_col] = {
            'type': 'conflict_adjusted',
            'conflict_col': conflict_col
        }
        
        logger.info(f"Created conflict adjusted prices for {price_col}")
        return data_copy
    
    @handle_errors
    def analyze_stationarity(
        self, data: pd.DataFrame, column: str, alpha: float = 0.05,
        plot: bool = True, figsize: Tuple[int, int] = (12, 8)
    ) -> Dict[str, Any]:
        """
        Analyze stationarity of a time series.
        
        Args:
            data: DataFrame containing the data.
            column: Column to analyze.
            alpha: Significance level for tests.
            plot: Whether to create plots.
            figsize: Figure size for plots.
            
        Returns:
            Dictionary containing stationarity analysis results.
            
        Raises:
            YemenAnalysisError: If the column is not found or the analysis fails.
        """
        logger.info(f"Analyzing stationarity of {column}")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        series = data[column].dropna()
        
        try:
            # Perform ADF test
            adf_result = adfuller(series)
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_critical_values = adf_result[4]
            adf_stationary = adf_pvalue < alpha
            
            # Perform KPSS test
            kpss_result = kpss(series)
            kpss_statistic = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_critical_values = kpss_result[3]
            kpss_stationary = kpss_pvalue > alpha
            
            # Create plots if requested
            if plot:
                fig, axes = plt.subplots(3, 1, figsize=figsize)
                
                # Plot time series
                axes[0].plot(series)
                axes[0].set_title(f"Time Series: {column}")
                axes[0].set_xlabel("Time")
                axes[0].set_ylabel("Value")
                
                # Plot ACF
                plot_acf(series, ax=axes[1], lags=40)
                axes[1].set_title(f"Autocorrelation Function (ACF): {column}")
                
                # Plot PACF
                plot_pacf(series, ax=axes[2], lags=40)
                axes[2].set_title(f"Partial Autocorrelation Function (PACF): {column}")
                
                plt.tight_layout()
                plt.show()
            
            # Compile results
            results = {
                'adf_test': {
                    'statistic': adf_statistic,
                    'pvalue': adf_pvalue,
                    'critical_values': adf_critical_values,
                    'stationary': adf_stationary
                },
                'kpss_test': {
                    'statistic': kpss_statistic,
                    'pvalue': kpss_pvalue,
                    'critical_values': kpss_critical_values,
                    'stationary': kpss_stationary
                },
                'conclusion': {
                    'stationary': adf_stationary and kpss_stationary,
                    'trend_stationary': not adf_stationary and kpss_stationary,
                    'difference_stationary': adf_stationary and not kpss_stationary,
                    'non_stationary': not adf_stationary and not kpss_stationary
                }
            }
            
            logger.info(f"Stationarity analysis results for {column}: ADF p-value={adf_pvalue:.4f}, KPSS p-value={kpss_pvalue:.4f}")
            return results
        except Exception as e:
            logger.error(f"Error analyzing stationarity: {e}")
            raise YemenAnalysisError(f"Error analyzing stationarity: {e}")
    
    @handle_errors
    def recommend_transformations(
        self, data: pd.DataFrame, column: str, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Recommend transformations for a time series.
        
        Args:
            data: DataFrame containing the data.
            column: Column to analyze.
            alpha: Significance level for tests.
            
        Returns:
            Dictionary containing recommended transformations.
            
        Raises:
            YemenAnalysisError: If the column is not found or the analysis fails.
        """
        logger.info(f"Recommending transformations for {column}")
        
        # Check if column exists
        if column not in data.columns:
            logger.error(f"Column {column} not found in data")
            raise YemenAnalysisError(f"Column {column} not found in data")
        
        # Get column data
        series = data[column].dropna()
        
        try:
            # Analyze original series
            original_results = self.analyze_stationarity(data, column, alpha, plot=False)
            
            # Initialize recommendations
            recommendations = {
                'original': original_results,
                'transformations': []
            }
            
            # Check if original series is stationary
            if original_results['conclusion']['stationary']:
                recommendations['transformations'].append({
                    'type': 'none',
                    'reason': 'Series is already stationary'
                })
                return recommendations
            
            # Try log transformation
            if series.min() > 0:
                log_data = self.log_transform(data, [column])
                log_results = self.analyze_stationarity(log_data, f"{column}_log", alpha, plot=False)
                
                recommendations['transformations'].append({
                    'type': 'log',
                    'results': log_results,
                    'recommended': log_results['conclusion']['stationary']
                })
            
            # Try first difference
            diff_data = self.difference(data, [column], periods=1)
            diff_results = self.analyze_stationarity(diff_data, f"{column}_diff1", alpha, plot=False)
            
            recommendations['transformations'].append({
                'type': 'difference',
                'periods': 1,
                'results': diff_results,
                'recommended': diff_results['conclusion']['stationary']
            })
            
            # Try seasonal difference if data has enough observations
            if len(series) >= 24:  # At least 2 years of monthly data
                sdiff_data = self.difference(data, [column], periods=1, seasonal=True, seasonal_periods=12)
                sdiff_results = self.analyze_stationarity(sdiff_data, f"{column}_diff1_sdiff12", alpha, plot=False)
                
                recommendations['transformations'].append({
                    'type': 'seasonal_difference',
                    'periods': 1,
                    'seasonal_periods': 12,
                    'results': sdiff_results,
                    'recommended': sdiff_results['conclusion']['stationary']
                })
            
            # Try log + first difference
            if series.min() > 0:
                log_diff_data = self.difference(log_data, [f"{column}_log"], periods=1)
                log_diff_results = self.analyze_stationarity(log_diff_data, f"{column}_log_diff1", alpha, plot=False)
                
                recommendations['transformations'].append({
                    'type': 'log_difference',
                    'periods': 1,
                    'results': log_diff_results,
                    'recommended': log_diff_results['conclusion']['stationary']
                })
            
            # Find best transformation
            best_transformation = None
            best_pvalue = 1.0
            
            for transformation in recommendations['transformations']:
                if transformation.get('recommended', False):
                    pvalue = transformation['results']['adf_test']['pvalue']
                    if pvalue < best_pvalue:
                        best_pvalue = pvalue
                        best_transformation = transformation
            
            recommendations['best_transformation'] = best_transformation
            
            logger.info(f"Recommended transformations for {column}: {best_transformation['type'] if best_transformation else 'none'}")
            return recommendations
        except Exception as e:
            logger.error(f"Error recommending transformations: {e}")
            raise YemenAnalysisError(f"Error recommending transformations: {e}")
    
    @handle_errors
    def apply_recommended_transformations(
        self, data: pd.DataFrame, columns: List[str], alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Apply recommended transformations to specified columns.
        
        Args:
            data: DataFrame containing the data.
            columns: Columns to transform.
            alpha: Significance level for tests.
            
        Returns:
            DataFrame with transformed data.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the transformation fails.
        """
        logger.info(f"Applying recommended transformations to columns: {columns}")
        
        # Check if columns exist
        for col in columns:
            if col not in data.columns:
                logger.error(f"Column {col} not found in data")
                raise YemenAnalysisError(f"Column {col} not found in data")
        
        # Create a copy of the data
        data_copy = data.copy()
        
        # Apply recommended transformations
        for col in columns:
            # Get recommendations
            recommendations = self.recommend_transformations(data_copy, col, alpha)
            
            # Apply best transformation
            best_transformation = recommendations.get('best_transformation')
            
            if best_transformation is None:
                logger.warning(f"No recommended transformation for {col}")
                continue
            
            if best_transformation['type'] == 'none':
                logger.info(f"No transformation needed for {col}")
                continue
            elif best_transformation['type'] == 'log':
                data_copy = self.log_transform(data_copy, [col])
            elif best_transformation['type'] == 'difference':
                data_copy = self.difference(data_copy, [col], periods=best_transformation['periods'])
            elif best_transformation['type'] == 'seasonal_difference':
                data_copy = self.difference(data_copy, [col], periods=best_transformation['periods'],
                                          seasonal=True, seasonal_periods=best_transformation['seasonal_periods'])
            elif best_transformation['type'] == 'log_difference':
                data_copy = self.log_transform(data_copy, [col])
                data_copy = self.difference(data_copy, [f"{col}_log"], periods=best_transformation['periods'])
            else:
                logger.warning(f"Unknown transformation type: {best_transformation['type']}")
        
        logger.info(f"Applied recommended transformations to {len(columns)} columns")
        return data_copy
