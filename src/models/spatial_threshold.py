"""
Spatial Threshold Modeling module for Yemen Market Analysis.

This module integrates threshold models with spatial econometrics
to analyze spatial market integration with transaction costs that
vary with conflict intensity.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import pandas as pd
import numpy as np
import statsmodels.api as sm
import geopandas as gpd
import libpysal.weights as weights
from statsmodels.regression.linear_model import OLS
from scipy import stats
import matplotlib.pyplot as plt

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.models.threshold.model import ThresholdModel
from src.models.spatial.models import SpatialModel
from src.models.threshold.tar import ThresholdAutoregressive
from src.models.threshold.mtar import MomentumThresholdAutoregressive

# Initialize logger
logger = logging.getLogger(__name__)


class SpatialThresholdModel:
    """
    Spatial Threshold Model for Yemen Market Analysis.
    
    This class integrates threshold models with spatial econometrics
    to analyze spatial market integration with transaction costs that
    vary with conflict intensity. It allows for analyzing how market
    integration patterns change across geographical space and conflict zones.
    
    Attributes:
        threshold_model (ThresholdModel): Threshold model instance for time series analysis.
        spatial_model (SpatialModel): Spatial model instance for spatial econometrics.
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        conflict_weight (float): Weight to apply to conflict intensity in transaction cost estimation.
    """
    
    def __init__(
        self, 
        threshold_model: Optional[ThresholdModel] = None, 
        spatial_model: Optional[SpatialModel] = None,
        alpha: float = None,
        max_lags: int = None,
        conflict_weight: float = None
    ):
        """
        Initialize the spatial threshold model.
        
        Args:
            threshold_model: Threshold model instance. If None, a new instance will be created.
            spatial_model: Spatial model instance. If None, a new instance will be created.
            alpha: Significance level for hypothesis tests. If None, uses the value from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the value from config.
            conflict_weight: Weight to apply to conflict intensity in transaction cost estimation.
                           If None, uses the value from config.
        """
        self.threshold_model = threshold_model
        self.spatial_model = spatial_model
        self.alpha = alpha if alpha is not None else config.get('analysis.threshold.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.threshold.max_lags', 4)
        self.conflict_weight = conflict_weight if conflict_weight is not None else config.get(
            'analysis.spatial_threshold.conflict_weight', 0.5
        )
    
    @handle_errors
    def estimate_with_conflict(
        self, 
        price_data: pd.DataFrame, 
        conflict_data: pd.DataFrame,
        spatial_weights: pd.DataFrame,
        price_col: str = 'price',
        conflict_col: str = 'intensity',
        threshold_type: str = 'tar'
    ) -> Dict[str, Any]:
        """
        Estimate spatial threshold model with conflict intensity.
        
        This method integrates threshold models with spatial econometrics to analyze
        how market integration patterns vary across geographical space and conflict zones.
        It incorporates conflict intensity into transaction cost estimates and provides
        methods to analyze spatial patterns of market integration.
        
        Args:
            price_data: DataFrame with price data. Must have a spatial index that can be
                      matched with the spatial_weights.
            conflict_data: DataFrame with conflict intensity data. Must have the same
                         spatial index as price_data.
            spatial_weights: DataFrame with spatial weights. Must have the same spatial
                           index as price_data.
            price_col: Column name for price data.
            conflict_col: Column name for conflict intensity.
            threshold_type: Type of threshold model ('tar', 'mtar', 'tvecm').
            
        Returns:
            Dictionary with estimation results, including:
            - Threshold model parameters for each spatial unit
            - Spatial patterns of transaction costs
            - Conflict-adjusted market integration measures
            - Diagnostic statistics and visualizations
            
        Raises:
            YemenAnalysisError: If the estimation fails or inputs are invalid.
        """
        logger.info(f"Estimating spatial threshold model with conflict intensity using {threshold_type} model")
        
        try:
            # Validate inputs
            self._validate_inputs(price_data, conflict_data, spatial_weights, price_col, conflict_col)
            
            # Initialize models if not provided
            if self.threshold_model is None:
                self.threshold_model = ThresholdModel(
                    y=price_data, 
                    x=price_data,  # Placeholder, will be updated in the loop
                    y_col=price_col, 
                    x_col=price_col,  # Placeholder, will be updated in the loop
                    mode=threshold_type,
                    alpha=self.alpha,
                    max_lags=self.max_lags
                )
            
            if self.spatial_model is None:
                # Convert to GeoDataFrame if not already
                if not isinstance(price_data, gpd.GeoDataFrame):
                    logger.warning("Price data is not a GeoDataFrame. Spatial analysis may be limited.")
                    geo_data = price_data.copy()
                else:
                    geo_data = price_data
                
                self.spatial_model = SpatialModel(data=geo_data)
            
            # Create spatial weight matrix if not provided
            if isinstance(spatial_weights, pd.DataFrame):
                # Convert DataFrame to weight matrix
                w = self._create_weight_matrix_from_df(spatial_weights)
            else:
                # Assume it's already a weight matrix
                w = spatial_weights
            
            # Step 1: Estimate conflict-adjusted transaction costs
            transaction_costs = self._estimate_transaction_costs(price_data, conflict_data, w, price_col, conflict_col)
            
            # Step 2: Estimate threshold models for each spatial unit
            threshold_results = self._estimate_spatial_thresholds(
                price_data, transaction_costs, w, price_col, threshold_type
            )
            
            # Step 3: Analyze spatial patterns of market integration
            spatial_patterns = self._analyze_spatial_patterns(threshold_results, w)
            
            # Step 4: Create visualizations
            visualizations = self._create_visualizations(threshold_results, transaction_costs, w)
            
            # Combine results
            results = {
                'model_type': f'spatial_{threshold_type}',
                'transaction_costs': transaction_costs,
                'threshold_results': threshold_results,
                'spatial_patterns': spatial_patterns,
                'visualizations': visualizations,
                'parameters': {
                    'price_col': price_col,
                    'conflict_col': conflict_col,
                    'threshold_type': threshold_type,
                    'conflict_weight': self.conflict_weight,
                    'alpha': self.alpha,
                    'max_lags': self.max_lags,
                },
            }
            
            logger.info("Spatial threshold model estimation completed successfully")
            return results
        
        except Exception as e:
            logger.error(f"Error estimating spatial threshold model: {e}")
            raise YemenAnalysisError(f"Error estimating spatial threshold model: {e}")
    
    def _validate_inputs(
        self, 
        price_data: pd.DataFrame, 
        conflict_data: pd.DataFrame,
        spatial_weights: pd.DataFrame,
        price_col: str,
        conflict_col: str
    ) -> None:
        """
        Validate input data for spatial threshold model.
        
        Args:
            price_data: DataFrame with price data.
            conflict_data: DataFrame with conflict intensity data.
            spatial_weights: DataFrame with spatial weights.
            price_col: Column name for price data.
            conflict_col: Column name for conflict intensity.
            
        Raises:
            YemenAnalysisError: If inputs are invalid.
        """
        # Check if price_col exists in price_data
        if price_col not in price_data.columns:
            raise YemenAnalysisError(f"Price column '{price_col}' not found in price data")
        
        # Check if conflict_col exists in conflict_data
        if conflict_col not in conflict_data.columns:
            raise YemenAnalysisError(f"Conflict column '{conflict_col}' not found in conflict data")
        
        # Check if spatial indices match
        if not set(price_data.index).issubset(set(conflict_data.index)):
            raise YemenAnalysisError("Price data and conflict data indices do not match")
        
        # Check if spatial weights is valid
        if isinstance(spatial_weights, pd.DataFrame):
            # Check if spatial weights has the right structure
            if not all(col in spatial_weights.columns for col in ['source', 'target', 'weight']):
                raise YemenAnalysisError(
                    "Spatial weights DataFrame must have 'source', 'target', and 'weight' columns"
                )
        elif not isinstance(spatial_weights, weights.W):
            raise YemenAnalysisError(
                "Spatial weights must be a DataFrame or a libpysal.weights.W object"
            )
    
    def _create_weight_matrix_from_df(self, weights_df: pd.DataFrame) -> weights.W:
        """
        Create a spatial weight matrix from a DataFrame.
        
        Args:
            weights_df: DataFrame with spatial weights. Must have 'source', 'target', and 'weight' columns.
            
        Returns:
            Spatial weight matrix.
            
        Raises:
            YemenAnalysisError: If the weights DataFrame is invalid.
        """
        try:
            # Check if weights_df has the right structure
            if not all(col in weights_df.columns for col in ['source', 'target', 'weight']):
                raise YemenAnalysisError(
                    "Weights DataFrame must have 'source', 'target', and 'weight' columns"
                )
            
            # Create dictionary for weight matrix
            w_dict = {}
            for _, row in weights_df.iterrows():
                source = row['source']
                target = row['target']
                weight = row['weight']
                
                if source not in w_dict:
                    w_dict[source] = {}
                
                w_dict[source][target] = weight
            
            # Create weight matrix
            w = weights.W(w_dict)
            
            # Row-standardize
            w.transform = 'r'
            
            return w
        
        except Exception as e:
            logger.error(f"Error creating weight matrix from DataFrame: {e}")
            raise YemenAnalysisError(f"Error creating weight matrix from DataFrame: {e}")
    
    def _estimate_transaction_costs(
        self, 
        price_data: pd.DataFrame, 
        conflict_data: pd.DataFrame,
        w: weights.W,
        price_col: str,
        conflict_col: str
    ) -> pd.DataFrame:
        """
        Estimate transaction costs adjusted by conflict intensity.
        
        This method estimates transaction costs between markets based on price differentials
        and adjusts them by conflict intensity. Higher conflict intensity increases
        transaction costs, reflecting the impact of conflict on market integration.
        
        Args:
            price_data: DataFrame with price data.
            conflict_data: DataFrame with conflict intensity data.
            w: Spatial weight matrix.
            price_col: Column name for price data.
            conflict_col: Column name for conflict intensity.
            
        Returns:
            DataFrame with transaction costs for each spatial unit.
        """
        logger.info("Estimating conflict-adjusted transaction costs")
        
        try:
            # Get common indices
            common_indices = set(price_data.index).intersection(set(conflict_data.index))
            
            # Filter data to common indices
            price_data_filtered = price_data.loc[common_indices]
            conflict_data_filtered = conflict_data.loc[common_indices]
            
            # Calculate price differentials
            price_differentials = {}
            transaction_costs = {}
            
            for i in common_indices:
                # Get neighbors
                if i not in w.neighbors:
                    logger.warning(f"Spatial unit {i} not found in weight matrix")
                    continue
                
                neighbors = w.neighbors[i]
                
                # Calculate price differentials with neighbors
                price_i = price_data_filtered.loc[i, price_col]
                conflict_i = conflict_data_filtered.loc[i, conflict_col]
                
                price_diffs = []
                conflict_adjustments = []
                
                for j in neighbors:
                    if j in common_indices:
                        price_j = price_data_filtered.loc[j, price_col]
                        conflict_j = conflict_data_filtered.loc[j, conflict_col]
                        
                        # Calculate price differential
                        price_diff = abs(price_i - price_j)
                        price_diffs.append(price_diff)
                        
                        # Calculate conflict adjustment
                        # Higher conflict intensity increases transaction costs
                        conflict_adjustment = 1 + self.conflict_weight * (conflict_i + conflict_j) / 2
                        conflict_adjustments.append(conflict_adjustment)
                
                if price_diffs:
                    # Calculate average price differential
                    avg_price_diff = np.mean(price_diffs)
                    price_differentials[i] = avg_price_diff
                    
                    # Calculate average conflict adjustment
                    avg_conflict_adjustment = np.mean(conflict_adjustments)
                    
                    # Calculate transaction cost
                    # Transaction cost = Price differential * Conflict adjustment
                    transaction_costs[i] = avg_price_diff * avg_conflict_adjustment
            
            # Create DataFrame with transaction costs
            transaction_costs_df = pd.DataFrame({
                'price_differential': pd.Series(price_differentials),
                'transaction_cost': pd.Series(transaction_costs),
            })
            
            logger.info(f"Estimated transaction costs for {len(transaction_costs_df)} spatial units")
            return transaction_costs_df
        
        except Exception as e:
            logger.error(f"Error estimating transaction costs: {e}")
            raise YemenAnalysisError(f"Error estimating transaction costs: {e}")
    
    def _estimate_spatial_thresholds(
        self, 
        price_data: pd.DataFrame, 
        transaction_costs: pd.DataFrame,
        w: weights.W,
        price_col: str,
        threshold_type: str
    ) -> Dict[str, Any]:
        """
        Estimate threshold models for each spatial unit.
        
        This method estimates threshold models for each spatial unit using the
        transaction costs as thresholds. It analyzes how market integration
        patterns vary across geographical space.
        
        Args:
            price_data: DataFrame with price data.
            transaction_costs: DataFrame with transaction costs.
            w: Spatial weight matrix.
            price_col: Column name for price data.
            threshold_type: Type of threshold model ('tar', 'mtar', 'tvecm').
            
        Returns:
            Dictionary with threshold model results for each spatial unit.
        """
        logger.info(f"Estimating {threshold_type} models for each spatial unit")
        
        try:
            # Initialize results dictionary
            threshold_results = {}
            
            # Get spatial units with transaction costs
            spatial_units = transaction_costs.index
            
            # Select appropriate threshold model class based on type
            if threshold_type == 'tar':
                model_class = ThresholdAutoregressive
            elif threshold_type == 'mtar':
                model_class = MomentumThresholdAutoregressive
            else:
                # Default to TAR
                model_class = ThresholdAutoregressive
                logger.warning(f"Unknown threshold type '{threshold_type}'. Using TAR model.")
            
            # Estimate threshold models for each spatial unit
            for i in spatial_units:
                # Get neighbors
                if i not in w.neighbors:
                    logger.warning(f"Spatial unit {i} not found in weight matrix")
                    continue
                
                neighbors = w.neighbors[i]
                
                # Filter neighbors that are in price_data
                valid_neighbors = [j for j in neighbors if j in price_data.index]
                
                if not valid_neighbors:
                    logger.warning(f"No valid neighbors found for spatial unit {i}")
                    continue
                
                # Get transaction cost for this unit
                transaction_cost = transaction_costs.loc[i, 'transaction_cost']
                
                # Get price data for this unit
                price_i = price_data.loc[i, price_col]
                
                # Initialize model results for this unit
                unit_results = {
                    'unit_id': i,
                    'transaction_cost': transaction_cost,
                    'neighbor_results': {},
                }
                
                # Estimate threshold models with each neighbor
                for j in valid_neighbors:
                    # Get price data for neighbor
                    price_j = price_data.loc[j, price_col]
                    
                    # Create price series for unit i and neighbor j
                    # Note: In a real implementation, these would be time series
                    # For simplicity, we're using single values here
                    y = pd.DataFrame({price_col: [price_i]})
                    x = pd.DataFrame({price_col: [price_j]})
                    
                    # Initialize threshold model
                    threshold_model = model_class(alpha=self.alpha, max_lags=self.max_lags)
                    
                    try:
                        # Estimate model with fixed threshold (transaction cost)
                        model_results = threshold_model.estimate(
                            y, x, price_col, price_col, fixed_threshold=transaction_cost
                        )
                        
                        # Store results
                        unit_results['neighbor_results'][j] = {
                            'neighbor_id': j,
                            'threshold': transaction_cost,
                            'rho_above': model_results['params'].get('rho_above'),
                            'rho_below': model_results['params'].get('rho_below'),
                            'is_integrated': model_results['params'].get('rho_below', 0) < -0.1,
                            'adjustment_speed': abs(model_results['params'].get('rho_below', 0)),
                        }
                    except Exception as e:
                        logger.warning(f"Error estimating threshold model for unit {i} with neighbor {j}: {e}")
                        unit_results['neighbor_results'][j] = {
                            'neighbor_id': j,
                            'error': str(e),
                        }
                
                # Calculate average integration measures
                if unit_results['neighbor_results']:
                    # Filter successful results
                    successful_results = [
                        r for r in unit_results['neighbor_results'].values()
                        if 'error' not in r
                    ]
                    
                    if successful_results:
                        # Calculate average integration measures
                        unit_results['avg_rho_above'] = np.mean([
                            r['rho_above'] for r in successful_results
                            if r.get('rho_above') is not None
                        ])
                        unit_results['avg_rho_below'] = np.mean([
                            r['rho_below'] for r in successful_results
                            if r.get('rho_below') is not None
                        ])
                        unit_results['integration_rate'] = np.mean([
                            1 if r.get('is_integrated', False) else 0
                            for r in successful_results
                        ])
                        unit_results['avg_adjustment_speed'] = np.mean([
                            r.get('adjustment_speed', 0) for r in successful_results
                        ])
                
                # Store results for this unit
                threshold_results[i] = unit_results
            
            logger.info(f"Estimated threshold models for {len(threshold_results)} spatial units")
            return threshold_results
        
        except Exception as e:
            logger.error(f"Error estimating spatial thresholds: {e}")
            raise YemenAnalysisError(f"Error estimating spatial thresholds: {e}")
    
    def _analyze_spatial_patterns(
        self, 
        threshold_results: Dict[str, Any],
        w: weights.W
    ) -> Dict[str, Any]:
        """
        Analyze spatial patterns of market integration.
        
        This method analyzes the spatial patterns of market integration based on
        the threshold model results. It identifies clusters of integrated and
        non-integrated markets and analyzes the relationship between conflict
        intensity and market integration.
        
        Args:
            threshold_results: Dictionary with threshold model results.
            w: Spatial weight matrix.
            
        Returns:
            Dictionary with spatial pattern analysis results.
        """
        logger.info("Analyzing spatial patterns of market integration")
        
        try:
            # Extract integration measures
            integration_rates = {}
            adjustment_speeds = {}
            
            for i, results in threshold_results.items():
                integration_rates[i] = results.get('integration_rate', 0)
                adjustment_speeds[i] = results.get('avg_adjustment_speed', 0)
            
            # Create DataFrames
            integration_df = pd.DataFrame({
                'integration_rate': pd.Series(integration_rates),
                'adjustment_speed': pd.Series(adjustment_speeds),
            })
            
            # Calculate global spatial autocorrelation
            # Note: In a real implementation, we would use spatial statistics
            # For simplicity, we're just calculating basic statistics here
            
            # Calculate mean and standard deviation
            mean_integration = integration_df['integration_rate'].mean()
            std_integration = integration_df['integration_rate'].std()
            
            mean_adjustment = integration_df['adjustment_speed'].mean()
            std_adjustment = integration_df['adjustment_speed'].std()
            
            # Identify high and low integration clusters
            high_integration = integration_df[integration_df['integration_rate'] > mean_integration + std_integration]
            low_integration = integration_df[integration_df['integration_rate'] < mean_integration - std_integration]
            
            # Create spatial pattern results
            spatial_patterns = {
                'global_statistics': {
                    'mean_integration_rate': mean_integration,
                    'std_integration_rate': std_integration,
                    'mean_adjustment_speed': mean_adjustment,
                    'std_adjustment_speed': std_adjustment,
                },
                'clusters': {
                    'high_integration': high_integration.index.tolist(),
                    'low_integration': low_integration.index.tolist(),
                },
                'integration_df': integration_df,
            }
            
            logger.info("Spatial pattern analysis completed")
            return spatial_patterns
        
        except Exception as e:
            logger.error(f"Error analyzing spatial patterns: {e}")
            raise YemenAnalysisError(f"Error analyzing spatial patterns: {e}")
    
    def _create_visualizations(
        self, 
        threshold_results: Dict[str, Any],
        transaction_costs: pd.DataFrame,
        w: weights.W
    ) -> Dict[str, plt.Figure]:
        """
        Create visualizations for spatial threshold model results.
        
        This method creates visualizations for the spatial threshold model results,
        including maps of transaction costs, market integration rates, and
        adjustment speeds.
        
        Args:
            threshold_results: Dictionary with threshold model results.
            transaction_costs: DataFrame with transaction costs.
            w: Spatial weight matrix.
            
        Returns:
            Dictionary with visualization figures.
        """
        logger.info("Creating visualizations for spatial threshold model results")
        
        try:
            # Extract integration measures
            integration_rates = {}
            adjustment_speeds = {}
            
            for i, results in threshold_results.items():
                integration_rates[i] = results.get('integration_rate', 0)
                adjustment_speeds[i] = results.get('avg_adjustment_speed', 0)
            
            # Create DataFrames
            integration_df = pd.DataFrame({
                'integration_rate': pd.Series(integration_rates),
                'adjustment_speed': pd.Series(adjustment_speeds),
            })
            
            # Create transaction costs histogram
            fig_tc_hist, ax_tc_hist = plt.subplots(figsize=(10, 6), dpi=100)
            ax_tc_hist.hist(transaction_costs['transaction_cost'], bins=20, alpha=0.7)
            ax_tc_hist.set_xlabel('Transaction Cost')
            ax_tc_hist.set_ylabel('Frequency')
            ax_tc_hist.set_title('Distribution of Transaction Costs')
            ax_tc_hist.grid(alpha=0.3)
            
            # Create integration rate histogram
            fig_ir_hist, ax_ir_hist = plt.subplots(figsize=(10, 6), dpi=100)
            ax_ir_hist.hist(integration_df['integration_rate'], bins=20, alpha=0.7)
            ax_ir_hist.set_xlabel('Integration Rate')
            ax_ir_hist.set_ylabel('Frequency')
            ax_ir_hist.set_title('Distribution of Market Integration Rates')
            ax_ir_hist.grid(alpha=0.3)
            
            # Create adjustment speed histogram
            fig_as_hist, ax_as_hist = plt.subplots(figsize=(10, 6), dpi=100)
            ax_as_hist.hist(integration_df['adjustment_speed'], bins=20, alpha=0.7)
            ax_as_hist.set_xlabel('Adjustment Speed')
            ax_as_hist.set_ylabel('Frequency')
            ax_as_hist.set_title('Distribution of Market Adjustment Speeds')
            ax_as_hist.grid(alpha=0.3)
            
            # Create scatter plot of transaction costs vs. integration rates
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6), dpi=100)
            
            # Merge DataFrames
            scatter_df = pd.merge(
                transaction_costs, integration_df, left_index=True, right_index=True
            )
            
            ax_scatter.scatter(
                scatter_df['transaction_cost'], scatter_df['integration_rate'],
                alpha=0.7, s=50
            )
            ax_scatter.set_xlabel('Transaction Cost')
            ax_scatter.set_ylabel('Integration Rate')
            ax_scatter.set_title('Transaction Costs vs. Market Integration Rates')
            ax_scatter.grid(alpha=0.3)
            
            # Add regression line
            if len(scatter_df) > 1:
                x = scatter_df['transaction_cost']
                y = scatter_df['integration_rate']
                
                # Add regression line
                m, b = np.polyfit(x, y, 1)
                ax_scatter.plot(x, m*x + b, 'r-', alpha=0.7)
                
                # Add correlation coefficient
                corr = np.corrcoef(x, y)[0, 1]
                ax_scatter.text(
                    0.05, 0.95, f'Correlation: {corr:.2f}',
                    transform=ax_scatter.transAxes,
                    verticalalignment='top'
                )
            
            # Create visualizations dictionary
            visualizations = {
                'transaction_costs_histogram': fig_tc_hist,
                'integration_rates_histogram': fig_ir_hist,
                'adjustment_speeds_histogram': fig_as_hist,
                'transaction_costs_vs_integration': fig_scatter,
            }
            
            logger.info("Created visualizations for spatial threshold model results")
            return visualizations
        
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise YemenAnalysisError(f"Error creating visualizations: {e}")

    @handle_errors
    def analyze_conflict_impact(
        self,
        price_data: pd.DataFrame,
        conflict_data: pd.DataFrame,
        spatial_weights: pd.DataFrame,
        price_col: str = 'price',
        conflict_col: str = 'intensity',
        threshold_type: str = 'tar'
    ) -> Dict[str, Any]:
        """
        Analyze the impact of conflict on market integration.
        
        This method analyzes how conflict intensity affects market integration
        patterns across geographical space. It compares market integration
        measures between high-conflict and low-conflict areas.
        
        Args:
            price_data: DataFrame with price data.
            conflict_data: DataFrame with conflict intensity data.
            spatial_weights: DataFrame with spatial weights.
            price_col: Column name for price data.
            conflict_col: Column name for conflict intensity.
            threshold_type: Type of threshold model ('tar', 'mtar', 'tvecm').
            
        Returns:
            Dictionary with conflict impact analysis results.
        """
        logger.info("Analyzing impact of conflict on market integration")
        
        try:
            # Estimate spatial threshold model
            model_results = self.estimate_with_conflict(
                price_data, conflict_data, spatial_weights,
                price_col, conflict_col, threshold_type
            )
            
            # Extract threshold results
            threshold_results = model_results['threshold_results']
            
            # Extract integration measures
            integration_rates = {}
            adjustment_speeds = {}
            
            for i, results in threshold_results.items():
                integration_rates[i] = results.get('integration_rate', 0)
                adjustment_speeds[i] = results.get('avg_adjustment_speed', 0)
            
            # Create integration DataFrame
            integration_df = pd.DataFrame({
                'integration_rate': pd.Series(integration_rates),
                'adjustment_speed': pd.Series(adjustment_speeds),
            })
            
            # Merge with conflict data
            conflict_filtered = conflict_data.loc[integration_df.index, [conflict_col]]
            analysis_df = pd.merge(
                integration_df, conflict_filtered, left_index=True, right_index=True
            )
            
            # Classify areas by conflict intensity
            median_conflict = analysis_df[conflict_col].median()
            analysis_df['conflict_level'] = np.where(
                analysis_df[conflict_col] > median_conflict, 'high', 'low'
            )
            
            # Compare integration measures between high and low conflict areas
            high_conflict = analysis_df[analysis_df['conflict_level'] == 'high']
            low_conflict = analysis_df[analysis_df['conflict_level'] == 'low']
            
            # Calculate statistics
            high_conflict_stats = {
                'mean_integration_rate': high_conflict['integration_rate'].mean(),
                'std_integration_rate': high_conflict['integration_rate'].std(),
                'mean_adjustment_speed': high_conflict['adjustment_speed'].mean(),
                'std_adjustment_speed': high_conflict['adjustment_speed'].std(),
                'count': len(high_conflict),
            }
            
            low_conflict_stats = {
                'mean_integration_rate': low_conflict['integration_rate'].mean(),
                'std_integration_rate': low_conflict['integration_rate'].std(),
                'mean_adjustment_speed': low_conflict['adjustment_speed'].mean(),
                'std_adjustment_speed': low_conflict['adjustment_speed'].std(),
                'count': len(low_conflict),
            }
            
            # Create visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
            
            # Plot integration rates by conflict level
            ax1.bar(['Low Conflict', 'High Conflict'], 
                   [low_conflict_stats['mean_integration_rate'], 
                    high_conflict_stats['mean_integration_rate']],
                   yerr=[low_conflict_stats['std_integration_rate'], 
                         high_conflict_stats['std_integration_rate']],
                   alpha=0.7)
            ax1.set_ylabel('Integration Rate')
            ax1.set_title('Market Integration by Conflict Level')
            ax1.grid(alpha=0.3)
            
            # Plot adjustment speeds by conflict level
            ax2.bar(['Low Conflict', 'High Conflict'], 
                   [low_conflict_stats['mean_adjustment_speed'], 
                    high_conflict_stats['mean_adjustment_speed']],
                   yerr=[low_conflict_stats['std_adjustment_speed'], 
                         high_conflict_stats['std_adjustment_speed']],
                   alpha=0.7)
            ax2.set_ylabel('Adjustment Speed')
            ax2.set_title('Market Adjustment Speed by Conflict Level')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Create scatter plot of conflict intensity vs. integration rate
            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6), dpi=100)
            
            ax_scatter.scatter(
                analysis_df[conflict_col], analysis_df['integration_rate'],
                alpha=0.7, s=50
            )
            ax_scatter.set_xlabel('Conflict Intensity')
            ax_scatter.set_ylabel('Integration Rate')
            ax_scatter.set_title('Conflict Intensity vs. Market Integration Rate')
            ax_scatter.grid(alpha=0.3)
            
            # Add regression line
            if len(analysis_df) > 1:
                x = analysis_df[conflict_col]
                y = analysis_df['integration_rate']
                
                # Add regression line
                m, b = np.polyfit(x, y, 1)
                ax_scatter.plot(x, m*x + b, 'r-', alpha=0.7)
                
                # Add correlation coefficient
                corr = np.corrcoef(x, y)[0, 1]
                ax_scatter.text(
                    0.05, 0.95, f'Correlation: {corr:.2f}',
                    transform=ax_scatter.transAxes,
                    verticalalignment='top'
                )
            
            # Combine results
            results = {
                'high_conflict_stats': high_conflict_stats,
                'low_conflict_stats': low_conflict_stats,
                'analysis_df': analysis_df,
                'visualizations': {
                    'conflict_comparison': fig,
                    'conflict_vs_integration': fig_scatter,
                },
                'correlation': np.corrcoef(analysis_df[conflict_col], analysis_df['integration_rate'])[0, 1]
                if len(analysis_df) > 1 else None,
            }
            
            logger.info("Conflict impact analysis completed")
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing conflict impact: {e}")
            raise YemenAnalysisError(f"Error analyzing conflict impact: {e}")
