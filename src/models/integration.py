"""
Integrated market analysis module combining time series and spatial econometrics.

This module provides a unified framework for analyzing market integration in
conflict-affected Yemen by connecting threshold cointegration analysis with
spatial econometric models.
"""
import pandas as pd
import numpy as np
import logging
import geopandas as gpd
from typing import Dict, Any, Optional, Union, List, Tuple
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import warnings

from yemen_market_integration.utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_geodataframe, validate_dataframe, validate_time_series, raise_if_invalid,
    validate_multiple_test_results,
    
    # Performance
    timer, m3_optimized, memory_usage_decorator, tiered_cache, 
    parallelize_dataframe, optimize_dataframe, configure_system_for_m3_performance,
    
    # Plotting utilities
    set_plotting_style, format_date_axis, plot_time_series, 
    plot_dual_axis, save_plot, plot_yemen_market_integration,
    
    # Configuration
    config
)

from src.models.cointegration import CointegrationTester
from src.models.threshold import ThresholdCointegration, calculate_asymmetric_adjustment
from src.models.spatial import SpatialEconometrics, market_integration_index, calculate_market_accessibility
from src.models.diagnostics import ModelDiagnostics
from src.models.panel_cointegration import PanelCointegrationTester
from src.models.network import MarketNetworkAnalysis
from src.utils.multiple_testing import apply_multiple_testing_correction, correct_threshold_cointegration_tests
import networkx as nx

# Initialize module logger
logger = logging.getLogger(__name__)


def _calculate_integration_score(
    cointegrated: bool,
    asymmetric: bool,
    half_life_below: Optional[float] = None,
    half_life_above: Optional[float] = None
) -> float:
    """
    Calculate a simplified market integration score.
    
    Parameters
    ----------
    cointegrated : bool
        Whether markets are cointegrated
    asymmetric : bool
        Whether adjustment is asymmetric
    half_life_below : float, optional
        Half-life of adjustment below threshold
    half_life_above : float, optional
        Half-life of adjustment above threshold
    
    Returns
    -------
    float
        Integration score between 0 and 1
    """
    # Base score from cointegration
    if not cointegrated:
        return 0.0
    
    score = 0.5  # Start with 0.5 if cointegrated
    
    # Add score based on adjustment speeds
    if half_life_below is not None and half_life_above is not None:
        # Calculate average half-life
        avg_half_life = (half_life_below + half_life_above) / 2
        
        # Lower half-life means better integration
        if avg_half_life <= 1:
            speed_score = 0.5  # Very fast adjustment
        elif avg_half_life <= 3:
            speed_score = 0.4  # Fast adjustment
        elif avg_half_life <= 5:
            speed_score = 0.3  # Moderate adjustment
        elif avg_half_life <= 10:
            speed_score = 0.2  # Slow adjustment
        else:
            speed_score = 0.1  # Very slow adjustment
        
        score += speed_score
    elif not asymmetric:
        # Simple adjustment with no asymmetry
        score += 0.3
    
    return min(score, 1.0)  # Cap at 1.0

# Get configuration values
DEFAULT_CONFLICT_COL = config.get('analysis.integration.conflict_column', 'conflict_intensity_normalized')
DEFAULT_PRICE_COL = config.get('analysis.integration.price_column', 'price')
DEFAULT_REGION_COL = config.get('analysis.integration.region_column', 'exchange_rate_regime')
DEFAULT_MARKET_ID_COL = config.get('analysis.integration.market_id_column', 'market_id')
DEFAULT_DATE_COL = config.get('analysis.integration.date_column', 'date')


@dataclass
class IntegrationResults:
    """
    Container for integrated market analysis results.
    
    This class stores and organizes the results from multiple analysis components,
    providing a unified structure for interpretation and visualization.
    
    Attributes
    ----------
    time_series_results : dict
        Results from time series analysis (cointegration, threshold)
    spatial_results : dict
        Results from spatial analysis
    integrated_metrics : dict
        Combined metrics across analysis types
    market_pairs : list
        List of analyzed market pairs
    visualization_hooks : dict
        Hooks for creating visualizations
    meta : dict
        Metadata about the analysis (dates, markets, parameters)
    """
    time_series_results: Dict[str, Any]
    spatial_results: Dict[str, Any]
    integrated_metrics: Dict[str, Any]
    market_pairs: List[Tuple[str, str]]
    visualization_hooks: Dict[str, Any]
    meta: Dict[str, Any]
    network_results: Optional[Dict[str, Any]] = None
    panel_results: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation of integration results."""
        n_pairs = len(self.market_pairs)
        n_markets = len(set([m for pair in self.market_pairs for m in pair]))
        
        result = (
            f"Yemen Market Integration Analysis\n"
            f"-------------------------------\n"
            f"Analyzed {n_pairs} market pairs across {n_markets} markets\n"
            f"Time period: {self.meta.get('start_date', 'N/A')} to {self.meta.get('end_date', 'N/A')}\n"
            f"Integration summary: {self.integrated_metrics.get('integration_level', 'N/A')}\n"
            f"Spatial autocorrelation: {self.spatial_results.get('moran_result', {}).get('significant_autocorrelation', False)}\n"
            f"Threshold effects: {self.integrated_metrics.get('threshold_effects_summary', 'N/A')}\n"
        )
        
        # Add network results if available
        if self.network_results:
            n_communities = self.network_results.get('n_communities', 0)
            result += f"Network communities: {n_communities}\n"
        
        # Add panel results if available
        if self.panel_results:
            panel_cointegrated = self.panel_results.get('panel_cointegrated', False)
            result += f"Panel cointegration: {'Yes' if panel_cointegrated else 'No'}\n"
        
        return result
        
    def get_market_pair_results(self, market1: str, market2: str) -> Dict[str, Any]:
        """
        Get results for a specific market pair.
        
        Parameters
        ----------
        market1 : str
            First market name
        market2 : str
            Second market name
            
        Returns
        -------
        dict
            Results for the specified market pair
        """
        # Check if pair exists in either order
        key = f"{market1}_{market2}"
        reverse_key = f"{market2}_{market1}"
        
        if key in self.time_series_results:
            return self.time_series_results[key]
        elif reverse_key in self.time_series_results:
            return self.time_series_results[reverse_key]
        else:
            raise KeyError(f"No results found for market pair {market1}-{market2}")
    
    def get_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of key integration metrics.
        
        Returns
        -------
        pd.DataFrame
            Summary table with key metrics for all market pairs
        """
        # Collect metrics for each market pair
        rows = []
        
        for market1, market2 in self.market_pairs:
            pair_key = f"{market1}_{market2}"
            
            # Get time series results if available
            ts_results = self.time_series_results.get(pair_key, {})
            cointegration = ts_results.get('cointegration', {})
            threshold = ts_results.get('threshold', {})
            asymm_adj = ts_results.get('asymmetric_adjustment', {})
            
            # Create summary row
            row = {
                'market1': market1,
                'market2': market2,
                'cointegrated': cointegration.get('cointegrated', False),
                'threshold': threshold.get('threshold', np.nan),
                'adjustment_below': asymm_adj.get('adjustment_below', np.nan),
                'adjustment_above': asymm_adj.get('adjustment_above', np.nan),
                'half_life_below': asymm_adj.get('half_life_below', np.nan),
                'half_life_above': asymm_adj.get('half_life_above', np.nan),
                'asymmetric': asymm_adj.get('asymmetric', False),
                'distance_km': ts_results.get('meta', {}).get('distance_km', np.nan),
                'conflict_barrier': ts_results.get('meta', {}).get('conflict_barrier', np.nan),
                'exchange_rate_diff': ts_results.get('meta', {}).get('exchange_rate_diff', np.nan),
                'integration_level': ts_results.get('meta', {}).get('integration_level', 'Unknown')
            }
            
            # Add multiple testing correction info if available
            if 'multiple_testing_correction' in ts_results.get('cointegration', {}):
                row['adjusted_p_value'] = ts_results['cointegration'].get('corrected_p_value', np.nan)
                row['significant_adjusted'] = ts_results['cointegration'].get('significant', False)
                
            # Add network info if available
            if self.network_results and 'communities' in self.network_results:
                communities = self.network_results['communities']
                if market1 in communities and market2 in communities:
                    row['community1'] = communities.get(market1)
                    row['community2'] = communities.get(market2)
                    row['same_community'] = communities.get(market1) == communities.get(market2)
            
            rows.append(row)
        
        # Create DataFrame
        summary_df = pd.DataFrame(rows)
        return summary_df
    
    def plot_integration_overview(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create overview plot of market integration results.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        matplotlib.figure.Figure
            Integration overview figure
        """
        # Set plotting style
        set_plotting_style()
        
        # Create figure with 2x3 subplots (added a column for network visualization)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Yemen Market Integration Analysis Overview", fontsize=16)
        
        # Plot 1: Market pairs map
        ax = axes[0, 0]
        if 'market_gdf' in self.visualization_hooks and 'edges_gdf' in self.visualization_hooks:
            try:
                market_gdf = self.visualization_hooks['market_gdf']
                edges_gdf = self.visualization_hooks['edges_gdf']
                
                # Plot base map
                market_gdf.plot(ax=ax, color='skyblue', alpha=0.7)
                
                # Plot edges colored by integration level
                edges_gdf.plot(ax=ax, column='integration_level_num', cmap='RdYlGn', 
                              linewidth=2, legend=True)
                
                ax.set_title("Market Integration Network")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
            except Exception as e:
                logger.warning(f"Could not create market pair map: {e}")
                ax.text(0.5, 0.5, "Map Not Available", 
                      ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, "Map Not Available", 
                  ha='center', va='center', fontsize=12)
        
        # Plot 2: Integration metrics by distance
        ax = axes[0, 1]
        summary_df = self.get_summary_table()
        
        try:
            # Calculate integration score (0-1)
            summary_df['integration_score'] = summary_df.apply(
                lambda x: _calculate_integration_score(
                    x['cointegrated'], 
                    x['asymmetric'], 
                    x['half_life_below'], 
                    x['half_life_above']
                ),
                axis=1
            )
            
            # Scatter plot of integration score vs distance
            scatter = ax.scatter(
                summary_df['distance_km'], 
                summary_df['integration_score'],
                c=summary_df['conflict_barrier'], 
                cmap='YlOrRd',
                alpha=0.7, 
                s=50
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Conflict Barrier Intensity')
            
            # Add best fit line
            if len(summary_df) > 1:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    summary_df['distance_km'].dropna(), 
                    summary_df['integration_score'].dropna()
                )
                x = np.array([min(summary_df['distance_km'].dropna()), 
                             max(summary_df['distance_km'].dropna())])
                ax.plot(x, intercept + slope * x, 'r', alpha=0.3)
                
                # Add R-squared
                ax.text(0.05, 0.95, f"R² = {r_value**2:.2f}", 
                      transform=ax.transAxes, ha='left', va='top')
            
            ax.set_title("Market Integration vs. Distance")
            ax.set_xlabel("Distance (km)")
            ax.set_ylabel("Integration Score (0-1)")
        except Exception as e:
            logger.warning(f"Could not create integration vs distance plot: {e}")
            ax.text(0.5, 0.5, "Plot Not Available", 
                  ha='center', va='center', fontsize=12)
        
        # Plot 3: Network visualization or community structure (New)
        ax = axes[0, 2]
        if self.network_results and 'communities' in self.network_results:
            try:
                # Plot network community sizes
                communities = self.network_results['communities']
                community_sizes = self.network_results.get('community_sizes', {})
                
                # Create bar chart of community sizes
                community_df = pd.DataFrame({
                    'Community': list(community_sizes.keys()),
                    'Size': list(community_sizes.values())
                }).sort_values('Community')
                
                community_df.plot.bar(x='Community', y='Size', ax=ax, color='skyblue')
                ax.set_title("Market Community Sizes")
                ax.set_xlabel("Community ID")
                ax.set_ylabel("Number of Markets")
                
            except Exception as e:
                logger.warning(f"Could not create network visualization: {e}")
                ax.text(0.5, 0.5, "Network Visualization Not Available", 
                      ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, "Network Analysis Not Available", 
                  ha='center', va='center', fontsize=12)
        
        # Plot 4: Histogram of adjustment speeds
        ax = axes[1, 0]
        try:
            # Filter for pairs with adjustment data
            adj_df = summary_df.dropna(subset=['adjustment_below', 'adjustment_above'])
            
            if not adj_df.empty:
                ax.hist(
                    [adj_df['adjustment_below'].abs(), adj_df['adjustment_above'].abs()],
                    bins=10, alpha=0.7, label=['Below Threshold', 'Above Threshold']
                )
                ax.set_title("Distribution of Adjustment Speeds")
                ax.set_xlabel("Absolute Adjustment Speed")
                ax.set_ylabel("Count")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No Adjustment Data Available", 
                      ha='center', va='center', fontsize=12)
        except Exception as e:
            logger.warning(f"Could not create adjustment speeds histogram: {e}")
            ax.text(0.5, 0.5, "Plot Not Available", 
                  ha='center', va='center', fontsize=12)
        
        # Plot 5: Spatial autocorrelation 
        ax = axes[1, 1]
        try:
            if 'moran_local_result' in self.spatial_results:
                local_moran_df = self.spatial_results['moran_local_result']
                
                # Count significant clusters
                cluster_counts = local_moran_df['cluster_type'].value_counts()
                
                # Exclude 'not_significant'
                if 'not_significant' in cluster_counts:
                    cluster_counts = cluster_counts.drop('not_significant')
                
                if not cluster_counts.empty:
                    cluster_counts.plot.bar(ax=ax, color='skyblue')
                    ax.set_title("Spatial Price Clusters")
                    ax.set_xlabel("Cluster Type")
                    ax.set_ylabel("Count")
                else:
                    ax.text(0.5, 0.5, "No Significant Clusters Found", 
                          ha='center', va='center', fontsize=12)
            else:
                ax.text(0.5, 0.5, "Local Moran's I Results Not Available", 
                      ha='center', va='center', fontsize=12)
        except Exception as e:
            logger.warning(f"Could not create spatial clusters plot: {e}")
            ax.text(0.5, 0.5, "Plot Not Available", 
                  ha='center', va='center', fontsize=12)
        
        # Plot 6: Panel cointegration results or central markets (New)
        ax = axes[1, 2]
        if self.network_results and 'centrality_measures' in self.network_results:
            try:
                # Plot top central markets
                centrality = self.network_results['centrality_measures']
                
                # Get top 10 markets by centrality
                top_markets = sorted(
                    centrality.keys(),
                    key=lambda m: centrality[m]['average'],
                    reverse=True
                )[:10]
                
                top_centrality = [centrality[m]['average'] for m in top_markets]
                
                # Create bar chart
                y_pos = np.arange(len(top_markets))
                ax.barh(y_pos, top_centrality, align='center', color='skyblue')
                ax.set_yticks(y_pos, labels=top_markets)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_title("Top 10 Central Markets")
                ax.set_xlabel("Centrality Score")
                
            except Exception as e:
                logger.warning(f"Could not create centrality plot: {e}")
                ax.text(0.5, 0.5, "Centrality Plot Not Available", 
                      ha='center', va='center', fontsize=12)
        elif self.panel_results:
            try:
                # Plot panel cointegration test statistics
                test_results = []
                for test_name, test_info in self.panel_results.items():
                    if isinstance(test_info, dict) and 'statistic' in test_info:
                        test_results.append({
                            'Test': test_name,
                            'Statistic': test_info['statistic'],
                            'Significant': test_info.get('significant', False)
                        })
                
                if test_results:
                    test_df = pd.DataFrame(test_results)
                    colors = ['green' if sig else 'red' for sig in test_df['Significant']]
                    
                    test_df.plot.bar(x='Test', y='Statistic', ax=ax, color=colors)
                    ax.set_title("Panel Cointegration Test Statistics")
                    ax.set_xlabel("Test")
                    ax.set_ylabel("Statistic")
                else:
                    ax.text(0.5, 0.5, "No Panel Test Results Available", 
                          ha='center', va='center', fontsize=12)
            except Exception as e:
                logger.warning(f"Could not create panel cointegration plot: {e}")
                ax.text(0.5, 0.5, "Panel Results Plot Not Available", 
                      ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, "Panel/Network Analysis Not Available", 
                  ha='center', va='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save if requested
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved integration overview plot to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save plot: {e}")
        
        return fig


class MarketIntegrationAnalysis:
    """
    Integrated market analysis for conflict-affected Yemen.
    
    This class provides a unified framework for analyzing market integration
    by combining time series methods (cointegration, threshold models) with
    spatial econometric approaches to capture both temporal price relationships
    and spatial connectivity patterns.
    """
    
    def __init__(
        self, 
        price_data: pd.DataFrame,
        spatial_data: Optional[gpd.GeoDataFrame] = None,
        conflict_data: Optional[pd.DataFrame] = None,
        price_col: str = DEFAULT_PRICE_COL,
        market_id_col: str = DEFAULT_MARKET_ID_COL,
        date_col: str = DEFAULT_DATE_COL,
        conflict_col: str = DEFAULT_CONFLICT_COL,
        region_col: str = DEFAULT_REGION_COL,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the integrated market analysis.
        
        Parameters
        ----------
        price_data : pd.DataFrame
            DataFrame with market price time series
        spatial_data : gpd.GeoDataFrame, optional
            GeoDataFrame with market locations and attributes
        conflict_data : pd.DataFrame, optional
            DataFrame with conflict intensity data
        price_col : str, optional
            Column containing price data
        market_id_col : str, optional
            Column identifying markets
        date_col : str, optional
            Column containing dates
        conflict_col : str, optional
            Column containing conflict intensity
        region_col : str, optional
            Column identifying exchange rate regions
        max_workers : int, optional
            Maximum number of parallel workers to use
        """
        # Configure system for optimal performance using M3 utilities
        try:
            configure_system_for_m3_performance()
        except:
            # Fall back to original configuration
            configure_system_for_performance()
        
        # Store column names
        self.price_col = price_col
        self.market_id_col = market_id_col
        self.date_col = date_col
        self.conflict_col = conflict_col
        self.region_col = region_col
        
        # Set max workers
        self.max_workers = max_workers
        if max_workers is None:
            import multiprocessing as mp
            if hasattr(mp, 'cpu_count'):
                self.max_workers = max(1, mp.cpu_count() - 1)
            else:
                self.max_workers = 4  # Default if cpu_count not available
        
        # Validate price data
        self._validate_price_data(price_data)
        self.price_data = optimize_dataframe(price_data)
        
        # Process spatial data if provided
        if spatial_data is not None:
            self._validate_spatial_data(spatial_data)
            self.spatial_data = spatial_data
            self.has_spatial = True
        else:
            self.has_spatial = False
            self.spatial_data = None
        
        # Process conflict data if provided
        if conflict_data is not None:
            self._validate_conflict_data(conflict_data)
            self.conflict_data = optimize_dataframe(conflict_data)
            self.has_conflict = True
        else:
            self.has_conflict = False
            self.conflict_data = None
        
        # Initialize components
        self.cointegration_tester = CointegrationTester()
        self.spatial_model = None
        self.network_model = None
        self.panel_model = None
        
        # Store market info
        self.markets = sorted(price_data[market_id_col].unique())
        self.n_markets = len(self.markets)
        
        # Store time range
        self.start_date = price_data[date_col].min()
        self.end_date = price_data[date_col].max()
        self.n_periods = price_data[date_col].nunique()
        
        # Prepare market pairs based on data availability
        self._prepare_market_pairs()
        
        # Log initialization
        logger.info(
            f"Initialized MarketIntegrationAnalysis with {self.n_markets} markets, "
            f"{len(self.market_pairs)} market pairs, and {self.n_periods} time periods"
        )
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _validate_price_data(self, price_data: pd.DataFrame) -> None:
        """
        Validate price data for integrated analysis.
        
        Parameters
        ----------
        price_data : pd.DataFrame
            Price data to validate
        """
        valid, errors = validate_dataframe(
            price_data,
            required_columns=[self.price_col, self.market_id_col, self.date_col],
            min_rows=30
        )
        raise_if_invalid(valid, errors, "Invalid price data for integrated analysis")
        
        # Check if date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(price_data[self.date_col]):
            try:
                # Try to convert to datetime
                pd.to_datetime(price_data[self.date_col])
            except:
                raise ValidationError(f"Column {self.date_col} must be convertible to datetime")
        
        # Check if we have at least 2 markets
        n_markets = price_data[self.market_id_col].nunique()
        if n_markets < 2:
            raise ValidationError(f"Need at least 2 markets, got {n_markets}")
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _validate_spatial_data(self, spatial_data: gpd.GeoDataFrame) -> None:
        """
        Validate spatial data for integrated analysis.
        
        Parameters
        ----------
        spatial_data : gpd.GeoDataFrame
            Spatial data to validate
        """
        valid, errors = validate_geodataframe(
            spatial_data,
            required_columns=[self.market_id_col],
            min_rows=1
        )
        raise_if_invalid(valid, errors, "Invalid spatial data for integrated analysis")
        
        # Check market ID correspondence
        price_markets = set(self.price_data[self.market_id_col].unique())
        spatial_markets = set(spatial_data[self.market_id_col].unique())
        
        missing_markets = price_markets - spatial_markets
        if missing_markets:
            logger.warning(
                f"The following markets in price data are missing from spatial data: "
                f"{', '.join(str(m) for m in missing_markets)}"
            )
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def _validate_conflict_data(self, conflict_data: pd.DataFrame) -> None:
        """
        Validate conflict data for integrated analysis.
        
        Parameters
        ----------
        conflict_data : pd.DataFrame
            Conflict data to validate
        """
        valid, errors = validate_dataframe(
            conflict_data,
            required_columns=[self.conflict_col, self.market_id_col],
            min_rows=1
        )
        raise_if_invalid(valid, errors, "Invalid conflict data for integrated analysis")
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _prepare_market_pairs(self) -> None:
        """
        Prepare market pairs for analysis based on data availability.
        
        This method identifies valid market pairs with sufficient
        price data for time series analysis. Optimized for M3 Pro
        using parallel processing for validation.
        """
        # Get unique markets
        markets = self.markets
        
        # Create all possible market pairs
        all_pairs = [(markets[i], markets[j]) 
                     for i in range(len(markets)) 
                     for j in range(i+1, len(markets))]
        
        # Filter for pairs with sufficient data using parallel processing
        valid_pairs = []
        min_observations = 30  # Econometric best practice for time series analysis
        
        # Use parallel processing for validation
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for market1, market2 in all_pairs:
                futures.append(
                    executor.submit(
                        self._validate_market_pair,
                        market1, 
                        market2,
                        min_observations
                    )
                )
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    valid_pairs.append(result)
        
        self.market_pairs = valid_pairs
        logger.info(f"Prepared {len(valid_pairs)} valid market pairs out of {len(all_pairs)} possible pairs")
    
    def _validate_market_pair(
        self, 
        market1: str, 
        market2: str, 
        min_observations: int
    ) -> Optional[Tuple[str, str]]:
        """
        Validate a market pair has sufficient data for analysis.
        
        Parameters
        ----------
        market1 : str
            First market ID
        market2 : str
            Second market ID
        min_observations : int
            Minimum number of common observations required
            
        Returns
        -------
        tuple or None
            Market pair tuple if valid, None otherwise
        """
        # Get price data for each market
        data1 = self.price_data[self.price_data[self.market_id_col] == market1]
        data2 = self.price_data[self.price_data[self.market_id_col] == market2]
        
        # Check for overlapping dates
        common_dates = set(data1[self.date_col]) & set(data2[self.date_col])
        
        # Check if we have at least min_observations common observations
        if len(common_dates) >= min_observations:
            return (market1, market2)
        else:
            logger.debug(
                f"Insufficient data for market pair {market1}-{market2}: "
                f"only {len(common_dates)} common dates"
            )
            return None
    
    @timer
    @memory_usage_decorator
    @m3_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def analyze_market_pair(
        self,
        market1: str,
        market2: str,
        estimation_method: str = 'ols',
        max_lags: int = 4,
        run_diagnostics: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a single market pair using threshold cointegration.
        
        Parameters
        ----------
        market1 : str
            First market ID
        market2 : str
            Second market ID
        estimation_method : str, optional
            Method for cointegration estimation ('ols', 'dols', 'fmols')
        max_lags : int, optional
            Maximum number of lags to consider
        run_diagnostics : bool, optional
            Whether to run model diagnostics
            
        Returns
        -------
        dict
            Analysis results for the market pair
        """
        # Get price data for each market
        data1 = self.price_data[self.price_data[self.market_id_col] == market1]
        data2 = self.price_data[self.price_data[self.market_id_col] == market2]
        
        # Sort by date
        data1 = data1.sort_values(self.date_col)
        data2 = data2.sort_values(self.date_col)
        
        # TODO: Complete the market_pair analysis implementation
        # This is a stub for now, as we're focusing on the panel and network methods
        
        return {"market1": market1, "market2": market2, "status": "pending"}
    
    @timer
    @memory_usage_decorator
    @m3_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def run_panel_cointegration_analysis(
        self,
        method: str = 'pedroni',
        trend: str = 'c',
        lag: int = 1,
        multiple_testing_correction: bool = True,
        correction_method: str = 'fdr_bh'
    ) -> Dict[str, Any]:
        """
        Run panel cointegration analysis across all markets.
        
        This method tests for cointegration relationships in the entire panel
        of market prices simultaneously, providing a system-wide view of
        market integration patterns.
        
        Parameters
        ----------
        method : str, default='pedroni'
            Panel cointegration test to use:
            - 'pedroni': Pedroni panel cointegration test
            - 'kao': Kao panel cointegration test
            - 'westerlund': Westerlund panel cointegration test
        trend : str, default='c'
            Deterministic trend specification ('n', 'c', 'ct')
        lag : int, default=1
            Number of lags to use in testing
        multiple_testing_correction : bool, default=True
            Whether to apply multiple testing correction
        correction_method : str, default='fdr_bh'
            Method for multiple testing correction if applied
            
        Returns
        -------
        Dict[str, Any]
            Panel cointegration test results
        """
        # Initialize panel cointegration tester if not already done
        if self.panel_model is None:
            self.panel_model = PanelCointegrationTester(
                data=self.price_data,
                market_col=self.market_id_col,
                time_col=self.date_col,
                price_col=self.price_col,
                max_workers=self.max_workers
            )
        
        # Run the specified panel cointegration test
        results = {}
        
        if method == 'pedroni':
            results = self.panel_model.test_pedroni(trend=trend, lag=lag)
        elif method == 'kao':
            results['kao'] = self.panel_model.test_kao(lag=lag)
            results['panel_test'] = 'kao'
            results['panel_cointegrated'] = results['kao'].significant
        elif method == 'westerlund':
            results = self.panel_model.test_westerlund(trend=trend, lag=lag)
        else:
            raise ValueError(f"Unknown panel cointegration test method: {method}")
        
        # Apply multiple testing correction if requested
        if multiple_testing_correction and method in ['pedroni', 'westerlund']:
            # Collect p-values from different test statistics
            p_values = []
            p_value_sources = []
            
            for key, value in results.items():
                if isinstance(value, dict) and 'p_value' in value:
                    p_values.append(value['p_value'])
                    p_value_sources.append(key)
            
            # Apply correction
            if p_values:
                reject, corrected_p = apply_multiple_testing_correction(
                    p_values, method=correction_method
                )
                
                # Update results with corrected p-values
                for i, key in enumerate(p_value_sources):
                    results[key]['corrected_p_value'] = corrected_p[i]
                    results[key]['passed_correction'] = bool(reject[i])
                
                # Update overall result based on corrected p-values
                results['panel_cointegrated'] = any(reject)
        
        logger.info(
            f"Panel cointegration test ({method}): "
            f"{'Cointegrated' if results.get('panel_cointegrated', False) else 'Not cointegrated'}"
        )
        
        return results
    
    @timer
    @memory_usage_decorator
    @m3_optimized(memory_intensive=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def run_network_analysis(
        self,
        integration_measure: str = 'half_life',
        weight_transform: str = 'inverse',
        weight_threshold: Optional[float] = None,
        community_method: str = 'louvain',
        run_resilience_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Run network analysis on market integration patterns.
        
        This method treats markets as nodes in a network with price transmission
        strength as edge weights, identifying central markets, communities, and
        structural patterns in the integration network.
        
        Parameters
        ----------
        integration_measure : str, default='half_life'
            Measure to use for edge weights:
            - 'half_life': Half-life of adjustment from threshold model
            - 'cointegration_stat': Cointegration test statistic
            - 'threshold': Threshold value
            - 'adjustment_speed': Speed of adjustment coefficient
            - 'p_value': P-value from cointegration test (reversed)
        weight_transform : str, default='inverse'
            Transformation to apply to the integration measure:
            - 'inverse': 1/(measure+1) (smaller half-life = stronger integration)
            - 'negative': -measure (more negative = stronger integration)
            - 'direct': measure (as-is)
        weight_threshold : float, optional
            Minimum weight threshold for including an edge
        community_method : str, default='louvain'
            Method for community detection:
            - 'louvain': Louvain method (modularity optimization)
            - 'leiden': Leiden method (improved Louvain)
            - 'label_propagation': Label propagation algorithm
        run_resilience_analysis : bool, default=True
            Whether to analyze network resilience to market disruptions
            
        Returns
        -------
        Dict[str, Any]
            Network analysis results
        """
        # We need time series results for market pairs to build the network
        if not hasattr(self, 'time_series_results') or not self.time_series_results:
            logger.warning(
                "No time series results available for network analysis. "
                "Run 'analyze_all_market_pairs' first."
            )
            return {}
        
        # Initialize network model
        self.network_model = MarketNetworkAnalysis(
            market_results=self.time_series_results,
            integration_measure=integration_measure,
            weight_transform=weight_transform,
            weight_threshold=weight_threshold,
            max_workers=self.max_workers
        )
        
        # Calculate centrality measures
        centrality = self.network_model.calculate_centrality()
        
        # Detect communities
        communities = self.network_model.detect_communities(method=community_method)
        
        # Convert communities to community sizes
        community_sizes = {}
        for community_id in set(communities.values()):
            community_sizes[community_id] = sum(1 for c in communities.values() if c == community_id)
        
        # Analyze cross-community integration
        cross_community = self.network_model.analyze_inter_community_integration()
        
        # Identify key connector markets
        connector_markets = self.network_model.identify_key_connector_markets(n=5)
        
        # Analyze network resilience if requested
        resilience_results = {}
        if run_resilience_analysis:
            resilience_results = self.network_model.compute_network_resilience()
        
        # Compile network analysis results
        results = {
            'network_density': nx.density(self.network_model.graph),
            'n_communities': len(community_sizes),
            'communities': communities,
            'community_sizes': community_sizes,
            'centrality_measures': {m: c.as_dict() for m, c in centrality.items()},
            'connector_markets': connector_markets,
            'cross_community_integration': cross_community,
            'modularity': cross_community.get('modularity', 0.0),
            'resilience': resilience_results
        }
        
        # Calculate average centrality for each market
        for market, measures in results['centrality_measures'].items():
            # Calculate average of degree, betweenness, eigenvector
            avg_centrality = (
                measures.get('degree', 0) + 
                measures.get('betweenness', 0) + 
                measures.get('eigenvector', 0)
            ) / 3
            results['centrality_measures'][market]['average'] = avg_centrality
        
        logger.info(
            f"Network analysis complete: "
            f"{len(communities)} communities detected with modularity {cross_community.get('modularity', 0.0):.4f}"
        )
        
        return results
