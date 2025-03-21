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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from yemen_market_integration.utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_geodataframe, validate_dataframe, validate_time_series, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, disk_cache, parallelize_dataframe,
    optimize_dataframe, configure_system_for_performance,
    
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

# Initialize module logger
logger = logging.getLogger(__name__)

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
    
    def __str__(self) -> str:
        """String representation of integration results."""
        n_pairs = len(self.market_pairs)
        n_markets = len(set([m for pair in self.market_pairs for m in pair]))
        
        return (
            f"Yemen Market Integration Analysis\n"
            f"-------------------------------\n"
            f"Analyzed {n_pairs} market pairs across {n_markets} markets\n"
            f"Time period: {self.meta.get('start_date', 'N/A')} to {self.meta.get('end_date', 'N/A')}\n"
            f"Integration summary: {self.integrated_metrics.get('integration_level', 'N/A')}\n"
            f"Spatial autocorrelation: {self.spatial_results.get('moran_result', {}).get('significant_autocorrelation', False)}\n"
            f"Threshold effects: {self.integrated_metrics.get('threshold_effects_summary', 'N/A')}\n"
        )
        
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
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
        
        # Plot 3: Histogram of adjustment speeds
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
        
        # Plot 4: Spatial autocorrelation 
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
        region_col: str = DEFAULT_REGION_COL
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
        """
        # Configure system for optimal performance
        configure_system_for_performance()
        
        # Store column names
        self.price_col = price_col
        self.market_id_col = market_id_col
        self.date_col = date_col
        self.conflict_col = conflict_col
        self.region_col = region_col
        
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
        price data for time series analysis.
        """
        # Get unique markets
        markets = self.markets
        
        # Create all possible market pairs
        all_pairs = [(markets[i], markets[j]) 
                     for i in range(len(markets)) 
                     for j in range(i+1, len(markets))]
        
        # Filter for pairs with sufficient data
        valid_pairs = []
        
        for market1, market2 in all_pairs:
            # Get price data for each market
            data1 = self.price_data[self.price_data[self.market_id_col] == market1]
            data2 = self.price_data[self.price_data[self.market_id_col] == market2]
            
            # Check for overlapping dates
            common_dates = set(data1[self.date_col]) & set(data2[self.date_col])
            
            # Check if we have at least 30 common observations
            if len(common_dates) >= 30:
                valid_pairs.append((market1, market2))
            else:
                logger.debug(
                    f"Insufficient data for market pair {market1}-{market2}: "
                    f"only {len(common_dates)} common dates"
                )
        
        self.market_pairs = valid_pairs
        logger.info(f"Prepared {len(valid_pairs)} valid market pairs out of {len(all_pairs)} possible pairs")
    
    @timer
    @memory_usage_decorator
    @m1_optimized(parallel=True)
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
        
        # Get common dates
        common_dates = sorted(set(data1[self.date_col]) & set(data2[self.date_col]))
        
        if len(common_dates) < 30:
            logger.warning(
                f"Insufficient data for market pair {market1}-{market2}: "
                f"only {len(common_dates)} common dates"
            )
            return {'error': 'Insufficient data'}
        
        # Filter for common dates
        data1 = data1[data1[self.date_col].isin(common_dates)]
        data2 = data2[data2[self.date_col].isin(common_dates)]
        
        # Extract price series
        prices1 = data1[self.price_col].values
        prices2 = data2[self.price_col].values
        dates = data1[self.date_col].values
        
        # Create ThresholdCointegration model
        tar_model = ThresholdCointegration(
            data1=prices1,
            data2=prices2,
            max_lags=max_lags,
            market1_name=str(market1),
            market2_name=str(market2)
        )
        
        # Run full analysis
        full_results = tar_model.run_full_analysis()
        
        # Add meta information
        result = {'market1': market1, 'market2': market2}
        result.update(full_results)
        
        # Add spatial information if available
        if self.has_spatial:
            spatial_meta = self._add_spatial_context(market1, market2)
            result['meta'] = spatial_meta
        
        # Add time range
        result['dates'] = {
            'start_date': dates[0],
            'end_date': dates[-1],
            'n_observations': len(dates)
        }
        
        # Run diagnostics if requested
        if run_diagnostics:
            try:
                diagnostics = ModelDiagnostics(
                    residuals=full_results['cointegration']['residuals'],
                    model_name=f"TAR_{market1}_{market2}"
                )
                result['diagnostics'] = diagnostics.run_all_tests()
            except Exception as e:
                logger.warning(f"Could not run diagnostics: {e}")
        
        logger.info(
            f"Completed analysis for market pair {market1}-{market2}: "
            f"cointegrated={full_results['cointegration']['cointegrated']}, "
            f"threshold={full_results['threshold']['threshold']:.4f}, "
            f"asymmetric={full_results['summary']['asymmetric_adjustment_tar']}"
        )
        
        return result
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _add_spatial_context(self, market1: str, market2: str) -> Dict[str, Any]:
        """
        Add spatial context to market pair analysis.
        
        Parameters
        ----------
        market1 : str
            First market ID
        market2 : str
            Second market ID
            
        Returns
        -------
        dict
            Spatial context for the market pair
        """
        meta = {}
        
        # Get market locations
        if self.spatial_data is not None:
            market1_data = self.spatial_data[self.spatial_data[self.market_id_col] == market1]
            market2_data = self.spatial_data[self.spatial_data[self.market_id_col] == market2]
            
            if not market1_data.empty and not market2_data.empty:
                # Calculate geographic distance
                point1 = market1_data.iloc[0].geometry
                point2 = market2_data.iloc[0].geometry
                
                try:
                    # Calculate distance in kilometers
                    distance = point1.distance(point2) / 1000
                    meta['distance_km'] = distance
                except Exception as e:
                    logger.warning(f"Could not calculate distance: {e}")
                
                # Add region information if available
                if self.region_col in market1_data.columns and self.region_col in market2_data.columns:
                    region1 = market1_data.iloc[0][self.region_col]
                    region2 = market2_data.iloc[0][self.region_col]
                    
                    meta['region1'] = region1
                    meta['region2'] = region2
                    meta['same_region'] = region1 == region2
                    
                    # Add exchange rate difference if cross-region
                    if region1 != region2:
                        try:
                            # Try to get exchange rate data
                            er_col = 'exchange_rate'
                            if er_col in market1_data.columns and er_col in market2_data.columns:
                                er1 = market1_data.iloc[0][er_col]
                                er2 = market2_data.iloc[0][er_col]
                                
                                if er1 > 0 and er2 > 0:
                                    er_diff = abs(er1 - er2) / min(er1, er2)
                                    meta['exchange_rate_diff'] = er_diff
                        except Exception as e:
                            logger.warning(f"Could not calculate exchange rate difference: {e}")
                
                # Add conflict barrier information if available
                if self.conflict_col in market1_data.columns and self.conflict_col in market2_data.columns:
                    conflict1 = market1_data.iloc[0][self.conflict_col]
                    conflict2 = market2_data.iloc[0][self.conflict_col]
                    
                    # Calculate average conflict intensity
                    meta['conflict_barrier'] = (conflict1 + conflict2) / 2
        
        return meta
    
    @timer
    @memory_usage_decorator
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def analyze_all_market_pairs(
        self,
        max_lags: int = 4,
        run_diagnostics: bool = True,
        n_workers: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all market pairs using threshold cointegration.
        
        Parameters
        ----------
        max_lags : int, optional
            Maximum number of lags to consider
        run_diagnostics : bool, optional
            Whether to run model diagnostics
        n_workers : int, optional
            Number of parallel workers to use
            
        Returns
        -------
        dict
            Results for all market pairs
        """
        # Set number of workers if not provided
        if n_workers is None:
            import multiprocessing as mp
            n_workers = max(1, mp.cpu_count() - 1)
        
        # Log that we're starting analysis
        logger.info(f"Starting analysis of {len(self.market_pairs)} market pairs using {n_workers} workers")
        
        # Use parallel processing for better performance
        all_results = {}
        
        # Create a ProcessPoolExecutor to parallelize the analysis
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit tasks for each market pair
            future_to_pair = {}
            for market1, market2 in self.market_pairs:
                future = executor.submit(
                    self.analyze_market_pair,
                    market1=market1,
                    market2=market2,
                    max_lags=max_lags,
                    run_diagnostics=run_diagnostics
                )
                future_to_pair[future] = (market1, market2)
            
            # Process results as they complete
            total_pairs = len(future_to_pair)
            completed = 0
            
            for future in future_to_pair:
                market1, market2 = future_to_pair[future]
                pair_key = f"{market1}_{market2}"
                
                try:
                    # Get result and store by pair key
                    result = future.result()
                    all_results[pair_key] = result
                    
                    # Log progress
                    completed += 1
                    if completed % max(1, total_pairs // 10) == 0:
                        logger.info(f"Analyzed {completed}/{total_pairs} market pairs ({completed/total_pairs:.1%})")
                    
                except Exception as e:
                    logger.error(f"Error analyzing market pair {market1}-{market2}: {e}")
                    all_results[pair_key] = {'error': str(e)}
        
        logger.info(f"Completed analysis of all market pairs. Found {sum(1 for r in all_results.values() if r.get('cointegration', {}).get('cointegrated', False))} cointegrated pairs")
        
        return all_results
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def run_spatial_analysis(
        self,
        conflict_adjusted: bool = True,
        k: int = 5,
        conflict_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run spatial analysis on market data.
        
        Parameters
        ----------
        conflict_adjusted : bool, optional
            Whether to adjust spatial weights by conflict intensity
        k : int, optional
            Number of nearest neighbors
        conflict_weight : float, optional
            Weight of conflict adjustment (0-1)
            
        Returns
        -------
        dict
            Spatial analysis results
        """
        # Make sure we have spatial data
        if not self.has_spatial:
            raise ValueError("Spatial data not provided, cannot run spatial analysis")
        
        # Make sure we have price data
        if not hasattr(self, 'price_data') or self.price_data is None:
            raise ValueError("Price data not available for spatial analysis")
        
        # Create SpatialEconometrics model
        self.spatial_model = SpatialEconometrics(self.spatial_data)
        
        # Create weight matrix
        weights = self.spatial_model.create_weight_matrix(
            k=k,
            conflict_adjusted=conflict_adjusted,
            conflict_col=self.conflict_col,
            conflict_weight=conflict_weight
        )
        
        # Run Moran's I test on prices
        try:
            # Get latest price for each market
            latest_date = self.price_data[self.date_col].max()
            latest_prices = self.price_data[self.price_data[self.date_col] == latest_date]
            
            # Merge with spatial data
            merged_data = self.spatial_data.merge(
                latest_prices[[self.market_id_col, self.price_col]],
                on=self.market_id_col,
                how='inner'
            )
            
            # Run Moran's I test
            moran_result = self.spatial_model.moran_i_test(self.price_col)
            
            # Run local Moran's I test
            local_moran_result = self.spatial_model.local_moran_test(self.price_col)
            
            # Run spatial regression for price determinants if conflict data is available
            spatial_reg_results = {}
            if self.has_conflict and self.conflict_col in merged_data.columns:
                # Prepare independent variables
                x_cols = [self.conflict_col]
                
                # Add region dummy if available
                if self.region_col in merged_data.columns:
                    # Create dummy variable
                    merged_data['region_dummy'] = (merged_data[self.region_col] == 'north').astype(int)
                    x_cols.append('region_dummy')
                
                # Run spatial lag model
                try:
                    lag_model = self.spatial_model.spatial_lag_model(
                        y_col=self.price_col,
                        x_cols=x_cols
                    )
                    spatial_reg_results['lag_model'] = {
                        'params': lag_model.betas.tolist(),
                        'rho': lag_model.rho,
                        'r2': lag_model.pr2,
                        'aic': lag_model.aic
                    }
                except Exception as e:
                    logger.warning(f"Could not estimate spatial lag model: {e}")
                
                # Run spatial error model
                try:
                    error_model = self.spatial_model.spatial_error_model(
                        y_col=self.price_col,
                        x_cols=x_cols
                    )
                    spatial_reg_results['error_model'] = {
                        'params': error_model.betas.tolist(),
                        'lambda': error_model.lam,
                        'r2': error_model.pr2,
                        'aic': error_model.aic
                    }
                except Exception as e:
                    logger.warning(f"Could not estimate spatial error model: {e}")
            
            # Compile results
            spatial_results = {
                'moran_result': moran_result,
                'local_moran_result': local_moran_result,
                'spatial_reg_results': spatial_reg_results,
                'weight_matrix_info': {
                    'k': k,
                    'conflict_adjusted': conflict_adjusted,
                    'conflict_weight': conflict_weight if conflict_adjusted else None
                }
            }
            
            logger.info(
                f"Spatial analysis completed. "
                f"Moran's I: {moran_result['I']:.4f} (p={moran_result['p_norm']:.4f}), "
                f"Significant: {moran_result['significant']}"
            )
            
            return spatial_results
            
        except Exception as e:
            logger.error(f"Error running spatial analysis: {e}")
            raise
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def calculate_market_accessibility(
        self,
        population_data: Optional[gpd.GeoDataFrame] = None,
        max_distance: float = 50000,
        distance_decay: float = 2.0,
        weight_col: str = 'population'
    ) -> gpd.GeoDataFrame:
        """
        Calculate market accessibility indices.
        
        Parameters
        ----------
        population_data : gpd.GeoDataFrame, optional
            GeoDataFrame with population centers
        max_distance : float, optional
            Maximum distance to consider (meters)
        distance_decay : float, optional
            Distance decay exponent
        weight_col : str, optional
            Population weight column
            
        Returns
        -------
        gpd.GeoDataFrame
            Market data with accessibility indices
        """
        # Make sure we have spatial data
        if not self.has_spatial:
            raise ValueError("Spatial data not provided, cannot calculate accessibility")
        
        # Check population data
        if population_data is None:
            raise ValueError("Population data required for accessibility calculation")
        
        # Calculate using the utility function
        accessibility_df = calculate_market_accessibility(
            markets_gdf=self.spatial_data,
            population_gdf=population_data,
            max_distance=max_distance,
            distance_decay=distance_decay,
            weight_col=weight_col
        )
        
        # Return the results
        logger.info(f"Calculated accessibility indices for {len(accessibility_df)} markets")
        return accessibility_df
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def run_market_integration_index(
        self,
        windows: Optional[List[int]] = None,
        return_components: bool = False
    ) -> pd.DataFrame:
        """
        Calculate market integration indices over time.
        
        Parameters
        ----------
        windows : list, optional
            List of window sizes for rolling calculations
        return_components : bool, optional
            Whether to return component metrics
            
        Returns
        -------
        pd.DataFrame
            Time series of integration indices
        """
        # Make sure we have spatial data
        if not self.has_spatial:
            raise ValueError("Spatial data not provided, cannot calculate integration index")
        
        # Make sure we have price data
        if not hasattr(self, 'price_data') or self.price_data is None:
            raise ValueError("Price data not available for integration index")
        
        # Create spatial weights if not yet created
        if not hasattr(self, 'spatial_model') or self.spatial_model is None:
            self.spatial_model = SpatialEconometrics(self.spatial_data)
            self.spatial_model.create_weight_matrix(
                conflict_adjusted=True,
                conflict_col=self.conflict_col
            )
        
        # Calculate using the utility function
        integration_df = market_integration_index(
            prices_df=self.price_data,
            weights_matrix=self.spatial_model.weights,
            market_id_col=self.market_id_col,
            price_col=self.price_col,
            time_col=self.date_col,
            windows=windows,
            return_components=return_components
        )
        
        # Return the results
        logger.info(f"Calculated market integration indices for {len(integration_df)} time periods")
        return integration_df
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
    def run_integrated_analysis(self) -> IntegrationResults:
        """
        Run a complete integrated market analysis.
        
        This method combines time series analysis (threshold cointegration)
        with spatial econometric analysis to provide a comprehensive view
        of market integration patterns in Yemen.
        
        Returns
        -------
        IntegrationResults
            Comprehensive integration analysis results
        """
        # Run time series analysis for all market pairs
        time_series_results = self.analyze_all_market_pairs(run_diagnostics=True)
        
        # Run spatial analysis if spatial data is available
        spatial_results = {}
        if self.has_spatial:
            spatial_results = self.run_spatial_analysis(conflict_adjusted=True)
        
        # Calculate integrated metrics
        integrated_metrics = self._calculate_integrated_metrics(
            time_series_results=time_series_results,
            spatial_results=spatial_results
        )
        
        # Create visualization hooks
        visualization_hooks = self._create_visualization_hooks(
            time_series_results=time_series_results,
            spatial_results=spatial_results
        )
        
        # Create meta information
        meta = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_markets': self.n_markets,
            'n_pairs': len(self.market_pairs),
            'n_cointegrated': sum(1 for r in time_series_results.values() 
                               if r.get('cointegration', {}).get('cointegrated', False)),
            'n_asymmetric': sum(1 for r in time_series_results.values() 
                             if r.get('mtar', {}).get('asymmetric', False) or 
                             r.get('tvecm', {}).get('asymmetric_adjustment', {}).get('asymmetric', False))
        }
        
        # Create IntegrationResults object
        results = IntegrationResults(
            time_series_results=time_series_results,
            spatial_results=spatial_results,
            integrated_metrics=integrated_metrics,
            market_pairs=self.market_pairs,
            visualization_hooks=visualization_hooks,
            meta=meta
        )
        
        logger.info(
            f"Completed integrated analysis of {meta['n_pairs']} market pairs. "
            f"Found {meta['n_cointegrated']} cointegrated pairs and {meta['n_asymmetric']} with asymmetric adjustment."
        )
        
        return results
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _calculate_integrated_metrics(
        self,
        time_series_results: Dict[str, Dict[str, Any]],
        spatial_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate integrated metrics combining time series and spatial analysis.
        
        Parameters
        ----------
        time_series_results : dict
            Results from time series analysis
        spatial_results : dict
            Results from spatial analysis
            
        Returns
        -------
        dict
            Integrated metrics
        """
        # Count cointegrated pairs
        n_pairs = len(time_series_results)
        n_cointegrated = sum(1 for r in time_series_results.values() 
                          if r.get('cointegration', {}).get('cointegrated', False))
        
        # Count pairs with significant thresholds
        n_threshold = sum(1 for r in time_series_results.values() 
                       if r.get('threshold_significance', {}).get('significant', False))
        
        # Count pairs with asymmetric adjustment
        n_asymmetric = sum(1 for r in time_series_results.values() 
                        if r.get('mtar', {}).get('asymmetric', False) or 
                        r.get('tvecm', {}).get('asymmetric_adjustment', {}).get('asymmetric', False))
        
        # Add spatial metrics if available
        spatial_metrics = {}
        if spatial_results:
            moran = spatial_results.get('moran_result', {})
            spatial_metrics = {
                'spatial_autocorrelation': moran.get('I', 0),
                'spatial_pvalue': moran.get('p_norm', 1),
                'significant_autocorrelation': moran.get('significant', False),
                'positive_autocorrelation': moran.get('positive_autocorrelation', False)
            }
        
        # Calculate average threshold
        thresholds = [r.get('threshold', {}).get('threshold', np.nan) 
                     for r in time_series_results.values()]
        avg_threshold = np.nanmean(thresholds) if thresholds else np.nan
        
        # Calculate average half-lives
        half_lives_below = [r.get('tvecm', {}).get('asymmetric_adjustment', {}).get('half_life_below', np.nan) 
                           for r in time_series_results.values()]
        half_lives_above = [r.get('tvecm', {}).get('asymmetric_adjustment', {}).get('half_life_above', np.nan) 
                           for r in time_series_results.values()]
        
        avg_half_life_below = np.nanmean(half_lives_below) if half_lives_below else np.nan
        avg_half_life_above = np.nanmean(half_lives_above) if half_lives_above else np.nan
        
        # Determine overall integration level
        if n_pairs == 0:
            integration_level = "Unknown"
        elif n_cointegrated / n_pairs < 0.2:
            integration_level = "Very Low"
        elif n_cointegrated / n_pairs < 0.4:
            integration_level = "Low"
        elif n_cointegrated / n_pairs < 0.6:
            integration_level = "Moderate"
        elif n_cointegrated / n_pairs < 0.8:
            integration_level = "High"
        else:
            integration_level = "Very High"
        
        # Determine threshold effects
        if n_threshold / n_pairs < 0.2:
            threshold_effects = "Very Low"
        elif n_threshold / n_pairs < 0.4:
            threshold_effects = "Low"
        elif n_threshold / n_pairs < 0.6:
            threshold_effects = "Moderate"
        elif n_threshold / n_pairs < 0.8:
            threshold_effects = "High"
        else:
            threshold_effects = "Very High"
        
        # Combine metrics
        integrated_metrics = {
            'cointegration_rate': n_cointegrated / n_pairs if n_pairs > 0 else 0,
            'threshold_effect_rate': n_threshold / n_pairs if n_pairs > 0 else 0,
            'asymmetric_rate': n_asymmetric / n_pairs if n_pairs > 0 else 0,
            'avg_threshold': avg_threshold,
            'avg_half_life_below': avg_half_life_below,
            'avg_half_life_above': avg_half_life_above,
            'integration_level': integration_level,
            'threshold_effects_summary': threshold_effects
        }
        
        # Add spatial metrics if available
        integrated_metrics.update(spatial_metrics)
        
        return integrated_metrics
    
    @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
    def _create_visualization_hooks(
        self,
        time_series_results: Dict[str, Dict[str, Any]],
        spatial_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create visualization hooks for integration results.
        
        Parameters
        ----------
        time_series_results : dict
            Results from time series analysis
        spatial_results : dict
            Results from spatial analysis
            
        Returns
        -------
        dict
            Visualization hooks
        """
        # Initialize hooks
        hooks = {}
        
        # Create market pair edges data if spatial data is available
        if self.has_spatial:
            try:
                # Create GeoDataFrame with market locations
                market_gdf = self.spatial_data.copy()
                
                # Create edges for market pairs
                edges = []
                
                for market1, market2 in self.market_pairs:
                    # Get result for this pair
                    pair_key = f"{market1}_{market2}"
                    result = time_series_results.get(pair_key, {})
                    
                    # Get market locations
                    market1_data = market_gdf[market_gdf[self.market_id_col] == market1]
                    market2_data = market_gdf[market_gdf[self.market_id_col] == market2]
                    
                    if not market1_data.empty and not market2_data.empty:
                        # Create line geometry
                        from shapely.geometry import LineString
                        line = LineString([
                            market1_data.iloc[0].geometry,
                            market2_data.iloc[0].geometry
                        ])
                        
                        # Get attributes from result
                        cointegrated = result.get('cointegration', {}).get('cointegrated', False)
                        threshold = result.get('threshold', {}).get('threshold', np.nan)
                        asymmetric = result.get('mtar', {}).get('asymmetric', False) or \
                                    result.get('tvecm', {}).get('asymmetric_adjustment', {}).get('asymmetric', False)
                        
                        # Create edge entry
                        edge = {
                            'market1': market1,
                            'market2': market2,
                            'cointegrated': cointegrated,
                            'threshold': threshold,
                            'asymmetric': asymmetric,
                            'geometry': line
                        }
                        
                        # Add integration level
                        if not cointegrated:
                            integration_level = "Not Integrated"
                            integration_level_num = 0
                        elif asymmetric:
                            integration_level = "Asymmetric Integration"
                            integration_level_num = 2
                        else:
                            integration_level = "Symmetric Integration"
                            integration_level_num = 1
                        
                        edge['integration_level'] = integration_level
                        edge['integration_level_num'] = integration_level_num
                        
                        edges.append(edge)
                
                # Create GeoDataFrame with edges
                import geopandas as gpd
                edges_gdf = gpd.GeoDataFrame(edges, geometry='geometry')
                
                # Set CRS to match market_gdf
                edges_gdf.crs = market_gdf.crs
                
                # Add to hooks
                hooks['market_gdf'] = market_gdf
                hooks['edges_gdf'] = edges_gdf
                
            except Exception as e:
                logger.warning(f"Could not create visualization hooks: {e}")
        
        return hooks


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def _calculate_integration_score(
    cointegrated: bool,
    asymmetric: bool,
    half_life_below: float,
    half_life_above: float
) -> float:
    """
    Calculate an integration score (0-1) from market pair results.
    
    Parameters
    ----------
    cointegrated : bool
        Whether markets are cointegrated
    asymmetric : bool
        Whether adjustment is asymmetric
    half_life_below : float
        Half-life below threshold
    half_life_above : float
        Half-life above threshold
        
    Returns
    -------
    float
        Integration score (0-1)
    """
    # Start with baseline score
    score = 0.0
    
    # Add cointegration component
    if cointegrated:
        score += 0.5
        
        # Add adjustment speed component
        try:
            # Calculate adjustment component from half-lives
            if np.isfinite(half_life_below) and half_life_below > 0:
                adj_below = min(1.0, 10.0 / half_life_below)
            else:
                adj_below = 0.0
                
            if np.isfinite(half_life_above) and half_life_above > 0:
                adj_above = min(1.0, 10.0 / half_life_above)
            else:
                adj_above = 0.0
            
            # Calculate effective adjustment speed (weight above more heavily)
            if asymmetric:
                # Above threshold adjustment is more important
                effective_adj = 0.25 * adj_below + 0.75 * adj_above
            else:
                # Equal weighting
                effective_adj = 0.5 * adj_below + 0.5 * adj_above
            
            # Add to score
            score += 0.5 * effective_adj
        except:
            # Default if calculation fails
            score += 0.2
    
    return score
