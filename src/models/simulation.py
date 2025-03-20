"""
Market Integration Simulation Module for the Yemen Market Integration Project.

This module provides simulation capabilities for testing the effects of various policy
interventions on market integration in Yemen, including exchange rate unification 
and improved connectivity due to reduced conflict.
"""
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Any, Union, Optional, List, Tuple, Callable
import gc
import psutil
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from collections import defaultdict
import itertools

from models.threshold import ThresholdCointegration, test_mtar_adjustment
from models.diagnostics import ModelDiagnostics


from utils import (
    # Error handling
    handle_errors, ValidationError,
    
    # Validation
    validate_dataframe, validate_geodataframe, raise_if_invalid,
    
    # Configuration
    config,
    
    # Data processing
    compute_price_differentials, normalize_columns, optimize_dataframe,
    
    # Spatial utilities
    create_conflict_adjusted_weights, calculate_distances,
    
    # Statistical utilities
    test_cointegration, estimate_threshold_model,
    
    # Performance
    m1_optimized, timer, memory_usage_decorator, disk_cache, memoize,
    parallelize_dataframe, configure_system_for_performance
)

logger = logging.getLogger(__name__)

# Configure system for optimal performance
configure_system_for_performance()

class MarketIntegrationSimulation:
    """
    Simulate policy interventions for market integration analysis in Yemen.
    
    This class provides methods to simulate the effects of exchange rate unification
    and improved connectivity on market integration, and to calculate welfare effects
    of these policy interventions.
    """
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def __init__(
        self, 
        data: gpd.GeoDataFrame, 
        threshold_model: Optional[Any] = None, 
        spatial_model: Optional[Any] = None
    ):
        """
        Initialize the MarketIntegrationSimulation with market data and models.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            GeoDataFrame containing market data with at least the following columns:
            - exchange_rate_regime: str, 'north' or 'south'
            - price: float, commodity price in local currency
            - exchange_rate: float, exchange rate to USD
            - conflict_intensity_normalized: float, normalized conflict intensity
        threshold_model : Any, optional
            Estimated threshold model for baseline comparison
        spatial_model : Any, optional
            Estimated spatial model for baseline comparison
        """
        # Validate input data
        self._validate_input_data(data)
        
        # Store data and models (optimize memory usage)
        self.data = optimize_dataframe(data.copy())
        self.threshold_model = threshold_model
        self.spatial_model = spatial_model
        
        # Store original data for comparison (optimize memory usage)
        self.original_data = optimize_dataframe(data.copy())
        
        # Initialize results storage
        self.results = {}
        
        # Get number of available workers based on CPU count
        self.n_workers = config.get('performance.n_workers', max(1, mp.cpu_count() - 1))
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"MarketIntegrationSimulation initialized with data of shape {data.shape}, memory: {memory_usage:.2f} MB")
    
    @handle_errors(logger=logger, error_type=ValidationError)
    def _validate_input_data(self, data: gpd.GeoDataFrame) -> None:
        """
        Validate input data for simulation.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            GeoDataFrame to validate
            
        Raises
        ------
        ValidationError
            If data does not meet requirements
        """
        # Check if data is a GeoDataFrame
        if not isinstance(data, gpd.GeoDataFrame):
            raise ValidationError("Input data must be a GeoDataFrame")
        
        # Validate required columns
        valid, errors = validate_geodataframe(
            data,
            required_columns=[
                'exchange_rate_regime', 
                'price', 
                'exchange_rate'
            ]
        )
        raise_if_invalid(valid, errors, "Invalid market data format")
        
        # Validate exchange rate regime values
        valid_regimes = set(data['exchange_rate_regime'].unique())
        if not all(regime in ['north', 'south'] for regime in valid_regimes):
            raise ValidationError(f"Exchange rate regime must be 'north' or 'south', got {valid_regimes}")
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger)
    def simulate_exchange_rate_unification(
        self, 
        target_rate: str = 'official', 
        reference_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate exchange rate unification using USD as cross-rate.
        
        The simulation follows these steps:
        1. Convert all prices to USD using region-specific exchange rates
        2. Apply a single unified USD-YER exchange rate across all regions
        3. Calculate new price differentials and re-estimate threshold models
        
        Parameters
        ----------
        target_rate : str, optional
            Method to determine the unified exchange rate:
            - 'official': Use official exchange rate
            - 'market': Use market exchange rate
            - 'average': Use average of north and south rates
            - Specific value: Use provided numerical value
        reference_date : str, optional
            Date to use for reference exchange rates (default: latest date)
            
        Returns
        -------
        Dict[str, Any]
            Simulation results including:
            - 'simulated_data': GeoDataFrame with simulated prices
            - 'unified_rate': Selected unified exchange rate
            - 'price_changes': DataFrame of price changes by region
            - 'threshold_model': Re-estimated threshold model
        """
        # Process input data in chunks for memory efficiency
        chunk_size = config.get('data.max_chunk_size', 10000)
        sim_data = optimize_dataframe(self.data.copy())
        
        # Track initial memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Convert all prices to USD
        self._convert_to_usd(sim_data)
        
        # Determine unified exchange rate
        unified_rate = self._determine_unified_rate(target_rate, reference_date)
        logger.info(f"Using unified exchange rate: {unified_rate:.2f}")
        
        # Apply unified exchange rate to convert back to YER
        sim_data['simulated_price'] = sim_data['usd_price'] * unified_rate
        
        # Calculate price changes
        price_changes = self._calculate_price_changes(
            original_prices=self.data['price'],
            simulated_prices=sim_data['simulated_price'],
            by_column='exchange_rate_regime'
        )
        
        # Re-estimate threshold model if provided
        threshold_model = None
        if self.threshold_model is not None:
            threshold_model = self._reestimate_threshold_model(sim_data)
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Exchange rate unification simulation complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Store results
        results = {
            'simulated_data': sim_data,
            'unified_rate': unified_rate,
            'price_changes': price_changes,
            'threshold_model': threshold_model
        }
        
        # Store in instance results
        self.results['exchange_rate_unification'] = results
        
        return results
    
    @handle_errors(logger=logger)
    def _convert_to_usd(self, data: gpd.GeoDataFrame) -> None:
        """
        Convert prices to USD using region-specific exchange rates.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            GeoDataFrame with price and exchange_rate columns
        """
        if 'exchange_rate' not in data.columns:
            raise ValidationError("Data must contain 'exchange_rate' column")
        
        # Convert prices to USD
        data['usd_price'] = data['price'] / data['exchange_rate']
        
        logger.debug("Converted prices to USD")
    
    @memoize
    @handle_errors(logger=logger)
    def _determine_unified_rate(
        self, 
        target_rate: str = 'official', 
        reference_date: Optional[str] = None
    ) -> float:
        """
        Determine the unified exchange rate to use in simulation.
        
        Parameters
        ----------
        target_rate : str
            Method to determine unified rate (official, market, average, or numeric value)
        reference_date : str, optional
            Date to use for reference rates
            
        Returns
        -------
        float
            Unified exchange rate value
        """
        # Try to interpret target_rate as numeric
        try:
            return float(target_rate)
        except ValueError:
            pass
        
        # Filter by reference date if provided
        data = self.data
        if reference_date is not None:
            if 'date' not in data.columns:
                logger.warning("Reference date provided but no date column exists")
            else:
                data = data[data['date'] == reference_date].copy()
                if data.empty:
                    logger.warning(f"No data found for reference date {reference_date}")
                    data = self.data  # Fallback to all data
        
        # Get rates by region
        north_rates = data[data['exchange_rate_regime'] == 'north']['exchange_rate']
        south_rates = data[data['exchange_rate_regime'] == 'south']['exchange_rate']
        
        # Calculate rate based on method
        if target_rate == 'official':
            # Use official exchange rate (assuming north is official)
            rate = north_rates.mean()
        elif target_rate == 'market':
            # Use market rate (assuming south is market)
            rate = south_rates.mean()
        elif target_rate == 'average':
            # Use average of north and south
            rate = (north_rates.mean() + south_rates.mean()) / 2
        else:
            raise ValueError(f"Invalid target_rate: {target_rate}")
        
        return rate
    
    @handle_errors(logger=logger)
    def _calculate_price_changes(
        self,
        original_prices: pd.Series,
        simulated_prices: pd.Series,
        by_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate price changes between original and simulated prices.
        
        Parameters
        ----------
        original_prices : pd.Series
            Original price series
        simulated_prices : pd.Series
            Simulated price series
        by_column : str, optional
            Column to group results by
            
        Returns
        -------
        pd.DataFrame
            DataFrame with price change statistics
        """
        # Calculate absolute and percentage changes
        abs_change = simulated_prices - original_prices
        pct_change = (abs_change / original_prices) * 100
        
        # Create DataFrame with changes
        changes_df = pd.DataFrame({
            'original_price': original_prices,
            'simulated_price': simulated_prices,
            'abs_change': abs_change,
            'pct_change': pct_change
        })
        
        # Group by column if provided
        if by_column is not None and by_column in self.data.columns:
            changes_df[by_column] = self.data[by_column].values
            
            # Calculate statistics by group
            stats = changes_df.groupby(by_column).agg({
                'original_price': ['mean', 'std'],
                'simulated_price': ['mean', 'std'],
                'abs_change': ['mean', 'std', 'min', 'max'],
                'pct_change': ['mean', 'std', 'min', 'max']
            })
            
            return stats
        
        # Otherwise return all changes
        return changes_df

    @disk_cache(cache_dir='.cache/simulations')
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger)
    def _reestimate_threshold_model(self, simulated_data: gpd.GeoDataFrame) -> Any:
        """
        Re-estimate threshold model with simulated prices.
        
        Parameters
        ----------
        simulated_data : gpd.GeoDataFrame
            GeoDataFrame with simulated prices
            
        Returns
        -------
        Any
            Re-estimated threshold model
        """
        logger.info("Re-estimating threshold model with simulated prices")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Optimize input data for memory efficiency
        simulated_data = optimize_dataframe(simulated_data.copy())
        
        # Get north and south prices
        north_prices = simulated_data[simulated_data['exchange_rate_regime'] == 'north']['simulated_price']
        south_prices = simulated_data[simulated_data['exchange_rate_regime'] == 'south']['simulated_price']
        
        # Check for large datasets that need chunking
        large_dataset = len(north_prices) > config.get('data.large_dataset_threshold', 10000)
        
        if large_dataset:
            logger.info(f"Processing large dataset with {len(north_prices)} observations using chunking")
            
            # Process in chunks
            chunk_size = config.get('data.chunk_size', 5000)
            reestimated_model = self._process_threshold_model_in_chunks(
                north_prices, south_prices, chunk_size
            )
        else:
            # Standard processing for smaller datasets
            logger.info(f"Processing dataset with {len(north_prices)} observations")
            
            # Re-estimate threshold model
            threshold_params = config.get_section('analysis.threshold')
            reestimated_model = estimate_threshold_model(
                north_prices, 
                south_prices,
                threshold_params=threshold_params
            )
        
        # Add M-TAR estimation if original model had it
        if self.threshold_model and hasattr(self.threshold_model, 'mtar_results'):
            try:
                # Create ThresholdCointegration object if not already that type
                if not isinstance(reestimated_model, ThresholdCointegration):
                    tc_model = ThresholdCointegration(
                        north_prices.values, 
                        south_prices.values,
                        market1_name="North", 
                        market2_name="South"
                    )
                    tc_model.estimate_cointegration()
                    tc_model.estimate_threshold()
                    tc_model.estimate_tvecm()
                    reestimated_mtar = tc_model.estimate_mtar()
                else:
                    reestimated_mtar = reestimated_model.estimate_mtar()
                
                # Store M-TAR results in model
                reestimated_model.mtar_results = reestimated_mtar
            except Exception as e:
                logger.warning(f"Could not estimate M-TAR model: {e}")
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Threshold model re-estimation complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return reestimated_model
    
    @m1_optimized(parallel=True)
    @handle_errors(logger=logger)
    def _process_threshold_model_in_chunks(
        self, 
        north_prices: pd.Series, 
        south_prices: pd.Series, 
        chunk_size: int
    ) -> Any:
        """
        Process threshold model estimation in chunks for large datasets.
        
        Parameters
        ----------
        north_prices : pd.Series
            Price series for northern markets
        south_prices : pd.Series
            Price series for southern markets
        chunk_size : int
            Size of each chunk for processing
        
        Returns
        -------
        Any
            Re-estimated threshold model
        """
        # Convert to numpy arrays if not already
        north_array = north_prices.values if isinstance(north_prices, pd.Series) else north_prices
        south_array = south_prices.values if isinstance(south_prices, pd.Series) else south_prices
        
        # Calculate number of chunks
        n = len(north_array)
        n_chunks = (n + chunk_size - 1) // chunk_size  # Ceiling division
        
        logger.info(f"Processing data in {n_chunks} chunks of size {chunk_size}")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                futures.append(executor.submit(
                    self._estimate_threshold_for_chunk,
                    north_array[i:end],
                    south_array[i:end],
                    i,
                    end
                ))
            
            # Collect results as they complete
            chunk_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        chunk_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
        
        # Combine results from all chunks
        if not chunk_results:
            raise ValueError("No valid threshold model estimations from any chunk")
        
        # Strategy 1: Return the model with the lowest SSR
        best_model = min(chunk_results, key=lambda x: x.get('ssr', float('inf')))
        
        logger.info(f"Selected best model from {len(chunk_results)} chunks based on SSR")
        
        return best_model
    
    @handle_errors(logger=logger)
    def _estimate_threshold_for_chunk(
        self, 
        north_chunk: np.ndarray, 
        south_chunk: np.ndarray, 
        start_idx: int, 
        end_idx: int
    ) -> Any:
        """
        Estimate threshold model for a data chunk.
        
        Parameters
        ----------
        north_chunk : np.ndarray
            Chunk of north prices
        south_chunk : np.ndarray
            Chunk of south prices
        start_idx : int
            Start index of chunk
        end_idx : int
            End index of chunk
        
        Returns
        -------
        Any
            Threshold model for this chunk
        """
        logger.debug(f"Estimating threshold model for chunk {start_idx}:{end_idx}")
        
        threshold_params = config.get_section('analysis.threshold')
        
        try:
            # Estimate threshold model for this chunk
            chunk_model = estimate_threshold_model(
                north_chunk,
                south_chunk,
                threshold_params=threshold_params
            )
            
            # Store chunk indices in model for reference
            chunk_model.chunk_indices = (start_idx, end_idx)
            
            return chunk_model
        except Exception as e:
            logger.warning(f"Error estimating threshold model for chunk {start_idx}:{end_idx}: {e}")
            return None
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger)
    def simulate_improved_connectivity(
        self, 
        reduction_factor: float = 0.5
    ) -> Dict[str, Any]:
        """
        Simulate improved connectivity by reducing conflict barriers.
        
        The simulation follows these steps:
        1. Reduce conflict intensity metrics by the specified factor
        2. Create new spatial weights with reduced conflict
        3. Re-estimate spatial models to assess impact
        
        Parameters
        ----------
        reduction_factor : float, optional
            Factor by which to reduce conflict intensity (0.0-1.0)
            
        Returns
        -------
        Dict[str, Any]
            Simulation results including:
            - 'simulated_data': GeoDataFrame with reduced conflict intensity
            - 'spatial_weights': Recalculated spatial weights
            - 'spatial_model': Re-estimated spatial model
        """
        # Validate reduction factor
        if not 0 <= reduction_factor <= 1:
            raise ValueError("Reduction factor must be between 0 and 1")
        
        # Check if conflict intensity column exists
        conflict_col = 'conflict_intensity_normalized'
        if conflict_col not in self.data.columns:
            raise ValidationError(f"Data must contain '{conflict_col}' column")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Copy data for simulation (with memory optimization)
        sim_data = optimize_dataframe(self.data.copy())
        
        # Reduce conflict intensity
        original_conflict = sim_data[conflict_col].copy()
        sim_data[conflict_col] = original_conflict * (1 - reduction_factor)
        
        logger.info(
            f"Reduced conflict intensity by factor {reduction_factor:.2f} (mean: {original_conflict.mean():.4f} -> {sim_data[conflict_col].mean():.4f})"
        )
        
        # Recalculate spatial weights
        spatial_weights = self._recalculate_spatial_weights(sim_data, conflict_col)
        
        # Re-estimate spatial model if provided
        spatial_model = None
        if self.spatial_model is not None:
            spatial_model = self._reestimate_spatial_model(sim_data, spatial_weights)
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Connectivity improvement simulation complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Store results
        results = {
            'simulated_data': sim_data,
            'reduction_factor': reduction_factor,
            'spatial_weights': spatial_weights,
            'spatial_model': spatial_model
        }
        
        # Store in instance results
        self.results['improved_connectivity'] = results
        
        return results
    
    @disk_cache(cache_dir='.cache/simulations')
    @handle_errors(logger=logger)
    @m1_optimized(use_numba=True)
    def _recalculate_spatial_weights(
        self, 
        data: gpd.GeoDataFrame, 
        conflict_col: str
    ) -> Any:
        """
        Recalculate spatial weights with adjusted conflict intensity.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            GeoDataFrame with adjusted conflict intensity
        conflict_col : str
            Column containing conflict intensity
            
        Returns
        -------
        Any
            Spatial weights matrix
        """
        logger.info("Recalculating spatial weights with adjusted conflict intensity")
        
        # Get parameters from config
        params = config.get_section('analysis.spatial')
        k = params.get('knn', 5)
        conflict_weight = params.get('conflict_weight', 0.5)
        
        # Check if there's a regime boundary to consider
        regime_boundary_penalty = config.get('analysis.simulation.regime_boundary_penalty', 1.5)
        
        # Create conflict-adjusted weights with regime boundary consideration
        if 'exchange_rate_regime' in data.columns:
            # Create a regime boundary matrix
            from scipy.spatial.distance import pdist, squareform
            regimes = data['exchange_rate_regime'].values
            n = len(regimes)
            regime_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if regimes[i] != regimes[j]:
                        regime_matrix[i, j] = 1
            
            # Create adjusted weights with enhanced conflict and regime penalties
            weights = create_conflict_adjusted_weights(
                data,
                k=k,
                conflict_col=conflict_col,
                conflict_weight=conflict_weight,
                additional_cost_matrix=regime_matrix,
                additional_weight=regime_boundary_penalty
            )
        else:
            # Standard conflict-adjusted weights
            weights = create_conflict_adjusted_weights(
                data,
                k=k,
                conflict_col=conflict_col,
                conflict_weight=conflict_weight
            )
        
        return weights
    
    @timer
    @handle_errors(logger=logger)
    def _reestimate_spatial_model(
        self, 
        data: gpd.GeoDataFrame, 
        weights: Any
    ) -> Any:
        """
        Re-estimate spatial model with simulated data.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            Simulated data with modified conflict levels
        weights : Any
            Recalculated spatial weights
            
        Returns
        -------
        Any
            Re-estimated spatial model
        """
        # This method is placeholder if not implemented in the provided code
        # We'll implement basic functionality based on similar methods
        
        logger.info("Re-estimating spatial model with simulated data")
        
        try:
            # Import SpatialEconometrics from spatial.py
            from models.spatial import SpatialEconometrics
            
            # Create a new model instance
            spatial_model = SpatialEconometrics(data)
            
            # Set the weights directly
            spatial_model.weights = weights
            
            # Run basic tests
            if 'price' in data.columns:
                moran_result = spatial_model.moran_i_test('price')
                logger.info(f"Moran's I for simulated data: {moran_result['I']:.4f}, p-value: {moran_result['p_norm']:.4f}")
            
            # Estimate models if price column is available
            if 'price' in data.columns or 'simulated_price' in data.columns:
                price_col = 'simulated_price' if 'simulated_price' in data.columns else 'price'
                
                # Find variables that might be predictors
                potential_predictors = [col for col in data.columns if col not in 
                                        [price_col, 'geometry', 'date', 'market_id']]
                
                if len(potential_predictors) > 0:
                    # Take a subset of predictors to avoid overfitting
                    predictors = potential_predictors[:min(3, len(potential_predictors))]
                    
                    # Estimate spatial lag model
                    lag_model = spatial_model.spatial_lag_model(price_col, predictors)
                    spatial_model.lag_model = lag_model
                    
                    # Estimate spatial error model
                    error_model = spatial_model.spatial_error_model(price_col, predictors)
                    spatial_model.error_model = error_model
            
            return spatial_model
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not re-estimate spatial model: {e}")
            return None
    
    @timer
    @handle_errors(logger=logger)
    def calculate_integration_index(
        self,
        policy_scenario: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate market integration index before and after policy intervention.
        
        Parameters
        ----------
        policy_scenario : str, optional
            Which policy scenario to analyze
            
        Returns
        -------
        Dict[str, float]
            Integration indices before and after, with percentage improvement
        """
        # Import required function from spatial.py
        from models.spatial import market_integration_index
        
        # Determine which results to use
        if policy_scenario is not None:
            if policy_scenario not in self.results:
                raise ValueError(f"Policy scenario '{policy_scenario}' not found in results")
            results = self.results[policy_scenario]
        else:
            # Use latest results
            if not self.results:
                raise ValueError("No simulation results available")
            policy_scenario = list(self.results.keys())[-1]
            results = self.results[policy_scenario]
        
        # Get original and simulated data
        original_data = self.original_data
        simulated_data = results['simulated_data']
        
        # Get original weights from spatial model if available
        original_weights = None
        simulated_weights = None
        
        if self.spatial_model is not None and hasattr(self.spatial_model, 'weights'):
            original_weights = self.spatial_model.weights
        else:
            # Create basic weights as fallback
            from libpysal.weights import KNN
            original_weights = KNN.from_dataframe(original_data, k=5)
        
        # Get simulated weights if available, otherwise use original
        if 'spatial_weights' in results and results['spatial_weights'] is not None:
            simulated_weights = results['spatial_weights']
        else:
            simulated_weights = original_weights
        
        # Calculate integration indices if data has time and market information
        integration_indices = {}
        
        if all(col in original_data.columns for col in ['date', 'market_id']):
            # Calculate original integration index
            original_index = market_integration_index(
                original_data,
                original_weights,
                market_id_col='market_id',
                price_col='price',
                time_col='date'
            )
            
            # Calculate simulated integration index
            simulated_index = market_integration_index(
                simulated_data,
                simulated_weights,
                market_id_col='market_id',
                price_col='simulated_price',
                time_col='date'
            )
            
            # Calculate average index values
            if not original_index.empty and not simulated_index.empty:
                orig_avg = original_index.filter(like='integration_index').mean().mean()
                sim_avg = simulated_index.filter(like='integration_index').mean().mean()
                
                # Calculate improvement
                abs_improvement = sim_avg - orig_avg
                pct_improvement = (abs_improvement / orig_avg * 100) if orig_avg != 0 else float('inf')
                
                integration_indices = {
                    'original_index': orig_avg,
                    'simulated_index': sim_avg,
                    'absolute_improvement': abs_improvement,
                    'percentage_improvement': pct_improvement,
                    'detailed_original': original_index,
                    'detailed_simulated': simulated_index
                }
        
        # Store results and return
        self.results[f'{policy_scenario}_integration'] = integration_indices
        return integration_indices

    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger)
    def simulate_combined_policy(
        self, 
        exchange_rate_target: str = 'official',
        conflict_reduction: float = 0.5,
        reference_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate combined exchange rate unification and improved connectivity.
        
        Parameters
        ----------
        exchange_rate_target : str, optional
            Method to determine unified exchange rate
        conflict_reduction : float, optional
            Factor by which to reduce conflict intensity
        reference_date : str, optional
            Date to use for reference exchange rates
            
        Returns
        -------
        Dict[str, Any]
            Combined simulation results
        """
        logger.info("Simulating combined policy effects")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Copy data for simulation (optimize memory usage)
        sim_data = optimize_dataframe(self.data.copy())
        
        # Apply exchange rate unification
        sim_data = self._apply_exchange_unification(
            sim_data, exchange_rate_target, reference_date
        )
        
        # Apply improved connectivity
        sim_data, spatial_weights = self._apply_improved_connectivity(
            sim_data, conflict_reduction
        )
        
        # Calculate price changes
        price_changes = self._calculate_price_changes(
            original_prices=self.data['price'],
            simulated_prices=sim_data['simulated_price'],
            by_column='exchange_rate_regime'
        )
        
        # Re-estimate models
        results = self._reestimate_models(sim_data, spatial_weights)
        
        # Compile all results
        combined_results = {
            'simulated_data': sim_data,
            'unified_rate': results.get('unified_rate'),
            'reduction_factor': conflict_reduction,
            'price_changes': price_changes,
            'spatial_weights': spatial_weights,
            'threshold_model': results.get('threshold_model'),
            'spatial_model': results.get('spatial_model')
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Combined policy simulation complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Store in instance results
        self.results['combined_policy'] = combined_results
        
        return combined_results
    
    @handle_errors(logger=logger)
    def _apply_exchange_unification(
        self, 
        data: gpd.GeoDataFrame,
        target_rate: str,
        reference_date: Optional[str]
    ) -> gpd.GeoDataFrame:
        """
        Apply exchange rate unification to data.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            Data to modify
        target_rate : str
            Method to determine unified rate
        reference_date : str, optional
            Reference date for rates
            
        Returns
        -------
        gpd.GeoDataFrame
            Modified data
        """
        # Convert to USD
        self._convert_to_usd(data)
        
        # Determine unified rate
        unified_rate = self._determine_unified_rate(target_rate, reference_date)
        
        # Apply unified rate
        data['simulated_price'] = data['usd_price'] * unified_rate
        data['unified_rate'] = unified_rate
        
        return data
    
    @handle_errors(logger=logger)
    def _apply_improved_connectivity(
        self,
        data: gpd.GeoDataFrame,
        reduction_factor: float
    ) -> Tuple[gpd.GeoDataFrame, Any]:
        """
        Apply improved connectivity by reducing conflict.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            Data to modify
        reduction_factor : float
            Factor to reduce conflict by
            
        Returns
        -------
        Tuple[gpd.GeoDataFrame, Any]
            Modified data and recalculated spatial weights
        """
        conflict_col = 'conflict_intensity_normalized'
        spatial_weights = None
        
        if conflict_col in data.columns:
            # Reduce conflict intensity
            data[conflict_col] = data[conflict_col] * (1 - reduction_factor)
            
            # Recalculate spatial weights
            spatial_weights = self._recalculate_spatial_weights(data, conflict_col)
        else:
            logger.warning(f"Column '{conflict_col}' not found, skipping conflict reduction")
        
        return data, spatial_weights
    
    @handle_errors(logger=logger)
    def _reestimate_models(
        self,
        data: gpd.GeoDataFrame,
        spatial_weights: Any
    ) -> Dict[str, Any]:
        """
        Re-estimate models with simulated data.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            Simulated data
        spatial_weights : Any
            Recalculated spatial weights
            
        Returns
        -------
        Dict[str, Any]
            Re-estimated models
        """
        results = {}
        
        # Re-estimate threshold model if available
        if self.threshold_model is not None:
            results['threshold_model'] = self._reestimate_threshold_model(data)
        
        # Re-estimate spatial model if available
        if self.spatial_model is not None and spatial_weights is not None:
            results['spatial_model'] = self._reestimate_spatial_model(
                data, spatial_weights
            )
        
        # Include unified rate if available
        if 'unified_rate' in data.columns:
            results['unified_rate'] = data['unified_rate'].iloc[0]
        
        return results
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger)
    def simulate_combined_policies(
        self, 
        policy_combinations: List[Dict[str, Any]],
        parallelize: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Simulate multiple policy combinations to analyze interactions.
        
        This enhanced method allows for simulating multiple policy combinations 
        simultaneously and analyzing their interactions. It can process different
        scenarios in parallel for improved performance.
        
        Parameters
        ----------
        policy_combinations : List[Dict[str, Any]]
            List of policy parameter dictionaries, each containing:
            - 'exchange_rate_target': str or float (optional)
            - 'conflict_reduction': float (optional)
            - 'policy_name': str (optional, for labeling)
        parallelize : bool, optional
            Whether to process policy combinations in parallel
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Results for each policy combination and their interactions
        """
        logger.info(f"Simulating {len(policy_combinations)} policy combinations")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # If no policy combinations provided, use defaults
        if not policy_combinations:
            default_exchange = config.get('analysis.simulation.exchange_rate_default', 'official')
            default_conflict = config.get('analysis.simulation.conflict_reduction_default', 0.5)
            
            policy_combinations = [
                {'exchange_rate_target': default_exchange, 'policy_name': 'exchange_rate_only'},
                {'conflict_reduction': default_conflict, 'policy_name': 'connectivity_only'},
                {'exchange_rate_target': default_exchange, 'conflict_reduction': default_conflict, 
                 'policy_name': 'combined'}
            ]
        
        # Generate names for unnamed policy combinations
        for i, policy in enumerate(policy_combinations):
            if 'policy_name' not in policy:
                components = []
                if 'exchange_rate_target' in policy:
                    components.append(f"er_{policy['exchange_rate_target']}")
                if 'conflict_reduction' in policy:
                    components.append(f"cr_{policy['conflict_reduction']}")
                
                policy['policy_name'] = f"policy_{i}_{'-'.join(components)}"
        
        # Create folder for results
        all_results = {}
        
        # Process policy combinations
        if parallelize and len(policy_combinations) > 1:
            # Use parallel processing for multiple policy combinations
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit tasks
                futures = {}
                for policy in policy_combinations:
                    future = executor.submit(
                        self._simulate_single_policy_combination,
                        policy
                    )
                    futures[future] = policy['policy_name']
                
                # Collect results and monitor progress
                total_policies = len(policy_combinations)
                completed = 0
                
                for future in as_completed(futures):
                    policy_name = futures[future]
                    try:
                        result = future.result()
                        all_results[policy_name] = result
                        
                        # Log progress
                        completed += 1
                        progress_pct = (completed / total_policies) * 100
                        logger.info(f"Completed policy simulation for '{policy_name}' ({completed}/{total_policies}, {progress_pct:.1f}%)")
                    except Exception as e:
                        logger.error(f"Error in policy simulation '{policy_name}': {e}")
        else:
            # Process sequentially
            for i, policy in enumerate(policy_combinations):
                policy_name = policy['policy_name']
                try:
                    logger.info(f"Processing policy combination {i+1}/{len(policy_combinations)}: '{policy_name}'")
                    result = self._simulate_single_policy_combination(policy)
                    all_results[policy_name] = result
                except Exception as e:
                    logger.error(f"Error in policy simulation '{policy_name}': {e}")
        
        # Calculate interaction effects if multiple policies
        if len(all_results) >= 2:
            interaction_results = self._analyze_policy_interactions(all_results)
            all_results['interaction_analysis'] = interaction_results
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Multi-policy simulation complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Store all results
        self.results['combined_policies'] = all_results
        
        return all_results
    
    @handle_errors(logger=logger)
    def _simulate_single_policy_combination(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a single policy combination.
        
        Parameters
        ----------
        policy : Dict[str, Any]
            Policy parameters
            
        Returns
        -------
        Dict[str, Any]
            Simulation results for this policy
        """
        # Extract parameters with defaults
        exchange_rate_target = policy.get('exchange_rate_target', None)
        conflict_reduction = policy.get('conflict_reduction', None)
        reference_date = policy.get('reference_date', None)
        
        # Determine which simulation to run
        if exchange_rate_target is not None and conflict_reduction is not None:
            # Combined policy
            result = self.simulate_combined_policy(
                exchange_rate_target=exchange_rate_target,
                conflict_reduction=conflict_reduction,
                reference_date=reference_date
            )
        elif exchange_rate_target is not None:
            # Exchange rate unification only
            result = self.simulate_exchange_rate_unification(
                target_rate=exchange_rate_target,
                reference_date=reference_date
            )
        elif conflict_reduction is not None:
            # Improved connectivity only
            result = self.simulate_improved_connectivity(
                reduction_factor=conflict_reduction
            )
        else:
            raise ValueError("Policy must specify at least one of: exchange_rate_target, conflict_reduction")
        
        # Calculate welfare effects
        policy_name = policy.get('policy_name', 'unnamed_policy')
        welfare_key = f"{policy_name}_welfare"
        
        # Store policy parameters in result
        result['policy_parameters'] = policy
        
        return result
    
    @handle_errors(logger=logger)
    def _analyze_policy_interactions(self, policy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze interactions between multiple policy interventions.
        
        Parameters
        ----------
        policy_results : Dict[str, Dict[str, Any]]
            Results from multiple policy simulations
            
        Returns
        -------
        Dict[str, Any]
            Interaction analysis
        """
        logger.info("Analyzing policy interactions")
        
        # Extract individual and combined policies
        exchange_only = None
        connectivity_only = None
        combined_policy = None
        
        for name, result in policy_results.items():
            params = result.get('policy_parameters', {})
            
            # Check if this is an exchange rate only policy
            if 'exchange_rate_target' in params and 'conflict_reduction' not in params:
                exchange_only = (name, result)
            
            # Check if this is a connectivity only policy
            elif 'conflict_reduction' in params and 'exchange_rate_target' not in params:
                connectivity_only = (name, result)
            
            # Check if this is a combined policy
            elif 'exchange_rate_target' in params and 'conflict_reduction' in params:
                combined_policy = (name, result)
        
        # If we have all three policy types, analyze interactions
        interaction_results = {}
        
        if exchange_only and connectivity_only and combined_policy:
            # Extract policy names for reference
            ex_name, ex_result = exchange_only
            con_name, con_result = connectivity_only
            comb_name, comb_result = combined_policy
            
            # Calculate expected combined effect (additive)
            expected_combined = {}
            
            # Compare price changes
            if all(key in policy_results for key in [ex_name, con_name, comb_name]):
                # Extract price changes
                try:
                    ex_changes = ex_result.get('price_changes', None)
                    con_changes = con_result.get('price_changes', None)
                    comb_changes = comb_result.get('price_changes', None)
                    
                    # If all have price changes, analyze interactions
                    if ex_changes is not None and con_changes is not None and comb_changes is not None:
                        # Calculate mean price changes
                        if isinstance(ex_changes, pd.DataFrame) and 'pct_change' in ex_changes.columns:
                            ex_mean = ex_changes['pct_change'].mean()
                            con_mean = con_changes['pct_change'].mean()
                            comb_mean = comb_changes['pct_change'].mean()
                            
                            # Expected combined effect (additive)
                            expected_mean = ex_mean + con_mean
                            
                            # Calculate interaction effect
                            interaction_effect = comb_mean - expected_mean
                            
                            interaction_results['price_changes'] = {
                                'exchange_only': ex_mean,
                                'connectivity_only': con_mean,
                                'combined_actual': comb_mean,
                                'combined_expected': expected_mean,
                                'interaction_effect': interaction_effect,
                                'synergy': interaction_effect > 0,
                                'percent_difference': (interaction_effect / abs(expected_mean) * 100) if expected_mean != 0 else np.nan
                            }
                except Exception as e:
                    logger.warning(f"Error calculating price change interactions: {e}")
            
            # Compare integration indices
            try:
                # Calculate integration indices if not already done
                for name, result in policy_results.items():
                    if f"{name}_integration" not in self.results:
                        self.calculate_integration_index(name)
                
                # Extract integration improvements
                ex_integration = self.results.get(f"{ex_name}_integration", {}).get('percentage_improvement', 0)
                con_integration = self.results.get(f"{con_name}_integration", {}).get('percentage_improvement', 0)
                comb_integration = self.results.get(f"{comb_name}_integration", {}).get('percentage_improvement', 0)
                
                # Expected combined effect (additive)
                expected_integration = ex_integration + con_integration
                
                # Calculate interaction effect
                interaction_effect = comb_integration - expected_integration
                
                interaction_results['integration_index'] = {
                    'exchange_only': ex_integration,
                    'connectivity_only': con_integration,
                    'combined_actual': comb_integration,
                    'combined_expected': expected_integration,
                    'interaction_effect': interaction_effect,
                    'synergy': interaction_effect > 0,
                    'percent_difference': (interaction_effect / abs(expected_integration) * 100) if expected_integration != 0 else np.nan
                }
            except Exception as e:
                logger.warning(f"Error calculating integration index interactions: {e}")
            
            # Generate interpretation of interactions
            interaction_results['interpretation'] = self._interpret_policy_interactions(interaction_results)
        
        return interaction_results
    
    @handle_errors(logger=logger)
    def _interpret_policy_interactions(self, interaction_results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate interpretation of policy interaction effects.
        
        Parameters
        ----------
        interaction_results : Dict[str, Dict[str, float]]
            Results of interaction analysis
            
        Returns
        -------
        str
            Interpretation of interaction effects
        """
        price_synergy = interaction_results.get('price_changes', {}).get('synergy', False)
        price_effect = interaction_results.get('price_changes', {}).get('interaction_effect', 0)
        
        integration_synergy = interaction_results.get('integration_index', {}).get('synergy', False)
        integration_effect = interaction_results.get('integration_index', {}).get('interaction_effect', 0)
        
        # Generate interpretation based on results
        if price_synergy and integration_synergy:
            interpretation = (
                "The combined policy shows strong positive synergies. "
                f"Price changes are {abs(price_effect):.2f}% better than expected, "
                f"and market integration is {abs(integration_effect):.2f}% better than the sum of individual policies. "
                "This suggests complementary effects between exchange rate unification and connectivity improvements."
            )
        elif price_synergy and not integration_synergy:
            interpretation = (
                "The combined policy shows mixed effects. "
                f"Price changes are {abs(price_effect):.2f}% better than expected, "
                f"but market integration is {abs(integration_effect):.2f}% worse than the sum of individual policies. "
                "This suggests that while prices converge more, market efficiency may be hindered by other factors."
            )
        elif not price_synergy and integration_synergy:
            interpretation = (
                "The combined policy shows mixed effects. "
                f"Price changes are {abs(price_effect):.2f}% worse than expected, "
                f"but market integration is {abs(integration_effect):.2f}% better than the sum of individual policies. "
                "This suggests improved efficiency despite smaller price convergence."
            )
        else:
            interpretation = (
                "The combined policy shows negative interactions. "
                f"Price changes are {abs(price_effect):.2f}% worse than expected, "
                f"and market integration is {abs(integration_effect):.2f}% worse than the sum of individual policies. "
                "This suggests conflicting effects between exchange rate unification and connectivity improvements."
            )
        
        return interpretation
    
    @timer
    @handle_errors(logger=logger)
    def calculate_welfare_effects(
        self, 
        policy_scenario: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate welfare effects of policy simulations.
        
        Parameters
        ----------
        policy_scenario : str, optional
            Which policy scenario to analyze:
            - 'exchange_rate_unification'
            - 'improved_connectivity'
            - 'combined_policy'
            If None, uses latest simulation results
            
        Returns
        -------
        Dict[str, Any]
            Welfare analysis results
        """
        # Determine which results to use
        if policy_scenario is not None:
            if policy_scenario not in self.results:
                raise ValueError(f"Policy scenario '{policy_scenario}' not found in results")
            results = self.results[policy_scenario]
        else:
            # Use latest results
            if not self.results:
                raise ValueError("No simulation results available")
            policy_scenario = list(self.results.keys())[-1]
            results = self.results[policy_scenario]
        
        logger.info(f"Calculating welfare effects for '{policy_scenario}' scenario")
        
        # Get simulated data
        sim_data = results['simulated_data']
        
        # Calculate regional price metrics
        regional_metrics = self._calculate_regional_welfare_metrics(sim_data)
        
        # Calculate commodity-specific effects if possible
        commodity_effects = self._calculate_commodity_effects(sim_data)
        
        # Calculate price convergence metrics
        price_convergence = self._calculate_price_convergence(
            self.data, sim_data, 'price', 'simulated_price', 'exchange_rate_regime'
        )
        
        # Check for M-TAR results for additional metrics
        mtar_metrics = {}
        if self.threshold_model and hasattr(self.threshold_model, 'mtar_results'):
            original_mtar = self.threshold_model.mtar_results
            
            # Check if simulated results have M-TAR
            if 'threshold_model' in results and hasattr(results['threshold_model'], 'mtar_results'):
                simulated_mtar = results['threshold_model'].mtar_results
                
                # Calculate changes in asymmetry
                mtar_metrics = {
                    'original_asymmetry': original_mtar.get('asymmetric', False),
                    'simulated_asymmetry': simulated_mtar.get('asymmetric', False),
                    'original_pvalue': original_mtar.get('p_value', 1.0),
                    'simulated_pvalue': simulated_mtar.get('p_value', 1.0),
                    'half_life_change_positive': simulated_mtar.get('half_life_positive', np.inf) - 
                                               original_mtar.get('half_life_positive', np.inf),
                    'half_life_change_negative': simulated_mtar.get('half_life_negative', np.inf) - 
                                               original_mtar.get('half_life_negative', np.inf)
                }
        
        # Compile welfare results
        welfare_results = {
            'policy_scenario': policy_scenario,
            'regional_metrics': regional_metrics,
            'price_convergence': price_convergence,
            'commodity_effects': commodity_effects,
            'mtar_metrics': mtar_metrics
        }
        
        # Store in results
        self.results[f'{policy_scenario}_welfare'] = welfare_results
        
        return welfare_results
    
    @timer
    @handle_errors(logger=logger)
    def calculate_policy_asymmetry_effects(
        self,
        policy_scenario: str
    ) -> Dict[str, Any]:
        """
        Analyze asymmetric policy response effects.
        
        Examines how policy interventions affect asymmetric price adjustment 
        patterns, which can reveal changes in market power or barriers.
        
        Parameters
        ----------
        policy_scenario : str
            Which policy scenario to analyze
            
        Returns
        -------
        Dict[str, Any]
            Asymmetric adjustment changes after policy
        """
        if policy_scenario not in self.results:
            raise ValueError(f"Policy scenario '{policy_scenario}' not found in results")
        
        results = self.results[policy_scenario]
        
        # Check if we have threshold models
        if 'threshold_model' not in results or results['threshold_model'] is None:
            return {'asymmetry_analysis_possible': False}
        
        # Original threshold model
        original_model = self.threshold_model
        simulated_model = results['threshold_model']
        
        # Extract regular TAR asymmetry results
        original_asymmetry = original_model.results.get('asymmetric_adjustment', {})
        simulated_asymmetry = simulated_model.results.get('asymmetric_adjustment', {})
        
        # Extract M-TAR results if available
        original_mtar = getattr(original_model, 'mtar_results', None)
        simulated_mtar = getattr(simulated_model, 'mtar_results', None)
        
        # Compare asymmetry metrics
        tar_comparison = {
            'original_asymmetric': original_asymmetry.get('asymmetry_1', 0),
            'simulated_asymmetric': simulated_asymmetry.get('asymmetry_1', 0),
            'change': simulated_asymmetry.get('asymmetry_1', 0) - original_asymmetry.get('asymmetry_1', 0),
            'half_life_below_change': (
                simulated_asymmetry.get('half_life_below_1', float('inf')) - 
                original_asymmetry.get('half_life_below_1', float('inf'))
            ),
            'half_life_above_change': (
                simulated_asymmetry.get('half_life_above_1', float('inf')) - 
                original_asymmetry.get('half_life_above_1', float('inf'))
            ),
        }
        
        # M-TAR comparison if available
        mtar_comparison = None
        if original_mtar is not None and simulated_mtar is not None:
            mtar_comparison = {
                'original_asymmetric': original_mtar.get('asymmetric', False),
                'simulated_asymmetric': simulated_mtar.get('asymmetric', False),
                'half_life_positive_change': (
                    simulated_mtar.get('half_life_positive', float('inf')) - 
                    original_mtar.get('half_life_positive', float('inf'))
                ),
                'half_life_negative_change': (
                    simulated_mtar.get('half_life_negative', float('inf')) - 
                    original_mtar.get('half_life_negative', float('inf'))
                ),
            }
        
        # Interpretation - did policy reduce asymmetry?
        interpretation = self._interpret_asymmetry_changes(tar_comparison, mtar_comparison)
        
        asymmetry_effects = {
            'tar_comparison': tar_comparison,
            'mtar_comparison': mtar_comparison,
            'interpretation': interpretation
        }
        
        # Store in policy results
        self.results[f'{policy_scenario}_asymmetry'] = asymmetry_effects
        
        return asymmetry_effects

    @handle_errors(logger=logger)
    def _interpret_asymmetry_changes(
        self,
        tar_comparison: Dict[str, Any],
        mtar_comparison: Optional[Dict[str, Any]] = None
    ) -> str:
        """Interpret changes in asymmetric adjustment after policy intervention."""
        tar_asymmetry_reduced = tar_comparison.get('change', 0) < 0
        tar_hl_below_reduced = tar_comparison.get('half_life_below_change', 0) < 0
        tar_hl_above_reduced = tar_comparison.get('half_life_above_change', 0) < 0
        
        if tar_asymmetry_reduced:
            main_finding = "The policy reduced asymmetric price adjustment, indicating more balanced market integration."
        else:
            main_finding = "The policy did not reduce asymmetric price adjustment, suggesting persistent market barriers."
        
        if tar_hl_below_reduced and tar_hl_above_reduced:
            speed_finding = "Adjustment speeds increased in both regimes, suggesting overall improved market efficiency."
        elif tar_hl_below_reduced:
            speed_finding = "Adjustment speeds increased only for small price differentials, suggesting partial barrier reduction."
        elif tar_hl_above_reduced:
            speed_finding = "Adjustment speeds increased only for large price differentials, suggesting reduced transaction costs."
        else:
            speed_finding = "No improvement in adjustment speeds was observed in either regime."
        
        mtar_insight = ""
        if mtar_comparison is not None:
            mtar_improved = (
                mtar_comparison.get('half_life_positive_change', 0) < 0 or 
                mtar_comparison.get('half_life_negative_change', 0) < 0
            )
            if mtar_improved:
                direction = "rising" if mtar_comparison.get('half_life_positive_change', 0) < 0 else "falling"
                mtar_insight = f" M-TAR analysis reveals improved adjustment to {direction} prices, indicating directional integration improvements."
        
        return f"{main_finding} {speed_finding}{mtar_insight}"
    
    @handle_errors(logger=logger)
    def _calculate_regional_welfare_metrics(
        self, 
        sim_data: gpd.GeoDataFrame
    ) -> Dict[str, Any]:
        """
        Calculate welfare metrics by region.
        
        Parameters
        ----------
        sim_data : gpd.GeoDataFrame
            Simulated data
            
        Returns
        -------
        Dict[str, Any]
            Regional welfare metrics
        """
        # Calculate regional price dispersion
        original_dispersion = self._calculate_price_dispersion(
            self.data, 'price', 'exchange_rate_regime'
        )
        
        simulated_dispersion = self._calculate_price_dispersion(
            sim_data, 'simulated_price', 'exchange_rate_regime'
        )
        
        # Calculate price changes by region
        price_changes = self._calculate_price_changes(
            original_prices=self.data['price'],
            simulated_prices=sim_data['simulated_price'],
            by_column='exchange_rate_regime'
        )
        
        return {
            'price_dispersion': {
                'original': original_dispersion,
                'simulated': simulated_dispersion,
                'change': {
                    region: {
                        'abs_change': simulated_dispersion.get(region, 0) - orig,
                        'pct_change': ((simulated_dispersion.get(region, 0) - orig) / orig * 100 
                                      if orig != 0 else np.nan)
                    }
                    for region, orig in original_dispersion.items()
                }
            },
            'price_changes': price_changes
        }
    
    @handle_errors(logger=logger)
    def _calculate_commodity_effects(
        self, 
        sim_data: gpd.GeoDataFrame
    ) -> Dict[str, Any]:
        """
        Calculate welfare effects by commodity.
        
        Parameters
        ----------
        sim_data : gpd.GeoDataFrame
            Simulated data
            
        Returns
        -------
        Dict[str, Any]
            Commodity-specific welfare effects
        """
        commodity_effects = {}
        
        if 'commodity' in sim_data.columns:
            for commodity in sim_data['commodity'].unique():
                # Filter data for this commodity
                commodity_sim = sim_data[sim_data['commodity'] == commodity]
                commodity_orig = self.data[self.data['commodity'] == commodity]
                
                # Calculate price changes
                changes = self._calculate_price_changes(
                    original_prices=commodity_orig['price'],
                    simulated_prices=commodity_sim['simulated_price'],
                    by_column='exchange_rate_regime'
                )
                
                # Calculate dispersion changes
                dispersion_change = self._calculate_price_dispersion_change(
                    commodity_orig, commodity_sim, 'price', 'simulated_price', 
                    'exchange_rate_regime'
                )
                
                commodity_effects[commodity] = {
                    'price_changes': changes,
                    'dispersion_change': dispersion_change
                }
        
        return commodity_effects
    
    @handle_errors(logger=logger)
    def _calculate_price_dispersion(
        self, 
        data: gpd.GeoDataFrame, 
        price_col: str, 
        group_col: str
    ) -> Dict[str, float]:
        """
        Calculate price dispersion by group.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            Data to analyze
        price_col : str
            Column containing prices
        group_col : str
            Column to group by
            
        Returns
        -------
        Dict[str, float]
            Price dispersion (coefficient of variation) by group
        """
        result = {}
        
        for group in data[group_col].unique():
            group_data = data[data[group_col] == group][price_col]
            # Calculate coefficient of variation (std/mean)
            cv = group_data.std() / group_data.mean() if group_data.mean() != 0 else np.nan
            result[group] = cv
        
        return result
    
    @handle_errors(logger=logger)
    def _calculate_price_dispersion_change(
        self, 
        original_data: gpd.GeoDataFrame, 
        simulated_data: gpd.GeoDataFrame, 
        original_price_col: str, 
        simulated_price_col: str, 
        group_col: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate changes in price dispersion between original and simulated data.
        
        Parameters
        ----------
        original_data : gpd.GeoDataFrame
            Original data
        simulated_data : gpd.GeoDataFrame
            Simulated data
        original_price_col : str
            Column with original prices
        simulated_price_col : str
            Column with simulated prices
        group_col : str
            Column to group by
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Changes in price dispersion by group
        """
        # Calculate original and simulated dispersion
        original_dispersion = self._calculate_price_dispersion(
            original_data, original_price_col, group_col
        )
        simulated_dispersion = self._calculate_price_dispersion(
            simulated_data, simulated_price_col, group_col
        )
        
        # Calculate changes
        result = {}
        for group in original_dispersion:
            orig = original_dispersion[group]
            sim = simulated_dispersion.get(group, np.nan)
            
            if np.isnan(orig) or np.isnan(sim):
                abs_change = np.nan
                pct_change = np.nan
            else:
                abs_change = sim - orig
                pct_change = (abs_change / orig) * 100 if orig != 0 else np.nan
            
            result[group] = {
                'original': orig,
                'simulated': sim,
                'abs_change': abs_change,
                'pct_change': pct_change
            }
        
        return result
    
    @handle_errors(logger=logger)
    def _calculate_price_convergence(
        self, 
        original_data: gpd.GeoDataFrame, 
        simulated_data: gpd.GeoDataFrame, 
        original_price_col: str, 
        simulated_price_col: str, 
        regime_col: str
    ) -> Dict[str, float]:
        """
        Calculate price convergence metrics between regions.
        
        Parameters
        ----------
        original_data : gpd.GeoDataFrame
            Original data
        simulated_data : gpd.GeoDataFrame
            Simulated data
        original_price_col : str
            Column with original prices
        simulated_price_col : str
            Column with simulated prices
        regime_col : str
            Column with exchange rate regime
            
        Returns
        -------
        Dict[str, float]
            Price convergence metrics
        """
        # Ensure we have north and south regimes
        regimes = original_data[regime_col].unique()
        if 'north' not in regimes or 'south' not in regimes:
            logger.warning(f"North and south regimes not both present in data: {regimes}")
            return {'convergence_possible': False}
        
        # Calculate average price differential between regions
        def calc_diff(data, price_col):
            north = data[data[regime_col] == 'north'][price_col].mean()
            south = data[data[regime_col] == 'south'][price_col].mean()
            abs_diff = abs(north - south)
            rel_diff = abs_diff / ((north + south) / 2) * 100 if (north + south) != 0 else np.nan
            return {'north': north, 'south': south, 'abs_diff': abs_diff, 'rel_diff': rel_diff}
        
        original_diff = calc_diff(original_data, original_price_col)
        simulated_diff = calc_diff(simulated_data, simulated_price_col)
        
        # Calculate convergence
        abs_convergence = original_diff['abs_diff'] - simulated_diff['abs_diff']
        rel_convergence = abs_convergence / original_diff['abs_diff'] * 100 if original_diff['abs_diff'] != 0 else np.nan
        
        return {
            'original_difference': original_diff,
            'simulated_difference': simulated_diff,
            'absolute_convergence': abs_convergence,
            'relative_convergence': rel_convergence
        }
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger)
    def test_robustness(
        self, 
        policy_scenario: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform robustness checks on simulation results.
        
        This method tests the robustness of simulation results by:
        1. Testing for structural breaks in price series before and after simulation
        2. Running comprehensive residual diagnostics on model results
        3. Testing model stability across different subsamples
        
        Parameters
        ----------
        policy_scenario : str, optional
            Which policy scenario to analyze
            
        Returns
        -------
        Dict[str, Any]
            Robustness test results
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Determine which results to use
        if policy_scenario is not None:
            if policy_scenario not in self.results:
                raise ValueError(f"Policy scenario '{policy_scenario}' not found in results")
            results = self.results[policy_scenario]
        else:
            # Use latest results
            if not self.results:
                raise ValueError("No simulation results available")
            policy_scenario = list(self.results.keys())[-1]
            results = self.results[policy_scenario]
        
        logger.info(f"Testing robustness of '{policy_scenario}' scenario")
        
        # Get original and simulated data
        original_data = optimize_dataframe(self.original_data.copy())
        simulated_data = optimize_dataframe(results['simulated_data'].copy())
        
        # Split robustness testing into smaller, focused tasks
        structural_break_results = self._test_structural_breaks(original_data, simulated_data)
        diagnostic_results = self._test_residual_diagnostics(results)
        stability_results = self._test_model_stability(original_data, simulated_data, results)
        
        # Compile all results
        robustness_results = {
            'structural_breaks': structural_break_results,
            'diagnostics': diagnostic_results,
            'stability': stability_results,
            'overall_assessment': self._assess_overall_robustness(
                structural_break_results, diagnostic_results, stability_results
            )
        }
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Robustness testing complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Store results and return
        self.results[f'{policy_scenario}_robustness'] = robustness_results
        return robustness_results
    
    @handle_errors(logger=logger)
    def _test_structural_breaks(
        self, 
        original_data: gpd.GeoDataFrame, 
        simulated_data: gpd.GeoDataFrame
    ) -> Dict[str, Any]:
        """
        Test for structural breaks in price series before and after simulation.
        
        Parameters
        ----------
        original_data : gpd.GeoDataFrame
            Original data
        simulated_data : gpd.GeoDataFrame
            Simulated data
            
        Returns
        -------
        Dict[str, Any]
            Structural break test results
        """
        # Import necessary class
        from models.unit_root import StructuralBreakTester
        
        logger.info("Testing for structural breaks")
        
        # Initialize results dictionary
        structural_break_results = {}
        
        # Create break tester
        structural_break_tester = StructuralBreakTester()
        
        try:
            # Test for structural breaks in original price series
            if 'price' in original_data.columns:
                original_breaks = structural_break_tester.test_bai_perron(
                    original_data['price'], 
                    min_size=10,
                    n_breaks=3
                )
                structural_break_results['original_structural_breaks'] = original_breaks
            
            # Test for structural breaks in simulated price series
            if 'simulated_price' in simulated_data.columns:
                simulated_breaks = structural_break_tester.test_bai_perron(
                    simulated_data['simulated_price'], 
                    min_size=10,
                    n_breaks=3
                )
                structural_break_results['simulated_structural_breaks'] = simulated_breaks
                
                # Compare break dates/locations to see if policy changed market structure
                original_bps = original_breaks.get('breakpoints', [])
                simulated_bps = simulated_breaks.get('breakpoints', [])
                
                structural_break_results['break_comparison'] = {
                    'original_breakpoints': original_bps,
                    'simulated_breakpoints': simulated_bps,
                    'structural_change': len(original_bps) != len(simulated_bps) or not all(abs(o - s) < 5 for o, s in zip(original_bps, simulated_bps))
                }
                
                # Calculate break significance
                original_significance = original_breaks.get('significant', False)
                simulated_significance = simulated_breaks.get('significant', False)
                
                structural_break_results['break_significance'] = {
                    'original_significant': original_significance,
                    'simulated_significant': simulated_significance,
                    'significance_changed': original_significance != simulated_significance
                }
                
            logger.info("Structural break testing complete")
            
        except Exception as e:
            logger.error(f"Error in structural break testing: {e}")
            structural_break_results['error'] = str(e)
        
        return structural_break_results
    
    @handle_errors(logger=logger)
    def _test_residual_diagnostics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run diagnostic tests on model residuals.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Simulation results with models
            
        Returns
        -------
        Dict[str, Any]
            Residual diagnostic test results
        """
        # Import necessary class
        from models.diagnostics import ModelDiagnostics
        
        logger.info("Running residual diagnostics")
        
        # Initialize results
        diagnostic_results = {}
        
        try:
            # Check if threshold model is available
            if 'threshold_model' in results and self.threshold_model is not None:
                # Create diagnostics object
                diagnostics = ModelDiagnostics(
                    model_name=f"threshold",
                    original_data=self.original_data
                )
                
                # Get residuals from original and simulated models
                if hasattr(self.threshold_model, 'eq_errors') and hasattr(results['threshold_model'], 'eq_errors'):
                    original_residuals = self.threshold_model.eq_errors
                    simulated_residuals = results['threshold_model'].eq_errors
                    
                    # Run diagnostics on original residuals
                    original_diagnostics = diagnostics.residual_tests(original_residuals)
                    diagnostic_results['original_residual_diagnostics'] = original_diagnostics
                    
                    # Run diagnostics on simulated residuals
                    diagnostics.residuals = simulated_residuals
                    simulated_diagnostics = diagnostics.residual_tests(simulated_residuals)
                    diagnostic_results['simulated_residual_diagnostics'] = simulated_diagnostics
                    
                    # Compare diagnostic results
                    diagnostic_results['diagnostic_comparison'] = self._compare_diagnostics(
                        original_diagnostics, simulated_diagnostics
                    )
                else:
                    logger.warning("Threshold models missing residuals for diagnostics")
            
            logger.info("Residual diagnostics complete")
            
        except Exception as e:
            logger.error(f"Error in residual diagnostics: {e}")
            diagnostic_results['error'] = str(e)
        
        return diagnostic_results
    
    @handle_errors(logger=logger)
    def _test_model_stability(
        self, 
        original_data: gpd.GeoDataFrame, 
        simulated_data: gpd.GeoDataFrame, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test model stability across different subsamples.
        
        Parameters
        ----------
        original_data : gpd.GeoDataFrame
            Original data
        simulated_data : gpd.GeoDataFrame
            Simulated data
        results : Dict[str, Any]
            Simulation results with models
            
        Returns
        -------
        Dict[str, Any]
            Model stability test results
        """
        logger.info("Testing model stability")
        
        # Initialize results
        stability_results = {}
        
        try:
            # Check if data has time information for subsample testing
            if 'date' in original_data.columns:
                # Test stability across time periods
                stability_results['time_stability'] = self._test_time_stability(
                    original_data, simulated_data, results
                )
            
            # Test stability across regions if exchange regime column exists
            if 'exchange_rate_regime' in original_data.columns:
                stability_results['regional_stability'] = self._test_regional_stability(
                    original_data, simulated_data, results
                )
            
            # Test parameter stability if threshold model is available
            if 'threshold_model' in results and self.threshold_model is not None:
                stability_results['parameter_stability'] = self._test_parameter_stability(
                    self.threshold_model, results['threshold_model']
                )
            
            logger.info("Model stability testing complete")
            
        except Exception as e:
            logger.error(f"Error in model stability testing: {e}")
            stability_results['error'] = str(e)
        
        return stability_results
    
    @handle_errors(logger=logger)
    def _test_time_stability(
        self, 
        original_data: gpd.GeoDataFrame, 
        simulated_data: gpd.GeoDataFrame, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test model stability across different time periods.
        
        Parameters
        ----------
        original_data : gpd.GeoDataFrame
            Original data
        simulated_data : gpd.GeoDataFrame
            Simulated data
        results : Dict[str, Any]
            Simulation results with models
            
        Returns
        -------
        Dict[str, Any]
            Time stability test results
        """
        # Get dates and sort
        dates = sorted(original_data['date'].unique())
        
        # Skip if too few time periods
        if len(dates) < 3:
            return {'sufficient_data': False, 'periods': len(dates)}
        
        # Split into early, middle, late periods
        early_dates = dates[:len(dates)//3]
        late_dates = dates[-len(dates)//3:]
        
        # Calculate price differentials for early and late periods
        early_original = original_data[original_data['date'].isin(early_dates)]
        early_simulated = simulated_data[simulated_data['date'].isin(early_dates)]
        
        late_original = original_data[original_data['date'].isin(late_dates)]
        late_simulated = simulated_data[simulated_data['date'].isin(late_dates)]
        
        # Calculate convergence for early and late periods
        early_convergence = self._calculate_price_convergence(
            early_original, early_simulated, 'price', 'simulated_price', 'exchange_rate_regime'
        )
        
        late_convergence = self._calculate_price_convergence(
            late_original, late_simulated, 'price', 'simulated_price', 'exchange_rate_regime'
        )
        
        # Compare convergence across periods
        early_rel_conv = early_convergence.get('relative_convergence', 0)
        late_rel_conv = late_convergence.get('relative_convergence', 0)
        
        # Calculate stability metrics
        abs_diff = abs(late_rel_conv - early_rel_conv)
        rel_diff = abs_diff / abs(early_rel_conv) * 100 if early_rel_conv != 0 else float('inf')
        
        # Determine if results are stable across time
        # (Below stability threshold from config)
        stability_threshold = config.get('analysis.simulation.stability_threshold', 20)  # 20% default
        is_stable = rel_diff < stability_threshold
        
        return {
            'early_period': {
                'dates': early_dates,
                'convergence': early_rel_conv
            },
            'late_period': {
                'dates': late_dates,
                'convergence': late_rel_conv
            },
            'stability_metrics': {
                'absolute_difference': abs_diff,
                'relative_difference': rel_diff,
                'is_stable': is_stable
            }
        }
    
    @handle_errors(logger=logger)
    def _test_regional_stability(
        self, 
        original_data: gpd.GeoDataFrame, 
        simulated_data: gpd.GeoDataFrame, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test model stability across different regions.
        
        Parameters
        ----------
        original_data : gpd.GeoDataFrame
            Original data
        simulated_data : gpd.GeoDataFrame
            Simulated data
        results : Dict[str, Any]
            Simulation results with models
            
        Returns
        -------
        Dict[str, Any]
            Regional stability test results
        """
        # Check for commodity column for further stratification
        if 'commodity' in original_data.columns:
            commodities = original_data['commodity'].unique()
            
            # Skip if too few commodities
            if len(commodities) < 2:
                return {'sufficient_commodities': False}
            
            # Calculate price changes by commodity
            regional_results = {}
            
            for commodity in commodities:
                commodity_orig = original_data[original_data['commodity'] == commodity]
                commodity_sim = simulated_data[simulated_data['commodity'] == commodity]
                
                # Calculate price changes for this commodity
                commodity_changes = self._calculate_price_changes(
                    original_prices=commodity_orig['price'],
                    simulated_prices=commodity_sim['simulated_price'],
                    by_column='exchange_rate_regime'
                )
                
                regional_results[commodity] = commodity_changes
            
            # Calculate variation in price changes across commodities
            north_changes = [
                results.loc[('north', 'pct_change'), 'mean'] 
                for commodity, results in regional_results.items()
                if ('north', 'pct_change') in results.index
            ]
            
            south_changes = [
                results.loc[('south', 'pct_change'), 'mean'] 
                for commodity, results in regional_results.items()
                if ('south', 'pct_change') in results.index
            ]
            
            # Calculate coefficient of variation (std/mean)
            north_cv = np.std(north_changes) / np.mean(north_changes) if north_changes and np.mean(north_changes) != 0 else np.nan
            south_cv = np.std(south_changes) / np.mean(south_changes) if south_changes and np.mean(south_changes) != 0 else np.nan
            
            # Determine if results are stable across commodities
            # (Below stability threshold from config)
            stability_threshold = config.get('analysis.simulation.regional_stability_threshold', 0.5)  # CV < 0.5 default
            is_stable = (not np.isnan(north_cv) and north_cv < stability_threshold and 
                         not np.isnan(south_cv) and south_cv < stability_threshold)
            
            return {
                'commodity_results': regional_results,
                'north_variation': north_cv,
                'south_variation': south_cv,
                'is_stable': is_stable
            }
        
        # If no commodity column, check for other regional dimensions
        elif 'market_id' in original_data.columns:
            # Similar analysis could be implemented for market-level stability
            return {'market_level_analysis': 'Not implemented'}
        
        else:
            return {'regional_dimensions': 'Insufficient for stability analysis'}
    
    @handle_errors(logger=logger)
    def _test_parameter_stability(
        self, 
        original_model: Any, 
        simulated_model: Any
    ) -> Dict[str, Any]:
        """
        Test stability of model parameters.
        
        Parameters
        ----------
        original_model : Any
            Original threshold model
        simulated_model : Any
            Simulated threshold model
            
        Returns
        -------
        Dict[str, Any]
            Parameter stability test results
        """
        # Extract relevant parameters
        original_params = {}
        simulated_params = {}
        
        # Check for threshold parameters
        if hasattr(original_model, 'threshold') and hasattr(simulated_model, 'threshold'):
            original_params['threshold'] = original_model.threshold
            simulated_params['threshold'] = simulated_model.threshold
        
        # Check for adjustment parameters
        if hasattr(original_model, 'results') and hasattr(simulated_model, 'results'):
            # Extract adjustment parameters below threshold
            original_params['adjustment_below'] = original_model.results.get('adjustment_below_1', None)
            simulated_params['adjustment_below'] = simulated_model.results.get('adjustment_below_1', None)
            
            # Extract adjustment parameters above threshold
            original_params['adjustment_above'] = original_model.results.get('adjustment_above_1', None)
            simulated_params['adjustment_above'] = simulated_model.results.get('adjustment_above_1', None)
        
        # Calculate parameter changes
        param_changes = {}
        for param in set(original_params.keys()) & set(simulated_params.keys()):
            orig_val = original_params[param]
            sim_val = simulated_params[param]
            
            if orig_val is not None and sim_val is not None:
                abs_change = sim_val - orig_val
                rel_change = (abs_change / abs(orig_val)) * 100 if orig_val != 0 else float('inf')
                
                param_changes[param] = {
                    'original': orig_val,
                    'simulated': sim_val,
                    'absolute_change': abs_change,
                    'relative_change': rel_change
                }
        
        # Determine if parameters are stable
        # (Below stability threshold from config)
        stability_threshold = config.get('analysis.simulation.parameter_stability_threshold', 50)  # 50% default
        
        param_stability = {}
        for param, changes in param_changes.items():
            rel_change = changes.get('relative_change', float('inf'))
            param_stability[param] = abs(rel_change) < stability_threshold
        
        # Overall stability assessment
        is_stable = all(param_stability.values()) if param_stability else False
        
        return {
            'parameter_changes': param_changes,
            'parameter_stability': param_stability,
            'is_stable': is_stable
        }
    
    @handle_errors(logger=logger)
    def _compare_diagnostics(
        self,
        original_diag: Dict[str, Any],
        simulated_diag: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare diagnostic results between original and simulated models.
        
        Parameters
        ----------
        original_diag : Dict[str, Any]
            Diagnostic results for original model
        simulated_diag : Dict[str, Any]
            Diagnostic results for simulated model
            
        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        comparison = {}
        
        # Compare normality
        if 'normality' in original_diag and 'normality' in simulated_diag:
            orig_normal = original_diag['normality'].get('normal', False)
            sim_normal = simulated_diag['normality'].get('normal', False)
            
            comparison['normality'] = {
                'original_normal': orig_normal,
                'simulated_normal': sim_normal,
                'improvement': not orig_normal and sim_normal,
                'deterioration': orig_normal and not sim_normal,
                'no_change': orig_normal == sim_normal
            }
        
        # Compare autocorrelation
        if 'autocorrelation' in original_diag and 'autocorrelation' in simulated_diag:
            orig_no_autocorr = original_diag['autocorrelation'].get('no_autocorrelation', False)
            sim_no_autocorr = simulated_diag['autocorrelation'].get('no_autocorrelation', False)
            
            comparison['autocorrelation'] = {
                'original_no_autocorr': orig_no_autocorr,
                'simulated_no_autocorr': sim_no_autocorr,
                'improvement': not orig_no_autocorr and sim_no_autocorr,
                'deterioration': orig_no_autocorr and not sim_no_autocorr,
                'no_change': orig_no_autocorr == sim_no_autocorr
            }
        
        # Compare heteroskedasticity
        if 'heteroskedasticity' in original_diag and 'heteroskedasticity' in simulated_diag:
            orig_homosked = original_diag['heteroskedasticity'].get('homoskedastic', False)
            sim_homosked = simulated_diag['heteroskedasticity'].get('homoskedastic', False)
            
            comparison['heteroskedasticity'] = {
                'original_homoskedastic': orig_homosked,
                'simulated_homoskedastic': sim_homosked,
                'improvement': not orig_homosked and sim_homosked,
                'deterioration': orig_homosked and not sim_homosked,
                'no_change': orig_homosked == sim_homosked
            }
        
        # Overall robustness assessment
        comparison['overall'] = {
            'improvements': sum(
                1 for key in ['normality', 'autocorrelation', 'heteroskedasticity'] 
                if key in comparison and comparison[key].get('improvement', False)
            ),
            'deteriorations': sum(
                1 for key in ['normality', 'autocorrelation', 'heteroskedasticity'] 
                if key in comparison and comparison[key].get('deterioration', False)
            )
        }
        
        return comparison
    
    @handle_errors(logger=logger)
    def _assess_overall_robustness(
        self,
        structural_break_results: Dict[str, Any],
        diagnostic_results: Dict[str, Any],
        stability_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess overall robustness of simulation results.
        
        Parameters
        ----------
        structural_break_results : Dict[str, Any]
            Structural break test results
        diagnostic_results : Dict[str, Any]
            Diagnostic test results
        stability_results : Dict[str, Any]
            Stability test results
            
        Returns
        -------
        Dict[str, Any]
            Overall robustness assessment
        """
        # Extract key robustness metrics
        structural_change = structural_break_results.get('break_comparison', {}).get('structural_change', False)
        
        diagnostic_improvements = diagnostic_results.get('diagnostic_comparison', {}).get('overall', {}).get('improvements', 0)
        diagnostic_deteriorations = diagnostic_results.get('diagnostic_comparison', {}).get('overall', {}).get('deteriorations', 0)
        
        time_stable = stability_results.get('time_stability', {}).get('stability_metrics', {}).get('is_stable', False)
        regional_stable = stability_results.get('regional_stability', {}).get('is_stable', False)
        parameter_stable = stability_results.get('parameter_stability', {}).get('is_stable', False)
        
        # Calculate overall robustness score
        # Each component contributes to the score
        robustness_score = 0
        total_components = 0
        
        # No structural change is good (+1)
        if not structural_change:
            robustness_score += 1
        total_components += 1
        
        # More diagnostic improvements than deteriorations is good (+1)
        if diagnostic_improvements > diagnostic_deteriorations:
            robustness_score += 1
        total_components += 1
        
        # Stable across time periods is good (+1)
        if time_stable:
            robustness_score += 1
        total_components += 1
        
        # Stable across regions is good (+1)
        if regional_stable:
            robustness_score += 1
        total_components += 1
        
        # Stable parameters is good (+1)
        if parameter_stable:
            robustness_score += 1
        total_components += 1
        
        # Calculate percentage score
        robustness_percentage = (robustness_score / total_components) * 100 if total_components > 0 else 0
        
        # Determine robustness level
        if robustness_percentage >= 80:
            robustness_level = "High"
        elif robustness_percentage >= 60:
            robustness_level = "Moderate"
        elif robustness_percentage >= 40:
            robustness_level = "Low"
        else:
            robustness_level = "Very Low"
        
        # Generate robustness summary
        robustness_summary = []
        
        if not structural_change:
            robustness_summary.append("No structural breaks introduced by the policy intervention.")
        else:
            robustness_summary.append("Policy intervention introduces structural changes in price dynamics.")
        
        if diagnostic_improvements > diagnostic_deteriorations:
            robustness_summary.append("Model diagnostics show improvements after the intervention.")
        elif diagnostic_improvements < diagnostic_deteriorations:
            robustness_summary.append("Model diagnostics deteriorate after the intervention.")
        else:
            robustness_summary.append("Model diagnostics show mixed or unchanged results after the intervention.")
        
        if time_stable:
            robustness_summary.append("Results are stable across different time periods.")
        else:
            robustness_summary.append("Results vary significantly across different time periods.")
        
        if regional_stable:
            robustness_summary.append("Results are consistent across different regions and commodities.")
        else:
            robustness_summary.append("Results show significant variation across regions and commodities.")
        
        if parameter_stable:
            robustness_summary.append("Model parameters remain stable before and after intervention.")
        else:
            robustness_summary.append("Model parameters change significantly after intervention.")
        
        return {
            'robustness_score': robustness_score,
            'total_components': total_components,
            'robustness_percentage': robustness_percentage,
            'robustness_level': robustness_level,
            'summary': robustness_summary
        }
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger)
    def run_sensitivity_analysis(
        self,
        sensitivity_type: str = 'conflict_reduction',
        param_values: Optional[List[float]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run sensitivity analysis by varying parameters and measuring impact.
        
        Parameters
        ----------
        sensitivity_type : str
            Type of sensitivity analysis:
            - 'conflict_reduction': Vary conflict reduction factor
            - 'exchange_rate': Vary exchange rate target values
        param_values : List[float], optional
            List of parameter values to test
            Default for conflict_reduction: [0.1, 0.25, 0.5, 0.75, 0.9]
            Default for exchange_rate: Multiple rates based on percentiles
        metrics : List[str], optional
            List of metrics to track
            
        Returns
        -------
        Dict[str, Any]
            Sensitivity analysis results
        """
        # Initialize default parameter values if not provided
        if param_values is None:
            if sensitivity_type == 'conflict_reduction':
                param_values = config.get('analysis.simulation.sensitivity_conflict_levels', 
                                         [0.1, 0.25, 0.5, 0.75, 0.9])
            elif sensitivity_type == 'exchange_rate':
                # Get percentiles of exchange rates
                if 'exchange_rate' in self.data.columns:
                    percentiles = [25, 50, 75]
                    param_values = [
                        np.percentile(self.data['exchange_rate'], p) 
                        for p in percentiles
                    ]
                    # Add official and market rates
                    north_rate = self.data[self.data['exchange_rate_regime'] == 'north']['exchange_rate'].mean()
                    south_rate = self.data[self.data['exchange_rate_regime'] == 'south']['exchange_rate'].mean()
                    param_values.extend([north_rate, south_rate])
                else:
                    param_values = [500, 600, 700, 800, 900]  # Default values if exchange_rate not available
            else:
                raise ValueError(f"Unknown sensitivity_type: {sensitivity_type}")
        
        # Initialize default metrics if not provided
        if metrics is None:
            metrics = config.get('analysis.simulation.sensitivity_metrics', 
                                ['price_convergence', 'integration_index', 'asymmetry'])
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Initialize results storage
        sensitivity_results = {
            'sensitivity_type': sensitivity_type,
            'param_values': param_values,
            'metrics': metrics,
            'results': {}
        }
        
        # Run simulations in parallel for better performance
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit sensitivity analysis tasks
            futures = {}
            for param_value in param_values:
                future = executor.submit(
                    self._run_sensitivity_analysis_for_param,
                    sensitivity_type, param_value, metrics
                )
                futures[future] = param_value
            
            # Collect results as they complete
            total_tasks = len(param_values)
            completed_tasks = 0
            
            for future in as_completed(futures):
                param_value = futures[future]
                try:
                    result = future.result()
                    sensitivity_results['results'][param_value] = result
                    
                    # Update progress
                    completed_tasks += 1
                    progress = (completed_tasks / total_tasks) * 100
                    logger.info(f"Sensitivity analysis progress: {completed_tasks}/{total_tasks} ({progress:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Error in sensitivity analysis for {param_value}: {e}")
                    sensitivity_results['results'][param_value] = {'error': str(e)}
        
        # Calculate summary statistics
        sensitivity_results['summary'] = self._calculate_sensitivity_summary(
            sensitivity_results['results'],
            metrics
        )
        
        # Generate plots for sensitivity analysis
        sensitivity_results['plots'] = self._generate_sensitivity_plots(
            sensitivity_results['results'],
            sensitivity_type,
            metrics
        )
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Sensitivity analysis complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Store in instance results
        self.results['sensitivity_analysis'] = sensitivity_results
        
        return sensitivity_results
    
    @handle_errors(logger=logger)
    def _run_sensitivity_analysis_for_param(
        self,
        sensitivity_type: str,
        param_value: float,
        metrics: List[str]
    ) -> Dict[str, float]:
        """
        Run a single sensitivity analysis simulation for a specific parameter value.
        
        Parameters
        ----------
        sensitivity_type : str
            Type of sensitivity analysis
        param_value : float
            Parameter value to test
        metrics : List[str]
            Metrics to calculate
            
        Returns
        -------
        Dict[str, float]
            Metrics for this parameter value
        """
        logger.info(f"Running {sensitivity_type} simulation with parameter value: {param_value}")
        
        # Run appropriate simulation based on sensitivity type
        if sensitivity_type == 'conflict_reduction':
            sim_result = self.simulate_improved_connectivity(reduction_factor=param_value)
            scenario_name = f"improved_connectivity_{param_value}"
        elif sensitivity_type == 'exchange_rate':
            sim_result = self.simulate_exchange_rate_unification(target_rate=str(param_value))
            scenario_name = f"exchange_rate_{param_value}"
        else:
            raise ValueError(f"Unknown sensitivity_type: {sensitivity_type}")
        
        # Store simulation result
        self.results[scenario_name] = sim_result
        
        # Calculate metrics
        metric_results = {}
        
        if 'price_convergence' in metrics:
            # Calculate price convergence metrics
            convergence = self._calculate_price_convergence(
                self.data, 
                sim_result['simulated_data'], 
                'price', 
                'simulated_price', 
                'exchange_rate_regime'
            )
            metric_results['price_convergence'] = convergence.get('relative_convergence', 0)
        
        if 'integration_index' in metrics:
            # Calculate integration index
            integration = self.calculate_integration_index(scenario_name)
            metric_results['integration_index'] = integration.get('percentage_improvement', 0)
        
        if 'asymmetry' in metrics and 'threshold_model' in sim_result:
            # Calculate asymmetry effect
            asymmetry = self.calculate_policy_asymmetry_effects(scenario_name)
            
            # Extract a numerical metric from asymmetry results
            if 'tar_comparison' in asymmetry:
                metric_results['asymmetry'] = asymmetry['tar_comparison'].get('change', 0)
        
        return metric_results
    
    @handle_errors(logger=logger)
    def _calculate_sensitivity_summary(
        self,
        sensitivity_results: Dict[float, Dict[str, float]],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for sensitivity analysis.
        
        Parameters
        ----------
        sensitivity_results : Dict[float, Dict[str, float]]
            Results of sensitivity analysis for each parameter value
        metrics : List[str]
            List of metrics tracked
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Summary statistics for each metric
        """
        summary = {}
        
        for metric in metrics:
            # Extract values for this metric across all parameter values
            values = [
                result.get(metric, np.nan) 
                for result in sensitivity_results.values()
                if metric in result and not isinstance(result, dict) or (
                    isinstance(result, dict) and 'error' not in result
                )
            ]
            
            if not values or all(np.isnan(values)):
                summary[metric] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'range': np.nan,
                    'coefficient_of_variation': np.nan
                }
                continue
            
            values = np.array([v for v in values if not np.isnan(v)])
            
            # Calculate summary statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            
            # Calculate coefficient of variation (CV) - std/mean
            # Higher CV indicates higher sensitivity to parameter changes
            cv = std_val / abs(mean_val) if mean_val != 0 else np.nan
            
            summary[metric] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'coefficient_of_variation': cv
            }
        
        # Calculate overall sensitivity score
        # Higher score means results are more sensitive to parameter changes
        valid_cvs = [
            stats['coefficient_of_variation'] 
            for stats in summary.values() 
            if not np.isnan(stats['coefficient_of_variation'])
        ]
        
        # Get sensitivity threshold from config
        high_sensitivity_threshold = config.get('analysis.simulation.high_sensitivity_threshold', 0.5)
        
        if valid_cvs:
            summary['overall'] = {
                'mean_cv': np.mean(valid_cvs),
                'max_cv': np.max(valid_cvs),
                'high_sensitivity': np.mean(valid_cvs) > high_sensitivity_threshold
            }
        else:
            summary['overall'] = {
                'mean_cv': np.nan,
                'max_cv': np.nan,
                'high_sensitivity': False
            }
        
        return summary
    
    @handle_errors(logger=logger)
    def _generate_sensitivity_plots(
        self,
        sensitivity_results: Dict[float, Dict[str, float]],
        sensitivity_type: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Generate plots for sensitivity analysis results.
        
        Parameters
        ----------
        sensitivity_results : Dict[float, Dict[str, float]]
            Results of sensitivity analysis for each parameter value
        sensitivity_type : str
            Type of sensitivity analysis
        metrics : List[str]
            List of metrics tracked
            
        Returns
        -------
        Dict[str, Any]
            Plot data for visualization
        """
        plot_data = {}
        
        # Extract parameter values and ensure they're sorted
        param_values = sorted(sensitivity_results.keys())
        
        for metric in metrics:
            # Extract values for this metric across all parameter values
            values = [
                sensitivity_results[param].get(metric, np.nan) 
                for param in param_values
                if isinstance(sensitivity_results[param], dict) and 'error' not in sensitivity_results[param]
            ]
            
            # Store x and y values for plotting
            plot_data[metric] = {
                'x': param_values,
                'y': values,
                'x_label': f"{sensitivity_type.replace('_', ' ').title()} Value",
                'y_label': f"{metric.replace('_', ' ').title()} Metric"
            }
        
        return plot_data