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
from typing import Dict, Any, Union, Optional, List, Tuple

from src.models.threshold import ThresholdCointegration, test_mtar_adjustment
from src.models.diagnostics import ModelDiagnostics


from src.utils import (
    # Error handling
    handle_errors, ValidationError,
    
    # Validation
    validate_dataframe, validate_geodataframe, raise_if_invalid,
    
    # Configuration
    config,
    
    # Data processing
    compute_price_differentials, normalize_columns,
    
    # Spatial utilities
    create_conflict_adjusted_weights, calculate_distances,
    
    # Statistical utilities
    test_cointegration, estimate_threshold_model,
    
    # Performance
    m1_optimized
)

logger = logging.getLogger(__name__)

class MarketIntegrationSimulation:
    """
    Simulate policy interventions for market integration analysis in Yemen.
    
    This class provides methods to simulate the effects of exchange rate unification
    and improved connectivity on market integration, and to calculate welfare effects
    of these policy interventions.
    """
    
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
        
        # Store data and models
        self.data = data.copy()
        self.threshold_model = threshold_model
        self.spatial_model = spatial_model
        
        # Store original data for comparison
        self.original_data = data.copy()
        
        # Initialize results storage
        self.results = {}
        
        logger.info("MarketIntegrationSimulation initialized with data of shape %s", data.shape)
    
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
        # Copy data for simulation
        sim_data = self.data.copy()
        
        # Convert all prices to USD
        self._convert_to_usd(sim_data)
        
        # Determine unified exchange rate
        unified_rate = self._determine_unified_rate(target_rate, reference_date)
        logger.info("Using unified exchange rate: %.2f", unified_rate)
        
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
        
        # Get north and south prices
        north_prices = simulated_data[simulated_data['exchange_rate_regime'] == 'north']['simulated_price']
        south_prices = simulated_data[simulated_data['exchange_rate_regime'] == 'south']['simulated_price']
        
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
        
        return reestimated_model
    
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
        
        # Copy data for simulation
        sim_data = self.data.copy()
        
        # Reduce conflict intensity
        original_conflict = sim_data[conflict_col].copy()
        sim_data[conflict_col] = original_conflict * (1 - reduction_factor)
        
        logger.info(
            "Reduced conflict intensity by factor %.2f (mean: %.4f -> %.4f)",
            reduction_factor,
            original_conflict.mean(),
            sim_data[conflict_col].mean()
        )
        
        # Recalculate spatial weights
        spatial_weights = self._recalculate_spatial_weights(sim_data, conflict_col)
        
        # Re-estimate spatial model if provided
        spatial_model = None
        if self.spatial_model is not None:
            spatial_model = self._reestimate_spatial_model(sim_data, spatial_weights)
        
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
        regime_boundary_penalty = 1.5  # Default penalty multiplier for cross-regime connections
        
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
        from src.models.spatial import market_integration_index
        
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
        
        # Copy data for simulation
        sim_data = self.data.copy()
        
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
        # Import necessary classes
        from src.models.diagnostics import ModelDiagnostics
        from src.models.unit_root import StructuralBreakTester
        
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
        original_data = self.original_data
        simulated_data = results['simulated_data']
        
        # Initialize robustness results dictionary
        robustness_results = {}
        
        # 1. Test for structural breaks using Bai-Perron tests
        structural_break_tester = StructuralBreakTester()
        
        # Test for structural breaks in original price series
        if 'price' in original_data.columns:
            original_breaks = structural_break_tester.test_bai_perron(
                original_data['price'], 
                min_size=10,
                n_breaks=3
            )
            robustness_results['original_structural_breaks'] = original_breaks
        
        # Test for structural breaks in simulated price series
        if 'simulated_price' in simulated_data.columns:
            simulated_breaks = structural_break_tester.test_bai_perron(
                simulated_data['simulated_price'], 
                min_size=10,
                n_breaks=3
            )
            robustness_results['simulated_structural_breaks'] = simulated_breaks
            
            # Compare break dates/locations to see if policy changed market structure
            original_bps = original_breaks.get('breakpoints', [])
            simulated_bps = simulated_breaks.get('breakpoints', [])
            
            robustness_results['break_comparison'] = {
                'original_breakpoints': original_bps,
                'simulated_breakpoints': simulated_bps,
                'structural_change': len(original_bps) != len(simulated_bps) or not all(abs(o - s) < 5 for o, s in zip(original_bps, simulated_bps))
            }
        
        # 2. Perform residual diagnostics on threshold model results
        if 'threshold_model' in results and self.threshold_model is not None:
            # Create diagnostics object
            diagnostics = ModelDiagnostics(
                model_name=f"{policy_scenario}_threshold",
                original_data=original_data
            )
            
            # Get residuals from original and simulated models
            if hasattr(self.threshold_model, 'eq_errors') and hasattr(results['threshold_model'], 'eq_errors'):
                original_residuals = self.threshold_model.eq_errors
                simulated_residuals = results['threshold_model'].eq_errors
                
                # Run diagnostics on original residuals
                original_diagnostics = diagnostics.residual_tests(original_residuals)
                robustness_results['original_residual_diagnostics'] = original_diagnostics
                
                # Run diagnostics on simulated residuals
                diagnostics.residuals = simulated_residuals
                simulated_diagnostics = diagnostics.residual_tests(simulated_residuals)
                robustness_results['simulated_residual_diagnostics'] = simulated_diagnostics
                
                # Compare diagnostic results
                robustness_results['diagnostic_comparison'] = self._compare_diagnostics(
                    original_diagnostics, simulated_diagnostics
                )
        
        # Store results and return
        self.results[f'{policy_scenario}_robustness'] = robustness_results
        return robustness_results
    
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
                param_values = [0.1, 0.25, 0.5, 0.75, 0.9]
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
            metrics = ['price_convergence', 'integration_index', 'asymmetry']
        
        # Initialize results storage
        sensitivity_results = {
            'sensitivity_type': sensitivity_type,
            'param_values': param_values,
            'metrics': metrics,
            'results': {}
        }
        
        # Run simulations for each parameter value
        for param_value in param_values:
            logger.info(f"Running {sensitivity_type} simulation with parameter value: {param_value}")
            
            # Run appropriate simulation based on sensitivity type
            if sensitivity_type == 'conflict_reduction':
                sim_result = self.simulate_improved_connectivity(reduction_factor=param_value)
                scenario_name = f"improved_connectivity_{param_value}"
            elif sensitivity_type == 'exchange_rate':
                sim_result = self.simulate_exchange_rate_unification(target_rate=str(param_value))
                scenario_name = f"exchange_rate_{param_value}"
            
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
            
            # Store metric results for this parameter value
            sensitivity_results['results'][param_value] = metric_results
        
        # Calculate summary statistics
        sensitivity_results['summary'] = self._calculate_sensitivity_summary(
            sensitivity_results['results'],
            metrics
        )
        
        # Store in instance results
        self.results['sensitivity_analysis'] = sensitivity_results
        
        return sensitivity_results
    
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
                if metric in result
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
        
        if valid_cvs:
            summary['overall'] = {
                'mean_cv': np.mean(valid_cvs),
                'max_cv': np.max(valid_cvs),
                'high_sensitivity': np.mean(valid_cvs) > 0.5
            }
        else:
            summary['overall'] = {
                'mean_cv': np.nan,
                'max_cv': np.nan,
                'high_sensitivity': False
            }
        
        return summary