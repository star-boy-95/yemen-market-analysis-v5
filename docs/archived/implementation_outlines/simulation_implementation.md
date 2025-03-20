# Policy Simulation Module Implementation Outline

This document provides a detailed implementation outline for the `simulation.py` module, which is an essential component of the Yemen Market Integration project. This implementation will follow the mathematical specifications outlined in the econometric_methods.md document.

## Overview

The simulation module implements methods for simulating the effects of policy interventions on market integration in Yemen, with a focus on exchange rate unification and conflict reduction scenarios. It provides tools for quantifying the potential impacts of these interventions on price convergence, welfare, and market efficiency.

## Class Structure

```python
class MarketIntegrationSimulation:
    """
    Simulate policy interventions for market integration analysis in Yemen.
    
    This class provides methods to simulate the effects of exchange rate unification
    and improved connectivity on market integration, and to calculate welfare effects
    of these policy interventions.
    """
    
    def __init__(self, data, threshold_model=None, spatial_model=None):
        """
        Initialize the simulation with market data and models.
        
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
        # Implementation
        
    def simulate_exchange_rate_unification(self, target_rate='official', reference_date=None):
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
        dict
            Simulation results including:
            - 'simulated_data': GeoDataFrame with simulated prices
            - 'unified_rate': Selected unified exchange rate
            - 'price_changes': DataFrame of price changes by region
            - 'threshold_model': Re-estimated threshold model
        """
        # Implementation
        
    def simulate_improved_connectivity(self, reduction_factor=0.5):
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
        dict
            Simulation results including:
            - 'simulated_data': GeoDataFrame with reduced conflict intensity
            - 'spatial_weights': Recalculated spatial weights
            - 'spatial_model': Re-estimated spatial model
        """
        # Implementation
        
    def simulate_combined_policy(self, exchange_rate_target='official', conflict_reduction=0.5, reference_date=None):
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
        dict
            Combined simulation results
        """
        # Implementation
        
    def simulate_combined_policies(self, policy_combinations=None, parallelize=True):
        """
        Simulate multiple policy combinations to analyze interactions.
        
        This enhanced method allows for simulating multiple policy combinations 
        simultaneously and analyzing their interactions. It can process different
        scenarios in parallel for improved performance.
        
        Parameters
        ----------
        policy_combinations : List[Dict[str, Any]], optional
            List of policy parameter dictionaries, each containing:
            - 'exchange_rate_target': str or float (optional)
            - 'conflict_reduction': float (optional)
            - 'policy_name': str (optional, for labeling)
        parallelize : bool, optional
            Whether to process policy combinations in parallel
            
        Returns
        -------
        dict
            Results for each policy combination and their interactions
        """
        # Implementation
        
    def calculate_welfare_effects(self, policy_scenario=None):
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
        dict
            Welfare analysis results
        """
        # Implementation
        
    def calculate_policy_asymmetry_effects(self, policy_scenario):
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
        dict
            Asymmetric adjustment changes after policy
        """
        # Implementation
        
    def calculate_integration_index(self, policy_scenario=None):
        """
        Calculate market integration index before and after policy intervention.
        
        Parameters
        ----------
        policy_scenario : str, optional
            Which policy scenario to analyze
            
        Returns
        -------
        dict
            Integration indices before and after, with percentage improvement
        """
        # Implementation
        
    def test_robustness(self, policy_scenario=None):
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
        dict
            Robustness test results
        """
        # Implementation
        
    def run_sensitivity_analysis(self, sensitivity_type='conflict_reduction', param_values=None, metrics=None):
        """
        Run sensitivity analysis by varying parameters and measuring impact.
        
        Parameters
        ----------
        sensitivity_type : str, optional
            Type of sensitivity analysis:
            - 'conflict_reduction': Vary conflict reduction factor
            - 'exchange_rate': Vary exchange rate target values
        param_values : List[float], optional
            List of parameter values to test
        metrics : List[str], optional
            List of metrics to track
            
        Returns
        -------
        dict
            Sensitivity analysis results
        """
        # Implementation
```

## Function Definitions

### 1. Exchange Rate Functions

```python
def _convert_to_usd(data, price_col='price', exchange_rate_col='exchange_rate'):
    """
    Convert prices to USD using region-specific exchange rates.
    
    Parameters
    ----------
    data : gpd.GeoDataFrame
        GeoDataFrame with price and exchange_rate columns
    price_col : str, optional
        Column name for prices
    exchange_rate_col : str, optional
        Column name for exchange rates
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with additional 'usd_price' column
    """
    # Implementation

def _determine_unified_rate(data, target_rate='official', reference_date=None):
    """
    Determine the unified exchange rate to use in simulation.
    
    Parameters
    ----------
    data : gpd.GeoDataFrame
        GeoDataFrame with exchange rate data
    target_rate : str, optional
        Method to determine unified rate ('official', 'market', 'average', or numeric value)
    reference_date : str, optional
        Date to use for reference rates
        
    Returns
    -------
    float
        Unified exchange rate value
    """
    # Implementation

def _calculate_price_changes(original_prices, simulated_prices, by_column=None):
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
    # Implementation
```

### 2. Conflict Adjustment Functions

```python
def _apply_improved_connectivity(data, reduction_factor=0.5, conflict_col='conflict_intensity_normalized'):
    """
    Apply improved connectivity by reducing conflict.
    
    Parameters
    ----------
    data : gpd.GeoDataFrame
        Data to modify
    reduction_factor : float, optional
        Factor to reduce conflict by
    conflict_col : str, optional
        Column name for conflict intensity
        
    Returns
    -------
    Tuple[gpd.GeoDataFrame, Any]
        Modified data and recalculated spatial weights
    """
    # Implementation

def _recalculate_spatial_weights(data, conflict_col):
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
    # Implementation
```

### 3. Welfare Analysis Functions

```python
def _calculate_regional_welfare_metrics(sim_data):
    """
    Calculate welfare metrics by region.
    
    Parameters
    ----------
    sim_data : gpd.GeoDataFrame
        Simulated data
        
    Returns
    -------
    dict
        Regional welfare metrics
    """
    # Implementation

def _calculate_commodity_effects(sim_data):
    """
    Calculate welfare effects by commodity.
    
    Parameters
    ----------
    sim_data : gpd.GeoDataFrame
        Simulated data
        
    Returns
    -------
    dict
        Commodity-specific welfare effects
    """
    # Implementation

def _calculate_price_convergence(original_data, simulated_data, original_price_col, simulated_price_col, regime_col):
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
    dict
        Price convergence metrics
    """
    # Implementation

def _calculate_price_dispersion(data, price_col, group_col):
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
    dict
        Price dispersion (coefficient of variation) by group
    """
    # Implementation
```

### 4. Robustness and Sensitivity Functions

```python
def _test_structural_breaks(original_data, simulated_data):
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
    dict
        Structural break test results
    """
    # Implementation

def _test_residual_diagnostics(results):
    """
    Run diagnostic tests on model residuals.
    
    Parameters
    ----------
    results : dict
        Simulation results with models
        
    Returns
    -------
    dict
        Residual diagnostic test results
    """
    # Implementation

def _test_model_stability(original_data, simulated_data, results):
    """
    Test model stability across different subsamples.
    
    Parameters
    ----------
    original_data : gpd.GeoDataFrame
        Original data
    simulated_data : gpd.GeoDataFrame
        Simulated data
    results : dict
        Simulation results with models
        
    Returns
    -------
    dict
        Model stability test results
    """
    # Implementation

def _run_sensitivity_analysis_for_param(sensitivity_type, param_value, metrics):
    """
    Run a single sensitivity analysis simulation for a specific parameter value.
    
    Parameters
    ----------
    sensitivity_type : str
        Type of sensitivity analysis
    param_value : float
        Parameter value to test
    metrics : list
        Metrics to calculate
        
    Returns
    -------
    dict
        Metrics for this parameter value
    """
    # Implementation

def _calculate_sensitivity_summary(sensitivity_results, metrics):
    """
    Calculate summary statistics for sensitivity analysis.
    
    Parameters
    ----------
    sensitivity_results : dict
        Results of sensitivity analysis for each parameter value
    metrics : list
        List of metrics tracked
        
    Returns
    -------
    dict
        Summary statistics for each metric
    """
    # Implementation
```

### 5. Policy Interaction Functions

```python
def _simulate_single_policy_combination(policy):
    """
    Simulate a single policy combination.
    
    Parameters
    ----------
    policy : dict
        Policy parameters
        
    Returns
    -------
    dict
        Simulation results for this policy
    """
    # Implementation

def _analyze_policy_interactions(policy_results):
    """
    Analyze interactions between multiple policy interventions.
    
    Parameters
    ----------
    policy_results : dict
        Results from multiple policy simulations
        
    Returns
    -------
    dict
        Interaction analysis
    """
    # Implementation

def _interpret_policy_interactions(interaction_results):
    """
    Generate interpretation of policy interaction effects.
    
    Parameters
    ----------
    interaction_results : dict
        Results of interaction analysis
        
    Returns
    -------
    str
        Interpretation of interaction effects
    """
    # Implementation

def _compare_diagnostics(original_diag, simulated_diag):
    """
    Compare diagnostic results between original and simulated models.
    
    Parameters
    ----------
    original_diag : dict
        Diagnostic results for original model
    simulated_diag : dict
        Diagnostic results for simulated model
        
    Returns
    -------
    dict
        Comparison results
    """
    # Implementation
```

## Implementation Dependencies

The implementation will rely on:

1. **Threshold Models**: Uses the ThresholdVECM and ThresholdCointegration classes for reestimation.
2. **Spatial Analysis**: Uses the SpatialEconometrics class for spatial analysis.
3. **Unit Root Testing**: Uses the StructuralBreakTester for robustness checks.
4. **Diagnostics**: Uses ModelDiagnostics for residual analysis.
5. **Performance Optimization**: Uses performance utilities to handle large datasets efficiently.

## Implementation Approaches

### Exchange Rate Unification Implementation Strategy

1. Convert all prices to USD using the existing exchange rates by region
2. Determine unified exchange rate based on the selected strategy
3. Apply unified exchange rate to convert USD prices back to YER
4. Re-estimate threshold models with simulated prices
5. Calculate price differential changes and other metrics
6. Analyze welfare impacts through regional metrics

### Connectivity Improvement Implementation Strategy

1. Reduce conflict intensity metrics by the specified reduction factor
2. Recalculate spatial weights with reduced conflict adjustment
3. Re-estimate spatial models
4. Calculate changes in market accessibility and isolation
5. Analyze impact on spatial market integration
6. Quantify welfare benefits of improved connectivity

### Combined Policy Implementation Strategy

1. Apply both exchange rate unification and connectivity improvement
2. Analyze interactions and synergies between the two policies
3. Identify whether policies complement or substitute one another
4. Calculate welfare effects of combined interventions
5. Perform robustness checks on combined policy results
6. Provide recommendations on optimal policy sequencing

### Sensitivity Analysis Implementation Strategy

1. Define parameter ranges for exchange rate targets and conflict reduction
2. Implement parallel processing for multiple parameter values
3. Calculate metrics for each parameter value
4. Analyze sensitivity of results to parameter changes
5. Identify critical thresholds for policy effectiveness
6. Provide confidence intervals for welfare metrics

## Error Handling and Performance Considerations

1. **Strict Input Validation**: Validate all inputs for:
   - Required columns in data
   - Valid exchange rate regimes (north/south)
   - Valid parameter ranges
   - Appropriate data types

2. **Memory Efficiency**:
   - Use optimized data structures for large datasets
   - Implement chunked processing for large simulations
   - Use memory-efficient data formats
   - Apply optimizations for M1/M2 hardware

3. **Error Handling**:
   - Use the `handle_errors` decorator for consistent error management
   - Implement specific error messages for common issues
   - Provide fallback options when possible

4. **Performance**:
   - Use parallel processing for simulating multiple policy scenarios
   - Cache intermediate results for reuse
   - Implement vectorized operations where possible
   - Apply memory optimizations for large GeoDataFrames

## Testing and Validation

1. **Unit Tests**:
   - Test exchange rate conversion functions
   - Test conflict adjustment algorithms
   - Verify welfare metrics calculations

2. **Integration Tests**:
   - Test interaction with threshold models and spatial models
   - Test combined policy simulations
   - Validate against theoretical expectations

3. **Performance Tests**:
   - Benchmark parallelized policy simulations
   - Test memory usage for large datasets
   - Validate optimization techniques

## Future Extensions

1. **Advanced Policy Scenarios**:
   - Gradual exchange rate convergence
   - Targeted conflict reduction in strategic areas
   - Trade subsidy and tariff simulations
   - Fiscal policy interventions

2. **Enhanced Welfare Analysis**:
   - Consumer surplus calculations
   - Producer welfare impacts
   - Distributional effects by income group
   - Food security implications

3. **Real-Time Monitoring**:
   - Dynamic policy simulation with real-time data
   - Early warning system for market fragmentation
   - Continuous monitoring of market integration
   - API for policy decision support
