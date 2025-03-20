# Yemen Market Integration Project: Simulation API Reference

This document provides detailed reference information for the policy simulation components implemented in the Yemen Market Integration project. These components enable the simulation of various policy interventions (exchange rate unification, conflict reduction, and combined policies) and the analysis of their impacts on market integration.

## MarketIntegrationSimulation Class

The `MarketIntegrationSimulation` class is the primary simulation interface, providing methods for simulating policy interventions and analyzing their effects.

### Initialization

```python
from src.models.simulation import MarketIntegrationSimulation

simulation = MarketIntegrationSimulation(
    data,                      # GeoDataFrame with market data
    threshold_model=None,      # Optional ThresholdCointegration model
    spatial_model=None         # Optional SpatialEconometrics model
)
```

#### Required Data Format

The input data must be a GeoDataFrame with at least the following columns:

- `exchange_rate_regime`: String, either 'north' or 'south'
- `price`: Float, commodity price in local currency
- `exchange_rate`: Float, exchange rate to USD
- `conflict_intensity_normalized`: Float, normalized conflict intensity (0-1)

### Policy Simulation Methods

#### `simulate_exchange_rate_unification`

Simulates exchange rate unification using USD as cross-rate.

```python
results = simulation.simulate_exchange_rate_unification(
    target_rate='official',     # Method to determine unified exchange rate
    reference_date=None         # Optional date for reference exchange rates
)
```

**Parameters:**

- `target_rate`: String or float, specifies how to determine the unified rate:
  - `'official'`: Use official exchange rate (north)
  - `'market'`: Use market exchange rate (south)
  - `'average'`: Use average of north and south rates
  - Numeric value: Use the provided value directly
- `reference_date`: Optional date string for reference exchange rates

**Returns:**
Dictionary with simulation results:

- `'simulated_data'`: GeoDataFrame with simulated prices
- `'unified_rate'`: Selected unified exchange rate
- `'price_changes'`: DataFrame of price changes by region
- `'threshold_model'`: Re-estimated threshold model (if original was provided)

#### `simulate_improved_connectivity`

Simulates improved connectivity by reducing conflict barriers.

```python
results = simulation.simulate_improved_connectivity(
    reduction_factor=0.5       # Factor by which to reduce conflict intensity
)
```

**Parameters:**

- `reduction_factor`: Float between 0 and 1, specifies the proportion by which to reduce conflict intensity

**Returns:**
Dictionary with simulation results:

- `'simulated_data'`: GeoDataFrame with reduced conflict intensity
- `'spatial_weights'`: Recalculated spatial weights
- `'spatial_model'`: Re-estimated spatial model (if original was provided)
- `'reduction_factor'`: The reduction factor used

#### `simulate_combined_policy`

Simulates combined exchange rate unification and improved connectivity.

```python
results = simulation.simulate_combined_policy(
    exchange_rate_target='official',  # Method to determine unified rate
    conflict_reduction=0.5,           # Factor for conflict reduction
    reference_date=None               # Optional reference date
)
```

**Parameters:**

- `exchange_rate_target`: Same as for `simulate_exchange_rate_unification`
- `conflict_reduction`: Same as for `simulate_improved_connectivity`
- `reference_date`: Optional date string for reference exchange rates

**Returns:**
Dictionary with combined simulation results including all outputs from both individual simulations.

#### `simulate_combined_policies`

Simulates multiple policy combinations for interaction analysis.

```python
results = simulation.simulate_combined_policies(
    policy_combinations=[      # List of policy parameter dictionaries
        {
            'exchange_rate_target': 'official',
            'policy_name': 'exchange_rate_only'
        },
        {
            'conflict_reduction': 0.5,
            'policy_name': 'connectivity_only'
        },
        {
            'exchange_rate_target': 'official',
            'conflict_reduction': 0.5,
            'policy_name': 'combined'
        }
    ],
    parallelize=True           # Whether to process combinations in parallel
)
```

**Parameters:**

- `policy_combinations`: List of dictionaries, each containing:
  - `'exchange_rate_target'`: Optional, method to determine unified rate
  - `'conflict_reduction'`: Optional, factor for conflict reduction
  - `'policy_name'`: Optional name for labeling the policy
  - `'reference_date'`: Optional reference date
- `parallelize`: Boolean, whether to process policy combinations in parallel

**Returns:**
Dictionary with results for each policy combination plus interaction analysis.

### Analysis Methods

#### `calculate_welfare_effects`

Calculates welfare effects of policy simulations.

```python
welfare = simulation.calculate_welfare_effects(
    policy_scenario='exchange_rate_unification'  # Which policy to analyze
)
```

**Parameters:**

- `policy_scenario`: String, which policy scenario to analyze:
  - `'exchange_rate_unification'`
  - `'improved_connectivity'`
  - `'combined_policy'`
  - If None, uses latest simulation results

**Returns:**
Dictionary with welfare analysis results:

- `'regional_metrics'`: Price dispersion and changes by region
- `'price_convergence'`: Price convergence metrics
- `'commodity_effects'`: Effects by commodity (if applicable)
- `'mtar_metrics'`: M-TAR asymmetry metrics (if applicable)

#### `calculate_policy_asymmetry_effects`

Analyzes asymmetric policy response effects.

```python
asymmetry = simulation.calculate_policy_asymmetry_effects(
    policy_scenario='exchange_rate_unification'  # Which policy to analyze
)
```

**Parameters:**

- `policy_scenario`: String, which policy scenario to analyze

**Returns:**
Dictionary with asymmetric adjustment analysis:

- `'tar_comparison'`: Changes in TAR asymmetry
- `'mtar_comparison'`: Changes in M-TAR asymmetry (if available)
- `'interpretation'`: Textual interpretation of asymmetry changes

#### `calculate_integration_index`

Calculates market integration index before and after policy intervention.

```python
integration = simulation.calculate_integration_index(
    policy_scenario='exchange_rate_unification'  # Which policy to analyze
)
```

**Parameters:**

- `policy_scenario`: String, which policy scenario to analyze

**Returns:**
Dictionary with integration index results:

- `'original_index'`: Average integration index before intervention
- `'simulated_index'`: Average integration index after intervention
- `'absolute_improvement'`: Absolute change in integration index
- `'percentage_improvement'`: Percentage change in integration index
- `'detailed_original'`: Detailed original integration indices
- `'detailed_simulated'`: Detailed simulated integration indices

#### `test_robustness`

Performs robustness checks on simulation results.

```python
robustness = simulation.test_robustness(
    policy_scenario='exchange_rate_unification'  # Which policy to analyze
)
```

**Parameters:**

- `policy_scenario`: String, which policy scenario to analyze

**Returns:**
Dictionary with robustness test results:

- `'structural_breaks'`: Structural break test results
- `'diagnostics'`: Residual diagnostic test results
- `'stability'`: Model stability test results
- `'overall_assessment'`: Summary assessment of robustness

#### `run_sensitivity_analysis`

Runs sensitivity analysis by varying parameters and measuring impact.

```python
sensitivity = simulation.run_sensitivity_analysis(
    sensitivity_type='conflict_reduction',   # Type of sensitivity analysis
    param_values=[0.1, 0.25, 0.5, 0.75, 0.9],  # Parameter values to test
    metrics=['price_convergence', 'integration_index']  # Metrics to track
)
```

**Parameters:**

- `sensitivity_type`: String, type of sensitivity analysis:
  - `'conflict_reduction'`: Vary conflict reduction factor
  - `'exchange_rate'`: Vary exchange rate target values
- `param_values`: List of parameter values to test
- `metrics`: List of metrics to track

**Returns:**
Dictionary with sensitivity analysis results:

- `'sensitivity_type'`: Type of sensitivity analysis
- `'param_values'`: List of parameter values tested
- `'metrics'`: List of metrics tracked
- `'results'`: Dictionary of results for each parameter value
- `'summary'`: Summary statistics for sensitivity analysis
- `'plots'`: Plot data for visualization

### Helper Methods

The class also provides several helper methods used internally by the main simulation methods:

#### `_validate_input_data`

Validates input data format and required columns.

#### `_convert_to_usd`

Converts prices to USD using region-specific exchange rates.

#### `_determine_unified_rate`

Determines the unified exchange rate to use in simulation.

#### `_calculate_price_changes`

Calculates price changes between original and simulated prices.

#### `_reestimate_threshold_model`

Re-estimates threshold model with simulated prices.

#### `_recalculate_spatial_weights`

Recalculates spatial weights with adjusted conflict intensity.

#### `_reestimate_spatial_model`

Re-estimates spatial model with simulated data.

#### `_calculate_regional_welfare_metrics`

Calculates welfare metrics by region.

#### `_calculate_commodity_effects`

Calculates welfare effects by commodity.

#### `_calculate_price_convergence`

Calculates price convergence metrics between regions.

#### `_calculate_price_dispersion`

Calculates price dispersion by group.

## Example Usage

### Exchange Rate Unification Simulation

```python
# Import required classes
from src.models.simulation import MarketIntegrationSimulation
from src.models.threshold import ThresholdCointegration

# Create threshold model with market data
threshold_model = ThresholdCointegration(north_prices, south_prices)
threshold_model.estimate_cointegration()
threshold_model.estimate_threshold()

# Initialize simulation with market data and threshold model
simulation = MarketIntegrationSimulation(
    market_data,
    threshold_model=threshold_model
)

# Simulate exchange rate unification
results = simulation.simulate_exchange_rate_unification(
    target_rate='official'
)

# Calculate welfare effects
welfare = simulation.calculate_welfare_effects('exchange_rate_unification')

# Print key results
print(f"Unified Rate: {results['unified_rate']}")
print(f"Price Convergence: {welfare['price_convergence']['relative_convergence']:.2f}%")
```

### Combined Policy Simulation with Sensitivity Analysis

```python
# Initialize simulation
simulation = MarketIntegrationSimulation(market_data)

# Run combined policy simulation
results = simulation.simulate_combined_policy(
    exchange_rate_target='official',
    conflict_reduction=0.5
)

# Run sensitivity analysis
sensitivity = simulation.run_sensitivity_analysis(
    sensitivity_type='conflict_reduction',
    param_values=[0.1, 0.25, 0.5, 0.75, 0.9],
    metrics=['price_convergence', 'integration_index', 'asymmetry']
)

# Calculate integration index
integration = simulation.calculate_integration_index('combined_policy')

# Print key sensitivity results
print("Sensitivity Analysis Summary:")
for metric, stats in sensitivity['summary'].items():
    if metric != 'overall':
        print(f"{metric}: CV = {stats.get('coefficient_of_variation', 'N/A')}")
