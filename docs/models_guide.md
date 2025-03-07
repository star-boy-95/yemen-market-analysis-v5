# Yemen Market Integration Models Guide

This guide explains how to use each model in the Yemen Market Integration project. It covers core concepts, usage patterns, and configuration options.

## Table of Contents
- [Unit Root Testing](#unit-root-testing)
- [Cointegration Testing](#cointegration-testing)
- [Threshold Models](#threshold-models)
- [Threshold VECM Models](#threshold-vecm-models)
- [Spatial Econometrics](#spatial-econometrics)
- [Model Diagnostics](#model-diagnostics)
- [Configuration Guidelines](#configuration-guidelines)

## Unit Root Testing

The `unit_root.py` module provides tools for testing stationarity in time series data.

### Key Classes:
- `UnitRootTester`: Tests for unit roots using various methods
- `StructuralBreakTester`: Tests for structural breaks in time series

### Usage Example:
```python
from src.models.unit_root import UnitRootTester

# Initialize tester
tester = UnitRootTester()

# Basic ADF test
adf_result = tester.test_adf(price_series, regression='c')
if adf_result['stationary']:
    print("Series is stationary")
else:
    print("Series has unit root")

# Run multiple tests and determine integration order
all_tests = tester.run_all_tests(price_series)
order = tester.determine_integration_order(price_series, max_order=2)
print(f"Series is integrated of order {order}")
```

### Configurable Parameters:
- Set `analysis.cointegration.max_lags` for lag selection
- Set `analysis.cointegration.alpha` for significance level

## Cointegration Testing

The `cointegration.py` module tests for long-run relationships between time series.

### Key Classes and Functions:
- `CointegrationTester`: Tests for cointegration using Engle-Granger and Johansen methods
- `estimate_cointegration_vector()`: Estimates the cointegration relationship
- `calculate_half_life()`: Calculates time for deviations to revert halfway to equilibrium

### Usage Example:
```python
from src.models.cointegration import CointegrationTester

# Initialize tester
tester = CointegrationTester()

# Test for cointegration between two price series
eg_result = tester.test_engle_granger(north_prices, south_prices)
print(f"Cointegrated: {eg_result['cointegrated']}")
print(f"Cointegration vector: {eg_result['beta']}")

# Or run combined tests
combined = tester.test_combined(north_prices, south_prices)
print(f"Half-life of deviations: {combined['half_life']} periods")
```

### Configurable Parameters:
- Set `analysis.cointegration.trend` for deterministic terms
- Set `analysis.cointegration.max_lags` for lag selection
- Set `analysis.cointegration.alpha` for significance level

## Threshold Models

The `threshold.py` module implements threshold cointegration models for nonlinear price transmission analysis.

### Key Classes and Functions:
- `ThresholdCointegration`: Implements two-regime threshold cointegration
- `calculate_asymmetric_adjustment()`: Quantifies asymmetric price transmission
- `test_asymmetric_adjustment()`: Tests for asymmetry in price adjustment

### Usage Example:
```python
from src.models.threshold import ThresholdCointegration

# Initialize with two price series
model = ThresholdCointegration(
    north_prices, 
    south_prices,
    market1_name="North", 
    market2_name="South"
)

# Test for cointegration
coint_result = model.estimate_cointegration()

# Estimate threshold and TVECM
threshold_result = model.estimate_threshold()
print(f"Threshold: {threshold_result['threshold']}")

tvecm_result = model.estimate_tvecm(run_diagnostics=True)
print(f"Adjustment below threshold: {tvecm_result['adjustment_below_1']}")
print(f"Adjustment above threshold: {tvecm_result['adjustment_above_1']}")

# Plot results
model.plot_regime_dynamics(save_path="results/threshold_dynamics.png")

# Run full workflow
full_results = model.run_full_analysis()
```

### Configurable Parameters:
- Set `analysis.threshold.alpha` for significance level
- Set `analysis.threshold.trim` for threshold search range
- Set `analysis.threshold.max_lags` for lag selection
- Set `analysis.threshold.n_bootstrap` for bootstrap replications

## Threshold VECM Models

The `threshold_vecm.py` module implements multivariate threshold vector error correction models.

### Key Classes and Functions:
- `ThresholdVECM`: Implements two-regime threshold VECM
- `calculate_regime_transition_matrix()`: Analyzes regime persistence
- `calculate_half_lives()`: Calculates speed of adjustment
- `test_threshold_significance()`: Tests significance of threshold effect
- `combine_tvecm_results()`: Compares multiple TVECM models

### Usage Example:
```python
from src.models.threshold_vecm import ThresholdVECM

# Initialize with price data (matrix with variables as columns)
model = ThresholdVECM(price_data, k_ar_diff=2)

# Estimate linear VECM first
linear_results = model.estimate_linear_vecm()

# Search for optimal threshold
threshold_results = model.grid_search_threshold(trim=0.15)
print(f"Threshold: {threshold_results['threshold']}")

# Estimate TVECM
tvecm_results = model.estimate_tvecm(run_diagnostics=True)
print(f"Alpha below: {tvecm_results['alpha_below']}")
print(f"Alpha above: {tvecm_results['alpha_above']}")

# Test significance of threshold effect
sig_test = model.test_threshold_significance(n_bootstrap=1000)
print(f"Threshold significant: {sig_test['significant']}")

# Plot regime dynamics
model.plot_regime_dynamics(save_path="results/vecm_dynamics.png")

# Run full workflow
full_results = model.run_full_analysis()
```

### Configurable Parameters:
- Set `analysis.threshold_vecm.alpha` for significance level
- Set `analysis.threshold_vecm.trim` for threshold search range
- Set `analysis.threshold_vecm.n_grid` for search grid size
- Set `analysis.threshold_vecm.k_ar_diff` for lag structure
- Set `analysis.threshold_vecm.bootstrap_reps` for bootstrap replications

## Spatial Econometrics

The `spatial.py` module provides tools for spatial analysis of market integration.

### Key Classes and Functions:
- `SpatialEconometrics`: Implements spatial econometric methods
- `calculate_market_accessibility()`: Measures market access given population
- `calculate_market_isolation()`: Quantifies market isolation due to conflict
- `find_nearest_points()`: Finds nearest features between two GeoDataFrames
- `simulate_improved_connectivity()`: Simulates reduced conflict scenarios
- `market_integration_index()`: Calculates time-varying integration metrics

### Usage Example:
```python
from src.models.spatial import SpatialEconometrics, simulate_improved_connectivity

# Initialize with market GeoDataFrame
model = SpatialEconometrics(markets_gdf)

# Create spatial weights (optionally adjusted for conflict)
weights = model.create_weight_matrix(
    k=5, 
    conflict_adjusted=True,
    conflict_col='conflict_intensity'
)

# Test for spatial autocorrelation
moran_result = model.moran_i_test('price')
print(f"Moran's I: {moran_result['I']}, p-value: {moran_result['p_norm']}")

# Identify local clusters
local_result = model.local_moran_test('price')
# local_result is a GeoDataFrame with cluster types

# Estimate spatial models
lag_model = model.spatial_lag_model('price', ['distance', 'population'])
error_model = model.spatial_error_model('price', ['distance', 'population'])

# Simulate improved connectivity
sim_result = simulate_improved_connectivity(
    markets_gdf,
    conflict_reduction=0.5,
    conflict_col='conflict_intensity',
    price_col='price'
)
print(f"Price convergence: {sim_result['metrics']['price_convergence_pct']}%")

# Visualize results
model.visualize_conflict_adjusted_weights(
    save_path="results/market_connectivity.png",
    node_color_col='price'
)
```

### Configurable Parameters:
- Set `analysis.spatial.knn` for number of neighbors
- Set `analysis.spatial.conflict_weight` for conflict adjustment factor
- Set `analysis.spatial.crs` for coordinate reference system

## Model Diagnostics

The `diagnostics.py` module provides comprehensive diagnostic tools for model validation.

### Key Classes and Functions:
- `ModelDiagnostics`: Performs diagnostics on model residuals
- `calculate_fit_statistics()`: Calculates model fit metrics
- `bootstrap_confidence_intervals()`: Generates bootstrap CIs
- `compute_prediction_intervals()`: Computes prediction intervals

### Usage Example:
```python
from src.models.diagnostics import ModelDiagnostics, calculate_fit_statistics

# Initialize diagnostics with model residuals
diagnostics = ModelDiagnostics(
    residuals=model_residuals,
    model_name="TVECM"
)

# Run basic residual tests
normality = diagnostics.test_normality()
autocorr = diagnostics.test_autocorrelation(lags=12)
hetero = diagnostics.test_heteroskedasticity()

# Run comprehensive diagnostics
all_tests = diagnostics.residual_tests()
print(f"Valid residuals: {all_tests['overall']['valid']}")

# Create diagnostic plots
plots = diagnostics.plot_diagnostics(
    save_path="results/model_diagnostics.png"
)

# Test model stability
stability = diagnostics.test_model_stability(window_size=20)
print(f"Stable parameters: {stability['stable']}")

# Calculate fit statistics
fit_stats = calculate_fit_statistics(
    observed=actual_values,
    predicted=model_predictions,
    n_params=5
)
print(f"R-squared: {fit_stats['r_squared']}")
print(f"RMSE: {fit_stats['rmse']}")
```

## Configuration Guidelines

The `settings.yaml` file controls parameters across all models. Here's how to configure it:

### Unit Root and Cointegration Settings
```yaml
analysis:
  cointegration:
    max_lags: 4      # Maximum lags for ADF and other tests
    alpha: 0.05      # Significance level
    trend: "c"       # Trend specification ('c', 'ct', 'n')
```

### Threshold Model Settings
```yaml
analysis:
  threshold:
    trim: 0.15       # Trimming parameter (0.0-0.5)
    n_grid: 300      # Grid points for threshold search
    max_lags: 4      # Maximum lags in models
    alpha: 0.05      # Significance level
    n_bootstrap: 1000  # Bootstrap replications
```

### Spatial Model Settings
```yaml
analysis:
  spatial:
    knn: 5           # Number of nearest neighbors
    conflict_weight: 0.5  # Weight of conflict in distance adjustment
    crs: 32638       # Coordinate reference system (UTM Zone 38N for Yemen)
```

### Visualization Settings
```yaml
visualization:
  color_palette: "viridis"  # Default color palette
  figure_dpi: 300           # Resolution for saved figures
  default_fig_width: 12     # Default figure width
  default_fig_height: 8     # Default figure height
  north_color: "#1f77b4"    # Color for northern regions
  south_color: "#ff7f0e"    # Color for southern regions
  conflict_color: "#d62728" # Color for conflict visualization
```

### Regional Definitions
```yaml
regions:
  north:
    - "amanat alasimah"
    - "sanaa"
    - "ibb"
    - "taiz"
  south:
    - "abyan"
    - "aden"
    - "lahj"
    - "shabwah"
```

To update configuration settings, modify the `settings.yaml` file directly. Changes will be automatically applied when the config module is imported.