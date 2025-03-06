## Integration with Other Modules

The spatial module works seamlessly with other modules in the project for comprehensive market analysis.

### Combining Spatial and Threshold Analysis

```python
import pandas as pd
import geopandas as gpd
from src.data import load_market_data
from src.models.spatial import SpatialEconometrics, calculate_market_isolation
from src.models.threshold import ThresholdCointegration

# Step 1: Load and prepare data
markets_gdf = load_market_data('unified_data.geojson')

# Filter for a specific commodity
commodity = 'wheat'
commodity_data = markets_gdf[markets_gdf['commodity'] == commodity]

# Step 2: Calculate market isolation
markets_with_isolation = calculate_market_isolation(
    commodity_data,
    conflict_col='conflict_intensity_normalized'
)

# Step 3: Create north and south time series
north_markets = commodity_data[commodity_data['exchange_rate_regime'] == 'north']
south_markets = commodity_data[commodity_data['exchange_rate_regime'] == 'south']

# Calculate average prices
north_price = north_markets.groupby('date')['price'].mean()
south_price = south_markets.groupby('date')['price'].mean()

# Step 4: Run threshold cointegration analysis
model = ThresholdCointegration(north_price, south_price)
coint_results = model.estimate_cointegration()
threshold_results = model.estimate_threshold()
tvecm_results = model.estimate_tvecm()

# Step 5: Calculate market-level spatial influence
spatial_model = SpatialEconometrics(commodity_data)
weights = spatial_model.create_weight_matrix(
    conflict_adjusted=True,
    conflict_col='conflict_intensity_normalized'
)
moran_result = spatial_model.moran_i_test('price')

# Step 6: Analyze how isolation affects integration
high_isolation = markets_with_isolation['isolation_index'] > markets_with_isolation['isolation_index'].median()
low_isolation = ~high_isolation

# Calculate price differences by isolation level
high_iso_diff = calculate_price_diff(commodity_data[high_isolation])
low_iso_diff = calculate_price_diff(commodity_data[low_isolation])

print(f"Price differential in high isolation markets: {high_iso_diff.mean():.2f}")
print(f"Price differential in low isolation markets: {low_iso_diff.mean():.2f}")
print(f"Threshold value: {threshold_results['threshold']:.2f}")

# Helper function
def calculate_price_diff(gdf):
    north = gdf[gdf['exchange_rate_regime'] == 'north']
    south = gdf[gdf['exchange_rate_regime'] == 'south']
    north_avg = north.groupby('date')['price'].mean()
    south_avg = south.groupby('date')['price'].mean()
    common_dates = north_avg.index.intersection(south_avg.index)
    return abs(north_avg.loc[common_dates] - south_avg.loc[common_dates])
```

### Using Spatial Results in Time Series Analysis

```python
import pandas as pd
import numpy as np
from src.data import load_market_data
from src.models.spatial import SpatialEconometrics
from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester

# Step 1: Load data and calculate spatial clusters
markets_gdf = load_market_data('unified_data.geojson')
commodity_data = markets_gdf[markets_gdf['commodity'] == 'wheat']

# Identify spatial clusters
spatial_model = SpatialEconometrics(commodity_data)
weights = spatial_model.create_weight_matrix(k=5)
local_moran = spatial_model.moran_local_test('price')  # Assumes this method exists

# Get high-price and low-price clusters
high_price_clusters = local_moran[local_moran['cluster_type'] == 'high-high'].index
low_price_clusters = local_moran[local_moran['cluster_type'] == 'low-low'].index

# Step 2: Analyze time series properties by cluster type
high_cluster_data = commodity_data[commodity_data.index.isin(high_price_clusters)]
low_cluster_data = commodity_data[commodity_data.index.isin(low_price_clusters)]

# Calculate average time series for each cluster
high_cluster_ts = high_cluster_data.groupby('date')['price'].mean()
low_cluster_ts = low_cluster_data.groupby('date')['price'].mean()

# Step 3: Test for unit roots
unit_root_tester = UnitRootTester()
high_cluster_result = unit_root_tester.test_adf(high_cluster_ts)
low_cluster_result = unit_root_tester.test_adf(low_cluster_ts)

# Step 4: Test for cointegration between clusters
cointegration_tester = CointegrationTester()
coint_result = cointegration_tester.test_engle_granger(high_cluster_ts, low_cluster_ts)

print(f"High-price cluster stationarity: {high_cluster_result['stationary']}")
print(f"Low-price cluster stationarity: {low_cluster_result['stationary']}")
print(f"Clusters are cointegrated: {coint_result['cointegrated']}")
```

These examples demonstrate how to integrate spatial analysis with threshold cointegration and unit root testing to gain deeper insights into market integration patterns.# Spatial Econometrics Guide

This guide explains how to use the spatial econometrics module for analyzing geographical relationships in market data.

## Table of Contents

1. [Introduction](#introduction)
2. [SpatialEconometrics Class](#spatialeconometrics-class)
3. [Market Accessibility Analysis](#market-accessibility-analysis)
4. [Market Isolation Analysis](#market-isolation-analysis)
5. [Examples](#examples)
6. [Integration with Other Modules](#integration-with-other-modules)

## Introduction

Spatial econometrics examines how market prices are influenced by geographical proximity and conflict-related barriers, crucial for understanding market integration in Yemen.

## SpatialEconometrics Class

The `SpatialEconometrics` class provides spatial analysis tools for market data.

### Basic Usage

```python
from src.models.spatial import SpatialEconometrics
import geopandas as gpd

# Load market data as GeoDataFrame
markets_gdf = gpd.read_file('data/markets.geojson')

# Initialize spatial model
spatial_model = SpatialEconometrics(markets_gdf)

# Create spatial weights matrix
# Option 1: Standard distance-based weights
weights = spatial_model.create_weight_matrix(k=5, conflict_adjusted=False)

# Option 2: Conflict-adjusted weights (reduces connectivity in conflict areas)
weights = spatial_model.create_weight_matrix(
    k=5, 
    conflict_adjusted=True,
    conflict_col='conflict_intensity_normalized',
    conflict_weight=0.5
)

# Test for spatial autocorrelation using Moran's I
moran_result = spatial_model.moran_i_test('price')
print(f"Moran's I: {moran_result['I']:.4f}, p-value: {moran_result['p_norm']:.4f}")
print(f"Significant spatial autocorrelation: {moran_result['significant']}")

# Estimate spatial lag model
lag_model = spatial_model.spatial_lag_model(
    y_col='price',
    x_cols=['conflict_intensity_normalized', 'exchange_regime_north', 'population']
)
print(f"Spatial lag coefficient (rho): {lag_model.rho:.4f}")

# Estimate spatial error model
error_model = spatial_model.spatial_error_model(
    y_col='price',
    x_cols=['conflict_intensity_normalized', 'exchange_regime_north', 'population']
)
print(f"Spatial error coefficient (lambda): {error_model.lam:.4f}")
```

### Interpreting Results

- **Moran's I**: Measures spatial autocorrelation
  - Values near 1: Strong positive spatial autocorrelation (similar values cluster)
  - Values near -1: Strong negative spatial autocorrelation (dissimilar values cluster)
  - Values near 0: No spatial autocorrelation

- **Spatial Lag Model**: Captures direct influence of neighboring markets
  - Significant positive ρ (rho): Prices influenced by neighboring markets
  - Larger ρ indicates stronger spatial price transmission

- **Spatial Error Model**: Captures spatial patterns in unobserved factors
  - Significant λ (lambda): Spatial structure in the error term
  - May indicate omitted spatial variables

## Market Accessibility Analysis

Calculate a market's accessibility based on surrounding population centers.

```python
from src.models.spatial import calculate_market_accessibility

# Calculate accessibility index
markets_with_accessibility = calculate_market_accessibility(
    markets_gdf,
    population_gdf,
    max_distance=50000,  # 50 km
    distance_decay=2.0,  # Squared distance decay
    weight_col='population'
)

# Higher index = more accessible market
print("Top 5 most accessible markets:")
print(markets_with_accessibility.sort_values('accessibility_index', ascending=False).head(5))
```

## Market Isolation Analysis

Calculate how isolated markets are based on distance to other markets and conflict.

```python
from src.models.spatial import calculate_market_isolation

# Calculate isolation index
markets_with_isolation = calculate_market_isolation(
    markets_gdf,
    conflict_col='conflict_intensity_normalized',
    max_distance=50000  # 50 km
)

# Higher index = more isolated market
print("Top 5 most isolated markets:")
print(markets_with_isolation.sort_values('isolation_index', ascending=False).head(5))
```

## Examples

### Example 1: Analyzing Spatial Price Patterns

```python
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_market_data
from src.models.spatial import SpatialEconometrics

# Load data
markets_gdf = load_market_data('unified_data.geojson')

# Filter for latest date and a specific commodity
latest_date = markets_gdf['date'].max()
commodity = 'beans (kidney red)'
current_markets = markets_gdf[
    (markets_gdf['date'] == latest_date) & 
    (markets_gdf['commodity'] == commodity)
]

# Initialize spatial model
spatial_model = SpatialEconometrics(current_markets)

# Create conflict-adjusted weights
weights = spatial_model.create_weight_matrix(
    k=5, 
    conflict_adjusted=True,
    conflict_col='conflict_intensity_normalized'
)

# Test for spatial autocorrelation in prices
moran_result = spatial_model.moran_i_test('price')
print(f"Moran's I test for {commodity} prices:")
print(f"I = {moran_result['I']:.4f}, p-value = {moran_result['p_norm']:.4f}")
print(f"Significant spatial autocorrelation: {moran_result['significant']}")

# Create dummy for exchange rate regime
current_markets['north_regime'] = (current_markets['exchange_rate_regime'] == 'north').astype(int)

# Estimate spatial lag model
lag_model = spatial_model.spatial_lag_model(
    y_col='price',
    x_cols=['conflict_intensity_normalized', 'north_regime']
)

print("\nSpatial Lag Model Results:")
print(f"Spatial autoregressive coefficient (rho): {lag_model.rho:.4f}")
print(f"Significance of rho (p-value): {lag_model.z_stat_rho[1]:.4f}")
print("\nCoefficients:")
for i, var in enumerate(['Constant'] + ['conflict_intensity_normalized', 'north_regime']):
    print(f"{var}: {lag_model.betas[i]:.4f} (p-value: {lag_model.z_stat[i][1]:.4f})")

print(f"\nModel fit: R² = {lag_model.pr2:.4f}, AIC = {lag_model.aic:.4f}")
```

### Example 2: Comparing Normal and Conflict-Adjusted Spatial Models

```python
import geopandas as gpd
import pandas as pd
from src.data import load_market_data
from src.models.spatial import SpatialEconometrics

# Load and filter data (as in Example 1)
markets_gdf = load_market_data('unified_data.geojson')
latest_date = markets_gdf['date'].max()
commodity = 'beans (kidney red)'
current_markets = markets_gdf[
    (markets_gdf['date'] == latest_date) & 
    (markets_gdf['commodity'] == commodity)
]
current_markets['north_regime'] = (current_markets['exchange_rate_regime'] == 'north').astype(int)

# Initialize spatial model
spatial_model = SpatialEconometrics(current_markets)

# Create standard weights (no conflict adjustment)
regular_weights = spatial_model.create_weight_matrix(
    k=5, 
    conflict_adjusted=False
)

# Test for spatial autocorrelation
regular_moran = spatial_model.moran_i_test('price')

# Estimate standard spatial lag model
regular_lag = spatial_model.spatial_lag_model(
    y_col='price',
    x_cols=['conflict_intensity_normalized', 'north_regime']
)

# Now with conflict adjustment
conflict_weights = spatial_model.create_weight_matrix(
    k=5, 
    conflict_adjusted=True,
    conflict_col='conflict_intensity_normalized'
)

# Test for spatial autocorrelation with conflict adjustment
conflict_moran = spatial_model.moran_i_test('price')

# Estimate conflict-adjusted spatial lag model
conflict_lag = spatial_model.spatial_lag_model(
    y_col='price',
    x_cols=['conflict_intensity_normalized', 'north_regime']
)

# Compare results
print("Comparing standard and conflict-adjusted spatial models:")
print(f"Standard Moran's I: {regular_moran['I']:.4f} (p-value: {regular_moran['p_norm']:.4f})")
print(f"Conflict-adjusted Moran's I: {conflict_moran['I']:.4f} (p-value: {conflict_moran['p_norm']:.4f})")
print(f"\nStandard spatial lag (rho): {regular_lag.rho:.4f}")
print(f"Conflict-adjusted spatial lag (rho): {conflict_lag.rho:.4f}")
print(f"\nStandard model fit (R²): {regular_lag.pr2:.4f}")
print(f"Conflict-adjusted model fit (R²): {conflict_lag.pr2:.4f}")
```

### Example 3: Market Accessibility Analysis

```python
import geopandas as gpd
import matplotlib.pyplot as plt
from src.data import load_market_data, load_population_data
from src.models.spatial import calculate_market_accessibility, calculate_market_isolation

# Load market and population data
markets_gdf = load_market_data('markets.geojson')
population_gdf = load_population_data('population.geojson')

# Calculate accessibility
markets_with_accessibility = calculate_market_accessibility(
    markets_gdf,
    population_gdf,
    max_distance=50000,  # 50 km
    distance_decay=2.0,
    weight_col='population'
)

# Calculate isolation
markets_with_isolation = calculate_market_isolation(
    markets_gdf,
    conflict_col='conflict_intensity_normalized',
    max_distance=50000  # 50 km
)

# Combine results
markets_analysis = markets_with_accessibility.copy()
markets_analysis['isolation_index'] = markets_with_isolation['isolation_index']

# Calculate accessibility-to-isolation ratio
markets_analysis['access_isolation_ratio'] = (
    markets_analysis['accessibility_index'] / 
    (markets_analysis['isolation_index'] + 0.001)  # Add small constant to avoid division by zero
)

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

markets_analysis.plot(
    column='accessibility_index',
    cmap='viridis',
    legend=True,
    ax=axes[0]
)
axes[0].set_title('Market Accessibility')

markets_analysis.plot(
    column='isolation_index',
    cmap='magma',
    legend=True,
    ax=axes[1]
)
axes[1].set_title('Market Isolation')

markets_analysis.plot(
    column='access_isolation_ratio',
    cmap='RdYlGn',
    legend=True,
    ax=axes[2]
)
axes[2].set_title('Accessibility-to-Isolation Ratio')

for ax in axes:
    ax.set_axis_off()

plt.tight_layout()
plt.savefig('figures/market_access_isolation.png', dpi=300)
plt.show()
```