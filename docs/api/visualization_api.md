# Yemen Market Integration Project: Visualization API Reference

This document provides detailed reference information for the specialized visualization components implemented in the Yemen Market Integration project. These components focus on creating publication-quality visualizations of asymmetric price adjustment patterns, threshold cointegration results, and spatial market integration.

## Table of Contents

1. [Time Series Visualization](#time-series-visualization)
2. [Asymmetric Adjustment Visualization](#asymmetric-adjustment-visualization)
3. [Spatial Integration Visualization](#spatial-integration-visualization)

## Time Series Visualization

The `TimeSeriesVisualizer` class provides enhanced time series visualization capabilities specifically designed for market price data.

### Class: `TimeSeriesVisualizer`

```python
from src.visualization.time_series import TimeSeriesVisualizer

visualizer = TimeSeriesVisualizer()
```

#### Methods

##### `plot_price_series`

Creates a time series plot of price data, optionally grouping by category.

```python
fig, ax = visualizer.plot_price_series(
    df,                      # DataFrame with price data
    price_col='price',       # Name of price column
    date_col='date',         # Name of date column
    group_col=None,          # Optional column to group by
    title=None,              # Optional plot title
    ylabel='Price',          # Y-axis label
    save_path=None           # Optional path to save figure
)
```

##### `plot_price_differentials`

Creates a plot of price differentials between north and south markets.

```python
fig, ax = visualizer.plot_price_differentials(
    df,                       # DataFrame with price data
    date_col='date',          # Name of date column
    north_col='north_price',  # Name of northern prices column
    south_col='south_price',  # Name of southern prices column
    diff_col=None,            # Optional name of price differential column
    title=None,               # Optional plot title
    save_path=None            # Optional path to save figure
)
```

##### `plot_interactive_time_series`

Creates an interactive time series plot using Plotly.

```python
fig = visualizer.plot_interactive_time_series(
    df,                      # DataFrame with price data
    price_col='price',       # Name of price column
    date_col='date',         # Name of date column
    group_col=None,          # Optional column to group by
    title=None,              # Optional plot title
    ylabel='Price',          # Y-axis label
    save_path=None           # Optional path to save figure
)
```

##### `plot_threshold_analysis`

Visualizes threshold model results.

```python
fig, axs = visualizer.plot_threshold_analysis(
    threshold_model,        # Threshold model object
    title=None,             # Optional plot title
    save_path=None          # Optional path to save figure
)
```

##### `plot_simulation_comparison`

Compares original and simulated price series.

```python
fig, axs = visualizer.plot_simulation_comparison(
    original_data,          # DataFrame with original prices
    simulated_data,         # DataFrame with simulated prices
    date_col='date',        # Name of date column
    price_cols=None,        # Optional list of price column names
    title=None,             # Optional plot title
    save_path=None          # Optional path to save figure
)
```

## Asymmetric Adjustment Visualization

The `AsymmetricAdjustmentVisualizer` class provides specialized visualizations for asymmetric price adjustment patterns in threshold cointegration models.

### Class: `AsymmetricAdjustmentVisualizer`

```python
from src.visualization.asymmetric_plots import AsymmetricAdjustmentVisualizer

visualizer = AsymmetricAdjustmentVisualizer()
```

#### Methods

##### `plot_regime_dynamics`

Visualizes price differential series with regime highlighting.

```python
fig, ax = visualizer.plot_regime_dynamics(
    price_diff,             # Time series of price differentials
    dates,                  # Corresponding dates for price differentials
    threshold,              # Threshold value for regime separation
    title=None,             # Optional plot title
    save_path=None          # Optional path to save figure
)
```

##### `plot_asymmetric_adjustment`

Visualizes asymmetric adjustment patterns from a threshold model.

```python
fig, axs = visualizer.plot_asymmetric_adjustment(
    threshold_model,        # Threshold model object with adjustment results
    title=None,             # Optional plot title
    save_path=None          # Optional path to save figure
)
```

##### `plot_regime_transitions`

Visualizes regime transitions over time.

```python
fig, axs = visualizer.plot_regime_transitions(
    price_diff,             # Time series of price differentials
    dates,                  # Corresponding dates for price differentials
    threshold,              # Threshold value for regime separation
    title=None,             # Optional plot title
    save_path=None          # Optional path to save figure
)
```

##### `compare_adjustment_patterns`

Compares asymmetric adjustment patterns before and after policy intervention.

```python
fig, axs = visualizer.compare_adjustment_patterns(
    original_model,         # Original threshold model before policy
    simulated_model,        # Simulated threshold model after policy
    title=None,             # Optional plot title
    save_path=None          # Optional path to save figure
)
```

## Spatial Integration Visualization

The `SpatialIntegrationVisualizer` class provides specialized visualizations for spatial market integration patterns.

### Class: `SpatialIntegrationVisualizer`

```python
from src.visualization.spatial_integration import SpatialIntegrationVisualizer

visualizer = SpatialIntegrationVisualizer()
```

#### Methods

##### `plot_market_network`

Visualizes market integration network.

```python
fig, ax = visualizer.plot_market_network(
    market_gdf,              # GeoDataFrame with market locations
    edges_gdf=None,          # Optional GeoDataFrame with market connections
    market_id_col='market_id',       # Column with market IDs
    market_name_col='market_name',   # Column with market names
    region_col='exchange_rate_regime',  # Column with exchange rate regime
    edge_color_col='integration_level_num',  # Column to color edges by
    edge_width_col=None,     # Optional column to set edge width by
    title=None,              # Optional plot title
    basemap=True,            # Whether to add a basemap
    save_path=None           # Optional path to save figure
)
```

##### `plot_integration_choropleth`

Creates a choropleth map for integration metrics.

```python
fig, ax = visualizer.plot_integration_choropleth(
    market_gdf,              # GeoDataFrame with market data
    metric_col,              # Column with integration metric for coloring
    market_id_col='market_id',  # Column with market IDs
    region_col='exchange_rate_regime',  # Column with exchange rate regime
    title=None,              # Optional plot title
    cmap='RdYlGn',           # Colormap name
    basemap=True,            # Whether to add a basemap
    save_path=None           # Optional path to save figure
)
```

##### `plot_conflict_adjusted_network`

Visualizes conflict-adjusted market network.

```python
fig, ax = visualizer.plot_conflict_adjusted_network(
    market_gdf,              # GeoDataFrame with market data
    conflict_col='conflict_intensity_normalized',  # Column with conflict intensity
    region_col='exchange_rate_regime',             # Column with exchange rate regime
    market_id_col='market_id',                     # Column with market IDs
    title=None,              # Optional plot title
    basemap=True,            # Whether to add a basemap
    save_path=None           # Optional path to save figure
)
```

##### `plot_market_integration_comparison`

Compares spatial integration patterns before and after policy intervention.

```python
fig, axs = visualizer.plot_market_integration_comparison(
    original_gdf,            # GeoDataFrame with original integration metrics
    simulated_gdf,           # GeoDataFrame with simulated integration metrics
    metric_col,              # Column with integration metric for comparison
    market_id_col='market_id',  # Column with market IDs
    region_col='exchange_rate_regime',  # Column with exchange rate regime
    title=None,              # Optional plot title
    cmap='RdYlGn',           # Colormap name
    basemap=True,            # Whether to add a basemap
    save_path=None           # Optional path to save figure
)
