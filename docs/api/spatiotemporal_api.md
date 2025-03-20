# Spatiotemporal Integration API

This document provides API documentation for the spatiotemporal integration module that connects spatial and time series models to analyze market integration across both geographic and temporal dimensions.

## SpatialTemporalIntegration

The primary class for performing integrated spatiotemporal analysis of market integration patterns.

### Constructor

```python
SpatialTemporalIntegration(
    market_data: pd.DataFrame, 
    spatial_data: Optional[gpd.GeoDataFrame] = None,
    date_col: str = "date",
    market_id_col: str = "market_id",
    price_col: str = "price",
    commodity_col: str = "commodity",
    region_col: Optional[str] = "exchange_rate_regime",
    conflict_col: Optional[str] = "conflict_intensity_normalized",
    geometry_col: str = "geometry"
)
```

**Parameters:**

- `market_data`: DataFrame containing time series price data with at minimum date, market ID, and price columns
- `spatial_data`: (Optional) GeoDataFrame containing spatial information for markets
- `date_col`: Name of date column in market_data
- `market_id_col`: Name of market ID column in both dataframes
- `price_col`: Name of price column in market_data
- `commodity_col`: Name of commodity column in market_data
- `region_col`: (Optional) Name of region/regime column in market_data
- `conflict_col`: (Optional) Name of conflict intensity column in market_data
- `geometry_col`: Name of geometry column in spatial_data

### Methods

#### calculate_market_distances

```python
calculate_market_distances(conflict_weight: Optional[float] = None) -> np.ndarray
```

Calculates geographical distances between all markets, optionally adjusting for conflict intensity.

**Parameters:**

- `conflict_weight`: Weight to apply to conflict intensity when calculating effective distances (0 = no effect, higher = more effect)

**Returns:**

- Distance matrix between markets (numpy array)

#### identify_market_pairs

```python
identify_market_pairs(
    max_distance: Optional[float] = None,
    max_pairs: Optional[int] = None,
    commodity: Optional[str] = None,
    region: Optional[str] = None,
    method: str = "distance"
) -> List[Tuple[str, str]]
```

Identifies market pairs for cointegration analysis using various selection methods.

**Parameters:**

- `max_distance`: Maximum distance between markets to consider as pairs
- `max_pairs`: Maximum number of pairs to analyze
- `commodity`: Filter to a specific commodity
- `region`: Filter to a specific region
- `method`: Method to use for selecting pairs:
  - "distance" - Select pairs based on distance
  - "correlation" - Select pairs based on price correlation
  - "both" - Consider both distance and correlation
  - "all" - Include all possible pairs

**Returns:**

- List of market pairs for analysis

#### analyze_market_cointegration

```python
analyze_market_cointegration(
    market_pairs: Optional[List[Tuple[str, str]]] = None,
    commodity: Optional[str] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[Tuple[str, str], Dict]
```

Analyzes cointegration between market pairs using threshold cointegration models.

**Parameters:**

- `market_pairs`: List of market pairs to analyze
- `commodity`: Filter to a specific commodity
- `max_workers`: Maximum number of parallel workers to use
- `progress_callback`: Function to call with progress updates

**Returns:**

- Dictionary of cointegration results by market pair

#### identify_spatial_clusters

```python
identify_spatial_clusters(
    commodity: Optional[str] = None,
    n_clusters: Optional[int] = None,
    eps: Optional[float] = None,
    min_samples: int = 3,
    scaling: bool = True,
    method: str = "dbscan"
) -> Dict[str, Any]
```

Identifies spatial clusters of integrated markets using machine learning clustering methods.

**Parameters:**

- `commodity`: Filter to a specific commodity
- `n_clusters`: Number of clusters (for KMeans)
- `eps`: Maximum distance between samples (for DBSCAN)
- `min_samples`: Minimum samples in a cluster (for DBSCAN)
- `scaling`: Whether to scale features before clustering
- `method`: Clustering method: "dbscan" or "kmeans"

**Returns:**

- Dictionary of clustering results

#### calculate_spatial_integration_metrics

```python
calculate_spatial_integration_metrics(
    commodity: Optional[str] = None,
    threshold_distance: Optional[float] = None
) -> pd.DataFrame
```

Calculates spatial integration metrics for markets, including cointegration ratios and adjustment speeds.

**Parameters:**

- `commodity`: Filter to a specific commodity
- `threshold_distance`: Maximum distance for considering markets as neighbors

**Returns:**

- DataFrame with integration metrics by market

### Private Methods

#### _calculate_price_correlation_matrix

```python
_calculate_price_correlation_matrix(data, commodity=None)
```

Calculates price correlation matrix between markets.

#### _calculate_clustering_features

```python
_calculate_clustering_features(data, commodity=None)
```

Calculates features for clustering markets based on price statistics and spatial coordinates.

## Usage Examples

### Basic Initialization

```python
import pandas as pd
import geopandas as gpd
from src.models.spatiotemporal import SpatialTemporalIntegration

# Load data
market_data = pd.read_csv("path/to/market_data.csv")
spatial_data = gpd.read_file("path/to/spatial_data.geojson")

# Create spatiotemporal model
model = SpatialTemporalIntegration(
    market_data=market_data,
    spatial_data=spatial_data
)
```

### Analyzing Market Integration with Conflict Adjustment

```python
# Calculate distances with conflict adjustment
distances = model.calculate_market_distances(conflict_weight=0.5)

# Identify market pairs within a maximum distance
market_pairs = model.identify_market_pairs(
    max_distance=200,  # km
    commodity="wheat",
    method="both"  # Use both distance and correlation
)

# Analyze cointegration
results = model.analyze_market_cointegration(
    market_pairs=market_pairs,
    max_workers=4  # Parallel processing
)

# Calculate integration metrics
metrics = model.calculate_spatial_integration_metrics(
    commodity="wheat",
    threshold_distance=150  # km
)

# Visualize metrics
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
metrics.plot(
    column='integration_ratio',
    cmap='viridis',
    legend=True,
    ax=ax
)
plt.title('Spatial Market Integration Ratio')
plt.show()
```

### Identifying Market Clusters

```python
# Identify market clusters using DBSCAN
clusters = model.identify_spatial_clusters(
    commodity="wheat",
    method="dbscan",
    eps=0.5,
    min_samples=3
)

# Access cluster information
for cluster_id, markets in clusters['market_clusters'].items():
    if cluster_id == -1:
        print(f"Outlier markets: {markets}")
    else:
        print(f"Cluster {cluster_id}: {markets}")
