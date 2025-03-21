"""
Spatial utilities for GIS operations in the Yemen Market Integration Project.
Optimized for M1 MacBook Pro.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import nearest_points, transform, unary_union
import pyproj
from pyproj import CRS, Transformer
from functools import partial
import rtree
import logging
from typing import Union, List, Dict, Any, Optional, Tuple, Callable
import warnings

from .error_handler import handle_errors
from .decorators import timer, m1_optimized

logger = logging.getLogger(__name__)

# Constants for Yemen
YEMEN_EPSG = 32638  # UTM Zone 38N, suitable for Yemen
WGS84_EPSG = 4326   # Standard GPS coordinates

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def reproject_geometry(
    geometry, 
    from_crs: Union[str, int, CRS] = WGS84_EPSG, 
    to_crs: Union[str, int, CRS] = YEMEN_EPSG
) -> Any:
    """
    Reproject a geometry from one CRS to another
    
    Parameters
    ----------
    geometry : Shapely geometry
        Geometry to reproject
    from_crs : str, int, or CRS, optional
        Source CRS
    to_crs : str, int, or CRS, optional
        Target CRS
        
    Returns
    -------
    Shapely geometry
        Reprojected geometry
    """
    # Create transformer
    if isinstance(from_crs, int):
        from_crs = f"EPSG:{from_crs}"
    if isinstance(to_crs, int):
        to_crs = f"EPSG:{to_crs}"
    
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    
    # Create the transformation function
    project = partial(transform, transformer.transform)
    
    # Transform the geometry
    return project(geometry)

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def reproject_gdf(
    gdf: gpd.GeoDataFrame, 
    to_crs: Union[str, int, CRS] = YEMEN_EPSG
) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to a new coordinate reference system
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to reproject
    to_crs : str, int, or CRS, optional
        Target CRS
        
    Returns
    -------
    geopandas.GeoDataFrame
        Reprojected GeoDataFrame
    """
    if isinstance(to_crs, int):
        to_crs = f"EPSG:{to_crs}"
    
    # Reproject using GeoDataFrame's built-in method
    # This is already optimized internally
    return gdf.to_crs(to_crs)

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def create_point_from_coords(
    x: float, 
    y: float,
    crs: Union[str, int, CRS] = WGS84_EPSG
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame containing a point from coordinates
    
    Parameters
    ----------
    x : float
        X-coordinate (longitude in WGS84)
    y : float
        Y-coordinate (latitude in WGS84)
    crs : str, int, or CRS, optional
        Coordinate reference system of the input coordinates
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with point geometry
    """
    if isinstance(crs, int):
        crs = f"EPSG:{crs}"
    
    # Create point geometry
    point = Point(x, y)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': [point]}, crs=crs)
    
    return gdf

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def create_buffer(
    gdf: gpd.GeoDataFrame, 
    distance: float, 
    unit: str = 'meters',
    dissolve: bool = False
) -> gpd.GeoDataFrame:
    """
    Create a buffer around geometries in a GeoDataFrame
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame
    distance : float
        Buffer distance
    unit : str, optional
        Distance unit (only used for WGS84 CRS)
    dissolve : bool, optional
        Whether to dissolve buffers
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with buffer geometries
    """
    # Check if projection is geographic (WGS84)
    is_geographic = gdf.crs and (
        str(gdf.crs).lower().startswith('epsg:4326') or 
        (hasattr(gdf.crs, 'is_geographic') and gdf.crs.is_geographic)
    )
    
    if is_geographic:
        # For geographic CRS, convert distance to degrees
        if unit == 'meters':
            # Approximate conversion (varies by latitude)
            avg_lat = gdf.geometry.centroid.y.mean()
            # Convert meters to degrees based on latitude
            # At the equator: 1 degree ≈ 111,320 meters
            # At higher latitudes, 1 degree of longitude becomes less than 111,320 meters
            lat_rad = np.radians(avg_lat)
            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(lat_rad)
            avg_meters_per_degree = (meters_per_degree_lat + meters_per_degree_lon) / 2
            distance_degrees = distance / avg_meters_per_degree
            
            logger.warning(
                f"Using approximate conversion from meters to degrees at latitude {avg_lat}. "
                f"For more accurate buffers, reproject data to a projected CRS first."
            )
            
            # Create buffer
            buffer_gdf = gdf.copy()
            buffer_gdf.geometry = gdf.geometry.buffer(distance_degrees)
            
        else:
            raise ValueError(f"Unsupported unit for WGS84: {unit}")
            
    else:
        # For projected CRS, use buffer directly
        buffer_gdf = gdf.copy()
        buffer_gdf.geometry = gdf.geometry.buffer(distance)
    
    # Dissolve buffers if requested
    if dissolve:
        buffer_gdf = buffer_gdf.dissolve()
    
    return buffer_gdf

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
@timer
def find_nearest_points(
    source_gdf: gpd.GeoDataFrame, 
    target_gdf: gpd.GeoDataFrame,
    target_col: str = 'geometry',
    max_distance: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Find nearest point in target_gdf for each point in source_gdf
    
    Parameters
    ----------
    source_gdf : geopandas.GeoDataFrame
        Source points
    target_gdf : geopandas.GeoDataFrame
        Target points
    target_col : str, optional
        Column name in target_gdf for the point geometries
    max_distance : float, optional
        Maximum distance to consider
        
    Returns
    -------
    geopandas.GeoDataFrame
        Source points with nearest target point information
    """
    # Ensure same CRS
    if source_gdf.crs != target_gdf.crs:
        logger.warning(f"CRS mismatch: {source_gdf.crs} vs {target_gdf.crs}. Reprojecting target.")
        target_gdf = target_gdf.to_crs(source_gdf.crs)
    
    # Build spatial index for target points
    target_idx = rtree.index.Index()
    for idx, geometry in enumerate(target_gdf.geometry):
        target_idx.insert(idx, geometry.bounds)
    
    # Prepare result
    source_gdf = source_gdf.copy()
    source_gdf['nearest_idx'] = -1
    source_gdf['distance'] = np.nan
    
    # For each source point, find nearest target
    for idx, source_point in enumerate(source_gdf.geometry):
        # Find potential nearest points using the spatial index
        potential_matches = list(target_idx.nearest(source_point.bounds, 10))
        
        if not potential_matches:
            continue
        
        # Calculate actual distances to find the nearest
        distances = [source_point.distance(target_gdf.geometry.iloc[match_idx]) 
                    for match_idx in potential_matches]
        
        # Find nearest
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        nearest_idx = potential_matches[min_distance_idx]
        
        # Check maximum distance if specified
        if max_distance is not None and min_distance > max_distance:
            continue
        
        # Store result
        source_gdf.at[idx, 'nearest_idx'] = nearest_idx
        source_gdf.at[idx, 'distance'] = min_distance
    
    # Filter points with no match
    source_gdf = source_gdf[source_gdf['nearest_idx'] >= 0].copy()
    
    # Join with target information
    source_gdf = source_gdf.merge(
        target_gdf.reset_index(drop=True),
        left_on='nearest_idx',
        right_index=True,
        suffixes=('', '_target')
    )
    
    return source_gdf

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def overlay_layers(
    base_gdf: gpd.GeoDataFrame, 
    overlay_gdf: gpd.GeoDataFrame,
    how: str = 'intersection',
    keep_columns: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Overlay two GeoDataFrames
    
    Parameters
    ----------
    base_gdf : geopandas.GeoDataFrame
        Base layer
    overlay_gdf : geopandas.GeoDataFrame
        Overlay layer
    how : str, optional
        Overlay operation ('intersection', 'union', 'difference', etc.)
    keep_columns : list, optional
        Columns to keep from overlay_gdf
        
    Returns
    -------
    geopandas.GeoDataFrame
        Result of overlay operation
    """
    # Ensure same CRS
    if base_gdf.crs != overlay_gdf.crs:
        logger.warning(f"CRS mismatch: {base_gdf.crs} vs {overlay_gdf.crs}. Reprojecting overlay.")
        overlay_gdf = overlay_gdf.to_crs(base_gdf.crs)
    
    # Filter columns if specified
    if keep_columns is not None:
        # Ensure 'geometry' is in keep_columns
        if 'geometry' not in keep_columns:
            keep_columns = keep_columns + ['geometry']
        
        # Filter columns that exist in overlay_gdf
        valid_columns = [col for col in keep_columns if col in overlay_gdf.columns]
        overlay_gdf = overlay_gdf[valid_columns]
    
    # Perform overlay
    result = gpd.overlay(base_gdf, overlay_gdf, how=how)
    
    return result

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def calculate_distances(
    origin_gdf: gpd.GeoDataFrame, 
    destination_gdf: gpd.GeoDataFrame,
    origin_id_col: str,
    dest_id_col: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate distances between points in two GeoDataFrames
    
    Parameters
    ----------
    origin_gdf : geopandas.GeoDataFrame
        Origin points
    destination_gdf : geopandas.GeoDataFrame
        Destination points
    origin_id_col : str
        Column with origin IDs
    dest_id_col : str
        Column with destination IDs
    output_path : str, optional
        Path to save the distances matrix
        
    Returns
    -------
    pandas.DataFrame
        Distance matrix
    """
    # Ensure same CRS
    if origin_gdf.crs != destination_gdf.crs:
        destination_gdf = destination_gdf.to_crs(origin_gdf.crs)
    
    # Check if CRS is geographic
    is_geographic = origin_gdf.crs and (
        str(origin_gdf.crs).lower().startswith('epsg:4326') or 
        (hasattr(origin_gdf.crs, 'is_geographic') and origin_gdf.crs.is_geographic)
    )
    
    # Create empty distance matrix
    origins = origin_gdf[origin_id_col].values
    destinations = destination_gdf[dest_id_col].values
    
    distances = np.zeros((len(origins), len(destinations)))
    
    # Calculate distances
    for i, origin in enumerate(origin_gdf.geometry):
        for j, dest in enumerate(destination_gdf.geometry):
            if is_geographic:
                # Use great-circle distance for geographic coordinates
                from geopy.distance import great_circle
                
                # Note: Point coordinates are (x, y) = (lon, lat)
                # But great_circle expects (lat, lon)
                origin_point = (origin.y, origin.x)
                dest_point = (dest.y, dest.x)
                
                distance = great_circle(origin_point, dest_point).meters
            else:
                # Use Euclidean distance for projected coordinates
                distance = origin.distance(dest)
            
            distances[i, j] = distance
    
    # Create DataFrame
    distance_df = pd.DataFrame(distances, index=origins, columns=destinations)
    
    # Save to file if requested
    if output_path:
        distance_df.to_csv(output_path)
    
    return distance_df

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def calculate_distance_matrix(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    method: str = 'euclidean',
    crs: Optional[Union[str, int]] = YEMEN_EPSG,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate distance matrix between points in a GeoDataFrame
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with point geometries
    id_col : str
        Column with point IDs
    method : str, optional
        Distance calculation method ('euclidean', 'great_circle')
    crs : str or int, optional
        CRS to use for distance calculation
    output_path : str, optional
        Path to save the distance matrix
        
    Returns
    -------
    pandas.DataFrame
        Distance matrix
    """
    # Reproject if CRS is specified
    if crs is not None:
        gdf = gdf.to_crs(crs)
    
    # Create empty distance matrix
    ids = gdf[id_col].values
    n = len(ids)
    distances = np.zeros((n, n))
    
    # Calculate distances
    if method == 'euclidean':
        # Use Euclidean distance (faster)
        for i in range(n):
            for j in range(i+1, n):
                dist = gdf.geometry.iloc[i].distance(gdf.geometry.iloc[j])
                distances[i, j] = dist
                distances[j, i] = dist
    
    elif method == 'great_circle':
        # Use great-circle distance (more accurate for long distances)
        from geopy.distance import great_circle
        
        for i in range(n):
            for j in range(i+1, n):
                # Extract coordinates: note the order (lat, lon) for great_circle
                p1 = (gdf.geometry.iloc[i].y, gdf.geometry.iloc[i].x)
                p2 = (gdf.geometry.iloc[j].y, gdf.geometry.iloc[j].x)
                
                dist = great_circle(p1, p2).meters
                distances[i, j] = dist
                distances[j, i] = dist
    
    else:
        raise ValueError(f"Unknown distance calculation method: {method}")
    
    # Create DataFrame
    distance_df = pd.DataFrame(distances, index=ids, columns=ids)
    
    # Save to file if requested
    if output_path:
        distance_df.to_csv(output_path)
    
    return distance_df

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def create_spatial_weight_matrix(
    gdf: gpd.GeoDataFrame,
    method: str = 'knn',
    k: int = 5,
    distance_threshold: Optional[float] = None,
    conflict_col: Optional[str] = None,
    conflict_weight: float = 0.5
) -> Dict[int, Dict[str, Any]]:
    """
    Create spatial weight matrix for spatial econometric analysis
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometries
    method : str, optional
        Weight matrix type ('knn', 'distance', 'contiguity')
    k : int, optional
        Number of nearest neighbors for k-nearest neighbors
    distance_threshold : float, optional
        Distance threshold for distance-based weights
    conflict_col : str, optional
        Column with conflict intensity
    conflict_weight : float, optional
        Weight of conflict in the calculation
        
    Returns
    -------
    dict
        Spatial weight matrix in PySAL format
    """
    from libpysal.weights import KNN, Kernel, Queen, Rook, W
    
    # Create basic weight matrix
    if method == 'knn':
        # K-nearest neighbors
        w = KNN.from_dataframe(gdf, k=k)
        
    elif method == 'distance':
        # Distance-based weights
        if distance_threshold is None:
            raise ValueError("distance_threshold must be provided for distance-based weights")
        
        # Calculate distance-based weights
        w = Kernel.from_dataframe(gdf, bandwidth=distance_threshold)
        
    elif method == 'queen':
        # Queen contiguity (shared edges or vertices)
        w = Queen.from_dataframe(gdf)
        
    elif method == 'rook':
        # Rook contiguity (shared edges only)
        w = Rook.from_dataframe(gdf)
        
    else:
        raise ValueError(f"Unknown spatial weight matrix method: {method}")
    
    # Adjust weights by conflict intensity if requested
    if conflict_col is not None and conflict_col in gdf.columns:
        # Normalize conflict intensity to [0, 1]
        conflict = gdf[conflict_col].values
        if conflict.min() != conflict.max():
            conflict = (conflict - conflict.min()) / (conflict.max() - conflict.min())
        
        # Convert w to dict format for easier manipulation
        new_weights = {}
        
        for i, neighbors in w.weights.items():
            new_weights[i] = []
            for j, weight in zip(w.neighbors[i], neighbors):
                # Calculate conflict barrier: higher conflict = lower weight
                conflict_i = conflict[i]
                conflict_j = conflict[j]
                avg_conflict = (conflict_i + conflict_j) / 2
                conflict_barrier = 1 - conflict_weight * avg_conflict
                
                # Adjust weight by conflict barrier
                new_weight = weight * conflict_barrier
                new_weights[i].append(new_weight)
        
        # Create new weight matrix with adjusted weights
        new_w = W(w.neighbors, new_weights)
        return new_w
    
    return w

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def extract_area_of_interest(
    gdf: gpd.GeoDataFrame,
    area_name: str,
    area_col: str = 'admin1'
) -> gpd.GeoDataFrame:
    """
    Extract a specific area from a GeoDataFrame
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame
    area_name : str
        Name of the area to extract
    area_col : str, optional
        Column containing area names
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame filtered to the specified area
    """
    # Check if area_col exists
    if area_col not in gdf.columns:
        raise ValueError(f"Column {area_col} not found in GeoDataFrame")
    
    # Filter to the specified area
    area_gdf = gdf[gdf[area_col] == area_name].copy()
    
    if len(area_gdf) == 0:
        raise ValueError(f"No features found with {area_col} = {area_name}")
    
    return area_gdf

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def aggregate_points_to_polygons(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    value_col: str,
    agg_func: str = 'mean',
    polygon_id_col: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Aggregate point values to polygons
    
    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        Points with values
    polygons_gdf : geopandas.GeoDataFrame
        Polygons to aggregate to
    value_col : str
        Column with values to aggregate
    agg_func : str, optional
        Aggregation function ('mean', 'sum', 'count', 'min', 'max')
    polygon_id_col : str, optional
        Column with polygon identifiers
        
    Returns
    -------
    geopandas.GeoDataFrame
        Polygons with aggregated values
    """
    # Ensure same CRS
    if points_gdf.crs != polygons_gdf.crs:
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)
    
    # Get polygon ID column
    if polygon_id_col is None:
        # If no ID column is provided, use the index
        polygons_gdf = polygons_gdf.reset_index().rename(columns={'index': 'polygon_id'})
        polygon_id_col = 'polygon_id'
    elif polygon_id_col not in polygons_gdf.columns:
        raise ValueError(f"Column {polygon_id_col} not found in polygons GeoDataFrame")
    
    # Spatial join points to polygons
    joined = gpd.sjoin(points_gdf, polygons_gdf[[polygon_id_col, 'geometry']], how='inner', op='within')
    
    # Check if any points were joined
    if len(joined) == 0:
        logger.warning("No points were found within polygons")
        # Add empty aggregation column to polygons
        result = polygons_gdf.copy()
        agg_col = f"{value_col}_{agg_func}"
        result[agg_col] = np.nan
        return result
    
    # Aggregate values
    if agg_func == 'mean':
        agg = joined.groupby(polygon_id_col)[value_col].mean()
    elif agg_func == 'sum':
        agg = joined.groupby(polygon_id_col)[value_col].sum()
    elif agg_func == 'count':
        agg = joined.groupby(polygon_id_col)[value_col].count()
    elif agg_func == 'min':
        agg = joined.groupby(polygon_id_col)[value_col].min()
    elif agg_func == 'max':
        agg = joined.groupby(polygon_id_col)[value_col].max()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")
    
    # Add aggregated values to polygons
    result = polygons_gdf.copy()
    agg_col = f"{value_col}_{agg_func}"
    result[agg_col] = result[polygon_id_col].map(agg)
    
    return result

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
@m1_optimized(parallel=True)
def compute_accessibility_index(
    markets_gdf: gpd.GeoDataFrame,
    population_gdf: gpd.GeoDataFrame,
    max_distance: float = 50000,
    distance_decay: float = 2.0,
    weight_col: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Compute accessibility index for markets
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market locations
    population_gdf : geopandas.GeoDataFrame
        Population centers
    max_distance : float, optional
        Maximum distance to consider in meters
    distance_decay : float, optional
        Distance decay exponent (higher = stronger decay)
    weight_col : str, optional
        Column with population weights
        
    Returns
    -------
    geopandas.GeoDataFrame
        Markets with accessibility index
    """
    # Ensure same CRS
    if markets_gdf.crs != population_gdf.crs:
        population_gdf = population_gdf.to_crs(markets_gdf.crs)
    
    # Check if CRS is projected
    if not markets_gdf.crs or hasattr(markets_gdf.crs, 'is_geographic') and markets_gdf.crs.is_geographic:
        logger.warning(
            "CRS appears to be geographic. Distance calculations may be inaccurate. "
            "Consider reprojecting to a projected CRS."
        )
    
    # Make a copy of markets GeoDataFrame
    result = markets_gdf.copy()
    
    # Initialize accessibility column
    result['accessibility'] = 0.0
    
    # Get weights if specified
    if weight_col and weight_col in population_gdf.columns:
        weights = population_gdf[weight_col].values
    else:
        # Default: equal weights
        weights = np.ones(len(population_gdf))
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Build spatial index for population centers
    population_idx = rtree.index.Index()
    for idx, geometry in enumerate(population_gdf.geometry):
        population_idx.insert(idx, geometry.bounds)
    
    # Compute accessibility for each market
    for i, market in enumerate(result.geometry):
        accessibility = 0.0
        
        # Find population centers within max_distance
        potential_matches = list(population_idx.nearest(market.bounds, 100))
        
        for j in potential_matches:
            pop_point = population_gdf.geometry.iloc[j]
            distance = market.distance(pop_point)
            
            if distance <= max_distance:
                # Apply distance decay function
                accessibility += weights[j] / (1 + distance ** distance_decay)
        
        result.at[i, 'accessibility'] = accessibility
    
    # Normalize accessibility scores
    if not result['accessibility'].empty and result['accessibility'].max() > 0:
        result['accessibility'] = result['accessibility'] / result['accessibility'].max()
    
    return result

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def create_exchange_regime_boundaries(
    gdf: gpd.GeoDataFrame,
    regime_col: str = 'exchange_rate_regime',
    dissolve: bool = True,
    simplify_tolerance: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Create boundaries between exchange rate regimes
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with exchange rate regime information
    regime_col : str, optional
        Column with regime information
    dissolve : bool, optional
        Whether to dissolve geometries by regime
    simplify_tolerance : float, optional
        Tolerance for simplifying boundaries
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with regime boundaries
    """
    # Check if regime column exists
    if regime_col not in gdf.columns:
        raise ValueError(f"Column {regime_col} not found in GeoDataFrame")
    
    # Get unique regimes
    regimes = gdf[regime_col].unique()
    
    if len(regimes) < 2:
        logger.warning(f"Only one regime found: {regimes[0]}")
        return gdf
    
    # Create GeoDataFrame for each regime
    regime_gdfs = []
    
    for regime in regimes:
        regime_gdf = gdf[gdf[regime_col] == regime].copy()
        
        if dissolve:
            regime_gdf = regime_gdf.dissolve(by=regime_col)
            # Reset index to get regime as a column
            regime_gdf = regime_gdf.reset_index()
        
        if simplify_tolerance is not None:
            # Simplify geometries
            regime_gdf['geometry'] = regime_gdf.geometry.simplify(simplify_tolerance)
        
        regime_gdfs.append(regime_gdf)
    
    # Find boundaries between regimes
    boundaries = []
    
    for i, regime1_gdf in enumerate(regime_gdfs):
        for j in range(i+1, len(regime_gdfs)):
            regime2_gdf = regime_gdfs[j]
            
            # Find intersection of the boundaries
            # First, get the boundary of each regime's area
            boundary1 = regime1_gdf.boundary.unary_union
            boundary2 = regime2_gdf.boundary.unary_union
            
            # Find the intersection (the shared boundary)
            shared_boundary = boundary1.intersection(boundary2)
            
            if not shared_boundary.is_empty:
                # Create GeoDataFrame with the shared boundary
                boundary_gdf = gpd.GeoDataFrame(
                    {
                        'regime1': regime1_gdf[regime_col].iloc[0],
                        'regime2': regime2_gdf[regime_col].iloc[0],
                        'geometry': [shared_boundary]
                    }, 
                    crs=gdf.crs
                )
                
                boundaries.append(boundary_gdf)
    
    if not boundaries:
        logger.warning("No boundaries found between regimes")
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)
    
    # Combine all boundaries
    result = pd.concat(boundaries, ignore_index=True)
    
    return result

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def calculate_market_isolation(
    markets_gdf: gpd.GeoDataFrame,
    transport_network_gdf: Optional[gpd.GeoDataFrame] = None,
    population_gdf: Optional[gpd.GeoDataFrame] = None,
    conflict_col: Optional[str] = None,
    max_distance: float = 50000
) -> gpd.GeoDataFrame:
    """
    Calculate market isolation index based on distance to infrastructure, 
    population, and conflict barriers
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market locations
    transport_network_gdf : geopandas.GeoDataFrame, optional
        Transportation network
    population_gdf : geopandas.GeoDataFrame, optional
        Population centers
    conflict_col : str, optional
        Column with conflict intensity
    max_distance : float, optional
        Maximum distance to consider in meters
        
    Returns
    -------
    geopandas.GeoDataFrame
        Markets with isolation index
    """
    # Make a copy of markets GeoDataFrame
    result = markets_gdf.copy()
    
    # Initialize isolation components
    components = []
    
    # 1. Distance to transportation network
    if transport_network_gdf is not None:
        # Ensure same CRS
        if markets_gdf.crs != transport_network_gdf.crs:
            transport_network_gdf = transport_network_gdf.to_crs(markets_gdf.crs)
        
        # Calculate distance to nearest road/transport infrastructure
        result['dist_to_transport'] = float('inf')
        
        for i, market in enumerate(result.geometry):
            min_distance = float('inf')
            
            for line in transport_network_gdf.geometry:
                distance = market.distance(line)
                min_distance = min(min_distance, distance)
            
            result.at[i, 'dist_to_transport'] = min_distance
        
        # Normalize distances
        max_dist = result['dist_to_transport'].max()
        if max_dist > 0:
            result['transport_isolation'] = result['dist_to_transport'] / max_dist
        else:
            result['transport_isolation'] = 0
        
        components.append('transport_isolation')
    
    # 2. Distance to population centers
    if population_gdf is not None:
        # Ensure same CRS
        if markets_gdf.crs != population_gdf.crs:
            population_gdf = population_gdf.to_crs(markets_gdf.crs)
        
        # Find nearest population center
        nearest_pop = find_nearest_points(result, population_gdf)
        result['dist_to_population'] = nearest_pop['distance']
        
        # Normalize distances
        max_dist = result['dist_to_population'].max()
        if max_dist > 0:
            result['population_isolation'] = result['dist_to_population'] / max_dist
        else:
            result['population_isolation'] = 0
        
        components.append('population_isolation')
    
    # 3. Conflict barrier
    if conflict_col and conflict_col in result.columns:
        # Normalize conflict intensity
        conflict = result[conflict_col].values
        if conflict.min() != conflict.max():
            result['conflict_isolation'] = (conflict - conflict.min()) / (conflict.max() - conflict.min())
        else:
            result['conflict_isolation'] = 0
        
        components.append('conflict_isolation')
    
    # Calculate overall isolation index as average of components
    if components:
        result['isolation_index'] = result[components].mean(axis=1)
    else:
        result['isolation_index'] = 0
    
    return result

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def assign_exchange_rate_regime(
    points_gdf: gpd.GeoDataFrame,
    regime_polygons_gdf: gpd.GeoDataFrame,
    regime_col: str = 'exchange_rate_regime',
    default_regime: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Assign exchange rate regime to points based on spatial location
    
    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        Point locations
    regime_polygons_gdf : geopandas.GeoDataFrame
        Polygons with exchange rate regime information
    regime_col : str, optional
        Column with regime information
    default_regime : str, optional
        Default regime for points outside all polygons
        
    Returns
    -------
    geopandas.GeoDataFrame
        Points with assigned exchange rate regime
    """
    # Check if regime column exists in polygons
    if regime_col not in regime_polygons_gdf.columns:
        raise ValueError(f"Column {regime_col} not found in regime polygons GeoDataFrame")
    
    # Ensure same CRS
    if points_gdf.crs != regime_polygons_gdf.crs:
        points_gdf = points_gdf.to_crs(regime_polygons_gdf.crs)
    
    # Make a copy of points GeoDataFrame
    result = points_gdf.copy()
    
    # Perform spatial join
    joined = gpd.sjoin(result, regime_polygons_gdf[[regime_col, 'geometry']], how='left', op='within')
    
    # Rename joined column to avoid confusion
    joined = joined.rename(columns={f"{regime_col}_right": regime_col})
    
    # Fill NaN values with default regime if provided
    if default_regime is not None:
        joined[regime_col] = joined[regime_col].fillna(default_regime)
    
    # Keep original columns plus regime column
    result_cols = points_gdf.columns.tolist()
    if regime_col not in result_cols:
        result_cols.append(regime_col)
    
    # Return result with assigned regimes
    result = joined[result_cols]
    
    return result

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def create_market_catchments(
    markets_gdf: gpd.GeoDataFrame,
    population_gdf: gpd.GeoDataFrame,
    market_id_col: str,
    population_weight_col: Optional[str] = None,
    max_distance: float = 50000,
    distance_decay: float = 2.0
) -> gpd.GeoDataFrame:
    """
    Create market catchment areas based on gravity model
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        Market locations
    population_gdf : geopandas.GeoDataFrame
        Population points
    market_id_col : str
        Column with market identifiers
    population_weight_col : str, optional
        Column with population weights
    max_distance : float, optional
        Maximum distance to consider in meters
    distance_decay : float, optional
        Distance decay exponent
        
    Returns
    -------
    geopandas.GeoDataFrame
        Population points with assigned market catchments
    """
    # Ensure same CRS
    if markets_gdf.crs != population_gdf.crs:
        population_gdf = population_gdf.to_crs(markets_gdf.crs)
    
    # Make a copy of population GeoDataFrame
    result = population_gdf.copy()
    
    # Get market weights (default: equal weights)
    markets_gdf = markets_gdf.copy()
    markets_gdf['market_weight'] = 1.0
    
    # Get population weights if specified
    if population_weight_col and population_weight_col in result.columns:
        result['pop_weight'] = result[population_weight_col]
    else:
        result['pop_weight'] = 1.0
    
    # Initialize columns for market assignment
    result['nearest_market'] = None
    result['market_distance'] = float('inf')
    result['market_gravity'] = 0.0
    
    # Build spatial index for markets
    market_idx = rtree.index.Index()
    for idx, geometry in enumerate(markets_gdf.geometry):
        market_idx.insert(idx, geometry.bounds)
    
    # For each population point, find the market with highest gravity
    for i, pop_point in enumerate(result.geometry):
        max_gravity = 0.0
        best_market = None
        best_distance = float('inf')
        
        # Find markets within max_distance
        potential_markets = list(market_idx.nearest(pop_point.bounds, 10))
        
        for j in potential_markets:
            market_point = markets_gdf.geometry.iloc[j]
            distance = pop_point.distance(market_point)
            
            if distance <= max_distance:
                # Calculate gravity using inverse power function
                market_weight = markets_gdf['market_weight'].iloc[j]
                gravity = market_weight / (distance ** distance_decay)
                
                if gravity > max_gravity:
                    max_gravity = gravity
                    best_market = markets_gdf[market_id_col].iloc[j]
                    best_distance = distance
        
        if best_market is not None:
            result.at[i, 'nearest_market'] = best_market
            result.at[i, 'market_distance'] = best_distance
            result.at[i, 'market_gravity'] = max_gravity
    
    # Remove unassigned points
    result = result[result['nearest_market'].notna()].copy()
    
    return result

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def create_conflict_adjusted_weights(
    gdf: gpd.GeoDataFrame,
    k: int = 5,
    conflict_col: str = 'conflict_intensity_normalized',
    conflict_weight: float = 0.5
) -> Any:  # Using Any because libpysal might not be imported in typing
    """
    Create spatial weight matrix adjusted by conflict intensity.
    
    In Yemen's context, conflict creates barriers between markets
    that aren't captured by geographic distance alone.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with point geometries
    k : int, optional
        Number of nearest neighbors
    conflict_col : str, optional
        Column with conflict intensity values (0-1 normalized)
    conflict_weight : float, optional
        Weight of conflict in distance adjustment (0-1)
        
    Returns
    -------
    libpysal.weights.W
        Spatial weight matrix adjusted by conflict
    """
    from libpysal.weights import KNN
    import copy
    
    # Validate input
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame")
    
    # Create base weights using KNN
    knn = KNN.from_dataframe(gdf, k=k)
    
    # If no conflict adjustment requested, return base weights
    if conflict_col not in gdf.columns or conflict_weight == 0:
        return knn
    
    # Create conflict-adjusted weights
    conflict_values = gdf[conflict_col].values
    w = copy.deepcopy(knn)
    
    # Adjust weights based on conflict
    for idx, (i, neighbors) in enumerate(w.neighbors.items()):
        new_weights = []
        for j_idx, j in enumerate(neighbors):
            # Get the actual indices in the conflict_values array
            # Use the position in the GeoDataFrame, not the original index
            i_pos = list(w.neighbors.keys()).index(i)
            j_pos = list(w.neighbors.keys()).index(j)
            
            # Ensure indices are within bounds
            if i_pos < len(conflict_values) and j_pos < len(conflict_values):
                # Calculate average conflict intensity between regions
                conflict_factor = (conflict_values[i_pos] + conflict_values[j_pos]) / 2
                # Reduce weight (increase distance) based on conflict
                original_weight = w.weights[i][j_idx]
                new_weight = original_weight * (1 - conflict_weight * conflict_factor)
                new_weights.append(new_weight)
            else:
                # If indices are out of bounds, keep original weight
                logger.warning(f"Index out of bounds in conflict adjustment: {i_pos}, {j_pos}, max={len(conflict_values)-1}")
                new_weights.append(w.weights[i][j_idx])
        
        w.weights[i] = new_weights
    
    # Ensure symmetry
    w = w.symmetrize()
    
    return w

@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def calculate_exchange_rate_boundary(
    markets_gdf: gpd.GeoDataFrame,
    regime_col: str = 'exchange_rate_regime',
    buffer_distance: float = 5000  # 5km buffer
) -> gpd.GeoDataFrame:
    """
    Calculate the boundary between different exchange rate regimes in Yemen.
    
    Parameters
    ----------
    markets_gdf : geopandas.GeoDataFrame
        GeoDataFrame with market locations and regime information
    regime_col : str, optional
        Column with exchange rate regime information
    buffer_distance : float, optional
        Buffer distance in coordinate units (e.g., meters if projected)
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with the boundary geometry
    """
    from shapely.ops import unary_union
    
    # Validate input
    if not isinstance(markets_gdf, gpd.GeoDataFrame):
        raise ValueError("Input must be a GeoDataFrame")
    
    if regime_col not in markets_gdf.columns:
        raise ValueError(f"Column {regime_col} not found in GeoDataFrame")
    
    # Check if input is projected
    if not markets_gdf.crs or markets_gdf.crs.is_geographic:
        logger.warning(
            "Input GeoDataFrame uses geographic coordinates. "
            "Consider reprojecting to a projected CRS for accurate distances."
        )
    
    # Split markets by regime
    regimes = markets_gdf[regime_col].unique()
    
    if len(regimes) < 2:
        raise ValueError(f"Found only one regime: {regimes[0]}. Need at least two regimes.")
    
    # Create regime polygons by buffering and dissolving
    regime_polygons = {}
    
    for regime in regimes:
        # Get markets for this regime
        regime_markets = markets_gdf[markets_gdf[regime_col] == regime]
        
        # Buffer points and dissolve
        buffered = regime_markets.copy()
        buffered['geometry'] = regime_markets.geometry.buffer(buffer_distance)
        dissolved = unary_union(buffered.geometry)
        
        regime_polygons[regime] = dissolved
    
    # Create boundary by intersecting the boundaries of each regime's area
    boundaries = []
    regime_list = list(regime_polygons.keys())
    
    for i in range(len(regime_list)):
        for j in range(i+1, len(regime_list)):
            regime1 = regime_list[i]
            regime2 = regime_list[j]
            
            # Get boundary of each regime area
            boundary1 = regime_polygons[regime1].boundary
            boundary2 = regime_polygons[regime2].boundary
            
            # Find intersection (the shared boundary)
            shared = boundary1.intersection(boundary2)
            
            if not shared.is_empty:
                boundaries.append({
                    'geometry': shared,
                    'regime1': regime1,
                    'regime2': regime2
                })
    
    # Create GeoDataFrame from boundaries
    if not boundaries:
        logger.warning("No boundaries found between regimes")
        return gpd.GeoDataFrame(geometry=[], crs=markets_gdf.crs)
    
    boundary_gdf = gpd.GeoDataFrame(boundaries, crs=markets_gdf.crs)
    
    return boundary_gdf