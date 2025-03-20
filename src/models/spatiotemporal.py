"""
Spatiotemporal integration module for Yemen Market Integration Analysis.

This module connects spatial and time series models to analyze market
integration patterns across both space and time dimensions. It implements
optimized algorithms for large-scale analysis of market integration.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from functools import partial
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans
import statsmodels.api as sm

from src.utils.validation import (
    validate_dataframe, 
    validate_spatial_dataframe,
    validate_column_presence
)
from src.utils.error_handler import handle_errors
from src.utils.performance_utils import (
    optimize_memory_usage, 
    timer, 
    parallelize, 
    chunk_process
)
from src.models.threshold import ThresholdCointegration
from src.models.spatial import SpatialEconometrics, calculate_spatial_weights
from src.models.unit_root import UnitRootTester


class SpatialTemporalIntegration:
    """
    Integrated analysis of market integration across space and time.
    
    This class combines spatial and time series models to analyze
    market integration patterns, with optimized methods for handling
    large datasets efficiently.
    """
    
    def __init__(
        self, 
        market_data: pd.DataFrame, 
        spatial_data: Optional[gpd.GeoDataFrame] = None,
        date_col: str = "date",
        market_id_col: str = "market_id",
        price_col: str = "price",
        commodity_col: str = "commodity",
        region_col: Optional[str] = "exchange_rate_regime",
        conflict_col: Optional[str] = "conflict_intensity_normalized",
        geometry_col: str = "geometry"
    ):
        """
        Initialize the spatiotemporal integration model.
        
        Parameters
        ----------
        market_data : pandas.DataFrame
            Time series data for market prices, containing at minimum
            date, market ID, and price columns
        spatial_data : geopandas.GeoDataFrame, optional
            Spatial data for markets, with market IDs and geometries
        date_col : str, default="date"
            Name of date column in market_data
        market_id_col : str, default="market_id"
            Name of market ID column in both dataframes
        price_col : str, default="price"
            Name of price column in market_data
        commodity_col : str, default="commodity"
            Name of commodity column in market_data
        region_col : str, optional, default="exchange_rate_regime"
            Name of region/regime column in market_data
        conflict_col : str, optional, default="conflict_intensity_normalized"
            Name of conflict intensity column in market_data
        geometry_col : str, default="geometry"
            Name of geometry column in spatial_data
        """
        # Validate market data
        validate_dataframe(market_data)
        required_market_cols = [date_col, market_id_col, price_col]
        validate_column_presence(market_data, required_market_cols)
        
        # Convert date column to datetime if needed
        if pd.api.types.is_object_dtype(market_data[date_col]):
            market_data = market_data.copy()
            market_data[date_col] = pd.to_datetime(market_data[date_col])
        
        # Check if spatial data is provided
        if spatial_data is not None:
            validate_spatial_dataframe(spatial_data)
            required_spatial_cols = [market_id_col, geometry_col]
            validate_column_presence(spatial_data, required_spatial_cols)
            self.has_spatial_data = True
        else:
            self.has_spatial_data = False
        
        # Store data and column names
        self.market_data = optimize_memory_usage(market_data)
        self.spatial_data = spatial_data
        self.date_col = date_col
        self.market_id_col = market_id_col
        self.price_col = price_col
        self.commodity_col = commodity_col
        self.region_col = region_col
        self.conflict_col = conflict_col
        self.geometry_col = geometry_col
        
        # Initialize internal attributes
        self._market_pairs = None
        self._distance_matrix = None
        self._cointegration_results = {}
        self._spatial_clusters = None
        self._price_correlation_matrix = None
        
        # Extract unique markets, dates, and commodities
        self.markets = sorted(market_data[market_id_col].unique())
        self.dates = sorted(market_data[date_col].unique())
        
        if commodity_col in market_data.columns:
            self.commodities = sorted(market_data[commodity_col].unique())
        else:
            self.commodities = ["Unknown"]
        
        # Create spatial models if spatial data is available
        if self.has_spatial_data:
            self.spatial_model = SpatialEconometrics(
                spatial_data, 
                id_col=market_id_col, 
                geometry_col=geometry_col
            )
    
    @timer
    def calculate_market_distances(
        self, 
        conflict_weight: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate distances between all markets.
        
        Parameters
        ----------
        conflict_weight : float, optional
            Weight to apply to conflict intensity when calculating
            effective distances (0 = no effect, higher = more effect)
        
        Returns
        -------
        numpy.ndarray
            Distance matrix between markets
        """
        if not self.has_spatial_data:
            raise ValueError("Spatial data is required to calculate market distances")
        
        # Check if we've already calculated the distance matrix
        if self._distance_matrix is not None:
            return self._distance_matrix
        
        # Create market ID to index mapping
        market_indices = {market: i for i, market in enumerate(self.markets)}
        
        # Initialize distance matrix
        n_markets = len(self.markets)
        distances = np.zeros((n_markets, n_markets))
        
        # Calculate geographic distances
        market_points = self.spatial_data.set_index(self.market_id_col)[self.geometry_col]
        
        # Optimize by using vectorized operations where possible
        for i, market1 in enumerate(self.markets):
            if market1 not in market_points.index:
                continue
                
            point1 = market_points.loc[market1]
            
            for j in range(i+1, n_markets):
                market2 = self.markets[j]
                
                if market2 not in market_points.index:
                    continue
                    
                point2 = market_points.loc[market2]
                dist = point1.distance(point2)
                
                # Store distance in matrix (symmetric)
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Apply conflict weights if specified
        if conflict_weight is not None and self.conflict_col in self.market_data.columns:
            # Calculate average conflict intensity by market
            conflict_by_market = self.market_data.groupby(self.market_id_col)[self.conflict_col].mean()
            
            # Apply conflict adjustment to distances
            for i, market1 in enumerate(self.markets):
                if market1 not in conflict_by_market.index:
                    continue
                    
                conflict1 = conflict_by_market.loc[market1]
                
                for j in range(i+1, n_markets):
                    market2 = self.markets[j]
                    
                    if market2 not in conflict_by_market.index:
                        continue
                        
                    conflict2 = conflict_by_market.loc[market2]
                    
                    # Average conflict between markets
                    avg_conflict = (conflict1 + conflict2) / 2
                    
                    # Increase distance based on conflict
                    conflict_factor = 1 + (conflict_weight * avg_conflict)
                    
                    # Apply to both i,j and j,i (symmetric)
                    distances[i, j] *= conflict_factor
                    distances[j, i] *= conflict_factor
        
        # Store the distance matrix
        self._distance_matrix = distances
        
        return distances
    
    @timer
    def identify_market_pairs(
        self, 
        max_distance: Optional[float] = None,
        max_pairs: Optional[int] = None,
        commodity: Optional[str] = None,
        region: Optional[str] = None,
        method: str = "distance"
    ) -> List[Tuple[str, str]]:
        """
        Identify market pairs for cointegration analysis.
        
        Parameters
        ----------
        max_distance : float, optional
            Maximum distance between markets to consider as pairs
        max_pairs : int, optional
            Maximum number of pairs to analyze
        commodity : str, optional
            Filter to a specific commodity
        region : str, optional
            Filter to a specific region
        method : str, default="distance"
            Method to use for selecting pairs:
            "distance" - Select pairs based on distance
            "correlation" - Select pairs based on price correlation
            "both" - Consider both distance and correlation
        
        Returns
        -------
        List[Tuple[str, str]]
            List of market pairs for analysis
        """
        # Filter market data if commodity or region specified
        filtered_data = self.market_data.copy()
        
        if commodity is not None and self.commodity_col in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[self.commodity_col] == commodity]
        
        if region is not None and self.region_col in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[self.region_col] == region]
        
        # Get unique markets after filtering
        filtered_markets = sorted(filtered_data[self.market_id_col].unique())
        
        # Limit to markets that have spatial data if method involves distance
        if method in ["distance", "both"] and self.has_spatial_data:
            spatial_markets = set(self.spatial_data[self.market_id_col])
            filtered_markets = [m for m in filtered_markets if m in spatial_markets]
        
        # Calculate all possible pairs
        all_pairs = []
        for i, market1 in enumerate(filtered_markets):
            for j in range(i+1, len(filtered_markets)):
                market2 = filtered_markets[j]
                all_pairs.append((market1, market2))
        
        # If no distance or correlation criteria, return all pairs (up to max_pairs)
        if method == "all":
            if max_pairs is not None and len(all_pairs) > max_pairs:
                # Randomly sample pairs
                selected_indices = np.random.choice(
                    range(len(all_pairs)), 
                    size=max_pairs, 
                    replace=False
                )
                selected_pairs = [all_pairs[i] for i in selected_indices]
                return selected_pairs
            else:
                return all_pairs
        
        # Score pairs based on selected method
        pair_scores = []
        
        if method in ["distance", "both"] and self.has_spatial_data:
            # Calculate distances if not already done
            if self._distance_matrix is None:
                self.calculate_market_distances()
            
            # Create indices for filtered markets
            market_indices = {m: i for i, m in enumerate(self.markets)}
            
            # Score based on distance (closer is better)
            for market1, market2 in all_pairs:
                idx1 = market_indices.get(market1)
                idx2 = market_indices.get(market2)
                
                if idx1 is None or idx2 is None:
                    continue
                
                distance = self._distance_matrix[idx1, idx2]
                
                # Skip if beyond max distance
                if max_distance is not None and distance > max_distance:
                    continue
                
                # Closer is better (negative score)
                pair_scores.append(((market1, market2), -distance))
        
        if method in ["correlation", "both"]:
            # Calculate price correlation if not done already
            if self._price_correlation_matrix is None:
                self._price_correlation_matrix = self._calculate_price_correlation_matrix(
                    filtered_data, commodity
                )
            
            corr_matrix = self._price_correlation_matrix
            
            # Create indices for markets in correlation matrix
            corr_markets = list(corr_matrix.index)
            
            # Score based on correlation (higher is better)
            for market1, market2 in all_pairs:
                if market1 not in corr_markets or market2 not in corr_markets:
                    continue
                
                correlation = corr_matrix.loc[market1, market2]
                
                # Higher correlation is better
                if method == "correlation":
                    pair_scores.append(((market1, market2), correlation))
                elif method == "both" and ((market1, market2), -distance) in pair_scores:
                    # Combine distance and correlation (normalize both between 0-1)
                    distance_idx = next(i for i, (pair, score) in enumerate(pair_scores) 
                                      if pair == (market1, market2))
                    distance_score = pair_scores[distance_idx][1]
                    
                    # Simple combination (can be adjusted)
                    combined_score = correlation - (distance_score / max(abs(s[1]) for s in pair_scores))
                    pair_scores[distance_idx] = ((market1, market2), combined_score)
        
        # Sort by score (higher is better)
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max_pairs
        if max_pairs is not None and len(pair_scores) > max_pairs:
            pair_scores = pair_scores[:max_pairs]
        
        # Extract pairs from scores
        selected_pairs = [pair for pair, _ in pair_scores]
        
        # Store selected pairs
        self._market_pairs = selected_pairs
        
        return selected_pairs
    
    @staticmethod
    def _calculate_price_correlation_matrix(data, commodity=None):
        """
        Calculate price correlation matrix between markets.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Market data to use
        commodity : str, optional
            Commodity to filter on
        
        Returns
        -------
        pandas.DataFrame
            Correlation matrix between markets
        """
        # Filter by commodity if specified
        if commodity is not None and 'commodity' in data.columns:
            data = data[data['commodity'] == commodity]
        
        # Pivot to wide format (markets as columns)
        pivot_data = data.pivot_table(
            index='date',
            columns='market_id',
            values='price'
        )
        
        # Calculate correlation matrix
        corr_matrix = pivot_data.corr()
        
        return corr_matrix
    
    @timer
    def analyze_market_cointegration(
        self, 
        market_pairs: Optional[List[Tuple[str, str]]] = None,
        commodity: Optional[str] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Analyze cointegration between market pairs.
        
        Parameters
        ----------
        market_pairs : List[Tuple[str, str]], optional
            List of market pairs to analyze (if None, uses previously identified)
        commodity : str, optional
            Filter to a specific commodity
        max_workers : int, optional
            Maximum number of parallel workers to use
        progress_callback : callable, optional
            Function to call with progress updates
        
        Returns
        -------
        Dict[Tuple[str, str], Dict]
            Dictionary of cointegration results by market pair
        """
        # Use previously identified pairs if none provided
        if market_pairs is None:
            if self._market_pairs is None:
                # Identify pairs using default method
                market_pairs = self.identify_market_pairs()
            else:
                market_pairs = self._market_pairs
        
        # Filter market data if commodity specified
        filtered_data = self.market_data.copy()
        if commodity is not None and self.commodity_col in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[self.commodity_col] == commodity]
        
        # Create function to analyze a single pair
        def analyze_pair(pair):
            market1, market2 = pair
            
            # Get price series for each market
            market1_prices = filtered_data[filtered_data[self.market_id_col] == market1]
            market2_prices = filtered_data[filtered_data[self.market_id_col] == market2]
            
            # Ensure the dates match
            common_dates = sorted(set(market1_prices[self.date_col]) & 
                                 set(market2_prices[self.date_col]))
            
            # Skip if insufficient data points
            if len(common_dates) < 30:
                return pair, {
                    'error': f"Insufficient data points for analysis (n={len(common_dates)})"
                }
            
            # Create aligned price series
            market1_series = market1_prices[market1_prices[self.date_col].isin(common_dates)]
            market2_series = market2_prices[market2_prices[self.date_col].isin(common_dates)]
            
            # Sort by date
            market1_series = market1_series.sort_values(by=self.date_col)
            market2_series = market2_series.sort_values(by=self.date_col)
            
            # Extract prices
            price1 = market1_series[self.price_col].values
            price2 = market2_series[self.price_col].values
            dates = market1_series[self.date_col].values
            
            # Create threshold cointegration model
            model = ThresholdCointegration(
                price1, price2,
                market1_name=market1,
                market2_name=market2
            )
            
            try:
                # Estimate cointegration
                coint_result = model.estimate_cointegration()
                
                # If cointegrated, estimate threshold and TAR/M-TAR models
                if coint_result['cointegrated']:
                    # Estimate threshold
                    threshold_result = model.estimate_threshold()
                    
                    # Store dates and price differentials for visualization
                    model.dates = dates
                    model.price_diff = price1 - (price2 * model.beta)
                    
                    # Estimate M-TAR model
                    mtar_result = model.estimate_mtar()
                    
                    # Calculate integration metrics
                    half_lives = model.calculate_half_lives()
                    
                    # Combine results
                    result = {
                        'cointegration': coint_result,
                        'threshold': threshold_result,
                        'mtar': mtar_result,
                        'half_lives': half_lives,
                        'model': model
                    }
                else:
                    # Just return cointegration result
                    result = {'cointegration': coint_result}
                
                return pair, result
                
            except Exception as e:
                return pair, {'error': str(e)}
        
        # Process pairs in parallel if max_workers specified
        if max_workers and max_workers > 1:
            results = parallelize(
                analyze_pair, 
                market_pairs,
                max_workers=max_workers,
                progress_callback=progress_callback
            )
        else:
            # Process sequentially
            results = []
            total = len(market_pairs)
            
            for i, pair in enumerate(market_pairs):
                result = analyze_pair(pair)
                results.append(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, total)
        
        # Convert results to dictionary
        cointegration_results = dict(results)
        
        # Store results
        self._cointegration_results = cointegration_results
        
        return cointegration_results
    
    @timer
    def identify_spatial_clusters(
        self, 
        commodity: Optional[str] = None,
        n_clusters: Optional[int] = None,
        eps: Optional[float] = None,
        min_samples: int = 3,
        scaling: bool = True,
        method: str = "dbscan"
    ) -> Dict[str, Any]:
        """
        Identify spatial clusters of integrated markets.
        
        Parameters
        ----------
        commodity : str, optional
            Filter to a specific commodity
        n_clusters : int, optional
            Number of clusters (for KMeans)
        eps : float, optional
            Maximum distance between samples (for DBSCAN)
        min_samples : int, default=3
            Minimum samples in a cluster (for DBSCAN)
        scaling : bool, default=True
            Whether to scale features before clustering
        method : str, default="dbscan"
            Clustering method: "dbscan" or "kmeans"
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of clustering results
        """
        if not self.has_spatial_data:
            raise ValueError("Spatial data is required for spatial clustering")
        
        # Filter market data if commodity specified
        filtered_data = self.market_data.copy()
        if commodity is not None and self.commodity_col in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[self.commodity_col] == commodity]
        
        # Calculate features for clustering
        features = self._calculate_clustering_features(filtered_data, commodity)
        
        # Apply scaling if requested
        if scaling:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
        else:
            features_scaled = features
        
        # Perform clustering
        if method == "dbscan":
            # Auto-determine eps if not provided
            if eps is None:
                # Use heuristic: average of 5th percentile of distances
                distances = pdist(features_scaled)
                eps = np.percentile(distances, 5)
            
            # Apply DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)
            
            # Extract labels
            labels = clustering.labels_
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Calculate silhouette score if more than one cluster
            silhouette = None
            if n_clusters_found > 1:
                silhouette = silhouette_score(features_scaled, labels)
            
            result = {
                'labels': labels,
                'n_clusters': n_clusters_found,
                'eps': eps,
                'min_samples': min_samples,
                'silhouette': silhouette,
                'method': 'dbscan'
            }
            
        elif method == "kmeans":
            # Auto-determine n_clusters if not provided
            if n_clusters is None:
                # Try different values and use silhouette score
                silhouette_scores = []
                n_range = range(2, min(len(features), 10))
                
                for n in n_range:
                    kmeans = KMeans(n_clusters=n, random_state=42).fit(features_scaled)
                    score = silhouette_score(features_scaled, kmeans.labels_)
                    silhouette_scores.append((n, score))
                
                # Choose n with highest score
                n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            
            # Apply KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features_scaled)
            
            # Extract labels and centers
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # Calculate silhouette score
            silhouette = silhouette_score(features_scaled, labels)
            
            result = {
                'labels': labels,
                'n_clusters': n_clusters,
                'centers': centers,
                'silhouette': silhouette,
                'method': 'kmeans'
            }
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Add market information to result
        result['markets'] = self.markets
        
        # Group markets by cluster
        market_clusters = {}
        for market, label in zip(self.markets, labels):
            if label not in market_clusters:
                market_clusters[label] = []
            market_clusters[label].append(market)
        
        result['market_clusters'] = market_clusters
        
        # Store result
        self._spatial_clusters = result
        
        return result
    
    def _calculate_clustering_features(self, data, commodity=None):
        """
        Calculate features for clustering markets.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Market data to use
        commodity : str, optional
            Commodity to filter on
        
        Returns
        -------
        numpy.ndarray
            Feature matrix for clustering
        """
        # Extract spatial features
        spatial_features = []
        
        if self.has_spatial_data:
            # Get coordinates
            market_coords = {}
            for _, row in self.spatial_data.iterrows():
                market_id = row[self.market_id_col]
                if hasattr(row[self.geometry_col], 'x') and hasattr(row[self.geometry_col], 'y'):
                    market_coords[market_id] = (row[self.geometry_col].x, row[self.geometry_col].y)
            
            # Create coordinate features
            for market in self.markets:
                if market in market_coords:
                    x, y = market_coords[market]
                    spatial_features.append([x, y])
                else:
                    # Use NaN for missing coordinates
                    spatial_features.append([np.nan, np.nan])
        
        # Extract price features
        price_features = []
        
        # Filter by commodity if specified
        if commodity is not None and self.commodity_col in data.columns:
            data = data[data[self.commodity_col] == commodity]
        
        # Calculate price statistics by market
        price_stats = data.groupby(self.market_id_col)[self.price_col].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        # Create price features
        price_stats_dict = {
            row[self.market_id_col]: [row['mean'], row['std'], row['min'], row['max']]
            for _, row in price_stats.iterrows()
        }
        
        for market in self.markets:
            if market in price_stats_dict:
                price_features.append(price_stats_dict[market])
            else:
                # Use NaN for missing price stats
                price_features.append([np.nan, np.nan, np.nan, np.nan])
        
        # Combine features
        if spatial_features and price_features:
            # Both spatial and price features
            features = np.hstack([spatial_features, price_features])
        elif spatial_features:
            # Only spatial features
            features = np.array(spatial_features)
        elif price_features:
            # Only price features
            features = np.array(price_features)
        else:
            raise ValueError("No features available for clustering")
        
        # Handle missing values
        features = np.nan_to_num(features)
        
        return features
    
    @timer
    def calculate_spatial_integration_metrics(
        self,
        commodity: Optional[str] = None,
        threshold_distance: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Calculate spatial integration metrics for markets.
        
        Parameters
        ----------
        commodity : str, optional
            Filter to a specific commodity
        threshold_distance : float, optional
            Maximum distance for considering markets as neighbors
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with integration metrics by market
        """
        if not self.has_spatial_data:
            raise ValueError("Spatial data is required for spatial integration metrics")
        
        # Filter market data if commodity specified
        filtered_data = self.market_data.copy()
        if commodity is not None and self.commodity_col in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[self.commodity_col] == commodity]
        
        # Calculate distance matrix if not already done
        if self._distance_matrix is None:
            self.calculate_market_distances()
        
        # Use cointegration results if available
        if self._cointegration_results:
            cointegration_results = self._cointegration_results
        else:
            # Identify pairs and analyze cointegration
            market_pairs = self.identify_market_pairs(
                max_distance=threshold_distance,
                commodity=commodity
            )
            cointegration_results = self.analyze_market_cointegration(
                market_pairs=market_pairs,
                commodity=commodity
            )
        
        # Calculate spatial weights if threshold_distance provided
        if threshold_distance is not None:
            spatial_weights = np.zeros((len(self.markets), len(self.markets)))
            
            for i, market1 in enumerate(self.markets):
                for j, market2 in enumerate(self.markets):
                    if i == j:
                        continue
                    
                    if self._distance_matrix[i, j] <= threshold_distance:
                        spatial_weights[i, j] = 1
        else:
            # Use inverse distance weights
            spatial_weights = 1 / (self._distance_matrix + 1e-10)  # Add small value to avoid division by zero
            np.fill_diagonal(spatial_weights, 0)  # Zero out diagonal
        
        # Normalize weights by row
        row_sums = spatial_weights.sum(axis=1)
        spatial_weights = spatial_weights / row_sums[:, np.newaxis]
        
        # Initialize metrics
        market_metrics = []
        
        for i, market in enumerate(self.markets):
            # Calculate local integration metrics
            neighbors = [j for j, weight in enumerate(spatial_weights[i]) if weight > 0]
            neighbor_markets = [self.markets[j] for j in neighbors]
            
            # Count cointegrated neighbors
            cointegrated_neighbors = 0
            average_adjustment_speed = 0
            cointegrated_count = 0
            
            for neighbor in neighbor_markets:
                # Check both directions of market pair
                pair1 = (market, neighbor)
                pair2 = (neighbor, market)
                
                if pair1 in cointegration_results:
                    pair_result = cointegration_results[pair1]
                elif pair2 in cointegration_results:
                    pair_result = cointegration_results[pair2]
                else:
                    continue
                
                # Check if cointegrated
                if 'cointegration' in pair_result and pair_result['cointegration'].get('cointegrated', False):
                    cointegrated_neighbors += 1
                    
                    # Extract adjustment speed if available
                    if 'mtar' in pair_result:
                        mtar = pair_result['mtar']
                        # Use average of positive and negative adjustment
                        adj_speed = (abs(mtar.get('adjustment_positive', 0)) + 
                                     abs(mtar.get('adjustment_negative', 0))) / 2
                        average_adjustment_speed += adj_speed
                        cointegrated_count += 1
            
            # Calculate average adjustment speed
            if cointegrated_count > 0:
                average_adjustment_speed /= cointegrated_count
            
            # Calculate integration ratio
            integration_ratio = cointegrated_neighbors / len(neighbor_markets) if neighbor_markets else 0
            
            # Store metrics
            metrics = {
                'market_id': market,
                'cointegrated_neighbors': cointegrated_neighbors,
                'total_neighbors': len(neighbor_markets),
                'integration_ratio': integration_ratio,
                'average_adjustment_speed': average_adjustment_speed
            }
            market_metrics.append(metrics)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(market_metrics)
        
        # Merge with spatial data if available
        if self.has_spatial_data:
            metrics_df = metrics_df.merge(self.spatial_data, on='market_id', how='left')
            metrics_df = gpd.GeoDataFrame(metrics_df, geometry=self.geometry_col)
        
        return metrics_df
u