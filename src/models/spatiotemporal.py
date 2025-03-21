"""
Spatiotemporal integration module for Yemen Market Integration project.

This module provides functions for integrating time series and spatial analysis
results to create a comprehensive market integration analysis framework.
"""

import numpy as np
import pandas as pd
import logging
from yemen_market_integration.utils import handle_errors

# Create logger
logger = logging.getLogger(__name__)


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def integrate_time_series_spatial_results(time_series_results, spatial_results, commodity):
    """
    Integrate time series and spatial analysis results for comprehensive market analysis.
    
    Parameters
    ----------
    time_series_results : dict
        Results from threshold cointegration analysis
    spatial_results : dict
        Results from spatial econometric analysis
    commodity : str
        Commodity name
        
    Returns
    -------
    dict
        Integrated analysis results
    """
    integrated_results = {}
    
    # Extract key metrics from time series analysis
    if time_series_results and 'tvecm' in time_series_results:
        tvecm_result = time_series_results['tvecm']
        integrated_results['threshold'] = tvecm_result.get('threshold')
        integrated_results['adjustment_below'] = tvecm_result.get('adjustment_below_1')
        integrated_results['adjustment_above'] = tvecm_result.get('adjustment_above_1')
        
        # Calculate half-lives if adjustment parameters are available
        adj_below = tvecm_result.get('adjustment_below_1')
        adj_above = tvecm_result.get('adjustment_above_1')
        
        if adj_below is not None:
            if adj_below != 0:
                half_life_below = np.log(0.5) / np.log(1 + abs(adj_below))
                integrated_results['half_life_below'] = half_life_below
            else:
                integrated_results['half_life_below'] = float('inf')
        else:
            integrated_results['half_life_below'] = None
                
        if adj_above is not None:
            if adj_above != 0:
                half_life_above = np.log(0.5) / np.log(1 + abs(adj_above))
                integrated_results['half_life_above'] = half_life_above
            else:
                integrated_results['half_life_above'] = float('inf')
        else:
            integrated_results['half_life_above'] = None
    
    # Extract key metrics from spatial analysis
    if spatial_results and 'global_moran' in spatial_results:
        global_moran = spatial_results['global_moran']
        if global_moran:
            integrated_results['spatial_autocorrelation'] = global_moran.get('I')
            integrated_results['spatial_p_value'] = global_moran.get('p')
        
        if 'lag_model' in spatial_results and spatial_results['lag_model']:
            lag_model = spatial_results['lag_model']
            integrated_results['spatial_dependence'] = getattr(lag_model, 'rho', None)
            integrated_results['spatial_r_squared'] = getattr(lag_model, 'r2', None)
    
    # Calculate integrated metrics
    integrated_results['integration_index'] = calculate_integration_index(
        time_series_results, spatial_results
    )
    
    integrated_results['market_clusters'] = identify_market_clusters(
        time_series_results, spatial_results
    )
    
    integrated_results['regime_boundary_effects'] = analyze_regime_boundaries(
        time_series_results, spatial_results
    )
    
    return integrated_results


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def calculate_integration_index(time_series_results, spatial_results):
    """
    Calculate a composite market integration index combining time series and spatial metrics.
    
    Parameters
    ----------
    time_series_results : dict
        Results from threshold cointegration analysis
    spatial_results : dict
        Results from spatial econometric analysis
        
    Returns
    -------
    float
        Composite market integration index (0-1 scale)
    """
    # Initialize weights and components
    weights = {
        'cointegration': 0.3,
        'threshold': 0.2,
        'spatial_autocorrelation': 0.2,
        'spatial_dependence': 0.3
    }
    
    components = {}
    
    # Calculate cointegration component (0-1 scale)
    if time_series_results and 'cointegration' in time_series_results:
        coint_result = time_series_results['cointegration']
        if coint_result.get('cointegrated', False):
            # Stronger cointegration = higher value
            p_value = coint_result.get('p_value', 0.5)
            components['cointegration'] = 1 - min(p_value * 20, 0.99)  # Transform p-value to 0-1 scale
        else:
            components['cointegration'] = 0
    else:
        # No cointegration results available
        components['cointegration'] = 0
        weights = {k: v / (1 - weights['cointegration']) for k, v in weights.items() if k != 'cointegration'}
        weights['cointegration'] = 0
    
    # Calculate threshold component (0-1 scale)
    if time_series_results and 'tvecm' in time_series_results:
        tvecm_result = time_series_results['tvecm']
        if 'threshold' in tvecm_result:
            # Lower threshold = higher integration
            threshold = tvecm_result['threshold']
            # Normalize threshold (assuming typical range 0-0.5)
            components['threshold'] = max(0, 1 - threshold * 2)
        else:
            components['threshold'] = 0
    else:
        # No threshold results available
        components['threshold'] = 0
        weights = {k: v / (1 - weights['threshold']) for k, v in weights.items() if k != 'threshold'}
        weights['threshold'] = 0
    
    # Calculate spatial autocorrelation component (0-1 scale)
    if spatial_results and 'global_moran' in spatial_results:
        global_moran = spatial_results['global_moran']
        if global_moran:
            # Higher positive autocorrelation = higher integration
            moran_i = global_moran.get('I', 0)
            # Transform Moran's I (-1 to 1) to 0-1 scale
            components['spatial_autocorrelation'] = (moran_i + 1) / 2
        else:
            components['spatial_autocorrelation'] = 0.5  # Neutral value
    else:
        # No spatial autocorrelation results available
        components['spatial_autocorrelation'] = 0.5  # Neutral value
        weights = {k: v / (1 - weights['spatial_autocorrelation']) for k, v in weights.items() if k != 'spatial_autocorrelation'}
        weights['spatial_autocorrelation'] = 0
    
    # Calculate spatial dependence component (0-1 scale)
    if spatial_results and 'lag_model' in spatial_results and spatial_results['lag_model']:
        lag_model = spatial_results['lag_model']
        # Higher spatial dependence = higher integration
        rho = getattr(lag_model, 'rho', 0)
        # Transform rho (typically 0-1) to 0-1 scale
        components['spatial_dependence'] = min(max(rho, 0), 1)
    else:
        # No spatial dependence results available
        components['spatial_dependence'] = 0
        weights = {k: v / (1 - weights['spatial_dependence']) for k, v in weights.items() if k != 'spatial_dependence'}
        weights['spatial_dependence'] = 0
    
    # Calculate weighted average
    integration_index = sum(weights[k] * components[k] for k in components)
    
    return integration_index


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def identify_market_clusters(time_series_results, spatial_results):
    """
    Identify market clusters based on integration patterns.
    
    Parameters
    ----------
    time_series_results : dict
        Results from threshold cointegration analysis
    spatial_results : dict
        Results from spatial econometric analysis
        
    Returns
    -------
    dict
        Market clusters with integration characteristics
    """
    clusters = {}
    
    # Extract local indicators of spatial association if available
    if spatial_results and 'model' in spatial_results and hasattr(spatial_results['model'], 'gdf'):
        gdf = spatial_results['model'].gdf
        
        # Check for cluster_type column (from local Moran's I analysis)
        if 'cluster_type' in gdf.columns:
            # Count markets in each cluster type
            cluster_counts = gdf['cluster_type'].value_counts().to_dict()
            
            # Create cluster summary
            clusters['spatial'] = {
                'counts': cluster_counts,
                'description': 'Markets grouped by price similarity and spatial proximity'
            }
            
            # Calculate average price by cluster
            if 'price' in gdf.columns:
                clusters['spatial']['avg_price'] = {
                    cluster_type: gdf[gdf['cluster_type'] == cluster_type]['price'].mean()
                    for cluster_type in cluster_counts.keys()
                }
        # Fall back to lisa_cluster if it exists (for backward compatibility)
        elif 'lisa_cluster' in gdf.columns:
            # Count markets in each cluster type
            cluster_counts = gdf['lisa_cluster'].value_counts().to_dict()
            
            # Map cluster codes to descriptions
            cluster_descriptions = {
                0: 'Not Significant',
                1: 'High-High (Hotspot)',
                2: 'Low-Low (Coldspot)',
                3: 'High-Low (Outlier)',
                4: 'Low-High (Outlier)'
            }
            
            # Create cluster summary
            clusters['spatial'] = {
                'counts': {cluster_descriptions.get(k, f"Unknown ({k})"): v for k, v in cluster_counts.items()},
                'description': 'Markets grouped by price similarity and spatial proximity'
            }
            
            # Calculate average price by cluster
            if 'price' in gdf.columns:
                clusters['spatial']['avg_price'] = {
                    cluster_descriptions.get(k, f"Unknown ({k})"): gdf[gdf['lisa_cluster'] == k]['price'].mean()
                    for k in cluster_counts.keys()
                }
    
    # Add exchange rate regime clustering
    if spatial_results and 'model' in spatial_results and hasattr(spatial_results['model'], 'gdf'):
        gdf = spatial_results['model'].gdf
        
        if 'exchange_rate_regime' in gdf.columns:
            # Count markets in each regime
            regime_counts = gdf['exchange_rate_regime'].value_counts().to_dict()
            
            # Create regime summary
            clusters['regime'] = {
                'counts': regime_counts,
                'description': 'Markets grouped by exchange rate regime'
            }
            
            # Calculate average price by regime
            if 'price' in gdf.columns:
                clusters['regime']['avg_price'] = {
                    regime: gdf[gdf['exchange_rate_regime'] == regime]['price'].mean()
                    for regime in regime_counts.keys()
                }
    
    # Add threshold-based clustering if available
    if time_series_results and 'tvecm' in time_series_results:
        tvecm_result = time_series_results['tvecm']
        
        if 'threshold' in tvecm_result:
            threshold = tvecm_result['threshold']
            
            clusters['threshold'] = {
                'threshold_value': threshold,
                'description': 'Markets grouped by price differential relative to threshold'
            }
    
    return clusters


@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
def analyze_regime_boundaries(time_series_results, spatial_results):
    """
    Analyze price transmission across exchange rate regime boundaries.
    
    Parameters
    ----------
    time_series_results : dict
        Results from threshold cointegration analysis
    spatial_results : dict
        Results from spatial econometric analysis
        
    Returns
    -------
    dict
        Analysis of regime boundary effects
    """
    boundary_effects = {}
    
    # Check if we have spatial model with regime information
    if spatial_results and 'model' in spatial_results and hasattr(spatial_results['model'], 'gdf'):
        gdf = spatial_results['model'].gdf
        
        if 'exchange_rate_regime' in gdf.columns:
            # Calculate price differential across regimes
            north_price = gdf[gdf['exchange_rate_regime'] == 'north']['price'].mean()
            south_price = gdf[gdf['exchange_rate_regime'] == 'south']['price'].mean()
            
            price_diff = abs(north_price - south_price)
            price_ratio = max(north_price, south_price) / min(north_price, south_price)
            
            boundary_effects['price_differential'] = price_diff
            boundary_effects['price_ratio'] = price_ratio
            
            # Calculate exchange rate adjusted price differential
            if 'usdprice' in gdf.columns:
                north_usd = gdf[gdf['exchange_rate_regime'] == 'north']['usdprice'].mean()
                south_usd = gdf[gdf['exchange_rate_regime'] == 'south']['usdprice'].mean()
                
                usd_diff = abs(north_usd - south_usd)
                usd_ratio = max(north_usd, south_usd) / min(north_usd, south_usd)
                
                boundary_effects['usd_price_differential'] = usd_diff
                boundary_effects['usd_price_ratio'] = usd_ratio
                
                # Calculate boundary effect (how much of price differential is due to exchange rate)
                if price_diff > 0:
                    exchange_rate_effect = (price_diff - usd_diff) / price_diff
                    boundary_effects['exchange_rate_effect'] = exchange_rate_effect
    
    # Add threshold information if available
    if time_series_results and 'tvecm' in time_series_results:
        tvecm_result = time_series_results['tvecm']
        
        if 'threshold' in tvecm_result:
            threshold = tvecm_result['threshold']
            
            # Compare threshold to price differential
            if 'price_differential' in boundary_effects:
                price_diff = boundary_effects['price_differential']
                
                # Check if price differential exceeds threshold
                boundary_effects['exceeds_threshold'] = price_diff > threshold
                boundary_effects['diff_to_threshold_ratio'] = price_diff / threshold if threshold > 0 else float('inf')
    
    return boundary_effects