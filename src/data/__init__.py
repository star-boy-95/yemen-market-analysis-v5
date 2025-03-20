"""
Data processing module for Yemen market integration analysis.

This module provides tools for loading, preprocessing, and integrating
market data from Yemen with spatial and conflict information.

Examples
--------
Basic data loading and preprocessing:

    >>> from src.data import load_market_data, preprocess_data
    >>> 
    >>> # Load raw market data
    >>> raw_data = load_market_data('unified_data.geojson')
    >>> 
    >>> # Preprocess the data
    >>> processed_data = preprocess_data(raw_data)
    >>> 
    >>> # Get price differentials
    >>> from src.data import calculate_price_differentials
    >>> differentials = calculate_price_differentials(processed_data)

Working with exchange rate regimes:

    >>> from src.data import split_by_exchange_regime
    >>> 
    >>> # Split data by north/south exchange rate regimes
    >>> north_data, south_data = split_by_exchange_regime(processed_data)
    >>> 
    >>> # Get time series for a specific region and commodity
    >>> from src.data import get_time_series
    >>> abyan_beans = get_time_series(processed_data, 'abyan', 'beans (kidney red)')

Integration with other data sources:

    >>> from src.data import integrate_conflict_data
    >>> 
    >>> # Integrate with conflict data
    >>> integrated_data = integrate_conflict_data(processed_data, 'conflict_data.geojson')
"""

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.integration import DataIntegrator


# Create singleton instances for easy access
_loader = DataLoader()
_preprocessor = DataPreprocessor()
_integrator = DataIntegrator()


# Convenience functions
def load_market_data(filename):
    """Load market data from GeoJSON file."""
    return _loader.load_geojson(filename)


def save_processed_data(data, filename):
    """Save processed data to file."""
    return _loader.save_processed_data(data, filename)


def preprocess_data(data):
    """Preprocess raw market data."""
    return _preprocessor.preprocess_geojson(data)


def calculate_price_differentials(data):
    """Calculate price differentials between exchange rate regimes."""
    return _preprocessor.calculate_price_differentials(data)


def split_by_exchange_regime(data):
    """Split data by exchange rate regime."""
    return _loader.split_by_exchange_regime(data)


def get_time_series(data, region, commodity):
    """Get time series for a specific region and commodity."""
    return _loader.get_time_series(data, region, commodity)


def get_commodity_list(data):
    """Get list of unique commodities in the data."""
    return _loader.get_commodity_list(data)


def get_region_list(data):
    """Get list of unique regions in the data."""
    return _loader.get_region_list(data)


def integrate_conflict_data(data, conflict_file):
    """Integrate market data with conflict data."""
    return _integrator.integrate_conflict_data(data, conflict_file)


def integrate_exchange_rates(data, exchange_file):
    """Integrate market data with exchange rate data."""
    return _integrator.integrate_exchange_rates(data, exchange_file)


def get_spatial_boundaries(boundary_file):
    """Load administrative boundaries for spatial analysis."""
    return _integrator.get_spatial_boundaries(boundary_file)


# Make classes and functions available directly
__all__ = [
    # Convenience functions
    'load_market_data',
    'save_processed_data',
    'preprocess_data',
    'calculate_price_differentials',
    'split_by_exchange_regime',
    'get_time_series',
    'get_commodity_list',
    'get_region_list',
    'integrate_conflict_data',
    'integrate_exchange_rates',
    'get_spatial_boundaries',
    
    # Classes
    'DataLoader',
    'DataPreprocessor',
    'DataIntegrator'
]