"""Yemen Market Analysis package.

This package provides tools for analyzing market integration in conflict-affected Yemen.
It includes modules for data loading and preprocessing, econometric analysis, visualization,
and reporting.
"""

__version__ = '0.1.0'

# Import main modules
from src.config import config
from src.data import loader, preprocessor, integration
from src.models import unit_root, cointegration, threshold, spatial, panel
from src.visualization import time_series, maps, econometric_tables
from src.utils import error_handling, validation, performance, statistics