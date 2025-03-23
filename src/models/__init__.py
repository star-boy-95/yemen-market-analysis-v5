"""
Economic modeling and statistical testing modules for market integration analysis.
"""

# Unit root testing
from src.models.unit_root import UnitRootTester, StructuralBreakTester

# Cointegration testing
from src.models.cointegration import (
    CointegrationTester,
    estimate_cointegration_vector,
    calculate_half_life
)

# Model diagnostics
from src.models.diagnostics import (
    ModelDiagnostics,
    calculate_fit_statistics,
    bootstrap_confidence_interval
)

# Threshold models - import from new unified implementation
from src.models.threshold_model import ThresholdModel

# Maintain backward compatibility
from src.models.threshold import ThresholdCointegration
from src.models.threshold_fixed import ThresholdFixed
from src.models.threshold_vecm import ThresholdVECM, combine_tvecm_results

# Helper functions from threshold_model
from src.models.threshold_model import (
    process_threshold,
    process_chunk
)

# Threshold reporting modules
from src.models.threshold_reporter import (
    ThresholdReporter,
    AcademicThresholdReporter
)

# Spatial models
from src.models.spatial import (
    SpatialEconometrics,
    calculate_market_accessibility,
    calculate_market_isolation,
    find_nearest_points,
    simulate_improved_connectivity,
    market_integration_index
)

# Simulation models
from src.models.simulation import (
    MarketIntegrationSimulation
)

# Define complete public API
__all__ = [
    # Unit root testing
    'UnitRootTester',
    'StructuralBreakTester',
    
    # Cointegration testing
    'CointegrationTester',
    'estimate_cointegration_vector',
    'calculate_half_life',
    
    # Model diagnostics
    'ModelDiagnostics',
    'calculate_fit_statistics',
    'bootstrap_confidence_interval',
    
    # Threshold models
    'ThresholdModel',  # New unified model
    'ThresholdCointegration',  # Backward compatibility
    'ThresholdFixed',  # Backward compatibility
    'ThresholdVECM',  # Backward compatibility
    'combine_tvecm_results',
    'process_threshold',
    'process_chunk',
    
    # Threshold reporting
    'ThresholdReporter',
    'AcademicThresholdReporter',
    
    # Spatial models
    'SpatialEconometrics',
    'calculate_market_accessibility',
    'calculate_market_isolation',
    'find_nearest_points',
    'simulate_improved_connectivity',
    'market_integration_index',
    
    # Simulation models
    'MarketIntegrationSimulation'
]