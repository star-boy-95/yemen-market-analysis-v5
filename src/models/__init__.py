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
    bootstrap_confidence_intervals,
    compute_prediction_intervals
)

# Threshold models
from src.models.threshold import (
    ThresholdCointegration,
    calculate_asymmetric_adjustment,
    calculate_half_life,
    test_asymmetric_adjustment
)

from src.models.threshold_vecm import (
    ThresholdVECM, 
    calculate_regime_transition_matrix,
    calculate_half_lives,
    test_threshold_significance,
    combine_tvecm_results
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
    'bootstrap_confidence_intervals',
    'compute_prediction_intervals',
    
    # Threshold models
    'ThresholdCointegration',
    'calculate_asymmetric_adjustment',
    'test_asymmetric_adjustment',
    'ThresholdVECM',
    'calculate_regime_transition_matrix',
    'calculate_half_lives',
    'test_threshold_significance',
    'combine_tvecm_results',
    
    # Spatial models
    'SpatialEconometrics',
    'calculate_market_accessibility',
    'calculate_market_isolation',
    'find_nearest_points',
    'simulate_improved_connectivity',
    'market_integration_index'
]