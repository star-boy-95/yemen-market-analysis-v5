"""
Computation module for Yemen Market Analysis.
"""
from .device_manager import DeviceManager, get_device_manager
from .parallel import parallel_process, ParallelProcessor
from .numerical import (
    stabilize_matrix, robust_inversion, robust_ols,
    robust_correlation, scale_matrix, NumericalStability
)
from .performance import (
    PerformanceTracker, performance_tracker, track_performance,
    optimize_computation_settings, chunk_large_computation
)

__all__ = [
    'DeviceManager', 'get_device_manager',
    'parallel_process', 'ParallelProcessor',
    'stabilize_matrix', 'robust_inversion', 'robust_ols',
    'robust_correlation', 'scale_matrix', 'NumericalStability',
    'PerformanceTracker', 'performance_tracker', 'track_performance',
    'optimize_computation_settings', 'chunk_large_computation'
]