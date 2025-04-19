"""
Utility modules for Yemen Market Analysis.

This package provides utility modules for the Yemen Market Analysis package.
It includes modules for error handling, validation, performance optimization,
and statistical analysis.
"""

from src.utils.error_handling import YemenAnalysisError, handle_errors, log_execution
from src.utils.validation import validate_data, validate_model_parameters
from src.utils.performance import MemoryManager, ParallelProcessor
from src.utils.statistics import (
    descriptive_statistics, correlation_analysis, detect_outliers,
    normality_test, heteroskedasticity_test, autocorrelation_test,
    granger_causality, stationarity_test, bootstrap_statistic
)

__all__ = [
    # Error handling
    'YemenAnalysisError',
    'handle_errors',
    'log_execution',

    # Validation
    'validate_data',
    'validate_model_parameters',

    # Performance
    'MemoryManager',
    'ParallelProcessor',

    # Statistics
    'descriptive_statistics',
    'correlation_analysis',
    'detect_outliers',
    'normality_test',
    'heteroskedasticity_test',
    'autocorrelation_test',
    'granger_causality',
    'stationarity_test',
    'bootstrap_statistic',
]