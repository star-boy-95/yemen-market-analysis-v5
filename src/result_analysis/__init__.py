"""
Result Analysis module for Yemen Market Integration Project.

This module provides components for enhancing statistical analysis and academic
reporting capabilities, including:
- Statistical testing and significance indicators
- Confidence interval calculations
- Comprehensive diagnostic tests
- Academic output formatting
- Cross-validation utilities
"""

# Import key components for easy access
from .statistical_tests import (
    calculate_significance_indicators,
    hypothesis_test,
    wald_test,
    joint_significance_test,
    threshold_significance_test,
    get_significance_level
)

from .diagnostics import (
    heteroskedasticity_test,
    serial_correlation_test,
    normality_test,
    model_specification_test,
    run_comprehensive_diagnostics
)

from .academic_formatting import AcademicTableFormatter

from .confidence_intervals import (
    calculate_confidence_interval,
    bootstrap_confidence_interval
)

from .cross_validation import (
    cross_validate_threshold,
    calculate_prediction_metrics
)

from .significance import (
    format_with_significance,
    add_significance_indicators,
    generate_significance_note
)

# Version
__version__ = '0.1.0'