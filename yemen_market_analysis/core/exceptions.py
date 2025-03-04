"""
Custom exception classes for Yemen Market Analysis.
"""


class YemenMarketError(Exception):
    """Base exception for all Yemen Market Analysis errors."""
    pass


class ConfigurationError(YemenMarketError):
    """Error in configuration settings."""
    pass


class ValidationError(YemenMarketError):
    """Error in data validation."""
    pass


class DataProcessingError(YemenMarketError):
    """Error during data processing."""
    pass


class ModelError(YemenMarketError):
    """Base class for model-related errors."""
    pass


class ThresholdModelError(ModelError):
    """Error in threshold model estimation."""
    pass


class StatisticalTestError(ModelError):
    """Error in statistical testing."""
    pass


class ComputationError(YemenMarketError):
    """Error in computation operations."""
    pass


class DeviceError(ComputationError):
    """Error related to computation device (CPU/GPU)."""
    pass


class VisualizationError(YemenMarketError):
    """Error in data visualization."""
    pass


class ReportingError(YemenMarketError):
    """Error in report generation."""
    pass


class AnalysisError(YemenMarketError):
    """Error in market analysis."""
    pass


class InputError(YemenMarketError):
    """Error in user input or command line arguments."""
    pass