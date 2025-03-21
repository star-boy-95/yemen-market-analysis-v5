# Error Handling Standards

This document outlines the standard error handling practices for the Yemen Market Analysis project.

## Error Types

Each module has its own error types that should be used for specific error scenarios:

- **DataError**: For errors related to data loading, validation, and processing
- **ModelError**: For errors in model fitting, prediction, and evaluation
- **VisualizationError**: For errors in creating and displaying visualizations
- **SimulationError**: For errors in simulation execution and analysis
- **ValidationError**: For errors in data validation

## Error Handling Decorator

The `@handle_errors` decorator should be used consistently across the codebase:

```python
@handle_errors(logger=logger, error_type=(ValueError, TypeError, ModuleSpecificError), reraise=True/False)
```

### Guidelines for `reraise` Parameter

- **Utility Functions**: Set `reraise=True` to propagate errors to calling functions
  ```python
  @handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)
  def utility_function():
      # Implementation
  ```

- **Analysis Functions**: Set `reraise=False` to log errors but continue execution
  ```python
  @handle_errors(logger=logger, error_type=(ValueError, TypeError, ModelError), reraise=False)
  def analysis_function():
      # Implementation
  ```

### Module-Specific Error Types

Use these error types based on the module:

- **Data Module**: `(ValueError, TypeError, DataError, IOError)`
- **Models Module**: `(ValueError, TypeError, ModelError)`
- **Visualization Module**: `(ValueError, TypeError, VisualizationError)`
- **Simulation Module**: `(ValueError, TypeError, SimulationError)`
- **Utils Module**: `(ValueError, TypeError, OSError)`

## Error Messages

Error messages should be clear, specific, and actionable:

- Include the context of what was being attempted
- Specify what went wrong
- Suggest how to fix the issue if possible

Example:
```python
raise DataError(f"GeoJSON file not found: {file_path}. Please ensure the file exists and you have read permissions.")
```

## Error Logging

Errors should be logged with appropriate severity levels:

- **DEBUG**: Detailed information for debugging
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication that something unexpected happened, but the application can continue
- **ERROR**: Due to a more serious problem, the application couldn't perform a function
- **CRITICAL**: A serious error indicating that the application may be unable to continue running

Example:
```python
try:
    # Operation that might fail
except Exception as e:
    logger.error(f"Failed to process data: {str(e)}")
    raise DataError(f"Data processing failed: {str(e)}") from e
```
