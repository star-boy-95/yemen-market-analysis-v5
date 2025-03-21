#!/usr/bin/env python3
"""
Script to standardize error handling in the Yemen Market Analysis codebase.

This script updates @handle_errors decorators to use consistent error types
and reraise behavior across the codebase.
"""
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

# Define error handling patterns to standardize
ERROR_HANDLING_PATTERNS = [
    # Utility functions - log and reraise
    (r'@handle_errors\(logger=logger\)', 
     r'@handle_errors(logger=logger, error_type=(ValueError, TypeError, OSError), reraise=True)'),
    
    # Utility functions with specific error types but no reraise
    (r'@handle_errors\(logger=logger, error_type=\(([^)]+)\)\)', 
     r'@handle_errors(logger=logger, error_type=(\1), reraise=True)'),
    
    # Analysis functions - log detailed errors but don't reraise
    (r'@handle_errors\(logger=logger, error_type=\((ValueError, TypeError)\)\)', 
     r'@handle_errors(logger=logger, error_type=(ValueError, TypeError, ModelError), reraise=False)'),
    
    # Generic exception handlers - make more specific
    (r'@handle_errors\(logger=logger, error_type=\(Exception,\), reraise=False\)', 
     r'@handle_errors(logger=logger, error_type=(ValueError, TypeError, IOError, ModelError), reraise=False)'),
]

# Define module-specific error types
MODULE_ERROR_TYPES = {
    'data': '(ValueError, TypeError, DataError, IOError)',
    'models': '(ValueError, TypeError, ModelError)',
    'visualization': '(ValueError, TypeError, VisualizationError)',
    'simulation': '(ValueError, TypeError, SimulationError)',
    'utils': '(ValueError, TypeError, OSError)',
}

def get_module_from_path(file_path: Path) -> Optional[str]:
    """
    Extract module name from file path.
    
    Parameters
    ----------
    file_path : Path
        Path to the file
        
    Returns
    -------
    Optional[str]
        Module name or None if not found
    """
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part == 'src' and i + 1 < len(parts):
            return parts[i + 1]
    return None

def standardize_error_handling_in_file(file_path: Path) -> Tuple[int, List[str]]:
    """
    Standardize error handling in a single file.
    
    Parameters
    ----------
    file_path : Path
        Path to the file to fix
        
    Returns
    -------
    Tuple[int, List[str]]
        Number of replacements made and list of patterns replaced
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    replacements_made = []
    
    # Get module-specific error types
    module = get_module_from_path(file_path)
    module_error_types = MODULE_ERROR_TYPES.get(module, '(ValueError, TypeError)')
    
    # Apply standard patterns
    for pattern, replacement in ERROR_HANDLING_PATTERNS:
        # Replace module-specific error types
        if '{module_error_types}' in replacement:
            replacement = replacement.replace('{module_error_types}', module_error_types)
        
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            content = new_content
            replacements_made.append(f"{pattern} -> {replacement} ({count} replacements)")
    
    # Apply module-specific patterns
    if module:
        # Replace generic error types with module-specific ones
        pattern = r'@handle_errors\(logger=logger, error_type=\((ValueError, TypeError)\)\)'
        replacement = f'@handle_errors(logger=logger, error_type={module_error_types})'
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            content = new_content
            replacements_made.append(f"{pattern} -> {replacement} ({count} replacements)")
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return len(replacements_made), replacements_made
    
    return 0, []

def standardize_error_handling_in_directory(directory: Path, file_pattern: str = "*.py") -> Dict[str, List[str]]:
    """
    Standardize error handling in all Python files in a directory.
    
    Parameters
    ----------
    directory : Path
        Directory to process
    file_pattern : str, optional
        Glob pattern for files to process, default "*.py"
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping file paths to lists of replacements made
    """
    results = {}
    
    for file_path in directory.glob(f"**/{file_pattern}"):
        if file_path.is_file():
            count, replacements = standardize_error_handling_in_file(file_path)
            if count > 0:
                results[str(file_path)] = replacements
    
    return results

def create_error_handling_guide(output_path: Path):
    """
    Create a guide for error handling standards.
    
    Parameters
    ----------
    output_path : Path
        Path to write the guide
    """
    guide_content = """# Error Handling Standards

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
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"Created error handling guide at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Standardize error handling in Python files")
    parser.add_argument("--directory", "-d", type=str, default="src",
                        help="Directory to process (default: src)")
    parser.add_argument("--pattern", "-p", type=str, default="*.py",
                        help="File pattern to match (default: *.py)")
    parser.add_argument("--create-guide", "-g", action="store_true",
                        help="Create error handling guide")
    parser.add_argument("--guide-path", type=str, default="error_handling_guide.md",
                        help="Path for error handling guide (default: error_handling_guide.md)")
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    print(f"Processing files in {directory}...")
    results = standardize_error_handling_in_directory(directory, args.pattern)
    
    print(f"\nStandardized error handling in {len(results)} files:")
    for file_path, replacements in results.items():
        print(f"\n{file_path}:")
        for replacement in replacements:
            print(f"  - {replacement}")
    
    print(f"\nTotal files modified: {len(results)}")
    
    if args.create_guide:
        create_error_handling_guide(Path(args.guide_path))

if __name__ == "__main__":
    main()