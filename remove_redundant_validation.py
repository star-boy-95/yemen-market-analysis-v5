#!/usr/bin/env python3
"""
Script to remove redundant validation code in the Yemen Market Analysis codebase.

This script identifies and removes duplicate validation checks in methods,
particularly in the cointegration module.
"""
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set

def remove_redundant_validation(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Remove redundant validation code in a file.
    
    Parameters
    ----------
    file_path : Path
        Path to the file to fix
        
    Returns
    -------
    Tuple[bool, List[str]]
        Whether any changes were made and a list of changes
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Pattern for duplicate length validation in test_engle_granger
    if 'test_engle_granger' in content:
        # Find the method definition
        method_match = re.search(r'def\s+test_engle_granger\s*\([^)]*\):', content)
        if method_match:
            method_start = method_match.start()
            
            # Find the method end (next def or end of file)
            next_def = re.search(r'def\s+', content[method_start+1:])
            method_end = next_def.start() + method_start + 1 if next_def else len(content)
            
            method_content = content[method_start:method_end]
            
            # Check for duplicate length validation
            if method_content.count('len(y) != len(x)') > 1:
                # Remove the second check
                pattern = r'# Check lengths\s*if len\(y\) != len\(x\):[^}]*?raise ValueError\([^)]*\)'
                new_method_content, count = re.subn(pattern, '# Length already validated above', method_content, flags=re.DOTALL)
                
                if count > 0:
                    content = content[:method_start] + new_method_content + content[method_end:]
                    changes.append("Removed redundant length validation in test_engle_granger")
    
    # Pattern for redundant validation in other methods
    # Look for methods with multiple validate_* calls
    method_pattern = r'def\s+(\w+)\s*\([^)]*\):[^}]*?validate_([^(]+)\([^)]*\)[^}]*?validate_\2\([^)]*\)'
    for match in re.finditer(method_pattern, content, re.DOTALL):
        method_name = match.group(1)
        validation_type = match.group(2)
        
        method_start = match.start()
        
        # Find the method end (next def or end of file)
        next_def = re.search(r'def\s+', content[method_start+1:])
        method_end = next_def.start() + method_start + 1 if next_def else len(content)
        
        method_content = content[method_start:method_end]
        
        # Find all validate_* calls
        validation_calls = re.findall(r'validate_' + validation_type + r'\([^)]*\)', method_content)
        
        if len(validation_calls) > 1:
            # Keep the first validation call, remove others
            for call in validation_calls[1:]:
                # Replace with a comment
                new_method_content = method_content.replace(call, f'# Validation already performed above: {call}')
                
                if new_method_content != method_content:
                    content = content[:method_start] + new_method_content + content[method_end:]
                    method_content = new_method_content
                    changes.append(f"Removed redundant {validation_type} validation in {method_name}")
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    
    return False, []

def remove_redundant_validation_in_directory(directory: Path, file_pattern: str = "*.py") -> Dict[str, List[str]]:
    """
    Remove redundant validation code in all Python files in a directory.
    
    Parameters
    ----------
    directory : Path
        Directory to process
    file_pattern : str, optional
        Glob pattern for files to process, default "*.py"
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping file paths to lists of changes made
    """
    results = {}
    
    for file_path in directory.glob(f"**/{file_pattern}"):
        if file_path.is_file():
            changed, changes = remove_redundant_validation(file_path)
            if changed:
                results[str(file_path)] = changes
    
    return results

def create_validation_guide(output_path: Path):
    """
    Create a guide for validation standards.
    
    Parameters
    ----------
    output_path : Path
        Path to write the guide
    """
    guide_content = """# Validation Standards

This document outlines the standard validation practices for the Yemen Market Integration project.

## Validation Principles

1. **Validate Early**: Perform validation as early as possible in the function
2. **Validate Once**: Avoid redundant validation of the same condition
3. **Use Utility Functions**: Use the provided validation utilities for consistency
4. **Be Specific**: Provide clear error messages that explain what went wrong

## Common Validation Utilities

- **validate_dataframe**: Validates DataFrame structure and content
- **validate_geodataframe**: Validates GeoDataFrame structure and content
- **validate_time_series**: Validates time series data for analysis
- **raise_if_invalid**: Raises an exception if validation fails

## Example of Good Validation

```python
def process_data(df, column):
    # Validate inputs once at the beginning
    valid, errors = validate_dataframe(
        df,
        required_columns=[column],
        check_nulls=True
    )
    raise_if_invalid(valid, errors, f"Invalid data for processing: {column}")
    
    # Process data knowing it's valid
    result = df[column].mean()
    return result
```

## Example of Poor Validation (Redundant)

```python
def process_data(df, column):
    # First validation
    valid, errors = validate_dataframe(
        df,
        required_columns=[column],
        check_nulls=False
    )
    raise_if_invalid(valid, errors, "Invalid DataFrame")
    
    # Redundant validation - already checked above
    if column not in df.columns:
        raise ValueError(f"Column {column} not found")
    
    # Process data
    result = df[column].mean()
    return result
```

## Special Cases

- When validating multiple inputs, validate each one separately for clear error messages
- For complex validations, consider creating a custom validation function
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"Created validation standards guide at {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Remove redundant validation code in Python files")
    parser.add_argument("--directory", "-d", type=str, default="src",
                        help="Directory to process (default: src)")
    parser.add_argument("--pattern", "-p", type=str, default="*.py",
                        help="File pattern to match (default: *.py)")
    parser.add_argument("--create-guide", "-g", action="store_true",
                        help="Create validation standards guide")
    parser.add_argument("--guide-path", type=str, default="validation_standards.md",
                        help="Path for validation guide (default: validation_standards.md)")
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    print(f"Processing files in {directory}...")
    results = remove_redundant_validation_in_directory(directory, args.pattern)
    
    print(f"\nRemoved redundant validation in {len(results)} files:")
    for file_path, changes in results.items():
        print(f"\n{file_path}:")
        for change in changes:
            print(f"  - {change}")
    
    print(f"\nTotal files modified: {len(results)}")
    
    if args.create_guide:
        create_validation_guide(Path(args.guide_path))

if __name__ == "__main__":
    main()