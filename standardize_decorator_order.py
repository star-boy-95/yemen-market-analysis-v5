#!/usr/bin/env python3
"""
Script to standardize the order of decorators in the Yemen Market Analysis codebase.

This script ensures that decorators are applied in the optimal order for performance and error handling.
The recommended order is:
1. @disk_cache (outermost)
2. @memory_usage_decorator
3. @handle_errors
4. @m1_optimized
5. @timer (innermost)
"""
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Define the decorator patterns to match
DECORATOR_PATTERNS = [
    r'@disk_cache\([^)]*\)',
    r'@memory_usage_decorator',
    r'@handle_errors\([^)]*\)',
    r'@m1_optimized\([^)]*\)',
    r'@timer'
]

def extract_decorators(content: str, start_pos: int) -> Tuple[List[str], int, int]:
    """
    Extract decorators from a method definition.
    
    Parameters
    ----------
    content : str
        File content
    start_pos : int
        Starting position to search from
        
    Returns
    -------
    Tuple[List[str], int, int]
        List of decorators, start position, and end position
    """
    # Find the method definition
    method_match = re.search(r'(\s*)def\s+\w+\s*\(', content[start_pos:])
    if not method_match:
        return [], -1, -1
    
    method_start = start_pos + method_match.start()
    indent = method_match.group(1)
    
    # Look backwards for decorators
    decorators = []
    decorator_start = method_start
    
    # Search backwards for decorators with the same indentation
    pos = method_start - 1
    while pos >= 0:
        # Find the previous line
        line_start = content.rfind('\n', 0, pos) + 1
        line = content[line_start:pos+1].strip()
        
        # Check if it's a decorator with the same indentation
        if line.startswith('@') and content[line_start:line_start+len(indent)] == indent:
            decorators.insert(0, line)
            decorator_start = line_start
            pos = line_start - 1
        else:
            break
    
    return decorators, decorator_start, method_start

def standardize_decorator_order(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Standardize decorator order in a file.
    
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
    
    # Find all method definitions with multiple decorators
    pos = 0
    while True:
        decorators, start_pos, end_pos = extract_decorators(content, pos)
        if start_pos == -1 or len(decorators) <= 1:
            if start_pos == -1:
                break
            pos = end_pos
            continue
        
        # Check if decorators need reordering
        ordered_decorators = []
        for pattern in DECORATOR_PATTERNS:
            for decorator in decorators:
                if re.match(pattern, decorator):
                    ordered_decorators.append(decorator)
        
        # Add any decorators that didn't match our patterns
        for decorator in decorators:
            if decorator not in ordered_decorators:
                ordered_decorators.append(decorator)
        
        # If the order changed, update the content
        if ordered_decorators != decorators:
            # Get the indentation from the first decorator
            indent_match = re.match(r'^(\s*)', content[start_pos:start_pos+10])
            indent = indent_match.group(1) if indent_match else ''
            
            # Create the new decorator block
            new_decorators = '\n'.join(indent + decorator for decorator in ordered_decorators)
            
            # Replace the old decorators with the new ones
            content = content[:start_pos] + new_decorators + content[end_pos:]
            
            changes.append(f"Reordered decorators for method at position {start_pos}")
            
            # Update position to continue search
            pos = start_pos + len(new_decorators)
        else:
            pos = end_pos
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    
    return False, []

def standardize_decorator_order_in_directory(directory: Path, file_pattern: str = "*.py") -> Dict[str, List[str]]:
    """
    Standardize decorator order in all Python files in a directory.
    
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
            changed, changes = standardize_decorator_order(file_path)
            if changed:
                results[str(file_path)] = changes
    
    return results

def create_decorator_guide(output_path: Path):
    """
    Create a guide for decorator order standards.
    
    Parameters
    ----------
    output_path : Path
        Path to write the guide
    """
    guide_content = """# Decorator Order Standards

This document outlines the standard decorator order for the Yemen Market Integration project.

## Recommended Decorator Order

Decorators should be applied in the following order (from outermost to innermost):

1. **@disk_cache** - Outermost decorator to avoid redundant computation
2. **@memory_usage_decorator** - Track memory usage of the function
3. **@handle_errors** - Handle errors and exceptions
4. **@m1_optimized** - Apply M1-specific optimizations
5. **@timer** - Innermost decorator to accurately measure execution time

## Example

```python
@disk_cache(cache_dir='.cache/yemen_market_integration/cointegration')
@memory_usage_decorator
@handle_errors(logger=logger, error_type=(ValueError, TypeError), reraise=True)
@m1_optimized(parallel=True)
@timer
def test_johansen(self, data, det_order=1, k_ar_diff=4):
    # Implementation
```

## Rationale

This order ensures:

1. **Caching happens first** - Prevents unnecessary computation if results are cached
2. **Memory tracking is accurate** - Captures memory usage of the actual function, not just cache lookups
3. **Error handling is comprehensive** - Catches errors from the function and any inner decorators
4. **Optimization is applied correctly** - M1 optimization is applied to the core function
5. **Timing is accurate** - Measures the actual execution time of the optimized function

## Special Cases

- If a function doesn't need all decorators, maintain the relative order of the ones used
- Custom decorators should be placed based on their functionality relative to the standard ones
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"Created decorator order guide at {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standardize decorator order in Python files")
    parser.add_argument("--directory", "-d", type=str, default="src",
                        help="Directory to process (default: src)")
    parser.add_argument("--pattern", "-p", type=str, default="*.py",
                        help="File pattern to match (default: *.py)")
    parser.add_argument("--create-guide", "-g", action="store_true",
                        help="Create decorator order guide")
    parser.add_argument("--guide-path", type=str, default="decorator_order_guide.md",
                        help="Path for decorator order guide (default: decorator_order_guide.md)")
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    print(f"Processing files in {directory}...")
    results = standardize_decorator_order_in_directory(directory, args.pattern)
    
    print(f"\nStandardized decorator order in {len(results)} files:")
    for file_path, changes in results.items():
        print(f"\n{file_path}:")
        for change in changes:
            print(f"  - {change}")
    
    print(f"\nTotal files modified: {len(results)}")
    
    if args.create_guide:
        create_decorator_guide(Path(args.guide_path))

if __name__ == "__main__":
    main()