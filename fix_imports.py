#!/usr/bin/env python3
"""
Script to fix import inconsistencies in the Yemen Market Analysis codebase.

This script replaces relative imports with absolute imports across all Python files
in the specified directories.
"""
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Define the import patterns to replace
IMPORT_PATTERNS = [
    # From relative to absolute imports
    (r'from utils import ([\s\S]*?)\)', r'from yemen_market_integration.utils import \1)'),
    (r'from utils import ([\w, _]+)', r'from yemen_market_integration.utils import \1'),
    (r'from src\.utils import ([\s\S]*?)\)', r'from yemen_market_integration.utils import \1)'),
    (r'from src\.utils import ([\w, _]+)', r'from yemen_market_integration.utils import \1'),
    
    # Break up grouped imports into specific module imports
    (r'from yemen_market_integration\.utils import (\s*# Error handling\s*)([\w, _]+)', 
     r'from yemen_market_integration.utils.error_handler import \2'),
    
    (r'from yemen_market_integration\.utils import (\s*# Validation\s*)([\w, _]+)', 
     r'from yemen_market_integration.utils.validation import \2'),
    
    (r'from yemen_market_integration\.utils import (\s*# Performance\s*)([\w, _]+)', 
     r'from yemen_market_integration.utils.performance_utils import \2'),
    
    (r'from yemen_market_integration\.utils import (\s*# Configuration\s*)([\w, _]+)', 
     r'from yemen_market_integration.utils.config import \2'),
    
    # Fix cache directories
    (r'@disk_cache\(cache_dir=\'\.cache/(\w+)\'\)', 
     r'@disk_cache(cache_dir=\'.cache/yemen_market_integration/\1\')'),
]

def fix_imports_in_file(file_path: Path) -> Tuple[int, List[str]]:
    """
    Fix imports in a single file.
    
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
    
    for pattern, replacement in IMPORT_PATTERNS:
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            content = new_content
            replacements_made.append(f"{pattern} -> {replacement} ({count} replacements)")
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return len(replacements_made), replacements_made
    
    return 0, []

def fix_imports_in_directory(directory: Path, file_pattern: str = "*.py") -> Dict[str, List[str]]:
    """
    Fix imports in all Python files in a directory.
    
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
            count, replacements = fix_imports_in_file(file_path)
            if count > 0:
                results[str(file_path)] = replacements
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Fix import inconsistencies in Python files")
    parser.add_argument("--directory", "-d", type=str, default="src",
                        help="Directory to process (default: src)")
    parser.add_argument("--pattern", "-p", type=str, default="*.py",
                        help="File pattern to match (default: *.py)")
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    print(f"Processing files in {directory}...")
    results = fix_imports_in_directory(directory, args.pattern)
    
    print(f"\nFixed imports in {len(results)} files:")
    for file_path, replacements in results.items():
        print(f"\n{file_path}:")
        for replacement in replacements:
            print(f"  - {replacement}")
    
    print(f"\nTotal files fixed: {len(results)}")

if __name__ == "__main__":
    main()