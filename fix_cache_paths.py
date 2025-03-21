#!/usr/bin/env python3
"""
Script to fix escaped single quotes in cache directory paths.
"""
import os
import re
from pathlib import Path

def fix_cache_paths(file_path):
    """
    Fix escaped single quotes in cache directory paths.
    
    Parameters
    ----------
    file_path : str
        Path to the file to fix
        
    Returns
    -------
    bool
        Whether any changes were made
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace escaped single quotes with double quotes
    pattern = r"@disk_cache\(cache_dir=\\'(.*?)\\'\)"
    replacement = r'@disk_cache(cache_dir="\1")'
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count > 0:
        print(f"Fixed {count} cache paths in {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    
    return False

def main():
    """
    Main function to fix cache paths in all Python files.
    """
    src_dir = Path('src')
    
    # Find all Python files
    python_files = list(src_dir.glob('**/*.py'))
    
    # Fix cache paths in all Python files
    fixed_files = 0
    for file_path in python_files:
        if fix_cache_paths(file_path):
            fixed_files += 1
    
    print(f"\nFixed cache paths in {fixed_files} files")

if __name__ == "__main__":
    main()