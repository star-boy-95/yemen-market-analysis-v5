#!/usr/bin/env python3
"""
Script to standardize cache directories in the Yemen Market Analysis codebase.

This script updates all @disk_cache decorators to use a consistent cache directory
pattern across the codebase.
"""
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Define the cache directory pattern to replace
CACHE_PATTERN = r'@disk_cache\(cache_dir=\'\.cache/([^/\']+)\'\)'
CACHE_REPLACEMENT = r'@disk_cache(cache_dir=\'.cache/yemen_market_integration/\1\')'

# Define version for cache versioning
CACHE_VERSION = "1.0.0"

def standardize_cache_in_file(file_path: Path, add_version: bool = False) -> Tuple[int, List[str]]:
    """
    Standardize cache directories in a single file.
    
    Parameters
    ----------
    file_path : Path
        Path to the file to fix
    add_version : bool, optional
        Whether to add version to cache directories, default False
        
    Returns
    -------
    Tuple[int, List[str]]
        Number of replacements made and list of patterns replaced
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    replacements = []
    
    # Replace cache directories
    if add_version:
        # With version
        pattern = CACHE_PATTERN
        replacement = r'@disk_cache(cache_dir=\'.cache/yemen_market_integration/\1/v' + CACHE_VERSION + r'\')'
    else:
        # Without version
        pattern = CACHE_PATTERN
        replacement = CACHE_REPLACEMENT
    
    new_content, count = re.subn(pattern, replacement, content)
    
    if count > 0:
        replacements.append(f"Replaced {count} cache directories")
        content = new_content
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return count, replacements
    
    return 0, []

def standardize_cache_in_directory(directory: Path, file_pattern: str = "*.py", add_version: bool = False) -> Dict[str, List[str]]:
    """
    Standardize cache directories in all Python files in a directory.
    
    Parameters
    ----------
    directory : Path
        Directory to process
    file_pattern : str, optional
        Glob pattern for files to process, default "*.py"
    add_version : bool, optional
        Whether to add version to cache directories, default False
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping file paths to lists of replacements made
    """
    results = {}
    
    for file_path in directory.glob(f"**/{file_pattern}"):
        if file_path.is_file():
            count, replacements = standardize_cache_in_file(file_path, add_version)
            if count > 0:
                results[str(file_path)] = replacements
    
    return results

def create_cache_cleanup_script(output_path: Path, max_age_days: int = 30):
    """
    Create a script to clean up old cache files.
    
    Parameters
    ----------
    output_path : Path
        Path to write the cleanup script
    max_age_days : int, optional
        Maximum age of cache files in days, default 30
    """
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Cache cleanup script for Yemen Market Analysis.

This script removes cache files older than {max_age_days} days.
\"\"\"
import os
import time
from pathlib import Path
import argparse
import shutil

def cleanup_old_caches(cache_dir: Path, max_age_days: int = {max_age_days}, dry_run: bool = False):
    \"\"\"
    Remove cache files older than specified days.
    
    Parameters
    ----------
    cache_dir : Path
        Cache directory to clean
    max_age_days : int, optional
        Maximum age of cache files in days, default {max_age_days}
    dry_run : bool, optional
        If True, only print what would be deleted without actually deleting
    \"\"\"
    if not cache_dir.exists():
        print(f"Cache directory {{cache_dir}} does not exist.")
        return
        
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    deleted_files = 0
    deleted_dirs = 0
    freed_space = 0
    
    # First, remove old cache files
    for cache_file in cache_dir.glob('**/*.cache'):
        try:
            file_age_seconds = current_time - cache_file.stat().st_mtime
            if file_age_seconds > max_age_seconds:
                size = cache_file.stat().st_size
                if dry_run:
                    print(f"Would delete: {{cache_file}} ({{file_age_seconds / (24 * 60 * 60):.1f}} days old)")
                else:
                    cache_file.unlink()
                    deleted_files += 1
                    freed_space += size
        except Exception as e:
            print(f"Error processing {{cache_file}}: {{e}}")
    
    # Then, remove empty directories
    for dirpath, dirnames, filenames in os.walk(cache_dir, topdown=False):
        dir_path = Path(dirpath)
        if dir_path != cache_dir and not any(dir_path.iterdir()):
            if dry_run:
                print(f"Would remove empty directory: {{dir_path}}")
            else:
                try:
                    dir_path.rmdir()
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Error removing directory {{dir_path}}: {{e}}")
    
    # Print summary
    if dry_run:
        print(f"\\nDry run summary:")
        print(f"Would delete {{deleted_files}} files and {{deleted_dirs}} directories")
        print(f"Would free approximately {{freed_space / (1024 * 1024):.2f}} MB")
    else:
        print(f"\\nCleanup summary:")
        print(f"Deleted {{deleted_files}} files and {{deleted_dirs}} directories")
        print(f"Freed approximately {{freed_space / (1024 * 1024):.2f}} MB")

def main():
    parser = argparse.ArgumentParser(description="Clean up old cache files")
    parser.add_argument("--cache-dir", type=str, default=".cache/yemen_market_integration",
                        help="Cache directory to clean (default: .cache/yemen_market_integration)")
    parser.add_argument("--max-age", type=int, default={max_age_days},
                        help=f"Maximum age of cache files in days (default: {max_age_days})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be deleted without actually deleting")
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    
    print(f"Cleaning up cache files older than {{args.max_age}} days in {{cache_dir}}...")
    cleanup_old_caches(cache_dir, args.max_age, args.dry_run)

if __name__ == "__main__":
    main()
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_path, 0o755)
    
    print(f"Created cache cleanup script at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Standardize cache directories in Python files")
    parser.add_argument("--directory", "-d", type=str, default="src",
                        help="Directory to process (default: src)")
    parser.add_argument("--pattern", "-p", type=str, default="*.py",
                        help="File pattern to match (default: *.py)")
    parser.add_argument("--add-version", "-v", action="store_true",
                        help="Add version to cache directories")
    parser.add_argument("--create-cleanup", "-c", action="store_true",
                        help="Create cache cleanup script")
    parser.add_argument("--cleanup-script", type=str, default="cleanup_cache.py",
                        help="Path for cleanup script (default: cleanup_cache.py)")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Maximum age of cache files in days for cleanup script (default: 30)")
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    print(f"Processing files in {directory}...")
    results = standardize_cache_in_directory(directory, args.pattern, args.add_version)
    
    print(f"\nStandardized cache directories in {len(results)} files:")
    for file_path, replacements in results.items():
        print(f"\n{file_path}:")
        for replacement in replacements:
            print(f"  - {replacement}")
    
    print(f"\nTotal files modified: {len(results)}")
    
    if args.create_cleanup:
        create_cache_cleanup_script(Path(args.cleanup_script), args.max_age)

if __name__ == "__main__":
    main()