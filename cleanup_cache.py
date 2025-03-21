#!/usr/bin/env python3
"""
Cache cleanup script for Yemen Market Analysis.

This script removes cache files older than 30 days.
"""
import os
import time
from pathlib import Path
import argparse
import shutil

def cleanup_old_caches(cache_dir: Path, max_age_days: int = 30, dry_run: bool = False):
    """
    Remove cache files older than specified days.
    
    Parameters
    ----------
    cache_dir : Path
        Cache directory to clean
    max_age_days : int, optional
        Maximum age of cache files in days, default 30
    dry_run : bool, optional
        If True, only print what would be deleted without actually deleting
    """
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} does not exist.")
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
                    print(f"Would delete: {cache_file} ({file_age_seconds / (24 * 60 * 60):.1f} days old)")
                else:
                    cache_file.unlink()
                    deleted_files += 1
                    freed_space += size
        except Exception as e:
            print(f"Error processing {cache_file}: {e}")
    
    # Then, remove empty directories
    for dirpath, dirnames, filenames in os.walk(cache_dir, topdown=False):
        dir_path = Path(dirpath)
        if dir_path != cache_dir and not any(dir_path.iterdir()):
            if dry_run:
                print(f"Would remove empty directory: {dir_path}")
            else:
                try:
                    dir_path.rmdir()
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")
    
    # Print summary
    if dry_run:
        print(f"\nDry run summary:")
        print(f"Would delete {deleted_files} files and {deleted_dirs} directories")
        print(f"Would free approximately {freed_space / (1024 * 1024):.2f} MB")
    else:
        print(f"\nCleanup summary:")
        print(f"Deleted {deleted_files} files and {deleted_dirs} directories")
        print(f"Freed approximately {freed_space / (1024 * 1024):.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Clean up old cache files")
    parser.add_argument("--cache-dir", type=str, default=".cache/yemen_market_integration",
                        help="Cache directory to clean (default: .cache/yemen_market_integration)")
    parser.add_argument("--max-age", type=int, default=30,
                        help=f"Maximum age of cache files in days (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be deleted without actually deleting")
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    
    print(f"Cleaning up cache files older than {args.max_age} days in {cache_dir}...")
    cleanup_old_caches(cache_dir, args.max_age, args.dry_run)

if __name__ == "__main__":
    main()
