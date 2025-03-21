#!/usr/bin/env python3
"""
Script to run all code improvements for the Yemen Market Analysis codebase.

This script runs all the improvement scripts in the correct order to ensure
that the codebase is optimized for performance, error handling, and maintainability.
"""
import os
import subprocess
import argparse
from pathlib import Path
import time
import sys
from typing import List, Dict

# Define the improvement scripts to run
IMPROVEMENT_SCRIPTS = [
    {
        "name": "Fix Cache Paths",
        "script": "fix_cache_paths.py",
        "args": []
    },
    {
        "name": "Standardize Cache Directories",
        "script": "standardize_cache_dirs.py",
        "args": ["--add-version", "--create-cleanup"]
    },
    {
        "name": "Fix Imports",
        "script": "fix_imports.py",
        "args": []
    },
    {
        "name": "Standardize Error Handling",
        "script": "standardize_error_handling.py",
        "args": ["--create-guide"]
    },
    {
        "name": "Enhance M1 Decorator",
        "script": "enhance_m1_decorator.py",
        "args": []
    },
    {
        "name": "Standardize Decorator Order",
        "script": "standardize_decorator_order.py",
        "args": ["--create-guide"]
    },
    {
        "name": "Remove Redundant Validation",
        "script": "remove_redundant_validation.py",
        "args": ["--create-guide"]
    },
    {
        "name": "Enhance Memory Management",
        "script": "enhance_memory_management.py",
        "args": ["--create-guide"]
    }
]

def run_script(script_path: Path, args: List[str] = None) -> bool:
    """
    Run a Python script with the given arguments.
    
    Parameters
    ----------
    script_path : Path
        Path to the script to run
    args : List[str], optional
        Arguments to pass to the script
        
    Returns
    -------
    bool
        Whether the script ran successfully
    """
    if args is None:
        args = []
    
    if not script_path.exists():
        print(f"Error: Script {script_path} does not exist")
        return False
    
    try:
        cmd = [sys.executable, str(script_path)] + args
        print(f"Running: {' '.join(cmd)}")
        
        # Run the script and capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Print output
        print(result.stdout)
        
        if result.stderr:
            print(f"Warnings/Errors:\n{result.stderr}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def run_all_improvements(directory: Path, skip: List[str] = None) -> Dict[str, bool]:
    """
    Run all improvement scripts.
    
    Parameters
    ----------
    directory : Path
        Directory to process
    skip : List[str], optional
        List of script names to skip
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping script names to success status
    """
    if skip is None:
        skip = []
    
    results = {}
    
    for improvement in IMPROVEMENT_SCRIPTS:
        name = improvement["name"]
        script = improvement["script"]
        args = improvement["args"]
        
        if script in skip:
            print(f"Skipping {name} ({script})")
            results[name] = None
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Running {name} ({script})")
        print(f"{'=' * 80}\n")
        
        script_path = Path(script)
        
        # Add directory argument if not already present
        if "--directory" not in args and "-d" not in args:
            args = args + ["--directory", str(directory)]
        
        start_time = time.time()
        success = run_script(script_path, args)
        end_time = time.time()
        
        results[name] = success
        
        print(f"\nCompleted {name} in {end_time - start_time:.2f} seconds")
        print(f"Result: {'Success' if success else 'Failed'}")
        
        # Pause between scripts
        if name != IMPROVEMENT_SCRIPTS[-1]["name"]:
            print("\nPausing before next script...")
            time.sleep(1)
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run all code improvements")
    parser.add_argument("--directory", "-d", type=str, default="src",
                        help="Directory to process (default: src)")
    parser.add_argument("--skip", "-s", type=str, nargs="+",
                        help="Scripts to skip")
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    print(f"Running all improvements on {directory}...")
    results = run_all_improvements(directory, args.skip)
    
    print("\n\nSummary of Results:")
    print(f"{'=' * 80}")
    
    all_success = True
    for name, success in results.items():
        if success is None:
            status = "Skipped"
        elif success:
            status = "Success"
        else:
            status = "Failed"
            all_success = False
        
        print(f"{name}: {status}")
    
    print(f"{'=' * 80}")
    print(f"Overall Result: {'Success' if all_success else 'Some improvements failed'}")
    
    # Create a summary file
    summary_path = Path("improvement_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Code Improvement Summary\n\n")
        f.write("The following improvements were applied to the Yemen Market Analysis codebase:\n\n")
        
        for name, success in results.items():
            if success is None:
                status = "⏭️ Skipped"
            elif success:
                status = "✅ Success"
            else:
                status = "❌ Failed"
            
            f.write(f"- **{name}**: {status}\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review the generated guide files for best practices\n")
        f.write("2. Run the test suite to ensure all functionality still works\n")
        f.write("3. Commit the changes to version control\n")
    
    print(f"\nSummary written to {summary_path}")

if __name__ == "__main__":
    main()