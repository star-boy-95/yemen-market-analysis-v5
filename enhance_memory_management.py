#!/usr/bin/env python3
"""
Script to enhance memory management in visualization methods.

This script adds proper memory cleanup to visualization methods,
particularly for interactive visualizations using Plotly.
"""
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set

def enhance_memory_management(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Enhance memory management in a file.
    
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
    
    # Check if this is a visualization file
    if 'visualization' in str(file_path) and ('plot_' in content or 'visualize' in content):
        # Pattern for interactive visualization methods using Plotly
        if 'plotly' in content and 'import go' in content:
            # Find interactive visualization methods
            interactive_methods = re.finditer(r'def\s+(plot_interactive_[^(]*)\s*\([^)]*\):', content)
            
            for method in interactive_methods:
                method_name = method.group(1)
                method_start = method.start()
                
                # Find the method end (next def or end of file)
                next_def = re.search(r'def\s+', content[method_start+1:])
                method_end = next_def.start() + method_start + 1 if next_def else len(content)
                
                method_content = content[method_start:method_end]
                
                # Check if the method already has memory cleanup
                if 'gc.collect()' not in method_content:
                    # Find the return statement
                    return_match = re.search(r'(\s*)return\s+(\w+)', method_content)
                    if return_match:
                        indent = return_match.group(1)
                        fig_var = return_match.group(2)
                        
                        # Add memory cleanup before return
                        memory_cleanup = f"\n{indent}# Force garbage collection to free memory\n{indent}gc.collect()\n"
                        
                        # Insert before return
                        return_pos = method_start + return_match.start()
                        new_content = content[:return_pos] + memory_cleanup + content[return_pos:]
                        content = new_content
                        
                        changes.append(f"Added memory cleanup to {method_name}")
                        
                        # If import gc is not present, add it
                        if 'import gc' not in content:
                            import_pos = content.find('import')
                            if import_pos >= 0:
                                content = content[:import_pos] + 'import gc\n' + content[import_pos:]
                                changes.append("Added import gc")
        
        # Pattern for matplotlib visualization methods
        if 'matplotlib' in content and ('plt.' in content or 'import plt' in content):
            # Find visualization methods that create figures
            fig_methods = re.finditer(r'def\s+([^(]*)\s*\([^)]*\):[^}]*?fig,\s*ax\s*=\s*plt\.subplots', content, re.DOTALL)
            
            for method in fig_methods:
                method_name = method.group(1)
                method_start = method.start()
                
                # Find the method end (next def or end of file)
                next_def = re.search(r'def\s+', content[method_start+1:])
                method_end = next_def.start() + method_start + 1 if next_def else len(content)
                
                method_content = content[method_start:method_end]
                
                # Check if the method already has figure cleanup
                if 'plt.close(fig)' not in method_content and 'plt.close("all")' not in method_content:
                    # Find the save_plot call if it exists
                    save_match = re.search(r'(\s*)save_plot\(', method_content)
                    
                    if save_match:
                        indent = save_match.group(1)
                        
                        # Add figure cleanup after save_plot
                        save_line_end = method_content.find('\n', save_match.start())
                        if save_line_end > 0:
                            save_line_end = save_line_end - method_start
                            
                            # Insert after save_plot line
                            cleanup_code = f"\n{indent}# Close figure to free memory if not needed\n{indent}if not config.get('visualization.keep_figures', False):\n{indent}    plt.close(fig)\n"
                            
                            insert_pos = method_start + save_line_end + 1
                            new_content = content[:insert_pos] + cleanup_code + content[insert_pos:]
                            content = new_content
                            
                            changes.append(f"Added figure cleanup to {method_name}")
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    
    return False, []

def enhance_memory_management_in_directory(directory: Path, file_pattern: str = "*.py") -> Dict[str, List[str]]:
    """
    Enhance memory management in all Python files in a directory.
    
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
            changed, changes = enhance_memory_management(file_path)
            if changed:
                results[str(file_path)] = changes
    
    return results

def create_memory_management_guide(output_path: Path):
    """
    Create a guide for memory management in visualization.
    
    Parameters
    ----------
    output_path : Path
        Path to write the guide
    """
    guide_content = """# Memory Management in Visualization

This document outlines best practices for memory management in visualization code.

## Memory Leaks in Visualization

Visualization libraries like Matplotlib and Plotly can consume significant memory,
especially when creating many figures or interactive visualizations. Common issues include:

1. **Unclosed Figures**: Matplotlib figures that aren't explicitly closed
2. **Reference Cycles**: Plotly figures with circular references
3. **Large Data**: Visualizations of large datasets without downsampling
4. **Cached Objects**: Objects stored in memory for reuse

## Best Practices

### For Matplotlib

```python
def plot_time_series(data):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    ax.plot(data)
    
    # Save if needed
    if save_path:
        save_plot(fig, save_path)
        
    # Close figure to free memory if not needed
    if not config.get('visualization.keep_figures', False):
        plt.close(fig)
    
    return fig, ax
```

### For Plotly

```python
def plot_interactive(data):
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(x=data.index, y=data.values))
    
    # Update layout
    fig.update_layout(title="Interactive Plot")
    
    # Save if needed
    if save_path:
        fig.write_html(save_path)
    
    # Force garbage collection to free memory
    gc.collect()
    
    return fig
```

## Configuration Options

Add these options to your configuration file:

```yaml
visualization:
  keep_figures: false  # Set to true during debugging
  max_figures: 10      # Maximum number of figures to keep in memory
  downsample_threshold: 1000  # Downsample data with more points than this
```

## Memory Profiling

Use the `memory_usage_decorator` to track memory usage:

```python
@memory_usage_decorator
def create_visualization(data):
    # Implementation
```

This will log memory usage before and after the function execution.
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"Created memory management guide at {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhance memory management in visualization code")
    parser.add_argument("--directory", "-d", type=str, default="src/visualization",
                        help="Directory to process (default: src/visualization)")
    parser.add_argument("--pattern", "-p", type=str, default="*.py",
                        help="File pattern to match (default: *.py)")
    parser.add_argument("--create-guide", "-g", action="store_true",
                        help="Create memory management guide")
    parser.add_argument("--guide-path", type=str, default="memory_management_guide.md",
                        help="Path for memory guide (default: memory_management_guide.md)")
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    print(f"Processing files in {directory}...")
    results = enhance_memory_management_in_directory(directory, args.pattern)
    
    print(f"\nEnhanced memory management in {len(results)} files:")
    for file_path, changes in results.items():
        print(f"\n{file_path}:")
        for change in changes:
            print(f"  - {change}")
    
    print(f"\nTotal files modified: {len(results)}")
    
    if args.create_guide:
        create_memory_management_guide(Path(args.guide_path))

if __name__ == "__main__":
    main()