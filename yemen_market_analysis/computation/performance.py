"""
Performance optimization utilities for Yemen Market Analysis.
"""
import time
import logging
import functools
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from core.decorators import error_handler
from core.exceptions import ComputationError

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track performance metrics for code optimization."""
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.timings = {}
        self.call_counts = {}
        self.memory_usage = {}
    
    def start_timer(self, name: str) -> None:
        """Start a timer for a named operation."""
        if name not in self.timings:
            self.timings[name] = []
            self.call_counts[name] = 0
        
        setattr(self, f"_start_{name}", time.time())
        self.call_counts[name] += 1
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed time."""
        start_time = getattr(self, f"_start_{name}", None)
        
        if start_time is None:
            logger.warning(f"Timer {name} was not started")
            return 0.0
        
        elapsed = time.time() - start_time
        self.timings[name].append(elapsed)
        
        return elapsed
    
    def get_average_time(self, name: str) -> float:
        """Get average time for a named operation."""
        if name not in self.timings or not self.timings[name]:
            return 0.0
            
        return np.mean(self.timings[name])
    
    def get_total_time(self, name: str) -> float:
        """Get total time for a named operation."""
        if name not in self.timings or not self.timings[name]:
            return 0.0
            
        return np.sum(self.timings[name])
    
    def get_call_count(self, name: str) -> int:
        """Get number of calls for a named operation."""
        return self.call_counts.get(name, 0)
    
    def reset(self) -> None:
        """Reset all performance tracking data."""
        self.timings = {}
        self.call_counts = {}
        self.memory_usage = {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        summary = {}
        
        for name in self.timings:
            if not self.timings[name]:
                continue
                
            summary[name] = {
                "count": self.call_counts[name],
                "total_time": np.sum(self.timings[name]),
                "average_time": np.mean(self.timings[name]),
                "min_time": np.min(self.timings[name]),
                "max_time": np.max(self.timings[name])
            }
        
        return summary
    
    def print_summary(self) -> None:
        """Print a summary of performance metrics."""
        summary = self.get_summary()
        
        if not summary:
            logger.info("No performance data available")
            return
        
        logger.info("Performance Summary:")
        
        # Sort by total time
        sorted_ops = sorted(summary.keys(), key=lambda x: summary[x]["total_time"], reverse=True)
        
        for name in sorted_ops:
            stats = summary[name]
            logger.info(
                f"{name}: {stats['count']} calls, "
                f"total: {stats['total_time']:.3f}s, "
                f"avg: {stats['average_time']:.3f}s, "
                f"min: {stats['min_time']:.3f}s, "
                f"max: {stats['max_time']:.3f}s"
            )


# Global performance tracker instance
performance_tracker = PerformanceTracker()


def track_performance(name: Optional[str] = None):
    """
    Decorator to track performance of a function.
    
    Args:
        name: Optional custom name for the function
    """
    def decorator(func):
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            performance_tracker.start_timer(func_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = performance_tracker.stop_timer(func_name)
                logger.debug(f"{func_name} completed in {elapsed:.3f}s")
        
        return wrapper
    
    return decorator


@error_handler(fallback_value=None)
def optimize_computation_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize computation settings based on system capabilities.
    
    Args:
        config: Current configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    import os
    import psutil
    
    # Get system information
    cpu_count = os.cpu_count() or 4
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Optimize parallelization
    parallel_workers = max(1, min(cpu_count - 1, config.get('commodity_parallel_processes', 4)))
    config['commodity_parallel_processes'] = parallel_workers
    
    # Optimize memory usage
    if memory_gb < 4:
        # Low memory system
        config['optimization_level'] = 'low_memory'
        config['use_gpu'] = False
    elif memory_gb < 8:
        # Medium memory system
        config['optimization_level'] = 'medium'
    else:
        # High memory system
        config['optimization_level'] = 'high'
    
    # Check GPU capability
    try:
        from .device_manager import get_device_manager
        device_manager = get_device_manager()
        
        if device_manager.is_gpu_available():
            config['use_gpu'] = True
            
            # Specific GPU optimizations
            if device_manager.is_cuda:
                config['gpu_type'] = 'cuda'
            elif device_manager.is_mps:
                config['gpu_type'] = 'mps'
    except Exception as e:
        logger.warning(f"Error detecting GPU capabilities: {str(e)}")
        config['use_gpu'] = False
    
    logger.info(
        f"Optimized computation settings: "
        f"workers={config['commodity_parallel_processes']}, "
        f"memory_level={config['optimization_level']}, "
        f"use_gpu={config.get('use_gpu', False)}"
    )
    
    return config


@error_handler(fallback_value=None)
def chunk_large_computation(
    data: np.ndarray,
    chunk_size: int,
    process_func: Callable[[np.ndarray], Any]
) -> List[Any]:
    """
    Process large datasets in chunks to optimize memory usage.
    
    Args:
        data: Large data array
        chunk_size: Number of rows per chunk
        process_func: Function to process each chunk
        
    Returns:
        List of processed chunk results
    """
    if len(data) <= chunk_size:
        return [process_func(data)]
    
    results = []
    n_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    logger.info(f"Processing {len(data)} rows in {n_chunks} chunks of size {chunk_size}")
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        
        chunk = data[start_idx:end_idx]
        
        try:
            result = process_func(chunk)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {str(e)}")
    
    return results