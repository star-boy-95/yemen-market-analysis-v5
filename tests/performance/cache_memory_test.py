"""
Memory-Mapped Array and Cache Optimization Test

Tests the memory reduction capabilities specifically using memory-mapped arrays
and the tiered caching system, which are key components of the M3 optimizations.
"""
import os
import gc
import sys
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
import psutil
import logging
import time
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.m3_utils import (
    create_mmap_array,
    TieredCache,
    configure_system_for_m3_performance
)

def test_mmap_array(use_mmap=True):
    """Test memory usage with and without memory-mapped arrays."""
    # Force garbage collection
    gc.collect()
    
    # Create a large array that will exceed memory limits if processed directly
    rows, cols = 20000, 5000  # 100 million elements, ~800MB for float64
    
    def run_operations():
        # Create large array
        logger.info(f"Creating array of shape ({rows}, {cols})...")
        X = np.random.normal(0, 1, (rows, cols))
        
        if use_mmap:
            # Create memory-mapped copy
            logger.info("Creating memory-mapped array...")
            X_mmap, mmap_file = create_mmap_array(X)
            
            # Process the memory-mapped array
            logger.info("Processing memory-mapped array...")
            result = np.sum(X_mmap, axis=1)
            
            # Clean up memory map
            if hasattr(mmap_file, 'close'):
                mmap_file.close()
            return result
        else:
            # Process directly in memory
            logger.info("Processing array directly in memory...")
            return np.sum(X, axis=1)
    
    # Measure memory usage
    mem_usage = max(memory_usage((run_operations, ()), interval=0.1))
    
    return mem_usage

def test_large_matrix_operations(use_optimization=True):
    """Test memory usage for large matrix operations."""
    # Force garbage collection
    gc.collect()
    
    size = 10000  # Create a 10K x 10K matrix (800MB for float64)
    
    def run_operations():
        logger.info(f"Creating {size}x{size} matrix...")
        X = np.random.normal(0, 1, (size, size))
        
        if use_optimization:
            # Process in chunks to reduce memory usage
            logger.info("Processing matrix in chunks...")
            chunk_size = 2000
            result = np.zeros(size)
            
            # Process matrix multiplication in chunks
            for i in range(0, size, chunk_size):
                end = min(i + chunk_size, size)
                chunk = X[i:end, :]
                # Calculate column sums for this chunk
                result[i:end] = np.sum(chunk, axis=1)
                
                # Force garbage collection after each chunk
                if i % (chunk_size * 2) == 0:
                    gc.collect()
        else:
            # Calculate entire sum at once
            logger.info("Processing entire matrix at once...")
            result = np.sum(X, axis=1)
            
        return result
    
    # Measure memory usage
    mem_usage = max(memory_usage((run_operations, ()), interval=0.1))
    
    return mem_usage

def test_tiered_cache():
    """Test memory usage reduction with the TieredCache system."""
    # Force garbage collection
    gc.collect()
    
    # Create a temporary directory for disk cache
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test without cache
        def run_without_cache():
            # Simulate processing large datasets with repeated access
            results = {}
            for i in range(20):
                # Create a "large" dataset (~40MB each)
                key = f"dataset_{i % 5}"  # Only 5 unique datasets, to test caching
                data = np.random.normal(0, 1, (1000, 1000))
                
                # Process the data
                processed = np.exp(data) + np.sin(data)
                results[key] = np.mean(processed)
            return results
        
        # Measure memory usage without cache
        no_cache_mem = max(memory_usage((run_without_cache, ()), interval=0.1))
        logger.info(f"Memory usage without cache: {no_cache_mem:.1f} MB")
        
        # Test with cache
        def run_with_cache():
            # Create tiered cache
            cache = TieredCache(
                maxsize=10,
                disk_cache_dir=temp_dir,
                memory_limit_mb=100
            )
            
            # Simulate processing with cache
            results = {}
            for i in range(20):
                key = f"dataset_{i % 5}"  # Only 5 unique datasets
                
                # Check if in cache
                cached_data = cache.get(key)
                if cached_data is None:
                    # Not in cache, create and process
                    data = np.random.normal(0, 1, (1000, 1000))
                    processed = np.exp(data) + np.sin(data)
                    
                    # Store in cache
                    cache.set(key, processed)
                else:
                    processed = cached_data
                
                results[key] = np.mean(processed)
            return results
        
        # Measure memory usage with cache
        with_cache_mem = max(memory_usage((run_with_cache, ()), interval=0.1))
        logger.info(f"Memory usage with cache: {with_cache_mem:.1f} MB")
        
        # Calculate reduction
        reduction = ((no_cache_mem - with_cache_mem) / no_cache_mem) * 100
        logger.info(f"Memory reduction with tiered cache: {reduction:.1f}%")
        
        return no_cache_mem, with_cache_mem, reduction
    
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

def main():
    """Run memory optimization benchmarks focused on M3 memory techniques."""
    # Configure system for optimized performance
    configure_system_for_m3_performance()
    
    # Test memory-mapped arrays
    logger.info("Testing large array processing without memory mapping...")
    non_mmap_mem = test_mmap_array(use_mmap=False)
    logger.info(f"Memory usage without memory mapping: {non_mmap_mem:.1f} MB")
    
    logger.info("Testing large array processing with memory mapping...")
    mmap_mem = test_mmap_array(use_mmap=True)
    logger.info(f"Memory usage with memory mapping: {mmap_mem:.1f} MB")
    
    mmap_reduction = ((non_mmap_mem - mmap_mem) / non_mmap_mem) * 100
    logger.info(f"Memory reduction with memory mapping: {mmap_reduction:.1f}%")
    
    # Test large matrix operations
    logger.info("\nTesting large matrix operations without chunking...")
    non_chunked_mem = test_large_matrix_operations(use_optimization=False)
    logger.info(f"Memory usage without chunking: {non_chunked_mem:.1f} MB")
    
    logger.info("Testing large matrix operations with chunking...")
    chunked_mem = test_large_matrix_operations(use_optimization=True)
    logger.info(f"Memory usage with chunking: {chunked_mem:.1f} MB")
    
    chunk_reduction = ((non_chunked_mem - chunked_mem) / non_chunked_mem) * 100
    logger.info(f"Memory reduction with chunking: {chunk_reduction:.1f}%")
    
    # Test tiered cache
    logger.info("\nTesting tiered cache system...")
    no_cache_mem, with_cache_mem, cache_reduction = test_tiered_cache()
    
    # Report overall results
    overall_reduction = (
        (non_mmap_mem + non_chunked_mem + no_cache_mem - mmap_mem - chunked_mem - with_cache_mem) / 
        (non_mmap_mem + non_chunked_mem + no_cache_mem)
    ) * 100
    
    logger.info("\n--- OVERALL RESULTS ---")
    logger.info(f"Memory mapping reduction: {mmap_reduction:.1f}%")
    logger.info(f"Chunking reduction: {chunk_reduction:.1f}%")
    logger.info(f"Cache reduction: {cache_reduction:.1f}%")
    logger.info(f"Overall memory reduction: {overall_reduction:.1f}%")
    logger.info(f"60% Reduction Goal: {'MET' if overall_reduction >= 60 else 'NOT MET'}")

if __name__ == "__main__":
    main()