"""
Simple Memory Optimization Test for M3 Pro

This script provides a direct test of memory optimizations implemented in 
m3_utils.py, focusing specifically on array computation optimizations.
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
import functools

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
    m3_optimized, 
    optimize_array_computation, 
    configure_system_for_m3_performance,
    create_mmap_array
)

def create_large_arrays(size=1000):
    """Create large arrays for testing memory optimization."""
    # Create a large array
    X = np.random.normal(0, 1, (size, size))
    Y = np.random.normal(0, 1, (size, size))
    return X, Y

def test_matrix_multiplications(optimized=True):
    """Test memory usage for matrix multiplication."""
    
    # Force garbage collection
    gc.collect()
    
    # Create matrices
    size = 3000  # Large enough to show memory impact
    X, Y = create_large_arrays(size)
    
    def run_operations():
        # Perform memory-intensive operations
        if optimized:
            # Use optimized function - passing np.dot as the operation
            result = optimize_array_computation(X, operation=lambda arr: np.dot(arr, Y))
        else:
            # Direct computation
            result = np.dot(X, Y)
        
        return result
    
    # Measure memory usage
    mem_usage = max(memory_usage((run_operations, ()), interval=0.1))
    
    return mem_usage

def test_svd_decomposition(optimized=True):
    """Test memory usage for SVD decomposition."""
    
    # Force garbage collection
    gc.collect()
    
    # Create matrices
    size = 2000  # Large enough to show memory impact
    X, _ = create_large_arrays(size)
    
    def run_operations():
        if optimized:
            # Use optimized function with SVD
            result = optimize_array_computation(X, operation=lambda arr: np.linalg.svd(arr, full_matrices=False))
        else:
            # Direct computation
            result = np.linalg.svd(X, full_matrices=False)
        
        return result
    
    # Measure memory usage
    mem_usage = max(memory_usage((run_operations, ()), interval=0.1))
    
    return mem_usage

def test_array_chunking(optimized=True):
    """Test memory usage for operations with chunking."""
    
    # Force garbage collection
    gc.collect()
    
    # Create a large array
    size = 5000
    X = np.random.normal(0, 1, (size, size))
    
    # Define a memory-intensive operation
    def memory_intensive_op(arr):
        # This operation forces several temporary arrays to be created
        result = np.exp(arr) + np.sin(arr) + np.cos(arr)
        result = np.dot(result, result.T)
        return np.mean(result, axis=1)
    
    def run_operations():
        if optimized:
            # Use optimized chunking
            return optimize_array_computation(X, chunk_size=1000, operation=memory_intensive_op)
        else:
            # Direct computation
            return memory_intensive_op(X)
    
    # Measure memory usage
    mem_usage = max(memory_usage((run_operations, ()), interval=0.1))
    
    return mem_usage

def test_with_decorator(optimized=True):
    """Test the m3_optimized decorator's memory impact."""
    
    # Define two versions of the same function, one with the decorator
    if optimized:
        @m3_optimized(memory_intensive=True)
        def process_array(arr):
            result = np.exp(arr) + np.sin(arr)
            result = np.dot(result, result.T)
            return np.sum(result)
    else:
        def process_array(arr):
            result = np.exp(arr) + np.sin(arr)
            result = np.dot(result, result.T)
            return np.sum(result)
    
    # Create a large array
    size = 3000
    X, _ = create_large_arrays(size)
    
    # Force garbage collection
    gc.collect()
    
    # Measure memory usage
    mem_usage = max(memory_usage((process_array, (X,)), interval=0.1))
    
    return mem_usage

def main():
    """Run memory optimization benchmarks."""
    # Configure system for optimized performance
    configure_system_for_m3_performance()
    
    # Test matrix multiplications
    logger.info("Testing matrix multiplications without optimization...")
    non_opt_mul_mem = test_matrix_multiplications(optimized=False)
    logger.info(f"Non-optimized matrix multiplication peak memory: {non_opt_mul_mem:.1f} MB")
    
    logger.info("Testing matrix multiplications with optimization...")
    opt_mul_mem = test_matrix_multiplications(optimized=True)
    logger.info(f"Optimized matrix multiplication peak memory: {opt_mul_mem:.1f} MB")
    
    mul_reduction = ((non_opt_mul_mem - opt_mul_mem) / non_opt_mul_mem) * 100
    logger.info(f"Matrix multiplication memory reduction: {mul_reduction:.1f}%")
    
    # Test SVD decomposition
    logger.info("\nTesting SVD decomposition without optimization...")
    non_opt_svd_mem = test_svd_decomposition(optimized=False)
    logger.info(f"Non-optimized SVD peak memory: {non_opt_svd_mem:.1f} MB")
    
    logger.info("Testing SVD decomposition with optimization...")
    opt_svd_mem = test_svd_decomposition(optimized=True)
    logger.info(f"Optimized SVD peak memory: {opt_svd_mem:.1f} MB")
    
    svd_reduction = ((non_opt_svd_mem - opt_svd_mem) / non_opt_svd_mem) * 100
    logger.info(f"SVD memory reduction: {svd_reduction:.1f}%")
    
    # Test array chunking
    logger.info("\nTesting array operations without chunking...")
    non_opt_chunk_mem = test_array_chunking(optimized=False)
    logger.info(f"Non-optimized operation peak memory: {non_opt_chunk_mem:.1f} MB")
    
    logger.info("Testing array operations with chunking...")
    opt_chunk_mem = test_array_chunking(optimized=True)
    logger.info(f"Optimized chunked operation peak memory: {opt_chunk_mem:.1f} MB")
    
    chunk_reduction = ((non_opt_chunk_mem - opt_chunk_mem) / non_opt_chunk_mem) * 100
    logger.info(f"Chunking memory reduction: {chunk_reduction:.1f}%")
    
    # Test decorator
    logger.info("\nTesting function without m3_optimized decorator...")
    non_opt_dec_mem = test_with_decorator(optimized=False)
    logger.info(f"Non-decorated function peak memory: {non_opt_dec_mem:.1f} MB")
    
    logger.info("Testing function with m3_optimized decorator...")
    opt_dec_mem = test_with_decorator(optimized=True)
    logger.info(f"Decorated function peak memory: {opt_dec_mem:.1f} MB")
    
    dec_reduction = ((non_opt_dec_mem - opt_dec_mem) / non_opt_dec_mem) * 100
    logger.info(f"Decorator memory reduction: {dec_reduction:.1f}%")
    
    # Report overall results
    total_non_opt = non_opt_mul_mem + non_opt_svd_mem + non_opt_chunk_mem + non_opt_dec_mem
    total_opt = opt_mul_mem + opt_svd_mem + opt_chunk_mem + opt_dec_mem
    overall_reduction = ((total_non_opt - total_opt) / total_non_opt) * 100
    
    logger.info("\n--- OVERALL RESULTS ---")
    logger.info(f"Overall memory reduction: {overall_reduction:.1f}%")
    logger.info(f"60% Reduction Goal: {'MET' if overall_reduction >= 60 else 'NOT MET'}")

if __name__ == "__main__":
    main()