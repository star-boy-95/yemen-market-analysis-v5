"""
Performance optimization module for Yemen Market Analysis.

This module provides utilities for optimizing the performance of the Yemen Market
Analysis package. It includes classes for memory management and parallel processing.
"""
import logging
import gc
import os
import psutil
import multiprocessing as mp
from typing import Dict, List, Optional, Union, Any, Callable, TypeVar, cast
from functools import partial

import pandas as pd
import numpy as np

from src.utils.error_handling import YemenAnalysisError, handle_errors

# Type variable for generic type hints
T = TypeVar('T')
R = TypeVar('R')

# Initialize logger
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Memory manager for Yemen Market Analysis.

    This class provides methods for managing memory usage during analysis.

    Attributes:
        threshold (float): Memory usage threshold (as a fraction of total memory).
        verbose (bool): Whether to log memory usage information.
    """

    def __init__(self, threshold: float = 0.8, verbose: bool = False):
        """
        Initialize the memory manager.

        Args:
            threshold: Memory usage threshold (as a fraction of total memory).
            verbose: Whether to log memory usage information.
        """
        self.threshold = threshold
        self.verbose = verbose

    @handle_errors
    def check_memory(self) -> Dict[str, float]:
        """
        Check current memory usage.

        Returns:
            Dictionary containing memory usage information.
        """
        # Get memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Get system memory
        system_memory = psutil.virtual_memory()

        # Calculate memory usage
        memory_usage = memory_info.rss / system_memory.total

        # Log memory usage
        if self.verbose:
            logger.info(f"Memory usage: {memory_usage:.2%}")

        # Check if memory usage is above threshold
        if memory_usage > self.threshold:
            logger.warning(f"Memory usage ({memory_usage:.2%}) is above threshold ({self.threshold:.2%})")

        # Return memory usage information
        return {
            'memory_usage': memory_usage,
            'memory_used': memory_info.rss,
            'memory_total': system_memory.total,
            'above_threshold': memory_usage > self.threshold,
        }

    @handle_errors
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage of a DataFrame.

        Args:
            df: DataFrame to optimize.

        Returns:
            Optimized DataFrame.
        """
        # Make a copy of the DataFrame
        df_optimized = df.copy()

        # Get memory usage before optimization
        memory_before = df_optimized.memory_usage(deep=True).sum()

        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=['int']).columns:
            # Get column min and max
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()

            # Convert to smallest possible integer type
            if col_min >= 0:
                if col_max < 2**8:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif col_max < 2**16:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif col_max < 2**32:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.uint64)
            else:
                if col_min > -2**7 and col_max < 2**7:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif col_min > -2**15 and col_max < 2**15:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif col_min > -2**31 and col_max < 2**31:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.int64)

        # Optimize float columns
        for col in df_optimized.select_dtypes(include=['float']).columns:
            df_optimized[col] = df_optimized[col].astype(np.float32)

        # Optimize object columns (convert to category if cardinality is low)
        for col in df_optimized.select_dtypes(include=['object']).columns:
            # Get number of unique values
            n_unique = df_optimized[col].nunique()

            # Convert to category if cardinality is low
            if n_unique / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')

        # Get memory usage after optimization
        memory_after = df_optimized.memory_usage(deep=True).sum()

        # Log memory usage
        if self.verbose:
            logger.info(f"Memory usage reduced from {memory_before / 1e6:.2f} MB to {memory_after / 1e6:.2f} MB ({1 - memory_after / memory_before:.2%} reduction)")

        return df_optimized

    @handle_errors
    def clear_memory(self) -> None:
        """
        Clear memory by running garbage collection.
        """
        # Run garbage collection
        gc.collect()

        # Log memory usage
        if self.verbose:
            logger.info("Memory cleared")

        # Check memory usage
        self.check_memory()


class ParallelProcessor:
    """
    Parallel processor for Yemen Market Analysis.

    This class provides methods for parallel processing of data.

    Attributes:
        n_jobs (int): Number of processes to use.
        verbose (bool): Whether to log processing information.
    """

    def __init__(self, n_jobs: Optional[int] = None, verbose: bool = False):
        """
        Initialize the parallel processor.

        Args:
            n_jobs: Number of processes to use. If None, uses the number of CPU cores.
            verbose: Whether to log processing information.
        """
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
        self.verbose = verbose
        
    @handle_errors
    def process(self, func: Callable[[T], R], items: List[T], *args: Any, **kwargs: Any) -> List[R]:
        """
        Process a list of items in parallel using the specified function.
        
        This is a general-purpose method for parallel processing of data.
        
        Args:
            func: Function to apply to each item.
            items: List of items to process.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.
            
        Returns:
            List of results from processing each item.
        """
        if not items:
            return []
            
        if self.verbose:
            logger.info(f"Processing {len(items)} items in parallel with {self.n_jobs} processes")
            
        if args or kwargs:
            # If we have additional arguments, use partial to bind them
            process_func = partial(func, *args, **kwargs)
        else:
            process_func = func
            
        # Use multiprocessing Pool for parallel execution
        with mp.Pool(processes=self.n_jobs) as pool:
            results = pool.map(process_func, items)
            
        if self.verbose:
            logger.info(f"Parallel processing complete, got {len(results)} results")
            
        return results

    @handle_errors
    def map(self, func: Callable[[T], R], items: List[T]) -> List[R]:
        """
        Apply a function to each item in a list in parallel.

        Args:
            func: Function to apply.
            items: List of items to process.

        Returns:
            List of results.
        """
        # Log processing information
        if self.verbose:
            logger.info(f"Processing {len(items)} items using {self.n_jobs} processes")

        # Create a pool of processes
        with mp.Pool(processes=self.n_jobs) as pool:
            # Apply function to each item in parallel
            results = pool.map(func, items)

        # Log processing information
        if self.verbose:
            logger.info(f"Processed {len(items)} items")

        return results

    @handle_errors
    def map_with_args(
        self, func: Callable[..., R], items: List[Any], *args: Any, **kwargs: Any
    ) -> List[R]:
        """
        Apply a function to each item in a list in parallel, with additional arguments.

        Args:
            func: Function to apply.
            items: List of items to process.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            List of results.
        """
        # Create a partial function with the additional arguments
        partial_func = partial(func, *args, **kwargs)

        # Apply the partial function to each item in parallel
        return self.map(partial_func, items)

    @handle_errors
    def apply_to_dataframe(
        self, df: pd.DataFrame, func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Apply a function to a DataFrame in parallel.

        Args:
            df: DataFrame to process.
            func: Function to apply.

        Returns:
            Processed DataFrame.
        """
        # Log processing information
        if self.verbose:
            logger.info(f"Processing DataFrame with {len(df)} rows using {self.n_jobs} processes")

        # Split DataFrame into chunks
        chunks = np.array_split(df, self.n_jobs)

        # Apply function to each chunk in parallel
        results = self.map(func, chunks)

        # Combine results
        result = pd.concat(results, ignore_index=True)

        # Log processing information
        if self.verbose:
            logger.info(f"Processed DataFrame with {len(df)} rows")

        return result