"""
Parallel processing utilities for Yemen Market Analysis.
"""
import os
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from core.decorators import error_handler, performance_tracker
from core.exceptions import ComputationError

logger = logging.getLogger(__name__)


@error_handler(fallback_value={})
@performance_tracker()
def parallel_process(
    items: List[Any],
    process_func: Callable[[Any, Any], Any],
    shared_data: Any = None,
    max_workers: Optional[int] = None,
    progress: bool = True,
    chunk_size: int = 1,
    timeout: Optional[float] = None,
    retry_failed: bool = True,
    max_retries: int = 2
) -> Dict[Any, Any]:
    """
    Process items in parallel with sophisticated error handling and retries.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each item
        shared_data: Data shared across all processes
        max_workers: Maximum number of workers
        progress: Whether to log progress
        chunk_size: Size of chunks for processing
        timeout: Timeout for each item
        retry_failed: Whether to retry failed items
        max_retries: Maximum number of retries
        
    Returns:
        Dictionary mapping items to results
    """
    if not items:
        return {}
    
    # Determine optimal number of workers
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = max(1, min(cpu_count - 1, len(items)))
    
    # Initialize results and failed items tracking
    results = {}
    failed_items = []
    retries = {}
    
    # Log start of parallel processing
    logger.info(f"Starting parallel processing of {len(items)} items with {max_workers} workers")
    start_time = time.time()
    
    # Create a wrapper function to handle shared data
    def worker(item):
        try:
            return item, process_func(item, shared_data)
        except Exception as e:
            logger.warning(f"Error processing item {item}: {str(e)}")
            return item, None
    
    # Process items in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(worker, item): item for item in items}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_item):
            try:
                item, result = future.result(timeout=timeout)
                
                if result is not None:
                    # Successfully processed
                    results[item] = result
                else:
                    # Failed to process
                    failed_items.append(item)
                    retries[item] = 0
            except Exception as e:
                # Future raised exception
                item = future_to_item[future]
                logger.warning(f"Future exception for item {item}: {str(e)}")
                failed_items.append(item)
                retries[item] = 0
            
            # Update progress
            completed += 1
            if progress and completed % max(1, len(items) // 10) == 0:
                elapsed = time.time() - start_time
                percent = (completed / len(items)) * 100
                logger.info(f"Progress: {percent:.1f}% ({completed}/{len(items)}) - {elapsed:.1f}s elapsed")
    
    # Retry failed items if needed
    if retry_failed and failed_items:
        retry_round = 0
        while retry_round < max_retries and failed_items:
            retry_round += 1
            retry_items = failed_items.copy()
            failed_items = []
            
            logger.info(f"Retry round {retry_round}: Retrying {len(retry_items)} failed items")
            
            # Process retries
            for item in retry_items:
                try:
                    retries[item] += 1
                    result = process_func(item, shared_data)
                    
                    if result is not None:
                        results[item] = result
                    else:
                        failed_items.append(item)
                except Exception as e:
                    logger.warning(f"Retry {retry_round} failed for item {item}: {str(e)}")
                    failed_items.append(item)
    
    # Log completion stats
    elapsed = time.time() - start_time
    success_count = len(results)
    fail_count = len(failed_items)
    logger.info(
        f"Parallel processing completed in {elapsed:.1f}s - "
        f"{success_count} succeeded, {fail_count} failed"
    )
    
    # Add metadata to results
    results_with_meta = {
        "data": results,
        "metadata": {
            "total_items": len(items),
            "success_count": success_count,
            "fail_count": fail_count,
            "failed_items": failed_items,
            "elapsed_time": elapsed
        }
    }
    
    return results_with_meta


class ParallelProcessor:
    """Class-based parallel processor for more complex workflows."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        progress: bool = True,
        timeout: Optional[float] = None,
        retry_failed: bool = True,
        max_retries: int = 2
    ):
        """Initialize the parallel processor with settings."""
        self.max_workers = max_workers
        self.progress = progress
        self.timeout = timeout
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        
        # Determine optimal number of workers
        if self.max_workers is None:
            cpu_count = os.cpu_count() or 4
            self.max_workers = max(1, cpu_count - 1)
    
    @error_handler(fallback_value={})
    @performance_tracker()
    def process(
        self,
        items: List[Any],
        process_func: Callable[[Any, Any], Any],
        shared_data: Any = None
    ) -> Dict[Any, Any]:
        """Process items in parallel."""
        return parallel_process(
            items=items,
            process_func=process_func,
            shared_data=shared_data,
            max_workers=self.max_workers,
            progress=self.progress,
            timeout=self.timeout,
            retry_failed=self.retry_failed,
            max_retries=self.max_retries
        )
    
    @error_handler(fallback_value={})
    @performance_tracker()
    def map(
        self,
        func: Callable[[Any], Any],
        items: List[Any]
    ) -> List[Any]:
        """Simple parallel map operation."""
        if not items:
            return []
        
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(func, item) for item in items]
            
            # Collect results in order
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Error in parallel map: {str(e)}")
                    results.append(None)
        
        return results