# Yemen Market Integration Project: Coding Standards

This document outlines the coding standards and practices required for the Yemen Market Integration Project, with special emphasis on utility usage, code organization, and performance.

## Utility Usage Requirements

### 1. Always Use Project Utilities

Never reimplement functionality that's already available in the `src.utils` package.

```python
# CORRECT
from src.utils import read_csv, clean_column_names
df = read_csv('data.csv')
df = clean_column_names(df)

# INCORRECT
import pandas as pd
df = pd.read_csv('data.csv')  # Missing error handling
df.columns = [col.lower().replace(' ', '_') for col in df.columns]  # Reimplementing
```

### 2. Error Handling is Mandatory

Use the `handle_errors` decorator on all functions that perform I/O operations or complex computations.

```python
from src.utils.error_handler import handle_errors
import logging

logger = logging.getLogger(__name__)

@handle_errors(logger=logger, error_type=(FileNotFoundError, ValueError))
def process_market_data(file_path):
    # Implementation with proper error handling
```

### 3. Input Validation Before Processing

Always validate inputs before processing to fail early and provide clear error messages.

```python
from src.utils import validate_dataframe, raise_if_invalid

def analyze_market_data(df):
    # Validate inputs
    valid, errors = validate_dataframe(
        df, 
        required_columns=['date', 'price', 'market'],
        column_types={'price': float}
    )
    raise_if_invalid(valid, errors, "Invalid market data format")
    
    # Continue with validated data
```

### 4. Use Configuration System

Access all configurable parameters through the config system, not hard-coded values.

```python
# CORRECT
from src.utils import config
threshold = config.get('analysis.threshold', 0.05)  # With default

# INCORRECT
threshold = 0.05  # Hard-coded value
```

### 5. Specialized Time Series Utilities

For time series analysis, use the project's specialized functions.

```python
from src.utils import test_stationarity, test_cointegration

# Test before modeling
result = test_stationarity(price_series)
if result['stationary']:
    # Proceed with stationary series analysis
else:
    # Difference or transform the series
```

## Code Organization Rules

### 1. Function Length Limit

Functions must not exceed 25 lines of code (excluding docstrings and comments).

```python
# GOOD: Focused function doing one thing
def calculate_price_index(prices, base_period):
    """Calculate price index relative to base period."""
    base_price = prices[base_period]
    return prices / base_price * 100

# BAD: Too long, doing multiple things
def process_prices(prices, base_period, adjustments):
    # 30+ lines of code doing multiple operations
```

### 2. Module Length Limit

Modules must not exceed 300 lines of code. Split larger modules into focused components.

### 3. Argument Limit

Functions must accept a maximum of 5 arguments. Use dictionaries or config objects for additional parameters.

```python
# GOOD: Limited arguments
def analyze_price_transmission(north_prices, south_prices, lag=1):
    # Implementation

# BAD: Too many arguments
def analyze_price_transmission(north_prices, south_prices, lag, alpha, method, trim, max_iter, tol):
    # Implementation

# BETTER: Using config for additional parameters
def analyze_price_transmission(north_prices, south_prices, params=None):
    """
    Analyze price transmission between markets.
    
    Parameters
    ----------
    north_prices : array-like
        Prices in northern markets
    south_prices : array-like
        Prices in southern markets
    params : dict, optional
        Additional parameters including:
        - lag: int
        - alpha: float
        - method: str
        - trim: float
    """
    # Get parameters from config or use defaults
    params = params or {}
    lag = params.get('lag', 1)
    alpha = params.get('alpha', 0.05)
    # Implementation
```

### 4. Composition Over Inheritance

Use function composition rather than complex class hierarchies.

```python
# GOOD: Functional composition
def preprocess(data):
    """Preprocess raw data."""
    return clean_data(data)

def analyze(data):
    """Analyze preprocessed data."""
    return fit_model(data)

def process_pipeline(raw_data):
    """Full processing pipeline."""
    preprocessed = preprocess(raw_data)
    return analyze(preprocessed)

# AVOID: Deep inheritance hierarchies
class BaseProcessor:
    # Implementation
    
class DataCleaner(BaseProcessor):
    # Implementation
    
class MarketAnalyzer(DataCleaner):
    # Implementation
```

### 5. Separation of Concerns

Each module should focus on a single aspect of functionality:

- `loaders.py`: Data loading and initial parsing
- `processors.py`: Data cleaning and transformation
- `models.py`: Statistical modeling and analysis
- `visualizers.py`: Data visualization

## Performance Requirements

### 1. M1 Optimization

Apply the `@m1_optimized` decorator to all computation-heavy functions.

```python
from src.utils.decorators import m1_optimized

@m1_optimized(use_numba=True, parallel=True)
def calculate_price_correlations(price_matrix):
    """Calculate pairwise correlations between price series."""
    # Computationally intensive implementation
```

### 2. Process Large Data in Chunks

Use chunked processing for all large files (>100MB).

```python
from src.utils.file_utils import read_large_csv_chunks

def process_large_dataset(file_path):
    results = []
    for chunk in read_large_csv_chunks(file_path, chunk_size=10000):
        # Process each chunk
        processed = process_chunk(chunk)
        results.append(processed)
    
    # Combine results
    return pd.concat(results)
```

### 3. Memory Usage Optimization

Use memory optimization techniques for large datasets.

```python
from src.utils.performance_utils import optimize_dataframe, memory_usage_decorator

@memory_usage_decorator
def process_large_dataframe(df):
    # Optimize memory usage first
    df = optimize_dataframe(df, downcast=True, category_min_size=50)
    
    # Process the optimized dataframe
    # ...
```

### 4. Parallel Processing

Use parallel processing for independent operations.

```python
from src.utils.performance_utils import parallelize_dataframe

def analyze_markets(df):
    def process_market(market_df):
        # Process a single market
        return result_df
    
    # Process markets in parallel
    return parallelize_dataframe(df, process_market, n_workers=4)
```

## Documentation Requirements

### 1. NumPy-Style Docstrings

Use NumPy-style docstrings for all functions and classes.

```python
def calculate_price_index(prices, base_period):
    """
    Calculate price index relative to base period.
    
    Parameters
    ----------
    prices : array-like
        Time series of prices
    base_period : int
        Index of the base period
        
    Returns
    -------
    array-like
        Price index values
    
    Examples
    --------
    >>> prices = np.array([100, 102, 105, 108])
    >>> calculate_price_index(prices, 0)
    array([100., 102., 105., 108.])
    """
```

### 2. Module-Level Docstrings

Include a descriptive docstring at the top of each module.

```python
"""
Market integration analysis module for the Yemen Market Integration Project.

This module provides functions for analyzing price transmission between markets
using threshold vector error correction models.
"""
```

### 3. Type Hints

Use type hints for function arguments and return values.

```python
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

def analyze_cointegration(
    series1: Union[pd.Series, np.ndarray],
    series2: Union[pd.Series, np.ndarray],
    lag: int = 1
) -> Dict[str, Any]:
    """
    Analyze cointegration between two time series.
    
    Parameters
    ----------
    series1 : array-like
        First time series
    series2 : array-like
        Second time series
    lag : int, optional
        Lag order
        
    Returns
    -------
    dict
        Dictionary of test results
    """
```

## Testing Standards

### 1. Unit Tests for All Utilities

Every utility function must have corresponding unit tests.

```python
# In test_data_utils.py
def test_clean_column_names():
    """Test column name cleaning function."""
    df = pd.DataFrame({'Raw Column': [1, 2], 'Another-Column': [3, 4]})
    result = clean_column_names(df)
    assert 'raw_column' in result.columns
    assert 'another_column' in result.columns
```

### 2. Integration Tests for Workflows

Create integration tests for end-to-end workflows.

```python
def test_price_transmission_workflow():
    """Test the full price transmission analysis workflow."""
    # Load test data
    north_data = read_csv('tests/data/north_prices.csv')
    south_data = read_csv('tests/data/south_prices.csv')
    
    # Run workflow
    result = analyze_price_transmission(north_data, south_data)
    
    # Assert expected outcomes
    assert 'threshold' in result
    assert 'cointegrated' in result
    assert isinstance(result['half_life'], float)
```

### 3. Performance Tests

Include tests for performance on large datasets.

```python
@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance on large dataset."""
    # Generate large test dataset
    large_df = generate_large_test_df(rows=100000)
    
    # Measure execution time
    start_time = time.time()
    result = process_large_dataframe(large_df)
    execution_time = time.time() - start_time
    
    # Assert performance expectations
    assert execution_time < 10.0  # Should complete in less than 10 seconds
```

## Logging Standards

### 1. Module-Level Loggers

Each module should define its own logger.

```python
import logging

logger = logging.getLogger(__name__)

def process_data():
    logger.info("Starting data processing")
    # Implementation
    logger.debug("Processed %d records", len(records))
```

### 2. Contextual Logging

Add context to log messages for better traceability.

```python
from src.utils.logging_setup import get_logger_with_context

def process_market_data(market, commodity):
    # Create logger with context
    logger = get_logger_with_context(
        __name__, 
        {'market': market, 'commodity': commodity}
    )
    
    logger.info("Processing market data")  # Will include market and commodity
    # Implementation
```

### 3. Appropriate Log Levels

Use appropriate log levels:
- ERROR: Exception conditions
- WARNING: Unusual conditions but not exceptions
- INFO: General operations
- DEBUG: Detailed information for troubleshooting

## Code Review Checklist

Before submitting code for review, ensure:

- [ ] All functions use appropriate utility functions
- [ ] Error handling is implemented for all I/O and computation
- [ ] Input validation is present
- [ ] Functions are short and focused
- [ ] Types are properly hinted
- [ ] Docstrings are complete
- [ ] Unit tests exist
- [ ] Logging is appropriate
- [ ] M1 optimization is applied where appropriate
- [ ] Large data is processed in chunks
- [ ] All configuration uses the config module
- [ ] Code meets performance requirements

Refer to `utils_guide.md` for full utility documentation and examples.