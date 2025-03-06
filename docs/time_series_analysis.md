# Time Series Analysis Guide

This document provides guidance on using the unit root and cointegration testing modules for market integration analysis in Yemen.

## Table of Contents

1. [Introduction](#introduction)
2. [Unit Root Testing](#unit-root-testing)
3. [Cointegration Testing](#cointegration-testing)
4. [Recommended Workflow](#recommended-workflow)
5. [Examples](#examples)

## Introduction

Time series analysis is crucial for understanding market integration. Before we can analyze relationships between market prices, we need to determine if the series are stationary and if they share common trends (cointegration).

## Unit Root Testing

The `unit_root.py` module provides tools for testing stationarity of time series.

### Available Tests

- **Augmented Dickey-Fuller (ADF)**: Tests the null hypothesis that a time series has a unit root
- **KPSS Test**: Tests the null hypothesis that a time series is stationary
- **ADF-GLS Test**: More powerful variant of ADF test
- **Zivot-Andrews Test**: Tests for unit root with a structural break

### Basic Usage

```python
from src.models.unit_root import UnitRootTester, determine_integration_order

# Initialize the tester
tester = UnitRootTester()

# Test a time series for stationarity
result = tester.test_adf(price_series)
if result['stationary']:
    print("Series is stationary")
else:
    print("Series has a unit root (non-stationary)")
    
# Determine order of integration
order = determine_integration_order(price_series)
print(f"Series is integrated of order {order}")
```

### Test Result Interpretation

- **ADF/ADF-GLS**: Reject null hypothesis (stationary) if p-value < alpha (typically 0.05)
- **KPSS**: Fail to reject null hypothesis (stationary) if p-value > alpha
- **Zivot-Andrews**: Reject null hypothesis (unit root) if p-value < alpha
 
For market integration analysis, we typically want to confirm that price series are I(1) - integrated of order 1 (stationary after first difference).

## Cointegration Testing

The `cointegration.py` module provides tools for testing cointegration between time series.

### Available Tests

- **Engle-Granger Test**: Two-step approach for two series
- **Johansen Test**: System-based approach for multiple series

### Basic Usage

```python
from src.models.cointegration import CointegrationTester, calculate_half_life

# Initialize the tester
tester = CointegrationTester()

# Test for cointegration between two series
eg_result = tester.test_engle_granger(north_prices, south_prices)
if eg_result['cointegrated']:
    print("Series are cointegrated")
    
    # Calculate half-life of deviations
    half_life = calculate_half_life(eg_result['residuals'])
    print(f"Half-life of deviations: {half_life:.2f} periods")
else:
    print("Series are not cointegrated")
    
# Test for cointegration among multiple series
data = np.column_stack([series1, series2, series3])
jo_result = tester.test_johansen(data)
print(f"Number of cointegrating relations: {jo_result['rank_trace']}")
```

### Test Result Interpretation

- **Engle-Granger**: Series are cointegrated if residuals are stationary
- **Johansen**: The rank indicates the number of cointegrating relationships

## Recommended Workflow

1. **Check data quality**: Ensure no missing values, sufficient observations, etc.
2. **Test for stationarity**: Determine integration order of individual series
3. **Test for cointegration**: If series are I(1), test for cointegration
4. **Analyze adjustment dynamics**: For cointegrated series, analyze speed of adjustment

```python
import pandas as pd
from src.data import load_market_data
from src.models.unit_root import UnitRootTester, determine_integration_order
from src.models.cointegration import CointegrationTester, calculate_half_life

# Load data
df = load_market_data('unified_data.geojson')

# Extract time series for different markets
north_prices = df[df['exchange_rate_regime'] == 'north']['price']
south_prices = df[df['exchange_rate_regime'] == 'south']['price']

# 1. Test for stationarity
unit_root_tester = UnitRootTester()
north_order = determine_integration_order(north_prices)
south_order = determine_integration_order(south_prices)

# 2. Test for cointegration if both series are I(1)
if north_order == 1 and south_order == 1:
    cointegration_tester = CointegrationTester()
    result = cointegration_tester.test_engle_granger(north_prices, south_prices)
    
    # 3. Analyze dynamics if cointegrated
    if result['cointegrated']:
        half_life = calculate_half_life(result['residuals'])
        print(f"Markets are integrated with half-life of {half_life:.2f} periods")
    else:
        print("Markets are not integrated")
else:
    print("Cannot test for cointegration: series are not I(1)")
```

## Examples

### Example 1: Testing Market Integration between Regions

```python
import pandas as pd
import numpy as np
from src.data import load_market_data, get_time_series
from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester

# Load data
df = load_market_data('unified_data.geojson')

# Get price series for wheat in two regions
abyan_wheat = get_time_series(df, 'abyan', 'wheat')['price']
aden_wheat = get_time_series(df, 'aden', 'wheat')['price']

# Initialize testers
unit_root_tester = UnitRootTester()
cointegration_tester = CointegrationTester()

# Full workflow
print("Testing stationarity:")
abyan_result = unit_root_tester.run_all_tests(abyan_wheat)
aden_result = unit_root_tester.run_all_tests(aden_wheat)

# Concordance of unit root tests
abyan_stationary = [test['stationary'] for test in abyan_result.values()]
aden_stationary = [test['stationary'] for test in aden_result.values()]

if not any(abyan_stationary) and not any(aden_stationary):
    print("Both series appear to be non-stationary, testing for cointegration")
    
    # Test for cointegration
    coint_result = cointegration_tester.test_combined(abyan_wheat, aden_wheat)
    
    if coint_result['cointegrated']:
        print("Markets are integrated!")
        
        # Get cointegrating vector
        beta = coint_result['engle_granger']['beta']
        print(f"Cointegrating relationship: price_abyan = {beta[0]:.2f} + {beta[1]:.2f} * price_aden")
        
        # Calculate speed of adjustment
        residuals = coint_result['engle_granger']['residuals']
        half_life = calculate_half_life(residuals)
        print(f"Half-life of deviations: {half_life:.2f} months")
    else:
        print("Markets are not integrated despite similar price movements")
else:
    print("Cannot reliably test for cointegration - stationarity results are mixed")
```

### Example 2: Testing for Structural Breaks

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data import load_market_data
from src.models.unit_root import UnitRootTester

# Load data
df = load_market_data('unified_data.geojson')

# Get price series
price_series = df[df['commodity'] == 'wheat']['price']

# Test for structural break
tester = UnitRootTester()
za_result = tester.test_zivot_andrews(price_series)

if za_result['stationary']:
    print(f"Series is stationary when accounting for structural break")
    
    # Get break date
    break_index = za_result['breakpoint']
    break_date = price_series.index[break_index]
    
    print(f"Structural break detected at: {break_date}")
    
    # Visualize the break
    plt.figure(figsize=(10, 6))
    plt.plot(price_series)
    plt.axvline(x=break_date, color='red', linestyle='--')
    plt.title('Wheat Prices with Structural Break')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
else:
    print("No significant structural break detected")
```