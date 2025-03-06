# Threshold Cointegration Analysis Guide

This guide explains how to use the threshold cointegration modules for analyzing market integration with transaction costs in Yemen.

## Table of Contents

1. [Introduction](#introduction)
2. [ThresholdCointegration Class](#thresholdcointegration-class)
3. [ThresholdVECM Class](#thresholdvecm-class)
4. [Asymmetric Adjustment Analysis](#asymmetric-adjustment-analysis)
5. [Examples](#examples)

## Introduction

Threshold cointegration models identify different adjustment regimes in price transmission between markets with transaction costs.

## ThresholdCointegration Class

Simple two-price series implementation using the Engle-Granger approach with threshold effect.

### Basic Usage

```python
from src.models.threshold import ThresholdCointegration, calculate_asymmetric_adjustment

# Initialize and run complete analysis
model = ThresholdCointegration(north_prices, south_prices)
coint_results = model.estimate_cointegration()
threshold_results = model.estimate_threshold()
tvecm_results = model.estimate_tvecm()
adjustment = calculate_asymmetric_adjustment(tvecm_results)

# Print key results
print(f"Cointegration relationship: y = {coint_results['beta0']:.4f} + {coint_results['beta1']:.4f} * x")
print(f"Threshold: {threshold_results['threshold']:.4f}")
print(f"Half-life below threshold: {adjustment['half_life_below_1']:.2f} periods")
print(f"Half-life above threshold: {adjustment['half_life_above_1']:.2f} periods")
```

### Interpreting Results

- **Threshold Value**: Defines two regimes:
  - Below threshold: Price differential < transaction costs
  - Above threshold: Price differential > transaction costs

- **Adjustment Speeds**: Should be faster above threshold when arbitrage is profitable

## ThresholdVECM Class

More sophisticated implementation based on Hansen & Seo (2002) for multi-variable systems.

### Basic Usage

```python
from src.models.threshold_vecm import ThresholdVECM, calculate_half_lives
import pandas as pd

# Create DataFrame with price series
data = pd.DataFrame({
    'north_price': north_prices,
    'south_price': south_prices
})

# Initialize and run analysis
model = ThresholdVECM(data, k_ar_diff=2)
linear_results = model.estimate_linear_vecm()
threshold_results = model.grid_search_threshold()
tvecm_results = model.estimate_tvecm()
half_lives = calculate_half_lives(tvecm_results)

# Print key results
print(f"Threshold: {threshold_results['threshold']:.4f}")
print(f"Half-lives below threshold: {half_lives['below_regime']}")
print(f"Half-lives above threshold: {half_lives['above_regime']}")
```

## Example: North-South Price Transmission

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_market_data
from src.models.threshold import ThresholdCointegration

# Load data
df = load_market_data('unified_data.geojson')

# Get wheat prices for north and south
north_df = df[df['exchange_rate_regime'] == 'north']
south_df = df[df['exchange_rate_regime'] == 'south']

north_wheat = north_df[north_df['commodity'] == 'wheat'].groupby('date')['price'].mean()
south_wheat = south_df[south_df['commodity'] == 'wheat'].groupby('date')['price'].mean()

# Ensure dates align
common_dates = north_wheat.index.intersection(south_wheat.index)
north_wheat = north_wheat.loc[common_dates]
south_wheat = south_wheat.loc[common_dates]

# Run threshold cointegration analysis
model = ThresholdCointegration(north_wheat, south_wheat)
coint_results = model.estimate_cointegration()
threshold_results = model.estimate_threshold()
tvecm_results = model.estimate_tvecm()

# Print results
print(f"Threshold: {threshold_results['threshold']:.2f}")
print(f"North market adjustment below threshold: {tvecm_results['adjustment_below_1']:.4f}")
print(f"North market adjustment above threshold: {tvecm_results['adjustment_above_1']:.4f}")
```

For more detailed examples, see the test files in `tests/test_threshold.py`.