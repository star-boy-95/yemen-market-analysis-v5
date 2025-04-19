# Error Handling Guide

This document provides guidance on how to handle errors in the Yemen Market Analysis project, with a particular focus on handling insufficient data.

## Insufficient Data Handling

The Yemen Market Analysis project implements various econometric models that require a minimum number of observations to produce reliable results. When data is insufficient, the models will raise a `YemenAnalysisError` exception with a clear error message instead of returning mock results.

### Minimum Data Requirements

Here are the minimum data requirements for each model:

| Model | Minimum Observations | Notes |
|-------|---------------------|-------|
| ADF Test | 4 | Augmented Dickey-Fuller test for unit roots |
| KPSS Test | 4 | Kwiatkowski-Phillips-Schmidt-Shin test for stationarity |
| Phillips-Perron Test | 4 | Phillips-Perron test for unit roots |
| Zivot-Andrews Test | 6 | Zivot-Andrews test for unit roots with structural breaks |
| DF-GLS Test | 8 | Dickey-Fuller GLS test for unit roots |
| Engle-Granger Test | 10 | Engle-Granger test for cointegration |
| Gregory-Hansen Test | 20 | Gregory-Hansen test for cointegration with structural breaks |
| Johansen Test | 12 | Johansen test for cointegration |
| TAR Model | 20 | Threshold Autoregressive model |
| M-TAR Model | 20 | Momentum Threshold Autoregressive model |
| TVECM | 30 | Threshold Vector Error Correction Model |

### Handling Exceptions

When working with the Yemen Market Analysis project, you should handle `YemenAnalysisError` exceptions appropriately. Here's an example of how to handle insufficient data exceptions:

```python
from src.utils.error_handling import YemenAnalysisError
from src.models.unit_root import UnitRootTester

def analyze_market_data(data):
    try:
        # Attempt to run unit root test
        tester = UnitRootTester()
        results = tester.test_adf(data, column='price')
        return results
    except YemenAnalysisError as e:
        # Handle the error appropriately
        print(f"Analysis error: {e}")
        # Log the error
        logger.warning(f"Could not analyze market data: {e}")
        # Return None or an empty result
        return None
```

### Best Practices for Handling Insufficient Data

1. **Validate data before analysis**: Check if your data meets the minimum requirements before attempting to run models.

2. **Provide clear error messages to users**: When catching `YemenAnalysisError` exceptions, provide clear and helpful error messages to users.

3. **Log errors**: Always log errors for debugging and monitoring purposes.

4. **Graceful degradation**: If a specific analysis fails due to insufficient data, try to continue with other analyses that can be performed.

5. **Alternative approaches**: Consider using simpler models or aggregating data when detailed analysis is not possible.

## Other Common Errors

### Data Quality Errors

The project includes validation checks for data quality. These checks may raise exceptions when:

- Data contains missing values
- Data has incorrect types
- Data contains outliers
- Data has inconsistent timestamps

### Computational Errors

Some models may encounter computational issues, such as:

- Convergence failures in optimization routines
- Singular matrices in regression models
- Numerical instability in threshold estimation

### File I/O Errors

When working with files, you may encounter:

- File not found errors
- Permission errors
- Incorrect file format errors

## Testing Error Handling

The project includes a test script (`test_insufficient_data.py`) that verifies the error handling behavior for insufficient data. Run this script to ensure that the models correctly raise exceptions when data is insufficient:

```bash
python test_insufficient_data.py
```

## Reporting Issues

If you encounter unexpected errors or have suggestions for improving error handling, please report them by:

1. Opening an issue on the project's GitHub repository
2. Including a detailed description of the error
3. Providing a minimal reproducible example
4. Attaching relevant logs or error messages
