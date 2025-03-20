# Cointegration Module Implementation Outline

This document provides a detailed implementation outline for the `cointegration.py` module, which is a critical component of the Yemen Market Integration project. This implementation will follow the mathematical specifications outlined in the econometric_methods.md document.

## Overview

The cointegration module implements methods for testing and estimating long-run equilibrium relationships between price series across different markets. It is a core component for market integration analysis, providing the foundation for threshold models and policy simulations.

## Class Structure

```python
class CointegrationTester:
    """
    Test for cointegration between time series using various methods.
    
    This class provides tools for testing cointegration relationships between
    price series from different markets, with specialized methods for
    conflict-affected settings like Yemen.
    """
    
    def __init__(self):
        """Initialize the cointegration tester."""
        # Configuration and setup logic
        
    def test_engle_granger(self, y, x, trend='c', max_lags=None, alpha=0.05):
        """
        Test for cointegration using the Engle-Granger two-step method.
        
        Parameters
        ----------
        y : array_like
            Dependent variable (e.g., price series from one market)
        x : array_like
            Independent variable(s) (e.g., price series from another market)
        trend : str, optional
            Deterministic trend specification ('n', 'c', 'ct')
        max_lags : int, optional
            Maximum number of lags for ADF test on residuals
        alpha : float, optional
            Significance level
            
        Returns
        -------
        dict
            Cointegration test results including:
            - cointegrated: bool, whether series are cointegrated
            - statistic: ADF test statistic
            - p_value: p-value of the test
            - critical_values: critical values at different significance levels
            - beta: cointegrating vector (if cointegrated)
            - half_life: half-life of deviations (if cointegrated)
        """
        # Implementation
        
    def test_johansen(self, data, det_order=1, k_ar_diff=2, alpha=0.05):
        """
        Test for cointegration in a multivariate system using Johansen procedure.
        
        Parameters
        ----------
        data : array_like
            Multivariate time series data (e.g., prices from multiple markets)
        det_order : int, optional
            Deterministic trend specification:
            - 0: no deterministic terms
            - 1: constant term
            - 2: constant and trend
        k_ar_diff : int, optional
            Number of lagged differences in the VECM
        alpha : float, optional
            Significance level
            
        Returns
        -------
        dict
            Cointegration test results including:
            - rank: int, estimated cointegration rank
            - trace_stat: trace statistics
            - max_eig_stat: maximum eigenvalue statistics
            - critical_values: critical values
            - eigenvectors: cointegrating vectors (if cointegrated)
            - eigenvalues: eigenvalues from decomposition
        """
        # Implementation
        
    def test_combined(self, y, x, method='engle-granger', **kwargs):
        """
        Run multiple cointegration tests and combine results.
        
        Parameters
        ----------
        y : array_like
            Dependent variable
        x : array_like
            Independent variable(s)
        method : str, optional
            Primary testing method ('engle-granger', 'johansen')
        **kwargs : dict
            Additional arguments passed to specific test methods
            
        Returns
        -------
        dict
            Combined test results including:
            - cointegrated: bool, whether series are cointegrated
            - method: method used for primary test
            - test_results: full results from individual tests
            - summary: summarized interpretation
        """
        # Implementation
```

## Function Definitions

### 1. Estimation Functions

```python
def estimate_cointegration_vector(y, x, method='ols', trend='c'):
    """
    Estimate cointegrating relationship between time series.
    
    Parameters
    ----------
    y : array_like
        Dependent variable (e.g., price series from one market)
    x : array_like
        Independent variable(s) (e.g., price series from another market)
    method : str, optional
        Estimation method ('ols', 'fm-ols', 'ccr')
    trend : str, optional
        Deterministic trend specification ('n', 'c', 'ct')
        
    Returns
    -------
    dict
        Estimation results including:
        - beta: cointegrating vector
        - t_stats: t-statistics for coefficients
        - p_values: p-values for coefficients
        - r_squared: goodness of fit
        - std_err: standard errors for coefficients
        - residuals: residuals from cointegrating regression
    """
    # Implementation
```

### 2. Diagnostic Functions

```python
def calculate_half_life(alpha, method='standard'):
    """
    Calculate half-life of deviations from long-run equilibrium.
    
    Parameters
    ----------
    alpha : float
        Adjustment speed coefficient from ECM/VECM
    method : str, optional
        Method for calculation ('standard', 'threshold')
        
    Returns
    -------
    float
        Half-life in time periods
    """
    # Implementation

def test_granger_causality(y, x, max_lags=4, alpha=0.05):
    """
    Test for Granger causality between cointegrated variables.
    
    Parameters
    ----------
    y : array_like
        First time series
    x : array_like
        Second time series
    max_lags : int, optional
        Maximum number of lags to include
    alpha : float, optional
        Significance level
        
    Returns
    -------
    dict
        Causality test results including:
        - y_causes_x: bool, whether y Granger-causes x
        - x_causes_y: bool, whether x Granger-causes y
        - optimal_lag: int, optimal lag length by AIC
        - f_stats: F-statistics for tests
        - p_values: p-values for tests
    """
    # Implementation
```

### 3. Helper Functions

```python
def _select_lag_length(data, max_lags=10, ic='aic'):
    """
    Select optimal lag length based on information criteria.
    
    Parameters
    ----------
    data : array_like
        Time series data
    max_lags : int, optional
        Maximum number of lags to test
    ic : str, optional
        Information criterion ('aic', 'bic', 'hqic')
        
    Returns
    -------
    int
        Optimal lag length
    """
    # Implementation

def _check_same_integration_order(series_list, max_order=2):
    """
    Check if all series have the same integration order.
    
    Parameters
    ----------
    series_list : list of array_like
        List of time series
    max_order : int, optional
        Maximum integration order to test
        
    Returns
    -------
    bool
        True if all series have the same integration order
    """
    # Implementation
```

## Implementation Dependencies

The implementation will rely on:

1. **Unit Root Testing**: Uses the `UnitRootTester` class from `unit_root.py` for testing stationarity of residuals.
2. **Validation Utilities**: Uses validation functions from `utils/validation.py` to ensure valid inputs.
3. **Diagnostics**: Depends on `diagnostics.py` for testing model adequacy.
4. **Memory Optimization**: Uses performance utilities to handle large datasets efficiently.

## Implementation Approaches

### Engle-Granger Implementation Strategy

1. Validate that series have same integration order using `UnitRootTester`
2. Estimate cointegrating relationship using OLS
3. Extract residuals
4. Test residuals for stationarity using ADF test
5. Calculate relevant statistics and metrics
6. Optional: Estimate Error Correction Model if cointegrated

### Johansen Implementation Strategy

1. Validate input data format and properties
2. Determine optimal lag length using information criteria
3. Construct VAR model and convert to VECM form
4. Perform eigenvalue decomposition
5. Calculate trace and maximum eigenvalue statistics
6. Compare with critical values to determine cointegration rank
7. Extract cointegrating vectors if cointegrated

## Error Handling and Performance Considerations

1. **Strict Input Validation**: Validate all time series for:
   - Length (minimum 30 observations)
   - No missing values
   - Same integration order
   - Same frequency

2. **Memory Efficiency**:
   - For large datasets, implement chunked processing
   - Use memory-optimized functions for matrix operations
   - Apply optimizations for M1/M2 hardware

3. **Error Handling**: 
   - Use the `handle_errors` decorator for consistent error management
   - Implement specific error messages for common issues
   - Provide fallback options when possible

4. **Performance**:
   - Cache intermediate results for reuse
   - Use parallel processing for bootstrap and simulation components
   - Implement vectorized operations where possible

## Testing and Validation

1. **Unit Tests**:
   - Test with known cointegrated and non-cointegrated series
   - Test with different trend specifications
   - Verify critical values match literature values

2. **Integration Tests**:
   - Test interaction with unit root testing module
   - Test with real Yemen market data
   - Validate results against established econometric software

3. **Performance Tests**:
   - Benchmark performance on large datasets
   - Measure memory usage for optimization
   - Test with both small and large dimensions

## Future Extensions

1. **Advanced Methods**:
   - Gregory-Hansen test for cointegration with structural breaks
   - Regime-switching cointegration models
   - Bounds testing approach for mixed integration orders

2. **Visualization**:
   - Integration with visualization module for cointegration plots
   - Interactive exploration of cointegrating relationships

3. **API Development**:
   - Simplified API for common use cases
   - Integration with spatial components for spatial cointegration
