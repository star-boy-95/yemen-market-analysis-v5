# Yemen Market Integration Project: Econometrics API Reference

This document provides detailed reference information for the econometric model components implemented in the Yemen Market Integration project. These components provide the core statistical functionality for analyzing market integration in Yemen, including unit root testing, cointegration analysis, threshold models, and diagnostic testing.

## Table of Contents

1. [Unit Root Testing](#unit-root-testing)
2. [Cointegration Analysis](#cointegration-analysis)
3. [Threshold Models](#threshold-models)
4. [Threshold VECM](#threshold-vecm)
5. [Model Diagnostics](#model-diagnostics)

## Unit Root Testing <a id="unit-root-testing"></a>

The unit root testing module provides functionality to test for stationarity and determine the integration order of time series data.

### Class: `UnitRootTester`

```python
from src.models.unit_root import UnitRootTester

tester = UnitRootTester()
```

#### Methods

##### `test_adf`

Performs the Augmented Dickey-Fuller test for unit roots.

```python
result = tester.test_adf(
    series,                # Time series to test
    regression='c',        # Regression type: 'c' (constant), 'ct' (constant & trend), 'nc' (none)
    lags=None,             # Number of lags (None for automatic selection)
    alpha=0.05             # Significance level
)
```

**Returns:**
Dictionary with test results:

- `'test_statistic'`: ADF test statistic
- `'critical_value'`: Critical value at specified alpha
- `'p_value'`: p-value of the test
- `'stationary'`: Boolean indicating stationarity
- `'lags'`: Number of lags used

##### `test_zivot_andrews`

Performs the Zivot-Andrews test for unit roots with a single structural break.

```python
result = tester.test_zivot_andrews(
    series,                # Time series to test
    model='both',          # Model type: 'trend', 'intercept', or 'both'
    lags=None,             # Number of lags (None for automatic selection)
    alpha=0.05             # Significance level
)
```

**Returns:**
Dictionary with test results:

- `'test_statistic'`: Zivot-Andrews test statistic
- `'critical_value'`: Critical value at specified alpha
- `'p_value'`: p-value of the test
- `'stationary'`: Boolean indicating stationarity
- `'break_point'`: Estimated break point
- `'break_date'`: Break date (if dates provided)
- `'lags'`: Number of lags used

##### `determine_integration_order`

Determines the integration order of a time series.

```python
order = tester.determine_integration_order(
    series,                # Time series to test
    max_order=2,           # Maximum integration order to consider
    regression='c',        # Regression type for ADF test
    alpha=0.05             # Significance level
)
```

**Returns:**
Integer indicating the integration order (0 for stationary, 1 for I(1), etc.)

### Class: `StructuralBreakTester`

```python
from src.models.unit_root import StructuralBreakTester

tester = StructuralBreakTester()
```

#### Methods

##### `test_gregory_hansen`

Tests for cointegration with a regime shift using the Gregory-Hansen approach.

```python
result = tester.test_gregory_hansen(
    y,                     # First time series
    x,                     # Second time series
    model='regime',        # Model type: 'level', 'trend', or 'regime'
    lags=None,             # Number of lags (None for automatic selection)
    alpha=0.05             # Significance level
)
```

**Returns:**
Dictionary with test results:

- `'test_statistic'`: Gregory-Hansen test statistic
- `'critical_value'`: Critical value at specified alpha
- `'cointegrated'`: Boolean indicating cointegration
- `'break_point'`: Estimated break point
- `'break_date'`: Break date (if dates provided)
- `'cointegrating_vector'`: Estimated cointegrating vector

## Cointegration Analysis <a id="cointegration-analysis"></a>

The cointegration module provides methods for testing and analyzing cointegration relationships between time series.

### Class: `CointegrationTester`

```python
from src.models.cointegration import CointegrationTester

tester = CointegrationTester()
```

#### Methods

##### `test_engle_granger`

Performs the Engle-Granger two-step test for cointegration.

```python
result = tester.test_engle_granger(
    y,                     # Dependent time series
    x,                     # Independent time series or matrix
    trend='c',             # Trend specification: 'c', 'ct', 'ctt', 'nc'
    alpha=0.05             # Significance level
)
```

**Returns:**
Dictionary with test results:

- `'test_statistic'`: Engle-Granger test statistic
- `'critical_value'`: Critical value at specified alpha
- `'p_value'`: p-value of the test
- `'cointegrated'`: Boolean indicating cointegration
- `'residuals'`: Residuals from cointegrating regression
- `'cointegrating_vector'`: Estimated cointegrating vector

##### `test_johansen`

Performs Johansen's procedure for testing cointegration in a multivariate system.

```python
result = tester.test_johansen(
    data,                  # Matrix of time series (columns are variables)
    det_order=0,           # Deterministic term specification (0-5)
    k_ar=2,                # Lag order
    alpha=0.05             # Significance level
)
```

**Returns:**
Dictionary with test results:

- `'trace_stats'`: Trace test statistics
- `'max_eig_stats'`: Maximum eigenvalue test statistics
- `'trace_crit'`: Critical values for trace test
- `'max_eig_crit'`: Critical values for maximum eigenvalue test
- `'n_cointegrating'`: Number of cointegrating relationships
- `'eigenvectors'`: Cointegrating vectors (eigenvectors)
- `'eigenvalues'`: Eigenvalues

##### `estimate_vecm`

Estimates a Vector Error Correction Model (VECM).

```python
result = tester.estimate_vecm(
    data,                  # Matrix of time series
    n_cointegrating=1,     # Number of cointegrating relationships
    k_ar=2,                # Lag order
    det_order=0            # Deterministic term specification
)
```

**Returns:**
Dictionary with VECM estimation results:

- `'alpha'`: Adjustment speed parameters
- `'beta'`: Cointegrating vectors
- `'gamma'`: Short-run parameters
- `'residuals'`: Residuals from the VECM
- `'fitted'`: Fitted values
- `'aic'`: Akaike Information Criterion
- `'bic'`: Bayesian Information Criterion

## Threshold Models <a id="threshold-models"></a>

The threshold module implements threshold autoregressive (TAR) and momentum-TAR (M-TAR) models for analyzing asymmetric adjustments.

### Class: `ThresholdCointegration`

```python
from src.models.threshold import ThresholdCointegration

model = ThresholdCointegration(
    series1,               # First time series
    series2,               # Second time series
    market1_name="Market1",  # Optional name for first market
    market2_name="Market2"   # Optional name for second market
)
```

#### Methods

##### `estimate_cointegration`

Estimates the cointegration relationship between the two series.

```python
result = model.estimate_cointegration(
    trend='c',             # Trend specification: 'c', 'ct', 'ctt', 'nc'
    alpha=0.05             # Significance level
)
```

**Returns:**
Dictionary with cointegration test results:

- `'test_statistic'`: Cointegration test statistic
- `'critical_value'`: Critical value at specified alpha
- `'p_value'`: p-value of the test
- `'cointegrated'`: Boolean indicating cointegration
- `'beta'`: Cointegrating vector coefficient

##### `estimate_threshold`

Estimates the threshold parameter for TAR model via grid search.

```python
result = model.estimate_threshold(
    threshold_min=None,    # Minimum threshold value (None for auto)
    threshold_max=None,    # Maximum threshold value (None for auto)
    n_thresh=100,          # Number of threshold values to test
    trimming=0.15          # Trimming percentage for threshold range
)
```

**Returns:**
Dictionary with threshold estimation results:

- `'threshold'`: Estimated optimal threshold
- `'significant'`: Boolean indicating threshold significance
- `'threshold_min'`: Minimum threshold value tested
- `'threshold_max'`: Maximum threshold value tested
- `'aic_values'`: AIC values for each threshold
- `'residual_var'`: Residual variance for each threshold

##### `estimate_tar`

Estimates the Threshold Autoregressive (TAR) model with the estimated threshold.

```python
result = model.estimate_tar(
    lags=1                 # Number of lags in the TAR model
)
```

**Returns:**
Dictionary with TAR model results:

- `'asymmetric'`: Boolean indicating asymmetric adjustment
- `'adjustment_below'`: Adjustment coefficient for below-threshold regime
- `'adjustment_middle'`: Adjustment coefficient for middle regime
- `'adjustment_above'`: Adjustment coefficient for above-threshold regime
- `'model_below'`: Regression model for below-threshold regime
- `'model_middle'`: Regression model for middle regime
- `'model_above'`: Regression model for above-threshold regime

##### `estimate_mtar`

Estimates the Momentum-Threshold Autoregressive (M-TAR) model.

```python
result = model.estimate_mtar(
    lags=1                 # Number of lags in the M-TAR model
)
```

**Returns:**
Dictionary with M-TAR model results:

- `'asymmetric'`: Boolean indicating asymmetric adjustment
- `'adjustment_negative'`: Adjustment coefficient for negative changes
- `'adjustment_positive'`: Adjustment coefficient for positive changes
- `'model_negative'`: Regression model for negative momentum regime
- `'model_positive'`: Regression model for positive momentum regime

##### `calculate_half_lives`

Calculates the half-life of price deviations in each regime.

```python
half_lives = model.calculate_half_lives()
```

**Returns:**
Dictionary with half-life values:

- `'half_life_below'`: Half-life in below-threshold regime
- `'half_life_middle'`: Half-life in middle regime
- `'half_life_above'`: Half-life in above-threshold regime

##### `predict_adjustment`

Predicts the price adjustment given current prices.

```python
adjustment = model.predict_adjustment(
    price1,                # Current price in first market
    price2                 # Current price in second market
)
```

**Returns:**
Float representing the predicted price adjustment in the first market.

## Threshold VECM <a id="threshold-vecm"></a>

The threshold VECM module implements multivariate threshold cointegration models.

### Class: `ThresholdVECM`

```python
from src.models.threshold_vecm import ThresholdVECM

model = ThresholdVECM(
    data,                  # Matrix of time series (variables in columns)
    k_ar_diff=2,           # Number of lags in differenced terms
    dates=None,            # Optional array of dates
    variable_names=None    # Optional variable names
)
```

#### Methods

##### `estimate_linear_vecm`

Estimates a linear VECM (no threshold).

```python
result = model.estimate_linear_vecm(
    r=1,                   # Number of cointegrating relationships
    deterministic='co'     # Deterministic specification
)
```

**Returns:**
Dictionary with linear VECM results:

- `'alpha'`: Adjustment speed parameters
- `'beta'`: Cointegrating vectors
- `'gamma'`: Short-run parameters
- `'aic'`: Akaike Information Criterion
- `'bic'`: Bayesian Information Criterion
- `'residuals'`: Model residuals

##### `grid_search_threshold`

Performs grid search to find the optimal threshold.

```python
result = model.grid_search_threshold(
    beta=None,             # Optional cointegrating vector (None to use estimated)
    threshold_range=None,  # Range of threshold values to search
    n_thresh=100,          # Number of threshold values to consider
    trimming=0.15          # Trimming percentage
)
```

**Returns:**
Dictionary with threshold search results:

- `'threshold'`: Optimal threshold value
- `'likelihood_ratio'`: Likelihood ratio test statistic
- `'p_value'`: p-value from bootstrap distribution
- `'significant'`: Boolean indicating threshold significance
- `'beta'`: Cointegrating vector

##### `estimate_tvecm`

Estimates the Threshold Vector Error Correction Model.

```python
result = model.estimate_tvecm(
    threshold=None,        # Threshold value (None to use estimated)
    beta=None,             # Cointegrating vector (None to use estimated)
    r=1                    # Number of cointegrating relationships
)
```

**Returns:**
Dictionary with TVECM results:

- `'alpha_below'`: Adjustment speeds in lower regime
- `'alpha_above'`: Adjustment speeds in upper regime
- `'gamma_below'`: Short-run parameters in lower regime
- `'gamma_above'`: Short-run parameters in upper regime
- `'threshold'`: Threshold value used
- `'beta'`: Cointegrating vector used
- `'residuals'`: Model residuals

##### `calculate_half_lives`

Calculates half-lives of price deviations in each regime.

```python
half_lives = model.calculate_half_lives()
```

**Returns:**
Dictionary with half-life values for each variable in each regime.

## Model Diagnostics <a id="model-diagnostics"></a>

The model diagnostics module provides functions for validating econometric models.

### Class: `ModelDiagnostics`

```python
from src.models.diagnostics import ModelDiagnostics

diagnostics = ModelDiagnostics(
    residuals,             # Residuals from estimated model
    X=None                 # Optional design matrix
)
```

#### Methods

##### `residual_tests`

Runs a suite of residual diagnostic tests.

```python
results = diagnostics.residual_tests()
```

**Returns:**
Dictionary with test results:

- `'normality'`: Jarque-Bera test for normality
- `'autocorrelation'`: Ljung-Box test for autocorrelation
- `'heteroskedasticity'`: White's test for heteroskedasticity (if X provided)

##### `model_selection_criteria`

Calculates model selection criteria.

```python
criteria = diagnostics.model_selection_criteria(
    observed,              # Observed values
    predicted,             # Predicted values from model
    n_params               # Number of model parameters
)
```

**Returns:**
Dictionary with selection criteria:

- `'aic'`: Akaike Information Criterion
- `'bic'`: Bayesian Information Criterion
- `'hqic'`: Hannan-Quinn Information Criterion
- `'r_squared'`: R-squared
- `'adj_r_squared'`: Adjusted R-squared
- `'rmse'`: Root Mean Square Error

##### `parameter_stability`

Tests for parameter stability across subsamples.

```python
result = diagnostics.parameter_stability(
    breakpoint=None,       # Break point (None for middle of sample)
    test_type='chow'       # Test type: 'chow', 'quandt', or 'nyblom'
)
```

**Returns:**
Dictionary with stability test results:

- `'test_statistic'`: Test statistic
- `'p_value'`: p-value
- `'stable'`: Boolean indicating parameter stability
- `'breakpoint'`: Breakpoint used or identified

##### `summary`

Generates a comprehensive model diagnostic summary.

```python
summary = diagnostics.summary(
    observed,              # Observed values
    predicted,             # Predicted values
    n_params               # Number of model parameters
)
```

**Returns:**
Dictionary with comprehensive diagnostics:

- `'residual_tests'`: Results of residual diagnostic tests
- `'fit_statistics'`: Model fit statistics
- `'parameter_stability'`: Parameter stability test results (if X provided)
- `'overall_assessment'`: Text summary of model adequacy

### Standalone Functions

#### `calculate_fit_statistics`

Calculates model fit statistics.

```python
from src.models.diagnostics import calculate_fit_statistics

stats = calculate_fit_statistics(
    observed,              # Observed values
    predicted,             # Predicted values
    n_params               # Number of model parameters
)
```

**Returns:**
Dictionary with fit statistics (same as `model_selection_criteria`).

#### `test_residual_normality`

Tests residuals for normality using the Jarque-Bera test.

```python
from src.models.diagnostics import test_residual_normality

result = test_residual_normality(residuals)
```

**Returns:**
Dictionary with test results:

- `'test_statistic'`: Jarque-Bera test statistic
- `'p_value'`: p-value
- `'normal'`: Boolean indicating normality

#### `test_residual_autocorrelation`

Tests residuals for autocorrelation using the Ljung-Box test.

```python
from src.models.diagnostics import test_residual_autocorrelation

result = test_residual_autocorrelation(
    residuals,             # Model residuals
    lags=10                # Number of lags to test
)
```

**Returns:**
Dictionary with test results:

- `'test_statistic'`: Ljung-Box Q statistic
- `'p_value'`: p-value
- `'autocorrelated'`: Boolean indicating autocorrelation

#### `test_heteroskedasticity`

Tests residuals for heteroskedasticity using White's test.

```python
from src.models.diagnostics import test_heteroskedasticity

result = test_heteroskedasticity(
    residuals,             # Model residuals
    X                      # Design matrix
)
```

**Returns:**
Dictionary with test results:

- `'test_statistic'`: White's test statistic
- `'p_value'`: p-value
- `'heteroskedastic'`: Boolean indicating heteroskedasticity

#### `test_parameter_stability`

Tests for parameter stability using the Chow test.

```python
from src.models.diagnostics import test_parameter_stability

result = test_parameter_stability(
    y,                     # Dependent variable
    X,                     # Design matrix
    breakpoint,            # Breakpoint index
    test_type='chow'       # Test type: 'chow', 'quandt', or 'nyblom'
)
```

**Returns:**
Dictionary with test results:

- `'test_statistic'`: Test statistic
- `'p_value'`: p-value
- `'stable'`: Boolean indicating parameter stability
