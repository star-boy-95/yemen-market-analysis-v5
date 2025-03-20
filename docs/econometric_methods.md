# Yemen Market Integration Project: Econometric Methods

This document provides detailed mathematical specifications for the econometric methods implemented in the Yemen Market Integration project. It serves as a reference for understanding the underlying statistical techniques and their implementation.

## Table of Contents

1. [Unit Root Testing](#unit-root-testing)
2. [Cointegration Analysis](#cointegration-analysis)
3. [Threshold Models](#threshold-models)
4. [Threshold VECM](#threshold-vecm)
5. [Spatial Econometrics](#spatial-econometrics)
6. [Model Diagnostics](#model-diagnostics)
7. [Policy Simulation](#policy-simulation)

## Unit Root Testing

### Augmented Dickey-Fuller (ADF) Test

The ADF test examines the null hypothesis that a time series $y_t$ contains a unit root against the alternative that it is stationary.

**Model Specification:**
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

where:

- $\Delta y_t$ is the first difference of the series
- $\alpha$ is a constant
- $\beta t$ is a time trend
- $p$ is the lag order
- $\varepsilon_t$ is white noise

**Null Hypothesis:** $H_0: \gamma = 0$ (unit root present)  
**Alternative Hypothesis:** $H_1: \gamma < 0$ (stationary)

**Implementation:**

```python
from src.models.unit_root import UnitRootTester

tester = UnitRootTester()
result = tester.test_adf(series, regression='ct', lags=4)
```

### Zivot-Andrews Test

The Zivot-Andrews test allows for a single structural break in the series, addressing a limitation of the standard ADF test when structural breaks are present (common in conflict settings).

**Model Specification:**
$$\Delta y_t = \alpha + \beta t + \theta DU_t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

where:

- $DU_t$ is a dummy variable that captures the structural break at time $TB$
- $DU_t = 1$ if $t > TB$, and 0 otherwise

**Null Hypothesis:** $H_0: \gamma = 0$ (unit root present with no structural break)  
**Alternative Hypothesis:** $H_1: \gamma < 0$ (stationary with a structural break)

**Implementation:**

```python
from src.models.unit_root import UnitRootTester

tester = UnitRootTester()
result = tester.test_zivot_andrews(series)
```

### Integration Order Determination

The integration order $d$ of a time series indicates how many times the series needs to be differenced to achieve stationarity. This is important for cointegration analysis, as most tests require series to be integrated of the same order.

**Implementation:**

```python
from src.models.unit_root import UnitRootTester

tester = UnitRootTester()
order = tester.determine_integration_order(series, max_order=2)
```

## Cointegration Analysis

### Engle-Granger Two-Step Method

The Engle-Granger method tests for cointegration between two time series by:

1. Estimating the long-run equilibrium relationship
2. Testing if the residuals are stationary

**Model Specification:**

1. Long-run equilibrium: $y_t = \beta_0 + \beta_1 x_t + u_t$
2. Test stationarity of $u_t$ using ADF test

**Implementation:**

```python
from src.models.cointegration import CointegrationTester

tester = CointegrationTester()
result = tester.test_engle_granger(y_series, x_series)
```

### Johansen Cointegration Test

The Johansen test examines cointegration in a multivariate system, allowing for multiple cointegrating relationships.

**Model Specification:**
Vector Error Correction Model (VECM) form:
$$\Delta X_t = \Pi X_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta X_{t-i} + \varepsilon_t$$

where:

- $X_t$ is a vector of variables
- $\Pi$ is the long-run impact matrix
- $\Gamma_i$ captures short-run dynamics

Cointegration is determined by the rank of $\Pi$.

**Implementation:**

```python
from src.models.cointegration import CointegrationTester

tester = CointegrationTester()
result = tester.test_johansen(market_data)
```

### Gregory-Hansen Test

The Gregory-Hansen test extends cointegration testing to account for structural breaks, which is crucial for conflict-affected settings where regime changes may occur.

**Model Specification:**
$$y_t = \beta_0 + \beta_1 D_t + \beta_2 x_t + \beta_3 D_t x_t + u_t$$

where:

- $D_t$ is a dummy variable for the regime change
- $\beta_3$ captures the change in cointegrating relationship

**Implementation:**

```python
from src.models.unit_root import StructuralBreakTester

tester = StructuralBreakTester()
result = tester.test_gregory_hansen(y_series, x_series)
```

## Threshold Models

### Threshold Autoregressive (TAR) Model

The TAR model captures nonlinear dynamics by allowing the autoregressive process to follow different regimes depending on the value of the threshold variable.

**Model Specification:**
$$y_t =
\begin{cases}
\alpha_1 + \sum_{i=1}^{p} \beta_{1i} y_{t-i} + \varepsilon_{1t}, & \text{if } z_t \leq \gamma \\
\alpha_2 + \sum_{i=1}^{p} \beta_{2i} y_{t-i} + \varepsilon_{2t}, & \text{if } z_t > \gamma
\end{cases}$$

where:
- $z_t$ is the threshold variable
- $\gamma$ is the threshold parameter
- $p$ is the autoregressive order

**Implementation:**
```python
from src.models.threshold import ThresholdCointegration

model = ThresholdCointegration(y_series, x_series)
result = model.estimate_tar()
```

### Momentum-Threshold Autoregressive (M-TAR) Model

The M-TAR model extends the TAR model by allowing the threshold to depend on the rate of change (momentum) of the series, capturing directional asymmetry.

**Model Specification:**
$$\Delta y_t =
\begin{cases}
\alpha_1 + \rho_1 y_{t-1} + \sum_{i=1}^{p} \beta_{1i} \Delta y_{t-i} + \varepsilon_{1t}, & \text{if } \Delta y_{t-1} \leq \gamma \\
\alpha_2 + \rho_2 y_{t-1} + \sum_{i=1}^{p} \beta_{2i} \Delta y_{t-i} + \varepsilon_{2t}, & \text{if } \Delta y_{t-1} > \gamma
\end{cases}$$

**Implementation:**
```python
from src.models.threshold import ThresholdCointegration

model = ThresholdCointegration(y_series, x_series)
result = model.estimate_mtar()
```

### Threshold Cointegration

Threshold cointegration extends cointegration analysis to allow for different adjustment processes depending on the size of the deviation from equilibrium. This is particularly relevant for modeling transaction costs in market integration.

**Model Specification:**
$$\Delta y_t =
\begin{cases}
\alpha_1 (y_{t-1} - \beta x_{t-1}) + \varepsilon_{1t}, & \text{if } |y_{t-1} - \beta x_{t-1}| \leq \gamma \\
\alpha_2 (y_{t-1} - \beta x_{t-1}) + \varepsilon_{2t}, & \text{if } |y_{t-1} - \beta x_{t-1}| > \gamma
\end{cases}$$

where:
- $\gamma$ represents the transaction cost threshold
- $\alpha_1$ and $\alpha_2$ are adjustment speeds in different regimes

**Implementation:**
```python
from src.models.threshold import ThresholdCointegration

model = ThresholdCointegration(north_price, south_price)
model.estimate_cointegration()
result = model.estimate_threshold()
```

## Threshold VECM

### Hansen & Seo (2002) Threshold VECM

The Threshold Vector Error Correction Model (TVECM) extends the standard VECM to include regime-switching behavior based on the error correction term.

**Model Specification:**
$$\Delta X_t =
\begin{cases}
A_1 X_{t-1} + u_{1t}, & \text{if } w_t(\beta) \leq \gamma \\
A_2 X_{t-1} + u_{2t}, & \text{if } w_t(\beta) > \gamma
\end{cases}$$

where:
- $X_t$ is a vector of variables
- $w_t(\beta)$ is the error correction term
- $\gamma$ is the threshold parameter

**Implementation:**
```python
from src.models.threshold_vecm import ThresholdVECM

model = ThresholdVECM(market_data, k_ar_diff=2)
model.estimate_linear_vecm()
threshold_results = model.grid_search_threshold()
tvecm_results = model.estimate_tvecm()
```

### Half-Life Calculation

The half-life measures how long it takes for a deviation from equilibrium to dissipate by half, providing an intuitive measure of adjustment speed:

$$\text{Half-life} = \frac{\ln(0.5)}{\ln(1+\alpha)}$$

where $\alpha$ is the adjustment coefficient.

**Implementation:**
```python
from src.models.threshold_vecm import ThresholdVECM

model = ThresholdVECM(market_data)
# After estimating TVECM
half_lives = model.calculate_half_lives()
```

## Spatial Econometrics

### Spatial Weight Matrix

The spatial weight matrix $W$ captures the connectivity between markets, with elements $w_{ij}$ representing the influence of market $j$ on market $i$.

In Yemen's context, the standard distance-based weights are adjusted for conflict intensity:

$$w_{ij}^* = w_{ij} \times (1 + \alpha \times \text{conflict}_{ij})$$

where:
- $w_{ij}$ is the original weight based on distance
- $\text{conflict}_{ij}$ is conflict intensity between markets
- $\alpha$ is the conflict adjustment parameter

**Implementation:**
```python
from src.models.spatial import SpatialEconometrics

model = SpatialEconometrics(markets_gdf)
weights = model.create_weight_matrix(
    k=5,
    conflict_adjusted=True,
    conflict_col='conflict_intensity'
)
```

### Spatial Lag Model (SLM)

The Spatial Lag Model incorporates spatial dependence in the dependent variable:

$$y = \rho W y + X\beta + \varepsilon$$

where:
- $y$ is the vector of prices
- $W$ is the spatial weight matrix
- $\rho$ is the spatial autoregressive parameter
- $X$ is a matrix of explanatory variables

**Implementation:**
```python
from src.models.spatial import SpatialEconometrics

model = SpatialEconometrics(markets_gdf)
result = model.spatial_lag_model('price', ['distance', 'population'])
```

### Spatial Error Model (SEM)

The Spatial Error Model incorporates spatial dependence in the error term:

$$y = X\beta + u, \quad u = \lambda W u + \varepsilon$$

where:
- $\lambda$ is the spatial error parameter

**Implementation:**
```python
from src.models.spatial import SpatialEconometrics

model = SpatialEconometrics(markets_gdf)
result = model.spatial_error_model('price', ['distance', 'population'])
```

### Market Accessibility Index

The Market Accessibility Index measures market access considering population, distance, and conflict barriers:

$$A_i = \sum_{j} \frac{P_j}{(d_{ij} \times (1 + \alpha \times \text{conflict}_{ij}))^{\beta}}$$

where:
- $A_i$ is the accessibility index for market $i$
- $P_j$ is the population in location $j$
- $d_{ij}$ is the distance between locations
- $\beta$ is the distance decay parameter

**Implementation:**
```python
from src.models.spatial import SpatialEconometrics

model = SpatialEconometrics(markets_gdf)
accessibility = model.compute_accessibility_index(
    markets_gdf,
    population_gdf,
    distance_decay=2.0
)
```

## Model Diagnostics

### Residual Diagnostics

Comprehensive diagnostic tests for model validation:

1. **Normality Test**: Jarque-Bera test for residual normality
2. **Autocorrelation Test**: Ljung-Box test for residual autocorrelation
3. **Heteroskedasticity Test**: White's test for heteroskedasticity

**Implementation:**
```python
from src.models.diagnostics import ModelDiagnostics

diagnostics = ModelDiagnostics(residuals=model_residuals)
all_tests = diagnostics.residual_tests()
```

### Model Selection Criteria

Information criteria for model selection:

1. **Akaike Information Criterion (AIC)**: $\text{AIC} = -2\ln(L) + 2k$
2. **Bayesian Information Criterion (BIC)**: $\text{BIC} = -2\ln(L) + k\ln(n)$
3. **Hannan-Quinn Criterion (HQC)**: $\text{HQC} = -2\ln(L) + 2k\ln(\ln(n))$

where $L$ is the likelihood, $k$ is the number of parameters, and $n$ is the sample size.

**Implementation:**
```python
from src.models.diagnostics import calculate_fit_statistics

fit_stats = calculate_fit_statistics(
    observed=actual_values,
    predicted=model_predictions,
    n_params=5
)
```

## Policy Simulation

### Exchange Rate Unification Simulation

The simulation follows these steps:
1. Convert all prices to USD using regional exchange rates
2. Apply a unified exchange rate across all regions
3. Recalculate price differentials and market integration metrics

**Mathematical Formulation:**
$$p_{i,USD} = \frac{p_i}{e_i}$$
$$p_{i,YER}^{unified} = p_{i,USD} \times e_{unified}$$

where:
- $p_i$ is the original price in market $i$
- $e_i$ is the exchange rate in region containing market $i$
- $e_{unified}$ is the unified exchange rate
- $p_{i,YER}^{unified}$ is the simulated price after unification

**Implementation:**
```python
from src.models.simulation import MarketIntegrationSimulation

sim = MarketIntegrationSimulation(markets_data)
result = sim.simulate_exchange_rate_unification(target_rate='official')
```

### Connectivity Improvement Simulation

Simulates reduction in conflict barriers with these steps:
1. Reduce conflict intensity metrics
2. Recalculate spatial weights with reduced conflict
3. Re-estimate spatial models

**Mathematical Formulation:**
$$\text{conflict}_{ij}^* = \text{conflict}_{ij} \times (1 - \alpha)$$
$$w_{ij}^* = w_{ij} \times (1 + \beta \times \text{conflict}_{ij}^*)$$

where:
- $\alpha$ is the conflict reduction factor
- $\beta$ is the conflict weight factor

**Implementation:**
```python
from src.models.simulation import MarketIntegrationSimulation

sim = MarketIntegrationSimulation(markets_data)
result = sim.simulate_improved_connectivity(reduction_factor=0.5)
```

### Welfare Analysis

Quantifies benefits of policy interventions through several metrics:

1. **Price Convergence**: $\text{Convergence} = \frac{|p_{north} - p_{south}|_{before} - |p_{north} - p_{south}|_{after}}{|p_{north} - p_{south}|_{before}} \times 100\%$

2. **Integration Index**: Improvement in market integration metrics after policy intervention

**Implementation:**
```python
from src.models.simulation import MarketIntegrationSimulation

sim = MarketIntegrationSimulation(markets_data)
# After running a simulation
welfare = sim.calculate_welfare_effects('exchange_rate_unification')
