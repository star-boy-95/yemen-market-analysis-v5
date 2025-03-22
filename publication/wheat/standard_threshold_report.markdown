# Threshold Model Analysis Report

## Metadata

- **Model Type:** ThresholdModel
- **Model Mode:** standard
- **Market 1:** North
- **Market 2:** South
- **Generated:** 2025-03-22T14:46:35.213805

## Interpretation

### Cointegration

The markets North and South are not cointegrated (p-value: 0.8835). This suggests that these markets do not share a stable long-run equilibrium relationship, possibly due to significant barriers to trade such as conflict zones, political fragmentation, or transportation constraints.

### Threshold

The estimated threshold is -0.1038, representing the transaction cost barrier between North and South. When price differentials are below this threshold (44.7% of observations), arbitrage is not profitable due to transaction costs. When price differentials exceed the threshold (55.3% of observations), arbitrage becomes profitable and drives prices back toward equilibrium. In Yemen's conflict context, this threshold captures barriers including security checkpoints, conflict zones, and dual exchange rate effects.

### Market Integration

The markets North and South are not integrated. The absence of cointegration indicates that prices do not share a long-run equilibrium relationship, suggesting these markets operate independently. This is likely due to significant barriers to trade such as conflict zones, political fragmentation, or transportation constraints that prevent effective arbitrage.

## Statistical Results

### Cointegration Analysis

| Parameter | Value | Description |
| --- | --- | --- |
| Intercept (β₀) | 0.2363 | Long-run equilibrium intercept |
| Slope (β₁) | 0.4783 | Long-run price transmission elasticity |
| Cointegration Equation | North = 0.2363 + 0.4783 × South | Long-run equilibrium relationship |

### Threshold Estimation

| Parameter | Value | Description |
| --- | --- | --- |
| Threshold | -0.1038 | Estimated transaction cost threshold |
| Below Threshold | 44.7% | Proportion of observations below threshold |
| Above Threshold | 55.3% | Proportion of observations above threshold |

## Visualizations

### Threshold Regime Dynamics

Visualization of price adjustment dynamics in different regimes

![Threshold Regime Dynamics](standard_threshold_report_regime_dynamics.png)

### Market Price Time Series

Time series of prices in both markets

![Market Price Time Series](standard_threshold_report_time_series.png)

### Equilibrium Error Dynamics

Deviations from long-run equilibrium with threshold

![Equilibrium Error Dynamics](standard_threshold_report_equilibrium_error.png)

