# Yemen Market Analysis: Methodology Implementation Plan

This document outlines a comprehensive plan for implementing all missing methodologies in the Yemen Market Analysis project, based on the documentation in the `docs` folder and the current state of the codebase.

## Table of Contents

1. [Overview](#overview)
2. [Unit Root and Cointegration Module](#unit-root-and-cointegration-module)
3. [Threshold Models Module](#threshold-models-module)
4. [Spatial Econometrics Module](#spatial-econometrics-module)
5. [Policy Simulation Module](#policy-simulation-module)
6. [Model Diagnostics Module](#model-diagnostics-module)
7. [Implementation Timeline](#implementation-timeline)
8. [Testing Strategy](#testing-strategy)

## Overview

The Yemen Market Analysis project requires several advanced econometric methodologies to analyze market integration in conflict-affected settings. Based on the documentation, several key methodologies need to be implemented or completed:

1. **Unit Root and Cointegration Testing**: Methods for testing stationarity and cointegration relationships
2. **Threshold Models**: Nonlinear models for analyzing asymmetric price adjustments
3. **Spatial Econometrics**: Methods for analyzing geographic dependencies in market integration
4. **Policy Simulation**: Tools for simulating the effects of policy interventions
5. **Model Diagnostics**: Comprehensive tools for validating econometric models

## Unit Root and Cointegration Module

### Current Status

The unit root testing module has basic functionality but is missing several advanced methods and has validation issues.

### Implementation Tasks

1. **Fix Validation Issues**
   - Update the `validate_time_series()` function to handle the `custom_validators` parameter
   - Implement proper error handling for missing data and invalid inputs
   - Add validation for minimum series length and frequency

2. **Implement Zivot-Andrews Test**
   ```python
   def test_zivot_andrews(self, series, max_lags=None, trend='both'):
       """
       Test for unit root with a structural break using Zivot-Andrews test.
       
       Parameters
       ----------
       series : array_like
           Time series to test
       max_lags : int, optional
           Maximum number of lags
       trend : str, optional
           Trend specification ('intercept', 'trend', 'both')
           
       Returns
       -------
       dict
           Test results including:
           - statistic: test statistic
           - p_value: p-value
           - critical_values: critical values
           - breakpoint: estimated breakpoint
           - stationary: bool, whether series is stationary
       """
       # Validate input
       valid, errors = validate_time_series(series)
       if not valid:
           raise ValidationError(f"Invalid time series: {errors}")
       
       # Determine lag order if not specified
       if max_lags is None:
           max_lags = int(np.ceil(12 * (len(series)/100)**(1/4)))
       
       # Initialize variables
       t = len(series)
       y = series.values if isinstance(series, pd.Series) else series
       
       # Trim series to exclude endpoints (typically 15% from each end)
       trim = int(0.15 * t)
       
       # Initialize results
       min_stat = np.inf
       breakpoint = None
       
       # Loop through potential breakpoints
       for tb in range(trim, t - trim):
           # Create dummy variables
           du = np.zeros(t)
           dt = np.zeros(t)
           du[tb:] = 1
           dt[tb:] = np.arange(1, t - tb + 1)
           
           # Create regression matrix based on trend specification
           if trend == 'intercept':
               x = np.column_stack((np.ones(t), np.arange(1, t + 1), du, y[:-1]))
           elif trend == 'trend':
               x = np.column_stack((np.ones(t), np.arange(1, t + 1), dt, y[:-1]))
           else:  # 'both'
               x = np.column_stack((np.ones(t), np.arange(1, t + 1), du, dt, y[:-1]))
           
           # Add lagged differences
           for j in range(1, max_lags + 1):
               x = np.column_stack((x, np.diff(y, n=1)[j-1:t-1+j]))
           
           # Adjust sample for lagged differences
           y_diff = np.diff(y, n=1)[max_lags:]
           x = x[max_lags:, :]
           
           # OLS regression
           beta = np.linalg.lstsq(x, y_diff, rcond=None)[0]
           resid = y_diff - x @ beta
           ssr = np.sum(resid**2)
           
           # Calculate t-statistic for unit root coefficient
           se = np.sqrt(ssr / (len(y_diff) - x.shape[1]))
           t_stat = beta[3] / (se * np.sqrt(np.linalg.inv(x.T @ x)[3, 3]))
           
           # Update minimum statistic
           if t_stat < min_stat:
               min_stat = t_stat
               breakpoint = tb
       
       # Critical values (from Zivot and Andrews, 1992)
       critical_values = {
           'intercept': {'1%': -5.34, '5%': -4.80, '10%': -4.58},
           'trend': {'1%': -4.93, '5%': -4.42, '10%': -4.11},
           'both': {'1%': -5.57, '5%': -5.08, '10%': -4.82}
       }
       
       # Determine p-value (approximation)
       p_value = self._approximate_za_pvalue(min_stat, trend)
       
       # Determine if stationary
       stationary = min_stat < critical_values[trend]['5%']
       
       return {
           'statistic': min_stat,
           'p_value': p_value,
           'critical_values': critical_values[trend],
           'breakpoint': breakpoint,
           'stationary': stationary
       }
   
   def _approximate_za_pvalue(self, statistic, trend):
       """
       Approximate p-value for Zivot-Andrews test.
       
       Parameters
       ----------
       statistic : float
           Test statistic
       trend : str
           Trend specification
           
       Returns
       -------
       float
           Approximate p-value
       """
       # Approximation based on simulated critical values
       # This is a simplified approach; more accurate methods exist
       if trend == 'intercept':
           if statistic < -5.34:
               return 0.01
           elif statistic < -4.80:
               return 0.05
           elif statistic < -4.58:
               return 0.10
           else:
               return 0.10 + 0.90 * (1 - np.exp(-(statistic + 4.58)))
       elif trend == 'trend':
           if statistic < -4.93:
               return 0.01
           elif statistic < -4.42:
               return 0.05
           elif statistic < -4.11:
               return 0.10
           else:
               return 0.10 + 0.90 * (1 - np.exp(-(statistic + 4.11)))
       else:  # 'both'
           if statistic < -5.57:
               return 0.01
           elif statistic < -5.08:
               return 0.05
           elif statistic < -4.82:
               return 0.10
           else:
               return 0.10 + 0.90 * (1 - np.exp(-(statistic + 4.82)))
   ```

3. **Implement Integration Order Determination**
   ```python
   def determine_integration_order(self, series, max_order=2, alpha=0.05):
       """
       Determine the integration order of a time series.
       
       Parameters
       ----------
       series : array_like
           Time series to test
       max_order : int, optional
           Maximum integration order to test
       alpha : float, optional
           Significance level
           
       Returns
       -------
       int
           Integration order (0, 1, 2, ..., or max_order if higher)
       """
       # Validate input
       valid, errors = validate_time_series(series)
       if not valid:
           raise ValidationError(f"Invalid time series: {errors}")
       
       # Test original series
       adf_result = self.test_adf(series)
       if adf_result['stationary']:
           return 0
       
       # Test first difference
       for d in range(1, max_order + 1):
           diff_series = np.diff(series, n=d)
           adf_result = self.test_adf(diff_series)
           if adf_result['stationary']:
               return d
       
       # If still not stationary after max_order differences
       return max_order
   ```

4. **Implement Gregory-Hansen Test**
   ```python
   def test_gregory_hansen(self, y, x, model='regime_shift', trend='c', max_lags=None, alpha=0.05):
       """
       Test for cointegration with structural breaks using Gregory-Hansen test.
       
       Parameters
       ----------
       y : array_like
           Dependent variable
       x : array_like
           Independent variable(s)
       model : str, optional
           Model type ('level_shift', 'regime_shift', 'trend_shift')
       trend : str, optional
           Deterministic trend specification ('n', 'c', 'ct')
       max_lags : int, optional
           Maximum number of lags for ADF test
       alpha : float, optional
           Significance level
           
       Returns
       -------
       dict
           Test results including:
           - statistic: ADF test statistic
           - p_value: p-value
           - critical_values: critical values
           - breakpoint: estimated breakpoint
           - cointegrated: bool, whether series are cointegrated
           - beta: cointegrating vector (if cointegrated)
       """
       # Implementation similar to Zivot-Andrews but for cointegration
       # This is a complex test that requires careful implementation
   ```

5. **Enhance Johansen Test**
   ```python
   def test_johansen(self, data, det_order=1, k_ar_diff=2, alpha=0.05):
       """
       Test for cointegration in a multivariate system using Johansen procedure.
       
       Parameters
       ----------
       data : array_like
           Multivariate time series data
       det_order : int, optional
           Deterministic trend specification
       k_ar_diff : int, optional
           Number of lagged differences in the VECM
       alpha : float, optional
           Significance level
           
       Returns
       -------
       dict
           Test results including:
           - rank: estimated cointegration rank
           - trace_stat: trace statistics
           - max_eig_stat: maximum eigenvalue statistics
           - critical_values: critical values
           - eigenvectors: cointegrating vectors (if cointegrated)
           - eigenvalues: eigenvalues from decomposition
       """
       # Enhanced implementation with better error handling and validation
   ```

6. **Implement Half-Life Calculation**
   ```python
   def calculate_half_life(self, alpha, method='standard'):
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
       if method == 'standard':
           # Standard half-life calculation
           if alpha >= 0:
               return float('inf')  # No convergence
           return np.log(0.5) / np.log(1 + alpha)
       elif method == 'threshold':
           # Threshold-specific calculation
           # This would depend on the specific threshold model
           # Implementation would vary based on model structure
           pass
       else:
           raise ValueError(f"Unknown method: {method}")
   ```

## Threshold Models Module

### Current Status

The threshold models module has basic functionality but is missing several advanced methods for asymmetric adjustment and regime-switching.

### Implementation Tasks

1. **Implement Threshold Autoregressive (TAR) Model**
   ```python
   def estimate_tar(self, threshold_var=None, max_lags=4, trim=0.15, threshold_range=None):
       """
       Estimate Threshold Autoregressive (TAR) model.
       
       Parameters
       ----------
       threshold_var : array_like, optional
           Threshold variable (uses lagged dependent variable if None)
       max_lags : int, optional
           Maximum number of lags
       trim : float, optional
           Trimming percentage for threshold search
       threshold_range : tuple, optional
           Range for threshold search (min, max)
           
       Returns
       -------
       dict
           Estimation results including:
           - threshold: estimated threshold value
           - coefficients_below: coefficients for lower regime
           - coefficients_above: coefficients for upper regime
           - residuals: model residuals
           - aic: Akaike Information Criterion
           - bic: Bayesian Information Criterion
           - threshold_effect: test for threshold effect
       """
       # Implementation
   ```

2. **Implement Momentum-Threshold Autoregressive (M-TAR) Model**
   ```python
   def estimate_mtar(self, max_lags=4, trim=0.15, threshold_range=None):
       """
       Estimate Momentum-Threshold Autoregressive (M-TAR) model.
       
       Parameters
       ----------
       max_lags : int, optional
           Maximum number of lags
       trim : float, optional
           Trimming percentage for threshold search
       threshold_range : tuple, optional
           Range for threshold search (min, max)
           
       Returns
       -------
       dict
           Estimation results including:
           - threshold: estimated threshold value
           - coefficients_below: coefficients for lower regime
           - coefficients_above: coefficients for upper regime
           - residuals: model residuals
           - aic: Akaike Information Criterion
           - bic: Bayesian Information Criterion
           - threshold_effect: test for threshold effect
       """
       # Implementation
   ```

3. **Enhance Threshold Cointegration**
   ```python
   def estimate_threshold(self, threshold_var=None, trim=0.15, max_iter=300, tol=1e-6):
       """
       Estimate threshold cointegration model.
       
       Parameters
       ----------
       threshold_var : array_like, optional
           Threshold variable (uses error correction term if None)
       trim : float, optional
           Trimming percentage for threshold search
       max_iter : int, optional
           Maximum number of iterations for estimation
       tol : float, optional
           Convergence tolerance
           
       Returns
       -------
       dict
           Estimation results including:
           - threshold: estimated threshold value
           - adjustment_below: adjustment coefficient below threshold
           - adjustment_above: adjustment coefficient above threshold
           - coefficients_below: coefficients for lower regime
           - coefficients_above: coefficients for upper regime
           - residuals: model residuals
           - aic: Akaike Information Criterion
           - bic: Bayesian Information Criterion
           - threshold_effect: test for threshold effect
       """
       # Enhanced implementation with better search algorithm
   ```

4. **Implement Hansen & Seo (2002) Threshold VECM**
   ```python
   def estimate_tvecm(self, beta=None, threshold=None, trim=0.15, max_iter=300, tol=1e-6):
       """
       Estimate Threshold Vector Error Correction Model (TVECM).
       
       Parameters
       ----------
       beta : array_like, optional
           Cointegrating vector (estimated if None)
       threshold : float, optional
           Threshold value (estimated if None)
       trim : float, optional
           Trimming percentage for threshold search
       max_iter : int, optional
           Maximum number of iterations for estimation
       tol : float, optional
           Convergence tolerance
           
       Returns
       -------
       dict
           Estimation results including:
           - threshold: estimated threshold value
           - beta: cointegrating vector
           - adjustment_below_1: adjustment coefficient for first variable below threshold
           - adjustment_above_1: adjustment coefficient for first variable above threshold
           - adjustment_below_2: adjustment coefficient for second variable below threshold
           - adjustment_above_2: adjustment coefficient for second variable above threshold
           - coefficients_below: coefficients for lower regime
           - coefficients_above: coefficients for upper regime
           - residuals: model residuals
           - aic: Akaike Information Criterion
           - bic: Bayesian Information Criterion
           - threshold_effect: test for threshold effect
       """
       # Implementation
   ```

5. **Implement Impulse Response Functions**
   ```python
   def calculate_impulse_responses(self, periods=24, shocks=None, regime=None):
       """
       Calculate impulse response functions for threshold models.
       
       Parameters
       ----------
       periods : int, optional
           Number of periods for impulse responses
       shocks : array_like, optional
           Custom shock values (uses unit shocks if None)
       regime : str, optional
           Regime to calculate IRFs for ('below', 'above', 'both')
           
       Returns
       -------
       dict
           Impulse response functions including:
           - irf_below: IRFs for lower regime
           - irf_above: IRFs for upper regime
           - irf_combined: Combined IRFs (if regime='both')
       """
       # Implementation
   ```

## Spatial Econometrics Module

### Current Status

The spatial econometrics module has basic functionality but is missing several advanced methods for conflict-adjusted spatial analysis.

### Implementation Tasks

1. **Fix Validation Issues**
   - Update the `validate_geodataframe()` function to handle the `check_crs` parameter
   - Fix the Numba compilation issue in `create_conflict_adjusted_weights()`
   - Implement proper error handling for missing data and invalid inputs

2. **Implement Conflict-Adjusted Spatial Weights**
   ```python
   def create_conflict_adjusted_weights(self, gdf, k=5, conflict_col=None, conflict_weight=0.5):
       """
       Create spatial weights adjusted for conflict intensity.
       
       Parameters
       ----------
       gdf : geopandas.GeoDataFrame
           GeoDataFrame with points
       k : int, optional
           Number of nearest neighbors
       conflict_col : str, optional
           Column name for conflict intensity
       conflict_weight : float, optional
           Weight for conflict in distance adjustment
           
       Returns
       -------
       libpysal.weights.W
           Conflict-adjusted spatial weights matrix
       """
       # Implementation without Numba to avoid compilation issues
       # Use pure Python implementation instead
   ```

3. **Implement Spatial Lag Model**
   ```python
   def spatial_lag_model(self, y_col, x_cols, weights=None, method='ML'):
       """
       Estimate Spatial Lag Model (SLM).
       
       Parameters
       ----------
       y_col : str
           Column name for dependent variable
       x_cols : list of str
           Column names for independent variables
       weights : libpysal.weights.W, optional
           Spatial weights matrix (uses self.weights if None)
       method : str, optional
           Estimation method ('ML', 'GMM')
           
       Returns
       -------
       dict
           Model results including:
           - rho: spatial autoregressive parameter
           - betas: coefficient estimates
           - std_err: standard errors
           - t_stats: t-statistics
           - p_values: p-values
           - r2: R-squared
           - log_likelihood: log-likelihood
           - aic: Akaike Information Criterion
       """
       # Implementation
   ```

4. **Implement Spatial Error Model**
   ```python
   def spatial_error_model(self, y_col, x_cols, weights=None, method='ML'):
       """
       Estimate Spatial Error Model (SEM).
       
       Parameters
       ----------
       y_col : str
           Column name for dependent variable
       x_cols : list of str
           Column names for independent variables
       weights : libpysal.weights.W, optional
           Spatial weights matrix (uses self.weights if None)
       method : str, optional
           Estimation method ('ML', 'GMM')
           
       Returns
       -------
       dict
           Model results including:
           - lambda: spatial error parameter
           - betas: coefficient estimates
           - std_err: standard errors
           - t_stats: t-statistics
           - p_values: p-values
           - r2: R-squared
           - log_likelihood: log-likelihood
           - aic: Akaike Information Criterion
       """
       # Implementation
   ```

5. **Implement Spatial Durbin Model**
   ```python
   def spatial_durbin_model(self, y_col, x_cols, weights=None, method='ML'):
       """
       Estimate Spatial Durbin Model (SDM).
       
       Parameters
       ----------
       y_col : str
           Column name for dependent variable
       x_cols : list of str
           Column names for independent variables
       weights : libpysal.weights.W, optional
           Spatial weights matrix (uses self.weights if None)
       method : str, optional
           Estimation method ('ML', 'GMM')
           
       Returns
       -------
       dict
           Model results including:
           - rho: spatial autoregressive parameter
           - betas: coefficient estimates
           - thetas: spatial lag coefficient estimates
           - std_err: standard errors
           - t_stats: t-statistics
           - p_values: p-values
           - r2: R-squared
           - log_likelihood: log-likelihood
           - aic: Akaike Information Criterion
       """
       # Implementation
   ```

6. **Implement Direct and Indirect Effects Calculation**
   ```python
   def calculate_impacts(self, model='lag'):
       """
       Calculate direct, indirect, and total effects for spatial models.
       
       Parameters
       ----------
       model : str, optional
           Model type ('lag', 'error', 'durbin')
           
       Returns
       -------
       dict
           Impact measures including:
           - direct: direct effects
           - indirect: indirect (spillover) effects
           - total: total effects
       """
       # Implementation
   ```

7. **Implement Market Accessibility Index**
   ```python
   def compute_accessibility_index(self, population_gdf, max_distance=None, distance_decay=2.0, weight_col='population'):
       """
       Compute market accessibility index considering population and distance.
       
       Parameters
       ----------
       population_gdf : geopandas.GeoDataFrame
           GeoDataFrame with population points/polygons
       max_distance : float, optional
           Maximum distance to consider
       distance_decay : float, optional
           Distance decay parameter (power)
       weight_col : str, optional
           Column name in population_gdf for weights
           
       Returns
       -------
       geopandas.GeoDataFrame
           Original GeoDataFrame with additional column 'accessibility_index'
       """
       # Implementation
   ```

## Policy Simulation Module

### Current Status

The policy simulation module is missing most of its core functionality, with only placeholder implementations.

### Implementation Tasks

1. **Implement Exchange Rate Unification Simulation**
   ```python
   def simulate_exchange_rate_unification(self, target_rate='official', reference_date=None):
       """
       Simulate exchange rate unification using USD as cross-rate.
       
       Parameters
       ----------
       target_rate : str or float, optional
           Method to determine the unified exchange rate:
           - 'official': Use official exchange rate
           - 'market': Use market exchange rate
           - 'average': Use average of north and south rates
           - Specific value: Use provided numerical value
       reference_date : str, optional
           Date to use for reference exchange rates (default: latest date)
           
       Returns
       -------
       dict
           Simulation results including:
           - 'simulated_data': GeoDataFrame with simulated prices
           - 'unified_rate': Selected unified exchange rate
           - 'price_changes': DataFrame of price changes by region
           - 'threshold_model': Re-estimated threshold model
       """
       # Validate input data
       self._validate_input_data()
       
       # Convert prices to USD
       data_with_usd = self._convert_to_usd(self.data)
       
       # Determine unified exchange rate
       unified_rate = self._determine_unified_rate(data_with_usd, target_rate, reference_date)
       
       # Apply unified rate to convert back to YER
       simulated_data = data_with_usd.copy()
       simulated_data['simulated_price'] = simulated_data['usd_price'] * unified_rate
       
       # Calculate price changes
       price_changes = self._calculate_price_changes(
           self.data['price'], 
           simulated_data['simulated_price'],
           by_column='exchange_rate_regime'
       )
       
       # Re-estimate threshold model if available
       threshold_model = None
       if self.threshold_model is not None:
           # Extract north and south prices
           north_prices = simulated_data[simulated_data['exchange_rate_regime'] == 'north']['simulated_price']
           south_prices = simulated_data[simulated_data['exchange_rate_regime'] == 'south']['simulated_price']
           
           # Re-estimate threshold model
           from src.models.threshold import ThresholdCointegration
           threshold_model = ThresholdCointegration(north_prices, south_prices)
           threshold_model.estimate_cointegration()
           threshold_model.estimate_threshold()
       
       return {
           'simulated_data': simulated_data,
           'unified_rate': unified_rate,
           'price_changes': price_changes,
           'threshold_model': threshold_model
       }
   ```

2. **Implement Improved Connectivity Simulation**
   ```python
   def simulate_improved_connectivity(self, reduction_factor=0.5):
       """
       Simulate improved connectivity by reducing conflict barriers.
       
       Parameters
       ----------
       reduction_factor : float, optional
           Factor by which to reduce conflict intensity (0.0-1.0)
           
       Returns
       -------
       dict
           Simulation results including:
           - 'simulated_data': GeoDataFrame with reduced conflict intensity
           - 'spatial_weights': Recalculated spatial weights
           - 'spatial_model': Re-estimated spatial model
       """
       # Validate input data
       self._validate_input_data()
       
       # Apply improved connectivity
       simulated_data, new_weights = self._apply_improved_connectivity(
           self.data, 
           reduction_factor=reduction_factor,
           conflict_col='conflict_intensity_normalized'
       )
       
       # Re-estimate spatial model if available
       spatial_model = None
       if self.spatial_model is not None:
           # Create new spatial model with simulated data
           from src.models.spatial import SpatialEconometrics
           spatial_model = SpatialEconometrics(simulated_data)
           spatial_model.weights = new_weights
           
           # Re-estimate model
           if hasattr(self.spatial_model, 'y_col') and hasattr(self.spatial_model, 'x_cols'):
               spatial_model.spatial_lag_model(
                   y_col=self.spatial_model.y_col,
                   x_cols=self.spatial_model.x_cols
               )
       
       return {
           'simulated_data': simulated_data,
           'spatial_weights': new_weights,
           'spatial_model': spatial_model
       }
   ```

3. **Implement Combined Policy Simulation**
   ```python
   def simulate_combined_policies(self, reduction_factor=0.5, unification_method='official', reference_date=None):
       """
       Simulate combined exchange rate unification and improved connectivity.
       
       Parameters
       ----------
       reduction_factor : float, optional
           Factor by which to reduce conflict intensity
       unification_method : str, optional
           Method to determine unified exchange rate
       reference_date : str, optional
           Date to use for reference exchange rates
           
       Returns
       -------
       dict
           Combined simulation results
       """
       # Validate input data
       self._validate_input_data()
       
       # First simulate improved connectivity
       connectivity_results = self.simulate_improved_connectivity(reduction_factor)
       
       # Then simulate exchange rate unification on the connectivity-adjusted data
       # Create a temporary simulation object with the connectivity-adjusted data
       temp_sim = MarketIntegrationSimulation(
           data=connectivity_results['simulated_data'],
           threshold_model=self.threshold_model,
           spatial_model=connectivity_results['spatial_model']
       )
       
       # Run exchange rate unification simulation
       exchange_results = temp_sim.simulate_exchange_rate_unification(
           target_rate=unification_method,
           reference_date=reference_date
       )
       
       # Combine results
       combined_results = {
           'simulated_data': exchange_results['simulated_data'],
           'unified_rate': exchange_results['unified_rate'],
           'conflict_reduction': reduction_factor,
           'price_changes': exchange_results['price_changes'],
           'threshold_model': exchange_results['threshold_model'],
           'spatial_model': connectivity_results['spatial_model'],
           'spatial_weights': connectivity_results['spatial_weights']
       }
       
       return combined_results
   ```

4. **Implement Welfare Effects Calculation**
   ```python
   def calculate_welfare_effects(self, policy_scenario=None):
       """
       Calculate welfare effects of policy simulations.
       
       Parameters
       ----------
       policy_scenario : str, optional
           Which policy scenario to analyze:
           - 'exchange_rate_unification'
           - 'improved_connectivity'
           - 'combined_policy'
           If None, uses latest simulation results
           
       Returns
       -------
       dict
           Welfare analysis results
       """
       # Implementation
   ```

5. **Implement Sensitivity Analysis**
   ```python
   def run_sensitivity_analysis(self, sensitivity_type='conflict_reduction', param_values=None, metrics=None):
       """
       Run sensitivity analysis by varying parameters and measuring impact.
       
       Parameters
       ----------
       sensitivity_type : str, optional
           Type of sensitivity analysis:
           - 'conflict_reduction': Vary conflict reduction factor
           - 'exchange_rate': Vary exchange rate target values
       param_values : List[float], optional
           List of parameter values to test
       metrics : List[str], optional
           List of metrics to track
           
       Returns
       -------
       dict
           Sensitivity analysis results
       """
       # Implementation
   ```

## Model Diagnostics Module

### Current Status

The model diagnostics module has basic functionality but is missing several advanced methods for comprehensive model validation.

### Implementation Tasks

1. **Implement Residual Diagnostics**
   ```python
   def residual_tests(self, residuals, lags=10):
       """
       Run comprehensive diagnostic tests on model residuals.
       
       Parameters
       ----------
       residuals : array_like
           Model residuals
       lags : int, optional
           Number of lags for autocorrelation tests
           
       Returns
       -------
       dict
           Test results including:
           - normality: Jarque-Bera test results
           - autocorrelation: Ljung-Box test results
           - heteroskedasticity: White test results
           - overall: summary of all tests
       """
       # Implementation
   ```

2. **Implement Model Selection Criteria**
   ```python
   def calculate_information_criteria(self, model):
       """
       Calculate information criteria for model selection.
       
       Parameters
       ----------
       model : object
           Estimated model object
           
       Returns
       -------
       dict
           Information criteria including:
           - aic: Akaike Information Criterion
           - bic: Bayesian Information Criterion
           - hqic: Hannan-Quinn Information Criterion
       """
       # Implementation
   ```

3. **Implement Parameter Stability Tests**
   ```python
   def test_parameter_stability(self, model, data):
       """
       Test for parameter stability over time.
       
       Parameters
       ----------
       model : object
           Estimated model object
       data : pandas.DataFrame
           Data used for estimation
           
       Returns
       -------
       dict
           Stability test results including:
           - cusum: CUSUM test results
           - cusum_sq: CUSUM of squares test results
           - recursive: recursive estimation results
           - chow: Chow test results
       """
       # Implementation
   ```

4. **Implement Model Comparison**
   ```python
   def compare_models(self, models, data):
       """
       Compare multiple models using various criteria.
       
       Parameters
       ----------
       models : list
           List of estimated model objects
       data : pandas.DataFrame
           Data used for estimation
           
       Returns
       -------
       dict
           Comparison results including:
           - information_criteria: AIC, BIC, HQIC for each model
           - likelihood_ratio: likelihood ratio test results
           - forecast_accuracy: forecast accuracy metrics
           - recommendation: recommended model
       """
       # Implementation
   ```

## Implementation Timeline

### Phase 1: Core Functionality (Weeks 1-2)

1. Fix validation issues in Unit Root and Spatial modules
2. Implement basic Threshold Cointegration functionality
3. Implement Exchange Rate Unification simulation

### Phase 2: Advanced Methods (Weeks 3-4)

1. Implement Zivot-Andrews and Gregory-Hansen tests
2. Implement Hansen & Seo Threshold VECM
3. Implement Conflict-Adjusted Spatial Weights
4. Implement Improved Connectivity simulation

### Phase 3: Integration and Optimization (Weeks 5-6)

1. Implement Combined Policy simulation
2. Implement Welfare Effects calculation
3. Optimize performance for large datasets
4. Implement parallel processing for simulations

### Phase 4: Diagnostics and Validation (Weeks 7-8)

1. Implement comprehensive Residual Diagnostics
2. Implement Parameter Stability Tests
3. Implement Model Comparison functionality
4. Implement Sensitivity Analysis

## Testing Strategy

### Unit Tests

1. Test each method with known data and expected results
2. Test edge cases and error handling
3. Test with different parameter configurations

### Integration Tests

1. Test interaction between different modules
2. Test end-to-end workflows
3. Test with real Yemen market data

### Performance Tests

1. Test with large datasets
2. Benchmark parallel processing performance
3. Test memory usage optimization

### Validation Tests

1. Compare results with established econometric software
2. Validate against theoretical expectations
3. Test robustness to different data characteristics