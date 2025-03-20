# Methodology Implementation Plan for Yemen Market Integration Analysis

## Overview

This document outlines the econometric methodology and implementation plan for the Yemen Market Integration project. It focuses on the technical aspects of implementing the econometric methods, including threshold cointegration, spatial econometrics, and policy simulation.

## 1. Econometric Framework

### 1.1 Time Series Analysis

#### Unit Root Testing
- **Implementation Status**: Complete
- **Methods**: 
  - Augmented Dickey-Fuller (ADF) test
  - KPSS test
  - Zivot-Andrews test for structural breaks
- **Implementation Details**:
  - Use the `UnitRootTester` class in `src/models/unit_root.py`
  - Implement tests with appropriate lag selection
  - Account for structural breaks due to conflict events

#### Cointegration Analysis
- **Implementation Status**: Complete
- **Methods**:
  - Engle-Granger two-step method
  - Johansen procedure for multivariate analysis
  - Gregory-Hansen test for cointegration with structural breaks
- **Implementation Details**:
  - Use the `CointegrationTester` class in `src/models/cointegration.py`
  - Implement tests with appropriate model specification
  - Calculate half-life of deviations from equilibrium

#### Threshold Cointegration
- **Implementation Status**: Complete
- **Methods**:
  - Threshold Autoregressive (TAR) models
  - Momentum-TAR (M-TAR) models
  - Threshold Vector Error Correction Models (TVECM)
- **Implementation Details**:
  - Use the `ThresholdCointegration` class in `src/models/threshold.py`
  - Implement grid search for threshold estimation
  - Calculate asymmetric adjustment parameters

### 1.2 Spatial Econometrics

#### Spatial Weight Matrix
- **Implementation Status**: Complete
- **Methods**:
  - K-nearest neighbors weights
  - Conflict-adjusted weights
  - Exchange rate regime boundary weights
- **Implementation Details**:
  - Use the `SpatialEconometrics` class in `src/models/spatial.py`
  - Implement conflict adjustment using conflict intensity data
  - Create specialized weights for cross-regime connections

#### Spatial Regression Models
- **Implementation Status**: Complete with recent updates
- **Methods**:
  - Spatial Lag Model (SLM)
  - Spatial Error Model (SEM)
  - Direct, indirect, and total effects calculation
- **Implementation Details**:
  - Use the `spatial_lag_model` and `spatial_error_model` methods
  - Implement the `calculate_impacts` method for effect decomposition
  - Store models as class attributes for further analysis

#### Spatial Autocorrelation
- **Implementation Status**: Complete
- **Methods**:
  - Global Moran's I
  - Local Indicators of Spatial Association (LISA)
  - Spatial clusters and outliers
- **Implementation Details**:
  - Use the `moran_i_test` and `local_moran_test` methods
  - Implement cluster classification
  - Visualize spatial patterns

### 1.3 Policy Simulation

#### Exchange Rate Unification
- **Implementation Status**: Needs implementation
- **Methods**:
  - Official rate unification
  - Market rate unification
  - Average rate unification
- **Implementation Details**:
  - Implement in `MarketIntegrationSimulation` class
  - Convert prices to USD using regional exchange rates
  - Apply unified exchange rate across all regions
  - Recalculate price differentials and integration metrics

#### Conflict Reduction
- **Implementation Status**: Needs implementation
- **Methods**:
  - Uniform conflict reduction
  - Targeted conflict reduction
  - Gradual vs. immediate reduction
- **Implementation Details**:
  - Implement in `MarketIntegrationSimulation` class
  - Reduce conflict intensity by specified factor
  - Recalculate spatial weights and market connections
  - Estimate impact on price transmission

#### Combined Policy Analysis
- **Implementation Status**: Needs implementation
- **Methods**:
  - Sequential policy implementation
  - Simultaneous policy implementation
  - Interaction effects analysis
- **Implementation Details**:
  - Implement in `MarketIntegrationSimulation` class
  - Combine exchange rate unification and conflict reduction
  - Analyze synergies and trade-offs
  - Calculate comprehensive welfare effects

## 2. Implementation Priorities

### 2.1 Immediate Priorities

1. **Complete Spatial Impact Calculation**
   - Status: Complete
   - Details: Implemented the `calculate_impacts` method in `SpatialEconometrics` class

2. **Implement Exchange Rate Unification Simulation**
   - Status: Needs implementation
   - Details: Replace placeholder with actual implementation in `MarketIntegrationSimulation` class

3. **Implement Conflict Reduction Simulation**
   - Status: Needs implementation
   - Details: Replace placeholder with actual implementation in `MarketIntegrationSimulation` class

4. **Implement Combined Policy Simulation**
   - Status: Needs implementation
   - Details: Replace placeholder with actual implementation in `MarketIntegrationSimulation` class

### 2.2 Secondary Priorities

1. **Enhance Welfare Analysis**
   - Status: Needs implementation
   - Details: Replace simplified calculations with comprehensive welfare metrics

2. **Implement Sensitivity Analysis**
   - Status: Needs implementation
   - Details: Replace placeholder with actual parameter sensitivity analysis

3. **Improve Visualization**
   - Status: Partially implemented
   - Details: Enhance visualization of spatial impacts and policy effects

### 2.3 Long-term Priorities

1. **Implement Advanced Econometric Methods**
   - Status: Future work
   - Details: Add panel data methods, dynamic spatial models, etc.

2. **Develop Interactive Dashboard**
   - Status: Future work
   - Details: Create interactive visualization of results

3. **Extend to Additional Commodities**
   - Status: Future work
   - Details: Apply methodology to more commodities and markets

## 3. Technical Implementation Details

### 3.1 Spatial Impact Calculation

```python
def calculate_impacts(self, model_type='lag'):
    """
    Calculate direct, indirect, and total effects for spatial models.
    
    Parameters
    ----------
    model_type : str, optional
        Type of model to calculate impacts for ('lag' or 'error')
        
    Returns
    -------
    dict
        Dictionary of direct, indirect, and total effects
    """
    # Check if we have the specified model
    if model_type == 'lag' and hasattr(self, 'lag_model'):
        # Calculate impacts for lag model
        impacts = self._calculate_lag_impacts()
        return impacts
    elif model_type == 'error' and hasattr(self, 'error_model'):
        # Calculate impacts for error model
        impacts = self._calculate_error_impacts()
        return impacts
    else:
        raise ValueError(f"Model type '{model_type}' not available or not estimated")
```

### 3.2 Exchange Rate Unification Simulation

```python
def simulate_exchange_rate_unification(self, method='official'):
    """
    Simulate exchange rate unification across regions.
    
    Parameters
    ----------
    method : str, optional
        Method for determining unified rate ('official', 'market', or 'average')
        
    Returns
    -------
    dict
        Simulation results
    """
    # Get current exchange rates
    north_rate = self.data[self.data['exchange_rate_regime'] == 'north']['exchange_rate'].mean()
    south_rate = self.data[self.data['exchange_rate_regime'] == 'south']['exchange_rate'].mean()
    
    # Determine unified rate based on method
    if method == 'official':
        unified_rate = north_rate  # Assuming north rate is official
    elif method == 'market':
        unified_rate = south_rate  # Assuming south rate is market-driven
    elif method == 'average':
        unified_rate = (north_rate + south_rate) / 2
    else:
        raise ValueError(f"Unknown unification method: {method}")
    
    # Convert all prices to USD
    self.data['usd_price'] = self.data['price'] / self.data['exchange_rate']
    
    # Apply unified exchange rate
    self.data['unified_price'] = self.data['usd_price'] * unified_rate
    
    # Calculate metrics
    results = self._calculate_simulation_metrics('unified_price')
    results['unified_rate'] = unified_rate
    results['method'] = method
    
    return results
```

### 3.3 Conflict Reduction Simulation

```python
def simulate_improved_connectivity(self, reduction_factor=0.5):
    """
    Simulate improved market connectivity by reducing conflict barriers.
    
    Parameters
    ----------
    reduction_factor : float, optional
        Percentage reduction in conflict (0-1)
        
    Returns
    -------
    dict
        Simulation results
    """
    # Create copy of data for simulation
    sim_data = self.data.copy()
    
    # Apply conflict reduction
    conflict_col = 'conflict_intensity_normalized'
    reduced_col = f'{conflict_col}_reduced'
    sim_data[reduced_col] = sim_data[conflict_col] * (1 - reduction_factor)
    
    # Create spatial model with reduced conflict
    spatial_model = SpatialEconometrics(sim_data)
    
    # Create weight matrices with reduced conflict
    spatial_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=reduced_col
    )
    
    # Estimate spatial model with reduced conflict
    y_col = 'price'
    x_cols = ['usdprice', 'conflict_intensity_normalized', 'distance_to_port']
    
    lag_model = spatial_model.spatial_lag_model(y_col, x_cols)
    
    # Calculate impacts
    impacts = spatial_model.calculate_impacts(model_type='lag')
    
    # Calculate metrics
    results = self._calculate_simulation_metrics('price')
    results['reduction_factor'] = reduction_factor
    results['impacts'] = impacts
    
    return results
```

### 3.4 Combined Policy Simulation

```python
def simulate_combined_policy(self, reduction_factor=0.5, exchange_rate_method='official'):
    """
    Simulate combined policies (conflict reduction + exchange rate unification).
    
    Parameters
    ----------
    reduction_factor : float, optional
        Percentage reduction in conflict (0-1)
    exchange_rate_method : str, optional
        Method for determining unified rate ('official', 'market', or 'average')
        
    Returns
    -------
    dict
        Simulation results
    """
    # Create copy of data for simulation
    sim_data = self.data.copy()
    
    # Apply conflict reduction
    conflict_col = 'conflict_intensity_normalized'
    reduced_col = f'{conflict_col}_reduced'
    sim_data[reduced_col] = sim_data[conflict_col] * (1 - reduction_factor)
    
    # Get current exchange rates
    north_rate = sim_data[sim_data['exchange_rate_regime'] == 'north']['exchange_rate'].mean()
    south_rate = sim_data[sim_data['exchange_rate_regime'] == 'south']['exchange_rate'].mean()
    
    # Determine unified rate based on method
    if exchange_rate_method == 'official':
        unified_rate = north_rate
    elif exchange_rate_method == 'market':
        unified_rate = south_rate
    elif exchange_rate_method == 'average':
        unified_rate = (north_rate + south_rate) / 2
    else:
        raise ValueError(f"Unknown unification method: {exchange_rate_method}")
    
    # Convert all prices to USD
    sim_data['usd_price'] = sim_data['price'] / sim_data['exchange_rate']
    
    # Apply unified exchange rate
    sim_data['unified_price'] = sim_data['usd_price'] * unified_rate
    
    # Create spatial model with reduced conflict and unified exchange rate
    spatial_model = SpatialEconometrics(sim_data)
    
    # Create weight matrices with reduced conflict
    spatial_model.create_weight_matrix(
        conflict_adjusted=True,
        conflict_col=reduced_col
    )
    
    # Estimate spatial model
    y_col = 'unified_price'
    x_cols = ['usd_price', reduced_col, 'distance_to_port']
    
    lag_model = spatial_model.spatial_lag_model(y_col, x_cols)
    
    # Calculate impacts
    impacts = spatial_model.calculate_impacts(model_type='lag')
    
    # Calculate metrics
    results = self._calculate_simulation_metrics('unified_price')
    results['reduction_factor'] = reduction_factor
    results['unified_rate'] = unified_rate
    results['exchange_rate_method'] = exchange_rate_method
    results['impacts'] = impacts
    
    return results
```

## 4. Testing and Validation

### 4.1 Unit Testing

- Test each econometric method individually
- Verify results against known examples
- Check edge cases and error handling

### 4.2 Integration Testing

- Test the full econometric pipeline
- Verify that components work together correctly
- Check that results are consistent across methods

### 4.3 Validation

- Compare results with economic theory
- Validate against empirical evidence
- Check robustness to parameter changes

## 5. Documentation

### 5.1 Code Documentation

- Add comprehensive docstrings to all methods
- Document parameters, returns, and exceptions
- Include examples of usage

### 5.2 Methodology Documentation

- Document econometric methods in detail
- Explain assumptions and limitations
- Provide references to academic literature

### 5.3 User Documentation

- Create user guides for running analyses
- Document configuration options
- Provide interpretation guidelines for results

## Conclusion

This methodology implementation plan provides a comprehensive roadmap for implementing the econometric methods required for the Yemen Market Integration project. By following this plan, we can ensure that the implementation is complete, accurate, and well-documented.