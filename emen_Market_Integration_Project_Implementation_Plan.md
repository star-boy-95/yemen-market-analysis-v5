# Yemen Market Integration Project: Implementation Plan

## Overview

This plan outlines the implementation of an econometric analysis project for studying market integration in conflict-affected Yemen, based on the provided research paper and data samples. The project uses threshold cointegration and spatial econometric techniques to analyze price transmission barriers and simulate potential policy interventions.

## Phase 1: Project Setup (Week 1)

### 1.1 Environment Setup

```bash
# Create a new directory for the project
mkdir -p yemen-market-integration
cd yemen-market-integration

# Initialize git repository
git init

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create initial project structure
mkdir -p data/{raw,processed}
mkdir -p src/{data,models,visualization,utils}
mkdir -p notebooks
mkdir -p tests
```

### 1.2 Project Structure Creation

```bash
# Create empty __init__.py files for proper Python packaging
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
```

### 1.3 Dependency Management

Create `requirements.txt`:

```
# Core data processing
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0

# Geospatial analysis
geopandas>=0.10.0
pysal>=2.4.0
folium>=0.12.0

# Econometrics
statsmodels>=0.13.0
arch>=5.0.0
pmdarima>=1.8.0  # For ARIMA models

# Spatial econometrics
spreg>=1.2.4     # Spatial regression models
libpysal>=4.5.0  # Core PySAL functionality
esda>=2.4.1      # Exploratory spatial data analysis

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
mapclassify>=2.4.0

# Development tools
jupyter>=1.0.0
pytest>=6.2.0
black>=21.5b2
flake8>=3.9.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 1.4 VS Code Configuration

Create `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

### 1.5 Git Configuration

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# VS Code
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json

# Data files
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# OS specific
.DS_Store
Thumbs.db
```

Initial commit:

```bash
git add .
git commit -m "Initial project setup"
```

## Phase 2: Data Processing (Week 2)

### 2.1 Create Data Loader Module

Create `src/data/loader.py`:

```python
"""
Data loading utilities for Yemen market integration analysis.
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path


class DataLoader:
    """Data loader for GeoJSON market data."""
    
    def __init__(self, data_path="./data"):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        data_path : str or Path
            Path to the data directory
        """
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
        self.processed_path = self.data_path / "processed"
    
    def load_geojson(self, filename):
        """
        Load GeoJSON data file into a GeoDataFrame.
        
        Parameters
        ----------
        filename : str
            Name of the GeoJSON file
            
        Returns
        -------
        geopandas.GeoDataFrame
            Loaded GeoJSON data
        """
        file_path = self.raw_path / filename
        return gpd.read_file(file_path)
    
    def save_processed_data(self, gdf, filename):
        """
        Save processed data to the processed directory.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Processed GeoDataFrame
        filename : str
            Output filename
        """
        output_path = self.processed_path / filename
        gdf.to_file(output_path, driver="GeoJSON")
        
    def split_by_exchange_regime(self, gdf):
        """
        Split data by exchange rate regime.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        tuple
            (north_gdf, south_gdf)
        """
        north = gdf[gdf['exchange_rate_regime'] == 'north']
        south = gdf[gdf['exchange_rate_regime'] == 'south']
        return north, south
    
    def get_time_series(self, gdf, admin_region, commodity):
        """
        Extract time series for specific region and commodity.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
        admin_region : str
            Administrative region name
        commodity : str
            Commodity name
            
        Returns
        -------
        pandas.DataFrame
            Filtered and sorted time series
        """
        mask = (gdf['admin1'] == admin_region) & (gdf['commodity'] == commodity)
        return gdf[mask].sort_values('date')
    
    def get_commodity_list(self, gdf):
        """
        Get list of available commodities in the data.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        list
            List of unique commodity names
        """
        return sorted(gdf['commodity'].unique())
    
    def get_region_list(self, gdf):
        """
        Get list of available administrative regions.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        list
            List of unique region names
        """
        return sorted(gdf['admin1'].unique())
```

### 2.2 Create Data Preprocessor Module

Create `src/data/preprocessor.py`:

```python
"""
Data preprocessing utilities for Yemen market integration analysis.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime


class DataPreprocessor:
    """Preprocess raw GeoJSON market data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
    
    def preprocess_geojson(self, gdf):
        """
        Preprocess the raw GeoJSON data.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Raw GeoJSON data
            
        Returns
        -------
        geopandas.GeoDataFrame
            Preprocessed data
        """
        # Make a copy to avoid modifying the original
        processed = gdf.copy()
        
        # Convert date strings to datetime objects
        processed['date'] = pd.to_datetime(processed['date'])
        
        # Handle missing values
        processed = self._handle_missing_values(processed)
        
        # Create additional features
        processed = self._create_features(processed)
        
        return processed
    
    def _handle_missing_values(self, gdf):
        """
        Handle missing values in the data.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        geopandas.GeoDataFrame
            DataFrame with handled missing values
        """
        # Check for missing values
        missing_values = gdf.isnull().sum()
        
        # Forward fill date-based missing values (for time series)
        for col in ['price', 'usdprice']:
            if missing_values[col] > 0:
                gdf[col] = gdf.groupby(['admin1', 'commodity'])[col].fillna(method='ffill')
                # If still missing, backward fill
                gdf[col] = gdf.groupby(['admin1', 'commodity'])[col].fillna(method='bfill')
        
        # For conflict data, fill remaining NAs with zeros
        conflict_cols = [col for col in gdf.columns if 'conflict' in col]
        for col in conflict_cols:
            if missing_values[col] > 0:
                gdf[col] = gdf[col].fillna(0)
        
        return gdf
    
    def _create_features(self, gdf):
        """
        Create additional features for analysis.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        geopandas.GeoDataFrame
            DataFrame with additional features
        """
        # Extract year and month
        gdf['year'] = gdf['date'].dt.year
        gdf['month'] = gdf['date'].dt.month
        
        # Create price log returns for volatility analysis
        gdf['price_log'] = np.log(gdf['price'])
        
        # Group by admin1, commodity, and date, then calculate log returns
        gdf = gdf.sort_values(['admin1', 'commodity', 'date'])
        gdf['price_return'] = gdf.groupby(['admin1', 'commodity'])['price_log'].diff()
        
        # Calculate price differential between exchange rate regimes
        # This requires a more complex approach - joining data across regimes
        # We'll implement this in a separate method
        
        return gdf
    
    def calculate_price_differentials(self, gdf):
        """
        Calculate price differentials between north and south exchange rate regimes.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with price differentials
        """
        # Get unique commodities and dates
        commodities = gdf['commodity'].unique()
        dates = gdf['date'].unique()
        
        # Create empty list to store differentials
        differentials = []
        
        # For each commodity and date, calculate north-south differential
        for commodity in commodities:
            for date in dates:
                # Filter data for this commodity and date
                mask = (gdf['commodity'] == commodity) & (gdf['date'] == date)
                data = gdf[mask]
                
                # Get average prices by regime
                north_price = data[data['exchange_rate_regime'] == 'north']['price'].mean()
                south_price = data[data['exchange_rate_regime'] == 'south']['price'].mean()
                
                # Calculate differential
                if not (np.isnan(north_price) or np.isnan(south_price)):
                    differential = {
                        'commodity': commodity,
                        'date': date,
                        'north_price': north_price,
                        'south_price': south_price,
                        'price_diff': north_price - south_price,
                        'price_diff_pct': (north_price - south_price) / south_price * 100
                    }
                    differentials.append(differential)
        
        return pd.DataFrame(differentials)
```

### 2.3 Implement Data Integration

Create `src/data/integration.py`:

```python
"""
Data integration utilities for combining data from different sources.
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path


class DataIntegrator:
    """Integrate data from multiple sources."""
    
    def __init__(self, data_path="./data"):
        """
        Initialize the data integrator.
        
        Parameters
        ----------
        data_path : str or Path
            Path to the data directory
        """
        self.data_path = Path(data_path)
    
    def integrate_conflict_data(self, market_gdf, conflict_file):
        """
        Integrate conflict data with market data.
        
        Parameters
        ----------
        market_gdf : geopandas.GeoDataFrame
            Market data
        conflict_file : str
            Filename for conflict data
            
        Returns
        -------
        geopandas.GeoDataFrame
            Integrated dataset
        """
        # This is a placeholder - actual implementation would depend
        # on the structure of your conflict data
        # For now, we assume the conflict data is already in the GeoJSON
        return market_gdf
    
    def integrate_exchange_rates(self, market_gdf, exchange_file):
        """
        Integrate exchange rate data with market data.
        
        Parameters
        ----------
        market_gdf : geopandas.GeoDataFrame
            Market data
        exchange_file : str
            Filename for exchange rate data
            
        Returns
        -------
        geopandas.GeoDataFrame
            Integrated dataset
        """
        # This is a placeholder - actual implementation would depend
        # on the structure of your exchange rate data
        # For now, we assume the exchange rate data is already in the GeoJSON
        return market_gdf
```

## Phase 3: Unit Root and Cointegration Testing (Week 3)

### 3.1 Create Unit Root Testing Module

Create `src/models/unit_root.py`:

```python
"""
Unit root testing module for time series analysis.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import arch.unitroot as unitroot


class UnitRootTester:
    """Perform unit root tests on time series data."""
    
    def __init__(self):
        """Initialize the unit root tester."""
        pass
    
    def test_adf(self, series, regression='c', lags=None):
        """
        Perform Augmented Dickey-Fuller test.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        regression : str, optional
            Constant and trend order to include in regression
            'c' : constant only (default)
            'ct' : constant and trend
            'ctt' : constant, and linear and quadratic trend
            'nc' : no constant, no trend
        lags : int, optional
            Number of lags to use in the ADF regression
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        result = adfuller(series, regression=regression, maxlag=lags)
        return {
            'statistic': result[0],
            'pvalue': result[1],
            'usedlag': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'icbest': result[5],
            'stationary': result[1] < 0.05
        }
    
    def test_adf_gls(self, series, lags=None):
        """
        Perform ADF-GLS test (Elliot, Rothenberg, Stock).
        
        Parameters
        ----------
        series : array_like
            The time series to test
        lags : int, optional
            Number of lags to use in the ADF regression
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        result = unitroot.DFGLS(series, lags=lags)
        return {
            'statistic': result.stat,
            'pvalue': result.pvalue,
            'critical_values': result.critical_values,
            'lags': result.lags,
            'stationary': result.pvalue < 0.05
        }
    
    def test_kpss(self, series, regression='c', lags=None):
        """
        Perform KPSS test for stationarity.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        regression : str, optional
            'c' : constant only (default)
            'ct' : constant and trend
        lags : int, optional
            Number of lags to use in the KPSS regression
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        result = kpss(series, regression=regression, nlags=lags)
        return {
            'statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[3],
            'stationary': result[1] > 0.05  # Note: opposite from ADF
        }
    
    def test_zivot_andrews(self, series):
        """
        Perform Zivot-Andrews test for unit root with structural break.
        
        Parameters
        ----------
        series : array_like
            The time series to test
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        result = unitroot.ZivotAndrews(series)
        return {
            'statistic': result.stat,
            'pvalue': result.pvalue,
            'critical_values': result.critical_values,
            'stationary': result.pvalue < 0.05,
            'breakpoint': result.breakpoint
        }
    
    def run_all_tests(self, series, lags=None):
        """
        Run all unit root tests.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        lags : int, optional
            Number of lags
            
        Returns
        -------
        dict
            Dictionary with results of all tests
        """
        return {
            'adf': self.test_adf(series, lags=lags),
            'adf_gls': self.test_adf_gls(series, lags=lags),
            'kpss': self.test_kpss(series, lags=lags),
            'zivot_andrews': self.test_zivot_andrews(series)
        }
```

### 3.2 Create Cointegration Testing Module

Create `src/models/cointegration.py`:

```python
"""
Cointegration testing module for time series analysis.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint


class CointegrationTester:
    """Perform cointegration tests on time series data."""
    
    def __init__(self):
        """Initialize the cointegration tester."""
        pass
    
    def test_engle_granger(self, y, x, trend='c', maxlag=None, autolag='AIC'):
        """
        Perform Engle-Granger two-step cointegration test.
        
        Parameters
        ----------
        y : array_like
            First time series
        x : array_like
            Second time series
        trend : str, optional
            'c' : constant (default)
            'ct' : constant and trend
            'ctt' : constant, linear and quadratic trend
            'nc' : no constant, no trend
        maxlag : int, optional
            Maximum lag to be used
        autolag : str, optional
            Method to use for automatic lag selection
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        result = coint(y, x, trend=trend, maxlag=maxlag, autolag=autolag)
        return {
            'statistic': result[0],
            'pvalue': result[1],
            'critical_values': result[2],
            'cointegrated': result[1] < 0.05
        }
    
    def test_johansen(self, data, det_order=0, k_ar_diff=1):
        """
        Perform Johansen cointegration test.
        
        Parameters
        ----------
        data : array_like
            Matrix of time series
        det_order : int, optional
            0 : no deterministic terms
            1 : constant term
            2 : constant and trend
        k_ar_diff : int, optional
            Number of lagged differences in the VAR model
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Extract trace and max eigenvalue statistics
        trace_stat = result.lr1
        trace_crit = result.cvt
        max_stat = result.lr2
        max_crit = result.cvm
        
        # Determine the cointegration rank
        rank_trace = sum(trace_stat > trace_crit[:, 0])  # 5% significance level
        rank_max = sum(max_stat > max_crit[:, 0])  # 5% significance level
        
        return {
            'trace_statistics': trace_stat,
            'trace_critical_values': trace_crit,
            'max_statistics': max_stat,
            'max_critical_values': max_crit,
            'rank_trace': rank_trace,
            'rank_max': rank_max,
            'cointegration_vectors': result.evec
        }
```

## Phase 4: Threshold Cointegration Development (Week 4-5)

### 4.1 Create Threshold Cointegration Module

Create `src/models/threshold.py`:

```python
"""
Threshold cointegration module for market integration analysis.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from arch.unitroot.cointegration import engle_granger


class ThresholdCointegration:
    """Threshold cointegration model implementation."""
    
    def __init__(self, data1, data2, max_lags=10):
        """
        Initialize the threshold cointegration model.
        
        Parameters
        ----------
        data1 : array_like
            First time series
        data2 : array_like
            Second time series
        max_lags : int, optional
            Maximum number of lags to consider
        """
        self.data1 = np.asarray(data1)
        self.data2 = np.asarray(data2)
        self.max_lags = max_lags
        self.results = None
    
    def estimate_cointegration(self):
        """
        Estimate the cointegration relationship.
        
        Returns
        -------
        dict
            Cointegration test results
        """
        result = engle_granger(self.data1, self.data2, trend='c', lags=self.max_lags)
        
        # Store the cointegration vector
        self.beta0 = result.coef[0]  # Intercept
        self.beta1 = result.coef[1]  # Slope
        
        # Calculate the equilibrium errors
        self.eq_errors = self.data1 - (self.beta0 + self.beta1 * self.data2)
        
        return {
            'statistic': result.stat,
            'pvalue': result.pvalue,
            'critical_values': result.critical_values,
            'cointegrated': result.pvalue < 0.05,
            'beta0': self.beta0,
            'beta1': self.beta1
        }
    
    def estimate_threshold(self, n_grid=300, trim=0.15):
        """
        Estimate the threshold parameter using grid search.
        
        Parameters
        ----------
        n_grid : int, optional
            Number of grid points
        trim : float, optional
            Trimming percentage
            
        Returns
        -------
        dict
            Threshold estimation results
        """
        # Make sure we have cointegration results
        if not hasattr(self, 'eq_errors'):
            self.estimate_cointegration()
        
        # Identify candidates for threshold
        sorted_errors = np.sort(self.eq_errors)
        lower_idx = int(len(sorted_errors) * trim)
        upper_idx = int(len(sorted_errors) * (1 - trim))
        candidates = sorted_errors[lower_idx:upper_idx]
        
        # Grid search for threshold
        if len(candidates) > n_grid:
            step = len(candidates) // n_grid
            candidates = candidates[::step]
        
        # Initialize variables for grid search
        best_ssr = np.inf
        best_threshold = None
        ssrs = []
        thresholds = []
        
        # Grid search
        for threshold in candidates:
            ssr = self._compute_ssr_for_threshold(threshold)
            ssrs.append(ssr)
            thresholds.append(threshold)
            
            if ssr < best_ssr:
                best_ssr = ssr
                best_threshold = threshold
        
        self.threshold = best_threshold
        self.ssr = best_ssr
        
        return {
            'threshold': best_threshold,
            'ssr': best_ssr,
            'all_thresholds': thresholds,
            'all_ssrs': ssrs
        }
    
    def _compute_ssr_for_threshold(self, threshold):
        """
        Compute sum of squared residuals for a given threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value
            
        Returns
        -------
        float
            Sum of squared residuals
        """
        # Indicator function
        below = self.eq_errors <= threshold
        above = ~below
        
        # Prepare data for regression
        y = np.diff(self.data1)
        X1 = np.column_stack([
            self.eq_errors[:-1] * below[:-1],
            self.eq_errors[:-1] * above[:-1]
        ])
        
        # Add lagged differences
        for lag in range(1, self.max_lags + 1):
            if lag < len(y):
                lag_d1 = np.roll(np.diff(self.data1), lag)[:-1]
                lag_d2 = np.roll(np.diff(self.data2), lag)[:-1]
                X1 = np.column_stack([X1, lag_d1, lag_d2])
        
        # Add constant
        X1 = sm.add_constant(X1)
        
        # Fit the model
        model = sm.OLS(y, X1)
        results = model.fit()
        
        return results.ssr
    
    def estimate_tvecm(self):
        """
        Estimate the Threshold Vector Error Correction Model.
        
        Returns
        -------
        dict
            TVECM estimation results
        """
        # Make sure we have a threshold
        if not hasattr(self, 'threshold'):
            self.estimate_threshold()
        
        # Indicator function
        below = self.eq_errors <= self.threshold
        above = ~below
        
        # Prepare data for regression
        y1 = np.diff(self.data1)
        y2 = np.diff(self.data2)
        
        X1 = np.column_stack([
            self.eq_errors[:-1] * below[:-1],
            self.eq_errors[:-1] * above[:-1]
        ])
        
        # Add lagged differences
        for lag in range(1, self.max_lags + 1):
            if lag < len(y1):
                lag_d1 = np.roll(np.diff(self.data1), lag)[:-1]
                lag_d2 = np.roll(np.diff(self.data2), lag)[:-1]
                X1 = np.column_stack([X1, lag_d1, lag_d2])
        
        # Add constant
        X1 = sm.add_constant(X1)
        
        # Fit the models
        model1 = sm.OLS(y1, X1)
        model2 = sm.OLS(y2, X1)
        
        results1 = model1.fit()
        results2 = model2.fit()
        
        # Extract adjustment speeds
        adj_below_1 = results1.params[1]
        adj_above_1 = results1.params[2]
        adj_below_2 = results2.params[1]
        adj_above_2 = results2.params[2]
        
        self.results = {
            'equation1': results1,
            'equation2': results2,
            'adjustment_below_1': adj_below_1,
            'adjustment_above_1': adj_above_1,
            'adjustment_below_2': adj_below_2,
            'adjustment_above_2': adj_above_2,
            'threshold': self.threshold,
            'cointegration_beta': self.beta1
        }
        
        return self.results
```

### 4.2 Create Threshold VECM Module

Create `src/models/threshold_vecm.py`:

```python
"""
Threshold Vector Error Correction Model implementation.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM


class ThresholdVECM:
    """
    Threshold Vector Error Correction Model (TVECM) implementation.
    
    This class implements a two-regime threshold VECM following
    Hansen & Seo (2002) methodology.
    """
    
    def __init__(self, data, k_ar_diff=1, deterministic="ci"):
        """
        Initialize the TVECM model.
        
        Parameters
        ----------
        data : array_like or pandas DataFrame
            The endogenous variables
        k_ar_diff : int, optional
            Number of lagged differences in the model
        deterministic : str, optional
            "n" - no deterministic terms
            "co" - constant outside the cointegration relation
            "ci" - constant inside the cointegration relation
            "lo" - linear trend outside the cointegration relation
            "li" - linear trend inside the cointegration relation
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.k_ar_diff = k_ar_diff
        self.deterministic = deterministic
        self.results = None
    
    def estimate_linear_vecm(self):
        """
        Estimate the linear VECM model (no threshold).
        
        Returns
        -------
        statsmodels.tsa.vector_ar.vecm.VECMResults
            Linear VECM estimation results
        """
        self.linear_model = VECM(
            self.data, 
            k_ar_diff=self.k_ar_diff, 
            deterministic=self.deterministic
        )
        self.linear_results = self.linear_model.fit()
        return self.linear_results
    
    def grid_search_threshold(self, trim=0.15, n_grid=300, verbose=False):
        """
        Perform grid search to find the optimal threshold.
        
        Parameters
        ----------
        trim : float, optional
            Trimming percentage
        n_grid : int, optional
            Number of grid points
        verbose : bool, optional
            Whether to print progress
            
        Returns
        -------
        dict
            Threshold estimation results
        """
        # Ensure linear VECM is estimated
        if not hasattr(self, 'linear_results'):
            self.estimate_linear_vecm()
        
        # Get cointegration relation
        beta = self.linear_results.beta
        
        # Calculate equilibrium errors
        y = self.data.values
        if self.deterministic == "ci":
            z = np.column_stack([np.ones(len(y)), y])[:, :-1]
        else:
            z = y
        
        eq_errors = z @ beta
        
        # Identify candidates for threshold
        sorted_errors = np.sort(eq_errors.flatten())
        lower_idx = int(len(sorted_errors) * trim)
        upper_idx = int(len(sorted_errors) * (1 - trim))
        candidates = sorted_errors[lower_idx:upper_idx]
        
        # Grid search for threshold
        if len(candidates) > n_grid:
            step = len(candidates) // n_grid
            candidates = candidates[::step]
        
        # Initialize variables for grid search
        best_llf = -np.inf
        best_threshold = None
        llfs = []
        thresholds = []
        
        # Grid search
        for i, threshold in enumerate(candidates):
            if verbose and (i % 10 == 0):
                print(f"Searching threshold: {i+1}/{len(candidates)}")
            
            llf = self._compute_llf_for_threshold(threshold, eq_errors)
            llfs.append(llf)
            thresholds.append(threshold)
            
            if llf > best_llf:
                best_llf = llf
                best_threshold = threshold
        
        self.threshold = best_threshold
        self.llf = best_llf
        
        return {
            'threshold': best_threshold,
            'llf': best_llf,
            'all_thresholds': thresholds,
            'all_llfs': llfs
        }
    
    def _compute_llf_for_threshold(self, threshold, eq_errors):
        """
        Compute log-likelihood for a given threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value
        eq_errors : array_like
            Equilibrium errors
            
        Returns
        -------
        float
            Log-likelihood
        """
        # This is a simplified implementation
        # A full implementation would be more complex
        
        # Indicator function
        below = eq_errors <= threshold
        above = ~below
        
        # Prepare data
        y = np.diff(self.data.values, axis=0)
        
        # Create matrices for the two regimes
        X_below = np.column_stack([
            np.ones(len(y)) * below[:-1],
            eq_errors[:-1] * below[:-1]
        ])
        
        X_above = np.column_stack([
            np.ones(len(y)) * above[:-1],
            eq_errors[:-1] * above[:-1]
        ])
        
        # Add lagged differences
        for lag in range(1, self.k_ar_diff + 1):
            if lag < len(y):
                lag_diff = np.roll(y, lag, axis=0)[:-1]
                X_below = np.column_stack([X_below, lag_diff * below[:-1, np.newaxis]])
                X_above = np.column_stack([X_above, lag_diff * above[:-1, np.newaxis]])
        
        # Combine matrices
        X = np.column_stack([X_below, X_above])
        
        # Fit model
        model = sm.OLS(y, X)
        results = model.fit()
        
        return results.llf
    
    def estimate_tvecm(self):
        """
        Estimate the Threshold VECM.
        
        Returns
        -------
        dict
            TVECM estimation results
        """
        # Ensure threshold is estimated
        if not hasattr(self, 'threshold'):
            self.grid_search_threshold()
        
        # Get cointegration relation
        beta = self.linear_results.beta
        
        # Calculate equilibrium errors
        y = self.data.values
        if self.deterministic == "ci":
            z = np.column_stack([np.ones(len(y)), y])[:, :-1]
        else:
            z = y
        
        eq_errors = z @ beta
        
        # Indicator function
        below = eq_errors <= self.threshold
        above = ~below
        
        # Prepare data
        y_diff = np.diff(self.data.values, axis=0)
        
        # Create matrices for the two regimes
        X_below = np.column_stack([
            np.ones(len(y_diff)) * below[:-1],
            eq_errors[:-1] * below[:-1]
        ])
        
        X_above = np.column_stack([
            np.ones(len(y_diff)) * above[:-1],
            eq_errors[:-1] * above[:-1]
        ])
        
        # Add lagged differences
        for lag in range(1, self.k_ar_diff + 1):
            if lag < len(y_diff):
                lag_diff = np.roll(y_diff, lag, axis=0)[:-1]
                X_below = np.column_stack([X_below, lag_diff * below[:-1, np.newaxis]])
                X_above = np.column_stack([X_above, lag_diff * above[:-1, np.newaxis]])
        
        # Combine matrices
        X = np.column_stack([X_below, X_above])
        
        # Fit model
        model = sm.OLS(y_diff, X)
        results = model.fit()
        
        # Extract parameters for each regime
        n_vars = self.data.shape[1]
        n_params_per_regime = 2 + self.k_ar_diff * n_vars
        
        params_below = results.params[:n_params_per_regime]
        params_above = results.params[n_params_per_regime:]
        
        # Extract adjustment speeds
        adj_below = params_below[1:1+n_vars]
        adj_above = params_above[1:1+n_vars]
        
        self.results = {
            'model': results,
            'threshold': self.threshold,
            'cointegration_beta': beta,
            'adjustment_below': adj_below,
            'adjustment_above': adj_above,
            'params_below': params_below,
            'params_above': params_above
        }
        
        return self.results
```

## Phase 5: Spatial Econometrics Development (Week 6)

### 5.1 Create Spatial Analysis Module

Create `src/models/spatial.py`:

```python
"""
Spatial econometric models for market integration analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN, Kernel, W
from esda.moran import Moran
from spreg import OLS, ML_Lag, ML_Error


class SpatialEconometrics:
    """Spatial econometric analysis for market integration."""
    
    def __init__(self, gdf):
        """
        Initialize with a GeoDataFrame.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data
        """
        self.gdf = gdf
        self.weights = None
    
    def create_weight_matrix(self, k=5, conflict_adjusted=True):
        """
        Create spatial weights matrix.
        
        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors
        conflict_adjusted : bool, optional
            If True, adjust weights by conflict intensity
            
        Returns
        -------
        libpysal.weights.W
            Spatial weights matrix
        """
        # Basic KNN weights
        knn = KNN.from_dataframe(self.gdf, k=k)
        
        if conflict_adjusted:
            # Adjust weights based on conflict intensity
            # Higher conflict = lower weight (more economic distance)
            adj_weights = {}
            for i, neighbors in knn.neighbors.items():
                weights = []
                for j in neighbors:
                    # Base weight (inverse distance)
                    base_weight = knn.weights[i][knn.neighbors[i].index(j)]
                    
                    # Get conflict intensity for both regions
                    conflict_i = self.gdf.iloc[i]['conflict_intensity_normalized']
                    conflict_j = self.gdf.iloc[j]['conflict_intensity_normalized']
                    
                    # Average conflict intensity along the path
                    avg_conflict = (conflict_i + conflict_j) / 2
                    
                    # Adjust weight: higher conflict = lower weight
                    adjusted_weight = base_weight * (1 - avg_conflict)
                    weights.append(adjusted_weight)
                
                adj_weights[i] = weights
            
            # Create new weight matrix with adjusted weights
            self.weights = W(knn.neighbors, adj_weights)
        else:
            self.weights = knn
        
        return self.weights
    
    def moran_i_test(self, variable):
        """
        Test for spatial autocorrelation using Moran's I.
        
        Parameters
        ----------
        variable : str
            Column name in GeoDataFrame to test
            
        Returns
        -------
        dict
            Moran's I test results
        """
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Calculate Moran's I
        moran = Moran(self.gdf[variable], self.weights)
        return {
            'I': moran.I,
            'p_norm': moran.p_norm,
            'p_sim': moran.p_sim,
            'z_norm': moran.z_norm
        }
    
    def spatial_lag_model(self, y_col, x_cols):
        """
        Estimate a spatial lag model.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name
        x_cols : list
            List of independent variable column names
            
        Returns
        -------
        spreg.ML_Lag
            Spatial lag model results
        """
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Prepare data
        y = self.gdf[y_col].values
        X = self.gdf[x_cols].values
        
        # Estimate model
        model = ML_Lag(y, X, self.weights, name_y=y_col, name_x=x_cols)
        return model
    
    def spatial_error_model(self, y_col, x_cols):
        """
        Estimate a spatial error model.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name
        x_cols : list
            List of independent variable column names
            
        Returns
        -------
        spreg.ML_Error
            Spatial error model results
        """
        if self.weights is None:
            raise ValueError("Weight matrix not created. Call create_weight_matrix first.")
        
        # Prepare data
        y = self.gdf[y_col].values
        X = self.gdf[x_cols].values
        
        # Estimate model
        model = ML_Error(y, X, self.weights, name_y=y_col, name_x=x_cols)
        return model
```

## Phase 6: Policy Simulation Development (Week 7)

### 6.1 Create Simulation Module

Create `src/models/simulation.py`:

```python
"""
Policy simulation models for market integration analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd


class MarketIntegrationSimulation:
    """Simulate policy interventions for market integration."""
    
    def __init__(self, data, threshold_model=None, spatial_model=None):
        """
        Initialize the simulation model.
        
        Parameters
        ----------
        data : pandas.DataFrame or geopandas.GeoDataFrame
            Market data
        threshold_model : ThresholdVECM, optional
            Estimated threshold model
        spatial_model : SpatialEconometrics, optional
            Estimated spatial model
        """
        self.data = data.copy()
        self.threshold_model = threshold_model
        self.spatial_model = spatial_model
        self.results = {}
    
    def simulate_exchange_rate_unification(self):
        """
        Simulate exchange rate unification by setting differential to zero.
        
        Returns
        -------
        dict
            Simulation results
        """
        # Create a copy of data with unified exchange rate
        unified_data = self.data.copy()
        
        # Get unique dates
        dates = unified_data['date'].unique()
        
        # For each date, calculate average exchange rate and apply it to all markets
        for date in dates:
            mask = unified_data['date'] == date
            avg_rate = unified_data.loc[mask, 'usdprice'].mean()
            unified_data.loc[mask, 'usdprice'] = avg_rate
        
        # Re-estimate threshold model with unified data
        if self.threshold_model is not None:
            # This would depend on the specific implementation
            # of your threshold model
            pass
        
        # Re-estimate spatial model with unified data
        if self.spatial_model is not None:
            # This would depend on the specific implementation
            # of your spatial model
            pass
        
        self.results['exchange_rate_unification'] = {
            'data': unified_data,
            # Additional results would be stored here
        }
        
        return self.results['exchange_rate_unification']
    
    def simulate_improved_connectivity(self, reduction_factor=0.5):
        """
        Simulate improved connectivity by reducing conflict-related barriers.
        
        Parameters
        ----------
        reduction_factor : float, optional
            Factor to reduce conflict intensity by (0-1)
            
        Returns
        -------
        dict
            Simulation results
        """
        if self.spatial_model is None:
            raise ValueError("Spatial model required for connectivity simulation")
        
        # Create a copy of data with reduced conflict intensity
        reduced_conflict_data = self.data.copy()
        
        # Reduce conflict intensity by the specified factor
        conflict_cols = [col for col in reduced_conflict_data.columns if 'conflict_intensity' in col]
        for col in conflict_cols:
            reduced_conflict_data[col] = reduced_conflict_data[col] * reduction_factor
        
        # Re-create weight matrix with reduced conflict
        # This would depend on the specific implementation
        # of your spatial model
        
        # Re-estimate spatial model
        # This would depend on the specific implementation
        # of your spatial model
        
        self.results['improved_connectivity'] = {
            'data': reduced_conflict_data,
            # Additional results would be stored here
        }
        
        return self.results['improved_connectivity']
```

## Phase 7: Visualization Tools Development (Week 8)

### 7.1 Create Time Series Visualization Module

Create `src/visualization/time_series.py`:

```python
"""
Time series visualization utilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TimeSeriesVisualizer:
    """Create time series visualizations for market data."""
    
    def __init__(self, style='seaborn'):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        style : str, optional
            Matplotlib style to use
        """
        plt.style.use(style)
    
    def plot_price_series(self, df, price_col='price', date_col='date', 
                          group_col=None, title=None, figsize=(12, 6)):
        """
        Plot price time series.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data to plot
        price_col : str, optional
            Column name for price
        date_col : str, optional
            Column name for date
        group_col : str, optional
            Column name for grouping (e.g., 'commodity', 'admin1')
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if group_col is not None:
            for name, group in df.groupby(group_col):
                ax.plot(group[date_col], group[price_col], label=name)
            ax.legend()
        else:
            ax.plot(df[date_col], df[price_col])
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(title or 'Price Time Series')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_price_differentials(self, df, date_col='date', 
                                north_col='north_price', south_col='south_price',
                                diff_col='price_diff', group_col=None,
                                title=None, figsize=(12, 8)):
        """
        Plot price differentials between north and south.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data with price differentials
        date_col : str, optional
            Column name for date
        north_col : str, optional
            Column name for north price
        south_col : str, optional
            Column name for south price
        diff_col : str, optional
            Column name for price differential
        group_col : str, optional
            Column name for grouping (e.g., 'commodity')
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        if group_col is not None:
            # Create a subplot for each group
            unique_groups = df[group_col].unique()
            fig, axes = plt.subplots(len(unique_groups), 1, figsize=figsize, sharex=True)
            
            for i, group_val in enumerate(unique_groups):
                group_df = df[df[group_col] == group_val]
                ax = axes[i]
                
                # Plot prices
                ax.plot(group_df[date_col], group_df[north_col], label='North')
                ax.plot(group_df[date_col], group_df[south_col], label='South')
                
                # Plot differential on secondary axis
                ax2 = ax.twinx()
                ax2.plot(group_df[date_col], group_df[diff_col], 'r--', label='Differential')
                
                ax.set_ylabel('Price')
                ax2.set_ylabel('Differential')
                ax.set_title(f'{group_val}')
                
                # Only show legend on first subplot
                if i == 0:
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            # Create a single plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot prices
            ax.plot(df[date_col], df[north_col], label='North')
            ax.plot(df[date_col], df[south_col], label='South')
            
            # Plot differential on secondary axis
            ax2 = ax.twinx()
            ax2.plot(df[date_col], df[diff_col], 'r--', label='Differential')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax2.set_ylabel('Differential')
            ax.set_title(title or 'Price Differentials: North vs South')
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_time_series(self, df, price_col='price', date_col='date',
                                    group_col=None, title=None):
        """
        Create an interactive time series plot using Plotly.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data to plot
        price_col : str, optional
            Column name for price
        date_col : str, optional
            Column name for date
        group_col : str, optional
            Column name for grouping
        title : str, optional
            Plot title
            
        Returns
        -------
        plotly.graph_objects.Figure
            The created figure
        """
        if group_col is not None:
            fig = px.line(df, x=date_col, y=price_col, color=group_col,
                        title=title or f'Price Time Series by {group_col}')
        else:
            fig = px.line(df, x=date_col, y=price_col,
                        title=title or 'Price Time Series')
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title=group_col or '',
            hovermode='closest'
        )
        
        return fig
```

### 7.2 Create Spatial Visualization Module

Create `src/visualization/market_maps.py`:

```python
"""
Spatial visualization utilities for market data.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import folium
from folium.plugins import MarkerCluster
import mapclassify


class MarketMapVisualizer:
    """Create spatial visualizations for market data."""
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def plot_static_map(self, gdf, column=None, cmap='viridis', figsize=(12, 10),
                         title=None, add_basemap=True, scheme='quantiles',
                         k=5, legend=True):
        """
        Create a static map using matplotlib.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data to plot
        column : str, optional
            Column name to use for choropleth coloring
        cmap : str, optional
            Colormap name
        figsize : tuple, optional
            Figure size
        title : str, optional
            Map title
        add_basemap : bool, optional
            Whether to add a basemap
        scheme : str, optional
            Classification scheme for choropleth map
        k : int, optional
            Number of classes for classification
        legend : bool, optional
            Whether to include a legend
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if column is not None:
            # Create choropleth map
            gdf.plot(column=column, cmap=cmap, ax=ax, scheme=scheme, k=k, legend=legend)
        else:
            # Simple plot without coloring
            gdf.plot(ax=ax)
        
        if add_basemap:
            # Add basemap if requested and if the CRS is appropriate
            try:
                ctx.add_basemap(ax, crs=gdf.crs)
            except Exception as e:
                print(f"Could not add basemap: {e}")
        
        ax.set_title(title or 'Market Map')
        ax.set_axis_off()
        
        return fig
    
    def create_interactive_map(self, gdf, column=None, popup_cols=None,
                             title=None, tiles='OpenStreetMap'):
        """
        Create an interactive map using folium.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data to map
        column : str, optional
            Column to use for coloring
        popup_cols : list, optional
            Columns to include in popups
        title : str, optional
            Map title
        tiles : str, optional
            Basemap tiles
            
        Returns
        -------
        folium.Map
            The created map
        """
        # Ensure gdf is in WGS84 for folium
        if gdf.crs and gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        
        # Get the center of the data
        center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
        
        # Create the map
        m = folium.Map(location=center, zoom_start=7, tiles=tiles)
        
        if title:
            # Add a title
            title_html = f'''
                <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
        
        if gdf.geometry.type.iloc[0] == 'Point':
            # For point data, use markers
            if column is not None:
                # Create a colormap
                min_val = gdf[column].min()
                max_val = gdf[column].max()
                
                # Function to determine marker color
                def get_color(value):
                    # Simple linear scale from green to red
                    norm_val = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    return f'#{int(255 * (1-norm_val)):02x}{int(255 * norm_val):02x}00'
                
                # Add marker cluster
                marker_cluster = MarkerCluster().add_to(m)
                
                for idx, row in gdf.iterrows():
                    # Prepare popup HTML
                    popup_html = ""
                    if popup_cols:
                        popup_html = "<table>"
                        for col in popup_cols:
                            popup_html += f"<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>"
                        popup_html += "</table>"
                    
                    # Create marker
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=folium.Popup(popup_html, max_width=300) if popup_html else None,
                        icon=folium.Icon(color='white', icon_color=get_color(row[column]), icon='info-sign')
                    ).add_to(marker_cluster)
            else:
                # Simple markers without coloring
                marker_cluster = MarkerCluster().add_to(m)
                
                for idx, row in gdf.iterrows():
                    # Prepare popup HTML
                    popup_html = ""
                    if popup_cols:
                        popup_html = "<table>"
                        for col in popup_cols:
                            popup_html += f"<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>"
                        popup_html += "</table>"
                    
                    # Create marker
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=folium.Popup(popup_html, max_width=300) if popup_html else None
                    ).add_to(marker_cluster)
        else:
            # For polygon data, use choropleth
            if column is not None:
                # Add choropleth layer
                folium.Choropleth(
                    geo_data=gdf,
                    name='choropleth',
                    data=gdf,
                    columns=[gdf.index.name or 'index', column],
                    key_on=f"feature.{gdf.index.name or 'id'}",
                    fill_color='YlOrRd',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name=column
                ).add_to(m)
            else:
                # Simple polygon layer without coloring
                folium.GeoJson(
                    gdf,
                    name='geojson'
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def plot_price_heatmap(self, gdf, commodity=None, date=None, price_col='price',
                          cmap='YlOrRd', figsize=(12, 10), title=None):
        """
        Create a price heatmap for a specific commodity and date.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial market data
        commodity : str, optional
            Commodity to filter for
        date : str or datetime, optional
            Date to filter for
        price_col : str, optional
            Column containing price values
        cmap : str, optional
            Colormap name
        figsize : tuple, optional
            Figure size
        title : str, optional
            Map title
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        # Filter data
        filtered = gdf.copy()
        
        if commodity is not None:
            filtered = filtered[filtered['commodity'] == commodity]
        
        if date is not None:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            filtered = filtered[filtered['date'] == date]
        
        # Create the map
        fig, ax = plt.subplots(figsize=figsize)
        
        filtered.plot(column=price_col, cmap=cmap, ax=ax, legend=True)
        
        # Try to add basemap
        try:
            ctx.add_basemap(ax, crs=filtered.crs)
        except Exception as e:
            print(f"Could not add basemap: {e}")
        
        # Set title
        if title:
            ax.set_title(title)
        elif commodity and date:
            ax.set_title(f'Price of {commodity} on {date.strftime("%Y-%m-%d")}')
        elif commodity:
            ax.set_title(f'Price of {commodity}')
        else:
            ax.set_title('Price Map')
        
        ax.set_axis_off()
        
        return fig
```

## Phase 8: Testing and Documentation (Week 9)

### 8.1 Create Unit Tests

Create `tests/test_data.py`:

```python
"""
Unit tests for data modules.
"""
import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor


class TestDataLoader(unittest.TestCase):
    """Tests for the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader('./data')
        
        # Create a temporary test GeoDataFrame
        self.test_data = gpd.GeoDataFrame({
            'admin1': ['abyan', 'abyan', 'aden', 'aden'],
            'commodity': ['beans', 'beans', 'beans', 'rice'],
            'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-01-01', '2020-02-01']),
            'price': [100, 110, 120, 130],
            'usdprice': [1.0, 1.1, 1.2, 1.3],
            'exchange_rate_regime': ['north', 'north', 'south', 'south'],
            'geometry': [
                gpd.points_from_xy([45.0, 45.0, 46.0, 46.0], [13.0, 13.0, 12.0, 12.0])[i]
                for i in range(4)
            ]
        })
    
    def test_split_by_exchange_regime(self):
        """Test splitting by exchange rate regime."""
        north, south = self.loader.split_by_exchange_regime(self.test_data)
        
        self.assertEqual(len(north), 2)
        self.assertEqual(len(south), 2)
        self.assertTrue(all(north['exchange_rate_regime'] == 'north'))
        self.assertTrue(all(south['exchange_rate_regime'] == 'south'))
    
    def test_get_time_series(self):
        """Test getting time series for a specific region and commodity."""
        ts = self.loader.get_time_series(self.test_data, 'abyan', 'beans')
        
        self.assertEqual(len(ts), 2)
        self.assertTrue(all(ts['admin1'] == 'abyan'))
        self.assertTrue(all(ts['commodity'] == 'beans'))
        self.assertTrue(ts['date'].is_monotonic)
    
    def test_get_commodity_list(self):
        """Test getting list of commodities."""
        commodities = self.loader.get_commodity_list(self.test_data)
        
        self.assertEqual(len(commodities), 2)
        self.assertIn('beans', commodities)
        self.assertIn('rice', commodities)
    
    def test_get_region_list(self):
        """Test getting list of regions."""
        regions = self.loader.get_region_list(self.test_data)
        
        self.assertEqual(len(regions), 2)
        self.assertIn('abyan', regions)
        self.assertIn('aden', regions)


class TestDataPreprocessor(unittest.TestCase):
    """Tests for the DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create a temporary test GeoDataFrame
        self.test_data = gpd.GeoDataFrame({
            'admin1': ['abyan', 'abyan', 'aden', 'aden'],
            'commodity': ['beans', 'beans', 'beans', 'rice'],
            'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-01-01', '2020-02-01']),
            'price': [100, 110, 120, 130],
            'usdprice': [1.0, 1.1, 1.2, 1.3],
            'exchange_rate_regime': ['north', 'north', 'south', 'south'],
            'conflict_intensity': [0.5, 0.6, 0.7, 0.8],
            'geometry': [
                gpd.points_from_xy([45.0, 45.0, 46.0, 46.0], [13.0, 13.0, 12.0, 12.0])[i]
                for i in range(4)
            ]
        })
        
        # Add some missing values
        self.test_data_with_na = self.test_data.copy()
        self.test_data_with_na.loc[1, 'price'] = np.nan
        self.test_data_with_na.loc[2, 'conflict_intensity'] = np.nan
    
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        processed = self.preprocessor._handle_missing_values(self.test_data_with_na)
        
        # Check if NaN values were filled
        self.assertFalse(processed['price'].isna().any())
        self.assertFalse(processed['conflict_intensity'].isna().any())
    
    def test_create_features(self):
        """Test creation of additional features."""
        processed = self.preprocessor._create_features(self.test_data)
        
        # Check if new columns were created
        self.assertIn('year', processed.columns)
        self.assertIn('month', processed.columns)
        self.assertIn('price_log', processed.columns)
        self.assertIn('price_return', processed.columns)
        
        # Check values
        self.assertEqual(processed['year'].iloc[0], 2020)
        self.assertEqual(processed['month'].iloc[0], 1)
    
    def test_calculate_price_differentials(self):
        """Test calculation of price differentials."""
        differentials = self.preprocessor.calculate_price_differentials(self.test_data)
        
        # Check if the DataFrame has the expected columns
        self.assertIn('commodity', differentials.columns)
        self.assertIn('date', differentials.columns)
        self.assertIn('north_price', differentials.columns)
        self.assertIn('south_price', differentials.columns)
        self.assertIn('price_diff', differentials.columns)
        self.assertIn('price_diff_pct', differentials.columns)
        
        # Check if the calculations are correct
        beans_row = differentials[differentials['commodity'] == 'beans'].iloc[0]
        self.assertEqual(beans_row['north_price'], 105.0)  # (100 + 110) / 2
        self.assertEqual(beans_row['south_price'], 120.0)
        self.assertEqual(beans_row['price_diff'], -15.0)  # 105 - 120


if __name__ == '__main__':
    unittest.main()
```

Create `tests/test_models.py`:

```python
"""
Unit tests for model modules.
"""
import unittest
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester
from src.models.threshold import ThresholdCointegration
from src.models.spatial import SpatialEconometrics


class TestUnitRootTester(unittest.TestCase):
    """Tests for the UnitRootTester class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = UnitRootTester()
        
        # Create test time series
        np.random.seed(42)
        # Non-stationary series (random walk)
        self.nonstationary = np.cumsum(np.random.normal(0, 1, 100))
        # Stationary series
        self.stationary = np.random.normal(0, 1, 100)
    
    def test_adf_test(self):
        """Test ADF test."""
        # Test on nonstationary series
        result_ns = self.tester.test_adf(self.nonstationary)
        
        # Test on stationary series
        result_s = self.tester.test_adf(self.stationary)
        
        # Check expected outcomes
        self.assertFalse(result_ns['stationary'])
        self.assertTrue(result_s['stationary'])
    
    def test_kpss_test(self):
        """Test KPSS test."""
        # Test on nonstationary series
        result_ns = self.tester.test_kpss(self.nonstationary)
        
        # Test on stationary series
        result_s = self.tester.test_kpss(self.stationary)
        
        # Check expected outcomes
        self.assertFalse(result_ns['stationary'])
        self.assertTrue(result_s['stationary'])
    
    def test_run_all_tests(self):
        """Test running all tests."""
        all_results = self.tester.run_all_tests(self.stationary)
        
        # Check if all test results are present
        self.assertIn('adf', all_results)
        self.assertIn('adf_gls', all_results)
        self.assertIn('kpss', all_results)
        self.assertIn('zivot_andrews', all_results)


class TestCointegrationTester(unittest.TestCase):
    """Tests for the CointegrationTester class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = CointegrationTester()
        
        # Create cointegrated series
        np.random.seed(42)
        # Common trend
        common_trend = np.cumsum(np.random.normal(0, 1, 100))
        # Two series with the same trend but different noise
        self.series1 = common_trend + np.random.normal(0, 0.5, 100)
        self.series2 = 2 * common_trend + np.random.normal(0, 0.5, 100)
        
        # Create non-cointegrated series
        self.series3 = np.cumsum(np.random.normal(0, 1, 100))
        self.series4 = np.cumsum(np.random.normal(0, 1, 100))
    
    def test_engle_granger(self):
        """Test Engle-Granger cointegration test."""
        # Test cointegrated series
        result_co = self.tester.test_engle_granger(self.series1, self.series2)
        
        # Test non-cointegrated series
        result_nonco = self.tester.test_engle_granger(self.series3, self.series4)
        
        # Check expected outcomes
        self.assertTrue(result_co['cointegrated'])
        self.assertFalse(result_nonco['cointegrated'])
    
    def test_johansen(self):
        """Test Johansen cointegration test."""
        # Create data matrix
        data_co = np.column_stack([self.series1, self.series2])
        data_nonco = np.column_stack([self.series3, self.series4])
        
        # Test cointegrated series
        result_co = self.tester.test_johansen(data_co)
        
        # Test non-cointegrated series
        result_nonco = self.tester.test_johansen(data_nonco)
        
        # Check expected outcomes
        self.assertEqual(result_co['rank_trace'], 1)  # Should find 1 cointegrating relation
        self.assertEqual(result_nonco['rank_trace'], 0)  # Should find 0 cointegrating relations


class TestSpatialEconometrics(unittest.TestCase):
    """Tests for the SpatialEconometrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test GeoDataFrame
        self.test_data = gpd.GeoDataFrame({
            'admin1': ['abyan', 'aden', 'sanaa', 'taiz'],
            'price': [100, 120, 110, 90],
            'conflict_intensity_normalized': [0.5, 0.7, 0.6, 0.4],
            'geometry': gpd.points_from_xy([45.0, 46.0, 47.0, 44.0], [13.0, 12.0, 15.0, 14.0])
        })
        
        self.spatial = SpatialEconometrics(self.test_data)
    
    def test_create_weight_matrix(self):
        """Test creation of weight matrix."""
        # Test without conflict adjustment
        w = self.spatial.create_weight_matrix(k=2, conflict_adjusted=False)
        
        # Check basic properties
        self.assertEqual(len(w), 4)  # 4 regions
        self.assertEqual(w.n, 4)
        
        # Test with conflict adjustment
        w_adj = self.spatial.create_weight_matrix(k=2, conflict_adjusted=True)
        
        # Check if weights were adjusted
        # In an adjusted matrix, weights should be lower due to conflict
        self.assertEqual(w_adj.n, 4)
        
        # Average weight should be lower with conflict adjustment
        avg_weight_orig = sum(sum(nn) for nn in w.weights.values()) / sum(len(nn) for nn in w.weights.values())
        avg_weight_adj = sum(sum(nn) for nn in w_adj.weights.values()) / sum(len(nn) for nn in w_adj.weights.values())
        
        self.assertLess(avg_weight_adj, avg_weight_orig)


if __name__ == '__main__':
    unittest.main()
```

### 8.2 Create Jupyter Notebooks

Create `notebooks/01_exploratory_analysis.ipynb`:

```python
# Sample notebook content outline
# This would be expanded with actual code and analysis

# Import necessary libraries
import sys
import os
# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.visualization.time_series import TimeSeriesVisualizer
from src.visualization.market_maps import MarketMapVisualizer

# Load data
loader = DataLoader()
gdf = loader.load_geojson('unified_data.geojson')

# Preprocess data
preprocessor = DataPreprocessor()
processed_gdf = preprocessor.preprocess_geojson(gdf)

# Explore data
# Summary statistics
processed_gdf.describe()

# Number of observations by region and commodity
processed_gdf.groupby(['admin1', 'commodity']).size()

# Visualize price trends
time_vis = TimeSeriesVisualizer()
fig = time_vis.plot_price_series(
    processed_gdf[processed_gdf['commodity'] == 'beans (kidney red)'],
    group_col='admin1',
    title='Price Trends for Kidney Beans by Region'
)

# Explore price differentials
differentials = preprocessor.calculate_price_differentials(processed_gdf)
fig = time_vis.plot_price_differentials(
    differentials[differentials['commodity'] == 'beans (kidney red)'],
    title='Price Differentials: North vs South (Kidney Beans)'
)

# Spatial visualization
map_vis = MarketMapVisualizer()
fig = map_vis.plot_static_map(
    processed_gdf[
        (processed_gdf['commodity'] == 'beans (kidney red)') & 
        (processed_gdf['date'] == processed_gdf['date'].max())
    ],
    column='price',
    title='Price Distribution of Kidney Beans (Latest Date)'
)

# Create interactive map
m = map_vis.create_interactive_map(
    processed_gdf[
        (processed_gdf['commodity'] == 'beans (kidney red)') & 
        (processed_gdf['date'] == processed_gdf['date'].max())
    ],
    column='price',
    popup_cols=['admin1', 'price', 'usdprice', 'conflict_intensity_normalized'],
    title='Interactive Price Map for Kidney Beans'
)
```

## Phase 9: Integration and Finalization (Week 10)

### 9.1 Create Main Script

Create `src/main.py`:

```python
"""
Main script for Yemen market integration analysis.
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import logging

from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.unit_root import UnitRootTester
from models.cointegration import CointegrationTester
from models.threshold import ThresholdCointegration
from models.threshold_vecm import ThresholdVECM
from models.spatial import SpatialEconometrics
from models.simulation import MarketIntegrationSimulation
from visualization.time_series import TimeSeriesVisualizer
from visualization.market_maps import MarketMapVisualizer


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('yemen_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Yemen Market Integration Analysis')
    parser.add_argument('--data', type=str, default='./data/raw/unified_data.geojson',
                        help='Path to the GeoJSON data file')
    parser.add_argument('--output', type=str, default='./output',
                        help='Path to save output files')
    parser.add_argument('--commodity', type=str, default='beans (kidney red)',
                        help='Commodity to analyze')
    parser.add_argument('--threshold', action='store_true',
                        help='Run threshold cointegration analysis')
    parser.add_argument('--spatial', action='store_true',
                        help='Run spatial econometric analysis')
    parser.add_argument('--simulation', action='store_true',
                        help='Run policy simulations')
    
    return parser.parse_args()


def main():
    """Main entry point for the analysis."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Yemen market integration analysis")
    
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    loader = DataLoader()
    try:
        gdf = loader.load_geojson(args.data)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = DataPreprocessor()
    processed_gdf = preprocessor.preprocess_geojson(gdf)
    
    # Save processed data
    processed_file = output_path / 'processed_data.geojson'
    processed_gdf.to_file(processed_file, driver='GeoJSON')
    logger.info(f"Saved processed data to {processed_file}")
    
    # Calculate price differentials
    logger.info("Calculating price differentials between north and south")
    differentials = preprocessor.calculate_price_differentials(processed_gdf)
    differentials.to_csv(output_path / 'price_differentials.csv', index=False)
    
    # Visualize data
    logger.info("Creating visualizations")
    time_vis = TimeSeriesVisualizer()
    map_vis = MarketMapVisualizer()
    
    # Time series plots
    commodity_data = processed_gdf[processed_gdf['commodity'] == args.commodity]
    fig = time_vis.plot_price_series(
        commodity_data,
        group_col='admin1',
        title=f'Price Trends for {args.commodity} by Region'
    )
    fig.savefig(output_path / f'{args.commodity.replace(" ", "_")}_price_trends.png')
    
    # Price differential plots
    commodity_diff = differentials[differentials['commodity'] == args.commodity]
    fig = time_vis.plot_price_differentials(
        commodity_diff,
        title=f'Price Differentials: North vs South ({args.commodity})'
    )
    fig.savefig(output_path / f'{args.commodity.replace(" ", "_")}_price_differentials.png')
    
    # Spatial visualization
    latest_date = processed_gdf['date'].max()
    latest_data = processed_gdf[
        (processed_gdf['commodity'] == args.commodity) & 
        (processed_gdf['date'] == latest_date)
    ]
    fig = map_vis.plot_static_map(
        latest_data,
        column='price',
        title=f'Price Distribution of {args.commodity} ({latest_date.strftime("%Y-%m-%d")})'
    )
    fig.savefig(output_path / f'{args.commodity.replace(" ", "_")}_price_map.png')
    
    # Interactive map
    m = map_vis.create_interactive_map(
        latest_data,
        column='price',
        popup_cols=['admin1', 'price', 'usdprice', 'conflict_intensity_normalized'],
        title=f'Interactive Price Map for {args.commodity}'
    )
    m.save(output_path / f'{args.commodity.replace(" ", "_")}_interactive_map.html')
    
    # Time series analysis
    if args.threshold:
        logger.info("Running threshold cointegration analysis")
        
        # Get data for north and south
        north_data = processed_gdf[
            (processed_gdf['commodity'] == args.commodity) & 
            (processed_gdf['exchange_rate_regime'] == 'north')
        ]
        south_data = processed_gdf[
            (processed_gdf['commodity'] == args.commodity) & 
            (processed_gdf['exchange_rate_regime'] == 'south')
        ]
        
        # Aggregate to monthly average prices
        north_monthly = north_data.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()
        south_monthly = south_data.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()
        
        # Ensure dates align
        merged = pd.merge(
            north_monthly, south_monthly,
            on='date', suffixes=('_north', '_south')
        )
        
        # Unit root tests
        logger.info("Testing for unit roots")
        unit_root_tester = UnitRootTester()
        
        north_unit_root = unit_root_tester.run_all_tests(merged['price_north'])
        south_unit_root = unit_root_tester.run_all_tests(merged['price_south'])
        
        # Cointegration tests
        logger.info("Testing for cointegration")
        cointegration_tester = CointegrationTester()
        
        eg_result = cointegration_tester.test_engle_granger(
            merged['price_north'], merged['price_south']
        )
        
        # Threshold cointegration
        logger.info("Estimating threshold cointegration model")
        threshold_model = ThresholdCointegration(
            merged['price_north'], merged['price_south'], max_lags=4
        )
        
        cointegration_result = threshold_model.estimate_cointegration()
        threshold_result = threshold_model.estimate_threshold()
        tvecm_result = threshold_model.estimate_tvecm()
        
        # Save results
        with open(output_path / f'{args.commodity.replace(" ", "_")}_threshold_results.txt', 'w') as f:
            f.write("UNIT ROOT TESTS\n")
            f.write("===============\n\n")
            f.write("North price:\n")
            f.write(str(north_unit_root) + "\n\n")
            f.write("South price:\n")
            f.write(str(south_unit_root) + "\n\n")
            
            f.write("COINTEGRATION TESTS\n")
            f.write("===================\n\n")
            f.write("Engle-Granger:\n")
            f.write(str(eg_result) + "\n\n")
            
            f.write("THRESHOLD COINTEGRATION\n")
            f.write("=======================\n\n")
            f.write("Cointegration result:\n")
            f.write(str(cointegration_result) + "\n\n")
            f.write("Threshold result:\n")
            f.write(str(threshold_result) + "\n\n")
            f.write("TVECM result:\n")
            f.write("Threshold: " + str(tvecm_result['threshold']) + "\n")
            f.write("Adjustment below (north): " + str(tvecm_result['adjustment_below_1']) + "\n")
            f.write("Adjustment above (north): " + str(tvecm_result['adjustment_above_1']) + "\n")
            f.write("Adjustment below (south): " + str(tvecm_result['adjustment_below_2']) + "\n")
            f.write("Adjustment above (south): " + str(tvecm_result['adjustment_above_2']) + "\n")
    
    # Spatial analysis
    if args.spatial:
        logger.info("Running spatial econometric analysis")
        
        # Get latest data for spatial analysis
        spatial_data = processed_gdf[
            (processed_gdf['commodity'] == args.commodity) & 
            (processed_gdf['date'] == latest_date)
        ]
        
        # Create spatial econometrics model
        spatial_model = SpatialEconometrics(spatial_data)
        
        # Create weight matrices
        w_standard = spatial_model.create_weight_matrix(
            k=5, conflict_adjusted=False
        )
        w_conflict = spatial_model.create_weight_matrix(
            k=5, conflict_adjusted=True
        )
        
        # Test for spatial autocorrelation
        moran_standard = spatial_model.moran_i_test('price')
        
        # Reset weights to conflict-adjusted
        spatial_model.weights = w_conflict
        moran_conflict = spatial_model.moran_i_test('price')
        
        # Estimate spatial lag model
        x_vars = ['conflict_intensity_normalized']
        if 'exchange_rate_regime' in spatial_data.columns:
            # Create dummy for exchange rate regime
            spatial_data['north_regime'] = (spatial_data['exchange_rate_regime'] == 'north').astype(int)
            x_vars.append('north_regime')
        
        spatial_lag = spatial_model.spatial_lag_model('price', x_vars)
        
        # Save results
        with open(output_path / f'{args.commodity.replace(" ", "_")}_spatial_results.txt', 'w') as f:
            f.write("SPATIAL AUTOCORRELATION\n")
            f.write("======================\n\n")
            f.write("Standard weights:\n")
            f.write(str(moran_standard) + "\n\n")
            f.write("Conflict-adjusted weights:\n")
            f.write(str(moran_conflict) + "\n\n")
            
            f.write("SPATIAL LAG MODEL\n")
            f.write("================\n\n")
            f.write(str(spatial_lag.summary) + "\n")
    
    # Policy simulations
    if args.simulation and args.threshold:
        logger.info("Running policy simulations")
        
        # Create simulation model
        sim_model = MarketIntegrationSimulation(
            processed_gdf, threshold_model=threshold_model
        )
        
        # Simulate exchange rate unification
        unified_result = sim_model.simulate_exchange_rate_unification()
        
        # Save results
        with open(output_path / f'{args.commodity.replace(" ", "_")}_simulation_results.txt', 'w') as f:
            f.write("EXCHANGE RATE UNIFICATION SIMULATION\n")
            f.write("==================================\n\n")
            # Actual content would depend on the specific implementation of the simulation
            f.write("Simulation completed. Results saved to processed data files.\n")
        
        # Save unified data
        unified_data = unified_result['data']
        unified_data.to_file(
            output_path / 'unified_exchange_rate_data.geojson',
            driver='GeoJSON'
        )
    
    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()
```

### 9.2 Create Setup Script

Create `setup.py`:

```python
"""
Setup script for yemen-market-integration package.
"""
from setuptools import setup, find_packages

setup(
    name="yemen-market-integration",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "geopandas>=0.10.0",
        "pysal>=2.4.0",
        "folium>=0.12.0",
        "statsmodels>=0.13.0",
        "arch>=5.0.0",
        "pmdarima>=1.8.0",
        "spreg>=1.2.4",
        "libpysal>=4.5.0",
        "plotly>=5.3.0",
        "mapclassify>=2.4.0",
        "contextily>=1.2.0",
        "esda>=2.4.1"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Econometric analysis of market integration in Yemen",
    keywords="market integration, conflict economics, threshold cointegration, spatial econometrics",
    python_requires=">=3.8",
)
```

### 9.3 Create README File

Create `README.md`:

```markdown
# Yemen Market Integration Analysis

## Overview

This project implements econometric methodologies for analyzing market integration in conflict-affected Yemen. It examines how conflict-induced transaction costs impede arbitrage and impact dual exchange rate regimes, using threshold cointegration and spatial econometric techniques.

## Features

- **Data Processing**: Tools for handling GeoJSON market data with spatial attributes
- **Time Series Analysis**: Unit root testing, cointegration analysis, and threshold modeling
- **Spatial Econometrics**: Tools for analyzing geographic dependencies in market integration
- **Policy Simulation**: Tools for simulating market integration scenarios (e.g., exchange rate unification)
- **Visualization**: Comprehensive visualization tools for both time series and spatial data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yemen-market-integration.git
cd yemen-market-integration

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Basic Analysis

```python
# Import key modules
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.threshold import ThresholdCointegration

# Load and preprocess data
loader = DataLoader()
gdf = loader.load_geojson('data/raw/unified_data.geojson')
preprocessor = DataPreprocessor()
processed_gdf = preprocessor.preprocess_geojson(gdf)

# Run threshold cointegration analysis
north_data = processed_gdf[processed_gdf['exchange_rate_regime'] == 'north']['price']
south_data = processed_gdf[processed_gdf['exchange_rate_regime'] == 'south']['price']
threshold_model = ThresholdCointegration(north_data, south_data)
results = threshold_model.estimate_tvecm()
```

### Running the Full Analysis

```bash
# Run the full analysis pipeline
python src/main.py --data data/raw/unified_data.geojson --output results --commodity "beans (kidney red)" --threshold --spatial --simulation
```

## Project Structure

```
yemen-market-integration/
│
├── data/                          # Data storage
│   ├── raw/                       # Original GeoJSON data
│   └── processed/                 # Cleaned and transformed data
│
├── src/                           # Source code
│   ├── data/                      # Data processing modules
│   ├── models/                    # Econometric models
│   ├── visualization/             # Visualization tools
│   └── utils/                     # Utility functions
│
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── results/                       # Analysis results
├── requirements.txt               # Project dependencies
├── setup.py                       # Package installation
└── README.md                      # Project documentation
```

## Documentation

For detailed documentation, refer to the following Jupyter notebooks:

- `notebooks/01_exploratory_analysis.ipynb`: Data exploration and visualization
- `notebooks/02_threshold_cointegration.ipynb`: Threshold cointegration analysis
- `notebooks/03_spatial_analysis.ipynb`: Spatial econometric analysis
- `notebooks/04_policy_simulations.ipynb`: Policy simulation scenarios

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```

## Phase 10: Documentation and Reproducibility (Week 11)

### 10.1 Create Documentation Notebooks

Create a series of detailed Jupyter notebooks with step-by-step explanations:

1. **Data Exploration Notebook**: Comprehensive analysis of the dataset
2. **Threshold Cointegration Notebook**: Detailed implementation and interpretation
3. **Spatial Analysis Notebook**: Spatial econometric techniques and visualizations
4. **Policy Simulation Notebook**: Exchange rate unification and connectivity simulations

### 10.2 Create Reproducibility Guide

Create `REPRODUCE.md`:

```markdown
# Reproducibility Guide

This document provides detailed instructions for reproducing all analyses in the Yemen Market Integration project.

## Environment Setup

We recommend using conda to manage dependencies:

```bash
# Create conda environment
conda create -n yemen-market python=3.9
conda activate yemen-market

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Data Preparation

1. Place the raw GeoJSON file in `data/raw/unified_data.geojson`
2. Run the preprocessing script:

```bash
python -m src.data.preprocessor
```

## Running the Analysis Pipeline

For a complete analysis of a specific commodity:

```bash
python src/main.py --data data/raw/unified_data.geojson --output results --commodity "beans (kidney red)" --threshold --spatial --simulation
```

## Step-by-Step Reproduction of Key Results

### 1. Threshold Cointegration Analysis

```python
# Run this in a Python console or Jupyter notebook
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.threshold import ThresholdCointegration
import pandas as pd

# Load and prepare data
loader = DataLoader()
gdf = loader.load_geojson('data/raw/unified_data.geojson')
preprocessor = DataPreprocessor()
processed_gdf = preprocessor.preprocess_geojson(gdf)

# Filter data for a specific commodity
commodity_data = processed_gdf[processed_gdf['commodity'] == 'beans (kidney red)']

# Get data for north and south
north_data = commodity_data[commodity_data['exchange_rate_regime'] == 'north']
south_data = commodity_data[commodity_data['exchange_rate_regime'] == 'south']

# Aggregate to monthly average prices
north_monthly = north_data.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()
south_monthly = south_data.groupby(pd.Grouper(key='date', freq='M'))['price'].mean().reset_index()

# Ensure dates align
merged = pd.merge(
    north_monthly, south_monthly,
    on='date', suffixes=('_north', '_south')
)

# Run threshold cointegration
threshold_model = ThresholdCointegration(
    merged['price_north'], merged['price_south'], max_lags=4
)

cointegration_result = threshold_model.estimate_cointegration()
threshold_result = threshold_model.estimate_threshold()
tvecm_result = threshold_model.estimate_tvecm()

# Print key results
print(f"Cointegration p-value: {cointegration_result['pvalue']:.4f}")
print(f"Threshold parameter: {threshold_result['threshold']:.2f}")
print(f"Adjustment below (north): {tvecm_result['adjustment_below_1']:.4f}")
print(f"Adjustment above (north): {tvecm_result['adjustment_above_1']:.4f}")
```

### 2. Spatial Analysis Reproduction

```python
# Run this in a Python console or Jupyter notebook
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.spatial import SpatialEconometrics

# Load and prepare data
loader = DataLoader()
gdf = loader.load_geojson('data/raw/unified_data.geojson')
preprocessor = DataPreprocessor()
processed_gdf = preprocessor.preprocess_geojson(gdf)

# Filter for latest date
latest_date = processed_gdf['date'].max()
spatial_data = processed_gdf[
    (processed_gdf['commodity'] == 'beans (kidney red)') & 
    (processed_gdf['date'] == latest_date)
]

# Create spatial model
spatial_model = SpatialEconometrics(spatial_data)

# Create weight matrices
w_standard = spatial_model.create_weight_matrix(k=5, conflict_adjusted=False)
w_conflict = spatial_model.create_weight_matrix(k=5, conflict_adjusted=True)

# Test for spatial autocorrelation
moran_standard = spatial_model.moran_i_test('price')
spatial_model.weights = w_conflict
moran_conflict = spatial_model.moran_i_test('price')

# Print results
print(f"Moran's I (standard weights): {moran_standard['I']:.4f}, p-value: {moran_standard['p_norm']:.4f}")
print(f"Moran's I (conflict weights): {moran_conflict['I']:.4f}, p-value: {moran_conflict['p_norm']:.4f}")
```

## Verifying Results

After running the analysis, compare your results with the expected outputs in the `results/expected` directory. Note that random seed initialization may cause slight variations in some results.

```

## Implementation Timeline

| Phase | Task | Duration | Deliverables |
|-------|------|----------|--------------|
| 1 | Project Setup | Week 1 | Project structure, environment, dependencies |
| 2 | Data Processing | Week 2 | Data loader, preprocessor modules |
| 3 | Unit Root and Cointegration | Week 3 | Testing modules, unit tests |
| 4-5 | Threshold Cointegration | Weeks 4-5 | Threshold models, VECM implementation |
| 6 | Spatial Econometrics | Week 6 | Spatial analysis modules |
| 7 | Policy Simulation | Week 7 | Simulation modules |
| 8 | Visualization Tools | Week 8 | Time series and spatial visualization tools |
| 9 | Integration | Week 10 | Main script, setup script |
| 10 | Documentation | Week 11 | Notebooks, reproducibility guide |
| 11 | Testing & Refinement | Week 12 | Final testing, bug fixes |

## Getting Started

To get started with the implementation:

1. **Set up VS Code workspace**:
   - Create a new workspace folder
   - Open the folder in VS Code
   - Initialize Git repository
   - Set up Python extension and linting

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Create project structure**:

   ```bash
   mkdir -p data/{raw,processed}
   mkdir -p src/{data,models,visualization,utils}
   mkdir -p notebooks tests
   touch requirements.txt setup.py README.md
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Import GeoJSON data**:
   - Place the `unified_data.geojson` file in the `data/raw` directory
   - Start implementing the data loader module

6. **Follow the phase-by-phase implementation**:
   - Complete each module as outlined in the phases above
   - Test each component as you build

By following this implementation plan, you'll create a robust Python package for analyzing market integration in conflict-affected Yemen using advanced econometric techniques.
