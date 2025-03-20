# Yemen Market Integration Project: Enhanced Implementation Plan

## Overview

This plan outlines the implementation of an econometric analysis project for studying market integration in conflict-affected Yemen. The project uses threshold cointegration and spatial econometric techniques to analyze price transmission barriers and simulate potential policy interventions.

## Phase 1: Project Setup and Infrastructure (Week 1-2)

### 1.1 Environment Setup

```bash
# Create a new directory for the project
mkdir -p yemen-market-integration
cd yemen-market-integration

# Initialize git repository with appropriate branching strategy
git init
git checkout -b main
git branch develop
git checkout develop

# Set up Docker for reproducible environment
echo "FROM python:3.9-slim
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["bash"]" > Dockerfile

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create project structure
mkdir -p data/{raw,processed,external}
mkdir -p src/{data,models,visualization,utils,tests}
mkdir -p notebooks
mkdir -p config
```

### 1.2 Project Structure Creation

```bash
# Create module structure with __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py

# Create testing framework
mkdir -p tests/{unit,integration,performance}
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py 
touch tests/performance/__init__.py

# Create configuration files
touch config/settings.yaml
touch config/logging.yaml
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
pyproj>=3.0.0
pysal>=2.4.0
folium>=0.12.0
contextily>=1.1.0

# Econometrics and statistics
statsmodels>=0.13.0
scikit-learn>=1.0.0
arch>=5.0.0
pmdarima>=1.8.0

# Spatial econometrics
spreg>=1.2.4
libpysal>=4.5.0
esda>=2.4.1
splot>=1.1.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
mapclassify>=2.4.0

# Performance optimization
numba>=0.55.0
joblib>=1.1.0
dask>=2022.1.0
swifter>=1.0.0

# Data acquisition and APIs
requests>=2.27.0
beautifulsoup4>=4.10.0
openpyxl>=3.0.0
xlrd>=2.0.0

# Testing and quality assurance
pytest>=6.2.0
pytest-cov>=2.12.0
flake8>=3.9.0
black>=21.5b2
pylint>=2.8.0

# Documentation
sphinx>=4.3.0
sphinx-rtd-theme>=1.0.0
nbsphinx>=0.8.0

# CI/CD
pre-commit>=2.17.0

# Environment management
python-dotenv>=0.19.0
pyyaml>=6.0
```

### 1.4 Continuous Integration Setup

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Lint with flake8
      run: |
        flake8 src tests
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.1.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 22.1.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
    -   id: flake8
```

## Phase 2: Data Acquisition and Management (Week 3-4)

### 2.1 Create Data Source Connectors

Create `src/data/sources.py`:

```python
"""
Data source connectors for acquiring Yemen market integration data.
"""
import os
import logging
import requests
from datetime import datetime
import pandas as pd
import geopandas as gpd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

class WFPDataConnector:
    """Connector for World Food Programme (WFP) price data."""
    
    def __init__(self, api_key=None):
        """Initialize WFP connector with optional API key."""
        self.api_key = api_key or os.getenv("WFP_API_KEY")
        self.base_url = "https://api.wfp.org/vam-data/markets"
    
    def fetch_commodity_prices(self, country="Yemen", 
                               start_date="2020-01-01", 
                               end_date=None):
        """
        Fetch commodity prices for Yemen from WFP API.
        
        Parameters
        ----------
        country : str
            Country name
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns
        -------
        pandas.DataFrame
            Commodity price data
        """
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching WFP data for {country} from {start_date} to {end_date}")
        
        # Implementation details would depend on WFP API documentation
        # This is a placeholder
        try:
            response = requests.get(
                f"{self.base_url}/prices",
                params={
                    "country": country,
                    "start_date": start_date,
                    "end_date": end_date,
                },
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data["data"])
            
            logger.info(f"Successfully fetched {len(df)} records")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching WFP data: {e}")
            raise


class ACLEDConflictConnector:
    """Connector for Armed Conflict Location & Event Data Project (ACLED)."""
    
    def __init__(self, api_key=None):
        """Initialize ACLED connector with optional API key."""
        self.api_key = api_key or os.getenv("ACLED_API_KEY")
        self.base_url = "https://api.acleddata.com/acled/read"
        
    def fetch_conflict_data(self, country="Yemen", 
                           start_date="2020-01-01", 
                           end_date=None):
        """
        Fetch conflict data for Yemen from ACLED API.
        
        Parameters
        ----------
        country : str
            Country name
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns
        -------
        pandas.DataFrame
            Conflict event data
        """
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching ACLED data for {country} from {start_date} to {end_date}")
        
        try:
            response = requests.get(
                self.base_url,
                params={
                    "country": country,
                    "event_date": f"{start_date}|{end_date}",
                    "email": self.api_key
                }
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data["data"])
            
            # Process coordinates and create points
            df["latitude"] = pd.to_numeric(df["latitude"])
            df["longitude"] = pd.to_numeric(df["longitude"])
            
            logger.info(f"Successfully fetched {len(df)} conflict events")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ACLED data: {e}")
            raise


class CentralBankConnector:
    """Connector for Central Bank of Yemen exchange rate data."""
    
    def __init__(self):
        """Initialize Central Bank connector."""
        self.sana_url = "https://cbysana.org/rates"  # Placeholder URL
        self.aden_url = "https://cby-aden.com/rates"  # Placeholder URL
    
    def fetch_exchange_rates(self, start_date="2020-01-01", end_date=None):
        """
        Fetch exchange rates from both Sana'a and Aden central banks.
        
        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns
        -------
        pandas.DataFrame
            Exchange rate data with columns for both regimes
        """
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching exchange rates from {start_date} to {end_date}")
        
        # Implementation would depend on actual data sources
        # This is a placeholder for web scraping or file download
        
        try:
            # Fetch Sana'a rates (example using web scraping)
            sana_rates = self._scrape_exchange_rates(self.sana_url)
            
            # Fetch Aden rates
            aden_rates = self._scrape_exchange_rates(self.aden_url)
            
            # Combine and process data
            combined = pd.merge(
                sana_rates.rename(columns={"rate": "sana_rate"}),
                aden_rates.rename(columns={"rate": "aden_rate"}),
                on="date", how="outer"
            )
            
            # Calculate differential
            combined["rate_differential"] = combined["aden_rate"] - combined["sana_rate"]
            
            logger.info(f"Successfully fetched {len(combined)} exchange rate records")
            return combined
            
        except Exception as e:
            logger.error(f"Error fetching exchange rate data: {e}")
            raise
    
    def _scrape_exchange_rates(self, url):
        """Helper method to scrape exchange rates from a website."""
        # This is a placeholder for web scraping implementation
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML and extract rates (simplified example)
        soup = BeautifulSoup(response.text, 'html.parser')
        rates_data = []
        
        # Implementation would depend on website structure
        # ...
        
        return pd.DataFrame(rates_data)


class ACAPSGeographicConnector:
    """Connector for ACAPS Yemen Analysis Hub geographic data."""
    
    def __init__(self):
        """Initialize ACAPS connector."""
        self.base_url = "https://data.humdata.org/dataset/yemen-administrative-boundaries"
    
    def fetch_admin_boundaries(self, level=1):
        """
        Fetch administrative boundaries for Yemen.
        
        Parameters
        ----------
        level : int
            Administrative level (1=governorate, 2=district)
            
        Returns
        -------
        geopandas.GeoDataFrame
            Administrative boundaries
        """
        logger.info(f"Fetching admin level {level} boundaries")
        
        # This would typically download a shapefile or GeoJSON
        # For now, we'll assume the file is locally available
        try:
            # Placeholder for actual implementation
            admin_file = f"data/external/yemen_admin{level}.shp"
            
            if not os.path.exists(admin_file):
                # Download logic would go here
                pass
            
            gdf = gpd.read_file(admin_file)
            logger.info(f"Successfully loaded {len(gdf)} administrative regions")
            return gdf
            
        except Exception as e:
            logger.error(f"Error fetching administrative boundaries: {e}")
            raise
```

### 2.2 Create Data Integration and Version Management

Create `src/data/integration.py`:

```python
"""
Data integration module for combining data from different sources.
"""
import os
import pandas as pd
import geopandas as gpd
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataVersionManager:
    """Manages data versioning and history."""
    
    def __init__(self, data_dir="./data"):
        """Initialize with data directory path."""
        self.data_dir = Path(data_dir)
        self.version_file = self.data_dir / "version_history.json"
        self._load_version_history()
    
    def _load_version_history(self):
        """Load version history from file."""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {}
            self._save_version_history()
    
    def _save_version_history(self):
        """Save version history to file."""
        with open(self.version_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def compute_hash(self, file_path):
        """Compute hash of a file for versioning."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def register_file(self, file_path, source, description=None):
        """
        Register a new file version in the history.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the file
        source : str
            Data source identifier
        description : str, optional
            Description of this version
            
        Returns
        -------
        str
            Version identifier
        """
        file_path = Path(file_path)
        file_hash = self.compute_hash(file_path)
        timestamp = datetime.now().isoformat()
        
        # Create version entry
        version_id = f"{source}_{timestamp.replace(':', '-')}"
        
        if source not in self.history:
            self.history[source] = []
        
        self.history[source].append({
            "version_id": version_id,
            "timestamp": timestamp,
            "path": str(file_path),
            "hash": file_hash,
            "description": description
        })
        
        self._save_version_history()
        logger.info(f"Registered new version {version_id} for {source}")
        
        return version_id
    
    def get_latest_version(self, source):
        """Get the latest version for a given source."""
        if source in self.history and self.history[source]:
            return self.history[source][-1]
        return None


class DataIntegrator:
    """Integrates data from multiple sources into unified datasets."""
    
    def __init__(self, data_dir="./data"):
        """Initialize with data directory path."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.version_manager = DataVersionManager(data_dir)
    
    def integrate_market_data(self, commodity_file, exchange_file, conflict_file,
                             admin_file, output_file="unified_data.geojson"):
        """
        Integrate market data from multiple sources.
        
        Parameters
        ----------
        commodity_file : str or Path
            Path to commodity price data
        exchange_file : str or Path
            Path to exchange rate data
        conflict_file : str or Path
            Path to conflict data
        admin_file : str or Path
            Path to administrative boundaries
        output_file : str, optional
            Output filename
            
        Returns
        -------
        geopandas.GeoDataFrame
            Integrated dataset
        """
        logger.info("Starting data integration process")
        
        # Load datasets
        commodity_df = pd.read_csv(commodity_file)
        exchange_df = pd.read_csv(exchange_file)
        conflict_df = pd.read_csv(conflict_file)
        admin_gdf = gpd.read_file(admin_file)
        
        # Ensure date columns are datetime
        for df in [commodity_df, exchange_df, conflict_df]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # Merge exchange rate data with commodity prices
        logger.info("Merging exchange rate data with commodity prices")
        df = pd.merge(
            commodity_df,
            exchange_df,
            on=['date', 'exchange_rate_regime'],
            how='left'
        )
        
        # Aggregate conflict data by admin region and date
        logger.info("Aggregating conflict data")
        conflict_agg = conflict_df.groupby(['admin1', 'date']).agg({
            'events': 'sum',
            'fatalities': 'sum'
        }).reset_index()
        
        # Merge conflict data
        logger.info("Merging conflict data")
        df = pd.merge(
            df,
            conflict_agg,
            on=['admin1', 'date'],
            how='left'
        )
        
        # Fill missing conflict data with zeros
        df[['events', 'fatalities']] = df[['events', 'fatalities']].fillna(0)
        
        # Calculate conflict intensity metrics
        logger.info("Calculating conflict intensity metrics")
        df = self._calculate_conflict_intensity(df)
        
        # Create spatial data by merging with admin boundaries
        logger.info("Creating spatial dataset")
        gdf = pd.merge(
            df,
            admin_gdf[['admin1', 'geometry', 'population']],
            on='admin1',
            how='left'
        )
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=admin_gdf.crs)
        
        # Save integrated data
        output_path = self.processed_dir / output_file
        gdf.to_file(output_path, driver='GeoJSON')
        
        # Register the new version
        self.version_manager.register_file(
            output_path,
            "integrated_data",
            "Integrated market, exchange, and conflict data"
        )
        
        logger.info(f"Integration complete. Saved to {output_path}")
        return gdf
    
    def _calculate_conflict_intensity(self, df):
        """
        Calculate conflict intensity metrics.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with events and fatalities columns
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added conflict intensity metrics
        """
        # Create a composite conflict intensity metric
        # Events and fatalities are weighted differently
        df['conflict_intensity'] = (df['events'] * 0.3) + (df['fatalities'] * 0.7)
        
        # Population-weighted intensity
        if 'population' in df.columns:
            df['conflict_intensity_weighted'] = df['conflict_intensity'] / df['population'] * 100000
        
        # Normalize to 0-1 scale
        max_intensity = df['conflict_intensity'].max()
        df['conflict_intensity_normalized'] = df['conflict_intensity'] / max_intensity if max_intensity > 0 else 0
        
        # Create lagged conflict intensity for time series analysis
        for lag in range(1, 4):
            df[f'conflict_intensity_lag{lag}'] = df.groupby(['admin1'])['conflict_intensity_normalized'].shift(lag)
        
        return df
```

### 2.3 Create Data Validation Module

Create `src/data/validation.py`:

```python
"""
Data validation module for ensuring data quality.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from typing import Tuple, Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data quality for Yemen market integration analysis."""
    
    def __init__(self):
        """Initialize validator."""
        pass
    
    def validate_commodity_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate commodity price data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Commodity price data
            
        Returns
        -------
        tuple
            (is_valid, validation_results)
        """
        results = {
            "missing_values": {},
            "outliers": {},
            "invalid_dates": False,
            "invalid_prices": False,
            "invalid_regions": False,
            "errors": []
        }
        
        # Check required columns
        required_cols = ["date", "price", "commodity", "admin1", "exchange_rate_regime"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            results["errors"].append(f"Missing required columns: {missing_cols}")
            return False, results
        
        # Check for missing values
        for col in required_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                results["missing_values"][col] = missing
        
        # Check date validity
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            results["errors"].append(f"Invalid date format: {str(e)}")
            results["invalid_dates"] = True
        
        # Check price validity (must be positive)
        if (df["price"] <= 0).any():
            results["invalid_prices"] = True
            results["errors"].append("Found non-positive prices")
        
        # Check for outliers using z-score method
        for col in ["price"]:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3].index.tolist()
            if outliers:
                results["outliers"][col] = len(outliers)
        
        # Check exchange rate regime validity
        valid_regimes = ["north", "south"]
        invalid_regimes = df[~df["exchange_rate_regime"].isin(valid_regimes)]
        if not invalid_regimes.empty:
            results["errors"].append(f"Invalid exchange rate regimes: {invalid_regimes['exchange_rate_regime'].unique().tolist()}")
            results["invalid_regions"] = True
        
        # Determine overall validity
        is_valid = not (results["invalid_dates"] or 
                        results["invalid_prices"] or 
                        results["invalid_regions"] or 
                        results["errors"])
        
        return is_valid, results
    
    def validate_conflict_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate conflict data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Conflict data
            
        Returns
        -------
        tuple
            (is_valid, validation_results)
        """
        results = {
            "missing_values": {},
            "outliers": {},
            "errors": []
        }
        
        # Check required columns
        required_cols = ["date", "admin1", "events", "fatalities"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            results["errors"].append(f"Missing required columns: {missing_cols}")
            return False, results
        
        # Check for missing values
        for col in required_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                results["missing_values"][col] = missing
        
        # Check date validity
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            results["errors"].append(f"Invalid date format: {str(e)}")
        
        # Check for negative values in count columns
        for col in ["events", "fatalities"]:
            if (df[col] < 0).any():
                results["errors"].append(f"Negative values found in {col}")
        
        # Check for outliers using z-score method
        for col in ["events", "fatalities"]:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > 3].index.tolist()
            if outliers:
                results["outliers"][col] = len(outliers)
        
        # Determine overall validity
        is_valid = len(results["errors"]) == 0
        
        return is_valid, results
    
    def validate_integrated_data(self, gdf: gpd.GeoDataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate integrated spatial data.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Integrated spatial data
            
        Returns
        -------
        tuple
            (is_valid, validation_results)
        """
        results = {
            "missing_values": {},
            "outliers": {},
            "invalid_geometries": False,
            "errors": []
        }
        
        # Check if GeoDataFrame
        if not isinstance(gdf, gpd.GeoDataFrame):
            results["errors"].append("Data is not a GeoDataFrame")
            return False, results
        
        # Check required columns
        required_cols = [
            "date", "price", "commodity", "admin1", 
            "exchange_rate_regime", "events", "fatalities",
            "conflict_intensity", "geometry"
        ]
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        
        if missing_cols:
            results["errors"].append(f"Missing required columns: {missing_cols}")
            return False, results
        
        # Check geometry validity
        invalid_geoms = gdf[~gdf.geometry.is_valid].index.tolist()
        if invalid_geoms:
            results["invalid_geometries"] = True
            results["errors"].append(f"Found {len(invalid_geoms)} invalid geometries")
        
        # Check CRS
        if gdf.crs is None:
            results["errors"].append("Missing coordinate reference system (CRS)")
        
        # Check for proper conflict intensity calculation
        if "conflict_intensity" in gdf.columns and "conflict_intensity_normalized" in gdf.columns:
            # Ensure normalized values are between 0 and 1
            if (gdf["conflict_intensity_normalized"] < 0).any() or (gdf["conflict_intensity_normalized"] > 1).any():
                results["errors"].append("Normalized conflict intensity values outside [0,1] range")
        
        # Determine overall validity
        is_valid = len(results["errors"]) == 0 and not results["invalid_geometries"]
        
        return is_valid, results
```

## Phase 3: Statistical Testing Framework (Week 5-6)

### 3.1 Enhanced Unit Root Testing Module

Enhance `src/models/unit_root.py`:

```python
"""
Unit root testing module with comprehensive test suite and structural break detection.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import arch.unitroot as unitroot
from arch.unitroot import ADF, DFGLS, PhillipsPerron, KPSS, ZivotAndrews
import logging
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

class UnitRootTester:
    """Perform comprehensive unit root tests on time series data."""
    
    def __init__(self):
        """Initialize the unit root tester."""
        pass
    
    def test_adf(self, series: Union[pd.Series, np.ndarray], 
                regression: str = 'c', lags: Optional[int] = None,
                autolag: str = 'AIC') -> Dict[str, Any]:
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
        autolag : str, optional
            Method to use for lag selection if lags is None
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        logger.info(f"Running ADF test with regression='{regression}'")
        
        try:
            result = adfuller(series, regression=regression, maxlag=lags, autolag=autolag)
            test_result = {
                'statistic': result[0],
                'pvalue': result[1],
                'usedlag': result[2],
                'nobs': result[3],
                'critical_values': result[4],
                'icbest': result[5],
                'stationary': result[1] < 0.05
            }
            
            logger.info(f"ADF test result: statistic={test_result['statistic']:.4f}, p-value={test_result['pvalue']:.4f}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error in ADF test: {str(e)}")
            raise
    
    def test_adf_gls(self, series: Union[pd.Series, np.ndarray], 
                    lags: Optional[int] = None,
                    autolag: str = 'AIC') -> Dict[str, Any]:
        """
        Perform ADF-GLS test (Elliot, Rothenberg, Stock).
        
        Parameters
        ----------
        series : array_like
            The time series to test
        lags : int, optional
            Number of lags to use in the regression
        autolag : str, optional
            Method to use for lag selection if lags is None
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        logger.info("Running ADF-GLS test")
        
        try:
            # Use the DFGLS class for better control
            result = DFGLS(series, lags=lags, trend='c')
            test_result = {
                'statistic': result.stat,
                'pvalue': result.pvalue,
                'critical_values': result.critical_values,
                'lags': result.lags,
                'stationary': result.pvalue < 0.05
            }
            
            logger.info(f"ADF-GLS test result: statistic={test_result['statistic']:.4f}, p-value={test_result['pvalue']:.4f}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error in ADF-GLS test: {str(e)}")
            raise
    
    def test_pp(self, series: Union[pd.Series, np.ndarray],
               lags: Optional[int] = None,
               regression: str = 'c') -> Dict[str, Any]:
        """
        Perform Phillips-Perron test for unit root.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        lags : int, optional
            Number of lags to use in the test
        regression : str, optional
            Specification of the deterministic regression terms
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        logger.info(f"Running Phillips-Perron test with regression='{regression}'")
        
        try:
            result = PhillipsPerron(series, lags=lags, trend=regression)
            test_result = {
                'statistic': result.stat,
                'pvalue': result.pvalue,
                'critical_values': result.critical_values,
                'lags': result.lags,
                'stationary': result.pvalue < 0.05
            }
            
            logger.info(f"Phillips-Perron test result: statistic={test_result['statistic']:.4f}, p-value={test_result['pvalue']:.4f}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error in Phillips-Perron test: {str(e)}")
            raise
    
    def test_kpss(self, series: Union[pd.Series, np.ndarray],
                 regression: str = 'c', lags: Optional[int] = None) -> Dict[str, Any]:
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
        logger.info(f"Running KPSS test with regression='{regression}'")
        
        try:
            result = KPSS(series, lags=lags, trend=regression)
            test_result = {
                'statistic': result.stat,
                'pvalue': result.pvalue,
                'critical_values': result.critical_values,
                'lags': result.lags,
                'stationary': result.pvalue > 0.05  # Note: opposite from ADF
            }
            
            logger.info(f"KPSS test result: statistic={test_result['statistic']:.4f}, p-value={test_result['pvalue']:.4f}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error in KPSS test: {str(e)}")
            raise
    
    def test_zivot_andrews(self, series: Union[pd.Series, np.ndarray],
                          model: str = 'both') -> Dict[str, Any]:
        """
        Perform Zivot-Andrews test for unit root with structural break.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        model : str, optional
            Model to use:
            'a' : tests for a break in intercept
            'b' : tests for a break in trend
            'both' : tests for a break in both intercept and trend
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        logger.info(f"Running Zivot-Andrews test with model='{model}'")
        
        try:
            result = ZivotAndrews(series, model=model)
            test_result = {
                'statistic': result.stat,
                'pvalue': result.pvalue,
                'critical_values': result.critical_values,
                'stationary': result.pvalue < 0.05,
                'breakpoint': result.breakpoint
            }
            
            # Convert break index to date if series is a pandas Series with datetime index
            if isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex):
                test_result['breakpoint_date'] = series.index[result.breakpoint]
            
            logger.info(f"Zivot-Andrews test result: statistic={test_result['statistic']:.4f}, "
                      f"p-value={test_result['pvalue']:.4f}, breakpoint={test_result['breakpoint']}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error in Zivot-Andrews test: {str(e)}")
            raise
    
    def run_all_tests(self, series: Union[pd.Series, np.ndarray], 
                     lags: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
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
        logger.info("Running comprehensive unit root test suite")
        
        return {
            'adf': self.test_adf(series, lags=lags),
            'adf_gls': self.test_adf_gls(series, lags=lags),
            'pp': self.test_pp(series, lags=lags),
            'kpss': self.test_kpss(series, lags=lags),
            'zivot_andrews': self.test_zivot_andrews(series)
        }
    
    def determine_integration_order(self, series: Union[pd.Series, np.ndarray], 
                                   max_order: int = 2) -> int:
        """
        Determine the order of integration for a time series.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        max_order : int, optional
            Maximum order of integration to check
            
        Returns
        -------
        int
            Order of integration (0, 1, 2, ...)
        """
        logger.info(f"Determining integration order (max_order={max_order})")
        
        # Check if the original series is stationary
        adf_result = self.test_adf(series)
        if adf_result['stationary']:
            logger.info("Series is stationary - integration order 0")
            return 0
        
        # Check first difference
        for d in range(1, max_order + 1):
            # Take d-th difference
            diff_series = pd.Series(series).diff(d).dropna()
            
            # Test stationarity of differenced series
            adf_result = self.test_adf(diff_series)
            
            if adf_result['stationary']:
                logger.info(f"Series is I({d}) - integrated of order {d}")
                return d
        
        logger.info(f"Series is not stationary after {max_order} differences")
        return max_order + 1  # Higher than max_order


class StructuralBreakTester:
    """Test for structural breaks in time series data."""
    
    def __init__(self):
        """Initialize the structural break tester."""
        pass
    
    def test_bai_perron(self, series: Union[pd.Series, np.ndarray],
                       max_breaks: int = 5, trim: float = 0.15) -> Dict[str, Any]:
        """
        Perform Bai-Perron multiple structural break test.
        
        Parameters
        ----------
        series : array_like
            The time series to test
        max_breaks : int, optional
            Maximum number of breaks to test for
        trim : float, optional
            Trimming percentage
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        logger.info(f"Running Bai-Perron test for up to {max_breaks} breaks")
        
        try:
            # This requires external 'ruptures' package
            import ruptures as rpt
            
            # Convert to numpy array if it's a pandas Series
            data = np.asarray(series)
            
            # Fit Pelt algorithm
            algo = rpt.Pelt(model="l2").fit(data)
            result = algo.predict(pen=np.log(len(data)) * 2)  # BIC penalty
            
            # Limit to max_breaks
            if len(result) > max_breaks + 1:  # +1 because result includes start and end points
                # Find largest cost reduction
                costs = [algo.cost.sum(data[:result[i]], data[result[i]:result[i+1]]) 
                        for i in range(len(result)-1)]
                
                # Keep only the most significant breaks
                sorted_idx = np.argsort(costs)[:max_breaks]
                breakpoints = sorted(result[i] for i in sorted_idx)
                
                # Add start and end points
                breakpoints = [0] + breakpoints + [len(data)]
            else:
                breakpoints = result
            
            # Convert break indices to dates if series is a pandas Series with datetime index
            break_dates = None
            if isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex):
                break_dates = [series.index[bp] for bp in breakpoints if bp < len(series)]
            
            test_result = {
                'breakpoints': breakpoints,
                'break_dates': break_dates,
                'num_breaks': len(breakpoints) - 2  # Subtract start and end points
            }
            
            logger.info(f"Bai-Perron test result: {test_result['num_breaks']} breaks detected")
            return test_result
            
        except ImportError:
            logger.warning("ruptures package not installed. Using simplified break detection.")
            # Simplified version using Zivot-Andrews
            za_result = self.test_zivot_andrews(series)
            return {
                'breakpoints': [0, za_result['breakpoint'], len(series)],
                'break_dates': [series.index[za_result['breakpoint']]] if isinstance(series, pd.Series) else None,
                'num_breaks': 1
            }
        except Exception as e:
            logger.error(f"Error in Bai-Perron test: {str(e)}")
            raise
    
    def test_gregory_hansen(self, y: Union[pd.Series, np.ndarray], 
                          x: Union[pd.Series, np.ndarray],
                          model: str = 'regshift') -> Dict[str, Any]:
        """
        Perform Gregory-Hansen test for cointegration with regime shifts.
        
        Parameters
        ----------
        y : array_like
            First time series
        x : array_like
            Second time series
        model : str, optional
            Type of structural change:
            'regshift' : level shift (C)
            'trend' : level shift with trend (C/T)
            'regshift_trend' : regime shift (C/S)
            
        Returns
        -------
        dict
            Dictionary with test results
        """
        logger.info(f"Running Gregory-Hansen test with model='{model}'")
        
        try:
            # Create lagged values of x (assuming it's exogenous)
            x_lag = np.roll(x, 1)
            x_lag[0] = x_lag[1]  # Fix the first value
            
            # Store the number of observations
            n = len(y)
            min_size = int(n * 0.15)  # Trim 15% from both ends
            
            # Initialize variables to store minimum test statistic and breakpoint
            min_adf = np.inf
            bp = None
            
            # Calculate GH test statistic for range of potential breaks
            for i in range(min_size, n - min_size):
                # Create dummy variable for break
                d = np.zeros(n)
                d[i:] = 1
                
                # Create regression variables based on model
                if model == 'regshift':
                    X = np.column_stack([np.ones(n), d, x_lag])
                elif model == 'trend':
                    trend = np.arange(n)
                    X = np.column_stack([np.ones(n), trend, d, x_lag])
                elif model == 'regshift_trend':
                    X = np.column_stack([np.ones(n), d, x_lag, d * x_lag])
                else:
                    raise ValueError(f"Unknown model type: {model}")
                
                # Run regression
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                
                # Calculate residuals
                resid = y - X @ beta
                
                # Run ADF test on residuals
                adf_result = adfuller(resid, regression='nc')
                
                # Store minimum test statistic
                if adf_result[0] < min_adf:
                    min_adf = adf_result[0]
                    bp = i
                    p_value = adf_result[1]
                    crit_vals = adf_result[4]
            
            # Determine if cointegrated
            is_cointegrated = min_adf < crit_vals['5%']  # Using 5% critical value
            
            # Convert break index to date if series are pandas Series with datetime index
            break_date = None
            if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
                break_date = y.index[bp]
            
            test_result = {
                'statistic': min_adf,
                'pvalue': p_value,
                'critical_values': crit_vals,
                'breakpoint': bp,
                'break_date': break_date,
                'cointegrated': is_cointegrated
            }
            
            logger.info(f"Gregory-Hansen test result: statistic={test_result['statistic']:.4f}, "
                      f"breakpoint={test_result['breakpoint']}, cointegrated={test_result['cointegrated']}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error in Gregory-Hansen test: {str(e)}")
            raise
```

### 3.2 Enhanced Cointegration Testing Module

Enhance `src/models/cointegration.py`:

```python
"""
Enhanced cointegration testing module with diagnostics.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank
from statsmodels.tsa.stattools import coint
import logging
from typing import Dict, Any, Tuple, List, Union, Optional

logger = logging.getLogger(__name__)

class CointegrationTester:
    """Perform comprehensive cointegration tests on time series data."""
    
    def __init__(self):
        """Initialize the cointegration tester."""
        pass
    
    def test_engle_granger(self, y: Union[pd.Series, np.ndarray], 
                          x: Union[pd.Series, np.ndarray], 
                          trend: str = 'c', maxlag: Optional[int] = None,
                          autolag: str = 'AIC') -> Dict[str, Any]:
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
        logger.info(f"Running Engle-Granger test with trend='{trend}'")
        
        try:
            # Create X with constant if needed
            if trend == 'c':
                X = np.column_stack([np.ones(len(x)), x])
            elif trend == 'ct':
                X = np.column_stack([np.ones(len(x)), np.arange(len(x)), x])
            elif trend == 'ctt':
                t = np.arange(len(x))
                X = np.column_stack([np.ones(len(x)), t, t**2, x])
            elif trend == 'nc':
                X = x.reshape(-1, 1) if hasattr(x, 'reshape') else np.array(x).reshape(-1, 1)
            
            # Step 1: Estimate cointegrating relationship
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                logger.error("Linear algebra error in estimating cointegration relation")
                raise
            
            # Calculate residuals
            if trend == 'c':
                residuals = y - (beta[0] + beta[1] * x)
            elif trend == 'ct':
                residuals = y - (beta[0] + beta[1] * np.arange(len(x)) + beta[2] * x)
            elif trend == 'ctt':
                t = np.arange(len(x))
                residuals = y - (beta[0] + beta[1] * t + beta[2] * t**2 + beta[3] * x)
            elif trend == 'nc':
                residuals = y - beta[0] * x
            
            # Step 2: Test for unit root in residuals
            result = coint(y, x, trend=trend, maxlag=maxlag, autolag=autolag)
            
            test_result = {
                'statistic': result[0],
                'pvalue': result[1],
                'critical_values': result[2],
                'cointegrated': result[1] < 0.05,
                'beta': beta,
                'residuals': residuals
            }
            
            logger.info(f"Engle-Granger test result: statistic={test_result['statistic']:.4f}, "
                      f"p-value={test_result['pvalue']:.4f}, cointegrated={test_result['cointegrated']}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error in Engle-Granger test: {str(e)}")
            raise
    
    def test_johansen(self, data: Union[pd.DataFrame, np.ndarray], 
                     det_order: int = 0, k_ar_diff: int = 1) -> Dict[str, Any]:
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
        logger.info(f"Running Johansen test with det_order={det_order}, k_ar_diff={k_ar_diff}")
        
        try:
            # Ensure data is a 2D numpy array
            if isinstance(data, pd.DataFrame):
                data_arr = data.values
            else:
                data_arr = np.asarray(data)
            
            if data_arr.ndim == 1:
                logger.warning("Single time series provided to Johansen test. At least two series are required.")
                return {
                    'error': "Single time series provided. At least two series are required."
                }
            
            # Run Johansen test
            result = coint_johansen(data_arr, det_order=det_order, k_ar_diff=k_ar_diff)
            
            # Extract trace and max eigenvalue statistics
            trace_stat = result.lr1
            trace_crit = result.cvt
            max_stat = result.lr2
            max_crit = result.cvm
            
            # Determine the cointegration rank
            rank_trace = sum(trace_stat > trace_crit[:, 0])  # 5% significance level
            rank_max = sum(max_stat > max_crit[:, 0])  # 5% significance level
            
            # Calculate p-values (approximate via chi-squared)
            import scipy.stats as stats
            # Degrees of freedom for trace test at each rank
            df_trace = np.array([(data_arr.shape[1] - r) for r in range(data_arr.shape[1])])
            p_values_trace = [1 - stats.chi2.cdf(trace_stat[i], df_trace[i]) for i in range(len(trace_stat))]
            
            # Extract eigenvectors
            eigenvectors = result.evec
            
            test_result = {
                'trace_statistics': trace_stat,
                'trace_critical_values': trace_crit,
                'max_statistics': max_stat,
                'max_critical_values': max_crit,
                'rank_trace': rank_trace,
                'rank_max': rank_max,
                'cointegration_vectors': eigenvectors,
                'p_values_trace': p_values_trace,
                'cointegrated': rank_trace > 0 or rank_max > 0
            }
            
            logger.info(f"Johansen test result: rank_trace={rank_trace}, rank_max={rank_max}, "
                      f"cointegrated={test_result['cointegrated']}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error in Johansen test: {str(e)}")
            raise
    
    def calculate_half_life(self, residuals: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate the half-life of deviations from the cointegrating relationship.
        
        Parameters
        ----------
        residuals : array_like
            Residuals from the cointegrating relationship
            
        Returns
        -------
        float
            Half-life in periods
        """
        logger.info("Calculating half-life of deviations")
        
        try:
            # Ensure residuals is a numpy array
            residuals = np.asarray(residuals)
            
            # Estimate AR(1) model: Δe_t = ρ*e_{t-1} + ε_t
            X = residuals[:-1].reshape(-1, 1)
            y = np.diff(residuals)
            
            # Add constant
            X = np.column_stack([np.ones(X.shape[0]), X])
            
            # Estimate coefficients
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                logger.error("Linear algebra error in estimating AR(1) parameters")
                raise
            
            # Extract speed of adjustment
            rho = beta[1]
            
            # Calculate half-life: ln(0.5) / ln(1 + ρ)
            # ρ should be negative for convergence
            if rho >= 0:
                logger.warning("Adjustment coefficient is non-negative, indicating no convergence")
                return float('inf')
            
            half_life = np.log(0.5) / np.log(1 + rho)
            
            logger.info(f"Half-life calculation: coefficient={rho:.4f}, half-life={half_life:.2f} periods")
            return half_life
            
        except Exception as e:
            logger.error(f"Error in half-life calculation: {str(e)}")
            raise
    
    def test_combined(self, y: Union[pd.Series, np.ndarray], 
                     x: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Run multiple cointegration tests and combine results.
        
        Parameters
        ----------
        y : array_like
            First time series
        x : array_like
            Second time series
            
        Returns
        -------
        dict
            Combined test results
        """
        logger.info("Running combined cointegration tests")
        
        # Run Engle-Granger test
        eg_result = self.test_engle_granger(y, x)
        
        # Run Johansen test
        # Stack series into a matrix
        data = np.column_stack([y, x])
        jo_result = self.test_johansen(data)
        
        # Combine results
        combined_result = {
            'engle_granger': eg_result,
            'johansen': jo_result,
            'cointegrated': eg_result['cointegrated'] or jo_result['cointegrated'],
            'half_life': self.calculate_half_life(eg_result['residuals']) if eg_result['cointegrated'] else None
        }
        
        logger.info(f"Combined cointegration test result: cointegrated={combined_result['cointegrated']}")
        
        return combined_result
```

### 3.3 Create Comprehensive Diagnostics Module

Create `src/models/diagnostics.py`:

```python
"""
Comprehensive diagnostics module for econometric models.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_white
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, Tuple, Union, Optional, List

logger = logging.getLogger(__name__)

class ModelDiagnostics:
    """
    Comprehensive diagnostics for econometric models.
    """
    
    def __init__(self):
        """Initialize the diagnostics."""
        pass
    
    def residual_tests(self, residuals: Union[pd.Series, np.ndarray], 
                      n_lags: int = 12) -> Dict[str, Any]:
        """
        Run comprehensive tests on model residuals.
        
        Parameters
        ----------
        residuals : array_like
            Model residuals
        n_lags : int, optional
            Number of lags for autocorrelation test
            
        Returns
        -------
        dict
            Test results
        """
        logger.info(f"Running residual diagnostics tests with n_lags={n_lags}")
        
        try:
            # Ensure residuals is a numpy array
            residuals = np.asarray(residuals)
            
            # Normality (Jarque-Bera test)
            jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals)
            
            # Autocorrelation (Breusch-Godfrey test)
            # Create X matrix with lagged residuals
            X = np.ones((len(residuals), 1))  # Constant term
            bg_lm_stat, bg_lm_pval, bg_fstat, bg_fpval = acorr_breusch_godfrey(residuals, X, nlags=n_lags)
            
            # Heteroskedasticity (White test)
            # Squared residuals, X is just a constant
            white_stat, white_pval, f_stat, f_pval = het_white(residuals, X)
            
            # Basic statistics
            mean = np.mean(residuals)
            std = np.std(residuals)
            
            results = {
                'normality': {
                    'jb_statistic': jb_stat,
                    'jb_pvalue': jb_pval,
                    'skewness': skew,
                    'kurtosis': kurtosis,
                    'normal': jb_pval > 0.05
                },
                'autocorrelation': {
                    'bg_lm_statistic': bg_lm_stat,
                    'bg_lm_pvalue': bg_lm_pval,
                    'bg_f_statistic': bg_fstat,
                    'bg_f_pvalue': bg_fpval,
                    'no_autocorrelation': bg_lm_pval > 0.05
                },
                'heteroskedasticity': {
                    'white_statistic': white_stat,
                    'white_pvalue': white_pval,
                    'white_f_statistic': f_stat,
                    'white_f_pvalue': f_pval,
                    'homoskedastic': white_pval > 0.05
                },
                'statistics': {
                    'mean': mean,
                    'std': std,
                    'min': np.min(residuals),
                    'max': np.max(residuals)
                }
            }
            
            # Overall assessment
            results['overall'] = {
                'valid': (results['normality']['normal'] and 
                        results['autocorrelation']['no_autocorrelation'] and 
                        results['heteroskedasticity']['homoskedastic']),
                'issues': []
            }
            
            # Identify issues
            if not results['normality']['normal']:
                results['overall']['issues'].append("Non-normal residuals")
            if not results['autocorrelation']['no_autocorrelation']:
                results['overall']['issues'].append("Autocorrelation detected")
            if not results['heteroskedasticity']['homoskedastic']:
                results['overall']['issues'].append("Heteroskedasticity detected")
            
            logger.info(f"Residual diagnostics: normal={results['normality']['normal']}, "
                      f"no_autocorr={results['autocorrelation']['no_autocorrelation']}, "
                      f"homoskedastic={results['heteroskedasticity']['homoskedastic']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in residual diagnostics: {str(e)}")
            raise
    
    def plot_diagnostics(self, residuals: Union[pd.Series, np.ndarray], 
                        title: str = "Model Diagnostics") -> plt.Figure:
        """
        Create diagnostic plots for model residuals.
        
        Parameters
        ----------
        residuals : array_like
            Model residuals
        title : str, optional
            Plot title
            
        Returns
        -------
        matplotlib.figure.Figure
            Diagnostic plots
        """
        logger.info("Creating diagnostic plots")
        
        try:
            # Convert to pandas Series if not already
            if not isinstance(residuals, pd.Series):
                residuals = pd.Series(residuals)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(title, fontsize=14)
            
            # Time series plot of residuals
            axes[0, 0].plot(residuals.index if hasattr(residuals, 'index') else np.arange(len(residuals)), 
                          residuals)
            axes[0, 0].axhline(y=0, color='r', linestyle='-')
            axes[0, 0].set_title("Residuals")
            
            # Histogram of residuals with normal curve
            axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7)
            
            # Add normal curve
            from scipy import stats
            x = np.linspace(min(residuals), max(residuals), 100)
            mu, std = stats.norm.fit(residuals)
            p = stats.norm.pdf(x, mu, std)
            axes[0, 1].plot(x, p, 'k', linewidth=2)
            axes[0, 1].set_title("Histogram of Residuals")
            
            # QQ plot
            from statsmodels.graphics.gofplots import qqplot
            qqplot(residuals, line='s', ax=axes[1, 0])
            axes[1, 0].set_title("QQ Plot")
            
            # Autocorrelation plot
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(residuals, lags=20, ax=axes[1, 1])
            axes[1, 1].set_title("Autocorrelation")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in diagnostic plots: {str(e)}")
            raise
    
    def stability_test(self, model, data: Union[pd.DataFrame, np.ndarray], 
                      window_size: int = 30, step_size: int = 10) -> Dict[str, Any]:
        """
        Test parameter stability using rolling estimation.
        
        Parameters
        ----------
        model : callable
            Function to estimate model (takes data, returns parameters)
        data : array_like
            Time series data
        window_size : int, optional
            Size of the rolling window
        step_size : int, optional
            Step size for rolling window
            
        Returns
        -------
        dict
            Stability test results
        """
        logger.info(f"Testing model stability with window_size={window_size}, step_size={step_size}")
        
        try:
            # Ensure data is a numpy array
            if isinstance(data, pd.DataFrame):
                data_arr = data.values
            else:
                data_arr = np.asarray(data)
            
            # Get total sample size
            n = len(data_arr)
            
            # Calculate number of windows
            n_windows = (n - window_size) // step_size + 1
            
            # Create array to store parameters
            params_history = []
            dates = []
            
            # Rolling estimation
            for i in range(n_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                # Get window data
                window_data = data_arr[start_idx:end_idx]
                
                # Estimate model on window
                params = model(window_data)
                params_history.append(params)
                
                # Store corresponding date if available
                if hasattr(data, 'index') and isinstance(data.index, pd.DatetimeIndex):
                    dates.append(data.index[end_idx - 1])
                else:
                    dates.append(end_idx - 1)
            
            # Convert to numpy array
            params_history = np.array(params_history)
            
            # Calculate statistics
            mean_params = np.mean(params_history, axis=0)
            std_params = np.std(params_history, axis=0)
            cv_params = std_params / np.abs(mean_params)  # Coefficient of variation
            
            # Determine if parameters are stable (CV < 0.5 is stable)
            is_stable = (cv_params < 0.5).all()
            
            results = {
                'params_history': params_history,
                'dates': dates,
                'mean_params': mean_params,
                'std_params': std_params,
                'cv_params': cv_params,
                'stable': is_stable
            }
            
            logger.info(f"Stability test result: stable={results['stable']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in stability test: {str(e)}")
            raise
```

## Phase 4: Enhanced Threshold Cointegration Development (Week 7-9)

### 4.1 Create Threshold Cointegration Module

Enhance `src/models/threshold.py`:

```python
"""
Threshold cointegration module with comprehensive implementation.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.unitroot.cointegration import engle_granger
import logging
from typing import Dict, Any, Tuple, Union, Optional, List, Callable
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

class ThresholdCointegration:
    """
    Threshold cointegration model implementation.
    
    This model implements the Balke and Fomby (1997) threshold cointegration model
    for analyzing nonlinear price adjustment in the presence of transaction costs.
    """
    
    def __init__(self, data1: Union[pd.Series, np.ndarray], 
                data2: Union[pd.Series, np.ndarray], 
                max_lags: int = 10,
                trim: float = 0.15):
        """
        Initialize the threshold cointegration model.
        
        Parameters
        ----------
        data1 : array_like
            First time series (dependent variable)
        data2 : array_like
            Second time series (independent variable)
        max_lags : int, optional
            Maximum number of lags to consider
        trim : float, optional
            Trimming percentage for threshold estimation
        """
        # Store the data
        self.data1 = np.asarray(data1)
        self.data2 = np.asarray(data2)
        self.max_lags = max_lags
        self.trim = trim
        
        # Store dates if available
        if isinstance(data1, pd.Series) and isinstance(data1.index, pd.DatetimeIndex):
            self.dates = data1.index
        else:
            self.dates = np.arange(len(data1))
        
        # Initialize results
        self.coint_result = None
        self.threshold_result = None
        self.tvecm_result = None
        self.diagnostics_result = None
        
        logger.info(f"Initialized ThresholdCointegration with {len(data1)} observations, max_lags={max_lags}")
    
    def estimate_cointegration(self, trend: str = 'c') -> Dict[str, Any]:
        """
        Estimate the cointegration relationship.
        
        Parameters
        ----------
        trend : str, optional
            'c' : constant (default)
            'ct' : constant and trend
            'ctt' : constant, linear and quadratic trend
            'nc' : no constant, no trend
            
        Returns
        -------
        dict
            Cointegration test results
        """
        logger.info(f"Estimating cointegration relationship with trend='{trend}'")
        
        try:
            # Run Engle-Granger cointegration test
            result = engle_granger(self.data1, self.data2, trend=trend, lags=self.max_lags)
            
            # Store coefficients
            if trend == 'c':
                self.beta0 = result.coef[0]  # Intercept
                self.beta1 = result.coef[1]  # Slope
            elif trend == 'ct':
                self.beta0 = result.coef[0]  # Intercept
                self.beta1 = result.coef[1]  # Trend
                self.beta2 = result.coef[2]  # Slope
            elif trend == 'nc':
                self.beta0 = 0  # No intercept
                self.beta1 = result.coef[0]  # Slope
            
            # Calculate equilibrium errors
            if trend == 'c':
                self.eq_errors = self.data1 - (self.beta0 + self.beta1 * self.data2)
            elif trend == 'ct':
                t = np.arange(len(self.data1))
                self.eq_errors = self.data1 - (self.beta0 + self.beta1 * t + self.beta2 * self.data2)
            elif trend == 'nc':
                self.eq_errors = self.data1 - (self.beta1 * self.data2)
            
            # Store the result
            self.coint_result = {
                'statistic': result.stat,
                'pvalue': result.pvalue,
                'critical_values': result.critical_values,
                'cointegrated': result.pvalue < 0.05,
                'beta0': self.beta0,
                'beta1': self.beta1 if trend != 'ct' else self.beta2,
                'trend_coef': self.beta1 if trend == 'ct' else None,
                'residuals': self.eq_errors,
                'trend': trend
            }
            
            logger.info(f"Cointegration test result: statistic={result.stat:.4f}, "
                      f"p-value={result.pvalue:.4f}, cointegrated={self.coint_result['cointegrated']}")
            return self.coint_result
            
        except Exception as e:
            logger.error(f"Error in cointegration estimation: {str(e)}")
            raise
    
    def estimate_threshold(self, n_grid: int = 300, 
                          parallel: bool = True, n_jobs: int = -1) -> Dict[str, Any]:
        """
        Estimate the threshold parameter using grid search.
        
        Parameters
        ----------
        n_grid : int, optional
            Number of grid points to evaluate
        parallel : bool, optional
            Whether to use parallel processing
        n_jobs : int, optional
            Number of parallel jobs (default: use all cores)
            
        Returns
        -------
        dict
            Threshold estimation results
        """
        logger.info(f"Estimating threshold with n_grid={n_grid}, parallel={parallel}")
        
        # Ensure we have cointegration results
        if not hasattr(self, 'eq_errors'):
            logger.warning("No cointegration relationship estimated. Running with default parameters.")
            self.estimate_cointegration()
        
        try:
            # Identify candidates for threshold
            sorted_errors = np.sort(self.eq_errors)
            lower_idx = int(len(sorted_errors) * self.trim)
            upper_idx = int(len(sorted_errors) * (1 - self.trim))
            candidates = sorted_errors[lower_idx:upper_idx]
            
            # Grid search for threshold
            if len(candidates) > n_grid:
                # Choose subset of candidates
                indices = np.linspace(0, len(candidates) - 1, n_grid, dtype=int)
                candidates = candidates[indices]
            
            logger.info(f"Grid search over {len(candidates)} threshold candidates")
            
            # Compute SSR for each threshold
            if parallel:
                # Parallel computation
                ssrs = Parallel(n_jobs=n_jobs)(
                    delayed(self._compute_ssr_for_threshold)(threshold) 
                    for threshold in candidates
                )
            else:
                # Sequential computation
                ssrs = [self._compute_ssr_for_threshold(threshold) for threshold in candidates]
            
            # Find best threshold
            best_idx = np.argmin(ssrs)
            best_threshold = candidates[best_idx]
            best_ssr = ssrs[best_idx]
            
            # Store results
            self.threshold = best_threshold
            self.ssr = best_ssr
            
            # Calculate threshold as percentage of transactions
            threshold_pct = np.mean(self.eq_errors <= best_threshold) * 100
            
            # Calculate AIC and BIC
            n = len(self.data1) - 1  # Adjust for differencing
            k = 2 * (2 + self.max_lags * 2)  # Parameters in the model (2 regimes)
            aic = n * np.log(best_ssr / n) + 2 * k
            bic = n * np.log(best_ssr / n) + k * np.log(n)
            
            # Test significance of threshold effect
            linear_ssr = self._compute_linear_ssr()
            lr_stat = n * (np.log(linear_ssr) - np.log(best_ssr))
            
            # Store threshold results
            self.threshold_result = {
                'threshold': best_threshold,
                'ssr': best_ssr,
                'linear_ssr': linear_ssr,
                'thresholds': candidates,
                'ssrs': ssrs,
                'percentage_below': threshold_pct,
                'aic': aic,
                'bic': bic,
                'lr_statistic': lr_stat
            }
            
            logger.info(f"Threshold estimation result: threshold={best_threshold:.4f}, "
                      f"SSR={best_ssr:.4f}, {threshold_pct:.1f}% of observations below threshold")
            
            # Bootstrap p-value for threshold effect
            if lr_stat > 0:
                logger.info("Bootstrap test for threshold effect")
                bootstrap_pval = self._bootstrap_threshold_test(n_bootstrap=200, lr_stat=lr_stat)
                self.threshold_result['bootstrap_pvalue'] = bootstrap_pval
                self.threshold_result['significant_threshold'] = bootstrap_pval < 0.05
                
                logger.info(f"Bootstrap test result: p-value={bootstrap_pval:.4f}, "
                          f"significant={self.threshold_result['significant_threshold']}")
            
            return self.threshold_result
            
        except Exception as e:
            logger.error(f"Error in threshold estimation: {str(e)}")
            raise
    
    def _compute_ssr_for_threshold(self, threshold: float) -> float:
        """
        Compute sum of squared residuals for a given threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value to evaluate
            
        Returns
        -------
        float
            Sum of squared residuals
        """
        try:
            # Indicator function
            below = self.eq_errors[:-1] <= threshold
            
            # Build regressor matrix X for the TAR model
            X = []
            
            # Error correction terms (split by threshold)
            X.append(self.eq_errors[:-1] * below)  # ECT for regime 1
            X.append(self.eq_errors[:-1] * (~below))  # ECT for regime 2
            
            # Add lagged differences
            for lag in range(1, self.max_lags + 1):
                if lag < len(self.data1) - 1:
                    # Lag of Δy
                    lag_dy = np.roll(np.diff(self.data1), lag)[:-1]
                    # Lag of Δx
                    lag_dx = np.roll(np.diff(self.data2), lag)[:-1]
                    
                    X.append(lag_dy)
                    X.append(lag_dx)
            
            # Convert to numpy array and add constant
            X = np.column_stack(X)
            X = sm.add_constant(X)
            
            # Dependent variable: Δy
            y = np.diff(self.data1)
            
            # Fit model
            model = sm.OLS(y, X)
            results = model.fit()
            
            return results.ssr
            
        except Exception as e:
            logger.error(f"Error computing SSR for threshold={threshold}: {str(e)}")
            raise
    
    def _compute_linear_ssr(self) -> float:
        """
        Compute SSR for linear VECM (no threshold).
        
        Returns
        -------
        float
            Sum of squared residuals for linear model
        """
        try:
            # Build regressor matrix X for the linear VECM
            X = []
            
            # Error correction term (no threshold)
            X.append(self.eq_errors[:-1])
            
            # Add lagged differences
            for lag in range(1, self.max_lags + 1):
                if lag < len(self.data1) - 1:
                    # Lag of Δy
                    lag_dy = np.roll(np.diff(self.data1), lag)[:-1]
                    # Lag of Δx
                    lag_dx = np.roll(np.diff(self.data2), lag)[:-1]
                    
                    X.append(lag_dy)
                    X.append(lag_dx)
            
            # Convert to numpy array and add constant
            X = np.column_stack(X)
            X = sm.add_constant(X)
            
            # Dependent variable: Δy
            y = np.diff(self.data1)
            
            # Fit model
            model = sm.OLS(y, X)
            results = model.fit()
            
            return results.ssr
            
        except Exception as e:
            logger.error(f"Error computing linear SSR: {str(e)}")
            raise
    
    def _bootstrap_threshold_test(self, n_bootstrap: int = 200, 
                                lr_stat: float = None) -> float:
        """
        Bootstrap test for threshold effect.
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap samples
        lr_stat : float, optional
            LR statistic from actual data
            
        Returns
        -------
        float
            Bootstrap p-value
        """
        try:
            # Get residuals from linear VECM
            X = []
            
            # Error correction term (no threshold)
            X.append(self.eq_errors[:-1])
            
            # Add lagged differences
            for lag in range(1, self.max_lags + 1):
                if lag < len(self.data1) - 1:
                    lag_dy = np.roll(np.diff(self.data1), lag)[:-1]
                    lag_dx = np.roll(np.diff(self.data2), lag)[:-1]
                    X.append(lag_dy)
                    X.append(lag_dx)
            
            X = np.column_stack(X)
            X = sm.add_constant(X)
            y = np.diff(self.data1)
            
            model = sm.OLS(y, X)
            results = model.fit()
            
            # Get residuals
            residuals = results.resid
            
            # Get fitted values
            fitted = results.fittedvalues
            
            # Bootstrap loop
            lr_stats = []
            
            for i in range(n_bootstrap):
                # Sample residuals with replacement
                boot_resid = np.random.choice(residuals, size=len(residuals))
                
                # Create bootstrap sample
                boot_y = fitted + boot_resid
                
                # Compute SSR for linear model
                boot_model = sm.OLS(boot_y, X)
                boot_results = boot_model.fit()
                linear_ssr = boot_results.ssr
                
                # Grid search for threshold model
                sorted_errors = np.sort(self.eq_errors[:-1])
                lower_idx = int(len(sorted_errors) * self.trim)
                upper_idx = int(len(sorted_errors) * (1 - self.trim))
                candidates = sorted_errors[lower_idx:upper_idx]
                
                # Randomly select candidates for efficiency
                if len(candidates) > 50:
                    candidates = np.random.choice(candidates, size=50, replace=False)
                
                ssrs = []
                for threshold in candidates:
                    # Indicator function
                    below = self.eq_errors[:-1] <= threshold
                    
                    # Create threshold regressor
                    X_threshold = np.column_stack([
                        np.ones(X.shape[0]),
                        self.eq_errors[:-1] * below,
                        self.eq_errors[:-1] * (~below),
                        X[:, 2:]  # Skip constant and EC term in original X
                    ])
                    
                    # Fit threshold model
                    boot_threshold_model = sm.OLS(boot_y, X_threshold)
                    boot_threshold_results = boot_threshold_model.fit()
                    ssrs.append(boot_threshold_results.ssr)
                
                # Get minimum SSR
                threshold_ssr = min(ssrs)
                
                # Calculate LR statistic
                n = len(boot_y)
                boot_lr_stat = n * (np.log(linear_ssr) - np.log(threshold_ssr))
                lr_stats.append(boot_lr_stat)
            
            # Calculate p-value
            p_value = np.mean(np.array(lr_stats) > lr_stat)
            
            return p_value
            
        except Exception as e:
            logger.error(f"Error in bootstrap threshold test: {str(e)}")
            raise
    
    def estimate_tvecm(self) -> Dict[str, Any]:
        """
        Estimate the Threshold Vector Error Correction Model.
        
        Returns
        -------
        dict
            TVECM estimation results
        """
        logger.info("Estimating Threshold VECM")
        
        # Ensure we have a threshold
        if not hasattr(self, 'threshold'):
            logger.warning("No threshold estimated. Running threshold estimation with default parameters.")
            self.estimate_threshold()
        
        try:
            # Indicator function for threshold
            below = self.eq_errors[:-1] <= self.threshold
            above = ~below
            
            # Build regressor matrix for threshold model
            X = []
            
            # Error correction terms (split by threshold)
            X.append(self.eq_errors[:-1] * below)  # ECT for regime 1
            X.append(self.eq_errors[:-1] * above)  # ECT for regime 2
            
            # Add lagged differences
            for lag in range(1, self.max_lags + 1):
                if lag < len(self.data1) - 1:
                    lag_dy = np.roll(np.diff(self.data1), lag)[:-1]
                    lag_dx = np.roll(np.diff(self.data2), lag)[:-1]
                    X.append(lag_dy)
                    X.append(lag_dx)
            
            # Convert to numpy array and add constant
            X = np.column_stack(X)
            X = sm.add_constant(X)
            
            # Dependent variables: Δy and Δx
            dy = np.diff(self.data1)
            dx = np.diff(self.data2)
            
            # Fit models for both equations
            model1 = sm.OLS(dy, X)
            model2 = sm.OLS(dx, X)
            
            results1 = model1.fit()
            results2 = model2.fit()
            
            # Extract adjustment speeds
            adj_below_1 = results1.params[1]  # Adjustment in lower regime for equation 1
            adj_above_1 = results1.params[2]  # Adjustment in upper regime for equation 1
            adj_below_2 = results2.params[1]  # Adjustment in lower regime for equation 2
            adj_above_2 = results2.params[2]  # Adjustment in upper regime for equation 2
            
            # Calculate half-lives
            half_life_below_1 = np.log(0.5) / np.log(1 + adj_below_1) if adj_below_1 < 0 else float('inf')
            half_life_above_1 = np.log(0.5) / np.log(1 + adj_above_1) if adj_above_1 < 0 else float('inf')
            half_life_below_2 = np.log(0.5) / np.log(1 + adj_below_2) if adj_below_2 < 0 else float('inf')
            half_life_above_2 = np.log(0.5) / np.log(1 + adj_above_2) if adj_above_2 < 0 else float('inf')
            
            # Store results
            self.tvecm_result = {
                'equation1': results1,
                'equation2': results2,
                'adjustment_below_1': adj_below_1,
                'adjustment_above_1': adj_above_1,
                'adjustment_below_2': adj_below_2,
                'adjustment_above_2': adj_above_2,
                'half_life_below_1': half_life_below_1,
                'half_life_above_1': half_life_above_1,
                'half_life_below_2': half_life_below_2,
                'half_life_above_2': half_life_above_2,
                'threshold': self.threshold,
                'cointegration_beta0': self.beta0,
                'cointegration_beta1': self.beta1,
                'n_below': sum(below),
                'n_above': sum(above),
                'pct_below': 100 * sum(below) / len(below),
                'pct_above': 100 * sum(above) / len(above)
            }
            
            logger.info(f"TVECM estimation complete. Adjustment speeds: "
                      f"below_1={adj_below_1:.4f}, above_1={adj_above_1:.4f}, "
                      f"below_2={adj_below_2:.4f}, above_2={adj_above_2:.4f}")
            
            # Calculate t-statistics for adjustment speeds
            self.tvecm_result['t_adj_below_1'] = results1.tvalues[1]
            self.tvecm_result['t_adj_above_1'] = results1.tvalues[2]
            self.tvecm_result['t_adj_below_2'] = results2.tvalues[1]
            self.tvecm_result['t_adj_above_2'] = results2.tvalues[2]
            
            # Calculate p-values for adjustment speeds
            self.tvecm_result['p_adj_below_1'] = results1.pvalues[1]
            self.tvecm_result['p_adj_above_1'] = results1.pvalues[2]
            self.tvecm_result['p_adj_below_2'] = results2.pvalues[1]
            self.tvecm_result['p_adj_above_2'] = results2.pvalues[2]
            
            # Calculate asymmetry test (H0: adj_below = adj_above)
            from scipy import stats
            
            # For equation 1
            diff1 = adj_below_1 - adj_above_1
            se1 = np.sqrt(results1.cov_params.iloc[1, 1] + results1.cov_params.iloc[2, 2] - 
                        2 * results1.cov_params.iloc[1, 2])
            t_stat1 = diff1 / se1
            p_value1 = 2 * (1 - stats.t.cdf(abs(t_stat1), results1.df_resid))
            
            # For equation 2
            diff2 = adj_below_2 - adj_above_2
            se2 = np.sqrt(results2.cov_params.iloc[1, 1] + results2.cov_params.iloc[2, 2] - 
                        2 * results2.cov_params.iloc[1, 2])
            t_stat2 = diff2 / se2
            p_value2 = 2 * (1 - stats.t.cdf(abs(t_stat2), results2.df_resid))
            
            self.tvecm_result['asymmetry_test'] = {
                'diff1': diff1,
                't_stat1': t_stat1,
                'p_value1': p_value1,
                'asymmetric1': p_value1 < 0.05,
                'diff2': diff2,
                't_stat2': t_stat2,
                'p_value2': p_value2,
                'asymmetric2': p_value2 < 0.05
            }
            
            logger.info(f"Asymmetry test: p-value1={p_value1:.4f}, p-value2={p_value2:.4f}")
            
            return self.tvecm_result
            
        except Exception as e:
            logger.error(f"Error in TVECM estimation: {str(e)}")
            raise
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostic tests on the TVECM model.
        
        Returns
        -------
        dict
            Diagnostic test results
        """
        logger.info("Running TVECM diagnostics")
        
        # Ensure we have TVECM results
        if self.tvecm_result is None:
            logger.warning("No TVECM estimated. Running TVECM with default parameters.")
            self.estimate_tvecm()
        
        try:
            from src.models.diagnostics import ModelDiagnostics
            diagnostics = ModelDiagnostics()
            
            # Get residuals from both equations
            residuals1 = self.tvecm_result['equation1'].resid
            residuals2 = self.tvecm_result['equation2'].resid
            
            # Run residual diagnostics
            diag1 = diagnostics.residual_tests(residuals1)
            diag2 = diagnostics.residual_tests(residuals2)
            
            # Store diagnostic results
            self.diagnostics_result = {
```python
                'equation1': diag1,
                'equation2': diag2,
                'combined': {
                    'valid': diag1['overall']['valid'] and diag2['overall']['valid'],
                    'issues': diag1['overall']['issues'] + diag2['overall']['issues']
                }
            }
            
            # Create diagnostic plots
            self.diagnostics_result['plots'] = {
                'equation1': diagnostics.plot_diagnostics(residuals1, 
                                                       title="Diagnostics - Equation 1"),
                'equation2': diagnostics.plot_diagnostics(residuals2, 
                                                       title="Diagnostics - Equation 2")
            }
            
            # Test parameter stability
            # Need to define a function that estimates parameters on subsample
            def estimate_params(data_subset):
                y1 = np.diff(data_subset[:, 0])
                y2 = np.diff(data_subset[:, 1])
                
                # Calculate cointegrating residuals
                if hasattr(self, 'beta0') and hasattr(self, 'beta1'):
                    errors = data_subset[:, 0] - (self.beta0 + self.beta1 * data_subset[:, 1])
                    errors = errors[:-1]  # Adjust for differencing
                    
                    below = errors <= self.threshold
                    above = ~below
                    
                    # Build regressor matrix (simplified)
                    X = np.column_stack([
                        np.ones(len(y1)),
                        errors * below,
                        errors * above
                    ])
                    
                    # Fit models
                    model1 = sm.OLS(y1, X)
                    results1 = model1.fit()
                    
                    # Return key parameters (adjustment speeds)
                    return results1.params[1:3]  # Return adjustment speeds only
                else:
                    return np.array([np.nan, np.nan])
            
            # Run stability test
            data_arr = np.column_stack([self.data1, self.data2])
            self.diagnostics_result['stability'] = diagnostics.stability_test(
                estimate_params, data_arr
            )
            
            logger.info(f"Diagnostics complete. Model valid: {self.diagnostics_result['combined']['valid']}")
            
            return self.diagnostics_result
            
        except Exception as e:
            logger.error(f"Error in diagnostics: {str(e)}")
            raise
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete threshold cointegration analysis.
        
        Returns
        -------
        dict
            Complete analysis results
        """
        logger.info("Running full threshold cointegration analysis")
        
        try:
            # Step 1: Estimate cointegration relationship
            self.estimate_cointegration()
            
            # If not cointegrated, may not proceed
            if not self.coint_result['cointegrated']:
                logger.warning("Series are not cointegrated. Threshold estimation may not be valid.")
            
            # Step 2: Estimate threshold
            self.estimate_threshold()
            
            # Step 3: Estimate TVECM
            self.estimate_tvecm()
            
            # Step 4: Run diagnostics
            self.run_diagnostics()
            
            # Combine all results
            results = {
                'cointegration': self.coint_result,
                'threshold': self.threshold_result,
                'tvecm': self.tvecm_result,
                'diagnostics': self.diagnostics_result,
                'data': {
                    'n_obs': len(self.data1),
                    'dates': self.dates
                }
            }
            
            logger.info("Full analysis complete")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis: {str(e)}")
            raise
```

### 4.2 Enhanced Threshold VECM Module

```python
"""
Enhanced Threshold Vector Error Correction Model implementation.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM
import logging
from typing import Dict, Any, Tuple, Union, Optional, List
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

class ThresholdVECM:
    """
    Enhanced Threshold Vector Error Correction Model (TVECM) implementation.
    
    This class implements a two-regime threshold VECM following
    Hansen & Seo (2002) methodology with comprehensive diagnostics
    and bootstrap inference.
    """
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], 
                k_ar_diff: int = 1, deterministic: str = "ci", 
                coint_rank: int = 1, trim: float = 0.15):
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
        coint_rank : int, optional
            Cointegration rank
        trim : float, optional
            Trimming percentage for threshold estimation
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
            self.dates = data.index if isinstance(data.index, pd.DatetimeIndex) else None
            self.variable_names = data.columns.tolist()
        else:
            self.data = pd.DataFrame(data)
            self.dates = None
            self.variable_names = [f'y{i+1}' for i in range(data.shape[1])]
        
        self.k_ar_diff = k_ar_diff
        self.deterministic = deterministic
        self.coint_rank = coint_rank
        self.trim = trim
        
        # Initialize results
        self.linear_results = None
        self.threshold_results = None
        self.tvecm_results = None
        
        logger.info(f"Initialized ThresholdVECM with {len(self.data)} observations, "
                  f"{self.data.shape[1]} variables, k_ar_diff={k_ar_diff}")
    
    def estimate_linear_vecm(self) -> Any:
        """
        Estimate the linear VECM model (no threshold).
        
        Returns
        -------
        statsmodels.tsa.vector_ar.vecm.VECMResults
            Linear VECM estimation results
        """
        logger.info(f"Estimating linear VECM with deterministic='{self.deterministic}', "
                  f"k_ar_diff={self.k_ar_diff}, coint_rank={self.coint_rank}")
        
        try:
            # Estimate linear VECM
            model = VECM(
                self.data, 
                k_ar_diff=self.k_ar_diff, 
                deterministic=self.deterministic,
                coint_rank=self.coint_rank
            )
            
            self.linear_model = model
            self.linear_results = model.fit()
            
            # Extract the cointegration vector
            self.beta = self.linear_results.beta
            
            # Extract loading matrix (adjustment speeds)
            self.alpha = self.linear_results.alpha
            
            # Calculate cointegration relation
            self._calculate_coint_relation()
            
            logger.info(f"Linear VECM estimation complete. "
                      f"Log-likelihood: {self.linear_results.llf:.4f}")
            
            return self.linear_results
            
        except Exception as e:
            logger.error(f"Error in linear VECM estimation: {str(e)}")
            raise
    
    def _calculate_coint_relation(self):
        """Calculate the cointegration relation (error correction term)."""
        try:
            # Get data matrix
            y = self.data.values
            
            # Create Z matrix based on deterministic specification
            if self.deterministic == "ci":
                z = np.column_stack([np.ones(len(y)), y])[:, :-1]  # Remove last column (constant)
            elif self.deterministic == "li":
                trend = np.arange(len(y)).reshape(-1, 1)
                z = np.column_stack([np.ones(len(y)), trend, y])[:, :-1]
            else:
                z = y
            
            # Calculate cointegration relation
            self.coint_relation = z @ self.beta
            
            logger.info(f"Calculated cointegration relation with shape {self.coint_relation.shape}")
            
        except Exception as e:
            logger.error(f"Error calculating cointegration relation: {str(e)}")
            raise
    
    def grid_search_threshold(self, trim: float = None, n_grid: int = 300, 
                             parallel: bool = True, n_jobs: int = -1,
                             verbose: bool = False) -> Dict[str, Any]:
        """
        Perform grid search to find the optimal threshold.
        
        Parameters
        ----------
        trim : float, optional
            Trimming percentage (overrides the instance attribute)
        n_grid : int, optional
            Number of grid points
        parallel : bool, optional
            Whether to use parallel processing
        n_jobs : int, optional
            Number of parallel jobs
        verbose : bool, optional
            Whether to print progress
            
        Returns
        -------
        dict
            Threshold estimation results
        """
        # Ensure linear VECM is estimated
        if not hasattr(self, 'beta'):
            logger.warning("Linear VECM not estimated. Running with default parameters.")
            self.estimate_linear_vecm()
        
        # Set trim if provided
        trim = trim or self.trim
        
        logger.info(f"Grid search for threshold with trim={trim}, n_grid={n_grid}, parallel={parallel}")
        
        try:
            # Sort cointegration relation values
            sorted_errors = np.sort(self.coint_relation.flatten())
            
            # Apply trimming
            lower_idx = int(len(sorted_errors) * trim)
            upper_idx = int(len(sorted_errors) * (1 - trim))
            candidates = sorted_errors[lower_idx:upper_idx]
            
            # Reduce grid if needed
            if len(candidates) > n_grid:
                indices = np.linspace(0, len(candidates) - 1, n_grid, dtype=int)
                candidates = candidates[indices]
            
            logger.info(f"Evaluating {len(candidates)} threshold candidates")
            
            # Grid search for threshold
            if parallel:
                # Parallel computation
                llfs = Parallel(n_jobs=n_jobs)(
                    delayed(self._compute_llf_for_threshold)(threshold) 
                    for threshold in candidates
                )
            else:
                # Sequential computation
                llfs = []
                for i, threshold in enumerate(candidates):
                    if verbose and (i % 10 == 0):
                        logger.info(f"Processing threshold candidate {i+1}/{len(candidates)}")
                    llfs.append(self._compute_llf_for_threshold(threshold))
            
            # Find best threshold (maximum likelihood)
            best_idx = np.argmax(llfs)
            best_threshold = candidates[best_idx]
            best_llf = llfs[best_idx]
            
            # Store threshold
            self.threshold = best_threshold
            self.llf = best_llf
            
            # Calculate threshold as percentage of observations
            threshold_pct = np.mean(self.coint_relation <= best_threshold) * 100
            
            # Calculate AIC and BIC
            n = len(self.data)
            k = 2 * (self.data.shape[1] * (1 + self.k_ar_diff))  # Parameters in the model (2 regimes)
            aic = -2 * best_llf + 2 * k
            bic = -2 * best_llf + k * np.log(n)
            
            # Calculate LR test against linear model
            lr_stat = 2 * (best_llf - self.linear_results.llf)
            
            # Store threshold results
            self.threshold_results = {
                'threshold': best_threshold,
                'llf': best_llf,
                'thresholds': candidates,
                'llfs': llfs,
                'percentage_below': threshold_pct,
                'aic': aic,
                'bic': bic,
                'lr_statistic': lr_stat,
                'linear_llf': self.linear_results.llf
            }
            
            logger.info(f"Threshold estimation result: threshold={best_threshold:.4f}, "
                      f"LLF={best_llf:.4f}, {threshold_pct:.1f}% observations below threshold")
            
            # Bootstrap test for threshold effect
            if lr_stat > 0:
                logger.info("Bootstrap test for threshold effect")
                p_value = self._bootstrap_threshold_test(n_bootstrap=200)
                self.threshold_results['bootstrap_pvalue'] = p_value
                self.threshold_results['significant_threshold'] = p_value < 0.05
                
                logger.info(f"Bootstrap test result: p-value={p_value:.4f}, "
                          f"significant={self.threshold_results['significant_threshold']}")
            
            return self.threshold_results
            
        except Exception as e:
            logger.error(f"Error in threshold grid search: {str(e)}")
            raise
    
    def _compute_llf_for_threshold(self, threshold: float) -> float:
        """
        Compute log-likelihood for a given threshold.
        
        Parameters
        ----------
        threshold : float
            Threshold value
            
        Returns
        -------
        float
            Log-likelihood
        """
        try:
            # Indicator functions
            below = self.coint_relation <= threshold
            above = ~below
            
            # Count observations in each regime
            n_below = sum(below)
            n_above = sum(above)
            
            # Skip if any regime has too few observations
            min_obs = 10 + self.data.shape[1] * self.k_ar_diff
            if n_below < min_obs or n_above < min_obs:
                return -np.inf
            
            # Calculate EC term with lag
            ec_term = np.roll(self.coint_relation, 1)
            ec_term[0] = ec_term[1]  # Fix first observation
            
            # Prepare data differences
            y_diff = np.diff(self.data.values, axis=0)
            y_diff = np.vstack([y_diff[0], y_diff])  # Add first observation to match dimensions
            
            # Create matrices for both regimes
            X_below = []
            X_below.append(np.ones(len(self.data)) * below)  # Constant for regime 1
            X_below.append(ec_term * below)  # EC term for regime 1
            
            X_above = []
            X_above.append(np.ones(len(self.data)) * above)  # Constant for regime 2
            X_above.append(ec_term * above)  # EC term for regime 2
            
            # Add lagged differences
            for lag in range(1, self.k_ar_diff + 1):
                if lag < len(y_diff):
                    lag_diff = np.roll(y_diff, lag, axis=0)
                    lag_diff[:lag] = lag_diff[lag]  # Fix initial observations
                    
                    for col in range(y_diff.shape[1]):
                        X_below.append(lag_diff[:, col] * below)
                        X_above.append(lag_diff[:, col] * above)
            
            # Combine matrices
            X = np.column_stack(X_below + X_above)
            
            # Calculate log-likelihood for each equation
            llf_total = 0
            
            for eq in range(y_diff.shape[1]):
                y = y_diff[:, eq]
                
                # Fit model
                model = sm.OLS(y, X)
                results = model.fit()
                
                # Add log-likelihood
                llf_total += results.llf
            
            return llf_total
            
        except Exception as e:
            logger.error(f"Error computing log-likelihood for threshold={threshold}: {str(e)}")
            return -np.inf
    
    def _bootstrap_threshold_test(self, n_bootstrap: int = 200) -> float:
        """
        Bootstrap test for threshold effect.
        
        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap replications
            
        Returns
        -------
        float
            Bootstrap p-value
        """
        try:
            # Get residuals from linear VECM
            residuals = self.linear_results.resid.values
            
            # Get predicted values from linear VECM
            y_diff = np.diff(self.data.values, axis=0)
            X = self.linear_results.predict().values
            
            # LR statistic from actual data
            lr_stat = self.threshold_results['lr_statistic']
            
            # Bootstrap loop
            lr_stats = []
            
            for i in range(n_bootstrap):
                if i % 10 == 0:
                    logger.info(f"Bootstrap iteration {i+1}/{n_bootstrap}")
                
                # Sample residuals with replacement
                boot_residuals = np.zeros_like(residuals)
                for eq in range(residuals.shape[1]):
                    # Sample residuals for each equation
                    boot_residuals[:, eq] = np.random.choice(residuals[:, eq], size=len(residuals))
                
                # Create bootstrap sample
                boot_y_diff = X + boot_residuals
                
                # Cumulative sum to get levels (reverse differencing)
                boot_y = np.zeros((len(boot_y_diff) + 1, boot_y_diff.shape[1]))
                boot_y[0] = self.data.values[0]  # Use actual first observation
                boot_y[1:] = self.data.values[0] + np.cumsum(boot_y_diff, axis=0)
                
                # Create bootstrap data
                boot_data = pd.DataFrame(boot_y, columns=self.variable_names)
                
                # Create bootstrap model
                boot_model = ThresholdVECM(
                    boot_data,
                    k_ar_diff=self.k_ar_diff,
                    deterministic=self.deterministic,
                    coint_rank=self.coint_rank,
                    trim=self.trim
                )
                
                # Estimate linear VECM
                boot_model.estimate_linear_vecm()
                
                # Grid search for threshold (simplified)
                boot_result = boot_model.grid_search_threshold(
                    n_grid=50,  # Use fewer grid points for speed
                    parallel=False
                )
                
                # Store LR statistic
                lr_stats.append(boot_result['lr_statistic'])
            
            # Calculate p-value
            p_value = np.mean(np.array(lr_stats) > lr_stat)
            
            return p_value
            
        except Exception as e:
            logger.error(f"Error in bootstrap threshold test: {str(e)}")
            raise
    
    def estimate_tvecm(self) -> Dict[str, Any]:
        """
        Estimate the Threshold VECM.
        
        Returns
        -------
        dict
            TVECM estimation results
        """
        logger.info("Estimating Threshold VECM")
        
        # Ensure threshold is estimated
        if not hasattr(self, 'threshold'):
            logger.warning("No threshold estimated. Running grid search with default parameters.")
            self.grid_search_threshold()
        
        try:
            # Indicator functions
            below = self.coint_relation <= self.threshold
            above = ~below
            
            # Calculate EC term with lag
            ec_term = np.roll(self.coint_relation, 1)
            ec_term[0] = ec_term[1]  # Fix first observation
            
            # Prepare data differences
            y_diff = np.diff(self.data.values, axis=0)
            y_diff = np.vstack([y_diff[0], y_diff])  # Add first observation to match dimensions
            
            # Create matrices for both regimes
            X_below = []
            X_below.append(np.ones(len(self.data)) * below)  # Constant for regime 1
            X_below.append(ec_term * below)  # EC term for regime 1
            
            X_above = []
            X_above.append(np.ones(len(self.data)) * above)  # Constant for regime 2
            X_above.append(ec_term * above)  # EC term for regime 2
            
            # Add lagged differences
            for lag in range(1, self.k_ar_diff + 1):
                if lag < len(y_diff):
                    lag_diff = np.roll(y_diff, lag, axis=0)
                    lag_diff[:lag] = lag_diff[lag]  # Fix initial observations
                    
                    for col in range(y_diff.shape[1]):
                        X_below.append(lag_diff[:, col] * below)
                        X_above.append(lag_diff[:, col] * above)
            
            # Combine matrices
            X = np.column_stack(X_below + X_above)
            
            # Estimate equations
            results = []
            
            for eq in range(y_diff.shape[1]):
                y = y_diff[:, eq]
                
                # Fit model
                model = sm.OLS(y, X)
                eq_results = model.fit()
                results.append(eq_results)
            
            # Extract parameters
            n_vars = y_diff.shape[1]
            n_params_below = 2 + self.k_ar_diff * n_vars  # Constant, EC term, lagged differences
            
            regime_params = {'below': {}, 'above': {}}
            
            for eq in range(n_vars):
                # Extract parameters for each regime
                params_below = results[eq].params[:n_params_below]
                params_above = results[eq].params[n_params_below:]
                
                regime_params['below'][f'equation{eq+1}'] = params_below
                regime_params['above'][f'equation{eq+1}'] = params_above
            
            # Extract adjustment speeds (alpha)
            alpha_below = np.array([results[eq].params[1] for eq in range(n_vars)])
            alpha_above = np.array([results[eq].params[n_params_below+1] for eq in range(n_vars)])
            
            # Calculate half-lives
            half_lives_below = []
            half_lives_above = []
            
            for eq in range(n_vars):
                a_below = alpha_below[eq]
                a_above = alpha_above[eq]
                
                # Calculate half-lives if adjustment coefficients have correct sign
                hl_below = np.log(0.5) / np.log(1 + a_below) if a_below < 0 else np.inf
                hl_above = np.log(0.5) / np.log(1 + a_above) if a_above < 0 else np.inf
                
                half_lives_below.append(hl_below)
                half_lives_above.append(hl_above)
            
            # Store results
            self.tvecm_results = {
                'equations': results,
                'threshold': self.threshold,
                'cointegration_beta': self.beta,
                'adjustment_below': alpha_below,
                'adjustment_above': alpha_above,
                'half_lives_below': half_lives_below,
                'half_lives_above': half_lives_above,
                'params_below': regime_params['below'],
                'params_above': regime_params['above'],
                'n_below': sum(below),
                'n_above': sum(above),
                'pct_below': 100 * sum(below) / len(below),
                'pct_above': 100 * sum(above) / len(above)
            }
            
            logger.info(f"TVECM estimation complete")
            
            # Calculate asymmetry tests
            self._calculate_asymmetry_tests()
            
            return self.tvecm_results
            
        except Exception as e:
            logger.error(f"Error in TVECM estimation: {str(e)}")
            raise
    
    def _calculate_asymmetry_tests(self):
        """Calculate asymmetry tests for adjustment speeds."""
        try:
            from scipy import stats
            
            n_vars = len(self.tvecm_results['adjustment_below'])
            
            asymmetry_tests = []
            
            for eq in range(n_vars):
                # Get adjustment speeds
                a_below = self.tvecm_results['adjustment_below'][eq]
                a_above = self.tvecm_results['adjustment_above'][eq]
                
                # Get standard errors
                se_below = self.tvecm_results['equations'][eq].bse[1]
                se_above = self.tvecm_results['equations'][eq].bse[1 + len(self.tvecm_results['params_below'][f'equation{eq+1}'])]
                
                # Calculate test statistic
                diff = a_below - a_above
                se_diff = np.sqrt(se_below**2 + se_above**2)  # Assuming independence
                t_stat = diff / se_diff
                
                # Calculate p-value
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(self.data) - 2 * len(self.tvecm_results['params_below'][f'equation{eq+1}'])))
                
                asymmetry_tests.append({
                    'diff': diff,
                    'se_diff': se_diff,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'asymmetric': p_value < 0.05
                })
            
            self.tvecm_results['asymmetry_tests'] = asymmetry_tests
            
        except Exception as e:
            logger.error(f"Error calculating asymmetry tests: {str(e)}")
            raise
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostic tests on TVECM model.
        
        Returns
        -------
        dict
            Diagnostic test results
        """
        logger.info("Running TVECM diagnostics")
        
        # Ensure TVECM is estimated
        if not hasattr(self, 'tvecm_results'):
            logger.warning("No TVECM estimated. Running TVECM with default parameters.")
            self.estimate_tvecm()
        
        try:
            from src.models.diagnostics import ModelDiagnostics
            diagnostics = ModelDiagnostics()
            
            # Get residuals from each equation
            residuals = [results.resid for results in self.tvecm_results['equations']]
            
            # Run diagnostics on each equation
            diagnostic_results = []
            
            for eq, resid in enumerate(residuals):
                eq_diag = diagnostics.residual_tests(resid)
                diagnostic_results.append(eq_diag)
            
            # Determine overall model validity
            valid = all(diag['overall']['valid'] for diag in diagnostic_results)
            
            # Collect all issues
            issues = []
            for eq, diag in enumerate(diagnostic_results):
                for issue in diag['overall']['issues']:
                    issues.append(f"Equation {eq+1}: {issue}")
            
            # Store diagnostics
            self.diagnostic_results = {
                'equations': diagnostic_results,
                'valid': valid,
                'issues': issues
            }
            
            logger.info(f"Diagnostics complete. Model valid: {valid}")
            
            return self.diagnostic_results
            
        except Exception as e:
            logger.error(f"Error in diagnostics: {str(e)}")
            raise
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete TVECM analysis.
        
        Returns
        -------
        dict
            Complete analysis results
        """
        logger.info("Running full TVECM analysis")
        
        try:
            # Step 1: Estimate linear VECM
            self.estimate_linear_vecm()
            
            # Step 2: Grid search for threshold
            self.grid_search_threshold()
            
            # Step 3: Estimate TVECM
            self.estimate_tvecm()
            
            # Step 4: Run diagnostics
            self.run_diagnostics()
            
            # Combine results
            results = {
                'linear_vecm': {
                    'llf': self.linear_results.llf,
                    'aic': self.linear_results.aic,
                    'bic': self.linear_results.bic
                },
                'threshold': self.threshold_results,
                'tvecm': self.tvecm_results,
                'diagnostics': self.diagnostic_results,
                'data': {
                    'n_obs': len(self.data),
                    'variables': self.variable_names,
                    'dates': self.dates
                }
            }
            
            logger.info("Full TVECM analysis complete")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis: {str(e)}")
            raise
```

## Phase 5: Enhanced Spatial Econometrics Development (Week 10-11)

### 5.1 Create Spatial Weight Matrix Module

```python
"""
Enhanced spatial weights module for market integration analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN, Kernel, W, WSP
from libpysal.weights.util import attach_islands
import logging
from typing import Dict, Any, Union, Optional
import warnings

logger = logging.getLogger(__name__)

class SpatialWeightMatrix:
    """
    Creates and manages spatial weight matrices with conflict adjustment.
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        """
        Initialize with a GeoDataFrame.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data with market locations
        """
        self.gdf = gdf
        self.weights = None
        self.weight_type = None
        
        logger.info(f"Initialized SpatialWeightMatrix with {len(gdf)} locations")
    
    def create_knn_weights(self, k: int = 5, conflict_adjusted: bool = False,
                          conflict_col: str = 'conflict_intensity_normalized',
                          conflict_weight: float = 0.5,
                          id_col: Optional[str] = None) -> W:
        """
        Create k-nearest neighbors weight matrix.
        
        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors
        conflict_adjusted : bool, optional
            Whether to adjust weights by conflict intensity
        conflict_col : str, optional
            Column containing conflict intensity values
        conflict_weight : float, optional
            Weight given to conflict adjustment (0-1)
        id_col : str, optional
            Column to use as ID variable (default: index)
            
        Returns
        -------
        libpysal.weights.W
            Spatial weights matrix
        """
        logger.info(f"Creating KNN weights with k={k}, conflict_adjusted={conflict_adjusted}")
        
        try:
            # Check if conflict adjustment is requested but column doesn't exist
            if conflict_adjusted and conflict_col not in self.gdf.columns:
                logger.warning(f"Conflict column '{conflict_col}' not found. Using standard weights.")
                conflict_adjusted = False
            
            # Create KNN weights
            if id_col:
                knn = KNN.from_dataframe(self.gdf, k=k, ids=self.gdf[id_col])
            else:
                knn = KNN.from_dataframe(self.gdf, k=k)
            
            self.weight_type = "knn"
            
            if conflict_adjusted:
                logger.info(f"Adjusting weights using conflict intensity with weight={conflict_weight}")
                
                # Adjust weights based on conflict intensity
                adj_weights = {}
                
                for i, neighbors in knn.neighbors.items():
                    weights = []
                    
                    for j in neighbors:
                        # Base weight (inverse distance)
                        base_weight = knn.weights[i][knn.neighbors[i].index(j)]
                        
                        # Get conflict intensity for both regions
                        conflict_i = self.gdf.iloc[i][conflict_col] if i < len(self.gdf) else 0
                        conflict_j = self.gdf.iloc[j][conflict_col] if j < len(self.gdf) else 0
                        
                        # Average conflict intensity along the path
                        avg_conflict = (conflict_i + conflict_j) / 2
                        
                        # Adjust weight: higher conflict = lower weight
                        adjusted_weight = base_weight * (1 - (conflict_weight * avg_conflict))
                        weights.append(max(adjusted_weight, 0.001))  # Ensure positive weight
                    
                    adj_weights[i] = weights
                
                # Create new weight matrix with adjusted weights
                self.weights = W(knn.neighbors, adj_weights)
                logger.info("Created conflict-adjusted KNN weights")
                
            else:
                self.weights = knn
                logger.info("Created standard KNN weights")
            
            # Handle islands
            if len(self.weights.islands) > 0:
                logger.warning(f"Found {len(self.weights.islands)} islands in the weight matrix")
                self.weights = attach_islands(self.weights)
                
            # Row-standardize the weights
            self.weights.transform = 'R'
            
            return self.weights
            
        except Exception as e:
            logger.error(f"Error creating KNN weights: {str(e)}")
            raise
    
    def create_distance_weights(self, threshold: float = None, 
                               conflict_adjusted: bool = False,
                               conflict_col: str = 'conflict_intensity_normalized',
                               conflict_weight: float = 0.5,
                               alpha: float = -1.0,
                               id_col: Optional[str] = None) -> W:
        """
        Create distance-based weight matrix.
        
        Parameters
        ----------
        threshold : float, optional
            Distance threshold (if None, use max distance)
        conflict_adjusted : bool, optional
            Whether to adjust weights by conflict intensity
        conflict_col : str, optional
            Column containing conflict intensity values
        conflict_weight : float, optional
            Weight given to conflict adjustment (0-1)
        alpha : float, optional
            Distance decay parameter (-1 = inverse distance)
        id_col : str, optional
            Column to use as ID variable (default: index)
            
        Returns
        -------
        libpysal.weights.W
            Spatial weights matrix
        """
        logger.info(f"Creating distance-based weights with threshold={threshold}, "
                  f"conflict_adjusted={conflict_adjusted}, alpha={alpha}")
        
        try:
            # Check if conflict adjustment is requested but column doesn't exist
            if conflict_adjusted and conflict_col not in self.gdf.columns:
                logger.warning(f"Conflict column '{conflict_col}' not found. Using standard weights.")
                conflict_adjusted = False
            
            # Create distance-based weights
            if id_col:
                if threshold:
                    dist = Kernel.from_dataframe(self.gdf, fixed=True, k=None, 
                                              function='triangular', 
                                              bandwidth=threshold,
                                              ids=self.gdf[id_col])
                else:
                    # Use adaptive bandwidth if no threshold
                    dist = Kernel.from_dataframe(self.gdf, fixed=False, k=10, 
                                              function='triangular',
                                              ids=self.gdf[id_col])
            else:
                if threshold:
                    dist = Kernel.from_dataframe(self.gdf, fixed=True, k=None, 
                                              function='triangular', 
                                              bandwidth=threshold)
                else:
                    # Use adaptive bandwidth if no threshold
                    dist = Kernel.from_dataframe(self.gdf, fixed=False, k=10, 
                                              function='triangular')
            
            self.weight_type = "distance"
            
            if conflict_adjusted:
                logger.info(f"Adjusting weights using conflict intensity with weight={conflict_weight}")
                
                # Adjust weights based on conflict intensity
                adj_weights = {}
                
                for i, neighbors in dist.neighbors.items():
                    weights = []
                    
                    for j in neighbors:
                        # Base weight (kernel)
                        base_weight = dist.weights[i][dist.neighbors[i].index(j)]
                        
                        # Get conflict intensity for both regions
                        conflict_i = self.gdf.iloc[i][conflict_col] if i < len(self.gdf) else 0
                        conflict_j = self.gdf.iloc[j][conflict_col] if j < len(self.gdf) else 0
                        
                        # Average conflict intensity along the path
                        avg_conflict = (conflict_i + conflict_j) / 2
                        
                        # Adjust weight: higher conflict = lower weight
                        adjusted_weight = base_weight * (1 - (conflict_weight * avg_conflict))
                        weights.append(max(adjusted_weight, 0.001))  # Ensure positive weight
                    
                    adj_weights[i] = weights
                
                # Create new weight matrix with adjusted weights
                self.weights = W(dist.neighbors, adj_weights)
                logger.info("Created conflict-adjusted distance weights")
                
            else:
                self.weights = dist
                logger.info("Created standard distance weights")
            
            # Handle islands
            if len(self.weights.islands) > 0:
                logger.warning(f"Found {len(self.weights.islands)} islands in the weight matrix")
                self.weights = attach_islands(self.weights)
                
            # Apply distance decay if alpha is provided
            if alpha != -1.0:
                logger.info(f"Applying distance decay with alpha={alpha}")
                self.weights.transform = 'v'
                
                # Apply custom power transformation
                new_weights = {}
                for i, w_i in self.weights.weights.items():
                    new_w_i = []
                    for w in w_i:
                        # Apply power transformation
                        new_w = w ** alpha if w > 0 else 0
                        new_w_i.append(new_w)
                    new_weights[i] = new_w_i
                
                self.weights = W(self.weights.neighbors, new_weights)
            
            # Row-standardize the weights
            self.weights.transform = 'R'
            
            return self.weights
            
        except Exception as e:
            logger.error(f"Error creating distance weights: {str(e)}")
            raise
    
    def create_connectivity_matrix(self, transport_network: gpd.GeoDataFrame = None,
                                  conflict_adjusted: bool = False,
                                  conflict_col: str = 'conflict_intensity_normalized',
                                  conflict_weight: float = 0.5) -> np.ndarray:
        """
        Create a connectivity matrix that incorporates transportation network.
        
        Parameters
        ----------
        transport_network : geopandas.GeoDataFrame, optional
            Road or transportation network
        conflict_adjusted : bool, optional
            Whether to adjust weights by conflict intensity
        conflict_col : str, optional
            Column containing conflict intensity values
        conflict_weight : float, optional
            Weight given to conflict adjustment (0-1)
            
        Returns
        -------
        numpy.ndarray
            Connectivity matrix
        """
        logger.info("Creating connectivity matrix")
        
        try:
            if transport_network is None:
                logger.warning("No transport network provided. Using distance-based connectivity.")
                
                # If no transport network, create a distance-based connectivity matrix
                if self.weights is None:
                    self.create_knn_weights(k=5, conflict_adjusted=conflict_adjusted,
                                         conflict_col=conflict_col, 
                                         conflict_weight=conflict_weight)
                
                # Convert to dense matrix
                conn_matrix = self.weights.full()[0]
                
            else:
                logger.info("Using transport network to create connectivity matrix")
                
                # Create connectivity matrix based on the transportation network
                # This requires a specialized algorithm that considers:
                # 1. Whether markets are connected by roads
                # 2. Quality/type of roads
                # 3. Conflict density along roads
                
                # Placeholder implementation (would need to be expanded)
                # ...
                
                # For now, just return a placeholder
                conn_matrix = np.zeros((len(self.gdf), len(self.gdf)))
                warnings.warn("Transport network connectivity not fully implemented")
            
            return conn_matrix
            
        except Exception as e:
            logger.error(f"Error creating connectivity matrix: {str(e)}")
            raise
    
    def get_sparse_weights(self) -> WSP:
        """
        Convert weights to sparse format.
        
        Returns
        -------
        libpysal.weights.WSP
            Sparse weights matrix
        """
        if self.weights is None:
            raise ValueError("Weights matrix not created yet")
            
        return WSP(self.weights.sparse)
    
    def get_full_matrix(self) -> np.ndarray:
        """
        Get the full weights matrix as a numpy array.
        
        Returns
        -------
        numpy.ndarray
            Full weights matrix
        """
        if self.weights is None:
            raise ValueError("Weights matrix not created yet")
            
        return self.weights.full()[0]
    
    def plot_weight_connections(self, ax=None, figsize=(10, 10),
                              linewidth=0.5, alpha=0.5):
        """
        Plot the weight connections on a map.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple, optional
            Figure size
        linewidth : float, optional
            Line width for connections
        alpha : float, optional
            Transparency for connections
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with weight connections
        """
        import matplotlib.pyplot as plt
        from libpysal.weights.util import plot_spatial_weights
        
        if self.weights is None:
            raise ValueError("Weights matrix not created yet")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Plot the GeoDataFrame
        self.gdf.plot(ax=ax, color='blue', markersize=5)
        
        # Plot the connections
        plot_spatial_weights(self.weights, self.gdf, 
                           coords_transform=None,
                           ax=ax, linewidth=linewidth, alpha=alpha)
        
        ax.set_title(f"Spatial Weight Connections ({self.weight_type.upper()} weights)")
        
        return fig
```

### 5.2 Enhanced Spatial Econometrics Models

```python
"""
Spatial econometric models for market integration analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from typing import Dict, Any, List, Union, Optional, Tuple
from libpysal.weights import W, WSP
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error, GM_Lag, GM_Error

logger = logging.getLogger(__name__)

class SpatialEconometrics:
    """
    Comprehensive spatial econometric analysis for market integration.
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame):
        """
        Initialize with a GeoDataFrame.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data
        """
        self.gdf = gdf
        self.weights = None
        self.models = {}
        
        logger.info(f"Initialized SpatialEconometrics with {len(gdf)} observations")
    
    def set_weights(self, weights: Union[W, WSP]) -> None:
        """
        Set spatial weights matrix.
        
        Parameters
        ----------
        weights : libpysal.weights.W or libpysal.weights.WSP
            Spatial weights matrix
        """
        self.weights = weights
        logger.info(f"Set spatial weights matrix with {len(weights.neighbors)} units")
    
    def moran_i_test(self, variable: str, permutations: int = 999) -> Dict[str, Any]:
        """
        Test for spatial autocorrelation using Moran's I.
        
        Parameters
        ----------
        variable : str
            Column name in GeoDataFrame to test
        permutations : int, optional
            Number of permutations for significance testing
            
        Returns
        -------
        dict
            Moran's I test results
        """
        logger.info(f"Running Moran's I test for {variable}")
        
        if self.weights is None:
            raise ValueError("Weight matrix not set. Call set_weights first.")
        
        try:
            # Calculate Moran's I
            moran = Moran(self.gdf[variable], self.weights, permutations=permutations)
            
            result = {
                'I': moran.I,
                'p_norm': moran.p_norm,
                'p_sim': moran.p_sim,
                'z_norm': moran.z_norm,
                'significant': moran.p_sim < 0.05,
                'permutations': permutations
            }
            
            logger.info(f"Moran's I result: I={result['I']:.4f}, p-value={result['p_sim']:.4f}, "
                      f"significant={result['significant']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Moran's I test: {str(e)}")
            raise
    
    def local_moran_test(self, variable: str, permutations: int = 999) -> pd.DataFrame:
        """
        Calculate Local Indicators of Spatial Association (LISA).
        
        Parameters
        ----------
        variable : str
            Column name in GeoDataFrame to test
        permutations : int, optional
            Number of permutations for significance testing
            
        Returns
        -------
        pandas.DataFrame
            Local Moran's I statistics
        """
        logger.info(f"Running Local Moran's I test for {variable}")
        
        if self.weights is None:
            raise ValueError("Weight matrix not set. Call set_weights first.")
        
        try:
            # Calculate Local Moran's I
            local_moran = Moran_Local(self.gdf[variable], self.weights, permutations=permutations)
            
            # Create DataFrame with results
            lisa_df = pd.DataFrame({
                'Is': local_moran.Is,
                'p_sim': local_moran.p_sim,
                'p_z_sim': local_moran.p_z_sim,
                'significant': local_moran.p_sim < 0.05,
                'cluster_type': pd.Series(local_moran.q)
            })
            
            # Map cluster types to descriptive labels
            cluster_labels = {
                1: 'high-high',
                2: 'low-high',
                3: 'low-low',
                4: 'high-low'
            }
            
            lisa_df['cluster_label'] = lisa_df['cluster_type'].map(cluster_labels)
            
            # Add geometry for mapping
            lisa_gdf = gpd.GeoDataFrame(
                lisa_df,
                geometry=self.gdf.geometry.copy(),
                crs=self.gdf.crs
            )
            
            logger.info(f"Local Moran's I test complete. "
                      f"{sum(lisa_df['significant'])} significant locations.")
            
            return lisa_gdf
            
        except Exception as e:
            logger.error(f"Error in Local Moran's I test: {str(e)}")
            raise
    
    def spatial_lag_model(self, y_col: str, x_cols: List[str],
                         robust: bool = False, name_y: str = None,
                         name_x: List[str] = None, white_test: bool = True) -> Any:
        """
        Estimate a spatial lag model.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name
        x_cols : list
            List of independent variable column names
        robust : bool, optional
            Whether to use robust (GM) estimation
        name_y : str, optional
            Custom name for dependent variable
        name_x : list, optional
            Custom names for independent variables
        white_test : bool, optional
            Whether to perform White's test for heteroskedasticity
            
        Returns
        -------
        spreg.ML_Lag or spreg.GM_Lag
            Spatial lag model results
        """
        logger.info(f"Estimating spatial lag model with y={y_col}, x={x_cols}")
        
        if self.weights is None:
            raise ValueError("Weight matrix not set. Call set_weights first.")
        
        try:
            # Prepare data
            y = self.gdf[y_col].values
            X = self.gdf[x_cols].values
            
            # Add constant
            if not (X.ndim == 2 and X.shape[1] > 0 and np.all(X[:, 0] == 1)):
                X = np.column_stack((np.ones(len(y)), X))
                if name_x:
                    name_x = ['CONSTANT'] + name_x
            
            # Set names
            name_y = name_y or y_col
            name_x = name_x or ['CONSTANT'] + x_cols
            
            # Estimate model
            if robust:
                model = GM_Lag(y, X, self.weights, name_y=name_y, name_x=name_x)
                logger.info("Estimated robust GM_Lag model")
            else:
                model = ML_Lag(y, X, self.weights, name_y=name_y, name_x=name_x)
                logger.info("Estimated maximum likelihood ML_Lag model")
            
            # Store model
            self.models['lag'] = model
            
            # Perform White test for heteroskedasticity if requested
            if white_test:
                from statsmodels.stats.diagnostic import het_white
                
                # Get residuals
                resid = model.u
                
                # Standard OLS for comparison
                ols = OLS(y, X, name_y=name_y, name_x=name_x)
                
                # White test
                white_results = het_white(resid, ols.x)
                
                # Add to model attributes
                model.white_test = {
                    'statistic': white_results[0],
                    'p_value': white_results[1],
                    'heteroskedastic': white_results[1] < 0.05
                }
                
                logger.info(f"White test result: heteroskedastic={model.white_test['heteroskedastic']}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error in spatial lag model: {str(e)}")
            raise
    
    def spatial_error_model(self, y_col: str, x_cols: List[str],
                           robust: bool = False, name_y: str = None,
                           name_x: List[str] = None, white_test: bool = True) -> Any:
        """
        Estimate a spatial error model.
        
        Parameters
        ----------
        y_col : str
            Dependent variable column name
        x_cols : list
            List of independent variable column names
        robust : bool, optional
            Whether to use robust (GM) estimation
        name_y : str, optional
            Custom name for dependent variable
        name_x : list, optional
            Custom names for independent variables
        white_test : bool, optional
            Whether to perform White's test for heteroskedasticity
            
        Returns
        -------
        spreg.ML_Error or spreg.GM_Error
            Spatial error model results
        """
        logger.info(f"Estimating spatial error model with y={y_col}, x={x_cols}")
        
        if self.weights is None:
            raise ValueError("Weight matrix not set. Call set_weights first.")
        
        try:
            # Prepare data
            y = self.gdf[y_col].values
            X = self.gdf[x_cols].values
            
            # Add constant
            if not (X.ndim == 2 and X.shape[1] > 0 and np.all(X[:, 0] == 1)):
                X = np.column_stack((np.ones(len(y)), X))
                if name_x:
                    name_x = ['CONSTANT'] + name_x
            
            # Set names
            name_y = name_y or y_col
            name_x = name_x or ['CONSTANT'] + x_cols
            
            # Estimate model
            if robust:
                model = GM_Error(y, X, self.weights, name_y=name_y, name_x=name_x)
                logger.info("Estimated robust GM_Error model")
            else:
                model = ML_Error(y, X, self.weights, name_y=name_y, name_x=name_x)
                logger.info("Estimated maximum likelihood ML_Error model")
            
            # Store model
            self.models['error'] = model
            
            # Perform White test for heteroskedasticity if requested
            if white_test:
                from statsmodels.stats.diagnostic import het_white
                
                # Get residuals
                resid = model.u
                
                # Standard OLS for comparison
                ols = OLS(y, X, name_y=name_y, name_x=name_x)
                
                # White test
                white_results = het_white(resid, ols.x)
                
                # Add to model attributes
                model.white_test = {
                    'statistic': white_results[0],
                    'p_value': white_results[1],
                    'heteroskedastic': white_results[1] < 0.05
                }
                
                logger.info(f"White test result: heteroskedastic={model.white_test['heteroskedastic']}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error in spatial error model: {str(e)}")
            raise
    
    def calculate_market_isolation(self, conflict_col: str = 'conflict_intensity_normalized',
                                  max_distance: float = 50000) -> gpd.GeoDataFrame:
        """
        Calculate market isolation index.
        
        Parameters
        ----------
        conflict_col : str, optional
            Column containing conflict intensity values
        max_distance : float, optional
            Maximum distance in projection units
            
        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with isolation index
        """
        logger.info(f"Calculating market isolation index with max_distance={max_distance}")
        
        try:
            if conflict_col not in self.gdf.columns:
                logger.warning(f"Conflict column '{conflict_col}' not found. Using zero values.")
                self.gdf[conflict_col] = 0
            
            # Calculate distance matrix
            from scipy.spatial.distance import pdist, squareform
            
            # Extract coordinates
            coords = np.array([(p.x, p.y) for p in self.gdf.geometry])
            
            # Calculate distance matrix
            dist_matrix = squareform(pdist(coords))
            
            # For each market, calculate isolation index
            isolation_indices = []
            
            for i in range(len(self.gdf)):
                # Get distances to other markets
                distances = dist_matrix[i]
                
                # Filter by max distance
                mask = distances <= max_distance
                
                if sum(mask) <= 1:
                    # If no other markets within range, high isolation
                    isolation_index = 1.0
                else:
                    # Calculate weighted average of conflict intensity
                    # and normalized distance for nearby markets
                    nearby_dists = distances[mask]
                    nearby_dists = nearby_dists[nearby_dists > 0]  # Exclude self
                    
                    # Normalize distances
                    norm_dists = nearby_dists / max_distance
                    
                    # Get conflict intensities for nearby markets
                    nearby_conflicts = np.array([
                        self.gdf.iloc[j][conflict_col] 
                        for j in range(len(self.gdf)) 
                        if mask[j] and i != j
                    ])
                    
                    # Calculate isolation index as weighted sum
                    # Distance weight: 0.6, Conflict weight: 0.4
                    dist_factor = np.mean(norm_dists)
                    conflict_factor = np.mean(nearby_conflicts)
                    
                    isolation_index = 0.6 * dist_factor + 0.4 * conflict_factor
                
                isolation_indices.append(isolation_index)
            
            # Add to GeoDataFrame
            result_gdf = self.gdf.copy()
            result_gdf['isolation_index'] = isolation_indices
            
            # Add classification
            result_gdf['isolation_category'] = pd.qcut(
                result_gdf['isolation_index'], 
                q=5, 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            
            logger.info("Market isolation index calculation complete")
            
            return result_gdf
            
        except Exception as e:
            logger.error(f"Error calculating market isolation: {str(e)}")
            raise
    
    def compute_accessibility_index(self, population_gdf: gpd.GeoDataFrame,
                                   max_distance: float = 50000,
                                   distance_decay: float = 2.0,
                                   weight_col: str = 'population') -> gpd.GeoDataFrame:
        """
        Compute market accessibility index based on nearby population.
        
        Parameters
        ----------
        population_gdf : geopandas.GeoDataFrame
            GeoDataFrame with population data
        max_distance : float, optional
            Maximum distance in projection units
        distance_decay : float, optional
            Distance decay parameter
        weight_col : str, optional
            Column containing population or weight values
            
        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with accessibility index
        """
        logger.info(f"Computing market accessibility index with max_distance={max_distance}")
        
        try:
            # Calculate distances from markets to population centers
            n_markets = len(self.gdf)
            n_pop = len(population_gdf)
            
            accessibility_indices = []
            
            # For each market
            for i in range(n_markets):
                market_geom = self.gdf.iloc[i].geometry
                
                # Calculate distances to all population centers
                distances = np.array([
                    market_geom.distance(pop_geom) 
                    for pop_geom in population_gdf.geometry
                ])
                
                # Filter by max distance
                mask = distances <= max_distance
                
                if sum(mask) == 0:
                    # If no population centers within range, zero accessibility
                    accessibility_index = 0.0
                else:
                    # Calculate weighted sum of population / distance^decay
                    nearby_dists = distances[mask]
                    nearby_pops = population_gdf.iloc[mask][weight_col].values
                    
                    # Avoid division by zero
                    nearby_dists = np.maximum(nearby_dists, 1.0)
                    
                    # Calculate accessibility as sum(pop / distance^decay)
                    accessibility = np.sum(nearby_pops / (nearby_dists ** distance_decay))
                    
                    # Normalize by theoretical maximum (population center at zero distance)
                    max_pop = np.max(population_gdf[weight_col])
                    theoretical_max = max_pop  # At distance=1
                    
                    accessibility_index = accessibility / theoretical_max
                
                accessibility_indices.append(accessibility_index)
            
            # Add to GeoDataFrame
            result_gdf = self.gdf.copy()
            result_gdf['accessibility_index'] = accessibility_indices
            
            # Scale to 0-1 range
            max_access = max(accessibility_indices)
            result_gdf['accessibility_index'] = result_gdf['accessibility_index'] / max_access
            
            # Add classification
            result_gdf['accessibility_category'] = pd.qcut(
                result_gdf['accessibility_index'], 
                q=5, 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            
            logger.info("Market accessibility index calculation complete")
            
            return result_gdf
            
        except Exception as e:
            logger.error(f"Error computing accessibility index: {str(e)}")
            raise
```

## Phase 6: Policy Simulation Module (Week 12)

```python
"""
Enhanced policy simulation module for market integration analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from typing import Dict, Any, List, Union, Optional, Tuple
import copy

logger = logging.getLogger(__name__)

class MarketIntegrationSimulation:
    """
    Simulate policy interventions for market integration analysis.
    """
    
    def __init__(self, data: Union[pd.DataFrame, gpd.GeoDataFrame], 
                threshold_model=None, spatial_model=None):
        """
        Initialize the simulation model.
        
        Parameters
        ----------
        data : pandas.DataFrame or geopandas.GeoDataFrame
            Market data
        threshold_model : ThresholdVECM or ThresholdCointegration, optional
            Estimated threshold model
        spatial_model : SpatialEconometrics, optional
            Estimated spatial model
        """
        self.data = data.copy()
        self.threshold_model = threshold_model
        self.spatial_model = spatial_model
        self.results = {}
        
        logger.info(f"Initialized MarketIntegrationSimulation with {len(data)} observations")
    
    def simulate_exchange_rate_unification(self, target_rate: str = 'average',
                                         reference_date: str = None) -> Dict[str, Any]:
        """
        Simulate exchange rate unification by setting differential to zero.
        
        Parameters
        ----------
        target_rate : str or float, optional
            How to set unified rate: 'average', 'north', 'south', or specific value
        reference_date : str, optional
            Reference date for rate calculation (default: latest date)
            
        Returns
        -------
        dict
            Simulation results
        """
        logger.info(f"Simulating exchange rate unification with target_rate={target_rate}")
        
        try:
            # Check if data contains exchange rate information
            if 'exchange_rate_regime' not in self.data.columns:
                logger.error("Data does not contain exchange rate regime information")
                raise ValueError("Data does not contain exchange rate regime information")
            
            # Create a copy of data
            unified_data = self.data.copy()
            
            # Get reference date
            if reference_date:
                if 'date' not in self.data.columns:
                    logger.warning("No date column found. Using all data for rate calculation.")
                    date_mask = slice(None)
                else:
                    date_mask = self.data['date'] == pd.to_datetime(reference_date)
                    if date_mask.sum() == 0:
                        logger.warning(f"Reference date {reference_date} not found. Using latest date.")
                        latest_date = self.data['date'].max()
                        date_mask = self.data['date'] == latest_date
            else:
                if 'date' in self.data.columns:
                    latest_date = self.data['date'].max()
                    date_mask = self.data['date'] == latest_date
                else:
                    date_mask = slice(None)
            
            # Calculate target exchange rate
            if target_rate == 'average':
                # Use average of north and south rates
                north_rate = self.data.loc[
                    date_mask & (self.data['exchange_rate_regime'] == 'north'), 'usdprice'
                ].mean()
                
                south_rate = self.data.loc[
                    date_mask & (self.data['exchange_rate_regime'] == 'south'), 'usdprice'
                ].mean()
                
                unified_rate = (north_rate + south_rate) / 2
                
            elif target_rate == 'north':
                # Use north rate
                unified_rate = self.data.loc[
                    date_mask & (self.data['exchange_rate_regime'] == 'north'), 'usdprice'
                ].mean()
                
            elif target_rate == 'south':
                # Use south rate
                unified_rate = self.data.loc[
                    date_mask & (self.data['exchange_rate_regime'] == 'south'), 'usdprice'
                ].mean()
                
            elif isinstance(target_rate, (int, float)):
                # Use specified rate
                unified_rate = float(target_rate)
                
            else:
                logger.error(f"Invalid target_rate: {target_rate}")
                raise ValueError(f"Invalid target_rate: {target_rate}")
            
            logger.info(f"Calculated unified exchange rate: {unified_rate:.4f}")
            
            # Apply unified rate
            if 'date' in self.data.columns:
                # For each date, apply unified rate
                for date in unified_data['date'].unique():
                    mask = unified_data['date'] == date
                    unified_data.loc[mask, 'usdprice'] = unified_rate
            else:
                # Apply to all data
                unified_data['usdprice'] = unified_rate
            
            # Calculate price changes
            # Original prices in local currency divided by original USD price
            # multiplied by new USD price
            if 'price' in unified_data.columns:
                # Calculate original local currency amounts
                if 'original_price' not in unified_data.columns:
                    unified_data['original_price'] = unified_data['price'].copy()
                
                # Calculate local currency conversion rate
                unified_data['local_currency_per_usd'] = unified_data['original_price'] / unified_data['usdprice']
                
                # Apply unified rate to recalculate prices
                unified_data['price'] = unified_data['local_currency_per_usd'] * unified_rate
            
            # Estimate new threshold model if available
            new_threshold_model = None
            if self.threshold_model is not None:
                logger.info("Re-estimating threshold model with unified rates")
                
                # This would depend on the specific implementation of your threshold model
                # Example for ThresholdCointegration
                if hasattr(self.threshold_model, 'run_full_analysis'):
                    # Extract north and south price series
                    north_prices = unified_data[
                        unified_data['exchange_rate_regime'] == 'north'
                    ].groupby('date')['price'].mean()
                    
                    south_prices = unified_data[
                        unified_data['exchange_rate_regime'] == 'south'
                    ].groupby('date')['price'].mean()
                    
                    # Ensure the series have the same dates
                    common_dates = north_prices.index.intersection(south_prices.index)
                    north_prices = north_prices.loc[common_dates]
                    south_prices = south_prices.loc[common_dates]
                    
                    # Create and estimate new model
                    from src.models.threshold import ThresholdCointegration
                    new_threshold_model = ThresholdCointegration(
                        north_prices.values, south_prices.values
                    )
                    new_model_results = new_threshold_model.run_full_analysis()
                else:
                    logger.warning("Unsupported threshold model type for re-estimation")
                    new_model_results = "Unsupported threshold model type"
            
            # Store results
            self.results['exchange_rate_unification'] = {
                'data': unified_data,
                'unified_rate': unified_rate,
                'original_data': self.data,
                'threshold_model': new_threshold_model,
                'model_results': new_model_results if 'new_model_results' in locals() else None,
                'policy_scenario': 'exchange_rate_unification',
                'target_rate': target_rate,
                'reference_date': reference_date
            }
            
            logger.info("Exchange rate unification simulation complete")
            
            return self.results['exchange_rate_unification']
            
        except Exception as e:
            logger.error(f"Error in exchange rate unification simulation: {str(e)}")
            raise
    
    def simulate_improved_connectivity(self, reduction_factor: float = 0.5,
                                     infrastructure_gdf: gpd.GeoDataFrame = None,
                                     scenario_name: str = "improved_roads") -> Dict[str, Any]:
        """
        Simulate improved connectivity by reducing conflict-related barriers.
        
        Parameters
        ----------
        reduction_factor : float, optional
            Factor to reduce conflict intensity by (0-1)
        infrastructure_gdf : geopandas.GeoDataFrame, optional
            New infrastructure data (e.g., roads, bridges)
        scenario_name : str, optional
            Name for the simulation scenario
            
        Returns
        -------
        dict
            Simulation results
        """
        logger.info(f"Simulating improved connectivity with reduction_factor={reduction_factor}")
        
        try:
            # Check if spatial model is available
            if self.spatial_model is None:
                logger.error("Spatial model required for connectivity simulation")
                raise ValueError("Spatial model required for connectivity simulation")
            
            # Create a copy of data
            improved_data = self.data.copy()
            
            # Check if data contains conflict information
            conflict_cols = [col for col in improved_data.columns 
                           if 'conflict' in col.lower()]
            
            if not conflict_cols:
                logger.error("Data does not contain conflict intensity information")
                raise ValueError("Data does not contain conflict intensity information")
            
            # Reduce conflict intensity
            for col in conflict_cols:
                if 'lag' not in col.lower():  # Don't modify lagged values
                    improved_data[col] = improved_data[col] * reduction_factor
                    logger.info(f"Reduced {col} by factor {reduction_factor}")
            
            # Re-create spatial weights with reduced conflict
            if hasattr(self.spatial_model, 'weights') and self.spatial_model.weights is not None:
                logger.info("Re-creating weight matrix with reduced conflict")
                
                from src.models.spatial import SpatialWeightMatrix
                
                # Create new weight matrix handler
                weight_handler = SpatialWeightMatrix(improved_data)
                
                # Create new weights
                if hasattr(self.spatial_model, 'weight_type') and self.spatial_model.weight_type == 'knn':
                    # K-nearest neighbors
                    k = getattr(self.spatial_model, 'k', 5)
                    new_weights = weight_handler.create_knn_weights(
                        k=k,
                        conflict_adjusted=True,
                        conflict_col='conflict_intensity_normalized',
                        conflict_weight=0.5
                    )
                else:
                    # Default to distance weights
                    new_weights = weight_handler.create_distance_weights(
                        conflict_adjusted=True,
                        conflict_col='conflict_intensity_normalized',
                        conflict_weight=0.5
                    )
                
                # Create new spatial model
                from src.models.spatial import SpatialEconometrics
                new_spatial_model = SpatialEconometrics(improved_data)
                new_spatial_model.set_weights(new_weights)
                
                # Re-estimate spatial models
                if hasattr(self.spatial_model, 'models') and 'lag' in self.spatial_model.models:
                    logger.info("Re-estimating spatial lag model")
                    
                    # Get original parameters
                    orig_model = self.spatial_model.models['lag']
                    y_col = orig_model.name_y
                    x_cols = orig_model.name_x[1:]  # Skip constant
                    
                    # Re-estimate model
                    new_lag_model = new_spatial_model.spatial_lag_model(
                        y_col=y_col,
                        x_cols=x_cols
                    )
                    
                    # Store model
                    new_spatial_model.models['lag'] = new_lag_model
                
                if hasattr(self.spatial_model, 'models') and 'error' in self.spatial_model.models:
                    logger.info("Re-estimating spatial error model")
                    
                    # Get original parameters
                    orig_model = self.spatial_model.models['error']
                    y_col = orig_model.name_y
                    x_cols = orig_model.name_x[1:]  # Skip constant
                    
                    # Re-estimate model
                    new_error_model = new_spatial_model.spatial_error_model(
                        y_col=y_col,
                        x_cols=x_cols
                    )
                    
                    # Store model
                    new_spatial_model.models['error'] = new_error_model
                
            else:
                logger.warning("No existing weights matrix found. Creating new spatial model.")
                
                # Create new spatial model
                from src.models.spatial import SpatialEconometrics
                new_spatial_model = SpatialEconometrics(improved_data)
                
                # Default weights
                from src.models.spatial import SpatialWeightMatrix
                weight_handler = SpatialWeightMatrix(improved_data)
                new_weights = weight_handler.create_knn_weights(
                    k=5,
                    conflict_adjusted=True,
                    conflict_col='conflict_intensity_normalized',
                    conflict_weight=0.5
                )
                
                new_spatial_model.set_weights(new_weights)
            
            # Calculate market integration metrics
            if hasattr(new_spatial_model, 'calculate_market_isolation'):
                logger.info("Calculating new market isolation indices")
                isolation_gdf = new_spatial_model.calculate_market_isolation(
                    conflict_col='conflict_intensity_normalized'
                )
                
                # Store isolation indices
                improved_data['isolation_index'] = isolation_gdf['isolation_index']
                improved_data['isolation_category'] = isolation_gdf['isolation_category']
            
            # Store results
            self.results['improved_connectivity'] = {
                'data': improved_data,
                'original_data': self.data,
                'spatial_model': new_spatial_model,
                'policy_scenario': scenario_name,
                'reduction_factor': reduction_factor
            }
            
            # Add model comparisons if available
            if 'new_lag_model' in locals() and hasattr(self.spatial_model, 'models') and 'lag' in self.spatial_model.models:
                orig_model = self.spatial_model.models['lag']
                new_model = new_lag_model
                
                self.results['improved_connectivity']['model_comparison'] = {
                    'original_rho': orig_model.rho,
                    'new_rho': new_model.rho,
                    'original_betas': orig_model.betas,
                    'new_betas': new_model.betas,
                    'original_std_err': orig_model.std_err,
                    'new_std_err': new_model.std_err,
                    'original_r2': orig_model.pr2,
                    'new_r2': new_model.pr2
                }
            
            logger.info("Improved connectivity simulation complete")
            
            return self.results['improved_connectivity']
            
        except Exception as e:
            logger.error(f"Error in improved connectivity simulation: {str(e)}")
            raise
    
    def simulate_combined_policy(self, exchange_rate_target: str = 'average',
                               conflict_reduction: float = 0.5) -> Dict[str, Any]:
        """
        Simulate combined exchange rate unification and improved connectivity.
        
        Parameters
        ----------
        exchange_rate_target : str or float, optional
            Target rate for unification
        conflict_reduction : float, optional
            Factor to reduce conflict intensity
            
        Returns
        -------
        dict
            Simulation results
        """
        logger.info(f"Simulating combined policy with exchange_rate_target={exchange_rate_target}, "
                  f"conflict_reduction={conflict_reduction}")
        
        try:
            # Run exchange rate unification first
            er_results = self.simulate_exchange_rate_unification(target_rate=exchange_rate_target)
            unified_data = er_results['data']
            
            # Create a temporary simulation object with unified data
            temp_sim = MarketIntegrationSimulation(
                unified_data,
                threshold_model=er_results.get('threshold_model'),
                spatial_model=self.spatial_model
            )
            
            # Run connectivity improvement on the unified data
            conn_results = temp_sim.simulate_improved_connectivity(
                reduction_factor=conflict_reduction,
                scenario_name="combined_policy"
            )
            
            # Combined data is the result of both simulations
            combined_data = conn_results['data']
            
            # Store results
            self.results['combined_policy'] = {
                'data': combined_data,
                'original_data': self.data,
                'exchange_rate_results': er_results,
                'connectivity_results': conn_results,
                'policy_scenario': 'combined_policy',
                'exchange_rate_target': exchange_rate_target,
                'conflict_reduction': conflict_reduction
            }
            
            logger.info("Combined policy simulation complete")
            
            return self.results['combined_policy']
            
        except Exception as e:
            logger.error(f"Error in combined policy simulation: {str(e)}")
            raise
    
    def calculate_welfare_effects(self, policy_scenario: str = None) -> Dict[str, Any]:
        """
        Calculate welfare effects of a policy simulation.
        
        Parameters
        ----------
        policy_scenario : str, optional
            Name of policy scenario to analyze
            
        Returns
        -------
        dict
            Welfare analysis results
        """
        logger.info(f"Calculating welfare effects for policy scenario: {policy_scenario}")
        
        try:
            # Get policy results
            if policy_scenario:
                if policy_scenario not in self.results:
                    logger.error(f"Policy scenario '{policy_scenario}' not found")
                    raise ValueError(f"Policy scenario '{policy_scenario}' not found")
                
                policy_results = self.results[policy_scenario]
            else:
                # Use the first available scenario
                if not self.results:
                    logger.error("No policy simulations available")
                    raise ValueError("No policy simulations available")
                
                policy_scenario = next(iter(self.results.keys()))
                policy_results = self.results[policy_scenario]
            
            # Get original and simulated data
            original_data = policy_results['original_data']
            simulated_data = policy_results['data']
            
            # Calculate price changes
            welfare_results = {
                'policy_scenario': policy_scenario,
                'regional_effects': {},
                'commodity_effects': {},
                'aggregate_effects': {}
            }
            
            # Check if price data is available
            if 'price' in original_data.columns and 'price' in simulated_data.columns:
                # Calculate price changes
                # Merge the datasets by common columns
                common_cols = [col for col in original_data.columns 
                             if col in simulated_data.columns and col != 'price']
                
                merged_data = pd.merge(
                    original_data[common_cols + ['price']].rename(columns={'price': 'original_price'}),
                    simulated_data[common_cols + ['price']].rename(columns={'price': 'simulated_price'}),
                    on=common_cols
                )
                
                # Calculate price changes
                merged_data['price_change'] = (
                    merged_data['simulated_price'] - merged_data['original_price']
                )
                merged_data['price_change_pct'] = (
                    (merged_data['simulated_price'] - merged_data['original_price']) / 
                    merged_data['original_price'] * 100
                )
                
                # Calculate aggregate effects
                welfare_results['aggregate_effects'] = {
                    'mean_price_change': merged_data['price_change'].mean(),
                    'mean_price_change_pct': merged_data['price_change_pct'].mean(),
                    'median_price_change': merged_data['price_change'].median(),
                    'median_price_change_pct': merged_data['price_change_pct'].median(),
                    'max_price_change': merged_data['price_change'].max(),
                    'min_price_change': merged_data['price_change'].min(),
                    'price_change_std': merged_data['price_change'].std()
                }
                
                # Calculate regional effects if admin1 column is available
                if 'admin1' in merged_data.columns:
                    regional_effects = merged_data.groupby('admin1').agg({
                        'price_change': ['mean', 'median', 'std'],
                        'price_change_pct': ['mean', 'median', 'std']
                    }).reset_index()
                    
                    regional_effects.columns = [
                        '_'.join(col).strip() if col[1] else col[0] 
                        for col in regional_effects.columns.values
                    ]
                    
                    welfare_results['regional_effects'] = regional_effects.to_dict(orient='records')
                
                # Calculate commodity effects if commodity column is available
                if 'commodity' in merged_data.columns:
                    commodity_effects = merged_data.groupby('commodity').agg({
                        'price_change': ['mean', 'median', 'std'],
                        'price_change_pct': ['mean', 'median', 'std']
                    }).reset_index()
                    
                    commodity_effects.columns = [
                        '_'.join(col).strip() if col[1] else col[0] 
                        for col in commodity_effects.columns.values
                    ]
                    
                    welfare_results['commodity_effects'] = commodity_effects.to_dict(orient='records')
            else:
                logger.warning("Price data not available for welfare analysis")
                welfare_results['error'] = "Price data not available"
            
            # Store welfare results
            self.results[f"{policy_scenario}_welfare"] = welfare_results
            
            logger.info(f"Welfare analysis complete for {policy_scenario}")
            
            return welfare_results
            
        except Exception as e:
            logger.error(f"Error in welfare analysis: {str(e)}")
            raise
```

## Phase 7: Visualization Tools Enhancement (Week 13-14)

### 7.1 Enhanced Time Series Visualization

```python
"""
Enhanced time series visualization utilities.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any, List, Union, Optional, Tuple

logger = logging.getLogger(__name__)

class TimeSeriesVisualizer:
    """Enhanced time series visualizations for market data."""
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        style : str, optional
            Matplotlib style to use
        """
        plt.style.use(style)
        self.style = style
        logger.info(f"Initialized TimeSeriesVisualizer with style='{style}'")
    
    def plot_price_series(self, df: pd.DataFrame, price_col: str = 'price',
                         date_col: str = 'date', group_col: Optional[str] = None,
                         title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6),
                         color_palette: str = 'viridis', 
                         marker: Optional[str] = 'o', linestyle: str = '-',
                         alpha: float = 0.8, legend_loc: str = 'best') -> Tuple[plt.Figure, plt.Axes]:
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
        color_palette : str, optional
            Matplotlib/Seaborn color palette
        marker : str, optional
            Marker style (set to None for no markers)
        linestyle : str, optional
            Line style
        alpha : float, optional
            Transparency
        legend_loc : str, optional
            Legend location
            
        Returns
        -------
        tuple
            (Figure, Axes)
        """
        logger.info(f"Plotting price series with group_col={group_col}")
        
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Ensure date column is datetime
            if df[date_col].dtype != 'datetime64[ns]':
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
            
            if group_col is not None:
                # Get unique groups and color palette
                groups = df[group_col].unique()
                colors = sns.color_palette(color_palette, n_colors=len(groups))
                
                # Plot each group
                for i, (name, group) in enumerate(df.groupby(group_col)):
                    ax.plot(group[date_col], group[price_col], label=name,
                          marker=marker, linestyle=linestyle, alpha=alpha,
                          color=colors[i])
                
                # Add legend
                ax.legend(title=group_col.capitalize(), loc=legend_loc)
                
            else:
                ax.plot(df[date_col], df[price_col], 
                      marker=marker, linestyle=linestyle, alpha=alpha)
            
            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(title or 'Price Time Series')
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            return fig, ax
            
        except Exception as e:
            logger.error(f"Error plotting price series: {str(e)}")
            raise
    
    def plot_price_differentials(self, df: pd.DataFrame, date_col: str = 'date',
                               north_col: str = 'north_price', south_col: str = 'south_price',
                               diff_col: str = 'price_diff', group_col: Optional[str] = None,
                               title: Optional[str] = None, figsize: Tuple[int, int] = (12, 8),
                               color_palette: str = 'tab10') -> Tuple[plt.Figure, List[plt.Axes]]:
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
        color_palette : str, optional
            Matplotlib/Seaborn color palette
            
        Returns
        -------
        tuple
            (Figure, List of Axes)
        """
        logger.info(f"Plotting price differentials with group_col={group_col}")
        
        try:
            # Ensure date column is datetime
            if df[date_col].dtype != 'datetime64[ns]':
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
            
            if group_col is not None:
                # Create a subplot for each group
                unique_groups = sorted(df[group_col].unique())
                n_groups = len(unique_groups)
                
                # Create figure with subplots
                fig, axes = plt.subplots(n_groups, 1, figsize=figsize, sharex=True)
                if n_groups == 1:
                    axes = [axes]  # Ensure axes is a list
                
                for i, group_val in enumerate(unique_groups):
                    group_df = df[df[group_col] == group_val]
                    ax = axes[i]
                    
                    # Plot prices
                    ax.plot(group_df[date_col], group_df[north_col], label='North', 
                          color='tab:blue', alpha=0.8)
                    ax.plot(group_df[date_col], group_df[south_col], label='South', 
                          color='tab:orange', alpha=0.8)
                    
                    # Plot differential on secondary axis
                    ax2 = ax.twinx()
                    ax2.plot(group_df[date_col], group_df[diff_col], 'r--', label='Differential')
                    
                    # Add zero line
                    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    
                    # Set labels
                    ax.set_ylabel('Price')
                    ax2.set_ylabel('Differential')
                    ax.set_title(f'{group_val}')
                    
                    # Add legend on first subplot
                    if i == 0:
                        lines1, labels1 = ax.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                # Set common x-axis label
                axes[-1].set_xlabel('Date')
                
                # Set overall title if provided
                if title:
                    fig.suptitle(title, fontsize=14)
                    fig.subplots_adjust(top=0.9)
                
            else:
                # Create a single plot
                fig, ax = plt.subplots(figsize=figsize)
                
                # Plot prices
                ax.plot(df[date_col], df[north_col], label='North', color='tab:blue', alpha=0.8)
                ax.plot(df[date_col], df[south_col], label='South', color='tab:orange', alpha=0.8)
                
                # Plot differential on secondary axis
                ax2 = ax.twinx()
                ax2.plot(df[date_col], df[diff_col], 'r--', label='Differential')
                
                # Add zero line
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                # Set labels
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax2.set_ylabel('Differential')
                ax.set_title(title or 'Price Differentials: North vs South')
                
                # Add legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                axes = [ax]
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            return

Let me continue with the remaining methods for the TimeSeriesVisualizer class:

```python
            return fig, axes
            
        except Exception as e:
            logger.error(f"Error plotting price differentials: {str(e)}")
            raise
    
    def plot_interactive_time_series(self, df: pd.DataFrame, price_col: str = 'price', 
                                    date_col: str = 'date', group_col: Optional[str] = None,
                                    title: Optional[str] = None, height: int = 600, 
                                    width: int = 900) -> go.Figure:
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
        height : int, optional
            Plot height
        width : int, optional
            Plot width
            
        Returns
        -------
        plotly.graph_objects.Figure
            The interactive figure
        """
        logger.info(f"Creating interactive time series plot with group_col={group_col}")
        
        try:
            # Ensure date column is datetime
            if df[date_col].dtype != 'datetime64[ns]':
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
            
            # Create plot
            if group_col is not None:
                fig = px.line(
                    df, x=date_col, y=price_col, color=group_col,
                    title=title or f'Price Time Series by {group_col}',
                    height=height, width=width,
                    labels={price_col: 'Price', date_col: 'Date', group_col: group_col.capitalize()}
                )
            else:
                fig = px.line(
                    df, x=date_col, y=price_col,
                    title=title or 'Price Time Series',
                    height=height, width=width,
                    labels={price_col: 'Price', date_col: 'Date'}
                )
            
            # Customize layout
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='closest',
                legend_title=group_col.capitalize() if group_col else '',
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive time series plot: {str(e)}")
            raise
    
    def plot_interactive_differentials(self, df: pd.DataFrame, date_col: str = 'date',
                                     north_col: str = 'north_price', south_col: str = 'south_price',
                                     diff_col: str = 'price_diff', group_col: Optional[str] = None,
                                     title: Optional[str] = None, height: int = 600, 
                                     width: int = 900) -> go.Figure:
        """
        Create an interactive plot of price differentials using Plotly.
        
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
        height : int, optional
            Plot height
        width : int, optional
            Plot width
            
        Returns
        -------
        plotly.graph_objects.Figure
            The interactive figure
        """
        logger.info(f"Creating interactive differential plot with group_col={group_col}")
        
        try:
            # Ensure date column is datetime
            if df[date_col].dtype != 'datetime64[ns]':
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
            
            if group_col is not None:
                # Create subplots for each group
                unique_groups = sorted(df[group_col].unique())
                n_groups = len(unique_groups)
                
                fig = make_subplots(
                    rows=n_groups, cols=1,
                    shared_xaxes=True,
                    subplot_titles=[str(g) for g in unique_groups],
                    specs=[[{"secondary_y": True}] for _ in range(n_groups)],
                    vertical_spacing=0.1 / n_groups
                )
                
                for i, group_val in enumerate(unique_groups):
                    group_df = df[df[group_col] == group_val]
                    row = i + 1
                    
                    # Add north price
                    fig.add_trace(
                        go.Scatter(
                            x=group_df[date_col], y=group_df[north_col],
                            name=f'North {group_val}' if i == 0 else f'North',
                            mode='lines', line=dict(color='blue'),
                            showlegend=(i == 0)
                        ),
                        row=row, col=1, secondary_y=False
                    )
                    
                    # Add south price
                    fig.add_trace(
                        go.Scatter(
                            x=group_df[date_col], y=group_df[south_col],
                            name=f'South {group_val}' if i == 0 else f'South',
                            mode='lines', line=dict(color='orange'),
                            showlegend=(i == 0)
                        ),
                        row=row, col=1, secondary_y=False
                    )
                    
                    # Add differential
                    fig.add_trace(
                        go.Scatter(
                            x=group_df[date_col], y=group_df[diff_col],
                            name=f'Differential {group_val}' if i == 0 else f'Differential',
                            mode='lines', line=dict(color='red', dash='dash'),
                            showlegend=(i == 0)
                        ),
                        row=row, col=1, secondary_y=True
                    )
                    
                    # Add zero line for differential
                    fig.add_hline(
                        y=0, line=dict(color='gray', width=1, dash='dot'),
                        row=row, col=1, secondary_y=True
                    )
                    
                    # Set y-axis titles
                    fig.update_yaxes(title_text="Price", row=row, col=1, secondary_y=False)
                    fig.update_yaxes(title_text="Differential", row=row, col=1, secondary_y=True)
                
                # Set common x-axis title
                fig.update_xaxes(title_text="Date", row=n_groups, col=1)
                
            else:
                # Create a single plot with dual y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add north price
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col], y=df[north_col],
                        name='North',
                        mode='lines', line=dict(color='blue')
                    ),
                    secondary_y=False
                )
                
                # Add south price
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col], y=df[south_col],
                        name='South',
                        mode='lines', line=dict(color='orange')
                    ),
                    secondary_y=False
                )
                
                # Add differential
                fig.add_trace(
                    go.Scatter(
                        x=df[date_col], y=df[diff_col],
                        name='Differential',
                        mode='lines', line=dict(color='red', dash='dash')
                    ),
                    secondary_y=True
                )
                
                # Add zero line for differential
                fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), secondary_y=True)
                
                # Set axes titles
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Price", secondary_y=False)
                fig.update_yaxes(title_text="Differential", secondary_y=True)
            
            # Update layout
            fig.update_layout(
                title=title or "Price Differentials: North vs South",
                height=height,
                width=width,
                template='plotly_white',
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive differential plot: {str(e)}")
            raise
    
    def plot_threshold_analysis(self, threshold_model: Any, title: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Visualize threshold model results.
        
        Parameters
        ----------
        threshold_model : ThresholdCointegration or ThresholdVECM
            Estimated threshold model
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        tuple
            (Figure, List of Axes)
        """
        logger.info("Creating threshold analysis visualization")
        
        try:
            # Check if threshold model is available
            if not hasattr(threshold_model, 'threshold_result') or threshold_model.threshold_result is None:
                logger.error("Threshold model results not available")
                raise ValueError("Threshold model results not available")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Plot 1: Cointegration relationship
            ax1 = axes[0, 0]
            
            # Extract data from the model
            if hasattr(threshold_model, 'coint_result') and hasattr(threshold_model, 'data1') and hasattr(threshold_model, 'data2'):
                ax1.scatter(threshold_model.data2, threshold_model.data1, alpha=0.6, s=30)
                
                # Create regression line
                x_range = np.linspace(min(threshold_model.data2), max(threshold_model.data2), 100)
                y_range = threshold_model.beta0 + threshold_model.beta1 * x_range
                
                ax1.plot(x_range, y_range, 'r-', linewidth=2, 
                       label=f'y = {threshold_model.beta0:.2f} + {threshold_model.beta1:.2f}x')
                
                ax1.set_xlabel('Series 2')
                ax1.set_ylabel('Series 1')
                ax1.set_title('Cointegration Relationship')
                ax1.legend()
                
            # Plot 2: Error correction term with threshold
            ax2 = axes[0, 1]
            
            if hasattr(threshold_model, 'eq_errors') and hasattr(threshold_model, 'threshold'):
                # Plot error correction term
                if hasattr(threshold_model, 'dates'):
                    ax2.plot(threshold_model.dates, threshold_model.eq_errors, 'b-', alpha=0.7)
                else:
                    ax2.plot(threshold_model.eq_errors, 'b-', alpha=0.7)
                
                # Add threshold line
                ax2.axhline(y=threshold_model.threshold, color='r', linestyle='--', 
                          label=f'Threshold = {threshold_model.threshold:.4f}')
                
                # Add zero line
                ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Error Correction Term')
                ax2.set_title('Error Correction Term with Threshold')
                ax2.legend()
                
            # Plot 3: Threshold estimation
            ax3 = axes[1, 0]
            
            if hasattr(threshold_model, 'threshold_result') and 'thresholds' in threshold_model.threshold_result and 'ssrs' in threshold_model.threshold_result:
                # Plot SSR against threshold values
                thresholds = threshold_model.threshold_result['thresholds']
                ssrs = threshold_model.threshold_result['ssrs']
                
                ax3.plot(thresholds, ssrs, 'b-', alpha=0.7)
                
                # Mark minimum SSR
                best_idx = np.argmin(ssrs)
                best_threshold = thresholds[best_idx]
                best_ssr = ssrs[best_idx]
                
                ax3.scatter([best_threshold], [best_ssr], color='r', s=80, 
                          zorder=3, label=f'Optimal threshold = {best_threshold:.4f}')
                
                ax3.set_xlabel('Threshold Value')
                ax3.set_ylabel('Sum of Squared Residuals')
                ax3.set_title('Threshold Estimation')
                ax3.legend()
                
            # Plot 4: Adjustment speeds by regime
            ax4 = axes[1, 1]
            
            if hasattr(threshold_model, 'tvecm_result'):
                # Extract adjustment speeds
                adj_below_1 = threshold_model.tvecm_result.get('adjustment_below_1', 0)
                adj_above_1 = threshold_model.tvecm_result.get('adjustment_above_1', 0)
                adj_below_2 = threshold_model.tvecm_result.get('adjustment_below_2', 0)
                adj_above_2 = threshold_model.tvecm_result.get('adjustment_above_2', 0)
                
                # Create bar chart
                regimes = ['Below Threshold', 'Above Threshold']
                eq1_speeds = [adj_below_1, adj_above_1]
                eq2_speeds = [adj_below_2, adj_above_2]
                
                x = np.arange(len(regimes))
                width = 0.35
                
                ax4.bar(x - width/2, eq1_speeds, width, label='Equation 1')
                ax4.bar(x + width/2, eq2_speeds, width, label='Equation 2')
                
                ax4.set_xlabel('Regime')
                ax4.set_ylabel('Adjustment Speed')
                ax4.set_title('Adjustment Speeds by Regime')
                ax4.set_xticks(x)
                ax4.set_xticklabels(regimes)
                ax4.legend()
                
                # Add dashed line at y=0
                ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                
                # Add half-life annotations if available
                if 'half_life_below_1' in threshold_model.tvecm_result and 'half_life_above_1' in threshold_model.tvecm_result:
                    hl_below_1 = threshold_model.tvecm_result.get('half_life_below_1', float('inf'))
                    hl_above_1 = threshold_model.tvecm_result.get('half_life_above_1', float('inf'))
                    
                    hl_below_1_str = f"{hl_below_1:.1f}" if hl_below_1 < 1000 else "∞"
                    hl_above_1_str = f"{hl_above_1:.1f}" if hl_above_1 < 1000 else "∞"
                    
                    ax4.annotate(f'Half-life: {hl_below_1_str}', 
                               xy=(x[0] - width/2, eq1_speeds[0]),
                               xytext=(x[0] - width/2, eq1_speeds[0] + 0.02),
                               ha='center', va='bottom', rotation=90)
                    
                    ax4.annotate(f'Half-life: {hl_above_1_str}', 
                               xy=(x[1] - width/2, eq1_speeds[1]),
                               xytext=(x[1] - width/2, eq1_speeds[1] + 0.02),
                               ha='center', va='bottom', rotation=90)
            
            # Set common title if provided
            if title:
                fig.suptitle(title, fontsize=16)
                fig.subplots_adjust(top=0.9)
            
            plt.tight_layout()
            
            return fig, axes
            
        except Exception as e:
            logger.error(f"Error creating threshold analysis visualization: {str(e)}")
            raise
    
    def plot_simulation_comparison(self, original_data: pd.DataFrame, 
                                 simulated_data: pd.DataFrame,
                                 price_col: str = 'price', date_col: str = 'date', 
                                 group_col: Optional[str] = None,
                                 region_col: Optional[str] = 'admin1',
                                 title: Optional[str] = None, 
                                 figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Compare original and simulated price series.
        
        Parameters
        ----------
        original_data : pandas.DataFrame
            Original market data
        simulated_data : pandas.DataFrame
            Simulated market data
        price_col : str, optional
            Column name for price
        date_col : str, optional
            Column name for date
        group_col : str, optional
            Column for grouping (e.g., 'commodity')
        region_col : str, optional
            Column for regional grouping
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        tuple
            (Figure, List of Axes)
        """
        logger.info(f"Creating simulation comparison plot with group_col={group_col}")
        
        try:
            # Ensure date columns are datetime
            original = original_data.copy()
            simulated = simulated_data.copy()
            
            if original[date_col].dtype != 'datetime64[ns]':
                original[date_col] = pd.to_datetime(original[date_col])
            
            if simulated[date_col].dtype != 'datetime64[ns]':
                simulated[date_col] = pd.to_datetime(simulated[date_col])
            
            # Calculate price difference
            original['data_type'] = 'Original'
            simulated['data_type'] = 'Simulated'
            
            # Add identifier columns for merging
            merge_cols = [col for col in original.columns 
                        if col in simulated.columns 
                        and col not in [price_col, 'data_type']]
            
            # Create combined dataset for plotting
            combined = pd.concat([original, simulated])
            
            # Determine plot structure
            if group_col is not None:
                # Get unique groups
                groups = sorted(combined[group_col].unique())
                n_groups = len(groups)
                
                # Create figure with subplots
                fig, axes = plt.subplots(n_groups, 2, figsize=figsize, 
                                      gridspec_kw={'width_ratios': [3, 1]})
                
                for i, group_val in enumerate(groups):
                    # Filter data for this group
                    group_data = combined[combined[group_col] == group_val]
                    
                    # First plot: Time series comparison
                    ax1 = axes[i, 0]
                    
                    # Plot original and simulated data
                    for dtype, marker, color in [
                        ('Original', 'o', 'blue'), 
                        ('Simulated', 's', 'red')
                    ]:
                        subset = group_data[group_data['data_type'] == dtype]
                        ax1.plot(subset[date_col], subset[price_col], 
                              marker, label=dtype, alpha=0.7, color=color)
                    
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Price')
                    ax1.set_title(f'{group_val}')
                    
                    if i == 0:
                        ax1.legend()
                    
                    # Second plot: Price distribution
                    ax2 = axes[i, 1]
                    
                    for dtype, color in [('Original', 'blue'), ('Simulated', 'red')]:
                        subset = group_data[group_data['data_type'] == dtype]
                        sns.kdeplot(subset[price_col], ax=ax2, label=dtype if i == 0 else "", 
                                  color=color, fill=True, alpha=0.3)
                    
                    ax2.set_xlabel('Price')
                    ax2.set_ylabel('Density')
                    
                    if i == 0:
                        ax2.legend()
                
            elif region_col in original.columns and region_col in simulated.columns:
                # Create comparison by region
                # Get unique regions
                regions = sorted(combined[region_col].unique())
                n_regions = min(len(regions), 6)  # Limit to 6 regions for readability
                
                # Create figure with subplots
                fig, axes = plt.subplots(n_regions, 1, figsize=figsize, sharex=True)
                
                for i, region_val in enumerate(regions[:n_regions]):
                    # Filter data for this region
                    region_data = combined[combined[region_col] == region_val]
                    
                    # Plot time series comparison
                    for dtype, marker, color in [
                        ('Original', 'o', 'blue'), 
                        ('Simulated', 's', 'red')
                    ]:
                        subset = region_data[region_data['data_type'] == dtype]
                        axes[i].plot(subset[date_col], subset[price_col], 
                                  marker, label=dtype if i == 0 else "", 
                                  alpha=0.7, color=color)
                    
                    axes[i].set_ylabel('Price')
                    axes[i].set_title(f'{region_val}')
                    
                    if i == 0:
                        axes[i].legend()
                
                # Set common x-axis label
                axes[-1].set_xlabel('Date')
                
            else:
                # Simple comparison without grouping
                fig, axes = plt.subplots(2, 1, figsize=figsize)
                
                # First plot: Time series comparison
                ax1 = axes[0]
                
                # Plot original and simulated data
                for dtype, marker, color in [
                    ('Original', 'o', 'blue'), 
                    ('Simulated', 's', 'red')
                ]:
                    subset = combined[combined['data_type'] == dtype]
                    ax1.plot(subset[date_col], subset[price_col], 
                          marker, label=dtype, alpha=0.7, color=color)
                
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price')
                ax1.set_title('Price Comparison: Original vs Simulated')
                ax1.legend()
                
                # Second plot: Price distribution
                ax2 = axes[1]
                
                for dtype, color in [('Original', 'blue'), ('Simulated', 'red')]:
                    subset = combined[combined['data_type'] == dtype]
                    sns.kdeplot(subset[price_col], ax=ax2, label=dtype, 
                              color=color, fill=True, alpha=0.3)
                
                ax2.set_xlabel('Price')
                ax2.set_ylabel('Density')
                ax2.set_title('Price Distribution')
                ax2.legend()
            
            # Set common title if provided
            if title:
                fig.suptitle(title, fontsize=16)
                fig.subplots_adjust(top=0.9)
            
            plt.tight_layout()
            
            return fig, axes.flat if hasattr(axes, 'flat') else axes
            
        except Exception as e:
            logger.error(f"Error creating simulation comparison plot: {str(e)}")
            raise
```

### 7.2 Enhanced Spatial Visualization Module

```python
"""
Enhanced spatial visualization utilities for market data.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import contextily as ctx
import folium
from folium.plugins import MarkerCluster, HeatMap
import mapclassify
import logging
from typing import Dict, Any, List, Union, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

class MarketMapVisualizer:
    """Enhanced spatial visualizations for market data."""
    
    def __init__(self, backend: str = 'matplotlib'):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        backend : str, optional
            Visualization backend ('matplotlib' or 'folium')
        """
        self.backend = backend
        logger.info(f"Initialized MarketMapVisualizer with backend='{backend}'")
    
    def plot_static_map(self, gdf: gpd.GeoDataFrame, column: Optional[str] = None, 
                      cmap: str = 'viridis', figsize: Tuple[int, int] = (12, 10),
                      title: Optional[str] = None, add_basemap: bool = True, 
                      scheme: str = 'quantiles', k: int = 5, legend: bool = True,
                      legend_kwargs: Optional[Dict[str, Any]] = None) -> plt.Figure:
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
        legend_kwargs : dict, optional
            Additional keyword arguments for legend
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        logger.info(f"Creating static map with column={column}, scheme={scheme}")
        
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Set default legend kwargs if not provided
            if legend_kwargs is None:
                legend_kwargs = {'loc': 'best', 'frameon': True}
            
            if column is not None:
                # Create choropleth map
                try:
                    gdf.plot(column=column, cmap=cmap, ax=ax, scheme=scheme, 
                           k=k, legend=legend, legend_kwds=legend_kwargs)
                except Exception as e:
                    logger.warning(f"Error with classification scheme: {e}")
                    # Fallback to natural breaks if classification fails
                    gdf.plot(column=column, cmap=cmap, ax=ax, legend=legend,
                           legend_kwds=legend_kwargs)
            else:
                # Simple plot without coloring
                gdf.plot(ax=ax, color='blue', alpha=0.7)
            
            if add_basemap:
                # Add basemap if requested and if the CRS is appropriate
                try:
                    # Check if CRS is Web Mercator projection
                    if gdf.crs and gdf.crs.to_string() != 'EPSG:3857':
                        # Try to reproject to Web Mercator for basemap
                        gdf_web = gdf.to_crs('EPSG:3857')
                        ax.set_xlim(gdf_web.total_bounds[[0, 2]])
                        ax.set_ylim(gdf_web.total_bounds[[1, 3]])
                        ctx.add_basemap(ax, crs=gdf_web.crs.to_string())
                    else:
                        ctx.add_basemap(ax)
                        
                except Exception as e:
                    logger.warning(f"Could not add basemap: {e}")
            
            # Set title and remove axis
            ax.set_title(title or 'Market Map')
            ax.set_axis_off()
            
            # Improve legend if it exists
            if legend and column is not None and hasattr(ax, 'get_legend'):
                leg = ax.get_legend()
                if leg:
                    leg.set_title(column.capitalize())
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating static map: {str(e)}")
            raise
    
    def create_interactive_map(self, gdf: gpd.GeoDataFrame, column: Optional[str] = None,
                             popup_cols: Optional[List[str]] = None, title: Optional[str] = None,
                             tiles: str = 'OpenStreetMap', width: int = 800, 
                             height: int = 600) -> folium.Map:
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
        width : int, optional
            Map width in pixels
        height : int, optional
            Map height in pixels
            
        Returns
        -------
        folium.Map
            The created map
        """
        logger.info(f"Creating interactive map with column={column}")
        
        try:
            # Ensure gdf is in WGS84 for folium
            if gdf.crs and gdf.crs.to_string() != 'EPSG:4326':
                logger.info(f"Reprojecting from {gdf.crs} to EPSG:4326")
                gdf = gdf.to_crs('EPSG:4326')
            
            # Get the center of the data
            center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
            
            # Create the map
            m = folium.Map(location=center, zoom_start=7, tiles=tiles, 
                         width=width, height=height)
            
            if title:
                # Add a title
                title_html = f'''
                    <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
                '''
                m.get_root().html.add_child(folium.Element(title_html))
            
            # Prepare colormap if a column is specified
            if column is not None and column in gdf.columns:
                # Create a colormap
                min_val = gdf[column].min()
                max_val = gdf[column].max()
                
                # Function to determine color
                def get_color(value):
                    # Simple linear scale from green to red
                    cmap = plt.cm.get_cmap('viridis')
                    norm_val = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                    rgba = cmap(norm_val)
                    return colors.rgb2hex(rgba)
            
            if 'Point' in gdf.geometry.type.unique():
                # For point data
                marker_cluster = MarkerCluster().add_to(m)
                
                for idx, row in gdf.iterrows():
                    # Skip non-point geometries
                    if row.geometry.type != 'Point':
                        continue
                        
                    # Prepare popup HTML
                    popup_html = ""
                    if popup_cols:
                        popup_html = "<table class='table table-striped'>"
                        for col in popup_cols:
                            if col in row:
                                value = row[col]
                                # Format dates nicely
                                if pd.api.types.is_datetime64_any_dtype(type(value)):
                                    value = value.strftime('%Y-%m-%d')
                                # Format numbers nicely
                                elif isinstance(value, (int, float)):
                                    value = f"{value:,.2f}"
                                popup_html += f"<tr><td><b>{col.capitalize()}</b></td><td>{value}</td></tr>"
                        popup_html += "</table>"
                    
                    # Create marker
                    if column is not None and column in row:
                        # Colored marker based on value
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=8,
                            popup=folium.Popup(popup_html, max_width=300) if popup_html else None,
                            color='black',
                            weight=1,
                            fill=True,
                            fill_color=get_color(row[column]),
                            fill_opacity=0.7,
                            tooltip=f"{column}: {row[column]}"
                        ).add_to(marker_cluster)
                    else:
                        # Simple marker
                        folium.Marker(
                            location=[row.geometry.y, row.geometry.x],
                            popup=folium.Popup(popup_html, max_width=300) if popup_html else None,
                            icon=folium.Icon(color='blue', icon='info-sign')
                        ).add_to(marker_cluster)
            
            elif 'Polygon' in gdf.geometry.type.unique() or 'MultiPolygon' in gdf.geometry.type.unique():
                # For polygon data
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
                    
                    # Add popup functionality to polygons
                    style_function = lambda x: {'fillColor': '#ffffff', 
                                              'color': '#000000', 
                                              'fillOpacity': 0.1, 
                                              'weight': 0.5}
                    highlight_function = lambda x: {'fillColor': '#000000', 
                                                  'color': '#000000', 
                                                  'fillOpacity': 0.50, 
                                                  'weight': 1}
                    
                    # Create popup content
                    fields = popup_cols if popup_cols else [column]
                    aliases = [field.capitalize() for field in fields]
                    
                    NIL = folium.features.GeoJson(
                        gdf,
                        style_function=style_function,
                        control=False,
                        highlight_function=highlight_function,
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=fields,
                            aliases=aliases,
                            style=("background-color: white; color: #333333; "
                                  "font-family: arial; font-size: 12px; "
                                  "padding: 10px;")
                        )
                    )
                    m.add_child(NIL)
                    m.keep_in_front(NIL)
                    
                else:
                    # Simple polygon layer without coloring
                    folium.GeoJson(
                        gdf,
                        name='geojson',
                        style_function=lambda x: {
                            'fillColor': 'blue',
                            'color': 'blue',
                            'weight': 1,
                            'fillOpacity': 0.5
                        },
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=popup_cols if popup_cols else [gdf.index.name or 'index'],
                            aliases=[col.capitalize() for col in (popup_cols if popup_cols else [gdf.index.name or 'index'])],
                            style=("background-color: white; color: #333333; "
                                   "font-family: arial; font-size: 12px; "
                                   "padding: 10px;")
                        )
                    ).add_to(m)
            
            else:
                # For other geometry types
                folium.GeoJson(
                    gdf,
                    name='geojson',
                    style_function=lambda x: {
                        'color': 'blue',
                        'weight': 3,
                        'opacity': 0.5
                    }
                ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating interactive map: {str(e)}")
            raise
    
    def plot_price_heatmap(self, gdf: gpd.GeoDataFrame, commodity: Optional[str] = None,
                         date: Optional[Union[str, pd.Timestamp]] = None, 
                         price_col: str = 'price', cmap: str = 'YlOrRd',
                         figsize: Tuple[int, int] = (12, 10), title: Optional[str] = None,
                         add_basemap: bool = True, scheme: str = 'quantiles', 
                         k: int = 5) -> plt.Figure:
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
        add_basemap : bool, optional
            Whether to add a basemap
        scheme : str, optional
            Classification scheme
        k : int, optional
            Number of classes
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        logger.info(f"Creating price heatmap for commodity={commodity}, date={date}")
        
        try:
            # Filter data
            filtered = gdf.copy()
            
            if commodity is not None and 'commodity' in filtered.columns:
                filtered = filtered[filtered['commodity'] == commodity]
                if len(filtered) == 0:
                    logger.warning(f"No data for commodity '{commodity}'")
                    return None
            
            if date is not None and 'date' in filtered.columns:
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                filtered = filtered[filtered['date'] == date]
                if len(filtered) == 0:
                    logger.warning(f"No data for date '{date}'")
                    return None
            
            # Create the map
            fig, ax = plt.subplots(figsize=figsize)
            
            # Handle empty data
            if len(filtered) == 0:
                ax.text(0.5, 0.5, "No data available for the selected filters",
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes, fontsize=14)
                if title:
                    ax.set_title(title)
                else:
                    if commodity and date:
                        ax.set_title(f'No data for {commodity} on {date.strftime("%Y-%m-%d")}')
                    elif commodity:
                        ax.set_title(f'No data for {commodity}')
                    else:
                        ax.set_title('No data for selected filters')
                
                ax.set_axis_off()
                plt.tight_layout()
                return fig
            
            # Plot with specified color scheme
            try:
                filtered.plot(column=price_col, cmap=cmap, ax=ax, 
                           scheme=scheme, k=k, legend=True,
                           legend_kwds={'title': price_col.capitalize(), 'loc': 'best'})
            except Exception as e:
                logger.warning(f"Error with classification scheme: {e}")
                # Fallback to natural breaks if classification fails
                filtered.plot(column=price_col, cmap=cmap, ax=ax, legend=True,
                           legend_kwds={'title': price_col.capitalize(), 'loc': 'best'})
            
            # Add basemap if requested
            if add_basemap:
                try:
                    # Check if CRS is Web Mercator projection
                    if filtered.crs and filtered.crs.to_string() != 'EPSG:3857':
                        # Try to reproject to Web Mercator for basemap
                        gdf_web = filtered.to_crs('EPSG:3857')
                        ax.set_xlim(gdf_web.total_bounds[[0, 2]])
                        ax.set_ylim(gdf_web.total_bounds[[1, 3]])
                        ctx.add_basemap(ax, crs=gdf_web.crs.to_string())
                    else:
                        ctx.add_basemap(ax)
                except Exception as e:
                    logger.warning(f"Could not add basemap: {e}")
            
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
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price heatmap: {str(e)}")
            raise
    
    def create_conflict_map(self, gdf: gpd.GeoDataFrame, 
                          conflict_col: str = 'conflict_intensity_normalized',
                          date: Optional[Union[str, pd.Timestamp]] = None,
                          interactive: bool = False, figsize: Tuple[int, int] = (12, 10),
                          title: Optional[str] = None) -> Union[plt.Figure, folium.Map]:
        """
        Create a conflict intensity map.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial data with conflict information
        conflict_col : str, optional
            Column containing conflict intensity values
        date : str or datetime, optional
            Date to filter for
        interactive : bool, optional
            Whether to create an interactive map
        figsize : tuple, optional
            Figure size for static map
        title : str, optional
            Map title
            
        Returns
        -------
        matplotlib.figure.Figure or folium.Map
            The created map
        """
        logger.info(f"Creating conflict map with conflict_col={conflict_col}, interactive={interactive}")
        
        try:
            # Filter data
            filtered = gdf.copy()
            
            if conflict_col not in filtered.columns:
                logger.error(f"Conflict column '{conflict_col}' not found in data")
                raise ValueError(f"Conflict column '{conflict_col}' not found in data")
            
            if date is not None and 'date' in filtered.columns:
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                filtered = filtered[filtered['date'] == date]
                if len(filtered) == 0:
                    logger.warning(f"No data for date '{date}'")
                    return None
            
            # Create title if not provided
            if title is None:
                title = 'Conflict Intensity Map'
                if date:
                    title += f' ({date.strftime("%Y-%m-%d")})'
            
            # Create map
            if interactive:
                # Create interactive map with folium
                map_obj = self.create_interactive_map(
                    filtered,
                    column=conflict_col,
                    popup_cols=[conflict_col, 'events', 'fatalities', 'admin1'] 
                    if all(c in filtered.columns for c in ['events', 'fatalities', 'admin1']) 
                    else [conflict_col],
                    title=title,
                    tiles='CartoDB positron'
                )
                
                # Add heat map layer if point data
                if 'Point' in filtered.geometry.type.unique():
                    # Extract coordinates and values for heat map
                    heat_data = []
                    for idx, row in filtered.iterrows():
                        if row.geometry.type == 'Point':
                            heat_data.append([
                                row.geometry.y, 
                                row.geometry.x, 
                                row[conflict_col]
                            ])
                    
                    # Add heat map layer
                    HeatMap(
                        heat_data,
                        name='Conflict Heatmap',
                        radius=15,
                        blur=10,
                        gradient={0.4: 'blue', 0.65: 'yellow', 1: 'red'}
                    ).add_to(map_obj)
                
                return map_obj
                
            else:
                # Create static map with matplotlib
                fig = self.plot_static_map(
                    filtered,
                    column=conflict_col,
                    cmap='inferno_r',  # Reversed inferno colormap (darker = more conflict)
                    figsize=figsize,
                    title=title,
                    add_basemap=True,
                    scheme='fisher_jenks',
                    k=5,
                    legend=True,
                    legend_kwargs={'title': 'Conflict Intensity'}
                )
                
                return fig
                
        except Exception as e:
            logger.error(f"Error creating conflict map: {str(e)}")
            raise
    
    def create_market_integration_map(self, gdf: gpd.GeoDataFrame, 
                                    threshold_values: Optional[Dict[str, float]] = None,
                                    isolation_col: str = 'isolation_index',
                                    date: Optional[Union[str, pd.Timestamp]] = None,
                                    interactive: bool = False, 
                                    figsize: Tuple[int, int] = (12, 10),
                                    title: Optional[str] = None) -> Union[plt.Figure, folium.Map]:
        """
        Create a market integration map showing isolation and integration.
        
        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Spatial market data
        threshold_values : dict, optional
            Dictionary of threshold values by commodity
        isolation_col : str, optional
            Column containing isolation index
        date : str or datetime, optional
            Date to filter for
        interactive : bool, optional
            Whether to create an interactive map
        figsize : tuple, optional
            Figure size for static map
        title : str, optional
            Map title
            
        Returns
        -------
        matplotlib.figure.Figure or folium.Map
            The created map
        """
        logger.info(f"Creating market integration map with isolation_col={isolation_col}")
        
        try:
            # Filter data
            filtered = gdf.copy()
            
            if isolation_col not in filtered.columns:
                logger.warning(f"Isolation column '{isolation_col}' not found in data")
                # Try to calculate isolation if not present
                if 'conflict_intensity_normalized' in filtered.columns:
                    logger.info("Calculating isolation index")
                    from src.models.spatial import SpatialEconometrics
                    spatial_model = SpatialEconometrics(filtered)
                    result_gdf = spatial_model.calculate_market_isolation(
                        conflict_col='conflict_intensity_normalized'
                    )
                    filtered[isolation_col] = result_gdf[isolation_col]
                else:
                    logger.error("Cannot calculate isolation index without conflict data")
                    raise ValueError("Isolation index not available and cannot be calculated")
            
            if date is not None and 'date' in filtered.columns:
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                filtered = filtered[filtered['date'] == date]
                if len(filtered) == 0:
                    logger.warning(f"No data for date '{date}'")
                    return None
            
            # Create title if not provided
            if title is None:
                title = 'Market Integration Map'
                if date:
                    title += f' ({date.strftime("%Y-%m-%d")})'
            
            # Add threshold boundaries if provided
            if threshold_values and 'commodity' in filtered.columns:
                # Create a new column for threshold status
                filtered['threshold_status'] = 'unknown'
                
                # For each commodity, set threshold status
                for commodity, threshold in threshold_values.items():
                    mask = filtered['commodity'] == commodity
                    if 'price_diff' in filtered.columns:
                        filtered.loc[mask & (filtered['price_diff'].abs() <= threshold), 'threshold_status'] = 'below'
                        filtered.loc[mask & (filtered['price_diff'].abs() > threshold), 'threshold_status'] = 'above'
            
            # Create map
            if interactive:
                # Create interactive map with folium
                popup_cols = [isolation_col, 'admin1', 'commodity', 'price']
                popup_cols = [col for col in popup_cols if col in filtered.columns]
                
                if 'threshold_status' in filtered.columns:
                    popup_cols.append('threshold_status')
                
                map_obj = self.create_interactive_map(
                    filtered,
                    column=isolation_col,
                    popup_cols=popup_cols,
                    title=title,
                    tiles='CartoDB positron'
                )
                
                return map_obj
                
            else:
                # Create static map with matplotlib
                fig = self.plot_static_map(
                    filtered,
                    column=isolation_col,
                    cmap='RdYlGn_r',  # Reversed RdYlGn (red = isolated, green = integrated)
                    figsize=figsize,
                    title=title,
                    add_basemap=True,
                    scheme='fisher_jenks',
                    k=5,
                    legend=True,
                    legend_kwargs={'title': 'Market Isolation'}
                )
                
                return fig
                
        except Exception as e:
            logger.error(f"Error creating market integration map: {str(e)}")
            raise
    
    def plot_policy_impact_map(self, original_gdf: gpd.GeoDataFrame, 
                             simulated_gdf: gpd.GeoDataFrame,
                             metric_col: str = 'price', date: Optional[Union[str, pd.Timestamp]] = None, 
                             figsize: Tuple[int, int] = (16, 8), title: Optional[str] = None) -> plt.Figure:
        """
        Create a map showing policy impact by comparing original and simulated data.
        
        Parameters
        ----------
        original_gdf : geopandas.GeoDataFrame
            Original spatial data
        simulated_gdf : geopandas.GeoDataFrame
            Simulated spatial data after policy intervention
        metric_col : str, optional
            Column to compare (e.g., 'price', 'isolation_index')
        date : str or datetime, optional
            Date to filter for
        figsize : tuple, optional
            Figure size
        title : str, optional
            Map title
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        logger.info(f"Creating policy impact map for metric_col={metric_col}")
        
        try:
            # Filter data
            orig_filtered = original_gdf.copy()
            sim_filtered = simulated_gdf.copy()
            
            if metric_col not in orig_filtered.columns or metric_col not in sim_filtered.columns:
                logger.error(f"Metric column '{metric_col}' not found in data")
                raise ValueError(f"Metric column '{metric_col}' not found in data")
            
            if date is not None and 'date' in orig_filtered.columns and 'date' in sim_filtered.columns:
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                orig_filtered = orig_filtered[orig_filtered['date'] == date]
                sim_filtered = sim_filtered[sim_filtered['date'] == date]
                if len(orig_filtered) == 0 or len(sim_filtered) == 0:
                    logger.warning(f"No data for date '{date}'")
                    return None
            
            # Create a figure with two subplots side by side
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Plot 1: Original data
            orig_filtered.plot(column=metric_col, cmap='viridis', ax=axes[0], 
                            legend=True, legend_kwds={'title': metric_col.capitalize()})
            
            if title:
                axes[0].set_title(f"Original {metric_col.capitalize()}")
            else:
                axes[0].set_title(f"Original {metric_col.capitalize()}")
            
            # Plot 2: Simulated data
            sim_filtered.plot(column=metric_col, cmap='viridis', ax=axes[1], 
                           legend=True, legend_kwds={'title': metric_col.capitalize()})
            
            if title:
                axes[1].set_title(f"Simulated {metric_col.capitalize()}")
            else:
                axes[1].set_title(f"Simulated {metric_col.capitalize()}")
            
            # Plot 3: Difference
            # Need to ensure the data can be joined correctly
            common_cols = [col for col in orig_filtered.columns 
                         if col in sim_filtered.columns 
                         and col != metric_col
                         and col != 'geometry']
            
            if not common_cols:
                logger.warning("No common columns found for joining data")
                axes[2].text(0.5, 0.5, "Cannot calculate differences: no common columns for joining",
                          horizontalalignment='center', verticalalignment='center',
                          transform=axes[2].transAxes, fontsize=10)
                axes[2].set_title("Difference")
                axes[2].set_axis_off()
            else:
                # Join the data
                join_col = common_cols[0]  # Use first common column for join
                
                # Create DataFrames without geometry for joining
                orig_df = pd.DataFrame(orig_filtered.drop(columns='geometry'))
                sim_df = pd.DataFrame(sim_filtered.drop(columns='geometry'))
                
                # Rename columns to avoid conflicts
                orig_df = orig_df.rename(columns={metric_col: f"orig_{metric_col}"})
                sim_df = sim_df.rename(columns={metric_col: f"sim_{metric_col}"})
                
                # Join the DataFrames
                merged = pd.merge(orig_df, sim_df, on=join_col, suffixes=('_orig', '_sim'))
                
                # Calculate difference
                merged['difference'] = merged[f"sim_{metric_col}"] - merged[f"orig_{metric_col}"]
                merged['pct_change'] = (merged['difference'] / merged[f"orig_{metric_col}"]) * 100
                
                # Join back to geometry
                diff_gdf = orig_filtered[[join_col, 'geometry']].merge(
                    merged[[join_col, 'difference', 'pct_change']], 
                    on=join_col
                )
                
                # Plot difference
                diff_gdf.plot(column='difference', cmap='RdBu_r', ax=axes[2], 
                           legend=True, legend_kwds={'title': 'Difference'})
                
                axes[2].set_title(f"Difference in {metric_col.capitalize()}")
            
            # Turn off axes
            for ax in axes:
                ax.set_axis_off()
                
                # Try to add basemap
                try:
                    ctx.add_basemap(ax)
                except Exception as e:
                    pass
            
            # Set common title if provided
            if title:
                fig.suptitle(title, fontsize=16)
                fig.subplots_adjust(top=0.9)
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating policy impact map: {str(e)}")
            raise
```

## Phase 8: Documentation and Testing (Week 15-16)

For this phase, I'll provide key documentation files and test implementations that correspond to the enhanced modules we've developed.

### 8.1 Create Documentation for Threshold Models

Create a sample `docs/threshold_models.md` documentation file:

```markdown
# Threshold Cointegration Models Guide

This guide explains how to use the enhanced threshold cointegration modules for analyzing price transmission in Yemen's fragmented markets.

## Overview

The threshold cointegration models implemented in this project are designed to capture nonlinear price adjustment processes that occur when transaction costs create a "band of inaction" where small price differences persist without triggering arbitrage.

## Available Models

### ThresholdCointegration

Simple implementation for a pair of price series using the Engle-Granger approach.

```python
from src.models.threshold import ThresholdCointegration

# Initialize with two price series
model = ThresholdCointegration(north_prices, south_prices, max_lags=4)

# Run full analysis
results = model.run_full_analysis()

# Access key results
threshold = results['threshold']['threshold']
adjustment_below = results['tvecm']['adjustment_below_1']
adjustment_above = results['tvecm']['adjustment_above_1']
print(f"Threshold: {threshold:.4f}")
print(f"Adjustment below threshold: {adjustment_below:.4f}")
print(f"Adjustment above threshold: {adjustment_above:.4f}")
```

### ThresholdVECM

More comprehensive implementation based on Hansen & Seo (2002) for multivariate systems.

```python
from src.models.threshold_vecm import ThresholdVECM

# Initialize with price matrix
model = ThresholdVECM(market_data, k_ar_diff=2, deterministic="ci")

# Run full analysis
results = model.run_full_analysis()

# Access key results
threshold = results['threshold']['threshold']
adjustment_below = results['tvecm']['adjustment_below']
adjustment_above = results['tvecm']['adjustment_above']
```

## Key Concepts

### 1. Threshold Parameter

The threshold parameter represents the minimum price differential that must be exceeded before arbitrage becomes profitable. It can be interpreted as a measure of transaction costs (transportation, security fees, trade barriers).

### 2. Adjustment Speeds

Adjustment speeds indicate how quickly prices respond to deviations from equilibrium:

- **Below threshold**: In the "band of inaction," adjustment is typically slow or non-existent
- **Above threshold**: Once price differentials exceed transaction costs, arbitrage becomes profitable and adjustment is faster

### 3. Half-lives

Half-lives represent the time required for a deviation to be reduced by 50%:

- Calculated as: ln(0.5) / ln(1 + α), where α is the adjustment speed
- Shorter half-lives indicate faster market integration

## Diagnostic Tests

The models include comprehensive diagnostics:

1. **Residual tests**: Check for normality, autocorrelation, heteroskedasticity
2. **Stability tests**: Verify parameter stability across subsamples
3. **Bootstrap tests**: Statistical significance of threshold effect

## Visualization

Visualize threshold model results:

```python
from src.visualization.time_series import TimeSeriesVisualizer

# Create visualizer
viz = TimeSeriesVisualizer()

# Plot threshold analysis
fig, axes = viz.plot_threshold_analysis(threshold_model)
```

## Interpretation Guide

### Integrated Markets

- Significant adjustment above threshold
- Faster adjustment above than below threshold
- Threshold aligns with estimated transaction costs

### Fragmented Markets

- Slow or insignificant adjustment in both regimes
- High threshold value
- Long half-lives

### Partially Integrated Markets

- Significant adjustment above threshold
- Very slow adjustment below threshold
- Threshold higher than expected transaction costs

## Policy Simulation

Use simulation models to assess potential interventions:

```python
from src.models.simulation import MarketIntegrationSimulation

# Create simulation
sim = MarketIntegrationSimulation(data, threshold_model=model)

# Simulate exchange rate unification
unified_results = sim.simulate_exchange_rate_unification()

# Compare threshold parameters and adjustment speeds
original_threshold = model.threshold
new_threshold = unified_results['threshold_model'].threshold
print(f"Threshold reduction: {original_threshold - new_threshold:.4f}")
```


### 8.2 Create Key Test Files

Create `tests/unit/test_threshold.py`:

```python
"""
Unit tests for threshold cointegration models.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.threshold import ThresholdCointegration
from src.models.threshold_vecm import ThresholdVECM

class TestThresholdCointegration(unittest.TestCase):
    """Tests for the ThresholdCointegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic cointegrated series with threshold effects
        np.random.seed(42)
        
        # Common trend
        trend = np.cumsum(np.random.normal(0, 1, 200))
        
        # First series
        self.y1 = trend + np.random.normal(0, 0.5, 200)
        
        # Second series with adjustment process that depends on threshold
        self.y2 = np.zeros(200)
        self.y2[0] = trend[0] + np.random.normal(0, 0.5)
        
        # True threshold value
        self.true_threshold = 1.0
        
        # Generate series with threshold adjustment
        for t in range(1, 200):
            # Calculate error correction term
            ect = self.y1[t-1] - self.y2[t-1]
            
            # Adjustment depends on threshold
            if abs(ect) <= self.true_threshold:
                # Slow adjustment below threshold
                adj_speed = -0.05
            else:
                # Fast adjustment above threshold
                adj_speed = -0.3
            
            # Apply error correction
            self.y2[t] = self.y2[t-1] + adj_speed * ect + np.random.normal(0, 0.5)
        
        # Create test dates
        self.dates = pd.date_range('2020-01-01', periods=200)
        
        # Create test model
        self.model = ThresholdCointegration(self.y1, self.y2, max_lags=2)
    
    def test_cointegration_estimation(self):
        """Test estimation of cointegration relationship."""
        result = self.model.estimate_cointegration()
        
        # Check for cointegration
        self.assertTrue(result['cointegrated'], 
                      "Series should be cointegrated")
        
        # Check coefficients are not zero
        self.assertNotEqual(result['beta0'], 0, 
                         "Intercept should not be zero")
        self.assertNotEqual(result['beta1'], 0, 
                         "Slope should not be zero")
    
    def test_threshold_estimation(self):
        """Test estimation of threshold parameter."""
        # First ensure cointegration relationship is estimated
        self.model.estimate_cointegration()
        
        # Estimate threshold
        result = self.model.estimate_threshold(n_grid=100)
        
        # Check threshold is reasonably close to true value
        self.assertAlmostEqual(result['threshold'], self.true_threshold, delta=0.5, 
                             msg="Estimated threshold should be close to true value")
        
        # Check threshold search results
        self.assertIn('thresholds', result, 
                    "Result should contain threshold candidates")
        self.assertIn('ssrs', result, 
                    "Result should contain SSR values")
        self.assertEqual(len(result['thresholds']), len(result['ssrs']), 
                       "Length of thresholds and SSRs should match")
    
    def test_tvecm_estimation(self):
        """Test estimation of TVECM."""
        # Run cointegration and threshold estimation first
        self.model.estimate_cointegration()
        self.model.estimate_threshold()
        
        # Estimate TVECM
        result = self.model.estimate_tvecm()
        
        # Check results
        self.assertIn('adjustment_below_1', result, 
                    "Result should contain adjustment speed below threshold")
        self.assertIn('adjustment_above_1', result, 
                    "Result should contain adjustment speed above threshold")
        
        # Check asymmetry: adjustment should be faster above threshold
        self.assertLess(result['adjustment_above_1'], result['adjustment_below_1'], 
                      "Adjustment should be faster (more negative) above threshold")
    
    def test_diagnostics(self):
        """Test model diagnostics."""
        # Run full analysis
        self.model.run_full_analysis()
        
        # Run diagnostics
        result = self.model.run_diagnostics()
        
        # Check diagnostic results
        self.assertIn('equation1', result, 
                    "Diagnostics should contain results for equation 1")
        self.assertIn('equation2', result, 
                    "Diagnostics should contain results for equation 2")
    
    def test_full_analysis(self):
        """Test full analysis pipeline."""
        result = self.model.run_full_analysis()
        
        # Check all components are present
        self.assertIn('cointegration', result, 
                    "Result should contain cointegration analysis")
        self.assertIn('threshold', result, 
                    "Result should contain threshold analysis")
        self.assertIn('tvecm', result, 
                    "Result should contain TVECM results")
        self.assertIn('diagnostics', result, 
                    "Result should contain diagnostics")


class TestThresholdVECM(unittest.TestCase):
    """Tests for the ThresholdVECM class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic cointegrated series with threshold effects
        np.random.seed(42)
        
        # Common trend
        trend = np.cumsum(np.random.normal(0, 1, 200))
        
        # Series with different loadings on the trend
        y1 = trend + np.random.normal(0, 0.5, 200)
        y2 = 2 * trend + np.random.normal(0, 0.5, 200)
        
        # Create DataFrame for VECM
        self.data = pd.DataFrame({
            'y1': y1,
            'y2': y2
        })
        
        # Create test model
        self.model = ThresholdVECM(self.data, k_ar_diff=1, deterministic="ci", coint_rank=1)
    
    def test_linear_vecm(self):
        """Test estimation of linear VECM."""
        result = self.model.estimate_linear_vecm()
        
        # Check that model is estimated
        self.assertIsNotNone(result, "VECM result should not be None")
        
        # Check cointegration vector
        self.assertTrue(hasattr(self.model, 'beta'), 
                      "Model should have beta attribute")
        self.assertIsNotNone(self.model.beta, 
                          "Cointegration vector should not be None")
    
    def test_threshold_grid_search(self):
        """Test grid search for threshold parameter."""
        # First estimate linear VECM
        self.model.estimate_linear_vecm()
        
        # Estimate threshold
        result = self.model.grid_search_threshold(n_grid=50)
        
        # Check results
        self.assertIn('threshold', result, 
                    "Result should contain threshold value")
        self.assertIn('llf', result, 
                    "Result should contain log-likelihood")
        self.assertIn('thresholds', result, 
                    "Result should contain threshold candidates")
        self.assertIn('llfs', result, 
                    "Result should contain log-likelihood values")
    
    def test_tvecm_estimation(self):
        """Test estimation of TVECM."""
        # Run linear VECM and threshold search first
        self.model.estimate_linear_vecm()
        self.model.grid_search_threshold()
        
        # Estimate TVECM
        result = self.model.estimate_tvecm()
        
        # Check results
        self.assertIn('adjustment_below', result, 
                    "Result should contain adjustment speeds below threshold")
        self.assertIn('adjustment_above', result, 
                    "Result should contain adjustment speeds above threshold")
        self.assertIn('threshold', result, 
                    "Result should contain threshold value")
    
    def test_full_analysis(self):
        """Test full analysis pipeline."""
        result = self.model.run_full_analysis()
        
        # Check all components are present
        self.assertIn('linear_vecm', result, 
                    "Result should contain linear VECM results")
        self.assertIn('threshold', result, 
                    "Result should contain threshold results")
        self.assertIn('tvecm', result, 
                    "Result should contain TVECM results")
        self.assertIn('diagnostics', result, 
                    "Result should contain diagnostics")

if __name__ == '__main__':
    unittest.main()
```

Create `tests/unit/test_spatial.py`:

```python
"""
Unit tests for spatial econometric models.
"""
import unittest
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.spatial import SpatialWeightMatrix, SpatialEconometrics

class TestSpatialWeightMatrix(unittest.TestCase):
    """Tests for the SpatialWeightMatrix class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test GeoDataFrame
        points = [
            Point(0, 0),
            Point(0, 1),
            Point(1, 1),
            Point(1, 0),
            Point(0.5, 0.5)
        ]
        
        data = {
            'market_id': range(1, 6),
            'price': [100, 110, 120, 130, 115],
            'conflict_intensity_normalized': [0.2, 0.5, 0.8, 0.3, 0.1]
        }
        
        self.gdf = gpd.GeoDataFrame(data, geometry=points, crs="EPSG:4326")
        
        # Create weight matrix handler
        self.weight_handler = SpatialWeightMatrix(self.gdf)
    
    def test_create_knn_weights(self):
        """Test creation of k-nearest neighbors weights."""
        # Create standard KNN weights
        weights = self.weight_handler.create_knn_weights(k=2)
        
        # Check basic properties
        self.assertEqual(len(weights), 5, "Weight matrix should have 5 observations")
        self.assertEqual(weights.n, 5, "Weight matrix should have 5 observations")
        
        # Check that each unit has exactly 2 neighbors
        for i in range(5):
            self.assertEqual(len(weights.neighbors[i]), 2, 
                          f"Unit {i} should have 2 neighbors")
        
        # Check row-standardization
        for i in range(5):
            self.assertAlmostEqual(sum(weights.weights[i]), 1.0, places=10,
                               msg="Weights should be row-standardized")
    
    def test_conflict_adjusted_weights(self):
        """Test creation of conflict-adjusted weights."""
        # Create standard KNN weights
        std_weights = self.weight_handler.create_knn_weights(k=2, conflict_adjusted=False)
        
        # Create conflict-adjusted weights
        adj_weights = self.weight_handler.create_knn_weights(
            k=2, 
            conflict_adjusted=True, 
            conflict_col='conflict_intensity_normalized'
        )
        
        # Neighbors should be the same
        for i in range(5):
            self.assertEqual(std_weights.neighbors[i], adj_weights.neighbors[i],
                          f"Conflict adjustment should not change neighbors for unit {i}")
        
        # But weights should be different
        weight_differences = []
        for i in range(5):
            for j in range(len(std_weights.weights[i])):
                diff = std_weights.weights[i][j] - adj_weights.weights[i][j]
                weight_differences.append(diff)
        
        # At least some weights should differ
        self.assertTrue(any(abs(diff) > 1e-10 for diff in weight_differences),
                     "At least some weights should differ with conflict adjustment")
    
    def test_distance_weights(self):
        """Test creation of distance-based weights."""
        # Create distance weights
        weights = self.weight_handler.create_distance_weights(threshold=1.5)
        
        # Check basic properties
        self.assertEqual(len(weights), 5, "Weight matrix should have 5 observations")
        
        # Central point should be connected to all others
        central_idx = self.gdf[self.gdf.geometry == Point(0.5, 0.5)].index[0]
        self.assertEqual(len(weights.neighbors[central_idx]), 4,
                      "Central point should be connected to all other points")
        
        # Row-standardization
        for i in range(5):
            if len(weights.weights[i]) > 0:
                self.assertAlmostEqual(sum(weights.weights[i]), 1.0, places=10,
                                   msg="Weights should be row-standardized")


class TestSpatialEconometrics(unittest.TestCase):
    """Tests for the SpatialEconometrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test GeoDataFrame
        np.random.seed(42)
        
        # Create grid of points
        x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        x = x.flatten()
        y = y.flatten()
        
        # Create points
        points = [Point(xy) for xy in zip(x, y)]
        
        # Create values with spatial autocorrelation
        base_values = np.random.normal(0, 1, len(points))
        
        # Add spatial structure: nearby points have similar values
        values = np.zeros_like(base_values)
        for i in range(len(points)):
            # For each point, add weighted values of nearby points
            weights = []
            for j in range(len(points)):
                if i != j:
                    # Weight inversely by distance
                    dist = points[i].distance(points[j])
                    weight = 1 / (1 + dist * 5)  # Scale distance effect
                    weights.append((j, weight))
            
            # Normalize weights
            total_weight = sum(w for _, w in weights)
            norm_weights = [(j, w / total_weight) for j, w in weights]
            
            # Weighted average of own value and neighbors
            values[i] = 0.3 * base_values[i] + 0.7 * sum(base_values[j] * w for j, w in norm_weights)
        
        # Scale to reasonable range for prices
        prices = 100 + 20 * values
        
        # Add conflict intensity
        conflict = 0.2 + 0.6 * np.random.beta(2, 5, len(points))
        
        # Create DataFrame
        data = {
            'market_id': range(1, len(points) + 1),
            'price': prices,
            'conflict_intensity_normalized': conflict,
            'other_var': np.random.normal(0, 1, len(points))
        }
        
        self.gdf = gpd.GeoDataFrame(data, geometry=points, crs="EPSG:4326")
        
        # Create spatial model
        self.spatial_model = SpatialEconometrics(self.gdf)
        
        # Create and set weights
        weight_handler = SpatialWeightMatrix(self.gdf)
        weights = weight_handler.create_knn_weights(k=4)
        self.spatial_model.set_weights(weights)
    
    def test_moran_i_test(self):
        """Test Moran's I test for spatial autocorrelation."""
        # Test on price variable
        result = self.spatial_model.moran_i_test('price', permutations=99)
        
        # Check result structure
        self.assertIn('I', result, "Result should contain Moran's I statistic")
        self.assertIn('p_sim', result, "Result should contain p-value")
        self.assertIn('significant', result, "Result should indicate significance")
        
        # Value should be between -1 and 1
        self.assertTrue(-1 <= result['I'] <= 1, 
                      "Moran's I should be between -1 and 1")
    
    def test_local_moran_test(self):
        """Test Local Moran's I (LISA) test."""
        # Test on price variable
        result = self.spatial_model.local_moran_test('price', permutations=99)
        
        # Check result structure
        self.assertIsInstance(result, gpd.GeoDataFrame, 
                           "Result should be a GeoDataFrame")
        self.assertIn('Is', result.columns, 
                    "Result should contain local Moran's I statistics")
        self.assertIn('p_sim', result.columns, 
                    "Result should contain p-values")
        self.assertIn('significant', result.columns, 
                    "Result should indicate significance")
        self.assertIn('cluster_type', result.columns, 
                    "Result should contain cluster types")
        
        # Check same number of observations
        self.assertEqual(len(result), len(self.gdf), 
                       "Result should have same number of observations as input")
    
    def test_spatial_lag_model(self):
        """Test spatial lag model estimation."""
        # Estimate model
        model = self.spatial_model.spatial_lag_model(
            y_col='price',
            x_cols=['conflict_intensity_normalized', 'other_var']
        )
        
        # Check model attributes
        self.assertTrue(hasattr(model, 'rho'), 
                      "Model should have spatial coefficient (rho)")
        self.assertTrue(hasattr(model, 'betas'), 
                      "Model should have coefficient estimates")
        self.assertTrue(hasattr(model, 'std_err'), 
                      "Model should have standard errors")
        
        # Check model is stored
        self.assertIn('lag', self.spatial_model.models, 
                    "Model should be stored in models dictionary")
    
    def test_spatial_error_model(self):
        """Test spatial error model estimation."""
        # Estimate model
        model = self.spatial_model.spatial_error_model(
            y_col='price',
            x_cols=['conflict_intensity_normalized', 'other_var']
        )
        
        # Check model attributes
        self.assertTrue(hasattr(model, 'lam'), 
                      "Model should have spatial error coefficient (lambda)")
        self.assertTrue(hasattr(model, 'betas'), 
                      "Model should have coefficient estimates")
        self.assertTrue(hasattr(model, 'std_err'), 
                      "Model should have standard errors")
        
        # Check model is stored
        self.assertIn('error', self.spatial_model.models, 
                    "Model should be stored in models dictionary")
    
    def test_calculate_market_isolation(self):
        """Test calculation of market isolation index."""
        # Calculate isolation
        result = self.spatial_model.calculate_market_isolation(
            conflict_col='conflict_intensity_normalized'
        )
        
        # Check result
        self.assertIsInstance(result, gpd.GeoDataFrame, 
                           "Result should be a GeoDataFrame")
        self.assertIn('isolation_index', result.columns, 
                    "Result should contain isolation index")
        self.assertIn('isolation_category', result.columns, 
                    "Result should contain isolation category")
        
        # Check values in expected range
        self.assertTrue(all(0 <= val <= 1 for val in result['isolation_index']), 
                      "Isolation index should be between 0 and 1")

if __name__ == '__main__':
    unittest.main()
```

## Phase 9: Finishing Touches and Documentation (Week 17)

### 9.1 Add Continuous Integration Configuration

Create `.github/workflows/python-package.yml`:

```yaml
name: Python Package

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.8', '3.9', '3.10']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install GDAL dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y gdal-bin libgdal-dev
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install flake8 pytest pytest-cov black
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install -e .
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
    - name: Check formatting with black
      run: |
        black --check src tests
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src
```

### 9.2 Create Comprehensive README.md

```markdown
# Yemen Market Integration Analysis

## Overview

This project implements an econometric framework for analyzing market integration in conflict-affected Yemen. Using advanced threshold cointegration and spatial econometric techniques, we quantify how conflict-induced barriers affect price transmission across politically fragmented territories.

## Key Features

- **Comprehensive Data Pipeline**: Robust data acquisition, cleaning, and integration
- **Advanced Econometric Models**: Threshold cointegration, spatial lag/error models
- **Conflict-Adjusted Analysis**: Integration of conflict data for transaction cost estimation
- **Policy Simulations**: Exchange rate unification and connectivity improvement scenarios
- **Interactive Visualizations**: Time series and spatial visualizations for enhanced analysis

## Installation

### Requirements

- Python 3.8+
- GDAL library (for spatial analysis)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/yemen-market-integration.git
cd yemen-market-integration

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Docker Setup (Alternative)

```bash
# Build Docker image
docker build -t yemen-market-integration .

# Run container
docker run -it --rm -v $(pwd):/app yemen-market-integration bash
```

## Usage

### Data Processing

```python
from src.data.sources import WFPDataConnector, ACLEDConflictConnector
from src.data.integration import DataIntegrator

# Fetch price data
wfp = WFPDataConnector()
prices = wfp.fetch_commodity_prices(country="Yemen", start_date="2020-01-01")

# Fetch conflict data
acled = ACLEDConflictConnector()
conflicts = acled.fetch_conflict_data(country="Yemen", start_date="2020-01-01")

# Integrate data sources
integrator = DataIntegrator()
integrated_data = integrator.integrate_market_data(
    commodity_file="data/raw/prices.csv",
    conflict_file="data/raw/conflicts.csv",
    exchange_file="data/raw/exchange_rates.csv",
    admin_file="data/raw/admin_boundaries.geojson"
)
```

### Threshold Analysis

```python
from src.models.threshold import ThresholdCointegration
from src.visualization.time_series import TimeSeriesVisualizer

# Initialize and run threshold cointegration analysis
model = ThresholdCointegration(north_prices, south_prices, max_lags=4)
results = model.run_full_analysis()

# Visualize results
viz = TimeSeriesVisualizer()
fig, axes = viz.plot_threshold_analysis(model)
```

### Spatial Analysis

```python
from src.models.spatial import SpatialWeightMatrix, SpatialEconometrics
from src.visualization.market_maps import MarketMapVisualizer

# Create spatial weights
weight_handler = SpatialWeightMatrix(market_gdf)
weights = weight_handler.create_knn_weights(
    k=5, 
    conflict_adjusted=True,
    conflict_col='conflict_intensity_normalized'
)

# Perform spatial analysis
spatial_model = SpatialEconometrics(market_gdf)
spatial_model.set_weights(weights)
moran_result = spatial_model.moran_i_test('price')
isolation_gdf = spatial_model.calculate_market_isolation()

# Visualize results
map_viz = MarketMapVisualizer()
map_fig = map_viz.create_market_integration_map(isolation_gdf)
```

### Policy Simulation

```python
from src.models.simulation import MarketIntegrationSimulation

# Initialize simulation model
sim = MarketIntegrationSimulation(data, threshold_model=model, spatial_model=spatial_model)

# Simulate exchange rate unification
unified_results = sim.simulate_exchange_rate_unification(target_rate='average')

# Simulate improved connectivity
connectivity_results = sim.simulate_improved_connectivity(reduction_factor=0.5)

# Simulate combined policies
combined_results = sim.simulate_combined_policy()

# Calculate welfare effects
welfare = sim.calculate_welfare_effects(policy_scenario='combined_policy')
```

## Project Structure

```
yemen-market-integration/
│
├── data/                          # Data storage
│   ├── raw/                       # Original data
│   ├── processed/                 # Cleaned data
│   └── external/                  # External data sources
│
├── src/                           # Source code
│   ├── data/                      # Data processing
│   │   ├── sources.py             # Data acquisition
│   │   ├── integration.py         # Data integration
│   │   └── validation.py          # Data validation
│   │
│   ├── models/                    # Econometric models
│   │   ├── threshold.py           # Threshold cointegration
│   │   ├── threshold_vecm.py      # Threshold VECM
│   │   ├── spatial.py             # Spatial econometrics
│   │   ├── simulation.py          # Policy simulations
│   │   ├── unit_root.py           # Unit root testing
│   │   ├── cointegration.py       # Cointegration testing
│   │   └── diagnostics.py         # Model diagnostics
│   │
│   ├── visualization/             # Visualization tools
│   │   ├── time_series.py         # Time series visualization
│   │   └── market_maps.py         # Spatial visualization
│   │
│   └── utils/                     # Utility functions
│
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
│
├── notebooks/                     # Jupyter notebooks
├── docs/                          # Documentation
├── config/                        # Configuration files
│
├── requirements.txt               # Dependencies
├── setup.py                       # Package setup
└── README.md                      # Project documentation
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Threshold Models Guide](docs/threshold_models.md)
- [Spatial Analysis Guide](docs/spatial_analysis.md)
- [Policy Simulation Guide](docs/simulations.md)
- [Visualization Guide](docs/visualization.md)

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- World Food Programme (WFP) for market price data
- Armed Conflict Location & Event Data Project (ACLED) for conflict data
- Central Bank of Yemen for exchange rate data
```

### 9.3 Final Checklist Before Deployment

1. Ensure all docstrings and comments are comprehensive
2. Check for consistent code style across modules
3. Verify test coverage for all core functionality
4. Validate documentation for all major components
5. Add licensing information and contributor guidelines
6. Create containerization support for easier deployment
7. Add example notebooks for major use cases

This enhanced implementation plan provides a robust framework for analyzing market integration in conflict-affected Yemen, with comprehensive data handling, advanced econometric modeling, and powerful visualization tools.