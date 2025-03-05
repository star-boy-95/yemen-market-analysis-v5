# Yemen Market Integration Analysis

## Overview

This project implements econometric methodologies for analyzing market integration in conflict-affected Yemen. It examines how conflict-induced transaction costs impede arbitrage and impact dual exchange rate regimes, using threshold cointegration and spatial econometric techniques.

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

## Features

- **Data Processing**: Tools for handling GeoJSON market data with spatial attributes
- **Time Series Analysis**: Unit root testing, cointegration analysis, and threshold modeling
- **Spatial Econometrics**: Tools for analyzing geographic dependencies in market integration
- **Policy Simulation**: Tools for simulating market integration scenarios (e.g., exchange rate unification)
- **Visualization**: Comprehensive visualization tools for both time series and spatial data

## Usage

### Basic Data Processing

```python
# Import key modules
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

# Load GeoJSON data
loader = DataLoader()
gdf = loader.load_geojson('unified_data.geojson')

# Preprocess data
preprocessor = DataPreprocessor()
processed_gdf = preprocessor.preprocess_geojson(gdf)

# Save processed data
loader.save_processed_data(processed_gdf, 'processed_data.geojson')

# Calculate price differentials
differentials = preprocessor.calculate_price_differentials(processed_gdf)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_data_loader.py
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
│   │   ├── loader.py              # Data loading utilities
│   │   ├── preprocessor.py        # Data preprocessing utilities
│   │   └── integration.py         # Data integration utilities
│   │
│   ├── models/                    # Econometric models
│   │   ├── unit_root.py           # Unit root testing module
│   │   ├── cointegration.py       # Cointegration testing module
│   │   ├── threshold.py           # Threshold cointegration module
│   │   ├── threshold_vecm.py      # Threshold VECM implementation
│   │   └── spatial.py             # Spatial econometrics models
│   │
│   ├── visualization/             # Visualization tools
│   │   ├── time_series.py         # Time series visualization tools
│   │   └── market_maps.py         # Spatial visualization tools
│   │
│   └── utils/                     # Utility functions
│       ├── error_handler.py       # Error handling utilities
│       ├── config.py              # Configuration management
│       └── validation.py          # Data validation utilities
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb    # Data exploration
│   ├── 02_threshold_cointegration.ipynb # Cointegration analysis
│   ├── 03_spatial_analysis.ipynb        # Spatial econometrics
│   └── 04_policy_simulations.ipynb      # Policy simulations
│
├── tests/                         # Unit tests
│   ├── test_data_loader.py        # Tests for data loader
│   ├── test_preprocessor.py       # Tests for preprocessor
│   └── test_models.py             # Tests for econometric models
│
├── config/                        # Configuration files
│   └── settings.yaml              # Project settings
│
├── logs/                          # Log files
├── figures/                       # Output figures
│
├── requirements.txt               # Project dependencies
├── setup.py                       # Package installation script
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation