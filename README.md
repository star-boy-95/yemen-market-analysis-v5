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
├── requirements.txt               # Project dependencies
├── setup.py                       # Package installation
└── README.md                      # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.