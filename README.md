# Yemen Market Integration Analysis

## Overview

This project implements advanced econometric methodologies for analyzing market integration in conflict-affected Yemen. It examines how conflict-induced transaction costs impede arbitrage and impact dual exchange rate regimes, using threshold cointegration and spatial econometric techniques.

The analysis addresses key questions:
- How do conflict barriers affect price transmission between markets?
- What are the economic impacts of the dual exchange rate regime?
- How would exchange rate unification and conflict reduction affect market integration?
- Which regions and commodities show most resistance to integration?

## Installation

```bash
# Clone the repository
git clone https://github.com/star-boy-95/yemen-market-analysis-v5.git
cd yemen-market-analysis-v5

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Features

- **Data Processing**: Comprehensive tools for loading, cleaning, and transforming market data from various sources, including WFP and ACLED. Supports GeoJSON format with spatial attributes.
- **Econometric Analysis**: 
  - **Unit Root Testing**: ADF, Zivot-Andrews, and other tests with structural break detection
  - **Cointegration Analysis**: Engle-Granger, Johansen, Gregory-Hansen methods
  - **Threshold Models**: TAR, M-TAR, and Threshold VECM implementations
  - **Model Diagnostics**: Comprehensive residual analysis and model validation
- **Spatial Econometrics**: Implementation of spatial econometric models including:
  - Spatial weight matrix creation with conflict-adjustment
  - Spatial lag models (SLM) and spatial error models (SEM)
  - Integration of spatial and time series results
- **Policy Simulation**: Tools for simulating various policy interventions:
  - Exchange rate unification scenarios
  - Connectivity improvement simulation
  - Market integration enhancement strategies
- **Visualization**: Comprehensive visualization tools:
  - Time series plots with regime highlighting
  - Interactive spatial maps for market integration
  - Diagnostic plots for model evaluation
- **Integrated Analysis Pipeline**: End-to-end workflow from data processing to result interpretation

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

### Econometric Analysis

```python
# Unit Root and Cointegration Testing
from src.models.unit_root import UnitRootTester
from src.models.cointegration import CointegrationTester

# Test for unit roots
ur_tester = UnitRootTester(price_series)
ur_results = ur_tester.test_adf(lags=4)

# Test for cointegration
coint_tester = CointegrationTester(north_prices, south_prices)
eg_results = coint_tester.test_engle_granger()
johansen_results = coint_tester.test_johansen(k_ar_diff=2)

# Threshold Cointegration Analysis
from src.models.threshold import ThresholdCointegration
from src.models.threshold_vecm import ThresholdVECM

# Standard threshold model
threshold_model = ThresholdCointegration(north_price, south_price)
threshold_model.estimate_cointegration()
tar_result = threshold_model.estimate_threshold()

# Threshold VECM
tvecm = ThresholdVECM(north_price, south_price)
tvecm_result = tvecm.estimate(k_ar_diff=2, trim=0.15)
```

### Spatial Analysis and Simulations

```python
# Spatial Econometrics
from src.models.spatial import SpatialEconometrics

# Create and analyze spatial relationships
spatial_model = SpatialEconometrics(markets_gdf)
weights = spatial_model.create_weight_matrix(
    k=5,
    conflict_adjusted=True,
    conflict_col='conflict_intensity'
)
moran_results = spatial_model.moran_test('price', weights)
spatial_lag_model = spatial_model.spatial_lag_model('price', ['distance', 'conflict'])

# Policy Simulation
from src.models.simulation import MarketIntegrationSimulation

# Simulate exchange rate unification
sim = MarketIntegrationSimulation(markets_data)
unification_results = sim.simulate_exchange_rate_unification(target_rate='official')
```

### Integration and Reporting

```python
# Import integration modules
from src.models.spatiotemporal import integrate_time_series_spatial_results
from src.models.interpretation import interpret_threshold_results
from src.models.reporting import generate_comprehensive_report

# Integrate time series and spatial results
integrated_results = integrate_time_series_spatial_results(
    time_series_results={
        'unit_root': unit_root_results,
        'cointegration': cointegration_results,
        'tvecm': threshold_results['tvecm']
    },
    spatial_results=spatial_results,
    commodity='beans (kidney red)'
)

# Interpret threshold model results
interpretation = interpret_threshold_results(threshold_results, 'beans (kidney red)')
print(f"Summary: {interpretation['summary']}")
for implication in interpretation['implications']:
    print(f"- {implication}")

# Generate comprehensive report
report_path = generate_comprehensive_report(
    all_results=all_results,
    commodity='beans (kidney red)',
    output_path=Path('output'),
    logger=logger
)
```

### Running the Full Analysis Pipeline

```bash
# Run the integrated analysis for a specific commodity
python run_yemen_analysis.py --commodity "beans (kidney red)" --output results

# Run with additional parameters
python run_yemen_analysis.py --commodity "wheat" --output results --max-lags 6 --k-neighbors 8 --report-format markdown
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_integrated_analysis.py

# Run tests with coverage report
pytest tests --cov=src --cov-report=term --cov-report=html
```

## Development Tools

This project includes several development tools to ensure code quality:

```bash
# Format code with Black
black src tests

# Lint with Flake8
flake8 src tests

# Run Jupyter notebook server for analysis
jupyter notebook --notebook-dir=notebooks
```

You can also run these tools using VS Code tasks:
- "Format Code with Black"
- "Lint with Flake8"
- "Run Tests"
- "Run Tests with Coverage"
- "Run Notebook Server"

## Documentation

The project includes comprehensive documentation:

- **Project Overview**: `docs/project_overview.md` - High-level project goals and methodology
- **Implementation Plan**: `docs/implementation_plan.md` - Development roadmap and status
- **Econometric Methods**: `docs/econometric_methods.md` - Mathematical specifications
- **Data Dictionary**: `docs/data_dictionary.md` - Data sources and structure
- **Integration Modules**: `docs/integration_modules.md` - Guide to the integration modules
- **Models Guide**: `docs/models_guide.md` - Guide to econometric models
- **Utils Guide**: `docs/utils_guide.md` and `UTILS_GUIDE.md` - Utility functions reference
- **Best Practices**: `docs/best_practices.md` - Code examples and patterns
- **Coding Standards**: `docs/coding_standards.md` - Project coding standards

## Project Structure

```
yemen-market-integration/
│
├── data/                          # Data storage
│   ├── raw/                       # Original data
│   └── processed/                 # Cleaned and transformed data
│
├── src/                           # Source code
│   ├── data/                      # Data processing modules
│   │   ├── loader.py              # Data loading utilities
│   │   ├── preprocessor.py        # Data preprocessing utilities
│   │   └── integration.py         # Data integration utilities
│   │
│   ├── models/                    # Econometric models
│   │   ├── unit_root.py           # Unit root testing
│   │   ├── cointegration.py       # Cointegration testing
│   │   ├── threshold.py           # Threshold cointegration
│   │   ├── threshold_vecm.py      # TVECM implementation
│   │   ├── threshold_fixed.py     # Fixed threshold models
│   │   ├── threshold_model.py     # Base threshold model class
│   │   ├── threshold_reporter.py  # Threshold results reporting
│   │   ├── spatial.py             # Spatial econometrics
│   │   ├── spatiotemporal.py      # Combined spatial-time series
│   │   ├── interpretation.py      # Results interpretation
│   │   ├── reporting.py           # Report generation
│   │   ├── simulation.py          # Policy simulation
│   │   └── diagnostics.py         # Model diagnostics
│   │
│   ├── visualization/             # Visualization tools
│   │   ├── time_series.py         # Time series visualization
│   │   ├── maps.py                # Spatial visualization
│   │   ├── asymmetric_plots.py    # Asymmetric regime plots
│   │   ├── spatial_integration.py # Spatial integration maps
│   │   └── dashboard.py           # Interactive dashboard
│   │
│   └── utils/                     # Utility functions
│       ├── error_handler.py       # Error handling
│       ├── config.py              # Configuration
│       ├── validation.py          # Data validation
│       ├── logging_setup.py       # Logging configuration
│       ├── performance_utils.py   # Performance optimization
│       ├── spatial_utils.py       # Spatial utilities
│       └── plotting_utils.py      # Plotting utilities
│
├── notebooks/                     # Jupyter notebooks
│   ├── 00_project_initialization.ipynb     # Setup
│   └── policy_simulation_walkthrough.ipynb # Simulation demo
│
├── tests/                         # Tests
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   ├── performance/               # Performance tests
│   └── visualization/             # Visualization tests
│
├── docs/                          # Documentation
│   ├── api/                       # API documentation
│   ├── best_practices.md          # Best practices
│   ├── coding_standards.md        # Coding standards
│   ├── data_dictionary.md         # Data dictionary
│   ├── implementation_plan.md     # Implementation plan
│   ├── models_guide.md            # Models guide
│   └── archived/                  # Archived documentation
│
├── config/                        # Configuration files
│   ├── settings.yaml              # Project settings
│   └── logging.yaml               # Logging configuration
│
├── logs/                          # Log files
├── output/                        # Analysis output files
├── results/                       # Analysis results
│
├── run_yemen_analysis.py          # Main entry point script
├── run_full_analysis.py           # Full analysis pipeline
├── generate_report.py             # Report generation script
├── requirements.txt               # Core dependencies
├── requirements-dev.txt           # Development dependencies
├── setup.py                       # Package installation
├── setup.cfg                      # Package configuration
├── build.sh                       # Build script
└── Makefile                       # Project makefile
```

## Dependencies

This project relies on several key Python packages:

**Core Data Processing**
- numpy, pandas, scipy

**Geospatial Analysis**
- geopandas, pyproj, pysal, folium, contextily, libpysal, esda, spreg, splot, networkx

**Econometrics and Statistics**
- statsmodels, scikit-learn, arch, pmdarima, linearmodels

**Spatial Econometrics**
- spreg, esda, spglm, spint

**Visualization**
- matplotlib, seaborn, plotly, mapclassify, bokeh

**Performance Optimization**
- numba, joblib, dask, swifter, psutil, cython

**Time Series Analysis**
- statsforecast, tsfresh, prophet

See `requirements.txt` for full dependency list.

## Contributing

We welcome contributions to this project! Please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Write clear, concise, and well-documented code.
4. Follow the project's coding standards (PEP 8).
5. Write unit tests for your code.
6. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
