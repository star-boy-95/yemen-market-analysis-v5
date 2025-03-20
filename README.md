# Yemen Market Integration Analysis

## Overview

This project implements econometric methodologies for analyzing market integration in conflict-affected Yemen. It examines how conflict-induced transaction costs impede arbitrage and impact dual exchange rate regimes, using threshold cointegration and spatial econometric techniques.

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Features

- **Data Processing**: Comprehensive tools for loading, cleaning, and transforming market data from various sources, including WFP and ACLED. Supports GeoJSON format with spatial attributes.
- **Time Series Analysis**: Advanced econometric methods for analyzing time series data, including unit root testing (ADF, Zivot-Andrews), cointegration analysis (Engle-Granger, Johansen, Gregory-Hansen), and threshold modeling (TAR, M-TAR).
- **Spatial Econometrics**: Implementation of spatial econometric models to analyze geographic dependencies in market integration, including spatial weight matrix creation, spatial lag models (SLM), and spatial error models (SEM).
- **Policy Simulation**: Tools for simulating various policy interventions, such as exchange rate unification and connectivity improvement, to assess their impact on market integration and welfare.
- **Visualization**: Comprehensive visualization tools for both time series and spatial data, including time series plots with regime highlighting, spatial maps for market integration visualization, and interactive dashboards for data exploration.
- **Integration Modules**: New modules for integrating time series and spatial results, interpreting analysis results, and generating comprehensive reports.

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

# Estimate Threshold Cointegration Model
from src.models.threshold import ThresholdCointegration
model = ThresholdCointegration(north_price, south_price)
model.estimate_cointegration()
result = model.estimate_threshold()

# Simulate Exchange Rate Unification
from src.models.simulation import MarketIntegrationSimulation
sim = MarketIntegrationSimulation(markets_data)
result = sim.simulate_exchange_rate_unification(target_rate='official')

# Create Spatial Weight Matrix
from src.models.spatial import SpatialEconometrics
model = SpatialEconometrics(markets_gdf)
weights = model.create_weight_matrix(
    k=5,
    conflict_adjusted=True,
    conflict_col='conflict_intensity'
)
```

### Using the New Integration Modules

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

### Running the Full Integrated Analysis

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
```

## Installation and Development

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/star-boy-95/yemen-market-analysis-v5.git
cd yemen-market-analysis-v5

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Development Installation

For development, install additional tools:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Documentation

For documentation development:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html
```

## Documentation

The project includes comprehensive documentation:

- **Project Overview**: `docs/project_overview.md` - High-level project goals and methodology
- **Implementation Plan**: `docs/implementation_plan.md` - Development roadmap and status
- **Econometric Methods**: `docs/econometric_methods.md` - Mathematical specifications
- **Data Dictionary**: `docs/data_dictionary.md` - Data sources and structure
- **Integration Modules**: `docs/integration_modules.md` - Guide to the new integration modules
- **API References**:
  - `docs/api/econometrics_api.md` - Econometric model API
  - `docs/api/simulation_api.md` - Simulation model API
  - `docs/api/visualization_api.md` - Visualization API
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
│   ├── simulation/                # Simulation pipeline
│   │   └── simulation_pipeline.py # End-to-end simulation
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
│   ├── guides/                    # User guides
│   └── archived/                  # Archived documentation
│
├── examples/                      # Example scripts
│   └── integrated_analysis_example.py  # Example of using integration modules
│
├── config/                        # Configuration files
│   ├── settings.yaml              # Project settings
│   └── logging.yaml               # Logging configuration
│
├── logs/                          # Log files
│
├── run_yemen_analysis.py          # Main entry point script
├── requirements.txt               # Core dependencies
├── requirements-dev.txt           # Development dependencies
├── setup.py                       # Package installation
├── setup.cfg                      # Package configuration
├── .env                           # Environment variables
├── .pre-commit-config.yaml        # Pre-commit hooks
├── build.sh                       # Build script
└── Makefile                       # Project makefile
```

## Contributing

We welcome contributions to this project! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Write clear, concise, and well-documented code.
4.  Follow the project's coding standards (PEP 8).
5.  Write unit tests for your code.
6.  Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
