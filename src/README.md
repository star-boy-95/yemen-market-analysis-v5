# Yemen Market Analysis

## Overview

This package provides a comprehensive suite of econometric tools for analyzing market integration in Yemen. It includes methods for time series analysis, spatial econometrics, and threshold models, with a focus on rigorous statistical inference and publication-quality reporting.

## Features

- **Data Loading and Preprocessing**: Tools for loading and preprocessing market price data from various sources.
- **Unit Root Testing**: Comprehensive unit root tests including ADF, KPSS, Phillips-Perron, and Zivot-Andrews tests.
- **Cointegration Analysis**: Methods for testing cointegration relationships, including Engle-Granger, Johansen, and Gregory-Hansen tests.
- **Threshold Models**: Implementation of threshold cointegration models, including TAR, M-TAR, and threshold VECM.
- **Spatial Analysis**: Tools for spatial econometrics, including spatial weight matrices, spatial autocorrelation tests, and spatial regression models.
- **Visualization**: Publication-quality visualizations for time series, spatial data, and econometric results.
- **Reporting**: Tools for generating comprehensive reports with formal hypothesis testing and statistical significance indicators.

## Installation

```bash
pip install yemen-market-analysis
```

For development:

```bash
pip install -e ".[dev]"
```

For spatial analysis features:

```bash
pip install -e ".[spatial]"
```

## Usage

### Basic Usage

```python
from yemen_market_analysis.data import DataLoader
from yemen_market_analysis.models import UnitRootTester, CointegrationTester, ThresholdModel
from yemen_market_analysis.visualization import TimeSeriesVisualizer

# Load data
loader = DataLoader()
data = loader.load_geojson("data/raw/unified_data.geojson")

# Preprocess data for a specific commodity
commodity_data = loader.preprocess_commodity_data(data, "wheat")

# Perform unit root tests
tester = UnitRootTester()
unit_root_results = tester.run_all_tests(commodity_data["market1"])

# Test for cointegration
coint_tester = CointegrationTester()
coint_results = coint_tester.test_engle_granger(
    commodity_data["market1"], 
    commodity_data["market2"]
)

# Estimate threshold model
model = ThresholdModel(
    commodity_data["market1"],
    commodity_data["market2"],
    mode="standard"
)
results = model.run_full_analysis()

# Visualize results
visualizer = TimeSeriesVisualizer()
fig = visualizer.plot_price_series(
    commodity_data,
    title="Wheat Prices in Different Markets"
)
fig.savefig("wheat_prices.png", dpi=300)
```

### Running a Complete Analysis

```python
from yemen_market_analysis.main import run_analysis

# Run a complete analysis for a specific commodity
results = run_analysis(
    commodity="wheat",
    threshold_modes=["standard", "fixed"],
    include_spatial=True,
    publication_quality=True
)

# Generate a comprehensive report
from yemen_market_analysis.models.reporting import generate_report

report = generate_report(
    results,
    format="latex",
    output_path="wheat_analysis_report.tex"
)
```

## Documentation

For detailed documentation, see the [API Reference](https://yemen-market-analysis.readthedocs.io/).

## Academic References

This package implements methods from the following academic papers:

- Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 251-276.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 1551-1580.
- Hansen, B. E., & Seo, B. (2002). Testing for two-regime threshold cointegration in vector error-correction models. *Journal of Econometrics*, 110(2), 293-318.
- Gregory, A. W., & Hansen, B. E. (1996). Residual-based tests for cointegration in models with regime shifts. *Journal of Econometrics*, 70(1), 99-126.
- Enders, W., & Siklos, P. L. (2001). Cointegration and threshold adjustment. *Journal of Business & Economic Statistics*, 19(2), 166-176.

## License

MIT License
