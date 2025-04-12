# Migration Guide: From `src` to `src_refactored`

This document provides guidance on migrating from the original Yemen Market Analysis codebase structure to the refactored version.

## Overview of Major Changes

The refactoring effort has made the following significant improvements:

1. **Object-Oriented Architecture**: Moved from procedural scripts to class-based design
2. **Modular Organization**: Reorganized code into logical, focused modules
3. **Enhanced Testing**: Added comprehensive test suite
4. **Standardized Configuration**: Centralized configuration management
5. **Improved Documentation**: Added API docs and usage examples

## Import Path Changes

### Old Structure:
```python
from src.data.loader import DataLoader
from src.models.unit_root import UnitRootTester
from src.models.threshold_model import ThresholdModel
from src.utils.config import get_config
```

### New Structure:
```python
from src_refactored.data.loader import DataLoader
from src_refactored.models.unit_root import UnitRootTester
from src_refactored.models.threshold import ThresholdModel
from src_refactored.config import config
```

## API Changes

### Running Analysis

#### Old Approach:
```python
import sys
import os
from pathlib import Path
import logging

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.run_integrated_analysis import parse_args, run_integrated_analysis, setup_logging

# Set up logging
logger = setup_logging()

# Parse arguments
args = parse_args()

# Run the analysis
run_integrated_analysis(args, logger)
```

#### New Approach:
```python
from src_refactored.main import run_analysis

# Run a complete analysis with simplified interface
results = run_analysis(
    data_path="data/raw/unified_data.geojson",
    commodity="wheat",
    threshold_modes=["standard", "fixed"],
    include_spatial=True,
    publication_quality=True,
    output_dir="results"
)
```

## Model Comparison

### Threshold Models

#### Old Structure:
The previous implementation spread threshold functionality across multiple files:
- `threshold_model.py`
- `threshold_fixed.py`
- `threshold_vecm.py`
- `threshold_reporter.py`

#### New Structure:
Threshold functionality is now organized in a cohesive subpackage:
- `models/threshold/model.py` - Core implementation
- `models/threshold/fixed.py` - Fixed threshold variant
- `models/threshold/vecm.py` - VECM implementation
- `models/threshold/reporting.py` - Results reporting
- `models/threshold/visualization.py` - Specialized visualizations

## Config Usage

### Old Approach:
```python
from src.utils.config import get_config

config = get_config()
max_lags = config.get('analysis', {}).get('max_lags', 4)
```

### New Approach:
```python
from src_refactored.config import config

max_lags = config.analysis.max_lags  # Defaults handled internally
```

## Testing Your Code

We've included an integration test suite to help verify that the refactored code produces equivalent results to the original implementation. Run the tests with:

```bash
cd src_refactored
python -m unittest discover tests
```

## Deprecation Timeline

- **Current**: Both `src` and `src_refactored` structures are maintained
- **Next Release**: The original `src` structure will emit deprecation warnings
- **Future Release**: The original `src` structure will be removed

## Getting Help

If you encounter issues during migration, please open an issue on our GitHub repository with details about the error and your use case.
