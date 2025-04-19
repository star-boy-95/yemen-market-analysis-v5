"""
Re-export of ThresholdCointegration class from threshold.py.

This module is created to avoid circular imports when importing ThresholdCointegration
in the test files.
"""

import sys
import os
import importlib.util

# Get the absolute path to the threshold.py file
threshold_path = os.path.join(os.path.dirname(__file__), 'threshold.py')

# Load the module directly from the file
spec = importlib.util.spec_from_file_location('threshold_direct', threshold_path)
threshold_direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(threshold_direct)

# Get the ThresholdCointegration class from the loaded module
ThresholdCointegration = threshold_direct.ThresholdCointegration

__all__ = ['ThresholdCointegration']