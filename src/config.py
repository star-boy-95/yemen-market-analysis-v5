"""
Configuration module for Yemen Market Analysis.

This module provides a centralized configuration system for the Yemen Market Analysis
package. It imports the configuration from src.config.__init__ to provide a unified
interface.
"""

# Import configuration from src.config.__init__
from src.config.__init__ import config, Config, ConfigDict

# Export the config object and Config class
__all__ = ['config', 'Config', 'ConfigDict']
