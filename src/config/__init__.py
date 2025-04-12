"""
Configuration module for Yemen Market Analysis.

This module provides access to the configuration settings for the Yemen Market Analysis
package. It reads the configuration from a YAML file and provides access to the
settings through a simple interface.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Configuration class that makes properties available via attribute access
class ConfigDict(dict):
    """Dictionary subclass that provides attribute-style access to settings."""
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
    def __getattr__(self, name):
        """Get attribute value, returning None if it doesn't exist."""
        return self.get(name)

class Config:
    """Configuration manager for Yemen Market Analysis."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config_path = Path(__file__).parent / "settings.yaml"
        self.config = self._load_config()
        
        # Add direct access to main sections
        for section, values in self.config.items():
            if isinstance(values, dict):
                setattr(self, section, ConfigDict(values))
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration settings.
        """
        if not self.config_path.exists():
            print(f"Configuration file not found: {self.config_path}")
            return {}
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, section_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-notation path.
        
        Args:
            section_path: Configuration path using dot notation (e.g., 'analysis.unit_root.alpha').
            default: Default value to return if the path is not found.
            
        Returns:
            Configuration value if found, otherwise the default value.
        """
        # Split the path into parts
        parts = section_path.split('.')
        
        # Start at the root of the config
        current = self.config
        
        # Navigate through the path
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current

# Create a singleton instance
config = Config()
