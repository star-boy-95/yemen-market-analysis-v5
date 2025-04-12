"""
Configuration module for Yemen Market Analysis.

This module provides a centralized configuration system for the Yemen Market Analysis
package. It loads configuration from a YAML file and provides access to configuration
values with type checking and default values.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple, TypeVar, cast

# Type variable for generic type hints
T = TypeVar('T')

# Initialize logger
logger = logging.getLogger(__name__)

class Config:
    """
    Configuration manager for Yemen Market Analysis.
    
    This class loads configuration from a YAML file and provides access to
    configuration values with type checking and default values.
    
    Attributes:
        config_path (Path): Path to the configuration file.
        config (Dict[str, Any]): Configuration dictionary.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses the default
                         path in the package directory.
        """
        if config_path is None:
            # Use default config path in package directory
            package_dir = Path(__file__).parent
            config_path = package_dir / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from the YAML file.
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the configuration file is not valid YAML.
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config = {}
                return
            
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            if self.config is None:
                self.config = {}
                
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = {}
    
    def get(self, key: str, default: Optional[T] = None) -> T:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key, using dot notation for nested keys.
            default: Default value to return if the key is not found.
            
        Returns:
            The configuration value, or the default value if the key is not found.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return cast(T, default)
        
        return cast(T, value)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key, using dot notation for nested keys.
            value: Value to set.
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration file. If None, uses the
                         current configuration path.
                         
        Raises:
            PermissionError: If the file cannot be written.
        """
        if config_path is None:
            config_path = self.config_path
        else:
            config_path = Path(config_path)
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

# Create a singleton instance
config = Config()

# Default configuration values
DEFAULT_CONFIG = {
    'data': {
        'path': './data',
        'raw_path': './data/raw',
        'processed_path': './data/processed',
    },
    'analysis': {
        'unit_root': {
            'alpha': 0.05,
            'max_lags': 4,
        },
        'cointegration': {
            'alpha': 0.05,
            'max_lags': 4,
            'trend': 'c',
        },
        'threshold': {
            'alpha': 0.05,
            'trim': 0.15,
            'n_grid': 300,
            'max_lags': 4,
            'n_bootstrap': 1000,
            'mtar_default_threshold': 0.0,
        },
        'threshold_vecm': {
            'k_ar_diff': 2,
            'deterministic': 'ci',
            'coint_rank': 1,
        },
        'spatial': {
            'conflict_column': 'conflict_intensity_normalized',
            'conflict_weight': 0.5,
            'price_column': 'price',
            'conflict_reduction': 0.5,
        },
        'simulation': {
            'policy_type': 'exchange_rate',
            'target_rate': 'official',
            'reduction_factor': 0.5,
        },
        'chunk_size': 5000,
        'gh': {
            'early_stop_threshold': -10.0,
        },
    },
    'performance': {
        'n_workers': os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1,
        'cache_dir': '.cache/yemen_market_analysis',
        'memory_limit': 0.8,  # Use up to 80% of available memory
    },
    'visualization': {
        'style': 'seaborn-v0_8-whitegrid',
        'figure_size': (10, 6),
        'font_family': 'serif',
        'font_size': 12,
        'dpi': 300,
    },
    'reporting': {
        'format': 'markdown',
        'style': 'world_bank',
        'confidence_level': 0.95,
        'significance_indicators': True,
    },
}

# Set default configuration values
for key, value in DEFAULT_CONFIG.items():
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            if isinstance(subvalue, dict):
                for subsubkey, subsubvalue in subvalue.items():
                    if config.get(f"{key}.{subkey}.{subsubkey}") is None:
                        config.set(f"{key}.{subkey}.{subsubkey}", subsubvalue)
            elif config.get(f"{key}.{subkey}") is None:
                config.set(f"{key}.{subkey}", subvalue)
    elif config.get(key) is None:
        config.set(key, value)
