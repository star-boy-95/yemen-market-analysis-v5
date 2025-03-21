"""
Configuration management for the Yemen Market Integration Project.
Handles loading, validating, and accessing configuration.
"""
import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import logging
from .error_handler import handle_errors, ConfigError

logger = logging.getLogger(__name__)

class Config:
    """
    Centralized configuration management with singleton pattern
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance._loaded_paths = []
        return cls._instance
    
    @handle_errors(logger=logger, error_type=(FileNotFoundError, PermissionError, yaml.YAMLError, json.JSONDecodeError), reraise=True)
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a file (yaml or json)
        
        Parameters
        ----------
        config_path : str or Path
            Path to the configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Track loaded paths to avoid duplicates
        if str(config_path) in self._loaded_paths:
            logger.warning(f"Configuration already loaded from {config_path}")
            return
            
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Merge with existing configuration
        self._update_nested_dict(self._config, config_data)
        self._loaded_paths.append(str(config_path))
        logger.info(f"Loaded configuration from {config_path}")
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from a dictionary
        
        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration
        """
        self._update_nested_dict(self._config, config_dict)
        logger.info("Loaded configuration from dictionary")
    
    def load_from_env(self, prefix: str = "YEMEN_") -> None:
        """
        Load configuration from environment variables with a specific prefix
        
        Parameters
        ----------
        prefix : str, optional
            Prefix for environment variables to include
        """
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase for consistency
                config_key = key[len(prefix):].lower()
                # Handle nested keys with double underscore
                if '__' in config_key:
                    parts = config_key.split('__')
                    current = env_config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = self._parse_env_value(value)
                else:
                    env_config[config_key] = self._parse_env_value(value)
        
        self._update_nested_dict(self._config, env_config)
        logger.info(f"Loaded configuration from environment variables with prefix {prefix}")
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable values to appropriate types"""
        # Try to convert to appropriate type
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        elif value.lower() == 'null' or value.lower() == 'none':
            return None
        
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If it's a comma-separated list
            if ',' in value:
                return [Config._parse_env_value(v.strip()) for v in value.split(',')]
            return value
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update a nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key
        
        Parameters
        ----------
        key : str
            Configuration key, can use dot notation for nested keys
        default : Any, optional
            Default value if the key is not found
            
        Returns
        -------
        Any
            Configuration value or default
        """
        if not key:
            return default
            
        if '.' not in key:
            return self._config.get(key, default)
        
        # Handle nested keys with dot notation
        parts = key.split('.')
        value = self._config
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section
        
        Parameters
        ----------
        section : str
            Section name
            
        Returns
        -------
        dict
            Section configuration or empty dict if not found
        """
        return self.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value
        
        Parameters
        ----------
        key : str
            Configuration key, can use dot notation for nested keys
        value : Any
            Value to set
        """
        if '.' not in key:
            self._config[key] = value
            return
        
        # Handle nested keys with dot notation
        parts = key.split('.')
        current = self._config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save current configuration to a file
        
        Parameters
        ----------
        file_path : str or Path
            Path to save the configuration file
        format : str, optional
            Format to save ('yaml' or 'json')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(self._config, f, default_flow_style=False)
            elif format.lower() == 'json':
                json.dump(self._config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to {file_path}")
    
    def reset(self) -> None:
        """Reset configuration to empty state"""
        self._config = {}
        self._loaded_paths = []
        logger.info("Configuration reset")
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model
        
        Parameters
        ----------
        model_name : str
            Name of the model
            
        Returns
        -------
        dict
            Model parameters or empty dict if not found
        """
        return self.get(f"models.{model_name}", {})
    
    def get_all(self) -> Dict[str, Any]:
        """Get a copy of the entire configuration"""
        return self._config.copy()

# Global instance for easy imports
config = Config()

def initialize_config(
    config_file: Optional[Union[str, Path]] = None,
    env_prefix: str = "YEMEN_",
    defaults: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Initialize configuration from various sources
    
    Parameters
    ----------
    config_file : str or Path, optional
        Path to configuration file
    env_prefix : str, optional
        Prefix for environment variables
    defaults : dict, optional
        Default configuration values
        
    Returns
    -------
    Config
        Initialized configuration object
    """
    # Start with defaults if provided
    if defaults:
        config.load_from_dict(defaults)
    
    # Load from environment
    config.load_from_env(prefix=env_prefix)
    
    # Load from file if provided (overrides environment and defaults)
    if config_file:
        config.load_from_file(config_file)
    
    return config