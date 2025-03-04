"""
Configuration management for Yemen Market Analysis.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        
        # Default configuration
        return {
            'directories': {
                'results_dir': 'results',
                'logs_dir': 'results/logs',
                'threshold_tracker': 'results/threshold_history/commodity_thresholds.json'
            },
            'parameters': {
                'threshold_cointegration': {
                    'grid_points': 50,
                    'nboot': 500,
                    'block_size': 5,
                    'min_regime_size': 0.1,
                    'use_adaptive_thresholds': True,
                    'conflict_adjustment_factor': 0.5
                },
                'max_price_volatility': 0.5,
                'max_interpolation_gap': 3,
                'min_periods': 20,
                'lag_periods': 8,
                'parallel_commodities': True,
                'commodity_parallel_processes': 4,
                'tsay_order': 2
            },
            'logging': {
                'log_level': 'INFO',
                'verbose_libraries': {
                    'matplotlib': 'WARNING',
                    'matplotlib.font_manager': 'ERROR'
                }
            },
            'model': {
                'version': '3.3'
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        result = self.config
        
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
                
        return result
    
    def get_path(self, key_path: str) -> Path:
        """Get a path from configuration, ensuring it exists."""
        path_str = self.get(key_path)
        if not path_str:
            raise ValueError(f"Path configuration '{key_path}' not found")
            
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        target = self.config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
                
        target[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path specified for saving configuration")
            
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Global configuration instance
config = Config()


def initialize_config(config_path: str) -> Config:
    """Initialize the global configuration."""
    global config
    config = Config(config_path)
    return config