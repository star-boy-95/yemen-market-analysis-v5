"""
Configuration Validation Test.

Verifies that the configuration system contains all required options and
that default values are appropriate.
"""
import sys
import os
import unittest
from pathlib import Path

# Add paths to both implementations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class ConfigurationValidationTest(unittest.TestCase):
    """Test that all required configuration options are present and valid."""
    
    def test_required_config_sections(self):
        """Verify all required config sections exist."""
        # Load config
        from src.config import config
        
        # Required sections for the application
        required_sections = ["analysis", "paths", "visualization", "data", "models"]
        
        # Transform config object to dict if needed
        if hasattr(config, '__dict__'):
            config_dict = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith('_')}
        else:
            config_dict = config
            
        for section in required_sections:
            self.assertIn(section, config_dict, f"Missing required config section: {section}")
    
    def test_required_config_parameters(self):
        """Verify all required config parameters exist."""
        # Load config
        from src.config import config
        
        # Critical parameters to check - add or modify based on your application needs
        required_params = [
            ("analysis", "max_lags"),
            ("analysis", "n_neighbors"),
            ("paths", "output_dir"),
            ("visualization", "dpi"),
            ("data", "default_path"),
            ("models", "default_mode")
        ]
        
        for section, param in required_params:
            # Check if section exists
            self.assertTrue(hasattr(config, section), f"Missing config section: {section}")
            
            # Get section and check if parameter exists
            section_obj = getattr(config, section)
            self.assertTrue(hasattr(section_obj, param) or param in section_obj, 
                          f"Missing required config parameter: {section}.{param}")
    
    def test_config_value_types(self):
        """Verify that config values have appropriate types."""
        # Load config
        from src.config import config
        
        # Type expectations for critical parameters
        type_expectations = [
            ("analysis.max_lags", int),
            ("analysis.n_neighbors", int),
            ("paths.output_dir", (str, Path)),
            ("visualization.dpi", int),
            ("models.threshold.default_mode", str)
        ]
        
        for param_path, expected_type in type_expectations:
            # Parse the parameter path
            parts = param_path.split('.')
            
            # Navigate through the config object
            curr = config
            for part in parts:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                elif isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    self.fail(f"Could not find parameter {param_path}")
                    break
            
            # Check the type
            if isinstance(expected_type, tuple):
                self.assertTrue(any(isinstance(curr, t) for t in expected_type),
                              f"Parameter {param_path} has incorrect type. Expected one of {expected_type}, got {type(curr)}")
            else:
                self.assertIsInstance(curr, expected_type,
                                   f"Parameter {param_path} has incorrect type. Expected {expected_type}, got {type(curr)}")


if __name__ == "__main__":
    unittest.main()
