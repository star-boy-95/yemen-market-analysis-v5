"""
Output file management for Yemen Market Analysis.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from core.decorators import error_handler, performance_tracker
from core.exceptions import ReportingError
from core.config import config

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


class OutputManager:
    """Manager for organizing and saving analysis outputs."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        module_name: str = 'unified_threshold_ecm',
        version: Optional[str] = None,
        analysis_name: str = 'analysis_all'
    ):
        """Initialize output manager with directory structure."""
        # Set base output directory
        self.base_dir = output_dir or config.get('directories.results_dir', 'results')
        
        # Set module name and version
        self.module_name = module_name
        self.version = version or config.get('model.version', 'v3.3')
        if not self.version.startswith('v'):
            self.version = f"v{self.version}"
        
        # Set analysis name
        self.analysis_name = analysis_name
        
        # Create timestamp for this analysis run
        self.timestamp = datetime.now().strftime("%Y%m%d")
        
        # Setup directory structure
        self.setup_directories()
        
        # Initialize manifest
        self.manifest = {
            "timestamp": self.timestamp,
            "module": self.module_name,
            "version": self.version,
            "analysis": self.analysis_name,
            "files": []
        }
    
    def setup_directories(self) -> None:
        """Create directory structure for outputs."""
        # Main module directory
        self.module_dir = os.path.join(self.base_dir, self.module_name)
        os.makedirs(self.module_dir, exist_ok=True)
        
        # Version directory
        self.version_dir = os.path.join(self.module_dir, self.version)
        os.makedirs(self.version_dir, exist_ok=True)
        
        # Analysis directory
        self.analysis_dir = os.path.join(self.version_dir, self.analysis_name)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Visualizations directory
        self.viz_dir = os.path.join(self.analysis_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        logger.info(f"Set up output directories at {self.analysis_dir}")
    
    def get_commodity_viz_dir(self, commodity: str) -> str:
        """Get directory for commodity-specific visualizations."""
        commodity_dir = os.path.join(self.viz_dir, commodity.replace(' ', '_').lower())
        os.makedirs(commodity_dir, exist_ok=True)
        return commodity_dir
    
    def get_policy_viz_dir(self) -> str:
        """Get directory for policy-related visualizations."""
        policy_dir = os.path.join(self.viz_dir, "policy")
        os.makedirs(policy_dir, exist_ok=True)
        return policy_dir
    
    @error_handler(fallback_value=False)
    def save_json(
        self, 
        data: Dict[str, Any], 
        commodity: Optional[str] = None,
        filename: Optional[str] = None
    ) -> bool:
        """Save data as JSON file."""
        # Create filename if not provided
        if filename is None:
            if commodity:
                filename = f"{self.timestamp}_{commodity.replace(' ', '_').lower()}.json"
            else:
                filename = f"{self.timestamp}.json"
        
        # Full file path
        file_path = os.path.join(self.analysis_dir, filename)
        
        # Save file
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
            
            # Add to manifest
            self.manifest["files"].append({
                "path": os.path.basename(file_path),
                "type": "json",
                "commodity": commodity,
                "timestamp": self.timestamp
            })
            
            logger.info(f"Saved JSON to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {str(e)}")
            return False
    
    @error_handler(fallback_value=False)
    def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Save metadata about the analysis run."""
        filename = f"{self.timestamp}.metadata.json"
        file_path = os.path.join(self.analysis_dir, filename)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(metadata, f, cls=NumpyEncoder, indent=2)
            
            # Add to manifest
            self.manifest["files"].append({
                "path": os.path.basename(file_path),
                "type": "metadata",
                "timestamp": self.timestamp
            })
            
            logger.info(f"Saved metadata to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving metadata to {file_path}: {str(e)}")
            return False
    
    @error_handler(fallback_value=False)
    def save_manifest(self) -> bool:
        """Save manifest file with listing of all outputs."""
        file_path = os.path.join(self.analysis_dir, "manifest.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.manifest, f, cls=NumpyEncoder, indent=2)
            
            logger.info(f"Saved manifest to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving manifest to {file_path}: {str(e)}")
            return False
    
    @error_handler(fallback_value=False)
    def save_csv(
        self, 
        df: pd.DataFrame, 
        filename: str
    ) -> bool:
        """Save DataFrame as CSV file."""
        file_path = os.path.join(self.analysis_dir, filename)
        
        try:
            df.to_csv(file_path, index=True)
            
            # Add to manifest
            self.manifest["files"].append({
                "path": os.path.basename(file_path),
                "type": "csv",
                "timestamp": self.timestamp
            })
            
            logger.info(f"Saved CSV to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving CSV to {file_path}: {str(e)}")
            return False
    
    @error_handler(fallback_value=False)
    def save_html(
        self, 
        html_content: str, 
        filename: str
    ) -> bool:
        """Save HTML content to file."""
        file_path = os.path.join(self.analysis_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Add to manifest
            self.manifest["files"].append({
                "path": os.path.basename(file_path),
                "type": "html",
                "timestamp": self.timestamp
            })
            
            logger.info(f"Saved HTML to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving HTML to {file_path}: {str(e)}")
            return False
    
    @error_handler(fallback_value=None)
    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load JSON data from file."""
        file_path = os.path.join(self.analysis_dir, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading JSON from {file_path}: {str(e)}")
            return None
    
    @error_handler(fallback_value=None)
    def load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load CSV data from file."""
        file_path = os.path.join(self.analysis_dir, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            logger.error(f"Error loading CSV from {file_path}: {str(e)}")
            return None