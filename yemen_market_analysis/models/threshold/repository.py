"""
Threshold model repository for Yemen Market Analysis.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from core.decorators import error_handler, performance_tracker
from core.exceptions import ThresholdModelError
from .tracker import ThresholdHistoryTracker

logger = logging.getLogger(__name__)


class ThresholdRepository:
    """Repository for threshold model results and history."""
    
    def __init__(self, base_path: str):
        """
        Initialize the threshold repository.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = base_path
        self.history_path = os.path.join(base_path, "threshold_history")
        self.results_path = os.path.join(base_path, "model_results")
        self.tracker = ThresholdHistoryTracker(
            os.path.join(self.history_path, "commodity_thresholds.json")
        )
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        os.makedirs(self.history_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
    
    @error_handler
    def add_result(
        self, 
        commodity: str, 
        model_results: Dict[str, Any],
        conflict_data: Optional[pd.Series] = None
    ) -> None:
        """
        Add model results and update threshold history.
        
        Args:
            commodity: Commodity name
            model_results: Threshold model results
            conflict_data: Optional conflict intensity data
        """
        # Extract key threshold values
        threshold = model_results.get('threshold')
        p_value = model_results.get('p_value')
        significant = model_results.get('threshold_significant', False)
        
        if threshold is None:
            logger.warning(f"No threshold value in model results for {commodity}")
            return
            
        # Add to history
        metadata = {
            'date': datetime.now().isoformat(),
            'p_value': p_value,
            'significant': significant,
            'model_type': model_results.get('model_type', 'unknown')
        }
        
        # If conflict data available, add average to metadata
        if conflict_data is not None and not conflict_data.empty:
            metadata['avg_conflict'] = float(conflict_data.mean())
            
        self.tracker.add_threshold(commodity, threshold, metadata)
        
        # Save model results for this run
        self._save_model_results(commodity, model_results)
    
    @error_handler
    def _save_model_results(self, commodity: str, results: Dict[str, Any]) -> None:
        """
        Save model results to disk.
        
        Args:
            commodity: Commodity name
            results: Model results
        """
        # Sanitize commodity name for filename
        safe_commodity = commodity.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{timestamp}_{safe_commodity}.json"
        filepath = os.path.join(self.results_path, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serialize)
            
        logger.debug(f"Saved model results for {commodity} to {filepath}")
    
    def _json_serialize(self, obj: Any) -> Any:
        """Custom JSON serializer for complex types."""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return str(obj)
    
    @error_handler(fallback_value=(None, None))
    def get_threshold_range(
        self, 
        commodity: str,
        conflict_data: Optional[pd.Series] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get recommended threshold range for a commodity.
        
        Args:
            commodity: Commodity name
            conflict_data: Optional conflict intensity data
            
        Returns:
            Tuple of (lower_bound, upper_bound) or (None, None) if no history
        """
        return self.tracker.get_threshold_range(commodity, conflict_data)
    
    @error_handler(fallback_value=[])
    def get_commodity_history(self, commodity: str) -> List[Dict[str, Any]]:
        """
        Get historical thresholds for a commodity.
        
        Args:
            commodity: Commodity name
            
        Returns:
            List of historical threshold records
        """
        if commodity not in self.tracker.history or not self.tracker.history[commodity]:
            return []
            
        history = []
        thresholds = self.tracker.history[commodity]
        metadata = self.tracker.metadata[commodity]
        
        for i, threshold in enumerate(thresholds):
            entry = {'threshold': threshold}
            
            # Add metadata if available
            if i < len(metadata):
                entry.update(metadata[i])
                
            history.append(entry)
            
        return history
    
    @error_handler(fallback_value={})
    def get_threshold_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all tracked thresholds.
        
        Returns:
            Dictionary with threshold statistics
        """
        return self.tracker.get_statistics()