"""
Threshold history tracking functionality for Yemen Market Analysis.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

from core.decorators import error_handler, performance_tracker
from core.exceptions import ThresholdModelError

logger = logging.getLogger(__name__)


class ThresholdHistoryTracker:
    """Maintains a history of threshold values by commodity to improve future threshold estimation."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the threshold history tracker.
        
        Args:
            storage_path: Path to storage file
        """
        self.history = {}  # Dict mapping commodity -> List of threshold values
        self.metadata = {}  # Dict mapping commodity -> List of metadata dicts
        self.storage_path = storage_path
        
        # Try to load existing history if storage path provided
        if storage_path and os.path.exists(storage_path):
            self.load()
    
    @error_handler
    def add_threshold(
        self, 
        commodity: str, 
        threshold: float, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new threshold value to the history for a commodity.
        
        Args:
            commodity: Commodity name
            threshold: Threshold value
            metadata: Optional metadata
        """
        if not np.isfinite(threshold):
            logger.warning(f"Attempted to add non-finite threshold ({threshold}) for {commodity}")
            return
            
        if commodity not in self.history:
            self.history[commodity] = []
            self.metadata[commodity] = []
        
        # Prepare metadata with timestamp
        entry_metadata = metadata or {}
        if 'timestamp' not in entry_metadata:
            entry_metadata['timestamp'] = datetime.now().isoformat()
        
        if 'date' not in entry_metadata:
            entry_metadata['date'] = datetime.now().isoformat()
            
        # Add to history
        self.history[commodity].append(threshold)
        self.metadata[commodity].append(entry_metadata)
        
        # Save updated history
        if self.storage_path:
            self.save()
    
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
        if commodity not in self.history or not self.history[commodity]:
            return None, None
        
        # With sufficient history, provide a range based on historical values
        if len(self.history[commodity]) >= 3:
            return self.compute_robust_threshold_range(commodity, None, conflict_data)
        
        # With limited history, provide a wider range around the mean
        thresholds = np.array(self.history[commodity])
        threshold_mean = np.mean(thresholds)
        threshold_std = np.std(thresholds) if len(thresholds) > 1 else threshold_mean * 0.2
        
        # Wider range with limited history
        lower = max(0, threshold_mean - 2 * threshold_std)
        upper = threshold_mean + 2 * threshold_std
        
        return lower, upper
    
    @error_handler(fallback_value=(0.05, 0.25))
    def compute_robust_threshold_range(
        self, 
        commodity: str, 
        new_residuals: Optional[np.ndarray] = None,
        conflict_data: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Compute robust threshold range using historical data and new residuals.
        
        Args:
            commodity: Commodity name
            new_residuals: Optional new residuals to consider
            conflict_data: Optional conflict intensity data
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if commodity not in self.history or not self.history[commodity]:
            # Default range if no history
            return 0.05, 0.25
        
        # Get historical thresholds
        history = self.history[commodity]
        
        # Calculate average conflict intensity if provided
        avg_conflict = 0.0
        if conflict_data is not None and len(conflict_data) > 0:
            # Normalize to 0-1 scale
            conflict_clean = conflict_data[~np.isnan(conflict_data)]
            if len(conflict_clean) > 0:
                conflict_min = np.min(conflict_clean)
                conflict_max = np.max(conflict_clean)
                if conflict_max > conflict_min:
                    normalized = (conflict_clean - conflict_min) / (conflict_max - conflict_min)
                    avg_conflict = np.mean(normalized)
        
        # With sufficient history, use bootstrapped confidence intervals
        if len(history) >= 5:
            hist_array = np.array(history)
            bootstrap_samples = np.random.choice(hist_array, size=(1000, len(hist_array)), replace=True)
            bootstrap_means = np.mean(bootstrap_samples, axis=1)
            
            # Calculate confidence intervals
            lower = np.percentile(bootstrap_means, 5)
            upper = np.percentile(bootstrap_means, 95)
            
            # Apply conflict adjustment if available
            if avg_conflict > 0:
                range_width = upper - lower
                conflict_adjustment = avg_conflict * 0.5  # Up to 50% wider
                lower = max(0, lower - range_width * conflict_adjustment / 2)
                upper = upper + range_width * conflict_adjustment / 2
                
            return lower, upper
        
        # With limited history, use mean and std with wider bounds
        threshold_mean = np.mean(history)
        threshold_std = np.std(history) if len(history) > 1 else threshold_mean * 0.2
        
        # Apply conflict adjustment
        conflict_factor = 1.0 + (avg_conflict * 0.5)  # Up to 50% wider
        
        lower = max(0, threshold_mean - 2 * threshold_std * conflict_factor)
        upper = threshold_mean + 2 * threshold_std * conflict_factor
        
        return lower, upper
    
    @error_handler
    def save(self) -> None:
        """Save threshold history to disk."""
        if not self.storage_path:
            logger.warning("No storage path specified for threshold history")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Convert to serializable format
        data = {
            'history': self.history,
            'metadata': self.metadata,
            'last_updated': datetime.now().isoformat()
        }
        
        # Save to file
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.debug(f"Saved threshold history to {self.storage_path}")
    
    @error_handler
    def load(self) -> None:
        """Load threshold history from disk."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            logger.warning("Threshold history file not found")
            return
            
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            self.history = data.get('history', {})
            self.metadata = data.get('metadata', {})
            
            logger.debug(f"Loaded threshold history from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load threshold history: {str(e)}")
    
    @error_handler(fallback_value={})
    def get_statistics(self, commodity: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about threshold history.
        
        Args:
            commodity: Optional commodity to get statistics for
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if commodity is not None:
            if commodity not in self.history:
                return {'commodity': commodity, 'count': 0}
                
            thresholds = np.array(self.history[commodity])
            stats = {
                'commodity': commodity,
                'count': len(thresholds),
                'mean': float(np.mean(thresholds)),
                'std': float(np.std(thresholds)) if len(thresholds) > 1 else 0.0,
                'min': float(np.min(thresholds)) if len(thresholds) > 0 else 0.0,
                'max': float(np.max(thresholds)) if len(thresholds) > 0 else 0.0,
                'last': float(thresholds[-1]) if len(thresholds) > 0 else 0.0
            }
        else:
            # Overall statistics
            stats = {
                'commodities': len(self.history),
                'total_entries': sum(len(v) for v in self.history.values()),
                'commodity_stats': {}
            }
            
            for comm in self.history:
                thresholds = np.array(self.history[comm])
                stats['commodity_stats'][comm] = {
                    'count': len(thresholds),
                    'mean': float(np.mean(thresholds)) if len(thresholds) > 0 else 0.0,
                    'last': float(thresholds[-1]) if len(thresholds) > 0 else 0.0
                }
        
        return stats