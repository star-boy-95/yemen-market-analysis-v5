"""
Main cointegration tester module for Yemen Market Analysis.

This module provides the main CointegrationTester class that integrates
all the individual cointegration test methods.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.models.cointegration.engle_granger import EngleGrangerTester
from src.models.cointegration.johansen import JohansenTester
from src.models.cointegration.gregory_hansen import GregoryHansenTester
from src.models.cointegration.error_correction import ErrorCorrectionModel

# Initialize logger
logger = logging.getLogger(__name__)

class CointegrationTester:
    """
    Main cointegration tester for Yemen Market Analysis.
    
    This class integrates all the individual cointegration test methods and provides
    a unified interface for running cointegration tests.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        eg_tester (EngleGrangerTester): Engle-Granger test implementation.
        johansen_tester (JohansenTester): Johansen test implementation.
        gh_tester (GregoryHansenTester): Gregory-Hansen test implementation.
        ecm (ErrorCorrectionModel): Error correction model implementation.
    """
    
    def __init__(self, alpha: float = None, max_lags: int = None):
        """
        Initialize the cointegration tester.
        
        Args:
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.alpha = alpha if alpha is not None else config.get('analysis.cointegration.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.cointegration.max_lags', 4)
        
        # Initialize individual testers
        self.eg_tester = EngleGrangerTester(alpha=self.alpha, max_lags=self.max_lags)
        self.johansen_tester = JohansenTester(alpha=self.alpha, max_lags=self.max_lags)
        self.gh_tester = GregoryHansenTester(alpha=self.alpha, max_lags=self.max_lags)
        self.ecm = ErrorCorrectionModel(alpha=self.alpha, max_lags=self.max_lags)
    
    @handle_errors
    def run_all_tests(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        trend: str = 'c', max_lags: Optional[int] = None, include_johansen: bool = True,
        include_gh: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all cointegration tests.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            trend: Trend to include in the tests.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            include_johansen: Whether to include the Johansen test.
            include_gh: Whether to include the Gregory-Hansen test.
            
        Returns:
            Dictionary mapping test names to test results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or any of the tests fail.
        """
        logger.info(f"Running all cointegration tests for {y_col} and {x_col}")
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
        
        # Run tests
        results = {}
        
        # Engle-Granger test
        eg_results = self.eg_tester.test(y, x, y_col, x_col, trend, max_lags)
        results['Engle-Granger'] = eg_results
        
        # Johansen test
        if include_johansen:
            # Combine data for Johansen test
            combined_data = pd.DataFrame({
                'y': y[y_col],
                'x': x[x_col]
            })
            
            # Determine deterministic order based on trend
            det_order = 0 if trend == 'n' else 1 if trend == 'c' else 2
            
            johansen_results = self.johansen_tester.test(combined_data, ['y', 'x'], det_order, 1, max_lags)
            results['Johansen'] = johansen_results
        
        # Gregory-Hansen test
        if include_gh:
            gh_results = self.gh_tester.test(y, x, y_col, x_col, trend, max_lags)
            results['Gregory-Hansen'] = gh_results
        
        # Determine overall cointegration
        is_cointegrated = results['Engle-Granger']['is_cointegrated']
        
        if include_johansen:
            is_cointegrated = is_cointegrated or results['Johansen']['is_cointegrated']
        
        if include_gh:
            is_cointegrated = is_cointegrated or results['Gregory-Hansen']['is_cointegrated']
        
        # Add overall result
        results['overall'] = {
            'is_cointegrated': is_cointegrated,
            'alpha': self.alpha,
        }
        
        logger.info(f"Overall cointegration result: {is_cointegrated}")
        return results
    
    @handle_errors
    def estimate_error_correction_model(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        max_lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Estimate an error correction model (ECM) for cointegrated series.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            max_lags: Maximum number of lags to consider. If None, uses the value
                     from the class.
            
        Returns:
            Dictionary containing the ECM results.
            
        Raises:
            YemenAnalysisError: If the columns are not found or the estimation fails.
        """
        logger.info(f"Estimating error correction model for {y_col} and {x_col}")
        
        # Set max_lags
        if max_lags is None:
            max_lags = self.max_lags
        
        # Estimate ECM
        return self.ecm.estimate(y, x, y_col, x_col, max_lags)
