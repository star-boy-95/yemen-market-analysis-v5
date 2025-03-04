#!/usr/bin/env python
"""
cointegration.py

Cointegration testing for Yemen Market Analysis project.
"""

import numpy as np
import pandas as pd
import logging
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from typing import Dict, List, Tuple, Optional, Union, Any

from yemen_market_analysis.core.decorators import smart_error_handler, performance_tracker

logger = logging.getLogger(__name__)

# Type alias
ArrayLike = Union[np.ndarray, List[float], pd.Series]

@smart_error_handler(fallback_value={"cointegrated": False, "status": "error"})
def test_cointegration(
    y: ArrayLike,
    x: ArrayLike,
    trend: str = 'c',
    method: str = 'aeg',
    max_lags: Optional[int] = None,
    return_diagnostics: bool = False
) -> Dict[str, Any]:
    """
    Test for cointegration between two time series using Engle-Granger method.
    
    This implements the Engle-Granger two-step approach:
    1. Estimate the long-run cointegrating relationship
    2. Test residuals for stationarity (unit root test)
    
    Args:
        y: First time series
        x: Second time series
        trend: Trend specification for ADF test ('c' for constant, 'ct' for constant and trend)
        method: Method to use ('aeg' for Augmented Engle-Granger)
        max_lags: Maximum lags for ADF test (None for automatic selection)
        return_diagnostics: Whether to return diagnostic information
        
    Returns:
        Dictionary with cointegration test results
    """
    y, x = pd.Series(y).dropna(), pd.Series(x).dropna()
    
    # Align series to common index if Series
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        y, x = y.align(x, join='inner')
    
    if len(y) < 20:
        return {"cointegrated": False, "status": "insufficient_data"}

    # Calculate cointegrating vector using OLS
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    beta = model.params[1]
    intercept = model.params[0]
    residuals = model.resid
    
    # Test residuals for stationarity (ADF test)
    if method.lower() == 'aeg':
        result = adfuller(residuals, regression=trend, maxlag=max_lags, autolag='AIC')
        cointegrated = result[1] < 0.05  # p-value < 0.05 means stationarity of residuals
        
        response = {
            "cointegrated": cointegrated,
            "p_value": float(result[1]),
            "t_stat": float(result[0]),
            "interpretation": "Cointegrated" if cointegrated else "Not cointegrated",
            "status": "success"
        }
    elif method.lower() == 'johansen':
        # Use statsmodels coint function (Phillips-Ouliaris test)
        # This is a placeholder for Johansen; actual Johansen would require VECM
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        # Need to stack y and x into a matrix
        data = np.column_stack([y, x])
        
        # For now, use Phillips-Ouliaris as fallback
        try:
            _, pvalue, _ = coint(y, x, trend=trend)
            cointegrated = pvalue < 0.05
            
            response = {
                "cointegrated": cointegrated,
                "p_value": float(pvalue),
                "interpretation": "Cointegrated" if cointegrated else "Not cointegrated",
                "status": "success",
                "note": "Phillips-Ouliaris test used as fallback for Johansen"
            }
        except Exception as e:
            logger.error(f"Cointegration test failed: {e}")
            return {"cointegrated": False, "status": "error", "error": str(e)}
    else:
        return {"cointegrated": False, "status": "error", "error": f"Unknown method: {method}"}
    
    # Add diagnostics if requested
    if return_diagnostics:
        response["diagnostics"] = {
            "beta": float(beta),
            "intercept": float(intercept),
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std()),
            "critical_values": result[4] if method.lower() == 'aeg' else None,
            "residuals": residuals.tolist()
        }
        
    return response

class CointegrationAnalysis:
    """Comprehensive cointegration analysis with multiple tests."""
    
    def __init__(self, y: ArrayLike, x: ArrayLike):
        """
        Initialize with two time series.
        
        Args:
            y: First time series
            x: Second time series
        """
        self.y, self.x = pd.Series(y).dropna(), pd.Series(x).dropna()
        
        # Align series if they have different indices
        if isinstance(self.y, pd.Series) and isinstance(self.x, pd.Series):
            self.y, self.x = self.y.align(self.x, join='inner')
            
        # Store properties
        self.n = len(self.y)
        self.is_valid = self.n >= 20
        
        # Initialize results
        self.results = {}
        
    @performance_tracker(level="debug")
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all available cointegration tests.
        
        Returns:
            Dictionary with test results
        """
        if not self.is_valid:
            return {"error": f"Insufficient observations: {self.n} < 20"}
        
        # Run Engle-Granger test
        eg_result = self.run_engle_granger()
        self.results["engle_granger"] = eg_result
        
        # Run Phillips-Ouliaris test
        po_result = self.run_phillips_ouliaris()
        self.results["phillips_ouliaris"] = po_result
        
        # Try to run Johansen test if possible
        try:
            johansen_result = self.run_johansen()
            self.results["johansen"] = johansen_result
        except Exception as e:
            logger.warning(f"Johansen test failed: {e}")
            self.results["johansen"] = {"error": str(e), "status": "error"}
        
        # Determine consensus
        self.results["consensus"] = self._determine_consensus()
        
        return self.results
    
    @smart_error_handler(fallback_value={"status": "error"})
    def run_engle_granger(self) -> Dict[str, Any]:
        """
        Run Engle-Granger test for cointegration.
        
        Returns:
            Dictionary with test results
        """
        if not self.is_valid:
            return {"error": f"Insufficient observations: {self.n} < 20"}
        
        # Estimate cointegrating relationship
        X = sm.add_constant(self.x)
        model = sm.OLS(self.y, X).fit()
        residuals = model.resid
        
        # Test residuals for stationarity
        adf_result = adfuller(residuals, regression='c', autolag='AIC')
        
        # Determine if cointegrated (residuals are stationary)
        p_value = adf_result[1]
        cointegrated = p_value < 0.05
        
        return {
            "cointegrated": cointegrated,
            "p_value": float(p_value),
            "t_stat": float(adf_result[0]),
            "critical_values": {k: float(v) for k, v in adf_result[4].items()},
            "params": {
                "beta": float(model.params[1]),
                "intercept": float(model.params[0])
            },
            "interpretation": "Cointegrated" if cointegrated else "Not cointegrated",
            "status": "success"
        }
    
    @smart_error_handler(fallback_value={"status": "error"})
    def run_phillips_ouliaris(self) -> Dict[str, Any]:
        """
        Run Phillips-Ouliaris test for cointegration.
        
        Returns:
            Dictionary with test results
        """
        if not self.is_valid:
            return {"error": f"Insufficient observations: {self.n} < 20"}
        
        # Use statsmodels coint function
        t_stat, p_value, critical_values = coint(self.y, self.x, trend='c')
        
        # Determine if cointegrated
        cointegrated = p_value < 0.05
        
        return {
            "cointegrated": cointegrated,
            "p_value": float(p_value),
            "t_stat": float(t_stat),
            "critical_values": {f"{int(key*100)}%": float(value) for key, value in zip([0.01, 0.05, 0.1], critical_values)},
            "interpretation": "Cointegrated" if cointegrated else "Not cointegrated",
            "status": "success"
        }
    
    @smart_error_handler(fallback_value={"status": "error"})
    def run_johansen(self) -> Dict[str, Any]:
        """
        Run Johansen test for cointegration.
        
        Returns:
            Dictionary with test results
        """
        if not self.is_valid:
            return {"error": f"Insufficient observations: {self.n} < 20"}
        
        # Need to import Johansen test
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        # Stack data into a matrix (columns are variables)
        data = np.column_stack([self.y, self.x])
        
        # Run Johansen test
        # det_order: 0=no deterministic terms, 1=constant, 2=constant and trend
        johansen = coint_johansen(data, det_order=1, k_ar_diff=1)
        
        # Extract statistics
        trace_stat = johansen.lr1[0]  # Trace statistic for H0: r=0
        max_stat = johansen.lr2[0]    # Max eigenvalue statistic for H0: r=0
        
        # Critical values
        cv_trace = johansen.cvt[:, 0]  # 90%, 95%, 99% for r=0
        cv_max = johansen.cvm[:, 0]    # 90%, 95%, 99% for r=0
        
        # Determine if cointegrated
        trace_cointegrated = trace_stat > cv_trace[1]  # Compare with 95% critical value
        max_cointegrated = max_stat > cv_max[1]        # Compare with 95% critical value
        
        # Cointegrated if either test indicates cointegration
        cointegrated = trace_cointegrated or max_cointegrated
        
        return {
            "cointegrated": cointegrated,
            "trace": {
                "statistic": float(trace_stat),
                "critical_values": {f"{int(100-(i+1)*10)}%": float(cv_trace[i]) for i in range(3)},
                "reject_null": bool(trace_cointegrated)
            },
            "max_eigenvalue": {
                "statistic": float(max_stat),
                "critical_values": {f"{int(100-(i+1)*10)}%": float(cv_max[i]) for i in range(3)},
                "reject_null": bool(max_cointegrated)
            },
            "interpretation": "Cointegrated" if cointegrated else "Not cointegrated",
            "status": "success"
        }
    
    def _determine_consensus(self) -> Dict[str, Any]:
        """
        Determine consensus from multiple cointegration tests.
        
        Returns:
            Dictionary with consensus result
        """
        if not self.results:
            return {"error": "No test results available"}
        
        # Count number of tests indicating cointegration
        tests = [
            result.get("cointegrated", False) 
            for name, result in self.results.items() 
            if name not in ["consensus"] and "status" in result and result["status"] == "success"
        ]
        
        if not tests:
            return {"error": "No valid test results available"}
        
        # Calculate proportion of tests indicating cointegration
        cointegration_pct = sum(tests) / len(tests)
        
        # Determine consensus
        if cointegration_pct >= 0.67:
            consensus = "Strong evidence of cointegration"
            cointegrated = True
        elif cointegration_pct >= 0.5:
            consensus = "Moderate evidence of cointegration"
            cointegrated = True
        elif cointegration_pct >= 0.33:
            consensus = "Weak evidence of cointegration"
            cointegrated = False
        else:
            consensus = "No evidence of cointegration"
            cointegrated = False
        
        return {
            "cointegrated": cointegrated,
            "consensus": consensus,
            "agreement_pct": float(cointegration_pct),
            "tests_cointegrated": int(sum(tests)),
            "tests_total": len(tests)
        }
        
@performance_tracker(level="info")
@smart_error_handler(fallback_value={"has_structural_break": None, "status": "error"})
def test_structural_breaks(
    series: ArrayLike,
    min_periods: int = 20,
    significance: float = 0.05
) -> Dict[str, Any]:
    """
    Test for structural breaks in cointegrating relationship.
    
    Args:
        series: Time series (usually residuals from cointegrating regression)
        min_periods: Minimum periods in each segment
        significance: Significance level for break detection
        
    Returns:
        Dictionary with structural break test results
    """
    series = pd.Series(series).dropna()
    
    if len(series) < min_periods * 2:
        return {
            "has_structural_break": None, 
            "status": "insufficient_data",
            "message": f"Need at least {min_periods * 2} observations, got {len(series)}"
        }
    
    try:
        # Import break tests
        from statsmodels.stats.diagnostic import breaks_cusumolsresid
        from statsmodels.tsa.stattools import adfuller
        
        # Run CUSUM test
        cusum = breaks_cusumolsresid(series.values, ddof=0)
        
        # Determine if structural break exists
        # Critical value for CUSUM test
        crit = 0.948  # Standard critical value
        break_detected = np.any(np.abs(cusum) > crit)
        
        # Try Zivot-Andrews test if available
        try:
            from statsmodels.tsa.stattools import zivot_andrews
            za_result = zivot_andrews(series.values, trim=0.15, maxlag=None)
            za_min_idx = np.argmin(za_result[0])
            za_min_pvalue = za_result[1][za_min_idx]
            za_break_idx = za_min_idx + int(len(series) * 0.15)
            za_break_date = series.index[za_break_idx] if isinstance(series.index, pd.DatetimeIndex) else za_break_idx
            
            # Check if break is significant
            za_break_detected = za_min_pvalue < significance
            
            # Run ADF test on segments
            if za_break_detected and za_break_idx > min_periods and len(series) - za_break_idx > min_periods:
                segment1 = series.iloc[:za_break_idx]
                segment2 = series.iloc[za_break_idx:]
                
                adf1 = adfuller(segment1.values, regression='c', autolag='AIC')
                adf2 = adfuller(segment2.values, regression='c', autolag='AIC')
                
                segment1_stationary = adf1[1] < significance
                segment2_stationary = adf2[1] < significance
                
                # Add segment specifics to results
                segments = [
                    {
                        "start_idx": 0,
                        "end_idx": za_break_idx,
                        "stationary": segment1_stationary,
                        "adf_pvalue": float(adf1[1])
                    },
                    {
                        "start_idx": za_break_idx,
                        "end_idx": len(series),
                        "stationary": segment2_stationary,
                        "adf_pvalue": float(adf2[1])
                    }
                ]
                
                # Add dates if available
                if isinstance(series.index, pd.DatetimeIndex):
                    segments[0]["start_date"] = series.index[0].strftime("%Y-%m-%d")
                    segments[0]["end_date"] = series.index[za_break_idx-1].strftime("%Y-%m-%d")
                    segments[1]["start_date"] = series.index[za_break_idx].strftime("%Y-%m-%d")
                    segments[1]["end_date"] = series.index[-1].strftime("%Y-%m-%d")
            else:
                segments = []
        except ImportError:
            # Zivot-Andrews test not available
            za_break_detected = None
            za_break_date = None
            segments = []
        
        return {
            "has_structural_break": break_detected or za_break_detected,
            "cusum_test": {
                "break_detected": bool(break_detected),
                "max_cusum": float(np.max(np.abs(cusum))),
                "critical_value": float(crit)
            },
            "zivot_andrews_test": {
                "break_detected": bool(za_break_detected) if za_break_detected is not None else None,
                "break_date": za_break_date if za_break_date is not None else None,
                "min_pvalue": float(za_min_pvalue) if 'za_min_pvalue' in locals() else None
            },
            "segments": segments,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Structural break test failed: {e}")
        return {"has_structural_break": None, "status": "error", "error": str(e)}