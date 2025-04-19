"""
Main threshold model module for Yemen Market Analysis.

This module provides the main ThresholdModel class that integrates
all the individual threshold model implementations.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.models.threshold.tar import ThresholdAutoregressive
from src.models.threshold.mtar import MomentumThresholdAutoregressive
from src.models.threshold.tvecm import ThresholdVECM

# Initialize logger
logger = logging.getLogger(__name__)

class ThresholdModel:
    """
    Main threshold model for Yemen Market Analysis.
    
    This class integrates all the individual threshold model implementations and provides
    a unified interface for estimating threshold models.
    
    Attributes:
        y (pd.DataFrame): DataFrame containing the dependent variable.
        x (pd.DataFrame): DataFrame containing the independent variable.
        y_col (str): Column name for the dependent variable.
        x_col (str): Column name for the independent variable.
        mode (str): Mode for the threshold model. Options are 'standard' (TAR),
                   'fixed' (TAR with fixed threshold), 'mtar' (M-TAR), and 'tvecm' (TVECM).
        alpha (float): Significance level for hypothesis tests.
        max_lags (int): Maximum number of lags to consider in tests.
        tar (ThresholdAutoregressive): TAR model implementation.
        mtar (MomentumThresholdAutoregressive): M-TAR model implementation.
        tvecm (ThresholdVECM): TVECM model implementation.
    """
    
    def __init__(
        self, y: pd.DataFrame, x: pd.DataFrame, y_col: str = 'price', x_col: str = 'price',
        mode: str = 'standard', alpha: float = None, max_lags: int = None
    ):
        """
        Initialize the threshold model.
        
        Args:
            y: DataFrame containing the dependent variable.
            x: DataFrame containing the independent variable.
            y_col: Column name for the dependent variable.
            x_col: Column name for the independent variable.
            mode: Mode for the threshold model. Options are 'standard' (TAR),
                 'fixed' (TAR with fixed threshold), 'mtar' (M-TAR), and 'tvecm' (TVECM).
            alpha: Significance level for hypothesis tests. If None, uses the value
                  from config.
            max_lags: Maximum number of lags to consider in tests. If None, uses the
                     value from config.
        """
        self.y = y
        self.x = x
        self.y_col = y_col
        self.x_col = x_col
        self.mode = mode
        self.alpha = alpha if alpha is not None else config.get('analysis.threshold.alpha', 0.05)
        self.max_lags = max_lags if max_lags is not None else config.get('analysis.threshold.max_lags', 4)
        
        # Initialize individual models
        self.tar = ThresholdAutoregressive(alpha=self.alpha, max_lags=self.max_lags)
        self.mtar = MomentumThresholdAutoregressive(alpha=self.alpha, max_lags=self.max_lags)
        self.tvecm = ThresholdVECM(alpha=self.alpha, max_lags=self.max_lags)
    
    @handle_errors
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run a full threshold model analysis.
        
        This method runs the appropriate threshold model based on the mode specified
        during initialization.
        
        Returns:
            Dictionary containing the analysis results.
            
        Raises:
            YemenAnalysisError: If the analysis fails.
        """
        logger.info(f"Running full threshold model analysis with mode={self.mode}")
        
        try:
            if self.mode == 'standard':
                # Run standard TAR model
                results = self.tar.estimate(self.y, self.x, self.y_col, self.x_col)
            elif self.mode == 'fixed':
                # Run TAR model with fixed threshold (0)
                threshold = config.get('analysis.threshold.mtar_default_threshold', 0.0)
                results = self.tar.estimate(self.y, self.x, self.y_col, self.x_col, fixed_threshold=threshold)
            elif self.mode == 'mtar':
                # Run M-TAR model
                results = self.mtar.estimate(self.y, self.x, self.y_col, self.x_col)
            elif self.mode == 'tvecm':
                # Run TVECM model
                k_ar_diff = config.get('analysis.threshold_vecm.k_ar_diff', 2)
                deterministic = config.get('analysis.threshold_vecm.deterministic', 'ci')
                coint_rank = config.get('analysis.threshold_vecm.coint_rank', 1)
                
                results = self.tvecm.estimate(
                    self.y, self.x, self.y_col, self.x_col,
                    k_ar_diff=k_ar_diff, deterministic=deterministic, coint_rank=coint_rank
                )
            else:
                logger.error(f"Invalid threshold model mode: {self.mode}")
                raise YemenAnalysisError(f"Invalid threshold model mode: {self.mode}")
            
            logger.info(f"Threshold model analysis completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error running threshold model analysis: {e}")
            raise YemenAnalysisError(f"Error running threshold model analysis: {e}")
    
    @handle_errors
    def run(self) -> Dict[str, Any]:
        """
        Run the threshold model analysis.
        
        This is an alias for run_full_analysis() to maintain a consistent interface.
        
        Returns:
            Dictionary containing the analysis results.
            
        Raises:
            YemenAnalysisError: If the analysis fails.
        """
        return self.run_full_analysis()
    
    @handle_errors
    def compare_models(self) -> Dict[str, Any]:
        """
        Compare different threshold model specifications.
        
        This method estimates all available threshold models and compares them
        based on information criteria.
        
        Returns:
            Dictionary containing the comparison results.
            
        Raises:
            YemenAnalysisError: If the comparison fails.
        """
        logger.info("Comparing different threshold model specifications")
        
        try:
            # Estimate all models
            tar_results = self.tar.estimate(self.y, self.x, self.y_col, self.x_col)
            
            threshold = config.get('analysis.threshold.mtar_default_threshold', 0.0)
            tar_fixed_results = self.tar.estimate(self.y, self.x, self.y_col, self.x_col, fixed_threshold=threshold)
            
            mtar_results = self.mtar.estimate(self.y, self.x, self.y_col, self.x_col)
            
            k_ar_diff = config.get('analysis.threshold_vecm.k_ar_diff', 2)
            deterministic = config.get('analysis.threshold_vecm.deterministic', 'ci')
            coint_rank = config.get('analysis.threshold_vecm.coint_rank', 1)
            
            tvecm_results = self.tvecm.estimate(
                self.y, self.x, self.y_col, self.x_col,
                k_ar_diff=k_ar_diff, deterministic=deterministic, coint_rank=coint_rank
            )
            
            # Extract information criteria
            models = {
                'TAR': tar_results,
                'TAR (fixed)': tar_fixed_results,
                'M-TAR': mtar_results,
                'TVECM': tvecm_results,
            }
            
            aic_values = {name: results['aic'] for name, results in models.items()}
            bic_values = {name: results['bic'] for name, results in models.items()}
            
            # Find best model
            best_aic = min(aic_values, key=aic_values.get)
            best_bic = min(bic_values, key=bic_values.get)
            
            # Create comparison results
            comparison_results = {
                'models': models,
                'aic_values': aic_values,
                'bic_values': bic_values,
                'best_aic': best_aic,
                'best_bic': best_bic,
                'recommendation': best_bic,  # Use BIC as default recommendation
            }
            
            logger.info(f"Model comparison completed. Best model (AIC): {best_aic}, Best model (BIC): {best_bic}")
            return comparison_results
        except Exception as e:
            logger.error(f"Error comparing threshold models: {e}")
            raise YemenAnalysisError(f"Error comparing threshold models: {e}")
    
    @handle_errors
    def run_with_structural_breaks(
        self, break_dates: Optional[List[str]] = None,
        detect_breaks: bool = False
    ) -> Dict[str, Any]:
        """
        Run threshold model analysis with structural breaks.
        
        This method extends the standard threshold model analysis to account for
        structural breaks in the cointegrating relationship. It implements a
        Gregory-Hansen type approach where the model is estimated separately
        for different regimes defined by the structural breaks.
        
        Args:
            break_dates: List of known break dates as strings in the format matching
                       the index of the data. If None and detect_breaks is True,
                       breaks will be detected automatically using the Zivot-Andrews test.
            detect_breaks: Whether to automatically detect breaks if not provided.
                         Uses the Zivot-Andrews test which is particularly valuable
                         for conflict-affected market data.
        
        Returns:
            Dictionary containing results for each regime and overall results.
            
        Raises:
            YemenAnalysisError: If the analysis fails or if no breaks are provided
                              and detect_breaks is False.
        """
        logger.info(f"Running threshold model analysis with structural breaks")
        
        try:
            # Check if we need to detect breaks
            if break_dates is None and not detect_breaks:
                logger.error("No break dates provided and detect_breaks is False")
                raise YemenAnalysisError(
                    "Either provide break_dates or set detect_breaks=True"
                )
            
            # Detect breaks if needed
            if break_dates is None and detect_breaks:
                logger.info("Detecting structural breaks using Zivot-Andrews test")
                
                # Import UnitRootTester to use Zivot-Andrews test
                from src.models.unit_root import UnitRootTester
                
                # Initialize unit root tester
                unit_root_tester = UnitRootTester(alpha=self.alpha, max_lags=self.max_lags)
                
                # Run Zivot-Andrews test on y and x
                y_za_results = unit_root_tester.test_za(self.y, self.y_col)
                x_za_results = unit_root_tester.test_za(self.x, self.x_col)
                
                # Get break dates
                y_break_date = y_za_results.get('break_date')
                x_break_date = x_za_results.get('break_date')
                
                # Use both breaks if they exist and are different
                detected_breaks = []
                if y_break_date is not None:
                    detected_breaks.append(y_break_date)
                if x_break_date is not None and x_break_date != y_break_date:
                    detected_breaks.append(x_break_date)
                
                # Sort breaks chronologically
                detected_breaks.sort()
                
                if not detected_breaks:
                    logger.warning("No structural breaks detected")
                    # Run standard analysis if no breaks detected
                    return self.run_full_analysis()
                
                break_dates = detected_breaks
                logger.info(f"Detected structural breaks at: {break_dates}")
            
            # Convert break dates to pandas datetime if they're strings
            break_dates_dt = []
            for date_str in break_dates:
                try:
                    # Try to convert to datetime - handle different formats
                    if isinstance(date_str, str):
                        break_dates_dt.append(pd.to_datetime(date_str))
                    else:
                        # If it's already a datetime or timestamp, use it directly
                        break_dates_dt.append(date_str)
                except Exception as e:
                    logger.error(f"Error converting break date {date_str} to datetime: {e}")
                    raise YemenAnalysisError(f"Error converting break date {date_str} to datetime: {e}")
            
            # Sort break dates
            break_dates_dt.sort()
            
            # Create time periods including before first break and after last break
            periods = []
            
            # Get min and max dates from data
            min_date = min(self.y.index.min(), self.x.index.min())
            max_date = max(self.y.index.max(), self.x.index.max())
            
            # Add period before first break
            periods.append((min_date, break_dates_dt[0]))
            
            # Add periods between breaks
            for i in range(len(break_dates_dt) - 1):
                periods.append((break_dates_dt[i], break_dates_dt[i + 1]))
            
            # Add period after last break
            periods.append((break_dates_dt[-1], max_date))
            
            # Initialize results dictionary
            results = {
                'model_type': 'threshold_with_breaks',
                'break_dates': break_dates,
                'regimes': {},
                'overall': {},
            }
            
            # Run threshold model for each period
            for i, (start_date, end_date) in enumerate(periods):
                logger.info(f"Running threshold model for regime {i+1}: {start_date} to {end_date}")
                
                # Filter data for this period
                y_period = self.y[(self.y.index >= start_date) & (self.y.index <= end_date)]
                x_period = self.x[(self.x.index >= start_date) & (self.x.index <= end_date)]
                
                # Check if we have enough data
                min_obs = 10  # Minimum observations needed for threshold model
                if len(y_period) < min_obs or len(x_period) < min_obs:
                    logger.warning(f"Insufficient data for regime {i+1} ({len(y_period)} obs). Skipping.")
                    results['regimes'][f'regime_{i+1}'] = {
                        'start_date': start_date,
                        'end_date': end_date,
                        'n_observations': len(y_period),
                        'status': 'skipped',
                        'reason': 'insufficient_data',
                    }
                    continue
                
                # Create a new threshold model for this period
                period_model = ThresholdModel(
                    y=y_period,
                    x=x_period,
                    y_col=self.y_col,
                    x_col=self.x_col,
                    mode=self.mode,
                    alpha=self.alpha,
                    max_lags=self.max_lags
                )
                
                # Run the model
                try:
                    period_results = period_model.run_full_analysis()
                    
                    # Add period information to results
                    period_results['start_date'] = start_date
                    period_results['end_date'] = end_date
                    period_results['n_observations'] = len(y_period)
                    period_results['status'] = 'success'
                    
                    # Store results for this regime
                    results['regimes'][f'regime_{i+1}'] = period_results
                except Exception as e:
                    logger.warning(f"Error running threshold model for regime {i+1}: {e}")
                    results['regimes'][f'regime_{i+1}'] = {
                        'start_date': start_date,
                        'end_date': end_date,
                        'n_observations': len(y_period),
                        'status': 'error',
                        'error': str(e),
                    }
            
            # Calculate overall results
            successful_regimes = [r for r in results['regimes'].values() if r.get('status') == 'success']
            
            if not successful_regimes:
                logger.warning("No successful regime models")
                results['overall']['status'] = 'error'
                results['overall']['error'] = 'No successful regime models'
                return results
            
            # Aggregate information criteria across regimes
            aic_values = [r.get('aic', float('inf')) for r in successful_regimes]
            bic_values = [r.get('bic', float('inf')) for r in successful_regimes]
            
            # Calculate weighted average of information criteria based on number of observations
            total_obs = sum(r.get('n_observations', 0) for r in successful_regimes)
            weighted_aic = sum(r.get('aic', 0) * r.get('n_observations', 0) / total_obs
                              for r in successful_regimes if 'aic' in r)
            weighted_bic = sum(r.get('bic', 0) * r.get('n_observations', 0) / total_obs
                              for r in successful_regimes if 'bic' in r)
            
            # Store overall results
            results['overall'] = {
                'status': 'success',
                'n_regimes': len(successful_regimes),
                'total_observations': total_obs,
                'aic_values': aic_values,
                'bic_values': bic_values,
                'weighted_aic': weighted_aic,
                'weighted_bic': weighted_bic,
            }
            
            # Compare with standard model (without breaks)
            try:
                standard_model = ThresholdModel(
                    y=self.y,
                    x=self.x,
                    y_col=self.y_col,
                    x_col=self.x_col,
                    mode=self.mode,
                    alpha=self.alpha,
                    max_lags=self.max_lags
                )
                standard_results = standard_model.run_full_analysis()
                
                # Add comparison
                results['comparison'] = {
                    'standard_model': {
                        'aic': standard_results.get('aic'),
                        'bic': standard_results.get('bic'),
                    },
                    'structural_break_model': {
                        'aic': weighted_aic,
                        'bic': weighted_bic,
                    },
                    'preferred_model': 'structural_break' if weighted_bic < standard_results.get('bic', float('inf'))
                                      else 'standard',
                }
            except Exception as e:
                logger.warning(f"Error running standard model for comparison: {e}")
            
            logger.info(f"Threshold model analysis with structural breaks completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error running threshold model with structural breaks: {e}")
            raise YemenAnalysisError(f"Error running threshold model with structural breaks: {e}")
    
    @handle_errors
    def compare_models_extended(
        self, criterion: str = 'bic',
        include_breaks: bool = True,
        include_spatial: bool = False
    ) -> Dict[str, Any]:
        """
        Extended model comparison with various specifications.
        
        This method extends the standard model comparison to include models with
        structural breaks and spatial effects, providing a more comprehensive
        analysis for conflict-affected markets.
        
        Args:
            criterion: Information criterion for comparison ('aic', 'bic', 'hqic').
                     Default is 'bic' which penalizes complexity more heavily.
            include_breaks: Whether to include models with structural breaks.
            include_spatial: Whether to include spatial threshold models.
        
        Returns:
            Dictionary with comparison results for all model specifications.
            
        Raises:
            YemenAnalysisError: If the comparison fails.
        """
        logger.info(f"Comparing extended threshold model specifications with criterion={criterion}")
        
        try:
            # Start with standard model comparison
            standard_comparison = self.compare_models()
            
            # Initialize extended comparison results
            extended_comparison = {
                'standard_models': standard_comparison['models'],
                'criterion': criterion,
                'criterion_values': {},
                'models_with_breaks': {},
                'spatial_models': {},
            }
            
            # Extract criterion values from standard models
            if criterion == 'aic':
                extended_comparison['criterion_values'] = standard_comparison['aic_values']
            else:  # Default to BIC
                extended_comparison['criterion_values'] = standard_comparison['bic_values']
            
            # Add models with structural breaks if requested
            if include_breaks:
                logger.info("Including models with structural breaks in comparison")
                
                # Import UnitRootTester to use Zivot-Andrews test
                from src.models.unit_root import UnitRootTester
                
                # Initialize unit root tester
                unit_root_tester = UnitRootTester(alpha=self.alpha, max_lags=self.max_lags)
                
                # Run Zivot-Andrews test on y and x to detect breaks
                y_za_results = unit_root_tester.test_za(self.y, self.y_col)
                x_za_results = unit_root_tester.test_za(self.x, self.x_col)
                
                # Get break dates
                y_break_date = y_za_results.get('break_date')
                x_break_date = x_za_results.get('break_date')
                
                # Use both breaks if they exist and are different
                detected_breaks = []
                if y_break_date is not None:
                    detected_breaks.append(y_break_date)
                if x_break_date is not None and x_break_date != y_break_date:
                    detected_breaks.append(x_break_date)
                
                # Sort breaks chronologically
                detected_breaks.sort()
                
                if detected_breaks:
                    logger.info(f"Detected structural breaks at: {detected_breaks}")
                    
                    # Run models with structural breaks
                    break_models = {}
                    criterion_values = {}
                    
                    # TAR with breaks
                    tar_break_model = ThresholdModel(
                        y=self.y, x=self.x, y_col=self.y_col, x_col=self.x_col,
                        mode='standard', alpha=self.alpha, max_lags=self.max_lags
                    )
                    tar_break_results = tar_break_model.run_with_structural_breaks(
                        break_dates=detected_breaks
                    )
                    break_models['TAR with breaks'] = tar_break_results
                    criterion_values['TAR with breaks'] = tar_break_results['overall'].get(
                        f'weighted_{criterion}', float('inf')
                    )
                    
                    # M-TAR with breaks
                    mtar_break_model = ThresholdModel(
                        y=self.y, x=self.x, y_col=self.y_col, x_col=self.x_col,
                        mode='mtar', alpha=self.alpha, max_lags=self.max_lags
                    )
                    mtar_break_results = mtar_break_model.run_with_structural_breaks(
                        break_dates=detected_breaks
                    )
                    break_models['M-TAR with breaks'] = mtar_break_results
                    criterion_values['M-TAR with breaks'] = mtar_break_results['overall'].get(
                        f'weighted_{criterion}', float('inf')
                    )
                    
                    # TVECM with breaks
                    tvecm_break_model = ThresholdModel(
                        y=self.y, x=self.x, y_col=self.y_col, x_col=self.x_col,
                        mode='tvecm', alpha=self.alpha, max_lags=self.max_lags
                    )
                    tvecm_break_results = tvecm_break_model.run_with_structural_breaks(
                        break_dates=detected_breaks
                    )
                    break_models['TVECM with breaks'] = tvecm_break_results
                    criterion_values['TVECM with breaks'] = tvecm_break_results['overall'].get(
                        f'weighted_{criterion}', float('inf')
                    )
                    
                    # Add to extended comparison
                    extended_comparison['models_with_breaks'] = break_models
                    extended_comparison['criterion_values'].update(criterion_values)
                else:
                    logger.info("No structural breaks detected")
            
            # Add spatial threshold models if requested
            if include_spatial:
                logger.info("Including spatial threshold models in comparison")
                
                try:
                    # Import spatial model classes
                    from src.models.spatial import SpatialModel
                    
                    # Check if data has spatial information
                    if hasattr(self.y, 'geometry') or hasattr(self.x, 'geometry'):
                        # Use the dataframe with geometry
                        spatial_data = self.y if hasattr(self.y, 'geometry') else self.x
                        
                        # Create spatial model
                        spatial_model = SpatialModel(data=spatial_data)
                        
                        # Create weight matrix
                        w = spatial_model.create_weight_matrix(type='queen')
                        
                        # Run spatial lag model
                        spatial_lag_results = spatial_model.spatial_lag_model(
                            y_variable=self.y_col,
                            x_variables=[self.x_col],
                            w=w
                        )
                        
                        # Run spatial error model
                        spatial_error_results = spatial_model.spatial_error_model(
                            y_variable=self.y_col,
                            x_variables=[self.x_col],
                            w=w
                        )
                        
                        # Add to extended comparison
                        extended_comparison['spatial_models'] = {
                            'Spatial Lag': spatial_lag_results,
                            'Spatial Error': spatial_error_results,
                        }
                        
                        # Add criterion values
                        extended_comparison['criterion_values']['Spatial Lag'] = spatial_lag_results.get(
                            criterion, float('inf')
                        )
                        extended_comparison['criterion_values']['Spatial Error'] = spatial_error_results.get(
                            criterion, float('inf')
                        )
                    else:
                        logger.warning("Data does not have spatial information. Skipping spatial models.")
                except ImportError:
                    logger.warning("Spatial module not available. Skipping spatial models.")
                except Exception as e:
                    logger.warning(f"Error running spatial models: {e}")
            
            # Find best model across all types
            if extended_comparison['criterion_values']:
                best_model = min(extended_comparison['criterion_values'],
                                key=extended_comparison['criterion_values'].get)
                extended_comparison['best_model'] = best_model
                extended_comparison['best_criterion_value'] = extended_comparison['criterion_values'][best_model]
            else:
                # Fall back to standard comparison
                if criterion == 'aic':
                    extended_comparison['best_model'] = standard_comparison['best_aic']
                else:
                    extended_comparison['best_model'] = standard_comparison['best_bic']
            
            logger.info(f"Extended model comparison completed. Best model: {extended_comparison['best_model']}")
            return extended_comparison
        except Exception as e:
            logger.error(f"Error comparing extended threshold models: {e}")
            raise YemenAnalysisError(f"Error comparing extended threshold models: {e}")
