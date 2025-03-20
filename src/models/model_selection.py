"""
Model selection and comparison module for the Yemen Market Integration Project.

This module provides tools for comparing different model specifications,
fitting models in parallel, and selecting the best model based on various
information criteria.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union, Optional, Callable
import gc
import psutil
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from utils import (
    # Error handling
    handle_errors, ModelError, ValidationError,
    
    # Validation
    validate_dataframe, validate_time_series, validate_model_inputs, raise_if_invalid,
    
    # Performance
    timer, m1_optimized, memory_usage_decorator, disk_cache, 
    parallelize_dataframe, configure_system_for_performance, optimize_dataframe,
    
    # Configuration
    config
)

# Initialize module logger
logger = logging.getLogger(__name__)

# Configure system for optimal performance
configure_system_for_performance()

@handle_errors(logger=logger, error_type=(ValueError, AttributeError, TypeError))
def calculate_information_criteria(model):
    """
    Calculate information criteria for model selection and comparison.
    
    Computes AIC, BIC, and HQC for the given model. If the model already has these
    attributes, they will be used directly. Otherwise, they will be calculated
    using the log-likelihood, number of parameters, and number of observations.
    
    Parameters
    ----------
    model : object
        Fitted model object. Should have attributes like 'llf', 'nobs', 'params',
        or methods like 'loglikelihood', 'get_params'.
        
    Returns
    -------
    dict
        Dictionary containing information criteria:
        - 'aic': Akaike Information Criterion
        - 'bic': Bayesian Information Criterion (Schwarz)
        - 'hqic': Hannan-Quinn Information Criterion
        - 'llf': Log-likelihood value
        - 'n_params': Number of parameters
        - 'n_obs': Number of observations
    """
    result = {}
    
    # Extract attributes that might already exist in the model
    # First check for common information criteria
    for criterion in ['aic', 'bic', 'hqic']:
        if hasattr(model, criterion):
            result[criterion] = getattr(model, criterion)
    
    # Get log-likelihood
    if hasattr(model, 'llf'):
        result['llf'] = model.llf
    elif hasattr(model, 'loglikelihood'):
        result['llf'] = model.loglikelihood
    elif hasattr(model, 'score'):
        # Some models use score instead of log-likelihood
        try:
            result['llf'] = model.score(getattr(model, 'X', None), getattr(model, 'y', None))
        except:
            pass
    
    # Get number of parameters
    if hasattr(model, 'params'):
        params = getattr(model, 'params')
        result['n_params'] = len(params) if hasattr(params, '__len__') else 1
    elif hasattr(model, 'nparams'):
        result['n_params'] = model.nparams
    elif hasattr(model, 'n_params'):
        result['n_params'] = model.n_params
    elif hasattr(model, 'coef_'):
        result['n_params'] = len(model.coef_) + (1 if hasattr(model, 'intercept_') else 0)
    elif hasattr(model, 'get_params'):
        try:
            result['n_params'] = len(model.get_params())
        except:
            # Default to 1 if we can't determine
            result['n_params'] = 1
    else:
        # Default to 1 if we can't determine
        result['n_params'] = 1
    
    # Get number of observations
    if hasattr(model, 'nobs'):
        result['n_obs'] = model.nobs
    elif hasattr(model, 'n_samples_'):
        result['n_obs'] = model.n_samples_
    elif hasattr(model, 'data') and hasattr(model.data, '__len__'):
        result['n_obs'] = len(model.data)
    else:
        # Try to infer from model attributes if possible
        # but will need default value later if not found
        pass
    
    # Calculate information criteria if they don't already exist
    if 'llf' in result and 'n_params' in result:
        
        # Ensure n_obs is available, use a default if not
        n_obs = result.get('n_obs', 100)  # Default to 100 if not found
        result['n_obs'] = n_obs  # Save in results
        
        k = result['n_params']
        llf = result['llf']
        
        # Calculate AIC if not already present
        if 'aic' not in result:
            result['aic'] = -2 * llf + 2 * k
        
        # Calculate BIC if not already present
        if 'bic' not in result:
            result['bic'] = -2 * llf + k * np.log(n_obs)
        
        # Calculate HQC if not already present
        if 'hqic' not in result:
            result['hqic'] = -2 * llf + 2 * k * np.log(np.log(n_obs))
    
    # Add model type information
    result['model_type'] = type(model).__name__
    
    return result

class ModelComparer:
    """
    Compare multiple model specifications for the best fit to data.
    
    This class facilitates parallel model estimation and comparison using
    various information criteria (AIC, BIC) and performance metrics.
    It optimizes memory usage and leverages parallel processing for
    computationally intensive tasks.
    """
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, TypeError))
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        model_specs: List[Dict[str, Any]],
        y_col: Optional[str] = None,
        X_cols: Optional[List[str]] = None,
        target_vars: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the model comparer with data and model specifications.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Data to use for model fitting. If DataFrame, columns can be referenced by name.
        model_specs : List[Dict[str, Any]]
            List of model specifications to compare. Each specification should include:
            - 'name': str, model name for identification
            - 'model_class': class or function, the model class to instantiate or function to call
            - 'parameters': dict, parameters to pass to the model constructor or function
            - 'fit_parameters': dict, optional parameters to pass to the fit method
        y_col : str, optional
            Column name for dependent variable (if data is DataFrame)
        X_cols : List[str], optional
            Column names for independent variables (if data is DataFrame)
        target_vars : List[str], optional
            Target variable names for multivariate models
        max_workers : int, optional
            Maximum number of worker processes for parallel execution.
            Defaults to CPU count - 1.
        """
        # Validate inputs
        self._validate_inputs(data, model_specs, y_col, X_cols)
        
        # Store data
        if isinstance(data, pd.DataFrame):
            # Optimize memory usage for DataFrame
            self.data = optimize_dataframe(data.copy())
            
            # Extract variables if specified
            if y_col is not None and X_cols is not None:
                self.y = self.data[y_col].values
                self.X = self.data[X_cols].values
                self.feature_names = X_cols
            elif target_vars is not None:
                # Multivariate case
                self.target_data = self.data[target_vars].values
                self.target_names = target_vars
                self.y = None
                self.X = None
            else:
                # Assume all preprocessing has been done
                self.y = None
                self.X = None
        else:
            # Data is already a numpy array
            self.data = data
            self.y = None
            self.X = None
        
        # Store model specifications
        self.model_specs = model_specs
        self.y_col = y_col
        self.X_cols = X_cols
        self.target_vars = target_vars
        
        # Set number of workers for parallel processing
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        
        # Initialize results storage
        self.results = {}
        self.fitted_models = {}
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"ModelComparer initialized with {len(model_specs)} model specifications. Memory usage: {memory_usage:.2f} MB")
    
    @handle_errors(logger=logger, error_type=ValidationError)
    def _validate_inputs(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        model_specs: List[Dict[str, Any]], 
        y_col: Optional[str], 
        X_cols: Optional[List[str]]
    ) -> None:
        """
        Validate input data and model specifications.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Data to validate
        model_specs : List[Dict[str, Any]]
            Model specifications to validate
        y_col : str, optional
            Column name for dependent variable
        X_cols : List[str], optional
            Column names for independent variables
            
        Raises
        ------
        ValidationError
            If inputs do not meet requirements
        """
        logger.debug("Validating inputs for model comparison")
        
        # Check data type
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValidationError(f"data must be a DataFrame or ndarray, got {type(data)}")
        
        # Validate model_specs
        if not isinstance(model_specs, list) or not model_specs:
            raise ValidationError("model_specs must be a non-empty list")
        
        for i, spec in enumerate(model_specs):
            if not isinstance(spec, dict):
                raise ValidationError(f"Model specification at index {i} must be a dictionary")
            
            # Check required keys
            required_keys = ['name', 'model_class']
            for key in required_keys:
                if key not in spec:
                    raise ValidationError(f"Model specification at index {i} is missing required key: {key}")
            
            # Model name should be unique
            for j, other_spec in enumerate(model_specs):
                if i != j and spec['name'] == other_spec['name']:
                    raise ValidationError(f"Duplicate model name '{spec['name']}' found")
        
        # If DataFrame, validate column names
        if isinstance(data, pd.DataFrame):
            if y_col is not None and y_col not in data.columns:
                raise ValidationError(f"y_col '{y_col}' not found in data columns")
                
            if X_cols is not None:
                for col in X_cols:
                    if col not in data.columns:
                        raise ValidationError(f"X_col '{col}' not found in data columns")
        
        logger.debug("Input validation passed for model comparison")
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    @m1_optimized(parallel=True)
    def fit_models(self) -> Dict[str, Any]:
        """
        Fit all model specifications in parallel.
        
        Leverages parallel processing for efficient model fitting, with
        memory optimization to handle large datasets. Progress and errors
        are tracked for each model.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of fitting results for each model, including:
            - 'model': The fitted model object
            - 'fit_time': Time taken to fit in seconds
            - 'memory_usage': Memory used during fitting in MB
            - 'fit_error': Error message if fitting failed
            - 'model_info': Information extracted from the model
        """
        # Track memory usage
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"Fitting {len(self.model_specs)} models in parallel with {self.max_workers} workers")
        
        # Use ProcessPoolExecutor for parallel model fitting
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit model fitting tasks
            futures = {}
            for i, spec in enumerate(self.model_specs):
                future = executor.submit(
                    self._fit_single_model,
                    spec,
                    i
                )
                futures[future] = spec['name']
            
            # Collect results as they complete
            results = {}
            completed = 0
            
            for future in as_completed(futures):
                model_name = futures[future]
                completed += 1
                
                try:
                    result = future.result()
                    results[model_name] = result
                    
                    logger.info(f"Fitted model '{model_name}' ({completed}/{len(self.model_specs)})")
                    
                    # Update fitted_models dictionary if successful
                    if 'model' in result and 'fit_error' not in result:
                        self.fitted_models[model_name] = result['model']
                        
                except Exception as e:
                    logger.error(f"Error fitting model '{model_name}': {str(e)}")
                    results[model_name] = {
                        'fit_error': str(e),
                        'fit_time': 0,
                        'memory_usage': 0
                    }
        
        # Store results
        self.results = results
        
        # Track memory after processing
        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
        memory_diff = end_mem - start_mem
        
        logger.info(f"Model fitting complete. Memory usage: {memory_diff:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        return results
    
    @handle_errors(logger=logger, error_type=(ValueError, RuntimeError))
    def _fit_single_model(self, spec: Dict[str, Any], model_idx: int) -> Dict[str, Any]:
        """
        Fit a single model according to its specification.
        
        Parameters
        ----------
        spec : Dict[str, Any]
            Model specification
        model_idx : int
            Index of the model in the specification list (for logging)
            
        Returns
        -------
        Dict[str, Any]
            Model fitting results
        """
        # Extract model specification
        model_name = spec['name']
        model_class = spec['model_class']
        parameters = spec.get('parameters', {})
        fit_parameters = spec.get('fit_parameters', {})
        
        # Track memory and time
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.time()
        
        try:
            logger.debug(f"Starting to fit model '{model_name}'")
            
            # Initialize the model
            if callable(model_class):
                model = model_class(**parameters)
            else:
                raise TypeError(f"model_class must be callable, got {type(model_class)}")
            
            # Fit the model
            if self.y is not None and self.X is not None:
                # Standard supervised learning case
                model.fit(self.X, self.y, **fit_parameters)
            elif hasattr(self, 'target_data') and self.target_data is not None:
                # Multivariate case
                model.fit(self.target_data, **fit_parameters)
            else:
                # Assume the model knows how to handle the data
                model.fit(self.data, **fit_parameters)
            
            # Calculate fitting time
            fit_time = time.time() - start_time
            
            # Calculate memory usage
            end_mem = process.memory_info().rss / (1024 * 1024)  # MB
            memory_usage = end_mem - start_mem
            
            # Extract model information
            model_info = self._extract_model_info(model)
            
            # Create result dictionary
            result = {
                'model': model,
                'fit_time': fit_time,
                'memory_usage': memory_usage,
                'model_info': model_info
            }
            
            logger.debug(f"Successfully fitted model '{model_name}' in {fit_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fitting model '{model_name}': {str(e)}")
            
            # Calculate time even for failed fits
            fit_time = time.time() - start_time
            
            return {
                'fit_error': str(e),
                'fit_time': fit_time,
                'memory_usage': 0
            }
    
    @handle_errors(logger=logger, error_type=(ValueError, AttributeError))
    def _extract_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Extract relevant information from a fitted model.
        
        Parameters
        ----------
        model : Any
            Fitted model object
            
        Returns
        -------
        Dict[str, Any]
            Extracted model information
        """
        info = {}
        
        # Check for common attributes across model types
        # Information criteria
        for criterion in ['aic', 'bic', 'hqic']:
            if hasattr(model, criterion):
                info[criterion] = getattr(model, criterion)
        
        # R-squared and adjusted R-squared
        if hasattr(model, 'rsquared'):
            info['rsquared'] = model.rsquared
        if hasattr(model, 'rsquared_adj'):
            info['rsquared_adj'] = model.rsquared_adj
        
        # Log-likelihood
        if hasattr(model, 'llf'):
            info['llf'] = model.llf
        elif hasattr(model, 'loglikelihood'):
            info['llf'] = model.loglikelihood
        
        # Number of parameters
        if hasattr(model, 'params'):
            params = getattr(model, 'params')
            info['n_params'] = len(params) if hasattr(params, '__len__') else 1
        elif hasattr(model, 'nparams'):
            info['n_params'] = model.nparams
        
        # Number of observations
        if hasattr(model, 'nobs'):
            info['n_obs'] = model.nobs
        
        # Model summary if available
        if hasattr(model, 'summary'):
            try:
                info['summary'] = str(model.summary())
            except:
                pass
        
        return info
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, KeyError))
    def compare_information_criteria(self) -> Dict[str, Any]:
        """
        Compare models using AIC, BIC, and other information criteria.
        
        Returns
        -------
        Dict[str, Any]
            Comparison results including:
            - 'best_model': Name of the best model based on selected criterion
            - 'criterion_values': Dictionary of criterion values by model
            - 'rankings': Dictionary of model rankings by criterion
            - 'best_by_criterion': Dictionary of best model by each criterion
        """
        if not self.results:
            logger.warning("No fitted models to compare. Call fit_models() first.")
            return {
                'error': "No fitted models available",
                'best_model': None
            }
        
        # Extract criteria values from all models
        criteria = ['aic', 'bic', 'hqic', 'llf', 'rsquared', 'rsquared_adj']
        criterion_values = {criterion: {} for criterion in criteria}
        
        for model_name, result in self.results.items():
            if 'fit_error' in result:
                logger.warning(f"Skipping model '{model_name}' due to fitting error")
                continue
                
            model_info = result.get('model_info', {})
            
            for criterion in criteria:
                if criterion in model_info:
                    criterion_values[criterion][model_name] = model_info[criterion]
        
        # Clean up empty criteria
        criterion_values = {k: v for k, v in criterion_values.items() if v}
        
        # Rank models by each criterion
        rankings = {}
        best_by_criterion = {}
        
        for criterion, values in criterion_values.items():
            if not values:
                continue
                
            # For AIC and BIC, lower is better; for others, higher is better
            reverse = criterion not in ['aic', 'bic', 'hqic']
            
            # Sort models by criterion value
            sorted_models = sorted(values.items(), key=lambda x: x[1], reverse=reverse)
            rankings[criterion] = {model: i+1 for i, (model, _) in enumerate(sorted_models)}
            
            # Store best model for this criterion
            best_by_criterion[criterion] = sorted_models[0][0]
        
        # Determine overall best model (default to AIC if available)
        best_model = None
        best_criterion = None
        for criterion in ['aic', 'bic', 'llf', 'rsquared_adj']:
            if criterion in best_by_criterion:
                best_model = best_by_criterion[criterion]
                best_criterion = criterion
                break
        
        # Create comparison result
        comparison = {
            'best_model': best_model,
            'best_criterion': best_criterion if best_model else None,
            'criterion_values': criterion_values,
            'rankings': rankings,
            'best_by_criterion': best_by_criterion
        }
        
        if best_model:
            logger.info(f"Best model based on {best_criterion}: '{best_model}'")
        else:
            logger.warning("Could not determine best model")
        
        return comparison
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, KeyError))
    def compare_performance_metrics(self) -> Dict[str, Any]:
        """
        Compare models based on computational performance metrics.
        
        Returns
        -------
        Dict[str, Any]
            Comparison results including:
            - 'fastest_model': Name of the model with shortest fitting time
            - 'most_memory_efficient': Name of the model using least memory
            - 'fit_times': Dictionary of fitting times by model
            - 'memory_usage': Dictionary of memory usage by model
            - 'performance_rankings': Dictionary of model rankings by metric
        """
        if not self.results:
            logger.warning("No fitted models to compare. Call fit_models() first.")
            return {
                'error': "No fitted models available",
                'fastest_model': None,
                'most_memory_efficient': None
            }
        
        # Extract performance metrics
        fit_times = {}
        memory_usage = {}
        
        for model_name, result in self.results.items():
            # Include all models, even those with fitting errors
            fit_times[model_name] = result.get('fit_time', float('inf'))
            memory_usage[model_name] = result.get('memory_usage', float('inf'))
        
        # Rank models by performance metrics
        performance_rankings = {
            'fit_time': {model: i+1 for i, (model, _) in enumerate(sorted(fit_times.items(), key=lambda x: x[1]))},
            'memory_usage': {model: i+1 for i, (model, _) in enumerate(sorted(memory_usage.items(), key=lambda x: x[1]))}
        }
        
        # Find fastest and most memory efficient models
        fastest_model = min(fit_times.items(), key=lambda x: x[1])[0]
        most_memory_efficient = min(memory_usage.items(), key=lambda x: x[1])[0]
        
        # Create comparison result
        comparison = {
            'fastest_model': fastest_model,
            'most_memory_efficient': most_memory_efficient,
            'fit_times': fit_times,
            'memory_usage': memory_usage,
            'performance_rankings': performance_rankings
        }
        
        logger.info(f"Fastest model: '{fastest_model}' ({fit_times[fastest_model]:.2f}s)")
        logger.info(f"Most memory efficient model: '{most_memory_efficient}' ({memory_usage[most_memory_efficient]:.2f} MB)")
        
        return comparison
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, KeyError))
    def evaluate_models(
        self,
        test_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        target_test: Optional[np.ndarray] = None,
        metrics: Optional[List[Callable]] = None,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate fitted models on test data using specified metrics.
        
        Parameters
        ----------
        test_data : Union[pd.DataFrame, np.ndarray], optional
            Test data to evaluate models on
        y_test : np.ndarray, optional
            Test target data (if using X/y format)
        X_test : np.ndarray, optional
            Test feature data (if using X/y format)
        target_test : np.ndarray, optional
            Test target data for multivariate models
        metrics : List[Callable], optional
            List of metric functions with signature metric(y_true, y_pred)
        metric_names : List[str], optional
            Names for metrics (should match length of metrics)
            
        Returns
        -------
        Dict[str, Any]
            Evaluation results including:
            - 'metric_values': Dictionary of metric values by model and metric
            - 'best_by_metric': Dictionary of best model by each metric
            - 'overall_best': Name of the overall best model
            - 'rankings': Dictionary of model rankings by metric
        """
        if not self.fitted_models:
            logger.warning("No fitted models to evaluate. Call fit_models() first.")
            return {
                'error': "No fitted models available",
                'overall_best': None
            }
        
        # Prepare test data
        if test_data is not None:
            # Handle DataFrame test data
            if isinstance(test_data, pd.DataFrame):
                if self.y_col is not None and self.X_cols is not None:
                    y_test = test_data[self.y_col].values
                    X_test = test_data[self.X_cols].values
                elif self.target_vars is not None:
                    target_test = test_data[self.target_vars].values
        
        # Validate that we have test data in some form
        if y_test is None and target_test is None and test_data is None:
            raise ValueError("No test data provided for evaluation")
        
        # Prepare metrics
        if metrics is None:
            # Default to mean squared error
            from sklearn.metrics import mean_squared_error
            metrics = [mean_squared_error]
            metric_names = ['mse']
        elif metric_names is None or len(metric_names) != len(metrics):
            # Generate default metric names if needed
            metric_names = [f"metric_{i}" for i in range(len(metrics))]
        
        # Evaluate each model
        metric_values = {name: {} for name in metric_names}
        
        for model_name, model in self.fitted_models.items():
            logger.debug(f"Evaluating model '{model_name}'")
            
            try:
                # Get predictions
                if hasattr(model, 'predict'):
                    if y_test is not None and X_test is not None:
                        # Standard supervised learning case
                        y_pred = model.predict(X_test)
                    elif target_test is not None:
                        # Multivariate case (assuming predict returns values for all targets)
                        y_pred = model.predict(target_test)
                    else:
                        # Assume the model knows how to handle the data
                        y_pred = model.predict(test_data)
                else:
                    logger.warning(f"Model '{model_name}' does not have predict method, skipping")
                    continue
                
                # Calculate metrics
                for metric, name in zip(metrics, metric_names):
                    # Handle different prediction scenarios
                    if y_test is not None:
                        # Standard case
                        score = metric(y_test, y_pred)
                    elif target_test is not None:
                        # For multivariate models, calculate average metric across all targets
                        scores = []
                        for i in range(target_test.shape[1]):
                            scores.append(metric(target_test[:, i], y_pred[:, i]))
                        score = np.mean(scores)
                    else:
                        # Skip if we can't calculate metric
                        continue
                    
                    metric_values[name][model_name] = score
                    
            except Exception as e:
                logger.warning(f"Error evaluating model '{model_name}': {str(e)}")
        
        # Clean up empty metrics
        metric_values = {k: v for k, v in metric_values.items() if v}
        
        # Rank models by each metric
        rankings = {}
        best_by_metric = {}
        
        for metric_name, values in metric_values.items():
            if not values:
                continue
                
            # For most metrics, lower is better (like MSE, MAE)
            # You can modify this logic if needed for specific metrics
            reverse = False
            
            # Sort models by metric value
            sorted_models = sorted(values.items(), key=lambda x: x[1], reverse=reverse)
            rankings[metric_name] = {model: i+1 for i, (model, _) in enumerate(sorted_models)}
            
            # Store best model for this metric
            best_by_metric[metric_name] = sorted_models[0][0]
        
        # Determine overall best model by average ranking
        if rankings:
            # Calculate average ranking for each model
            all_models = set()
            for metric_ranks in rankings.values():
                all_models.update(metric_ranks.keys())
            
            avg_rankings = {}
            for model in all_models:
                # Get model's rank for each metric (default to worst rank + 1 if not ranked)
                model_ranks = []
                for metric_name, metric_ranks in rankings.items():
                    worst_rank = max(metric_ranks.values()) if metric_ranks else 0
                    model_ranks.append(metric_ranks.get(model, worst_rank + 1))
                
                avg_rankings[model] = np.mean(model_ranks)
            
            # Model with lowest average rank is the best
            overall_best = min(avg_rankings.items(), key=lambda x: x[1])[0]
        else:
            overall_best = None
        
        # Create evaluation result
        evaluation = {
            'metric_values': metric_values,
            'best_by_metric': best_by_metric,
            'overall_best': overall_best,
            'rankings': rankings
        }
        
        if overall_best:
            logger.info(f"Best model overall: '{overall_best}'")
        else:
            logger.warning("Could not determine best model")
        
        return evaluation
    
    @timer
    @handle_errors(logger=logger, error_type=(ValueError, KeyError))
    def get_best_model(
        self, 
        criterion: str = 'aic', 
        return_model: bool = False
    ) -> Union[str, Any]:
        """
        Get the best model based on the specified criterion.
        
        Parameters
        ----------
        criterion : str, optional
            Criterion to use for model selection:
            - Information criteria: 'aic', 'bic', 'hqic'
            - Fit statistics: 'rsquared', 'rsquared_adj', 'llf'
            - Performance metrics: 'fit_time', 'memory_usage'
        return_model : bool, optional
            If True, returns the model object; otherwise, returns the model name
            
        Returns
        -------
        Union[str, Any]
            Name or object of the best model
        """
        if not self.results:
            logger.warning("No fitted models to compare. Call fit_models() first.")
            return None
        
        # Get model rankings
        if criterion in ['aic', 'bic', 'hqic', 'rsquared', 'rsquared_adj', 'llf']:
            # Compare using information criteria
            comparison = self.compare_information_criteria()
            if criterion in comparison['best_by_criterion']:
                best_model_name = comparison['best_by_criterion'][criterion]
            else:
                # Fall back to default criterion if requested one is not available
                best_model_name = comparison.get('best_model')
        elif criterion in ['fit_time', 'memory_usage']:
            # Compare using performance metrics
            comparison = self.compare_performance_metrics()
            if criterion == 'fit_time':
                best_model_name = comparison.get('fastest_model')
            else:
                best_model_name = comparison.get('most_memory_efficient')
        else:
            logger.warning(f"Unknown criterion: {criterion}")
            return None
        
        if not best_model_name:
            logger.warning(f"Could not determine best model for criterion: {criterion}")
            return None
        
        if return_model:
            return self.fitted_models.get(best_model_name)
        else:
            return best_model_name
    
    @timer
    @memory_usage_decorator
    @handle_errors(logger=logger, error_type=(ValueError, KeyError))
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of all model comparisons.
        
        Returns
        -------
        Dict[str, Any]
            Comprehensive report including:
            - 'information_criteria': Results from compare_information_criteria()
            - 'performance_metrics': Results from compare_performance_metrics()
            - 'best_models': Dictionary of best models by different criteria
            - 'model_details': Dictionary of detailed information for each model
            - 'recommendations': Textual recommendations based on results
        """
        if not self.results:
            logger.warning("No fitted models to report on. Call fit_models() first.")
            return {
                'error': "No fitted models available",
                'recommendations': "No models available for comparison"
            }
        
        # Get comparisons
        ic_comparison = self.compare_information_criteria()
        performance_comparison = self.compare_performance_metrics()
        
        # Collect best models by different criteria
        best_models = {}
        
        # From information criteria
        for criterion, model in ic_comparison.get('best_by_criterion', {}).items():
            best_models[criterion] = model
        
        # From performance metrics
        best_models['fastest'] = performance_comparison.get('fastest_model')
        best_models['memory_efficient'] = performance_comparison.get('most_memory_efficient')
        
        # Collect model details
        model_details = {}
        for model_name, result in self.results.items():
            # Skip models with fitting errors
            if 'fit_error' in result:
                model_details[model_name] = {
                    'status': 'error',
                    'error': result['fit_error'],
                    'fit_time': result.get('fit_time', float('inf'))
                }
                continue
            
            # Collect information criteria
            info = {}
            model_info = result.get('model_info', {})
            for key in ['aic', 'bic', 'hqic', 'llf', 'rsquared', 'rsquared_adj', 'n_params', 'n_obs']:
                if key in model_info:
                    info[key] = model_info[key]
            
            # Collect performance metrics
            perf = {
                'fit_time': result.get('fit_time', float('inf')),
                'memory_usage': result.get('memory_usage', float('inf'))
            }
            
            # Combine information
            model_details[model_name] = {
                'status': 'success',
                'information_criteria': info,
                'performance_metrics': perf
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(best_models, model_details)
        
        # Create comprehensive report
        report = {
            'information_criteria': ic_comparison,
            'performance_metrics': performance_comparison,
            'best_models': best_models,
            'model_details': model_details,
            'recommendations': recommendations
        }
        
        logger.info("Comprehensive model comparison report generated")
        
        return report
    
    @handle_errors(logger=logger, error_type=(ValueError, KeyError))
    def _generate_recommendations(
        self, 
        best_models: Dict[str, str],
        model_details: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate recommendations based on model comparison results.
        
        Parameters
        ----------
        best_models : Dict[str, str]
            Best models by different criteria
        model_details : Dict[str, Dict[str, Any]]
            Detailed information for each model
            
        Returns
        -------
        str
            Textual recommendations
        """
        # Count how many times each model is the best
        model_counts = {}
        for model in best_models.values():
            if model:
                model_counts[model] = model_counts.get(model, 0) + 1
        
        # Determine the overall best model
        if model_counts:
            overall_best = max(model_counts.items(), key=lambda x: x[1])[0]
            
            # Get model details
            details = model_details.get(overall_best, {})
            info = details.get('information_criteria', {})
            perf = details.get('performance_metrics', {})
            
            # Generate recommendation
            rec = f"The recommended model is '{overall_best}', which performed best across {model_counts[overall_best]} criteria.\n\n"
            
            if 'aic' in info:
                rec += f"AIC: {info['aic']:.2f}, "
            if 'bic' in info:
                rec += f"BIC: {info['bic']:.2f}, "
            if 'rsquared_adj' in info:
                rec += f"Adjusted R²: {info['rsquared_adj']:.4f}, "
            if 'fit_time' in perf:
                rec += f"Fit time: {perf['fit_time']:.2f}s, "
            if 'memory_usage' in perf:
                rec += f"Memory usage: {perf['memory_usage']:.2f} MB"
            
            # Add notes about other good models
            other_good = []
            for model, count in model_counts.items():
                if model != overall_best and count > 0:
                    other_good.append(model)
            
            if other_good:
                rec += f"\n\nOther models that performed well in some criteria: {', '.join(other_good)}."
            
            # Add specific recommendations based on model type
            rec += "\n\nRecommendations:"
            
            # Look at which specific criteria the best model excels at
            for criterion, model in best_models.items():
                if model == overall_best:
                    if criterion == 'aic':
                        rec += "\n- This model has the best balance of fit and complexity."
                    elif criterion == 'bic':
                        rec += "\n- This model is well-suited for larger datasets and is less likely to overfit."
                    elif criterion in ['rsquared', 'rsquared_adj']:
                        rec += "\n- This model explains the highest proportion of variance in the data."
                    elif criterion == 'fastest':
                        rec += "\n- This model has the fastest fitting time, making it suitable for real-time applications."
                    elif criterion == 'memory_efficient':
                        rec += "\n- This model uses the least memory, making it suitable for deployment on resource-constrained systems."
        else:
            rec = "No clear best model could be determined. Check the detailed results for more information."
        
        return rec