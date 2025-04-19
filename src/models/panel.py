"""
Panel data analysis module for Yemen Market Analysis.

This module provides functions for panel data analysis, including fixed effects,
random effects, and pooled OLS models.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, BetweenOLS, FirstDifferenceOLS
from linearmodels.panel.results import PanelResults
import matplotlib.pyplot as plt

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data

# Initialize logger
logger = logging.getLogger(__name__)

class PanelModel:
    """
    Panel data model for Yemen Market Analysis.
    
    This class provides methods for estimating panel data models, including
    fixed effects, random effects, and pooled OLS models.
    
    Attributes:
        data (pd.DataFrame): DataFrame containing panel data.
        entity_col (str): Column containing entity identifiers.
        time_col (str): Column containing time identifiers.
        model_type (str): Type of panel model.
        alpha (float): Significance level for hypothesis tests.
        has_data (bool): Whether data has been set.
        results (Dict[str, Any]): Model results.
    """
    
    def __init__(
        self, data: Optional[pd.DataFrame] = None,
        entity_col: str = 'market', time_col: str = 'date',
        model_type: str = 'fixed', alpha: float = 0.05
    ):
        """
        Initialize the panel data model.
        
        Args:
            data: DataFrame containing panel data.
            entity_col: Column containing entity identifiers.
            time_col: Column containing time identifiers.
            model_type: Type of panel model. Options are 'fixed', 'random', 'pooled',
                      'between', and 'first_difference'.
            alpha: Significance level for hypothesis tests.
        """
        self.data = data
        self.entity_col = entity_col
        self.time_col = time_col
        self.model_type = model_type
        self.alpha = alpha
        self.has_data = data is not None
        self.results = {}
    
    @handle_errors
    def set_data(
        self, data: pd.DataFrame, entity_col: Optional[str] = None,
        time_col: Optional[str] = None
    ) -> None:
        """
        Set the data for the model.
        
        Args:
            data: DataFrame containing panel data.
            entity_col: Column containing entity identifiers.
            time_col: Column containing time identifiers.
            
        Raises:
            YemenAnalysisError: If the data is invalid or the columns are not found.
        """
        logger.info("Setting data for panel model")
        
        # Set columns if provided
        if entity_col is not None:
            self.entity_col = entity_col
        
        if time_col is not None:
            self.time_col = time_col
        
        # Check if columns exist
        if self.entity_col not in data.columns:
            logger.error(f"Entity column {self.entity_col} not found in data")
            raise YemenAnalysisError(f"Entity column {self.entity_col} not found in data")
        
        if self.time_col not in data.columns:
            logger.error(f"Time column {self.time_col} not found in data")
            raise YemenAnalysisError(f"Time column {self.time_col} not found in data")
        
        # Set data
        self.data = data
        self.has_data = True
        
        logger.info(f"Set data with {len(self.data)} observations")
    
    @handle_errors
    def prepare_data(
        self, y_col: str, x_cols: List[str], drop_na: bool = True,
        standardize: bool = False
    ) -> pd.DataFrame:
        """
        Prepare data for panel model estimation.
        
        Args:
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            drop_na: Whether to drop rows with missing values.
            standardize: Whether to standardize the variables.
            
        Returns:
            DataFrame with prepared data.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the columns are not found.
        """
        logger.info(f"Preparing data for panel model with y_col={y_col}, x_cols={x_cols}")
        
        # Check if data has been set
        if not self.has_data:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Check if columns exist
        if y_col not in self.data.columns:
            logger.error(f"Dependent variable column {y_col} not found in data")
            raise YemenAnalysisError(f"Dependent variable column {y_col} not found in data")
        
        for col in x_cols:
            if col not in self.data.columns:
                logger.error(f"Independent variable column {col} not found in data")
                raise YemenAnalysisError(f"Independent variable column {col} not found in data")
        
        try:
            # Create a copy of the data
            data = self.data.copy()
            
            # Ensure time column is datetime
            if pd.api.types.is_datetime64_any_dtype(data[self.time_col]):
                # Already datetime, no need to convert
                pass
            else:
                # Try to convert to datetime
                data[self.time_col] = pd.to_datetime(data[self.time_col])
            
            # Select relevant columns
            cols = [self.entity_col, self.time_col, y_col] + x_cols
            data = data[cols]
            
            # Drop rows with missing values if requested
            if drop_na:
                data = data.dropna(subset=[y_col] + x_cols)
            
            # Standardize variables if requested
            if standardize:
                for col in [y_col] + x_cols:
                    data[col] = (data[col] - data[col].mean()) / data[col].std()
            
            # Create MultiIndex
            data = data.set_index([self.entity_col, self.time_col])
            
            logger.info(f"Prepared data with {len(data)} observations")
            return data
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise YemenAnalysisError(f"Error preparing data: {e}")
    
    @handle_errors
    def estimate(
        self, y_col: str, x_cols: List[str], model_type: Optional[str] = None,
        entity_effects: bool = True, time_effects: bool = False,
        cov_type: str = 'robust', **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate a panel data model.
        
        Args:
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            model_type: Type of panel model. Options are 'fixed', 'random', 'pooled',
                      'between', and 'first_difference'. If None, uses the value
                      from the class.
            entity_effects: Whether to include entity fixed effects.
            time_effects: Whether to include time fixed effects.
            cov_type: Type of covariance estimator. Options are 'robust', 'clustered',
                     'kernel', and 'unadjusted'.
            **kwargs: Additional arguments to pass to the model.
            
        Returns:
            Dictionary containing the model results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the model fails to estimate.
        """
        logger.info(f"Estimating {model_type or self.model_type} panel model for {y_col}")
        
        # Check if data has been set
        if not self.has_data:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        # Set model_type if provided
        if model_type is not None:
            self.model_type = model_type
        
        try:
            # Prepare data
            data = self.prepare_data(y_col, x_cols)
            
            # Get dependent and independent variables
            y = data[y_col]
            X = data[x_cols]
            
            # Add constant to X
            X = sm.add_constant(X)
            
            # Estimate model based on model_type
            if self.model_type == 'fixed':
                # Fixed effects model
                model = PanelOLS(
                    y, X,
                    entity_effects=entity_effects,
                    time_effects=time_effects,
                    **kwargs
                )
                results = model.fit(cov_type=cov_type)
            elif self.model_type == 'random':
                # Random effects model
                model = RandomEffects(y, X, **kwargs)
                results = model.fit(cov_type=cov_type)
            elif self.model_type == 'pooled':
                # Pooled OLS model
                model = PooledOLS(y, X, **kwargs)
                results = model.fit(cov_type=cov_type)
            elif self.model_type == 'between':
                # Between effects model
                model = BetweenOLS(y, X, **kwargs)
                results = model.fit(cov_type=cov_type)
            elif self.model_type == 'first_difference':
                # First difference model
                model = FirstDifferenceOLS(y, X, **kwargs)
                results = model.fit(cov_type=cov_type)
            else:
                logger.error(f"Invalid model type: {self.model_type}")
                raise YemenAnalysisError(f"Invalid model type: {self.model_type}")
            
            # Store results
            self.results = self._parse_results(results, y_col, x_cols, self.model_type)
            
            logger.info(f"Estimated {self.model_type} panel model with R-squared={results.rsquared:.4f}")
            return self.results
        except Exception as e:
            logger.error(f"Error estimating {self.model_type} panel model: {e}")
            raise YemenAnalysisError(f"Error estimating {self.model_type} panel model: {e}")
    
    def _parse_results(
        self, results: PanelResults, y_col: str, x_cols: List[str], model_type: str
    ) -> Dict[str, Any]:
        """
        Parse the results from a panel model.
        
        Args:
            results: Results from a panel model.
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            model_type: Type of panel model.
            
        Returns:
            Dictionary containing the parsed results.
        """
        # Create a dictionary to store the results
        parsed_results = {
            'model_type': model_type,
            'y_col': y_col,
            'x_cols': x_cols,
            'coefficients': results.params.to_dict(),
            'std_errors': results.std_errors.to_dict(),
            'p_values': results.pvalues.to_dict(),
            't_values': results.tstats.to_dict(),
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'f_statistic': results.f_statistic.stat,
            'f_p_value': results.f_statistic.pval,
            'n_obs': results.nobs,
            'n_entities': len(results.entity_effects) if hasattr(results, 'entity_effects') else None,
            'n_times': len(results.time_effects) if hasattr(results, 'time_effects') else None,
            'summary': results.summary,
        }
        
        return parsed_results
    
    @handle_errors
    def hausman_test(
        self, y_col: str, x_cols: List[str], sigmaul: Optional[float] = None,
        sigmare: Optional[float] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Perform Hausman test to choose between fixed and random effects.
        
        Args:
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            sigmaul: Variance of the random effects.
            sigmare: Variance of the residuals.
            **kwargs: Additional arguments to pass to the models.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the test fails.
        """
        logger.info("Performing Hausman test")
        
        # Check if data has been set
        if not self.has_data:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        try:
            # Prepare data
            data = self.prepare_data(y_col, x_cols)
            
            # Get dependent and independent variables
            y = data[y_col]
            X = data[x_cols]
            
            # Add constant to X
            X = sm.add_constant(X)
            
            # Estimate fixed effects model
            fe_model = PanelOLS(y, X, entity_effects=True, **kwargs)
            fe_results = fe_model.fit()
            
            # Estimate random effects model
            re_model = RandomEffects(y, X, **kwargs)
            re_results = re_model.fit(cov_type='unadjusted')
            
            # Perform Hausman test
            hausman = fe_results.compare(re_results, sigmaul=sigmaul, sigmare=sigmare)
            
            # Create results dictionary
            results = {
                'test': 'Hausman',
                'statistic': hausman.stat,
                'p_value': hausman.pval,
                'df': hausman.df,
                'recommended_model': 'fixed' if hausman.pval < self.alpha else 'random',
                'alpha': self.alpha,
                'fe_results': self._parse_results(fe_results, y_col, x_cols, 'fixed'),
                're_results': self._parse_results(re_results, y_col, x_cols, 'random'),
            }
            
            logger.info(f"Hausman test results: statistic={hausman.stat:.4f}, p_value={hausman.pval:.4f}, recommended_model={results['recommended_model']}")
            return results
        except Exception as e:
            logger.error(f"Error performing Hausman test: {e}")
            raise YemenAnalysisError(f"Error performing Hausman test: {e}")
    
    @handle_errors
    def f_test_fixed_effects(
        self, y_col: str, x_cols: List[str], entity_effects: bool = True,
        time_effects: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """
        Perform F-test for the presence of fixed effects.
        
        Args:
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            entity_effects: Whether to test for entity fixed effects.
            time_effects: Whether to test for time fixed effects.
            **kwargs: Additional arguments to pass to the models.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the test fails.
        """
        logger.info(f"Performing F-test for {'entity' if entity_effects else ''} {'time' if time_effects else ''} fixed effects")
        
        # Check if data has been set
        if not self.has_data:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        try:
            # Prepare data
            data = self.prepare_data(y_col, x_cols)
            
            # Get dependent and independent variables
            y = data[y_col]
            X = data[x_cols]
            
            # Add constant to X
            X = sm.add_constant(X)
            
            # Estimate fixed effects model
            fe_model = PanelOLS(
                y, X,
                entity_effects=entity_effects,
                time_effects=time_effects,
                **kwargs
            )
            fe_results = fe_model.fit()
            
            # Estimate pooled OLS model
            pooled_model = PooledOLS(y, X, **kwargs)
            pooled_results = pooled_model.fit()
            
            # Calculate F-statistic
            # SSR_pooled - SSR_fe
            ssr_pooled = pooled_results.resid.T @ pooled_results.resid
            ssr_fe = fe_results.resid.T @ fe_results.resid
            
            # Degrees of freedom
            n = len(y)
            k = len(x_cols) + 1  # +1 for constant
            n_effects = 0
            
            if entity_effects:
                n_effects += fe_results.df_model - k
            
            if time_effects:
                n_effects += fe_results.df_model - k - (n_effects if entity_effects else 0)
            
            # Calculate F-statistic
            f_stat = ((ssr_pooled - ssr_fe) / n_effects) / (ssr_fe / (n - k - n_effects))
            
            # Calculate p-value
            from scipy import stats
            p_value = 1 - stats.f.cdf(f_stat, n_effects, n - k - n_effects)
            
            # Create results dictionary
            results = {
                'test': 'F-test for fixed effects',
                'entity_effects': entity_effects,
                'time_effects': time_effects,
                'statistic': f_stat,
                'p_value': p_value,
                'df1': n_effects,
                'df2': n - k - n_effects,
                'are_fixed_effects_significant': p_value < self.alpha,
                'alpha': self.alpha,
                'fe_results': self._parse_results(fe_results, y_col, x_cols, 'fixed'),
                'pooled_results': self._parse_results(pooled_results, y_col, x_cols, 'pooled'),
            }
            
            logger.info(f"F-test results: statistic={f_stat:.4f}, p_value={p_value:.4f}, are_fixed_effects_significant={results['are_fixed_effects_significant']}")
            return results
        except Exception as e:
            logger.error(f"Error performing F-test for fixed effects: {e}")
            raise YemenAnalysisError(f"Error performing F-test for fixed effects: {e}")
    
    @handle_errors
    def breusch_pagan_test(
        self, y_col: str, x_cols: List[str], **kwargs
    ) -> Dict[str, Any]:
        """
        Perform Breusch-Pagan test for the presence of random effects.
        
        Args:
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            **kwargs: Additional arguments to pass to the models.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the test fails.
        """
        logger.info("Performing Breusch-Pagan test for random effects")
        
        # Check if data has been set
        if not self.has_data:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        try:
            # Prepare data
            data = self.prepare_data(y_col, x_cols)
            
            # Get dependent and independent variables
            y = data[y_col]
            X = data[x_cols]
            
            # Add constant to X
            X = sm.add_constant(X)
            
            # Estimate pooled OLS model
            pooled_model = PooledOLS(y, X, **kwargs)
            pooled_results = pooled_model.fit()
            
            # Get residuals
            residuals = pooled_results.resid
            
            # Group residuals by entity
            entity_residuals = residuals.groupby(level=0)
            
            # Calculate LM statistic
            n = len(residuals)
            n_entities = len(entity_residuals)
            T = n / n_entities  # Average number of time periods per entity
            
            sum_T = 0
            sum_sq = 0
            
            for entity, group in entity_residuals:
                T_i = len(group)
                sum_T += T_i
                sum_sq += (T_i * group.mean())**2
            
            lm_stat = n**2 / (2 * sum_T * (T - 1)) * sum_sq / (residuals**2).sum()
            
            # Calculate p-value
            from scipy import stats
            p_value = 1 - stats.chi2.cdf(lm_stat, 1)
            
            # Create results dictionary
            results = {
                'test': 'Breusch-Pagan LM',
                'statistic': lm_stat,
                'p_value': p_value,
                'df': 1,
                'are_random_effects_significant': p_value < self.alpha,
                'alpha': self.alpha,
                'pooled_results': self._parse_results(pooled_results, y_col, x_cols, 'pooled'),
            }
            
            logger.info(f"Breusch-Pagan test results: statistic={lm_stat:.4f}, p_value={p_value:.4f}, are_random_effects_significant={results['are_random_effects_significant']}")
            return results
        except Exception as e:
            logger.error(f"Error performing Breusch-Pagan test: {e}")
            raise YemenAnalysisError(f"Error performing Breusch-Pagan test: {e}")
    
    @handle_errors
    def cross_section_dependence_test(
        self, y_col: str, x_cols: List[str], test_type: str = 'cd', **kwargs
    ) -> Dict[str, Any]:
        """
        Perform test for cross-sectional dependence.
        
        Args:
            y_col: Column name for the dependent variable.
            x_cols: Column names for the independent variables.
            test_type: Type of test. Options are 'cd' (Pesaran CD) and 'lm' (Breusch-Pagan LM).
            **kwargs: Additional arguments to pass to the models.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the data has not been set or the test fails.
        """
        logger.info(f"Performing {test_type} test for cross-sectional dependence")
        
        # Check if data has been set
        if not self.has_data:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")
        
        try:
            # Prepare data
            data = self.prepare_data(y_col, x_cols)
            
            # Get dependent and independent variables
            y = data[y_col]
            X = data[x_cols]
            
            # Add constant to X
            X = sm.add_constant(X)
            
            # Estimate pooled OLS model
            pooled_model = PooledOLS(y, X, **kwargs)
            pooled_results = pooled_model.fit()
            
            # Get residuals
            residuals = pooled_results.resid
            
            # Create a wide-format DataFrame of residuals
            resid_wide = residuals.unstack(level=0)
            
            # Calculate cross-correlation matrix
            correlation_matrix = resid_wide.corr()
            
            # Get number of entities and time periods
            n_entities = len(correlation_matrix)
            n_times = len(residuals) / n_entities
            
            if test_type == 'cd':
                # Pesaran CD test
                # CD = sqrt(2 / (N * (N-1))) * sum(i=1 to N-1) sum(j=i+1 to N) sqrt(T_ij) * rho_ij
                cd_sum = 0
                
                for i in range(n_entities - 1):
                    for j in range(i + 1, n_entities):
                        entity_i = correlation_matrix.index[i]
                        entity_j = correlation_matrix.columns[j]
                        
                        # Get correlation
                        rho_ij = correlation_matrix.loc[entity_i, entity_j]
                        
                        # Calculate T_ij (common time periods)
                        T_ij = min(
                            len(resid_wide[entity_i].dropna()),
                            len(resid_wide[entity_j].dropna())
                        )
                        
                        cd_sum += np.sqrt(T_ij) * rho_ij
                
                cd_stat = np.sqrt(2 / (n_entities * (n_entities - 1))) * cd_sum
                
                # Calculate p-value assuming CD ~ N(0, 1)
                from scipy import stats
                p_value = 2 * (1 - stats.norm.cdf(abs(cd_stat)))
                
                # Create results dictionary
                results = {
                    'test': 'Pesaran CD',
                    'statistic': cd_stat,
                    'p_value': p_value,
                    'is_cross_sectionally_dependent': p_value < self.alpha,
                    'alpha': self.alpha,
                }
            elif test_type == 'lm':
                # Breusch-Pagan LM test
                # LM = sum(i=1 to N-1) sum(j=i+1 to N) T_ij * rho_ij^2
                lm_sum = 0
                
                for i in range(n_entities - 1):
                    for j in range(i + 1, n_entities):
                        entity_i = correlation_matrix.index[i]
                        entity_j = correlation_matrix.columns[j]
                        
                        # Get correlation
                        rho_ij = correlation_matrix.loc[entity_i, entity_j]
                        
                        # Calculate T_ij (common time periods)
                        T_ij = min(
                            len(resid_wide[entity_i].dropna()),
                            len(resid_wide[entity_j].dropna())
                        )
                        
                        lm_sum += T_ij * rho_ij**2
                
                lm_stat = lm_sum
                
                # Calculate p-value assuming LM ~ chi2(N * (N-1) / 2)
                from scipy import stats
                df = n_entities * (n_entities - 1) / 2
                p_value = 1 - stats.chi2.cdf(lm_stat, df)
                
                # Create results dictionary
                results = {
                    'test': 'Breusch-Pagan LM',
                    'statistic': lm_stat,
                    'p_value': p_value,
                    'df': df,
                    'is_cross_sectionally_dependent': p_value < self.alpha,
                    'alpha': self.alpha,
                }
            else:
                logger.error(f"Invalid test type: {test_type}")
                raise YemenAnalysisError(f"Invalid test type: {test_type}")
            
            logger.info(f"{test_type} test results: statistic={results['statistic']:.4f}, p_value={results['p_value']:.4f}, is_cross_sectionally_dependent={results['is_cross_sectionally_dependent']}")
            return results
        except Exception as e:
            logger.error(f"Error performing {test_type} test: {e}")
            raise YemenAnalysisError(f"Error performing {test_type} test: {e}")
    
    @handle_errors
    def plot_entity_effects(
        self, output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a plot of entity fixed effects.
        
        Args:
            output_path: Path to save the plot. If None, the plot is not saved.
            
        Returns:
            Matplotlib figure.
            
        Raises:
            YemenAnalysisError: If the model has not been estimated or the plot cannot be created.
        """
        logger.info("Creating entity effects plot")
        
        # Check if model has been estimated
        if not self.results:
            logger.error("Model has not been estimated")
            raise YemenAnalysisError("Model has not been estimated")
        
        # Check if model has entity effects
        if 'entity_effects' not in self.results or self.results['entity_effects'] is None:
            logger.error("Model does not have entity effects")
            raise YemenAnalysisError("Model does not have entity effects")
        
        try:
            # Get entity effects
            entity_effects = self.results['entity_effects']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # Plot entity effects
            ax.bar(entity_effects.index.astype(str), entity_effects.values)
            
            # Set title and labels
            ax.set_title("Entity Fixed Effects")
            ax.set_xlabel("Entity")
            ax.set_ylabel("Effect")
            
            # Rotate x-axis labels
            fig.autofmt_xdate()
            
            # Save plot if output_path is provided
            if output_path:
                fig.savefig(output_path)
                logger.info(f"Saved entity effects plot to {output_path}")
            
            logger.info("Created entity effects plot")
            return fig
        except Exception as e:
            logger.error(f"Error creating entity effects plot: {e}")
            raise YemenAnalysisError(f"Error creating entity effects plot: {e}")
    
    @handle_errors
    def plot_time_effects(
        self, output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a plot of time fixed effects.
        
        Args:
            output_path: Path to save the plot. If None, the plot is not saved.
            
        Returns:
            Matplotlib figure.
            
        Raises:
            YemenAnalysisError: If the model has not been estimated or the plot cannot be created.
        """
        logger.info("Creating time effects plot")
        
        # Check if model has been estimated
        if not self.results:
            logger.error("Model has not been estimated")
            raise YemenAnalysisError("Model has not been estimated")
        
        # Check if model has time effects
        if 'time_effects' not in self.results or self.results['time_effects'] is None:
            logger.error("Model does not have time effects")
            raise YemenAnalysisError("Model does not have time effects")
        
        try:
            # Get time effects
            time_effects = self.results['time_effects']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            
            # Plot time effects
            ax.plot(time_effects.index, time_effects.values, marker='o')
            
            # Set title and labels
            ax.set_title("Time Fixed Effects")
            ax.set_xlabel("Time")
            ax.set_ylabel("Effect")
            
            # Rotate x-axis labels
            fig.autofmt_xdate()
            
            # Save plot if output_path is provided
            if output_path:
                fig.savefig(output_path)
                logger.info(f"Saved time effects plot to {output_path}")
            
            logger.info("Created time effects plot")
            return fig
        except Exception as e:
            logger.error(f"Error creating time effects plot: {e}")
            raise YemenAnalysisError(f"Error creating time effects plot: {e}")
    
    @handle_errors
    def get_summary(self) -> str:
        """
        Get a summary of the model results.
        
        Returns:
            String containing the model summary.
            
        Raises:
            YemenAnalysisError: If the model has not been estimated.
        """
        logger.info("Getting model summary")
        
        # Check if model has been estimated
        if not self.results:
            logger.error("Model has not been estimated")
            raise YemenAnalysisError("Model has not been estimated")
        
        try:
            # Get summary from results
            if 'summary' in self.results:
                return str(self.results['summary'])
            
            # Otherwise, create a summary
            summary = f"{self.results['model_type'].capitalize()} Panel Model Summary\n"
            summary += "=" * 60 + "\n\n"
            
            summary += f"Dependent Variable: {self.results['y_col']}\n"
            summary += f"Independent Variables: {', '.join(self.results['x_cols'])}\n\n"
            
            summary += f"R-squared: {self.results['r_squared']:.4f}\n"
            summary += f"Adjusted R-squared: {self.results['adj_r_squared']:.4f}\n"
            summary += f"F-statistic: {self.results['f_statistic']:.4f} (p-value: {self.results['f_p_value']:.4f})\n"
            summary += f"Number of Observations: {self.results['n_obs']}\n"
            
            if self.results['n_entities'] is not None:
                summary += f"Number of Entities: {self.results['n_entities']}\n"
            
            if self.results['n_times'] is not None:
                summary += f"Number of Time Periods: {self.results['n_times']}\n"
            
            summary += "\nCoefficients:\n"
            summary += f"{'Variable':<15} {'Coefficient':<15} {'Std Error':<15} {'t-value':<15} {'p-value':<15}\n"
            summary += "-" * 75 + "\n"
            
            for var in self.results['coefficients'].keys():
                coef = self.results['coefficients'][var]
                std_err = self.results['std_errors'][var]
                t_value = self.results['t_values'][var]
                p_value = self.results['p_values'][var]
                
                summary += f"{var:<15} {coef:<15.4f} {std_err:<15.4f} {t_value:<15.4f} {p_value:<15.4f}\n"
            
            logger.info("Generated model summary")
            return summary
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            raise YemenAnalysisError(f"Error getting model summary: {e}")