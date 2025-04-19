"""
Model diagnostics module for Yemen Market Analysis.

This module provides functions for diagnosing and validating econometric models,
including tests for autocorrelation, heteroskedasticity, and normality.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import matplotlib.pyplot as plt

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)

class ModelDiagnostics:
    """
    Model diagnostics for Yemen Market Analysis.
    
    This class provides methods for diagnosing and validating econometric models.
    
    Attributes:
        alpha (float): Significance level for hypothesis tests.
        figsize (Tuple[int, int]): Figure size for diagnostic plots.
        dpi (int): DPI for diagnostic plots.
    """
    
    def __init__(
        self, alpha: float = 0.05, 
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100
    ):
        """
        Initialize the model diagnostics.
        
        Args:
            alpha: Significance level for hypothesis tests.
            figsize: Figure size for diagnostic plots.
            dpi: DPI for diagnostic plots.
        """
        self.alpha = alpha
        self.figsize = figsize
        self.dpi = dpi
    
    @handle_errors
    def test_heteroskedasticity(
        self, residuals: pd.Series, exog: pd.DataFrame, test: str = 'breusch_pagan'
    ) -> Dict[str, Any]:
        """
        Test for heteroskedasticity in model residuals.
        
        Args:
            residuals: Model residuals.
            exog: Exogenous variables.
            test: Test to use. Options are 'breusch_pagan' and 'white'.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the test fails or the test type is invalid.
        """
        logger.info(f"Testing for heteroskedasticity using {test} test")
        
        try:
            # Ensure exog is a numpy array
            exog_array = np.asarray(exog)
            
            # Run the selected test
            if test == 'breusch_pagan':
                # Breusch-Pagan test
                bp_test = het_breuschpagan(residuals, exog_array)
                
                # Create results dictionary
                results = {
                    'test': 'Breusch-Pagan',
                    'statistic': bp_test[0],
                    'p_value': bp_test[1],
                    'f_statistic': bp_test[2],
                    'f_p_value': bp_test[3],
                    'is_heteroskedastic': bp_test[1] < self.alpha,
                    'alpha': self.alpha,
                }
            elif test == 'white':
                # White test
                white_test = het_white(residuals, exog_array)
                
                # Create results dictionary
                results = {
                    'test': 'White',
                    'statistic': white_test[0],
                    'p_value': white_test[1],
                    'f_statistic': white_test[2],
                    'f_p_value': white_test[3],
                    'is_heteroskedastic': white_test[1] < self.alpha,
                    'alpha': self.alpha,
                }
            else:
                logger.error(f"Invalid heteroskedasticity test: {test}")
                raise YemenAnalysisError(f"Invalid heteroskedasticity test: {test}")
            
            logger.info(f"{test} test results: statistic={results['statistic']:.4f}, p_value={results['p_value']:.4f}, is_heteroskedastic={results['is_heteroskedastic']}")
            return results
        except Exception as e:
            logger.error(f"Error testing for heteroskedasticity: {e}")
            raise YemenAnalysisError(f"Error testing for heteroskedasticity: {e}")
    
    @handle_errors
    def test_autocorrelation(
        self, residuals: pd.Series, lags: int = 10, test: str = 'ljung_box'
    ) -> Dict[str, Any]:
        """
        Test for autocorrelation in model residuals.
        
        Args:
            residuals: Model residuals.
            lags: Number of lags to include in the test.
            test: Test to use. Options are 'ljung_box' and 'durbin_watson'.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the test fails or the test type is invalid.
        """
        logger.info(f"Testing for autocorrelation using {test} test")
        
        try:
            # Run the selected test
            if test == 'ljung_box':
                # Ljung-Box test
                lb_test = acorr_ljungbox(residuals, lags=lags)
                
                # Create results dictionary
                results = {
                    'test': 'Ljung-Box',
                    'statistics': lb_test[0].tolist(),
                    'p_values': lb_test[1].tolist(),
                    'lags': list(range(1, lags + 1)),
                    'is_autocorrelated': any(p < self.alpha for p in lb_test[1]),
                    'alpha': self.alpha,
                }
            elif test == 'durbin_watson':
                # Durbin-Watson test
                dw_statistic = durbin_watson(residuals)
                
                # Define p-value heuristic for Durbin-Watson
                if dw_statistic < 1.5:
                    p_value = 0.01  # Strong positive autocorrelation
                elif dw_statistic > 2.5:
                    p_value = 0.01  # Strong negative autocorrelation
                else:
                    p_value = 0.1  # No strong evidence of autocorrelation
                
                # Create results dictionary
                results = {
                    'test': 'Durbin-Watson',
                    'statistic': dw_statistic,
                    'p_value': p_value,  # Approximate p-value
                    'is_autocorrelated': dw_statistic < 1.5 or dw_statistic > 2.5,
                    'alpha': self.alpha,
                }
            else:
                logger.error(f"Invalid autocorrelation test: {test}")
                raise YemenAnalysisError(f"Invalid autocorrelation test: {test}")
            
            if test == 'ljung_box':
                logger.info(f"{test} test results: significant lags={sum(p < self.alpha for p in results['p_values'])}, is_autocorrelated={results['is_autocorrelated']}")
            else:
                logger.info(f"{test} test results: statistic={results['statistic']:.4f}, is_autocorrelated={results['is_autocorrelated']}")
            
            return results
        except Exception as e:
            logger.error(f"Error testing for autocorrelation: {e}")
            raise YemenAnalysisError(f"Error testing for autocorrelation: {e}")
    
    @handle_errors
    def test_normality(self, residuals: pd.Series, test: str = 'jarque_bera') -> Dict[str, Any]:
        """
        Test for normality of model residuals.
        
        Args:
            residuals: Model residuals.
            test: Test to use. Options are 'jarque_bera' and 'shapiro'.
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the test fails or the test type is invalid.
        """
        logger.info(f"Testing for normality using {test} test")
        
        try:
            # Run the selected test
            if test == 'jarque_bera':
                # Jarque-Bera test
                jb_test = jarque_bera(residuals)
                
                # Create results dictionary
                results = {
                    'test': 'Jarque-Bera',
                    'statistic': jb_test[0],
                    'p_value': jb_test[1],
                    'skew': jb_test[2],
                    'kurtosis': jb_test[3],
                    'is_normal': jb_test[1] >= self.alpha,
                    'alpha': self.alpha,
                }
            elif test == 'shapiro':
                # Shapiro-Wilk test
                sw_test = stats.shapiro(residuals)
                
                # Create results dictionary
                results = {
                    'test': 'Shapiro-Wilk',
                    'statistic': sw_test[0],
                    'p_value': sw_test[1],
                    'is_normal': sw_test[1] >= self.alpha,
                    'alpha': self.alpha,
                }
            else:
                logger.error(f"Invalid normality test: {test}")
                raise YemenAnalysisError(f"Invalid normality test: {test}")
            
            logger.info(f"{test} test results: statistic={results['statistic']:.4f}, p_value={results['p_value']:.4f}, is_normal={results['is_normal']}")
            return results
        except Exception as e:
            logger.error(f"Error testing for normality: {e}")
            raise YemenAnalysisError(f"Error testing for normality: {e}")
    
    @handle_errors
    def test_stationarity(
        self, residuals: pd.Series, max_lags: int = 10, trend: str = 'c'
    ) -> Dict[str, Any]:
        """
        Test for stationarity of model residuals.
        
        Args:
            residuals: Model residuals.
            max_lags: Maximum number of lags to include in the test.
            trend: Trend to include in the test. Options are 'c' (constant),
                 'ct' (constant and trend), 'ctt' (constant, linear and quadratic trend),
                 and 'n' (no trend).
            
        Returns:
            Dictionary containing the test results.
            
        Raises:
            YemenAnalysisError: If the test fails.
        """
        logger.info(f"Testing for stationarity using ADF test")
        
        try:
            # Import ADF test
            from statsmodels.tsa.stattools import adfuller
            
            # Run ADF test
            adf_test = adfuller(residuals, maxlag=max_lags, regression=trend)
            
            # Create results dictionary
            results = {
                'test': 'ADF',
                'statistic': adf_test[0],
                'p_value': adf_test[1],
                'n_lags': adf_test[2],
                'n_obs': adf_test[3],
                'critical_values': adf_test[4],
                'is_stationary': adf_test[1] < self.alpha,
                'alpha': self.alpha,
            }
            
            logger.info(f"ADF test results: statistic={results['statistic']:.4f}, p_value={results['p_value']:.4f}, is_stationary={results['is_stationary']}")
            return results
        except Exception as e:
            logger.error(f"Error testing for stationarity: {e}")
            raise YemenAnalysisError(f"Error testing for stationarity: {e}")
    
    @handle_errors
    def plot_residuals(
        self, residuals: pd.Series, dates: Optional[pd.Series] = None,
        title: str = "Residuals Plot"
    ) -> plt.Figure:
        """
        Create a plot of model residuals.
        
        Args:
            residuals: Model residuals.
            dates: Dates for the residuals. If None, uses index values.
            title: Title for the plot.
            
        Returns:
            Matplotlib figure.
            
        Raises:
            YemenAnalysisError: If the plot cannot be created.
        """
        logger.info("Creating residuals plot")
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot residuals
            if dates is not None:
                ax.plot(dates, residuals)
                ax.set_xlabel("Date")
            else:
                ax.plot(residuals)
                ax.set_xlabel("Index")
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='r', linestyle='-')
            
            # Set title and labels
            ax.set_title(title)
            ax.set_ylabel("Residuals")
            
            # Rotate x-axis labels if dates are provided
            if dates is not None:
                fig.autofmt_xdate()
            
            logger.info("Created residuals plot")
            return fig
        except Exception as e:
            logger.error(f"Error creating residuals plot: {e}")
            raise YemenAnalysisError(f"Error creating residuals plot: {e}")
    
    @handle_errors
    def plot_residuals_histogram(
        self, residuals: pd.Series, bins: int = 30,
        title: str = "Residuals Histogram"
    ) -> plt.Figure:
        """
        Create a histogram of model residuals.
        
        Args:
            residuals: Model residuals.
            bins: Number of bins for the histogram.
            title: Title for the plot.
            
        Returns:
            Matplotlib figure.
            
        Raises:
            YemenAnalysisError: If the plot cannot be created.
        """
        logger.info("Creating residuals histogram")
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot histogram
            ax.hist(residuals, bins=bins, alpha=0.5, density=True)
            
            # Add a normal distribution curve
            x = np.linspace(residuals.min(), residuals.max(), 100)
            y = stats.norm.pdf(x, residuals.mean(), residuals.std())
            ax.plot(x, y, 'r-', linewidth=2)
            
            # Set title and labels
            ax.set_title(title)
            ax.set_xlabel("Residuals")
            ax.set_ylabel("Density")
            
            # Add legend
            ax.legend(["Normal Distribution", "Residuals"])
            
            logger.info("Created residuals histogram")
            return fig
        except Exception as e:
            logger.error(f"Error creating residuals histogram: {e}")
            raise YemenAnalysisError(f"Error creating residuals histogram: {e}")
    
    @handle_errors
    def plot_residuals_qq(
        self, residuals: pd.Series, title: str = "Residuals Q-Q Plot"
    ) -> plt.Figure:
        """
        Create a Q-Q plot of model residuals.
        
        Args:
            residuals: Model residuals.
            title: Title for the plot.
            
        Returns:
            Matplotlib figure.
            
        Raises:
            YemenAnalysisError: If the plot cannot be created.
        """
        logger.info("Creating residuals Q-Q plot")
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Create Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax)
            
            # Set title
            ax.set_title(title)
            
            logger.info("Created residuals Q-Q plot")
            return fig
        except Exception as e:
            logger.error(f"Error creating residuals Q-Q plot: {e}")
            raise YemenAnalysisError(f"Error creating residuals Q-Q plot: {e}")
    
    @handle_errors
    def plot_residuals_vs_fitted(
        self, residuals: pd.Series, fitted: pd.Series,
        title: str = "Residuals vs. Fitted Values"
    ) -> plt.Figure:
        """
        Create a plot of residuals vs. fitted values.
        
        Args:
            residuals: Model residuals.
            fitted: Fitted values from the model.
            title: Title for the plot.
            
        Returns:
            Matplotlib figure.
            
        Raises:
            YemenAnalysisError: If the plot cannot be created.
        """
        logger.info("Creating residuals vs. fitted values plot")
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            
            # Plot residuals vs. fitted values
            ax.scatter(fitted, residuals, alpha=0.5)
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='r', linestyle='-')
            
            # Set title and labels
            ax.set_title(title)
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            
            logger.info("Created residuals vs. fitted values plot")
            return fig
        except Exception as e:
            logger.error(f"Error creating residuals vs. fitted values plot: {e}")
            raise YemenAnalysisError(f"Error creating residuals vs. fitted values plot: {e}")
    
    @handle_errors
    def plot_acf_pacf(
        self, residuals: pd.Series, lags: int = 40,
        title: str = "ACF and PACF of Residuals"
    ) -> plt.Figure:
        """
        Create ACF and PACF plots of model residuals.
        
        Args:
            residuals: Model residuals.
            lags: Number of lags to include in the plots.
            title: Title for the plot.
            
        Returns:
            Matplotlib figure.
            
        Raises:
            YemenAnalysisError: If the plot cannot be created.
        """
        logger.info("Creating ACF and PACF plots")
        
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
            
            # Plot ACF
            plot_acf(residuals, lags=lags, ax=ax1)
            ax1.set_title("Autocorrelation Function (ACF)")
            
            # Plot PACF
            plot_pacf(residuals, lags=lags, ax=ax2)
            ax2.set_title("Partial Autocorrelation Function (PACF)")
            
            # Set overall title
            fig.suptitle(title)
            
            # Adjust layout
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            
            logger.info("Created ACF and PACF plots")
            return fig
        except Exception as e:
            logger.error(f"Error creating ACF and PACF plots: {e}")
            raise YemenAnalysisError(f"Error creating ACF and PACF plots: {e}")
    
    @handle_errors
    def run_all_diagnostics(
        self, residuals: pd.Series, exog: pd.DataFrame,
        fitted: Optional[pd.Series] = None, dates: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run all diagnostic tests on model residuals.
        
        Args:
            residuals: Model residuals.
            exog: Exogenous variables.
            fitted: Fitted values from the model.
            dates: Dates for the residuals.
            
        Returns:
            Dictionary containing all diagnostic results.
            
        Raises:
            YemenAnalysisError: If any of the tests fail.
        """
        logger.info("Running all diagnostic tests")
        
        try:
            # Initialize results dictionary
            all_results = {}
            
            # Run heteroskedasticity tests
            bp_results = self.test_heteroskedasticity(residuals, exog, test='breusch_pagan')
            white_results = self.test_heteroskedasticity(residuals, exog, test='white')
            all_results['heteroskedasticity'] = {
                'breusch_pagan': bp_results,
                'white': white_results,
                'is_heteroskedastic': bp_results['is_heteroskedastic'] or white_results['is_heteroskedastic'],
            }
            
            # Run autocorrelation tests
            lb_results = self.test_autocorrelation(residuals, test='ljung_box')
            dw_results = self.test_autocorrelation(residuals, test='durbin_watson')
            all_results['autocorrelation'] = {
                'ljung_box': lb_results,
                'durbin_watson': dw_results,
                'is_autocorrelated': lb_results['is_autocorrelated'] or dw_results['is_autocorrelated'],
            }
            
            # Run normality tests
            jb_results = self.test_normality(residuals, test='jarque_bera')
            sw_results = self.test_normality(residuals, test='shapiro')
            all_results['normality'] = {
                'jarque_bera': jb_results,
                'shapiro': sw_results,
                'is_normal': jb_results['is_normal'] and sw_results['is_normal'],
            }
            
            # Run stationarity test
            adf_results = self.test_stationarity(residuals)
            all_results['stationarity'] = {
                'adf': adf_results,
                'is_stationary': adf_results['is_stationary'],
            }
            
            # Create plots
            plots = {}
            
            # Residuals plot
            plots['residuals'] = self.plot_residuals(residuals, dates)
            
            # Residuals histogram
            plots['histogram'] = self.plot_residuals_histogram(residuals)
            
            # Residuals Q-Q plot
            plots['qq'] = self.plot_residuals_qq(residuals)
            
            # Residuals vs. fitted values plot
            if fitted is not None:
                plots['vs_fitted'] = self.plot_residuals_vs_fitted(residuals, fitted)
            
            # ACF and PACF plots
            plots['acf_pacf'] = self.plot_acf_pacf(residuals)
            
            all_results['plots'] = plots
            
            # Create overall assessment
            all_results['overall'] = {
                'issues': [],
                'is_valid': True,
            }
            
            # Add issues based on test results
            if all_results['heteroskedasticity']['is_heteroskedastic']:
                all_results['overall']['issues'].append('Heteroskedasticity detected')
                all_results['overall']['is_valid'] = False
            
            if all_results['autocorrelation']['is_autocorrelated']:
                all_results['overall']['issues'].append('Autocorrelation detected')
                all_results['overall']['is_valid'] = False
            
            if not all_results['normality']['is_normal']:
                all_results['overall']['issues'].append('Non-normal residuals')
                all_results['overall']['is_valid'] = False
            
            if not all_results['stationarity']['is_stationary']:
                all_results['overall']['issues'].append('Non-stationary residuals')
                all_results['overall']['is_valid'] = False
            
            logger.info(f"All diagnostic tests completed: model valid? {all_results['overall']['is_valid']}")
            return all_results
        except Exception as e:
            logger.error(f"Error running all diagnostic tests: {e}")
            raise YemenAnalysisError(f"Error running all diagnostic tests: {e}")
    
    @handle_errors
    def create_diagnostic_report(
        self, results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """
        Create a diagnostic report from test results.
        
        Args:
            results: Results from run_all_diagnostics.
            output_path: Path to save the report to. If None, report is not saved.
            
        Returns:
            Report as a string.
            
        Raises:
            YemenAnalysisError: If the report cannot be created.
        """
        logger.info("Creating diagnostic report")
        
        try:
            # Create report header
            report = "# Model Diagnostics Report\n\n"
            
            # Add overall assessment
            report += "## Overall Assessment\n\n"
            
            if results['overall']['is_valid']:
                report += "✅ **The model passes all diagnostic tests.**\n\n"
            else:
                report += "❌ **The model fails some diagnostic tests.**\n\n"
                report += "Issues detected:\n"
                for issue in results['overall']['issues']:
                    report += f"- {issue}\n"
                report += "\n"
            
            # Add heteroskedasticity results
            report += "## Heteroskedasticity Tests\n\n"
            
            # Breusch-Pagan test
            bp = results['heteroskedasticity']['breusch_pagan']
            report += f"### Breusch-Pagan Test\n\n"
            report += f"- Statistic: {bp['statistic']:.4f}\n"
            report += f"- p-value: {bp['p_value']:.4f}\n"
            report += f"- Result: {'❌ Heteroskedasticity detected' if bp['is_heteroskedastic'] else '✅ No heteroskedasticity detected'}\n\n"
            
            # White test
            white = results['heteroskedasticity']['white']
            report += f"### White Test\n\n"
            report += f"- Statistic: {white['statistic']:.4f}\n"
            report += f"- p-value: {white['p_value']:.4f}\n"
            report += f"- Result: {'❌ Heteroskedasticity detected' if white['is_heteroskedastic'] else '✅ No heteroskedasticity detected'}\n\n"
            
            # Add autocorrelation results
            report += "## Autocorrelation Tests\n\n"
            
            # Ljung-Box test
            lb = results['autocorrelation']['ljung_box']
            report += f"### Ljung-Box Test\n\n"
            report += f"- Result: {'❌ Autocorrelation detected' if lb['is_autocorrelated'] else '✅ No autocorrelation detected'}\n"
            report += f"- Significant lags: {sum(p < self.alpha for p in lb['p_values'])}/{len(lb['p_values'])}\n\n"
            
            # Durbin-Watson test
            dw = results['autocorrelation']['durbin_watson']
            report += f"### Durbin-Watson Test\n\n"
            report += f"- Statistic: {dw['statistic']:.4f}\n"
            report += f"- Result: {'❌ Autocorrelation detected' if dw['is_autocorrelated'] else '✅ No autocorrelation detected'}\n"
            report += f"- Interpretation: {self._interpret_durbin_watson(dw['statistic'])}\n\n"
            
            # Add normality results
            report += "## Normality Tests\n\n"
            
            # Jarque-Bera test
            jb = results['normality']['jarque_bera']
            report += f"### Jarque-Bera Test\n\n"
            report += f"- Statistic: {jb['statistic']:.4f}\n"
            report += f"- p-value: {jb['p_value']:.4f}\n"
            report += f"- Skewness: {jb['skew']:.4f}\n"
            report += f"- Kurtosis: {jb['kurtosis']:.4f}\n"
            report += f"- Result: {'❌ Non-normal residuals' if not jb['is_normal'] else '✅ Normal residuals'}\n\n"
            
            # Shapiro-Wilk test
            sw = results['normality']['shapiro']
            report += f"### Shapiro-Wilk Test\n\n"
            report += f"- Statistic: {sw['statistic']:.4f}\n"
            report += f"- p-value: {sw['p_value']:.4f}\n"
            report += f"- Result: {'❌ Non-normal residuals' if not sw['is_normal'] else '✅ Normal residuals'}\n\n"
            
            # Add stationarity results
            report += "## Stationarity Test\n\n"
            
            # ADF test
            adf = results['stationarity']['adf']
            report += f"### Augmented Dickey-Fuller Test\n\n"
            report += f"- Statistic: {adf['statistic']:.4f}\n"
            report += f"- p-value: {adf['p_value']:.4f}\n"
            report += f"- Result: {'✅ Stationary residuals' if adf['is_stationary'] else '❌ Non-stationary residuals'}\n\n"
            
            # Add recommendations
            report += "## Recommendations\n\n"
            
            if results['overall']['is_valid']:
                report += "The model appears to be well-specified. No remedial actions are needed.\n\n"
            else:
                report += "Consider the following remedial actions:\n\n"
                
                if results['heteroskedasticity']['is_heteroskedastic']:
                    report += "### For Heteroskedasticity\n\n"
                    report += "- Use robust standard errors (HC0, HC1, HC2, HC3)\n"
                    report += "- Transform the dependent variable (e.g., log transformation)\n"
                    report += "- Use weighted least squares (WLS)\n\n"
                
                if results['autocorrelation']['is_autocorrelated']:
                    report += "### For Autocorrelation\n\n"
                    report += "- Include lagged dependent variables in the model\n"
                    report += "- Use HAC standard errors (Newey-West)\n"
                    report += "- Use ARIMA or ARIMAX models\n\n"
                
                if not results['normality']['is_normal']:
                    report += "### For Non-normality\n\n"
                    report += "- Transform the dependent variable (e.g., Box-Cox transformation)\n"
                    report += "- Check for outliers and influential observations\n"
                    report += "- Consider quantile regression\n\n"
                
                if not results['stationarity']['is_stationary']:
                    report += "### For Non-stationarity\n\n"
                    report += "- Difference the data\n"
                    report += "- Use cointegration techniques (if appropriate)\n"
                    report += "- Consider error correction models\n\n"
            
            # Save report if output_path is provided
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(report)
                logger.info(f"Saved diagnostic report to {output_path}")
            
            logger.info("Created diagnostic report")
            return report
        except Exception as e:
            logger.error(f"Error creating diagnostic report: {e}")
            raise YemenAnalysisError(f"Error creating diagnostic report: {e}")
    
    def _interpret_durbin_watson(self, dw_statistic: float) -> str:
        """
        Interpret the Durbin-Watson statistic.
        
        Args:
            dw_statistic: Durbin-Watson statistic.
            
        Returns:
            Interpretation of the statistic.
        """
        if dw_statistic < 1.0:
            return "Strong positive autocorrelation"
        elif dw_statistic < 1.5:
            return "Positive autocorrelation"
        elif dw_statistic < 2.0:
            return "Slight positive autocorrelation"
        elif dw_statistic < 2.5:
            return "Slight negative autocorrelation"
        elif dw_statistic < 3.0:
            return "Negative autocorrelation"
        else:
            return "Strong negative autocorrelation"