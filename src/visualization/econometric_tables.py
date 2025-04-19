"""Econometric tables module for Yemen Market Analysis.

This module provides functions for creating formatted tables of econometric results.
It includes functions for creating tables of unit root test results, cointegration
test results, threshold model results, spatial model results, and panel data model results.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors

# Initialize logger
logger = logging.getLogger(__name__)


class EconometricTableGenerator:
    """
    Econometric table generator for Yemen Market Analysis.

    This class provides methods for creating formatted tables of econometric results.

    Attributes:
        style (str): Table style. Options are 'markdown', 'latex', 'html', and 'plain'.
        significance_indicators (bool): Whether to include significance indicators.
        confidence_level (float): Confidence level for significance indicators.
        float_format (str): Format string for floating-point numbers.
    """

    def __init__(
        self,
        style: str = 'markdown',
        significance_indicators: bool = True,
        confidence_level: float = 0.95,
        float_format: str = '{:.4f}'
    ):
        """
        Initialize the econometric table generator.

        Args:
            style: Table style. Options are 'markdown', 'latex', 'html', and 'plain'.
            significance_indicators: Whether to include significance indicators.
            confidence_level: Confidence level for significance indicators.
            float_format: Format string for floating-point numbers.
        """
        self.style = style
        self.significance_indicators = significance_indicators
        self.confidence_level = confidence_level
        self.float_format = float_format

        # Set default tabulate format based on style
        if style == 'markdown':
            self.tabulate_format = 'pipe'
        elif style == 'latex':
            self.tabulate_format = 'latex'
        elif style == 'html':
            self.tabulate_format = 'html'
        else:
            self.tabulate_format = 'simple'

    @handle_errors
    def _add_significance_indicators(self, p_value: float) -> str:
        """
        Add significance indicators to a p-value.

        Args:
            p_value: P-value to add significance indicators to.

        Returns:
            String representation of the p-value with significance indicators.
        """
        if not self.significance_indicators:
            return self.float_format.format(p_value)

        if p_value < 0.01:
            return f"{self.float_format.format(p_value)}***"
        elif p_value < 0.05:
            return f"{self.float_format.format(p_value)}**"
        elif p_value < 0.1:
            return f"{self.float_format.format(p_value)}*"
        else:
            return self.float_format.format(p_value)

    @handle_errors
    def _format_float(self, value: float) -> str:
        """
        Format a floating-point number.

        Args:
            value: Floating-point number to format.

        Returns:
            Formatted string representation of the number.
        """
        return self.float_format.format(value)

    @handle_errors
    def _format_dict_to_table(self, data: Dict[str, Any], headers: List[str]) -> str:
        """
        Format a dictionary as a table.

        Args:
            data: Dictionary to format.
            headers: Table headers.

        Returns:
            Formatted table as a string.
        """
        # Convert dictionary to list of lists
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                for subkey, subvalue in value.items():
                    rows.append([f"{key}.{subkey}", self._format_value(subvalue)])
            else:
                rows.append([key, self._format_value(value)])

        # Generate table
        return tabulate(rows, headers=headers, tablefmt=self.tabulate_format)

    @handle_errors
    def _format_value(self, value: Any) -> str:
        """
        Format a value for display in a table.

        Args:
            value: Value to format.

        Returns:
            Formatted string representation of the value.
        """
        if isinstance(value, float):
            return self._format_float(value)
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (list, tuple)):
            return ', '.join(str(x) for x in value)
        elif isinstance(value, dict):
            return str(value)
        else:
            return str(value)

    @handle_errors
    def create_unit_root_table(
        self, results: Dict[str, Dict[str, Any]], title: Optional[str] = None
    ) -> str:
        """
        Create a table of unit root test results.

        Args:
            results: Dictionary mapping test names to test results.
            title: Table title.

        Returns:
            Formatted table as a string.
        """
        logger.info("Creating unit root test results table")

        # Create DataFrame for results
        data = []
        for test_name, test_result in results.items():
            if not isinstance(test_result, dict):
                continue

            row = {
                'Test': test_name,
                'Statistic': test_result.get('statistic'),
                'p-value': test_result.get('p_value'),
                'Lags': test_result.get('lags'),
                'Stationary': test_result.get('stationary', False)
            }

            # Add critical values if available
            if 'critical_values' in test_result and isinstance(test_result['critical_values'], dict):
                for level, value in test_result['critical_values'].items():
                    row[f'CV ({level}%)'] = value

            data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Format p-values with significance indicators
        if 'p-value' in df.columns:
            df['p-value'] = df['p-value'].apply(lambda x: self._add_significance_indicators(x) if pd.notnull(x) else '')

        # Format other numeric columns
        for col in df.columns:
            if col not in ['Test', 'p-value', 'Stationary', 'Lags']:
                df[col] = df[col].apply(lambda x: self._format_float(x) if pd.notnull(x) else '')

        # Convert DataFrame to table
        table = tabulate(df, headers='keys', tablefmt=self.tabulate_format, showindex=False)

        # Add title if provided
        if title:
            table = f"{title}\n\n{table}"

        # Add footnote for significance indicators
        if self.significance_indicators:
            footnote = "\n\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1"
            table = f"{table}{footnote}"

        logger.info("Unit root test results table created successfully")
        return table

    @handle_errors
    def create_cointegration_table(
        self, results: Dict[str, Dict[str, Any]], title: Optional[str] = None
    ) -> str:
        """
        Create a table of cointegration test results.

        Args:
            results: Dictionary mapping test names to test results.
            title: Table title.

        Returns:
            Formatted table as a string.
        """
        logger.info("Creating cointegration test results table")

        # Create DataFrame for results
        data = []
        for test_name, test_result in results.items():
            if not isinstance(test_result, dict):
                continue

            # Handle different test types
            if test_name.lower() == 'johansen':
                # Johansen test has multiple rows for different rank hypotheses
                if 'trace_stat' in test_result and 'trace_crit_vals' in test_result:
                    trace_stats = test_result['trace_stat']
                    trace_crit_vals = test_result['trace_crit_vals']
                    trace_pvals = test_result.get('trace_pvals', [None] * len(trace_stats))

                    for i, (stat, crit_val, pval) in enumerate(zip(trace_stats, trace_crit_vals, trace_pvals)):
                        row = {
                            'Test': f"{test_name} (Trace, r≤{i})",
                            'Statistic': stat,
                            'Critical Value': crit_val,
                            'p-value': pval,
                            'Cointegrated': stat > crit_val
                        }
                        data.append(row)

                if 'max_stat' in test_result and 'max_crit_vals' in test_result:
                    max_stats = test_result['max_stat']
                    max_crit_vals = test_result['max_crit_vals']
                    max_pvals = test_result.get('max_pvals', [None] * len(max_stats))

                    for i, (stat, crit_val, pval) in enumerate(zip(max_stats, max_crit_vals, max_pvals)):
                        row = {
                            'Test': f"{test_name} (Max, r≤{i})",
                            'Statistic': stat,
                            'Critical Value': crit_val,
                            'p-value': pval,
                            'Cointegrated': stat > crit_val
                        }
                        data.append(row)
            else:
                # Standard test with a single row
                row = {
                    'Test': test_name,
                    'Statistic': test_result.get('statistic'),
                    'p-value': test_result.get('p_value'),
                    'Lags': test_result.get('lags'),
                    'Cointegrated': test_result.get('cointegrated', False)
                }

                # Add critical values if available
                if 'critical_values' in test_result and isinstance(test_result['critical_values'], dict):
                    for level, value in test_result['critical_values'].items():
                        row[f'CV ({level}%)'] = value

                data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Format p-values with significance indicators
        if 'p-value' in df.columns:
            df['p-value'] = df['p-value'].apply(
                lambda x: self._add_significance_indicators(x) if pd.notnull(x) else ''
            )

        # Format other numeric columns
        for col in df.columns:
            if col not in ['Test', 'p-value', 'Cointegrated', 'Lags']:
                df[col] = df[col].apply(
                    lambda x: self._format_float(x) if pd.notnull(x) else ''
                )

        # Convert DataFrame to table
        table = tabulate(df, headers='keys', tablefmt=self.tabulate_format, showindex=False)

        # Add title if provided
        if title:
            table = f"{title}\n\n{table}"

        # Add footnote for significance indicators
        if self.significance_indicators:
            footnote = "\n\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1"
            table = f"{table}{footnote}"

        logger.info("Cointegration test results table created successfully")
        return table

    @handle_errors
    def create_threshold_model_table(
        self, results: Dict[str, Any], title: Optional[str] = None
    ) -> str:
        """
        Create a table of threshold model results.

        Args:
            results: Dictionary containing threshold model results.
            title: Table title.

        Returns:
            Formatted table as a string.
        """
        logger.info("Creating threshold model results table")

        # Extract model information
        model_type = results.get('model_type', 'Unknown')
        threshold_value = results.get('threshold', None)
        threshold_test = results.get('threshold_test', {})
        regime_1 = results.get('regime_1', {})
        regime_2 = results.get('regime_2', {})

        # Create summary table
        summary_data = [
            ['Model Type', model_type],
            ['Threshold Value', threshold_value],
            ['Threshold Test Statistic', threshold_test.get('statistic')],
            ['Threshold Test p-value', threshold_test.get('p_value')],
            ['Threshold Significant', threshold_test.get('p_value', 1.0) < 0.05],
            ['Number of Observations (Regime 1)', regime_1.get('n_obs')],
            ['Number of Observations (Regime 2)', regime_2.get('n_obs')],
        ]

        # Format p-values with significance indicators
        for i, row in enumerate(summary_data):
            if row[0] == 'Threshold Test p-value' and row[1] is not None:
                summary_data[i][1] = self._add_significance_indicators(row[1])
            elif isinstance(row[1], float):
                summary_data[i][1] = self._format_float(row[1])

        summary_table = tabulate(
            summary_data,
            headers=['Parameter', 'Value'],
            tablefmt=self.tabulate_format
        )

        # Create coefficient tables for each regime
        regime_tables = []

        for regime_name, regime_data in [('Regime 1', regime_1), ('Regime 2', regime_2)]:
            if not regime_data:
                continue

            coefficients = regime_data.get('coefficients', {})
            std_errors = regime_data.get('std_errors', {})
            t_stats = regime_data.get('t_stats', {})
            p_values = regime_data.get('p_values', {})

            coef_data = []
            for var_name in coefficients.keys():
                coef = coefficients.get(var_name)
                std_err = std_errors.get(var_name)
                t_stat = t_stats.get(var_name)
                p_value = p_values.get(var_name)

                row = [
                    var_name,
                    coef,
                    std_err,
                    t_stat,
                    p_value
                ]
                coef_data.append(row)

            # Format p-values with significance indicators
            for i, row in enumerate(coef_data):
                if row[4] is not None:  # p-value
                    coef_data[i][4] = self._add_significance_indicators(row[4])

                # Format other numeric values
                for j in range(1, 4):  # coef, std_err, t_stat
                    if row[j] is not None:
                        coef_data[i][j] = self._format_float(row[j])

            regime_table = tabulate(
                coef_data,
                headers=['Variable', 'Coefficient', 'Std. Error', 't-statistic', 'p-value'],
                tablefmt=self.tabulate_format
            )

            regime_tables.append(f"{regime_name}:\n\n{regime_table}")

        # Combine tables
        combined_table = f"Model Summary:\n\n{summary_table}"

        for regime_table in regime_tables:
            combined_table = f"{combined_table}\n\n{regime_table}"

        # Add title if provided
        if title:
            combined_table = f"{title}\n\n{combined_table}"

        # Add footnote for significance indicators
        if self.significance_indicators:
            footnote = "\n\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1"
            combined_table = f"{combined_table}{footnote}"

        logger.info("Threshold model results table created successfully")
        return combined_table

    @handle_errors
    def create_spatial_model_table(
        self, results: Dict[str, Any], title: Optional[str] = None
    ) -> str:
        """
        Create a table of spatial model results.

        Args:
            results: Dictionary containing spatial model results.
            title: Table title.

        Returns:
            Formatted table as a string.
        """
        logger.info("Creating spatial model results table")

        # Extract model information
        model_type = results.get('model_type', 'Unknown')
        n_obs = results.get('n_obs', None)
        r_squared = results.get('r_squared', None)
        adj_r_squared = results.get('adj_r_squared', None)
        log_likelihood = results.get('log_likelihood', None)
        aic = results.get('aic', None)
        bic = results.get('bic', None)

        # Create summary table
        summary_data = [
            ['Model Type', model_type],
            ['Number of Observations', n_obs],
            ['R-squared', r_squared],
            ['Adjusted R-squared', adj_r_squared],
            ['Log Likelihood', log_likelihood],
            ['AIC', aic],
            ['BIC', bic],
        ]

        # Format numeric values
        for i, row in enumerate(summary_data):
            if isinstance(row[1], float):
                summary_data[i][1] = self._format_float(row[1])

        summary_table = tabulate(
            summary_data,
            headers=['Parameter', 'Value'],
            tablefmt=self.tabulate_format
        )

        # Create coefficient table
        coefficients = results.get('coefficients', {})
        std_errors = results.get('std_errors', {})
        t_stats = results.get('t_stats', {})
        p_values = results.get('p_values', {})

        coef_data = []
        for var_name in coefficients.keys():
            coef = coefficients.get(var_name)
            std_err = std_errors.get(var_name)
            t_stat = t_stats.get(var_name)
            p_value = p_values.get(var_name)

            row = [
                var_name,
                coef,
                std_err,
                t_stat,
                p_value
            ]
            coef_data.append(row)

        # Format p-values with significance indicators
        for i, row in enumerate(coef_data):
            if row[4] is not None:  # p-value
                coef_data[i][4] = self._add_significance_indicators(row[4])

            # Format other numeric values
            for j in range(1, 4):  # coef, std_err, t_stat
                if row[j] is not None:
                    coef_data[i][j] = self._format_float(row[j])

        coef_table = tabulate(
            coef_data,
            headers=['Variable', 'Coefficient', 'Std. Error', 't-statistic', 'p-value'],
            tablefmt=self.tabulate_format
        )

        # Create diagnostic tests table
        diagnostics = results.get('diagnostics', {})

        diag_data = []
        for test_name, test_result in diagnostics.items():
            if not isinstance(test_result, dict):
                continue

            row = [
                test_name,
                test_result.get('statistic'),
                test_result.get('p_value'),
                test_result.get('null_hypothesis', '')
            ]
            diag_data.append(row)

        # Format p-values with significance indicators
        for i, row in enumerate(diag_data):
            if row[2] is not None:  # p-value
                diag_data[i][2] = self._add_significance_indicators(row[2])

            # Format statistic
            if row[1] is not None:
                diag_data[i][1] = self._format_float(row[1])

        diag_table = ""
        if diag_data:
            diag_table = tabulate(
                diag_data,
                headers=['Test', 'Statistic', 'p-value', 'Null Hypothesis'],
                tablefmt=self.tabulate_format
            )

        # Combine tables
        combined_table = f"Model Summary:\n\n{summary_table}\n\nCoefficients:\n\n{coef_table}"

        if diag_table:
            combined_table = f"{combined_table}\n\nDiagnostic Tests:\n\n{diag_table}"

        # Add title if provided
        if title:
            combined_table = f"{title}\n\n{combined_table}"

        # Add footnote for significance indicators
        if self.significance_indicators:
            footnote = "\n\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1"
            combined_table = f"{combined_table}{footnote}"

        logger.info("Spatial model results table created successfully")
        return combined_table

    @handle_errors
    def create_panel_model_table(
        self, results: Dict[str, Any], title: Optional[str] = None
    ) -> str:
        """
        Create a table of panel data model results.

        Args:
            results: Dictionary containing panel data model results.
            title: Table title.

        Returns:
            Formatted table as a string.
        """
        logger.info("Creating panel data model results table")

        # Extract model information
        model_type = results.get('model_type', 'Unknown')
        n_obs = results.get('n_obs', None)
        n_entities = results.get('n_entities', None)
        n_periods = results.get('n_periods', None)
        r_squared = results.get('r_squared', None)
        adj_r_squared = results.get('adj_r_squared', None)
        f_statistic = results.get('f_statistic', None)
        f_pvalue = results.get('f_pvalue', None)
        entity_effects = results.get('entity_effects', False)
        time_effects = results.get('time_effects', False)

        # Create summary table
        summary_data = [
            ['Model Type', model_type],
            ['Number of Observations', n_obs],
            ['Number of Entities', n_entities],
            ['Number of Time Periods', n_periods],
            ['R-squared', r_squared],
            ['Adjusted R-squared', adj_r_squared],
            ['F-statistic', f_statistic],
            ['F p-value', f_pvalue],
            ['Entity Effects', entity_effects],
            ['Time Effects', time_effects],
        ]

        # Format numeric values and p-values
        for i, row in enumerate(summary_data):
            if row[0] == 'F p-value' and row[1] is not None:
                summary_data[i][1] = self._add_significance_indicators(row[1])
            elif isinstance(row[1], float):
                summary_data[i][1] = self._format_float(row[1])

        summary_table = tabulate(
            summary_data,
            headers=['Parameter', 'Value'],
            tablefmt=self.tabulate_format
        )

        # Create coefficient table
        coefficients = results.get('coefficients', {})
        std_errors = results.get('std_errors', {})
        t_stats = results.get('t_stats', {})
        p_values = results.get('p_values', {})
        conf_int_lower = results.get('conf_int_lower', {})
        conf_int_upper = results.get('conf_int_upper', {})

        coef_data = []
        for var_name in coefficients.keys():
            coef = coefficients.get(var_name)
            std_err = std_errors.get(var_name)
            t_stat = t_stats.get(var_name)
            p_value = p_values.get(var_name)
            ci_lower = conf_int_lower.get(var_name)
            ci_upper = conf_int_upper.get(var_name)

            row = [
                var_name,
                coef,
                std_err,
                t_stat,
                p_value
            ]

            # Add confidence intervals if available
            if ci_lower is not None and ci_upper is not None:
                row.append(f"[{self._format_float(ci_lower)}, {self._format_float(ci_upper)}]")

            coef_data.append(row)

        # Format p-values with significance indicators
        for i, row in enumerate(coef_data):
            if row[4] is not None:  # p-value
                coef_data[i][4] = self._add_significance_indicators(row[4])

            # Format other numeric values
            for j in range(1, 4):  # coef, std_err, t_stat
                if row[j] is not None:
                    coef_data[i][j] = self._format_float(row[j])

        # Determine headers based on whether confidence intervals are included
        headers = ['Variable', 'Coefficient', 'Std. Error', 't-statistic', 'p-value']
        if len(coef_data[0]) > 5 if coef_data else False:
            headers.append('95% Conf. Interval')

        coef_table = tabulate(
            coef_data,
            headers=headers,
            tablefmt=self.tabulate_format
        )

        # Create tests table for panel-specific tests
        tests = results.get('tests', {})

        test_data = []
        for test_name, test_result in tests.items():
            if not isinstance(test_result, dict):
                continue

            row = [
                test_name,
                test_result.get('statistic'),
                test_result.get('p_value'),
                test_result.get('null_hypothesis', '')
            ]
            test_data.append(row)

        # Format p-values with significance indicators
        for i, row in enumerate(test_data):
            if row[2] is not None:  # p-value
                test_data[i][2] = self._add_significance_indicators(row[2])

            # Format statistic
            if row[1] is not None:
                test_data[i][1] = self._format_float(row[1])

        test_table = ""
        if test_data:
            test_table = tabulate(
                test_data,
                headers=['Test', 'Statistic', 'p-value', 'Null Hypothesis'],
                tablefmt=self.tabulate_format
            )

        # Combine tables
        combined_table = f"Model Summary:\n\n{summary_table}\n\nCoefficients:\n\n{coef_table}"

        if test_table:
            combined_table = f"{combined_table}\n\nSpecification Tests:\n\n{test_table}"

        # Add title if provided
        if title:
            combined_table = f"{title}\n\n{combined_table}"

        # Add footnote for significance indicators
        if self.significance_indicators:
            footnote = "\n\nSignificance levels: *** p<0.01, ** p<0.05, * p<0.1"
            combined_table = f"{combined_table}{footnote}"

        logger.info("Panel data model results table created successfully")
        return combined_table