"""
Academic formatting module for Yemen Market Integration analysis.

This module provides utilities for formatting econometric results according to
academic journal standards, with support for various output formats and journal styles.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List, Tuple, Callable
import logging
import re

from .statistical_tests import calculate_significance_indicators

logger = logging.getLogger(__name__)

class AcademicTableFormatter:
    """
    Format tables according to academic journal standards.
    
    This class provides utilities for creating publication-quality tables
    in various formats (LaTeX, Markdown, HTML) following academic journal standards.
    """
    
    def __init__(
        self,
        journal_style: str = 'general',
        include_significance: bool = True,
        confidence_level: float = 0.95,
        rounding: int = 4
    ):
        """
        Initialize the formatter.
        
        Parameters
        ----------
        journal_style : str
            Journal style to use ('general', 'econometrica', 'aer', 'world_bank')
        include_significance : bool
            Whether to include significance indicators
        confidence_level : float
            Confidence level for intervals (0.90, 0.95, 0.99)
        rounding : int
            Number of decimal places to round to
        """
        self.journal_style = journal_style
        self.include_significance = include_significance
        self.confidence_level = confidence_level
        self.rounding = rounding
        
        # Define journal-specific styles
        self.styles = {
            'general': {
                'significance_note': [
                    '* p<0.1', 
                    '** p<0.05', 
                    '*** p<0.01'
                ],
                'confidence_note': f'Values in parentheses represent {int(confidence_level*100)}% confidence intervals',
                'header_format': r'\textbf{{{}}}'
            },
            'econometrica': {
                'significance_note': [
                    '* p<0.1', 
                    '** p<0.05', 
                    '*** p<0.01'
                ],
                'confidence_note': f'Standard errors in parentheses',
                'header_format': r'\textsc{{{}}}'
            },
            'aer': {
                'significance_note': [
                    '* Significant at the 10 percent level', 
                    '** Significant at the 5 percent level', 
                    '*** Significant at the 1 percent level'
                ],
                'confidence_note': f'Standard errors in parentheses',
                'header_format': r'\textit{{{}}}'
            },
            'world_bank': {
                'significance_note': [
                    '* p<0.1', 
                    '** p<0.05', 
                    '*** p<0.01'
                ],
                'confidence_note': f'Robust standard errors in parentheses',
                'header_format': r'\textbf{{{}}}'
            }
        }
        
        # Use general style if requested style not found
        if journal_style not in self.styles:
            logger.warning(f"Journal style '{journal_style}' not found, using 'general'")
            self.journal_style = 'general'
    
    def format_econometric_table_latex(
        self,
        model_results: Dict[str, Any],
        title: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Create a LaTeX table for econometric results.
        
        Parameters
        ----------
        model_results : dict
            Dictionary containing model results
        title : str
            Table title
        notes : str, optional
            Additional table notes
            
        Returns
        -------
        str
            LaTeX table
        """
        # Extract style settings
        style = self.styles[self.journal_style]
        significance_note = style['significance_note']
        confidence_note = style['confidence_note']
        header_format = style['header_format']
        
        # Start building the LaTeX table
        latex = []
        
        # Table header
        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(fr'\caption{{{title}}}')
        
        # Determine number of columns based on models
        n_models = len(model_results.get('models', []))
        col_spec = 'l' + 'c' * n_models
        
        latex.append(fr'\begin{{tabular}}{{{col_spec}}}')
        latex.append(r'\toprule')
        
        # Create headers for models
        header_row = [''] + [f"{header_format.format(model['name'])}" for model in model_results.get('models', [])]
        latex.append(' & '.join(header_row) + r' \\')
        
        # Add parameters
        params = model_results.get('parameters', {})
        for param_name, param_results in params.items():
            # Parameter name
            row = [param_name]
            
            # Add results for each model
            for model in model_results.get('models', []):
                model_id = model.get('id')
                if model_id in param_results:
                    param_data = param_results[model_id]
                    
                    # Format parameter value with significance indicator
                    value = param_data.get('value', 0)
                    p_value = param_data.get('p_value', 1)
                    
                    # Add significance indicator if significant
                    sig_indicator = ''
                    if self.include_significance:
                        if p_value < 0.01:
                            sig_indicator = '^{***}'
                        elif p_value < 0.05:
                            sig_indicator = '^{**}'
                        elif p_value < 0.1:
                            sig_indicator = '^{*}'
                    
                    # Format value
                    value_formatted = f"{value:.{self.rounding}f}{sig_indicator}"
                    
                    # Format standard error or confidence interval
                    if 'std_err' in param_data:
                        se = param_data.get('std_err', 0)
                        parentheses = f"({se:.{self.rounding}f})"
                    elif 'ci_lower' in param_data and 'ci_upper' in param_data:
                        ci_lower = param_data.get('ci_lower', 0)
                        ci_upper = param_data.get('ci_upper', 0)
                        parentheses = f"[{ci_lower:.{self.rounding}f}, {ci_upper:.{self.rounding}f}]"
                    else:
                        parentheses = ""
                    
                    # Combine value and parentheses
                    cell = f"{value_formatted} \\\\ {parentheses}"
                    row.append(cell)
                else:
                    row.append('')
            
            # Add row to table
            latex.append(' & '.join(row) + r' \\')
        
        # Add horizontal line before statistics
        latex.append(r'\midrule')
        
        # Add model statistics
        stats = model_results.get('statistics', {})
        for stat_name, stat_results in stats.items():
            # Statistic name
            row = [stat_name]
            
            # Add results for each model
            for model in model_results.get('models', []):
                model_id = model.get('id')
                if model_id in stat_results:
                    stat_data = stat_results[model_id]
                    
                    # Format statistic value
                    value = stat_data.get('value', 0)
                    value_formatted = f"{value:.{self.rounding}f}"
                    
                    row.append(value_formatted)
                else:
                    row.append('')
            
            # Add row to table
            latex.append(' & '.join(row) + r' \\')
        
        # Add bottom line
        latex.append(r'\bottomrule')
        
        # Add notes
        latex.append(r'\multicolumn{' + str(n_models + 1) + r'}{l}{\footnotesize \textit{Notes:} ' + confidence_note + r'}\\')
        
        # Add significance notes
        if self.include_significance:
            for note in significance_note:
                latex.append(r'\multicolumn{' + str(n_models + 1) + r'}{l}{\footnotesize ' + note + r'}\\')
        
        # Add custom notes if provided
        if notes:
            latex.append(r'\multicolumn{' + str(n_models + 1) + r'}{l}{\footnotesize ' + notes + r'}\\')
        
        # Close table
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')
        
        return '\n'.join(latex)
    
    def format_threshold_table_latex(
        self,
        threshold_results: Dict[str, Any],
        title: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Create a LaTeX table for threshold model results.
        
        Parameters
        ----------
        threshold_results : dict
            Dictionary containing threshold model results
        title : str
            Table title
        notes : str, optional
            Additional table notes
            
        Returns
        -------
        str
            LaTeX table
        """
        # Extract style settings
        style = self.styles[self.journal_style]
        significance_note = style['significance_note']
        confidence_note = style['confidence_note']
        header_format = style['header_format']
        
        # Start building the LaTeX table
        latex = []
        
        # Table header
        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(fr'\caption{{{title}}}')
        
        # Table structure
        latex.append(r'\begin{tabular}{lcc}')
        latex.append(r'\toprule')
        
        # Headers
        latex.append(r'Parameter & Below Threshold & Above Threshold \\')
        latex.append(r'\midrule')
        
        # Extract results
        adjustment_below = threshold_results.get('adjustment_below', 0)
        adjustment_above = threshold_results.get('adjustment_above', 0)
        
        p_value_below = threshold_results.get('p_value_below', 1)
        p_value_above = threshold_results.get('p_value_above', 1)
        
        se_below = threshold_results.get('se_below', 0)
        se_above = threshold_results.get('se_above', 0)
        
        # Add significance indicators
        sig_below = ''
        sig_above = ''
        
        if self.include_significance:
            sig_below = calculate_significance_indicators(p_value_below)
            sig_above = calculate_significance_indicators(p_value_above)
            
            # Convert to LaTeX superscript format
            if sig_below:
                sig_below = f"^{{{sig_below}}}"
            if sig_above:
                sig_above = f"^{{{sig_above}}}"
        
        # Add adjustment parameters
        latex.append(fr'Adjustment Speed & {adjustment_below:.{self.rounding}f}{sig_below} & {adjustment_above:.{self.rounding}f}{sig_above} \\')
        latex.append(fr' & ({se_below:.{self.rounding}f}) & ({se_above:.{self.rounding}f}) \\')
        
        # Add half-lives if available
        if 'half_life_below' in threshold_results and 'half_life_above' in threshold_results:
            half_life_below = threshold_results.get('half_life_below', float('inf'))
            half_life_above = threshold_results.get('half_life_above', float('inf'))
            
            # Format half-lives
            if np.isinf(half_life_below):
                hl_below = r'\infty'
            else:
                hl_below = f"{half_life_below:.{self.rounding}f}"
                
            if np.isinf(half_life_above):
                hl_above = r'\infty'
            else:
                hl_above = f"{half_life_above:.{self.rounding}f}"
                
            latex.append(fr'Half-Life (periods) & {hl_below} & {hl_above} \\')
        
        # Add threshold value
        threshold = threshold_results.get('threshold', 0)
        threshold_se = threshold_results.get('threshold_se', 0)
        threshold_p = threshold_results.get('threshold_p_value', 1)
        
        # Add significance indicator for threshold
        threshold_sig = ''
        if self.include_significance:
            threshold_sig = calculate_significance_indicators(threshold_p)
            if threshold_sig:
                threshold_sig = f"^{{{threshold_sig}}}"
        
        latex.append(r'\midrule')
        latex.append(fr'Threshold & \multicolumn{{2}}{{c}}{{{threshold:.{self.rounding}f}{threshold_sig}}} \\')
        latex.append(fr' & \multicolumn{{2}}{{c}}{{({threshold_se:.{self.rounding}f})}} \\')
        
        # Add confidence interval if available
        if 'threshold_ci_lower' in threshold_results and 'threshold_ci_upper' in threshold_results:
            ci_lower = threshold_results.get('threshold_ci_lower', 0)
            ci_upper = threshold_results.get('threshold_ci_upper', 0)
            
            latex.append(fr'{int(self.confidence_level*100)}\% CI & \multicolumn{{2}}{{c}}{{[{ci_lower:.{self.rounding}f}, {ci_upper:.{self.rounding}f}]}} \\')
        
        # Add model statistics
        if 'statistics' in threshold_results:
            latex.append(r'\midrule')
            
            stats = threshold_results.get('statistics', {})
            for stat_name, stat_value in stats.items():
                latex.append(fr'{stat_name} & \multicolumn{{2}}{{c}}{{{stat_value:.{self.rounding}f}}} \\')
        
        # Add bottom line
        latex.append(r'\bottomrule')
        
        # Add notes
        latex.append(r'\multicolumn{3}{l}{\footnotesize \textit{Notes:} ' + confidence_note + r'}\\')
        
        # Add significance notes
        if self.include_significance:
            for note in significance_note:
                latex.append(r'\multicolumn{3}{l}{\footnotesize ' + note + r'}\\')
        
        # Add custom notes if provided
        if notes:
            latex.append(r'\multicolumn{3}{l}{\footnotesize ' + notes + r'}\\')
        
        # Close table
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')
        
        return '\n'.join(latex)
    
    def format_comparative_table_latex(
        self,
        model_comparison: Dict[str, Any],
        title: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Create a LaTeX table for model comparison results.
        
        Parameters
        ----------
        model_comparison : dict
            Dictionary containing model comparison results
        title : str
            Table title
        notes : str, optional
            Additional table notes
            
        Returns
        -------
        str
            LaTeX table
        """
        # Extract style settings
        style = self.styles[self.journal_style]
        header_format = style['header_format']
        
        # Start building the LaTeX table
        latex = []
        
        # Table header
        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(fr'\caption{{{title}}}')
        
        # Get information criteria
        ic_comparison = model_comparison.get('information_criteria', {})
        
        # Determine models
        models = list(ic_comparison.keys()) if ic_comparison else []
        
        # Determine criteria to include
        criteria = []
        for model_name, model_criteria in ic_comparison.items():
            criteria.extend(model_criteria.keys())
        criteria = sorted(list(set(criteria)))
        
        # Create table structure
        latex.append(r'\begin{tabular}{l' + 'r' * len(models) + '}')
        latex.append(r'\toprule')
        
        # Headers
        header_row = ['Model'] + [f"{header_format.format(model)}" for model in models]
        latex.append(' & '.join(header_row) + r' \\')
        latex.append(r'\midrule')
        
        # Add criteria rows
        for criterion in criteria:
            row = [criterion]
            
            for model in models:
                if model in ic_comparison and criterion in ic_comparison[model]:
                    value = ic_comparison[model][criterion]
                    row.append(f"{value:.{self.rounding}f}")
                else:
                    row.append('')
            
            latex.append(' & '.join(row) + r' \\')
        
        # Add best model row if available
        if 'best_model' in model_comparison:
            best_model = model_comparison['best_model']
            latex.append(r'\midrule')
            latex.append(fr'Best Model & \multicolumn{{{len(models)}}}{{c}}{{{best_model}}} \\')
        
        # Add bottom line
        latex.append(r'\bottomrule')
        
        # Add custom notes if provided
        if notes:
            latex.append(r'\multicolumn{' + str(len(models) + 1) + r'}{l}{\footnotesize \textit{Notes:} ' + notes + r'}\\')
        
        # Close table
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')
        
        return '\n'.join(latex)
    
    def format_diagnostic_table_latex(
        self,
        diagnostic_results: Dict[str, Any],
        title: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Create a LaTeX table for diagnostic test results.
        
        Parameters
        ----------
        diagnostic_results : dict
            Dictionary containing diagnostic test results
        title : str
            Table title
        notes : str, optional
            Additional table notes
            
        Returns
        -------
        str
            LaTeX table
        """
        # Extract style settings
        style = self.styles[self.journal_style]
        significance_note = style['significance_note']
        
        # Start building the LaTeX table
        latex = []
        
        # Table header
        latex.append(r'\begin{table}[htbp]')
        latex.append(r'\centering')
        latex.append(fr'\caption{{{title}}}')
        
        # Table structure
        latex.append(r'\begin{tabular}{lcrrl}')
        latex.append(r'\toprule')
        
        # Headers
        latex.append(r'Test & Statistic & p-value & Significance & Result \\')
        latex.append(r'\midrule')
        
        # Add heteroskedasticity tests
        if 'heteroskedasticity' in diagnostic_results:
            het_tests = diagnostic_results['heteroskedasticity']
            
            # Add header for section
            latex.append(r'\multicolumn{5}{l}{\textit{Heteroskedasticity Tests}} \\')
            
            # Add each test
            for test_name, test_results in het_tests.items():
                if isinstance(test_results, dict) and 'test_statistic' in test_results:
                    # Extract values
                    test_stat = test_results.get('test_statistic', 0)
                    p_value = test_results.get('p_value', 1)
                    sig = test_results.get('significance', '')
                    result = test_results.get('interpretation', '')
                    
                    # Format values
                    test_stat_str = f"{test_stat:.{self.rounding}f}"
                    p_value_str = f"{p_value:.{self.rounding}f}"
                    
                    # Convert significance to LaTeX format
                    if sig:
                        sig = f"$^{{{sig}}}$"
                    
                    # Add row
                    latex.append(f"{test_name} & {test_stat_str} & {p_value_str} & {sig} & {result} \\\\")
        
        # Add serial correlation tests
        if 'serial_correlation' in diagnostic_results:
            sc_tests = diagnostic_results['serial_correlation']
            
            # Add header for section
            latex.append(r'\midrule')
            latex.append(r'\multicolumn{5}{l}{\textit{Serial Correlation Tests}} \\')
            
            # Add each test
            for test_name, test_results in sc_tests.items():
                if isinstance(test_results, dict) and ('test_statistic' in test_results or test_name == 'durbin_watson'):
                    # Extract values
                    test_stat = test_results.get('test_statistic', 0)
                    p_value = test_results.get('p_value', None)
                    sig = test_results.get('significance', '')
                    result = test_results.get('interpretation', '')
                    
                    # Format values
                    test_stat_str = f"{test_stat:.{self.rounding}f}"
                    p_value_str = f"{p_value:.{self.rounding}f}" if p_value is not None else "N/A"
                    
                    # Convert significance to LaTeX format
                    if sig:
                        sig = f"$^{{{sig}}}$"
                    
                    # Add row
                    latex.append(f"{test_name} & {test_stat_str} & {p_value_str} & {sig} & {result} \\\\")
        
        # Add normality tests
        if 'normality' in diagnostic_results:
            norm_tests = diagnostic_results['normality']
            
            # Add header for section
            latex.append(r'\midrule')
            latex.append(r'\multicolumn{5}{l}{\textit{Normality Tests}} \\')
            
            # Add each test
            for test_name, test_results in norm_tests.items():
                if isinstance(test_results, dict) and 'test_statistic' in test_results:
                    # Extract values
                    test_stat = test_results.get('test_statistic', 0)
                    p_value = test_results.get('p_value', 1)
                    sig = test_results.get('significance', '')
                    result = test_results.get('interpretation', '')
                    
                    # Format values
                    test_stat_str = f"{test_stat:.{self.rounding}f}"
                    p_value_str = f"{p_value:.{self.rounding}f}"
                    
                    # Convert significance to LaTeX format
                    if sig:
                        sig = f"$^{{{sig}}}$"
                    
                    # Add row
                    latex.append(f"{test_name} & {test_stat_str} & {p_value_str} & {sig} & {result} \\\\")
        
        # Add specification tests
        if 'specification' in diagnostic_results:
            spec_tests = diagnostic_results['specification']
            
            # Add header for section
            latex.append(r'\midrule')
            latex.append(r'\multicolumn{5}{l}{\textit{Specification Tests}} \\')
            
            # Add each test
            for test_name, test_results in spec_tests.items():
                if isinstance(test_results, dict) and 'test_statistic' in test_results:
                    # Extract values
                    test_stat = test_results.get('test_statistic', 0)
                    p_value = test_results.get('p_value', 1)
                    sig = test_results.get('significance', '')
                    result = test_results.get('interpretation', '')
                    
                    # Format values
                    test_stat_str = f"{test_stat:.{self.rounding}f}"
                    p_value_str = f"{p_value:.{self.rounding}f}"
                    
                    # Convert significance to LaTeX format
                    if sig:
                        sig = f"$^{{{sig}}}$"
                    
                    # Add row
                    latex.append(f"{test_name} & {test_stat_str} & {p_value_str} & {sig} & {result} \\\\")
        
        # Add bottom line
        latex.append(r'\bottomrule')
        
        # Add notes
        if self.include_significance:
            latex.append(r'\multicolumn{5}{l}{\footnotesize \textit{Significance levels:}}\\')
            for note in significance_note:
                latex.append(r'\multicolumn{5}{l}{\footnotesize ' + note + r'}\\')
        
        # Add custom notes if provided
        if notes:
            latex.append(r'\multicolumn{5}{l}{\footnotesize \textit{Notes:} ' + notes + r'}\\')
        
        # Close table
        latex.append(r'\end{tabular}')
        latex.append(r'\end{table}')
        
        return '\n'.join(latex)
    
    def format_econometric_table_markdown(
        self,
        model_results: Dict[str, Any],
        title: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Create a Markdown table for econometric results.
        
        Parameters
        ----------
        model_results : dict
            Dictionary containing model results
        title : str
            Table title
        notes : str, optional
            Additional table notes
            
        Returns
        -------
        str
            Markdown table
        """
        # Start building the Markdown table
        markdown = []
        
        # Add title
        markdown.append(f"# {title}")
        markdown.append("")
        
        # Extract style settings
        style = self.styles[self.journal_style]
        significance_note = style['significance_note']
        confidence_note = style['confidence_note']
        
        # Determine number of columns
        n_models = len(model_results.get('models', []))
        
        # Create header row
        header = [""] + [model['name'] for model in model_results.get('models', [])]
        markdown.append("| " + " | ".join(header) + " |")
        
        # Create separator row
        separator = ["-" * len(h) for h in header]
        markdown.append("| " + " | ".join(separator) + " |")
        
        # Add parameters
        params = model_results.get('parameters', {})
        for param_name, param_results in params.items():
            # Parameter name
            row = [param_name]
            
            # Add results for each model
            for model in model_results.get('models', []):
                model_id = model.get('id')
                if model_id in param_results:
                    param_data = param_results[model_id]
                    
                    # Format parameter value with significance indicator
                    value = param_data.get('value', 0)
                    p_value = param_data.get('p_value', 1)
                    
                    # Add significance indicator if significant
                    sig_indicator = ''
                    if self.include_significance:
                        sig_indicator = calculate_significance_indicators(p_value)
                    
                    # Format value
                    value_formatted = f"{value:.{self.rounding}f}{sig_indicator}"
                    
                    # Format standard error or confidence interval
                    if 'std_err' in param_data:
                        se = param_data.get('std_err', 0)
                        parentheses = f"({se:.{self.rounding}f})"
                    elif 'ci_lower' in param_data and 'ci_upper' in param_data:
                        ci_lower = param_data.get('ci_lower', 0)
                        ci_upper = param_data.get('ci_upper', 0)
                        parentheses = f"[{ci_lower:.{self.rounding}f}, {ci_upper:.{self.rounding}f}]"
                    else:
                        parentheses = ""
                    
                    # Combine value and parentheses
                    cell = f"{value_formatted}<br>{parentheses}"
                    row.append(cell)
                else:
                    row.append("")
            
            # Add row to table
            markdown.append("| " + " | ".join(row) + " |")
        
        # Add model statistics
        stats = model_results.get('statistics', {})
        for stat_name, stat_results in stats.items():
            # Statistic name
            row = [stat_name]
            
            # Add results for each model
            for model in model_results.get('models', []):
                model_id = model.get('id')
                if model_id in stat_results:
                    stat_data = stat_results[model_id]
                    
                    # Format statistic value
                    value = stat_data.get('value', 0)
                    value_formatted = f"{value:.{self.rounding}f}"
                    
                    row.append(value_formatted)
                else:
                    row.append("")
            
            # Add row to table
            markdown.append("| " + " | ".join(row) + " |")
        
        # Add notes
        markdown.append("")
        markdown.append(f"**Notes:** {confidence_note}")
        
        # Add significance notes
        if self.include_significance:
            for note in significance_note:
                markdown.append(note)
        
        # Add custom notes if provided
        if notes:
            markdown.append(notes)
        
        return "\n".join(markdown)
    
    def format_econometric_table_html(
        self,
        model_results: Dict[str, Any],
        title: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Create an HTML table for econometric results.
        
        Parameters
        ----------
        model_results : dict
            Dictionary containing model results
        title : str
            Table title
        notes : str, optional
            Additional table notes
            
        Returns
        -------
        str
            HTML table
        """
        # Start building the HTML table
        html = []
        
        # Extract style settings
        style = self.styles[self.journal_style]
        significance_note = style['significance_note']
        confidence_note = style['confidence_note']
        
        # Add table start and title
        html.append("<table class='econometric-table'>")
        html.append(f"<caption>{title}</caption>")
        
        # Add header row
        html.append("<thead>")
        html.append("<tr>")
        html.append("<th></th>")  # Empty header for parameter names
        
        # Add model names
        for model in model_results.get('models', []):
            html.append(f"<th>{model['name']}</th>")
        
        html.append("</tr>")
        html.append("</thead>")
        
        # Add table body
        html.append("<tbody>")
        
        # Add parameters
        params = model_results.get('parameters', {})
        for param_name, param_results in params.items():
            html.append("<tr>")
            html.append(f"<td>{param_name}</td>")
            
            # Add results for each model
            for model in model_results.get('models', []):
                model_id = model.get('id')
                if model_id in param_results:
                    param_data = param_results[model_id]
                    
                    # Format parameter value with significance indicator
                    value = param_data.get('value', 0)
                    p_value = param_data.get('p_value', 1)
                    
                    # Add significance indicator if significant
                    sig_indicator = ''
                    if self.include_significance:
                        sig_indicator = calculate_significance_indicators(p_value)
                    
                    # Format value
                    value_formatted = f"{value:.{self.rounding}f}{sig_indicator}"
                    
                    # Format standard error or confidence interval
                    if 'std_err' in param_data:
                        se = param_data.get('std_err', 0)
                        parentheses = f"({se:.{self.rounding}f})"
                    elif 'ci_lower' in param_data and 'ci_upper' in param_data:
                        ci_lower = param_data.get('ci_lower', 0)
                        ci_upper = param_data.get('ci_upper', 0)
                        parentheses = f"[{ci_lower:.{self.rounding}f}, {ci_upper:.{self.rounding}f}]"
                    else:
                        parentheses = ""
                    
                    # Combine value and parentheses
                    cell = f"{value_formatted}<br>{parentheses}"
                    html.append(f"<td>{cell}</td>")
                else:
                    html.append("<td></td>")
            
            html.append("</tr>")
        
        # Add model statistics
        stats = model_results.get('statistics', {})
        for stat_name, stat_results in stats.items():
            html.append("<tr>")
            html.append(f"<td>{stat_name}</td>")
            
            # Add results for each model
            for model in model_results.get('models', []):
                model_id = model.get('id')
                if model_id in stat_results:
                    stat_data = stat_results[model_id]
                    
                    # Format statistic value
                    value = stat_data.get('value', 0)
                    value_formatted = f"{value:.{self.rounding}f}"
                    
                    html.append(f"<td>{value_formatted}</td>")
                else:
                    html.append("<td></td>")
            
            html.append("</tr>")
        
        html.append("</tbody>")
        html.append("</table>")
        
        # Add notes
        html.append("<div class='table-notes'>")
        html.append(f"<p><strong>Notes:</strong> {confidence_note}</p>")
        
        # Add significance notes
        if self.include_significance:
            html.append("<p>")
            html.append("<br>".join(significance_note))
            html.append("</p>")
        
        # Add custom notes if provided
        if notes:
            html.append(f"<p>{notes}</p>")
        
        html.append("</div>")
        
        return "\n".join(html)