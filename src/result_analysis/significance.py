"""
Significance formatting module for Yemen Market Integration analysis.

This module provides utilities for formatting statistical significance
indicators in econometric results according to academic standards.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List, Tuple, Callable
import logging
import re

logger = logging.getLogger(__name__)

def format_with_significance(
    value: float,
    p_value: float,
    format_str: str = "{:.4f}",
    include_significance: bool = True
) -> str:
    """
    Format a value with significance indicators.
    
    Parameters
    ----------
    value : float
        Value to format
    p_value : float
        p-value for significance testing
    format_str : str, optional
        Format string for the value
    include_significance : bool, optional
        Whether to include significance indicators
        
    Returns
    -------
    str
        Formatted value with significance indicators
    """
    # Format value according to format string
    formatted_value = format_str.format(value)
    
    # Add significance indicators if requested
    if include_significance:
        if p_value < 0.01:
            return f"{formatted_value}***"
        elif p_value < 0.05:
            return f"{formatted_value}**"
        elif p_value < 0.1:
            return f"{formatted_value}*"
    
    return formatted_value


def add_significance_indicators(
    df: pd.DataFrame,
    value_cols: List[str],
    p_value_cols: List[str],
    output_cols: Optional[List[str]] = None,
    format_str: str = "{:.4f}"
) -> pd.DataFrame:
    """
    Add significance indicators to values in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing values and p-values
    value_cols : list of str
        Column names for values
    p_value_cols : list of str
        Column names for p-values (should match value_cols in order)
    output_cols : list of str, optional
        Column names for formatted output (if None, uses value_cols)
    format_str : str, optional
        Format string for values
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns for formatted values
    """
    # Check that value_cols and p_value_cols have the same length
    if len(value_cols) != len(p_value_cols):
        raise ValueError("value_cols and p_value_cols must have the same length")
    
    # Set output columns to value_cols if not provided
    if output_cols is None:
        output_cols = [f"{col}_formatted" for col in value_cols]
    
    # Check that output_cols has the correct length
    if len(output_cols) != len(value_cols):
        raise ValueError("output_cols must have the same length as value_cols")
    
    # Create a copy of the DataFrame
    result_df = df.copy()
    
    # Add formatted columns
    for value_col, p_value_col, output_col in zip(value_cols, p_value_cols, output_cols):
        result_df[output_col] = result_df.apply(
            lambda row: format_with_significance(
                row[value_col],
                row[p_value_col],
                format_str
            ),
            axis=1
        )
    
    return result_df


def generate_significance_note(
    style: str = 'general',
    format: str = 'markdown'
) -> str:
    """
    Generate standard significance note for tables.
    
    Parameters
    ----------
    style : str, optional
        Style of significance note ('general', 'econometrica', 'aer', 'world_bank')
    format : str, optional
        Output format ('markdown', 'latex', 'html')
        
    Returns
    -------
    str
        Formatted significance note
    """
    # Define styles
    styles = {
        'general': [
            '* p<0.1', 
            '** p<0.05', 
            '*** p<0.01'
        ],
        'econometrica': [
            '* p<0.1', 
            '** p<0.05', 
            '*** p<0.01'
        ],
        'aer': [
            '* Significant at the 10 percent level', 
            '** Significant at the 5 percent level', 
            '*** Significant at the 1 percent level'
        ],
        'world_bank': [
            '* p<0.1', 
            '** p<0.05', 
            '*** p<0.01'
        ]
    }
    
    # Use general style if requested style not found
    if style not in styles:
        logger.warning(f"Style '{style}' not found, using 'general'")
        style = 'general'
    
    # Get significance notes for the selected style
    notes = styles[style]
    
    # Format notes according to output format
    if format == 'markdown':
        return '\n'.join(notes)
    
    elif format == 'latex':
        latex_notes = []
        for note in notes:
            latex_notes.append(note)
        return '\\\\'.join(latex_notes)
    
    elif format == 'html':
        html_notes = []
        for note in notes:
            html_notes.append(f"<span>{note}</span>")
        return '<br>'.join(html_notes)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def parse_latex_significance(latex_str: str) -> Tuple[float, str]:
    """
    Parse a LaTeX-formatted value with significance indicators.
    
    Parameters
    ----------
    latex_str : str
        LaTeX-formatted value with significance indicators
        
    Returns
    -------
    tuple
        (value, significance_indicator)
    """
    # Regular expression to match a LaTeX-formatted value with significance indicators
    # e.g., "2.345^{***}", "0.678^{**}", "1.234^{*}", "5.678"
    pattern = r'([+-]?\d+\.\d+)(?:\^\{([*]+)\})?'
    
    match = re.match(pattern, latex_str)
    if match:
        value_str, sig = match.groups()
        value = float(value_str)
        significance = sig if sig else ""
        return value, significance
    else:
        # Try to parse as a simple number
        try:
            value = float(latex_str)
            return value, ""
        except ValueError:
            logger.warning(f"Could not parse LaTeX string: {latex_str}")
            return np.nan, ""


def latex_to_significance_level(sig: str) -> float:
    """
    Convert LaTeX significance indicators to p-value level.
    
    Parameters
    ----------
    sig : str
        Significance indicator ('***', '**', '*', or '')
        
    Returns
    -------
    float
        Corresponding p-value level (0.01, 0.05, 0.1, or 1.0)
    """
    if sig == "***":
        return 0.01
    elif sig == "**":
        return 0.05
    elif sig == "*":
        return 0.1
    else:
        return 1.0


def format_p_value(
    p_value: float,
    format_str: str = "{:.4f}",
    threshold: float = 0.0001
) -> str:
    """
    Format p-value according to academic standards.
    
    Parameters
    ----------
    p_value : float
        p-value to format
    format_str : str, optional
        Format string for p-value
    threshold : float, optional
        Threshold for displaying p-value as "< [threshold]"
        
    Returns
    -------
    str
        Formatted p-value
    """
    if p_value < threshold:
        return f"<{threshold}"
    else:
        return format_str.format(p_value)


def significance_to_latex(sig: str) -> str:
    """
    Convert significance indicator to LaTeX format.
    
    Parameters
    ----------
    sig : str
        Significance indicator ('***', '**', '*', or '')
        
    Returns
    -------
    str
        LaTeX-formatted significance indicator
    """
    if not sig:
        return ""
    else:
        return f"^{{{sig}}}"


def significance_to_html(sig: str) -> str:
    """
    Convert significance indicator to HTML format.
    
    Parameters
    ----------
    sig : str
        Significance indicator ('***', '**', '*', or '')
        
    Returns
    -------
    str
        HTML-formatted significance indicator
    """
    if not sig:
        return ""
    else:
        return f"<sup>{sig}</sup>"