"""
Report generators for Yemen Market Analysis.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import datetime

from core.decorators import error_handler, performance_tracker
from core.exceptions import ReportingError
from .tables import (
    create_summary_table, create_conflict_summary_table,
    create_welfare_summary_table, format_summary_table_html
)

logger = logging.getLogger(__name__)


@error_handler(fallback_value='')
@performance_tracker()
def generate_html_summary_report(
    results_by_commodity: Dict[str, Dict[str, Any]],
    conflict_results_by_commodity: Optional[Dict[str, Dict[str, Any]]] = None,
    welfare_results_by_commodity: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "Market Integration Analysis Summary Report"
) -> str:
    """
    Generate HTML summary report for all commodities.
    
    Args:
        results_by_commodity: Dictionary with results keyed by commodity
        conflict_results_by_commodity: Optional conflict analysis results
        welfare_results_by_commodity: Optional welfare analysis results
        title: Report title
        
    Returns:
        HTML formatted report
    """
    if not results_by_commodity:
        return "<h1>No results available for report generation</h1>"
    
    # Create summary tables
    summary_df = create_summary_table(results_by_commodity)
    
    # Create conflict summary if available
    conflict_df = None
    if conflict_results_by_commodity:
        conflict_df = create_conflict_summary_table(conflict_results_by_commodity)
    
    # Create welfare summary if available
    welfare_df = None
    if welfare_results_by_commodity:
        welfare_df = create_welfare_summary_table(welfare_results_by_commodity)
    
    # Generate report date
    report_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Start HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .summary-table {{ margin-bottom: 40px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #777; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>Generated on: {report_date}</p>
        
        <h2>Market Integration Summary</h2>
"""
    
    # Add market integration summary table
    if not summary_df.empty:
        html += format_summary_table_html(
            summary_df,
            title="Threshold Model Parameters by Commodity",
            description="Key parameters from threshold cointegration analysis."
        )
    else:
        html += "<p>No market integration data available.</p>"
    
    # Add conflict analysis if available
    if conflict_df is not None and not conflict_df.empty:
        html += "<h2>Conflict Impact Analysis</h2>"
        html += format_summary_table_html(
            conflict_df,
            title="Conflict Sensitivity by Commodity",
            description="Analysis of market behavior under different conflict intensities."
        )
    
    # Add welfare analysis if available
    if welfare_df is not None and not welfare_df.empty:
        html += "<h2>Welfare Impact Analysis</h2>"
        html += format_summary_table_html(
            welfare_df,
            title="Welfare Effects by Commodity",
            description="Estimated welfare impacts of market fragmentation."
        )
    
    # Add footer and close HTML
    html += """
        <div class="footer">
            <p>Yemen Market Analysis - Threshold Cointegration Analysis</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html


@error_handler(fallback_value='')
@performance_tracker()
def generate_academic_results_report(
    results_by_commodity: Dict[str, Dict[str, Any]],
    title: str = "Formal Threshold Cointegration Results"
) -> str:
    """
    Generate formal academic-style results report.
    
    Args:
        results_by_commodity: Dictionary with results keyed by commodity
        title: Report title
        
    Returns:
        HTML formatted report with academic formatting
    """
    if not results_by_commodity:
        return "<h1>No results available for report generation</h1>"
    
    # Create summary table
    summary_df = create_summary_table(results_by_commodity)
    
    # Generate report date
    report_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Start HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Times New Roman', Times, serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #000; }}
        h1 {{ text-align: center; font-size: 18pt; }}
        h2 {{ font-size: 14pt; margin-top: 30px; }}
        h3 {{ font-size: 12pt; }}
        p {{ text-align: justify; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0 30px 0; font-size: 10pt; }}
        th, td {{ padding: 8px 10px; text-align: center; border: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .abstract {{ font-style: italic; margin: 20px 0; }}
        .footer {{ margin-top: 50px; padding-top: 20px; font-size: 10pt; text-align: center; }}
        .table-note {{ font-size: 9pt; text-align: left; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="abstract">
            This report presents the results of threshold cointegration analysis applied to market price data 
            in Yemen. The analysis implements Hansen & Seo (2002) and Enders & Siklos (2001) threshold models 
            to investigate spatial market integration and price transmission asymmetries.
        </p>
        
        <h2>1. Introduction</h2>
        <p>
            Market integration in conflict-affected economies is characterized by nonlinear price 
            transmission mechanisms and threshold effects. This report presents threshold cointegration 
            model results that capture these nonlinearities in the context of Yemen's fragmented markets.
        </p>
        
        <h2>2. Methodology</h2>
        <p>
            The analysis employs threshold vector error correction models (TVECM) that allow for 
            regime-dependent adjustment to long-run equilibrium. The threshold parameter represents 
            the minimum price differential required to trigger arbitrage between markets.
        </p>
        
        <h2>3. Results</h2>
        <p>
            Table 1 presents the estimated threshold parameters and adjustment coefficients for each commodity.
            The threshold values represent the minimum percentage price differential that must be exceeded 
            before arbitrage forces begin to restore equilibrium.
        </p>
        
        <h3>Table 1: Threshold Cointegration Results</h3>
"""
    
    # Add formatted results table
    if not summary_df.empty:
        # Format table columns for academic presentation
        academic_columns = [
            'threshold', 'p_value', 'threshold_significant',
            'alpha_down', 'alpha_up', 'half_life_down', 'half_life_up',
            'asymmetry_significant', 'integration_index'
        ]
        
        # Keep only columns that exist in the DataFrame
        academic_columns = [col for col in academic_columns if col in summary_df.columns]
        academic_df = summary_df[academic_columns]
        
        # Rename columns for academic presentation
        column_names = {
            'threshold': 'Threshold (γ)',
            'p_value': 'p-value',
            'threshold_significant': 'Significant',
            'alpha_down': 'α₁ (Lower)',
            'alpha_up': 'α₂ (Upper)',
            'half_life_down': 'Half-life₁',
            'half_life_up': 'Half-life₂',
            'asymmetry_significant': 'Asymmetry',
            'integration_index': 'Integration'
        }
        
        # Rename only columns that exist
        column_names = {col: name for col, name in column_names.items() if col in academic_df.columns}
        academic_df = academic_df.rename(columns=column_names)
        
        # Format for academic report
        styler = academic_df.style
        
        # Format numbers
        format_dict = {
            'Threshold (γ)': '{:.3f}',
            'p-value': '{:.3f}',
            'α₁ (Lower)': '{:.3f}',
            'α₂ (Upper)': '{:.3f}',
            'Half-life₁': '{:.1f}',
            'Half-life₂': '{:.1f}',
            'Integration': '{:.2f}'
        }
        
        # Apply formatting where columns exist
        format_dict = {col: fmt for col, fmt in format_dict.items() if col in academic_df.columns}
        styler = styler.format(format_dict)
        
        # Add the table to the report
        html += styler.to_html()
        
        # Add table notes
        html += """
        <p class="table-note">
            <i>Notes:</i> Threshold (γ) represents the percentage price differential required to trigger arbitrage.
            α₁ and α₂ are the adjustment parameters in the lower and upper regimes, respectively.
            Half-life values indicate the number of periods required to reduce deviation by 50%.
            The 'Significant' column indicates threshold effect significance at the 5% level,
            and 'Asymmetry' indicates significant differences between adjustment speeds.
        </p>
        """
    else:
        html += "<p>No market integration data available.</p>"
    
    # Add interpretation section
    html += """
        <h2>4. Interpretation</h2>
        <p>
            The results indicate varying degrees of market integration across commodities.
            The estimated thresholds represent the transaction costs and risk premiums that
            traders face when engaging in spatial arbitrage between markets. Higher thresholds
            suggest greater barriers to trade and less efficient market linkages.
        </p>
        <p>
            Asymmetric adjustment patterns, as indicated by differences between α₁ and α₂,
            suggest that prices adjust more rapidly to certain types of deviations from equilibrium.
            These asymmetries may reflect factors such as market power, information asymmetries,
            or physical constraints in the trading infrastructure.
        </p>
        
        <h2>5. Conclusion</h2>
        <p>
            The threshold cointegration analysis provides evidence of nonlinear price transmission
            in Yemen's markets. The significant threshold effects for several commodities confirm
            the presence of regime-dependent adjustment processes consistent with theoretical
            expectations in fragmented markets.
        </p>
        
        <div class="footer">
            <p>Yemen Market Analysis - Threshold Cointegration Analysis | Generated on: """ + report_date + """</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html


@error_handler(fallback_value='')
@performance_tracker()
def generate_policy_brief(
    results_by_commodity: Dict[str, Dict[str, Any]],
    policy_results_by_commodity: Dict[str, Dict[str, Any]],
    welfare_results_by_commodity: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "Policy Brief: Market Integration in Yemen"
) -> str:
    """
    Generate policy brief with recommendations.
    
    Args:
        results_by_commodity: Dictionary with results keyed by commodity
        policy_results_by_commodity: Dictionary with policy implications by commodity
        welfare_results_by_commodity: Optional welfare analysis results
        title: Brief title
        
    Returns:
        HTML formatted policy brief
    """
    if not results_by_commodity or not policy_results_by_commodity:
        return "<h1>No results available for policy brief generation</h1>"
    
    # Generate report date
    report_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Start HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .executive-summary {{ background-color: #f8f9fa; padding: 15px; border-left: 5px solid #3498db; margin: 20px 0; }}
        .recommendation {{ background-color: #f1f9f1; padding: 15px; border-left: 5px solid #2ecc71; margin: 15px 0; }}
        .priority-high {{ border-left-color: #e74c3c; }}
        .priority-medium {{ border-left-color: #f39c12; }}
        .priority-low {{ border-left-color: #2ecc71; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #777; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p><strong>Date:</strong> {report_date}</p>
        
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p>
                This policy brief summarizes the findings from market integration analysis in Yemen
                and provides actionable recommendations for improving market efficiency and reducing
                fragmentation. The analysis identifies varying degrees of market fragmentation across
                commodities, with significant implications for humanitarian response and economic policy.
            </p>
        </div>
        
        <h2>Key Findings</h2>
        <ul>
"""
    
    # Add key findings
    # Find commodities with highest and lowest integration
    integration_values = {
        commodity: results.get('integration', {}).get('integration_index', 0)
        for commodity, results in results_by_commodity.items()
    }
    
    if integration_values:
        most_integrated = max(integration_values.items(), key=lambda x: x[1])
        least_integrated = min(integration_values.items(), key=lambda x: x[1])
        
        html += f"""
            <li><strong>Market Fragmentation:</strong> Markets show varying degrees of integration, with
                {most_integrated[0]} markets showing the highest integration ({most_integrated[1]:.2f}) and
                {least_integrated[0]} markets showing the lowest ({least_integrated[1]:.2f}).</li>
        """
    
    # Add findings about price transmission
    html += """
            <li><strong>Price Transmission:</strong> Significant thresholds were detected, indicating that
                small price differences between markets may not trigger arbitrage due to high transaction costs.</li>
            <li><strong>Asymmetric Adjustment:</strong> Prices adjust at different speeds depending on whether
                they are above or below equilibrium, suggesting market power imbalances.</li>
    """
    
    # Add welfare findings if available
    if welfare_results_by_commodity:
        avg_dwl = np.mean([
            results.get('dwl_percent_of_market', 0)
            for results in welfare_results_by_commodity.values()
        ])
        
        html += f"""
            <li><strong>Welfare Losses:</strong> Market fragmentation leads to estimated welfare losses
                averaging {avg_dwl:.2f}% of market value across commodities.</li>
        """
    
    html += """
        </ul>
        
        <h2>Recommendations by Commodity</h2>
    """
    
    # Add recommendations by commodity
    for commodity, policy_results in policy_results_by_commodity.items():
        # Extract key info
        priority = policy_results.get('policy_priority', 'Medium')
        explanation = policy_results.get('explanation', '')
        recommendations = policy_results.get('recommendations', [])
        
        # Determine priority class
        priority_class = {
            'High': 'priority-high',
            'Medium-High': 'priority-high',
            'Medium': 'priority-medium',
            'Low': 'priority-low'
        }.get(priority, 'priority-medium')
        
        html += f"""
        <h3>{commodity}</h3>
        <div class="recommendation {priority_class}">
            <p><strong>Priority:</strong> {priority}</p>
            <p>{explanation}</p>
            <ul>
        """
        
        # Add each recommendation
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
            </ul>
        </div>
        """
    
    # Add implementation section
    html += """
        <h2>Implementation Considerations</h2>
        <p>
            Effective implementation of these recommendations requires coordination between
            humanitarian actors, local authorities, and market stakeholders. Interventions should
            be sequenced based on priority levels and should take into account seasonal factors,
            conflict dynamics, and existing market support programs.
        </p>
        
        <h3>Short-term Actions</h3>
        <ul>
            <li>Focus on reducing transaction costs for high-priority commodities through temporary market support</li>
            <li>Enhance market monitoring systems to detect changes in integration patterns</li>
            <li>Support trader financing in areas with significant arbitrage opportunities but limited capital</li>
        </ul>
        
        <h3>Medium-term Strategies</h3>
        <ul>
            <li>Invest in critical transportation infrastructure to reduce geographic barriers</li>
            <li>Develop information systems to address market information asymmetries</li>
            <li>Support policy dialogue on reducing administrative barriers to internal trade</li>
        </ul>
        
        <div class="footer">
            <p>Yemen Market Analysis - Threshold Cointegration Analysis</p>
            <p>For inquiries, please contact: [Contact Information]</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html