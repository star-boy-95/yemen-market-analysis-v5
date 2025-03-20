"""
Reporting module for Yemen Market Integration project.

This module provides functions for generating comprehensive reports and
executive summaries based on analysis results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from src.utils import handle_errors

# Create logger
logger = logging.getLogger(__name__)


@handle_errors(logger=logger)
def generate_comprehensive_report(all_results, commodity, output_path, logger):
    """
    Generate a comprehensive report of all analysis results.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all analysis results
    commodity : str
        Commodity name
    output_path : pathlib.Path or str
        Path to save the report
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pathlib.Path
        Path to the generated report
    """
    from src.models.interpretation import (
        interpret_unit_root_results,
        interpret_cointegration_results,
        interpret_threshold_results,
        interpret_spatial_results,
        interpret_simulation_results
    )
    
    # Ensure output_path is a Path object
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Generate interpretations
    interpretations = {}
    
    if 'unit_root_results' in all_results:
        interpretations['unit_root'] = interpret_unit_root_results(
            all_results['unit_root_results'], commodity
        )
    
    if 'cointegration_results' in all_results:
        interpretations['cointegration'] = interpret_cointegration_results(
            all_results['cointegration_results'], commodity
        )
    
    if 'threshold_results' in all_results:
        interpretations['threshold'] = interpret_threshold_results(
            all_results['threshold_results'], commodity
        )
    
    if 'spatial_results' in all_results:
        interpretations['spatial'] = interpret_spatial_results(
            all_results['spatial_results'], commodity
        )
    
    if 'simulation_results' in all_results:
        interpretations['simulation'] = interpret_simulation_results(
            all_results['simulation_results'], commodity
        )
    
    # Create report path
    report_path = output_path / f'{commodity.replace(" ", "_")}_comprehensive_report.md'
    
    # Write report
    with open(report_path, 'w') as f:
        f.write(f"# Yemen Market Integration Analysis: {commodity.capitalize()}\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Add key findings from interpretations
        if 'unit_root' in interpretations:
            f.write(f"- {interpretations['unit_root']['summary']}\n")
        
        if 'cointegration' in interpretations:
            f.write(f"- {interpretations['cointegration']['summary']}\n")
        
        if 'threshold' in interpretations:
            f.write(f"- {interpretations['threshold']['summary']}\n")
        
        if 'spatial' in interpretations:
            f.write(f"- {interpretations['spatial']['summary']}\n")
        
        if 'simulation' in interpretations:
            f.write(f"- {interpretations['simulation']['summary']}\n")
        
        # Add policy recommendations
        f.write("\n## Policy Recommendations\n\n")
        
        if 'simulation' in interpretations and 'policy_recommendations' in interpretations['simulation']:
            for recommendation in interpretations['simulation']['policy_recommendations']:
                f.write(f"- {recommendation}\n")
        
        # Add detailed results sections
        f.write("\n## Detailed Analysis Results\n\n")
        
        # Unit Root Analysis
        f.write("### Time Series Properties\n\n")
        if 'unit_root' in interpretations:
            f.write(f"{interpretations['unit_root']['summary']}\n\n")
            
            if 'details' in interpretations['unit_root']:
                for key, value in interpretations['unit_root']['details'].items():
                    f.write(f"- {value}\n")
            
            f.write("\n**Implications:**\n\n")
            for implication in interpretations['unit_root']['implications']:
                f.write(f"- {implication}\n")
        else:
            f.write("Unit root analysis not performed or insufficient data.\n")
        
        # Cointegration Analysis
        f.write("\n### Market Integration Analysis\n\n")
        if 'cointegration' in interpretations:
            f.write(f"{interpretations['cointegration']['summary']}\n\n")
            
            if 'details' in interpretations['cointegration']:
                for key, value in interpretations['cointegration']['details'].items():
                    f.write(f"- {value}\n")
            
            f.write("\n**Implications:**\n\n")
            for implication in interpretations['cointegration']['implications']:
                f.write(f"- {implication}\n")
        else:
            f.write("Cointegration analysis not performed or insufficient data.\n")
        
        # Threshold Analysis
        f.write("\n### Transaction Cost Analysis\n\n")
        if 'threshold' in interpretations:
            f.write(f"{interpretations['threshold']['summary']}\n\n")
            
            if 'details' in interpretations['threshold']:
                for key, value in interpretations['threshold']['details'].items():
                    if key == 'threshold':
                        f.write(f"- Estimated transaction cost threshold: {value:.4f}\n")
                    elif key == 'adjustment_below':
                        f.write(f"- Adjustment speed below threshold: {value:.4f}\n")
                    elif key == 'adjustment_above':
                        f.write(f"- Adjustment speed above threshold: {value:.4f}\n")
                    elif key == 'half_life_below':
                        f.write(f"- Half-life below threshold: {value:.2f} periods\n")
                    elif key == 'half_life_above':
                        f.write(f"- Half-life above threshold: {value:.2f} periods\n")
                    else:
                        f.write(f"- {value}\n")
            
            f.write("\n**Implications:**\n\n")
            for implication in interpretations['threshold']['implications']:
                f.write(f"- {implication}\n")
        else:
            f.write("Threshold analysis not performed or insufficient data.\n")
        
        # Spatial Analysis
        f.write("\n### Spatial Market Analysis\n\n")
        if 'spatial' in interpretations:
            f.write(f"{interpretations['spatial']['summary']}\n\n")
            
            if 'details' in interpretations['spatial']:
                for key, value in interpretations['spatial']['details'].items():
                    if key == 'morans_i':
                        f.write(f"- Moran's I statistic: {value:.4f}\n")
                    elif key == 'morans_p':
                        f.write(f"- Moran's I p-value: {value:.4f}\n")
                    elif key == 'spatial_dependence':
                        f.write(f"- Spatial dependence parameter: {value:.4f}\n")
                    elif key == 'model_fit':
                        f.write(f"- Spatial model R-squared: {value:.4f}\n")
                    else:
                        f.write(f"- {value}\n")
            
            f.write("\n**Implications:**\n\n")
            for implication in interpretations['spatial']['implications']:
                f.write(f"- {implication}\n")
        else:
            f.write("Spatial analysis not performed or insufficient data.\n")
        
        # Policy Simulation
        f.write("\n### Policy Simulation Results\n\n")
        if 'simulation' in interpretations:
            f.write(f"{interpretations['simulation']['summary']}\n\n")
            
            if 'welfare_effects' in interpretations['simulation']:
                welfare = interpretations['simulation']['welfare_effects']
                
                if 'best_policy' in welfare:
                    f.write(f"- Best policy option: {welfare['best_policy']}\n")
                
                if 'welfare_gain' in welfare:
                    f.write(f"- Estimated welfare gain: {welfare['welfare_gain']:.2f}\n")
                
                if 'distributional' in welfare:
                    dist = welfare['distributional']
                    
                    if 'gini_change' in dist and dist['gini_change'] is not None:
                        f.write(f"- Change in inequality (Gini): {dist['gini_change']:.4f}\n")
                    
                    if 'bottom_quintile_impact' in dist and dist['bottom_quintile_impact'] is not None:
                        f.write(f"- Price impact on bottom quintile: {dist['bottom_quintile_impact']:.2f}%\n")
                    
                    if 'food_security_improvement' in dist and dist['food_security_improvement'] is not None:
                        f.write(f"- Food security improvement: {dist['food_security_improvement']:.2f}%\n")
            
            f.write("\n**Implementation Considerations:**\n\n")
            for consideration in interpretations['simulation']['implementation_considerations']:
                f.write(f"- {consideration}\n")
        else:
            f.write("Policy simulation not performed or insufficient data.\n")
        
        # Methodology
        f.write("\n## Methodology\n\n")
        f.write("This analysis employed the following econometric methods:\n\n")
        f.write("1. **Unit Root Testing**: Augmented Dickey-Fuller (ADF), KPSS, and Zivot-Andrews tests to determine time series properties and detect structural breaks.\n")
        f.write("2. **Cointegration Analysis**: Engle-Granger, Johansen, and Gregory-Hansen tests to assess long-run equilibrium relationships between markets.\n")
        f.write("3. **Threshold Cointegration**: Threshold Vector Error Correction Models (TVECM) to estimate transaction costs and asymmetric price adjustment.\n")
        f.write("4. **Spatial Econometrics**: Spatial autocorrelation tests and spatial regression models to analyze geographic patterns of market integration.\n")
        f.write("5. **Policy Simulation**: Counterfactual analysis of exchange rate unification, conflict reduction, and combined policies.\n")
    
    logger.info(f"Comprehensive report generated at {report_path}")
    
    return report_path


@handle_errors(logger=logger)
def create_executive_summary(all_results, commodity, output_path, logger):
    """
    Create a concise executive summary of key findings.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all analysis results
    commodity : str
        Commodity name
    output_path : pathlib.Path or str
        Path to save the summary
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pathlib.Path
        Path to the generated summary
    """
    from src.models.interpretation import (
        interpret_unit_root_results,
        interpret_cointegration_results,
        interpret_threshold_results,
        interpret_spatial_results,
        interpret_simulation_results
    )
    
    # Ensure output_path is a Path object
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Generate interpretations
    interpretations = {}
    
    if 'unit_root_results' in all_results:
        interpretations['unit_root'] = interpret_unit_root_results(
            all_results['unit_root_results'], commodity
        )
    
    if 'cointegration_results' in all_results:
        interpretations['cointegration'] = interpret_cointegration_results(
            all_results['cointegration_results'], commodity
        )
    
    if 'threshold_results' in all_results:
        interpretations['threshold'] = interpret_threshold_results(
            all_results['threshold_results'], commodity
        )
    
    if 'spatial_results' in all_results:
        interpretations['spatial'] = interpret_spatial_results(
            all_results['spatial_results'], commodity
        )
    
    if 'simulation_results' in all_results:
        interpretations['simulation'] = interpret_simulation_results(
            all_results['simulation_results'], commodity
        )
    
    # Create summary path
    summary_path = output_path / f'{commodity.replace(" ", "_")}_executive_summary.md'
    
    # Write summary
    with open(summary_path, 'w') as f:
        f.write(f"# Executive Summary: {commodity.capitalize()} Market Integration in Yemen\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Add key findings from interpretations
        if 'unit_root' in interpretations:
            f.write(f"- **Time Series Properties**: {interpretations['unit_root']['summary']}\n")
        
        if 'cointegration' in interpretations:
            f.write(f"- **Market Integration**: {interpretations['cointegration']['summary']}\n")
        
        if 'threshold' in interpretations:
            f.write(f"- **Transaction Costs**: {interpretations['threshold']['summary']}\n")
        
        if 'spatial' in interpretations:
            f.write(f"- **Spatial Patterns**: {interpretations['spatial']['summary']}\n")
        
        # Add policy recommendations
        f.write("\n## Policy Recommendations\n\n")
        
        if 'simulation' in interpretations and 'policy_recommendations' in interpretations['simulation']:
            # Get top 3 recommendations
            top_recommendations = interpretations['simulation']['policy_recommendations'][:3]
            for recommendation in top_recommendations:
                f.write(f"- {recommendation}\n")
        
        # Add welfare effects
        f.write("\n## Expected Policy Impacts\n\n")
        
        if 'simulation' in interpretations and 'welfare_effects' in interpretations['simulation']:
            welfare = interpretations['simulation']['welfare_effects']
            
            if 'best_policy' in welfare:
                f.write(f"- **Recommended Approach**: {welfare['best_policy']}\n")
            
            if 'distributional' in welfare:
                dist = welfare['distributional']
                
                if 'food_security_improvement' in dist and dist['food_security_improvement'] is not None:
                    f.write(f"- **Food Security Impact**: {dist['food_security_improvement']:.1f}% improvement expected\n")
                
                if 'bottom_quintile_impact' in dist and dist['bottom_quintile_impact'] is not None:
                    f.write(f"- **Impact on Vulnerable Populations**: {abs(dist['bottom_quintile_impact']):.1f}% {'decrease' if dist['bottom_quintile_impact'] < 0 else 'increase'} in prices for poorest households\n")
    
    logger.info(f"Executive summary generated at {summary_path}")
    
    return summary_path


@handle_errors(logger=logger)
def export_results_for_publication(all_results, commodity, output_path, logger):
    """
    Format results for academic publication.
    
    Parameters
    ----------
    all_results : dict
        Dictionary containing all analysis results
    commodity : str
        Commodity name
    output_path : pathlib.Path or str
        Path to save the formatted results
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    pathlib.Path
        Path to the generated publication file
    """
    # Ensure output_path is a Path object
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # Create publication path
    publication_path = output_path / f'{commodity.replace(" ", "_")}_publication_results.tex'
    
    # Write LaTeX formatted results
    with open(publication_path, 'w') as f:
        f.write("\\section{Empirical Results}\n\n")
        
        # Unit Root Results
        f.write("\\subsection{Time Series Properties}\n\n")
        if 'unit_root_results' in all_results:
            unit_root = all_results['unit_root_results']
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Unit Root Test Results}\n")
            f.write("\\label{tab:unit_root}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\hline\n")
            f.write("Series & ADF Test & KPSS Test & Zivot-Andrews Test \\\\\n")
            f.write("& (p-value) & (p-value) & (p-value) \\\\\n")
            f.write("\\hline\n")
            
            # North series
            if 'north' in unit_root:
                north_adf = unit_root['north'].get('adf', {})
                north_kpss = unit_root['north'].get('kpss', {})
                north_za = unit_root['north'].get('zivot_andrews', {})
                
                f.write(f"North {commodity} & {north_adf.get('p_value', 'N/A'):.4f} & {north_kpss.get('p_value', 'N/A'):.4f} & {north_za.get('p_value', 'N/A'):.4f} \\\\\n")
            
            # South series
            if 'south' in unit_root:
                south_adf = unit_root['south'].get('adf', {})
                south_kpss = unit_root['south'].get('kpss', {})
                south_za = unit_root['south'].get('zivot_andrews', {})
                
                f.write(f"South {commodity} & {south_adf.get('p_value', 'N/A'):.4f} & {south_kpss.get('p_value', 'N/A'):.4f} & {south_za.get('p_value', 'N/A'):.4f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\small\n")
            f.write("\\item Note: ADF and Zivot-Andrews test the null hypothesis of a unit root. KPSS tests the null hypothesis of stationarity.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{table}\n\n")
        
        # Cointegration Results
        f.write("\\subsection{Cointegration Analysis}\n\n")
        if 'cointegration_results' in all_results:
            coint = all_results['cointegration_results']
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Cointegration Test Results}\n")
            f.write("\\label{tab:cointegration}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Test & Statistic & p-value \\\\\n")
            f.write("\\hline\n")
            
            # Engle-Granger
            if 'engle_granger' in coint:
                eg = coint['engle_granger']
                f.write(f"Engle-Granger & {eg.get('statistic', 'N/A'):.4f} & {eg.get('p_value', 'N/A'):.4f} \\\\\n")
            
            # Johansen
            if 'johansen' in coint:
                jo = coint['johansen']
                if 'trace_stat' in jo and 'trace_pval' in jo and len(jo['trace_stat']) > 0 and len(jo['trace_pval']) > 0:
                    f.write(f"Johansen (trace) & {jo['trace_stat'][0]:.4f} & {jo['trace_pval'][0]:.4f} \\\\\n")
            
            # Gregory-Hansen
            if 'gregory_hansen' in coint:
                gh = coint['gregory_hansen']
                f.write(f"Gregory-Hansen & {gh.get('statistic', 'N/A'):.4f} & {gh.get('p_value', 'N/A'):.4f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\small\n")
            f.write("\\item Note: All tests have the null hypothesis of no cointegration.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{table}\n\n")
        
        # Threshold Results
        f.write("\\subsection{Threshold Cointegration Analysis}\n\n")
        if 'threshold_results' in all_results and 'tvecm' in all_results['threshold_results']:
            tvecm = all_results['threshold_results']['tvecm']
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Threshold Vector Error Correction Model Results}\n")
            f.write("\\label{tab:threshold}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Parameter & Below Threshold & Above Threshold \\\\\n")
            f.write("\\hline\n")
            
            if 'threshold' in tvecm:
                f.write(f"Threshold & \\multicolumn{{2}}{{c}}{{{tvecm['threshold']:.4f}}} \\\\\n")
            
            if 'adjustment_below_1' in tvecm and 'adjustment_above_1' in tvecm:
                f.write(f"Adjustment Speed (North) & {tvecm['adjustment_below_1']:.4f} & {tvecm['adjustment_above_1']:.4f} \\\\\n")
            
            if 'adjustment_below_2' in tvecm and 'adjustment_above_2' in tvecm:
                f.write(f"Adjustment Speed (South) & {tvecm['adjustment_below_2']:.4f} & {tvecm['adjustment_above_2']:.4f} \\\\\n")
            
            # Calculate half-lives if adjustment parameters are available
            if 'adjustment_below_1' in tvecm and 'adjustment_above_1' in tvecm:
                adj_below = tvecm['adjustment_below_1']
                adj_above = tvecm['adjustment_above_1']
                
                if adj_below != 0:
                    half_life_below = np.log(0.5) / np.log(1 + abs(adj_below))
                    f.write(f"Half-life (North) & {half_life_below:.2f} & ")
                else:
                    f.write(f"Half-life (North) & $\\infty$ & ")
                    
                if adj_above != 0:
                    half_life_above = np.log(0.5) / np.log(1 + abs(adj_above))
                    f.write(f"{half_life_above:.2f} \\\\\n")
                else:
                    f.write(f"$\\infty$ \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\small\n")
            f.write("\\item Note: Adjustment speeds represent the rate at which deviations from equilibrium are corrected. Half-life is measured in periods.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{table}\n\n")
        
        # Spatial Results
        f.write("\\subsection{Spatial Econometric Analysis}\n\n")
        if 'spatial_results' in all_results:
            spatial = all_results['spatial_results']
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Spatial Econometric Results}\n")
            f.write("\\label{tab:spatial}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Parameter & Spatial Lag Model & Spatial Error Model \\\\\n")
            f.write("\\hline\n")
            
            if 'global_moran' in spatial and spatial['global_moran']:
                moran = spatial['global_moran']
                f.write(f"Moran's I & \\multicolumn{{2}}{{c}}{{{moran.get('I', 'N/A'):.4f} ({moran.get('p', 'N/A'):.4f})}} \\\\\n")
            
            if 'lag_model' in spatial and spatial['lag_model']:
                lag = spatial['lag_model']
                rho = getattr(lag, 'rho', 'N/A')
                r2 = getattr(lag, 'r2', 'N/A')
                
                if rho != 'N/A':
                    f.write(f"Spatial Dependence & {rho:.4f} & ")
                else:
                    f.write(f"Spatial Dependence & N/A & ")
                
                if 'error_model' in spatial and spatial['error_model']:
                    error = spatial['error_model']
                    lambda_val = getattr(error, 'lambda_', 'N/A')
                    if lambda_val != 'N/A':
                        f.write(f"{lambda_val:.4f} \\\\\n")
                    else:
                        f.write("N/A \\\\\n")
                else:
                    f.write("-- \\\\\n")
                
                if r2 != 'N/A':
                    f.write(f"R-squared & {r2:.4f} & ")
                else:
                    f.write(f"R-squared & N/A & ")
                
                if 'error_model' in spatial and spatial['error_model']:
                    error = spatial['error_model']
                    error_r2 = getattr(error, 'r2', 'N/A')
                    if error_r2 != 'N/A':
                        f.write(f"{error_r2:.4f} \\\\\n")
                    else:
                        f.write("N/A \\\\\n")
                else:
                    f.write("-- \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\small\n")
            f.write("\\item Note: Spatial dependence parameters measure the influence of neighboring markets on price formation.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{table}\n\n")
        
        # Simulation Results
        f.write("\\subsection{Policy Simulation Results}\n\n")
        if 'simulation_results' in all_results and 'welfare' in all_results['simulation_results']:
            welfare = all_results['simulation_results']['welfare']
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Policy Simulation Results}\n")
            f.write("\\label{tab:simulation}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\hline\n")
            f.write("Policy & Welfare Gain & Price Convergence & Food Security Impact \\\\\n")
            f.write("\\hline\n")
            
            # Exchange rate policies
            if 'exchange_rate' in welfare:
                for target, results in welfare['exchange_rate'].items():
                    for commodity_name, effects in results.items():
                        if commodity_name == commodity:
                            welfare_gain = effects.get('total_welfare', 'N/A')
                            price_conv = effects.get('price_convergence', {}).get('relative_convergence', 'N/A')
                            food_sec = effects.get('distributional', {}).get('food_security_improvement', 'N/A')
                            
                            if welfare_gain != 'N/A':
                                f.write(f"Exchange Rate ({target}) & {welfare_gain:.2f} & ")
                            else:
                                f.write(f"Exchange Rate ({target}) & N/A & ")
                                
                            if price_conv != 'N/A':
                                f.write(f"{price_conv:.2f}\\% & ")
                            else:
                                f.write(f"N/A & ")
                                
                            if food_sec != 'N/A':
                                f.write(f"{food_sec:.2f}\\% \\\\\n")
                            else:
                                f.write(f"N/A \\\\\n")
            
            # Connectivity policies
            if 'connectivity' in welfare:
                for scenario, results in welfare['connectivity'].items():
                    for commodity_name, effects in results.items():
                        if commodity_name == commodity:
                            reduction_pct = scenario.split('_')[-1]
                            welfare_gain = effects.get('total_welfare', 'N/A')
                            price_conv = effects.get('price_convergence', {}).get('relative_convergence', 'N/A')
                            food_sec = effects.get('distributional', {}).get('food_security_improvement', 'N/A')
                            
                            if welfare_gain != 'N/A':
                                f.write(f"Conflict Reduction ({reduction_pct}) & {welfare_gain:.2f} & ")
                            else:
                                f.write(f"Conflict Reduction ({reduction_pct}) & N/A & ")
                                
                            if price_conv != 'N/A':
                                f.write(f"{price_conv:.2f}\\% & ")
                            else:
                                f.write(f"N/A & ")
                                
                            if food_sec != 'N/A':
                                f.write(f"{food_sec:.2f}\\% \\\\\n")
                            else:
                                f.write(f"N/A \\\\\n")
            
            # Combined policies
            if 'combined' in welfare:
                for scenario, results in welfare['combined'].items():
                    for commodity_name, effects in results.items():
                        if commodity_name == commodity:
                            welfare_gain = effects.get('total_welfare', 'N/A')
                            price_conv = effects.get('price_convergence', {}).get('relative_convergence', 'N/A')
                            food_sec = effects.get('distributional', {}).get('food_security_improvement', 'N/A')
                            
                            if welfare_gain != 'N/A':
                                f.write(f"Combined ({scenario}) & {welfare_gain:.2f} & ")
                            else:
                                f.write(f"Combined ({scenario}) & N/A & ")
                                
                            if price_conv != 'N/A':
                                f.write(f"{price_conv:.2f}\\% & ")
                            else:
                                f.write(f"N/A & ")
                                
                            if food_sec != 'N/A':
                                f.write(f"{food_sec:.2f}\\% \\\\\n")
                            else:
                                f.write(f"N/A \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\small\n")
            f.write("\\item Note: Welfare gain is measured as a composite index. Price convergence shows the percentage reduction in price differentials. Food security impact represents the estimated percentage improvement in food security indicators.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{table}\n\n")
    
    logger.info(f"Publication-ready results exported to {publication_path}")
    
    return publication_path