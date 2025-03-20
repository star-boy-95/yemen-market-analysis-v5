"""
Result interpretation module for Yemen Market Integration project.

This module provides functions for interpreting econometric analysis results
and generating insights for policy recommendations.
"""

import numpy as np
from datetime import datetime
import logging
from src.utils import handle_errors

# Create logger
logger = logging.getLogger(__name__)


@handle_errors(logger=logger)
def interpret_unit_root_results(unit_root_results, commodity):
    """
    Provide contextual interpretation of unit root test results.
    
    Parameters
    ----------
    unit_root_results : dict
        Results from unit root testing
    commodity : str
        Commodity name
        
    Returns
    -------
    dict
        Interpretation of unit root results with economic insights
    """
    interpretation = {
        'commodity': commodity,
        'summary': '',
        'details': {},
        'implications': []
    }
    
    # Check if we have valid results
    if not unit_root_results:
        interpretation['summary'] = "Insufficient data for unit root analysis."
        return interpretation
    
    # Extract integration orders
    north_order = unit_root_results.get('north', {}).get('integration_order', None)
    south_order = unit_root_results.get('south', {}).get('integration_order', None)
    
    if north_order is None or south_order is None:
        interpretation['summary'] = "Incomplete unit root analysis results."
        return interpretation
    
    # Interpret integration orders
    if north_order == 1 and south_order == 1:
        interpretation['summary'] = f"Both north and south {commodity} price series are integrated of order 1 (I(1)), indicating non-stationary price levels but stationary price changes."
        interpretation['implications'].append("The I(1) nature of both series makes them suitable for cointegration analysis.")
        interpretation['implications'].append("Price shocks have permanent effects, suggesting market inefficiencies or structural barriers.")
    elif north_order == 0 and south_order == 0:
        interpretation['summary'] = f"Both north and south {commodity} price series are integrated of order 0 (I(0)), indicating stationary price levels."
        interpretation['implications'].append("The I(0) nature of both series suggests that price shocks are temporary and markets quickly return to equilibrium.")
        interpretation['implications'].append("Standard cointegration analysis is not applicable, but direct correlation analysis can be performed.")
    else:
        interpretation['summary'] = f"The north {commodity} price series is I({north_order}) while the south series is I({south_order}), indicating different time series properties."
        interpretation['implications'].append("The different integration orders suggest structural differences between north and south markets.")
        interpretation['implications'].append("Standard cointegration analysis may not be appropriate without further transformations.")
    
    # Check for structural breaks
    north_za = unit_root_results.get('north', {}).get('zivot_andrews', {})
    south_za = unit_root_results.get('south', {}).get('zivot_andrews', {})
    
    if north_za.get('stationary', False) or south_za.get('stationary', False):
        interpretation['details']['structural_breaks'] = "Structural breaks detected in the price series."
        
        if 'breakpoint' in north_za and 'merged_data' in unit_root_results and 'date' in unit_root_results['merged_data']:
            try:
                breakpoint_date = unit_root_results['merged_data']['date'].iloc[north_za['breakpoint']]
                interpretation['details']['north_break'] = f"North market experienced a structural break around {breakpoint_date.strftime('%Y-%m-%d')}."
            except (IndexError, AttributeError):
                interpretation['details']['north_break'] = "North market experienced a structural break, but the date could not be determined."
        
        if 'breakpoint' in south_za and 'merged_data' in unit_root_results and 'date' in unit_root_results['merged_data']:
            try:
                breakpoint_date = unit_root_results['merged_data']['date'].iloc[south_za['breakpoint']]
                interpretation['details']['south_break'] = f"South market experienced a structural break around {breakpoint_date.strftime('%Y-%m-%d')}."
            except (IndexError, AttributeError):
                interpretation['details']['south_break'] = "South market experienced a structural break, but the date could not be determined."
        
        interpretation['implications'].append("The presence of structural breaks suggests significant market disruptions, possibly due to conflict events, policy changes, or supply chain disruptions.")
        interpretation['implications'].append("Analysis should account for these breaks to avoid biased results.")
    
    return interpretation


@handle_errors(logger=logger)
def interpret_cointegration_results(cointegration_results, commodity):
    """
    Provide contextual interpretation of cointegration test results.
    
    Parameters
    ----------
    cointegration_results : dict
        Results from cointegration testing
    commodity : str
        Commodity name
        
    Returns
    -------
    dict
        Interpretation of cointegration results with economic insights
    """
    interpretation = {
        'commodity': commodity,
        'summary': '',
        'details': {},
        'implications': []
    }
    
    # Check if we have valid results
    if not cointegration_results:
        interpretation['summary'] = "Insufficient data for cointegration analysis."
        return interpretation
    
    # Extract test results
    eg_result = cointegration_results.get('engle_granger', {})
    jo_result = cointegration_results.get('johansen', {})
    gh_result = cointegration_results.get('gregory_hansen', {})
    
    # Determine overall cointegration status
    is_cointegrated = (eg_result.get('cointegrated', False) or 
                       jo_result.get('rank_trace', 0) > 0 or 
                       gh_result.get('cointegrated', False))
    
    if is_cointegrated:
        interpretation['summary'] = f"North and south {commodity} markets show evidence of cointegration, indicating a long-run equilibrium relationship despite short-term deviations."
        
        # Add details about which tests show cointegration
        cointegration_tests = []
        if eg_result.get('cointegrated', False):
            cointegration_tests.append("Engle-Granger")
        if jo_result.get('rank_trace', 0) > 0:
            cointegration_tests.append("Johansen")
        if gh_result.get('cointegrated', False):
            cointegration_tests.append("Gregory-Hansen")
        
        interpretation['details']['cointegration_tests'] = f"Cointegration detected by the following tests: {', '.join(cointegration_tests)}."
        
        # Add implications
        interpretation['implications'].append("The presence of cointegration suggests that despite conflict barriers, markets maintain long-run price relationships.")
        interpretation['implications'].append("Price deviations between markets are temporary, with economic forces driving prices back to equilibrium over time.")
        interpretation['implications'].append("Market integration exists, though the speed of adjustment may be affected by conflict and transaction costs.")
    else:
        interpretation['summary'] = f"North and south {commodity} markets do not show strong evidence of cointegration, suggesting market fragmentation."
        
        # Add details about which tests failed to show cointegration
        failed_tests = []
        if not eg_result.get('cointegrated', False):
            failed_tests.append("Engle-Granger")
        if jo_result.get('rank_trace', 0) == 0:
            failed_tests.append("Johansen")
        if not gh_result.get('cointegrated', False):
            failed_tests.append("Gregory-Hansen")
        
        interpretation['details']['failed_tests'] = f"Cointegration not detected by the following tests: {', '.join(failed_tests)}."
        
        # Add implications
        interpretation['implications'].append("The lack of cointegration suggests that conflict barriers have effectively fragmented markets.")
        interpretation['implications'].append("Price deviations between markets can persist indefinitely, indicating inefficient resource allocation.")
        interpretation['implications'].append("Policy interventions may be needed to restore market integration.")
    
    # Check for structural breaks in cointegration
    if gh_result.get('cointegrated', False):
        interpretation['details']['structural_break'] = "Cointegration with structural break detected."
        
        if 'breakpoint' in gh_result and 'merged_data' in cointegration_results and 'date' in cointegration_results['merged_data']:
            try:
                breakpoint_date = cointegration_results['merged_data']['date'].iloc[gh_result['breakpoint']]
                interpretation['details']['break_date'] = f"Cointegration relationship changed around {breakpoint_date.strftime('%Y-%m-%d')}."
            except (IndexError, AttributeError):
                interpretation['details']['break_date'] = "Cointegration relationship changed, but the date could not be determined."
        
        interpretation['implications'].append("The nature of market integration changed at some point, possibly due to conflict events or policy changes.")
        interpretation['implications'].append("Analysis should account for this structural change in the cointegration relationship.")
    
    return interpretation


@handle_errors(logger=logger)
def interpret_threshold_results(threshold_results, commodity):
    """
    Provide economic interpretation of threshold cointegration results.
    
    Parameters
    ----------
    threshold_results : dict
        Results from threshold cointegration analysis
    commodity : str
        Commodity name
        
    Returns
    -------
    dict
        Interpretation of threshold results with economic insights
    """
    interpretation = {
        'commodity': commodity,
        'summary': '',
        'details': {},
        'implications': []
    }
    
    # Check if we have valid results
    if not threshold_results or 'tvecm' not in threshold_results:
        interpretation['summary'] = "Insufficient data for threshold analysis."
        return interpretation
    
    tvecm_result = threshold_results['tvecm']
    
    # Check if threshold was found
    if 'threshold' not in tvecm_result:
        interpretation['summary'] = f"No significant threshold effect detected for {commodity} markets."
        interpretation['implications'].append("The absence of a threshold suggests that price adjustment is linear, without distinct regimes.")
        return interpretation
    
    # Extract threshold and adjustment parameters
    threshold = tvecm_result['threshold']
    
    interpretation['summary'] = f"A threshold effect was detected for {commodity} markets with a threshold value of {threshold:.4f}."
    interpretation['details']['threshold'] = threshold
    
    # Interpret threshold value
    interpretation['details']['threshold_interpretation'] = f"The threshold of {threshold:.4f} represents the price differential at which the adjustment mechanism changes. This can be interpreted as the transaction cost or barrier to arbitrage between north and south markets."
    
    # Check if we have adjustment parameters
    if 'adjustment_below_1' in tvecm_result and 'adjustment_above_1' in tvecm_result:
        adj_below = tvecm_result['adjustment_below_1']
        adj_above = tvecm_result['adjustment_above_1']
        
        interpretation['details']['adjustment_below'] = adj_below
        interpretation['details']['adjustment_above'] = adj_above
        
        # Calculate half-lives
        if adj_below != 0:
            half_life_below = np.log(0.5) / np.log(1 + abs(adj_below))
            interpretation['details']['half_life_below'] = half_life_below
        else:
            interpretation['details']['half_life_below'] = float('inf')
            
        if adj_above != 0:
            half_life_above = np.log(0.5) / np.log(1 + abs(adj_above))
            interpretation['details']['half_life_above'] = half_life_above
        else:
            interpretation['details']['half_life_above'] = float('inf')
        
        # Interpret adjustment speeds
        if abs(adj_below) < abs(adj_above):
            interpretation['details']['adjustment_pattern'] = "The adjustment speed is faster when price deviations exceed the threshold."
            interpretation['implications'].append("Large price differentials between north and south markets are corrected more quickly, while small deviations persist longer.")
            interpretation['implications'].append("This suggests that arbitrage only becomes profitable when price differences are large enough to overcome transaction costs.")
        elif abs(adj_below) > abs(adj_above):
            interpretation['details']['adjustment_pattern'] = "The adjustment speed is faster when price deviations are below the threshold."
            interpretation['implications'].append("Small price differentials between north and south markets are corrected quickly, while large deviations persist longer.")
            interpretation['implications'].append("This unusual pattern may indicate institutional barriers or conflict-related constraints that make large price differences difficult to arbitrage away.")
        else:
            interpretation['details']['adjustment_pattern'] = "The adjustment speeds are similar above and below the threshold."
            interpretation['implications'].append("Price deviations are corrected at similar rates regardless of their size, suggesting symmetric price transmission.")
    
    # Check if we have M-TAR results
    if 'mtar' in threshold_results:
        mtar_result = threshold_results['mtar']
        
        if mtar_result.get('asymmetric', False):
            interpretation['details']['directional_asymmetry'] = "Directional asymmetry detected in price adjustment."
            
            if 'adjustment_positive' in mtar_result and 'adjustment_negative' in mtar_result:
                adj_pos = mtar_result['adjustment_positive']
                adj_neg = mtar_result['adjustment_negative']
                
                interpretation['details']['adjustment_positive'] = adj_pos
                interpretation['details']['adjustment_negative'] = adj_neg
                
                if abs(adj_pos) < abs(adj_neg):
                    interpretation['details']['directional_pattern'] = "Price decreases are transmitted more quickly than price increases."
                    interpretation['implications'].append("The faster adjustment to price decreases suggests competitive market behavior.")
                elif abs(adj_pos) > abs(adj_neg):
                    interpretation['details']['directional_pattern'] = "Price increases are transmitted more quickly than price decreases."
                    interpretation['implications'].append("The faster adjustment to price increases suggests market power or information asymmetries.")
    
    # Add policy implications
    interpretation['implications'].append(f"The threshold of {threshold:.4f} provides a quantitative measure of transaction costs between markets that could be targeted by policy interventions.")
    interpretation['implications'].append("Reducing conflict barriers and transaction costs could lower this threshold and improve market integration.")
    
    return interpretation


@handle_errors(logger=logger)
def interpret_spatial_results(spatial_results, commodity):
    """
    Provide economic interpretation of spatial econometric results.
    
    Parameters
    ----------
    spatial_results : dict
        Results from spatial econometric analysis
    commodity : str
        Commodity name
        
    Returns
    -------
    dict
        Interpretation of spatial results with economic insights
    """
    interpretation = {
        'commodity': commodity,
        'summary': '',
        'details': {},
        'implications': []
    }
    
    # Check if we have valid results
    if not spatial_results:
        interpretation['summary'] = "Insufficient data for spatial analysis."
        return interpretation
    
    # Check for global spatial autocorrelation
    if 'global_moran' in spatial_results and spatial_results['global_moran']:
        global_moran = spatial_results['global_moran']
        
        interpretation['details']['morans_i'] = global_moran.get('I')
        interpretation['details']['morans_p'] = global_moran.get('p')
        
        if global_moran.get('p', 1.0) < 0.05:
            if global_moran.get('I', 0) > 0:
                interpretation['details']['spatial_pattern'] = "Significant positive spatial autocorrelation detected."
                interpretation['summary'] = f"{commodity} prices show significant spatial clustering, with similar price levels in nearby markets."
                interpretation['implications'].append("The spatial clustering of prices suggests that geographic proximity influences price formation despite conflict barriers.")
                interpretation['implications'].append("Markets within the same region tend to have similar price levels, indicating some degree of regional integration.")
            else:
                interpretation['details']['spatial_pattern'] = "Significant negative spatial autocorrelation detected."
                interpretation['summary'] = f"{commodity} prices show significant spatial dispersion, with dissimilar price levels in nearby markets."
                interpretation['implications'].append("The spatial dispersion of prices suggests that geographic proximity does not lead to price similarity.")
                interpretation['implications'].append("This unusual pattern may indicate strong local market segmentation or conflict barriers between adjacent markets.")
        else:
            interpretation['details']['spatial_pattern'] = "No significant spatial autocorrelation detected."
            interpretation['summary'] = f"{commodity} prices show no significant spatial pattern, suggesting random distribution of prices across space."
            interpretation['implications'].append("The lack of spatial pattern suggests that geographic proximity has little influence on price formation.")
            interpretation['implications'].append("Markets appear to be spatially fragmented, possibly due to conflict barriers disrupting normal spatial price relationships.")
    
    # Check for spatial lag model results
    if 'lag_model' in spatial_results and spatial_results['lag_model']:
        lag_model = spatial_results['lag_model']
        
        interpretation['details']['spatial_dependence'] = getattr(lag_model, 'rho', None)
        interpretation['details']['model_fit'] = getattr(lag_model, 'r2', None)
        
        rho = getattr(lag_model, 'rho', 0)
        if rho > 0:
            if rho < 0.3:
                strength = "weak"
            elif rho < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
                
            interpretation['details']['dependence_strength'] = f"{strength.capitalize()} spatial dependence detected."
            interpretation['implications'].append(f"The {strength} spatial dependence suggests that prices in neighboring markets influence each other to some degree.")
            interpretation['implications'].append("This indicates partial market integration despite conflict barriers.")
        else:
            interpretation['details']['dependence_strength'] = "No positive spatial dependence detected."
            interpretation['implications'].append("The lack of positive spatial dependence suggests that prices in neighboring markets do not influence each other.")
            interpretation['implications'].append("This indicates severe market fragmentation due to conflict barriers.")
    
    # Check for spillover effects
    if 'spillover_effects' in spatial_results:
        spillover = spatial_results['spillover_effects']
        
        if 'indirect' in spillover and 'conflict_intensity_normalized' in spillover['indirect']:
            conflict_indirect = spillover['indirect']['conflict_intensity_normalized']
            
            interpretation['details']['conflict_spillover'] = conflict_indirect
            
            if abs(conflict_indirect) > 0.1:
                interpretation['details']['conflict_effect'] = "Substantial indirect effect of conflict on prices."
                interpretation['implications'].append("Conflict in one area significantly affects prices in neighboring areas, suggesting conflict spillover effects on market integration.")
            else:
                interpretation['details']['conflict_effect'] = "Limited indirect effect of conflict on prices."
                interpretation['implications'].append("Conflict effects appear to be largely localized and do not significantly spill over to neighboring markets.")
    
    # Add policy implications
    interpretation['implications'].append("Spatial analysis provides insights for targeted interventions to improve market connectivity in specific regions.")
    interpretation['implications'].append("Reducing conflict in key market hubs could have multiplier effects on market integration due to spatial spillovers.")
    
    return interpretation


@handle_errors(logger=logger)
def interpret_simulation_results(simulation_results, commodity):
    """
    Provide policy recommendations based on simulation results.
    
    Parameters
    ----------
    simulation_results : dict
        Results from policy simulations
    commodity : str
        Commodity name
        
    Returns
    -------
    dict
        Interpretation of simulation results with policy recommendations
    """
    interpretation = {
        'commodity': commodity,
        'summary': '',
        'policy_recommendations': [],
        'welfare_effects': {},
        'implementation_considerations': []
    }
    
    # Check if we have valid results
    if not simulation_results:
        interpretation['summary'] = "Insufficient data for policy simulation analysis."
        return interpretation
    
    # Extract welfare results
    if 'welfare' in simulation_results:
        welfare = simulation_results['welfare']
        
        # Determine which policy has the highest welfare gain
        best_policy = None
        best_welfare = -float('inf')
        
        policies = {}
        
        # Check exchange rate results
        if 'exchange_rate' in welfare:
            for target, results in welfare['exchange_rate'].items():
                for commodity_name, effects in results.items():
                    if commodity_name == commodity and 'total_welfare' in effects:
                        policies[f'exchange_rate_{target}'] = effects['total_welfare']
                        if effects['total_welfare'] > best_welfare:
                            best_welfare = effects['total_welfare']
                            best_policy = f'Exchange rate unification ({target})'
        
        # Check connectivity results
        if 'connectivity' in welfare:
            for scenario, results in welfare['connectivity'].items():
                for commodity_name, effects in results.items():
                    if commodity_name == commodity and 'total_welfare' in effects:
                        # Extract reduction percentage from scenario key
                        reduction_pct = scenario.split('_')[-1]
                        policies[f'connectivity_{reduction_pct}'] = effects['total_welfare']
                        if effects['total_welfare'] > best_welfare:
                            best_welfare = effects['total_welfare']
                            best_policy = f'Conflict reduction ({reduction_pct}%)'
        
        # Check combined results
        if 'combined' in welfare:
            for scenario, results in welfare['combined'].items():
                for commodity_name, effects in results.items():
                    if commodity_name == commodity and 'total_welfare' in effects:
                        policies[f'combined_{scenario}'] = effects['total_welfare']
                        if effects['total_welfare'] > best_welfare:
                            best_welfare = effects['total_welfare']
                            best_policy = f'Combined policies ({scenario})'
        
        # Set summary based on best policy
        if best_policy:
            interpretation['summary'] = f"Based on simulation results, {best_policy} provides the highest welfare gains for {commodity} markets."
            interpretation['welfare_effects']['best_policy'] = best_policy
            interpretation['welfare_effects']['welfare_gain'] = best_welfare
            interpretation['welfare_effects']['all_policies'] = policies
            
            # Add policy recommendations based on best policy
            if 'Exchange rate' in best_policy:
                interpretation['policy_recommendations'].append("Prioritize exchange rate unification to reduce price distortions between north and south markets.")
                interpretation['policy_recommendations'].append("Coordinate monetary policy between authorities in different regions to achieve sustainable unification.")
                interpretation['policy_recommendations'].append("Implement supporting fiscal measures to manage potential inflation impacts during transition.")
                
                interpretation['implementation_considerations'].append("Exchange rate unification may face political resistance from authorities benefiting from the current dual system.")
                interpretation['implementation_considerations'].append("A phased approach with clear communication may reduce market disruptions during implementation.")
            elif 'Conflict reduction' in best_policy:
                interpretation['policy_recommendations'].append("Prioritize conflict reduction measures to improve physical market connectivity.")
                interpretation['policy_recommendations'].append("Focus on securing key trade routes between north and south markets.")
                interpretation['policy_recommendations'].append("Establish market protection zones to facilitate safe trade in conflict-affected areas.")
                
                interpretation['implementation_considerations'].append("Conflict reduction requires coordination among multiple stakeholders, including security forces and local communities.")
                interpretation['implementation_considerations'].append("Initial focus on high-volume trade corridors may maximize early benefits.")
            elif 'Combined policies' in best_policy:
                interpretation['policy_recommendations'].append("Implement both exchange rate unification and conflict reduction measures simultaneously for maximum benefit.")
                interpretation['policy_recommendations'].append("Coordinate monetary and security policies to ensure synergistic effects.")
                interpretation['policy_recommendations'].append("Establish a comprehensive market integration program with both economic and security components.")
                
                interpretation['implementation_considerations'].append("Combined approach requires more resources but offers higher returns.")
                interpretation['implementation_considerations'].append("Careful sequencing of interventions may be necessary for practical implementation.")
    
    # Add distributional effects if available
    if 'welfare' in simulation_results and 'combined' in simulation_results['welfare']:
        for scenario, results in simulation_results['welfare']['combined'].items():
            for commodity_name, effects in results.items():
                if commodity_name == commodity and 'distributional' in effects:
                    dist = effects['distributional']
                    
                    interpretation['welfare_effects']['distributional'] = {
                        'gini_change': dist.get('gini_change'),
                        'bottom_quintile_impact': dist.get('bottom_quintile_impact'),
                        'food_security_improvement': dist.get('food_security_improvement')
                    }
                    
                    # Add distributional implications
                    if 'gini_change' in dist and dist['gini_change'] < 0:
                        interpretation['policy_recommendations'].append("The policy reduces inequality, providing additional social benefits beyond market efficiency.")
                    
                    if 'bottom_quintile_impact' in dist and dist['bottom_quintile_impact'] < 0:
                        interpretation['policy_recommendations'].append("The policy has positive impacts on the poorest households, making it particularly valuable for poverty reduction.")
                    
                    if 'food_security_improvement' in dist and dist['food_security_improvement'] > 0:
                        interpretation['policy_recommendations'].append(f"The policy improves food security by approximately {dist['food_security_improvement']:.1f}%, addressing a critical humanitarian concern.")
    
    # Add general recommendations
    interpretation['policy_recommendations'].append("Monitor market integration metrics regularly to assess policy effectiveness and make adjustments as needed.")
    interpretation['policy_recommendations'].append("Complement market integration policies with targeted support for vulnerable populations during transition periods.")
    
    return interpretation