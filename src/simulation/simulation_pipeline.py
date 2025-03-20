import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.models.simulation import (
    ExchangeRateSimulation, 
    ConnectivitySimulation, 
    CombinedPolicySimulation,
    WelfareAnalysis
)

def perform_simulation_analysis(market_data, exchange_rates, geo_data, conflict_data, threshold_results, spatial_results, config, tables_dir, figures_dir, logger) -> Dict[str, Dict[str, Any]]:
    """
    Perform policy simulation analysis.
    
    This method simulates the impact of policy interventions on market integration,
    including exchange rate unification, improved connectivity, and combined policies.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary of simulation results
    """
    logger.info("Performing policy simulation analysis")
    
    simulation_results = {}
    
    try:
        # Check for required prerequisite results
        if not threshold_results:
            logger.warning("Threshold results are required for simulation analysis")
            raise ValueError("Threshold analysis must be performed before simulation")
        
        if not spatial_results:
            logger.warning("Spatial results are required for simulation analysis")
            raise ValueError("Spatial analysis must be performed before simulation")
        
        # Perform exchange rate unification simulation
        simulation_results['exchange_rate'] = simulate_exchange_rate_unification(market_data, exchange_rates, threshold_results, config, tables_dir, logger)
        
        # Perform connectivity improvement simulation
        simulation_results['connectivity'] = simulate_connectivity_improvement(market_data, geo_data, conflict_data, spatial_results, config, tables_dir, logger)
        
        # Perform combined policy simulation
        simulation_results['combined'] = simulate_combined_policies(market_data, exchange_rates, geo_data, conflict_data, threshold_results, spatial_results, config, tables_dir, logger)
        
        # Calculate welfare effects
        simulation_results['welfare'] = calculate_welfare_effects(market_data, exchange_rates, geo_data, simulation_results, config, tables_dir, logger)
        
        # Create simulation summary tables
        create_simulation_summary_tables(simulation_results, tables_dir, logger)
        
        # Create simulation visualizations
        create_simulation_visualizations(simulation_results, figures_dir, logger)
        
        logger.info("Successfully completed policy simulation analysis")
        
        return simulation_results
        
    except Exception as e:
        logger.error(f"Error in simulation analysis: {str(e)}")
        raise

def simulate_exchange_rate_unification(market_data, exchange_rates, threshold_results, config, tables_dir, logger) -> Dict[str, Any]:
    """
    Simulate the effects of exchange rate unification.
    
    This method simulates the impact of harmonizing the dual exchange rate regimes
    in Yemen on market integration.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of exchange rate unification simulation results
    """
    logger.info("Simulating exchange rate unification")
    
    results = {}
    
    # Get configuration options
    sim_config = config.get("simulation", {}).get("exchange_rate", {})
    
    # Define target rates for simulation (official, market, or weighted average)
    target_rates = sim_config.get("target_rates", ["official", "market", "average"])
    
    for target in target_rates:
        logger.info(f"Simulating unification to {target} rate")
        
        # Initialize exchange rate simulation
        simulator = ExchangeRateSimulation(
            market_data=market_data,
            exchange_rates=exchange_rates,
            threshold_results=threshold_results
        )
        
        # Run simulation
        sim_result = simulator.simulate(
            target_rate=target,
            recalculate_thresholds=sim_config.get("recalculate_thresholds", True)
        )
        
        # Store results
        results[target] = sim_result
        
        # Generate summary table
        create_exchange_rate_summary_table(target, sim_result, tables_dir, logger)
    
    return results

def simulate_connectivity_improvement(market_data, geo_data, conflict_data, spatial_results, config, tables_dir, logger) -> Dict[str, Any]:
    """
    Simulate the effects of improved market connectivity.
    
    This method simulates the impact of reduced conflict barriers on market connectivity
    and price integration.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of connectivity improvement simulation results
    """
    logger.info("Simulating improved market connectivity")
    
    results = {}
    
    # Get configuration options
    sim_config = config.get("simulation", {}).get("connectivity", {})
    
    # Define reduction factors for simulation
    reduction_factors = sim_config.get("reduction_factors", [0.25, 0.5, 0.75])
    
    for factor in reduction_factors:
        logger.info(f"Simulating connectivity improvement with {factor*100:.0f}% conflict reduction")
        
        # Initialize connectivity simulation
        simulator = ConnectivitySimulation(
            market_data=market_data,
            geo_data=geo_data,
            conflict_data=conflict_data,
            spatial_results=spatial_results
        )
        
        # Run simulation
        sim_result = simulator.simulate(
            reduction_factor=factor,
            recalculate_accessibility=sim_config.get("recalculate_accessibility", True)
        )
        
        # Store results
        results[f"reduction_{int(factor*100)}"] = sim_result
        
        # Generate summary table
        create_connectivity_summary_table(factor, sim_result, tables_dir, logger)
    
    return results

def simulate_combined_policies(market_data, exchange_rates, geo_data, conflict_data, threshold_results, spatial_results, config, tables_dir, logger) -> Dict[str, Any]:
    """
    Simulate the effects of combined policy interventions.
    
    This method simulates the impact of implementing both exchange rate unification
    and improved connectivity simultaneously.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of combined policy simulation results
    """
    logger.info("Simulating combined policy interventions")
    
    results = {}
    
    # Get configuration options
    sim_config = config.get("simulation", {}).get("combined", {})
    
    # Define parameter combinations
    exchange_targets = sim_config.get("exchange_targets", ["official"])
    connectivity_factors = sim_config.get("connectivity_factors", [0.5])
    
    for target in exchange_targets:
        for factor in connectivity_factors:
            scenario_name = f"{target}_exchange_rate_with_{int(factor*100)}pct_connectivity"
            logger.info(f"Simulating combined scenario: {scenario_name}")
            
            # Initialize combined simulation
            simulator = CombinedPolicySimulation(
                market_data=market_data,
                exchange_rates=exchange_rates,
                geo_data=geo_data,
                conflict_data=conflict_data,
                threshold_results=threshold_results,
                spatial_results=spatial_results
            )
            
            # Run simulation
            sim_result = simulator.simulate(
                target_rate=target,
                reduction_factor=factor
            )
            
            # Store results
            results[scenario_name] = sim_result
            
            # Generate summary table
            create_combined_summary_table(scenario_name, sim_result, tables_dir, logger)
    
    return results

def calculate_welfare_effects(market_data, exchange_rates, geo_data, simulation_results, config, tables_dir, logger) -> Dict[str, Any]:
    """
    Calculate welfare effects of policy interventions.
    
    This method quantifies the benefits of simulated policy interventions using
    various metrics such as price convergence and market integration.
    
    Parameters
    ----------
    simulation_results : Dict[str, Dict[str, Any]]
        Dictionary of simulation results
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of welfare analysis results
    """
    logger.info("Calculating welfare effects of policy interventions")
    
    welfare_results = {}
    
    # Initialize welfare analysis
    analyzer = WelfareAnalysis(
        market_data=market_data,
        exchange_rates=exchange_rates,
        geo_data=geo_data
    )
    
    # Calculate for exchange rate simulation
    if 'exchange_rate' in simulation_results:
        exchange_welfare = {}
        for target, result in simulation_results['exchange_rate'].items():
            exchange_welfare[target] = analyzer.calculate_welfare_effects(
                'exchange_rate', result
            )
        welfare_results['exchange_rate'] = exchange_welfare
    
    # Calculate for connectivity simulation
    if 'connectivity' in simulation_results:
        connectivity_welfare = {}
        for scenario, result in simulation_results['connectivity'].items():
            connectivity_welfare[scenario] = analyzer.calculate_welfare_effects(
                'connectivity', result
            )
        welfare_results['connectivity'] = connectivity_welfare
    
    # Calculate for combined simulation
    if 'combined' in simulation_results:
        combined_welfare = {}
        for scenario, result in simulation_results['combined'].items():
            combined_welfare[scenario] = analyzer.calculate_welfare_effects(
                'combined', result
            )
        welfare_results['combined'] = combined_welfare
    
    # Create comparative welfare summary
    create_welfare_summary_table(welfare_results, tables_dir, logger)
    
    return welfare_results

def create_exchange_rate_summary_table(target: str, results: Dict[str, Any], tables_dir, logger) -> None:
    """
    Create summary table for exchange rate unification simulation.
    
    Parameters
    ----------
    target : str
        Target exchange rate type
    results : Dict[str, Any]
        Simulation results
    """
    # Create summary DataFrame for price effects
    if 'price_effects' in results:
        price_data = []
        
        for commodity, effects in results['price_effects'].items():
            row = {
                'Commodity': commodity,
                'Avg Price Change (%)': effects['avg_pct_change'],
                'Max Price Change (%)': effects['max_pct_change'],
                'Price Dispersion Before': effects['dispersion_before'],
                'Price Dispersion After': effects['dispersion_after'],
                'Dispersion Reduction (%)': effects['dispersion_reduction_pct']
            }
            price_data.append(row)
        
        price_df = pd.DataFrame(price_data)
        
        # Save to CSV
        table_path = tables_dir / f"exchange_rate_{target}_price_effects.csv"
        price_df.to_csv(table_path, index=False)
        
        logger.info(f"Saved exchange rate price effects summary to {table_path}")
    
    # Create summary DataFrame for threshold effects
    if 'threshold_effects' in results:
        threshold_data = []
        
        for commodity, effects in results['threshold_effects'].items():
            row = {
                'Commodity': commodity,
                'Avg Threshold Before': effects['avg_threshold_before'],
                'Avg Threshold After': effects['avg_threshold_after'],
                'Threshold Reduction (%)': effects['threshold_reduction_pct'],
                'Markets Integrated Before': effects['integrated_pairs_before'],
                'Markets Integrated After': effects['integrated_pairs_after'],
                'Integration Improvement (%)': effects['integration_improvement_pct']
            }
            threshold_data.append(row)
        
        threshold_df = pd.DataFrame(threshold_data)
        
        # Save to CSV
        table_path = tables_dir / f"exchange_rate_{target}_threshold_effects.csv"
        threshold_df.to_csv(table_path, index=False)
        
        logger.info(f"Saved exchange rate threshold effects summary to {table_path}")

def create_connectivity_summary_table(factor: float, results: Dict[str, Any], tables_dir, logger) -> None:
    """
    Create summary table for connectivity improvement simulation.
    
    Parameters
    ----------
    factor : float
        Conflict reduction factor
    results : Dict[str, Any]
        Simulation results
    """
    # Create summary DataFrame for accessibility effects
    if 'accessibility_effects' in results:
        access_data = []
        
        for commodity, effects in results['accessibility_effects'].items():
            row = {
                'Commodity': commodity,
                'Avg Accessibility Before': effects['avg_accessibility_before'],
                'Avg Accessibility After': effects['avg_accessibility_after'],
                'Accessibility Improvement (%)': effects['accessibility_improvement_pct'],
                'Market Coverage Before (%)': effects['market_coverage_before'],
                'Market Coverage After (%)': effects['market_coverage_after'],
                'Coverage Improvement (%)': effects['coverage_improvement_pct']
            }
            access_data.append(row)
        
        access_df = pd.DataFrame(access_data)
        
        # Save to CSV
        table_path = tables_dir / f"connectivity_{int(factor*100)}_accessibility_effects.csv"
        access_df.to_csv(table_path, index=False)
        
        logger.info(f"Saved connectivity accessibility effects summary to {table_path}")
    
    # Create summary DataFrame for spatial lag effects
    if 'spatial_effects' in results:
        spatial_data = []
        
        for commodity, effects in results['spatial_effects'].items():
            row = {
                'Commodity': commodity,
                'Spatial Lag Before': effects['spatial_lag_before'],
                'Spatial Lag After': effects['spatial_lag_after'],
                'Spatial Dependence Change (%)': effects['spatial_dependence_change_pct'],
                'Price Transmission Before': effects['price_transmission_before'],
                'Price Transmission After': effects['price_transmission_after'],
                'Price Transmission Improvement (%)': effects['price_transmission_improvement_pct']
            }
            spatial_data.append(row)
        
        spatial_df = pd.DataFrame(spatial_data)
        
        # Save to CSV
        table_path = tables_dir / f"connectivity_{int(factor*100)}_spatial_effects.csv"
        spatial_df.to_csv(table_path, index=False)
        
        logger.info(f"Saved connectivity spatial effects summary to {table_path}")

def create_combined_summary_table(scenario: str, results: Dict[str, Any], tables_dir, logger) -> None:
    """
    Create summary table for combined policy simulation.
    
    Parameters
    ----------
    scenario : str
        Scenario name
    results : Dict[str, Any]
        Simulation results
    """
    # Create summary DataFrame for combined effects
    if 'combined_effects' in results:
        combined_data = []
        
        for commodity, effects in results['combined_effects'].items():
            row = {
                'Commodity': commodity,
                'Price Dispersion Before': effects['dispersion_before'],
                'Price Dispersion After': effects['dispersion_after'],
                'Dispersion Reduction (%)': effects['dispersion_reduction_pct'],
                'Market Integration Before (%)': effects['integration_before'],
                'Market Integration After (%)': effects['integration_after'],
                'Integration Improvement (%)': effects['integration_improvement_pct'],
                'Synergy Effect (%)': effects.get('synergy_effect_pct', np.nan)
            }
            combined_data.append(row)
        
        combined_df = pd.DataFrame(combined_data)
        
        # Save to CSV
        table_path = tables_dir / f"combined_{scenario}_effects.csv"
        combined_df.to_csv(table_path, index=False)
        
        logger.info(f"Saved combined policy effects summary to {table_path}")

def create_welfare_summary_table(welfare_results: Dict[str, Dict[str, Dict[str, Any]]], tables_dir, logger) -> None:
    """
    Create summary table for welfare analysis.
    
    Parameters
    ----------
    welfare_results : Dict[str, Dict[str, Dict[str, Any]]]
        Welfare analysis results
    """
    # Create summary DataFrame for comparative welfare effects
    welfare_data = []
    
    # Initialize list to store all policy scenarios
    scenarios = []
    
    # Process exchange rate welfare results
    if 'exchange_rate' in welfare_results:
        for target, results in welfare_results['exchange_rate'].items():
            scenario = f"Exchange Rate ({target})"
            scenarios.append(scenario)
            
            for commodity, effects in results.items():
                row = {
                    'Commodity': commodity,
                    'Policy Scenario': scenario,
                    'Price Convergence (%)': effects['price_convergence_pct'],
                    'Consumer Welfare Gain (%)': effects['consumer_welfare_gain_pct'],
                    'Market Integration Improvement (%)': effects['market_integration_improvement_pct'],
                    'Overall Welfare Impact': effects['overall_welfare_impact']
                }
                welfare_data.append(row)
        
        # Process connectivity welfare results
        if 'connectivity' in welfare_results:
            for scenario_key, results in welfare_results['connectivity'].items():
                # Extract reduction percentage from scenario key
                reduction_pct = scenario_key.split('_')[-1]
                scenario = f"Connectivity ({reduction_pct}%)"
                scenarios.append(scenario)
                
                for commodity, effects in results.items():
                    row = {
                        'Commodity': commodity,
                        'Policy Scenario': scenario,
                        'Price Convergence (%)': effects['price_convergence_pct'],
                        'Consumer Welfare Gain (%)': effects['consumer_welfare_gain_pct'],
                        'Market Integration Improvement (%)': effects['market_integration_improvement_pct'],
                        'Overall Welfare Impact': effects['overall_welfare_impact']
                    }
                    welfare_data.append(row)
        
        # Process combined welfare results
        if 'combined' in welfare_results:
            for scenario_key, results in welfare_results['combined'].items():
                scenario = f"Combined ({scenario_key})"
                scenarios.append(scenario)
                
                for commodity, effects in results.items():
                    row = {
                        'Commodity': commodity,
                        'Policy Scenario': scenario,
                        'Price Convergence (%)': effects['price_convergence_pct'],
                        'Consumer Welfare Gain (%)': effects['consumer_welfare_gain_pct'],
                        'Market Integration Improvement (%)': effects['market_integration_improvement_pct'],
                        'Overall Welfare Impact': effects['overall_welfare_impact']
                    }
                    welfare_data.append(row)
        
        # Create DataFrame
        welfare_df = pd.DataFrame(welfare_data)
        
        # Save to CSV
        table_path = tables_dir / "comparative_welfare_effects.csv"
        welfare_df.to_csv(table_path, index=False)
        
        # Also save a LaTeX version for the paper
        tex_path = tables_dir / "comparative_welfare_effects.tex"
        
        # Format the DataFrame for LaTeX
        latex_df = welfare_df.copy()
        
        # Format percentages
        for col in ['Price Convergence (%)', 'Consumer Welfare Gain (%)', 'Market Integration Improvement (%)']:
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
        
        # Convert to LaTeX
        latex_table = latex_df.to_latex(index=False, na_rep='N/A', float_format="%.2f")
        
        with open(tex_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Saved comparative welfare effects summary to {table_path} and {tex_path}")

def create_simulation_visualizations(simulation_results: Dict[str, Dict[str, Any]], figures_dir, logger) -> None:
    """
    Create visualizations of simulation results.
    
    Parameters
    ----------
    simulation_results : Dict[str, Dict[str, Any]]
        Dictionary of simulation results
    """
    # Create exchange rate visualizations
    if 'exchange_rate' in simulation_results:
        create_exchange_rate_visualizations(simulation_results['exchange_rate'], figures_dir, logger)
    
    # Create connectivity visualizations
    if 'connectivity' in simulation_results:
        create_connectivity_visualizations(simulation_results['connectivity'], figures_dir, logger)
    
    # Create combined policy visualizations
    if 'combined' in simulation_results:
        create_combined_policy_visualizations(simulation_results['combined'], figures_dir, logger)
    
    # Create comparative welfare visualizations
    if 'welfare' in simulation_results:
        create_welfare_visualizations(simulation_results['welfare'], figures_dir, logger)

def create_exchange_rate_visualizations(results: Dict[str, Dict[str, Any]], figures_dir, logger) -> None:
    """
    Create visualizations of exchange rate unification results.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Exchange rate simulation results
    """
    # Create price dispersion comparison plot
    create_price_dispersion_plot(results, figures_dir, logger)
    
    # Create threshold comparison plot
    create_threshold_comparison_plot(results, figures_dir, logger)

def create_price_dispersion_plot(results: Dict[str, Dict[str, Any]], figures_dir, logger) -> None:
    """
    Create price dispersion comparison plot for exchange rate scenarios.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Exchange rate simulation results
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for plotting
    targets = list(results.keys())
    commodities = list(results[targets[0]]['price_effects'].keys())
    
    # Setup plot positions
    x = np.arange(len(commodities))
    width = 0.35 / len(targets)
    offsets = np.linspace(-(len(targets)-1)/2*width, (len(targets)-1)/2*width, len(targets))
    
    # Plot bars for each target
    for i, target in enumerate(targets):
        before_values = [results[target]['price_effects'][c]['dispersion_before'] for c in commodities]
        after_values = [results[target]['price_effects'][c]['dispersion_after'] for c in commodities]
        
        ax.bar(x + offsets[i] - width/2, before_values, width, label=f'{target.capitalize()} (Before)', 
               alpha=0.7, color=f'C{i}')
        ax.bar(x + offsets[i] + width/2, after_values, width, label=f'{target.capitalize()} (After)', 
               alpha=0.4, hatch='///', color=f'C{i}')
    
    # Set labels and title
    ax.set_xlabel('Commodity')
    ax.set_ylabel('Price Dispersion (CV)')
    ax.set_title('Price Dispersion Before and After Exchange Rate Unification')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in commodities])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    fig_path = figures_dir / "exchange_rate_price_dispersion.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_threshold_comparison_plot(results: Dict[str, Dict[str, Any]], figures_dir, logger) -> None:
    """
    Create threshold comparison plot for exchange rate scenarios.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Exchange rate simulation results
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for plotting
    targets = list(results.keys())
    commodities = list(results[targets[0]]['threshold_effects'].keys())
    
    # Setup plot positions
    x = np.arange(len(commodities))
    width = 0.35 / len(targets)
    offsets = np.linspace(-(len(targets)-1)/2*width, (len(targets)-1)/2*width, len(targets))
    
    # Plot bars for each target
    for i, target in enumerate(targets):
        before_values = [results[target]['threshold_effects'][c]['avg_threshold_before'] for c in commodities]
        after_values = [results[target]['threshold_effects'][c]['avg_threshold_after'] for c in commodities]
        
        ax.bar(x + offsets[i] - width/2, before_values, width, label=f'{target.capitalize()} (Before)', 
               alpha=0.7, color=f'C{i}')
        ax.bar(x + offsets[i] + width/2, after_values, width, label=f'{target.capitalize()} (After)', 
               alpha=0.4, hatch='///', color=f'C{i}')
    
    # Set labels and title
    ax.set_xlabel('Commodity')
        ax.set_ylabel('Average Market Accessibility')
        ax.set_title('Market Accessibility Before and After Connectivity Improvement')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        fig_path = figures_dir / "connectivity_accessibility.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_spatial_dependence_plot(results: Dict[str, Dict[str, Any]], figures_dir, logger) -> None:
    """
    Create spatial dependence plot for connectivity scenarios.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Connectivity simulation results
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for plotting
    scenarios = list(results.keys())
    
    # Extract reduction percentages from scenario keys
    reduction_pcts = [int(s.split('_')[-1]) for s in scenarios]
    
    # Sort scenarios by reduction percentage
    sorted_indices = np.argsort(reduction_pcts)
    scenarios = [scenarios[i] for i in sorted_indices]
    reduction_pcts = [reduction_pcts[i] for i in sorted_indices]
    
    commodities = list(results[scenarios[0]]['spatial_effects'].keys())
    
    # Setup bar positions
    x = np.arange(len(reduction_pcts))
    width = 0.8 / len(commodities)
    offsets = np.linspace(-(len(commodities)-1)/2*width, (len(commodities)-1)/2*width, len(commodities))
    
    # Plot grouped bars for each commodity
    for i, commodity in enumerate(commodities):
        before_values = [results[s]['spatial_effects'][commodity]['spatial_lag_before'] for s in scenarios]
        after_values = [results[s]['spatial_effects'][commodity]['spatial_lag_after'] for s in scenarios]
        
        # Calculate change percentages for text labels
        changes = [(after - before) / before * 100 for before, after in zip(before_values, after_values)]
        
        # Plot bars
        bars1 = ax.bar(x + offsets[i], before_values, width, label=f'{commodity.capitalize()} (Before)' if i == 0 else "", 
                      alpha=0.7, color=f'C{i}')
        bars2 = ax.bar(x + offsets[i], after_values, width, bottom=before_values, 
                      label=f'{commodity.capitalize()} (After)' if i == 0 else "", 
                      alpha=0.4, hatch='///', color=f'C{i}')
        
        # Add percentage change labels
        for j, (b1, b2, change) in enumerate(zip(bars1, bars2, changes)):
            ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + b2.get_height() + 0.01, 
                   f'{change:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Set labels and title
        ax.set_xlabel('Conflict Reduction (%)')
        ax.set_ylabel('Spatial Dependence Parameter')
        ax.set_title('Spatial Dependence Before and After Connectivity Improvement')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}%" for p in reduction_pcts])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save figure
        fig_path = figures_dir / "connectivity_spatial_dependence.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_combined_policy_visualizations(results: Dict[str, Dict[str, Any]], figures_dir, logger) -> None:
    """
    Create visualizations of combined policy simulation results.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Combined policy simulation results
    """
    # Create integration improvement plot
    create_integration_improvement_plot(results, figures_dir, logger)
    
    # Create synergy effect plot
    create_synergy_effect_plot(results, figures_dir, logger)

def create_integration_improvement_plot(results: Dict[str, Dict[str, Any]], figures_dir, logger) -> None:
    """
    Create integration improvement plot for combined policy scenarios.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Combined policy simulation results
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for plotting
    scenarios = list(results.keys())
    commodities = list(results[scenarios[0]]['combined_effects'].keys())
    
    # Setup plot positions
    x = np.arange(len(commodities))
    width = 0.35
    
    # Plot bars
    before_values = [results[scenarios[0]]['combined_effects'][c]['integration_before'] for c in commodities]
    after_values = [results[scenarios[0]]['combined_effects'][c]['integration_after'] for c in commodities]
    
    # Calculate improvement percentages for text labels
    improvements = [results[scenarios[0]]['combined_effects'][c]['integration_improvement_pct'] for c in commodities]
    
    # Plot bars
    bars1 = ax.bar(x - width/2, before_values, width, label='Before', alpha=0.7)
    bars2 = ax.bar(x + width/2, after_values, width, label='After', alpha=0.7)
    
    # Add percentage improvement labels
    for i, (improvement) in enumerate(improvements):
        ax.text(i, max(before_values[i], after_values[i]) + 0.05, 
               f'+{improvement:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Commodity')
    ax.set_ylabel('Market Integration (%)')
    ax.set_title(f'Market Integration Before and After Combined Policy: {scenarios[0]}')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in commodities])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure
    fig_path = figures_dir / "combined_integration_improvement.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_synergy_effect_plot(results: Dict[str, Dict[str, Any]], figures_dir, logger) -> None:
    """
    Create synergy effect plot for combined policy scenarios.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Combined policy simulation results
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for plotting
    scenario = list(results.keys())[0]  # Get the first scenario
    commodities = list(results[scenario]['combined_effects'].keys())
    
    # Setup plot positions
    x = np.arange(len(commodities))
    
    # Plot bars for synergy effects
    synergy_values = [results[scenario]['combined_effects'][c].get('synergy_effect_pct', 0) for c in commodities]
    
    bars = ax.bar(x, synergy_values, color='teal', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(synergy_values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', va='bottom')
    
    # Set labels and title
    ax.set_xlabel('Commodity')
    ax.set_ylabel('Synergy Effect (%)')
    ax.set_title(f'Policy Synergy Effects for {scenario}')
    ax.set
