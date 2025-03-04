"""
Data export utilities for Yemen Market Analysis.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path

from core.decorators import error_handler, performance_tracker
from core.exceptions import ReportingError
from .output_manager import NumpyEncoder
from .tables import create_summary_table, create_conflict_summary_table, create_welfare_summary_table

logger = logging.getLogger(__name__)


@error_handler(fallback_value=False)
@performance_tracker()
def export_to_excel(
    results_by_commodity: Dict[str, Dict[str, Any]],
    output_path: str,
    conflict_results: Optional[Dict[str, Dict[str, Any]]] = None,
    welfare_results: Optional[Dict[str, Dict[str, Any]]] = None,
    include_metadata: bool = True
) -> bool:
    """
    Export analysis results to formatted Excel file.
    
    Args:
        results_by_commodity: Dictionary with results keyed by commodity
        output_path: Path for output Excel file
        conflict_results: Optional conflict analysis results
        welfare_results: Optional welfare analysis results
        include_metadata: Whether to include metadata sheet
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Create summary tables
            summary_df = create_summary_table(results_by_commodity)
            
            # Create formats
            header_format = workbook.add_format({
                'bold': True,
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            numeric_format = workbook.add_format({
                'num_format': '0.000',
                'border': 1
            })
            
            percent_format = workbook.add_format({
                'num_format': '0.0%',
                'border': 1
            })
            
            # Write summary table
            summary_df.to_excel(writer, sheet_name='Market Integration')
            
            # Format summary sheet
            sheet = writer.sheets['Market Integration']
            
            for col_num, col_name in enumerate(summary_df.columns):
                # Apply header format
                sheet.write(0, col_num + 1, col_name, header_format)
                
                # Apply number formats
                if col_name in ['threshold', 'alpha_down', 'alpha_up', 'p_value']:
                    for row_num in range(len(summary_df)):
                        sheet.write(row_num + 1, col_num + 1, summary_df.iloc[row_num][col_name], numeric_format)
                
                elif col_name in ['integration_index']:
                    for row_num in range(len(summary_df)):
                        sheet.write(row_num + 1, col_num + 1, summary_df.iloc[row_num][col_name], percent_format)
            
            # Set column widths
            sheet.set_column(0, 0, 18)  # Commodity column
            sheet.set_column(1, len(summary_df.columns), 15)  # Data columns
            
            # Add conflict results if available
            if conflict_results:
                conflict_df = create_conflict_summary_table(conflict_results)
                if not conflict_df.empty:
                    conflict_df.to_excel(writer, sheet_name='Conflict Impact')
                    
                    # Format conflict sheet
                    sheet = writer.sheets['Conflict Impact']
                    
                    for col_num, col_name in enumerate(conflict_df.columns):
                        # Apply header format
                        sheet.write(0, col_num + 1, col_name, header_format)
                    
                    # Set column widths
                    sheet.set_column(0, 0, 18)  # Commodity column
                    sheet.set_column(1, len(conflict_df.columns), 15)  # Data columns
            
            # Add welfare results if available
            if welfare_results:
                welfare_df = create_welfare_summary_table(welfare_results)
                if not welfare_df.empty:
                    welfare_df.to_excel(writer, sheet_name='Welfare Impact')
                    
                    # Format welfare sheet
                    sheet = writer.sheets['Welfare Impact']
                    
                    for col_num, col_name in enumerate(welfare_df.columns):
                        # Apply header format
                        sheet.write(0, col_num + 1, col_name, header_format)
                    
                    # Set column widths
                    sheet.set_column(0, 0, 18)  # Commodity column
                    sheet.set_column(1, len(welfare_df.columns), 15)  # Data columns
            
            # Add metadata if requested
            if include_metadata:
                # Create metadata
                metadata = {
                    'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'commodities': list(results_by_commodity.keys()),
                    'total_commodities': len(results_by_commodity)
                }
                
                # Create metadata sheet
                metadata_df = pd.DataFrame([metadata])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Exported analysis results to Excel: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export to Excel: {str(e)}")
        return False


@error_handler(fallback_value=False)
@performance_tracker()
def export_to_geojson(
    results_by_commodity: Dict[str, Dict[str, Any]],
    admin_boundaries: Dict[str, Any],
    output_path: str,
    regime_property: str = 'exchange_rate_regime'
) -> bool:
    """
    Export analysis results as GeoJSON for spatial visualization.
    
    Args:
        results_by_commodity: Dictionary with results keyed by commodity
        admin_boundaries: GeoJSON with administrative boundaries
        output_path: Path for output GeoJSON file
        regime_property: Property name for regime identification
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create feature collection based on admin boundaries
        geo_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # Process each feature in admin boundaries
        for feature in admin_boundaries.get("features", []):
            # Get region name and regime
            properties = feature.get("properties", {})
            admin_name = properties.get("admin1", "").lower()
            regime = properties.get(regime_property, "")
            
            # Create new feature with same geometry
            new_feature = {
                "type": "Feature",
                "geometry": feature.get("geometry", {}),
                "properties": {
                    "admin1": admin_name,
                    "regime": regime,
                    "commodities": {}
                }
            }
            
            # Add analysis results for each commodity
            for commodity, results in results_by_commodity.items():
                # Extract key metrics
                threshold = results.get('threshold', np.nan)
                integration_index = results.get('integration', {}).get('integration_index', np.nan)
                
                # Add to properties
                new_feature["properties"]["commodities"][commodity] = {
                    "threshold": float(threshold) if not np.isnan(threshold) else None,
                    "integration_index": float(integration_index) if not np.isnan(integration_index) else None
                }
            
            # Add to feature collection
            geo_data["features"].append(new_feature)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(geo_data, f, cls=NumpyEncoder, indent=2)
        
        logger.info(f"Exported GeoJSON to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export to GeoJSON: {str(e)}")
        return False


@error_handler(fallback_value=False)
@performance_tracker()
def export_to_dashboard_format(
    results_by_commodity: Dict[str, Dict[str, Any]],
    output_path: str,
    include_conflict: bool = True,
    include_welfare: bool = True
) -> bool:
    """
    Export analysis results in a format optimized for dashboard visualization.
    
    Args:
        results_by_commodity: Dictionary with results keyed by commodity
        output_path: Path for output JSON file
        include_conflict: Whether to include conflict metrics
        include_welfare: Whether to include welfare metrics
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dashboard_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "commodities": list(results_by_commodity.keys())
            },
            "integration_metrics": [],
            "threshold_metrics": [],
            "adjustment_metrics": []
        }
        
        # Process each commodity
        for commodity, results in results_by_commodity.items():
            # Skip if no results
            if not results:
                continue
            
            # Extract integration metrics
            integration = results.get('integration', {})
            integration_index = integration.get('integration_index', np.nan)
            integration_level = integration.get('integration_level', 'Unknown')
            
            dashboard_data["integration_metrics"].append({
                "commodity": commodity,
                "integration_index": float(integration_index) if not np.isnan(integration_index) else None,
                "integration_level": integration_level
            })
            
            # Extract threshold metrics
            threshold = results.get('threshold', np.nan)
            p_value = results.get('p_value', np.nan)
            threshold_significant = results.get('threshold_significant', False)
            
            dashboard_data["threshold_metrics"].append({
                "commodity": commodity,
                "threshold": float(threshold) if not np.isnan(threshold) else None,
                "p_value": float(p_value) if not np.isnan(p_value) else None,
                "significant": threshold_significant
            })
            
            # Extract adjustment metrics
            adjustment = results.get('adjustment_dynamics', {})
            alpha_down = adjustment.get('alpha_down', np.nan)
            alpha_up = adjustment.get('alpha_up', np.nan)
            half_life_down = adjustment.get('half_life_down', np.nan)
            half_life_up = adjustment.get('half_life_up', np.nan)
            
            dashboard_data["adjustment_metrics"].append({
                "commodity": commodity,
                "alpha_down": float(alpha_down) if not np.isnan(alpha_down) else None,
                "alpha_up": float(alpha_up) if not np.isnan(alpha_up) else None,
                "half_life_down": float(half_life_down) if not np.isnan(half_life_down) else None,
                "half_life_up": float(half_life_up) if not np.isnan(half_life_up) else None
            })
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, cls=NumpyEncoder, indent=2)
        
        logger.info(f"Exported dashboard data to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export dashboard data: {str(e)}")
        return False


@error_handler(fallback_value=False)
@performance_tracker()
def export_time_series_data(
    north_prices: Dict[str, pd.Series],
    south_prices: Dict[str, pd.Series],
    price_diffs: Dict[str, pd.DataFrame],
    output_dir: str,
    file_prefix: str = "timeseries_"
) -> bool:
    """
    Export time series data for each commodity.
    
    Args:
        north_prices: Dictionary of north price series by commodity
        south_prices: Dictionary of south price series by commodity
        price_diffs: Dictionary of price differential DataFrames by commodity
        output_dir: Directory for output files
        file_prefix: Prefix for output filenames
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each commodity
        for commodity in north_prices.keys():
            # Skip if missing data
            if commodity not in south_prices or commodity not in price_diffs:
                continue
            
            # Get data
            north = north_prices[commodity]
            south = south_prices[commodity]
            diff_df = price_diffs[commodity]
            
            # Create commodity-specific DataFrame
            if not north.empty and not south.empty:
                # Align indices
                common_index = north.index.intersection(south.index)
                north = north.loc[common_index]
                south = south.loc[common_index]
                
                # Create time series DataFrame
                ts_df = pd.DataFrame({
                    'date': common_index,
                    'north_price': north.values,
                    'south_price': south.values
                })
                
                # Add price differential if available
                if not diff_df.empty:
                    diff_common_index = common_index.intersection(diff_df.index)
                    if len(diff_common_index) > 0:
                        for col in ['diff', 'diff_pct', 'abs_diff', 'abs_diff_pct']:
                            if col in diff_df.columns:
                                ts_df[col] = diff_df.loc[diff_common_index, col].values
                
                # Export to CSV
                safe_commodity = commodity.replace(' ', '_').replace('(', '').replace(')', '').lower()
                output_path = os.path.join(output_dir, f"{file_prefix}{safe_commodity}.csv")
                ts_df.to_csv(output_path, index=False)
                
                logger.info(f"Exported time series data for {commodity} to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to export time series data: {str(e)}")
        return False