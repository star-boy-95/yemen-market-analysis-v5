"""
Dashboard components for the Yemen Market Integration project.

This module provides components and layouts for creating interactive
dashboards to visualize market integration analysis results.
"""
import os
import json
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any

from src.utils.validation import validate_dataframe, validate_file_path
from src.utils.error_handler import handle_errors


class MarketDashboard:
    """
    Interactive dashboard for visualizing Yemen market integration analysis.
    
    This class provides a Dash-based dashboard framework for interactive
    visualization of market integration analysis results, including
    time series plots, spatial maps, and policy simulation comparisons.
    """
    
    def __init__(self, title: str = "Yemen Market Integration Dashboard"):
        """
        Initialize the dashboard with the specified title.
        
        Parameters
        ----------
        title : str
            Dashboard title
        """
        # Initialize Dash app
        self.app = dash.Dash(__name__, 
                            suppress_callback_exceptions=True,
                            meta_tags=[{"name": "viewport", 
                                        "content": "width=device-width, initial-scale=1"}])
        
        # Set title
        self.app.title = title
        self.title = title
        
        # Initialize data containers
        self.market_data = None
        self.spatial_data = None
        self.simulation_results = None
        
        # Initialize dashboard layout
        self._create_layout()
        self._setup_callbacks()
    
    def _create_layout(self):
        """Create the dashboard layout."""
        self.app.layout = html.Div([
            # Dashboard header
            html.Div([
                html.H1(self.title, 
                        style={'textAlign': 'center', 'margin-bottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Label("Data Selection:"),
                        dcc.Dropdown(
                            id='data-file-dropdown',
                            placeholder='Select a data file...',
                            clearable=False
                        ),
                    ], className='six columns'),
                    
                    html.Div([
                        html.Label("Analysis Type:"),
                        dcc.Dropdown(
                            id='analysis-type-dropdown',
                            options=[
                                {'label': 'Time Series Analysis', 'value': 'time_series'},
                                {'label': 'Spatial Integration', 'value': 'spatial'},
                                {'label': 'Threshold Cointegration', 'value': 'threshold'},
                                {'label': 'Policy Simulation', 'value': 'simulation'}
                            ],
                            value='time_series',
                            clearable=False
                        ),
                    ], className='six columns'),
                ], className='row'),
                
                html.Hr()
            ]),
            
            # Filter panel
            html.Div([
                html.H3("Filters", style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Div([
                        html.Label("Commodity:"),
                        dcc.Dropdown(
                            id='commodity-dropdown',
                            placeholder='Select a commodity...',
                            clearable=False
                        ),
                    ], className='four columns'),
                    
                    html.Div([
                        html.Label("Exchange Rate Regime:"),
                        dcc.Dropdown(
                            id='regime-dropdown',
                            placeholder='Select a regime...',
                            clearable=True,
                            multi=True
                        ),
                    ], className='four columns'),
                    
                    html.Div([
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            clearable=True,
                        ),
                    ], className='four columns'),
                ], className='row'),
                
                html.Hr()
            ]),
            
            # Main content area with tabs
            html.Div([
                dcc.Tabs(id='main-tabs', value='visualization-tab', children=[
                    dcc.Tab(label='Visualization', value='visualization-tab', children=[
                        html.Div(id='visualization-content')
                    ]),
                    dcc.Tab(label='Analysis Results', value='analysis-tab', children=[
                        html.Div(id='analysis-content')
                    ]),
                    dcc.Tab(label='Data Table', value='data-tab', children=[
                        html.Div(id='data-content')
                    ]),
                ]),
            ]),
            
            # Store components for intermediate data
            dcc.Store(id='stored-market-data'),
            dcc.Store(id='stored-spatial-data'),
            dcc.Store(id='stored-analysis-results')
        ])

    def _setup_callbacks(self):
        """Set up dashboard callbacks for interactivity."""
        
        @self.app.callback(
            Output('stored-market-data', 'data'),
            Input('data-file-dropdown', 'value')
        )
        def load_market_data(selected_file):
            """Load market data when a file is selected."""
            if not selected_file:
                return None
            
            try:
                # Load data based on file type
                if selected_file.endswith('.csv'):
                    df = pd.read_csv(selected_file)
                elif selected_file.endswith('.json'):
                    df = pd.read_json(selected_file)
                else:
                    return {'error': f"Unsupported file format: {selected_file}"}
                
                # Convert date column to datetime if it exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Return data in format suitable for dcc.Store
                return df.to_json(date_format='iso', orient='split')
                
            except Exception as e:
                return {'error': f"Error loading data: {str(e)}"}
        
        @self.app.callback(
            [Output('commodity-dropdown', 'options'),
             Output('commodity-dropdown', 'value'),
             Output('regime-dropdown', 'options'),
             Output('regime-dropdown', 'value'),
             Output('date-range-picker', 'min_date_allowed'),
             Output('date-range-picker', 'max_date_allowed'),
             Output('date-range-picker', 'start_date'),
             Output('date-range-picker', 'end_date')],
            Input('stored-market-data', 'data')
        )
        def update_filter_options(json_data):
            """Update filter options based on loaded data."""
            if not json_data or isinstance(json_data, dict) and 'error' in json_data:
                # Return empty options if data not loaded or error
                return [], None, [], [], None, None, None, None
            
            try:
                # Parse the JSON data back to DataFrame
                df = pd.read_json(json_data, orient='split')
                
                # Create commodity options
                commodity_options = []
                if 'commodity' in df.columns:
                    commodities = sorted(df['commodity'].unique())
                    commodity_options = [{'label': c, 'value': c} for c in commodities]
                
                # Create regime options
                regime_options = []
                if 'exchange_rate_regime' in df.columns:
                    regimes = sorted(df['exchange_rate_regime'].unique())
                    regime_options = [{'label': r, 'value': r} for r in regimes]
                
                # Create date range
                min_date = None
                max_date = None
                if 'date' in df.columns:
                    min_date = df['date'].min().date()
                    max_date = df['date'].max().date()
                
                # Return updated filter options
                return (commodity_options, 
                        commodity_options[0]['value'] if commodity_options else None,
                        regime_options,
                        None,
                        min_date,
                        max_date,
                        min_date,
                        max_date)
                
            except Exception as e:
                # Return empty options on error
                print(f"Error updating filters: {str(e)}")
                return [], None, [], [], None, None, None, None
        
        @self.app.callback(
            Output('visualization-content', 'children'),
            [Input('analysis-type-dropdown', 'value'),
             Input('stored-market-data', 'data'),
             Input('commodity-dropdown', 'value'),
             Input('regime-dropdown', 'value'),
             Input('date-range-picker', 'start_date'),
             Input('date-range-picker', 'end_date')]
        )
        def update_visualization(analysis_type, json_data, commodity, regimes, start_date, end_date):
            """Update the visualization based on selected analysis type and filters."""
            if not json_data or isinstance(json_data, dict) and 'error' in json_data:
                return html.Div("No data loaded or error in data loading.")
            
            if not analysis_type:
                return html.Div("Please select an analysis type.")
            
            try:
                # Parse the JSON data back to DataFrame
                df = pd.read_json(json_data, orient='split')
                
                # Apply filters
                filtered_df = df.copy()
                
                if commodity and 'commodity' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['commodity'] == commodity]
                
                if regimes and 'exchange_rate_regime' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['exchange_rate_regime'].isin(regimes)]
                
                if start_date and 'date' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['date'] >= start_date]
                
                if end_date and 'date' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['date'] <= end_date]
                
                # Create visualization based on analysis type
                if analysis_type == 'time_series':
                    return self._create_time_series_visualization(filtered_df)
                elif analysis_type == 'spatial':
                    return self._create_spatial_visualization(filtered_df)
                elif analysis_type == 'threshold':
                    return self._create_threshold_visualization(filtered_df)
                elif analysis_type == 'simulation':
                    return self._create_simulation_visualization(filtered_df)
                else:
                    return html.Div(f"Unknown analysis type: {analysis_type}")
                
            except Exception as e:
                return html.Div(f"Error updating visualization: {str(e)}")
        
        @self.app.callback(
            Output('analysis-content', 'children'),
            [Input('analysis-type-dropdown', 'value'),
             Input('stored-market-data', 'data'),
             Input('commodity-dropdown', 'value'),
             Input('regime-dropdown', 'value')]
        )
        def update_analysis_results(analysis_type, json_data, commodity, regimes):
            """Update the analysis results based on selected analysis type and filters."""
            if not json_data or isinstance(json_data, dict) and 'error' in json_data:
                return html.Div("No data loaded or error in data loading.")
            
            if not analysis_type:
                return html.Div("Please select an analysis type.")
            
            try:
                # Parse the JSON data back to DataFrame
                df = pd.read_json(json_data, orient='split')
                
                # Apply filters
                filtered_df = df.copy()
                
                if commodity and 'commodity' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['commodity'] == commodity]
                
                if regimes and 'exchange_rate_regime' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['exchange_rate_regime'].isin(regimes)]
                
                # Show appropriate analysis results based on analysis type
                if analysis_type == 'time_series':
                    return self._create_time_series_analysis(filtered_df)
                elif analysis_type == 'spatial':
                    return self._create_spatial_analysis(filtered_df)
                elif analysis_type == 'threshold':
                    return self._create_threshold_analysis(filtered_df)
                elif analysis_type == 'simulation':
                    return self._create_simulation_analysis(filtered_df)
                else:
                    return html.Div(f"Unknown analysis type: {analysis_type}")
                
            except Exception as e:
                return html.Div(f"Error updating analysis results: {str(e)}")
        
        @self.app.callback(
            Output('data-content', 'children'),
            [Input('stored-market-data', 'data'),
             Input('commodity-dropdown', 'value'),
             Input('regime-dropdown', 'value'),
             Input('date-range-picker', 'start_date'),
             Input('date-range-picker', 'end_date')]
        )
        def update_data_table(json_data, commodity, regimes, start_date, end_date):
            """Update the data table based on filters."""
            if not json_data or isinstance(json_data, dict) and 'error' in json_data:
                return html.Div("No data loaded or error in data loading.")
            
            try:
                # Parse the JSON data back to DataFrame
                df = pd.read_json(json_data, orient='split')
                
                # Apply filters
                filtered_df = df.copy()
                
                if commodity and 'commodity' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['commodity'] == commodity]
                
                if regimes and 'exchange_rate_regime' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['exchange_rate_regime'].isin(regimes)]
                
                if start_date and 'date' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['date'] >= start_date]
                
                if end_date and 'date' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['date'] <= end_date]
                
                # Create data table
                return html.Div([
                    html.H4("Filtered Data"),
                    html.P(f"Showing {len(filtered_df)} records"),
                    html.Div(style={"overflow": "auto", "maxHeight": "500px"}, children=[
                        dash.dash_table.DataTable(
                            data=filtered_df.head(1000).to_dict('records'),
                            columns=[{"name": i, "id": i} for i in filtered_df.columns],
                            page_size=20,
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'minWidth': '100px', 
                                'width': '150px', 
                                'maxWidth': '300px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                            }
                        )
                    ])
                ])
                
            except Exception as e:
                return html.Div(f"Error updating data table: {str(e)}")
    
    def _create_time_series_visualization(self, df: pd.DataFrame) -> Any:
        """Create time series visualization for the filtered data."""
        if 'date' not in df.columns or 'price' not in df.columns:
            return html.Div("Data must contain 'date' and 'price' columns for time series visualization.")
        
        # Create time series plot
        if 'exchange_rate_regime' in df.columns:
            # Group by date and regime
            grouped = df.groupby(['date', 'exchange_rate_regime'])['price'].mean().reset_index()
            
            # Create time series plot with regimes
            fig = px.line(grouped, x='date', y='price', color='exchange_rate_regime',
                         title="Price Time Series by Exchange Rate Regime",
                         labels={'price': 'Price', 'date': 'Date', 
                                'exchange_rate_regime': 'Exchange Rate Regime'})
        else:
            # Just plot by date
            grouped = df.groupby('date')['price'].mean().reset_index()
            
            # Create simple time series plot
            fig = px.line(grouped, x='date', y='price',
                         title="Price Time Series",
                         labels={'price': 'Price', 'date': 'Date'})
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Exchange Rate Regime",
            height=500
        )
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.Div([
                html.P(f"Number of observations: {len(df)}"),
                html.P(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}"),
            ])
        ])
    
    def _create_spatial_visualization(self, df: pd.DataFrame) -> Any:
        """Create spatial visualization for the filtered data."""
        if 'geometry' not in df.columns or not all(
            col in df.columns for col in ['market_id', 'market_name']):
            return html.Div("Data must contain geometry and market information for spatial visualization.")
        
        # Check if we have price data
        if 'price' in df.columns:
            # Create scatter geo map with price information
            if all(hasattr(geom, 'x') and hasattr(geom, 'y') for geom in df['geometry']):
                # Extract coordinates
                df['lon'] = df['geometry'].apply(lambda g: g.x)
                df['lat'] = df['geometry'].apply(lambda g: g.y)
                
                # Use most recent data for each market
                if 'date' in df.columns:
                    latest_date = df['date'].max()
                    latest_df = df[df['date'] == latest_date]
                else:
                    latest_df = df
                
                # Group by market if needed
                if len(latest_df) > len(latest_df['market_id'].unique()):
                    latest_df = latest_df.groupby('market_id').agg({
                        'market_name': 'first',
                        'lon': 'first',
                        'lat': 'first',
                        'price': 'mean',
                        'exchange_rate_regime': 'first'
                    }).reset_index()
                
                # Create scatter map
                fig = px.scatter_mapbox(
                    latest_df,
                    lat='lat',
                    lon='lon',
                    color='price',
                    size='price',
                    hover_name='market_name',
                    hover_data=['market_id', 'price'],
                    color_continuous_scale=px.colors.cyclical.IceFire,
                    size_max=15,
                    zoom=7,
                    title="Market Prices by Location"
                )
                
                # Update to use OpenStreetMap
                fig.update_layout(
                    mapbox_style="open-street-map",
                    height=600,
                    margin={"r": 0, "t": 30, "l": 0, "b": 0}
                )
                
                return html.Div([
                    dcc.Graph(figure=fig),
                    html.Div([
                        html.P(f"Number of markets: {len(latest_df)}"),
                        html.P(f"Price range: {latest_df['price'].min():.2f} to {latest_df['price'].max():.2f}"),
                    ])
                ])
            
        # Fallback visualization if we can't create the map
        return html.Div("Spatial visualization requires valid geometry data with coordinates.")
    
    def _create_threshold_visualization(self, df: pd.DataFrame) -> Any:
        """Create threshold cointegration visualization for the filtered data."""
        # This would require a specific format of data with threshold analysis results
        # For now, we'll just provide a placeholder
        return html.Div([
            html.H3("Threshold Cointegration Visualization"),
            html.P("This visualization requires threshold cointegration analysis results."),
            html.P("Please run threshold analysis on the data first.")
        ])
    
    def _create_simulation_visualization(self, df: pd.DataFrame) -> Any:
        """Create policy simulation visualization for the filtered data."""
        # This would require simulation results data
        # For now, we'll just provide a placeholder
        return html.Div([
            html.H3("Policy Simulation Visualization"),
            html.P("This visualization requires policy simulation results."),
            html.P("Please run simulations on the data first.")
        ])
    
    def _create_time_series_analysis(self, df: pd.DataFrame) -> Any:
        """Create time series analysis results for the filtered data."""
        if 'date' not in df.columns or 'price' not in df.columns:
            return html.Div("Data must contain 'date' and 'price' columns for time series analysis.")
        
        # Calculate basic stats by exchange rate regime
        if 'exchange_rate_regime' in df.columns:
            regimes = df['exchange_rate_regime'].unique()
            stats_components = []
            
            for regime in regimes:
                regime_df = df[df['exchange_rate_regime'] == regime]
                
                # Calculate statistics
                mean_price = regime_df['price'].mean()
                std_price = regime_df['price'].std()
                min_price = regime_df['price'].min()
                max_price = regime_df['price'].max()
                
                # Create statistics component
                stats_components.append(html.Div([
                    html.H4(f"Statistics for {regime}"),
                    html.Table([
                        html.Tr([html.Td("Mean Price:"), html.Td(f"{mean_price:.2f}")]),
                        html.Tr([html.Td("Standard Deviation:"), html.Td(f"{std_price:.2f}")]),
                        html.Tr([html.Td("Minimum Price:"), html.Td(f"{min_price:.2f}")]),
                        html.Tr([html.Td("Maximum Price:"), html.Td(f"{max_price:.2f}")]),
                        html.Tr([html.Td("Number of Observations:"), html.Td(f"{len(regime_df)}")])
                    ])
                ]))
            
            # Create price differential analysis if we have multiple regimes
            if len(regimes) > 1:
                # Create histogram of price differentials
                if len(regimes) == 2:
                    # Calculate price differentials
                    df_pivot = df.pivot_table(
                        index='date', 
                        columns='exchange_rate_regime', 
                        values='price', 
                        aggfunc='mean'
                    ).reset_index()
                    
                    if df_pivot.shape[1] == 3:  # date + 2 regimes
                        regime_names = [col for col in df_pivot.columns if col != 'date']
                        df_pivot['price_diff'] = df_pivot[regime_names[0]] - df_pivot[regime_names[1]]
                        
                        # Create histogram
                        hist_fig = px.histogram(
                            df_pivot, 
                            x='price_diff',
                            nbins=20,
                            title=f"Histogram of Price Differentials ({regime_names[0]} - {regime_names[1]})"
                        )
                        
                        # Add to stats components
                        stats_components.append(html.Div([
                            html.H4("Price Differential Analysis"),
                            dcc.Graph(figure=hist_fig),
                            html.P(f"Mean differential: {df_pivot['price_diff'].mean():.2f}"),
                            html.P(f"Std deviation: {df_pivot['price_diff'].std():.2f}")
                        ]))
            
            # Combine all components
            return html.Div(stats_components)
        else:
            # Just calculate overall statistics
            mean_price = df['price'].mean()
            std_price = df['price'].std()
            min_price = df['price'].min()
            max_price = df['price'].max()
            
            return html.Div([
                html.H4("Overall Statistics"),
                html.Table([
                    html.Tr([html.Td("Mean Price:"), html.Td(f"{mean_price:.2f}")]),
                    html.Tr([html.Td("Standard Deviation:"), html.Td(f"{std_price:.2f}")]),
                    html.Tr([html.Td("Minimum Price:"), html.Td(f"{min_price:.2f}")]),
                    html.Tr([html.Td("Maximum Price:"), html.Td(f"{max_price:.2f}")]),
                    html.Tr([html.Td("Number of Observations:"), html.Td(f"{len(df)}")])
                ])
            ])
    
    def _create_spatial_analysis(self, df: pd.DataFrame) -> Any:
        """Create spatial analysis results for the filtered data."""
        # Placeholder for spatial analysis
        return html.Div([
            html.H3("Spatial Analysis Results"),
            html.P("This analysis requires geospatial data processing."),
            html.P("Detailed spatial analysis implementation pending.")
        ])
    
    def _create_threshold_analysis(self, df: pd.DataFrame) -> Any:
        """Create threshold cointegration analysis for the filtered data."""
        # Placeholder for threshold analysis
        return html.Div([
            html.H3("Threshold Cointegration Analysis"),
            html.P("This analysis requires time series data for market pairs."),
            html.P("Please select two markets to run threshold cointegration analysis.")
        ])
    
    def _create_simulation_analysis(self, df: pd.DataFrame) -> Any:
        """Create policy simulation analysis for the filtered data."""
        # Placeholder for simulation analysis
        return html.Div([
            html.H3("Policy Simulation Analysis"),
            html.P("This analysis requires running policy simulations."),
            html.P("Please select a policy scenario to run simulation analysis.")
        ])
    
    def load_data(self, market_data_path: str, spatial_data_path: Optional[str] = None):
        """
        Load market and spatial data for the dashboard.
        
        Parameters
        ----------
        market_data_path : str
            Path to the market data file (CSV or JSON)
        spatial_data_path : str, optional
            Path to the spatial data file (GeoJSON)
        """
        # Validate file paths
        validate_file_path(market_data_path)
        if spatial_data_path:
            validate_file_path(spatial_data_path)
        
        # Load market data
        if market_data_path.endswith('.csv'):
            self.market_data = pd.read_csv(market_data_path)
        elif market_data_path.endswith('.json'):
            self.market_data = pd.read_json(market_data_path)
        else:
            raise ValueError(f"Unsupported market data format: {market_data_path}")
        
        # Convert date column to datetime if it exists
        if 'date' in self.market_data.columns:
            self.market_data['date'] = pd.to_datetime(self.market_data['date'])
        
        # Load spatial data if provided
        if spatial_data_path:
            import geopandas as gpd
            self.spatial_data = gpd.read_file(spatial_data_path)
    
    def load_simulation_results(self, results_path: str):
        """
        Load policy simulation results for the dashboard.
        
        Parameters
        ----------
        results_path : str
            Path to the simulation results file (JSON)
        """
        # Validate file path
        validate_file_path(results_path)
        
        # Load simulation results
        with open(results_path, 'r') as f:
            self.simulation_results = json.load(f)
    
    def update_data_dropdown(self):
        """Update the data file dropdown with available data files."""
        # This would typically search a data directory for files
        # For now, we'll just use the loaded data paths
        options = []
        
        if hasattr(self, 'market_data_path') and self.market_data_path:
            options.append({'label': os.path.basename(self.market_data_path), 
                          'value': self.market_data_path})
        
        return options
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """
        Run the dashboard server.
        
        Parameters
        ----------
        debug : bool
            Whether to run in debug mode
        port : int
            Port to run the server on
        """
        self.app.run_server(debug=debug, port=port)


# Main function to run the dashboard when module is executed directly
def run_dashboard(market_data_path: str, 
                 spatial_data_path: Optional[str] = None, 
                 port: int = 8050):
    """
    Run the Yemen Market Integration dashboard with the specified data.
    
    Parameters
    ----------
    market_data_path : str
        Path to the market data file (CSV or JSON)
    spatial_data_path : str, optional
        Path to the spatial data file (GeoJSON)
    port : int
        Port to run the server on
    """
    dashboard = MarketDashboard()
    dashboard.load_data(market_data_path, spatial_data_path)
    dashboard.run_server(port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Yemen Market Integration Dashboard')
    parser.add_argument('--market-data', required=True, 
                       help='Path to market data file (CSV or JSON)')
    parser.add_argument('--spatial-data', 
                       help='Path to spatial data file (GeoJSON)')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port to run the dashboard on')
    
    args = parser.parse_args()
    
    run_dashboard(args.market_data, args.spatial_data, args.port)
