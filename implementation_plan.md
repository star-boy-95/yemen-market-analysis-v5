# Yemen Market Analysis Implementation Plan

This document outlines the remaining implementation tasks and improvements needed for the Yemen Market Analysis project. It identifies areas where temporary workarounds were implemented and provides a plan for proper implementation.

## 1. Unit Root Testing Module

### Issues Identified
- The `validate_time_series()` function is being called with an unsupported `custom_validators` parameter
- Error: `TypeError: validate_time_series() got an unexpected keyword argument 'custom_validators'`

### Implementation Plan
1. Update the `test_adf()` method in `src/models/unit_root.py` to remove the unsupported parameter or implement the missing functionality:
   ```python
   def test_adf(self, series, lags=None):
       # Remove custom_validators parameter or implement it
       valid, errors = validate_time_series(series)
       # Rest of the method...
   ```

2. Alternatively, update the `validate_time_series()` function in `src/utils/validation.py` to accept the `custom_validators` parameter:
   ```python
   def validate_time_series(series, custom_validators=None):
       # Implement custom validators functionality
       # ...
   ```

## 2. Spatial Analysis Module

### Issues Identified
- The `validate_dataframe()` function is being called with an unsupported `check_crs` parameter
- Error: `TypeError: validate_dataframe() got an unexpected keyword argument 'check_crs'`
- The `create_conflict_adjusted_weights()` function has an issue with Numba compilation
- Error: `UnsupportedBytecodeError: Use of unsupported opcode (IMPORT_NAME)`
- The `calculate_impacts()` method is missing from the `SpatialEconometrics` class
- The `data` attribute is missing from the `SpatialEconometrics` class (using `gdf` instead)

### Implementation Plan
1. Update the `validate_geodataframe()` function in `src/utils/validation.py` to handle the `check_crs` parameter:
   ```python
   def validate_geodataframe(gdf, check_crs=True, **kwargs):
       # Add check_crs parameter handling
       is_valid, errors = validate_dataframe(gdf, **{k: v for k, v in kwargs.items() if k != 'check_crs'})
       
       if check_crs and not gdf.crs:
           errors.append("GeoDataFrame missing coordinate reference system (CRS)")
       
       # Rest of the function...
   ```

2. Fix the Numba compilation issue in `create_conflict_adjusted_weights()` in `src/utils/spatial_utils.py`:
   - Remove the `@numba.jit` decorator if it's causing issues
   - Alternatively, refactor the function to avoid using Python imports inside Numba-compiled code
   - Consider implementing a pure Python version as a fallback

3. Implement the `calculate_impacts()` method in the `SpatialEconometrics` class:
   ```python
   def calculate_impacts(self, model='lag'):
       """
       Calculate direct, indirect, and total effects for spatial models.
       
       Parameters
       ----------
       model : str
           Type of model to calculate impacts for ('lag' or 'error')
           
       Returns
       -------
       dict
           Dictionary of direct, indirect, and total effects
       """
       if model == 'lag' and hasattr(self, 'lag_model'):
           # Calculate impacts for lag model
           # ...
       elif model == 'error' and hasattr(self, 'error_model'):
           # Calculate impacts for error model
           # ...
       else:
           raise ValueError(f"Model {model} not available or not estimated")
   ```

4. Standardize attribute naming in the `SpatialEconometrics` class:
   - Either rename `gdf` to `data` throughout the class
   - Or update all references to `data` to use `gdf` instead

## 3. Simulation Module

### Issues Identified
- The `MarketIntegrationSimulation` class doesn't accept a `commodity` parameter
- The `calculate_baseline()` method is missing from the `MarketIntegrationSimulation` class
- The `simulate_reduced_conflict()` method is missing
- The `simulate_exchange_unification()` method is missing
- The `simulate_combined_policies()` method is missing
- The `run_simulation()` method is missing (used in sensitivity analysis)

### Implementation Plan
1. Update the `MarketIntegrationSimulation` class constructor to accept the `commodity` parameter:
   ```python
   def __init__(self, data, commodity=None):
       self.data = data
       self.commodity = commodity
       # Rest of initialization...
   ```

2. Implement the `calculate_baseline()` method:
   ```python
   def calculate_baseline(self):
       """
       Calculate baseline scenario for market integration simulation.
       
       Returns
       -------
       dict
           Baseline scenario metrics
       """
       north_data = self.data[self.data['exchange_rate_regime'] == 'north']
       south_data = self.data[self.data['exchange_rate_regime'] == 'south']
       
       return {
           'avg_price_north': north_data['price'].mean(),
           'avg_price_south': south_data['price'].mean(),
           'price_differential': abs(north_data['price'].mean() - south_data['price'].mean()),
           'price_volatility': self.data['price_volatility'].mean(),
           'integration_index': self._calculate_integration_index()
       }
   ```

3. Implement the `simulate_reduced_conflict()` method:
   ```python
   def simulate_reduced_conflict(self, reduction_factor=0.5):
       """
       Simulate the impact of conflict reduction on market integration.
       
       Parameters
       ----------
       reduction_factor : float
           Conflict reduction factor (0-1)
           
       Returns
       -------
       dict
           Simulation results
       """
       baseline = self.calculate_baseline()
       
       # Implement actual simulation logic based on economic theory
       # This should use actual data and models rather than simple scaling
       
       return {
           'avg_price_north': baseline['avg_price_north'] * (1 - reduction_factor * 0.1),
           'avg_price_south': baseline['avg_price_south'] * (1 - reduction_factor * 0.1),
           'price_differential': baseline['price_differential'] * (1 - reduction_factor * 0.2),
           'price_volatility': baseline['price_volatility'] * (1 - reduction_factor * 0.15),
           'integration_index': baseline['integration_index'] * (1 + reduction_factor * 0.1)
       }
   ```

4. Implement the `simulate_exchange_unification()` method:
   ```python
   def simulate_exchange_unification(self, method='official'):
       """
       Simulate the impact of exchange rate unification on market integration.
       
       Parameters
       ----------
       method : str
           Exchange rate unification method ('official', 'market', or 'average')
           
       Returns
       -------
       dict
           Simulation results
       """
       baseline = self.calculate_baseline()
       
       # Implement actual simulation logic based on economic theory
       # This should use actual data and models rather than simple scaling
       
       # Apply different effects based on the unification method
       method_factor = 0.0
       if method == 'market':
           method_factor = 0.1
       elif method == 'average':
           method_factor = 0.05
       
       return {
           'avg_price_north': baseline['avg_price_north'] * (1 - 0.05),
           'avg_price_south': baseline['avg_price_south'] * (1 + 0.05),
           'price_differential': baseline['price_differential'] * 0.7,
           'price_volatility': baseline['price_volatility'] * 0.8,
           'integration_index': baseline['integration_index'] * (1.2 + method_factor)
       }
   ```

5. Implement the `simulate_combined_policies()` method:
   ```python
   def simulate_combined_policies(self, reduction_factor=0.5, unification_method='official'):
       """
       Simulate the combined impact of conflict reduction and exchange rate unification.
       
       Parameters
       ----------
       reduction_factor : float
           Conflict reduction factor (0-1)
       unification_method : str
           Exchange rate unification method ('official', 'market', or 'average')
           
       Returns
       -------
       dict
           Simulation results
       """
       baseline = self.calculate_baseline()
       
       # Implement actual simulation logic based on economic theory
       # This should account for interaction effects between policies
       
       # Apply different effects based on the unification method
       method_factor = 0.0
       if unification_method == 'market':
           method_factor = 0.1
       elif unification_method == 'average':
           method_factor = 0.05
       
       return {
           'avg_price_north': baseline['avg_price_north'] * (1 - reduction_factor * 0.15),
           'avg_price_south': baseline['avg_price_south'] * (1 - reduction_factor * 0.05),
           'price_differential': baseline['price_differential'] * (1 - reduction_factor * 0.3),
           'price_volatility': baseline['price_volatility'] * (1 - reduction_factor * 0.25),
           'integration_index': baseline['integration_index'] * (1 + reduction_factor * 0.2 * (1 + method_factor))
       }
   ```

6. Implement the `run_simulation()` method for sensitivity analysis:
   ```python
   def run_simulation(self, **params):
       """
       Run a simulation with the specified parameters.
       
       Parameters
       ----------
       **params : dict
           Simulation parameters
           
       Returns
       -------
       dict
           Simulation results
       """
       if 'reduction_factor' in params and 'exchange_rate_method' in params:
           return self.simulate_combined_policies(
               reduction_factor=params['reduction_factor'],
               unification_method=params['exchange_rate_method']
           )
       elif 'reduction_factor' in params:
           return self.simulate_reduced_conflict(reduction_factor=params['reduction_factor'])
       elif 'exchange_rate_method' in params:
           return self.simulate_exchange_unification(method=params['exchange_rate_method'])
       else:
           return self.calculate_baseline()
   ```

## 4. Visualization Module

### Issues Identified
- Several visualization parameters are commented out as unsupported:
  - `style='publication'`
  - `include_events=True`
  - `include_trend=True`
  - `annotate_outliers=True`
- Parameter name inconsistencies (e.g., `include_legend` vs `legend`)

### Implementation Plan
1. Implement the `style` parameter in visualization functions:
   ```python
   def plot_price_series(self, data, group_col='admin1', title=None, style=None):
       # Set matplotlib style based on the style parameter
       if style == 'publication':
           plt.style.use('seaborn-whitegrid')
           # Additional publication-quality settings
           plt.rcParams['font.family'] = 'serif'
           plt.rcParams['font.size'] = 12
           plt.rcParams['axes.labelsize'] = 14
           plt.rcParams['axes.titlesize'] = 16
           plt.rcParams['figure.figsize'] = (10, 6)
       
       # Rest of the function...
   ```

2. Implement the `include_events` parameter:
   ```python
   def plot_price_series(self, data, group_col='admin1', title=None, include_events=False):
       # Plot time series
       # ...
       
       if include_events:
           # Add vertical lines for significant events
           events = {
               '2020-03-01': 'COVID-19 Pandemic',
               '2021-06-01': 'Major Conflict Escalation',
               '2022-04-01': 'Ceasefire Agreement'
           }
           
           for date_str, label in events.items():
               date = pd.to_datetime(date_str)
               if date >= data['date'].min() and date <= data['date'].max():
                   plt.axvline(date, color='red', linestyle='--', alpha=0.7)
                   plt.text(date, plt.ylim()[1]*0.95, label, rotation=90, verticalalignment='top')
       
       # Rest of the function...
   ```

3. Implement the `include_trend` parameter:
   ```python
   def plot_price_differentials(self, data, title=None, include_trend=False):
       # Plot differentials
       # ...
       
       if include_trend:
           # Add trend line using polynomial regression
           x = np.array(range(len(data)))
           y = data['price_differential'].values
           
           # Fit polynomial of degree 2
           z = np.polyfit(x, y, 2)
           p = np.poly1d(z)
           
           # Add trend line to plot
           plt.plot(data['date'], p(x), 'r--', label='Trend')
           plt.legend()
       
       # Rest of the function...
   ```

4. Implement the `annotate_outliers` parameter:
   ```python
   def plot_static_map(self, gdf, column='price', title=None, cmap='viridis', legend=True, annotate_outliers=False):
       # Plot map
       # ...
       
       if annotate_outliers:
           # Identify outliers using IQR method
           q1 = gdf[column].quantile(0.25)
           q3 = gdf[column].quantile(0.75)
           iqr = q3 - q1
           outliers = gdf[(gdf[column] < q1 - 1.5*iqr) | (gdf[column] > q3 + 1.5*iqr)]
           
           # Annotate outliers on the map
           for idx, row in outliers.iterrows():
               plt.annotate(
                   f"{row['admin1']}: {row[column]:.2f}",
                   xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                   xytext=(10, 10),
                   textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
               )
       
       # Rest of the function...
   ```

5. Standardize parameter naming across visualization functions:
   - Use `legend` consistently instead of `include_legend`
   - Use `style` consistently for styling options
   - Use `annotate` consistently for annotation options

## 5. Validation and Error Handling

### Issues Identified
- Missing validation for required columns in data
- Inconsistent error handling for missing data
- Lack of proper validation for simulation parameters

### Implementation Plan
1. Implement comprehensive data validation in the `validate_data()` function:
   ```python
   def validate_data(gdf, logger):
       """
       Validate input data for analysis.
       
       Parameters
       ----------
       gdf : geopandas.GeoDataFrame
           Input data
       logger : logging.Logger
           Logger instance
           
       Returns
       -------
       bool
           True if data is valid, False otherwise
       """
       required_columns = [
           'date', 'commodity', 'price', 'admin1', 'geometry', 
           'exchange_rate_regime', 'conflict_intensity_normalized'
       ]
       
       missing_columns = set(required_columns) - set(gdf.columns)
       if missing_columns:
           logger.error(f"Missing required columns: {', '.join(missing_columns)}")
           return False
       
       # Check for missing values in critical columns
       for col in ['date', 'commodity', 'price', 'admin1', 'exchange_rate_regime']:
           if gdf[col].isna().any():
               logger.warning(f"Column '{col}' has {gdf[col].isna().sum()} missing values")
       
       # Check for valid geometry
       if gdf.geometry.isna().any():
           logger.warning(f"Found {gdf.geometry.isna().sum()} rows with missing geometry")
       
       # Check for valid date range
       date_range = gdf['date'].max() - gdf['date'].min()
       if date_range.days < 365:
           logger.warning(f"Data spans less than a year ({date_range.days} days)")
       
       return True
   ```

2. Implement proper error handling for missing data in simulation methods:
   ```python
   def calculate_baseline(self):
       north_data = self.data[self.data['exchange_rate_regime'] == 'north']
       south_data = self.data[self.data['exchange_rate_regime'] == 'south']
       
       if len(north_data) == 0 or len(south_data) == 0:
           raise ValueError("Missing data for north or south regions")
       
       # Rest of the method...
   ```

3. Implement parameter validation for simulation methods:
   ```python
   def simulate_reduced_conflict(self, reduction_factor=0.5):
       if not 0 <= reduction_factor <= 1:
           raise ValueError("Reduction factor must be between 0 and 1")
       
       # Rest of the method...
   
   def simulate_exchange_unification(self, method='official'):
       valid_methods = ['official', 'market', 'average']
       if method not in valid_methods:
           raise ValueError(f"Method must be one of {valid_methods}")
       
       # Rest of the method...
   ```

## 6. Performance Optimization

### Issues Identified
- Potential performance issues with large datasets
- Lack of parallel processing for computationally intensive operations
- Memory management issues with large GeoDataFrames

### Implementation Plan
1. Implement chunked processing for large datasets:
   ```python
   def process_large_dataset(data, chunk_size=1000):
       """
       Process a large dataset in chunks to reduce memory usage.
       
       Parameters
       ----------
       data : pandas.DataFrame
           Input data
       chunk_size : int
           Size of each chunk
           
       Returns
       -------
       pandas.DataFrame
           Processed data
       """
       results = []
       
       for i in range(0, len(data), chunk_size):
           chunk = data.iloc[i:i+chunk_size].copy()
           # Process chunk
           processed_chunk = process_chunk(chunk)
           results.append(processed_chunk)
       
       return pd.concat(results)
   ```

2. Implement parallel processing for computationally intensive operations:
   ```python
   def parallel_process(data, func, n_jobs=-1):
       """
       Process data in parallel using multiple cores.
       
       Parameters
       ----------
       data : list
           List of data items to process
       func : function
           Function to apply to each item
       n_jobs : int
           Number of jobs to run in parallel (-1 for all cores)
           
       Returns
       -------
       list
           Processed data
       """
       from concurrent.futures import ProcessPoolExecutor
       import multiprocessing as mp
       
       if n_jobs == -1:
           n_jobs = mp.cpu_count()
       
       with ProcessPoolExecutor(max_workers=n_jobs) as executor:
           results = list(executor.map(func, data))
       
       return results
   ```

3. Implement memory optimization for GeoDataFrames:
   ```python
   def optimize_geodataframe(gdf):
       """
       Optimize a GeoDataFrame to reduce memory usage.
       
       Parameters
       ----------
       gdf : geopandas.GeoDataFrame
           Input GeoDataFrame
           
       Returns
       -------
       geopandas.GeoDataFrame
           Optimized GeoDataFrame
       """
       # Convert float64 to float32
       float_cols = gdf.select_dtypes(include=['float64']).columns
       for col in float_cols:
           gdf[col] = gdf[col].astype('float32')
       
       # Convert int64 to int32
       int_cols = gdf.select_dtypes(include=['int64']).columns
       for col in int_cols:
           gdf[col] = gdf[col].astype('int32')
       
       # Convert object to category for columns with few unique values
       obj_cols = gdf.select_dtypes(include=['object']).columns
       for col in obj_cols:
           if gdf[col].nunique() / len(gdf) < 0.5:  # If less than 50% unique values
               gdf[col] = gdf[col].astype('category')
       
       return gdf
   ```

## 7. Documentation and Testing

### Issues Identified
- Incomplete or missing docstrings
- Lack of comprehensive unit tests
- Missing examples and usage documentation

### Implementation Plan
1. Complete docstrings for all functions and classes:
   ```python
   def function_name(param1, param2=None):
       """
       Brief description of the function.
       
       Detailed description of the function, including its purpose,
       behavior, and any important implementation details.
       
       Parameters
       ----------
       param1 : type
           Description of param1
       param2 : type, optional
           Description of param2, default is None
           
       Returns
       -------
       type
           Description of return value
           
       Raises
       ------
       ExceptionType
           Description of when this exception is raised
           
       Examples
       --------
       >>> function_name(1)
       Expected output
       
       >>> function_name(1, 2)
       Expected output
       """
   ```

2. Implement comprehensive unit tests:
   ```python
   def test_function_name():
       """Test the function_name function."""
       # Test with valid inputs
       result = function_name(1)
       assert result == expected_result
       
       # Test with edge cases
       result = function_name(0)
       assert result == expected_edge_case_result
       
       # Test with invalid inputs
       with pytest.raises(ValueError):
           function_name(-1)
   ```

3. Create usage examples and documentation:
   ```markdown
   # Function Name
   
   ## Description
   Brief description of the function.
   
   ## Usage
   ```python
   from module import function_name
   
   result = function_name(1)
   print(result)
   ```
   
   ## Parameters
   - `param1` (type): Description of param1
   - `param2` (type, optional): Description of param2, default is None
   
   ## Returns
   - (type): Description of return value
   
   ## Examples
   ```python
   # Example 1: Basic usage
   result = function_name(1)
   # Expected output
   
   # Example 2: With optional parameter
   result = function_name(1, 2)
   # Expected output
   ```
   ```

## 8. Integration with External Data Sources

### Issues Identified
- Lack of integration with external data sources
- Hard-coded data paths and formats
- Limited data validation for external sources

### Implementation Plan
1. Implement flexible data loading from various sources:
   ```python
   def load_data(source, **kwargs):
       """
       Load data from various sources.
       
       Parameters
       ----------
       source : str
           Data source type ('file', 'api', 'database')
       **kwargs : dict
           Source-specific parameters
           
       Returns
       -------
       pandas.DataFrame or geopandas.GeoDataFrame
           Loaded data
       """
       if source == 'file':
           path = kwargs.get('path')
           file_format = kwargs.get('format', 'csv')
           
           if file_format == 'csv':
               return pd.read_csv(path, **kwargs.get('options', {}))
           elif file_format == 'geojson':
               return gpd.read_file(path, **kwargs.get('options', {}))
           elif file_format == 'excel':
               return pd.read_excel(path, **kwargs.get('options', {}))
           else:
               raise ValueError(f"Unsupported file format: {file_format}")
       
       elif source == 'api':
           url = kwargs.get('url')
           headers = kwargs.get('headers', {})
           params = kwargs.get('params', {})
           
           import requests
           response = requests.get(url, headers=headers, params=params)
           response.raise_for_status()
           
           data_format = kwargs.get('format', 'json')
           if data_format == 'json':
               return pd.DataFrame(response.json())
           elif data_format == 'csv':
               import io
               return pd.read_csv(io.StringIO(response.text))
           else:
               raise ValueError(f"Unsupported API data format: {data_format}")
       
       elif source == 'database':
           connection_string = kwargs.get('connection_string')
           query = kwargs.get('query')
           
           import sqlalchemy
           engine = sqlalchemy.create_engine(connection_string)
           return pd.read_sql(query, engine)
       
       else:
           raise ValueError(f"Unsupported data source: {source}")
   ```

2. Implement data validation for external sources:
   ```python
   def validate_external_data(data, source_type, logger):
       """
       Validate data from external sources.
       
       Parameters
       ----------
       data : pandas.DataFrame
           Data to validate
       source_type : str
           Type of external source ('conflict', 'economic', 'demographic')
       logger : logging.Logger
           Logger instance
           
       Returns
       -------
       bool
           True if data is valid, False otherwise
       """
       if source_type == 'conflict':
           required_columns = ['date', 'location', 'intensity', 'type']
       elif source_type == 'economic':
           required_columns = ['date', 'location', 'indicator', 'value']
       elif source_type == 'demographic':
           required_columns = ['location', 'population', 'year']
       else:
           logger.error(f"Unknown source type: {source_type}")
           return False
       
       missing_columns = set(required_columns) - set(data.columns)
       if missing_columns:
           logger.error(f"Missing required columns for {source_type} data: {', '.join(missing_columns)}")
           return False
       
       # Check for missing values in critical columns
       for col in required_columns:
           if data[col].isna().any():
               logger.warning(f"Column '{col}' has {data[col].isna().sum()} missing values")
       
       return True
   ```

3. Implement data integration with external sources:
   ```python
   def integrate_external_data(base_data, external_data, join_columns, logger):
       """
       Integrate external data with base data.
       
       Parameters
       ----------
       base_data : pandas.DataFrame
           Base data
       external_data : pandas.DataFrame
           External data to integrate
       join_columns : list
           Columns to join on
       logger : logging.Logger
           Logger instance
           
       Returns
       -------
       pandas.DataFrame
           Integrated data
       """
       # Check for join column existence
       for df, name in [(base_data, 'base_data'), (external_data, 'external_data')]:
           missing_columns = set(join_columns) - set(df.columns)
           if missing_columns:
               logger.error(f"Missing join columns in {name}: {', '.join(missing_columns)}")
               return base_data
       
       # Perform join
       integrated_data = base_data.merge(
           external_data,
           on=join_columns,
           how='left',
           indicator=True
       )
       
       # Log join statistics
       match_count = (integrated_data['_merge'] == 'both').sum()
       logger.info(f"Matched {match_count} out of {len(base_data)} records ({match_count/len(base_data)*100:.2f}%)")
       
       # Remove indicator column
       integrated_data = integrated_data.drop(columns=['_merge'])
       
       return integrated_data
   ```

## 9. Deployment and Automation

### Issues Identified
- Lack of automated deployment process
- Manual execution of analysis scripts
- Limited scheduling and monitoring capabilities

### Implementation Plan
1. Implement a deployment script:
   ```python
   def deploy_analysis_pipeline(config_path):
       """
       Deploy the analysis pipeline.
       
       Parameters
       ----------
       config_path : str
           Path to configuration file
           
       Returns
       -------
       bool
           True if deployment was successful, False otherwise
       """
       import yaml
       import subprocess
       import os
       
       # Load configuration
       with open(config_path, 'r') as f:
           config = yaml.safe_load(f)
       
       # Create output directories
       for directory in config.get('output_directories', []):
           os.makedirs(directory, exist_ok=True)
       
       # Install dependencies
       if 'dependencies' in config:
           subprocess.run(['pip', 'install', '-r', config['dependencies']])
       
       # Set up environment variables
       for key, value in config.get('environment_variables', {}).items():
           os.environ[key] = value
       
       # Run setup scripts
       for script in config.get('setup_scripts', []):
           subprocess.run(['python', script])
       
       return True
   ```

2. Implement a scheduling script:
   ```python
   def schedule_analysis(config_path):
       """
       Schedule the analysis to run periodically.
       
       Parameters
       ----------
       config_path : str
           Path to configuration file
           
       Returns
       -------
       bool
           True if scheduling was successful, False otherwise
       """
       import yaml
       import schedule
       import time
       import subprocess
       import logging
       
       # Set up logging
       logging.basicConfig(
           level=logging.INFO,
           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
           filename='schedule.log'
       )
       logger = logging.getLogger('scheduler')
       
       # Load configuration
       with open(config_path, 'r') as f:
           config = yaml.safe_load(f)
       
       # Define job function
       def job():
           logger.info("Starting scheduled analysis")
           try:
               cmd = ['python', config['script']]
               cmd.extend(config.get('arguments', []))
               result = subprocess.run(cmd, capture_output=True, text=True)
               if result.returncode == 0:
                   logger.info("Analysis completed successfully")
               else:
                   logger.error(f"Analysis failed: {result.stderr}")
           except Exception as e:
               logger.error(f"Error running analysis: {e}")
       
       # Schedule job
       schedule_config = config.get('schedule', {})
       interval = schedule_config.get('interval', 'daily')
       time_str = schedule_config.get('time', '00:00')
       
       if interval == 'hourly':
           schedule.every().hour.do(job)
       elif interval == 'daily':
           schedule.every().day.at(time_str).do(job)
       elif interval == 'weekly':
           day = schedule_config.get('day', 'monday')
           schedule.every().week.at(f"{day} {time_str}").do(job)
       elif interval == 'monthly':
           day = schedule_config.get('day', 1)
           schedule.every().month.at(f"day {day} {time_str}").do(job)
       else:
           logger.error(f"Unknown interval: {interval}")
           return False
       
       logger.info(f"Analysis scheduled to run {interval} at {time_str}")
       
       # Run the scheduler
       while True:
           schedule.run_pending()
           time.sleep(60)
       
       return True
   ```

3. Implement a monitoring script:
   ```python
   def monitor_analysis(log_path, alert_threshold=5):
       """
       Monitor the analysis and send alerts if errors exceed threshold.
       
       Parameters
       ----------
       log_path : str
           Path to log file
       alert_threshold : int
           Number of errors before sending an alert
           
       Returns
       -------
       bool
           True if monitoring was successful, False otherwise
       """
       import re
       import time
       import smtplib
       import os
       from email.message import EmailMessage
       
       # Set up email configuration
       email_config = {
           'sender': os.environ.get('ALERT_EMAIL_SENDER'),
           'recipient': os.environ.get('ALERT_EMAIL_RECIPIENT'),
           'smtp_server': os.environ.get('ALERT_SMTP_SERVER'),
           'smtp_port': int(os.environ.get('ALERT_SMTP_PORT', 587)),
           'username': os.environ.get('ALERT_SMTP_USERNAME'),
           'password': os.environ.get('ALERT_SMTP_PASSWORD')
       }
       
       # Check if email configuration is complete
       if not all(email_config.values()):
           print("Email configuration incomplete, alerts will not be sent")
       
       # Monitor log file
       error_count = 0
       last_position = 0
       
       while True:
           try:
               with open(log_path, 'r') as f:
                   f.seek(last_position)
                   new_lines = f.readlines()
                   last_position = f.tell()
               
               # Count errors in new lines
               for line in new_lines:
                   if re.search(r'ERROR|CRITICAL', line):
                       error_count += 1
               
               # Send alert if error count exceeds threshold
               if error_count >= alert_threshold and all(email_config.values()):
                   msg = EmailMessage()
                   msg['Subject'] = 'Analysis Error Alert'
                   msg['From'] = email_config['sender']
                   msg['To'] = email_config['recipient']
                   msg.set_content(f"Analysis has encountered {error_count} errors. Please check the logs.")
                   
                   with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                       server.starttls()
                       server.login(email_config['username'], email_config['password'])
                       server.send_message(msg)
                   
                   print(f"Alert sent: {error_count} errors detected")
                   error_count = 0  # Reset error count after sending alert
               
               time.sleep(60)  # Check every minute
           except Exception as e:
               print(f"Error monitoring log file: {e}")
               time.sleep(300)  # Wait 5 minutes before retrying
       
       return True
   ```

## 10. User Interface and Reporting

### Issues Identified
- Limited user interface for analysis results
- Basic reporting capabilities
- Lack of interactive visualizations

### Implementation Plan
1. Implement a web-based dashboard:
   ```python
   def create_dashboard(data_path, output_path):
       """
       Create a web-based dashboard for analysis results.
       
       Parameters
       ----------
       data_path : str
           Path to data file
       output_path : str
           Path to save dashboard
           
       Returns
       -------
       str
           Path to dashboard HTML file
       """
       import dash
       import dash_core_components as dcc
       import dash_html_components as html
       import plotly.express as px
       import pandas as pd
       
       # Load data
       data = pd.read_json(data_path)
       
       # Create app
       app = dash.Dash(__name__)
       
       # Create layout
       app.layout = html.Div([
           html.H1("Yemen Market Integration Analysis"),
           
           html.Div([
               html.H2("Price Trends"),
               dcc.Graph(
                   id='price-trends',
                   figure=px.line(
                       data,
                       x='date',
                       y='price',
                       color='admin1',
                       title='Price Trends by Region'
                   )
               )
           ]),
           
           html.Div([
               html.H2("Price Differentials"),
               dcc.Graph(
                   id='price-differentials',
                   figure=px.line(
                       data.groupby(['date', 'exchange_rate_regime'])['price'].mean().reset_index(),
                       x='date',
                       y='price',
                       color='exchange_rate_regime',
                       title='Price Differentials: North vs South'
                   )
               )
           ]),
           
           html.Div([
               html.H2("Spatial Distribution"),
               dcc.Graph(
                   id='spatial-distribution',
                   figure=px.scatter_mapbox(
                       data,
                       lat='latitude',
                       lon='longitude',
                       color='price',
                       size='price_volatility',
                       hover_name='admin1',
                       mapbox_style='carto-positron',
                       zoom=5,
                       title='Spatial Distribution of Prices'
                   )
               )
           ])
       ])
       
       # Save to HTML
       html_path = f"{output_path}/dashboard.html"
       with open(html_path, 'w') as f:
           f.write(app.index_string)
       
       return html_path
   ```

2. Implement interactive visualizations:
   ```python
   def create_interactive_visualization(data, output_path):
       """
       Create interactive visualizations for analysis results.
       
       Parameters
       ----------
       data : pandas.DataFrame
           Data to visualize
       output_path : str
           Path to save visualizations
           
       Returns
       -------
       list
           Paths to visualization HTML files
       """
       import plotly.express as px
       import plotly.graph_objects as go
       from plotly.subplots import make_subplots
       
       # Create time series visualization
       fig_ts = px.line(
           data,
           x='date',
           y='price',
           color='admin1',
           title='Price Trends by Region',
           labels={'price': 'Price', 'date': 'Date', 'admin1': 'Region'},
           template='plotly_white'
       )
       
       fig_ts.update_layout(
           hovermode='x unified',
           legend=dict(
               orientation='h',
               yanchor='bottom',
               y=1.02,
               xanchor='right',
               x=1
           )
       )
       
       ts_path = f"{output_path}/price_trends_interactive.html"
       fig_ts.write_html(ts_path, include_plotlyjs='cdn')
       
       # Create map visualization
       fig_map = px.scatter_mapbox(
           data,
           lat='latitude',
           lon='longitude',
           color='price',
           size='price_volatility',
           hover_name='admin1',
           hover_data=['price', 'price_volatility', 'conflict_intensity_normalized'],
           mapbox_style='carto-positron',
           zoom=5,
           title='Spatial Distribution of Prices',
           labels={'price': 'Price', 'price_volatility': 'Volatility', 'conflict_intensity_normalized': 'Conflict Intensity'}
       )
       
       map_path = f"{output_path}/price_map_interactive.html"
       fig_map.write_html(map_path, include_plotlyjs='cdn')
       
       # Create combined visualization
       fig_combined = make_subplots(
           rows=2, cols=2,
           subplot_titles=(
               'Price Trends by Region',
               'Price Differentials: North vs South',
               'Price Volatility',
               'Conflict Intensity'
           ),
           specs=[
               [{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]
           ]
       )
       
       # Add traces
       for region in data['admin1'].unique():
           region_data = data[data['admin1'] == region]
           fig_combined.add_trace(
               go.Scatter(
                   x=region_data['date'],
                   y=region_data['price'],
                   name=region,
                   mode='lines'
               ),
               row=1, col=1
           )
       
       # Add price differentials
       north_data = data[data['exchange_rate_regime'] == 'north'].groupby('date')['price'].mean().reset_index()
       south_data = data[data['exchange_rate_regime'] == 'south'].groupby('date')['price'].mean().reset_index()
       
       fig_combined.add_trace(
           go.Scatter(
               x=north_data['date'],
               y=north_data['price'],
               name='North',
               mode='lines'
           ),
           row=1, col=2
       )
       
       fig_combined.add_trace(
           go.Scatter(
               x=south_data['date'],
               y=south_data['price'],
               name='South',
               mode='lines'
           ),
           row=1, col=2
       )
       
       # Add price volatility
       for region in data['admin1'].unique():
           region_data = data[data['admin1'] == region]
           fig_combined.add_trace(
               go.Scatter(
                   x=region_data['date'],
                   y=region_data['price_volatility'],
                   name=region,
                   mode='lines'
               ),
               row=2, col=1
           )
       
       # Add conflict intensity
       for region in data['admin1'].unique():
           region_data = data[data['admin1'] == region]
           fig_combined.add_trace(
               go.Scatter(
                   x=region_data['date'],
                   y=region_data['conflict_intensity_normalized'],
                   name=region,
                   mode='lines'
               ),
               row=2, col=2
           )
       
       fig_combined.update_layout(
           height=800,
           showlegend=False,
           title_text='Yemen Market Integration Analysis'
       )
       
       combined_path = f"{output_path}/combined_interactive.html"
       fig_combined.write_html(combined_path, include_plotlyjs='cdn')
       
       return [ts_path, map_path, combined_path]
   ```

3. Implement automated reporting:
   ```python
   def generate_report(data, output_path, template_path=None):
       """
       Generate a comprehensive report of analysis results.
       
       Parameters
       ----------
       data : dict
           Analysis results
       output_path : str
           Path to save report
       template_path : str, optional
           Path to report template
           
       Returns
       -------
       str
           Path to generated report
       """
       import jinja2
       import markdown
       import os
       
       # Load template
       if template_path:
           with open(template_path, 'r') as f:
               template_str = f.read()
       else:
           template_str = """
           # Yemen Market Integration Analysis
           
           ## Overview
           
           This report presents the results of market integration analysis for {{ commodity }} in Yemen.
           
           ## Threshold Cointegration Analysis
           
           {% if threshold_results %}
           The threshold cointegration analysis examines the price transmission between north and south markets.
           
           {% if threshold_results.cointegrated %}
           The markets are cointegrated with a threshold value of {{ threshold_results.threshold }}.
           
           The adjustment parameters are:
           - Below threshold (north): {{ threshold_results.adjustment_below_1 }}
           - Above threshold (north): {{ threshold_results.adjustment_above_1 }}
           - Below threshold (south): {{ threshold_results.adjustment_below_2 }}
           - Above threshold (south): {{ threshold_results.adjustment_above_2 }}
           {% else %}
           The markets are not cointegrated, indicating limited price transmission between north and south.
           {% endif %}
           {% else %}
           No threshold cointegration analysis was performed.
           {% endif %}
           
           ## Spatial Analysis
           
           {% if spatial_results %}
           The spatial analysis examines the geographic patterns of market integration.
           
           {% if spatial_results.global_moran %}
           Global Moran's I: {{ spatial_results.global_moran.I }}
           p-value: {{ spatial_results.global_moran.p }}
           
           {% if spatial_results.global_moran.p < 0.05 %}
           There is significant spatial autocorrelation in prices, indicating that prices in neighboring markets are related.
           {% else %}
           There is no significant spatial autocorrelation in prices, indicating that prices in neighboring markets are not strongly related.
           {% endif %}
           {% endif %}
           
           {% if spatial_results.lag_model %}
           The spatial lag model shows a spatial dependence parameter (Rho) of {{ spatial_results.lag_model.rho }}.
           {% endif %}
           
           {% if spatial_results.error_model %}
           The spatial error model shows a spatial error parameter (Lambda) of {{ spatial_results.error_model.lambda_ }}.
           {% endif %}
           {% else %}
           No spatial analysis was performed.
           {% endif %}
           
           ## Policy Simulation
           
           {% if simulation_results %}
           The policy simulation analysis examines the potential impacts of conflict reduction and exchange rate unification.
           
           ### Conflict Reduction
           
           A {{ simulation_results.reduction_factor * 100 }}% reduction in conflict intensity would result in:
           - {{ simulation_results.conflict_reduction.price_differential_change * 100 }}% reduction in price differentials
           - {{ simulation_results.conflict_reduction.price_volatility_change * 100 }}% reduction in price volatility
           - {{ simulation_results.conflict_reduction.integration_index_change * 100 }}% increase in market integration
           
           ### Exchange Rate Unification
           
           Exchange rate unification using the {{ simulation_results.unification_method }} method would result in:
           - {{ simulation_results.exchange_unification.price_differential_change * 100 }}% reduction in price differentials
           - {{ simulation_results.exchange_unification.price_volatility_change * 100 }}% reduction in price volatility
           - {{ simulation_results.exchange_unification.integration_index_change * 100 }}% increase in market integration
           
           ### Combined Policies
           
           Implementing both policies would result in:
           - {{ simulation_results.combined_policies.price_differential_change * 100 }}% reduction in price differentials
           - {{ simulation_results.combined_policies.price_volatility_change * 100 }}% reduction in price volatility
           - {{ simulation_results.combined_policies.integration_index_change * 100 }}% increase in market integration
           {% else %}
           No policy simulation was performed.
           {% endif %}
           
           ## Conclusion
           
           {% if threshold_results and threshold_results.cointegrated %}
           The markets are cointegrated, indicating long-run price transmission between north and south.
           {% elif threshold_results %}
           The markets are not cointegrated, indicating limited price transmission between north and south.
           {% endif %}
           
           {% if spatial_results and spatial_results.global_moran and spatial_results.global_moran.p < 0.05 %}
           There is significant spatial autocorrelation in prices, indicating that prices in neighboring markets are related.
           {% elif spatial_results and spatial_results.global_moran %}
           There is no significant spatial autocorrelation in prices, indicating that prices in neighboring markets are not strongly related.
           {% endif %}
           
           {% if simulation_results %}
           The policy simulations suggest that {{ 'combined policies' if simulation_results.combined_policies.integration_index_change > max(simulation_results.conflict_reduction.integration_index_change, simulation_results.exchange_unification.integration_index_change) else 'exchange rate unification' if simulation_results.exchange_unification.integration_index_change > simulation_results.conflict_reduction.integration_index_change else 'conflict reduction' }} would have the largest impact on market integration.
           {% endif %}
           """
       
       # Create template
       template = jinja2.Template(template_str)
       
       # Render template
       report_md = template.render(**data)
       
       # Convert to HTML
       report_html = markdown.markdown(report_md)
       
       # Save report
       md_path = f"{output_path}/report.md"
       html_path = f"{output_path}/report.html"
       
       with open(md_path, 'w') as f:
           f.write(report_md)
       
       with open(html_path, 'w') as f:
           f.write(f"""
           <!DOCTYPE html>
           <html>
           <head>
               <title>Yemen Market Integration Analysis</title>
               <style>
                   body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                   h1, h2, h3 {{ color: #333; }}
                   table {{ border-collapse: collapse; width: 100%; }}
                   th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                   th {{ background-color: #f2f2f2; }}
                   img {{ max-width: 100%; height: auto; }}
                   .figure {{ margin: 20px 0; text-align: center; }}
                   .figure img {{ max-width: 600px; }}
                   .figure .caption {{ font-style: italic; color: #666; }}
               </style>
           </head>
           <body>
               {report_html}
           </body>
           </html>
           """)
       
       return html_path
   ```

## Implementation Timeline

### Phase 1: Core Functionality (Weeks 1-2)
- Fix validation issues in Unit Root Testing Module
- Fix validation issues in Spatial Analysis Module
- Implement missing methods in Simulation Module
- Fix data attribute issues in Spatial Analysis Module

### Phase 2: Enhanced Functionality (Weeks 3-4)
- Implement missing visualization parameters
- Standardize parameter naming
- Implement comprehensive data validation
- Implement proper error handling

### Phase 3: Performance Optimization (Weeks 5-6)
- Implement chunked processing for large datasets
- Implement parallel processing for computationally intensive operations
- Implement memory optimization for GeoDataFrames

### Phase 4: Documentation and Testing (Weeks 7-8)
- Complete docstrings for all functions and classes
- Implement comprehensive unit tests
- Create usage examples and documentation

### Phase 5: Advanced Features (Weeks 9-10)
- Implement integration with external data sources
- Implement deployment and automation scripts
- Implement user interface and reporting features

## Conclusion

This implementation plan outlines the steps needed to complete the Yemen Market Analysis project. By addressing the identified issues and implementing the missing functionality, the project will be more robust, efficient, and user-friendly. The plan is organized into phases to prioritize critical fixes and enhancements, with a timeline that allows for thorough testing and documentation at each stage.