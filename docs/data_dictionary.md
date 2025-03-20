# Yemen Market Integration Project: Data Dictionary

This document provides a comprehensive reference for all data sources, structures, and transformations used in the Yemen Market Integration project. It serves as the authoritative guide for understanding the data components of the analysis.

## Table of Contents

1. [Market Data](#market-data)
2. [Conflict Data](#conflict-data)
3. [Exchange Rate Data](#exchange-rate-data)
4. [Geographic Information](#geographic-information)
5. [Derived Variables](#derived-variables)
6. [Data Transformations](#data-transformations)
7. [File Formats](#file-formats)

## Market Data

### WFP Market Price Data

**Source**: World Food Programme (WFP) Market Monitoring System  
**Frequency**: Weekly  
**Time Period**: 2016-2023  
**Geographic Coverage**: Multiple markets across Yemen's governorates  

#### Variables

| Variable | Type | Description | Units | Example Values |
|----------|------|-------------|-------|----------------|
| date | Date | Date of price observation | YYYY-MM-DD | 2022-06-01 |
| market_id | String | Unique market identifier | ID | YE_MKT_101 |
| market_name | String | Name of the market | Text | Central Sana'a Market |
| admin1 | String | Governorate name | Text | Sana'a |
| admin2 | String | District name | Text | As Sabain |
| longitude | Float | Market longitude coordinate | Decimal degrees | 44.2066 |
| latitude | Float | Market latitude coordinate | Decimal degrees | 15.3694 |
| commodity | String | Commodity type | Text | wheat, rice, sugar |
| price | Float | Price in local currency | YER | 650.00 |
| currency | String | Currency unit | Text | YER |
| unit | String | Quantity unit | Text | kg |
| quantity | Float | Quantity amount | Number | 1.0 |
| exchange_rate_regime | String | Exchange rate zone | Text | north, south |
| exchange_rate | Float | Local exchange rate to USD | YER/USD | 600.00 |

#### Processing Notes

- Price data is cleaned to remove outliers (values beyond 3σ from the mean)
- Missing values are interpolated using forward-fill for time series consistency
- Markets are classified into exchange rate regimes based on political control zones

## Conflict Data

### ACLED Conflict Event Data

**Source**: Armed Conflict Location & Event Data Project (ACLED)  
**Frequency**: Daily, aggregated to weekly  
**Time Period**: 2016-2023  
**Geographic Coverage**: Yemen nationwide  

#### Variables

| Variable | Type | Description | Units | Example Values |
|----------|------|-------------|-------|----------------|
| date | Date | Date of conflict event | YYYY-MM-DD | 2022-05-28 |
| event_id | String | Unique event identifier | ID | YEM38562 |
| event_type | String | Type of conflict event | Text | Battle, Violence against civilians |
| sub_event_type | String | More specific event categorization | Text | Armed clash, Shelling/artillery |
| admin1 | String | Governorate name | Text | Taiz |
| admin2 | String | District name | Text | Al Mukha |
| longitude | Float | Event longitude coordinate | Decimal degrees | 43.2500 |
| latitude | Float | Event latitude coordinate | Decimal degrees | 13.3167 |
| fatalities | Integer | Number of fatalities | Count | 5 |
| notes | String | Description of the event | Text | "Fighting between..." |

#### Derived Conflict Metrics

| Variable | Type | Description | Units | Calculation |
|----------|------|-------------|-------|-------------|
| conflict_intensity | Float | Weighted conflict intensity | Index (0-5) | Weighted sum of events by type and fatalities |
| conflict_intensity_normalized | Float | Normalized conflict intensity | 0-1 scale | Min-max scaling of conflict_intensity |
| conflict_buffer | Float | Spatial extent of conflict impact | km | Buffer around conflict events (10km default) |
| conflict_duration | Integer | Length of sustained conflict | Days | Consecutive days with conflict events |

#### Processing Notes

- Events are geo-coded and mapped to the nearest market
- Intensity is calculated using a weighted scheme: battles (weight=1.0), violence against civilians (weight=0.8), explosions (weight=1.2)
- For spatial analysis, conflict events create buffers that can intersect market catchment areas

## Exchange Rate Data

### Central Bank and Market Exchange Rates

**Source**: Central Bank of Yemen (Aden), Sana'a monetary authorities, and market surveys  
**Frequency**: Weekly  
**Time Period**: 2016-2023  
**Geographic Coverage**: Yemen nationwide, differentiated by control zones  

#### Variables

| Variable | Type | Description | Units | Example Values |
|----------|------|-------------|-------|----------------|
| date | Date | Date of exchange rate observation | YYYY-MM-DD | 2022-06-01 |
| official_rate_north | Float | Official exchange rate in north | YER/USD | 600.00 |
| official_rate_south | Float | Official exchange rate in south | YER/USD | 1200.00 |
| parallel_rate_north | Float | Parallel market rate in north | YER/USD | 610.00 |
| parallel_rate_south | Float | Parallel market rate in south | YER/USD | 1225.00 |
| rate_differential | Float | Difference between north and south rates | YER/USD | 600.00 |
| rate_differential_pct | Float | Percentage difference in rates | Percent | 100.00 |

#### Processing Notes

- Exchange rates are smoothed using a 7-day moving average to reduce volatility
- When official rates are unavailable, parallel market rates are used
- Rate differential is calculated as absolute and percentage differences

## Geographic Information

### Administrative Boundaries

**Source**: OCHA/UNOSAT Yemen administrative boundaries  
**Format**: GeoJSON/Shapefile  
**Coordinate Reference System**: EPSG:4326 (WGS84)  

#### Variables

| Variable | Type | Description | Example Values |
|----------|------|-------------|----------------|
| admin1Name | String | Governorate name | Sana'a |
| admin1Pcod | String | Governorate code | YE01 |
| admin2Name | String | District name | As Sabain |
| admin2Pcod | String | District code | YE0105 |
| admin_control | String | Political control classification | Internationally recognized government, Houthi |
| population | Integer | Population estimate | 25000 |
| area_sqkm | Float | Area in square kilometers | 120.5 |

### Market Locations

**Source**: WFP, merged with OCHA POI database  
**Format**: GeoJSON/Shapefile  
**Coordinate Reference System**: EPSG:4326 (WGS84)  

#### Variables

| Variable | Type | Description | Example Values |
|----------|------|-------------|----------------|
| market_id | String | Unique market identifier | YE_MKT_101 |
| market_name | String | Name of the market | Central Sana'a Market |
| market_type | String | Classification of market | Primary, Secondary, Terminal |
| market_size | String | Size of market | Small, Medium, Large |
| admin1 | String | Governorate name | Sana'a |
| admin2 | String | District name | As Sabain |
| longitude | Float | Market longitude coordinate | 44.2066 |
| latitude | Float | Market latitude coordinate | 15.3694 |
| geometry | Point | Spatial point geometry | POINT(44.2066 15.3694) |

## Derived Variables

### Market Integration Metrics

| Variable | Type | Description | Calculation |
|----------|------|-------------|-------------|
| price_differential | Float | Price difference between markets | price_market1 - price_market2 |
| price_differential_pct | Float | Percentage price difference | (price_market1 - price_market2) / price_market2 * 100 |
| price_ratio | Float | Ratio of prices between markets | price_market1 / price_market2 |
| integration_index | Float | Composite market integration metric | Weighted average of multiple integration indicators |
| half_life | Float | Half-life of price deviations | ln(0.5) / ln(1 + adjustment_speed) |
| threshold | Float | Estimated threshold parameter | Grid search from threshold cointegration |
| adjustment_speed_below | Float | Price adjustment below threshold | Coefficient from threshold model |
| adjustment_speed_above | Float | Price adjustment above threshold | Coefficient from threshold model |

### Spatial Integration Metrics

| Variable | Type | Description | Calculation |
|----------|------|-------------|-------------|
| market_accessibility | Float | Population-weighted market access | Sum of population / (distance^decay_parameter) |
| market_isolation | Float | Conflict-adjusted market isolation | Accessibility adjusted for conflict barriers |
| spatial_lag | Float | Spatial autoregressive effect | Coefficient from spatial models |
| price_spatial_correlation | Float | Moran's I for prices | Spatial autocorrelation statistic |
| regime_boundary | Boolean | Whether market pairs cross regime boundary | True if markets in different regimes |

## Data Transformations

### Time Series Transformations

| Transformation | Description | Purpose |
|----------------|-------------|---------|
| Logarithmic | Natural logarithm of price series | Stabilize variance, interpret as percentage changes |
| First Difference | Change between consecutive observations | Achieve stationarity for unit root series |
| Moving Average | N-period moving average smoothing | Reduce noise and highlight trends |
| Seasonal Adjustment | Remove seasonal patterns | Focus on trend and cycle components |
| Lag Creation | Creation of lagged variables | For autoregressive models |
| Price Index | Prices converted to index (base=100) | For better comparability across commodities |

### Spatial Transformations

| Transformation | Description | Purpose |
|----------------|-------------|---------|
| Coordinate Reprojection | Convert coordinates between reference systems | For accurate distance calculations |
| Distance Matrix | Calculate distances between all market pairs | For spatial relationship analysis |
| Spatial Weight Matrix | Weights based on proximity and conflict | For spatial econometric models |
| Conflict-Adjusted Distance | Increase effective distance by conflict intensity | To model conflict as a trade barrier |
| Market Catchments | Define market influence areas | For population served calculations |
| Border Effects | Account for administrative/political boundaries | For regime boundary effects |

## File Formats

### Raw Data Storage

| Data Type | Format | Location | Naming Convention | Example |
|-----------|--------|----------|-------------------|---------|
| Market Prices | CSV | data/raw/market/ | yyyy_mm_market_prices.csv | 2022_06_market_prices.csv |
| Conflict Events | GeoJSON | data/raw/conflict/ | acled_yemen_yyyy_mm.geojson | acled_yemen_2022_06.geojson |
| Exchange Rates | CSV | data/raw/exchange/ | exchange_rates_yyyy_mm.csv | exchange_rates_2022_06.csv |
| Geographic | Shapefile/GeoJSON | data/raw/geographic/ | yemen_admin_level_n.geojson | yemen_admin_level_1.geojson |

### Processed Data Storage

| Data Type | Format | Location | Naming Convention | Example |
|-----------|--------|----------|-------------------|---------|
| Integrated Dataset | GeoJSON | data/processed/ | unified_data_yyyy_mm.geojson | unified_data_2022_06.geojson |
| Analysis Results | CSV | data/processed/results/ | tvecm_results_commodity.csv | tvecm_results_wheat.csv |
| Simulation Results | CSV | data/processed/simulations/ | simulation_scenario_yyyy_mm.csv | simulation_exchangerate_2022_06.csv |
| Visualization Data | GeoJSON | data/processed/visualizations/ | visualization_type_yyyy_mm.geojson | visualization_markets_2022_06.geojson |
