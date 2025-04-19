"""
Conflict integration module for Yemen Market Analysis.

This module provides the ConflictIntegration class for analyzing the impact of
conflict on market integration.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import libpysal.weights as weights

from src.config import config
from src.utils.error_handling import YemenAnalysisError, handle_errors
from src.utils.validation import validate_data
from src.models.spatial.weights import SpatialWeightMatrix

# Initialize logger
logger = logging.getLogger(__name__)

class ConflictIntegration:
    """
    Conflict integration analysis for Yemen Market Analysis.

    This class provides methods for analyzing the impact of conflict on market integration.

    Attributes:
        data (gpd.GeoDataFrame): GeoDataFrame containing spatial data.
        w (weights.W): Spatial weight matrix.
        conflict_column (str): Column containing conflict intensity.
        price_column (str): Column containing prices.
        results (Dict[str, Any]): Analysis results.
    """

    def __init__(
        self, data: Optional[gpd.GeoDataFrame] = None,
        w: Optional[weights.W] = None,
        conflict_column: str = 'conflict_intensity',
        price_column: str = 'price'
    ):
        """
        Initialize the conflict integration analysis.

        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
            conflict_column: Column containing conflict intensity.
            price_column: Column containing prices.
        """
        self.data = data
        self.w = w
        self.conflict_column = conflict_column
        self.price_column = price_column
        self.results = {}

    @handle_errors
    def set_data(
        self, data: gpd.GeoDataFrame, w: Optional[weights.W] = None,
        conflict_column: Optional[str] = None, price_column: Optional[str] = None
    ) -> None:
        """
        Set the data for the analysis.

        Args:
            data: GeoDataFrame containing spatial data.
            w: Spatial weight matrix.
            conflict_column: Column containing conflict intensity.
            price_column: Column containing prices.

        Raises:
            YemenAnalysisError: If the data is invalid.
        """
        logger.info("Setting data for conflict integration analysis")

        # Validate data
        validate_data(data, 'spatial')

        # Set conflict column
        if conflict_column is not None:
            self.conflict_column = conflict_column

        # Set price column
        if price_column is not None:
            self.price_column = price_column

        # Check if columns exist
        if self.conflict_column not in data.columns:
            logger.error(f"Conflict column {self.conflict_column} not found in data")
            raise YemenAnalysisError(f"Conflict column {self.conflict_column} not found in data")

        if self.price_column not in data.columns:
            logger.error(f"Price column {self.price_column} not found in data")
            raise YemenAnalysisError(f"Price column {self.price_column} not found in data")

        # Set data
        self.data = data

        # Set spatial weight matrix
        if w is not None:
            self.w = w

        logger.info(f"Set data with {len(self.data)} observations")

    @handle_errors
    def create_conflict_weights(
        self, alpha: float = -1.0, normalize: bool = True
    ) -> weights.W:
        """
        Create conflict-adjusted spatial weights.

        Args:
            alpha: Distance decay parameter.
            normalize: Whether to row-normalize the weights.

        Returns:
            Conflict-adjusted spatial weight matrix.

        Raises:
            YemenAnalysisError: If the data has not been set.
        """
        logger.info(f"Creating conflict-adjusted weights with alpha={alpha}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        try:
            # Create spatial weight matrix
            weight_matrix = SpatialWeightMatrix(self.data)

            # Create conflict-adjusted weights
            self.w = weight_matrix.create_conflict_weights(
                conflict_column=self.conflict_column,
                alpha=alpha,
                normalize=normalize
            )

            logger.info(f"Created conflict-adjusted weights with {len(self.w.neighbors)} units")
            return self.w
        except Exception as e:
            logger.error(f"Error creating conflict-adjusted weights: {e}")
            raise YemenAnalysisError(f"Error creating conflict-adjusted weights: {e}")

    @handle_errors
    def analyze_price_dispersion(self) -> Dict[str, Any]:
        """
        Analyze the relationship between conflict and price dispersion.

        Returns:
            Dictionary containing the analysis results.

        Raises:
            YemenAnalysisError: If the data has not been set.
        """
        logger.info("Analyzing relationship between conflict and price dispersion")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        try:
            # Calculate price dispersion
            # Group by date and calculate coefficient of variation
            price_dispersion = self.data.groupby('date')[self.price_column].agg(['mean', 'std'])
            price_dispersion['cv'] = price_dispersion['std'] / price_dispersion['mean']

            # Calculate conflict intensity by date
            conflict_intensity = self.data.groupby('date')[self.conflict_column].mean()

            # Combine price dispersion and conflict intensity
            combined = pd.DataFrame({
                'price_dispersion': price_dispersion['cv'],
                'conflict_intensity': conflict_intensity
            })

            # Remove missing values
            combined = combined.dropna()

            # Estimate regression model
            X = sm.add_constant(combined['conflict_intensity'])
            model = OLS(combined['price_dispersion'], X)
            results = model.fit()

            # Store results
            self.results['price_dispersion'] = {
                'coefficients': {
                    'constant': results.params[0],
                    'conflict_intensity': results.params[1],
                },
                'std_errors': {
                    'constant': results.bse[0],
                    'conflict_intensity': results.bse[1],
                },
                'p_values': {
                    'constant': results.pvalues[0],
                    'conflict_intensity': results.pvalues[1],
                },
                't_values': {
                    'constant': results.tvalues[0],
                    'conflict_intensity': results.tvalues[1],
                },
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'f_statistic': results.fvalue,
                'f_p_value': results.f_pvalue,
                'n_obs': results.nobs,
                'data': combined,
            }

            logger.info(f"Analyzed relationship between conflict and price dispersion: R-squared={results.rsquared:.4f}")
            return self.results['price_dispersion']
        except Exception as e:
            logger.error(f"Error analyzing price dispersion: {e}")
            raise YemenAnalysisError(f"Error analyzing price dispersion: {e}")

    @handle_errors
    def analyze_market_integration(
        self, method: str = 'moran', time_periods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the relationship between conflict and market integration.

        Args:
            method: Method for measuring market integration. Options are 'moran'
                  (Moran's I), 'geary' (Geary's C), and 'correlation'.
            time_periods: List of time periods to analyze. If None, analyzes all time periods.

        Returns:
            Dictionary containing the analysis results.

        Raises:
            YemenAnalysisError: If the data has not been set or the method is invalid.
        """
        logger.info(f"Analyzing relationship between conflict and market integration with method={method}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if spatial weight matrix has been set
        if self.w is None:
            logger.error("Spatial weight matrix has not been set")
            raise YemenAnalysisError("Spatial weight matrix has not been set")

        try:
            # Get time periods
            if time_periods is None:
                time_periods = self.data['date'].unique()

            # Initialize results
            integration_results = {}

            # Analyze each time period
            for period in time_periods:
                # Get data for the time period
                period_data = self.data[self.data['date'] == period]

                # Skip if not enough data
                if len(period_data) < 3:
                    logger.warning(f"Not enough data for time period {period}")
                    continue

                # Calculate market integration
                if method == 'moran':
                    # Calculate Moran's I
                    from esda.moran import Moran

                    # Standardize prices
                    prices = period_data[self.price_column]
                    prices_std = (prices - prices.mean()) / prices.std()

                    # Calculate Moran's I
                    # Make sure the weights matrix matches the data
                    try:
                        moran = Moran(prices_std, self.w)
                    except ValueError:
                        # Create a new weights matrix for this subset of data
                        from src.models.spatial.weights import create_weights
                        subset_w = create_weights(period_data, weight_type='distance')
                        moran = Moran(prices_std, subset_w)

                    # Store results
                    integration_results[period] = {
                        'integration_measure': moran.I,
                        'p_value': moran.p_sim,
                        'z_value': moran.z_sim,
                        'method': 'Moran\'s I',
                    }
                elif method == 'geary':
                    # Calculate Geary's C
                    from esda.geary import Geary

                    # Standardize prices
                    prices = period_data[self.price_column]
                    prices_std = (prices - prices.mean()) / prices.std()

                    # Calculate Geary's C
                    # Make sure the weights matrix matches the data
                    try:
                        geary = Geary(prices_std, self.w)
                    except ValueError:
                        # Create a new weights matrix for this subset of data
                        from src.models.spatial.weights import create_weights
                        subset_w = create_weights(period_data, weight_type='distance')
                        geary = Geary(prices_std, subset_w)

                    # Store results
                    integration_results[period] = {
                        'integration_measure': 1 - geary.C,  # Convert to similarity measure
                        'p_value': geary.p_sim,
                        'z_value': geary.z_sim,
                        'method': 'Geary\'s C',
                    }
                elif method == 'correlation':
                    # Calculate correlation between prices and spatial lag
                    prices = period_data[self.price_column]

                    # Make sure the weights matrix matches the data
                    try:
                        spatial_lag = weights.lag_spatial(self.w, prices)
                    except ValueError:
                        # Create a new weights matrix for this subset of data
                        from src.models.spatial.weights import create_weights
                        subset_w = create_weights(period_data, weight_type='distance')
                        spatial_lag = weights.lag_spatial(subset_w, prices)

                    # Calculate correlation
                    correlation = np.corrcoef(prices, spatial_lag)[0, 1]

                    # Store results
                    integration_results[period] = {
                        'integration_measure': correlation,
                        'method': 'Correlation',
                    }
                else:
                    logger.error(f"Invalid method: {method}")
                    raise YemenAnalysisError(f"Invalid method: {method}")

                # Add conflict intensity
                integration_results[period]['conflict_intensity'] = period_data[self.conflict_column].mean()

            # Convert to DataFrame
            integration_df = pd.DataFrame.from_dict(integration_results, orient='index')

            # Estimate regression model
            X = sm.add_constant(integration_df['conflict_intensity'])
            model = OLS(integration_df['integration_measure'], X)
            results = model.fit()

            # Store results
            self.results['market_integration'] = {
                'coefficients': {
                    'constant': results.params.iloc[0],
                    'conflict_intensity': results.params.iloc[1],
                },
                'std_errors': {
                    'constant': results.bse.iloc[0],
                    'conflict_intensity': results.bse.iloc[1],
                },
                'p_values': {
                    'constant': results.pvalues.iloc[0],
                    'conflict_intensity': results.pvalues.iloc[1],
                },
                't_values': {
                    'constant': results.tvalues.iloc[0],
                    'conflict_intensity': results.tvalues.iloc[1],
                },
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'f_statistic': results.fvalue,
                'f_p_value': results.f_pvalue,
                'n_obs': results.nobs,
                'method': method,
                'data': integration_df,
            }

            logger.info(f"Analyzed relationship between conflict and market integration: R-squared={results.rsquared:.4f}")
            return self.results['market_integration']
        except Exception as e:
            logger.error(f"Error analyzing market integration: {e}")
            raise YemenAnalysisError(f"Error analyzing market integration: {e}")

    @handle_errors
    def analyze_price_transmission(
        self, reference_market: str, market_column: str = 'market',
        time_column: str = 'date'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the impact of conflict on price transmission between markets.

        Args:
            reference_market: Reference market for price transmission.
            market_column: Column containing market names.
            time_column: Column containing time periods.

        Returns:
            Dictionary mapping markets to price transmission results.

        Raises:
            YemenAnalysisError: If the data has not been set or the reference market is invalid.
        """
        logger.info(f"Analyzing impact of conflict on price transmission from {reference_market}")

        # Check if data has been set
        if self.data is None:
            logger.error("Data has not been set")
            raise YemenAnalysisError("Data has not been set")

        # Check if columns exist
        if market_column not in self.data.columns:
            logger.error(f"Market column {market_column} not found in data")
            raise YemenAnalysisError(f"Market column {market_column} not found in data")

        if time_column not in self.data.columns:
            logger.error(f"Time column {time_column} not found in data")
            raise YemenAnalysisError(f"Time column {time_column} not found in data")

        # Check if reference market exists
        if reference_market not in self.data[market_column].unique():
            logger.error(f"Reference market {reference_market} not found in data")
            raise YemenAnalysisError(f"Reference market {reference_market} not found in data")

        try:
            # Get reference market prices
            reference_prices = self.data[self.data[market_column] == reference_market]
            reference_prices = reference_prices.set_index(time_column)[self.price_column]

            # Get markets
            markets = self.data[market_column].unique()
            markets = [m for m in markets if m != reference_market]

            # Initialize results
            transmission_results = {}

            # Analyze each market
            for market in markets:
                # Get market prices
                market_prices = self.data[self.data[market_column] == market]
                market_prices = market_prices.set_index(time_column)

                # Get conflict intensity
                conflict_intensity = market_prices[self.conflict_column]

                # Get market prices
                market_prices = market_prices[self.price_column]

                # Align prices
                common_index = reference_prices.index.intersection(market_prices.index)
                ref_prices = reference_prices.loc[common_index]
                mkt_prices = market_prices.loc[common_index]
                conflict = conflict_intensity.loc[common_index]

                # Skip if not enough data
                if len(common_index) < 10:
                    logger.warning(f"Not enough data for market {market}")
                    continue

                # Create interaction term
                interaction = ref_prices * conflict

                # Estimate regression model
                X = sm.add_constant(pd.DataFrame({
                    'reference_price': ref_prices,
                    'conflict': conflict,
                    'interaction': interaction
                }))

                model = OLS(mkt_prices, X)
                results = model.fit()

                # Store results
                transmission_results[market] = {
                    'coefficients': {
                        'constant': results.params.iloc[0],
                        'reference_price': results.params.iloc[1],
                        'conflict': results.params.iloc[2],
                        'interaction': results.params.iloc[3],
                    },
                    'std_errors': {
                        'constant': results.bse.iloc[0],
                        'reference_price': results.bse.iloc[1],
                        'conflict': results.bse.iloc[2],
                        'interaction': results.bse.iloc[3],
                    },
                    'p_values': {
                        'constant': results.pvalues.iloc[0],
                        'reference_price': results.pvalues.iloc[1],
                        'conflict': results.pvalues.iloc[2],
                        'interaction': results.pvalues.iloc[3],
                    },
                    't_values': {
                        'constant': results.tvalues.iloc[0],
                        'reference_price': results.tvalues.iloc[1],
                        'conflict': results.tvalues.iloc[2],
                        'interaction': results.tvalues.iloc[3],
                    },
                    'r_squared': results.rsquared,
                    'adj_r_squared': results.rsquared_adj,
                    'f_statistic': results.fvalue,
                    'f_p_value': results.f_pvalue,
                    'n_obs': results.nobs,
                }

            # Store results
            self.results['price_transmission'] = transmission_results

            logger.info(f"Analyzed impact of conflict on price transmission for {len(transmission_results)} markets")
            return transmission_results
        except Exception as e:
            logger.error(f"Error analyzing price transmission: {e}")
            raise YemenAnalysisError(f"Error analyzing price transmission: {e}")

    @handle_errors
    def get_summary(self) -> str:
        """
        Get a summary of the analysis results.

        Returns:
            String containing the analysis summary.

        Raises:
            YemenAnalysisError: If no analyses have been performed.
        """
        logger.info("Getting analysis summary")

        # Check if any analyses have been performed
        if not self.results:
            logger.error("No analyses have been performed")
            raise YemenAnalysisError("No analyses have been performed")

        try:
            # Create summary
            summary = "Conflict Integration Analysis Summary\n"
            summary += "====================================\n\n"

            # Add price dispersion results
            if 'price_dispersion' in self.results:
                summary += "Price Dispersion Analysis\n"
                summary += "-----------------------\n"
                summary += f"R-squared: {self.results['price_dispersion']['r_squared']:.4f}\n"
                summary += f"Effect of conflict on price dispersion: {self.results['price_dispersion']['coefficients']['conflict_intensity']:.4f}"

                if self.results['price_dispersion']['p_values']['conflict_intensity'] < 0.05:
                    summary += " (significant at 5% level)\n"
                else:
                    summary += " (not significant at 5% level)\n"

                summary += "\n"

            # Add market integration results
            if 'market_integration' in self.results:
                summary += "Market Integration Analysis\n"
                summary += "--------------------------\n"
                summary += f"Method: {self.results['market_integration']['method']}\n"
                summary += f"R-squared: {self.results['market_integration']['r_squared']:.4f}\n"
                summary += f"Effect of conflict on market integration: {self.results['market_integration']['coefficients']['conflict_intensity']:.4f}"

                if self.results['market_integration']['p_values']['conflict_intensity'] < 0.05:
                    summary += " (significant at 5% level)\n"
                else:
                    summary += " (not significant at 5% level)\n"

                summary += "\n"

            # Add price transmission results
            if 'price_transmission' in self.results:
                summary += "Price Transmission Analysis\n"
                summary += "--------------------------\n"

                for market, results in self.results['price_transmission'].items():
                    summary += f"Market: {market}\n"
                    summary += f"R-squared: {results['r_squared']:.4f}\n"
                    summary += f"Price transmission coefficient: {results['coefficients']['reference_price']:.4f}"

                    if results['p_values']['reference_price'] < 0.05:
                        summary += " (significant at 5% level)\n"
                    else:
                        summary += " (not significant at 5% level)\n"

                    summary += f"Effect of conflict on price transmission: {results['coefficients']['interaction']:.4f}"

                    if results['p_values']['interaction'] < 0.05:
                        summary += " (significant at 5% level)\n"
                    else:
                        summary += " (not significant at 5% level)\n"

                    summary += "\n"

            logger.info("Generated analysis summary")
            return summary
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            raise YemenAnalysisError(f"Error getting analysis summary: {e}")
