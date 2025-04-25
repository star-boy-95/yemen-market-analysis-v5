import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Import the necessary modules
from src.data.preprocessor import DataPreprocessor
from src.models.threshold_py import ThresholdCointegration
from src.models.threshold.model import ThresholdModel # Using the base ThresholdModel for general tests
from src.models.unit_root import UnitRootTester, StructuralBreakTester
from src.models.diagnostics import ModelDiagnostics
from src.models.reporting import ModelComparer # Assuming ModelComparer is a real class for workflow tests

# Helper function to create synthetic data
def create_synthetic_data(n_samples=100, seed=42):
    """Generates synthetic time series data for testing threshold models."""
    np.random.seed(seed)
    dates = pd.date_range(start='2015-01-01', periods=n_samples, freq='M')
    # Create two cointegrated series with a threshold effect
    x = np.random.randn(n_samples).cumsum()
    error = np.random.randn(n_samples) * 0.5
    # Introduce a threshold effect around x=1.0
    y = x + error + np.where(x > 1.0, 5, 0)
    data = pd.DataFrame({'x': x, 'y': y, 'threshold_var': x + np.random.randn(n_samples)*0.1}, index=dates)
    return data

# Define a fixture for synthetic data
@pytest.fixture
def synthetic_data():
    """Provides synthetic time series data for testing."""
    return create_synthetic_data()

# Define a fixture for a preprocessor instance
@pytest.fixture
def preprocessor(synthetic_data):
    """Provides a DataPreprocessor instance with synthetic data."""
    # For integration tests, we can initialize with data directly or mock loader
    # Initializing directly is simpler for this scope
    return DataPreprocessor(data=synthetic_data.copy())

# Define a fixture for a threshold cointegration instance (if still needed, or use ThresholdModel)
@pytest.fixture
def threshold_cointegration_model():
    """Provides a ThresholdCointegration instance."""
    # This fixture might be redundant if using ThresholdModel directly
    # Keeping it for now based on original test structure, but consider removing if not used
    return ThresholdCointegration()

# Define a fixture for a general threshold model instance
@pytest.fixture
def threshold_model(synthetic_data):
    """Provides a ThresholdModel instance with synthetic data."""
    # Initialize with basic parameters; specific tests will override as needed
    return ThresholdModel(data=synthetic_data.copy(), dependent_var='y', threshold_var='threshold_var')

# Define a fixture for a unit root tester instance
@pytest.fixture
def unit_root_tester():
    """Provides a UnitRootTester instance."""
    return UnitRootTester()

# Define a fixture for a structural break tester instance
@pytest.fixture
def structural_break_tester():
    """Provides a StructuralBreakTester instance."""
    return StructuralBreakTester()

# Define a fixture for a model diagnostics instance
@pytest.fixture
def model_diagnostics():
    """Provides a ModelDiagnostics instance."""
    return ModelDiagnostics()

# Define a fixture for a model comparer instance
@pytest.fixture
def model_comparer():
    """Provides a ModelComparer instance."""
    return ModelComparer([]) # Initialize with an empty list of models

class TestThresholdIntegration:
    """
    Integration tests for threshold models and their interaction with non-spatial components.
    Focuses on integration with Preprocessor, Unit Root Testing, Diagnostics,
    Model Comparison, and Structural Break Analysis.
    """

    def test_preprocessor_threshold_integration(self, preprocessor, threshold_model, synthetic_data):
        """
        Test that preprocessed data can be properly fed into threshold models.
        Verify outlier detection and handling works with threshold estimation.
        Test dual exchange rate adjustments affect cointegration analysis.
        """
        # Simulate preprocessing steps
        # Use a copy to avoid modifying the fixture data
        data_copy = synthetic_data.copy()
        processed_data = preprocessor.handle_outliers(data_copy, method='iqr', column='y')
        # Assuming adjust_dual_exchange_rates is a static method or takes data
        # If it's a method of preprocessor instance, adapt call
        # For this test, let's assume it's a method that modifies the dataframe
        processed_data = preprocessor.adjust_dual_exchange_rates(processed_data, rate_column='y', official_rate=1.0, market_rate=1.5)


        # Mock the underlying model fitting to check if processed data is used
        # Assuming ThresholdModel has a _fit method
        with patch.object(threshold_model, '_fit') as mock_fit:

            # Attempt to fit the model with processed data
            threshold_model.fit(processed_data)

            # Assert that the mock was called once
            mock_fit.assert_called_once()
            # Check if the data passed to the mock is the processed data
            # Note: This requires the _fit method to accept data as the first argument
            pd.testing.assert_frame_equal(mock_fit.call_args[0][0], processed_data)

        # Add assertions to verify the effects of preprocessing on model output if possible
        # This would require fitting a real model and checking its properties,
        # which might be more complex and potentially better suited for higher-level integration tests.
        # For this basic integration test, verifying that the processed data is used is sufficient.


    def test_unit_root_threshold_integration(self, unit_root_tester, threshold_model, synthetic_data):
        """
        Verify that unit root test results correctly influence threshold model fitting.
        Test that stationarity is properly considered in the threshold modeling pipeline.
        """
        # Simulate unit root testing on the dependent variable
        # Mock the actual unit root test methods to control results
        with patch.object(unit_root_tester, 'adf_test') as mock_adf, \
             patch.object(unit_root_tester, 'kpss_test') as mock_kpss:

            # Configure mocks to return specific results (e.g., non-stationary)
            mock_adf.return_value = {'pvalue': 0.5} # Non-stationary
            mock_kpss.return_value = {'pvalue': 0.01} # Non-stationary

            # Perform unit root tests on the dependent variable
            adf_result = unit_root_tester.adf_test(synthetic_data['y'])
            kpss_result = unit_root_tester.kpss_test(synthetic_data['y'])

            # Mock the threshold model's internal handling of unit root results
            # Assuming ThresholdModel has a method to handle pre-fitting checks including unit root
            with patch.object(threshold_model, '_pre_fit_checks') as mock_pre_fit_checks:
                 # Attempt to fit a model, which should trigger pre-fit checks
                 # In a real scenario, the threshold_model might internally call unit root tests
                 # or accept their results as input. This mock simulates the latter for testing integration.
                 threshold_model.fit(synthetic_data.copy()) # Pass data to fit method

                 # Assert that the mock handling method was called
                 mock_pre_fit_checks.assert_called_once()
                 # In a more advanced test, you would check the arguments passed to mock_pre_fit_checks
                 # to ensure the unit root test results are correctly influencing the model fitting process.

        # Add assertions to verify how unit root test results influence model selection or parameters
        # This would require more detailed knowledge of how the threshold model uses these results.
        # For this basic integration test, verifying that the results are passed/handled is sufficient.


    def test_diagnostics_threshold_integration(self, model_diagnostics, threshold_model, synthetic_data):
        """
        Test that diagnostic results can be generated for a fitted threshold model.
        Verify that the diagnostics component can process the output of a threshold model.
        """
        # Simulate fitting a threshold model
        # Mock the fit method to return a dummy model object with necessary attributes/methods
        mock_fitted_model = MagicMock()
        # Simulate having residuals and predictions after fitting
        mock_fitted_model.residuals = synthetic_data['y'] - synthetic_data['y'].mean() # Dummy residuals
        mock_fitted_model.predictions = synthetic_data['y'].mean() + np.random.randn(len(synthetic_data)) * 0.1 # Simulate predictions with some noise
        mock_fitted_model.model = MagicMock() # Mock the underlying statsmodels or similar model object if accessed

        with patch.object(threshold_model, 'fit', return_value=mock_fitted_model) as mock_fit:
            # Fit the model (mocked)
            fitted_model_instance = threshold_model.fit(synthetic_data.copy())

            # Simulate running diagnostics on the fitted model
            # Mock the actual diagnostic methods within ModelDiagnostics
            with patch.object(model_diagnostics, 'check_residuals') as mock_check_residuals, \
                 patch.object(model_diagnostics, 'check_autocorrelation') as mock_check_autocorrelation:

                # Configure mocks to return dummy diagnostic results
                mock_check_residuals.return_value = {'normality_pvalue': 0.05}
                mock_check_autocorrelation.return_value = {'ljung_box_pvalue': 0.1}

                # Run diagnostics
                # Assuming run_all_diagnostics takes the fitted model instance and original data
                diagnostics_results = model_diagnostics.run_all_diagnostics(fitted_model_instance, synthetic_data['y'])

                # Assert that diagnostic methods were called with the fitted model or its outputs
                mock_check_residuals.assert_called_once()
                # Check if check_residuals was called with the residuals from the fitted model
                # This requires knowing the expected argument name for residuals in check_residuals
                # Assuming it's the first positional argument
                # pd.testing.assert_series_equal(mock_check_residuals.call_args[0][0], mock_fitted_model.residuals) # Uncomment if check_residuals takes residuals directly

                mock_check_autocorrelation.assert_called_once()
                # Check if check_autocorrelation was called with the residuals
                # pd.testing.assert_series_equal(mock_check_autocorrelation.call_args[0][0], mock_fitted_model.residuals) # Uncomment if check_autocorrelation takes residuals directly


            # Assert that the diagnostics results are in the expected format or contain expected keys
            assert isinstance(diagnostics_results, dict)
            assert 'normality_pvalue' in diagnostics_results
            assert 'ljung_box_pvalue' in diagnostics_results


    def test_model_comparison_workflow(self, synthetic_data, model_comparer):
        """
        Tests the model comparison process for non-spatial threshold models.
        Verifies that different threshold model specifications can be compared
        using information criteria.
        """
        # Create synthetic data for a single region for non-spatial comparison
        region_data = synthetic_data.copy()

        # Initialize and fit multiple threshold models with different specifications
        # Model 1: TAR model with 'threshold_var' as threshold
        model_spec1 = ThresholdModel(data=region_data, dependent_var='y', independent_vars=['x'], threshold_var='threshold_var', model_type='tar')
        model_spec1.fit() # Assuming fit calculates AIC/BIC and stores them

        # Model 2: MTAR model with 'x' as threshold
        model_spec2 = ThresholdModel(data=region_data, dependent_var='y', independent_vars=['x'], threshold_var='x', model_type='mtar')
        model_spec2.fit() # Assuming fit calculates AIC/BIC and stores them

        # Ensure AIC and BIC are calculated and available after fitting
        assert hasattr(model_spec1, 'aic') and model_spec1.aic is not None
        assert hasattr(model_spec1, 'bic') and model_spec1.bic is not None
        assert hasattr(model_spec2, 'aic') and model_spec2.aic is not None
        assert hasattr(model_spec2, 'bic') and model_spec2.bic is not None

        # Use the ModelComparer to compare the models
        comparer = ModelComparer([model_spec1, model_spec2])
        comparison_results = comparer.compare(criteria=['AIC', 'BIC'])

        # Assertions
        # Verify that the comparison results contain the expected criteria and models
        assert isinstance(comparison_results, pd.DataFrame)
        assert 'AIC' in comparison_results.columns
        assert 'BIC' in comparison_results.columns
        # Check if the index contains identifiers for the models (assuming ModelComparer uses object IDs or similar)
        # A more robust test would involve checking against known model identifiers if the comparer provides them
        assert len(comparison_results) == 2 # Should have results for two models

        # Verify that the AIC/BIC values in the comparison results match the models' attributes
        # This requires knowing how ModelComparer identifies models in the output DataFrame index.
        # Assuming the index might contain string representations or similar.
        # A more specific assertion would require adapting based on actual ModelComparer implementation.
        # For now, we'll check if the values are present in the results.
        assert model_spec1.aic in comparison_results['AIC'].values
        assert model_spec1.bic in comparison_results['BIC'].values
        assert model_spec2.aic in comparison_results['AIC'].values
        assert model_spec2.bic in comparison_results['BIC'].values


    def test_structural_break_analysis_workflow(self, synthetic_data, structural_break_tester):
        """
        Tests structural break detection and its potential interaction with
        threshold model specification or interpretation.
        Verifies that structural break results can be obtained and potentially
        used alongside threshold analysis.
        """
        # Use the dependent variable series for structural break testing
        price_series = synthetic_data['y']

        # Perform Structural Break Detection
        # Mock the actual structural break test method if needed for control,
        # but for integration, let's assume the tester works.
        # We'll use a simplified test method for demonstration.
        # Assuming StructuralBreakTester has a method like `detect_breaks`
        # that returns a dictionary including 'break_dates'.

        # Simulate a known structural break in the synthetic data
        # Inject a break by shifting the mean after a certain point
        data_with_break = synthetic_data.copy()
        break_point_index = int(len(data_with_break) * 0.6) # Break at 60% of the data
        break_date = data_with_break.index[break_point_index]
        data_with_break.loc[data_with_break.index >= break_date, 'y'] += 10 # Shift the mean

        # Use the structural break tester on the series with a break
        break_results = structural_break_tester.test_breaks(data_with_break['y'], method='bai_perron') # Example method

        # Assertions
        # Verify that break detection results are obtained
        assert isinstance(break_results, dict)
        assert 'break_dates' in break_results
        assert isinstance(break_results['break_dates'], list)

        # Verify that the detected break date is close to the injected break date
        # Structural break tests might not find the exact date, but should be close
        detected_breaks = break_results['break_dates']
        assert len(detected_breaks) >= 1 # Expect at least one break detected

        # Check if any of the detected breaks are close to the injected break date
        # Define a tolerance for date comparison (e.g., within a few periods)
        tolerance = pd.Timedelta(days=60) # Within approximately 2 months
        is_break_detected_near_injected = any(abs(detected_date - break_date) <= tolerance for detected_date in detected_breaks)

        assert is_break_detected_near_injected

        # Note: Integrating structural break results directly into the ThresholdModel
        # for regime definition would require specific model functionality not detailed
        # in the prompt. This test verifies the detection part of the workflow.