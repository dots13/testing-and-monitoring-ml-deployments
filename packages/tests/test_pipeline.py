from gradient_boosting_model import pipeline
from gradient_boosting_model.config.core import config
import numpy as np


def test_pipeline_drops_unnecessary_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    # Check that the feature to be dropped is in the original dataframe
    drop_feature = config.gradient_boosting_model_config.drop_features
    assert drop_feature in X_train.columns, (
        f"'{drop_feature}' should be in the original X_train columns."
    )

    # Fit the pipeline
    pipeline.price_pipe.fit(X_train, y_train)

    # When
    # Transform the training data and capture the result
    transformed_inputs = pipeline.price_pipe[:-1].transform(X_train)

    # Then
    # Check that the feature to be dropped is no longer in the transformed dataframe
    assert drop_feature not in transformed_inputs.columns, (
        f"'{drop_feature}' should have been dropped from transformed data."
    )

    # Verify that no other rows were dropped
    assert X_train.shape[0] == transformed_inputs.shape[0], (
        "Number of rows should remain the same after dropping features."
    )


def test_pipeline_transforms_temporal_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    # Ensure the temporal variable exists in the input data
    temporal_var = config.gradient_boosting_model_config.temporal_vars
    reference_var = config.gradient_boosting_model_config.drop_features
    assert temporal_var in X_train.columns, (
        f"'{temporal_var}' should be in X_train columns."
    )
    assert reference_var in X_train.columns, (
        f"'{reference_var}' should be in X_train columns."
    )

    # When
    # Transform the training data and capture the result
    transformed_inputs = pipeline.price_pipe[:-1].transform(X_train)

    # Then
    # Verify the transformation of the temporal feature
    transformed_temporal_value = transformed_inputs.iloc[0][temporal_var]
    expected_value = X_train.iloc[0][reference_var] - X_train.iloc[0][temporal_var]
    assert transformed_temporal_value == expected_value, (
        f"Temporal variable '{temporal_var}' should be transformed as the difference "
        f"between '{reference_var}' and '{temporal_var}'."
    )


def test_imputation(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    pipeline.price_pipe.fit(X_train, y_train)

    # When
    transformed_X_train = pipeline.price_pipe[:-1].transform(X_train)

    # Then
    for col in config.gradient_boosting_model_config.numerical_vars:
        assert transformed_X_train[col].isnull().sum() == 0, (
            f"Numerical {col} has missing values after imputation"
        )
    for col in config.gradient_boosting_model_config.categorical_vars:
        assert (transformed_X_train[col] == "missing").sum() == 0, (
            f"Categorical {col} has missing values after imputation"
        )


def test_encoding(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    pipeline.price_pipe.fit(X_train, y_train)

    # When
    transformed_X_train = pipeline.price_pipe[:-1].transform(X_train)

    # Then
    for col in config.gradient_boosting_model_config.categorical_vars:
        # Check that categorical variables are properly encoded
        assert transformed_X_train[col].isnull().sum() == 0, (
            f"{col} contains missing values before encoding."
        )

        # Ensure that categorical variables are encoded as integers,
        # even though sklearn OrdinalEncoder returns float64
        transformed_X_train[col] = transformed_X_train[col].astype('int64')
        assert transformed_X_train[col].dtype == 'int64', (
            f"{col} should be encoded as integers."
        )


def test_pipeline_model_fit_and_predict(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    # When
    pipeline.price_pipe.fit(X_train, y_train)
    predictions = pipeline.price_pipe.predict(X_test)

    # Then
    assert len(predictions) == len(y_test), (
        "The number of predictions must match the number of test samples"
    )
    assert all(~np.isnan(predictions)), (
        "Predictions should not contain NaN values"
    )
