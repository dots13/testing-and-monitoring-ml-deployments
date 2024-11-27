from gradient_boosting_model.config.core import config
from gradient_boosting_model.processing import preprocessors as pp
from sklearn.impute import SimpleImputer


def test_sklearn_transformer_wrapper_with_config_numerical_vars(pipeline_inputs):
    # Given
    X_train, _, _, _ = pipeline_inputs
    numerical_vars = config.gradient_boosting_model_config.numerical_vars
    assert all(var in X_train.columns for var in numerical_vars)  # Ensure variables exist in DataFrame

    transformer = pp.SklearnTransformerWrapper(
        variables=numerical_vars,
        transformer=SimpleImputer(strategy="most_frequent")
    )

    # When
    X_train_subset = X_train[config.gradient_boosting_model_config.numerical_vars]
    transformer.fit(X_train_subset)

    X_transformed = transformer.transform(X_train_subset)

    # Then
    for var in numerical_vars:
        assert not X_transformed[var].isna().any()  # Ensure NaNs are imputed for all numerical variables


def test_drop_unnecessary_features_transformer(pipeline_inputs):
    # Given
    X_train, _, _, _ = pipeline_inputs

    drop_features = config.gradient_boosting_model_config.drop_features

    if isinstance(drop_features, str):
        drop_features = [drop_features]

    assert isinstance(drop_features, list), f"drop_features should be a list, got {type(drop_features)}"
    assert all(feature in X_train.columns for feature in drop_features), \
        f"Not all features in {drop_features} are present in X_train columns"

    # Create the transformer
    transformer = pp.DropUnnecessaryFeatures(
        variables_to_drop=drop_features,
    )

    # When
    X_transformed = transformer.transform(X_train)

    # Then
    # Assert that each feature in drop_features is no longer in the transformed dataset
    for feature in drop_features:
        assert feature not in X_transformed.columns, f"Feature '{feature}' was not dropped"

    # Additionally, ensure no other features were unintentionally dropped or modified
    for column in X_train.columns:
        if column not in drop_features:
            assert column in X_transformed.columns, f"Feature '{column}' was unintentionally dropped or modified"

    # Check if the transformed DataFrame still contains the same number of rows
    assert X_train.shape[0] == X_transformed.shape[0], "Number of rows has changed after dropping features"


def test_temporal_variable_estimator(pipeline_inputs):
    # Given
    X_train, _, _, _ = pipeline_inputs
    # Ensure temporal variables are present in X_train
    temporal_vars = config.gradient_boosting_model_config.temporal_vars
    reference_variable = config.gradient_boosting_model_config.drop_features

    if isinstance(temporal_vars, str):
        temporal_vars = [temporal_vars]
    # Check that temporal_vars are in the DataFrame
    assert all(var in X_train.columns for var in temporal_vars), f"Missing temporal variables: {temporal_vars}"

    # Initialize the transformer
    transformer = pp.TemporalVariableEstimator(
        variables=temporal_vars,
        reference_variable=reference_variable,
    )

    # When
    X_transformed = transformer.transform(X_train)

    # Then
    # Check that the transformation is correct for each row
    for i in range(X_train.shape[0]):
        for var in temporal_vars:
            expected_difference = X_train.iloc[i][reference_variable] - X_train.iloc[i][var]
            assert X_transformed.iloc[i][var] == expected_difference, (
                f"Row {i} - {var} transformation mismatch: "
                f"Expected {expected_difference}, got {X_transformed.iloc[i][var]}"
            )

    # Ensure that no rows are missing in the transformed DataFrame
    assert X_train.shape[0] == X_transformed.shape[0], "Number of rows has changed after transformation"

