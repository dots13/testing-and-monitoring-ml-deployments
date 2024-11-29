from gradient_boosting_model.processing.validation import validate_inputs


def test_validate_inputs_valid_data(sample_input_data):
    """
    Ensure valid inputs pass through validation without errors
    and return the expected number of rows.
    """
    # When
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)

    # Then
    assert not errors
    assert len(sample_input_data) == 1459
    assert len(validated_inputs) == 1457  # 2 rows dropped


def test_validate_inputs_identifies_single_error(sample_input_data):
    """
    Ensure a single invalid value triggers validation errors
    with the correct error message.
    """
    # Given
    test_inputs = sample_input_data.copy()

    # Introduce an error
    test_inputs.at[1, "BldgType"] = 50  # Expecting a string

    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors
    assert len(errors) == 1
    assert errors[1] == {"BldgType": ["Not a valid string."]}
