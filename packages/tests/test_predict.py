from pyexpat import model

from gradient_boosting_model.predict import make_prediction
from gradient_boosting_model.config.core import config

from sklearn.metrics import mean_squared_error

from gradient_boosting_model import predict as alt_predict

def test_prediction_quality_against_another_model(
        raw_training_data,
        sample_input_data
):
    # Given
    input_df = raw_training_data.drop(
        config.gradient_boosting_model_config.target,
        axis=1
    )
    output_df = raw_training_data[config.gradient_boosting_model_config.target]
    current_predictions = make_prediction(input_data=input_df)


    alternative_predictions = alt_predict.make_prediction(input_data=input_df)

    # When
    current_mse = mean_squared_error(
        y_true=output_df.values, y_pred=current_predictions["predictions"]
    )

    alternative_mse = mean_squared_error(
        y_true=output_df.values, y_pred=alternative_predictions["predictions"]
    )

    # Then
    assert current_mse <= alternative_mse


def test_prediction_quality_against_benchmark(raw_training_data, sample_input_data):
    # Given
    input_df = raw_training_data.drop(
        config.gradient_boosting_model_config.target,
        axis=1
    )
    output_df = raw_training_data[config.gradient_boosting_model_config.target]

    # Generate rough benchmarks (you would tweak depending on your model)
    benchmark_flexibility = 50000
    # setting ndigits to -4 will round the value to the nearest 10,000 i.e. 210,000
    benchmark_lower_boundary = (
        round(output_df.iloc[0], ndigits=-4) - benchmark_flexibility
    )  # 210,000 - 50000 = 160000
    benchmark_upper_boundary = (
        round(output_df.iloc[0], ndigits=-4) + benchmark_flexibility
    )  # 210000 + 50000 = 260000

    # When
    subject = make_prediction(input_data=input_df[0:1])

    # Then
    assert subject is not None
    prediction = subject.get("predictions")[0]
    assert isinstance(prediction, float)
    assert prediction > benchmark_lower_boundary
    assert prediction < benchmark_upper_boundary


def test_prediction_range(raw_training_data):
    # Given known range of target
    min_target_value = raw_training_data[config.gradient_boosting_model_config.target].min()
    max_target_value = raw_training_data[config.gradient_boosting_model_config.target].max()

    # When
    subject = make_prediction(
        input_data=raw_training_data.drop(
            config.gradient_boosting_model_config.target, axis=1)
    )
    predictions = subject.get("predictions")

    # Then
    for pred in predictions:
        assert min_target_value * 0.9 <= pred <= max_target_value * 1.1, (
            "Prediction is out of the expected range"
        )
