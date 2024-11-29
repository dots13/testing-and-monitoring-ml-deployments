import logging
import typing as t
import pandas as pd
from sklearn.pipeline import Pipeline

from gradient_boosting_model import __version__ as _version
from gradient_boosting_model.config.core import config
from gradient_boosting_model.processing.data_management import load_pipeline
from gradient_boosting_model.processing.validation import validate_inputs

_logger = logging.getLogger(__name__)

# Explicitly define the type to show _price_pipe can be None
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe: t.Optional[Pipeline] = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if _price_pipe is not None and not errors:
        predictions = _price_pipe.predict(
            X=validated_data[config.gradient_boosting_model_config.features]
        )
        _logger.info(
            f"Making predictions with model version: {_version} "
            f"Predictions: {predictions}"
        )
        results = {"predictions": predictions, "version": _version, "errors": errors}
    else:
        _logger.error(
            "Model pipeline is None or errors in validation. "
            "Predictions cannot be made."
        )

    return results
