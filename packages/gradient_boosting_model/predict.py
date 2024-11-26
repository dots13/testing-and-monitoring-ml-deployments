import logging
import typing as t

import pandas as pd

from gradient_boosting_model import __version__ as _version
from gradient_boosting_model.config.core import config
from gradient_boosting_model.processing.data_management import load_pipeline
from gradient_boosting_model.processing.validation import validate_inputs

_logger = logging.getLogger(__name__)


def load_model_pipeline() -> t.Optional[object]:
    """Load the model pipeline and handle errors."""
    try:
        pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
        pipeline = load_pipeline(file_name=pipeline_file_name)
        return pipeline
    except FileNotFoundError as e:
        _logger.error(f"Pipeline file not found: {e}")
    except Exception as e:
        _logger.error(f"Error loading model pipeline: {e}")
    return None


_price_pipe = load_model_pipeline()


def make_prediction(input_data: t.Union[pd.DataFrame, t.Dict],) -> dict:
    """
    Make a prediction using a saved model pipeline.

    Args:
        input_data (pd.DataFrame or dict): Data for which predictions are to be made.

    Returns:
        dict: A dictionary with predictions, version, and any errors.
    """
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)

    validated_data, errors = validate_inputs(input_data=input_data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _price_pipe.predict(
            X=validated_data[config.model_config.features]
        )
        _logger.info(
            f"Making predictions with model version: {_version} "
            f"Predictions: {predictions}"
        )
        results["predictions"] = predictions

    return results

