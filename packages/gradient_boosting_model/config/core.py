from pathlib import Path
import typing as t

from pydantic import BaseModel, field_validator, ValidationInfo
from strictyaml import load, YAML

import gradient_boosting_model

# Project Directories
PACKAGE_ROOT = Path(gradient_boosting_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    training_data_file: str
    test_data_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    drop_features: str
    target: str
    variables_to_rename: t.Dict[str, str]
    features: t.Union[str, t.List[str]]
    numerical_vars: t.List[str]
    categorical_vars: t.Union[str, t.List[str]]
    temporal_vars: t.Union[str, t.List[str]]
    numerical_vars_with_na: t.List[str]
    numerical_na_not_allowed: t.List[str]
    test_size: float
    random_state: int
    n_estimators: int
    rare_label_n_categories: int
    rare_label_tol: float

    allowed_loss_functions: t.Tuple[str, ...]
    loss: str

    @field_validator("loss")
    def allowed_loss_function(cls, value: str, values: ValidationInfo) -> str:
        """
        Loss function to be optimized.

        `squared_error` refers to least squares regression.
        `absolute_error` (least absolute deviation)
        `huber` is a combination of the two.
        `quantile` allows quantile regression.

        Following the research phase, loss is restricted to
        `ls` and `huber` for this model.
        """

        # Accessing the allowed_loss_functions field directly
        allowed_loss_functions = values.data.get('allowed_loss_functions', [])
        if value in allowed_loss_functions:
            return value
        raise ValueError(
            f"the loss parameter specified: {value}, "
            f"is not in the allowed set: {allowed_loss_functions}"
        )


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    gradient_boosting_model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: t.Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()

    try:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found at: {cfg_path}") from e
    except Exception as e:
        raise ValueError(f"Error parsing YAML from {cfg_path}: {e}") from e


def create_and_validate_config(parsed_config: t.Optional[YAML] = None) -> Config:
    """
    Create and validate the configuration by parsing the provided YAML file.

    If no parsed configuration is provided, it loads the configuration from
    the default YAML file and validates it. The configuration is returned as
    a `Config` object with both application and model configurations.

    Args:
        parsed_config (Optional[YAML]): A parsed YAML configuration,
                                         or `None` to load the default configuration.

    Returns:
        Config: The validated configuration object.
    """
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        gradient_boosting_model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
