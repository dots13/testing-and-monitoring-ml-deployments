from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from feature_engine.encoding import RareLabelEncoder

from gradient_boosting_model.processing import preprocessors as pp
from gradient_boosting_model.config.core import config

import logging


_logger = logging.getLogger(__name__)


price_pipe = Pipeline(
    [
        (
            "numerical_imputer",
            pp.SklearnTransformerWrapper(
                variables=config.gradient_boosting_model_config.numerical_vars,
                transformer=SimpleImputer(strategy="most_frequent"),
            ),
        ),
        (
            "categorical_imputer",
            pp.SklearnTransformerWrapper(
                variables=config.gradient_boosting_model_config.categorical_vars,
                transformer=SimpleImputer(strategy="constant", fill_value="missing"),
            ),
        ),
        (
            "temporal_variable",
            pp.TemporalVariableEstimator(
                variables=config.gradient_boosting_model_config.temporal_vars,
                reference_variable=config.gradient_boosting_model_config.drop_features,
            ),
        ),
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=config.gradient_boosting_model_config.rare_label_tol,
                n_categories=config.gradient_boosting_model_config.rare_label_n_categories,
                variables=config.gradient_boosting_model_config.categorical_vars,
            ),
        ),
        (
            "categorical_encoder",
            pp.SklearnTransformerWrapper(
                variables=config.gradient_boosting_model_config.categorical_vars,
                transformer=OrdinalEncoder(),
            ),
        ),
        (
            "drop_features",
            pp.DropUnnecessaryFeatures(
                variables_to_drop=config.gradient_boosting_model_config.drop_features,
            ),
        ),
        (
            "gb_model",
            GradientBoostingRegressor(
                loss=config.gradient_boosting_model_config.loss,
                random_state=config.gradient_boosting_model_config.random_state,
                n_estimators=config.gradient_boosting_model_config.n_estimators,
            ),
        ),
    ]
)

