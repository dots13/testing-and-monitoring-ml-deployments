import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers,
    like the SimpleImputer() or OrdinalEncoder(), to allow
    the use of the transformer on a selected group of variables.
    """

    def __init__(self, variables=None, transformer=None):
        if variables is None or transformer is None:
            raise ValueError("Both 'variables' and 'transformer' must be provided.")
        self.variables = (
            variables if isinstance(variables, list) else [variables]
        )  # Ensure variables is always a list
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        The `fit` method allows scikit-learn transformers to
        learn the required parameters from the training dataset.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformations to the DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        X = X.copy()
        X[self.variables] = self.transformer.transform(X[self.variables])
        return X


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """
    Calculates the time difference between 2 temporal variables.
    """

    def __init__(self, variables=None, reference_variable=None):
        if variables is None or reference_variable is None:
            raise ValueError("Both 'variables' and 'reference_variable' must be provided.")
        self.variables = (
            variables if isinstance(variables, list) else [variables]
        )  # Ensure variables is always a list
        self.reference_variable = reference_variable

    def fit(self, X: pd.DataFrame, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time differences and return a modified DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X


class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):
    """
    Drops unnecessary or unused features from the dataset.
    """

    def __init__(self, variables_to_drop=None):
        if variables_to_drop is None:
            raise ValueError("'variables_to_drop' must be provided.")
        self.variables = (
            variables_to_drop if isinstance(variables_to_drop, list) else [variables_to_drop]
        )  # Ensure variables_to_drop is always a list

    def fit(self, X: pd.DataFrame, y=None):
        """
        The `fit` method is necessary to accommodate the
        scikit-learn pipeline functionality.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unnecessary or unused features from the dataset.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        X = X.copy()
        X.drop(self.variables, axis=1)
        return X
