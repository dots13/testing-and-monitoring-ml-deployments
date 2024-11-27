import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SklearnTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for Scikit-learn pre-processing transformers, like SimpleImputer or OrdinalEncoder,
    to apply the transformer to a specified set of variables.

    Parameters:
    ----------
    variables : list or str
        List of variables to transform. If a single variable, pass it as a string.
    transformer : sklearn Transformer
        A scikit-learn transformer instance (e.g., SimpleImputer, OrdinalEncoder).
    """

    def __init__(self, variables=None, transformer=None):
        if not variables or not transformer:
            raise ValueError("Both 'variables' and 'transformer' must be provided.")
        self.variables = variables if isinstance(variables, list) else [variables]
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the transformer to the selected variables.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : pd.Series, optional
            The target variable, by default None.

        Returns:
        -------
        self
        """
        self._validate_dataframe(X)
        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the selected variables.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame.

        Returns:
        -------
        pd.DataFrame
            Transformed DataFrame.
        """
        self._validate_dataframe(X)
        X = X.copy()
        X[self.variables] = self.transformer.transform(X[self.variables])
        return X

    @staticmethod
    def _validate_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """
    Calculates the time difference between temporal variables and a reference variable.

    Parameters:
    ----------
    variables : list or str
        List of temporal variables for which to calculate the time difference.
    reference_variable : str
        The reference temporal variable.
    """

    def __init__(self, variables=None, reference_variable=None):
        if not variables or not reference_variable:
            raise ValueError("Both 'variables' and 'reference_variable' must be provided.")
        self.variables = variables if isinstance(variables, list) else [variables]
        self.reference_variable = reference_variable

    def fit(self, X: pd.DataFrame, y=None):
        """
        No fitting needed; returns self for pipeline compatibility.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : pd.Series, optional
            The target variable, by default None.

        Returns:
        -------
        self
        """
        self._validate_dataframe(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the time difference and updates the DataFrame.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame.

        Returns:
        -------
        pd.DataFrame
            Transformed DataFrame with time differences.
        """
        self._validate_dataframe(X)
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X

    @staticmethod
    def _validate_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")


class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):
    """
    Drops unnecessary features from a DataFrame.

    Parameters:
    ----------
    variables_to_drop : list or str
        List of variables to drop. If a single variable, pass it as a string.
    """

    def __init__(self, variables_to_drop=None):
        if not variables_to_drop:
            raise ValueError("'variables_to_drop' must be provided.")
        self.variables = (
            variables_to_drop if isinstance(variables_to_drop, list) else [variables_to_drop]
        )

    def fit(self, X: pd.DataFrame, y=None):
        """
        No fitting needed; returns self for pipeline compatibility.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : pd.Series, optional
            The target variable, by default None.

        Returns:
        -------
        self
        """
        self._validate_dataframe(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the specified variables from the DataFrame.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame.

        Returns:
        -------
        pd.DataFrame
            DataFrame with specified variables dropped.
        """
        self._validate_dataframe(X)
        X = X.copy()
        return X.drop(columns=self.variables, errors="ignore")

    @staticmethod
    def _validate_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

