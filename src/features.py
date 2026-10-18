"""Feature selection as an sklearn transformer so it lives inside the Pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import FEATURES


class SelectFeatures(BaseEstimator, TransformerMixin):
    """Pick the model's input columns from a raw observation DataFrame.

    Stateless (nothing is learned in fit), so it cannot leak information. Having
    it as the first Pipeline step means the saved model accepts the raw CSV
    columns, including extras such as grid_x / grid_y, and ignores the rest.
    """

    def __init__(self, features=None):
        self.features = features

    def _cols(self):
        return list(self.features) if self.features is not None else list(FEATURES)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self._cols()].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self._cols(), dtype=object)


def drop_correlated(df: pd.DataFrame, features, threshold: float = 0.9):
    """Greedy de-duplication: walk features in order and drop any whose |Pearson r|
    with an already-kept feature exceeds `threshold`. Call on training data only."""
    corr = df[features].corr().abs()
    kept, dropped = [], {}
    for f in features:
        clash = [k for k in kept if corr.loc[f, k] > threshold]
        if clash:
            dropped[f] = (clash[0], round(float(corr.loc[f, clash[0]]), 3))
        else:
            kept.append(f)
    return kept, dropped
