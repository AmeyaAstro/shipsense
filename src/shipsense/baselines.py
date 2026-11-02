"""Classical baselines. A deep model has to beat these to justify itself."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from shipsense.config import CATEGORICAL, NUMERIC, SEED


class ShippingModeRule:
    """Non-ML baseline: predict each order's late rate from its shipping mode alone.

    If a model can't beat this one-liner, it hasn't learned anything useful.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.rates_ = y.groupby(X["shipping_mode"]).mean()
        self.default_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame):
        p = X["shipping_mode"].map(self.rates_).fillna(self.default_).to_numpy()
        return pd.DataFrame({0: 1 - p, 1: p}).to_numpy()


def logistic_regression() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def gradient_boosting() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        categorical_features=list(range(len(CATEGORICAL))),
        early_stopping=True, random_state=SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])
