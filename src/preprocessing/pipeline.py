from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=0.01, upper_percentile=0.99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.limits_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.limits_[col] = (
                X[col].quantile(self.lower_percentile),
                X[col].quantile(self.upper_percentile)
            )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            if col in self.limits_:
                lower, upper = self.limits_[col]
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X

def create_preprocessor(config: dict) -> ColumnTransformer:
    """
    Builds a ColumnTransformer based on config.

    Config example:
    preprocessing:
      numerical:
        features: ["Price", "Quantity"]
        steps:
          - name: imputer
            type: simple_imputer
            args: {strategy: "median"}
          - name: scaler
            type: standard_scaler
      categorical:
        features: ["Category", "Location"]
        steps:
          - name: imputer
            type: simple_imputer
            args: {strategy: "most_frequent"}
          - name: encoder
            type: one_hot
            args: {handle_unknown: "ignore"}
    """
    transformers = []

    # Numerical Pipeline
    num_config = config.get("numerical", {})
    if num_config and num_config.get("features"):
        steps = []
        for step in num_config.get("steps", []):
            if step["type"] == "simple_imputer":
                steps.append((step["name"], SimpleImputer(**step.get("args", {}))))
            elif step["type"] == "standard_scaler":
                steps.append((step["name"], StandardScaler(**step.get("args", {}))))
            elif step["type"] == "min_max_scaler":
                steps.append((step["name"], MinMaxScaler(**step.get("args", {}))))
            elif step["type"] == "outlier_clipper":
                steps.append((step["name"], OutlierClipper(**step.get("args", {}))))

        if steps:
            transformers.append(("num", Pipeline(steps), num_config["features"]))

    # Categorical Pipeline
    cat_config = config.get("categorical", {})
    if cat_config and cat_config.get("features"):
        steps = []
        for step in cat_config.get("steps", []):
            if step["type"] == "simple_imputer":
                steps.append((step["name"], SimpleImputer(**step.get("args", {}))))
            elif step["type"] == "one_hot":
                steps.append((step["name"], OneHotEncoder(**step.get("args", {}))))
            elif step["type"] == "ordinal":
                steps.append((step["name"], OrdinalEncoder(**step.get("args", {}))))

        if steps:
            transformers.append(("cat", Pipeline(steps), cat_config["features"]))

    return ColumnTransformer(transformers=transformers, remainder="drop")

def build_full_pipeline(preprocessor: ColumnTransformer, model=None):
    steps = [("preprocessor", preprocessor)]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)
