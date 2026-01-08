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

    def get_feature_names_out(self, input_features=None):
        return input_features

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
            elif step["type"] == "robust_scaler":
                steps.append((step["name"], RobustScaler(**step.get("args", {}))))
            elif step["type"] == "outlier_clipper":
                steps.append((step["name"], OutlierClipper(**step.get("args", {}))))

        if steps:
            transformers.append(("num", Pipeline(steps), num_config["features"]))
        else:
             # If no steps defined but features listed, pass them through
             transformers.append(("num", "passthrough", num_config["features"]))

    # Categorical Pipeline
    cat_config = config.get("categorical", {})
    if cat_config and cat_config.get("features"):
        steps = []
        for step in cat_config.get("steps", []):
            if step["type"] == "simple_imputer":
                steps.append((step["name"], SimpleImputer(**step.get("args", {}))))
            elif step["type"] == "one_hot":
                # Ensure sparse_output is False if we want pandas output
                args = step.get("args", {}).copy()
                if "sparse" in args: # Old sklearn
                    args["sparse"] = False
                if "sparse_output" not in args: # New sklearn
                    args["sparse_output"] = False
                steps.append((step["name"], OneHotEncoder(**args)))
            elif step["type"] == "ordinal":
                steps.append((step["name"], OrdinalEncoder(**step.get("args", {}))))

        if steps:
            transformers.append(("cat", Pipeline(steps), cat_config["features"]))
        else:
            # If no steps defined but features listed, pass them through (warning: strings might break some models)
            # XGBoost handles categories if enable_categorical=True or if they are encoded.
            # However, ColumnTransformer converts output to object array if mixed types.
            # XGBoost expects pandas dataframe or categorical type for enable_categorical=True,
            # NOT numpy object array of strings.
            # We must use OrdinalEncoder as a fallback for "raw" strings if we want to support models that don't support strings.
            # OR we ensure output is pandas DF? ColumnTransformer can return pandas if set_output is used?
            # But set_output is newer sklearn.

            # Better fallback: Use OrdinalEncoder with 'unknown' support for baseline?
            # Or just pass 'passthrough' and rely on the model.
            # The issue is that ColumnTransformer stacks everything into a numpy array.
            # Numpy array of strings + floats = Numpy array of objects/strings.
            # XGBoost fails on numpy array of strings ("could not convert string to float").
            # It needs Pandas DataFrame with Category dtype or specific DMatrix.

            # Since we are returning numpy array from preprocessor, we MUST encode strings.
            # Let's enforce OrdinalEncoder for "passthrough" categorical if model expects numbers.
            # But "baseline" implies raw.
            # If we want to support "passthrough", we must return a DataFrame from 'cached_fit_transform'.

            transformers.append(("cat", "passthrough", cat_config["features"]))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    # Force pandas output to support mixed types (strings + floats) for XGBoost
    ct.set_output(transform="pandas")
    return ct

def _ensure_category_dtype(X):
    """
    Helper to convert object columns to category dtype for XGBoost.
    """
    if hasattr(X, "select_dtypes"):
        obj_cols = X.select_dtypes(include=['object']).columns
        for col in obj_cols:
            X[col] = X[col].astype("category")
    return X

def build_full_pipeline(preprocessor: ColumnTransformer, model=None, memory=None):
    """
    Builds the full pipeline with optional memory caching.
    """
    steps = [("preprocessor", preprocessor)]
    if model is not None:
        steps.append(("model", model))

    return Pipeline(steps, memory=memory)
