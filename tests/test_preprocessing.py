import pytest
import pandas as pd
import numpy as np
from src.preprocessing.pipeline import create_preprocessor, OutlierClipper
from sklearn.pipeline import Pipeline

def test_outlier_clipper():
    X = pd.DataFrame({"a": [1, 2, 3, 100]})
    # quantile(0.75) of [1, 2, 3, 100] is 27.25.
    clipper = OutlierClipper(lower_percentile=0, upper_percentile=0.75)
    clipper.fit(X)

    X_trans = clipper.transform(X)
    assert X_trans["a"].max() <= 27.3
    assert X_trans["a"].iloc[-1] == 27.25

def test_create_preprocessor():
    config = {
        "numerical": {
            "features": ["num"],
            "steps": [
                {"name": "imputer", "type": "simple_imputer", "args": {"strategy": "mean"}},
                {"name": "scaler", "type": "standard_scaler"}
            ]
        },
        "categorical": {
            "features": ["cat"],
            "steps": [
                {"name": "imputer", "type": "simple_imputer", "args": {"strategy": "most_frequent"}},
                {"name": "encoder", "type": "one_hot", "args": {"handle_unknown": "ignore", "sparse_output": False}}
            ]
        }
    }

    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan],
        "cat": ["a", "b", np.nan]
    })

    pre = create_preprocessor(config)
    out = pre.fit_transform(df)

    assert out.shape[0] == 3
    assert out.shape[1] >= 2 # 1 num + at least 1 cat
