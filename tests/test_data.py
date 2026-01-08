import pytest
import pandas as pd
import numpy as np
from src.data.ingestion import load_data, validate_types
from src.data.splitting import Splitter
import os

# Fixtures
@pytest.fixture
def sample_csv(tmp_path):
    d = {
        "col1": [1, 2, 3, 4, 5],
        "col2": ["a", "b", "c", "d", "e"],
        "date": pd.date_range("2023-01-01", periods=5),
        "target": [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(d)
    filepath = tmp_path / "test.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)

def test_load_data(sample_csv):
    df = load_data(sample_csv)
    assert df.shape == (5, 4)
    assert "col1" in df.columns

def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent.csv")

def test_validate_types():
    df = pd.DataFrame({"num": ["1", "2"], "date": ["2020-01-01", "not_a_date"]})
    type_map = {"num": "numeric", "date": "datetime"}
    df_clean = validate_types(df, type_map)
    assert pd.api.types.is_float_dtype(df_clean["num"]) or pd.api.types.is_integer_dtype(df_clean["num"])
    assert pd.api.types.is_datetime64_any_dtype(df_clean["date"])
    assert pd.isna(df_clean["date"][1])

def test_splitter_random():
    df = pd.DataFrame({
        "feature": range(100),
        "target": [0, 1] * 50
    })
    splitter = Splitter(strategy="random", test_size=0.2, random_state=42)
    train, test = splitter.split(df, target_col="target")
    assert len(train) == 80
    assert len(test) == 20
    # Check stratification roughly
    assert abs(train["target"].mean() - 0.5) < 0.1

def test_splitter_time():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=100),
        "val": range(100)
    })
    splitter = Splitter(strategy="time", test_size=0.2, date_col="date")
    train, test = splitter.split(df)
    assert len(train) == 80
    assert len(test) == 20
    assert train["date"].max() < test["date"].min()
