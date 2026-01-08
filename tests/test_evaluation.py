import pytest
import pandas as pd
import numpy as np
from src.evaluation.metrics import calculate_metrics
from src.evaluation.experiment import ExperimentRunner
from src.models.factory import create_model

def test_metrics_classification():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    metrics = calculate_metrics(y_true, y_pred, task_type="classification")
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.75

def test_metrics_fairness():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    sensitive = ["a", "a", "b", "b"]
    metrics = calculate_metrics(y_true, y_pred, task_type="classification", sensitive_features=sensitive)
    assert "demographic_parity_diff" in metrics
    assert metrics["demographic_parity_diff"] == 0.0

def test_model_factory():
    model = create_model({"type": "logistic_regression", "args": {"C": 0.1}})
    assert model.C == 0.1

def test_experiment_runner(tmp_path):
    # Create dummy data
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4] * 5,
        "f2": ["a", "b", "a", "b"] * 5,
        "target": [0, 1, 0, 1] * 5,
        "group": ["x", "x", "y", "y"] * 5
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    config = {
        "data": {
            "path": str(data_path),
            "target": "target",
            "sensitive_attribute": "group",
            "task_type": "classification"
        },
        "splitting": {"strategy": "random"},
        "strategies": {
            "base": {
                "preprocessing": {
                    "numerical": {"features": ["f1"], "steps": [{"name": "imp", "type": "simple_imputer"}]},
                    "categorical": {"features": ["f2"], "steps": [{"name": "enc", "type": "one_hot"}]}
                }
            }
        },
        "models": {
            "lr": {"type": "logistic_regression"}
        }
    }

    runner = ExperimentRunner(config)
    runner.run()
    runner.save_results(str(tmp_path))

    assert (tmp_path / "results.csv").exists()
