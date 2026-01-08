from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference

def calculate_metrics(y_true, y_pred, y_prob=None, task_type="classification", sensitive_features=None) -> Dict[str, float]:
    """
    Computes performance and fairness metrics.
    """
    metrics = {}

    # Performance
    if task_type == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred, average="weighted")
        if y_prob is not None:
            # Handle multiclass or binary
            try:
                if len(np.unique(y_true)) > 2:
                     metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
                else:
                     metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            except ValueError:
                metrics["auc"] = 0.0 # Fallback
    else:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["r2"] = r2_score(y_true, y_pred)

    # Fairness
    if sensitive_features is not None:
        try:
            if task_type == "classification":
                metrics["demographic_parity_diff"] = demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_features
                )
                metrics["equalized_odds_diff"] = equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sensitive_features
                )
            else:
                # Regression fairness: Difference in MAE across groups
                mf = MetricFrame(
                    metrics=mean_absolute_error,
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=sensitive_features
                )
                metrics["mae_diff"] = mf.difference()
                metrics["mae_ratio"] = mf.ratio()
        except Exception as e:
            # logging.warning(f"Fairness calculation failed: {e}")
            metrics["fairness_error"] = 1.0

    return metrics
