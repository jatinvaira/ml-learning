import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from fairlearn.metrics import MetricFrame

def evaluate_group_fairness(y_true, y_pred, sensitive: pd.Series):
    mf = MetricFrame(
        metrics={"mae": mean_absolute_error},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )  # MetricFrame can report overall and by-group metrics. [web:54]

    out = {
        "overall_mae": float(mf.overall["mae"]),
        "mae_by_group": mf.by_group["mae"].to_dict(),
        "mae_diff_between_groups": float(mf.difference(method="between_groups")["mae"]),
        "mae_ratio_between_groups": float(mf.ratio(method="between_groups")["mae"]),
    }  # MetricFrame offers group disparity aggregations like difference/ratio. [web:56]
    return out
