import time
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from preprocess import basic_clean, add_datetime_features

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def make_pipeline(model):
    # NOTE: after add_datetime_features, these exist
    numeric_features = ["Price_Per_Unit", "Quantity", "tx_dow", "tx_month", "tx_day"]
    categorical_features = ["Category", "Item", "Payment_Method", "Location", "Discount_Applied"]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )  # ColumnTransformer is designed for heterogeneous column preprocessing. [web:63]

    pipe = Pipeline(steps=[
        ("dt_features", FunctionTransformer(lambda df: add_datetime_features(df), feature_names_out="one-to-one")),
        ("preprocess", pre),
        ("model", model),
    ])
    return pipe

def main():
    df = pd.read_csv("data/raw/retail.csv")
    df = basic_clean(df)

    # Target
    y = df["Total_Spent"]
    X = df.drop(columns=["Total_Spent"])

    # Time-based split (transaction-level): sort by Transaction_Date to avoid leakage
    df2 = pd.concat([X, y], axis=1).sort_values("Transaction_Date")
    y = df2["Total_Spent"]
    X = df2.drop(columns=["Total_Spent"])

    # Candidate model families
    candidates = {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "hgb": HistGradientBoostingRegressor(random_state=42),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }

    tscv = TimeSeriesSplit(n_splits=5)

    results = []
    for name, model in candidates.items():
        pipe = make_pipeline(model)

        fold_metrics = []
        start_fit = time.perf_counter()
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]

            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)

            fold_metrics.append({
                "model": name,
                "fold": fold,
                "mae": float(mean_absolute_error(yte, pred)),
                "rmse": rmse(yte, pred),
            })

        fit_seconds = time.perf_counter() - start_fit
        for row in fold_metrics:
            row["fit_seconds_total_for_cv"] = fit_seconds

        results.extend(fold_metrics)

    pd.DataFrame(results).to_csv("runs/cv_results.csv", index=False)
    print("Wrote runs/cv_results.csv")

if __name__ == "__main__":
    main()
