import numpy as np
import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names (optional but helps)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Types
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")

    # Coerce numerics
    for c in ["Price_Per_Unit", "Quantity", "Total_Spent"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove impossible values (tune rules to your dataset)
    df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
    df.loc[df["Price_Per_Unit"] <= 0, "Price_Per_Unit"] = np.nan

    # Optional: keep Discount_Applied as categorical string
    if "Discount_Applied" in df.columns:
        df["Discount_Applied"] = df["Discount_Applied"].astype("string")

    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["Transaction_Date"]
    df["tx_dow"] = dt.dt.dayofweek
    df["tx_month"] = dt.dt.month
    df["tx_day"] = dt.dt.day
    return df
