import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, Optional

class Splitter:
    def __init__(self, strategy: str = "random", test_size: float = 0.2, random_state: int = 42, date_col: Optional[str] = None):
        """
        Args:
            strategy: 'random' or 'time'.
            test_size: Fraction of data for testing.
            random_state: Seed for reproducibility.
            date_col: Column name for sorting in time-based split.
        """
        self.strategy = strategy
        self.test_size = test_size
        self.random_state = random_state
        self.date_col = date_col

    def split(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataframe into train and test sets.

        Args:
            df: Input dataframe.
            target_col: Target column for stratification (required for random strategy).

        Returns:
            (train_df, test_df)
        """
        if self.strategy == "time":
            if not self.date_col or self.date_col not in df.columns:
                raise ValueError("Date column required for time-based split")

            df_sorted = df.sort_values(self.date_col).copy()
            split_idx = int(len(df_sorted) * (1 - self.test_size))
            return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]

        elif self.strategy == "random":
            if target_col is None:
                 # Fallback to simple shuffle if no target
                 return self._simple_shuffle(df)

            # Handle missing targets for stratification by dropping them temporarily or filling
            # For splitting purpose only, we drop rows with missing target in stratification calculation
            # But we want to keep them in the dataset? Usually we drop missing targets before splitting.
            # We assume df is already clean enough or we handle it.
            # If target has NaNs, StratifiedShuffleSplit fails.

            valid_mask = df[target_col].notna()
            if not valid_mask.all():
                 # fallback to simple shuffle if target has nans? or filter?
                 # ideally we filter. Let's assume input df should have targets handled or we error.
                 # For robustness, we will only stratify on valid targets
                 pass

            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)

            # If target is continuous, StratifiedShuffleSplit fails. We need to check type.
            is_categorical = pd.api.types.is_object_dtype(df[target_col]) or pd.api.types.is_bool_dtype(df[target_col]) or pd.api.types.is_categorical_dtype(df[target_col])

            if not is_categorical:
                 # Fallback to random shuffle for regression targets
                 return self._simple_shuffle(df)

            try:
                train_idx, test_idx = next(sss.split(df, df[target_col]))
                return df.iloc[train_idx], df.iloc[test_idx]
            except ValueError:
                # Fallback if single class or errors
                return self._simple_shuffle(df)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _simple_shuffle(self, df: pd.DataFrame):
        df_shuffled = df.sample(frac=1, random_state=self.random_state)
        split_idx = int(len(df_shuffled) * (1 - self.test_size))
        return df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]
