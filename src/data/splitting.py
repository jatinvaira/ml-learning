import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedShuffleSplit
from typing import Tuple, Optional, Generator

class Splitter:
    def __init__(self, strategy: str = "random", test_size: float = 0.2, random_state: int = 42, date_col: Optional[str] = None, n_splits: int = 3, gap: int = 0, min_train_size: Optional[int] = None):
        """
        Args:
            strategy: 'random' or 'time' or 'time_rolling'.
            test_size: Fraction of data for testing (for random/time) or size of test set (for rolling).
            random_state: Seed for reproducibility.
            date_col: Column name for sorting in time-based split.
            n_splits: Number of splits for 'time_rolling'.
            gap: Gap between train and test in 'time_rolling'.
            min_train_size: Minimum training size for 'time_rolling'.
        """
        self.strategy = strategy
        self.test_size = test_size
        self.random_state = random_state
        self.date_col = date_col
        self.n_splits = n_splits
        self.gap = gap
        self.min_train_size = min_train_size

    def split(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Legacy single split method.
        """
        if self.strategy == "time":
            if not self.date_col or self.date_col not in df.columns:
                raise ValueError("Date column required for time-based split")

            df_sorted = df.sort_values(self.date_col).copy()
            split_idx = int(len(df_sorted) * (1 - self.test_size))
            return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]

        elif self.strategy == "random":
             return self._random_split(df, target_col)

        elif self.strategy == "time_rolling":
             # Return the last split for backward compatibility if called via split()
             splits = list(self.get_rolling_splits(df))
             return splits[-1]

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def get_rolling_splits(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generator for time-based rolling splits.
        """
        if self.strategy != "time_rolling":
            # Just yield the single split if not rolling
            yield self.split(df)
            return

        if not self.date_col or self.date_col not in df.columns:
            raise ValueError("Date column required for time_rolling split")

        df_sorted = df.sort_values(self.date_col).reset_index(drop=True)

        # Calculate max_train_size if needed, but usually we expand.
        # TimeSeriesSplit expects n_splits.
        # Note: TimeSeriesSplit does not directly take test_size as float.
        # It splits into (n_splits + 1) chunks.

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap,
            max_train_size=None, # Expanding window
            test_size=None # Auto-calculated
        )

        for train_index, test_index in tscv.split(df_sorted):
            train_df = df_sorted.iloc[train_index]
            test_df = df_sorted.iloc[test_index]

            # Verify time ordering
            if train_df[self.date_col].max() > test_df[self.date_col].min():
                 raise ValueError("Train data contains dates after Test data start.")

            yield train_df, test_df

    def _random_split(self, df, target_col):
        # ... (Existing logic for random split) ...
        if target_col is None:
                return self._simple_shuffle(df)

        valid_mask = df[target_col].notna()
        if not valid_mask.all():
                pass

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)

        is_categorical = pd.api.types.is_object_dtype(df[target_col]) or pd.api.types.is_bool_dtype(df[target_col]) or pd.api.types.is_categorical_dtype(df[target_col])

        if not is_categorical:
                return self._simple_shuffle(df)

        try:
            train_idx, test_idx = next(sss.split(df, df[target_col]))
            return df.iloc[train_idx], df.iloc[test_idx]
        except ValueError:
            return self._simple_shuffle(df)

    def _simple_shuffle(self, df: pd.DataFrame):
        df_shuffled = df.sample(frac=1, random_state=self.random_state)
        split_idx = int(len(df_shuffled) * (1 - self.test_size))
        return df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]
