import unittest
import pandas as pd
import numpy as np
from src.data.splitting import Splitter

class TestRollingSplits(unittest.TestCase):
    def setUp(self):
        # Create dummy time series data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.df = pd.DataFrame({
            "date": dates,
            "value": np.random.randn(100),
            "target": np.random.randint(0, 2, 100)
        })

    def test_rolling_split_counts(self):
        n_splits = 3
        splitter = Splitter(strategy="time_rolling", date_col="date", n_splits=n_splits)
        splits = list(splitter.get_rolling_splits(self.df))

        self.assertEqual(len(splits), n_splits)

    def test_time_ordering(self):
        splitter = Splitter(strategy="time_rolling", date_col="date", n_splits=3)
        for train, test in splitter.get_rolling_splits(self.df):
            self.assertTrue(train["date"].max() <= test["date"].min())

    def test_gap(self):
        gap = 2
        splitter = Splitter(strategy="time_rolling", date_col="date", n_splits=3, gap=gap)
        for train, test in splitter.get_rolling_splits(self.df):
            # The difference between first test date and last train date should be > gap
            diff = (test["date"].min() - train["date"].max()).days
            self.assertGreater(diff, gap)

    def test_expanding_window(self):
        splitter = Splitter(strategy="time_rolling", date_col="date", n_splits=3)
        splits = list(splitter.get_rolling_splits(self.df))

        # Check that train size increases
        train_sizes = [len(s[0]) for s in splits]
        self.assertEqual(train_sizes, sorted(train_sizes))
        self.assertTrue(train_sizes[1] > train_sizes[0])

if __name__ == "__main__":
    unittest.main()
