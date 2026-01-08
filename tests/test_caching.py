import unittest
import pandas as pd
import numpy as np
import os
import shutil
import joblib
from src.evaluation.caching import cached_fit_transform, get_dataset_fingerprint

class TestCaching(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_cache_tmp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.cache_dir = os.path.join(self.test_dir, "joblib_cache")
        self.memory = joblib.Memory(self.cache_dir, verbose=0)
        self.cached_func = self.memory.cache(cached_fit_transform)

        # Create dummy dataset
        self.dataset_path = os.path.join(self.test_dir, "data.csv")
        df = pd.DataFrame({
            "A": np.random.rand(20),
            "B": np.random.choice(["x", "y"], 20),
            "target": np.random.randint(0, 2, 20)
        })
        df.to_csv(self.dataset_path, index=False)
        self.fp = get_dataset_fingerprint(self.dataset_path)

        self.config = {
            "preprocessing": {
                "numerical": {"features": ["A"], "steps": [{"type": "standard_scaler", "name": "scaler"}]},
                "categorical": {"features": ["B"], "steps": [{"type": "one_hot", "name": "encoder"}]}
            }
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_caching_behavior(self):
        # First Run
        start_1 = pd.Timestamp.now()
        res1 = self.cached_func(
            dataset_path=self.dataset_path,
            dataset_fingerprint=self.fp,
            split_train_idx=list(range(10)),
            split_test_idx=list(range(10, 20)),
            strategy_config=self.config,
            target_col="target",
            cache_version="v1"
        )
        end_1 = pd.Timestamp.now()

        # Second Run (Same inputs)
        start_2 = pd.Timestamp.now()
        res2 = self.cached_func(
            dataset_path=self.dataset_path,
            dataset_fingerprint=self.fp,
            split_train_idx=list(range(10)),
            split_test_idx=list(range(10, 20)),
            strategy_config=self.config,
            target_col="target",
            cache_version="v1"
        )
        end_2 = pd.Timestamp.now()

        # Check if results match
        np.testing.assert_array_almost_equal(res1["X_train_t"], res2["X_train_t"])

        # Check timings - 2nd run should be instant (only loading from disk cache)
        # Note: joblib loading from disk can take time, but 'fit' time inside result should be identical
        # because it returns the cached return value (which contains the ORIGINAL fit time).
        self.assertEqual(res1["timings"]["preprocess_fit_time"], res2["timings"]["preprocess_fit_time"])

        # Verify cache directory is populated
        self.assertTrue(os.path.exists(self.cache_dir))

    def test_leakage(self):
        # Ensure fit is only on train
        # We can't easily spy on the internal object, but we can check if transformation depends on test data.
        # e.g., if we use standard scaler, mean should be from train.

        # Manually calculate mean of train
        df = pd.read_csv(self.dataset_path)
        train_mean = df.iloc[:10]["A"].mean()
        train_std = df.iloc[:10]["A"].std(ddof=1) # Sklearn uses ddof=0? No, StandardScaler uses biased estimator (std(ddof=0))
        train_std_sklearn = df.iloc[:10]["A"].std(ddof=0)

        res = self.cached_func(
            dataset_path=self.dataset_path,
            dataset_fingerprint=self.fp,
            split_train_idx=list(range(10)),
            split_test_idx=list(range(10, 20)),
            strategy_config=self.config,
            target_col="target",
            cache_version="v1"
        )

        # Check first element of transformed train
        # (X - mean) / std
        val = df.iloc[0]["A"]
        expected = (val - train_mean) / train_std_sklearn

        # There might be small float differences
        # X_train_t is a DataFrame, so we use iloc
        self.assertAlmostEqual(res["X_train_t"].iloc[0, 0], expected, places=5)

if __name__ == "__main__":
    unittest.main()
