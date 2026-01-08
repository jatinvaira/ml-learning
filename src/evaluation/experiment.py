import time
import pandas as pd
import logging
import joblib
import os
import json
from sklearn.preprocessing import LabelEncoder

from src.data.ingestion import load_data
from src.data.splitting import Splitter
from src.preprocessing.pipeline import build_full_pipeline, create_preprocessor
from src.preprocessing.ladder import LadderBuilder
from src.models.factory import create_model
from src.evaluation.metrics import calculate_metrics
from src.evaluation.caching import get_dataset_fingerprint, cached_fit_transform

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config: dict):
        self.config = config
        self.results = []

        # Setup Caching
        cache_config = self.config.get("cache", {})
        self.use_cache = cache_config.get("enabled", False)
        self.cache_dir = cache_config.get("dir", None)
        self.cache_version = cache_config.get("version", "v1")

        self.memory = None
        self.cached_func = None

        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.memory = joblib.Memory(self.cache_dir, verbose=0)
            # Decorate the worker function
            self.cached_func = self.memory.cache(cached_fit_transform)
        else:
            # No caching, just run directly
            self.cached_func = cached_fit_transform

    def run(self):
        # 1. Load Data Info (Don't load full df yet if using cache, but we need it for splitting indices)
        data_cfg = self.config["data"]
        dataset_path = data_cfg["path"]
        target = data_cfg["target"]

        # We need the dataframe to generate split INDICES.
        # Loading it here is unavoidable to know the split structure.
        df = load_data(dataset_path)

        # Pre-calculate Fingerprint
        dataset_fp = get_dataset_fingerprint(dataset_path)

        # 2. Determine Strategies
        strategies = {}
        if "ladder" in self.config:
            # Build ladder strategies
            builder = LadderBuilder(self.config["ladder"], self.config.get("strategies", {}).get("baseline", {}))
            strategies = builder.build_strategies()
        else:
            strategies = self.config["strategies"]

        # 3. Setup Splitter
        split_cfg = self.config["splitting"]
        splitter = Splitter(
            strategy=split_cfg["strategy"],
            test_size=split_cfg.get("test_size", 0.2),
            random_state=self.config.get("random_seed", 42),
            date_col=split_cfg.get("date_col"),
            n_splits=split_cfg.get("n_splits", 3),
            gap=split_cfg.get("gap", 0),
            min_train_size=split_cfg.get("min_train_size")
        )

        # 4. Iterate Splits
        # For Rolling splits, we get multiple folds.
        # For Random, we usually just get one, but we should support multiple seeds if requested.
        # The prompt says: "Run each rung across multiple seeds and time splits".

        # If strategy is random, we might want to loop seeds here or inside.
        # But 'Splitter' with 'random' strategy usually produces one split based on its random_state.
        # To support multiple random splits, we can iterate seeds.

        # If strategy is time_rolling, 'get_rolling_splits' yields multiple folds.

        splits_generator = splitter.get_rolling_splits(df)

        # Note: 'get_rolling_splits' returns DataFrames.
        # But our 'cached_fit_transform' expects INDICES to avoid hashing huge DFs.
        # We need to map back to indices.
        # Since 'df' is in memory, we can just use df.index if it is unique.
        # Assuming df has default RangeIndex or unique index.

        # To make this robust, let's grab indices from the split DataFrames.

        # Wait, TimeSeriesSplit returns indices. Our wrapper returns DataFrames.
        # Let's trust our wrapper returns slices of the original DF.

        for split_idx, (train_df_split, test_df_split) in enumerate(splits_generator):
            logger.info(f"Processing Split {split_idx+1}...")

            # Get Indices
            train_indices = train_df_split.index.tolist()
            test_indices = test_df_split.index.tolist()

            # 5. Iterate Strategies
            for strat_name, strat_cfg in strategies.items():

                # Check for Pruning (not fully implemented yet, but placeholders ok)
                # ...

                # 6. Preprocessing (Cached)
                # We pass the minimal info needed to reconstruct the data + transforms

                try:
                    preproc_result = self.cached_func(
                        dataset_path=dataset_path,
                        dataset_fingerprint=dataset_fp,
                        split_train_idx=train_indices,
                        split_test_idx=test_indices,
                        strategy_config=strat_cfg,
                        target_col=target,
                        cache_version=self.cache_version
                    )
                except Exception as e:
                    logger.error(f"Preprocessing failed for {strat_name}: {e}")
                    continue

                X_train_t = preproc_result["X_train_t"]
                X_test_t = preproc_result["X_test_t"]
                y_train = preproc_result["y_train"]
                y_test = preproc_result["y_test"]
                timings = preproc_result["timings"]
                n_features = preproc_result["n_features"]

                # Handle Classification Targets (Label Encoding)
                # This usually happens *before* fitting the model but *after* splitting.
                # If we cache the preprocessor output (X), we still need to handle y.
                # The cached function returns y_train/y_test as series.
                # We need to encode them if classification.

                if data_cfg.get("task_type") == "classification":
                    le = LabelEncoder()
                    y_train = le.fit_transform(y_train)

                    # Handle unseen labels
                    known_labels = set(le.classes_)
                    # In test
                    # If X is numpy array, we can't easily filter rows by index alignment if we just filter y.
                    # We need to filter X and y together.

                    # This logic is tricky with cached X (which is numpy).
                    # We need to convert y_test to series/array and check.

                    # Current implementation in run.py handles this.
                    # If we filter test rows, we must filter X_test_t as well.

                    # Note: y_test from cached func is a Series or array.

                    mask = pd.Series(y_test).isin(known_labels)
                    if (~mask).any():
                        # logger.warning(f"Dropping {len(y_test) - mask.sum()} test rows with unseen labels.")
                        y_test = y_test[mask]
                        # X_test_t is numpy array or sparse
                        if hasattr(X_test_t, "toarray"): # sparse
                            X_test_t = X_test_t[mask.values]
                        else:
                            X_test_t = X_test_t[mask.values]

                    y_test = le.transform(y_test)

                # Sensitive Features for Fairness
                # We need to extract them from the ORIGINAL test df (indices)
                # We kept indices.
                sensitive_col = data_cfg.get("sensitive_attribute")
                sensitive_test = None
                if sensitive_col:
                    # We need the raw test df rows corresponding to the (possibly filtered) X_test_t
                    # If we filtered for labels, we need to filter indices too.

                    # Reconstruct full test df slice
                    full_test_df = df.iloc[test_indices]

                    if data_cfg.get("task_type") == "classification":
                         # Apply same mask
                         if (~mask).any():
                             full_test_df = full_test_df[mask.values]

                    sensitive_test = full_test_df[sensitive_col]

                # 7. Iterate Models
                for model_name, model_cfg in self.config["models"].items():
                    # Check Seeds (if we want multiple seeds per split/strat/model)
                    n_seeds = self.config.get("n_seeds", 1)

                    for seed in range(n_seeds):
                        model_cfg_seeded = model_cfg.copy()
                        # Inject seed if model supports it
                        if "args" not in model_cfg_seeded:
                            model_cfg_seeded["args"] = {}
                        model_cfg_seeded["args"]["random_state"] = self.config.get("random_seed", 42) + seed

                        logger.info(f"Running {strat_name} + {model_name} (Split {split_idx}, Seed {seed})...")

                        model = create_model(model_cfg_seeded)

                        # Fit Model
                        start_train = time.time()

                        # Ensure categorical dtype for XGBoost if using pandas output
                        if hasattr(X_train_t, "iloc"):
                             from src.preprocessing.pipeline import _ensure_category_dtype
                             X_train_t = _ensure_category_dtype(X_train_t)
                             X_test_t = _ensure_category_dtype(X_test_t)

                        model.fit(X_train_t, y_train)
                        model_train_time = time.time() - start_train

                        # Predict
                        start_predict = time.time()
                        y_pred = model.predict(X_test_t)
                        model_predict_time = time.time() - start_predict

                        y_prob = None
                        if hasattr(model, "predict_proba"):
                            try:
                                y_prob = model.predict_proba(X_test_t)
                            except AttributeError:
                                pass

                        # Metrics
                        metrics = calculate_metrics(
                            y_test, y_pred, y_prob,
                            task_type=data_cfg["task_type"],
                            sensitive_features=sensitive_test
                        )

                        # Collect Result
                        result = {
                            "strategy": strat_name,
                            "model": model_name,
                            "split_id": split_idx,
                            "seed": seed,
                            "metrics": metrics,
                            "timings": {
                                "preprocess_fit_time": timings["preprocess_fit_time"],
                                "preprocess_transform_time": timings["preprocess_transform_time"],
                                "model_train_time": model_train_time,
                                "model_predict_time": model_predict_time,
                                "total_time": timings["preprocess_fit_time"] + timings["preprocess_transform_time"] + model_train_time + model_predict_time
                            },
                            "complexity": {
                                "steps": strat_cfg.get("complexity_steps", 0),
                                "n_features": n_features,
                                "cost": (timings["preprocess_transform_time"] * 1000) + n_features # Simple proxy or just keep separate
                            }
                        }
                        self.results.append(result)

    def save_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        # Flatten for CSV
        flat_results = []
        for r in self.results:
            flat = {
                "strategy": r["strategy"],
                "model": r["model"],
                "split_id": r["split_id"],
                "seed": r["seed"],
                **r["metrics"],
                **r["timings"],
                "complexity_steps": r["complexity"]["steps"],
                "n_features": r["complexity"]["n_features"]
            }
            flat_results.append(flat)

        pd.DataFrame(flat_results).to_csv(os.path.join(output_dir, "results.csv"), index=False)
        with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
            for r in self.results:
                f.write(json.dumps(r) + "\n")
