import time
import pandas as pd
import logging
from src.data.ingestion import load_data, validate_types
from src.data.splitting import Splitter
from src.preprocessing.pipeline import create_preprocessor, build_full_pipeline
from src.models.factory import create_model
from src.evaluation.metrics import calculate_metrics
from sklearn.preprocessing import LabelEncoder
import json
import os

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config: dict):
        self.config = config
        self.results = []

    def run(self):
        # 1. Load Data
        data_cfg = self.config["data"]
        df = load_data(data_cfg["path"])

        # 2. Split
        split_cfg = self.config["splitting"]
        splitter = Splitter(
            strategy=split_cfg["strategy"],
            test_size=split_cfg.get("test_size", 0.2),
            random_state=self.config.get("random_seed", 42),
            date_col=split_cfg.get("date_col")
        )

        target = data_cfg["target"]
        train_df, test_df = splitter.split(df, target_col=target)

        # Clean targets (Drop NaNs)
        logger.info(f"Train shape before target cleanup: {train_df.shape}")
        train_df = train_df.dropna(subset=[target])
        test_df = test_df.dropna(subset=[target])
        logger.info(f"Train shape after target cleanup: {train_df.shape}")

        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]

        # Robust Target Encoding for Classification
        if data_cfg.get("task_type") == "classification":
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            # Handle unseen labels in test?
            # If test has labels not in train, transform raises error.
            # We filter test set to only include known labels.
            known_labels = set(le.classes_)
            mask = y_test.isin(known_labels)
            if (~mask).any():
                logger.warning(f"Dropping {len(y_test) - mask.sum()} test rows with unseen labels.")
                y_test = y_test[mask]
                X_test = X_test[mask]
                test_df = test_df[mask] # Update for sensitive attribute alignment

            y_test = le.transform(y_test)

        sensitive_col = data_cfg.get("sensitive_attribute")
        sensitive_test = test_df[sensitive_col] if sensitive_col else None

        # 3. Iterate Strategies
        for strat_name, strat_cfg in self.config["strategies"].items():
            preprocessor = create_preprocessor(strat_cfg["preprocessing"])

            for model_name, model_cfg in self.config["models"].items():
                logger.info(f"Running {strat_name} + {model_name}...")

                model = create_model(model_cfg)
                pipeline = build_full_pipeline(preprocessor, model)

                start_fit = time.time()
                pipeline.fit(X_train, y_train)
                train_time = time.time() - start_fit

                start_infer = time.time()
                y_pred = pipeline.predict(X_test)
                infer_time = time.time() - start_infer

                y_prob = None
                if hasattr(pipeline, "predict_proba"):
                    try:
                        y_prob = pipeline.predict_proba(X_test)
                    except AttributeError:
                        pass

                metrics = calculate_metrics(
                    y_test, y_pred, y_prob,
                    task_type=data_cfg["task_type"],
                    sensitive_features=sensitive_test
                )

                result = {
                    "strategy": strat_name,
                    "model": model_name,
                    "train_time": train_time,
                    "infer_time": infer_time,
                    "metrics": metrics
                }
                self.results.append(result)

    def save_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.json_normalize(self.results)
        results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
        with open(os.path.join(output_dir, "results.jsonl"), "w") as f:
            for r in self.results:
                f.write(json.dumps(r) + "\n")
