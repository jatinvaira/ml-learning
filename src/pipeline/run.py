import argparse
import yaml
import logging
import joblib
import json
import os
import pandas as pd
from src.data.ingestion import load_data
from src.data.splitting import Splitter
from src.preprocessing.pipeline import create_preprocessor, build_full_pipeline
from src.models.factory import create_model
from src.evaluation.metrics import calculate_metrics
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Single Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--strategy", type=str, help="Name of strategy to use")
    parser.add_argument("--model", type=str, help="Name of model to use")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Select strategy/model
    strat_name = args.strategy if args.strategy else list(config["strategies"].keys())[0]
    model_name = args.model if args.model else list(config["models"].keys())[0]

    if strat_name not in config["strategies"]:
        raise ValueError(f"Strategy {strat_name} not found in config.")
    if model_name not in config["models"]:
        raise ValueError(f"Model {model_name} not found in config.")

    strat_cfg = config["strategies"][strat_name]
    model_cfg = config["models"][model_name]

    logger.info(f"Running pipeline with Strategy={strat_name} and Model={model_name}")

    # Load and Split
    df = load_data(config["data"]["path"])
    target = config["data"]["target"]

    splitter = Splitter(
        strategy=config["splitting"]["strategy"],
        date_col=config["splitting"].get("date_col")
    )
    train_df, test_df = splitter.split(df, target_col=target)

    # Handle missing target in train/test
    train_df = train_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Handle classification targets robustly
    if config["data"].get("task_type") == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        # Check for unseen labels in test
        known_labels = set(le.classes_)
        mask = y_test.isin(known_labels)
        if (~mask).any():
            logger.warning(f"Dropping {len(y_test) - mask.sum()} test rows with unseen labels.")
            y_test = y_test[mask]
            X_test = X_test[mask]
            test_df = test_df[mask]
        y_test = le.transform(y_test)

    # Build and Fit
    preprocessor = create_preprocessor(strat_cfg["preprocessing"])
    model = create_model(model_cfg)
    pipeline = build_full_pipeline(preprocessor, model)

    pipeline.fit(X_train, y_train)

    # Predict and Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)
        except AttributeError:
            pass

    sensitive_col = config["data"].get("sensitive_attribute")
    sensitive_test = test_df[sensitive_col] if sensitive_col else None

    metrics = calculate_metrics(
        y_test, y_pred, y_prob,
        task_type=config["data"]["task_type"],
        sensitive_features=sensitive_test
    )

    # Save Artifacts
    os.makedirs(args.output, exist_ok=True)

    # 1. Pipeline
    joblib.dump(pipeline, os.path.join(args.output, "pipeline.pkl"))

    # 2. Metrics
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 3. Config (Reproducibility)
    run_config = {
        "strategy": strat_name,
        "model": model_name,
        "full_config": config
    }
    with open(os.path.join(args.output, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info(f"Artifacts saved to {args.output}")

if __name__ == "__main__":
    main()
