import hashlib
import os
import joblib
import pandas as pd
import time
from src.preprocessing.pipeline import create_preprocessor
import logging

logger = logging.getLogger(__name__)

def get_dataset_fingerprint(filepath: str) -> str:
    """
    Computes a SHA256 hash of the file content.
    """
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# We define the function to be cached.
# It needs to be at module level for joblib.

def cached_fit_transform(
    dataset_path: str,
    dataset_fingerprint: str,
    split_train_idx: list,
    split_test_idx: list,
    strategy_config: dict,
    target_col: str,
    cache_version: str
):
    """
    Fits the preprocessor on the train split and transforms both train and test.
    Returns transformed arrays and timings.

    Args:
        dataset_path: Path to the full dataset.
        dataset_fingerprint: Hash of the dataset file (ensures validity).
        split_train_idx: Indices for training rows.
        split_test_idx: Indices for test rows.
        strategy_config: Preprocessing configuration.
        target_col: Name of the target column.
        cache_version: String to invalidate cache if code changes.
    """
    # Load Data (we only load what we need if possible, but usually pandas loads all)
    # Using 'read_csv' with skiprows might be optimization, but let's stick to simple load first.
    # We assume the indices are integer positions (iloc).

    start_load = time.time()
    df = pd.read_csv(dataset_path) # Ingestion might be more complex, but we stick to basic here.
    # If ingestion has custom logic (like date parsing), we should reuse src.data.ingestion.load_data
    # But we can't easily import it here if it causes circular deps?
    # Actually, experiment.py calls this, so we can pass the data?
    # No, passing data breaks hashing efficiency.
    # Let's import load_data inside.
    from src.data.ingestion import load_data
    df = load_data(dataset_path)
    load_time = time.time() - start_load

    # Split
    train_df = df.iloc[split_train_idx]
    test_df = df.iloc[split_test_idx]

    # Separate Target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Create Preprocessor
    preprocessor = create_preprocessor(strategy_config["preprocessing"])

    # Fit
    start_fit = time.time()
    preprocessor.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Transform
    start_trans = time.time()
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    transform_time = time.time() - start_trans

    # Metadata
    # If feature selection was done, we would have fewer columns.
    # If the preprocessor is a ColumnTransformer, output is usually numpy array or sparse matrix.
    # We can get shape from it.

    n_features = X_train_t.shape[1]

    return {
        "X_train_t": X_train_t,
        "X_test_t": X_test_t,
        "y_train": y_train,
        "y_test": y_test,
        "timings": {
            "preprocess_fit_time": fit_time,
            "preprocess_transform_time": transform_time,
            "load_time": load_time
        },
        "n_features": n_features
    }
