import pandas as pd
import logging
from typing import Tuple, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads data from a CSV file and performs basic schema validation.

    Args:
        filepath: Path to the CSV file.
        required_columns: List of columns that must be present.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalize columns
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded data from {filepath} with shape {df.shape}")
    return df

def validate_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """
    Validates and coerces column types based on a map.

    Args:
        df: Input dataframe.
        type_map: Dictionary mapping column names to types (e.g., {'Price': 'float'}).

    Returns:
        pd.DataFrame: Dataframe with coerced types.
    """
    df = df.copy()
    for col, dtype in type_map.items():
        if col in df.columns:
            try:
                if dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == 'numeric':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Failed to convert {col} to {dtype}: {e}")
    return df
