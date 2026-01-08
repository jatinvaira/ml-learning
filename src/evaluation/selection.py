import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class SelectionEngine:
    def __init__(self, results_path: str):
        self.df = pd.read_json(results_path, lines=True)

    def aggregate_results(self, group_by: List[str] = ["strategy", "model", "complexity_steps"]) -> pd.DataFrame:
        """
        Aggregates results over splits/seeds.
        """
        # Expand metrics and timings columns if they are dicts (pd.read_json might load them as struct or dicts)
        # If read_json(lines=True) was used, nested dicts are columns containing dicts.

        # Helper to unpack dict columns
        def unpack_col(df, col_name, prefix=""):
            if col_name not in df.columns:
                return df

            # Check if it contains dicts
            if df[col_name].apply(lambda x: isinstance(x, dict)).any():
                expanded = pd.json_normalize(df[col_name])
                expanded.columns = [f"{prefix}{c}" for c in expanded.columns]
                # Join back using index
                expanded.index = df.index
                return pd.concat([df.drop(columns=[col_name]), expanded], axis=1)
            return df

        df_flat = unpack_col(self.df, "metrics", "")
        df_flat = unpack_col(df_flat, "timings", "")
        # In save_results, we construct flat_results for CSV but results.jsonl (self.df) is still nested.
        # "complexity" key exists in jsonl.
        df_flat = unpack_col(df_flat, "complexity", "comp_")

        # Ensure 'complexity_steps' is available if it was inside 'complexity' dict
        if "complexity_steps" not in df_flat.columns and "comp_steps" in df_flat.columns:
             df_flat["complexity_steps"] = df_flat["comp_steps"]

        # Numeric columns only for mean/std
        numeric_cols = df_flat.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude grouping cols from numeric_cols if they are there
        for g in group_by:
            if g in numeric_cols:
                numeric_cols.remove(g)

        agg_funcs = {c: ["mean", "std", "median"] for c in numeric_cols}

        summary = df_flat.groupby(group_by).agg(agg_funcs)

        # Flatten columns
        summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
        summary = summary.reset_index()

        return summary

    def compute_pareto_frontier(self, summary_df: pd.DataFrame, objectives: Dict[str, str]) -> pd.DataFrame:
        """
        Identifies the Pareto frontier based on objectives.
        objectives: dict mapping metric name -> 'min' or 'max'.
        e.g. {'f1_mean': 'max', 'demographic_parity_diff_mean': 'min', 'total_time_mean': 'min'}
        """
        subset = summary_df.copy()
        pareto_mask = np.ones(len(subset), dtype=bool)

        for i, row_i in subset.iterrows():
            for j, row_j in subset.iterrows():
                if i == j: continue

                # Check if j dominates i
                # j dominates i if j is at least as good as i in all objectives AND strictly better in at least one.

                is_worse_or_equal = True
                is_strictly_worse = False

                for obj, direction in objectives.items():
                    val_i = row_i[obj]
                    val_j = row_j[obj]

                    if direction == 'max':
                        if val_j < val_i: # j is worse than i
                            is_worse_or_equal = False
                            break
                        if val_j > val_i:
                            is_strictly_worse = True
                    else: # min
                        if val_j > val_i: # j is worse than i
                            is_worse_or_equal = False
                            break
                        if val_j < val_i:
                            is_strictly_worse = True

                if is_worse_or_equal and is_strictly_worse:
                    pareto_mask[i] = False
                    break

        return subset[pareto_mask]

    def filter_constraints(self, summary_df: pd.DataFrame, constraints: List[str]) -> pd.DataFrame:
        """
        Filters dataframe using query strings.
        e.g. "demographic_parity_diff_mean < 0.1"
        """
        df = summary_df.copy()
        for const in constraints:
            try:
                df = df.query(const)
            except Exception as e:
                print(f"Warning: Failed to apply constraint '{const}': {e}")
        return df

    def compute_weighted_utility(self, summary_df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """
        Computes a single utility score.
        """
        df = summary_df.copy()
        df["utility"] = 0.0

        # Normalize columns first (MinMax) to make weights comparable
        for col, weight in weights.items():
            if col not in df.columns:
                continue

            min_val = df[col].min()
            max_val = df[col].max()
            rng = max_val - min_val if max_val != min_val else 1.0

            # If weight is negative, we want to minimize, but usually weights are positive
            # and we handle direction by inverting the normalized value?
            # Or user provides negative weight for minimization.
            # Let's assume user provides weights for Maximization (positive) and Minimization (negative).

            norm_val = (df[col] - min_val) / rng
            df["utility"] += norm_val * weight

        return df.sort_values("utility", ascending=False)
