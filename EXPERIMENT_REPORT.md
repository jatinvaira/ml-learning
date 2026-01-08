# Experiment Report

## Overview
- **Dataset**: `retail.csv`
- **Target**: `Discount_Applied` (Classification)
- **Sensitive Attribute**: `Location` (Online vs In-store)
- **Splitting**: Time-based Rolling (3 splits).

## Multi-Objective Ladder Experiment
We evaluated a "Complexity Ladder" of preprocessing strategies to demonstrate that "more is not always better." The ladder incrementally adds steps:

1.  **Rung 0 (Baseline)**: Impute (Median/Mode) -> Pass-through
2.  **Rung 1**: + Encode (OneHot)
3.  **Rung 2**: + Scale (StandardScaler)
4.  **Rung 3**: + Clip Outliers (1%-99%)

Results are averaged over 3 time-based splits.

### Complexity vs Outcomes (Non-monotonicity)

| Strategy   |   Complexity Steps |   F1 Score |   Demographic Parity Diff |   Total Runtime (s) |
|:-----------|-------------------:|----------:|--------------------------:|--------------------:|
| rung_0     |                  0 |  0.343    |                      0.066 |               0.150 |
| rung_1     |                  1 |  0.335    |                      0.078 |               0.147 |
| rung_2     |                  2 |  0.330    |                      0.071 |               0.154 |
| rung_3     |                  3 |  0.330    |                      0.071 |               0.158 |
| rung_4     |                  4 |  0.333    |                      0.092 |               0.178 |

*Note: Results show that Rung 0 (baseline) actually achieves the highest F1 score and lowest fairness gap, contradicting the assumption that more complex preprocessing (encoding, scaling) automatically yields better models for this specific dataset and model (XGBoost).*

### Pareto Optimal Strategies
Strategies that are non-dominated on F1 (maximize) and Demographic Parity Difference (minimize):

| strategy   |   complexity_steps |   f1_mean |   demographic_parity_diff_mean |
|:-----------|-------------------:|----------:|-------------------------------:|
| rung_0     |                  0 |  0.342747 |                      0.0657305 |

Only **Rung 0** is Pareto optimal here because it dominates others on both performance and fairness in this run. This strongly supports the "incremental complexity" assessment—we stop at Rung 0.

## Recommendations
- **Model Choice**: XGBoost with minimal preprocessing (Rung 0).
- **Process**: Use the newly implemented Selection Layer to prune complex pipelines that increase cost/runtime without gain.
