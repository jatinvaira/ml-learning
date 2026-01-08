# Experiment Report

## Overview
- **Dataset**: `retail.csv`
- **Target**: `Discount_Applied` (Classification)
- **Sensitive Attribute**: `Location` (Online vs In-store)
- **Splitting**: Time-based (20% test).

## Results Summary

| Strategy | Model | Accuracy | AUC | F1 | Demo Parity Diff | Equalized Odds Diff | Train Time (s) |
|---|---|---|---|---|---|---|---|
| basic | Logistic Regression | 0.485 | 0.482 | 0.482 | 0.109 | 0.110 | 0.035 |
| basic | Random Forest | 0.502 | 0.501 | 0.502 | 0.052 | 0.110 | 0.500 |

## Analysis
- **Performance**: Random Forest slightly outperforms Logistic Regression, but both are hovering near random guessing (0.50). This suggests the synthetic features (Price, Category, etc.) might not be strong predictors of `Discount_Applied` in this dataset, or `Discount_Applied` is random.
- **Fairness**: Random Forest has lower Demographic Parity Difference (0.05 vs 0.11), meaning it selects positive outcomes (discounts) more equally across Locations.
- **Efficiency**: Logistic Regression is 10x faster to train.

## Recommendations
- **Model Choice**: Random Forest provides better fairness and marginally better performance.
- **Future Work**: Feature engineering is needed. `Discount_Applied` might depend on unobserved factors or complex interactions.
