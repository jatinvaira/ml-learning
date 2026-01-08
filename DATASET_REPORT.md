# Dataset Analysis Report

## Overview
- **Source**: `data/raw/retail.csv`
- **Rows**: 12,575
- **Columns**: 11

## Column Statistics

| Column Name | Type | Missing Count | Missing % | Cardinality | Notes |
|---|---|---|---|---|---|
| `Transaction ID` | Object | 0 | 0% | 12,575 | Unique ID, drop for training. |
| `Customer ID` | Object | 0 | 0% | 25 | High recurrence. Potential grouping key. |
| `Category` | Object | 0 | 0% | 8 | Low cardinality categorical. |
| `Item` | Object | 1,213 | ~9.6% | 200 | Medium cardinality. |
| `Price Per Unit` | Float | 609 | ~4.8% | - | Numeric. |
| `Quantity` | Float | 604 | ~4.8% | - | Numeric. |
| `Total Spent` | Float | 604 | ~4.8% | - | Numeric. Target (Regression). |
| `Payment Method` | Object | 0 | 0% | 3 | Protected Attribute Candidate. |
| `Location` | Object | 0 | 0% | 2 | Protected Attribute Candidate (Online vs In-store). |
| `Transaction Date` | Object | 0 | 0% | 1,114 | Date (2020-2024). Needs parsing. |
| `Discount Applied` | Object | 4,199 | ~33.4% | 2 | Target (Classification). High missingness. |

## Key Findings

1.  **Leakage Alert**: `Total Spent` has a correlation of **1.0** with `Price Per Unit * Quantity`.
    - *Implication*: If predicting `Total Spent`, `Quantity` **must** be excluded from features to avoid data leakage.
    - *Recommendation*: Use `Discount Applied` as the primary target for demonstration to ensure a meaningful modeling task.

2.  **Missing Values**:
    - `Discount Applied` is missing in 33% of rows. Strategies: Drop rows, impute (treat as separate category 'Unknown'), or model prediction.
    - `Item` is missing in ~10%.

3.  **Protected Attributes**:
    - `Location` (Online/In-store) is a suitable binary protected attribute for fairness evaluation.
    - `Payment Method` is also viable but multi-class.

## Configuration Recommendations

### Target: `Discount_Applied` (Classification)
- **Type**: Binary Classification
- **Positive Label**: `True`
- **Preprocessing**:
    - Drop rows where `Discount_Applied` is missing OR Impute.
    - Features: `Category`, `Price_Per_Unit`, `Quantity`, `Total_Spent`, `Location`, `Payment_Method`, `Time` features.
    - **Note**: `Total_Spent` can be a feature here.

### Target: `Total_Spent` (Regression)
- **Type**: Regression
- **Preprocessing**:
    - **CRITICAL**: Drop `Quantity` from features.
    - Drop rows where `Total_Spent` is missing.

### Splitting
- Dataset spans multiple years. **Time-based splitting** is recommended to simulate real-world deployment (train on past, predict future).
