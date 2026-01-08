# Retail Data Preprocessing & Experimentation Pipeline

## Overview
This repository contains a reproducible pipeline for preprocessing retail transaction data, training models, and evaluating them on performance, fairness, and efficiency.

## Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run a Single Pipeline
To train a single model and save the artifact:
```bash
python -m src.pipeline.run --config configs/experiment.yaml --output data/processed
```
This saves the fitted pipeline to `data/processed/pipeline.pkl`.

### 2. Run an Experiment
To compare multiple strategies and models:
```bash
python -m src.evaluation.run_experiment --config configs/experiment.yaml --output runs/latest
```
Results (CSV, JSONL) will be saved to `runs/latest`.

## Architecture
- **`src/data`**: Data ingestion and splitting (Random/Time-based).
- **`src/preprocessing`**: Configurable sklearn pipelines (Imputation, Encoding, Scaling).
- **`src/models`**: Factory for model instantiation.
- **`src/evaluation`**: Experiment runner and metrics (Fairness via Fairlearn).
- **`configs/`**: YAML configurations for experiments.

## Configuration
See `configs/experiment.yaml` for an example. You can define:
- Data path and target
- Splitting strategy
- Preprocessing steps (Numerical/Categorical)
- Model families to sweep

## Reports
- `DATASET_REPORT.md`: Analysis of the raw data.
- `EXPERIMENT_REPORT.md`: Summary of initial experiment results.
