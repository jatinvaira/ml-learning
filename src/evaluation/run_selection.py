import argparse
import pandas as pd
import yaml
import os
import sys
from src.evaluation.selection import SelectionEngine

def main():
    parser = argparse.ArgumentParser(description="Run Selection Layer")
    parser.add_argument("--results", type=str, required=True, help="Path to results.jsonl")
    parser.add_argument("--output", type=str, default="selection_output", help="Output directory")
    parser.add_argument("--config", type=str, help="Path to selection config (weights/constraints)")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file {args.results} not found.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    engine = SelectionEngine(args.results)

    # 1. Aggregate
    summary = engine.aggregate_results()
    summary.to_csv(os.path.join(args.output, "summary.csv"), index=False)

    # 2. Pareto Frontier
    # Default objectives if not in config
    objectives = config.get("pareto", {}).get("objectives", {
        "f1_mean": "max",
        "demographic_parity_diff_mean": "min",
        "comp_cost_mean": "min" # Use complexity cost (time + features)
    })

    # Ensure columns exist
    valid_objs = {}
    for k, v in objectives.items():
        if k in summary.columns:
            valid_objs[k] = v
        else:
            # Try to guess column name (e.g. if user said 'f1', map to 'f1_mean')
            if f"{k}_mean" in summary.columns:
                valid_objs[f"{k}_mean"] = v
            else:
                print(f"Warning: Objective {k} not found in summary columns.")

    if valid_objs:
        pareto = engine.compute_pareto_frontier(summary, valid_objs)
        pareto.to_json(os.path.join(args.output, "pareto.jsonl"), orient="records", lines=True)
        pareto.to_csv(os.path.join(args.output, "pareto.csv"), index=False)

    # 3. Constraints
    constraints = config.get("constraints", [])
    if constraints:
        filtered = engine.filter_constraints(summary, constraints)
        filtered.to_json(os.path.join(args.output, "selected_constraints.jsonl"), orient="records", lines=True)

    # 4. Generate Markdown Summary
    md_path = os.path.join(args.output, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Experiment Summary\n\n")

        f.write("## Pareto Optimal Strategies\n")
        f.write("Strategies that are non-dominated on: " + ", ".join([f"{k} ({v})" for k,v in valid_objs.items()]) + "\n\n")
        if valid_objs and not pareto.empty:
            # Select key columns
            cols = ["strategy", "complexity_steps"] + list(valid_objs.keys())
            f.write(pareto[cols].to_markdown(index=False))
        else:
            f.write("No Pareto frontier found (or empty).\n")

        f.write("\n\n## All Strategies (Aggregated)\n")
        # Sort by complexity
        if "comp_cost_mean" in summary.columns:
            summary = summary.sort_values("comp_cost_mean")
        elif "complexity_steps" in summary.columns: # Actually grouping col, so no mean suffix
            summary = summary.sort_values("complexity_steps")

        disp_cols = [c for c in summary.columns if "mean" in c or c in ["strategy", "complexity_steps"]]
        f.write(summary[disp_cols].to_markdown(index=False))

    print(f"Selection complete. Artifacts saved to {args.output}")

if __name__ == "__main__":
    main()
