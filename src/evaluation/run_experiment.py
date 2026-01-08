import argparse
import yaml
import logging
import sys
import os
import subprocess
from src.evaluation.experiment import ExperimentRunner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Experiment Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="runs/latest", help="Output directory")
    parser.add_argument("--auto-select", action="store_true", help="Run selection automatically after experiment")
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Starting experiment...")
    runner = ExperimentRunner(config)
    runner.run()

    logger.info(f"Saving results to {args.output}")
    runner.save_results(args.output)

    if args.auto_select:
        logger.info("Running auto-selection...")
        # Call selection script
        # We assume src.evaluation.run_selection is available
        results_path = os.path.join(args.output, "results.jsonl")
        select_output = os.path.join(args.output, "selection")

        cmd = [
            sys.executable, "-m", "src.evaluation.run_selection",
            "--results", results_path,
            "--output", select_output,
            "--config", args.config # Use same config for pareto/constraints
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Selection artifacts saved to {select_output}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Selection failed: {e}")

    logger.info("Done.")

if __name__ == "__main__":
    main()
