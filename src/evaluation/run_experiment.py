import argparse
import yaml
import logging
import sys
from src.evaluation.experiment import ExperimentRunner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Experiment Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="runs/latest", help="Output directory")
    args = parser.parse_args()

    logger.info(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Starting experiment...")
    runner = ExperimentRunner(config)
    runner.run()

    logger.info(f"Saving results to {args.output}")
    runner.save_results(args.output)
    logger.info("Done.")

if __name__ == "__main__":
    main()
