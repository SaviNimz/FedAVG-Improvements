import argparse
from pathlib import Path
import yaml

from analysis.comparison_runner import run_comparison
from analysis.reporting import save_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline FedAvg with FedAvg+KD"
    )
    parser.add_argument(
        "output",
        help="Directory where results should be saved",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--tag",
        help="Optional identifier for the run",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_path = Path(args.output)
    results = run_comparison(config, base_path, args.tag)

    run_dir = base_path / "results" / results.get("tag", "")
    save_results(results, run_dir)

    print(f"Artifacts stored in: {run_dir}")


if __name__ == "__main__":
    main()
