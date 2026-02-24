import json
import logging
from pathlib import Path

import numpy as np

from kpp.logging_config import setup_logging

logger = logging.getLogger("predictor.table")


def _load_results(models_path: Path, glob_pattern: str) -> list[tuple[str, float, float]]:
    results = []
    for config_file in models_path.glob(glob_pattern):
        with open(config_file, "r") as f:
            try:
                data = json.load(f)
                service = data.get("service", config_file.stem)
                best_mse = data.get("best_test_loss", None)
                if best_mse is not None:
                    best_rmse = np.sqrt(best_mse)
                    results.append((service, best_rmse, best_rmse * 100))
                else:
                    logger.warning(f"No 'best_test_loss' found in {config_file.name}")
            except json.JSONDecodeError:
                logger.error(f"Error reading JSON from {config_file.name}")
    results.sort(key=lambda x: x[0])
    return results


def _print_table(title: str, results: list[tuple[str, float, float]]) -> None:
    if not results:
        logger.warning(f"No valid results found for: {title}")
        return

    service_col_width = max(len("Microservice"), max(len(s) for s, _, _ in results))

    print(f"\n### {title}\n")
    print(f"| {'Microservice':<{service_col_width}} | Best Test RMSE | Error Margin |")
    print(f"|{'-' * (service_col_width + 2)}|----------------|--------------|")

    for service, rmse, pct in results:
        print(f"| {service:<{service_col_width}} | {rmse:.4f}         | {pct:>5.2f}%       |")


def generate_rmse_table(models_dir: str = "models") -> None:
    models_path = Path(models_dir)

    if not models_path.exists():
        logger.error(f"The directory '{models_dir}' does not exist.")
        return

    results = _load_results(models_path, "config_*.json")
    _print_table("PerformanceModel Results by Microservice", results)


if __name__ == "__main__":
    setup_logging("generate_table")
    generate_rmse_table()
