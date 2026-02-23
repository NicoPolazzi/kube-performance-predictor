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

    gru_results = _load_results(models_path, "config_*.json")
    baseline_results = _load_results(models_path, "linear_config_*.json")

    _print_table("GRU+Attention Performance by Microservice", gru_results)
    _print_table("LinearBaseline Performance by Microservice", baseline_results)

    if gru_results and baseline_results:
        gru_by_service = {s: rmse for s, rmse, _ in gru_results}
        baseline_by_service = {s: rmse for s, rmse, _ in baseline_results}
        shared = sorted(set(gru_by_service) & set(baseline_by_service))

        if shared:
            service_col_width = max(len("Microservice"), max(len(s) for s in shared))
            print("\n### GRU+Attention vs LinearBaseline Comparison\n")
            print(
                f"| {'Microservice':<{service_col_width}} | GRU+Attn RMSE | Linear RMSE | Winner       |"
            )
            print(
                f"|{'-' * (service_col_width + 2)}|---------------|-------------|--------------|"
            )
            for service in shared:
                gru_rmse = gru_by_service[service]
                lin_rmse = baseline_by_service[service]
                winner = "GRU+Attn" if gru_rmse <= lin_rmse else "Linear"
                print(
                    f"| {service:<{service_col_width}} | {gru_rmse:.4f}        | {lin_rmse:.4f}      | {winner:<12} |"
                )


if __name__ == "__main__":
    setup_logging("generate_table")
    generate_rmse_table()
