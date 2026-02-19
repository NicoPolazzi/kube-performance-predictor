import json
from pathlib import Path

import numpy as np


def generate_rmse_table(models_dir="models"):
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"Error: The directory '{models_dir}' does not exist.")
        return

    results = []

    for config_file in models_path.glob("config_*.json"):
        with open(config_file, "r") as f:
            try:
                data = json.load(f)

                # Extract the necessary data
                service = data.get("service", config_file.stem.replace("config_", ""))
                best_mse = data.get("best_test_loss", None)

                if best_mse is not None:
                    best_rmse = np.sqrt(best_mse)
                    error_percentage = best_rmse * 100
                    results.append((service, best_rmse, error_percentage))
                else:
                    print(f"Warning: No 'best_test_loss' found in {config_file.name}")

            except json.JSONDecodeError:
                print(f"Error reading JSON from {config_file.name}")

    if not results:
        print("No valid results found to generate a table.")
        return

    results.sort(key=lambda x: x[0])
    service_col_width = max(len("Microservice"), max(len(s) for s, _, _ in results))

    # Print the Markdown Table
    print("\n### Final Model Performance by Microservice\n")
    print(f"| {'Microservice':<{service_col_width}} | Best Test RMSE | Error Margin |")
    print(f"|{'-' * (service_col_width + 2)}|----------------|--------------|")

    for service, rmse, pct in results:
        print(f"| {service:<{service_col_width}} | {rmse:.4f}         | {pct:>5.2f}%       |")


if __name__ == "__main__":
    generate_rmse_table()
