import json
from pathlib import Path

import numpy as np


def generate_rmse_table(models_dir="models"):
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"Error: The directory '{models_dir}' does not exist.")
        return

    results = []

    # Iterate through all JSON configuration files in the folder
    for config_file in models_path.glob("config_*.json"):
        with open(config_file, "r") as f:
            try:
                data = json.load(f)

                # Extract the necessary data
                service = data.get("service", config_file.stem.replace("config_", ""))
                best_mse = data.get("best_test_loss", None)

                if best_mse is not None:
                    # Remember: The saved loss is MSE. We must square root it to get RMSE.
                    best_rmse = np.sqrt(best_mse)

                    # Because data is MinMax scaled (0 to 1), RMSE directly translates to a percentage
                    error_percentage = best_rmse * 100

                    results.append((service, best_rmse, error_percentage))
                else:
                    print(f"Warning: No 'best_test_loss' found in {config_file.name}")

            except json.JSONDecodeError:
                print(f"Error reading JSON from {config_file.name}")

    if not results:
        print("No valid results found to generate a table.")
        return

    # Sort the results alphabetically by service name
    results.sort(key=lambda x: x[0])

    # Print the Markdown Table
    print("\n### Final Model Performance by Microservice\n")
    print("| Microservice | Best Test RMSE | Error Margin |")
    print("|--------------|----------------|--------------|")

    for service, rmse, pct in results:
        # Format the numbers to be perfectly aligned and clean
        print(f"| {service:<20} | {rmse:.4f}         | {pct:>5.2f}%       |")


if __name__ == "__main__":
    generate_rmse_table()
