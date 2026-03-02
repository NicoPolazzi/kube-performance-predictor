import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from kpp.config import PredictorConfig
from kpp.logging_config import setup_logging
from kpp.predictor.model import PerformanceModel, evaluate, train_model
from kpp.predictor.pipeline import PerformanceDataPipeline
from torch.utils.data import DataLoader

logger = logging.getLogger("predictor")


def plot(
    real_predictions: np.ndarray,
    real_targets: np.ndarray,
    user_counts_int: np.ndarray,
    target_columns: list[str],
    target_indices: list[int],
    service_name: str,
) -> None:
    """Creates and saves the predictions plot to plots/{service_name}_predictions.png."""
    unique_users = sorted(np.unique(user_counts_int))

    num_targets = len(target_columns)
    _, axes = plt.subplots(num_targets, 1, figsize=(12, 4 * num_targets), sharex=True)

    if num_targets == 1:
        axes = [axes]

    for i, col_name in enumerate(target_columns):
        ax = axes[i]
        target_idx = target_indices[i]

        mean_pred_list: list[float] = []
        std_pred_list: list[float] = []
        mean_true_list: list[float] = []
        std_true_list: list[float] = []

        for u in unique_users:
            mask = user_counts_int == u
            mean_pred_list.append(real_predictions[mask, target_idx].mean())
            std_pred_list.append(real_predictions[mask, target_idx].std())
            mean_true_list.append(real_targets[mask, target_idx].mean())
            std_true_list.append(real_targets[mask, target_idx].std())

        x = np.array(unique_users)
        mean_pred = np.array(mean_pred_list)
        std_pred = np.array(std_pred_list)
        mean_true = np.array(mean_true_list)
        std_true = np.array(std_true_list)

        ax.plot(x, mean_true, label="Ground Truth", color="blue", linewidth=2)
        ax.fill_between(x, mean_true - std_true, mean_true + std_true, color="blue", alpha=0.15)

        ax.plot(x, mean_pred, label="Linear", color="green", linewidth=2)
        ax.fill_between(x, mean_pred - std_pred, mean_pred + std_pred, color="green", alpha=0.15)

        rmse = np.sqrt(np.mean((real_predictions[:, target_idx] - real_targets[:, target_idx]) ** 2))
        ax.set_title(f"{service_name} - {col_name}  |  RMSE: {rmse:.4f}")
        ax.set_ylabel(col_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.xlabel("Concurrent Users")
    plt.tight_layout()

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{service_name}_predictions.png"
    plt.savefig(file_path)
    plt.close()

    logger.info(f"Saved plot for {service_name} at {file_path}")


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


def main() -> None:
    setup_logging("predictor")
    config = PredictorConfig.from_yaml()

    csv_path = "dataset/performance_results_medium.csv"
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV data file not found: '{csv_path}'. Place your collected data file at this path."
        )

    target_cols = [
        "Response Time (s)",
        "Throughput (req/s)",
        "CPU Usage",
    ]

    pipeline = PerformanceDataPipeline(config.pipeline.sequence_length, target_cols)
    datasets = pipeline.run(
        csv_path,
        train_lower_percentile=config.pipeline.train_lower_percentile,
        train_upper_percentile=config.pipeline.train_upper_percentile,
    )

    # Derive feature list from the pipeline's schema, excluding non-numeric identifier columns.
    all_features = [
        col
        for col in PerformanceDataPipeline.REQUIRED_COLUMNS
        if col not in ("Timestamp", "Service")
    ] + PerformanceDataPipeline.DELTA_COLUMNS

    for service_name, data_split in datasets.items():
        logger.info(f"--- Service: {service_name} ---")

        train_dataset = data_split["train"]
        test_dataset = data_split["test"]

        logger.info(f"Train Shape: {train_dataset.tensors[0].shape} (Samples, Window, Features)")
        logger.info(f"Test Shape:  {test_dataset.tensors[0].shape}")

        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        output_size = train_dataset.tensors[1].shape[1]
        flat_input_size = train_dataset.tensors[0].shape[1] * train_dataset.tensors[0].shape[2]

        logger.info(f"Training PerformanceModel for {service_name}...")
        model = PerformanceModel(
            input_size=flat_input_size,
            output_size=output_size,
            hidden_size=config.model.hidden_size,
        )
        train_model(
            config,
            service_name,
            model,
            train_loader,
            test_loader,
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
        )

        logger.info(f"Loading the best saved model weights for {service_name}...")
        model_path = Path("models") / f"{service_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Training may not have saved a checkpoint "
                f"(no epoch improved on the historical best)."
            )
        model.load_state_dict(torch.load(model_path, weights_only=True))

        logger.info(f"Evaluating and plotting {service_name}...")

        service_scaler = pipeline.scalers.get(service_name)
        if service_scaler is None:
            logger.warning(f"No scaler found for {service_name}. Skipping evaluation.")
            continue

        real_predictions, real_targets, user_counts_int = evaluate(
            model=model,
            test_loader=test_loader,
            scaler=service_scaler,
            target_columns=target_cols,
            feature_names=all_features,
        )

        target_indices = [all_features.index(col) for col in target_cols]

        plot(
            real_predictions=real_predictions,
            real_targets=real_targets,
            user_counts_int=user_counts_int,
            target_columns=target_cols,
            target_indices=target_indices,
            service_name=service_name,
        )

    generate_rmse_table()


if __name__ == "__main__":
    main()
