import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from kpp.config import PredictorConfig
from kpp.logging_config import setup_logging
from kpp.predictor.model import PerformanceModel, evaluate, train_model
from kpp.predictor.pipeline import PerformanceDataPipeline

logger = logging.getLogger("predictor")


def compute_metrics(
    real_predictions: np.ndarray,
    real_targets: np.ndarray,
    target_columns: list[str],
    target_indices: list[int],
) -> dict[str, dict[str, float]]:
    """
    Returns {col_name: {"MAE": float, "MAPE": float}} for each target column.
    MAPE skips samples where |true| <= 1e-10 to avoid division by zero.
    """
    metrics: dict[str, dict[str, float]] = {}
    for col_name, idx in zip(target_columns, target_indices, strict=True):
        preds = real_predictions[:, idx]
        targets = real_targets[:, idx]
        mae = float(np.mean(np.abs(preds - targets)))
        valid_mask = np.abs(targets) > 1e-10
        if valid_mask.any():
            mape = float(
                np.mean(np.abs((preds[valid_mask] - targets[valid_mask]) / targets[valid_mask]))
                * 100
            )
        else:
            mape = float("nan")
        metrics[col_name] = {"MAE": mae, "MAPE": mape}
    return metrics


def plot(
    real_predictions: np.ndarray,
    real_targets: np.ndarray,
    user_counts_int: np.ndarray,
    target_columns: list[str],
    target_indices: list[int],
    service_name: str,
    metrics: dict[str, dict[str, float]],
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

        col_metrics = metrics.get(col_name, {})
        mae = col_metrics.get("MAE", float("nan"))
        mape = col_metrics.get("MAPE", float("nan"))
        ax.set_title(f"{service_name} - {col_name}  |  MAE: {mae:.4f}  |  MAPE: {mape:.2f}%")
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


def generate_metrics_table(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    target_columns: list[str],
) -> None:
    """Prints a table with rows=services and columns=MAE+MAPE per target metric."""
    if not all_metrics:
        logger.warning("No metrics to display.")
        return

    services = sorted(all_metrics.keys())
    service_col_width = max(len("Microservice"), max(len(s) for s in services))

    # Build header: one MAE + MAPE pair per target column
    col_headers = []
    for col in target_columns:
        short = col.replace(" (s)", "").replace(" (req/s)", "").replace(" ", "_")
        col_headers.append(f"{short}_MAE")
        col_headers.append(f"{short}_MAPE%")

    col_width = 12
    header_row = f"| {'Microservice':<{service_col_width}} |"
    for h in col_headers:
        header_row += f" {h:>{col_width}} |"

    sep_row = f"|{'-' * (service_col_width + 2)}|"
    for _ in col_headers:
        sep_row += f"{'-' * (col_width + 2)}|"

    print("\n### Prediction Metrics by Microservice\n")
    print(header_row)
    print(sep_row)

    for service in services:
        row = f"| {service:<{service_col_width}} |"
        for col in target_columns:
            col_metrics = all_metrics[service].get(col, {})
            mae = col_metrics.get("MAE", float("nan"))
            mape = col_metrics.get("MAPE", float("nan"))
            row += f" {mae:>{col_width}.4f} |"
            row += f" {mape:>{col_width}.2f} |"
        print(row)


def main() -> None:
    setup_logging("predictor")
    config = PredictorConfig.from_yaml()

    csv_path = "dataset/performance_results_normal.csv"
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
    datasets = pipeline.run(csv_path, train_ratio=config.pipeline.train_ratio, split_strategy=config.pipeline.split_strategy)

    # Derive feature list from the pipeline's schema, excluding non-numeric identifier columns.
    all_features = [
        col
        for col in PerformanceDataPipeline.REQUIRED_COLUMNS
        if col not in ("Timestamp", "Service")
    ] + PerformanceDataPipeline.DELTA_COLUMNS

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for service_name, data_split in datasets.items():
        logger.info(f"--- Service: {service_name} ---")

        train_dataset = data_split["train"]
        test_dataset = data_split["test"]

        logger.info(f"Train Shape: {train_dataset.tensors[0].shape} (Samples, Window, Features)")
        logger.info(f"Test Shape:  {test_dataset.tensors[0].shape}")

        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0
        )

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
        service_metrics = compute_metrics(
            real_predictions, real_targets, target_cols, target_indices
        )
        all_metrics[service_name] = service_metrics

        plot(
            real_predictions=real_predictions,
            real_targets=real_targets,
            user_counts_int=user_counts_int,
            target_columns=target_cols,
            target_indices=target_indices,
            service_name=service_name,
            metrics=service_metrics,
        )

    generate_metrics_table(all_metrics, target_cols)


if __name__ == "__main__":
    main()
