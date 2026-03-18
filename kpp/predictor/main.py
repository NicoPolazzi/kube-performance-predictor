import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from kpp.config import PredictorConfig
from kpp.logging_config import setup_logging
from kpp.predictor.classification import DEFAULT_TARGET_COLUMNS, validate_csv_path
from kpp.predictor.model import PerformanceModel, evaluate, train_model
from kpp.predictor.pipeline import PerformanceDataPipeline

logger = logging.getLogger("predictor")


def _shorten_column_name(col: str) -> str:
    """Strips unit suffixes and replaces spaces with underscores for table headers."""
    return _shorten_column_name(col)


def compute_metrics(
    real_predictions: np.ndarray,
    real_targets: np.ndarray,
    target_columns: list[str],
    target_indices: list[int],
) -> dict[str, dict[str, float]]:
    """
    Returns {col_name: {"MAE": float, "MAPE": float}} for each target column.
    MAPE skips samples where |true| <= 1e-6 to avoid division by zero.
    """
    metrics: dict[str, dict[str, float]] = {}
    for col_name, idx in zip(target_columns, target_indices, strict=True):
        preds = real_predictions[:, idx]
        targets = real_targets[:, idx]
        mae = float(np.mean(np.abs(preds - targets)))
        threshold = max(1e-6, 0.01 * np.mean(np.abs(targets)))
        valid_mask = np.abs(targets) > threshold
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
    output_dir: Path,
) -> None:
    """Creates and saves the predictions plot to output_dir/predictions/{service_name}_predictions.png."""
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

        ax.plot(x, mean_pred, label="Prediction", color="green", linewidth=2)
        ax.fill_between(x, mean_pred - std_pred, mean_pred + std_pred, color="green", alpha=0.15)

        col_metrics = metrics.get(col_name, {})
        mae = col_metrics.get("MAE", float("nan"))
        mape = col_metrics.get("MAPE", float("nan"))
        ax.set_title(f"{service_name} - {col_name}  |  MAE: {mae:.6f}  |  MAPE: {mape:.2f}%")
        ax.set_ylabel(col_name)
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.xlabel("Concurrent Users")
    plt.tight_layout()

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    file_path = predictions_dir / f"{service_name}_predictions.png"
    plt.savefig(file_path)
    plt.close()

    logger.info(f"Saved plot for {service_name} at {file_path}")


def generate_metrics_table(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    target_columns: list[str],
    output_dir: Path,
) -> None:
    """Prints a Markdown table with rows=services and columns=MAE+MAPE per target metric.

    Includes a Mean row at the bottom. Also writes the table to output_dir/metrics_table.md.
    """
    if not all_metrics:
        logger.warning("No metrics to display.")
        return

    services = sorted(all_metrics.keys())
    service_col_width = max(len("Microservice"), max(len(s) for s in services), len("**Mean**"))

    # Build header: one MAE + MAPE pair per target column
    col_headers = []
    for col in target_columns:
        short = _shorten_column_name(col)
        col_headers.append(f"{short}_MAE")
        col_headers.append(f"{short}_MAPE%")

    col_width = 12
    header_row = f"| {'Microservice':<{service_col_width}} |"
    for h in col_headers:
        header_row += f" {h:>{col_width}} |"

    sep_row = f"|{'-' * (service_col_width + 2)}|"
    for _ in col_headers:
        sep_row += f"{'-' * (col_width + 2)}|"

    lines = ["\n### Prediction Metrics by Microservice\n", header_row, sep_row]

    # Accumulators for mean computation
    col_sums: dict[str, float] = {h: 0.0 for h in col_headers}
    col_counts: dict[str, int] = {h: 0 for h in col_headers}

    for service in services:
        row = f"| {service:<{service_col_width}} |"
        for col in target_columns:
            col_metrics = all_metrics[service].get(col, {})
            mae = col_metrics.get("MAE", float("nan"))
            mape = col_metrics.get("MAPE", float("nan"))
            row += f" {mae:>{col_width}.6f} |"
            row += f" {mape:>{col_width}.2f} |"

            short = _shorten_column_name(col)
            if not np.isnan(mae):
                col_sums[f"{short}_MAE"] += mae
                col_counts[f"{short}_MAE"] += 1
            if not np.isnan(mape):
                col_sums[f"{short}_MAPE%"] += mape
                col_counts[f"{short}_MAPE%"] += 1
        lines.append(row)

    # Mean row
    mean_row = f"| {'**Mean**':<{service_col_width}} |"
    for h in col_headers:
        if col_counts[h] > 0:
            mean_val = col_sums[h] / col_counts[h]
            mean_row += (
                f" {mean_val:>{col_width}.6f} |" if "MAE" in h else f" {mean_val:>{col_width}.2f} |"
            )
        else:
            mean_row += f" {'nan':>{col_width}} |"
    lines.append(mean_row)

    table_text = "\n".join(lines)
    print(table_text)

    # Write Markdown version
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "metrics_table.md"
    md_path.write_text(table_text + "\n")
    logger.info(f"Markdown metrics table saved to {md_path}")


def plot_losses(
    train_losses: list[float],
    test_losses: list[float],
    service_name: str,
    output_dir: Path,
) -> None:
    """Plots train & test MAE loss curves and saves to output_dir/losses/{service_name}_losses.png."""
    epochs = range(1, len(train_losses) + 1)
    train_mae = np.array(train_losses)
    test_mae = np.array(test_losses)

    _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_mae, label="Train MAE", linewidth=2)
    ax.plot(epochs, test_mae, label="Test MAE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title(f"{service_name} — Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    losses_dir = output_dir / "losses"
    losses_dir.mkdir(parents=True, exist_ok=True)
    file_path = losses_dir / f"{service_name}_losses.png"
    plt.savefig(file_path)
    plt.close()

    logger.info(f"Saved loss plot for {service_name} at {file_path}")


def main() -> None:
    setup_logging("predictor")
    config = PredictorConfig.from_yaml()

    csv_path = "datasets/performance_results_normal.csv"
    validate_csv_path(csv_path)

    test_csv_path = (
        "datasets/performance_results_overload.csv"
        if config.pipeline.split_strategy in ("extrapolation", "merged")
        else None
    )
    if test_csv_path is not None:
        validate_csv_path(test_csv_path, description="Overload CSV data file")

    target_cols = DEFAULT_TARGET_COLUMNS

    pipeline = PerformanceDataPipeline(target_cols)
    datasets = pipeline.run(
        csv_path,
        train_ratio=config.pipeline.train_ratio,
        split_strategy=config.pipeline.split_strategy,
        test_csv_path=test_csv_path,
    )

    all_features = pipeline.feature_names

    output_dir = Path("results") / config.pipeline.split_strategy

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    torch.manual_seed(42)

    for service_name, data_split in datasets.items():
        logger.info(f"--- Service: {service_name} ---")

        train_dataset = data_split["train"]
        test_dataset = data_split["test"]

        logger.info(f"Train Shape: {train_dataset.tensors[0].shape} (Samples, Features)")
        logger.info(f"Test Shape:  {test_dataset.tensors[0].shape}")

        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0
        )

        output_size = train_dataset.tensors[1].shape[1]
        input_size = train_dataset.tensors[0].shape[1]

        logger.info(f"Training PerformanceModel for {service_name}...")
        model = PerformanceModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=config.model.hidden_size,
            hidden_size_2=config.model.hidden_size_2,
            head_hidden_size=config.model.head_hidden_size,
            dropout=config.model.dropout,
        )
        _, train_losses, test_losses = train_model(
            config,
            service_name,
            model,
            train_loader,
            test_loader,
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
        )

        plot_losses(train_losses, test_losses, service_name, output_dir)

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
            x_feature_names=pipeline.input_columns,
            log_transform_columns=PerformanceDataPipeline.LOG_TRANSFORM_COLUMNS,
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
            output_dir=output_dir,
        )

    generate_metrics_table(all_metrics, target_cols, output_dir)


if __name__ == "__main__":
    main()
