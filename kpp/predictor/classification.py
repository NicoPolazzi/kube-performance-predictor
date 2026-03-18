import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from kpp.config import ClassificationConfig, PredictorConfig
from kpp.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_TARGET_COLUMNS = ["Response Time (s)", "Throughput (req/s)", "CPU Usage"]


def cpu_to_label(cpu_pct_values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Converts CPU percentage values to class labels (0=good, 1=danger, 2=bottleneck)."""
    labels = np.zeros(len(cpu_pct_values), dtype=np.int64)
    labels[cpu_pct_values >= thresholds[0]] = 1
    labels[cpu_pct_values >= thresholds[1]] = 2
    return labels


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """Computes per-class and weighted precision/recall/F1.

    Returns {class_name: {"precision": float, "recall": float, "f1": float}, "weighted": {...}}.
    Weighted averages each class's metric by its support (number of true samples), so imbalanced
    classes contribute proportionally rather than equally.
    """
    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0.0
    )
    metrics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(class_names):
        metrics[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
        }
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0.0
    )
    metrics["weighted"] = {
        "precision": float(w_p),
        "recall": float(w_r),
        "f1": float(w_f1),
    }
    return metrics


def regression_to_classes(
    cpu_predictions: np.ndarray,
    cpu_requests: np.ndarray,
    replicas: np.ndarray,
    thresholds: list[float],
) -> np.ndarray:
    """Converts regression CPU predictions to class labels via percentage thresholds."""
    cpu_pct = cpu_predictions / (cpu_requests * replicas) * 100
    return cpu_to_label(cpu_pct, thresholds)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    service_name: str,
    title_suffix: str,
    output_dir: Path,
) -> None:
    """Saves a confusion matrix heatmap for a single service."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=labels,
        yticks=labels,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=f"{service_name} — {title_suffix}",
    )

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{service_name}_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(file_path)
    plt.close()
    logger.info(f"Saved confusion matrix for {service_name} at {file_path}")


def generate_classification_table(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    title: str = "Classification Metrics",
) -> None:
    """Prints and saves a Markdown table with weighted P/R/F1 per service."""
    if not all_metrics:
        logger.warning("No classification metrics to display.")
        return

    services = sorted(all_metrics.keys())
    service_col_width = max(len("Microservice"), max(len(s) for s in services), len("**Mean**"))

    col_headers = ["P", "R", "F1"]

    col_width = 10
    header_row = f"| {'Microservice':<{service_col_width}} |"
    for h in col_headers:
        header_row += f" {h:>{col_width}} |"

    sep_row = f"|{'-' * (service_col_width + 2)}|"
    for _ in col_headers:
        sep_row += f"{'-' * (col_width + 2)}|"

    lines = [f"\n### {title}\n", header_row, sep_row]

    col_sums: dict[str, float] = {h: 0.0 for h in col_headers}
    col_counts: dict[str, int] = {h: 0 for h in col_headers}

    for service in services:
        macro = all_metrics[service].get("weighted", {})
        mp = macro.get("precision", float("nan"))
        mr = macro.get("recall", float("nan"))
        mf1 = macro.get("f1", float("nan"))
        row = f"| {service:<{service_col_width}} |"
        row += f" {mp:>{col_width}.4f} |"
        row += f" {mr:>{col_width}.4f} |"
        row += f" {mf1:>{col_width}.4f} |"
        for key, val in [("P", mp), ("R", mr), ("F1", mf1)]:
            if not np.isnan(val):
                col_sums[key] += val
                col_counts[key] += 1
        lines.append(row)

    mean_row = f"| {'**Mean**':<{service_col_width}} |"
    for h in col_headers:
        if col_counts[h] > 0:
            mean_val = col_sums[h] / col_counts[h]
            mean_row += f" {mean_val:>{col_width}.4f} |"
        else:
            mean_row += f" {'nan':>{col_width}} |"
    lines.append(mean_row)

    table_text = "\n".join(lines)
    print(table_text)

    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{title.lower().replace(' ', '_')}.md"
    md_path.write_text(table_text + "\n")
    logger.info(f"Classification table saved to {md_path}")


def validate_csv_path(csv_path: str, description: str = "CSV data file") -> None:
    """Raises FileNotFoundError if the given CSV path does not exist."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{description} not found: '{csv_path}'. Place your collected data file at this path."
        )


def _run_classification_experiment(
    config: "PredictorConfig",
    cls_config: "ClassificationConfig",
    csv_path: str,
    overload_csv_path: str | None,
    split_strategy: str,
    output_dir: Path,
    table_title: str,
) -> None:
    from kpp.predictor.classifier import ClassificationModel, evaluate_classifier, train_classifier
    from kpp.predictor.pipeline import PerformanceDataPipeline

    pipeline = PerformanceDataPipeline(DEFAULT_TARGET_COLUMNS)
    datasets = pipeline.run_classification(
        csv_path,
        thresholds=cls_config.thresholds,
        train_ratio=config.pipeline.train_ratio,
        split_strategy=split_strategy,
        test_csv_path=overload_csv_path,
    )

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for service_name, data_split in sorted(datasets.items()):
        logger.info(f"--- Service: {service_name} ---")

        train_loader = DataLoader(
            data_split["train"], batch_size=config.training.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            data_split["test"], batch_size=config.training.batch_size, shuffle=False, num_workers=0
        )

        model = ClassificationModel(
            input_size=data_split["train"].tensors[0].shape[1],
            num_classes=cls_config.num_classes,
            hidden_size=config.model.hidden_size,
            hidden_size_2=config.model.hidden_size_2,
            head_hidden_size=config.model.head_hidden_size,
            dropout=config.model.dropout,
        )
        train_classifier(
            config, service_name, model, train_loader, test_loader,
            epochs=config.training.epochs, learning_rate=config.training.learning_rate,
            label_smoothing=cls_config.label_smoothing,
        )

        preds, labels = evaluate_classifier(model, test_loader)
        all_metrics[service_name] = compute_classification_metrics(
            labels, preds, cls_config.class_names
        )
        plot_confusion_matrix(
            labels, preds, cls_config.class_names, service_name, "Classifier",
            output_dir / "confusion",
        )

    generate_classification_table(all_metrics, output_dir, title=table_title)


def _extract_input_features(
    test_loader: DataLoader,
    input_columns: list[str],
    feature_names: list[str],
    scaler: "StandardScaler",
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cpu_requests, replicas) extracted from test_loader and inverse-transformed.

    Uses the dummy-array approach to invert only the two target columns while leaving
    the rest at zero, matching the pattern used in evaluate() in model.py.
    """
    from kpp.predictor.pipeline import PerformanceDataPipeline

    cpu_req_col = PerformanceDataPipeline.CPU_REQUEST_COL
    replicas_col = PerformanceDataPipeline.REPLICAS_COL

    cpu_req_idx_input = input_columns.index(cpu_req_col)
    replicas_idx_input = input_columns.index(replicas_col)
    cpu_req_idx_feat = feature_names.index(cpu_req_col)
    replicas_idx_feat = feature_names.index(replicas_col)
    num_features = len(feature_names)

    all_x: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            all_x.append(batch_x.cpu().numpy())

    x_array = np.concatenate(all_x, axis=0)

    dummy = np.zeros((len(x_array), num_features))
    dummy[:, cpu_req_idx_feat] = x_array[:, cpu_req_idx_input]
    dummy[:, replicas_idx_feat] = x_array[:, replicas_idx_input]

    inv = scaler.inverse_transform(dummy)
    return inv[:, cpu_req_idx_feat], inv[:, replicas_idx_feat]


def _run_regression_classification_experiment(
    config: "PredictorConfig",
    cls_config: "ClassificationConfig",
    csv_path: str,
    overload_csv_path: str | None,
    split_strategy: str,
    output_dir: Path,
    table_title: str,
) -> None:
    from kpp.predictor.model import PerformanceModel, evaluate, train_model
    from kpp.predictor.pipeline import PerformanceDataPipeline

    pipeline = PerformanceDataPipeline(DEFAULT_TARGET_COLUMNS)

    if split_strategy == "stratified":
        if overload_csv_path is None:
            raise ValueError("overload_csv_path is required for stratified split")
        datasets = pipeline.run_stratified_regression(
            csv_path,
            test_csv_path=overload_csv_path,
            thresholds=cls_config.thresholds,
            train_ratio=config.pipeline.train_ratio,
        )
    else:
        datasets = pipeline.run(
            csv_path,
            train_ratio=config.pipeline.train_ratio,
            split_strategy=split_strategy,
            test_csv_path=overload_csv_path,
        )

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for service_name, data_split in sorted(datasets.items()):
        logger.info(f"--- Service: {service_name} ---")

        train_loader = DataLoader(
            data_split["train"], batch_size=config.training.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            data_split["test"], batch_size=config.training.batch_size, shuffle=False, num_workers=0
        )

        input_size = data_split["train"].tensors[0].shape[1]
        output_size = data_split["train"].tensors[1].shape[1]

        model = PerformanceModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=config.model.hidden_size,
            hidden_size_2=config.model.hidden_size_2,
            head_hidden_size=config.model.head_hidden_size,
            dropout=config.model.dropout,
        )
        train_model(
            config, service_name, model, train_loader, test_loader,
            epochs=config.training.epochs, learning_rate=config.training.learning_rate,
        )

        real_predictions, real_targets, _ = evaluate(
            model=model,
            test_loader=test_loader,
            scaler=pipeline.scalers[service_name],
            target_columns=DEFAULT_TARGET_COLUMNS,
            feature_names=pipeline.feature_names,
            x_feature_names=pipeline.input_columns,
            log_transform_columns=list(PerformanceDataPipeline.LOG_TRANSFORM_COLUMNS),
        )

        cpu_feat_idx = pipeline.feature_names.index("CPU Usage")
        pred_cpu = real_predictions[:, cpu_feat_idx]
        true_cpu = real_targets[:, cpu_feat_idx]

        cpu_requests, replicas = _extract_input_features(
            test_loader,
            input_columns=pipeline.input_columns,
            feature_names=pipeline.feature_names,
            scaler=pipeline.scalers[service_name],
        )

        pred_classes = regression_to_classes(pred_cpu, cpu_requests, replicas, cls_config.thresholds)
        true_classes = regression_to_classes(true_cpu, cpu_requests, replicas, cls_config.thresholds)

        all_metrics[service_name] = compute_classification_metrics(
            true_classes, pred_classes, cls_config.class_names
        )
        plot_confusion_matrix(
            true_classes, pred_classes, cls_config.class_names, service_name, "Regression",
            output_dir / "confusion",
        )

    generate_classification_table(all_metrics, output_dir, title=table_title)


def main() -> None:  # pragma: no cover
    setup_logging("predictor")
    config = PredictorConfig.from_yaml()
    cls_config = config.classification or ClassificationConfig()

    csv_path = "datasets/performance_results_normal.csv"
    validate_csv_path(csv_path)

    overload_csv_path = "datasets/performance_results_overload.csv"
    validate_csv_path(overload_csv_path, description="Overload CSV data file")

    torch.manual_seed(42)

    # Experiment 1: extrapolation — train on normal load only, test on overload
    logger.info("=== Experiment: Extrapolation (normal → train, overload → test) ===")
    _run_classification_experiment(
        config=config,
        cls_config=cls_config,
        csv_path=csv_path,
        overload_csv_path=overload_csv_path,
        split_strategy="extrapolation",
        output_dir=Path("results") / "classification" / "extrapolation",
        table_title="Classifier — Extrapolation",
    )

    # Experiment 2: stratified split — combine both datasets, hold out 10% per class
    # This guarantees all three classes appear in the test set, unlike the middle-user-count
    # holdout ("merged") which selects normal-range user counts where bottleneck never occurs.
    logger.info("=== Experiment: Stratified (normal + overload, 10% holdout per class) ===")
    torch.manual_seed(42)
    _run_classification_experiment(
        config=config,
        cls_config=cls_config,
        csv_path=csv_path,
        overload_csv_path=overload_csv_path,
        split_strategy="stratified",
        output_dir=Path("results") / "classification" / "stratified",
        table_title="Classifier — Stratified",
    )

    # Experiment 3: interpolation split — train and test on normal dataset only,
    # holding out the middle user-count value(s).
    logger.info("=== Experiment: Interpolation (normal dataset only) ===")
    torch.manual_seed(42)
    _run_classification_experiment(
        config=config,
        cls_config=cls_config,
        csv_path=csv_path,
        overload_csv_path=None,
        split_strategy="interpolation",
        output_dir=Path("results") / "classification" / "interpolation",
        table_title="Classifier — Interpolation",
    )

    # Regression as classifier experiments — train regression model, convert CPU predictions
    # to classes via thresholds, compare against classifier.
    for strategy, test_path in [
        ("extrapolation", overload_csv_path),
        ("stratified", overload_csv_path),
        ("interpolation", None),
        ("merged", overload_csv_path),
    ]:
        logger.info(f"=== Experiment: Regression as Classifier — {strategy.title()} ===")
        torch.manual_seed(42)
        _run_regression_classification_experiment(
            config=config,
            cls_config=cls_config,
            csv_path=csv_path,
            overload_csv_path=test_path,
            split_strategy=strategy,
            output_dir=Path("results") / "classification" / f"regression_{strategy}",
            table_title=f"Regression as Classifier — {strategy.title()}",
        )


if __name__ == "__main__":
    main()
