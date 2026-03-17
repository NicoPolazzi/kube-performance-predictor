import numpy as np

from kpp.predictor.classification import (
    compute_classification_metrics,
    cpu_to_label,
    regression_to_classes,
)


def test_cpu_to_label_assigns_good_for_low_values():
    values = np.array([0.0, 10.0, 39.9])
    labels = cpu_to_label(values, thresholds=[40.0, 60.0])
    np.testing.assert_array_equal(labels, [0, 0, 0])


def test_cpu_to_label_assigns_danger_for_mid_values():
    values = np.array([40.0, 50.0, 59.9])
    labels = cpu_to_label(values, thresholds=[40.0, 60.0])
    np.testing.assert_array_equal(labels, [1, 1, 1])


def test_cpu_to_label_assigns_bottleneck_for_high_values():
    values = np.array([60.0, 80.0, 100.0])
    labels = cpu_to_label(values, thresholds=[40.0, 60.0])
    np.testing.assert_array_equal(labels, [2, 2, 2])


def test_compute_classification_metrics_returns_per_class_and_macro():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    class_names = ["good", "danger", "bottleneck"]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)

    assert set(metrics.keys()) == {"good", "danger", "bottleneck", "macro"}
    for name in class_names:
        assert metrics[name]["precision"] == 1.0
        assert metrics[name]["recall"] == 1.0
        assert metrics[name]["f1"] == 1.0
    assert metrics["macro"]["f1"] == 1.0


def test_regression_to_classes_thresholds_predictions():
    # cpu_pct = cpu_pred / (cpu_request * replicas) * 100
    # 0.02 / (0.1 * 1) * 100 = 20% → good (0)
    # 0.05 / (0.1 * 1) * 100 = 50% → danger (1)
    # 0.08 / (0.1 * 1) * 100 = 80% → bottleneck (2)
    cpu_preds = np.array([0.02, 0.05, 0.08])
    cpu_requests = np.array([0.1, 0.1, 0.1])
    replicas = np.array([1.0, 1.0, 1.0])
    thresholds = [40.0, 60.0]

    labels = regression_to_classes(cpu_preds, cpu_requests, replicas, thresholds)
    np.testing.assert_array_equal(labels, [0, 1, 2])
