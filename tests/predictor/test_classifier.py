import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from kpp.config import ModelConfig, PipelineConfig, PredictorConfig, SchedulerConfig, TrainingConfig
from kpp.predictor.classifier import ClassificationModel, evaluate_classifier, train_classifier

N_FEATURES = 8
N_SAMPLES = 20
NUM_CLASSES = 3


def _make_cls_loader(n_samples: int, n_features: int, num_classes: int, rng: np.random.Generator) -> DataLoader:
    X = rng.random((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n_samples).astype(np.int64)
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=4,
        num_workers=0,
    )


def test_classification_model_forward_produces_correct_shape():
    model = ClassificationModel(input_size=N_FEATURES, num_classes=NUM_CLASSES)
    x = torch.randn(8, N_FEATURES)
    output = model(x)
    assert output.shape == torch.Size([8, NUM_CLASSES])


def test_train_classifier_restores_best_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(42)
    train_loader = _make_cls_loader(N_SAMPLES, N_FEATURES, NUM_CLASSES, rng)
    test_loader = _make_cls_loader(N_SAMPLES, N_FEATURES, NUM_CLASSES, rng)

    model = ClassificationModel(input_size=N_FEATURES, num_classes=NUM_CLASSES)
    config = PredictorConfig(
        pipeline=PipelineConfig(train_ratio=0.8),
        model=ModelConfig(hidden_size=64, hidden_size_2=32),
        training=TrainingConfig(epochs=1, learning_rate=0.001, batch_size=4, weight_decay=0.003),
        scheduler=SchedulerConfig(factor=0.5, patience=10, min_lr=1e-6),
    )

    best_test_acc, train_losses, test_losses = train_classifier(
        config=config,
        service_name="frontend",
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=1,
    )

    assert model.training is False
    assert isinstance(best_test_acc, float)
    assert 0.0 <= best_test_acc <= 1.0
    assert len(train_losses) == 1
    assert len(test_losses) == 1


def test_evaluate_classifier_returns_integer_classes():
    rng = np.random.default_rng(42)
    test_loader = _make_cls_loader(N_SAMPLES, N_FEATURES, NUM_CLASSES, rng)

    model = ClassificationModel(input_size=N_FEATURES, num_classes=NUM_CLASSES)
    model.eval()

    preds, labels = evaluate_classifier(model, test_loader)

    assert preds.dtype == np.int64 or np.issubdtype(preds.dtype, np.integer)
    assert labels.dtype == np.int64 or np.issubdtype(labels.dtype, np.integer)
    assert len(preds) == N_SAMPLES
    assert len(labels) == N_SAMPLES
    assert set(preds.tolist()).issubset({0, 1, 2})
