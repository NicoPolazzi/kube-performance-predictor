from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from kpp.config import ModelConfig, PipelineConfig, PredictorConfig, SchedulerConfig, TrainingConfig
from kpp.predictor.model import PerformanceModel, evaluate, train_model
from kpp.predictor.pipeline import PerformanceDataPipeline

_FIXTURE = Path(__file__).parent.parent / "predictor" / "fixtures" / "small_sample.csv"

# Use fast settings so the e2e test is quick (< 30 s) but still exercises the full path.
_CONFIG = PredictorConfig(
    pipeline=PipelineConfig(sequence_length=3, train_ratio=0.7),
    model=ModelConfig(hidden_size=16, hidden_size_2=8),
    training=TrainingConfig(epochs=20, learning_rate=0.01, batch_size=8, weight_decay=0.001),
    scheduler=SchedulerConfig(factor=0.5, patience=5, min_lr=1e-6),
)

_TARGET_COLS = ["Response Time (s)", "Throughput (req/s)", "CPU Usage"]

# Quality gate: normalized MSE ceiling.  A value >= 0.5 means the model is
# performing no better than a constant predictor, signalling a broken pipeline.
_MSE_CEILING = 0.5


def test_predictor_pipeline_meets_quality_gate(tmp_path, monkeypatch):
    """Full run: pipeline → train → evaluate.  Checks output files and RMSE gate."""
    monkeypatch.chdir(tmp_path)

    pipeline = PerformanceDataPipeline(_CONFIG.pipeline.sequence_length, _TARGET_COLS)
    datasets = pipeline.run(str(_FIXTURE), train_ratio=_CONFIG.pipeline.train_ratio)

    assert datasets, "Pipeline returned no service datasets"

    for service_name, data_split in datasets.items():
        train_dataset = data_split["train"]
        test_dataset = data_split["test"]

        train_loader = DataLoader(
            train_dataset, batch_size=_CONFIG.training.batch_size, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=_CONFIG.training.batch_size, shuffle=False, num_workers=0
        )

        flat_input_size = train_dataset.tensors[0].shape[1] * train_dataset.tensors[0].shape[2]
        output_size = train_dataset.tensors[1].shape[1]

        model = PerformanceModel(
            input_size=flat_input_size,
            output_size=output_size,
            hidden_size=_CONFIG.model.hidden_size,
            hidden_size_2=_CONFIG.model.hidden_size_2,
        )
        best_test_loss = train_model(
            config=_CONFIG,
            service_name=service_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=_CONFIG.training.epochs,
            learning_rate=_CONFIG.training.learning_rate,
        )

        # --- No disk artefacts should be created ---
        assert not (tmp_path / "models").exists(), "models/ directory should not be created"

        # --- Quality gate ---
        assert best_test_loss < _MSE_CEILING, (
            f"Quality gate failed for '{service_name}': "
            f"best_test_loss={best_test_loss:.4f} >= {_MSE_CEILING} "
            f"(normalized MSE ceiling).  A code change may have broken the model."
        )

        # --- Evaluate: predictions must be finite ---
        scaler = pipeline.scalers[service_name]
        # Derive feature names from the scaler so the list matches the pipeline output
        # exactly (the fixture CSV has an extra "CPU Usage %" column not in REQUIRED_COLUMNS).
        all_features = list(scaler.feature_names_in_)

        real_predictions, real_targets, user_counts_int = evaluate(
            model=model,
            test_loader=test_loader,
            scaler=scaler,
            target_columns=_TARGET_COLS,
            feature_names=all_features,
        )

        assert real_predictions.shape == real_targets.shape
        assert np.all(np.isfinite(real_predictions)), f"Non-finite predictions for {service_name}"
        assert np.issubdtype(user_counts_int.dtype, np.integer)
