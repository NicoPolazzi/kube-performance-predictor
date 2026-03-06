import logging

import optuna
from torch.utils.data import DataLoader

from kpp.config import PredictorConfig
from kpp.predictor.model import PerformanceModel, train_model
from kpp.predictor.pipeline import PerformanceDataPipeline

logger = logging.getLogger("tune")


def objective(trial: optuna.Trial, config: PredictorConfig, datasets: dict) -> float:
    trial_lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    trial_weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    trial_batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    trial_hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    trial_hidden_size_2 = trial.suggest_categorical("hidden_size_2", [32, 64, 128, 256])
    trial_head_hidden_size = trial.suggest_categorical("head_hidden_size", [16, 32, 64, 128])

    if trial_hidden_size_2 > trial_hidden_size:
        raise optuna.TrialPruned()

    if trial_head_hidden_size > trial_hidden_size_2:
        raise optuna.TrialPruned()

    # 2. Override the config with the suggested values
    config.training.learning_rate = trial_lr
    config.training.weight_decay = trial_weight_decay
    config.model.hidden_size = trial_hidden_size
    config.model.hidden_size_2 = trial_hidden_size_2
    config.model.head_hidden_size = trial_head_hidden_size
    config.training.batch_size = trial_batch_size

    total_validation_loss = 0.0

    # 3. Train the model on each microservice and accumulate the loss
    for service_name, data_split in datasets.items():
        train_dataset = data_split["train"]
        test_dataset = data_split["test"]

        train_loader = DataLoader(train_dataset, batch_size=trial_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=trial_batch_size, shuffle=False)

        input_size = train_dataset.tensors[0].shape[1]
        output_size = train_dataset.tensors[1].shape[1]

        model = PerformanceModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=trial_hidden_size,
            hidden_size_2=trial_hidden_size_2,
            head_hidden_size=trial_head_hidden_size,
        )

        # We can use fewer epochs during tuning to save time (e.g., 50 instead of 200)
        best_loss = train_model(
            config,
            service_name,
            model,
            train_loader,
            test_loader,
            epochs=100,
            learning_rate=trial_lr,
        )

        total_validation_loss += best_loss

    # Return the average validation loss across all microservices
    return total_validation_loss / len(datasets)


def main():
    # Load config and data EXACTLY once to save time
    config = PredictorConfig.from_yaml()
    target_cols = ["Response Time (s)", "Throughput (req/s)", "CPU Usage"]
    pipeline = PerformanceDataPipeline(config.pipeline.sequence_length, target_cols)

    datasets = pipeline.run(
        "dataset/performance_results_normal.csv",
        train_ratio=config.pipeline.train_ratio,
        split_strategy=config.pipeline.split_strategy,
    )

    # Create the Optuna study
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Run the optimization for 50 trials
    print("Starting hyperparameter tuning...")
    study.optimize(lambda trial: objective(trial, config, datasets), n_trials=100)

    print("\n--- Tuning Complete ---")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Validation Loss: {study.best_value:.6f}")
    print("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
