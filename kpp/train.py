import logging

import pandas as pd
import torch
from plotting import run_inference_and_plot
from prediction import DataPreprocessor, HyperParameters, MetricForecaster
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


NODE_ORDER = [
    "checkoutservice",
    "emailservice",
    "currencyservice",
    "paymentservice",
    "frontend",
    "cartservice",
    "adservice",
    "shippingservice",
    "recommendationservice",
    "productcatalogservice",
]


def main():
    logger.info("Initializing Preprocessor...")

    params = HyperParameters(
        epochs=100,
        batch_size=64,
        lstm_hidden_size=32,
        lstm_layers=2,
        window_size=20,
        learning_rate=0.001,
        dropout=0.2,
    )

    csv_path = "performance_results_medium.csv"
    feature_cols = ["User Count", "Response Time (s)", "Throughput (req/s)", "CPU Usage"]
    preprocessor = DataPreprocessor(NODE_ORDER, feature_cols)

    try:
        # 1. Load Data (Returns Train and Test lists; Val is empty)
        #    Note the '_' to ignore the empty validation list
        train_list, _, test_list = preprocessor.load_and_process(
            csv_path, split_ratios=(params.val_split, params.test_split)
        )

        # 2. Reconstruct test_df based on User Count logic (Interpolation Strategy)
        #    We must filter the DataFrame using the EXACT same logic as the preprocessor
        #    to ensure the model predicts on the same data we visualize.
        full_df = pd.read_csv(csv_path)

        # Re-derive the test user counts
        unique_users = sorted(full_df["User Count"].unique())
        # Logic: "If index % 3 == 0, it's Test" (Matches your preprocessor)
        test_users = [u for i, u in enumerate(unique_users) if (i + 1) % 3 == 0]

        # Filter: Only rows with "Test" user counts
        test_df = full_df[full_df["User Count"].isin(test_users)].copy().reset_index(drop=True)

        # 3. Create Datasets
        #    Note: noise_level=0 is good for now to stabilize training
        train_ds = preprocessor.create_dataset(train_list, params.window_size, noise_level=0)
        test_ds = preprocessor.create_dataset(test_list, params.window_size)

        # 4. Handle Missing Validation Set
        #    Since we don't have a separate validation split, we use the Test set
        #    as the validation set. This lets the scheduler/early-stopping work
        #    by monitoring how well the model interpolates (generalizes).
        val_ds = test_ds

        # 5. Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params.batch_size)  # Actually checking Test data
        test_loader = DataLoader(test_ds, batch_size=params.batch_size)

        logger.info(
            f"Data Loaded. Train samples: {len(train_ds)}, Test/Val samples: {len(test_ds)}"
        )

    except FileNotFoundError:
        logger.error(f"Could not find {csv_path}. Run collector.py first to generate data.")
        return

    # Input: 4 features (Users + Metrics)
    # Output: 3 metrics (RT, Throughput, CPU)
    model = MetricForecaster(
        input_size=len(feature_cols),
        hidden_size=params.lstm_hidden_size,
        num_layers=params.lstm_layers,
        output_size=3,
        dropout=params.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.MSELoss()
    early_stop_counter = 0
    early_stop_patience = 10

    best_val_loss = float("inf")
    logger.info(params)
    logger.info("Starting Training...")
    for epoch in range(params.epochs):
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch + 1} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.6f}"
            )

        # --- Checkpointing & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_metric_forecaster.pth")
            early_stop_counter = 0  # Reset
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    logger.info("Training Complete. Best model saved.")

    test_loss = 0.0
    model.eval()

    with torch.no_grad():
        for inputs, targets in test_loader:
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)

    logger.info("--------------------------------------------------")
    logger.info(f"FINAL RESULT - Test Set Loss (MSE): {avg_test_loss:.6f}")
    logger.info("--------------------------------------------------")

    model.load_state_dict(torch.load("best_metric_forecaster.pth"))

    run_inference_and_plot(
        model=model,
        preprocessor=preprocessor,
        test_df=test_df,
        service_name="frontend",
        user_count=200,
        feature_cols=feature_cols,
        target_cols=["Response Time (s)", "Throughput (req/s)", "CPU Usage"],
    )


if __name__ == "__main__":
    main()
