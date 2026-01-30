import logging

from prediction import DataPreprocessor, DenseGATLSTM, GNNTrainer, HyperParameters
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

ADJ_MATRIX = [
    # fr, ad, rec,cat,chk,crt,shp,cur,pay,eml
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # frontend
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ad
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # recommendation
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # productcatalog
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],  # checkout
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # cart
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # shipping
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # currency
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # payment
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # email
]


def main():
    # 1. Setup Hyperparameters
    params = HyperParameters(epochs=100, batch_size=32, window_size=10, learning_rate=0.001)

    # 2. Define Features (Must match the columns in your CSV)
    # Based on PerformanceSample in collector.py
    feature_cols = ["User Count", "Response Time (s)", "Throughput (req/s)", "CPU Usage"]

    # 3. Process Data
    csv_path = "performance_results_FINAL.csv"

    logger.info("Initializing Preprocessor...")
    preprocessor = DataPreprocessor(NODE_ORDER, feature_cols)

    try:
        # Load and split data
        t_tensor, v_tensor, test_tensor = preprocessor.load_and_process(
            csv_path, split_ratios=(params.val_split, params.test_split)
        )

        # Create sliding windows for LSTM
        train_ds = preprocessor.create_windows(t_tensor, params.window_size)
        val_ds = preprocessor.create_windows(v_tensor, params.window_size)

        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params.batch_size)

        logger.info(f"Data Loaded. Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    except FileNotFoundError:
        logger.error(f"Could not find {csv_path}. Run collector.py first to generate data.")
        return

    # 4. Initialize Model
    model = DenseGATLSTM(
        num_nodes=len(NODE_ORDER),
        in_channels=len(feature_cols),
        out_channels=len(feature_cols),
        adj_matrix=ADJ_MATRIX,
        params=params,
    )

    # 5. Train
    trainer = GNNTrainer(model, params)
    trainer.train(train_loader, val_loader)

    logger.info("Training Run Complete.")


if __name__ == "__main__":
    main()
