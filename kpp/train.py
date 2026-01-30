import logging

import matplotlib.pyplot as plt
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


def plot_training_history(history: dict, save_path: str = "plots/training_loss.png") -> None:
    """
    Plots the training and validation loss from the training history.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Training Loss", color="blue")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color="orange")

    plt.title("Training Progress")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MAE)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(save_path)
    plt.close()
    logger.info(f"Loss plot saved to {save_path}")


def main():
    # 1. Setup Hyperparameters
    params = HyperParameters(epochs=50, batch_size=32, window_size=10, learning_rate=0.001)

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
    history = trainer.train(train_loader, val_loader)

    logger.info("Training Run Complete.")
    plot_training_history(history)


if __name__ == "__main__":
    main()


# TODO: implement the inference phase, with graphs to show the results. Also it is useful to plot graphs about the errors.
# def inspect_inference(model, loader, scaler, node_names, target_node='frontend'):
#     """
#     Runs inference, Inverse Scales, and Inverse Logs (expm1) to get real units.
#     """
#     model.eval()

#     x, y_true_scaled = next(iter(loader))
#     x, y_true_scaled = x.to(device), y_true_scaled.to(device)

#     with torch.no_grad():
#         y_pred_scaled = model(x)

#     y_pred_np = y_pred_scaled.cpu().numpy()
#     y_true_np = y_true_scaled.cpu().numpy()

#     y_pred_np = np.maximum(0, y_pred_np)

#     batch_size, num_nodes, num_features = y_pred_np.shape


#     y_pred_flat = y_pred_np.reshape(-1, num_features)
#     y_true_flat = y_true_np.reshape(-1, num_features)

#     y_pred_log = scaler.inverse_transform(y_pred_flat)
#     y_true_log = scaler.inverse_transform(y_true_flat)

#     # np.expm1 is the mathematical inverse of np.log1p
#     y_pred_real_flat = np.expm1(y_pred_log)
#     y_true_real_flat = np.expm1(y_true_log)

#     # 3. Reshape back to 3D
#     y_pred_real = y_pred_real_flat.reshape(batch_size, num_nodes, num_features)
#     y_true_real = y_true_real_flat.reshape(batch_size, num_nodes, num_features)

#     if target_node in node_names:
#         node_idx = node_names.index(target_node)
#     else:
#         node_idx = 0

#     print(f"\n--- Inference Snapshot for Node: {node_names[node_idx]} ---")
#     print(f"{'Metric':<20} | {'Actual':<12} | {'Predicted':<12} | {'Error'}")
#     print("-" * 60)

#     feature_names = ['User Count', 'Response Time', 'Throughput', 'CPU Usage']

#     sample_idx = 0
#     for feat_idx, feat_name in enumerate(feature_names):
#         actual = y_true_real[sample_idx, node_idx, feat_idx]
#         predicted = y_pred_real[sample_idx, node_idx, feat_idx]

#         if feat_name == 'User Count':
#             act_display = int(round(actual))
#             pred_display = int(round(predicted))
#             error = abs(act_display - pred_display)
#             print(f"{feat_name:<20} | {act_display:<12} | {pred_display:<12} | {error:<12}")
#         else:
#             error = abs(actual - predicted)
#             print(f"{feat_name:<20} | {actual:<12.4f} | {predicted:<12.4f} | {error:<.4f}")

# model.load_state_dict(torch.load('best_model_dense_gat_lstm.pth'))
# inspect_inference(model, test_loader, scaler, NODE_ORDER, target_node='frontend')
# inspect_inference(model, test_loader, scaler, NODE_ORDER, target_node='adservice')
# inspect_inference(model, test_loader, scaler, NODE_ORDER, target_node='adservice')
