import numpy as np
from matplotlib import pyplot as plt


def run_inference_and_plot(
    model,
    preprocessor,
    test_df,
    service_name,
    user_count,
    feature_cols,
    target_cols,
    window_size=20,
):
    """
    Runs inference, un-scales the predictions, and creates a subplot for each target metric.
    """

    # 1. Filter Data
    mask = (test_df["Service"] == service_name) & (test_df["User Count"] == user_count)
    subset_df = test_df[mask].copy()

    if len(subset_df) <= window_size:
        print(f"Error: Not enough data ({len(subset_df)}) for window size {window_size}.")
        return

    # 2. Scale the Input Data (CRITICAL STEP)
    # The model was trained on scaled data (0-1 range).
    # If we feed it raw numbers (e.g. 100 req/s), it will fail.
    try:
        raw_values = subset_df[feature_cols].values
        scaled_values = preprocessor.scaler.transform(raw_values)
    except Exception as e:
        print(f"Scaling Error: {e}")
        return

    # 3. Create Windows from SCALED data
    X_windows = []
    for i in range(len(scaled_values) - window_size):
        X_windows.append(scaled_values[i : i + window_size])

    X_batch = np.array(X_windows)

    # 4. Inference
    print(f"Running inference on {len(X_batch)} windows...")
    predictions_scaled = model.predict(X_batch)  # Shape: (N, 3)

    # 5. Inverse Scale the Predictions
    # The scaler expects 4 columns (User, RT, Thr, CPU), but model outputs 3 (RT, Thr, CPU).
    # We must build a dummy matrix to "trick" the inverse_transform method.

    num_samples = predictions_scaled.shape[0]
    num_features = len(feature_cols)  # Should be 4

    # Create a matrix of zeros [N, 4]
    dummy_matrix = np.zeros((num_samples, num_features))

    # Fill the prediction columns.
    # ASSUMPTION: feature_cols order is ["User Count", "RT", "Thr", "CPU"]
    # So we map model outputs 0,1,2 to dummy columns 1,2,3.
    dummy_matrix[:, 1] = predictions_scaled[:, 0]  # Response Time
    dummy_matrix[:, 2] = predictions_scaled[:, 1]  # Throughput
    dummy_matrix[:, 3] = predictions_scaled[:, 2]  # CPU Usage

    # Convert back to original units
    predictions_unscaled_full = preprocessor.scaler.inverse_transform(dummy_matrix)

    # Slice out only the metrics we care about (columns 1, 2, 3)
    predictions_final = predictions_unscaled_full[:, 1:]

    # 6. Plotting (Subplots)
    num_metrics = len(target_cols)
    fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(12, 4 * num_metrics), sharex=True)

    if num_metrics == 1:
        axes = [axes]

    # Align the actual data (raw values from dataframe)
    aligned_df = subset_df.iloc[window_size:].reset_index(drop=True)

    for i, col_name in enumerate(target_cols):
        ax = axes[i]

        # Plot Actual (Raw Data)
        ax.plot(aligned_df[col_name], label=f"Actual {col_name}", color="blue", alpha=0.6)

        # Plot Predicted (Unscaled)
        ax.plot(
            predictions_final[:, i],
            label=f"Predicted {col_name}",
            color="red",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_ylabel(col_name)
        ax.set_title(f"{col_name} Analysis for {service_name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("Sample Index (Time Steps)")
    plt.suptitle(
        f"Performance Predictions: {service_name} (Users: {user_count})", y=1.02, fontsize=16
    )
    plt.tight_layout()
    plt.show()
