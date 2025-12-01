import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# ===== CONFIGURATION =====
input_folder = "loss_log"        # Folder with input CSVs
plots_folder = "plots"           # Folder to save plots (PNGs)
points_folder = "plotted_points" # Folder to save CSVs of plotted points
window_size = 5                  # Rolling average window size for smoothing
step_group_size = 50             # Number of steps per averaged train loss group
# =========================

# Create output folders if they don't exist
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(points_folder, exist_ok=True)

# Get list of all CSV files in input_folder
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

for csv_file in csv_files:
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Read CSV
    df = pd.read_csv(csv_file)

    # Ensure val_loss is numeric
    df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')

    # Average train loss every `step_group_size` steps
    avg_train_loss = df['train_loss'].groupby(df.index // step_group_size).mean().reset_index(drop=True)
    val_loss = df['val_loss'].dropna().reset_index(drop=True)

    # Match length for train and val losses
    min_len = min(len(avg_train_loss), len(val_loss))
    steps = [i * step_group_size for i in range(min_len)]

    avg_train_loss = avg_train_loss.iloc[:min_len]
    val_loss = val_loss.iloc[:min_len]

    # Apply rolling average smoothing
    avg_train_loss_smooth = avg_train_loss.rolling(window=window_size, center=True, min_periods=1).mean()
    val_loss_smooth = val_loss.rolling(window=window_size, center=True, min_periods=1).mean()

    # Save the smoothed points for inspection
    points_df = pd.DataFrame({
        "step": steps,
        "avg_train_loss": avg_train_loss_smooth,
        "val_loss": val_loss_smooth
    })
    points_csv_name = os.path.join(points_folder, f"{base_name}_plotted_points.csv")
    points_df.to_csv(points_csv_name, index=False)

    # Plot smoothed loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(steps, avg_train_loss_smooth, label="Train Loss")
    plt.plot(steps, val_loss_smooth, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {base_name}")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_filename = os.path.join(plots_folder, f"{base_name}_loss_curve.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close plot to avoid memory issues when looping

    print(f"Processed '{base_name}': plot saved to '{plot_filename}', points saved to '{points_csv_name}'")
