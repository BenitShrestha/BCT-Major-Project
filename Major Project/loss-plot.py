import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("plotted_points_bpe16/loss_log_bpe16_plotted_points.csv")

# Ensure val_loss is numeric
df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')

# Average train loss every 50 steps
avg_train_loss = df['train_loss'].groupby(df.index // 50).mean().reset_index(drop=True)
val_loss = df['val_loss'].dropna().reset_index(drop=True)

# Steps based on number of available points
min_len = min(len(avg_train_loss), len(val_loss))
steps = [i * 50 for i in range(min_len)]

# Trim both series to match steps length
avg_train_loss = avg_train_loss.iloc[:min_len]
val_loss = val_loss.iloc[:min_len]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(steps, avg_train_loss, label="Train Loss")
plt.plot(steps, val_loss, label="Validation Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)

# Save plot
plt.savefig("loss_curve_uni50.png", dpi=300, bbox_inches='tight')
plt.show()
