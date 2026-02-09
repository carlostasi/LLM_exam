import pandas as pd
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
metrics_balanced_path = os.path.join(current_dir, "metrics_balanced.csv")
metrics_path = os.path.join(current_dir, "metrics.csv")

df_balanced = pd.read_csv(metrics_balanced_path)
df_standard = pd.read_csv(metrics_path)

df_standard_filtered = df_standard[
    df_standard["Experiment"].str.contains("eval_loss", na=False)
].copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Validation Loss plot
ax1.plot(
    df_balanced["Epoch"],
    df_balanced["Validation_Loss"],
    marker="o",
    label="Balanced Model",
    linewidth=2,
    markersize=8,
)
ax1.plot(
    df_standard_filtered["Epoch"],
    df_standard_filtered["Validation_Loss"],
    marker="s",
    label="Standard Model (6 epochs with eval_loss)",
    linewidth=2,
    markersize=8,
)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Validation Loss", fontsize=12)
ax1.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy - Bar Chart
categories = ["Balanced", "Imbalanced"]
accuracies = [
    df_balanced["Final_Model_Accuracy"].iloc[-1],
    df_standard_filtered["Final_Model_Accuracy"].iloc[-1],
]
colors = ["#2ecc71", "#e74c3c"]
bars = ax2.bar(
    categories, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{acc:.2f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax2.set_ylabel("Final Model Accuracy (%)", fontsize=12)
ax2.set_title("Accuracy comparison", fontsize=14, fontweight="bold")
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    os.path.join(current_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight"
)
print("Chart saved come 'metrics_comparison.png'")
plt.show()

