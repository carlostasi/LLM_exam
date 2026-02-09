import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


current_dir = os.getcwd()
file = pd.read_csv(os.path.join(current_dir, "BERT_metrics", "metrics.csv"))

# Extract features
experiments = file["Experiment"].unique()
# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("BERT Model Training Metrics Analysis", fontsize=16, fontweight="bold")

# Plot 1: Training Loss vs Epoch for all experiments
ax1 = axes[0, 0]
for exp in experiments:
    exp_data = file[file["Experiment"] == exp]
    ax1.plot(exp_data["Epoch"], exp_data["Training_Loss"], marker="o", label=exp)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss")
ax1.set_title("Training Loss per Epoch")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Validation Loss vs Epoch for all experiments
ax2 = axes[0, 1]
for exp in experiments:
    exp_data = file[file["Experiment"] == exp]
    ax2.plot(exp_data["Epoch"], exp_data["Validation_Loss"], marker="s", label=exp)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Loss")
ax2.set_title("Validation Loss per Epoch")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Step Accuracy vs Epoch for all experiments
ax3 = axes[1, 0]
for exp in experiments:
    exp_data = file[file["Experiment"] == exp]
    ax3.plot(exp_data["Epoch"], exp_data["Step_Accuracy"], marker="^", label=exp)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Step Accuracy")
ax3.set_title("Step Accuracy per Epoch")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Final Model Accuracy per Experiment
ax4 = axes[1, 1]
final_accuracies = file.groupby("Experiment")["Final_Model_Accuracy"].first()
colors = plt.cm.viridis(np.linspace(0, 1, len(final_accuracies)))
bars = ax4.bar(range(len(final_accuracies)), final_accuracies.values, color=colors)
ax4.set_xticks(range(len(final_accuracies)))
ax4.set_xticklabels(final_accuracies.index, rotation=45, ha="right", fontsize=8)
ax4.set_ylabel("Accuracy")
ax4.set_title("Final Model Accuracy per Experiment")
ax4.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(
    os.path.join(current_dir, "BERT_metrics", "metrics_visualization.png"),
    dpi=300,
    bbox_inches="tight",
)
print("\nChart saved as 'metrics_visualization.png'")
plt.show()
