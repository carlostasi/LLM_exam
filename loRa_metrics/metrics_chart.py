import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "loRa_metrics", "metrics.csv")

try:
    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()
    df["Experiment"] = df["Model_name"] + "\n(" + df["Balanced/Unblanced"] + ")"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "GPT-2 vs DistilGPT-2: Accuracy & Time Analysis", fontsize=16, fontweight="bold"
    )

    # --- PLOT 1: Accuracy Comparison ---
    ax1 = axes[0]
    colors_acc = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars1 = ax1.bar(
        df["Experiment"],
        df["Accuracy(%)"],
        color=colors_acc,
        alpha=0.85,
        edgecolor="black",
    )

    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    ax1.set_ylim(0, df["Accuracy(%)"].max() * 1.15)

    # Labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # --- PLOT 2: Training Time Comparison ---
    ax2 = axes[1]
    colors_time = plt.cm.plasma(np.linspace(0.2, 0.8, len(df)))

    bars2 = ax2.bar(
        df["Experiment"],
        df["Training_Time(min)"],
        color=colors_time,
        alpha=0.85,
        edgecolor="black",
    )

    ax2.set_ylabel("Training Time (min)", fontsize=11)
    ax2.set_title("Training Time Comparison", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, df["Training_Time(min)"].max() * 1.15)

    # More labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.2,
            f"{height:.1f} min",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    output_filename = "gpt_metrics_visualization_csv.png"
    plt.savefig(
        os.path.join(current_dir, "loRa_metrics", output_filename),
        dpi=300, 
        bbox_inches="tight")

    print(f"\nChart saved successfully as '{output_filename}'")
    plt.show()

except FileNotFoundError:
    print(
        f"Errore: Il file '{file_path}' non è stato trovato nella directory corrente."
    )
except KeyError as e:
    print(f"Errore: Colonna mancante nel CSV. Controlla i nomi. Dettaglio: {e}")
    print(f"Colonne trovate: {df.columns.tolist()}")
