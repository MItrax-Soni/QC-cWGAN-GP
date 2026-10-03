"""
Training Loss and Metric Visualization Engine.
Plots publication-quality curves for Critic Loss, Generator Loss, Wasserstein Distance,
and Gradient Penalty across training epochs.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(metrics_csv_path: str, output_fig_path: str, title_suffix: str = ""):
    """
    Plots a 4-panel dashboard of training dynamics from a metrics.csv file.
    """
    if not os.path.exists(metrics_csv_path):
        raise FileNotFoundError(f"Metrics CSV not found at: {metrics_csv_path}")

    df = pd.read_csv(metrics_csv_path)
    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    # 1. Critic Loss
    ax = axes[0, 0]
    ax.plot(epochs, df["critic_loss"], color="#e74c3c", linewidth=2.0, marker="o", label="Critic Loss (L_D)")
    ax.set_title("Critic Loss over Epochs", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 2. Generator Loss
    ax = axes[0, 1]
    ax.plot(epochs, df["generator_loss"], color="#2980b9", linewidth=2.0, marker="s", label="Generator Loss (L_G)")
    ax.set_title("Generator Loss over Epochs", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 3. Wasserstein Distance Estimate
    ax = axes[1, 0]
    ax.plot(epochs, df["wasserstein_estimate"], color="#27ae60", linewidth=2.0, marker="^", label="Wasserstein Estimate E[D(x)] - E[D(G(z))]")
    ax.set_title("Wasserstein Distance Estimate", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Estimated W-Distance", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # 4. Gradient Penalty
    ax = axes[1, 1]
    ax.plot(epochs, df["gradient_penalty"], color="#8e44ad", linewidth=2.0, marker="d", label="Gradient Penalty (GP)")
    ax.axhline(1.0, color="#7f8c8d", linestyle="--", label="Target Norm Level (1.0)")
    ax.set_title("Gradient Penalty (1-Lipschitz Constraint)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Penalty Value", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    suptitle = f"cWGAN-GP Training Dynamics {title_suffix}".strip()
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved successfully to: {output_fig_path}")


if __name__ == "__main__":
    import sys
    csv_p = sys.argv[1] if len(sys.argv) > 1 else r"d:\Projects\GAN\experiments\test_infra_pilot\metrics.csv"
    fig_p = sys.argv[2] if len(sys.argv) > 2 else r"d:\Projects\GAN\results\figures\pilot_loss_curves.png"
    plot_training_curves(csv_p, fig_p)
