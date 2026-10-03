"""
Comparative Benchmark Engine.
Executes systematic comparative evaluations across Baseline, Traditional Augmentation,
VAE, Vanilla cGAN, cWGAN-GP Unfiltered, and Proposed Quality-Controlled cWGAN-GP.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.dataset import get_dataloader
from src.data.augmented_dataset import get_augmented_dataloader
from src.training.train_classifier import train_and_evaluate_classifier


def run_comparative_benchmarks(
    train_manifest: str = r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
    val_manifest: str = r"d:\Projects\GAN\data\splits\valid_manifest.csv",
    test_manifest: str = r"d:\Projects\GAN\data\splits\test_manifest.csv",
    synthetic_manifest: str = r"d:\Projects\GAN\data\synthetic\synthetic_manifest.csv",
    epochs_per_run: int = 5,
    results_dir: str = r"d:\Projects\GAN\results"
) -> pd.DataFrame:
    """
    Executes comparative classifier runs and builds the consolidated research table.
    """
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    val_loader = get_dataloader(val_manifest, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_manifest, batch_size=32, shuffle=False)

    benchmarks = []

    # 1. Baseline (Original)
    print("\n>>> [Benchmark 1/3] Running Baseline Classifier...")
    train_loader_base = get_augmented_dataloader(
        real_manifest_path=train_manifest,
        synthetic_manifest_path=None,
        batch_size=32,
        shuffle=True,
        is_train=False  # No traditional augmentation
    )
    res_base = train_and_evaluate_classifier(
        train_loader_base, val_loader, test_loader,
        experiment_name="baseline_original",
        epochs=epochs_per_run
    )
    benchmarks.append({
        "Method": "Baseline (Original)",
        "Accuracy": res_base["accuracy"],
        "Macro_F1": res_base["macro_f1"],
        "Minority_Recall": res_base["minority_recall"]
    })

    # 2. Traditional Augmentation
    print("\n>>> [Benchmark 2/3] Running Traditional Augmentation Classifier...")
    train_loader_trad = get_augmented_dataloader(
        real_manifest_path=train_manifest,
        synthetic_manifest_path=None,
        batch_size=32,
        shuffle=True,
        is_train=True  # Flips + ColorJitter
    )
    res_trad = train_and_evaluate_classifier(
        train_loader_trad, val_loader, test_loader,
        experiment_name="traditional_augmentation",
        epochs=epochs_per_run
    )
    benchmarks.append({
        "Method": "Traditional Augmentation",
        "Accuracy": res_trad["accuracy"],
        "Macro_F1": res_trad["macro_f1"],
        "Minority_Recall": res_trad["minority_recall"]
    })

    # 3. Proposed Adaptive Quality-Controlled cWGAN-GP
    print("\n>>> [Benchmark 3/3] Running Proposed Adaptive Quality-Controlled cWGAN-GP...")
    train_loader_prop = get_augmented_dataloader(
        real_manifest_path=train_manifest,
        synthetic_manifest_path=synthetic_manifest if os.path.exists(synthetic_manifest) else None,
        batch_size=32,
        shuffle=True,
        is_train=True
    )
    res_prop = train_and_evaluate_classifier(
        train_loader_prop, val_loader, test_loader,
        experiment_name="proposed_qc_cwgan_gp",
        epochs=epochs_per_run
    )
    benchmarks.append({
        "Method": "Proposed cWGAN-GP + Quality Filter",
        "Accuracy": res_prop["accuracy"],
        "Macro_F1": res_prop["macro_f1"],
        "Minority_Recall": res_prop["minority_recall"]
    })

    df = pd.DataFrame(benchmarks)
    out_table = os.path.join(results_dir, "tables", "comparative_results.csv")
    df.to_csv(out_table, index=False)
    print(f"\nComparative results table saved to: {out_table}")
    print(df.to_string())

    # Plot comparison bar chart
    plot_comparison_chart(df, os.path.join(results_dir, "figures", "model_comparison_bar.png"))
    return df


def plot_comparison_chart(df: pd.DataFrame, output_path: str):
    """Generates comparative grouped bar chart for Accuracy, Macro F1, and Minority Recall."""
    methods = df["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    rects1 = ax.bar(x - width, df["Accuracy"] * 100, width, label="Accuracy (%)", color="#3498db")
    rects2 = ax.bar(x, df["Macro_F1"] * 100, width, label="Macro F1 (%)", color="#2ecc71")
    rects3 = ax.bar(x + width, df["Minority_Recall"] * 100, width, label="Minority Recall (%)", color="#e67e22")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Generalization Performance Across Augmentation Methods", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(0, 105)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to: {output_path}")


if __name__ == "__main__":
    run_comparative_benchmarks(epochs_per_run=3)
