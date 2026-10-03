"""
Comprehensive 3-Architecture Benchmark Runner.
Trains and evaluates downstream ResNet-18 classifiers across all 6 augmentation regimes
specified in Table 5.1 of the research project report:
1. Baseline (No Augmentation)
2. Traditional Augmentation (Flips + Jitter)
3. VAE-based Augmentation
4. Vanilla cGAN Augmentation
5. cWGAN-GP (Unfiltered)
6. Proposed Adaptive Quality-Controlled cWGAN-GP
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


def run_full_comparative_suite(
    train_manifest: str = r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
    val_manifest: str = r"d:\Projects\GAN\data\splits\valid_manifest.csv",
    test_manifest: str = r"d:\Projects\GAN\data\splits\test_manifest.csv",
    vae_manifest: str = r"d:\Projects\GAN\data\synthetic_vae\synthetic_manifest.csv",
    cgan_manifest: str = r"d:\Projects\GAN\data\synthetic_cgan\synthetic_manifest.csv",
    cwgan_raw_manifest: str = r"d:\Projects\GAN\data\synthetic_cwgan_unfiltered\synthetic_manifest.csv",
    cwgan_filtered_manifest: str = r"d:\Projects\GAN\data\synthetic\synthetic_manifest.csv",
    epochs_per_classifier: int = 5,
    results_dir: str = r"d:\Projects\GAN\results"
) -> pd.DataFrame:
    """
    Executes all 6 classifier benchmarks and generates the complete Table 5.1 report.
    """
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    val_loader = get_dataloader(val_manifest, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_manifest, batch_size=32, shuffle=False)

    benchmarks = []

    regimes = [
        ("No Augmentation (Baseline)", None, False, "baseline"),
        ("Traditional Augmentation", None, True, "traditional"),
        ("VAE-based Augmentation", vae_manifest, False, "vae"),
        ("Vanilla cGAN Augmentation", cgan_manifest, False, "cgan"),
        ("cWGAN-GP (No Filtering)", cwgan_raw_manifest, False, "cwgan_raw"),
        ("cWGAN-GP + Quality Filter (Proposed)", cwgan_filtered_manifest, True, "cwgan_proposed")
    ]

    for name, syn_path, trad_aug, exp_tag in regimes:
        print(f"\n=======================================================")
        print(f"Executing Benchmark: {name}")
        print(f"=======================================================")

        use_syn = syn_path if (syn_path and os.path.exists(syn_path)) else None

        train_loader = get_augmented_dataloader(
            real_manifest_path=train_manifest,
            synthetic_manifest_path=use_syn,
            batch_size=32,
            shuffle=True,
            is_train=trad_aug
        )

        res = train_and_evaluate_classifier(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            experiment_name=exp_tag,
            epochs=epochs_per_classifier
        )

        benchmarks.append({
            "Model": name,
            "Accuracy (%)": round(res["accuracy"] * 100, 2),
            "Macro F1 (%)": round(res["macro_f1"] * 100, 2),
            "Minority Recall (%)": round(res["minority_recall"] * 100, 2)
        })

    df = pd.DataFrame(benchmarks)
    table_path = os.path.join(results_dir, "tables", "three_architecture_comparison.csv")
    df.to_csv(table_path, index=False)
    print(f"\n[REPORT TABLE 5.1 SAVED] -> {table_path}\n")
    print(df.to_string())

    # Generate grouped comparison bar plot
    plot_six_regime_comparison(df, os.path.join(results_dir, "figures", "architecture_accuracy_comparison.png"))
    return df


def plot_six_regime_comparison(df: pd.DataFrame, output_path: str):
    """Plots comparative grouped bar chart for all 6 augmentation regimes."""
    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    rects1 = ax.bar(x - width, df["Accuracy (%)"], width, label="Accuracy (%)", color="#3498db")
    rects2 = ax.bar(x, df["Macro F1 (%)"], width, label="Macro F1 (%)", color="#2ecc71")
    rects3 = ax.bar(x + width, df["Minority Recall (%)"], width, label="Minority Recall (%)", color="#e67e22")

    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Downstream Classification Generalization (Table 5.1 Benchmark Replication)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, fontweight="bold", rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(0, 105)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"6-regime comparison chart saved to: {output_path}")


if __name__ == "__main__":
    run_full_comparative_suite(epochs_per_classifier=3)
