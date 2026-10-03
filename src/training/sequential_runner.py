"""
Automated Sequential Pipeline Runner.
Monitors VAE completion, launches Vanilla cGAN 1,000-epoch training,
and orchestrates the final comparative evaluation and visual reporting.
"""

import os
import sys
import time
import pandas as pd
import torch

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.training.train_cgan import train_cgan
from src.evaluation.generate_comparative_data import (
    generate_vae_synthetic_data,
    generate_cgan_synthetic_data,
    generate_cwgan_unfiltered_data
)
from src.visualization.visualize_architecture_comparison import generate_architecture_comparison_panel
from src.evaluation.run_three_model_benchmark import run_full_comparative_suite


def wait_for_vae_completion(
    metrics_path: str = r"d:\Projects\GAN\experiments\vae_1000ep\metrics.csv",
    target_epochs: int = 1000,
    poll_interval_sec: int = 30
):
    """Monitors VAE metrics.csv until target epoch is reached."""
    print(f"Monitoring VAE progress towards {target_epochs} epochs...", flush=True)
    while True:
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path)
                if len(df) > 0:
                    current_epoch = int(df["epoch"].iloc[-1])
                    print(f"[VAE Monitor] Current Epoch: {current_epoch} / {target_epochs}", flush=True)
                    if current_epoch >= target_epochs:
                        print(f"\n[VAE Monitor] VAE {target_epochs}-Epoch Training Complete!\n", flush=True)
                        break
            except Exception:
                pass
        time.sleep(poll_interval_sec)


def execute_remaining_comparative_pipeline(target_epochs: int = 1000):
    manifest = r"d:\Projects\GAN\data\splits\pilot_manifest.csv"

    # 1. Wait for VAE to complete
    wait_for_vae_completion(target_epochs=target_epochs)

    # 2. Train Vanilla cGAN to 1000 Epochs (fresh import)
    import src.training.train_cgan as tc
    import importlib
    importlib.reload(tc)

    print("\n" + "="*70, flush=True)
    print("STARTING VANILLA cGAN 1,000-EPOCH TRAINING", flush=True)
    print("="*70 + "\n", flush=True)
    tc.train_cgan(manifest_path=manifest, epochs=target_epochs, batch_size=32)

    # 3. Generate Comparative Synthetic Datasets
    print("\n>>> Generating Synthetic Minority Faces from All 3 Architectures...")
    vae_man = generate_vae_synthetic_data(samples_per_minority_class=30)
    cgan_man = generate_cgan_synthetic_data(samples_per_minority_class=30)
    cwgan_unfiltered_man = generate_cwgan_unfiltered_data(samples_per_minority_class=30)

    # 4. Generate 4-Row Side-by-Side Visual Comparison
    print("\n>>> Generating Visual Architecture Comparison Panel (Figure)...")
    generate_architecture_comparison_panel()

    # 5. Downstream ResNet-18 Classifier Benchmarks (Table 5.1 Replication)
    print("\n>>> Running Downstream ResNet-18 Evaluation on Held-Out Test Data...")
    run_full_comparative_suite(epochs_per_classifier=5)

    print("\n" + "="*70)
    print("ALL 3 ARCHITECTURES (cWGAN-GP, VAE, cGAN) 1,000-EPOCH BENCHMARK COMPLETE!")
    print("Results Table -> results/tables/three_architecture_comparison.csv")
    print("Visual Panel  -> results/figures/architecture_comparison_faces.png")
    print("Bar Chart     -> results/figures/architecture_accuracy_comparison.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    execute_remaining_comparative_pipeline()
