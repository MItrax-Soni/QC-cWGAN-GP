"""
Master Sequential Orchestrator for 3-Architecture Comparison.
Automates:
1. Training Conditional VAE (1,000 Epochs).
2. Training Vanilla Conditional GAN (1,000 Epochs).
3. Generating synthetic datasets from VAE, cGAN, and cWGAN-GP.
4. Executing 4-stage adaptive quality filtering on cWGAN-GP.
5. Creating 4-panel qualitative visual comparison grid.
6. Evaluating downstream ResNet-18 classifiers across all 6 regimes matching Table 5.1.
"""

import os
import sys
import argparse
import time

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.training.train_vae import train_vae
from src.training.train_cgan import train_cgan
from src.evaluation.generate_comparative_data import (
    generate_vae_synthetic_data,
    generate_cgan_synthetic_data,
    generate_cwgan_unfiltered_data
)
from src.filtering.adaptive_filter import run_adaptive_filtering_pipeline
from src.visualization.visualize_architecture_comparison import generate_architecture_comparison_panel
from src.evaluation.run_three_model_benchmark import run_full_comparative_suite


def run_comparative_pipeline(
    vae_epochs: int = 1000,
    cgan_epochs: int = 1000,
    classifier_epochs: int = 5,
    skip_training: bool = False
):
    print("\n" + "="*70)
    print("STARTING 3-ARCHITECTURE COMPARATIVE PIPELINE")
    print("Architectures: Conditional VAE | Vanilla cGAN | Adaptive cWGAN-GP")
    print("="*70 + "\n")

    manifest = r"d:\Projects\GAN\data\splits\pilot_manifest.csv"

    # Step 1: Train VAE
    if not skip_training:
        print("\n>>> [Step 1/5] Training Conditional VAE Baseline...")
        train_vae(manifest_path=manifest, epochs=vae_epochs, batch_size=32)

        print("\n>>> [Step 2/5] Training Vanilla Conditional GAN Baseline...")
        train_cgan(manifest_path=manifest, epochs=cgan_epochs, batch_size=32)
    else:
        print("\n>>> Skipping generative model training, using existing checkpoints...")

    # Step 3: Generate Synthetic Datasets
    print("\n>>> [Step 3/5] Generating Synthetic Minority Faces from All 3 Architectures...")
    vae_man = generate_vae_synthetic_data(samples_per_minority_class=30)
    cgan_man = generate_cgan_synthetic_data(samples_per_minority_class=30)
    cwgan_raw_man = generate_cwgan_unfiltered_data(samples_per_minority_class=30)

    # Step 4: Quality-Controlled Filtering on cWGAN-GP
    print("\n>>> [Step 4/5] Executing 4-Stage Adaptive Quality Filtering on cWGAN-GP...")
    run_adaptive_filtering_pipeline(
        checkpoint_path=r"d:\Projects\GAN\experiments\cwgan_gp_1000ep\checkpoints\latest.pt",
        candidates_per_class=30
    )

    # Step 5: Visual Panel & Downstream Classification Suite
    print("\n>>> [Step 5/5] Generating Comparative Visual Panel & Downstream Classification Suite...")
    generate_architecture_comparison_panel()
    run_full_comparative_suite(epochs_per_classifier=classifier_epochs)

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Report Table 5.1 -> results/tables/three_architecture_comparison.csv")
    print("Visual Panel     -> results/figures/architecture_comparison_faces.png")
    print("Bar Chart        -> results/figures/architecture_accuracy_comparison.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-Architecture Comparison Pipeline.")
    parser.add_argument("--vae_epochs", type=int, default=1000, help="Epochs for VAE")
    parser.add_argument("--cgan_epochs", type=int, default=1000, help="Epochs for cGAN")
    parser.add_argument("--classifier_epochs", type=int, default=5, help="Epochs for classifier")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training and use checkpoints")
    args = parser.parse_args()

    run_comparative_pipeline(
        vae_epochs=args.vae_epochs,
        cgan_epochs=args.cgan_epochs,
        classifier_epochs=args.classifier_epochs,
        skip_training=args.skip_training
    )
