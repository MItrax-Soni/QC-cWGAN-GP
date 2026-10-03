"""
Adaptive Quality-Controlled cWGAN-GP for Long-Tailed Facial Expression Recognition.
Master CLI Orchestrator and Pipeline Entrypoint.
"""

import os
import sys
import argparse
import unittest


def run_tests():
    """Runs all 18 discovery unit tests."""
    print("\n=======================================================")
    print("Running Full Research Unit Test Suite...")
    print("=======================================================\n")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("[FAIL] Unit tests failed!")
        sys.exit(1)
    print("\n[SUCCESS] All unit tests passed cleanly!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator: Adaptive Quality-Controlled cWGAN-GP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Modes:
  test               : Run full test suite (18 unit tests across models, filtering, dataset, evaluation).
  audit              : Audit system environment, hardware, and dataset structure.
  train_wgan         : Train the proposed cWGAN-GP with gradient penalty.
  train_vae          : Train the baseline Conditional VAE model.
  train_cgan         : Train the baseline Vanilla cGAN model.
  filter             : Run 4-stage Adaptive Quality-Controlled Filtering on minority expressions.
  generate           : Generate class-conditional synthetic face samples.
  train_classifier   : Train downstream ResNet-18 classifier across augmentation regimes.
  evaluate           : Generate comparative benchmark table and visualization charts.
  ablation           : Generate multi-stage ablation study metrics and plots.
  all                : Execute end-to-end pipeline: tests -> filtering -> classification -> evaluation.
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=[
            "test",
            "audit",
            "train_wgan",
            "train_vae",
            "train_cgan",
            "filter",
            "generate",
            "train_classifier",
            "evaluate",
            "ablation",
            "tsne",
            "compare_all",
            "all"
        ],
        help="Pipeline phase to execute."
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--checkpoint", type=str, default=r"d:\Projects\GAN\checkpoints\pilot_5ep\latest.pt", help="Path to checkpoint")
    parser.add_argument("--manifest", type=str, default=r"d:\Projects\GAN\data\splits\pilot_manifest.csv", help="Dataset manifest CSV path")
    parser.add_argument("--experiment_id", type=str, default="cwgan_gp_local", help="Experiment ID / directory name")

    args = parser.parse_args()

    if args.mode == "test":
        run_tests()

    elif args.mode == "audit":
        from src.utils.audit import run_full_audit
        run_full_audit()

    elif args.mode == "train_wgan":
        from src.training.train_wgan_gp import train_cwgan_gp
        train_cwgan_gp(
            manifest_path=args.manifest,
            experiment_id=args.experiment_id,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    elif args.mode == "train_vae":
        from src.training.train_vae import train_vae
        train_vae(epochs=args.epochs, batch_size=args.batch_size)

    elif args.mode == "train_cgan":
        from src.training.train_cgan import train_cgan
        train_cgan(epochs=args.epochs, batch_size=args.batch_size)

    elif args.mode == "filter":
        from src.filtering.adaptive_filter import run_adaptive_filtering_pipeline
        run_adaptive_filtering_pipeline(checkpoint_path=args.checkpoint)

    elif args.mode == "generate":
        from src.generation.generate_samples import generate_samples
        generate_samples(checkpoint_path=args.checkpoint)

    elif args.mode == "train_classifier":
        from src.training.train_classifier import train_and_evaluate_classifier
        from src.data.augmented_dataset import get_augmented_dataloader
        from src.data.dataset import get_dataloader

        train_loader = get_augmented_dataloader(
            real_manifest_path=r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
            synthetic_manifest_path=r"d:\Projects\GAN\data\synthetic\synthetic_manifest.csv",
            batch_size=args.batch_size,
            is_train=True
        )
        val_loader = get_dataloader(r"d:\Projects\GAN\data\splits\valid_manifest.csv", batch_size=args.batch_size, shuffle=False)
        test_loader = get_dataloader(r"d:\Projects\GAN\data\splits\test_manifest.csv", batch_size=args.batch_size, shuffle=False)

        train_and_evaluate_classifier(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            experiment_name="proposed_qc_cwgan_gp"
        )

    elif args.mode == "evaluate":
        from src.evaluation.comparison import run_comparative_benchmarks
        run_comparative_benchmarks(epochs_per_run=args.epochs)

    elif args.mode == "ablation":
        from src.evaluation.ablation import generate_ablation_summary
        generate_ablation_summary()

    elif args.mode == "tsne":
        from src.visualization.feature_visualization import generate_tsne_visualization
        generate_tsne_visualization()

    elif args.mode == "compare_all":
        from src.training.orchestrate_all import run_comparative_pipeline
        run_comparative_pipeline(classifier_epochs=args.epochs)

    elif args.mode == "all":
        run_tests()
        print("\n>>> [Pipeline 1/4] Running 4-Stage Adaptive Quality Filtering...")
        from src.filtering.adaptive_filter import run_adaptive_filtering_pipeline
        run_adaptive_filtering_pipeline(checkpoint_path=args.checkpoint)

        print("\n>>> [Pipeline 2/4] Running Downstream Classifier Benchmarks...")
        from src.evaluation.comparison import run_comparative_benchmarks
        run_comparative_benchmarks(epochs_per_run=args.epochs)

        print("\n>>> [Pipeline 3/4] Generating Ablation Study Summary...")
        from src.evaluation.ablation import generate_ablation_summary
        generate_ablation_summary()

        print("\n>>> [Pipeline 4/4] Pipeline Complete! Results saved in results/")


if __name__ == "__main__":
    main()
