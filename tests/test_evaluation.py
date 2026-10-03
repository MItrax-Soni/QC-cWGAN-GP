"""
Unit Tests for Evaluation and Augmented Dataset Modules.
Tests metric calculations, confusion matrix generation, and augmented dataset loading.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.classification_metrics import calculate_classification_metrics
from src.data.augmented_dataset import CombinedAugmentedDataset


class TestEvaluation(unittest.TestCase):

    def test_metrics_calculation(self):
        """Verifies accuracy, macro F1, and minority recall on known distributions."""
        y_true = np.array([0, 1, 2, 4, 4, 7, 7, 8])
        y_pred = np.array([0, 1, 2, 4, 0, 7, 7, 8])  # 7 / 8 correct

        metrics = calculate_classification_metrics(y_true, y_pred, minority_classes=[1, 2, 7])

        self.assertAlmostEqual(metrics["accuracy"], 7 / 8, places=4)
        self.assertTrue(0.0 <= metrics["macro_f1"] <= 1.0)
        self.assertAlmostEqual(metrics["minority_recall"], 1.0, places=4)  # 1, 2, 7 were 100% correct
        self.assertEqual(len(metrics["confusion_matrix"]), 9)
        self.assertEqual(len(metrics["confusion_matrix"][0]), 9)

    def test_augmented_dataset_loading(self):
        """Verifies CombinedAugmentedDataset integrates both real and synthetic entries."""
        real_manifest = r"d:\Projects\GAN\data\splits\pilot_manifest.csv"
        syn_manifest = r"d:\Projects\GAN\data\synthetic\synthetic_manifest.csv"

        if os.path.exists(real_manifest) and os.path.exists(syn_manifest):
            ds = CombinedAugmentedDataset(
                real_manifest_path=real_manifest,
                synthetic_manifest_path=syn_manifest,
                image_size=64,
                is_train=True
            )
            self.assertTrue(len(ds) > 450)
            img, lbl = ds[len(ds) - 1]
            self.assertEqual(img.shape, (3, 64, 64))
            self.assertTrue(0 <= lbl.item() <= 8)


if __name__ == "__main__":
    unittest.main()
