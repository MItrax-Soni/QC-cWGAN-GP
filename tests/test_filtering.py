"""
Unit Tests for Adaptive Quality-Controlled Filtering Pipeline.
Tests Stage 1 (Critic-Score), Stage 2 (Feature Similarity),
Stage 3 (pHash Duplicate Detection), and Stage 4 (Diversity Maximization).
"""

import os
import sys
import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.generator import ConditionalGenerator
from src.models.critic import ConditionalCritic
from src.models.resnet_classifier import ResNetClassifier
from src.filtering.critic_filter import CriticFilter
from src.filtering.similarity_filter import SimilarityFilter
from src.filtering.duplicate_filter import DuplicateFilter
from src.filtering.diversity_filter import DiversityFilter
from src.filtering.adaptive_filter import AdaptiveQualityFilter


class TestQualityFiltering(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.num_classes = 9
        self.latent_dim = 50

        # Lightweight models for testing
        self.netG = ConditionalGenerator(latent_dim=self.latent_dim, num_classes=self.num_classes, feature_maps=16).to(self.device)
        self.netD = ConditionalCritic(num_classes=self.num_classes, feature_maps=16, image_size=64).to(self.device)
        self.feature_extractor = ResNetClassifier(num_classes=self.num_classes, pretrained=False).to(self.device)

        # Create dummy validation loader with guaranteed representation of all classes
        val_imgs = torch.randn(27, 3, 64, 64)
        val_lbls = torch.tensor([i % self.num_classes for i in range(27)])
        self.val_loader = DataLoader(TensorDataset(val_imgs, val_lbls), batch_size=10)

        # Create dummy training loader with guaranteed representation of all classes
        train_imgs = torch.randn(27, 3, 64, 64)
        train_lbls = torch.tensor([i % self.num_classes for i in range(27)])
        self.train_loader = DataLoader(TensorDataset(train_imgs, train_lbls), batch_size=10)

    def test_stage1_critic_filter(self):
        """Verifies Stage 1 validation quantile calibration and candidate partitioning."""
        cfilter = CriticFilter(self.netD, quantile=0.5, device=self.device)
        threshs = cfilter.calibrate_thresholds(self.val_loader)
        self.assertTrue(len(threshs) > 0)

        cands = torch.randn(10, 3, 64, 64)
        acc, rej, scores, t = cfilter.filter_candidates(cands, class_id=0)

        self.assertEqual(len(acc) + len(rej), 10)
        self.assertEqual(len(scores), 10)

    def test_stage2_similarity_filter(self):
        """Verifies Stage 2 feature extraction and cosine similarity thresholding."""
        sfilter = SimilarityFilter(self.feature_extractor, min_cosine=0.1, max_cosine=0.99, device=self.device)
        sfilter.build_reference_pool(self.train_loader)

        cands = torch.randn(8, 3, 64, 64)
        acc, rej, sims = sfilter.filter_candidates(cands, class_id=0)
        self.assertEqual(len(acc) + len(rej), 8)
        self.assertEqual(len(sims), 8)

    def test_stage3_duplicate_filter(self):
        """Verifies Stage 3 perceptual hash filtering eliminates identical duplicates."""
        dfilter = DuplicateFilter(max_hamming_distance=2)

        torch.manual_seed(42)
        # Create 2 distinct structured images and 1 exact duplicate
        img1 = torch.randn(1, 3, 64, 64).clamp(-1.0, 1.0)
        img2 = torch.randn(1, 3, 64, 64).clamp(-1.0, 1.0)
        img3 = img1.clone()  # Exact duplicate of img1
        cands = torch.cat([img1, img2, img3], dim=0)

        acc, rej = dfilter.filter_candidates(cands)
        self.assertEqual(len(acc), 2)
        self.assertEqual(len(rej), 1)

    def test_stage4_diversity_filter(self):
        """Verifies Stage 4 greedy dispersion selects the requested number of diverse faces."""
        div_filter = DiversityFilter(self.feature_extractor, device=self.device)
        cands = torch.randn(15, 3, 64, 64)

        acc, rej = div_filter.select_diverse_subset(cands, target_count=5)
        self.assertEqual(len(acc), 5)
        self.assertEqual(len(rej), 10)
        # Ensure selected indices are unique
        self.assertEqual(len(set(acc.tolist())), 5)


if __name__ == "__main__":
    unittest.main()
