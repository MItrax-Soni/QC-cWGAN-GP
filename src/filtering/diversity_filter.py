"""
Stage 4: Diversity Maximization Quality Filtering.
Selects the most diverse subset of synthetic faces using greedy max-min dispersion
in ResNet-18 feature space to guarantee wide coverage of the expression manifold.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityFilter:
    """
    Selects a diverse subset of candidate synthetic faces via greedy facility location /
    max-min dispersion in feature space.
    """
    def __init__(self, feature_extractor: nn.Module, device: torch.device = torch.device("cpu")):
        self.feature_extractor = feature_extractor.to(device)
        self.device = device

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts normalized 512-dim features."""
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor.extract_features(images.to(self.device))
            norm_feats = F.normalize(feats, p=2, dim=1)
        return norm_feats.cpu()

    def select_diverse_subset(
        self,
        candidate_images: torch.Tensor,
        target_count: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects target_count most diverse samples from candidate_images.
        Args:
            candidate_images: Tensor of shape (P, 3, 64, 64).
            target_count: Target number of diverse images to retain.
        Returns:
            selected_indices: Tensor of selected indices.
            unselected_indices: Tensor of unselected indices.
        """
        num_candidates = candidate_images.size(0)
        if num_candidates <= target_count:
            return torch.arange(num_candidates, dtype=torch.long), torch.empty(0, dtype=torch.long)

        features = self.extract_features(candidate_images)  # (P, 512)

        # 1. Start with the sample closest to the mean centroid
        centroid = torch.mean(features, dim=0, keepdim=True)
        dist_to_centroid = torch.norm(features - centroid, dim=1)
        first_idx = int(torch.argmin(dist_to_centroid).item())

        selected = [first_idx]
        remaining = set(range(num_candidates))
        remaining.remove(first_idx)

        # Precompute distance matrix: (P, P)
        dist_matrix = torch.cdist(features, features, p=2)

        # 2. Greedily select samples that maximize min distance to already selected pool
        while len(selected) < target_count and len(remaining) > 0:
            best_cand = None
            max_min_dist = -1.0

            for cand in remaining:
                # Minimum distance from candidate to any already selected sample
                dists_to_selected = dist_matrix[cand, selected]
                min_dist = torch.min(dists_to_selected).item()

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_cand = cand

            selected.append(best_cand)
            remaining.remove(best_cand)

        selected_indices = torch.tensor(selected, dtype=torch.long)
        unselected_indices = torch.tensor(list(remaining), dtype=torch.long)

        return selected_indices, unselected_indices
