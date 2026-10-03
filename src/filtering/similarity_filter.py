"""
Stage 2: Feature-Space Similarity Quality Filtering.
Computes cosine similarity in ResNet-18 embedding space against real reference faces.
Filters out memorized replicas (excessive similarity) and semantic outliers (too distant).
"""

from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityFilter:
    """
    Evaluates cosine similarity of candidate faces against a reference pool of real faces.
    """
    def __init__(
        self,
        feature_extractor: nn.Module,
        min_cosine: float = 0.25,
        max_cosine: float = 0.95,
        device: torch.device = torch.device("cpu")
    ):
        self.feature_extractor = feature_extractor.to(device)
        self.min_cosine = min_cosine
        self.max_cosine = max_cosine
        self.device = device
        self.reference_embeddings: Dict[int, torch.Tensor] = {}

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts normalized 512-dim feature embeddings."""
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor.extract_features(images.to(self.device))
            norm_features = F.normalize(features, p=2, dim=1)
        return norm_features.cpu()

    def build_reference_pool(self, real_loader: torch.utils.data.DataLoader, max_per_class: int = 500) -> None:
        """
        Builds a normalized reference feature pool from real training images.
        """
        feats_by_class: Dict[int, List[torch.Tensor]] = {}

        for imgs, labels in real_loader:
            feats = self.extract_features(imgs)
            for f, lbl in zip(feats, labels.tolist()):
                if lbl not in feats_by_class:
                    feats_by_class[lbl] = []
                if len(feats_by_class[lbl]) < max_per_class:
                    feats_by_class[lbl].append(f)

        for cid, flist in feats_by_class.items():
            self.reference_embeddings[cid] = torch.stack(flist, dim=0)

    def filter_candidates(
        self,
        candidate_images: torch.Tensor,
        class_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Filters candidates based on cosine similarity to real faces of the same class.
        Args:
            candidate_images: Tensor of shape (M, 3, 64, 64).
            class_id: Target class label.
        Returns:
            accepted_indices: Tensor of accepted indices.
            rejected_indices: Tensor of rejected indices.
            max_similarities: Maximum cosine similarity for each candidate.
        """
        if class_id not in self.reference_embeddings:
            raise KeyError(f"Reference pool for class {class_id} has not been built!")

        cand_features = self.extract_features(candidate_images)
        ref_features = self.reference_embeddings[class_id]  # Shape: (R, 512)

        # Compute cosine similarity matrix: (M, R)
        sim_matrix = torch.matmul(cand_features, ref_features.T)
        max_sims, _ = torch.max(sim_matrix, dim=1)  # Nearest real neighbor similarity

        accepted_mask = (max_sims >= self.min_cosine) & (max_sims <= self.max_cosine)
        accepted_indices = torch.where(accepted_mask)[0]
        rejected_indices = torch.where(~accepted_mask)[0]

        return accepted_indices, rejected_indices, max_sims.tolist()
