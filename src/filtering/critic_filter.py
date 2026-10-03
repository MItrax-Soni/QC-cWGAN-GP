"""
Stage 1: Critic-Score Quality Filtering.
Filters candidate synthetic images using validation-derived, class-specific Wasserstein score distributions.
Does not treat raw critic scores as calibrated probabilities.
"""

from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn


class CriticFilter:
    """
    Computes class-specific score thresholds using validation data and filters
    low-fidelity synthetic candidate faces.
    """
    def __init__(self, critic: nn.Module, quantile: float = 0.25, device: torch.device = torch.device("cpu")):
        self.critic = critic
        self.quantile = quantile
        self.device = device
        self.class_thresholds: Dict[int, float] = {}

    def calibrate_thresholds(self, val_loader: torch.utils.data.DataLoader) -> Dict[int, float]:
        """
        Computes validation Wasserstein score distributions per class and selects
        the class threshold at the specified quantile (e.g. 25th percentile of real scores).
        """
        self.critic.eval()
        scores_by_class: Dict[int, List[float]] = {}

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                scores = self.critic(imgs, labels).squeeze(-1).cpu().tolist()

                for score, lbl in zip(scores, labels.cpu().tolist()):
                    if lbl not in scores_by_class:
                        scores_by_class[lbl] = []
                    scores_by_class[lbl].append(score)

        # Set threshold per class
        for cid, scores in scores_by_class.items():
            if len(scores) > 0:
                self.class_thresholds[cid] = float(np.percentile(scores, self.quantile * 100))
            else:
                self.class_thresholds[cid] = 0.0

        return self.class_thresholds

    def filter_candidates(
        self,
        candidate_images: torch.Tensor,
        class_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], float]:
        """
        Filters candidate synthetic images for a specific emotion class.
        Args:
            candidate_images: Tensor of shape (N, 3, 64, 64) in [-1, 1].
            class_id: Target class label.
        Returns:
            accepted_indices: Tensor of accepted sample indices.
            rejected_indices: Tensor of rejected sample indices.
            candidate_scores: List of all candidate scores.
            threshold: Applied critic score threshold.
        """
        self.critic.eval()
        N = candidate_images.size(0)
        labels = torch.full((N,), class_id, dtype=torch.long, device=self.device)

        with torch.no_grad():
            scores = self.critic(candidate_images.to(self.device), labels).squeeze(-1).cpu()

        threshold = self.class_thresholds.get(class_id, 0.0)
        accepted_mask = scores >= threshold
        
        # Adaptive Fallback: If strict validation threshold filters 100% of candidates 
        # (common in early or pilot training checkpoints), adapt to the top candidate percentile
        # to ensure downstream filtering stages remain active and verifiable.
        if not accepted_mask.any() and len(scores) > 0:
            adaptive_thresh = float(np.percentile(scores.numpy(), 50.0))  # Top 50% of candidates
            accepted_mask = scores >= adaptive_thresh
            threshold = adaptive_thresh

        accepted_indices = torch.where(accepted_mask)[0]
        rejected_indices = torch.where(~accepted_mask)[0]

        return accepted_indices, rejected_indices, scores.tolist(), threshold
