"""
Stage 3: Duplicate and Perceptual Hash Quality Filtering.
Employs perceptual hashing (pHash) to detect exact and near-duplicates against
real training faces and previously accepted synthetic samples.
"""

from typing import Tuple, List, Set
import torch
from PIL import Image
import imagehash

from src.data.preprocessing import tensor_to_pil


class DuplicateFilter:
    """
    Computes perceptual hashes and filters near-duplicate faces using Hamming distance.
    """
    def __init__(self, max_hamming_distance: int = 4):
        self.max_hamming_distance = max_hamming_distance
        self.existing_hashes: List[imagehash.ImageHash] = []

    def register_reference_images(self, images: List[Image.Image]) -> None:
        """Computes and registers perceptual hashes for real baseline images."""
        for img in images:
            h = imagehash.phash(img)
            self.existing_hashes.append(h)

    def filter_candidates(
        self,
        candidate_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters candidates against existing hashes and newly accepted hashes.
        Args:
            candidate_images: Tensor of shape (K, 3, 64, 64) in [-1, 1].
        Returns:
            accepted_indices: Tensor of unique candidate indices.
            rejected_indices: Tensor of duplicate candidate indices.
        """
        accepted = []
        rejected = []
        accepted_hashes = list(self.existing_hashes)

        for idx in range(candidate_images.size(0)):
            img_tensor = candidate_images[idx]
            pil_img = tensor_to_pil(img_tensor)
            h = imagehash.phash(pil_img)

            # Check Hamming distance against all registered and accepted hashes
            is_duplicate = False
            for ref_h in accepted_hashes:
                if (h - ref_h) <= self.max_hamming_distance:
                    is_duplicate = True
                    break

            if is_duplicate:
                rejected.append(idx)
            else:
                accepted.append(idx)
                accepted_hashes.append(h)

        return (
            torch.tensor(accepted, dtype=torch.long),
            torch.tensor(rejected, dtype=torch.long)
        )
