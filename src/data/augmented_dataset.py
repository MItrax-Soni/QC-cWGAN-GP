"""
Augmented Dataset Engine.
Combines real training images with accepted synthetic samples while preserving
validation and test sets completely untouched to prevent data leakage.
"""

import os
import sys
import pandas as pd
from typing import Optional, Union, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.dataset import FacialExpressionDataset
from src.data.preprocessing import get_classifier_transforms


class CombinedAugmentedDataset(Dataset):
    """
    Unified dataset supporting both zip-backed real images and disk-backed synthetic images.
    """
    def __init__(
        self,
        real_manifest_path: str,
        synthetic_manifest_path: Optional[str] = None,
        zip_path: str = r"C:\Users\mitra\Downloads\archive.zip",
        image_size: int = 64,
        is_train: bool = True
    ):
        self.real_dataset = FacialExpressionDataset(
            manifest=real_manifest_path,
            zip_path=zip_path,
            image_size=image_size
        )
        self.synthetic_df = None
        if synthetic_manifest_path and os.path.exists(synthetic_manifest_path):
            df_syn = pd.read_csv(synthetic_manifest_path)
            if len(df_syn) > 0:
                self.synthetic_df = df_syn.reset_index(drop=True)

        self.transform = get_classifier_transforms(image_size=image_size, is_train=is_train)
        self.real_len = len(self.real_dataset)
        self.syn_len = len(self.synthetic_df) if self.synthetic_df is not None else 0

    def __len__(self) -> int:
        return self.real_len + self.syn_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < self.real_len:
            # Load real image
            img_tensor, label_tensor = self.real_dataset[idx]
            # Convert [-1, 1] tensor back to PIL then apply classifier transform
            from src.data.preprocessing import tensor_to_pil
            pil_img = tensor_to_pil(img_tensor)
            return self.transform(pil_img), label_tensor
        else:
            # Load synthetic image from disk
            syn_idx = idx - self.real_len
            row = self.synthetic_df.iloc[syn_idx]
            img_path = row["image_path"]
            class_id = int(row["class_id"])
            pil_img = Image.open(img_path).convert("RGB")
            return self.transform(pil_img), torch.tensor(class_id, dtype=torch.long)


def get_augmented_dataloader(
    real_manifest_path: str,
    synthetic_manifest_path: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    is_train: bool = True
) -> DataLoader:
    """Factory helper returning DataLoader for augmented training set."""
    ds = CombinedAugmentedDataset(
        real_manifest_path=real_manifest_path,
        synthetic_manifest_path=synthetic_manifest_path,
        is_train=is_train
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
