"""
PyTorch Dataset and DataLoader for Facial Expression Recognition.
Implements lazy-loading from the dataset archive with on-the-fly face cropping,
resizing to 64x64, and [-1, 1] normalization.
"""

import os
import sys
import io
import zipfile
from typing import Optional, Union, Tuple
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.yolo_parser import yolo_to_pixel_box
from src.data.preprocessing import crop_face_region, get_gan_transforms


class FacialExpressionDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for facial expressions.
    Reads image files directly from zip archive or filesystem, crops face region
    from YOLO coordinates, and produces normalized tensors in [-1, 1].
    """
    def __init__(
        self,
        manifest: Union[str, pd.DataFrame],
        zip_path: str = r"C:\Users\mitra\Downloads\archive.zip",
        image_size: int = 64,
        margin_ratio: float = 0.05,
        transform = None
    ):
        if isinstance(manifest, str):
            if not os.path.exists(manifest):
                raise FileNotFoundError(f"Manifest CSV not found: {manifest}")
            self.df = pd.read_csv(manifest)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        else:
            raise TypeError("Manifest must be a file path string or pandas DataFrame.")

        self.zip_path = zip_path
        self.image_size = image_size
        self.margin_ratio = margin_ratio
        self.transform = transform if transform is not None else get_gan_transforms(image_size)
        
        # Zip handle cache per process
        self._zip_file: Optional[zipfile.ZipFile] = None

    def _get_zip(self) -> zipfile.ZipFile:
        """Maintains a worker-local ZipFile instance."""
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path, "r")
        return self._zip_file

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        class_id = int(row["class_id"])
        xc, yc, w, h = float(row["xc"]), float(row["yc"]), float(row["w"]), float(row["h"])

        # Load image bytes from zip
        z = self._get_zip()
        img_bytes = z.read(img_path)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = img.size

        # Convert YOLO box to pixel box and crop
        box = yolo_to_pixel_box(xc, yc, w, h, img_w, img_h, clamp=True)
        face_img = crop_face_region(img, box, margin_ratio=self.margin_ratio)

        # Apply transformation: Resize -> ToTensor -> Normalize([-1, 1])
        img_tensor = self.transform(face_img)
        label_tensor = torch.tensor(class_id, dtype=torch.long)

        return img_tensor, label_tensor

    def __del__(self):
        if self._zip_file is not None:
            try:
                self._zip_file.close()
            except Exception:
                pass


def get_dataloader(
    manifest: Union[str, pd.DataFrame],
    zip_path: str = r"C:\Users\mitra\Downloads\archive.zip",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    image_size: int = 64,
    margin_ratio: float = 0.05,
    transform = None,
    drop_last: bool = True
) -> DataLoader:
    """
    Factory function to construct a configured PyTorch DataLoader.
    """
    dataset = FacialExpressionDataset(
        manifest=manifest,
        zip_path=zip_path,
        image_size=image_size,
        margin_ratio=margin_ratio,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    return loader
