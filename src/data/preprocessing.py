"""
Image Preprocessing and Transformations for GAN and Classifier Pipelines.
Normalizes images to [-1, 1] for Tanh generator activation and handles cropping.
"""

from typing import Tuple
from PIL import Image
import torch
import torchvision.transforms as T


def crop_face_region(
    img: Image.Image,
    box: Tuple[int, int, int, int],
    margin_ratio: float = 0.05
) -> Image.Image:
    """
    Crops the face region from a PIL Image using the provided bounding box.
    Adds a configurable margin to preserve context around facial contours.
    """
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    
    img_w, img_h = img.size
    
    # Apply margin if requested
    if margin_ratio > 0.0:
        pad_w = int(width * margin_ratio)
        pad_h = int(height * margin_ratio)
        xmin = max(0, xmin - pad_w)
        ymin = max(0, ymin - pad_h)
        xmax = min(img_w, xmax + pad_w)
        ymax = min(img_h, ymax + pad_h)

    # Fallback if invalid dimensions
    if xmax <= xmin or ymax <= ymin:
        return img

    return img.crop((xmin, ymin, xmax, ymax))


def get_gan_transforms(image_size: int = 64) -> T.Compose:
    """
    Returns standard transforms for GAN training:
    Resizes cropped face to (image_size, image_size), converts to tensor,
    and normalizes RGB channels to [-1, 1].
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a normalized tensor in [-1, 1] back to [0, 1].
    Works on both (C, H, W) and (B, C, H, W).
    """
    denorm = tensor * 0.5 + 0.5
    return torch.clamp(denorm, 0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a single (3, H, W) PyTorch tensor in [-1, 1] to a PIL Image.
    """
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C, H, W), got shape: {tensor.shape}")

    denorm = denormalize_tensor(tensor.detach().cpu())
    to_pil = T.ToPILImage()
    return to_pil(denorm)


def get_classifier_transforms(image_size: int = 64, is_train: bool = True) -> T.Compose:
    """
    Returns image transformation pipeline for downstream ResNet-18 classifier.
    """
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
