"""
Generative Quality & Diversity Metrics Engine.
Computes empirical feature-space diversity, intra-class variance, and Fréchet distance
on extracted feature representations between real and synthetic face distributions.
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from PIL import Image

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.resnet_classifier import ResNetClassifier
from src.utils.device import get_device


def compute_feature_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes empirical mean vector and covariance matrix of feature embeddings."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Computes the Fréchet distance (FID equivalent) between two Gaussians:
    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*(sigma1*sigma2)^0.5)
    """
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary component check
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_feature_diversity(features: np.ndarray) -> float:
    """
    Computes the mean pairwise Euclidean distance across feature vectors,
    measuring distribution spread (mode diversity).
    """
    if len(features) < 2:
        return 0.0
    from scipy.spatial.distance import pdist
    distances = pdist(features, metric="euclidean")
    return float(np.mean(distances))


def extract_dataset_embeddings(
    image_tensors: torch.Tensor,
    device: torch.device = None
) -> np.ndarray:
    """
    Extracts 512-dimensional normalized embeddings using pretrained ResNet-18 backbone.
    """
    if device is None:
        device = get_device()

    model = ResNetClassifier(num_classes=9, pretrained=True).to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(image_tensors), 32):
            batch = image_tensors[i:i + 32].to(device)
            embs = model.extract_features(batch)
            all_embs.append(embs.cpu().numpy())

    return np.concatenate(all_embs, axis=0)
