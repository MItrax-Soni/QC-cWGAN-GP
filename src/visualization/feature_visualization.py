"""
Feature Space Visualizer.
Generates 2D dimensionality reduction plots (t-SNE or PCA) comparing feature distributions
of real facial expression crops versus quality-filtered synthetic samples.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.dataset import FacialExpressionDataset
from src.evaluation.generative_metrics import extract_dataset_embeddings


def generate_tsne_visualization(
    real_manifest_path: str = r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
    synthetic_manifest_path: str = r"d:\Projects\GAN\data\synthetic\synthetic_manifest.csv",
    output_fig_path: str = r"d:\Projects\GAN\results\figures\tsne_feature_space.png",
    max_samples: int = 200
):
    """
    Extracts features and plots 2D projection comparing real and synthetic faces.
    Uses t-SNE if scikit-learn is installed, otherwise uses SVD/PCA.
    """
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)

    print("Extracting features for feature-space visualization...")
    real_ds = FacialExpressionDataset(manifest=real_manifest_path, image_size=64)
    n_real = min(len(real_ds), max_samples)

    real_imgs = []
    real_labels = []
    for i in range(n_real):
        img, lbl = real_ds[i]
        real_imgs.append(img)
        real_labels.append(lbl.item())

    real_tensor = torch.stack(real_imgs, dim=0)
    real_embs = extract_dataset_embeddings(real_tensor)

    # Synthetic samples
    syn_embs = None
    if os.path.exists(synthetic_manifest_path):
        import pandas as pd
        from PIL import Image
        import torchvision.transforms as T

        syn_df = pd.read_csv(synthetic_manifest_path)
        if len(syn_df) > 0:
            syn_imgs = []
            transform = T.Compose([
                T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            for _, row in syn_df.iterrows():
                pil_img = Image.open(row["image_path"]).convert("RGB")
                syn_imgs.append(transform(pil_img))
            syn_tensor = torch.stack(syn_imgs, dim=0)
            syn_embs = extract_dataset_embeddings(syn_tensor)

    # Combine embeddings
    if syn_embs is not None and len(syn_embs) > 0:
        all_embs = np.concatenate([real_embs, syn_embs], axis=0)
        types = ["Real"] * len(real_embs) + ["Synthetic (Quality-Controlled)"] * len(syn_embs)
    else:
        all_embs = real_embs
        types = ["Real"] * len(real_embs)

    # Dimensionality reduction (t-SNE if sklearn available, else PCA via SVD)
    method_name = "t-SNE"
    try:
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, len(all_embs) - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        reduced = tsne.fit_transform(all_embs)
    except ImportError:
        method_name = "PCA"
        centered = all_embs - np.mean(all_embs, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        reduced = np.dot(centered, vt[:2].T)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    real_mask = [t == "Real" for t in types]
    syn_mask = [t != "Real" for t in types]

    ax.scatter(
        reduced[real_mask, 0],
        reduced[real_mask, 1],
        c="#3498db",
        alpha=0.6,
        s=35,
        label="Real Facial Crops",
        edgecolors="none"
    )

    if any(syn_mask):
        ax.scatter(
            reduced[syn_mask, 0],
            reduced[syn_mask, 1],
            c="#e74c3c",
            alpha=0.9,
            s=70,
            marker="*",
            label="Synthetic (Adaptive Quality-Filtered)",
            edgecolors="#c0392b"
        )

    ax.set_title(f"{method_name} Feature Space Distribution (ResNet-18 512-D Latent Space)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=10, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature space visualization saved to: {output_fig_path}")


if __name__ == "__main__":
    generate_tsne_visualization()
