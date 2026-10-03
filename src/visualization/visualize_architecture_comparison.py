"""
Visual Comparison of Generative Architectures.
Generates side-by-side visual panels comparing:
1. Real reference facial expression crops.
2. Conditional VAE samples (blurriness artifact demonstration).
3. Vanilla Conditional GAN samples (mode collapse / geometric distortion demonstration).
4. Proposed Adaptive Quality-Controlled cWGAN-GP samples (crisp facial features & expression fidelity).
"""

import os
import sys
import glob
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def generate_architecture_comparison_panel(
    real_sample_dir: str = r"d:\Projects\GAN\results\figures",
    vae_dir: str = r"d:\Projects\GAN\data\synthetic_vae\contempt",
    cgan_dir: str = r"d:\Projects\GAN\data\synthetic_cgan\contempt",
    cwgan_filtered_dir: str = r"d:\Projects\GAN\data\synthetic\contempt",
    output_fig_path: str = r"d:\Projects\GAN\results\figures\architecture_comparison_faces.png",
    num_samples: int = 6
):
    """
    Builds a 4-row comparative visual figure displaying real vs VAE vs cGAN vs Proposed cWGAN-GP.
    """
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)

    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2.2 + 1.5, 9.5), dpi=300)

    row_titles = [
        "(a) Real Ground-Truth\n(Reference Faces)",
        "(b) Conditional VAE\n(L2 Blur Artifact)",
        "(c) Vanilla cGAN\n(Mode Collapse)",
        "(d) Proposed cWGAN-GP\n(Filtered & Diverse)"
    ]

    # Collect images
    image_sources = [
        # 1. Real samples (load from pilot manifest crops)
        _get_real_pilot_images(num_samples, target_class=1),
        # 2. VAE samples
        _load_images_from_dir(vae_dir, num_samples),
        # 3. cGAN samples
        _load_images_from_dir(cgan_dir, num_samples),
        # 4. Filtered cWGAN-GP samples
        _load_images_from_dir(cwgan_filtered_dir, num_samples)
    ]

    for row_idx, (title, img_list) in enumerate(zip(row_titles, image_sources)):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            if col_idx < len(img_list) and img_list[col_idx] is not None:
                ax.imshow(img_list[col_idx])
            else:
                ax.text(0.5, 0.5, "Generating...", ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx == 0:
                ax.set_ylabel(title, fontsize=10, fontweight="bold", labelpad=12, rotation=0, ha="right", va="center")

    plt.suptitle("Qualitative Visual Comparison Across Generative Architectures (Minority Class: Contempt)",
                 fontsize=13, fontweight="bold", y=0.96)
    plt.tight_layout(rect=[0.12, 0.02, 1, 0.94])
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Architecture comparison panel saved to: {output_fig_path}")


def _load_images_from_dir(directory: str, n: int):
    """Loads up to n images from directory."""
    if not os.path.exists(directory):
        return [None] * n
    files = sorted(glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.jpg")))
    imgs = []
    for f in files[:n]:
        try:
            imgs.append(Image.open(f).convert("RGB").resize((64, 64)))
        except Exception:
            imgs.append(None)
    while len(imgs) < n:
        imgs.append(None)
    return imgs


def _get_real_pilot_images(n: int, target_class: int = 1):
    """Loads real cropped faces from pilot manifest for target class."""
    manifest_path = r"d:\Projects\GAN\data\splits\pilot_manifest.csv"
    if not os.path.exists(manifest_path):
        return [None] * n
    from src.data.dataset import FacialExpressionDataset
    from src.data.preprocessing import tensor_to_pil
    ds = FacialExpressionDataset(manifest=manifest_path, image_size=64)
    imgs = []
    for i in range(len(ds)):
        t_img, t_lbl = ds[i]
        if t_lbl.item() == target_class:
            imgs.append(tensor_to_pil(t_img))
            if len(imgs) == n:
                break
    while len(imgs) < n:
        imgs.append(None)
    return imgs


if __name__ == "__main__":
    generate_architecture_comparison_panel()
