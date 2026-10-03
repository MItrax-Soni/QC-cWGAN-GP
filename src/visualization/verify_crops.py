"""
Visual Verification Tool for Facial Bounding Box Cropping and Normalization.
Extracts representative face crops across all 9 expression classes and produces
a verified 3x3 reference grid.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.dataset import FacialExpressionDataset
from src.data.preprocessing import denormalize_tensor
from src.data.yolo_parser import id_to_class_name


def verify_and_plot_sample_crops(
    manifest_path: str = r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
    output_path: str = r"d:\Projects\GAN\results\figures\sample_face_crops_grid.png"
):
    print("Loading pilot dataset for visual verification...")
    dataset = FacialExpressionDataset(manifest=manifest_path, image_size=64)
    
    # Collect one sample per class
    samples = {}
    for idx in range(len(dataset)):
        img_tensor, label_tensor = dataset[idx]
        cid = label_tensor.item()
        if cid not in samples:
            samples[cid] = img_tensor
        if len(samples) == 9:
            break
            
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), dpi=300)
    for cid in range(9):
        ax = axes[cid // 3, cid % 3]
        if cid in samples:
            # Denormalize [-1, 1] -> [0, 1]
            img_disp = denormalize_tensor(samples[cid]).permute(1, 2, 0).numpy()
            ax.imshow(img_disp)
        else:
            ax.text(0.5, 0.5, "Missing Class", ha="center", va="center")
            
        cname = id_to_class_name(cid)
        ax.set_title(f"Class {cid}: {cname.capitalize()}", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle("Verified 64x64 Face Crops Across All 9 Classes", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Sample face crops grid saved successfully to: {output_path}")


if __name__ == "__main__":
    verify_and_plot_sample_crops()
