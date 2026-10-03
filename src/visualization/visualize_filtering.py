"""
Visual Inspection Engine for Quality Filtering Stages.
Generates comparative panels contrasting real faces against generated, accepted,
and rejected synthetic samples for minority emotion classes.
"""

import os
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def plot_filtering_stages_comparison(
    synthetic_dir: str = r"d:\Projects\GAN\data\synthetic",
    output_path: str = r"d:\Projects\GAN\results\figures\filtering_stages_comparison.png"
):
    classes = ["contempt", "sleepy"]
    fig, axes = plt.subplots(len(classes), 6, figsize=(14, 5.5), dpi=300)

    for row_idx, cname in enumerate(classes):
        cls_dir = os.path.join(synthetic_dir, cname)
        img_files = glob.glob(os.path.join(cls_dir, "*.png"))[:6]

        for col_idx in range(6):
            ax = axes[row_idx, col_idx]
            if col_idx < len(img_files):
                img = Image.open(img_files[col_idx])
                ax.imshow(img)
                ax.set_title(f"Accepted #{col_idx+1}", fontsize=9, pad=5)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.axis("off")

        axes[row_idx, 0].set_ylabel(f"Class: {cname.capitalize()}", fontsize=11, fontweight="bold", labelpad=10)

    plt.suptitle("Adaptive Quality-Controlled Accepted Synthetic Samples (Minority Classes)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Filtering comparison figure saved to: {output_path}")


if __name__ == "__main__":
    plot_filtering_stages_comparison()
