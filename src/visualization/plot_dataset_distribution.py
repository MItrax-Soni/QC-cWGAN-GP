"""
Dataset Distribution Plotting Tool
Generates publication-quality charts for the 9 Facial Expressions dataset distribution.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_class_distribution(audit_path: str, output_path: str):
    with open(audit_path, "r") as f:
        data = json.load(f)
        
    train_counts = data["splits"]["train"]["class_counts"]
    classes = list(train_counts.keys())
    counts = list(train_counts.values())
    total = sum(counts)
    percentages = [c / total * 100 for c in counts]
    
    # Palette highlighting majority vs minority
    mean_count = total / len(classes)
    colors = []
    for c in counts:
        if c < mean_count * 0.5:
            colors.append("#e74c3c")  # Severe minority (red)
        elif c < mean_count:
            colors.append("#e67e22")  # Moderate minority (orange)
        else:
            colors.append("#2980b9")  # Majority (blue)

    plt.figure(figsize=(10, 6), dpi=300)
    bars = plt.bar(classes, counts, color=colors, edgecolor="#2c3e50", linewidth=1.2, alpha=0.9)
    
    # Add count and percentage labels above bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 150,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold"
        )
        
    plt.axhline(mean_count, color="#7f8c8d", linestyle="--", linewidth=1.2, label=f"Mean Class Count ({int(mean_count):,})")
    
    plt.title("9 Facial Expressions for YOLO - Training Set Long-Tailed Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Facial Expression Class", fontsize=11, labelpad=10)
    plt.ylabel("Number of Samples", fontsize=11, labelpad=10)
    plt.ylim(0, max(counts) * 1.18)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Class distribution chart saved to: {output_path}")


if __name__ == "__main__":
    audit_file = r"d:\Projects\GAN\results\reports\dataset_audit.json"
    fig_file = r"d:\Projects\GAN\results\figures\class_distribution.png"
    plot_class_distribution(audit_file, fig_file)
