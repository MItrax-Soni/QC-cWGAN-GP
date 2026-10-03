"""
Quantitative Classification Evaluation Metrics and Confusion Matrix Engine.
Calculates Accuracy, Precision, Recall, F1, Macro F1, and Minority-Class Recall.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.yolo_parser import CLASS_NAMES, id_to_class_name


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minority_classes: List[int] = [1, 2, 7]  # contempt (1), disgust (2), sleepy (7)
) -> Dict[str, Any]:
    """
    Computes overall accuracy, macro F1, weighted F1, per-class recall, and minority recall.
    """
    num_classes = len(CLASS_NAMES)
    total_samples = len(y_true)
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion matrix: rows = true, cols = predicted
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    supports = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = np.sum(cm[c, :])
        supports[c] = int(support)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        cname = id_to_class_name(c)
        per_class_precision[cname] = round(prec, 4)
        per_class_recall[cname] = round(rec, 4)
        per_class_f1[cname] = round(f1, 4)

    # Macro averages
    macro_f1 = float(np.mean(list(per_class_f1.values())))
    macro_recall = float(np.mean(list(per_class_recall.values())))

    # Weighted F1
    weighted_f1 = float(sum(per_class_f1[id_to_class_name(c)] * supports[c] for c in range(num_classes)) / total_samples)

    # Minority recall (strictly defined across specified minority classes)
    minority_names = [id_to_class_name(c) for c in minority_classes]
    minority_recalls = [per_class_recall[name] for name in minority_names]
    minority_recall_mean = float(np.mean(minority_recalls))

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_recall": round(macro_recall, 4),
        "minority_recall": round(minority_recall_mean, 4),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True
) -> None:
    """
    Plots a publication-quality confusion matrix heatmap.
    """
    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    else:
        cm_norm = cm

    plt.figure(figsize=(9, 8), dpi=300)
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=12, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=9)

    fmt = ".2f" if normalize else "d"
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            color = "white" if val > thresh else "black"
            text_str = f"{val:{fmt}}" if normalize else f"{val}"
            plt.text(j, i, text_str, ha="center", va="center", color=color, fontsize=8)

    plt.ylabel("True Emotion Label", fontsize=11, labelpad=8)
    plt.xlabel("Predicted Emotion Label", fontsize=11, labelpad=8)
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")
