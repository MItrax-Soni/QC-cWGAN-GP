"""
ResNet-18 Classifier Training and Evaluation Pipeline.
Trains downstream emotion classifiers under different augmentation regimes and evaluates
generalization performance on held-out test data.
"""

import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.device import get_device
from src.models.resnet_classifier import ResNetClassifier
from src.data.augmented_dataset import get_augmented_dataloader
from src.data.dataset import get_dataloader
from src.evaluation.classification_metrics import calculate_classification_metrics, plot_confusion_matrix


def train_and_evaluate_classifier(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    experiment_name: str = "baseline_classifier",
    epochs: int = 15,
    lr: float = 0.0005,
    device: Optional[torch.device] = None,
    save_dir: str = r"d:\Projects\GAN\results"
) -> Dict[str, Any]:
    """
    Trains ResNet-18 classifier, tracks validation performance, and evaluates on test data.
    """
    if device is None:
        device = get_device()

    print(f"\n=======================================================")
    print(f"Training Downstream Classifier: '{experiment_name}'")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr}")
    print(f"=======================================================\n")

    model = ResNetClassifier(num_classes=9, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    # 1. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += imgs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation phase
        val_acc, val_f1 = _evaluate_validation(model, val_loader, device)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    # 2. Load best checkpoint and evaluate on held-out test split
    model.load_state_dict(best_weights)
    test_metrics = evaluate_classifier_on_test(model, test_loader, device)

    # 3. Save Confusion Matrix figure
    cm_fig_path = os.path.join(save_dir, "figures", f"confusion_matrix_{experiment_name}.png")
    plot_confusion_matrix(
        cm=np.array(test_metrics["confusion_matrix"]),
        output_path=cm_fig_path,
        title=f"Confusion Matrix ({experiment_name})"
    )

    print(f"\n[Test Results: {experiment_name}]")
    print(f"Overall Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro F1 Score:   {test_metrics['macro_f1']:.4f}")
    print(f"Minority Recall:  {test_metrics['minority_recall']*100:.2f}%\n")

    return test_metrics


def _evaluate_validation(model: nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device):
    """Computes validation accuracy and macro F1."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(lbls.tolist())
            y_pred.extend(preds.cpu().tolist())

    metrics = calculate_classification_metrics(np.array(y_true), np.array(y_pred))
    return metrics["accuracy"], metrics["macro_f1"]


def evaluate_classifier_on_test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluates classifier on test split and computes comprehensive metrics."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(lbls.tolist())
            y_pred.extend(preds.cpu().tolist())

    return calculate_classification_metrics(np.array(y_true), np.array(y_pred))
