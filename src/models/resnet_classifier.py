"""
ResNet-18 Facial Expression Classifier & Feature Extractor.
Used for downstream classification evaluation across original and augmented datasets,
and as a deep feature extractor for Stage 2 similarity filtering.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetClassifier(nn.Module):
    """
    ResNet-18 adapted for 9-class facial expression recognition and feature extraction.
    """
    def __init__(self, num_classes: int = 9, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Load ResNet-18 with weights handling
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            # Fallback to uninitialized weights if offline
            self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features  # 512
        # Replace final classification head
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning 9-class raw classification logits.
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 512-dimensional penultimate feature embeddings for Stage 2 similarity filtering.
        """
        # Pass through conv layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)  # (batch_size, 512)
        return features
