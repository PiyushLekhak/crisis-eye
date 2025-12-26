import torch
import torch.nn as nn
from torchvision import models


class ResNetImageClassifier(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=True):
        super().__init__()

        # Load pretrained ResNet50 backbone
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=weights)

        # Replace final FC layer first
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Store freeze flag
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Explicitly unfreeze classifier head (clarity)
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

            # Keep backbone in eval mode to freeze BN/Dropout statistics
            self.backbone.eval()

    def forward(self, pixel_values):
        return self.backbone(pixel_values)

    def train(self, mode: bool = True):
        """
        Override .train() to ensure backbone stays in eval mode if frozen.
        This prevents BatchNorm/Dropout layers from updating statistics.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
