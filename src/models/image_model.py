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

            # Unfreeze last residual stage (layer4) and the classifier head
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

            for param in self.backbone.fc.parameters():
                param.requires_grad = True

            # Do NOT call backbone.eval() here globally — control eval/train state in .train()
            # This ensures layer4's BatchNorm layers can be set to train() when desired.

    def forward(self, pixel_values):
        return self.backbone(pixel_values)

    def train(self, mode: bool = True):
        """
        Override .train() to:
        - keep frozen backbone layers in eval() so their BatchNorm stats don't update
        - allow layer4 and fc to be set to train(mode) so they can learn when training
        """
        super().train(mode)
        if self.freeze_backbone:
            # Put entire backbone in eval to freeze BatchNorm / Dropout behavior for frozen parts
            self.backbone.eval()

            # Explicitly set layer4 to train mode if overall mode is True (so its BN updates)
            # and set fc to train mode as well.
            for m in self.backbone.layer4.modules():
                m.train(mode)
            for m in self.backbone.fc.modules():
                m.train(mode)
        return self
