import torch
import torch.nn as nn
from src.models.text_model import DistilBertTextClassifier
from src.models.image_model import ResNetImageClassifier


class LateFusionModel(nn.Module):
    def __init__(
        self,
        num_classes=3,
        text_checkpoint=None,
        image_checkpoint=None,
        freeze_backbones=False,  # Change default to False
    ):
        super(LateFusionModel, self).__init__()

        # --- Text Branch (DistilBERT) ---
        self.text_model = DistilBertTextClassifier(num_classes=num_classes)
        if text_checkpoint:
            print(f"Loading Text weights from {text_checkpoint}")
            self.text_model.load_state_dict(
                torch.load(text_checkpoint, map_location="cpu")
            )

        # --- Image Branch (ResNet) ---
        self.image_model = ResNetImageClassifier(
            num_classes=num_classes, freeze_backbone=freeze_backbones
        )  # Pass freeze_backbones to ResNet
        if image_checkpoint:
            print(f"Loading Image weights from {image_checkpoint}")
            self.image_model.load_state_dict(
                torch.load(image_checkpoint, map_location="cpu")
            )

        # Remove the final classification layer from ResNet to get features (2048 dim)
        self.image_model.backbone.fc = nn.Identity()

        # --- Freeze Pretrained Backbones (CRITICAL for baseline fusion) ---
        if freeze_backbones:
            print("Freezing Text and Image backbones...")
            for param in self.text_model.encoder.parameters():
                param.requires_grad = False

            for param in self.image_model.backbone.parameters():
                param.requires_grad = False

            # Verify trainable params
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"Trainable params: {trainable:,} / {total:,}")

        # --- Fusion Head ---
        # Text [CLS] (768) + Image (2048) = 2816 features
        self.fusion_dim = 768 + 2048

        # Replacing BatchNorm with a simpler approach
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        # 1. Text Features
        # Pass through DistilBERT backbone
        txt_out = self.text_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_features = txt_out.last_hidden_state[:, 0, :]

        # 2. Image Features
        image_features = self.image_model.backbone(images)

        # 3. Concatenate
        combined_features = torch.cat((text_features, image_features), dim=1)

        # 4. Classify
        logits = self.classifier(combined_features)

        return logits
