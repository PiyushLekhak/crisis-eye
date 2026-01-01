import torch
import torch.nn as nn
from src.models.text_model import DistilBertTextClassifier
from src.models.image_model import ResNetImageClassifier


class AdvancedFusionModel(nn.Module):
    def __init__(
        self,
        num_classes=3,
        text_checkpoint=None,
        image_checkpoint=None,
        common_dim=512,
        dropout=0.3,
    ):
        super().__init__()

        # -------- 1. Backbones (Same as before) --------
        self.text_model = DistilBertTextClassifier(num_classes=num_classes)
        if text_checkpoint:
            print(f"Loading Text Backbone from {text_checkpoint}")
            self.text_model.load_state_dict(
                torch.load(text_checkpoint, map_location="cpu")
            )

        self.image_model = ResNetImageClassifier(
            num_classes=num_classes, freeze_backbone=False
        )
        if image_checkpoint:
            print(f"Loading Image Backbone from {image_checkpoint}")
            self.image_model.load_state_dict(
                torch.load(image_checkpoint, map_location="cpu")
            )

        self.image_model.backbone.fc = nn.Identity()

        # -------- 2. Freeze Logic (Same as before) --------
        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.image_model.parameters():
            p.requires_grad = False

        # Unfreeze last layers for fine-tuning
        for p in self.text_model.encoder.transformer.layer[-1].parameters():
            p.requires_grad = True
        for p in self.image_model.backbone.layer4.parameters():
            p.requires_grad = True

        # -------- 3. Projection Layers --------
        # We must project both to the SAME dimension to weigh them against each other
        self.text_projector = nn.Sequential(
            nn.Linear(768, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.image_projector = nn.Sequential(
            nn.Linear(2048, common_dim),
            nn.BatchNorm1d(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -------- 4. The Gating Mechanism (The "Brain") --------
        # This layer looks at both inputs and decides the weight 'z'
        # Input: Text(512) + Image(512) = 1024
        # Output: 1 scalar value (0 to 1) per sample
        self.gate_layer = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, 1),
            nn.Sigmoid(),  # Forces output between 0 and 1
        )

        # -------- 5. Classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        # 1. Get Raw Features
        txt_out = self.text_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        txt_emb = txt_out.last_hidden_state[:, 0, :]  # [B, 768]

        img_raw = self.image_model.backbone(images)  # [B, 2048]

        # 2. Project to Common Dimension
        txt_proj = self.text_projector(txt_emb)  # [B, 512]
        img_proj = self.image_projector(img_raw)  # [B, 512]

        # 3. Calculate the Gate 'z'
        # We concatenate them to let the gate compare them
        concat_features = torch.cat([txt_proj, img_proj], dim=1)  # [B, 1024]
        z = self.gate_layer(concat_features)  # [B, 1]

        # Stabilize the Gate early in training
        z = torch.clamp(z, 0.1, 0.9)

        # 4. Apply Weighted Fusion
        # If z is close to 1, we trust Text. If z is close to 0, we trust Image.
        # This allows the model to choose per-sample.
        fused_features = (z * txt_proj) + ((1 - z) * img_proj)  # [B, 512]

        # 5. Classify
        logits = self.classifier(fused_features)

        return logits

    def train(self, mode=True):
        super().train(mode)

        # Freeze frozen parts
        self.image_model.backbone.eval()
        self.text_model.encoder.eval()

        # Re-enable training for unfrozen layers
        self.text_model.encoder.transformer.layer[-1].train()
        self.image_model.backbone.layer4.train()

        return self
