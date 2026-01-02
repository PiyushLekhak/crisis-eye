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
        # CHANGE: BatchNorm1d → LayerNorm for stability with small batches
        self.text_projector = nn.Sequential(
            nn.Linear(768, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.image_projector = nn.Sequential(
            nn.Linear(2048, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -------- 4. The Gating Mechanism (The "Brain") --------
        self.gate_layer = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, 1),
            nn.Sigmoid(),
        )

        # -------- 5. Classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # NEW: Auxiliary classifiers for unimodal losses
        self.text_aux_classifier = nn.Linear(common_dim, num_classes)
        self.image_aux_classifier = nn.Linear(common_dim, num_classes)

        # Store auxiliary outputs for loss computation
        self.text_logits = None
        self.image_logits = None

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
        concat_features = torch.cat([txt_proj, img_proj], dim=1)  # [B, 1024]
        z = self.gate_layer(concat_features)  # [B, 1]

        # CHANGE: Affine transform instead of clamp to preserve gradients
        z = 0.8 * z + 0.1

        # 4. Apply Weighted Fusion
        fused_features = (z * txt_proj) + ((1 - z) * img_proj)  # [B, 512]

        # 5. Classify
        logits = self.classifier(fused_features)

        # NEW: Store auxiliary logits for loss computation (only during training)
        if self.training:
            self.text_logits = self.text_aux_classifier(txt_proj)
            self.image_logits = self.image_aux_classifier(img_proj)

        return logits

    def compute_loss(self, logits, labels, criterion):
        """
        Computes total loss including auxiliary unimodal losses.
        This method is called automatically if model is in training mode.

        Args:
            logits: Main fusion output
            labels: Ground truth labels
            criterion: Loss function (e.g., CrossEntropyLoss)

        Returns:
            Total loss with auxiliary losses (α = 0.1)
        """
        loss_fusion = criterion(logits, labels)

        if (
            self.training
            and self.text_logits is not None
            and self.image_logits is not None
        ):
            loss_text = criterion(self.text_logits, labels)
            loss_image = criterion(self.image_logits, labels)

            # Combined loss with auxiliary losses (α = 0.1)
            total_loss = loss_fusion + 0.1 * loss_text + 0.1 * loss_image

            # Clear cached logits
            self.text_logits = None
            self.image_logits = None

            return total_loss

        return loss_fusion

    def get_optimizer_params(self, base_lr=2e-5):
        """
        Returns parameter groups with different learning rates.

        Args:
            base_lr: Base learning rate (used for smallest components)

        Returns:
            List of parameter groups for optimizer
        """
        return [
            {
                "params": list(self.text_projector.parameters())
                + list(self.image_projector.parameters())
                + list(self.gate_layer.parameters())
                + list(self.classifier.parameters())
                + list(self.text_aux_classifier.parameters())
                + list(self.image_aux_classifier.parameters()),
                "lr": base_lr * 25,  # 5e-4 if base_lr = 2e-5
            },
            {
                "params": self.image_model.backbone.layer4.parameters(),
                "lr": base_lr * 5,  # 1e-4 if base_lr = 2e-5
            },
            {
                "params": self.text_model.encoder.transformer.layer[-1].parameters(),
                "lr": base_lr,  # 2e-5
            },
        ]

    def train(self, mode=True):
        super().train(mode)

        # Freeze frozen parts
        self.image_model.backbone.eval()
        self.text_model.encoder.eval()

        # Re-enable training for unfrozen layers
        self.text_model.encoder.transformer.layer[-1].train()
        self.image_model.backbone.layer4.train()

        return self
