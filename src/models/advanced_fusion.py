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

        # -------- Text Backbone --------
        self.text_model = DistilBertTextClassifier(num_classes=num_classes)
        if text_checkpoint:
            print(f"Loading Text Backbone from {text_checkpoint}")
            self.text_model.load_state_dict(
                torch.load(text_checkpoint, map_location="cpu")
            )

        # -------- Image Backbone --------
        # Initialize with freeze_backbone=False so we can manually control it
        self.image_model = ResNetImageClassifier(
            num_classes=num_classes, freeze_backbone=False
        )
        if image_checkpoint:
            print(f"Loading Image Backbone from {image_checkpoint}")
            self.image_model.load_state_dict(
                torch.load(image_checkpoint, map_location="cpu")
            )

        # Remove classifier head (we only want features)
        self.image_model.backbone.fc = nn.Identity()

        # -------- 1. FREEZE LOGIC --------
        # Freeze everything first
        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.image_model.parameters():
            p.requires_grad = False

        # Unfreeze ONLY specific high-level layers
        # Unfreeze last Transformer Encoder layer
        for p in self.text_model.encoder.transformer.layer[-1].parameters():
            p.requires_grad = True

        # Unfreeze Last ResNet Block (Layer 4)
        for p in self.image_model.backbone.layer4.parameters():
            p.requires_grad = True

        # -------- Projection Layers --------
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

        # -------- Cross-Attention Fusion --------
        # We treat this as a sequence of length 2: [Text, Image]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=common_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # -------- Classifier --------
        # INPUT DIMENSION CHANGE:
        # We concatenate (Original Image 2048) + (Fused Image Token 512) + (Fused Text Token 512)
        # This gives the model a "Safety Valve" to use raw image features if fusion is confusing.
        self.classifier = nn.Sequential(
            nn.Linear(2048 + common_dim * 2, 512),
            nn.BatchNorm1d(512),  # Added BN for stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        # 1. Extract Raw Features
        txt_out = self.text_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        txt_emb = txt_out.last_hidden_state[:, 0, :]  # [B, 768]

        img_raw = self.image_model.backbone(
            images
        )  # [B, 2048] (This is the strong baseline feature)

        # 2. Project to Common Space
        txt_proj = self.text_projector(txt_emb)  # [B, 512]
        img_proj = self.image_projector(img_raw)  # [B, 512]

        # 3. Cross Attention Fusion
        # Stack as sequence: [Text_Token, Image_Token]
        fusion_input = torch.stack([txt_proj, img_proj], dim=1)  # [B, 2, 512]
        fusion_out = self.fusion_encoder(fusion_input)  # [B, 2, 512]

        # 4. Flatten Fusion Output
        # shape becomes [B, 1024] (Text_Fused + Image_Fused)
        fusion_flat = fusion_out.reshape(fusion_out.size(0), -1)

        # 5. RESIDUAL CONNECTION (The "Secret Sauce")
        # Concatenate: [Raw_Image_Features (2048), Fused_Features (1024)]
        # This guarantees the model has access to the pure image signal
        # even if the text/fusion is noisy.
        combined = torch.cat([img_raw, fusion_flat], dim=1)

        return self.classifier(combined)

    def train(self, mode=True):
        """
        Custom train method to safe-guard Batch Normalization.
        We MUST keep the frozen parts of ResNet in eval mode to preserve
        ImageNet statistics, otherwise small batches (BS=16) destroy the features.
        """
        super().train(mode)

        # Force Backbones to EVAL mode (Freezes BN stats and Dropout)
        # We only want to train the weights of Layer4, but keep BN stats frozen.
        self.image_model.backbone.eval()
        self.text_model.encoder.eval()

        # Note: If you want Dropout active in backbones, you can granularly set it,
        # but for small-batch fine-tuning, full eval on backbones is safer.

        return self
