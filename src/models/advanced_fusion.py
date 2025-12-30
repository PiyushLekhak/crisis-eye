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
            self.text_model.load_state_dict(
                torch.load(text_checkpoint, map_location="cpu")
            )

        # -------- Image Backbone --------
        self.image_model = ResNetImageClassifier(
            num_classes=num_classes,
            freeze_backbone=False,
        )
        if image_checkpoint:
            self.image_model.load_state_dict(
                torch.load(image_checkpoint, map_location="cpu")
            )

        # Remove classifier head AFTER loading
        self.image_model.backbone.fc = nn.Identity()

        # -------- Freeze everything --------
        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.image_model.parameters():
            p.requires_grad = False

        # -------- Smart unfreezing --------
        for p in self.text_model.encoder.transformer.layer[-1].parameters():
            p.requires_grad = True

        for p in self.image_model.backbone.layer4.parameters():
            p.requires_grad = True

        # -------- Projection layers --------
        self.text_projector = nn.Sequential(
            nn.Linear(768, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
        )

        self.image_projector = nn.Sequential(
            nn.Linear(2048, common_dim),
            nn.LayerNorm(common_dim),
            nn.ReLU(),
        )

        # -------- Cross-attention --------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=common_dim,
            nhead=4,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # -------- Classifier --------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(common_dim, 256),  # Changed the input dim to common_dim
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, images):
        # Text features
        txt_out = self.text_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        txt_emb = txt_out.last_hidden_state[:, 0, :]

        # Image features
        img_emb = self.image_model.backbone(images)

        # Project
        txt_proj = self.text_projector(txt_emb)
        img_proj = self.image_projector(img_emb)

        # Stack [Text, Image]
        fusion_input = torch.stack([txt_proj, img_proj], dim=1)

        # Cross-attention
        fusion_out = self.fusion_encoder(fusion_input)

        # Apply mean pooling across the sequence dimension (dim=1)
        fusion_pooled = fusion_out.mean(dim=1)

        # Pass through the classifier
        return self.classifier(fusion_pooled)
