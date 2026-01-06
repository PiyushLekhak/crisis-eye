import torch
import torch.nn as nn
from src.models.text_model import DistilBertTextClassifier
from src.models.image_model import ResNetImageClassifier


class LateFusionModel(nn.Module):
    """
    Advanced late-fusion that:
      - projects text (768) and image (2048) to a common_dim
      - uses a learned gating scalar per sample to weight modalities
      - includes auxiliary unimodal classifiers and auxiliary losses
      - exposes get_optimizer_params() for differential LRs
      - respects the internal fair-freezing implemented in the backbone classes
    """

    def __init__(
        self,
        num_classes=3,
        text_checkpoint=None,
        image_checkpoint=None,
        common_dim=512,
        dropout=0.3,
    ):
        super().__init__()

        # -------- 1. Backbones (respect their internal freeze policy) --------
        # DistilBertTextClassifier: already freezes all but last transformer layer + head
        self.text_model = DistilBertTextClassifier(num_classes=num_classes)
        if text_checkpoint:
            print(f"Loading Text Backbone from {text_checkpoint}")
            self.text_model.load_state_dict(
                torch.load(text_checkpoint, map_location="cpu")
            )

        # ResNetImageClassifier: pass freeze_backbone=True so its constructor
        # applies the fair-freezing policy (layer4 + fc trainable).
        self.image_model = ResNetImageClassifier(
            num_classes=num_classes, freeze_backbone=True
        )
        if image_checkpoint:
            print(f"Loading Image Backbone from {image_checkpoint}")
            self.image_model.load_state_dict(
                torch.load(image_checkpoint, map_location="cpu")
            )

        # Replace ResNet fc with identity to obtain 2048-d features
        self.image_model.backbone.fc = nn.Identity()

        # -------- 2. Projection layers to common_dim (LayerNorm for stability) --------
        self.common_dim = common_dim
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

        # -------- 3. Gating mechanism (learned scalar per sample) --------
        # Input: concat(txt_proj, img_proj) -> scalar in (0,1)
        self.gate_layer = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.ReLU(),
            nn.Linear(common_dim, 1),
            nn.Sigmoid(),
        )

        # Affine shift/clipping to avoid z near 0/1 if desired
        # We'll apply a small affine after sigmoid in forward: z = 0.8*z + 0.1

        # -------- 4. Classifier (use LayerNorm for stability) --------
        self.classifier = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # -------- 5. Auxiliary unimodal classifiers --------
        self.text_aux_classifier = nn.Linear(common_dim, num_classes)
        self.image_aux_classifier = nn.Linear(common_dim, num_classes)

        # storage for aux logits during training
        self._text_logits = None
        self._image_logits = None

    def forward(self, input_ids, attention_mask, images):
        # 1) Get raw features from backbones
        txt_out = self.text_model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        txt_emb = txt_out.last_hidden_state[:, 0, :]  # [B, 768]

        img_raw = self.image_model.backbone(images)  # [B, 2048]

        # 2) Project both to common_dim
        txt_proj = self.text_projector(txt_emb)  # [B, common_dim]
        img_proj = self.image_projector(img_raw)  # [B, common_dim]

        # 3) Gate
        concat_feats = torch.cat([txt_proj, img_proj], dim=1)  # [B, 2*common_dim]
        z = self.gate_layer(concat_feats)  # [B,1]
        # small affine to avoid exact 0/1 (helps gradients/stability)
        z = 0.8 * z + 0.1  # now in [0.1,0.9]

        # 4) Weighted fusion
        fused = (z * txt_proj) + ((1.0 - z) * img_proj)  # [B, common_dim]

        # 5) Classify
        logits = self.classifier(fused)

        # Store auxiliary logits during training for compute_loss
        if self.training:
            self._text_logits = self.text_aux_classifier(txt_proj)
            self._image_logits = self.image_aux_classifier(img_proj)

        return logits

    def compute_loss(
        self,
        logits,
        labels,
        aux_labels_text,
        aux_labels_image,
        criterion,
        aux_alpha=0.1,
    ):

        loss_fusion = criterion(logits, labels)

        if self.training:
            total_aux = 0.0

            # Text auxiliary loss (uses TEXT labels)
            if self._text_logits is not None:
                loss_text = criterion(self._text_logits, aux_labels_text)
                total_aux += loss_text

            # Image auxiliary loss (uses IMAGE labels)
            if self._image_logits is not None:
                if (aux_labels_image != -100).any():  # avoid all-ignore batch
                    loss_image = criterion(self._image_logits, aux_labels_image)
                    total_aux += loss_image

            # clear stored aux logits
            self._text_logits = None
            self._image_logits = None

            return loss_fusion + aux_alpha * total_aux

        return loss_fusion

    def get_optimizer_params(self, base_lr=2e-5):
        """
        Return parameter groups for optimizer with differential LRs:
          - fusion parts (projectors, gate, classifier, aux heads): high LR
          - image layer4 params: medium LR
          - text last layer params: low LR
        This function assumes the backbones expose:
          - self.image_model.backbone.layer4
          - self.text_model.encoder.transformer.layer[-1] (or equivalent)
        """
        # collect fusion params
        fusion_params = (
            list(self.text_projector.parameters())
            + list(self.image_projector.parameters())
            + list(self.gate_layer.parameters())
            + list(self.classifier.parameters())
            + list(self.text_aux_classifier.parameters())
            + list(self.image_aux_classifier.parameters())
        )

        # text last-layer params (try few attribute layouts)
        text_last = None
        if hasattr(self.text_model.encoder, "transformer") and hasattr(
            self.text_model.encoder.transformer, "layer"
        ):
            text_last = self.text_model.encoder.transformer.layer[-1].parameters()
        elif hasattr(self.text_model.encoder, "distilbert") and hasattr(
            self.text_model.encoder.distilbert, "transformer"
        ):
            text_last = self.text_model.encoder.distilbert.transformer.layer[
                -1
            ].parameters()
        else:
            text_last = []

        # image last block
        image_last = (
            self.image_model.backbone.layer4.parameters()
            if hasattr(self.image_model.backbone, "layer4")
            else []
        )

        return [
            {"params": fusion_params, "lr": base_lr * 10},
            {"params": image_last, "lr": base_lr * 5},
            {"params": text_last, "lr": base_lr},
        ]

    def train(self, mode: bool = True):
        """
        Ensure frozen parts remain in eval() (so BN stats don't update),
        while last encoder blocks remain trainable and in train() mode.
        """
        super().train(mode)

        # Put backbone base parts in eval (this leaves their requires_grad as set by backbone constructors)
        # Then explicitly set last blocks to train mode so their BN layers update.
        try:
            # image: set whole backbone eval, then layer4 to train
            self.image_model.backbone.eval()
            if hasattr(self.image_model.backbone, "layer4"):
                for m in self.image_model.backbone.layer4.modules():
                    m.train(mode)
            # text: encoder typically uses LayerNorm (stateless) but we still set encoder eval to be safe
            if hasattr(self.text_model, "encoder"):
                try:
                    self.text_model.encoder.eval()
                    # set last transformer layer to train (if present)
                    if hasattr(self.text_model.encoder, "transformer") and hasattr(
                        self.text_model.encoder.transformer, "layer"
                    ):
                        for m in self.text_model.encoder.transformer.layer[
                            -1
                        ].modules():
                            m.train(mode)
                    elif hasattr(self.text_model.encoder, "distilbert") and hasattr(
                        self.text_model.encoder.distilbert, "transformer"
                    ):
                        for m in self.text_model.encoder.distilbert.transformer.layer[
                            -1
                        ].modules():
                            m.train(mode)
                except Exception:
                    pass
        except Exception:
            pass

        return self
