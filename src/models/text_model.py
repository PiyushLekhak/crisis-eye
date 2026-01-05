import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertTextClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

        # ---- Fair freezing policy: freeze all encoder params except last transformer layer ----
        # Freeze entire encoder first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Try to locate the last transformer layer in possible attribute layouts
        last_layer = None
        # Common: encoder.transformer.layer (DistilBertModel may expose transformer)
        if hasattr(self.encoder, "transformer") and hasattr(
            self.encoder.transformer, "layer"
        ):
            last_layer = self.encoder.transformer.layer[-1]
        # Alternate layout: encoder.distilbert.transformer.layer (older/newer wrappers)
        elif (
            hasattr(self.encoder, "distilbert")
            and hasattr(self.encoder.distilbert, "transformer")
            and hasattr(self.encoder.distilbert.transformer, "layer")
        ):
            last_layer = self.encoder.distilbert.transformer.layer[-1]

        # Unfreeze parameters in the identified last transformer block
        if last_layer is not None:
            for p in last_layer.parameters():
                p.requires_grad = True

        # Ensure classifier head is trainable
        for p in self.classifier.parameters():
            p.requires_grad = True
        # --------------------------------------------------------------------------------------

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls_embedding)

        return logits
