import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertTextClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS]
        logits = self.classifier(cls_embedding)

        return logits
