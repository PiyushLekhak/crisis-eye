import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CrisisTextDataset(Dataset):
    def __init__(self, tsv_file, max_len=128):
        self.data = pd.read_csv(tsv_file, sep="\t")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_len = max_len

        self.label_map = {
            "injured_or_dead_people": 0,
            "missing_or_found_people": 0,
            "affected_individuals": 0,
            "infrastructure_and_utility_damage": 1,
            "vehicle_damage": 1,
            "rescue_volunteering_or_donation_effort": 1,
            "not_humanitarian": 2,
            "other_relevant_information": 2,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = str(row["tweet_text"])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_str = (
            row["label_text"]
            if pd.notna(row["label_text"])
            else row.get("label_image", "not_humanitarian")
        )
        label = self.label_map.get(label_str, 2)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
