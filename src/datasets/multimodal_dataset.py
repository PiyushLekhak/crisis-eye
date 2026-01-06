import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class CrisisMultimodalDataset(Dataset):
    def __init__(self, tsv_file, img_dir, max_len=128, split="train"):
        self.data = pd.read_csv(tsv_file, sep="\t")
        self.img_dir = img_dir
        self.max_len = max_len
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if self.split == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
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
        img_path = os.path.join(self.img_dir, row["image"])
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transforms(image)
        except Exception:
            image = torch.zeros((3, 224, 224))
        text = str(row["tweet_text"])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        final_label_str = (
            row["label_text"]
            if pd.notna(row["label_text"])
            else row.get("label_image", "not_humanitarian")
        )
        final_label = self.label_map.get(final_label_str, 2)
        img_lbl_str = row.get("label_image", None)
        aux_label_image = (
            self.label_map.get(img_lbl_str, 2) if pd.notna(img_lbl_str) else -100
        )
        txt_lbl_str = row.get("label_text", None)
        aux_label_text = (
            self.label_map.get(txt_lbl_str, 2) if pd.notna(txt_lbl_str) else final_label
        )
        return {
            "image": image,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(final_label, dtype=torch.long),
            "aux_label_text": torch.tensor(aux_label_text, dtype=torch.long),
            "aux_label_image": torch.tensor(aux_label_image, dtype=torch.long),
        }
