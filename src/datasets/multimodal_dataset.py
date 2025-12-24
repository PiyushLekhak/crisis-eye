import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import DistilBertTokenizer


class CrisisDataset(Dataset):
    def __init__(self, tsv_file, img_dir, max_len=128, mode="train"):
        """
        Args:
            tsv_file (str): Path to the .tsv file (e.g., task_humanitarian_train.tsv)
            img_dir (str): Root folder containing images (e.g., data/data_image)
            tokenizer: HuggingFace tokenizer (DistilBertTokenizer)
            max_len (int): Max sequence length for text (128)
            mode (str): 'train' or 'test'. Used for label mapping.
        """
        self.data = pd.read_csv(tsv_file, sep="\t")
        self.img_dir = img_dir
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.max_len = max_len
        self.mode = mode

        # --- IMAGE TRANSFORMS ---
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # ResNet standard size
                transforms.ToTensor(),  # Convert to 0-1 Tensor
                transforms.Normalize(  # ImageNet Stats
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # --- LABEL MAPPING  ---
        self.label_map = {
            # High Urgency (0): Direct Human Impact / Suffering
            "injured_or_dead_people": 0,
            "missing_or_found_people": 0,
            "affected_individuals": 0,
            # Medium Urgency (1): Infrastructure, Response, & Logistics
            "infrastructure_and_utility_damage": 1,
            "vehicle_damage": 1,
            "rescue_volunteering_or_donation_effort": 1,
            # Low Urgency (2): Irrelevant / Informational
            "not_humanitarian": 2,
            "other_relevant_information": 2,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Get row
        row = self.data.iloc[idx]

        # 2. Load Image
        img_path = os.path.join(self.img_dir, row["image"])

        try:
            image = Image.open(img_path).convert("RGB")  # Ensure 3 channels
            image = self.transforms(image)
        except Exception as e:
            # Fallback for missing/corrupt images (Safety Net)
            print(f"Warning: Could not load {img_path}. Using black image.")
            image = torch.zeros((3, 224, 224))

        # 3. Process Text
        text = str(row["tweet_text"])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",  # Return PyTorch tensors
        )

        # 4. Process Label
        raw_label = row["label_image"]  # Using image label as ground truth
        label = self.label_map.get(
            raw_label, 2
        )  # Default to 'Low Urgency' if label unknown

        return {
            "image": image,
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }
