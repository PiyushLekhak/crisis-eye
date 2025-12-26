import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd


class CrisisImageDataset(Dataset):
    def __init__(self, tsv_file, img_dir, transform=None):
        """
        Args:
            tsv_file (str): Path to the TSV file (train or dev).
            img_dir (str): Root directory where images are stored.
            transform (callable, optional): Torchvision transforms (Resize, Normalize, etc.)
        """

        self.data = pd.read_csv(tsv_file, sep="\t")
        self.img_dir = img_dir
        self.transform = transform

        # Same mapping as your text dataset
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

        # Image loading logic
        img_name = row["image"]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            # Open and convert to RGB (handles occasional grayscale/PNG alpha channels)
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for corrupt images (rare but possible in big datasets)
            # Create a black image so training doesn't crash
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # Using label_image as ground truth
        label = self.label_map.get(row["label_image"], 2)

        return {
            "pixel_values": image,  # Convention: 'pixel_values' for images
            "label": torch.tensor(label, dtype=torch.long),
        }
