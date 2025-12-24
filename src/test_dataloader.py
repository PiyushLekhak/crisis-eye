from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from dataset import CrisisDataset

# Paths
TSV_FILE = "../data/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv"
IMG_DIR = "../data"

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Dataset
dataset = CrisisDataset(tsv_file=TSV_FILE, img_dir=IMG_DIR, tokenizer=tokenizer)

# DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Test one batch
batch = next(iter(loader))

print("Image batch:", batch["image"].shape)
print("Input IDs:", batch["input_ids"].shape)
print("Attention mask:", batch["attention_mask"].shape)
print("Labels:", batch["label"])
