import time
import random
import numpy as np
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import amp

from sklearn.metrics import classification_report, f1_score


from src.datasets.multimodal_dataset import CrisisMultimodalDataset
from src.models.advanced_fusion import AdvancedFusionModel

# -------------------- CONFIG --------------------
SEED = 42
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# Hyperparameters (Adjusted for Unfrozen Training)
BATCH_SIZE = 16
MAX_EPOCHS = 10
PATIENCE = 5  # Increased slightly as unfrozen models need time to settle
LR = 2e-5  # LOWER LR (Critical for unfrozen backbones)
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
NUM_CLASSES = 3

# Data / output paths
TRAIN_PATH = "data/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv"
VAL_PATH = "data/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv"
IMG_DIR = "data/"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "advanced_fusion_best.pt"
REPORT_PATH = CHECKPOINT_DIR / "advanced_fusion_report.txt"

# DataLoader speedups
NUM_WORKERS = 4
PIN_MEMORY = True if DEVICE == "cuda" else False

scaler = amp.GradScaler(enabled=USE_CUDA)
# ---------------------------------------------------


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_class_weights(dataset, num_classes=NUM_CLASSES):
    """Inverse frequency weights to handle class imbalance"""
    labels = [int(dataset[i]["label"]) for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Mixed precision
        with amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=USE_CUDA
        ):
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=USE_CUDA,
            ):
                logits = model(input_ids, attention_mask, images)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds, target_names=["High", "Medium", "Low"], digits=4
    )
    return macro_f1, report


def main(args):
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print("Loading datasets...")

    train_dataset = CrisisMultimodalDataset(TRAIN_PATH, IMG_DIR, split="train")
    val_dataset = CrisisMultimodalDataset(VAL_PATH, IMG_DIR, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    print("Class weights:", class_weights.tolist())

    # --- Initialize Advanced Fusion Model ---
    print(f"\nInitializing Advanced Fusion Model (Cross-Attention)...")
    print(f"Text checkpoint: {args.text_checkpoint}")
    print(f"Image checkpoint: {args.image_checkpoint}")

    model = AdvancedFusionModel(
        num_classes=NUM_CLASSES,
        text_checkpoint=args.text_checkpoint,
        image_checkpoint=args.image_checkpoint,
        common_dim=512,
        dropout=0.3,
    ).to(DEVICE)

    # Check trainable params (Should be much higher than late fusion, but lower than full finetune)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print("(Includes Fusion Head + Last DistilBERT Layer + Last ResNet Block)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with higher weight decay for regularization
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1 = 0.0
    patience_counter = 0

    with open(REPORT_PATH, "w") as f:
        f.write("Advanced Fusion Model Training Report\n")
        f.write(f"Config: LR={LR}, BATCH={BATCH_SIZE}, WD={WEIGHT_DECAY}\n")
        f.write(f"Trainable: {trainable_params:,}\n\n")

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        t_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, GRAD_CLIP
        )
        val_macro_f1, report = evaluate(model, val_loader, DEVICE)

        print(
            f"\nEpoch {epoch}/{MAX_EPOCHS} — Loss: {train_loss:.4f} — Val F1: {val_macro_f1:.4f}"
        )
        print(report)

        with open(REPORT_PATH, "a") as f:
            f.write(f"Epoch {epoch}: Loss={train_loss:.4f}, F1={val_macro_f1:.4f}\n")
            f.write(report + "\n" + "=" * 50 + "\n")

        if val_macro_f1 > best_f1 + 1e-6:
            best_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved Best Model: {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        scheduler.step(val_macro_f1)

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Time: {(time.time() - t_start):.1f}s")

    print(f"\nDone. Best F1: {best_f1:.4f}. Total time: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_checkpoint", type=str, default="checkpoints/text_baseline_best.pt"
    )
    parser.add_argument(
        "--image_checkpoint", type=str, default="checkpoints/image_baseline_best.pt"
    )
    args = parser.parse_args()
    main(args)
