import time
import random
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import amp
import subprocess

from torchvision import transforms

from sklearn.metrics import classification_report, f1_score

from src.datasets.image_dataset import CrisisImageDataset
from src.models.image_model import ResNetImageClassifier


# -------------------- CONFIG --------------------
SEED = 42
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# Hyperparameters (vision-specific baseline standards)
BATCH_SIZE = 32  # Standard for ResNet50 on modern GPUs
MAX_EPOCHS = 15  # Images need more epochs than text
PATIENCE = 5  # Higher patience for CNNs (they learn in plateaus)
LR = 1e-4  # Higher LR for CNNs (vs 2e-5 for BERT)
WEIGHT_DECAY = 1e-4  # Standard L2 regularization for vision
GRAD_CLIP = 1.0
NUM_CLASSES = 3

# Data / output paths
TRAIN_PATH = "data/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv"
VAL_PATH = "data/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv"
IMG_DIR = "data/"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "image_baseline_best.pt"
REPORT_PATH = CHECKPOINT_DIR / "image_baseline_report.txt"

# DataLoader speedups
NUM_WORKERS = 4  # Lower than text (image loading is I/O bound)
PIN_MEMORY = True if DEVICE == "cuda" else False

# AMP scaler (mixed precision)
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


def get_transforms():
    """
    Standard ImageNet preprocessing + light augmentation for baseline.
    Train: RandomCrop + HorizontalFlip (standard baseline augmentation)
    Val: Center crop (no augmentation)
    """
    # ImageNet statistics (ResNet50 was pretrained on these)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to slightly larger
            transforms.RandomCrop((224, 224)),  # Then random crop to 224
            transforms.RandomHorizontalFlip(p=0.5),  # 50% chance flip
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),  # Resize to 256
            transforms.CenterCrop(224),  # Center crop to 224
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, val_transform


def compute_class_weights(dataset, num_classes=NUM_CLASSES):
    """Same as text baseline - count labels and compute inverse frequency"""
    labels = [int(dataset[i]["label"]) for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid div0
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        # Images use 'pixel_values' key (convention from image_dataset.py)
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        # Mixed precision context
        with amp.autocast(device_type=DEVICE, enabled=USE_CUDA):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
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
            images = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            with amp.autocast(device_type=DEVICE, enabled=USE_CUDA):
                logits = model(images)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds, target_names=["High", "Medium", "Low"], digits=4
    )
    return macro_f1, report


def main():
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print("Loading datasets...")

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Load datasets
    train_dataset = CrisisImageDataset(TRAIN_PATH, IMG_DIR, transform=train_transform)
    val_dataset = CrisisImageDataset(VAL_PATH, IMG_DIR, transform=val_transform)

    # ----------------- WeightedRandomSampler setup -----------------
    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    print("Class weights:", class_weights.tolist())

    labels = [int(train_dataset[i]["label"]) for i in range(len(train_dataset))]
    sample_weights = torch.tensor(
        [class_weights.cpu()[int(l)].item() for l in labels], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
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
    # ---------------------------------------------------------------

    # Model with frozen backbone (baseline standard)
    model = ResNetImageClassifier(NUM_CLASSES, freeze_backbone=True).to(DEVICE)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Criterion: use plain CrossEntropy when using sampler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1 = 0.0
    patience_counter = 0

    # Initialize report file
    with open(REPORT_PATH, "w") as f:
        f.write("Image Baseline Training Report (ResNet50)\n")
        f.write(f"Seed: {SEED}\n")
        f.write(
            f"Config: LR={LR}, BATCH_SIZE={BATCH_SIZE}, WEIGHT_DECAY={WEIGHT_DECAY}\n"
        )
        f.write(f"Trainable params: {trainable_params:,} / {total_params:,}\n\n")

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        t_epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, grad_clip=GRAD_CLIP
        )

        val_macro_f1, report = evaluate(model, val_loader, DEVICE)

        print(
            f"\nEpoch {epoch}/{MAX_EPOCHS} — Train loss: {train_loss:.4f} — Val macro-f1: {val_macro_f1:.4f}"
        )
        print(report)

        # Log to file
        with open(REPORT_PATH, "a") as f:
            f.write(f"\nEpoch {epoch}\n")
            f.write(f"Train loss: {train_loss:.4f}\n")
            f.write(f"Val macro-f1: {val_macro_f1:.4f}\n")
            f.write(report)
            f.write("\n" + "=" * 50 + "\n")

        # Save only BEST model
        if val_macro_f1 > best_f1 + 1e-6:
            best_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved to: {BEST_MODEL_PATH} (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        # Scheduler step
        scheduler.step(val_macro_f1)

        # Early stopping
        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)."
            )
            break

        t_epoch_end = time.time()
        print(f"Epoch time: {(t_epoch_end - t_epoch_start):.1f} sec")

    total_time_min = (time.time() - t0) / 60.0
    print(
        f"\nTraining finished. Total time: {total_time_min:.2f} minutes. Best Val macro-F1: {best_f1:.4f}"
    )

    # Final report append
    with open(REPORT_PATH, "a") as f:
        f.write(
            f"\nTraining finished. Best Val macro-F1: {best_f1:.4f}\nTotal time (min): {total_time_min:.2f}\n"
        )

    save_artifacts(best_f1)


def save_artifacts(best_f1):
    """
    Save transforms, label mapping, and frozen run configuration
    for the ResNet50 image baseline.
    """
    artifacts_dir = Path("artifacts/image_baseline")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ---- Label mapping (same as text) ----
    label_map = {
        "label2id": {"High": 0, "Medium": 1, "Low": 2},
        "id2label": {"0": "High", "1": "Medium", "2": "Low"},
    }
    with open(artifacts_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # ---- Transform info (for documentation) ----
    transform_info = {
        "train": {
            "resize": [256, 256],
            "crop": [224, 224],
            "horizontal_flip": 0.5,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "val": {
            "resize": [256],
            "center_crop": [224, 224],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
    }
    with open(artifacts_dir / "transforms.json", "w") as f:
        json.dump(transform_info, f, indent=2)

    # ---- Git commit ----
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_sha = "unknown"

    # ---- Run config ----
    run_config = {
        "model": "resnet50",
        "pretrained_weights": "IMAGENET1K_V1",
        "frozen_backbone": True,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "grad_clip": GRAD_CLIP,
        "seed": SEED,
        "best_val_macro_f1": best_f1,
        "checkpoint_path": str(BEST_MODEL_PATH),
        "train_data": TRAIN_PATH,
        "val_data": VAL_PATH,
        "image_dir": IMG_DIR,
        "git_commit": git_sha,
    }

    with open(artifacts_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Artifacts saved to {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
