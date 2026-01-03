import time
import random
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import amp
import subprocess
from transformers import DistilBertTokenizer


from sklearn.metrics import classification_report, f1_score

from src.datasets.text_dataset import CrisisTextDataset
from src.models.text_model import DistilBertTextClassifier


# -------------------- CONFIG --------------------
SEED = 42
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# Hyperparameters (sensible defaults + easy to tune)
BATCH_SIZE = 16
MAX_EPOCHS = 10
PATIENCE = 3  # early stopping patience
LR = 2e-5  # BERT-style LR (sane default)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
MAX_LEN = 128  # EDA showed 95% tokens <= 57; 128 is safe
NUM_CLASSES = 3

# Data / output paths
TRAIN_PATH = "data/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv"
VAL_PATH = "data/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "text_baseline_best.pt"
REPORT_PATH = CHECKPOINT_DIR / "text_baseline_report.txt"

# DataLoader speedups
NUM_WORKERS = 6
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


def compute_class_weights(dataset, num_classes=NUM_CLASSES):
    # dataset[i]["label"] must return integer 0..num_classes-1
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Mixed precision context
        with amp.autocast(device_type=DEVICE, enabled=USE_CUDA):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Gradient clipping (unscale then clip)
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
            labels = batch["label"].to(device)

            with amp.autocast(device_type=DEVICE, enabled=USE_CUDA):
                logits = model(input_ids, attention_mask)

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

    train_dataset = CrisisTextDataset(TRAIN_PATH, max_len=MAX_LEN)
    val_dataset = CrisisTextDataset(VAL_PATH, max_len=MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    # class weights and criterion
    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    print("Class weights:", class_weights.tolist())

    model = DistilBertTextClassifier(NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1 = 0.0
    patience_counter = 0

    # Initialize / overwrite report file
    with open(REPORT_PATH, "w") as f:
        f.write("Text Baseline Training Report\n")
        f.write(f"Seed: {SEED}\n")
        f.write(
            f"Config: LR={LR}, BATCH_SIZE={BATCH_SIZE}, MAX_LEN={MAX_LEN}, WEIGHT_DECAY={WEIGHT_DECAY}\n\n"
        )

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

        # Scheduler step uses the validation metric
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
    Save tokenizer, label mapping, and frozen run configuration
    for the DistilBERT text baseline.
    Called automatically if training is run again.
    """

    artifacts_dir = Path("artifacts/text_baseline")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(artifacts_dir / "tokenizer")

    # ---- Label mapping ----
    label_map = {
        "label2id": {"High": 0, "Medium": 1, "Low": 2},
        "id2label": {"0": "High", "1": "Medium", "2": "Low"},
    }
    with open(artifacts_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

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
        "model": "distilbert-base-uncased",
        "num_classes": NUM_CLASSES,
        "max_len": MAX_LEN,
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
        "git_commit": git_sha,
    }

    with open(artifacts_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Artifacts saved to {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
