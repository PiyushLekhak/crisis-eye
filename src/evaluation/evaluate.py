import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms

# -------------------------
# Imports from the project
# -------------------------

from src.datasets.text_dataset import CrisisTextDataset
from src.datasets.image_dataset import CrisisImageDataset
from src.models.text_model import DistilBertTextClassifier
from src.models.image_model import ResNetImageClassifier
from transformers import AutoTokenizer
from src.datasets.multimodal_dataset import CrisisMultimodalDataset
from src.models.fusion_model import LateFusionModel

# -------------------------
# CONSTANTS
# -------------------------

TEST_DATA_PATH = "data/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv"
IMG_DIR = "data/"


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_test_transforms():
    """
    FAIRNESS CHECK:
    Must be identical to Validation transforms: Resize(256) -> CenterCrop(224).
    NO RandomCrop or Flip here.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# -------------------------
# Core evaluation
# -------------------------
@torch.no_grad()
def evaluate_model(model, dataloader, device, modality):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in dataloader:
        labels = batch["label"].to(device)

        if modality == "text":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask)

        elif modality == "image":
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values)

        elif modality == "fusion":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            outputs = model(input_ids, attention_mask, images)

        else:
            raise ValueError("Unsupported modality")

        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return all_preds, all_labels


# -------------------------
# Main
# -------------------------
def main(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Evaluating Modality: {args.modality}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Dataset & Model setup
    # -------------------------
    if args.modality == "text":
        # Load tokenizer (must match model)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        test_dataset = CrisisTextDataset(
            TEST_DATA_PATH,
            max_len=args.max_len,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        model = DistilBertTextClassifier(num_classes=args.num_classes)

    elif args.modality == "image":
        # FAIRNESS: Use exact same val transforms
        test_transform = get_test_transforms()

        test_dataset = CrisisImageDataset(
            tsv_file=TEST_DATA_PATH, img_dir=IMG_DIR, transform=test_transform
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # Initialize structure (freeze_backbone=True matches training state)
        model = ResNetImageClassifier(
            num_classes=args.num_classes, freeze_backbone=True
        )

    elif args.modality == "fusion":
        test_dataset = CrisisMultimodalDataset(
            tsv_file=TEST_DATA_PATH,
            img_dir=IMG_DIR,
            max_len=args.max_len,
            split="test",
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        model = LateFusionModel(
            num_classes=args.num_classes,
            text_checkpoint=args.text_checkpoint,
            image_checkpoint=args.image_checkpoint,
        )

    else:
        raise ValueError("Modality must be one of: text | image | fusion")

    # -------------------------
    # Load checkpoint
    # -------------------------
    print(f"Loading weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    # -------------------------
    # Evaluation
    # -------------------------
    preds, labels = evaluate_model(model, test_loader, device, args.modality)

    # -------------------------
    # Metrics
    # -------------------------
    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2]
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    class_names = ["High", "Medium", "Low"]
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True
    )

    # Print readable report to console
    print("\n" + "=" * 30)
    print(f"TEST REPORT: {args.modality.upper()}")
    print("=" * 30)
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    print(f"Macro F1: {macro_f1:.4f}")

    metrics = {
        "modality": args.modality,
        "checkpoint": args.checkpoint_path,
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": {
            class_names[i]: {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
    }

    # -------------------------
    # Save outputs
    # -------------------------
    # 1. Metrics JSON
    metrics_path = output_dir / f"{args.modality}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 2. Predictions CSV (Useful for error analysis later)
    df = pd.DataFrame({"true_label": labels, "pred_label": preds})
    df["label_name_true"] = df["true_label"].map({0: "High", 1: "Medium", 2: "Low"})
    df["label_name_pred"] = df["pred_label"].map({0: "High", 1: "Medium", 2: "Low"})
    df_path = output_dir / f"{args.modality}_test_predictions.csv"
    df.to_csv(df_path, index=False)

    # 3. Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_path = output_dir / f"{args.modality}_test_confusion_matrix.png"
    save_confusion_matrix(cm, class_names, cm_path)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["text", "image", "fusion"],
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="artifacts/eval_results")

    parser.add_argument(
        "--text_checkpoint",
        type=str,
        required=False,
        help="Path to text baseline checkpoint (fusion only)",
    )
    parser.add_argument(
        "--image_checkpoint",
        type=str,
        required=False,
        help="Path to image baseline checkpoint (fusion only)",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)

# python -m src.evaluation.evaluate --modality text --checkpoint_path checkpoints/text_baseline_best.pt
# python -m src.evaluation.evaluate --modality image --checkpoint_path checkpoints/image_baseline_best.pt
# python -m src.evaluation.evaluate --modality fusion --checkpoint_path checkpoints/fusion_best.pt --text_checkpoint checkpoints/text_baseline_best.pt --image_checkpoint checkpoints/image_baseline_best.pt
