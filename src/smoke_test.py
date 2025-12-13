import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision.models import resnet50, ResNet50_Weights
import time
import os

# --- Configuration for Higher Smoke Test ---
TEST_BATCH_SIZE = 16

MAX_SEQ_LENGTH = 128
EMBEDDING_DIM = 768
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_gpu():
    """Checks for GPU and reports available VRAM."""
    if DEVICE.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {total_mem:.2f} GB. Testing Batch Size: {TEST_BATCH_SIZE}")
    else:
        print("GPU not found.")
    return DEVICE


def print_vram_usage(step_name):
    """Prints current allocated and cached VRAM in GB."""
    if DEVICE.type == "cuda":
        allocated = torch.cuda.memory_allocated(DEVICE) / (1024**3)
        cached = torch.cuda.memory_reserved(DEVICE) / (1024**3)
        print(
            f"   [VRAM USAGE - {step_name}]: Allocated: {allocated:.2f} GB / Cached: {cached:.2f} GB"
        )


class LateFusionModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.image_model.fc = nn.Identity()
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Non-Freezing (High-Risk Mode)
        for param in self.image_model.parameters():
            param.requires_grad = True
        for param in self.text_model.parameters():
            param.requires_grad = True

        fusion_dim = 2048 + 768
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(EMBEDDING_DIM, output_dim),
        )

    def forward(self, image_input, text_input_ids, text_attention_mask):
        image_features = self.image_model(image_input)
        text_output = self.text_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        fused_features = torch.cat((image_features, text_features), dim=1)
        output = self.fusion_head(fused_features)
        return output


def run_smoke_test(device):
    if device.type == "cpu":
        return

    try:
        # --- 1. Initialize Model ---
        print("\n[STEP 1] Initializing Model (ResNet50 + DistilBERT)...")
        model = LateFusionModel(NUM_CLASSES).to(device)
        model.train()

        # Use PyTorch's native AMP utility
        scaler = torch.amp.GradScaler("cuda")
        print("Mixed Precision (FP16) GradScaler Initialized.")

        # --- 2. Create Dummy Data ---
        print(f"[STEP 2] Generating Dummy Data (Batch Size: {TEST_BATCH_SIZE})...")
        dummy_images = torch.randn(TEST_BATCH_SIZE, 3, 224, 224).to(device)
        dummy_input_ids = torch.randint(0, 1000, (TEST_BATCH_SIZE, MAX_SEQ_LENGTH)).to(
            device
        )
        dummy_attention_mask = torch.ones(TEST_BATCH_SIZE, MAX_SEQ_LENGTH).to(device)
        dummy_labels = torch.randint(0, NUM_CLASSES, (TEST_BATCH_SIZE,)).to(device)

        # --- 3. Forward Pass (High VRAM Check) ---
        print("[STEP 3] Performing Forward Pass...")
        print_vram_usage("Before Forward")

        with torch.amp.autocast("cuda"):
            outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, dummy_labels)

        print_vram_usage("After Forward")

        # --- 4. Backward Pass (Peak VRAM Check) ---
        print("[STEP 4] Performing Backward Pass (Gradient Calculation)...")

        # Scaled backward pass
        scaler.scale(loss).backward()

        print_vram_usage("After Backward (PEAK)")

        # Cleanup
        del (
            outputs,
            loss,
            dummy_images,
            dummy_input_ids,
            dummy_attention_mask,
            dummy_labels,
            model,
        )
        torch.cuda.empty_cache()
        print(" Cleanup successful. VRAM usage released.")

        print(f"\n[VERDICT B={TEST_BATCH_SIZE}]")
        print(f"✨ SUCCESS: Batch Size {TEST_BATCH_SIZE} is stable. ✨")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("-" * 60)
            print(f"OUT-OF-MEMORY ERROR at Batch Size {TEST_BATCH_SIZE}!")
            print(
                f"The VRAM ceiling is the previous successful batch size (B_max = {TEST_BATCH_SIZE - 2} or {TEST_BATCH_SIZE - 4})."
            )
            print("Your plan must use this B_max with Gradient Accumulation.")
            print("-" * 60)
            torch.cuda.empty_cache()
        else:
            raise e


if __name__ == "__main__":
    device = check_gpu()
    run_smoke_test(device)
