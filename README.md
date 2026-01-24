# CRISIS-EYE  

## Reliability-Aware Gated Late-Fusion for Crisis Urgency Classification

## Project Overview

This repository contains the codebase and experimental artifacts for a multimodal crisis urgency classification system. Social media image–text pairs are classified into three urgency levels: **High**, **Medium**, and **Low**. The system is designed to improve operational reliability under noisy and highly imbalanced conditions by prioritizing precision for the High-urgency class.

The project was developed and evaluated using the CrisisMMD dataset ([official dataset page](https://crisisnlp.qcri.org/crisismmd.html#data_version2.0)). A reliability-aware gated late-fusion architecture was employed to selectively integrate textual and visual information on a per-sample basis.

## Key Technical Contributions

- Exploratory Data Analysis revealed strong class imbalance and substantial disagreement between text and image modalities.
- Unimodal baselines were implemented using DistilBERT (text) and ResNet-50 (image).
- A gated late-fusion architecture was introduced, using a learned per-sample scalar gate to weight modality contributions.
- Auxiliary unimodal classification heads were used during training to prevent modality collapse.
- Evaluation demonstrated that gated fusion preserved aggregate performance while improving precision for the High-urgency class.

## Repository Structure

```plaintext
CRISIS-EYE/
├── artifacts/                  # Stored metrics and experiment outputs
├── checkpoints/                # Checkpoint reports (model files not committed)
├── data/                       # Dataset files (not included)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_results_analysis.ipynb
├── src/
│   ├── datasets/               # Dataset loaders
│   ├── evaluation/             # Evaluation utilities
│   ├── models/                 # Model definitions
│   ├── training/               # Training scripts
│   └── cleanup_files.py
├── requirements.txt
└── README.md
```

## Model Weights

Trained model checkpoints are not committed to the repository.
The best-performing gated late-fusion model (fusion_best.pt, ~351 MB) is provided via [Releases page](https://github.com/PiyushLekhak/crisis-eye/releases/tag/v1.0).

## Environment Setup

Project dependencies are listed in **requirements.txt.**

A typical setup can be created using:

```bash
pip install -r requirements.txt
```

Model development and evaluation were conducted using PyTorch and HuggingFace Transformers. GPU acceleration is recommended but not strictly required for evaluation.

## Training and Evaluation

Training scripts for all models are located in `src/training/:`

- train_text.py

- train_image.py

- train_fusion.py

Evaluation utilities are provided in `src/evaluation/.`

Metrics include **Macro F1 and per-class precision and recall**, with an emphasis on **High-urgency false positives**.

**Example workflows and result analysis are demonstrated in the provided notebooks.**

## Inference

After downloading the model checkpoint from the [Releases page](https://github.com/PiyushLekhak/crisis-eye/releases/tag/v1.0), you can load it for inference as follows:

### Prerequisites

```bash
pip install -r requirements.txt
```

### Loading the Model

```python
import torch
from src.models.fusion_model import LateFusionModel

# Initialize model architecture (must match training configuration)
model = LateFusionModel(
    num_classes=3,
    text_checkpoint=None,  # Not needed if loading full fusion checkpoint
    image_checkpoint=None  # Not needed if loading full fusion checkpoint
)

# Load trained weights
checkpoint = torch.load('path/to/fusion_best.pt', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
```

### Running Inference on a Single Sample

```python
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms

# Preprocessing (must match training pipeline)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def predict_urgency(image_path, text):
    # Process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Process text
    inputs = tokenizer(text, return_tensors='pt', 
                      max_length=128, padding='max_length', 
                      truncation=True)
    
    # Run inference
    with torch.no_grad():
        logits = model(inputs['input_ids'], 
                      inputs['attention_mask'], 
                      image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    urgency_levels = ['High', 'Medium', 'Low']
    return urgency_levels[pred_class], probs.numpy()

# Example usage
image_path = 'example_disaster_image.jpg'
text = "People trapped in collapsed building, urgent help needed"
urgency, confidence = predict_urgency(image_path, text)
print(f"Predicted urgency: {urgency} (Confidence: {confidence})")
```

### Batch Inference

For batch processing, modify the preprocessing to handle multiple images/texts and use appropriate batching in the DataLoader from src/datasets/multimodal_dataset.py.

**Note:** Ensure your preprocessing exactly matches the validation pipeline used during training (center-crop, no random augmentations).

## Notes and Limitations

- For posts containing multiple images, only the first image was used during experiments.

- Evaluation was conducted offline on a held-out test split of CrisisMMD; generalization to unseen disasters or platforms is not guaranteed.

- Operational deployment would require calibration of decision thresholds and human oversight.
