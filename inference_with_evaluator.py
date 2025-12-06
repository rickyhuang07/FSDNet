"""
Standalone Test/Inference Script using Evaluator
Supports both:
1. Single folder (no labels, pure inference)
2. Subfolders real/ and fake/ (ground-truth labels, metrics computed)
"""

import os
import argparse
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import logging

from config import config  
from models.fsdnet import FSDNet
from evaluation.evaluator import Evaluator
from utils.device import setup_device, to_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# Dataset for inference
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Detect subfolders real/ and fake/
        real_dir = os.path.join(image_dir, "real")
        fake_dir = os.path.join(image_dir, "fake")
        if os.path.exists(real_dir) and os.path.exists(fake_dir):
            # Option 2: labeled
            for f in os.listdir(real_dir):
                self.image_paths.append(os.path.join(real_dir, f))
                self.labels.append(0)
            for f in os.listdir(fake_dir):
                self.image_paths.append(os.path.join(fake_dir, f))
                self.labels.append(1)
            self.has_labels = True
            logger.info(f"Detected real/ and fake/ subfolders: {len(self.image_paths)} images")
        else:
            # Option 1: unlabeled
            for f in os.listdir(image_dir):
                self.image_paths.append(os.path.join(image_dir, f))
                self.labels.append(-1)
            self.has_labels = False
            logger.info(f"No subfolders detected, using single folder: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# -----------------------------
# Main test function
# -----------------------------
def test_model(checkpoint_path: str, image_dir: str, model_type: str = "fsdnet",
               batch_size: int = 16, device_str: str = "auto",
               threshold_path: str = None, output_file: str = "inference_predictions.csv"):

    # Device
    device = setup_device(device_str)
    logger.info(f"Using device: {device}")

    # Load threshold
    if threshold_path is None:
        threshold_path = os.path.join(config.logging.output_dir, "evaluation", "optimal_threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            threshold = float(json.load(f).get("threshold", 0.5))
        logger.info(f"Loaded optimal threshold: {threshold}")
    else:
        threshold = 0.5
        logger.warning(f"No threshold file found. Using default 0.5")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor()
    ])

    # Dataset and loader
    dataset = TestDataset(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = FSDNet(config)

    # -----------------------------
    # Load checkpoint safely
    # -----------------------------
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        # PyTorch 2.6+ safe load
        with torch.serialization.safe_globals([config.FSDNetConfig]):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ SUCCESSFULLY LOADED CHECKPOINT (weights_only=True)")
    except Exception as e:
        logger.warning(f"Safe weights_only load failed: {e}")
        # Fallback
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ SUCCESSFULLY LOADED CHECKPOINT (weights_only=False)")

    model = to_device(model, device)
    model.eval()

    # Evaluator
    evaluator = Evaluator(config, model, device, model_type=model_type)

    # Collect probabilities and predictions
    all_probs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = to_device(images, device)
            outputs = model(images)
            if hasattr(outputs, "logits"):  # HF model case
                logits = outputs.logits
            else:
                logits = outputs  # torch model case

            probs = torch.softmax(logits, dim=1)[:, 1]  # probability of fake class

            preds = (probs >= threshold).int()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    # Save predictions to CSV
    df = pd.DataFrame({
        "image_path": all_paths,
        "prob_fake": all_probs,
        "pred_label": [int(p >= threshold) for p in all_probs],
        "pred_class": ["fake" if p >= threshold else "real" for p in all_probs],
        "true_label": all_labels
    })
    df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

    # If ground-truth labels exist, compute metrics
    if dataset.has_labels:
        true = [l for l in all_labels if l != -1]
        pred = [int(p >= threshold) for i, p in enumerate(all_probs) if all_labels[i] != -1]
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = accuracy_score(true, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary', zero_division=0)
        logger.info(f"Metrics (ground truth available): Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FSDNet model using Evaluator")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image-dir", required=True, help="Directory with test images (single folder or real/fake subfolders)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--threshold-path", default=None, help="Path to optimal_threshold.json")
    parser.add_argument("--output-file", default="test_predictions.csv", help="CSV output file")
    args = parser.parse_args()

    test_model(
        checkpoint_path=args.checkpoint,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        device_str=args.device,
        threshold_path=args.threshold_path,
        output_file=args.output_file
    )
