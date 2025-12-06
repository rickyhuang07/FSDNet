"""
Evaluation module for FSDNet project.
Handles model evaluation on test datasets.
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from utils.device import to_device
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class.
    """

    def __init__(self, config, model: nn.Module, device: torch.device, model_type: str = "fsdnet"):
        """
        Initialize evaluator.

        Args:
            config: Configuration object
            model: Trained FSDNet, ResNet50, or Efficientnet-B0 model
            device: Device to evaluate on
            model_type: 'FSDNet' or 'resnet50' or 'efficientnet-b0'
        """
        self.config = config
        self.model = model
        self.device = device
        self.model_type = model_type.lower()

        # Move model to device
        self.model = to_device(self.model, self.device)
        self.model.eval()

        logger.info(f" Evaluator initialized for model type: {self.model_type}")
    
    def evaluate(self, test_loader: DataLoader, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate model on test dataset.

        Args:
            test_loader: Test data loader
            threshold: Optional probability threshold for classifying the 'fake' class.
                   If None, uses argmax as before.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
    
        all_predictions = []
        all_labels = []
        all_probabilities = []
    
        with torch.no_grad():
            for images, labels in test_loader:
                images = to_device(images, self.device)
                labels = to_device(labels, self.device)
            
                outputs = self.model(images)
                if hasattr(outputs, "logits"):   # Hugging Face AutoModel case
                    outputs = outputs.logits
                probabilities = torch.softmax(outputs, dim=1)
            
                # Apply threshold if provided
                if threshold is not None:
                    predictions = (probabilities[:, 1] >= threshold).long()
                else:
                    _, predictions = outputs.max(1)
            
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
    
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
    
        logger.info("Evaluation completed!")
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
    
        return metrics

    
    def _calculate_metrics(self, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            probabilities: Predicted probabilities for fake class
        
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Precision, recall, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(labels, probabilities)
        except ValueError:
            auc_roc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }
    
    def collect_probs_and_labels(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Collect probabilities for fake class and labels from a loader."""
        probs = []
        labels = []
        with torch.no_grad():
            for images, y in loader:
                images = to_device(images, self.device)
                outputs = self.model(images)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                p = torch.softmax(outputs, dim=1)[:, 1]
                probs.extend(p.cpu().numpy())
                labels.extend(y.numpy())
        return np.array(probs), np.array(labels)

    def find_optimal_threshold(self, loader: DataLoader, strategy: str = "youden") -> Dict[str, float]:
        """
        Find an optimal probability threshold on a validation set.
        strategy: 'youden' (maximize TPR-FPR) or 'f1' (maximize F1)
        Returns dict with threshold and metrics at that threshold.
        """
        from sklearn.metrics import roc_curve
        probs, labels = self.collect_probs_and_labels(loader)
        # Scan thresholds
        thresholds = np.linspace(0.0, 1.0, num=201)
        best = {"threshold": 0.5, "score": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for t in thresholds:
            preds = (probs >= t).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            if strategy == 'f1':
                score = f1
            else:
                fpr, tpr, _ = roc_curve(labels, probs >= t)
                # Simplify Youden by computing directly from confusion
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
                tpr2 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr2 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                score = tpr2 - fpr2
            if score > best["score"]:
                best = {"threshold": float(t), "score": float(score), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
        return best

    def save_optimal_threshold(self, best: Dict[str, float], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "optimal_threshold.json")
        with open(path, 'w') as f:
            json.dump(best, f, indent=2)
        logger.info(f"Saved optimal threshold to {path}: {best}")
        return path

    def load_optimal_threshold(self, json_path: str) -> float:
        """
        Load optimal threshold from JSON file.
        Falls back to 0.5 if file is missing or invalid.
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                threshold = float(data.get("threshold", 0.5))
                logger.info(f"Loaded optimal threshold from {json_path}: {threshold}")
                return threshold
        except Exception as e:
            logger.warning(f"Failed to load threshold from {json_path}, using default 0.5. Error: {e}")
            return 0.5

    def plot_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray, save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], 
                    yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, labels: np.ndarray, probabilities: np.ndarray, save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            labels: True labels
            probabilities: Predicted probabilities for fake class
            save_path: Path to save the plot
        """
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, test_loader: DataLoader, output_dir: str, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Generate comprehensive evaluation report using optional threshold.
        """
        os.makedirs(output_dir, exist_ok=True)
    
        # Evaluate model with threshold
        metrics = self.evaluate(test_loader, threshold=threshold)
    
        # Collect probabilities and predictions for plots
        all_predictions = []
        all_labels = []
        all_probabilities = []
    
        with torch.no_grad():
            for images, labels in test_loader:
                images = to_device(images, self.device)
                labels = to_device(labels, self.device)
            
                outputs = self.model(images)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                probabilities = torch.softmax(outputs, dim=1)
            
                if threshold is not None:
                    predictions = (probabilities[:, 1] >= threshold).long()
                else:
                    _, predictions = outputs.max(1)
            
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
    
        # Generate plots
        self.plot_confusion_matrix(all_labels, all_predictions, os.path.join(output_dir, 'confusion_matrix.png'))
        self.plot_roc_curve(all_labels, all_probabilities, os.path.join(output_dir, 'roc_curve.png'))
    
        # Save metrics
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("FSDNet Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
    
        logger.info(f"Evaluation report saved to {output_dir}")
    
        return metrics

