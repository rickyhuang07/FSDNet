"""
Main entry point for FSDNet Project.
Handles training, validation, and evaluation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
from models.fsdnet import FSDNet
from transformers import AutoModelForImageClassification

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config
from utils.device import setup_device, get_device_info
from data.dataset import create_data_loaders, create_combined_data_loaders
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    log_format = "%(_asctime)s - %(name)s - %(levelname)s - %(message)s".replace("_", "")
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

def train_model(config, device, model_type: str = "fsdnet"):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training process for {model_type.upper()} ")

    # Create data loaders
    if getattr(config.data, 'use_video_frames', False):
        logger.info("Using combined image+video-frame dataset with 70/15/15 stratified split")
        train_loader, val_loader, test_loader = create_combined_data_loaders(config)
    else:
        train_loader, val_loader, test_loader = create_data_loaders(config)

    # Select model
    if model_type == "fsdnet":
        model = FSDNet(config)
    elif model_type == "resnet-50":
        # Weights options: None, ResNet50_Weights.DEFAULT, "IMAGENET1K_V1"
        model = resnet50(weights=ResNet50_Weights.DEFAULT)  
        model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary classification
    elif model_type == "efficientnet-b0":  # EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif model_type == "mobilenet-v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # Binary classification
    elif model_type == "hf-df1":  
        model = AutoModelForImageClassification.from_pretrained(
        "dima806/deepfake_vs_real_image_detection", #hf1
        num_labels=2  # binary classification
        )
    elif model_type == "hf-df2":  
        model = AutoModelForImageClassification.from_pretrained(
        "prithivMLmods/deepfake-detector-model-v1", #hf2
        num_labels=2  # binary classification
        )
    elif model_type == "hf-df3":  
        model = AutoModelForImageClassification.from_pretrained(
        "prithivMLmods/Deep-Fake-Detector-v2-Model", #hf3
        num_labels=2  # binary classification
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = Trainer(config, model, device)

    if config.training.load_best == True:
        # Load checkpoint for continue training
        checkpoint_path = "checkpoints/best_checkpoint.pth"  # or 'best_checkpoint.pth'
        trainer.load_checkpoint(checkpoint_path)
        logger.info("best_checkpoints.pth loaded successfully!")

    # Train model
    trainer.train(train_loader, val_loader)

    # Threshold tuning 
    evaluator = Evaluator(config, model, device, model_type=model_type)
    best = evaluator.find_optimal_threshold(val_loader, strategy="youden")
    thresh_path = evaluator.save_optimal_threshold(best, os.path.join(config.logging.output_dir, "evaluation"))
    logger.info(f"Optimal threshold (validation): {best}")

    # Evaluate on test set
    test_metrics = evaluator.evaluate(test_loader)

    # Generate evaluation report
    evaluation_dir = os.path.join(config.logging.output_dir, "evaluation")
    evaluator.generate_evaluation_report(test_loader, evaluation_dir)

    logger.info("Training and evaluation completed!")
    return model, test_metrics

def evaluate_model(config, device, checkpoint_path: str, model_type: str = "fsdnet"):
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {model_type.upper()} from {checkpoint_path}")

    checkpoint_name = os.path.basename(checkpoint_path)
    logger.info("=" * 60)
    logger.info(f"EVALUATING CHECKPOINT: {checkpoint_name} ({model_type})")
    logger.info("=" * 60)

    # Create data loaders
    if getattr(config.data, 'use_video_frames', False):
        _, val_loader, test_loader = create_combined_data_loaders(config)
    else:
        _, _, test_loader = create_data_loaders(config)
        val_loader = None

    # Select model
    if model_type == "fsdnet":
        model = FSDNet(config)
    elif model_type == "resnet-50":
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif model_type == "efficientnet-b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif model_type == "mobilenet-v2":
        model = mobilenet_v2(weights=None)  # No pretrained weights
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif model_type == "hf-df1":  
        model = AutoModelForImageClassification.from_pretrained(
        "dima806/deepfake_vs_real_image_detection", #hf1
        num_labels=2  # binary classification
        )
    elif model_type == "hf-df2":  
        model = AutoModelForImageClassification.from_pretrained(
        "prithivMLmods/deepfake-detector-model-v1", #hf2
         num_labels=2  # binary classification
        )
    elif model_type == "hf-df3":  
        model = AutoModelForImageClassification.from_pretrained(
        "prithivMLmods/Deep-Fake-Detector-v2-Model", #hf3
        num_labels=2  # binary classification
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_num = checkpoint.get('epoch', 'Unknown')
        logger.info(f"✓ SUCCESSFULLY LOADED CHECKPOINT FROM EPOCH {epoch_num} (secure mode)")
    except Exception as e:
        logger.warning(f"Secure loading failed: {e}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_num = checkpoint.get('epoch', 'Unknown')
        logger.info(f"✓ SUCCESSFULLY LOADED CHECKPOINT FROM EPOCH {epoch_num} (legacy mode)")

    # Evaluate
    evaluator = Evaluator(config, model, device, model_type=model_type)
    if val_loader is not None:
        best = evaluator.find_optimal_threshold(val_loader, strategy='youden')
        evaluator.save_optimal_threshold(best, os.path.join(config.logging.output_dir, "evaluation"))
    
    # Load threshold from previous run
    best_threshold = None
    if getattr(config.evaluation, "load_optimal_threshold", False):
        threshold_path = getattr(config.evaluation, "optimal_threshold_path", None)
        if threshold_path and os.path.exists(threshold_path):
            best_threshold = evaluator.load_optimal_threshold(threshold_path)
            logger.info(f"Loaded optimal threshold from: {threshold_path}")
        else:
            logger.warning(f"Optimal threshold path not found: {threshold_path}")
    else:
        logger.info("Skipping optimal threshold loading (disabled in config).")

    # Evaluate and generate report using loaded threshold
    test_metrics = evaluator.evaluate(test_loader, threshold=best_threshold)
    evaluation_dir = os.path.join(config.logging.output_dir, "evaluation")
    evaluator.generate_evaluation_report(test_loader, evaluation_dir, threshold=best_threshold)

    logger.info("=" * 60)
    logger.info(f"EVALUATION COMPLETED FOR EPOCH {epoch_num} ({model_type})")
    logger.info("=" * 60)
    return test_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="FSDNet: Frequency-Spatial Deepfake Detection Network")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train",
                       help="Mode to run: train or evaluate")
    parser.add_argument("--model-type", choices=["fsdnet", "resnet-50", "efficientnet-b0", "mobilenet-v2", "hf-df1","hf-df2", "hf-df3"], default="fsdnet",
                   help="Model type: fsdnet, resnet-50, or efficientnet-b0, mobilenet-v2, hf-df1, hf-df2, hf-df3")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for evaluation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str, help="Log file path")
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Log device information
    device_info = get_device_info()
    logger.info("Device Information:")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")
    
    # Setup device
    device = setup_device(config.hardware.device)
    logger.info(f"Using device: {device}")
    
    try:
        if args.mode == "train":
            model, metrics = train_model(config, device, model_type=args.model_type)
            logger.info("Training completed successfully!")

        elif args.mode == "evaluate":
            # Evaluation mode
            if not args.checkpoint:
                raise ValueError("Checkpoint path is required for evaluation mode")
            
            metrics = evaluate_model(config, device, args.checkpoint, model_type=args.model_type)
            logger.info("Evaluation completed successfully!")
            
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
            
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
