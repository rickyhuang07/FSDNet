"""
Training trainer for FSDNet project.
Handles training loop, validation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from utils.device import setup_device, to_device
import os
import time
try:
    import wandb
except Exception:
    wandb = None
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for FSDNet model.
    """
    
    def __init__(self, config, model: nn.Module, device: torch.device):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model: FSDNet model
            device: Device to train on
        """
        self.config = config
        self.model = model
        self.device = device

        # Wandb flag initialized early to avoid attribute errors
        self.use_wandb = bool(getattr(self.config.logging, 'wandb', False) and (wandb is not None))
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Move model to device (optimize memory/layout)
        self.model = to_device(self.model, self.device)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # torch.compile for speed (CUDA only, safe fallback)
        self._maybe_compile_model()
        
        # Mixed precision scaler (if enabled & on CUDA)
        self.use_amp = bool(getattr(self.config.hardware, "mixed_precision", False) and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Setup W&B
        self._setup_wandb()

        logger.info("FSDNet trainer initialized")

    def _setup_wandb(self):
        """Initialize Weights & Biases if enabled in config."""
        if self.use_wandb:
            try:
                wandb.init(project="fsdnet", config={
                    "lr": self.config.training.learning_rate,
                    "batch_size": self.config.data.batch_size,
                    "optimizer": self.config.training.optimizer,
                    "scheduler": self.config.training.scheduler,
                    "use_video_frames": getattr(self.config.data, 'use_video_frames', False),
                })
            except Exception as e:
                logger.warning(f"W&B init failed, disabling wandb: {e}")
                self.use_wandb = False

    def _setup_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.training.optimizer == "adam":
            return optim.Adam(params, lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)
        elif self.config.training.optimizer == "sgd":
            return optim.SGD(params, lr=self.config.training.learning_rate, momentum=0.9, weight_decay=self.config.training.weight_decay)
        else:
            return optim.Adam(params, lr=self.config.training.learning_rate, weight_decay=self.config.training.weight_decay)

    def _setup_scheduler(self):
        if self.config.training.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.training.epochs)
        elif self.config.training.scheduler == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, self.config.training.epochs // 3), gamma=0.1)
        elif self.config.training.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)
        else:
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.training.epochs)

    def _setup_loss_function(self):
        if self.config.training.loss_function == "cross_entropy":
            class_weights = None
            if getattr(self.config.training, 'loss_class_weights', None):
                rw, fw = self.config.training.loss_class_weights
                class_weights = torch.tensor([rw, fw], dtype=torch.float32, device=self.device)
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.config.training.label_smoothing)
        else:
            return nn.CrossEntropyLoss(label_smoothing=self.config.training.label_smoothing)

    def _maybe_compile_model(self):
        # Only attempt compile on CUDA; disable on MPS/CPU due to stability issues
        if not getattr(self.config.training, 'use_compile', False):
            return
        if self.device.type != 'cuda':
            logger.info("Skipping torch.compile on non-CUDA device")
            return
        try:
            self.model = torch.compile(self.model, mode='reduce-overhead', dynamic=False)
            logger.info("Model compiled with torch.compile (CUDA)")
        except Exception as e:
            logger.warning(f"torch.compile failed; continuing without compilation: {e}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            # Move data to device
            non_blocking = self.device.type == 'cuda'
            images = images.to(self.device, non_blocking=non_blocking, memory_format=torch.channels_last if self.device.type == 'cuda' else torch.contiguous_format)
            labels = labels.to(self.device, non_blocking=non_blocking)
            
            # Forward + backward (with optional AMP + GradScaler)
            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.autocast(device_type='cuda', enabled=True):
                    outputs = self.model(images)
                    if hasattr(outputs, "logits"):
                        outputs = outputs.logits
                    loss = self.criterion(outputs, labels)
                # scale, backward, step, update scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / max(1, len(train_loader))
        accuracy = 100. * correct / max(1, total)

        if self.use_wandb:
            try:
                wandb.log({'train/loss': avg_loss, 'train/accuracy': accuracy, 'epoch': self.current_epoch + 1})
            except Exception:
                pass
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Move data to device
                non_blocking = self.device.type == 'cuda'
                images = images.to(self.device, non_blocking=non_blocking, memory_format=torch.channels_last if self.device.type == 'cuda' else torch.contiguous_format)
                labels = labels.to(self.device, non_blocking=non_blocking)
                
                # Forward pass
                if self.use_amp:
                    with torch.autocast(device_type='cuda', enabled=True):
                        outputs = self.model(images)
                        if hasattr(outputs, "logits"):
                            outputs = outputs.logits
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    if hasattr(outputs, "logits"):
                        outputs = outputs.logits 
                    loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / max(1, len(val_loader))
        accuracy = 100. * correct / max(1, total)

        if self.use_wandb:
            try:
                wandb.log({'val/loss': avg_loss, 'val/accuracy': accuracy, 'epoch': self.current_epoch + 1})
            except Exception:
                pass
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        # Unwrap model if wrapped by DataParallel or torch.compile
        raw_model = getattr(self.model, "_orig_mod", getattr(self.model, "module", self.model))
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.logging.checkpoint_dir, 
            f'checkpoint_epoch_{self.current_epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if this is the best so far
        if is_best:
            best_path = os.path.join(
                self.config.logging.checkpoint_dir, 
                'best_checkpoint.pth'
            )
            torch.save(checkpoint, best_path)
            logger.info(f"New best checkpoint saved with validation accuracy: {self.best_val_accuracy:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Check if this is the best model
            is_best = val_acc > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_acc
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping check
            if len(self.val_accuracies) >= self.config.training.patience:
                recent_accuracies = self.val_accuracies[-self.config.training.patience:]
                if max(recent_accuracies) - min(recent_accuracies) < self.config.training.min_delta:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
