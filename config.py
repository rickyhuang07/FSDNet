"""
main configuration file 
contains all hyperparameters + settings for training, evaluation, and inference
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class DataConfig:
    """everything related to dataset and preprocessing"""
    real_images_dir: str = "dataset/real"
    fake_images_dir: str = "dataset/fake"
    # optional frame extraction from videos
    use_video_frames: bool = False
    video_frames_dir: str = "data_video_frames"  # expects {train,val,test}/{real,fake}/...
    video_real_videos_dir: str = "data_video/real"
    video_fake_videos_dir: str = "data_video/fake"
    
    # video extraction settings
    video_inner_percent: float = 0.7     # use inner 70% of frames by default
    video_frames_per_second: float = 2.0   # sample 2 frames per second

    # data splits (used when explicit split dirs absent or for video frame extraction)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # image preprocessing settings
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 8
    prefetch_factor: int = 4   # this just keeps gpu fed with data, preps 8 x 4 = 32 batches in advance
    persistent_workers: bool = True   # prevents workers from constantly being created/destroyed each epoch
    
    # data augmentation
    use_augmentation: bool = True
    blur_probability: float = 0.5
    blur_sigma_range: Tuple[float, float] = (0.0, 3.0)
    jpeg_probability: float = 0.5
    jpeg_quality_range: Tuple[int, int] = (30, 100)

@dataclass
class ModelConfig:
    """model architecture config"""
    # radial pooling spectral pooling (RPSP) path
    rpsp_radii_count: int = 128
    rpsp_max_radius: float = 100.0
    
    # backbone path (basically a cnn/ViT)
    pretrained: bool = True   # use pretrained weights  
  
    # classifier
    num_classes: int = 2   # real or fake
    dropout_rate: float = 0.5   # prevents overfitting

@dataclass
class TrainingConfig:
    """training config"""
    # basic training params
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # optimizer
    optimizer: str = "adam"  # adam, sgd
    scheduler: str = "cosine"  # cosine, step, plateau (controls how learning rate changes during training)
    use_compile: bool = True
    
    # loss
    loss_function: str = "cross_entropy"
    label_smoothing: float = 0.1  # prevent overconfidence -> improve generalization

    # class balancing
    balance_classes: bool = True         
    sampler_balance: bool = True  # WeightedRandomSampler
    loss_class_weights: Optional[Tuple[float, float]] = None  # (real_weight, fake_weight)
    
    # checkpointing
    save_every: int = 10
    save_best: bool = True
    load_best: bool = False
    
    # early stopping
    patience: int = 5
    min_delta: float = 0.001

class EvaluationConfig:
    load_optimal_threshold: bool = False  # flag to enable/disable loading
    optimal_threshold_path: str = "outputs/evaluation/optimal_threshold.json"  

@dataclass
class HardwareConfig:
    """hardware and device config"""
    device: str = "auto" # auto, cuda, mps, cpu
    num_gpus: int = 1
    mixed_precision: bool = True # memory usage / numerical precision tradeoff
    
    # memory management
    gradient_accumulation_steps: int = 1
    max_memory_usage: Optional[float] = None

@dataclass
class LoggingConfig:
    """logging and output configuration"""
    # output dirs
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # logging
    log_every: int = 100
    tensorboard: bool = True
    wandb: bool = True
    
    # visualization
    save_samples: bool = True
    num_samples: int = 16

@dataclass
class FSDNetConfig:
    """main config class combining everything"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """create necessary directories after initialization"""
        os.makedirs(self.logging.output_dir, exist_ok=True)
        os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logging.log_dir, exist_ok=True)

# default config instance
config = FSDNetConfig()
