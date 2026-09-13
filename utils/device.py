"""
device utilities, handles cuda, mps, and cpu device detection + setup
"""

import torch
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

def get_device(device: str = "auto") -> torch.device:
    # options: 'auto', 'cuda', 'mps', 'cpu'

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("MPS (Apple Silicon) available")
        else:
            device = "cpu"
            logger.info("Using CPU")
    
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    if device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        logger.warning("MPS requested but not available, falling back to CPU")
        device = "cpu"
    
    return torch.device(device)

def setup_device(config_device: str = "auto") -> torch.device:
    # grab info about device
    device = get_device(config_device)
    
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(idx)}")
        props = torch.cuda.get_device_properties(idx)
        logger.info(f"CUDA Memory: {props.total_memory / 1e9:.2f} GB")
        
        # set memory fraction if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    elif device.type == "mps":
        logger.info("Using MPS (Apple Silicon GPU)")
    
    else:
        logger.info("Using CPU")
    
    return device

def to_device(tensor_or_module: Union[torch.Tensor, torch.nn.Module], device: torch.device):
    """
    move tensor/module to specified device
    """
    return tensor_or_module.to(device)

def get_device_info() -> dict:
    # grabs comprehensive device info
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "device_count": 0,
        "device_names": [],
        "memory_info": {}
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_names"] = [torch.cuda.get_device_name(i) for i in range(info["device_count"])]
        
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["memory_info"][f"cuda:{i}"] = {
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}"
            }
    
    return info
