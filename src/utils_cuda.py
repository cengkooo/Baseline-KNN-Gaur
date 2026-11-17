"""
CUDA and Hardware Utility Module for NVIDIA GeForce RTX 3070 Ti & PyTorch.
Provides device selection, VRAM memory logging, and tensor helpers.
"""

from typing import Optional, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_torch_device(prefer_cuda: bool = True) -> 'torch.device':
    """
    Returns the optimal PyTorch device.
    Prioritizes CUDA on NVIDIA GeForce RTX 3070 Ti when available.
    """
    if not TORCH_AVAILABLE:
        return 'cpu'
        
    if prefer_cuda and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"[⚡ GPU ACTIVE] Using: {device_name} | CUDA: {cuda_version}")
        return torch.device('cuda:0')
    else:
        print("[ℹ️ CPU ACTIVE] CUDA not available or not preferred. Using CPU.")
        return torch.device('cpu')


def log_vram_usage(prefix: str = "[RTX 3070 Ti VRAM]"):
    """
    Prints currently allocated and reserved GPU memory in Megabytes.
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"{prefix} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        print(f"{prefix} GPU memory stats unavailable (running on CPU).")


def to_torch_tensor(arr: Union[np.ndarray, list], device: Optional[Union[str, 'torch.device']] = None, dtype: Optional['torch.dtype'] = None) -> 'torch.Tensor':
    """
    Converts a NumPy array or list to a PyTorch tensor on the requested device.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in the current environment.")
        
    if dtype is None:
        dtype = torch.float32
        
    if isinstance(arr, np.ndarray):
        tensor = torch.from_numpy(arr).to(dtype=dtype)
    else:
        tensor = torch.tensor(arr, dtype=dtype)
        
    if device is not None:
        tensor = tensor.to(device)
    return tensor
