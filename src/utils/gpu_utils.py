"""GPU utilities and initialization."""
import logging
import cupy as cp
import torch
from typing import Tuple, Optional

def get_cuda_device() -> Optional[cp.cuda.Device]:
    """Get CUDA device if available."""
    try:
        device = cp.cuda.Device()
        device_name = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode('utf-8')
        logging.info(f"Using CUDA Device: {device_name}")
        return device
    except Exception as e:
        logging.warning(f"CUDA device initialization failed: {str(e)}")
        return None

def get_torch_device() -> torch.device:
    """Get PyTorch device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.warning("PyTorch using CPU - GPU not available")
    return device

def log_gpu_memory() -> None:
    """Log GPU memory usage."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            cached = torch.cuda.memory_reserved() / 1024**3
            logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    except Exception as e:
        logging.warning(f"Could not get GPU memory info: {str(e)}")

def setup_gpu() -> Tuple[Optional[cp.cuda.Device], torch.device]:
    """Initialize and configure GPU devices."""
    logging.info("Initializing GPU...")
    
    # Get CUDA device for CuPy
    cuda_device = get_cuda_device()
    
    # Get PyTorch device
    torch_device = get_torch_device()
    
    # Log memory usage
    log_gpu_memory()
    
    return cuda_device, torch_device

def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cp.cuda.runtime.getDeviceCount() > 0:
            cp.get_default_memory_pool().free_all_blocks()
        logging.info("GPU memory cache cleared")
    except Exception as e:
        logging.warning(f"Error clearing GPU memory: {str(e)}")