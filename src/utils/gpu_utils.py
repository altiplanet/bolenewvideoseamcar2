"""GPU utilities and initialization."""
import logging
import cupy as cp
import torch

def setup_gpu():
    """Initialize and configure GPU devices."""
    try:
        # Initialize CUDA device
        device = cp.cuda.Device()
        logging.info(f"Using CUDA Device: {device.name}")
        
        # Check available GPU memory
        mem_info = device.mem_info
        total_memory = mem_info[1] / (1024**3)  # Convert to GB
        free_memory = mem_info[0] / (1024**3)
        logging.info(f"Total GPU Memory: {total_memory:.2f} GB")
        logging.info(f"Free GPU Memory: {free_memory:.2f} GB")
        
        # Set up PyTorch device
        torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"PyTorch using device: {torch_device}")
        
        return device, torch_device
    except Exception as e:
        logging.error(f"GPU initialization failed: {str(e)}")
        raise