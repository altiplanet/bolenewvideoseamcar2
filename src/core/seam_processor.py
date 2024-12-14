"""Seam processing operations using GPU acceleration."""
import cupy as cp
import numpy as np
import logging
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from ..utils.gpu_utils import clear_gpu_memory

class SeamProcessor:
    def __init__(self, cuda_device: Optional[cp.cuda.Device], torch_device: torch.device):
        self.cuda_device = cuda_device
        self.torch_device = torch_device
        self.use_gpu = cuda_device is not None
        logging.info(f"Initialized SeamProcessor with GPU support: {self.use_gpu}")
    
    def _to_gpu(self, array: np.ndarray) -> cp.ndarray:
        """Transfer array to GPU if available."""
        return cp.asarray(array) if self.use_gpu else array
    
    def _to_cpu(self, array: cp.ndarray) -> np.ndarray:
        """Transfer array back to CPU if needed."""
        return cp.asnumpy(array) if self.use_gpu else array
    
    def find_vertical_seam(self, energy_map: np.ndarray) -> np.ndarray:
        """Find vertical seam using dynamic programming."""
        try:
            height, width = energy_map.shape
            if self.use_gpu:
                energy_gpu = self._to_gpu(energy_map)
                cum_energy = cp.zeros_like(energy_gpu)
                cum_energy[0] = energy_gpu[0]
                
                # Calculate cumulative minimum energy
                for i in range(1, height):
                    left = cp.roll(cum_energy[i-1], 1)
                    right = cp.roll(cum_energy[i-1], -1)
                    left[:, 0] = cp.inf
                    right[:, -1] = cp.inf
                    
                    cum_energy[i] = energy_gpu[i] + cp.minimum(
                        cp.minimum(left, cum_energy[i-1]),
                        right
                    )
                
                # Find optimal seam
                seam = cp.zeros(height, dtype=cp.int32)
                seam[-1] = cp.argmin(cum_energy[-1])
                
                # Backtrack
                for i in range(height-2, -1, -1):
                    prev_x = seam[i+1]
                    left = prev_x - 1 if prev_x > 0 else 0
                    right = prev_x + 2 if prev_x < width - 1 else width
                    seam[i] = left + cp.argmin(cum_energy[i, left:right])
                
                return self._to_cpu(seam)
            else:
                # CPU fallback implementation
                cum_energy = np.zeros_like(energy_map)
                cum_energy[0] = energy_map[0]
                # ... (rest of CPU implementation)
                return seam
                
        except Exception as e:
            logging.error(f"Error finding vertical seam: {str(e)}")
            raise
        finally:
            if self.use_gpu:
                clear_gpu_memory()
    
    # ... (rest of the class implementation)