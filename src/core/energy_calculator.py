"""Energy map calculation using GPU acceleration."""
import cupy as cp
import cv2
import numpy as np
import logging
from typing import List

class EnergyCalculator:
    def __init__(self, device):
        self.device = device
        self.sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        logging.info("Initialized EnergyCalculator with GPU support")
    
    def calculate_frame_energy(self, frame: np.ndarray) -> np.ndarray:
        """Calculate energy map for a single frame using GPU."""
        try:
            # Transfer frame to GPU and convert to grayscale
            frame_gpu = cp.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
            # Calculate gradients
            grad_x = cp.abs(cp.correlate2d(frame_gpu, self.sobel_x, mode='same'))
            grad_y = cp.abs(cp.correlate2d(frame_gpu, self.sobel_y, mode='same'))
            
            # Calculate energy map
            energy = cp.sqrt(grad_x**2 + grad_y**2)
            
            return cp.asnumpy(energy)
        except Exception as e:
            logging.error(f"Error calculating frame energy: {str(e)}")
            raise
    
    def calculate_temporal_energy(self, frames: List[np.ndarray]) -> np.ndarray:
        """Calculate temporal energy between consecutive frames."""
        try:
            temporal_energy = cp.zeros_like(cp.asarray(frames[0]), dtype=cp.float32)
            
            for i in range(len(frames) - 1):
                frame1_gpu = cp.asarray(frames[i])
                frame2_gpu = cp.asarray(frames[i + 1])
                diff = cp.abs(frame2_gpu - frame1_gpu)
                temporal_energy += diff
            
            temporal_energy /= (len(frames) - 1)
            return cp.asnumpy(temporal_energy)
        except Exception as e:
            logging.error(f"Error calculating temporal energy: {str(e)}")
            raise