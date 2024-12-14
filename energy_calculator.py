import numpy as np
import cupy as cp
import cv2

class EnergyCalculator:
    def __init__(self):
        self.device = cp.cuda.Device()
    
    def calculate_frame_energy(self, frame):
        """Calculate energy map for a single frame using GPU acceleration."""
        # Transfer frame to GPU
        frame_gpu = cp.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # Calculate gradients on GPU
        sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad_x = cp.abs(cp.correlate2d(frame_gpu, sobel_x, mode='same'))
        grad_y = cp.abs(cp.correlate2d(frame_gpu, sobel_y, mode='same'))
        
        # Calculate energy map
        energy = cp.sqrt(grad_x**2 + grad_y**2)
        
        # Transfer back to CPU
        return cp.asnumpy(energy)
    
    def calculate_temporal_energy(self, frames):
        """Calculate temporal energy between consecutive frames."""
        temporal_energy = np.zeros_like(frames[0], dtype=np.float32)
        
        for i in range(len(frames) - 1):
            frame1_gpu = cp.asarray(frames[i])
            frame2_gpu = cp.asarray(frames[i + 1])
            diff = cp.abs(frame2_gpu - frame1_gpu)
            temporal_energy += cp.asnumpy(diff)
            
        return temporal_energy / (len(frames) - 1)