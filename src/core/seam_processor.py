"""Seam processing operations using GPU acceleration."""
import cupy as cp
import numpy as np
import logging
from typing import List, Tuple
import torch
import torch.nn.functional as F

class SeamProcessor:
    def __init__(self, device):
        self.device = device
        logging.info("Initialized SeamProcessor with GPU support")
    
    def find_vertical_seam(self, energy_map: np.ndarray) -> np.ndarray:
        """Find vertical seam using dynamic programming on GPU."""
        try:
            # Transfer to GPU
            energy_gpu = cp.asarray(energy_map)
            height, width = energy_map.shape
            
            # Initialize cumulative energy map
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
            
            return cp.asnumpy(seam)
        except Exception as e:
            logging.error(f"Error finding vertical seam: {str(e)}")
            raise
    
    def expand_frame(self, frame: np.ndarray, seam: np.ndarray) -> np.ndarray:
        """Expand frame by duplicating pixels along seam."""
        try:
            height, width = frame.shape[:2]
            expanded = np.zeros((height, width + 1, frame.shape[2]), dtype=frame.dtype)
            
            for y in range(height):
                x = seam[y]
                expanded[y, :x] = frame[y, :x]
                expanded[y, x] = frame[y, x]  # Duplicate pixel
                expanded[y, x+1:] = frame[y, x:]
            
            return expanded
        except Exception as e:
            logging.error(f"Error expanding frame: {str(e)}")
            raise
    
    def process_frame_batch(self, frames: List[np.ndarray], target_size: Tuple[int, int]) -> List[np.ndarray]:
        """Process a batch of frames for expansion."""
        try:
            processed_frames = []
            current_height, current_width = frames[0].shape[:2]
            target_height, target_width = target_size
            
            # Convert frames to torch tensors for batch processing
            frames_tensor = torch.stack([
                torch.from_numpy(frame).cuda() 
                for frame in frames
            ])
            
            while current_width < target_width or current_height < target_height:
                if current_width < target_width:
                    # Process vertical expansion
                    seam = self.find_vertical_seam(frames_tensor.mean(dim=0).cpu().numpy())
                    frames = [self.expand_frame(frame, seam) for frame in frames]
                    current_width += 1
                    logging.info(f"Expanded width to {current_width}/{target_width}")
                
                if current_height < target_height:
                    # Process horizontal expansion (transpose frames)
                    frames = [frame.transpose(1, 0, 2) for frame in frames]
                    seam = self.find_vertical_seam(frames_tensor.mean(dim=0).cpu().numpy())
                    frames = [self.expand_frame(frame, seam).transpose(1, 0, 2) for frame in frames]
                    current_height += 1
                    logging.info(f"Expanded height to {current_height}/{target_height}")
            
            return frames
        except Exception as e:
            logging.error(f"Error processing frame batch: {str(e)}")
            raise