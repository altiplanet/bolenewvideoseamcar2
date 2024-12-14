"""Seam processing operations using GPU acceleration."""
import cupy as cp
import numpy as np
import logging
from typing import List, Tuple, Optional
import torch
from utils.gpu_utils import clear_gpu_memory
from utils.progress import create_progress_bar

class SeamProcessor:
    def __init__(self, cuda_device: Optional[cp.cuda.Device]):
        self.cuda_device = cuda_device
        self.use_gpu = cuda_device is not None
        logging.info(f"Initialized SeamProcessor with GPU support: {self.use_gpu}")
    
    def process_frame_batch(self, frames: List[np.ndarray], target_size: Tuple[int, int]) -> List[np.ndarray]:
        """Process a batch of frames to target size."""
        target_height, target_width = target_size
        processed_frames = []
        
        pbar = create_progress_bar(len(frames), "Processing frames")
        
        for frame in frames:
            try:
                # Process frame
                processed = self._process_single_frame(frame, target_height, target_width)
                processed_frames.append(processed)
                pbar.update(1)
            except Exception as e:
                logging.error(f"Error processing frame: {str(e)}")
                raise
        
        pbar.close()
        return processed_frames
    
    def _process_single_frame(self, frame: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """Process a single frame to target dimensions."""
        current = frame.copy()
        height, width = current.shape[:2]
        
        # Handle width changes
        if width < target_width:
            current = self._expand_width(current, target_width - width)
        elif width > target_width:
            current = self._reduce_width(current, width - target_width)
            
        # Handle height changes
        if height < target_height:
            current = self._expand_height(current, target_height - height)
        elif height > target_height:
            current = self._reduce_height(current, height - target_height)
            
        return current
    
    # ... rest of the implementation ...