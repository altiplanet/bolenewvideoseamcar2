import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from energy_calculator import EnergyCalculator
from seam_finder import SeamFinder

class FrameProcessor:
    def __init__(self, num_workers=4):
        self.energy_calculator = EnergyCalculator()
        self.seam_finder = SeamFinder()
        self.num_workers = num_workers
        
    def remove_seam(self, frame, seam):
        """Remove a seam from the frame."""
        height, width = frame.shape[:2]
        new_frame = np.zeros((height, width-1, frame.shape[2]), dtype=frame.dtype)
        
        for y in range(height):
            new_frame[y, :seam[y]] = frame[y, :seam[y]]
            new_frame[y, seam[y]:] = frame[y, seam[y]+1:]
            
        return new_frame
    
    def process_frame_group(self, frames):
        """Process a group of consecutive frames."""
        # Calculate energy maps
        energy_maps = []
        temporal_energy = self.energy_calculator.calculate_temporal_energy(frames)
        
        for frame in frames:
            spatial_energy = self.energy_calculator.calculate_frame_energy(frame)
            total_energy = spatial_energy + temporal_energy
            energy_maps.append(total_energy)
            
        # Find optimal seam
        seam = self.seam_finder.find_seam(np.mean(energy_maps, axis=0))
        
        # Remove seam from all frames
        processed_frames = []
        for frame in frames:
            processed_frames.append(self.remove_seam(frame, seam))
            
        return processed_frames
    
    def process_video(self, frames, target_width):
        """Process entire video in parallel."""
        num_seams = frames[0].shape[1] - target_width
        current_frames = frames
        
        for i in range(num_seams):
            print(f"Removing seam {i+1}/{num_seams}")
            
            # Process frames in groups
            frame_groups = [current_frames[i:i+5] for i in range(0, len(current_frames), 5)]
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                processed_groups = list(executor.map(self.process_frame_group, frame_groups))
                
            # Flatten processed groups
            current_frames = [frame for group in processed_groups for frame in group]
            
        return current_frames