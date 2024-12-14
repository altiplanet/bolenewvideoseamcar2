"""Progress tracking utilities."""
from tqdm.notebook import tqdm
import logging

class ProgressTracker:
    """Track and display progress for long-running operations."""
    
    @staticmethod
    def frame_reader(total_frames):
        """Create a progress bar for frame reading."""
        return tqdm(
            total=total_frames,
            desc="Reading frames",
            unit="frames"
        )
    
    @staticmethod
    def frame_processor(total_frames):
        """Create a progress bar for frame processing."""
        return tqdm(
            total=total_frames,
            desc="Processing frames",
            unit="frames"
        )
    
    @staticmethod
    def frame_writer(total_frames):
        """Create a progress bar for frame writing."""
        return tqdm(
            total=total_frames,
            desc="Writing frames",
            unit="frames"
        )
    
    @staticmethod
    def log_progress(message, current, total):
        """Log progress with percentage."""
        percentage = (current / total) * 100
        logging.info(f"{message}: {current}/{total} ({percentage:.1f}%)")