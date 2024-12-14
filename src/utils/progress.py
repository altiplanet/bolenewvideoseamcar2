"""Progress tracking utilities for video processing."""
import logging
from tqdm import tqdm  # Changed from tqdm.notebook to tqdm for better compatibility

class ProgressBar:
    """Custom progress bar with logging."""
    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc)
        self.total = total
        self.current = 0
        self.desc = desc

    def update(self, n=1):
        self.current += n
        self.pbar.update(n)
        if self.current % (self.total // 10) == 0:  # Log every 10%
            percentage = (self.current / self.total) * 100
            logging.info(f"{self.desc}: {self.current}/{self.total} ({percentage:.1f}%)")

    def close(self):
        self.pbar.close()

def create_progress_bar(total: int, desc: str) -> ProgressBar:
    """Create a progress bar with the given parameters."""
    return ProgressBar(total, desc)