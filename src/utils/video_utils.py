"""Video processing utilities."""
import cv2
import logging
import numpy as np
from typing import Tuple, List
from .progress import ProgressTracker

def read_video(path: str) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """Read video file and return frames, fps, and dimensions."""
    logging.info(f"Reading video: {path}")
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Video properties - FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
    
    # Create progress bar
    pbar = ProgressTracker.frame_reader(total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    ProgressTracker.log_progress("Frames read", len(frames), total_frames)
    return frames, fps, (width, height)

def write_video(path: str, frames: List[np.ndarray], fps: float):
    """Write frames to video file."""
    if not frames:
        raise ValueError("No frames to write")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    logging.info(f"Writing video to {path}")
    pbar = ProgressTracker.frame_writer(len(frames))
    
    for frame in frames:
        out.write(frame)
        pbar.update(1)
    
    pbar.close()
    out.release()
    logging.info(f"Successfully wrote {len(frames)} frames to {path}")