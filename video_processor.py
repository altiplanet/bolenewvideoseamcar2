import cv2
import numpy as np
from frame_processor import FrameProcessor

class VideoProcessor:
    def __init__(self, input_path, output_path, target_width, target_height, num_workers=4):
        self.input_path = input_path
        self.output_path = output_path
        self.target_width = target_width
        self.target_height = target_height
        self.frame_processor = FrameProcessor(num_workers)
        
    def read_video(self):
        """Read video frames into memory."""
        cap = cv2.VideoCapture(self.input_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        return frames, cap.get(cv2.CAP_PROP_FPS)
    
    def write_video(self, frames, fps):
        """Write processed frames to output video."""
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
    
    def process(self):
        """Process the entire video."""
        print("Reading video...")
        frames, fps = self.read_video()
        
        # Process width reduction
        if self.target_width < frames[0].shape[1]:
            print("Processing horizontal seams...")
            frames = self.frame_processor.process_video(frames, self.target_width)
        
        # Process height reduction
        if self.target_height < frames[0].shape[0]:
            print("Processing vertical seams...")
            frames = [cv2.transpose(f) for f in frames]
            frames = self.frame_processor.process_video(frames, self.target_height)
            frames = [cv2.transpose(f) for f in frames]
        
        print("Writing output video...")
        self.write_video(frames, fps)