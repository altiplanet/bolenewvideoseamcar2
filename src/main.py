"""Main entry point for video seam carving application."""
import argparse
import logging
import os
from utils.logger import setup_logging
from utils.gpu_utils import setup_gpu
from utils.video_utils import read_video, write_video
from core.energy_calculator import EnergyCalculator
from core.seam_processor import SeamProcessor

def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated video seam carving")
    parser.add_argument("input_path", help="Path to input video")
    parser.add_argument("output_path", help="Path to output video")
    parser.add_argument("target_width", type=int, help="Target width")
    parser.add_argument("target_height", type=int, help="Target height")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of frames to process in parallel")
    parser.add_argument("--notebook", action="store_true", help="Running in Jupyter notebook")
    args = parser.parse_args()
    
    # Initialize logging
    setup_logging(notebook_mode=args.notebook)
    logging.info("Starting video seam carving process")
    
    try:
        # Initialize GPU
        device, torch_device = setup_gpu()
        
        # Read input video
        frames, fps, (orig_width, orig_height) = read_video(args.input_path)
        logging.info(f"Processing video expansion from {orig_width}x{orig_height} to {args.target_width}x{args.target_height}")
        
        # Initialize processors
        energy_calc = EnergyCalculator(device)
        seam_processor = SeamProcessor(device)
        
        # Process video in batches
        processed_frames = []
        for i in range(0, len(frames), args.batch_size):
            batch = frames[i:i + args.batch_size]
            processed_batch = seam_processor.process_frame_batch(
                batch, 
                (args.target_height, args.target_width)
            )
            processed_frames.extend(processed_batch)
        
        # Write output video
        write_video(args.output_path, processed_frames, fps)
        logging.info("Video processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error during video processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()