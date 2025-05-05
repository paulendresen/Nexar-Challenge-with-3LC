import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def extract_frames_and_metadata(args):
    """Extract frames and return video metadata"""
    video_path, output_dir = args
    try:
        # Create output directory for this video
        video_name = Path(video_path).stem
        # Ensure video_id is zero-padded to 5 digits
        video_id = f"{int(video_name):05d}"
        frames_dir = os.path.join(output_dir, video_id)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to 256x256 using high quality Lanczos interpolation
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            # Save frame as JPEG
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        cap.release()
        
        # Return metadata with zero-padded video_id
        return {
            'video_id': video_id,  # Using zero-padded ID
            'fps': fps,
            'total_frames': frame_count,
            'duration': duration
        }
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None

def process_videos(input_dir, output_dir, num_workers=8):
    """Process videos and return metadata DataFrame"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all MP4 files
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} videos to process")
    print("First few video paths:", video_files[:5])
    
    # Create arguments list for parallel processing
    args_list = [(video_path, output_dir) for video_path in video_files]
    
    # Process videos in parallel and collect metadata
    metadata_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(extract_frames_and_metadata, args_list),
            total=len(video_files),
            desc="Processing videos"
        ))
        
        # Filter out None results and collect metadata
        metadata_list = [meta for meta in results if meta is not None]
    
    # Create DataFrame from metadata
    df = pd.DataFrame(metadata_list)
    print("\nFirst few rows of metadata:")
    print(df.head())
    print("\nVideo ID examples:", df['video_id'].values[:5])
    return df

if __name__ == "__main__":
    # Process training videos
    print("Processing training videos...")
    train_metadata = process_videos(
        input_dir='./train',
        output_dir='./train_frames256',
        num_workers=32
    )
    train_metadata.to_csv('train_frames_metadata.csv', index=False)
    
    # Process test videos if they exist
    if os.path.exists('./test'):
        print("\nProcessing test videos...")
        test_metadata = process_videos(
            input_dir='./test',
            output_dir='./test_frames256',
            num_workers=32
        )
        test_metadata.to_csv('test_frames_metadata.csv', index=False)
    
    print("\nPreprocessing completed!") 