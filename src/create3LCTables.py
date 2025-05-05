from pathlib import Path
import tlc
import pandas as pd
import os
from tqdm import tqdm
from preprocess_videos import process_videos
import cv2
from sklearn.model_selection import train_test_split

def ensure_frames_exist(video_dir, frames_dir):
    """Ensure frames are extracted from videos"""
    if not Path(frames_dir).exists():
        print(f"Frames directory not found. Extracting frames from videos...")
        metadata_df = process_videos(
            input_dir=video_dir,
            output_dir=frames_dir,
            num_workers=8
        )
        return metadata_df
    return None

def get_frame_path(frames_dir, video_id, frame_num):
    """Get the full path to a specific frame"""
    # Convert to absolute path using Path.resolve()
    return str(Path(frames_dir).resolve() / f"{video_id}" / f"frame_{frame_num:05d}.jpg")

def get_video_fps(video_path):
    """Get FPS from video file"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps)
    except Exception as e:
        print(f"Error getting FPS from video {video_path}: {e}")
        return 30.0  # Default FPS if unable to read

def create_3lc_tables(video_dir, frames_dir, metadata_csv, dataset_name, table_name, video_ids=None, 
                     project_name="NexarChallenge", 
                     description=None,
                     max_videos=None):
    """Create 3LC tables for video frames
    
    Args:
        video_dir: Directory containing videos
        frames_dir: Directory containing extracted frames
        metadata_csv: Path to metadata CSV
        dataset_name: Either 'train' or 'val'
        table_name: Name of the table to create
        video_ids: List of video IDs to process (if None, process all)
        project_name: Name of the project (default: "NexarChallenge")
        description: Custom description for the table (default: auto-generated)
        max_videos: Maximum number of videos to process (default: None, process all)
    """
    # Ensure frames exist
    frames_metadata = ensure_frames_exist(video_dir, frames_dir)
    
    # Load metadata
    df = pd.read_csv(metadata_csv)
    
    # Filter by video_ids if provided
    if video_ids is not None:
        df = df[df['id'].apply(lambda x: f"{int(x):05d}").isin(video_ids)]
    
    # Limit number of videos if max_videos is specified
    if max_videos is not None:
        df = df.head(max_videos)
    
    # Specify the schemas for the columns
    schemas = {
        "video_id": tlc.Schema(value=tlc.StringValue(), writable=False),
        "frame": tlc.ImagePath,
        "frame_number": tlc.Schema(value=tlc.Int32Value(), writable=False),
        "fps": tlc.Schema(value=tlc.Float32Value(), writable=False),
        "event_occurs": tlc.Schema(value=tlc.BoolValue(), writable=False),
        "time_to_alert": tlc.Schema(value=tlc.Float32Value(), writable=False),
        "time_to_event": tlc.Schema(value=tlc.Float32Value(), writable=False),
        "has_event": tlc.Schema(value=tlc.BoolValue(), writable=False),
        "weight": tlc.SampleWeightSchema(),
    }

    # Create table writer
    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=dataset_name,
        project_name=project_name,
        description=description or f"{dataset_name} split: Video frames with collision probability based on 16 frames leading up to this frame",
        column_schemas=schemas,
    )

    # Process each video
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating 3LC table"):
        video_id = f"{int(row['id']):05d}"
        video_path = Path(video_dir) / f"{video_id}.mp4"
        
        # Get FPS from video file
        fps = get_video_fps(video_path)
        
        # Get frames directory for this video
        frames_path = Path(frames_dir) / video_id
        if not frames_path.exists():
            print(f"Warning: No frames found for video {video_id}")
            continue

        # Get all frame files and sort them
        frame_files = sorted(frames_path.glob("frame_*.jpg"))
        
        # Calculate event frame if it's a crash video
        event_frame = None
        alert_frame = None
        
        if row['target'] == 1 and 'time_of_event' in row:
            event_frame = int(row['time_of_event'] * fps)
            alert_frame = int(row['time_of_alert'] * fps)

        # Skip first frames as each frame/training sample when loaded loads 16 frames with step 4 before it
        for frame_file in frame_files[63:]:
            # Extract frame number from filename
            frame_num = int(frame_file.stem.split('_')[1])
            
            # Skip frames if less than 0.3s away from event OR more than 0.7s after alert
            if (event_frame is not None and frame_num > event_frame):
                break  # Break, we only want to train up until collision

            # Calculate collision probability and time deltas
            event_occurs = False

            if event_frame is not None:
                if (frame_num >= alert_frame):
                    event_occurs = True

                time_to_alert = (alert_frame - frame_num) / fps
                time_to_event = (event_frame - frame_num) / fps
            else:
                time_to_alert = 0.0
                time_to_event = 0.0

            # Add row to table
            table_writer.add_row({
                "video_id": video_id,
                "frame": str(frame_file.resolve()),
                "frame_number": frame_num,
                "fps": fps,
                "event_occurs": event_occurs,
                "time_to_alert": time_to_alert,
                "time_to_event": time_to_event,
                "has_event": event_frame is not None,
                "weight": 1.0,
            })

    # Finalize and return the table
    table = table_writer.finalize()
    print(f"Created table with {len(table)} rows from {len(df)} videos")
    return table

if __name__ == "__main__":
    video_dir = './train'
    frames_dir = './train_frames256'
    metadata_csv = 'train.csv'
    
    pathIamIn = str(Path(__file__).resolve().parent.parent)
    print(pathIamIn)

    # ALIAS so images can be loaded from different locations - this yaml file will reside in the 3LC projects/NexarChallenge folder
    tlc.register_project_url_alias("NEXAR_CHALLENGE", pathIamIn, project="NexarChallenge")

    # Load metadata to get video IDs
    df = pd.read_csv(metadata_csv)
    all_video_ids = [f"{int(x):05d}" for x in df['id']]
    
    # Create train/val split (20% validation)
    train_ids, val_ids = train_test_split(all_video_ids, test_size=0.02, random_state=42)
    
    # Create train table
    train_table = create_3lc_tables(
        video_dir, 
        frames_dir,
        metadata_csv, 
        "train", 
        "final",
        train_ids,
        project_name="NexarChallenge",
        description="Training split for collision prediction model"
    )

    print(f"Created train table with {len(train_table)} rows")
    
    # Create validation table
    val_table = create_3lc_tables(
        video_dir, 
        frames_dir,
        metadata_csv, 
        "val", 
        "final",
        val_ids,
        project_name="NexarChallenge",
        description="Validation split for collision prediction model",
    )

    print(f"Created validation table with {len(val_table)} rows")
