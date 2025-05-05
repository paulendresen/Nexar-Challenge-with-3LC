import tlc
import numpy as np
import torch
from tqdm import tqdm
import csv
from scipy.special import expit

run = tlc.Run.from_url("/home/ubuntu/PaulsDisk/3LC/NexarChallenge/runs/del025weight_61_acc_93.65_CP_81.80_CR_52.01_NCP_94.51_NCR_98.62.pt")
table = tlc.Table.from_url("/home/ubuntu/PaulsDisk/3LC/NexarChallenge/datasets/test/tables/initial")
# Convert predictions to numpy array
predictions = np.array(run.metrics_tables[0].get_column('event_probs'))

print(len(predictions))
print(len(table))

# Configuration flags
skip_initial_frames = True
frames_to_skip = 64
process_last_n_frames = 1  # Set to None to process all frames

# First pass: collect all frames per video
video_frames = {}
for idx, row in enumerate(tqdm(table.table_rows, desc="Collecting video frames")):
    video_id = row['video_id']
    if video_id not in video_frames:
        video_frames[video_id] = []
    video_frames[video_id].append({
        'idx': idx,
        'pred': predictions[idx]
    })

# Process videos with frame constraints
video_predictions = {}
for video_id, frames in tqdm(video_frames.items(), desc="Processing videos"):
    max_pred = 0.0
    
    # Calculate which absolute indices to process
    if process_last_n_frames is not None:
        frame_indices = [len(frames) - 1]  # Just the last frame
    else:
        frame_indices = range(frames_to_skip if skip_initial_frames else 0, len(frames))

    # Process frames
    for abs_idx in frame_indices:
        current_pred = frames[abs_idx]['pred']# np.percentile(preds_window, 94)
        max_pred = max(max_pred, current_pred)
    
    video_predictions[video_id] = max_pred

# Write predictions to CSV
with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'target'])  # Header
    for video_id, pred in video_predictions.items():
        writer.writerow([video_id, pred])

print(f"Processed {len(video_predictions)} videos and saved predictions to submission.csv")




    