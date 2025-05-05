import os
import tlc
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import torch.nn as nn
import numpy as np
from pathlib import Path
# Set environment variables for multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

from tqdm import tqdm

from model import NewCrashPredictor
from train import TrainingScheduler
from utils import MapFn

def main():
    # Configuration
    config = {
        # Model parameters
        'num_frames': 16,  
        
        # Training parameters
        'batch_size': 16,
        'num_workers': 8,
        'learning_rate': 1e-4,  # Initial learning rate for cosine annealing
        'weight_decay': 1e-4,
        'patience': 100,
        'max_grad_norm': 1.0,
        'epochs': 100,
        'gradient_accumulation_steps': 1,

        # Scheduler parameters
        'scheduler_type': 'cosine',  # or 'cosine_warm_restarts'
        'T_max': 100,  # Same as epochs, for CosineAnnealingLR
        'T_0': 4,  # For CosineAnnealingWarmRestarts
        'T_mult': 2,  # For CosineAnnealingWarmRestarts
        'min_lr': 1e-6,  # Minimum learning rate
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load TLC tables
    train_table = tlc.Table.from_url("/home/ubuntu/PaulsDisk/3LC/NexarChallenge/datasets/train/tables/final").latest()
    val_table = tlc.Table.from_url("/home/ubuntu/PaulsDisk/3LC/NexarChallenge/datasets/val/tables/final").latest()

    print ("Url to train table: ", train_table.url) 
    print ("Url to val table: ", val_table.url)

    # Apply mapping functions
    train_table = train_table.map(MapFn(is_train=True))
    val_table = val_table.map(MapFn(is_train=False))

    weights = np.array(train_table.get_column("weight").to_numpy(), copy=True)

    # Train on 6000 weighted samples from the full set for each epoch
    train_sampler = WeightedRandomSampler(
        weights=np.array(weights),
        num_samples=6000, 
        replacement=True
    )

    # Group frames by video_id
    video_frames = {}
    for row_index, row in enumerate(tqdm(val_table.table_rows, desc="Processing validation rows")):
        video_id = row['video_id']
        if video_id not in video_frames:
            video_frames[video_id] = []
        video_frames[video_id].append((row_index, row['event_occurs']))

    validation_indices = []
    
    # reduce number of validation samples, take every 4th no collision and every 1 collision - on larger validation sets I used 64 and 4
    # Define sampling parameters
    NO_COLLISION_STEP = 4  # Frames to check/skip when no collision
    COLLISION_STEP = 1      # Frames to sample during collision

    for video_id, frames in video_frames.items():
        current_idx = 0
        collision_started = False
        
        while current_idx < len(frames):
            row_idx, event_occurs = frames[current_idx]
            
            if not collision_started:
                if event_occurs == False:
                    validation_indices.append(row_idx)
                    # Check next NO_COLLISION_STEP frames for collision, stop if found
                    next_frames = range(current_idx + 1, min(current_idx + NO_COLLISION_STEP, len(frames)))
                    found_collision = False
                    for check_idx in next_frames:
                        if frames[check_idx][1] > 0:
                            current_idx = check_idx  # Move to first collision frame
                            found_collision = True
                            break
                    if not found_collision:
                        current_idx += NO_COLLISION_STEP  # Only jump if no collision found
                else:
                    collision_started = True
                    validation_indices.append(row_idx)
                    current_idx += COLLISION_STEP  # Start sampling when collision begins
            else:
                validation_indices.append(row_idx)
                current_idx += COLLISION_STEP  # Continue sampling after collision starts
                
    randomValidationIndices = np.array(validation_indices)
    
    val_sampler = SubsetRandomSampler(indices=randomValidationIndices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_table,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_table,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Initialize model without num_classes parameter
    print("Initializing model...")
    #model = VideoMAE_CrashPrediction() 
    model = NewCrashPredictor()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Create training scheduler
    scheduler = TrainingScheduler(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    # Create output directories
    os.makedirs('checkpoints_newmodel', exist_ok=True)
    
    # Train the model
    print("Starting training...")
    scheduler.train(epochs=config['epochs'])
    print("Training completed!")

if __name__ == "__main__":
    main() 