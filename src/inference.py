import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import tlc
import pacmap
from torch.utils.data import DataLoader
from model import NewCrashPredictor
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from PIL import Image
from utils import MapFn
from train import event_loss

def load_model(model_path):
    """Load the trained model"""
    model = NewCrashPredictor()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model

def get_embeddings(model, dataloader, device):
    """Extract embeddings and predictions from the model"""
    model.eval()
    all_embeddings = []
    all_predictions = []
    losses = []
    event_probs = []
    #sigmoid = nn.Sigmoid()
    
    with torch.no_grad():
        for videos, event_occurs in tqdm(dataloader, desc="Extracting embeddings"):
            short_frames = videos
            short_frames = short_frames.to(device)
            event_occurs = event_occurs.to(device)

            # Get model outputs
            event_logits, embeddings_batch = model(short_frames, return_embeddings=True)

            event_logits = event_logits.squeeze(1)  # Only squeeze dimension 1
            
            loss = event_loss(event_logits, event_occurs, mean=False)
            
            # Convert event_logits to probabilities, keeping the same shape
            event_prob = torch.sigmoid(event_logits)
            
            # Make sure to convert to numpy array with at least 1 dimension
            event_probs.extend(event_prob.cpu().numpy().reshape(-1))

            all_embeddings.extend(embeddings_batch.cpu().numpy())
            losses.extend(loss.cpu().numpy())

    return np.array(all_embeddings),  np.array(losses), np.array(event_probs)

def main():
    # Configuration
    config = {
        'num_frames': 16,
        'batch_size': 32,
        'num_workers': 16
    }
    
    # Initialize 3LC run
    run = tlc.init(
        "NexarChallenge",
        run_name="del025weight_61", 
        description="Analyze val set with trained model",
        parameters=config
    )
    
    # Register the dataset root path as an alias
    pathIamIn = str(Path(__file__).resolve().parent.parent)
    tlc.register_url_alias("NEXAR_DATA", pathIamIn)

    # Load TLC tables
    #train_table = tlc.Table.from_url("/home/ubuntu/PaulsDisk/3LC/NexarChallenge/datasets/train/tables/initialTTE").latest()
    train_table = tlc.Table.from_url("/home/ubuntu/PaulsDisk/3LC/NexarChallenge/datasets/train/tables/final").latest()
    val_table = tlc.Table.from_url("/home/ubuntu/PaulsDisk/3LC/NexarChallenge/datasets/val/tables/initialTTE").latest()
    test_table = tlc.Table.from_url("/home/ubuntu/PaulsDisk/3LC/NexarChallenge/datasets/test/tables/initial")

    # Apply mapping functions
    train_table = train_table.map(MapFn())
    val_table = val_table.map(MapFn())
    test_table = test_table.map(MapFn(is_inference=True))

    # Create dataloaders
    train_loader = DataLoader(
       train_table,
       batch_size=config['batch_size'],
       shuffle=False,
       num_workers=config['num_workers'],
       pin_memory=True,
       persistent_workers=True,
       prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_table,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_table,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('checkpoints_newmodel/del025weight_61_acc_93.65_CP_81.80_CR_52.01_NCP_94.51_NCR_98.62.pt') #Score = 0.898
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Get embeddings and predictions
    # print("Processing training data...")
    # train_embeddings, train_losses, train_event_probs = get_embeddings(model, train_loader, device)
    
    # print("Processing validation data...")
    # val_embeddings, val_losses, val_event_probs = get_embeddings(model, val_loader, device)

    print("Processing test data...")
    test_embeddings, test_losses, test_event_probs = get_embeddings(model, test_loader, device)
      
    # Reduce dimensionality with PaCMAP
    print("Reducing embedding dimensions with PaCMAP...")

    # Reshape embeddings if they are more than 2D
    # train_embeddings_reshaped = train_embeddings.reshape(train_embeddings.shape[0], -1)
    #val_embeddings_reshaped = val_embeddings.reshape(val_embeddings.shape[0], -1)
    test_embeddings_reshaped = test_embeddings.reshape(test_embeddings.shape[0], -1)
    
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=10)
    #train_embeddings = reducer.fit_transform(train_embeddings_reshaped)
    #val_embeddings = reducer.transform(val_embeddings_reshaped, train_embeddings_reshaped)

    #val_embeddings = reducer.fit_transform(val_embeddings_reshaped)
    test_embeddings = reducer.fit_transform(test_embeddings_reshaped)

    #Add metrics to 3LC run
    # train_metrics = {
    #     "embeddings": list(train_embeddings),
    #     "loss": train_losses.flatten(),
    #     "event_probs": train_event_probs.flatten(),
    # }

    # val_metrics = {
    #     "embeddings": list(val_embeddings),
    #     "loss": val_losses.flatten(),
    #     "event_probs": val_event_probs.flatten(),
    # }

    test_metrics = {
        "embeddings": list(test_embeddings),
        "loss": test_losses.flatten(),
        "event_probs": test_event_probs.flatten(),
    }
        # Add metrics to 3LC tables
    #run.add_metrics(train_metrics, foreign_table_url=train_table.url)
    #run.add_metrics(val_metrics, foreign_table_url=val_table.url)
    run.add_metrics(test_metrics, foreign_table_url=test_table.url)
    
    print("Inference and analysis completed!")

if __name__ == "__main__":
    main() 
