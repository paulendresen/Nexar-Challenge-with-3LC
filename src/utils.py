import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

class MapFn:
    def __init__(self, is_train=False, is_inference=False):
        self.is_train = is_train
        self.is_inference = is_inference

    def __call__(self, sample):
        # Get current frame path and collision probability
        current_frame = sample['frame']

        if self.is_inference:
            event_occurs = 0.0
        else:
            if sample['event_occurs']:
                event_occurs = 1.0

            else:
                event_occurs = 0.0

        current_frame_num = int(Path(current_frame).stem.split('_')[1])
        frame_dir = str(Path(current_frame).parent)
        
        # Get frame paths for both sequences
        frame_paths  = []  # step 1 sequence
        
        # Get previous 15 frames with step 1 (short sequence)
        for i in range(current_frame_num - 15 * 4, current_frame_num + 1, 4):
            frame_num = max(1, i)  # Ensure frame number is at least 1
            frame_path = str(Path(frame_dir) / f"frame_{frame_num:05d}.jpg")
            frame_paths.append(frame_path)
        
        # Set the spatial transform parameters outside the loop for consistent augmentation
        if self.is_train:
            # Get random rotation angle between -10 and 10 degrees
            angle = float(torch.rand(1) * 20 - 10)  # rand gives [0,1], scale to [-10,10]
            
            # Get random scale factor between 224/256 and 1.0
            scale_factor = float(torch.rand(1) * (1.0 - 224/256) + 224/256)
            
            # Calculate new size after scaling from 256x256
            new_size = (int(256 * scale_factor), int(256 * scale_factor))
            
            # Calculate crop parameters directly
            max_x = new_size[0] - 224
            max_y = new_size[1] - 224
            i = int(torch.rand(1) * max_y)  # random y start
            j = int(torch.rand(1) * max_x)  # random x start
            h, w = 224, 224  # fixed crop size
            
            do_hflip = torch.rand(1) < 0.5
            
            # Color jitter params
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            color_jitter_params = [
                param[0].item() if isinstance(param, torch.Tensor) else param
                for param in color_jitter.get_params(
                    color_jitter.brightness,
                    color_jitter.contrast,
                    color_jitter.saturation,
                    color_jitter.hue
                )
            ]
        
        frames = []
        
        # Process short sequence (step 1)
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            
            if self.is_train:
                # First rotate
                frame = transforms.functional.rotate(frame, angle)
                
                # Then scale
                frame = transforms.functional.resize(frame, new_size)
                
                # Then random crop
                frame = transforms.functional.crop(frame, i, j, h, w)
                
                # Rest of the transforms
                frame = transforms.functional.hflip(frame) if do_hflip else frame
                frame = transforms.functional.adjust_brightness(frame, color_jitter_params[0])
                frame = transforms.functional.adjust_contrast(frame, color_jitter_params[1])
                frame = transforms.functional.adjust_saturation(frame, color_jitter_params[2])
                frame = transforms.ToTensor()(frame)
                frame = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])(frame)
            else:
                frame = transforms.functional.resize(frame, (224, 224))
                frame = transforms.ToTensor()(frame)
                frame = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])(frame)
            
            frames.append(frame)
        
        # Stack and arrange both sequences
        frames = torch.stack(frames)  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        return frames, torch.tensor(event_occurs, dtype=torch.float32) 