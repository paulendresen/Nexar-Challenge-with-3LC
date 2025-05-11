import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import math

class NewCrashPredictor(nn.Module):
    def __init__(self, num_frames=16):
        super().__init__()

        self.num_frames = num_frames

        # Load pretrained backbone with proper weights
        self.backbone_rgb = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
        self.backbone_rgb.head = nn.Identity()

        # Output dimension from MViT V2 Small
        hidden_size = 768

        self.event_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Binary output: will event occur or not
        )

    def forward(self, x, return_embeddings=False):
        # Process through modified backbone
        features = self.backbone_rgb(x)

        event_logits = self.event_classifier(features) 

        if return_embeddings:
            return event_logits,features

        return event_logits
