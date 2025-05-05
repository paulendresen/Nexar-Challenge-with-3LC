import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import math

class AttentionFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),  # Two attention scores (for short & long)
            nn.Softmax(dim=1)  # Normalize attention weights
        )
    
    def forward(self, short_feats, long_feats):
        combined = torch.cat([short_feats, long_feats], dim=1)
        attn_weights = self.attn(combined)  # Shape: (batch, 2)
        
        # Apply attention
        fused = attn_weights[:, 0:1] * short_feats + attn_weights[:, 1:2] * long_feats
        return fused

class NewCrashPredictor(nn.Module):
    def __init__(self, num_frames=16):
        super().__init__()

        self.num_frames = num_frames

        # Load pretrained backbone with proper weights
        self.backbone_rgb = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT)
        self.backbone_rgb.head = nn.Identity()

        # Output dimension from MViT V2 Small
        hidden_size = 768

        # Attention-based fusion
        self.fusion = AttentionFusion(hidden_size)

        # Stage 1: Binary classifier for event occurrence
        self.event_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Binary output: will event occur or not
        )

        # Stage 2: TTE predictor (only used when event is predicted to occur)
        self.tte_predictor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Weibull parameters (lambda, k)
        )

    def forward(self, x, return_embeddings=False):
        short_frames = x
        
        # Process through modified backbone
        short_features = self.backbone_rgb(short_frames)
       # long_features = self.backbone_rgb(long_frames)

        # Fuse features with attention mechanism
        fused_features = short_features  #self.fusion(short_features, long_features)

        # Stage 1: Predict if event will occur (return logits, not probability)
        event_logits = self.event_classifier(fused_features) 

        if return_embeddings:
            return event_logits,fused_features

        return event_logits
