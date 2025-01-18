import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoModel


class MultiModalModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64):
        super().__init__()
        
        # Image Encoder
        # Takes MNIST images (1 channel) and converts to embedding dimension
        self.image_encoder = nn.Sequential(
            # First conv block: 1->32 channels, downsample by 2
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # Output: 32 x 14 x 14
            nn.BatchNorm2d(32),  # Normalize activations
            nn.ReLU(),  # Non-linearity
            
            # Second conv block: 32->64 channels, downsample by 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: 64 x 7 x 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Global average pooling to get fixed size representation
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: 64 x 1 x 1
            nn.Flatten(),  # Convert to vector: 64
            
            # Project to embedding dimension
            nn.Linear(64, emb_dim),  # Output: emb_dim
            nn.LayerNorm(emb_dim)  # Final normalization
        )
        
        # Text Encoder
        # Transforms tokenized text to same embedding space
        self.text_embedding = nn.Embedding(vocab_size, emb_dim)  # Word embeddings
        self.text_encoder = nn.Sequential(
            # Expand dimension for better representation
            nn.Linear(emb_dim, emb_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            
            # Project back to embedding dimension
            nn.Linear(emb_dim*2, emb_dim),
            nn.LayerNorm(emb_dim)  # Match image embedding normalization
        )
        
        # Learnable temperature parameter for similarity scaling
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(self, images, text):
        # Process images
        img_features = self.image_encoder(images)  # Get image embeddings
        img_features = F.normalize(img_features, dim=-1)  # L2 normalize
        
        # Process text
        text_emb = self.text_embedding(text).mean(dim=1)  # Average word embeddings
        text_features = self.text_encoder(text_emb)  # Encode text
        text_features = F.normalize(text_features, dim=-1)  # L2 normalize
        
        return img_features, text_features