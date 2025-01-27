

# Multimodal MNIST Matching

This project implements a neural network that learns to match MNIST digit images with their textual descriptions. The model learns joint embeddings for both images and text, enabling bidirectional matching between the two modalities.

## Architecture

### Image Encoder
- Input: MNIST images (28×28×1)
- Two convolutional blocks with batch normalization
- Global pooling
- Linear projection to 64d embedding
- LayerNorm

### Text Encoder
- Input: Tokenized text descriptions
- Word embedding layer (64d)
- Mean pooling
- Two-layer MLP with dropout
- LayerNorm

## Training Details
- Contrastive learning with temperature scaling
- Bidirectional loss (I2T + T2I)
- Label smoothing (0.1)
- Balanced batch sampling
- Early stopping

## Results
- Achieved ~90% accuracy on both I2T and T2I matching
- Training converges in ~30 epochs
- Stable training with minimal overfitting

## Requirements
```
torch
torchvision
transformers
numpy
matplotlib
tqdm
```

## Usage
```python
# Train model
python main.py

# Parameters can be modified in the Trainer initialization:
trainer = Trainer(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=50,
    device='cuda'
)
```

## Visualizations
- Training curves (loss and accuracy)
- Similarity matrices
- Example predictions
- Architecture diagrams

## Key Features
1. Efficient architecture with minimal parameters
2. Robust text encoding with vocabulary mapping
3. Balanced sampling for better training
4. Comprehensive visualization tools 

