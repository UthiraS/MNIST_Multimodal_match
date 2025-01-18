import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

from model import *
from helpers import * 

class Trainer:
    def __init__(self, learning_rate: float, batch_size: int, num_epochs: int, device: str = 'cuda:0'):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        # Get digit mappings
        self.int_to_str, vocab_size = create_digit_mappings()
        print(f"Vocabulary size: {vocab_size}")
        
        # Initialize model
        self.model = MultiModalModel(vocab_size=vocab_size).to(device)
        
        # Set different learning rates for different components
        params = [
            {'params': self.model.image_encoder.parameters(), 'lr': learning_rate},
            {'params': self.model.text_embedding.parameters(), 'lr': learning_rate * 0.1},
            {'params': self.model.text_encoder.parameters(), 'lr': learning_rate * 0.1}
        ]
        self.optimizer = torch.optim.AdamW(params)
        
        # Metrics tracking (single source of truth)
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_i2t_acc': [], 'train_t2i_acc': [],
            'val_i2t_acc': [], 'val_t2i_acc': []
        }
        
        # Early stopping 
        self.patience = 5
        self.best_val_acc = 0.0
        self.counter = 0
        
        # Create checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)

    def train_step(self, images, labels):
        self.model.train()
        images = images.to(self.device)
        text_indices = torch.stack([self.int_to_str[l.item()] for l in labels]).to(self.device)
        
        # Forward pass
        image_embeddings, text_embeddings = self.model(images, text_indices)
        loss = self.compute_loss(image_embeddings, text_embeddings)
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            similarity = torch.matmul(image_embeddings, text_embeddings.t())
            i2t_acc, t2i_acc = self.compute_accuracy(similarity)
        
        return loss.item(), i2t_acc, t2i_acc
    
  

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_i2t_acc = 0
        total_t2i_acc = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Training')
        for images, labels in pbar:
            loss, i2t_acc, t2i_acc = self.train_step(images, labels)
            
            total_loss += loss
            total_i2t_acc += i2t_acc
            total_t2i_acc += t2i_acc
            num_batches += 1
            
            pbar.set_postfix({
                'loss': total_loss/num_batches,
                'i2t_acc': total_i2t_acc/num_batches,
                't2i_acc': total_t2i_acc/num_batches
            })
        
        return {
            'loss': total_loss / num_batches,
            'i2t_acc': total_i2t_acc / num_batches,
            't2i_acc': total_t2i_acc / num_batches
        }

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_i2t_acc = 0
        total_t2i_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Get embeddings
                images = images.to(self.device)
                text_indices = torch.stack([self.int_to_str[l.item()] 
                                         for l in labels]).to(self.device)
                
                image_embeddings, text_embeddings = self.model(images, text_indices)
                
                # Compute loss
                loss = self.compute_loss(image_embeddings, text_embeddings)
                
                # Compute accuracy
                similarity = torch.matmul(image_embeddings, text_embeddings.t())
                i2t_acc, t2i_acc = self.compute_accuracy(similarity)
                
                total_loss += loss.item()
                total_i2t_acc += i2t_acc
                total_t2i_acc += t2i_acc
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'i2t_acc': total_i2t_acc / num_batches,
            't2i_acc': total_t2i_acc / num_batches
        }

    def fit(self):
        # Data transforms for MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard normalization
        ])
        
        # Load datasets
        train_dataset = datasets.MNIST('../data/', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data/', train=False, download=True, transform=transform)
        
        # Use balanced sampling
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=BalancedBatchSampler(train_dataset, self.batch_size),
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_sampler=BalancedBatchSampler(test_dataset, self.batch_size),
            num_workers=4
        )
        
        for epoch in range(self.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(test_loader)
            
            # Update metrics history
            for k in ['loss', 'i2t_acc', 't2i_acc']:
                self.metrics[f'train_{k}'].append(train_metrics[k])
                self.metrics[f'val_{k}'].append(val_metrics[k])
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}")
            print(f"Train - I2T Acc: {train_metrics['i2t_acc']:.4f}")
            print(f"Train - T2I Acc: {train_metrics['t2i_acc']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}")
            print(f"Val - I2T Acc: {val_metrics['i2t_acc']:.4f}")
            print(f"Val - T2I Acc: {val_metrics['t2i_acc']:.4f}")
            
            # Early stopping check
            if val_metrics['i2t_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['i2t_acc']
                self.counter = 0
                self.save_checkpoint(epoch, val_metrics, 'best_model.pth')
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered!")
                    break
            
            # Save visualizations periodically
            if (epoch + 1) % 5 == 0:
                self.plot_metrics()
                self.create_presentation_visuals(test_loader, epoch)

    def compute_loss(self, image_embeddings, text_embeddings):
        similarity = torch.matmul(image_embeddings, text_embeddings.t())
        # Add temperature scaling with stability
        similarity = similarity / self.model.temperature.clamp(min=1e-4)
        
        labels = torch.arange(similarity.size(0)).to(self.device)
        
        # Add label smoothing
        loss_i2t = F.cross_entropy(similarity, labels, label_smoothing=0.1)
        loss_t2i = F.cross_entropy(similarity.t(), labels, label_smoothing=0.1)
        
        return (loss_i2t + loss_t2i) / 2

    def compute_accuracy(self, similarity_matrix):
        batch_size = similarity_matrix.size(0)
        i2t_matches = similarity_matrix.argmax(dim=1)
        t2i_matches = similarity_matrix.argmax(dim=0)
        
        diagonal = torch.arange(batch_size).to(self.device)
        i2t_accuracy = (i2t_matches == diagonal).float().mean().item()
        t2i_accuracy = (t2i_matches == diagonal).float().mean().item()
        
        return i2t_accuracy, t2i_accuracy

   
    def save_checkpoint(self, epoch, metrics, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = os.path.join('checkpoints', filename)
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint - {str(e)}")

    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['train_i2t_acc'], label='Train I2T')
        plt.plot(self.metrics['train_t2i_acc'], label='Train T2I')
        plt.plot(self.metrics['val_i2t_acc'], label='Val I2T')
        plt.plot(self.metrics['val_t2i_acc'], label='Val T2I')
        plt.title('Accuracy Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

    def create_presentation_visuals(self, loader, epoch):
        " Create visuals "
        self.model.eval()
        
        # Get a batch of images
        images, labels = next(iter(loader))
        images = images.to(self.device)
        text_indices = torch.stack([self.int_to_str[l.item()] 
                                  for l in labels]).to(self.device)
        
        with torch.no_grad():
            image_embeddings, text_embeddings = self.model(images, text_indices)
            similarity = torch.matmul(image_embeddings, text_embeddings.t())
        
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        similarity_np = similarity.cpu().numpy()
        plt.imshow(similarity_np, cmap='viridis')
        plt.colorbar()
        plt.title(f'Similarity Matrix - Epoch {epoch+1}')
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, f'{similarity_np[i,j]:.2f}',
                        ha='center', va='center')
        
        # Add labels
        plt.xticks(range(len(labels)), labels.tolist())
        plt.yticks(range(len(labels)), labels.tolist())
        plt.xlabel('Text Labels')
        plt.ylabel('Image Labels')
        
        plt.savefig(f'similarity_epoch_{epoch+1}.png')
        plt.close()
        
        # Print sample predictions
        print("\nSample Predictions:")
        for i in range(min(5, len(images))):
            pred_idx = similarity[i].argmax().item()
            print(f"\nImage {i+1}:")
            print(f"True Label: {labels[i].item()}")
            print(f"Predicted Label: {labels[pred_idx].item()}")
            print(f"Confidence: {similarity[i].max().item():.4f}")

        # Save some example images
        plt.figure(figsize=(15, 5))
        for i in range(min(5, len(images))):
            plt.subplot(1, 5, i+1)
            plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            plt.title(f'Label: {labels[i].item()}\nPred: {labels[similarity[i].argmax()].item()}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'samples_epoch_{epoch+1}.png')
        plt.close()

    
if __name__ == '__main__':
    
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainer = Trainer(
        learning_rate=1e-5,  
        batch_size=32,
        num_epochs=50,
        device=device
    )
    trainer.fit()