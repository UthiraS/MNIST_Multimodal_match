import random
import torch
from transformers import AutoTokenizer
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
import random
import wandb
from torch.utils.data import DataLoader



def create_digit_mappings(vocab_size=100):
    " Create mappings for digits to token sequences "
    # Start with special tokens
    tokens = {
        '<pad>': 0,
        '<start>': 1,
        '<end>': 2,
        'digit': 3,
        'number': 4
    }
    
    # Add number words and digits
    number_words = ['zero', 'one', 'two', 'three', 'four', 
                   'five', 'six', 'seven', 'eight', 'nine']
    for i, word in enumerate(number_words):
        tokens[word] = i + 10
        tokens[str(i)] = i + 20  # numerical representations
    
    # Create digit to token sequence mapping
    int_to_str = {}
    for i in range(10):
        # Use word representation for simplicity
        seq = torch.tensor([tokens['<start>'], tokens['digit'], 
                           tokens[number_words[i]], tokens['<end>']])
        int_to_str[i] = seq
    
    return int_to_str, len(tokens)


class TextVocabulary:
    def __init__(self):
        # Create vocabulary from digit words and templates
        self.digit_words = ['zero', 'one', 'two', 'three', 'four',
                          'five', 'six', 'seven', 'eight', 'nine']
        self.templates = [
            "digit {}",
            "{} in handwriting",
            "handwritten {}",
            "number {}",
            "{} drawn",
            "this is {}",
            "showing {}",
            "written {}"
        ]
        
        # Build vocabulary
        self.word_to_idx = {}
        self.idx_to_word = {}
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        idx = 0
        for token in special_tokens:
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            idx += 1
            
        # Add digit words and numbers
        for word in self.digit_words + [str(i) for i in range(10)]:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
                
        # Add template words
        for template in self.templates:
            for word in template.split():
                if word != '{}' and word not in self.word_to_idx:
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    idx += 1
    
    def get_vocab_size(self):
        return len(self.word_to_idx)
    
    def encode_text(self, text):
        words = text.lower().split()
        return torch.tensor([self.word_to_idx.get(word, self.word_to_idx['<unk>']) 
                           for word in words])
    
    def decode_text(self, indices):
        return ' '.join([self.idx_to_word[idx.item()] for idx in indices])



class BalancedBatchSampler:
    " Custom batch sampler for balanced class distribution "
    def __init__(self, dataset, batch_size=8):
        self.labels = dataset.targets
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        
        # Create index mapping for each class
        self.label_indices = {}
        for label in range(10):  # MNIST has 10 classes
            self.label_indices[label] = (self.labels == label).nonzero().squeeze(1)
            
    def __iter__(self):
        for _ in range(self.num_batches):
            # Randomly select classes for this batch
            selected_labels = torch.randperm(10)[:self.batch_size]
            
            # Select one random example from each selected class
            batch_indices = []
            for label in selected_labels:
                label_indices = self.label_indices[label.item()]
                selected_idx = label_indices[torch.randint(len(label_indices), (1,))].item()
                batch_indices.append(selected_idx)
                
            yield torch.tensor(batch_indices)
    
    def __len__(self):
        return self.num_batches


