"""
DATA AUGMENTATION FOR LIGHT CURVES

Techniques to increase effective dataset size for Deep Learning:
1. Jittering: Add Gaussian noise to flux
2. Scaling: Multiply flux by random factor
3. Time Warping: Stretch/compress time axis
4. Dropout: Zero out random points
"""

import numpy as np
import torch

class LightCurveAugmenter:
    def __init__(self):
        pass
        
    def jitter(self, sequence, sigma=0.05):
        """Add Gaussian noise to flux"""
        # sequence shape: (timesteps, bands, features)
        # feature 0 is flux
        noise = np.random.normal(0, sigma, sequence.shape)
        # Apply only to flux channel (idx 0)
        augmented = sequence.copy()
        augmented[:, :, 0] += noise[:, :, 0]
        return augmented
        
    def scale(self, sequence, sigma=0.1):
        """Multiply flux by random factor"""
        factor = np.random.normal(1.0, sigma)
        augmented = sequence.copy()
        augmented[:, :, 0] *= factor
        return augmented
        
    def time_warp(self, sequence, sigma=0.2):
        """
        Stretch/compress time
        For fixed-length sequences, this is tricky. 
        Instead of real warping, we can shift the sequence or crop-and-pad.
        Simple approach: Random crop and pad
        """
        # (timesteps, ...)
        timesteps = sequence.shape[0]
        full_len = timesteps
        
        # Simple shift
        shift = np.random.randint(-10, 10)
        augmented = np.zeros_like(sequence)
        
        if shift > 0:
            augmented[shift:] = sequence[:-shift]
        elif shift < 0:
            augmented[:shift] = sequence[-shift:]
        else:
            augmented = sequence.copy()
            
        return augmented
        
    def dropout(self, sequence, p=0.1):
        """Randomly zero out time steps"""
        mask = np.random.rand(sequence.shape[0]) > p
        augmented = sequence.copy()
        augmented[~mask] = 0
        return augmented
    
    def augment(self, sequence):
        """Apply random augmentations"""
        # Always jitter
        aug = self.jitter(sequence)
        
        # Randomly scale
        if np.random.rand() > 0.5:
            aug = self.scale(aug)
            
        # Randomly shift
        if np.random.rand() > 0.5:
            aug = self.time_warp(aug)
            
        # Randomly dropout
        if np.random.rand() > 0.5:
            aug = self.dropout(aug)
            
        return aug

# PyTorch Dataset Wrapper
from torch.utils.data import Dataset

class AugmentedLightCurveDataset(Dataset):
    def __init__(self, sequences, labels=None, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.augmenter = LightCurveAugmenter()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        
        if self.augment:
            seq = self.augmenter.augment(seq)
            
        # Convert to tensor
        seq_tensor = torch.FloatTensor(seq).permute(1, 0, 2) # (timesteps, bands, feats) -> (bands, timesteps, feats)
        # Flatten bands*feats for LSTM input -> (timesteps, bands*feats)
        # But wait, our LSTM expects (batch, timesteps, input_size)
        # Let's check lstm_model_v2.LightCurveDataset format
        # It does: `sequence.reshape(n_timesteps, -1)` = (200, 18)
        
        # Original shape: (200, 6, 3)
        # Let's reshape correctly
        seq_flat = seq.reshape(seq.shape[0], -1) # (200, 18)
        seq_tensor = torch.FloatTensor(seq_flat) # (200, 18)
        
        if self.labels is not None:
            return seq_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return seq_tensor

