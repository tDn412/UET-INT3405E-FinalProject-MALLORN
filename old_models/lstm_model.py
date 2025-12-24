"""
DEEP LEARNING: LSTM Model for TDE Classification

CONCEPT: Long Short-Term Memory (LSTM)
- Processes sequential data (light curves)
- Remembers long-term dependencies
- Better than statistical features for time series

Architecture:
- Input: Raw light curve sequences (timesteps, bands, features)
- LSTM layers: Learn temporal patterns
- Output: Binary classification (TDE or not)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LSTM MODEL FOR TDE CLASSIFICATION")
print("="*70)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Custom Dataset
class LightCurveDataset(Dataset):
    """
    Dataset for light curves
    
    CONCEPT: PyTorch Dataset
    - Wraps data for efficient batching
    - Returns (sequence, label) pairs
    """
    def __init__(self, sequences, labels=None, masks=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.masks = torch.FloatTensor(masks) if masks is not None else None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            if self.masks is not None:
                return self.sequences[idx], self.labels[idx], self.masks[idx]
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]

# LSTM Model
class TDE_LSTM(nn.Module):
    """
    LSTM Model for TDE Classification
    
    Architecture:
    - LSTM layers: Process time series
    - Dropout: Prevent overfitting
    - Fully connected: Classification
    
    CONCEPT: Bidirectional LSTM
    - Processes sequence forward AND backward
    - Captures context from both directions
    - Better for astronomical events
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(TDE_LSTM, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Process both directions!
        )
        
        # Attention mechanism (optional but helpful)
        self.attention = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification
        
    def forward(self, x, mask=None):
        # x shape: (batch, timesteps, features)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch, timesteps, hidden_size*2)
        
        # Attention mechanism
        if mask is not None:
            # Apply mask to attention
            attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
            # Mask padded positions
            attn_weights = attn_weights * mask
            # Renormalize
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
        else:
            attn_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        
        # Apply attention
        # attn_weights shape: (batch, timesteps)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        # context shape: (batch, hidden_size*2)
        
        # Fully connected layers
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        if len(batch) == 3:
            sequences, labels, masks = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
        else:
            sequences, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            masks = None
        
        # Forward
        outputs = model(sequences, masks)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Predictions
        probs = F.softmax(outputs, dim=1)[:, 1]  # Probability of class 1 (TDE)
        all_preds.extend(probs.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)

# Validation function
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                sequences, labels, masks = batch
                sequences = sequences.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
            else:
                sequences, labels = batch
                sequences = sequences.to(device)
                labels = labels.to(device)
                masks = None
            
            outputs = model(sequences, masks)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)

# Main training script will go here...
print("\nLSTM Model defined successfully!")
print("Waiting for PyTorch installation to complete...")
