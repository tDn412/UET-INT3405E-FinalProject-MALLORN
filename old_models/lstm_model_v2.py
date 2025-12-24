"""
LSTM MODEL v2 - WITH FIXES FOR NaN ISSUE

Fixes implemented:
1. Gradient clipping (prevent explosion)
2. Batch normalization (stabilize training)
3. Lower learning rate (0.0001 instead of 0.001)
4. Reduced class weights (sqrt instead of raw ratio)
5. Better initialization
6. Input normalization check
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Dataset class (same as before)
class LightCurveDataset(Dataset):
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

# IMPROVED LSTM Model
class TDE_LSTM_v2(nn.Module):
    """
    Improved LSTM with fixes for numerical stability
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(TDE_LSTM_v2, self).__init__()
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Smaller, more stable LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # Reduced from 128 to 64
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Batch norm after LSTM
        self.lstm_bn = nn.BatchNorm1d(hidden_size * 2)
        
        # Simpler attention
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Simpler FC layers
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(32, 2)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stability"""
        for name, param in self.named_parameters():
            # Skip batch norm parameters
            if 'bn' in name or 'BatchNorm' in name:
                continue
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:  # Only for 2D+ tensors
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x, mask=None):
        # x shape: (batch, timesteps, features)
        batch_size, seq_len, features = x.shape
        
        # Input normalization per feature
        x_reshaped = x.transpose(1, 2)  # (batch, features, timesteps)
        x_normalized = self.input_bn(x_reshaped)
        x = x_normalized.transpose(1, 2)  # Back to (batch, timesteps, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, timesteps, hidden*2)
        
        # Apply batch norm
        lstm_out_transposed = lstm_out.transpose(1, 2)  # (batch, hidden*2, timesteps)
        lstm_out_bn = self.lstm_bn(lstm_out_transposed)
        lstm_out = lstm_out_bn.transpose(1, 2)  # (batch, timesteps, hidden*2)
        
        # Simple attention (mean pooling if issues)
        try:
            attn_weights = torch.sigmoid(self.attention(lstm_out).squeeze(-1))
            
            if mask is not None:
                attn_weights = attn_weights * mask
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
            else:
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        except:
            # Fallback to mean pooling if attention fails
            if mask is not None:
                masked_lstm = lstm_out * mask.unsqueeze(-1)
                context = masked_lstm.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                context = lstm_out.mean(dim=1)
        
        # FC layers
        x = F.relu(self.fc1(context))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

# Training functions with gradient clipping
def train_epoch_v2(model, dataloader, criterion, optimizer, device, clip_value=1.0):
    """Train with gradient clipping"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        if len(batch) == 3:
            sequences, labels, masks = batch
            masks = masks.to(device)
        else:
            sequences, labels = batch
            masks = None
        
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Check for NaN in input
        if torch.isnan(sequences).any():
            print("WARNING: NaN in input sequences!")
            continue
        
        # Forward
        outputs = model(sequences, masks)
        loss = criterion(outputs, labels)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("WARNING: NaN loss detected! Skipping batch.")
            continue
        
        # Backward with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Predictions
        probs = F.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    return avg_loss, np.array(all_preds), np.array(all_labels)

def validate_v2(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                sequences, labels, masks = batch
                masks = masks.to(device)
            else:
                sequences, labels = batch
                masks = None
            
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            if torch.isnan(sequences).any():
                continue
            
            outputs = model(sequences, masks)
            loss = criterion(outputs, labels)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    return avg_loss, np.array(all_preds), np.array(all_labels)

print("LSTM v2 model with stability fixes loaded successfully!")
