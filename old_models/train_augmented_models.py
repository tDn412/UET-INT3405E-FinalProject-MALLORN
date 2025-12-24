"""
TRAIN AUGMENTED DEEP LEARNING MODELS (Phase 3)

Strategy:
- Train LSTM & CNN using on-the-fly data augmentation
- Jittering, Scaling, Time-shifting
- effectively increases dataset size infinitely
- Train for longer (40 epochs)
- Generate Test Predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
from lstm_model_v2 import TDE_LSTM_v2
from cnn_model import TDE_CNN
from augmentations import AugmentedLightCurveDataset
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING AUGMENTED MODELS (PHASE 3)")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
print("\n[1] Loading data...")
X_train_seq = np.load('X_train_sequences.npy')
y_train = np.load('y_train.npy')
X_test_seq = np.load('X_test_sequences.npy')

X_train_seq = np.nan_to_num(X_train_seq, nan=0.0).clip(-10, 10)
X_test_seq = np.nan_to_num(X_test_seq, nan=0.0).clip(-10, 10)

# Do NOT flatten yet - augmentations need (timesteps, bands, features) structure
# X_train_seq shape: (N, 200, 6, 3)

# Datasets
# Train with augmentation
train_dset = AugmentedLightCurveDataset(X_train_seq, y_train, augment=True)
# Test without augmentation
test_dset = AugmentedLightCurveDataset(X_test_seq, augment=False)

train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=64, shuffle=False)

# Training Params
EPOCHS = 40
LR = 0.0005

def train_model(model, name):
    print(f"\nTraining Augmented {name}...")
    model = model.to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device)) # Adjusted weights
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for seq, lab in train_loader:
            seq, lab = seq.to(device), lab.to(device)
            out = model(seq)
            loss = crit(out, lab)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}: Loss {total_loss/len(train_loader):.4f}")
            
    return model

def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for seq in loader:
            seq = seq.to(device)
            out = model(seq)
            prob = torch.softmax(out, dim=1)[:, 1]
            preds.extend(prob.cpu().numpy())
    return np.array(preds)

# 1. Train LSTM
lstm = TDE_LSTM_v2(18, 96, 2, 0.25)
lstm = train_model(lstm, "LSTM")
lstm_preds = predict(lstm, test_loader)

# 2. Train CNN
cnn = TDE_CNN(18, 64, 0.3)
cnn = train_model(cnn, "CNN")
cnn_preds = predict(cnn, test_loader)

# Ensemble
avg_preds = (lstm_preds + cnn_preds) / 2

# Save
with open('test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

# Create submission
# Threshold 0.5 (models trained with class weights)
binary = (avg_preds >= 0.5).astype(int)
sub = pd.DataFrame({'object_id': test_ids, 'prediction': binary})
sub.to_csv('submission_augmented_dl.csv', index=False)

print(f"\n✓ Saved submission_augmented_dl.csv (TDEs: {binary.sum()})")
