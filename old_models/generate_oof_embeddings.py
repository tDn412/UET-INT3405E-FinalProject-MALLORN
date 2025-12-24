"""
GENERATE OUT-OF-FOLD EMBEDDINGS (Fixing Hybrid Leakage)

Phase 2 of High Score Strategy:
- Train DL models (LSTM, CNN) on 5-fold CV
- Extract embeddings for VALIDATION sets (Out-of-Fold)
- Extract embeddings for TEST set (average of 5 folds)
- Save for use in Hybrid Stacking

Result: "Clean" embeddings without overfitting/leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle
from lstm_model_v2 import LightCurveDataset, TDE_LSTM_v2, train_epoch_v2, validate_v2
from cnn_model import TDE_CNN
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GENERATING OOF EMBEDDINGS (PHASE 2)")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load Data
print("\n[1] Loading data...")
X_train_seq = np.load('X_train_sequences.npy')
y_train = np.load('y_train.npy')
X_test_seq = np.load('X_test_sequences.npy')

# Clean
X_train_seq = np.nan_to_num(X_train_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_test_seq = np.nan_to_num(X_test_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_train_seq = np.clip(X_train_seq, -10, 10)
X_test_seq = np.clip(X_test_seq, -10, 10)

# Reshape
n_samples, n_timesteps, n_bands, n_features = X_train_seq.shape
X_train_flat = X_train_seq.reshape(n_samples, n_timesteps, n_bands * n_features)
X_test_flat = X_test_seq.reshape(X_test_seq.shape[0], n_timesteps, n_bands * n_features)

print(f"Data shape: {X_train_flat.shape}")

# Setup for OOF
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Placeholders
lstm_oof_emb = np.zeros((len(X_train_flat), 192)) # 96*2
cnn_oof_emb = np.zeros((len(X_train_flat), 256))  # 64*4

lstm_test_folds = []
cnn_test_folds = []

# Training params
BATCH_SIZE = 64
EPOCHS = 15  # 15 is enough for embeddings
LR = 0.001

def extract_lstm_features(model, X):
    dataset = LightCurveDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for sequences in loader:
            sequences = sequences.to(device)
            x = sequences.transpose(1, 2)
            x = model.input_bn(x)
            x = x.transpose(1, 2)
            _, (hidden, _) = model.lstm(x)
            emb = torch.cat([hidden[-2], hidden[-1]], dim=1)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

def extract_cnn_features(model, X):
    dataset = LightCurveDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for sequences in loader:
            sequences = sequences.to(device)
            # Forward pass excluding final FC
            x = sequences.transpose(1, 2)
            x = model.input_bn(x)
            x = F.relu(model.bn1(model.conv1(x)))
            x = F.max_pool1d(x, 2)
            x = F.relu(model.bn2(model.conv2(x)))
            x = F.max_pool1d(x, 2)
            x = F.relu(model.bn3(model.conv3(x)))
            avg_pool = model.global_avg_pool(x).squeeze(-1)
            max_pool = model.global_max_pool(x).squeeze(-1)
            emb = torch.cat([avg_pool, max_pool], dim=1)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

# CV Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_flat, y_train), 1):
    print(f"\nProcessing Fold {fold}/{n_folds}...")
    
    # Data
    X_tr, y_tr = X_train_flat[train_idx], y_train[train_idx]
    X_val, y_val = X_train_flat[val_idx], y_train[val_idx]
    
    train_dset = LightCurveDataset(X_tr, y_tr)
    val_dset = LightCurveDataset(X_val, y_val)
    train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Train LSTM ---
    print("  Training LSTM...")
    lstm = TDE_LSTM_v2(18, 96, 2, 0.25).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(lstm.parameters(), lr=0.0001)
    
    for _ in range(EPOCHS):
        lstm.train()
        for seq, lab in train_loader:
            seq, lab = seq.to(device), lab.to(device)
            out = lstm(seq)
            loss = crit(out, lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            opt.step()
            
    # Extract
    lstm.eval()
    lstm_oof_emb[val_idx] = extract_lstm_features(lstm, X_val)
    lstm_test_folds.append(extract_lstm_features(lstm, X_test_flat))
    
    # --- Train CNN ---
    print("  Training CNN...")
    cnn = TDE_CNN(18, 64, 0.3).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
    
    for _ in range(EPOCHS):
        cnn.train() # Custom loop
        for seq, lab in train_loader:
            seq, lab = seq.to(device), lab.to(device)
            out = cnn(seq)
            loss = crit(out, lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)
            opt.step()
            
    # Extract
    cnn.eval()
    cnn_oof_emb[val_idx] = extract_cnn_features(cnn, X_val)
    cnn_test_folds.append(extract_cnn_features(cnn, X_test_flat))

print("\nAggregating results...")
lstm_test_avg = np.mean(lstm_test_folds, axis=0)
cnn_test_avg = np.mean(cnn_test_folds, axis=0)

# Save
np.save('lstm_embeddings_oof.npy', lstm_oof_emb)
np.save('cnn_embeddings_oof.npy', cnn_oof_emb)
np.save('lstm_embeddings_test.npy', lstm_test_avg)
np.save('cnn_embeddings_test.npy', cnn_test_avg)

print("✓ Saved OOF and Test embeddings.")
print("Done!")
