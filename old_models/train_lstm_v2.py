"""
TRAIN LSTM v2 - WITH FIXES

Key improvements:
- Gradient clipping
- Batch normalization
- Lower learning rate
- Reduced class weights
- Input validation
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
from lstm_model_v2 import LightCurveDataset, TDE_LSTM_v2, train_epoch_v2, validate_v2
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING LSTM v2 (WITH STABILITY FIXES)")
print("="*70)

# Load and validate data
print("\n[1] Loading and validating data...")
X_train_seq = np.load('X_train_sequences.npy')
y_train = np.load('y_train.npy')
X_test_seq = np.load('X_test_sequences.npy')
masks_train = np.load('masks_train.npy')

# Check for NaN/Inf in data
print(f"Checking data quality...")
print(f"  Training NaN: {np.isnan(X_train_seq).sum()}")
print(f"  Training Inf: {np.isinf(X_train_seq).sum()}")
print(f"  Training range: [{X_train_seq.min():.2f}, {X_train_seq.max():.2f}]")

# Replace any NaN/Inf
X_train_seq = np.nan_to_num(X_train_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_test_seq = np.nan_to_num(X_test_seq, nan=0.0, posinf=0.0, neginf=0.0)

# Clip extreme values
X_train_seq = np.clip(X_train_seq, -10, 10)
X_test_seq = np.clip(X_test_seq, -10, 10)

print(f"  After cleaning: [{X_train_seq.min():.2f}, {X_train_seq.max():.2f}]")

# Reshape
n_samples, n_timesteps, n_bands, n_features = X_train_seq.shape
X_train_lstm = X_train_seq.reshape(n_samples, n_timesteps, n_bands * n_features)
X_test_lstm = X_test_seq.reshape(X_test_seq.shape[0], n_timesteps, n_bands * n_features)
masks_train_flat = masks_train.max(axis=2)

print(f"\nData shape:")
print(f"  Train: {X_train_lstm.shape}")
print(f"  Test: {X_test_lstm.shape}")

# IMPROVED Parameters
input_size = n_bands * n_features
hidden_size = 64  # Smaller
num_layers = 2
dropout = 0.2  # Less dropout
batch_size = 64  # Larger batch (more stable)
learning_rate = 0.0001  # 10x smaller!
num_epochs = 30  # Fewer epochs
patience = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
print(f"\nHyperparameters:")
print(f"  Hidden size: {hidden_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Batch size: {batch_size}")
print(f"  Gradient clip: 1.0")

# Train with ONE fold first (faster debugging)
print("\n[2] Training with 1-fold validation (for debugging)...")

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val, masks_tr, masks_val = train_test_split(
    X_train_lstm, y_train, masks_train_flat, 
    test_size=0.2, random_state=42, stratify=y_train
)

train_dataset = LightCurveDataset(X_tr, y_tr, masks_tr)
val_dataset = LightCurveDataset(X_val, y_val, masks_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = TDE_LSTM_v2(input_size, hidden_size, num_layers, dropout).to(device)

# REDUCED class weights (use sqrt to make less extreme)
class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
weight_pos = np.sqrt(class_ratio)  # ~4.4 instead of 19.6
class_weights = torch.FloatTensor([1.0, weight_pos]).to(device)
print(f"\nClass weights: [1.0, {weight_pos:.2f}] (reduced from {class_ratio:.1f})")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Training loop
print(f"\n{'='*70}")
print("TRAINING")
print(f"{'='*70}")

best_val_f1 = 0
patience_counter = 0
best_model_state = model.state_dict()

for epoch in range(num_epochs):
    train_loss, train_preds, train_labels = train_epoch_v2(
        model, train_loader, criterion, optimizer, device, clip_value=1.0
    )
    val_loss, val_preds, val_labels = validate_v2(
        model, val_loader, criterion, device
    )
    
    # Check for NaN
    if np.isnan(train_loss) or np.isnan(val_loss):
        print(f"Epoch {epoch+1}: NaN detected! train_loss={train_loss}, val_loss={val_loss}")
        print("Stopping training.")
        break
    
    # Find best threshold
    best_threshold = 0.5
    best_f1 = 0
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(val_labels, (val_preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_F1={best_f1:.4f} (t={best_threshold:.2f}), lr={current_lr:.6f}")
    
    # Early stopping
    if best_f1 > best_val_f1:
        best_val_f1 = best_f1
        patience_counter = 0
        best_model_state = model.state_dict()
        print(f"  → New best F1: {best_val_f1:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\n{'='*70}")
print(f"BEST VALIDATION F1: {best_val_f1:.4f}")
print(f"{'='*70}")

# If reasonable performance, train on full data
if best_val_f1 > 0.1:  # At least some signal
    print("\n[3] Model shows promise! Training on full data...")
    
    model.load_state_dict(best_model_state)
    
    full_dataset = LightCurveDataset(X_train_lstm, y_train, masks_train_flat)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    final_model = TDE_LSTM_v2(input_size, hidden_size, num_layers, dropout).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    for epoch in range(15):
        train_loss, _, _ = train_epoch_v2(final_model, full_loader, criterion, final_optimizer, device)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/15: loss={train_loss:.4f}")
        if np.isnan(train_loss):
            print("NaN detected in final training!")
            break
    
    # Predict
    print("\n[4] Generating predictions...")
    test_dataset = LightCurveDataset(X_test_lstm)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    final_model.eval()
    test_preds = []
    
    with torch.no_grad():
        for sequences in test_loader:
            sequences = sequences.to(device)
            outputs = final_model(sequences)
            probs = F.softmax(outputs, dim=1)[:, 1]
            test_preds.extend(probs.cpu().numpy())
    
    test_preds = np.array(test_preds)
    
    # Load test IDs
    with open('test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    
    # Create submission
    threshold = best_threshold if best_threshold > 0 else 0.5
    binary = (test_preds >= threshold).astype(int)
    
    sub = pd.DataFrame({
        'object_id': test_ids,
        'prediction': binary
    })
    sub.to_csv('submission_lstm_v2.csv', index=False)
    
    tde_count = binary.sum()
    print(f"\n✓ submission_lstm_v2.csv")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")
    print(f"  Val F1: {best_val_f1:.4f}")
    
    # Save model
    torch.save(final_model.state_dict(), 'lstm_model_v2.pth')
    print(f"✓ Model saved!")
    
else:
    print(f"\n❌ Model performance too low (F1={best_val_f1:.4f})")
    print("LSTM may not be suitable for this dataset.")
    print("Recommendation: Use XGBoost (0.4146 proven)")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
