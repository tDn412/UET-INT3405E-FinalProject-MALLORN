"""
TRAIN LSTM MODEL ON RAW LIGHT CURVES

This will take longer than XGBoost but should capture temporal patterns better!
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
from lstm_model import LightCurveDataset, TDE_LSTM, train_epoch, validate
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING LSTM ON RAW LIGHT CURVES")
print("="*70)

# Load preprocessed sequences
print("\n[1] Loading preprocessed sequences...")
X_train_seq = np.load('X_train_sequences.npy')
y_train = np.load('y_train.npy')
X_test_seq = np.load('X_test_sequences.npy')
masks_train = np.load('masks_train.npy')

print(f"Training sequences: {X_train_seq.shape}")  # (samples, timesteps, bands, features)
print(f"Test sequences: {X_test_seq.shape}")

# Reshape for LSTM: (samples, timesteps, features)
# Flatten bands and features into single feature dimension
n_samples, n_timesteps, n_bands, n_features = X_train_seq.shape
X_train_lstm = X_train_seq.reshape(n_samples, n_timesteps, n_bands * n_features)
X_test_lstm = X_test_seq.reshape(X_test_seq.shape[0], n_timesteps, n_bands * n_features)

# Flatten masks too
masks_train_flat = masks_train.max(axis=2)  # Any band valid = timestep valid

print(f"\nReshaped for LSTM:")
print(f"  Train: {X_train_lstm.shape}")
print(f"  Test: {X_test_lstm.shape}")
print(f"  Input features: {n_bands * n_features}")

# Model parameters
input_size = n_bands * n_features  # 6 bands * 3 features = 18
hidden_size = 128
num_layers = 2
dropout = 0.3
batch_size = 32
learning_rate = 0.001
num_epochs = 50
patience = 10  # Early stopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 5-Fold Cross-Validation
print("\n[2] Training with 5-fold CV...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_scores = []
all_val_preds = []
all_val_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_lstm, y_train), 1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold}/5")
    print(f"{'='*70}")
    
    # Split data
    X_tr = X_train_lstm[train_idx]
    y_tr = y_train[train_idx]
    masks_tr = masks_train_flat[train_idx]
    
    X_val = X_train_lstm[val_idx]
    y_val = y_train[val_idx]
    masks_val = masks_train_flat[val_idx]
    
    # Create datasets and loaders
    train_dataset = LightCurveDataset(X_tr, y_tr, masks_tr)
    val_dataset = LightCurveDataset(X_val, y_val, masks_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TDE_LSTM(input_size, hidden_size, num_layers, dropout).to(device)
    
    # Loss with class weights for imbalance
    class_weights = torch.FloatTensor([1.0, 19.6]).to(device)  # Weight for minority class
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = model.state_dict()  # Initialize with current state
    
    for epoch in range(num_epochs):
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Find best threshold on validation
        best_threshold = 0.5
        best_f1 = 0
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(val_labels, (val_preds >= t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_F1={best_f1:.4f} (t={best_threshold:.2f})")
        
        # Early stopping
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            patience_counter = 0
            # Save best model for this fold
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and get final predictions
    model.load_state_dict(best_model_state)
    _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    # Store for overall CV score
    all_val_preds.extend(val_preds)
    all_val_labels.extend(val_labels)
    
    cv_f1_scores.append(best_val_f1)
    print(f"\nFold {fold} Best F1: {best_val_f1:.4f}")

# Overall CV results
all_val_preds = np.array(all_val_preds)
all_val_labels = np.array(all_val_labels)

# Find best threshold overall
best_overall_threshold = 0.5
best_overall_f1 = 0
for t in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(all_val_labels, (all_val_preds >= t).astype(int))
    if f1 > best_overall_f1:
        best_overall_f1 = f1
        best_overall_threshold = t

print(f"\n{'='*70}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*70}")
print(f"Mean CV F1: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
print(f"Best Overall Threshold: {best_overall_threshold:.3f}")
print(f"Best Overall F1: {best_overall_f1:.4f}")

# Train final model on all data
print(f"\n[3] Training final model on all data...")

full_dataset = LightCurveDataset(X_train_lstm, y_train, masks_train_flat)
full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

final_model = TDE_LSTM(input_size, hidden_size, num_layers, dropout).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)

for epoch in range(30):  # Fewer epochs for final model
    train_loss, train_preds, train_labels = train_epoch(final_model, full_loader, criterion, optimizer, device)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30: loss={train_loss:.4f}")

# Predict on test set
print("\n[4] Generating test predictions...")

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

# Create submissions
print("\n[5] Creating submissions...")

thresholds = [
    (best_overall_threshold, 'submission_lstm_optimal.csv'),
    (best_overall_threshold - 0.1, 'submission_lstm_lower.csv'),
    (best_overall_threshold + 0.1, 'submission_lstm_higher.csv'),
]

for threshold, filename in thresholds:
    binary = (test_preds >= threshold).astype(int)
    sub = pd.DataFrame({
        'object_id': test_ids,
        'prediction': binary
    })
    sub.to_csv(filename, index=False)
    
    tde_count = binary.sum()
    print(f"✓ {filename}")
    print(f"  Threshold: {threshold:.3f}, TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")

# Save model
torch.save(final_model.state_dict(), 'lstm_model.pth')
print(f"\n✓ Model saved to lstm_model.pth")

print("\n" + "="*70)
print("LSTM TRAINING COMPLETE!")
print("="*70)

print(f"""
RESULTS:
- CV F1: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}
- Best Threshold: {best_overall_threshold:.3f}
- Best Overall F1: {best_overall_f1:.4f}

COMPARISON:
- XGBoost (58 features): CV F1 = 0.4147, Test = 0.4146
- LSTM (raw sequences): CV F1 = {np.mean(cv_f1_scores):.4f}

Main File: submission_lstm_optimal.csv

CONCEPT LEARNED:
LSTM processes raw time series directly without manual feature engineering!
Attention mechanism helps focus on important timesteps (like TDE peak).
""")
