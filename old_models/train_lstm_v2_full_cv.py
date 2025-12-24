"""
TRAIN LSTM v2 - FULL 5-FOLD CV

Train properly with full cross-validation for better performance
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
print("TRAINING LSTM v2 - FULL 5-FOLD CV")
print("="*70)

# Load and validate data
print("\n[1] Loading data...")
X_train_seq = np.load('X_train_sequences.npy')
y_train = np.load('y_train.npy')
X_test_seq = np.load('X_test_sequences.npy')
masks_train = np.load('masks_train.npy')

# Clean data
X_train_seq = np.nan_to_num(X_train_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_test_seq = np.nan_to_num(X_test_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_train_seq = np.clip(X_train_seq, -10, 10)
X_test_seq = np.clip(X_test_seq, -10, 10)

# Reshape
n_samples, n_timesteps, n_bands, n_features = X_train_seq.shape
X_train_lstm = X_train_seq.reshape(n_samples, n_timesteps, n_bands * n_features)
X_test_lstm = X_test_seq.reshape(X_test_seq.shape[0], n_timesteps, n_bands * n_features)
masks_train_flat = masks_train.max(axis=2)

print(f"Data: {X_train_lstm.shape}, Test: {X_test_lstm.shape}")

# Parameters (IMPROVED)
input_size = n_bands * n_features
hidden_size = 96  # Slightly larger
num_layers = 2
dropout = 0.25
batch_size = 64
learning_rate = 0.0001
num_epochs = 40  # More epochs
patience = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 5-Fold CV
print("\n[2] Training with 5-fold CV...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_scores = []
all_val_preds = []
all_val_labels = []
fold_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_lstm, y_train), 1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold}/5")
    print(f"{'='*70}")
    
    # Split
    X_tr = X_train_lstm[train_idx]
    y_tr = y_train[train_idx]
    masks_tr = masks_train_flat[train_idx]
    
    X_val = X_train_lstm[val_idx]
    y_val = y_train[val_idx]
    masks_val = masks_train_flat[val_idx]
    
    # Datasets
    train_dataset = LightCurveDataset(X_tr, y_tr, masks_tr)
    val_dataset = LightCurveDataset(X_val, y_val, masks_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = TDE_LSTM_v2(input_size, hidden_size, num_layers, dropout).to(device)
    
    # Loss
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    weight_pos = np.sqrt(class_ratio)
    class_weights = torch.FloatTensor([1.0, weight_pos]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)
    
    # Train
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = model.state_dict()
    
    for epoch in range(num_epochs):
        train_loss, train_preds, train_labels = train_epoch_v2(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_preds, val_labels = validate_v2(
            model, val_loader, criterion, device
        )
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"NaN at epoch {epoch+1}! Stopping fold.")
            break
        
        # Find best threshold
        best_f1 = 0
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(val_labels, (val_preds >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}/{val_loss:.4f}, F1={best_f1:.4f}, lr={lr:.6f}")
        
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break
    
    # Load best and predict
    model.load_state_dict(best_model_state)
    _, val_preds, val_labels = validate_v2(model, val_loader, criterion, device)
    
    all_val_preds.extend(val_preds)
    all_val_labels.extend(val_labels)
    cv_f1_scores.append(best_val_f1)
    fold_models.append(best_model_state)
    
    print(f"Fold {fold} Best F1: {best_val_f1:.4f}")

# Overall results
all_val_preds = np.array(all_val_preds)
all_val_labels = np.array(all_val_labels)

best_threshold = 0.5
best_f1 = 0
for t in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(all_val_labels, (all_val_preds >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n{'='*70}")
print("5-FOLD CV RESULTS")
print(f"{'='*70}")
print(f"Mean F1: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best Overall F1: {best_f1:.4f}")

# Ensemble predictions from all folds
if np.mean(cv_f1_scores) > 0.15:
    print("\n[3] Creating ensemble predictions...")
    
    test_dataset = LightCurveDataset(X_test_lstm)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_test_preds = []
    
    for fold_idx, fold_state in enumerate(fold_models, 1):
        model = TDE_LSTM_v2(input_size, hidden_size, num_layers, dropout).to(device)
        model.load_state_dict(fold_state)
        model.eval()
        
        test_preds = []
        with torch.no_grad():
            for sequences in test_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                probs = F.softmax(outputs, dim=1)[:, 1]
                test_preds.extend(probs.cpu().numpy())
        
        all_test_preds.append(test_preds)
        print(f"Fold {fold_idx} predictions generated")
    
    # Ensemble: Average predictions
    ensemble_preds = np.mean(all_test_preds, axis=0)
    
    # Load test IDs
    with open('test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    
    # Create submission
    binary = (ensemble_preds >= best_threshold).astype(int)
    
    sub = pd.DataFrame({
        'object_id': test_ids,
        'prediction': binary
    })
    sub.to_csv('submission_lstm_v2_ensemble.csv', index=False)
    
    tde_count = binary.sum()
    print(f"\n✓ submission_lstm_v2_ensemble.csv")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")
    print(f"  CV F1: {np.mean(cv_f1_scores):.4f}")
    
    # Save best fold model
    best_fold_idx = np.argmax(cv_f1_scores)
    final_model = TDE_LSTM_v2(input_size, hidden_size, num_layers, dropout)
    final_model.load_state_dict(fold_models[best_fold_idx])
    torch.save(final_model.state_dict(), 'lstm_v2_cv_best.pth')
    print(f"✓ Saved best fold model (fold {best_fold_idx+1})")

else:
    print(f"\n❌ Performance too low: {np.mean(cv_f1_scores):.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
