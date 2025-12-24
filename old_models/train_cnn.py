"""
TRAIN 1D CNN MODEL

Faster alternative to LSTM
Often works better for time series classification
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
from lstm_model_v2 import LightCurveDataset, train_epoch_v2, validate_v2
from cnn_model import TDE_CNN
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING 1D CNN MODEL")
print("="*70)

# Load data
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
X_train_cnn = X_train_seq.reshape(n_samples, n_timesteps, n_bands * n_features)
X_test_cnn = X_test_seq.reshape(X_test_seq.shape[0], n_timesteps, n_bands * n_features)

print(f"Data: {X_train_cnn.shape}")

# Parameters
input_size = n_bands * n_features
num_filters = 64
dropout = 0.3
batch_size = 64
learning_rate = 0.001  # Slightly higher for CNN
num_epochs = 30
patience = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Single fold validation (faster)
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_cnn, y_train, test_size=0.2, random_state=42, stratify=y_train
)

train_dataset = LightCurveDataset(X_tr, y_tr)
val_dataset = LightCurveDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model
model = TDE_CNN(input_size, num_filters, dropout).to(device)

# Loss
class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
weight_pos = np.sqrt(class_ratio)
class_weights = torch.FloatTensor([1.0, weight_pos]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Train
print(f"\n{'='*70}")
print("TRAINING CNN")
print(f"{'='*70}")

best_val_f1 = 0
patience_counter = 0
best_model_state = model.state_dict()

for epoch in range(num_epochs):
    # CNN doesn't use masks - use simple training loop
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        sequences, labels = batch
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        probs = F.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss_total = 0
    val_preds = []
    val_labels_list = []
    
    with torch.no_grad():
        for batch in val_loader:
            sequences, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item()
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            val_preds.extend(probs.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())
    
    val_loss = val_loss_total / len(val_loader)
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels_list)
    train_loss, train_preds, train_labels = train_loss, np.array(all_preds), np.array(all_labels)
    
    if np.isnan(train_loss) or np.isnan(val_loss):
        print(f"NaN at epoch {epoch+1}!")
        break
    
    # Find best threshold
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(val_labels, (val_preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{num_epochs}: loss={train_loss:.4f}/{val_loss:.4f}, F1={best_f1:.4f} (t={best_threshold:.2f}), lr={lr:.6f}")
    
    if best_f1 > best_val_f1:
        best_val_f1 = best_f1
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"  → New best!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break

print(f"\n{'='*70}")
print(f"BEST VAL F1: {best_val_f1:.4f}")
print(f"{'='*70}")

# Predict if reasonable
if best_val_f1 > 0.1:
    print("\n[3] Generating predictions...")
    
    model.load_state_dict(best_model_state)
    
    # Train on full data
    full_dataset = LightCurveDataset(X_train_cnn, y_train)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    final_model = TDE_CNN(input_size, num_filters, dropout).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate*0.5, weight_decay=1e-5)
    
    for epoch in range(15):
        final_model.train()
        total_loss = 0
        for batch in full_loader:
            sequences, labels = batch
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = final_model(sequences)
            loss = criterion(outputs, labels)
            
            final_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            final_optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(full_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/15: loss={train_loss:.4f}")
    
    # Test predictions
    test_dataset = LightCurveDataset(X_test_cnn)
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
    binary = (test_preds >= best_threshold).astype(int)
    
    sub = pd.DataFrame({
        'object_id': test_ids,
        'prediction': binary
    })
    sub.to_csv('submission_cnn.csv', index=False)
    
    tde_count = binary.sum()
    print(f"\n✓ submission_cnn.csv")
    print(f"  Val F1: {best_val_f1:.4f}")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")
    
    # Save model
    torch.save(final_model.state_dict(), 'cnn_model.pth')
    print("✓ Model saved!")

else:
    print(f"\n❌ Performance too low: {best_val_f1:.4f}")

print("\n" + "="*70)
print("CNN TRAINING COMPLETE!")
print("="*70)

print(f"""
COMPARISON:
- XGBoost:  0.4146 (test) ✅ BEST
- LSTM v2:  0.1538 (val, 1-fold)
- CNN:      {best_val_f1:.4f} (val)
""")
