"""
HYBRID APPROACH: Deep Learning Features + XGBoost

Strategy:
1. Extract embeddings from trained LSTM & CNN
2. Combine with original 58 features
3. Train XGBoost on combined features
4. Hypothesis: DL learns different patterns than manual features

Expected: Potential to beat 0.4146!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import pickle
from lstm_model_v2 import LightCurveDataset, TDE_LSTM_v2
from cnn_model import TDE_CNN
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HYBRID: DL FEATURES + XGBOOST")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load sequences
print("\n[1] Loading sequences...")
X_train_seq = np.load('X_train_sequences.npy')
y_train = np.load('y_train.npy')
X_test_seq = np.load('X_test_sequences.npy')

X_train_seq = np.nan_to_num(X_train_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_test_seq = np.nan_to_num(X_test_seq, nan=0.0, posinf=0.0, neginf=0.0)
X_train_seq = np.clip(X_train_seq, -10, 10)
X_test_seq = np.clip(X_test_seq, -10, 10)

n_samples, n_timesteps, n_bands, n_features = X_train_seq.shape
X_train_lstm = X_train_seq.reshape(n_samples, n_timesteps, n_bands * n_features)
X_test_lstm = X_test_seq.reshape(X_test_seq.shape[0], n_timesteps, n_bands * n_features)

print(f"Data: {X_train_lstm.shape}")

# Load original features
print("\n[2] Loading original features...")
train_meta = pd.read_csv('train_log.csv')
test_meta = pd.read_csv('test_log.csv')
train_lc = pd.read_csv('lightcurve_features_train.csv')
test_lc = pd.read_csv('lightcurve_features_test.csv')

# Original features (58)
X_orig_train = train_meta[['Z', 'EBV']].copy()
X_orig_train['Z_EBV_ratio'] = X_orig_train['Z'] / (X_orig_train['EBV'] + 1e-5)
X_orig_train['Z_squared'] = X_orig_train['Z'] ** 2
X_orig_train['EBV_squared'] = X_orig_train['EBV'] ** 2
X_orig_train['Z_EBV_interaction'] = X_orig_train['Z'] * X_orig_train['EBV']

X_orig_test = test_meta[['Z', 'EBV']].copy()
X_orig_test['Z_EBV_ratio'] = X_orig_test['Z'] / (X_orig_test['EBV'] + 1e-5)
X_orig_test['Z_squared'] = X_orig_test['Z'] ** 2
X_orig_test['EBV_squared'] = X_orig_test['EBV'] ** 2
X_orig_test['Z_EBV_interaction'] = X_orig_test['Z'] * X_orig_test['EBV']

# Add LC features
for col in train_lc.columns:
    if col != 'object_id':
        X_orig_train[f'lc_{col}'] = train_lc[col].values
        X_orig_test[f'lc_{col}'] = test_lc[col].values

X_orig_train = X_orig_train.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])
X_orig_test = X_orig_test.fillna(0).replace([np.inf, -np.inf], [1e10, -1e10])

print(f"Original features: {X_orig_train.shape[1]}")

# Extract LSTM embeddings
print("\n[3] Extracting LSTM embeddings...")

input_size = n_bands * n_features
lstm_model = TDE_LSTM_v2(input_size, hidden_size=96, num_layers=2, dropout=0.25).to(device)
lstm_model.load_state_dict(torch.load('lstm_v2_cv_best.pth'))
lstm_model.eval()

def extract_lstm_features(model, X, batch_size=64):
    """Extract hidden representations from LSTM"""
    dataset = LightCurveDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    
    with torch.no_grad():
        for sequences in loader:
            sequences = sequences.to(device)
            
            # Forward through LSTM
            x = sequences.transpose(1, 2)
            x = model.input_bn(x)
            x = x.transpose(1, 2)
            
            lstm_out, (hidden, cell) = model.lstm(x)
            
            # Use final hidden state as embedding
            # hidden shape: (num_layers*2, batch, hidden_size)
            # Concatenate forward and backward from last layer
            embedding = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_size*2)
            
            embeddings.append(embedding.cpu().numpy())
    
    return np.vstack(embeddings)

lstm_train_emb = extract_lstm_features(lstm_model, X_train_lstm)
lstm_test_emb = extract_lstm_features(lstm_model, X_test_lstm)

print(f"LSTM embeddings: {lstm_train_emb.shape}")

# Extract CNN embeddings
print("\n[4] Extracting CNN embeddings...")

cnn_model = TDE_CNN(input_size, num_filters=64, dropout=0.3).to(device)
cnn_model.load_state_dict(torch.load('cnn_model.pth'))
cnn_model.eval()

def extract_cnn_features(model, X, batch_size=64):
    """Extract features before final FC layer"""
    dataset = LightCurveDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    
    with torch.no_grad():
        for sequences in loader:
            sequences = sequences.to(device)
            
            # Forward through CNN (exclude final FC)
            x = sequences.transpose(1, 2)
            x = model.input_bn(x)
            
            x = model.conv1(x)
            x = model.bn1(x)
            x = F.relu(x)
            x = F.max_pool1d(x, 2)
            
            x = model.conv2(x)
            x = model.bn2(x)
            x = F.relu(x)
            x = F.max_pool1d(x, 2)
            
            x = model.conv3(x)
            x = model.bn3(x)
            x = F.relu(x)
            
            # Global pooling
            avg_pool = model.global_avg_pool(x).squeeze(-1)
            max_pool = model.global_max_pool(x).squeeze(-1)
            embedding = torch.cat([avg_pool, max_pool], dim=1)
            
            features.append(embedding.cpu().numpy())
    
    return np.vstack(features)

cnn_train_emb = extract_cnn_features(cnn_model, X_train_lstm)
cnn_test_emb = extract_cnn_features(cnn_model, X_test_lstm)

print(f"CNN embeddings: {cnn_train_emb.shape}")

# Combine all features
print("\n[5] Combining features...")

# Create feature names
orig_feature_names = list(X_orig_train.columns)
lstm_feature_names = [f'lstm_emb_{i}' for i in range(lstm_train_emb.shape[1])]
cnn_feature_names = [f'cnn_emb_{i}' for i in range(cnn_train_emb.shape[1])]

all_feature_names = orig_feature_names + lstm_feature_names + cnn_feature_names

X_hybrid_train = np.hstack([X_orig_train.values, lstm_train_emb, cnn_train_emb])
X_hybrid_test = np.hstack([X_orig_test.values, lstm_test_emb, cnn_test_emb])

print(f"Total hybrid features: {X_hybrid_train.shape[1]}")
print(f"  - Original: {X_orig_train.shape[1]}")
print(f"  - LSTM embeddings: {lstm_train_emb.shape[1]}")
print(f"  - CNN embeddings: {cnn_train_emb.shape[1]}")

# Train XGBoost on hybrid features
print("\n[6] Training XGBoost on hybrid features...")

# Cross-validation with F1 optimization
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_scores = []
all_val_preds = []
all_val_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_hybrid_train, y_train), 1):
    X_tr = X_hybrid_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_hybrid_train[val_idx]
    y_val = y_train[val_idx]
    
    scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.1,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    
    model.fit(X_tr, y_tr, verbose=False)
    val_proba = model.predict_proba(X_val)[:, 1]
    
    all_val_preds.extend(val_proba)
    all_val_labels.extend(y_val)
    
    # Find best threshold
    best_f1 = 0
    for t in np.arange(0.05, 0.5, 0.01):
        f1 = f1_score(y_val, (val_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    
    cv_f1_scores.append(best_f1)
    print(f"Fold {fold}: F1={best_f1:.4f}")

all_val_preds = np.array(all_val_preds)
all_val_labels = np.array(all_val_labels)

# Find best threshold overall
best_threshold = 0.1
best_f1 = 0
for t in np.arange(0.05, 0.5, 0.005):
    f1 = f1_score(all_val_labels, (all_val_preds >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n{'='*70}")
print(f"HYBRID MODEL RESULTS")
print(f"{'='*70}")
print(f"Mean CV F1: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best Overall F1: {best_f1:.4f}")
print(f"\nCOMPARISON:")
print(f"  XGBoost only:     0.4146")
print(f"  Hybrid DL+XGB:    {best_f1:.4f}")
print(f"  Improvement:      {(best_f1/0.4146 - 1)*100:+.2f}%")

# Train final model
if best_f1 >= 0.41:  # Only if competitive
    print("\n[7] Training final hybrid model...")
    
    final_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.1,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    
    final_model.fit(X_hybrid_train, y_train)
    test_proba = final_model.predict_proba(X_hybrid_test)[:, 1]
    
    # Load test IDs
    with open('test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    
    # Create submission
    binary = (test_proba >= best_threshold).astype(int)
    
    sub = pd.DataFrame({
        'object_id': test_ids,
        'prediction': binary
    })
    sub.to_csv('submission_hybrid.csv', index=False)
    
    tde_count = binary.sum()
    print(f"\n✓ submission_hybrid.csv")
    print(f"  CV F1: {best_f1:.4f}")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  TDEs: {tde_count} ({tde_count/len(binary)*100:.2f}%)")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': all_feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 features:")
    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    pickle.dump(final_model, open('hybrid_model.pkl', 'wb'))
    print(f"\n✓ Model saved!")

else:
    print(f"\n❌ Hybrid not better than baseline (0.4146)")
    print("Skipping final training.")

print("\n" + "="*70)
print("HYBRID EXPERIMENT COMPLETE!")
print("="*70)
