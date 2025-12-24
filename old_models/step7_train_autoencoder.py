"""
STEP 7: UNSUPERVISED LSTM AUTOENCODER (PYTORCH) - FIX V2
Goal: Learn robust "Shape Embeddings" from ALL data (~10k objects).
Reason: 148 TDEs is too few. 10k objects = Physics.

Architecture:
- Input: Multi-band Light Curve (Binned to 100 steps)
- Encoder -> Bottleneck (Embedding) -> Decoder
- Loss: MSE
- Fixes: Log normalization, Gradient Clipping, NaN handling.
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SELF-SUPERVISED LEARNING: LSTM AUTOENCODER (PYTORCH V2)")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Data
with open('raw_lightcurves_train_corrected.pkl', 'rb') as f:
    train_lcs = pickle.load(f)
with open('raw_lightcurves_test_corrected.pkl', 'rb') as f:
    test_lcs = pickle.load(f)

all_lcs = {**train_lcs, **test_lcs}
ids = list(all_lcs.keys())

# 2. Preprocessing
MAX_LEN = 100
N_BANDS = 6

def process_lc_stack(lc_df):
    if len(lc_df) == 0: return None
    
    df = lc_df.sort_values('Time (MJD)')
    
    # Simple Log1p Transform of Flux first?
    # No, bin first then log.
    
    t_min = df['Time (MJD)'].min()
    t_max = df['Time (MJD)'].max()
    duration = t_max - t_min
    if duration == 0: duration = 1.0
    
    bins = np.linspace(t_min, t_max + 1e-5, MAX_LEN + 1)
    df['bin_idx'] = np.digitize(df['Time (MJD)'], bins) - 1
    
    seq = np.zeros((MAX_LEN, N_BANDS), dtype=np.float32)
    mask = np.zeros((MAX_LEN, N_BANDS), dtype=np.float32)
    
    filter_map = {'u':0, 'g':1, 'r':2, 'i':3, 'z':4, 'y':5}
    
    for b_code, b_name in enumerate(['u','g','r','i','z','y']):
        b_data = df[df['Filter'] == b_name]
        if len(b_data) == 0: continue
        
        grp = b_data.groupby('bin_idx')['Flux'].mean()
        for idx, flx in grp.items():
            if 0 <= idx < MAX_LEN:
                seq[idx, b_code] = flx
                mask[idx, b_code] = 1.0
                
    # Log Normalization: sign(x) * log1p(abs(x))
    # This handles negative flux (noise) gracefully
    seq = np.sign(seq) * np.log1p(np.abs(seq))
    
    # Replace NaNs (if any)
    seq = np.nan_to_num(seq)
    
    return seq

print("\n[2] Preprocessing Sequences (Log1p)...")
X_all_list = []
valid_ids = []

for idx, oid in enumerate(ids):
    seq = process_lc_stack(all_lcs[oid])
    if seq is not None and not np.isnan(seq).any():
        X_all_list.append(seq)
        valid_ids.append(oid)

X_all = np.array(X_all_list)
print(f"  Data Shape: {X_all.shape}")

# Dataset
class LCDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

train_loader = DataLoader(LCDataset(X_all), batch_size=64, shuffle=True)
all_loader = DataLoader(LCDataset(X_all), batch_size=64, shuffle=False)

# 3. Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=100, n_features=6, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, 64, batch_first=True, bidirectional=True)
        self.bottleneck = nn.Linear(128, embedding_dim)
        
        self.decoder_rnn = nn.LSTM(embedding_dim, 64, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(128, n_features)
        
    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)
        # Concat fwd/bwd hidden
        h = torch.cat((hidden[-2], hidden[-1]), dim=1) # (B, 128)
        emb = self.bottleneck(h) # (B, 64)
        
        # Decoder
        # Expand embedding to sequence
        # (B, 64) -> (B, 1, 64) -> (B, 100, 64)
        rep = emb.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        out, _ = self.decoder_rnn(rep) # (B, 100, 128)
        recon = self.output_layer(out) # (B, 100, 6)
        
        return recon, emb

model = LSTMAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Train
print("\n[4] Training...")
epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        if torch.isnan(loss):
            print("NaN Loss detected!")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
        optimizer.step()
        total_loss += loss.item()
        
    avg = total_loss / len(train_loader)
    print(f"  Epoch {epoch+1}: Loss = {avg:.6f}")

# 5. Extract
print("\n[5] Extracting Embeddings...")
model.eval()
embs = []
with torch.no_grad():
    for batch in all_loader:
        batch = batch.to(device)
        _, emb = model(batch)
        embs.append(emb.cpu().numpy())

final_embs = np.concatenate(embs, axis=0)
emb_df = pd.DataFrame(final_embs, columns=[f'emb_{i}' for i in range(64)])
emb_df['object_id'] = valid_ids
emb_df.to_csv('embeddings_autoencoder.csv', index=False)

print("✓ Saved embeddings_autoencoder.csv")
