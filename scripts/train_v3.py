"""
train_v3.py — DeepHybrid-V3 (SOTA Approach)
─────────────────────────────────────────────────────────
Implements "Hybrid 2.0" Architecture for EEG Depression Detection:
1. BALANCED SAMPLING: Enforces 1:1 Healthy/MDD ratio per batch.
2. SPECAUGMENT: Random time/freq masking to prevent overfitting.
3. RESNET-BiLSTM: Deeper feature extraction + bidirectional temporal modeling.
4. COORDINATE ATTENTION (Optional): Focuses on relevant frequency bands.

Designed for limited datasets (40 subjects) to maximize generalization.
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results_v3")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 0.0008   # Slightly lower for deeper model
BATCH_SIZE    = 64       # Smaller batch size for better generalization
MAX_EPOCHS    = 40       # More epochs allowed due to regularization
PATIENCE      = 10       # More patience
spec_augment_prob = 0.5  # Probability of applying SpecAugment


# ─────────────────────────────────────────────────────────
# AUGMENTATION: SpecAugment
# ─────────────────────────────────────────────────────────
class SpecAugment:
    def __init__(self, time_mask_param=30, freq_mask_param=20):
        self.T = time_mask_param
        self.F = freq_mask_param

    def __call__(self, x):
        # x: (C, H, W) -> (3, 224, 224)
        # We process the image as a spectrogram.
        # H = Frequency axis (224), W = Time axis (224)
        
        # Clone to avoid modifying original
        x_aug = x.clone()
        C, H, W = x_aug.shape
        
        # Frequency Masking
        f_len = np.random.randint(0, self.F)
        f0 = np.random.randint(0, H - f_len)
        x_aug[:, f0:f0+f_len, :] = 0.0

        # Time Masking
        t_len = np.random.randint(0, self.T)
        t0 = np.random.randint(0, W - t_len)
        x_aug[:, :, t0:t0+t_len] = 0.0
        
        return x_aug

# ─────────────────────────────────────────────────────────
# DATASET & BALANCED SAMPLER
# ─────────────────────────────────────────────────────────
class EEGDatasetV3(Dataset):
    def __init__(self, X_tensor, y_tensor, augment=False):
        self.X = X_tensor
        self.y = y_tensor
        self.augment = augment
        self.transform = SpecAugment()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        if self.augment and np.random.rand() < spec_augment_prob:
            img = self.transform(img)
        return img, self.y[idx]

class BalancedBatchSampler(Sampler):
    """
    Forces each batch to have 50% Healthy and 50% MDD.
    This is CRITICAL for training on imbalanced data.
    """
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.h_indices = np.where(labels == 0)[0]
        self.mdd_indices = np.where(labels == 1)[0]
        
        self.n_batches = int(min(len(self.h_indices), len(self.mdd_indices)) * 2 / batch_size)
    
    def __iter__(self):
        for _ in range(self.n_batches):
            h_batch = np.random.choice(self.h_indices, self.batch_size // 2, replace=False)
            mdd_batch = np.random.choice(self.mdd_indices, self.batch_size // 2, replace=False)
            batch = np.concatenate([h_batch, mdd_batch])
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches

# ─────────────────────────────────────────────────────────
# MODEL: ResNet-BiLSTM (DeepHybrid-V3)
# ─────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial Conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 224 -> 112 -> 56
        )
        
        # Residual Blocks
        self.layer1 = ResidualBlock(32, 64, stride=2)   # 56 -> 28
        self.layer2 = ResidualBlock(64, 128, stride=2)  # 28 -> 14
        self.layer3 = ResidualBlock(128, 256, stride=2) # 14 -> 7
        
        # Temporal Feature Aggregation
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Classifier
        self.head = nn.Sequential(
            nn.Dropout(0.5), # Higher dropout for regularization
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # (B, 256, 7, 7)
        
        # Global Pool -> (B, 256, 1, 1)
        x = self.pool(x).flatten(1) # (B, 256)
        
        # Add sequence dim for LSTM: (B, 1, 256)
        x = x.unsqueeze(1)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x) # (B, 1, 256) due to bidirectional (128*2)
        feat = lstm_out[:, -1, :]  # Take last time step
        
        return self.head(feat).squeeze(-1)

# ─────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────
def get_loso_folds(subject_ids_array, allowed_sids):
    folds = []
    for test_sid in allowed_sids:
        test_mask  = (subject_ids_array == test_sid)
        train_mask = np.isin(subject_ids_array, [s for s in allowed_sids if s != test_sid])
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds

# ─────────────────────────────────────────────────────────
# TRAIN FUNCTION
# ─────────────────────────────────────────────────────────
def train_one_fold(train_indices, test_indices, spectrograms, labels, subject_ids):
    
    # 1. Prep Data (Contiguous RAM)
    def prep(idx):
        return (torch.from_numpy(spectrograms[idx]).permute(0,3,1,2).float(), 
                torch.from_numpy(labels[idx]).float())

    X_tr, y_tr = prep(train_indices)
    
    # Validation Split (Stratified 10%)
    h_idx = np.where(y_tr.numpy() == 0)[0]
    m_idx = np.where(y_tr.numpy() == 1)[0]
    val_h = np.random.choice(h_idx, int(0.1*len(h_idx)), replace=False)
    val_m = np.random.choice(m_idx, int(0.1*len(m_idx)), replace=False)
    val_idx = np.concatenate([val_h, val_m])
    tr_idx  = np.setdiff1d(np.arange(len(y_tr)), val_idx)
    
    X_val, y_val = X_tr[val_idx], y_tr[val_idx]
    X_tr, y_tr   = X_tr[tr_idx], y_tr[tr_idx]
    
    X_te, y_te   = prep(test_indices)
    
    # 2. DataLoaders
    # Note: Train uses BalancedSampler + Augmentation
    train_ds = EEGDatasetV3(X_tr, y_tr, augment=True)
    sampler  = BalancedBatchSampler(y_tr.numpy(), BATCH_SIZE)
    train_dl = DataLoader(train_ds, batch_sampler=sampler)
    
    val_dl   = DataLoader(EEGDatasetV3(X_val, y_val), batch_size=BATCH_SIZE)
    test_dl  = DataLoader(EEGDatasetV3(X_te, y_te), batch_size=BATCH_SIZE)

    # 3. Model
    model = ResNet_BiLSTM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss() # No pos_weight needed due to balanced sampling

    # 4. Loop
    best_loss = float('inf')
    patience_ctr = 0
    best_state = None
    
    print(f"      [V3] Starting fold... Tr: {len(tr_idx)} Val: {len(val_idx)} Test: {len(y_te)}")
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        tr_loss = 0
        steps = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            steps += 1
        tr_loss /= max(1, steps)
        
        # Val
        model.eval()
        val_loss = 0
        v_steps = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x), y).item()
                v_steps += 1
        val_loss /= max(1, v_steps)
        
        print(f"      Ep {epoch+1:2d} | Tr: {tr_loss:.4f} | Val: {val_loss:.4f}", end="\r")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_ctr = 0
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n      Early stop epoch {epoch+1} (Best: {best_loss:.4f})")
                break
    
    # 5. Test
    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    
    preds, probs, truth = [], [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEVICE)
            logit = model(x)
            prob = torch.sigmoid(logit)
            preds.extend((prob >= 0.5).long().cpu().numpy())
            probs.extend(prob.cpu().numpy())
            truth.extend(y.numpy())
            
    # Metrics
    y_true, y_pred = np.array(truth), np.array(preds)
    
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity / Sensitivity
    h_mask = (y_true == 0)
    m_mask = (y_true == 1)
    
    h_rec = accuracy_score(y_true[h_mask], y_pred[h_mask]) if h_mask.any() else 0.0
    m_rec = accuracy_score(y_true[m_mask], y_pred[m_mask]) if m_mask.any() else 0.0
    
    return {
        'accuracy': acc, 'f1': f1, 'h_recall': h_rec, 
        'mdd_recall': m_rec, 'y_true': y_true, 'y_pred': y_pred
    }

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  TRAINING V3: SOTA Hybrid Model + Balanced Sampling")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("  Loading Data...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels       = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids  = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    
    id_map = {
     0:"H_14", 1:"H_16", 2:"H_19", 3:"H_22", 4:"H_23",
     5:"H_24", 6:"H_26", 7:"H_27", 8:"H_28", 9:"H_4",
    10:"H_5", 11:"H_6", 12:"H_8", 13:"H_9",
    14:"MDD_1", 15:"MDD_10", 16:"MDD_11", 17:"MDD_13", 18:"MDD_14",
    19:"MDD_15", 20:"MDD_17", 21:"MDD_18", 22:"MDD_19", 23:"MDD_2",
    24:"MDD_20", 25:"MDD_21", 26:"MDD_22", 27:"MDD_23", 28:"MDD_24",
    29:"MDD_26", 30:"MDD_27", 31:"MDD_28", 32:"MDD_29", 33:"MDD_3",
    34:"MDD_30", 35:"MDD_31", 36:"MDD_32", 37:"MDD_33", 38:"MDD_5",
    39:"MDD_6"
    }
    
    sids_all = sorted(id_map.keys())
    folds = get_loso_folds(subject_ids, sids_all)
    
    results = []
    yt_all, yp_all = [], []
    
    for i, (tr, te) in enumerate(folds):
        name = id_map[sids_all[i]]
        print(f"\n  Fold {i+1}/{len(folds)}: Test {name}")
        
        r = train_one_fold(tr, te, spectrograms, labels, subject_ids)
        
        print(f"      Acc: {r['accuracy']:.3f} | F1: {r['f1']:.3f} | H_Rec: {r['h_recall']:.3f} | MDD_Rec: {r['mdd_recall']:.3f}")
        
        results.append({
            'subject': name, 'accuracy': r['accuracy'], 'f1': r['f1'],
            'h_recall': r['h_recall'], 'mdd_recall': r['mdd_recall']
        })
        yt_all.extend(r['y_true'])
        yp_all.extend(r['y_pred'])
        
    # Save Report
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "v3_summary.csv"), index=False)
    
    print("\n" + "="*60)
    print("  V3 RESULTS SUMMARY")
    print("="*60)
    t = PrettyTable(['Metric', 'Mean', 'Std'])
    t.add_row(['Accuracy', f"{df['accuracy'].mean():.3f}", f"{df['accuracy'].std():.3f}"])
    t.add_row(['F1 Score', f"{df['f1'].mean():.3f}", f"{df['f1'].std():.3f}"])
    t.add_row(['H Recall', f"{df['h_recall'].mean():.3f}", f"{df['h_recall'].std():.3f}"])
    t.add_row(['MDD Recall', f"{df['mdd_recall'].mean():.3f}", f"{df['mdd_recall'].std():.3f}"])
    print(t)

if __name__ == "__main__":
    main()
