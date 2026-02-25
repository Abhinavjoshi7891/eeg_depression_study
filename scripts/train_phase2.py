"""
train_phase2.py (Subject-Aware Mixup)
─────────────────────────────────────
Hypothesis: 
Phase 1 (100 samples/subject) underfit because of data starvation.
Phase 2 restores data (300 samples/subject) but uses WITHIN-CLASS MIXUP
to prevent overfitting to subject signatures. By blending two MDD brains,
we create a "synthetic MDD brain" that forces the model to learn features
common to the class, not the individual.

Changes:
1. MAX_SAMPLES_PER_SUBJECT = 300 (3x Phase 1)
2. Implemented mixup_data() function
3. Training loop uses mixup with probability 0.5
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
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
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "phase2_mixup")
LOG_FILE     = os.path.join(RESULTS_DIR, "training_log.txt")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LEARNING_RATE = 0.001
BATCH_SIZE    = 128
MAX_SAMPLES_PER_SUBJECT = 50  # <--- decreased from 300
MAX_EPOCHS    = 35             # Slightly longer training for mixup
PATIENCE      = 8

LSTM_HIDDEN   = 128
LSTM_LAYERS   = 1

SUBJECT_ID_TO_KEY = {
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
SUBJECT_KEY_TO_ID = {v:k for k,v in SUBJECT_ID_TO_KEY.items()}

def log_msg(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ─────────────────────────────────────────────────────────
# MIXUP UTILS
# ─────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4, device=DEVICE):
    """
    Performs WITHIN-CLASS mixup.
    We separate indices for Class 0 and Class 1, shuffle them independently,
    and then mix.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    
    # Identify classes
    idx0 = (y == 0).nonzero(as_tuple=True)[0]
    idx1 = (y == 1).nonzero(as_tuple=True)[0]
    
    # Create permutation that respects class boundaries
    perm = torch.arange(batch_size).to(device)
    
    if len(idx0) > 0:
        perm[idx0] = idx0[torch.randperm(len(idx0)).to(device)]
    if len(idx1) > 0:
        perm[idx1] = idx1[torch.randperm(len(idx1)).to(device)]
        
    mixed_x = lam * x + (1 - lam) * x[perm]
    
    # Labels don't change because we mixed with SAME class!
    return mixed_x, y

# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─────────────────────────────────────────────────────────
# MODEL (CNN + LSTM) - Same as Phase 1
# ─────────────────────────────────────────────────────────
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.lstm = nn.LSTM(input_size=1, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(LSTM_HIDDEN, 1)
        )

    def forward(self, x):
        feat = self.cnn(x).view(x.size(0), -1).unsqueeze(-1)
        _, (h, _) = self.lstm(feat)
        return self.head(h.squeeze(0)).squeeze(-1)

# ─────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────
def plot_cm(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.set_context("notebook", font_scale=1.2)
    cmap = sns.light_palette("#2ecc71", as_cmap=True) if "Healthy" in title else sns.light_palette("#e74c3c", as_cmap=True)
    if "Aggregated" in title: cmap = "Blues"
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=['H','MDD'], yticklabels=['H','MDD'])
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    plt.xlabel('Pred'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def plot_acc(accs, labels, path, title):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(max(12, len(accs)*0.35), 6))
    colors = ['#3498db' if 'H' in l else '#e74c3c' for l in labels]
    ax.bar(range(len(accs)), accs, color=colors, alpha=0.9, width=0.7)
    ax.axhline(np.mean(accs), color='#2c3e50', linestyle='--', linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    plt.title(title, fontsize=14, fontweight='bold'); plt.ylim(0, 1.15)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# ─────────────────────────────────────────────────────────
# TRAIN ONE FOLD
# ─────────────────────────────────────────────────────────
def train_one_fold(train_indices, test_indices, spectrograms, labels, subject_ids):
    
    # 1. Subsample (300 per subject)
    sid_train_all = subject_ids[train_indices]
    keep_indices = []
    
    for sid in np.unique(sid_train_all):
        rel_sid_indices = np.where(sid_train_all == sid)[0]
        if len(rel_sid_indices) > MAX_SAMPLES_PER_SUBJECT:
            chosen_rel = np.random.choice(rel_sid_indices, MAX_SAMPLES_PER_SUBJECT, replace=False)
        else:
            chosen_rel = rel_sid_indices
        keep_indices.extend(train_indices[chosen_rel])
    
    keep_indices = np.array(keep_indices)
    y_keep = labels[keep_indices]

    # 2. Stratified Split (90/10)
    h_idx   = np.where(y_keep == 0)[0]
    mdd_idx = np.where(y_keep == 1)[0]
    val_h_rel   = np.random.choice(h_idx,  max(1, int(0.1*len(h_idx))),  replace=False)
    val_mdd_rel = np.random.choice(mdd_idx, max(1, int(0.1*len(mdd_idx))), replace=False)
    val_rel_idx = np.concatenate([val_h_rel, val_mdd_rel])
    tr_rel_idx  = np.setdiff1d(np.arange(len(y_keep)), val_rel_idx)

    tr_indices  = keep_indices[tr_rel_idx]
    val_indices = keep_indices[val_rel_idx]
    
    # 3. Data Loading
    t_prep = time.time()
    def prep_data(indices):
        X_sub = torch.from_numpy(spectrograms[indices]).permute(0, 3, 1, 2).float()
        y_sub = torch.from_numpy(labels[indices]).float()
        return X_sub, y_sub

    X_tr, y_tr = prep_data(tr_indices)
    X_val, y_val = prep_data(val_indices)
    X_te, y_te = prep_data(test_indices)
    
    log_msg(f"      [Log {time.strftime('%H:%M:%S')}] Data prep done: tr={len(X_tr)}, val={len(X_val)} ({time.time()-t_prep:.1f}s)")

    train_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(EEGDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    # 4. Train
    model = CNN_LSTM().to(DEVICE)
    
    n_h   = int(np.sum(labels[tr_indices] == 0))
    n_mdd = int(np.sum(labels[tr_indices] == 1))
    pw    = torch.tensor([n_h / n_mdd], dtype=torch.float32).to(DEVICE)
    
    log_msg(f"      [Log {time.strftime('%H:%M:%S')}] pos_weight={pw.item():.3f} (n_H={n_h}, n_MDD={n_mdd})")
    log_msg(f"      [Log {time.strftime('%H:%M:%S')}] Starting training loop...")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None

    for epoch in range(MAX_EPOCHS):
        t_start = time.time()
        model.train()
        tr_loss, tr_steps = 0.0, 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            # Mixup (50% chance)
            if np.random.rand() < 0.5:
                imgs, lbls = mixup_data(imgs, lbls)
                
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_steps += 1
        
        tr_loss /= max(1, tr_steps)

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                val_loss += criterion(model(imgs), lbls).item()
                val_steps += 1
        val_loss /= max(1, val_steps)
        
        log_msg(f"      [Log {time.strftime('%H:%M:%S')}] Ep {epoch+1:2d}/{MAX_EPOCHS} | tr_loss: {tr_loss:.4f} | val_loss: {val_loss:.4f} | {time.time()-t_start:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log_msg(f"      [Log {time.strftime('%H:%M:%S')}] Early stop epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break
    print(f"")
    
    # 5. Evaluate
    if best_state: model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    
    preds, truth = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            p = torch.sigmoid(model(imgs))
            preds.extend((p >= 0.5).long().cpu().numpy())
            truth.extend(lbls.numpy())

    y_true, y_pred = np.array(truth), np.array(preds)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    
    h_mask = (y_true == 0)
    m_mask = (y_true == 1)
    h_rec = accuracy_score(y_true[h_mask], y_pred[h_mask]) if h_mask.any() else 0.0
    m_rec = accuracy_score(y_true[m_mask], y_pred[m_mask]) if m_mask.any() else 0.0

    return dict(accuracy=acc, f1=f1, h_recall=h_rec, mdd_recall=m_rec,
                y_true=y_true, y_pred=y_pred)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  TRAINING Phase 2: within-Class Mixup + 300 samples/subj")
    print("="*70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE) # Clear old log
    
    print("  Loading spectrograms...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels       = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids  = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    
    sids_all = sorted(SUBJECT_ID_TO_KEY.keys())
    folds = list(range(len(sids_all)))
    
    results = []
    yt_all, yp_all, accs, labels_plot = [], [], [], []

    for i in folds:
        test_sid = sids_all[i]
        name = SUBJECT_ID_TO_KEY[test_sid]
        
        tr_mask = (subject_ids != test_sid)
        te_mask = (subject_ids == test_sid)
        
        log_msg(f"\nFold {i+1}/{len(folds)} — Test: {name}")
        t0 = time.time()
        
        r = train_one_fold(np.where(tr_mask)[0], np.where(te_mask)[0], 
                           spectrograms, labels, subject_ids)
        
        elapsed = time.time() - t0
        log_msg(f"      acc={r['accuracy']:.3f}  f1={r['f1']:.3f}  "
                f"H_rec={r['h_recall']:.3f}  MDD_rec={r['mdd_recall']:.3f}  "
                f"[{elapsed:.0f}s]")
        
        results.append({'test_subject': name, **{k:v for k,v in r.items() if k not in ('y_true','y_pred')}})
        yt_all.extend(r['y_true']); yp_all.extend(r['y_pred'])
        accs.append(r['accuracy']); labels_plot.append(name)

    # Report
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    
    with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w") as f:
        f.write("# 📊 Phase 2 Report: Mixup (300/subj)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("| Metric | Mean | Std Dev |\n|---|---|---|\n")
        f.write(f"| Accuracy | **{df['accuracy'].mean():.3f}** | {df['accuracy'].std():.3f} |\n")
        f.write(f"| H Recall | {df['h_recall'].mean():.3f} | {df['h_recall'].std():.3f} |\n")
        f.write(f"| MDD Recall | {df['mdd_recall'].mean():.3f} | {df['mdd_recall'].std():.3f} |\n")
        
        struggles = df[df['accuracy'] < 0.50]
        if not struggles.empty:
            f.write("\n## Struggles\n")
            f.write(struggles[['test_subject', 'accuracy']].to_markdown(index=False))

    plot_cm(yt_all, yp_all, os.path.join(RESULTS_DIR, "confusion_matrix.png"), "Phase 2 - Mixup CM")
    plot_acc(accs, labels_plot, os.path.join(RESULTS_DIR, "accuracy_plot.png"), "Phase 2 - Per Fold Accuracy")
    
    # Console
    t = PrettyTable(['Metric', 'Mean', 'Std'])
    t.add_row(['Accuracy', f"{df['accuracy'].mean():.3f}", f"{df['accuracy'].std():.3f}"])
    t.add_row(['H Recall', f"{df['h_recall'].mean():.3f}", f"{df['h_recall'].std():.3f}"])
    t.add_row(['MDD Recall', f"{df['mdd_recall'].mean():.3f}", f"{df['mdd_recall'].std():.3f}"])
    print("\n" + str(t))

if __name__ == '__main__':
    main()
