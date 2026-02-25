"""
train_v4.py — Balanced Subject Training (14 H + 14 MDD = 28 subjects)
───────────────────────────────────────────────────────────────────────
Hypothesis:
Using perfectly balanced subjects (14 vs 14) should eliminate class imbalance
at the subject level and may improve generalization.

Key changes:
1. Use only 28 subjects (all 14 Healthy + 14 MDD)
2. 500 samples per subject (standard amount)
3. No fancy sampling - just pos_weight for batch class imbalance
4. Simple CNN_LSTM architecture
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "v4_balanced_28subj")
LOG_FILE     = os.path.join(RESULTS_DIR, "training_log.txt")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE    = 128
MAX_EPOCHS    = 30
PATIENCE      = 7
MAX_SAMPLES_PER_SUBJECT = 500

LSTM_HIDDEN = 128
LSTM_LAYERS = 1

# All 14 Healthy subjects
H_SUBJECTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Select 14 MDD subjects (first 14 by ID)
MDD_SUBJECTS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

# Combined 28 subjects
SELECTED_SUBJECTS = H_SUBJECTS + MDD_SUBJECTS

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

# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════
def log_msg(msg):
    timestamp = time.strftime('%H:%M:%S')
    formatted = f"[Log {timestamp}] {msg}"
    print(formatted)
    if os.path.exists(os.path.dirname(LOG_FILE)):
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")

# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ═══════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'MDD'], yticklabels=['Healthy', 'MDD'],
                linewidths=2, linecolor='black', annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('True', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_per_fold_accuracy(accs, labels, path, title):
    fig, ax = plt.subplots(figsize=(max(12, len(accs)*0.4), 6))
    colors = ['#3498db' if 'H_' in l else '#e74c3c' for l in labels]
    ax.bar(range(len(accs)), accs, color=colors, alpha=0.85, edgecolor='black')
    ax.axhline(np.mean(accs), color='#2c3e50', linestyle='--', linewidth=2.5, label=f'Mean={np.mean(accs):.3f}')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ═══════════════════════════════════════════════════════════
# TRAIN ONE FOLD
# ═══════════════════════════════════════════════════════════
def train_one_fold(train_indices, test_indices, spectrograms, labels, subject_ids, fold_num, subject_name):
    log_msg(f"\nFold {fold_num}/28 — Test: {subject_name}")
    
    # 1. Subsample training data
    t_prep = time.time()
    sid_train = subject_ids[train_indices]
    keep_idx = []
    
    for sid in np.unique(sid_train):
        sid_mask = np.where(sid_train == sid)[0]
        if len(sid_mask) > MAX_SAMPLES_PER_SUBJECT:
            chosen = np.random.choice(sid_mask, MAX_SAMPLES_PER_SUBJECT, replace=False)
        else:
            chosen = sid_mask
        keep_idx.extend(train_indices[chosen])
    
    keep_idx = np.array(keep_idx)
    y_keep = labels[keep_idx]
    
    # 2. Stratified val split (10%)
    h_idx = np.where(y_keep == 0)[0]
    mdd_idx = np.where(y_keep == 1)[0]
    val_h = np.random.choice(h_idx, max(1, int(0.1 * len(h_idx))), replace=False)
    val_mdd = np.random.choice(mdd_idx, max(1, int(0.1 * len(mdd_idx))), replace=False)
    val_rel = np.concatenate([val_h, val_mdd])
    tr_rel = np.setdiff1d(np.arange(len(y_keep)), val_rel)
    
    tr_idx = keep_idx[tr_rel]
    val_idx = keep_idx[val_rel]
    
    # 3. Prepare tensors
    def prep(indices):
        X = torch.from_numpy(spectrograms[indices]).permute(0, 3, 1, 2).float()
        y = torch.from_numpy(labels[indices]).float()
        return X, y
    
    X_tr, y_tr = prep(tr_idx)
    X_val, y_val = prep(val_idx)
    X_te, y_te = prep(test_indices)
    
    n_h = int((y_tr == 0).sum())
    n_mdd = int((y_tr == 1).sum())
    
    log_msg(f"Data prep done: tr={len(X_tr)}, val={len(X_val)}, te={len(X_te)} ({time.time()-t_prep:.1f}s)")
    log_msg(f"pos_weight={n_h/n_mdd:.3f} (n_H={n_h}, n_MDD={n_mdd})")
    
    train_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EEGDataset(X_te, y_te), batch_size=BATCH_SIZE)
    
    # 4. Model
    model = CNN_LSTM().to(DEVICE)
    pw = torch.tensor([n_h / n_mdd], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val = float('inf')
    patience_ctr = 0
    best_state = None
    
    log_msg("Starting training loop...")
    
    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        
        # Train
        model.train()
        tr_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                val_loss += criterion(model(imgs), lbls).item()
        val_loss /= len(val_loader)
        
        log_msg(f"Ep {epoch+1:2d}/{MAX_EPOCHS} | tr_loss: {tr_loss:.4f} | val_loss: {val_loss:.4f} | {time.time()-t0:.1f}s")
        
        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log_msg(f"Early stop epoch {epoch+1} (best val_loss={best_val:.4f})")
                break
    
    # 5. Evaluate
    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    
    preds, probs, truth = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            p = torch.sigmoid(model(imgs))
            preds.extend((p >= 0.5).long().cpu().numpy())
            probs.extend(p.cpu().numpy())
            truth.extend(lbls.numpy())
    
    y_true, y_pred, y_prob = np.array(truth).astype(int), np.array(preds), np.array(probs)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    h_mask, mdd_mask = (y_true == 0), (y_true == 1)
    h_rec = accuracy_score(y_true[h_mask], y_pred[h_mask]) if h_mask.any() else 0.0
    mdd_rec = accuracy_score(y_true[mdd_mask], y_pred[mdd_mask]) if mdd_mask.any() else 0.0
    
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except:
        auc_val = 0.0
    
    log_msg(f"acc={acc:.3f}  f1={f1:.3f}  H_rec={h_rec:.3f}  MDD_rec={mdd_rec:.3f}")
    
    return {
        'subject': subject_name, 'accuracy': acc, 'f1': f1, 'auc': auc_val,
        'h_recall': h_rec, 'mdd_recall': mdd_rec,
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob
    }

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n" + "="*70)
    print("  TRAIN V4: Balanced Subjects (14 H + 14 MDD = 28)")
    print("="*70 + "\n")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    # Load data
    print("  Loading spectrograms...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    
    # Filter to selected subjects
    mask = np.isin(subject_ids, SELECTED_SUBJECTS)
    print(f"  Original samples: {len(spectrograms):,}")
    print(f"  Selected subjects: {len(SELECTED_SUBJECTS)} (14 H + 14 MDD)")
    print(f"  Selected samples: {mask.sum():,}")
    
    # LOSO across 28 subjects
    results = []
    all_yt, all_yp = [], []
    accs, subj_names = [], []
    
    t_total = time.time()
    
    for i, test_sid in enumerate(SELECTED_SUBJECTS):
        subject_name = SUBJECT_ID_TO_KEY[test_sid]
        
        # Train on other 27 subjects, test on this one
        train_sids = [s for s in SELECTED_SUBJECTS if s != test_sid]
        
        tr_mask = np.isin(subject_ids, train_sids)
        te_mask = (subject_ids == test_sid)
        
        r = train_one_fold(
            np.where(tr_mask)[0], np.where(te_mask)[0],
            spectrograms, labels, subject_ids,
            fold_num=i+1, subject_name=subject_name
        )
        
        results.append({k: v for k, v in r.items() if k not in ['y_true', 'y_pred', 'y_prob']})
        all_yt.extend(r['y_true'])
        all_yp.extend(r['y_pred'])
        accs.append(r['accuracy'])
        subj_names.append(subject_name)
    
    total_time = time.time() - t_total
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)
    
    # Plots
    plot_confusion_matrix(all_yt, all_yp, os.path.join(RESULTS_DIR, "confusion_matrix.png"),
                          "V4: Balanced 28 Subjects - Confusion Matrix")
    plot_per_fold_accuracy(accs, subj_names, os.path.join(RESULTS_DIR, "accuracy_plot.png"),
                           "V4: Per-Fold Accuracy (28 LOSO)")
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    
    t = PrettyTable(['Metric', 'Mean', 'Std', 'Min', 'Max'])
    t.add_row(['Accuracy', f"{df['accuracy'].mean():.3f}", f"{df['accuracy'].std():.3f}",
               f"{df['accuracy'].min():.3f}", f"{df['accuracy'].max():.3f}"])
    t.add_row(['F1', f"{df['f1'].mean():.3f}", f"{df['f1'].std():.3f}",
               f"{df['f1'].min():.3f}", f"{df['f1'].max():.3f}"])
    t.add_row(['H Recall', f"{df['h_recall'].mean():.3f}", f"{df['h_recall'].std():.3f}",
               f"{df['h_recall'].min():.3f}", f"{df['h_recall'].max():.3f}"])
    t.add_row(['MDD Recall', f"{df['mdd_recall'].mean():.3f}", f"{df['mdd_recall'].std():.3f}",
               f"{df['mdd_recall'].min():.3f}", f"{df['mdd_recall'].max():.3f}"])
    print(t)
    
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Results: {RESULTS_DIR}\n")

if __name__ == '__main__':
    main()
