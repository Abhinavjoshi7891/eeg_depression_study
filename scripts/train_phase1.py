"""
train_phase1.py  (Aggressive Subsampling)
─────────────────────────────────────────
Hypothesis: Reducing samples per subject from 500 -> 100 will reduce
subject-specific overfitting and improve generalization (especially H recall).

Changes from train.py:
1. MAX_SAMPLES_PER_SUBJECT = 100 (was 500)
2. Dropout = 0.5 (was 0.3)
3. Integrated enhanced reporting (Markdown, Plots, PrettyTable)
4. Timestamps in logs
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

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "phase1_subsample100")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LEARNING_RATE = 0.001
BATCH_SIZE    = 128
MAX_SAMPLES_PER_SUBJECT = 100  # <--- CRITICAL CHANGE
MAX_EPOCHS    = 30
PATIENCE      = 7

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

# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor  # (N, 3, 224, 224)
        self.y = y_tensor  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─────────────────────────────────────────────────────────
# MODEL (CNN + LSTM)
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
            nn.Dropout(0.5), # <--- CHANGED from 0.3
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
    
    # Custom vibrant cmap
    cmap = sns.light_palette("#2ecc71", as_cmap=True) if "Healthy" in title else sns.light_palette("#e74c3c", as_cmap=True)
    if "Aggregated" in title:
        cmap = "Blues"

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Healthy (0)', 'MDD (1)'], 
                yticklabels=['Healthy (0)', 'MDD (1)'],
                linewidths=2, linecolor='white', cbar=False)
    
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_acc(accs, labels, path, title):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(max(12, len(accs)*0.35), 6))
    colors = ['#3498db' if 'H' in l else '#e74c3c' for l in labels]
    bars = ax.bar(range(len(accs)), accs, color=colors, alpha=0.9, width=0.7, zorder=3)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

    mean_val = np.mean(accs)
    ax.axhline(mean_val, color='#2c3e50', linestyle='--', linewidth=2, zorder=4, label=f'Mean: {mean_val:.3f}')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_xlabel('Test Subject', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.15)
    
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#3498db', lw=4, label='Healthy'),
        Line2D([0], [0], color='#e74c3c', lw=4, label='MDD'),
        Line2D([0], [0], color='#2c3e50', lw=2, linestyle='--', label=f'Mean')
    ]
    ax.legend(handles=custom_lines, loc='upper right')
    sns.despine(left=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────────────────
# TRAIN ONE FOLD
# ─────────────────────────────────────────────────────────
def train_one_fold(train_indices, test_indices, spectrograms, labels, subject_ids):
    
    # 1. Subsample (Aggressive: 100 per subject)
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
    
    # 3. Data Loading (Fast Contiguous)
    t_prep = time.time()
    def prep_data(indices):
        X_sub = torch.from_numpy(spectrograms[indices]).permute(0, 3, 1, 2).float()
        y_sub = torch.from_numpy(labels[indices]).float()
        return X_sub, y_sub

    X_tr, y_tr = prep_data(tr_indices)
    X_val, y_val = prep_data(val_indices)
    X_te, y_te = prep_data(test_indices)
    print(f"      [Log {time.strftime('%H:%M:%S')}] Data prep: tr={len(X_tr)}, val={len(X_val)} ({time.time()-t_prep:.1f}s)")

    train_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(EEGDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    # 4. Train
    model = CNN_LSTM().to(DEVICE)
    
    n_h   = int(np.sum(labels[tr_indices] == 0))
    n_mdd = int(np.sum(labels[tr_indices] == 1))
    pw    = torch.tensor([n_h / n_mdd], dtype=torch.float32).to(DEVICE)
    print(f"      [Log {time.strftime('%H:%M:%S')}] pos_weight={pw.item():.3f} (n_H={n_h}, n_MDD={n_mdd})")
    
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
        
        print(f"      [Log {time.strftime('%H:%M:%S')}] Ep {epoch+1:2d} | Tr: {tr_loss:.4f} | Val: {val_loss:.4f} | {time.time()-t_start:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"      [Log {time.strftime('%H:%M:%S')}] Early stop epoch {epoch+1} (Best: {best_val_loss:.4f})")
                break

    # 5. Evaluate
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    
    preds, probs, truth = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            p = torch.sigmoid(model(imgs))
            preds.extend((p >= 0.5).long().cpu().numpy())
            probs.extend(p.cpu().numpy())
            truth.extend(lbls.numpy())

    y_true, y_pred = np.array(truth), np.array(preds)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    
    h_mask = (y_true == 0)
    m_mask = (y_true == 1)
    h_rec = accuracy_score(y_true[h_mask], y_pred[h_mask]) if h_mask.any() else 0.0
    m_rec = accuracy_score(y_true[m_mask], y_pred[m_mask]) if m_mask.any() else 0.0

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, 
                h_recall=h_rec, mdd_recall=m_rec,
                y_true=y_true, y_pred=y_pred)

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  TRAINING Phase 1: Aggressive Subsampling (100 samples/subject)")
    print("="*70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("  Loading spectrograms...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels       = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids  = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    
    # LOSO
    sids_all = sorted(SUBJECT_ID_TO_KEY.keys())
    folds = list(range(len(sids_all)))
    
    results = []
    yt_all, yp_all = [], []
    accs, labels_plot = [], []

    for i in folds:
        test_sid = sids_all[i]
        name = SUBJECT_ID_TO_KEY[test_sid]
        
        # Test mask
        test_mask = (subject_ids == test_sid)
        tr_mask   = (subject_ids != test_sid) # Train on everyone else
        tr_idx    = np.where(tr_mask)[0]
        te_idx    = np.where(test_mask)[0]
        
        print(f"\n  Fold {i+1}/{len(folds)} — Test: {name}")
        t0 = time.time()
        
        r = train_one_fold(tr_idx, te_idx, spectrograms, labels, subject_ids)
        
        elapsed = time.time() - t0
        print(f"      Result: Acc={r['accuracy']:.3f} H_Rec={r['h_recall']:.3f} MDD_Rec={r['mdd_recall']:.3f} [{elapsed:.0f}s]")
        
        res_entry = {k:v for k,v in r.items() if k not in ('y_true','y_pred')}
        res_entry['test_subject'] = name
        results.append(res_entry)
        
        yt_all.extend(r['y_true'])
        yp_all.extend(r['y_pred'])
        accs.append(r['accuracy'])
        labels_plot.append(name)

    # ─────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    
    # Report Markdown
    with open(os.path.join(RESULTS_DIR, "REPORT.md"), "w") as f:
        f.write("# 📊 Phase 1 Report: Aggressive Subsampling (100/subj)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write("| Metric | Mean | Std Dev |\n|---|---|---|\n")
        f.write(f"| Accuracy | **{df['accuracy'].mean():.3f}** | {df['accuracy'].std():.3f} |\n")
        f.write(f"| H Recall | {df['h_recall'].mean():.3f} | {df['h_recall'].std():.3f} |\n")
        f.write(f"| MDD Recall | {df['mdd_recall'].mean():.3f} | {df['mdd_recall'].std():.3f} |\n\n")
        
        f.write("## 2. High Variance Subjects\n")
        struggles = df[df['accuracy'] < 0.50]
        if not struggles.empty:
            f.write(struggles[['test_subject', 'accuracy', 'h_recall', 'mdd_recall']].to_markdown(index=False))
        else:
            f.write("None.")

    # Plots
    plot_cm(yt_all, yp_all, os.path.join(RESULTS_DIR, "confusion_matrix.png"), "Phase 1 - Aggregated CM")
    plot_acc(accs, labels_plot, os.path.join(RESULTS_DIR, "accuracy_plot.png"), "Phase 1 - Per Fold Accuracy")
    
    # Console Table
    t = PrettyTable(['Metric', 'Mean', 'Std'])
    t.add_row(['Accuracy', f"{df['accuracy'].mean():.3f}", f"{df['accuracy'].std():.3f}"])
    t.add_row(['H Recall', f"{df['h_recall'].mean():.3f}", f"{df['h_recall'].std():.3f}"])
    t.add_row(['MDD Recall', f"{df['mdd_recall'].mean():.3f}", f"{df['mdd_recall'].std():.3f}"])
    print("\n" + str(t))
    print(f"  Results saved to: {RESULTS_DIR}")

if __name__ == '__main__':
    main()
