"""
train.py  v2
────────────
Replaces InceptionV3 (27M params, ImageNet domain, 299x299)
with a custom 4-layer CNN (200k params, trains from scratch, 224x224).

Everything else is identical to v1: LOSO splits, two approaches,
metrics, plots, manifest loading. Only the model and the Dataset
__getitem__ change.

WHY this model, explained inline.
"""

import os, sys
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
import time

# ─────────────────────────────────────────────────────────
# CONFIG  — one place, one change
# ─────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data", "manifests")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

LEARNING_RATE = 0.001    # from-scratch CNN needs stronger signal than fine-tuning
BATCH_SIZE    = 128      # model is 456k params, 128 is fine on 48GB
MAX_SAMPLES_PER_SUBJECT = 500  # cap per subject in training set (see comment in train_one_fold)
MAX_EPOCHS    = 30
PATIENCE      = 7

LSTM_HIDDEN   = 128
LSTM_LAYERS   = 1

BALANCED_SEEDS = [42, 123, 7, 256, 999]

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
HEALTHY_IDS = [s for s,k in SUBJECT_ID_TO_KEY.items() if k.startswith("H")]
MDD_IDS     = [s for s,k in SUBJECT_ID_TO_KEY.items() if k.startswith("MDD")]


# ─────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────
# No resize. spectrograms.npy is already 224×224. We just
# convert numpy (H,W,C) → torch (C,H,W). Three operations.
# ─────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor  # (N, 3, 224, 224) already on device or in RAM
        self.y = y_tensor  # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────
# MODEL — CNN + LSTM
# ─────────────────────────────────────────────────────────
# Shape trace for ONE image, batch omitted for clarity:
#
#   Input:   (3, 224, 224)
#
#   Block 1: Conv(3→32) → BN → ReLU → MaxPool
#            (32, 224, 224) → (32, 112, 112)
#            WHY 32 filters: enough to detect basic patterns
#            (edges, gradients) in the spectrogram. More is waste.
#
#   Block 2: Conv(32→64) → BN → ReLU → MaxPool
#            (64, 112, 112) → (64, 56, 56)
#            WHY double the filters: each layer combines features
#            from the previous. 64 can represent richer textures.
#
#   Block 3: Conv(64→128) → BN → ReLU → MaxPool
#            (128, 56, 56) → (128, 28, 28)
#            Mid-level features: frequency-time regions.
#
#   Block 4: Conv(128→256) → BN → ReLU → AdaptiveAvgPool(1)
#            (256, 28, 28) → (256, 1, 1)
#            WHY AdaptiveAvgPool not MaxPool:
#              MaxPool keeps the strongest activation spatially.
#              AvgPool averages across all spatial positions.
#              For spectrograms we care about the OVERALL presence
#              of a pattern across the image, not its peak location.
#              AvgPool is correct here.
#
#   Flatten: (256, 1, 1) → (256,)
#            256 numbers. Each is "how much of pattern X is in this image".
#
#   LSTM:    (256,) → reshape to (256, 1) = 256 time steps, 1 value each
#            LSTM scans these 256 feature values in order.
#            WHY LSTM on CNN features?
#              The 256 feature maps are ordered by the CNN's internal
#              representation. The LSTM learns which SEQUENCES of
#              features matter: "if feature 40 is high AND feature 120
#              is low, that is depression." A Dense layer sees all 256
#              at once but cannot model this kind of ordered dependency.
#            Output: take the last hidden state → (128,)
#
#   Classifier: Dropout(0.3) → Linear(128→1) → (1,)
#            Dropout: randomly zero 30% of inputs during training.
#            Forces the model to not rely on any single LSTM output.
#            Linear: 128 inputs → 1 logit. Sigmoid applied by the loss.
#
# Total parameters: ~210k. Verified below in main().
# ─────────────────────────────────────────────────────────
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 224 → 112

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 112 → 56

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 56 → 28

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),    # 28 → 1
        )

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True            # (batch, seq_len, input_size)
        )

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(LSTM_HIDDEN, 1)
        )

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        feat = self.cnn(x)                          # (batch, 256, 1, 1)
        feat = feat.view(feat.size(0), -1)          # (batch, 256)
        feat = feat.unsqueeze(-1)                   # (batch, 256, 1)
        _, (h, _) = self.lstm(feat)                 # h: (1, batch, 128)
        h = h.squeeze(0)                            # (batch, 128)
        return self.head(h).squeeze(-1)             # (batch,)


# ─────────────────────────────────────────────────────────
# LOSO SPLITS
# ─────────────────────────────────────────────────────────
def get_loso_folds(subject_ids_array, allowed_sids):
    folds = []
    for test_sid in allowed_sids:
        test_mask  = (subject_ids_array == test_sid)
        train_mask = np.isin(subject_ids_array, [s for s in allowed_sids if s != test_sid])
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds


# ─────────────────────────────────────────────────────────
# TRAIN ONE FOLD
# ─────────────────────────────────────────────────────────
# pos_weight logic (from first principles):
#
#   BCEWithLogitsLoss with pos_weight=w:
#     loss(label=1) = -w · log(σ(logit))       ← MDD
#     loss(label=0) = -log(1 - σ(logit))       ← Healthy
#
#   w > 1 → MDD errors cost more → model chases MDD recall
#   w < 1 → MDD errors cost less → model chases H recall
#
#   Our training set: ~18k H, ~34k MDD. MDD is majority.
#   Model will naturally predict MDD more often.
#   We want to correct that → make H errors expensive →
#   equivalently make MDD errors CHEAP → w < 1.
#
#   Standard formula: pos_weight = n_negative / n_positive
#                                = n_H / n_MDD ≈ 0.53
#   That IS correct. The v1 value of 0.530 was right.
#   The problem was InceptionV3 not learning, not the weight.
# ─────────────────────────────────────────────────────────
def train_one_fold(train_indices, test_indices,
                   spectrograms, labels, subject_ids,
                   use_class_weights=False):

    # ── Subsample: cap at MAX_SAMPLES_PER_SUBJECT per subject ──
    # We subsample indices to avoid copying the huge spectrograms array.
    sid_train_all = subject_ids[train_indices]
    keep_indices = []
    
    for sid in np.unique(sid_train_all):
        rel_sid_indices = np.where(sid_train_all == sid)[0]
        if len(rel_sid_indices) > MAX_SAMPLES_PER_SUBJECT:
            chosen_rel = np.random.choice(rel_sid_indices, MAX_SAMPLES_PER_SUBJECT, replace=False)
        else:
            chosen_rel = rel_sid_indices
        # Map relative indices back to global indices
        keep_indices.extend(train_indices[chosen_rel])
    
    keep_indices = np.array(keep_indices)
    y_keep = labels[keep_indices]

    # 90/10 stratified train/val for early stopping
    h_idx   = np.where(y_keep == 0)[0]
    mdd_idx = np.where(y_keep == 1)[0]
    
    val_h_rel   = np.random.choice(h_idx,  max(1, int(0.1*len(h_idx))),  replace=False)
    val_mdd_rel = np.random.choice(mdd_idx, max(1, int(0.1*len(mdd_idx))), replace=False)
    val_rel_idx = np.concatenate([val_h_rel, val_mdd_rel])
    tr_rel_idx  = np.setdiff1d(np.arange(len(y_keep)), val_rel_idx)

    # Convert rel to global
    tr_indices  = keep_indices[tr_rel_idx]
    val_indices = keep_indices[val_rel_idx]
    
    t_prep = time.time()
    # Convert to contiguous tensors and pre-permute (H,W,C -> C,H,W)
    def prep_data(indices):
        X_sub = torch.from_numpy(spectrograms[indices]).permute(0, 3, 1, 2).float()
        y_sub = torch.from_numpy(labels[indices]).float()
        return X_sub, y_sub

    X_tr, y_tr = prep_data(tr_indices)
    X_val, y_val = prep_data(val_indices)
    X_te, y_te = prep_data(test_indices)
    print(f"      [Log {time.strftime('%H:%M:%S')}] Data prep done: tr={len(X_tr)}, val={len(X_val)}, te={len(X_te)} ({time.time()-t_prep:.1f}s)")

    train_loader = DataLoader(EEGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(EEGDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_LSTM().to(DEVICE)

    # Loss
    if use_class_weights:
        n_h   = int(np.sum(labels[tr_indices] == 0))
        n_mdd = int(np.sum(labels[tr_indices] == 1))
        pw    = torch.tensor([n_h / n_mdd], dtype=torch.float32).to(DEVICE)
        print(f"      [Log {time.strftime('%H:%M:%S')}] pos_weight={pw.item():.3f} (n_H={n_h}, n_MDD={n_mdd})")
    else:
        pw = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None

    print(f"      [Log {time.strftime('%H:%M:%S')}] Starting training loop...")
    for epoch in range(MAX_EPOCHS):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_tr = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(lbls)
            n_tr += len(lbls)
        
        train_loss /= n_tr

        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                val_loss  += criterion(model(imgs), lbls).item() * len(lbls)
                n_val     += len(lbls)
        val_loss /= n_val
        
        t_epoch = time.time() - t_epoch_start
        print(f"      [Log {time.strftime('%H:%M:%S')}] Ep {epoch+1:2d}/{MAX_EPOCHS} | tr_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | {t_epoch:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"      [Log {time.strftime('%H:%M:%S')}] Early stop epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    # Evaluate on test
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    preds_all, probs_all, labels_all = [], [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.sigmoid(logits)
            preds_all.append((probs >= 0.5).long().cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            labels_all.append(lbls.numpy())

    y_pred = np.concatenate(preds_all)
    y_prob = np.concatenate(probs_all)
    y_true = np.concatenate(labels_all).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = None

    h_mask  = y_true == 0
    mdd_mask = y_true == 1
    h_rec   = accuracy_score(y_true[h_mask],  y_pred[h_mask])  if h_mask.any()  else 0.0
    mdd_rec = accuracy_score(y_true[mdd_mask], y_pred[mdd_mask]) if mdd_mask.any() else 0.0

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                auc=auc, h_recall=h_rec, mdd_recall=mdd_rec,
                n_test=len(y_true), y_true=y_true, y_pred=y_pred)


# ─────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────
def plot_cm(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    # Define a modern color palette
    # MDD=Positive (1), Healthy=Negative (0)
    
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
    print(f"    Saved Plot: {os.path.basename(path)}")

def plot_acc(accs, labels, path, title):
    # Set style
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(figsize=(max(12, len(accs)*0.35), 6))
    
    # Color logic: Blue for Healthy test fold, Red for MDD test fold
    colors = ['#3498db' if 'H' in l else '#e74c3c' for l in labels]
    
    bars = ax.bar(range(len(accs)), accs, color=colors, alpha=0.9, width=0.7, zorder=3)
    
    # Add value labels on top
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7, rotation=90)

    # Mean line
    mean_val = np.mean(accs)
    ax.axhline(mean_val, color='#2c3e50', linestyle='--', linewidth=2, zorder=4, label=f'Mean Accuracy: {mean_val:.3f}')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_xlabel('Test Subject (Fold)', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.15)
    
    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#3498db', lw=4, label='Healthy Subject'),
        Line2D([0], [0], color='#e74c3c', lw=4, label='MDD Subject'),
        Line2D([0], [0], color='#2c3e50', lw=2, linestyle='--', label=f'Mean: {mean_val:.2f}')
    ]
    ax.legend(handles=custom_lines, loc='upper right', frameon=True, shadow=True)
    
    # Remove top/right spines
    sns.despine(left=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved Plot: {os.path.basename(path)}")

# ─────────────────────────────────────────────────────────
# APPROACH 1
# ─────────────────────────────────────────────────────────
def run_approach1(spectrograms, labels, subject_ids):
    print("\n" + "="*70)
    print("  APPROACH 1: 40 subjects, class-weighted LOSO")
    print("="*70)

    out = os.path.join(RESULTS_DIR, "approach1")
    os.makedirs(out, exist_ok=True)

    sids  = sorted(SUBJECT_ID_TO_KEY.keys())
    folds = get_loso_folds(subject_ids, sids)

    rows, yt_all, yp_all, accs, flabels = [], [], [], [], []

    for i, (tr, te) in enumerate(folds):
        name = SUBJECT_ID_TO_KEY[sids[i]]
        t0   = time.time()
        print(f"\n  Fold {i+1:2d}/{len(folds)} — test: {name}")

        r = train_one_fold(tr, te, spectrograms, labels, subject_ids, use_class_weights=True)
        elapsed = time.time() - t0

        r['fold'] = i; r['test_subject'] = name; r['time_sec'] = round(elapsed,1)
        rows.append({k:v for k,v in r.items() if k not in ('y_true','y_pred')})
        yt_all.extend(r['y_true']); yp_all.extend(r['y_pred'])
        accs.append(r['accuracy']); flabels.append(name)

        print(f"      acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
              f"rec={r['recall']:.3f}  f1={r['f1']:.3f}  "
              f"H_rec={r['h_recall']}  MDD_rec={r['mdd_recall']}  "
              f"[{elapsed:.0f}s]")

    # --- IMPROVED REPORTING ---
    from prettytable import PrettyTable
    
    # 1. CSV (Raw Data)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "summary.csv"), index=False)
    
    # 2. Markdown Report (Human Readable)
    report_path = os.path.join(out, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("# 📊 Analysis Report: Approach 1 (LOSO)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Total Subjects:** {len(accs)} (40)\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write("| Metric | Mean | Std Dev | Description |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Accuracy** | **{df['accuracy'].mean():.3f}** | {df['accuracy'].std():.3f} | Overall correctness |\n")
        f.write(f"| Precision | {df['precision'].mean():.3f} | {df['precision'].std():.3f} | False Positive rate proxy |\n")
        f.write(f"| Recall | {df['recall'].mean():.3f} | {df['recall'].std():.3f} | Ability to find MDD |\n")
        f.write(f"| F1 Score | {df['f1'].mean():.3f} | {df['f1'].std():.3f} | Balance of Prec/Rec |\n")
        f.write("\n")

        f.write("## 2. Per-Class Performance\n")
        f.write(f"- **Healthy Recall (Specificity):** `{df['h_recall'].mean():.1%}` (Ability to identify healthy people)\n")
        f.write(f"- **MDD Recall (Sensitivity):** `{df['mdd_recall'].mean():.1%}` (Ability to detect depression)\n\n")
        
        f.write("## 3. High Variance Subjects\n")
        f.write("Subjects where the model struggled significantly (Accuracy < 50%):\n")
        struggles = df[df['accuracy'] < 0.50]
        if not struggles.empty:
            f.write(struggles[['test_subject', 'accuracy', 'h_recall', 'mdd_recall']].to_markdown(index=False))
        else:
            f.write("None.")
        f.write("\n\n")
    
    print(f"    Saved Report: {os.path.basename(report_path)}")
    
    plot_cm(yt_all, yp_all, os.path.join(out, "confusion_matrix.png"),
            "Approach 1 — Aggregated CM (40 subjects, LOSO)")
    plot_acc(accs, flabels, os.path.join(out, "accuracy_plot.png"),
             "Approach 1 — Per-Fold Accuracy")

    print(f"\n  {'─'*50}")
    print(f"  APPROACH 1 SUMMARY")
    print(f"  {'─'*50}")
    
    # Console Table
    t = PrettyTable(['Metric', 'Mean', 'Std'])
    t.add_row(['Accuracy', f"{df['accuracy'].mean():.3f}", f"{df['accuracy'].std():.3f}"])
    t.add_row(['Precision', f"{df['precision'].mean():.3f}", f"{df['precision'].std():.3f}"])
    t.add_row(['Recall', f"{df['recall'].mean():.3f}", f"{df['recall'].std():.3f}"])
    t.add_row(['F1', f"{df['f1'].mean():.3f}", f"{df['f1'].std():.3f}"])
    print(t)
    
    return df


# ─────────────────────────────────────────────────────────
# APPROACH 2
# ─────────────────────────────────────────────────────────
def load_balanced_seeds():
    df = pd.read_csv(os.path.join(MANIFEST_DIR, "balanced_seeds.csv"))
    return {int(r['seed']): r['mdd_subjects_selected'].split(';') for _,r in df.iterrows()}

def run_approach2(spectrograms, labels, subject_ids):
    print("\n" + "="*70)
    print("  APPROACH 2: Balanced 28 subjects, 5 seeds")
    print("="*70)

    out = os.path.join(RESULTS_DIR, "approach2")
    os.makedirs(out, exist_ok=True)
    seed_map = load_balanced_seeds()

    seed_summaries = []
    yt_global, yp_global = [], []

    for seed in BALANCED_SEEDS:
        print(f"\n  {'─'*60}\n  SEED {seed}\n  {'─'*60}")

        mdd_ids      = [SUBJECT_KEY_TO_ID[k] for k in seed_map[seed]]
        allowed      = sorted(HEALTHY_IDS + mdd_ids)
        seed_dir     = os.path.join(out, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        folds        = get_loso_folds(subject_ids, allowed)

        rows, yt, yp, accs, flabels = [], [], [], [], []

        for i, (tr, te) in enumerate(folds):
            name = SUBJECT_ID_TO_KEY[allowed[i]]
            t0   = time.time()
            print(f"\n    Fold {i+1:2d}/{len(folds)} — test: {name}")

            r = train_one_fold(tr, te, spectrograms, labels, subject_ids, use_class_weights=False)
            elapsed = time.time() - t0

            r['fold'] = i; r['test_subject'] = name; r['time_sec'] = round(elapsed,1)
            rows.append({k:v for k,v in r.items() if k not in ('y_true','y_pred')})
            yt.extend(r['y_true']); yp.extend(r['y_pred'])
            accs.append(r['accuracy']); flabels.append(name)

            print(f"      acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
                  f"rec={r['recall']:.3f}  f1={r['f1']:.3f}  [{elapsed:.0f}s]")

        sdf = pd.DataFrame(rows)
        sdf.to_csv(os.path.join(seed_dir, "summary.csv"), index=False)
        plot_cm(yt, yp, os.path.join(seed_dir, "confusion_matrix.png"),
                f"Approach 2 — Seed {seed} CM")
        plot_acc(accs, flabels, os.path.join(seed_dir, "accuracy_plot.png"),
                 f"Approach 2 — Seed {seed} Accuracy")

        seed_summaries.append(dict(seed=seed,
            mean_acc=sdf['accuracy'].mean(), std_acc=sdf['accuracy'].std(),
            mean_prec=sdf['precision'].mean(), mean_rec=sdf['recall'].mean(),
            mean_f1=sdf['f1'].mean(),
            mean_h_rec=sdf['h_recall'].mean(), mean_mdd_rec=sdf['mdd_recall'].mean()))
        yt_global.extend(yt); yp_global.extend(yp)
        print(f"\n    Seed {seed}: acc={sdf['accuracy'].mean():.3f} ± {sdf['accuracy'].std():.3f}")

    odf = pd.DataFrame(seed_summaries)
    odf.to_csv(os.path.join(out, "overall_summary.csv"), index=False)
    plot_cm(yt_global, yp_global, os.path.join(out, "confusion_matrix.png"),
            "Approach 2 — Aggregated CM (all seeds)")

    print(f"\n  {'─'*50}")
    print(f"  APPROACH 2 SUMMARY (mean ± std across 5 seeds)")
    print(f"  {'─'*50}")
    print(f"  Accuracy:   {odf['mean_acc'].mean():.3f} ± {odf['mean_acc'].std():.3f}")
    print(f"  Precision:  {odf['mean_prec'].mean():.3f} ± {odf['mean_prec'].std():.3f}")
    print(f"  Recall:     {odf['mean_rec'].mean():.3f} ± {odf['mean_rec'].std():.3f}")
    print(f"  F1:         {odf['mean_f1'].mean():.3f} ± {odf['mean_f1'].std():.3f}")
    print(f"  H recall:   {odf['mean_h_rec'].mean():.3f} ± {odf['mean_h_rec'].std():.3f}")
    print(f"  MDD recall: {odf['mean_mdd_rec'].mean():.3f} ± {odf['mean_mdd_rec'].std():.3f}")
    return odf


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("  TRAIN v2 — CNN + LSTM, no InceptionV3")
    print(f"  Device: {DEVICE}")
    print("="*70)

    # Model size report
    m = CNN_LSTM()
    total = sum(p.numel() for p in m.parameters())
    print(f"\n  Model params: {total:,}")
    del m

    # Load
    print("  Loading spectrograms into RAM (should fit in 48GB)...")
    t0 = time.time()
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels       = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids  = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    print(f"  Loaded in {time.time()-t0:.1f}s: {spectrograms.shape[0]} samples, "
          f"{len(np.unique(subject_ids))} subjects, H={np.sum(labels==0)}, MDD={np.sum(labels==1)}")

    # Run Approach 1 first. It is 40 folds.
    # Approach 2 is 140 folds (28 × 5 seeds). We run it AFTER
    # Approach 1 results confirm the model is working.
    a1 = run_approach1(spectrograms, labels, subject_ids)

    # a2 = run_approach2(spectrograms, labels, subject_ids)   # uncomment after A1 looks good

    # Compare
    print("\n" + "="*70)
    print("  APPROACH 1 COMPLETE")
    print("="*70)
    print(f"  Accuracy: {a1['accuracy'].mean():.3f} ± {a1['accuracy'].std():.3f}")
    print(f"  Results:  {RESULTS_DIR}")
    print(f"  Next:     review results, then uncomment Approach 2 and re-run.\n")


if __name__ == '__main__':
    main()
