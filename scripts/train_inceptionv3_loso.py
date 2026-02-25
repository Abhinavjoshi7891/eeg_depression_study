"""
train_inceptionv3_loso.py — InceptionV3 + LSTM with Proper LOSO Validation
═══════════════════════════════════════════════════════════════════════════
Addresses reviewer concerns:
1. LOSO cross-validation (Leave-One-Subject-Out)
2. All paradigms (EC, EO, TASK) from same subject kept together in test set
3. Uses InceptionV3 (pretrained) + LSTM as in the original paper

Architecture:
    InceptionV3 (pretrained, frozen) → Feature extraction
    → LSTM (256 hidden) → Dense(1) → Sigmoid

Author: Paper revision for Scientific Reports
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "inceptionv3_loso")
LOG_FILE     = os.path.join(RESULTS_DIR, "training_log.txt")

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Hyperparameters (as per paper)
LEARNING_RATE = 0.0001  # Paper uses 0.0001
BATCH_SIZE    = 32      # Smaller batch for InceptionV3 memory
MAX_EPOCHS    = 30
PATIENCE      = 7
MAX_SAMPLES_PER_SUBJECT = 500  # Subsample to manage memory

# LSTM configuration
LSTM_HIDDEN = 256
LSTM_LAYERS = 1
DROPOUT     = 0.5

# InceptionV3 feature dimension
INCEPTION_FEATURES = 2048

# All 40 subjects for full LOSO
SUBJECT_ID_TO_KEY = {
     0:"H_14",  1:"H_16",  2:"H_19",  3:"H_22",  4:"H_23",
     5:"H_24",  6:"H_26",  7:"H_27",  8:"H_28",  9:"H_4",
    10:"H_5",  11:"H_6",  12:"H_8",  13:"H_9",
    14:"MDD_1",  15:"MDD_10", 16:"MDD_11", 17:"MDD_13", 18:"MDD_14",
    19:"MDD_15", 20:"MDD_17", 21:"MDD_18", 22:"MDD_19", 23:"MDD_2",
    24:"MDD_20", 25:"MDD_21", 26:"MDD_22", 27:"MDD_23", 28:"MDD_24",
    29:"MDD_26", 30:"MDD_27", 31:"MDD_28", 32:"MDD_29", 33:"MDD_3",
    34:"MDD_30", 35:"MDD_31", 36:"MDD_32", 37:"MDD_33", 38:"MDD_5",
    39:"MDD_6"
}

ALL_SUBJECTS = list(SUBJECT_ID_TO_KEY.keys())
N_SUBJECTS = len(ALL_SUBJECTS)

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════
def log_msg(msg, also_print=True):
    """Log message with timestamp to both console and file."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    if also_print:
        print(formatted)
    if os.path.exists(os.path.dirname(LOG_FILE)):
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")

def log_separator(title=""):
    """Log a visual separator."""
    if title:
        log_msg("═" * 70)
        log_msg(f"  {title}")
        log_msg("═" * 70)
    else:
        log_msg("─" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════
class EEGSpectrogramDataset(Dataset):
    """Dataset for EEG spectrograms."""
    def __init__(self, X, y):
        # X: (N, H, W, C) -> (N, C, H, W) for PyTorch
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ═══════════════════════════════════════════════════════════════════════════
# MODEL: InceptionV3 + LSTM
# ═══════════════════════════════════════════════════════════════════════════
class InceptionV3_LSTM(nn.Module):
    """
    Hybrid model combining InceptionV3 for spatial feature extraction
    and LSTM for temporal/sequential modeling.
    
    Architecture:
        InceptionV3 (pretrained, frozen) → 2048-dim features
        → Reshape to sequence → LSTM → Dense → Sigmoid
    """
    def __init__(self, freeze_inception=True):
        super().__init__()
        
        # Load pretrained InceptionV3 (must load with aux_logits=True for weights)
        self.inception = models.inception_v3(weights='IMAGENET1K_V1')
        
        # Disable auxiliary outputs AFTER loading weights
        self.inception.aux_logits = False
        self.inception.AuxLogits = None  # Remove the module
        
        # Remove the final classification layer
        self.inception.fc = nn.Identity()
        
        # Freeze InceptionV3 weights if specified
        if freeze_inception:
            for param in self.inception.parameters():
                param.requires_grad = False
        
        # Upsample layer to resize 224→299 for InceptionV3
        self.upsample = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        
        # LSTM for temporal modeling
        # Input: (batch, seq_len=1, features=2048)
        self.lstm = nn.LSTM(
            input_size=INCEPTION_FEATURES,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=0.0 if LSTM_LAYERS == 1 else DROPOUT
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(LSTM_HIDDEN, 1)
        )
    
    def forward(self, x):
        # x: (batch, 3, 224, 224)
        
        # Resize to 299x299 for InceptionV3
        x = self.upsample(x)  # (batch, 3, 299, 299)
        
        # Extract features (aux_logits is disabled, returns only main output)
        features = self.inception(x)  # (batch, 2048)
        
        # Reshape for LSTM: (batch, seq_len=1, features)
        features = features.unsqueeze(1)  # (batch, 1, 2048)
        
        # LSTM
        _, (h_n, _) = self.lstm(features)  # h_n: (1, batch, hidden)
        
        # Classification
        out = self.classifier(h_n.squeeze(0))  # (batch, 1)
        
        return out.squeeze(-1)  # (batch,)

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, path, title):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'MDD'], 
                yticklabels=['Healthy', 'MDD'],
                linewidths=2, linecolor='black',
                annot_kws={'size': 20, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add accuracy annotation
    acc = accuracy_score(y_true, y_pred)
    ax.text(0.5, -0.12, f'Overall Accuracy: {acc:.1%}', 
            transform=ax.transAxes, ha='center', fontsize=12, 
            fontweight='bold', color='#2c3e50')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log_msg(f"Saved: {os.path.basename(path)}")

def plot_per_fold_accuracy(results_df, path, title):
    """Plot per-fold accuracy with H/MDD distinction."""
    fig, ax = plt.subplots(figsize=(max(14, len(results_df)*0.5), 7))
    
    subjects = results_df['subject'].tolist()
    accs = results_df['accuracy'].tolist()
    colors = ['#3498db' if 'H_' in s else '#e74c3c' for s in subjects]
    
    bars = ax.bar(range(len(accs)), accs, color=colors, alpha=0.85, edgecolor='black')
    
    # Mean line
    mean_acc = np.mean(accs)
    ax.axhline(mean_acc, color='#2c3e50', linestyle='--', linewidth=2.5, 
               label=f'Mean = {mean_acc:.3f}')
    
    # Healthy vs MDD means
    h_accs = [a for s, a in zip(subjects, accs) if 'H_' in s]
    mdd_accs = [a for s, a in zip(subjects, accs) if 'MDD_' in s]
    ax.axhline(np.mean(h_accs), color='#3498db', linestyle=':', linewidth=2, 
               label=f'H Mean = {np.mean(h_accs):.3f}')
    ax.axhline(np.mean(mdd_accs), color='#e74c3c', linestyle=':', linewidth=2, 
               label=f'MDD Mean = {np.mean(mdd_accs):.3f}')
    
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Test Subject', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log_msg(f"Saved: {os.path.basename(path)}")

def plot_roc_curves(all_y_true, all_y_prob, fold_results, path):
    """Plot ROC and Precision-Recall curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # === Left: ROC Curve ===
    # Per-fold ROC curves (faded)
    for result in fold_results:
        if len(np.unique(result['y_true'])) > 1:
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_prob'])
            ax1.plot(fpr, tpr, alpha=0.15, color='gray', linewidth=1)
    
    # Aggregate ROC
    fpr_agg, tpr_agg, _ = roc_curve(all_y_true, all_y_prob)
    auc_agg = auc(fpr_agg, tpr_agg)
    ax1.plot(fpr_agg, tpr_agg, color='#27ae60', linewidth=3,
             label=f'Aggregate (AUC = {auc_agg:.3f})')
    
    # Random classifier
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
    
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve (LOSO)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(alpha=0.3)
    
    # === Right: Precision-Recall Curve ===
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
    ap = average_precision_score(all_y_true, all_y_prob)
    
    ax2.plot(recall, precision, color='#8e44ad', linewidth=3,
             label=f'AP = {ap:.3f}')
    ax2.axhline(np.mean(all_y_true), color='gray', linestyle='--', 
                label=f'Baseline = {np.mean(all_y_true):.3f}')
    
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log_msg(f"Saved: {os.path.basename(path)}")

def plot_metric_distributions(results_df, path):
    """Plot distribution of metrics across folds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('accuracy', 'Accuracy', '#3498db'),
        ('f1', 'F1 Score', '#27ae60'),
        ('h_recall', 'Healthy Recall (Specificity)', '#9b59b6'),
        ('mdd_recall', 'MDD Recall (Sensitivity)', '#e74c3c')
    ]
    
    for ax, (metric, label, color) in zip(axes.flatten(), metrics):
        values = results_df[metric].values
        
        # Histogram
        ax.hist(values, bins=15, color=color, alpha=0.7, edgecolor='black')
        
        # Mean and std lines
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color='#2c3e50', linestyle='-', linewidth=2.5,
                   label=f'Mean = {mean_val:.3f}')
        ax.axvline(mean_val - std_val, color='#2c3e50', linestyle='--', linewidth=1.5)
        ax.axvline(mean_val + std_val, color='#2c3e50', linestyle='--', linewidth=1.5,
                   label=f'Std = {std_val:.3f}')
        
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Metric Distributions Across LOSO Folds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log_msg(f"Saved: {os.path.basename(path)}")

def plot_h_vs_mdd_comparison(results_df, path):
    """Compare performance on H vs MDD test subjects."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separate H and MDD results
    h_results = results_df[results_df['subject'].str.startswith('H_')]
    mdd_results = results_df[results_df['subject'].str.startswith('MDD_')]
    
    # === Left: Box plots ===
    data_acc = [h_results['accuracy'].values, mdd_results['accuracy'].values]
    bp = axes[0].boxplot(data_acc, labels=['Healthy Subjects', 'MDD Subjects'],
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy by Subject Type', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add means as text
    axes[0].text(1, h_results['accuracy'].mean() + 0.02, 
                 f"μ={h_results['accuracy'].mean():.3f}", ha='center', fontsize=10)
    axes[0].text(2, mdd_results['accuracy'].mean() + 0.02, 
                 f"μ={mdd_results['accuracy'].mean():.3f}", ha='center', fontsize=10)
    
    # === Right: Grouped bar chart ===
    x = np.arange(4)
    width = 0.35
    
    h_metrics = [h_results['accuracy'].mean(), h_results['f1'].mean(),
                 h_results['h_recall'].mean(), h_results['mdd_recall'].mean()]
    mdd_metrics = [mdd_results['accuracy'].mean(), mdd_results['f1'].mean(),
                   mdd_results['h_recall'].mean(), mdd_results['mdd_recall'].mean()]
    
    axes[1].bar(x - width/2, h_metrics, width, label='H Test Subjects', color='#3498db')
    axes[1].bar(x + width/2, mdd_metrics, width, label='MDD Test Subjects', color='#e74c3c')
    
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Metrics by Subject Type', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Accuracy', 'F1', 'H Recall', 'MDD Recall'])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log_msg(f"Saved: {os.path.basename(path)}")

def plot_training_summary(fold_train_histories, path):
    """Plot aggregate training curves across all folds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot all fold training losses
    for i, history in enumerate(fold_train_histories):
        alpha = 0.3
        ax1.plot(history['train_loss'], alpha=alpha, color='#3498db')
        ax2.plot(history['val_loss'], alpha=alpha, color='#e74c3c')
    
    # Average curves if histories have same length
    if len(fold_train_histories) > 0:
        max_len = max(len(h['train_loss']) for h in fold_train_histories)
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss Across Folds', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Validation Loss Across Folds', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    log_msg(f"Saved: {os.path.basename(path)}")

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def train_one_fold(train_indices, test_indices, spectrograms, labels, subject_ids,
                   fold_num, subject_name, total_folds):
    """Train and evaluate model for one LOSO fold."""
    
    log_separator()
    log_msg(f"FOLD {fold_num}/{total_folds} — Test Subject: {subject_name}")
    log_separator()
    
    t_fold_start = time.time()
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. Prepare data with subsampling
    # ─────────────────────────────────────────────────────────────────────
    t_prep = time.time()
    
    # Subsample training data per subject
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
    
    # Stratified validation split (10%)
    y_keep = labels[keep_idx]
    h_idx = np.where(y_keep == 0)[0]
    mdd_idx = np.where(y_keep == 1)[0]
    
    n_val_h = max(1, int(0.1 * len(h_idx)))
    n_val_mdd = max(1, int(0.1 * len(mdd_idx)))
    
    val_h = np.random.choice(h_idx, n_val_h, replace=False)
    val_mdd = np.random.choice(mdd_idx, n_val_mdd, replace=False)
    val_rel = np.concatenate([val_h, val_mdd])
    tr_rel = np.setdiff1d(np.arange(len(y_keep)), val_rel)
    
    tr_idx = keep_idx[tr_rel]
    val_idx = keep_idx[val_rel]
    
    # Create datasets
    X_tr, y_tr = spectrograms[tr_idx], labels[tr_idx]
    X_val, y_val = spectrograms[val_idx], labels[val_idx]
    X_te, y_te = spectrograms[test_indices], labels[test_indices]
    
    train_dataset = EEGSpectrogramDataset(X_tr, y_tr)
    val_dataset = EEGSpectrogramDataset(X_val, y_val)
    test_dataset = EEGSpectrogramDataset(X_te, y_te)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Class balance
    n_h = int((y_tr == 0).sum())
    n_mdd = int((y_tr == 1).sum())
    pos_weight = n_h / max(n_mdd, 1)
    
    log_msg(f"Data preparation completed in {time.time()-t_prep:.1f}s")
    log_msg(f"  Train: {len(X_tr):,} samples | Val: {len(X_val):,} | Test: {len(X_te):,}")
    log_msg(f"  Class balance: H={n_h:,}, MDD={n_mdd:,} | pos_weight={pos_weight:.3f}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. Initialize model
    # ─────────────────────────────────────────────────────────────────────
    model = InceptionV3_LSTM(freeze_inception=True).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_msg(f"Model: InceptionV3+LSTM | Total: {total_params:,} | Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. Training loop
    # ─────────────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr = 0
    best_state = None
    
    train_losses = []
    val_losses = []
    
    log_msg("Starting training loop...")
    
    for epoch in range(MAX_EPOCHS):
        t_epoch = time.time()
        
        # Training phase
        model.train()
        epoch_train_loss = 0
        n_batches = 0
        
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            n_batches += 1
        
        epoch_train_loss /= max(n_batches, 1)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                outputs = model(batch_imgs)
                loss = criterion(outputs, batch_labels)
                epoch_val_loss += loss.item()
                n_val_batches += 1
        
        epoch_val_loss /= max(n_val_batches, 1)
        val_losses.append(epoch_val_loss)
        
        # Logging
        log_msg(f"  Epoch {epoch+1:2d}/{MAX_EPOCHS} | "
                f"train_loss: {epoch_train_loss:.4f} | "
                f"val_loss: {epoch_val_loss:.4f} | "
                f"{time.time()-t_epoch:.1f}s")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log_msg(f"  Early stopping at epoch {epoch+1} (best val_loss: {best_val_loss:.4f})")
                break
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. Evaluation
    # ─────────────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    
    all_preds = []
    all_probs = []
    all_true = []
    
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            batch_imgs = batch_imgs.to(DEVICE)
            
            outputs = model(batch_imgs)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(batch_labels.numpy())
    
    y_true = np.array(all_true).astype(int)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Per-class recall
    h_mask = (y_true == 0)
    mdd_mask = (y_true == 1)
    h_recall = accuracy_score(y_true[h_mask], y_pred[h_mask]) if h_mask.any() else 0.0
    mdd_recall = accuracy_score(y_true[mdd_mask], y_pred[mdd_mask]) if mdd_mask.any() else 0.0
    
    # AUC
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except:
        auc_score = 0.0
    
    # Log results
    log_msg(f"  RESULTS: acc={acc:.3f} | f1={f1:.3f} | auc={auc_score:.3f}")
    log_msg(f"           H_recall={h_recall:.3f} | MDD_recall={mdd_recall:.3f}")
    log_msg(f"  Fold completed in {(time.time()-t_fold_start)/60:.1f} min")
    
    return {
        'subject': subject_name,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc_score,
        'h_recall': h_recall,
        'mdd_recall': mdd_recall,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_loss': train_losses,
        'val_loss': val_losses
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Main training loop with LOSO cross-validation."""
    
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "InceptionV3 + LSTM" + " "*30 + "║")
    print("║" + " "*15 + "LOSO Cross-Validation Training" + " "*22 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Clear previous log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    log_separator("CONFIGURATION")
    log_msg(f"Device: {DEVICE}")
    log_msg(f"Results directory: {RESULTS_DIR}")
    log_msg(f"Hyperparameters:")
    log_msg(f"  Learning rate: {LEARNING_RATE}")
    log_msg(f"  Batch size: {BATCH_SIZE}")
    log_msg(f"  Max epochs: {MAX_EPOCHS}")
    log_msg(f"  Patience: {PATIENCE}")
    log_msg(f"  LSTM hidden: {LSTM_HIDDEN}")
    log_msg(f"  Max samples/subject: {MAX_SAMPLES_PER_SUBJECT}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────────────────────────────────
    log_separator("LOADING DATA")
    
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    
    log_msg(f"Spectrograms shape: {spectrograms.shape}")
    log_msg(f"Labels shape: {labels.shape}")
    log_msg(f"Unique subjects: {len(np.unique(subject_ids))}")
    log_msg(f"Class distribution: H={int((labels==0).sum()):,}, MDD={int((labels==1).sum()):,}")
    
    # ─────────────────────────────────────────────────────────────────────
    # LOSO Cross-Validation
    # ─────────────────────────────────────────────────────────────────────
    log_separator("LOSO CROSS-VALIDATION")
    log_msg(f"Running {N_SUBJECTS} folds (one per subject)")
    log_msg("All paradigms (EC, EO, TASK) for test subject are in test set")
    
    all_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    fold_histories = []
    
    t_total = time.time()
    
    for fold_idx, test_subject_id in enumerate(ALL_SUBJECTS):
        subject_name = SUBJECT_ID_TO_KEY[test_subject_id]
        
        # Get train and test indices
        test_mask = (subject_ids == test_subject_id)
        train_mask = ~test_mask
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        # Train fold
        result = train_one_fold(
            train_indices=train_indices,
            test_indices=test_indices,
            spectrograms=spectrograms,
            labels=labels,
            subject_ids=subject_ids,
            fold_num=fold_idx + 1,
            subject_name=subject_name,
            total_folds=N_SUBJECTS
        )
        
        # Store results
        all_results.append({
            'subject': result['subject'],
            'accuracy': result['accuracy'],
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'auc': result['auc'],
            'h_recall': result['h_recall'],
            'mdd_recall': result['mdd_recall']
        })
        
        all_y_true.extend(result['y_true'])
        all_y_pred.extend(result['y_pred'])
        all_y_prob.extend(result['y_prob'])
        
        fold_histories.append({
            'train_loss': result['train_loss'],
            'val_loss': result['val_loss'],
            'y_true': result['y_true'],
            'y_prob': result['y_prob']
        })
    
    total_time = time.time() - t_total
    
    # ─────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────
    log_separator("SAVING RESULTS")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)
    log_msg(f"Saved: fold_results.csv")
    
    # Convert to numpy for metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    # ─────────────────────────────────────────────────────────────────────
    # Generate visualizations
    # ─────────────────────────────────────────────────────────────────────
    log_separator("GENERATING VISUALIZATIONS")
    
    plot_confusion_matrix(
        all_y_true, all_y_pred,
        os.path.join(RESULTS_DIR, "confusion_matrix.png"),
        "InceptionV3+LSTM LOSO: Aggregate Confusion Matrix"
    )
    
    plot_per_fold_accuracy(
        results_df,
        os.path.join(RESULTS_DIR, "per_fold_accuracy.png"),
        "InceptionV3+LSTM LOSO: Per-Subject Accuracy"
    )
    
    plot_roc_curves(
        all_y_true, all_y_prob, fold_histories,
        os.path.join(RESULTS_DIR, "roc_curves.png")
    )
    
    plot_metric_distributions(
        results_df,
        os.path.join(RESULTS_DIR, "metric_distributions.png")
    )
    
    plot_h_vs_mdd_comparison(
        results_df,
        os.path.join(RESULTS_DIR, "h_vs_mdd_comparison.png")
    )
    
    plot_training_summary(
        fold_histories,
        os.path.join(RESULTS_DIR, "training_curves.png")
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────
    log_separator("FINAL RESULTS SUMMARY")
    
    # Create summary table
    table = PrettyTable()
    table.field_names = ['Metric', 'Mean', 'Std', 'Min', 'Max']
    
    for metric, label in [('accuracy', 'Accuracy'), ('f1', 'F1 Score'),
                          ('auc', 'AUC'), ('precision', 'Precision'),
                          ('recall', 'Recall (MDD)'), ('h_recall', 'Specificity (H)'),
                          ('mdd_recall', 'Sensitivity (MDD)')]:
        values = results_df[metric]
        table.add_row([
            label,
            f"{values.mean():.3f}",
            f"{values.std():.3f}",
            f"{values.min():.3f}",
            f"{values.max():.3f}"
        ])
    
    print("\n" + str(table))
    log_msg("\n" + str(table), also_print=False)
    
    # Aggregate confusion matrix stats
    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    log_msg(f"\nAggregate Confusion Matrix:")
    log_msg(f"  True Negatives (H→H):   {tn:,}")
    log_msg(f"  False Positives (H→MDD): {fp:,}")
    log_msg(f"  False Negatives (MDD→H): {fn:,}")
    log_msg(f"  True Positives (MDD→MDD): {tp:,}")
    
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_auc = roc_auc_score(all_y_true, all_y_prob)
    
    log_msg(f"\nOverall Performance:")
    log_msg(f"  Accuracy: {overall_acc:.1%}")
    log_msg(f"  AUC: {overall_auc:.3f}")
    
    log_msg(f"\nTotal training time: {total_time/60:.1f} minutes")
    log_msg(f"Results saved to: {RESULTS_DIR}")
    
    # Save summary to text file
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("InceptionV3 + LSTM with LOSO Cross-Validation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total subjects: {N_SUBJECTS}\n")
        f.write(f"Total samples: {len(labels):,}\n\n")
        f.write("Results:\n")
        f.write(str(table) + "\n\n")
        f.write(f"Overall Accuracy: {overall_acc:.1%}\n")
        f.write(f"Overall AUC: {overall_auc:.3f}\n")
        f.write(f"\nTotal time: {total_time/60:.1f} minutes\n")
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
