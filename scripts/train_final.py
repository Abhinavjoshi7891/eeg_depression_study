"""
train_final.py — Full Dataset + Balanced Sampling + Enhanced Visualizations
────────────────────────────────────────────────────────────────────────────

FINAL SOLUTION with comprehensive logging and visualizations:
1. Use ALL data (~1500 samples per subject) — no subsampling
2. BalancedBatchSampler ensures 50% H / 50% MDD per batch
3. Keep proven CNN_LSTM architecture (456k params)
4. Enhanced logging: terminal + file with timestamps
5. Rich visualizations: 10+ plots for deep analysis

Expected Results:
- Overall Accuracy: 77-83%
- H Recall: 65-75%
- MDD Recall: 75-85%
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

# ═════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "final_full_balanced")
LOG_FILE     = os.path.join(RESULTS_DIR, "training_log.txt")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE    = 128
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

# ═════════════════════════════════════════════════════════
# ENHANCED LOGGING SYSTEM
# ═════════════════════════════════════════════════════════
def log_msg(msg, to_file=True, to_console=True):
    """
    Enhanced logging with timestamp to both console and file.
    Format: [Log HH:MM:SS] message
    """
    timestamp = time.strftime('%H:%M:%S')
    formatted_msg = f"[Log {timestamp}] {msg}"
    
    if to_console:
        print(formatted_msg)
    
    if to_file and LOG_FILE and os.path.exists(os.path.dirname(LOG_FILE)):
        with open(LOG_FILE, "a") as f:
            f.write(formatted_msg + "\n")

def log_section(title):
    """Log a section separator"""
    sep = "─" * 70
    log_msg(sep)
    log_msg(title)
    log_msg(sep)

# ═════════════════════════════════════════════════════════
# BALANCED BATCH SAMPLER
# ═════════════════════════════════════════════════════════
class BalancedBatchSampler(Sampler):
    """Forces each batch to have 50% Healthy and 50% MDD samples."""
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        
        self.h_indices = np.where(labels == 0)[0]
        self.mdd_indices = np.where(labels == 1)[0]
        
        samples_per_class = batch_size // 2
        self.n_batches = min(len(self.h_indices), len(self.mdd_indices)) // samples_per_class
    
    def __iter__(self):
        for _ in range(self.n_batches):
            h_batch = np.random.choice(self.h_indices, self.batch_size // 2, replace=False)
            mdd_batch = np.random.choice(self.mdd_indices, self.batch_size // 2, replace=False)
            
            batch = np.concatenate([h_batch, mdd_batch])
            np.random.shuffle(batch)
            
            yield batch.tolist()

    def __len__(self):
        return self.n_batches

# ═════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════
class EEGDataset(Dataset):
    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ═════════════════════════════════════════════════════════
# MODEL: CNN + LSTM
# ═════════════════════════════════════════════════════════
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

# ═════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, path, title):
    """Confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'MDD'], 
                yticklabels=['Healthy', 'MDD'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='black',
                annot_kws={'size': 16, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_fold_accuracy(accuracies, labels, path, title):
    """Bar plot of accuracy per fold"""
    fig, ax = plt.subplots(figsize=(max(14, len(accuracies)*0.4), 6))
    
    colors = ['#3498db' if 'H_' in label else '#e74c3c' for label in labels]
    bars = ax.bar(range(len(accuracies)), accuracies, color=colors, 
                   alpha=0.85, edgecolor='black', linewidth=0.8)
    
    mean_acc = np.mean(accuracies)
    ax.axhline(mean_acc, color='#2c3e50', linestyle='--', linewidth=2.5, 
               label=f'Mean = {mean_acc:.3f}')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Test Subject (LOSO Fold)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Healthy fold'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='MDD fold'),
        plt.Line2D([0], [0], color='#2c3e50', linestyle='--', linewidth=2.5, label=f'Mean')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(all_y_true, all_y_prob, fold_results, path):
    """ROC curve with per-fold and aggregate"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Individual fold ROC curves (faded)
    for result in fold_results:
        if len(result['y_true']) > 0 and len(np.unique(result['y_true'])) > 1:
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_prob'])
            fold_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, alpha=0.15, color='gray', linewidth=1)
    
    # Aggregate ROC
    fpr_agg, tpr_agg, _ = roc_curve(all_y_true, all_y_prob)
    auc_agg = auc(fpr_agg, tpr_agg)
    ax1.plot(fpr_agg, tpr_agg, color='#2ecc71', linewidth=3, 
             label=f'Aggregate (AUC = {auc_agg:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random (AUC = 0.500)')
    
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve (All Folds)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Right: Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_prob)
    ax2.plot(recall, precision, color='#3498db', linewidth=3)
    ax2.fill_between(recall, precision, alpha=0.2, color='#3498db')
    
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(all_histories, path):
    """Training and validation loss curves across all folds"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual fold curves (faded)
    for history in all_histories:
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], color='#27ae60', alpha=0.1, linewidth=1)
        ax.plot(epochs, history['val_loss'], color='#e74c3c', alpha=0.1, linewidth=1)
    
    # Calculate and plot mean curves
    max_epochs = max(len(h['train_loss']) for h in all_histories)
    train_losses_padded = []
    val_losses_padded = []
    
    for h in all_histories:
        train_padded = list(h['train_loss']) + [np.nan] * (max_epochs - len(h['train_loss']))
        val_padded = list(h['val_loss']) + [np.nan] * (max_epochs - len(h['val_loss']))
        train_losses_padded.append(train_padded)
        val_losses_padded.append(val_padded)
    
    mean_train = np.nanmean(train_losses_padded, axis=0)
    mean_val = np.nanmean(val_losses_padded, axis=0)
    
    epochs = range(1, max_epochs + 1)
    ax.plot(epochs, mean_train, color='#27ae60', linewidth=3, label='Mean Train Loss')
    ax.plot(epochs, mean_val, color='#e74c3c', linewidth=3, label='Mean Val Loss')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
    ax.set_title('Training Dynamics (All 40 Folds)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_performance_comparison(df, path):
    """Bar chart comparing H vs MDD recall"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    means = [df['h_recall'].mean(), df['mdd_recall'].mean()]
    stds = [df['h_recall'].std(), df['mdd_recall'].std()]
    
    bars = ax.bar(x, means, yerr=stds, capsize=10, 
                   color=['#3498db', '#e74c3c'], alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Healthy Recall', 'MDD Recall'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f} ± {std:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_distributions(df, path):
    """Box plots for all metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'h_recall', 'mdd_recall']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'H Recall', 'MDD Recall']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        data = df[metric].values
        bp = ax.boxplot([data], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor='#3498db', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))
        
        # Add scatter points
        y = data
        x = np.random.normal(1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.4, s=30, color='darkblue')
        
        ax.set_ylabel(name, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks([])
        
        # Add statistics text
        stats_text = f"μ={data.mean():.3f}\nσ={data.std():.3f}"
        ax.text(1.15, 0.5, stats_text, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Metric Distributions Across 40 LOSO Folds', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_subject_performance(df, path):
    """Heatmap of metrics per subject"""
    metrics_subset = ['accuracy', 'f1', 'h_recall', 'mdd_recall']
    data_matrix = df[metrics_subset].values.T
    
    fig, ax = plt.subplots(figsize=(max(16, len(df)*0.4), 6))
    
    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_yticks(range(len(metrics_subset)))
    ax.set_yticklabels(['Accuracy', 'F1 Score', 'H Recall', 'MDD Recall'], fontsize=10)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['subject'].values, rotation=45, ha='right', fontsize=8)
    
    ax.set_title('Per-Subject Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(metrics_subset)):
        for j in range(len(df)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_healthy_vs_mdd_folds(df, path):
    """Side-by-side comparison of H folds vs MDD folds"""
    h_folds = df[df['subject'].str.contains('H_')]
    mdd_folds = df[df['subject'].str.contains('MDD_')]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Healthy folds
    ax = axes[0]
    x = range(len(h_folds))
    ax.bar(x, h_folds['accuracy'].values, color='#3498db', alpha=0.8, edgecolor='black')
    ax.axhline(h_folds['accuracy'].mean(), color='red', linestyle='--', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(h_folds['subject'].values, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Healthy Folds (n={len(h_folds)}, μ={h_folds["accuracy"].mean():.3f})', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # MDD folds
    ax = axes[1]
    x = range(len(mdd_folds))
    ax.bar(x, mdd_folds['accuracy'].values, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.axhline(mdd_folds['accuracy'].mean(), color='blue', linestyle='--', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(mdd_folds['subject'].values, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'MDD Folds (n={len(mdd_folds)}, μ={mdd_folds["accuracy"].mean():.3f})', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_matrix(df, path):
    """Correlation matrix of all metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'h_recall', 'mdd_recall']
    corr_matrix = df[metrics].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 10, 'weight': 'bold'})
    
    ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_histogram(df, path):
    """Histogram of accuracy distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    accuracies = df['accuracy'].values
    
    n, bins, patches = ax.hist(accuracies, bins=15, color='#3498db', 
                                alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Color code bins
    mean_acc = accuracies.mean()
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < mean_acc:
            patch.set_facecolor('#e74c3c')  # Red for below mean
        else:
            patch.set_facecolor('#27ae60')  # Green for above mean
    
    ax.axvline(mean_acc, color='black', linestyle='--', linewidth=2.5,
               label=f'Mean = {mean_acc:.3f}')
    ax.axvline(np.median(accuracies), color='blue', linestyle=':', linewidth=2.5,
               label=f'Median = {np.median(accuracies):.3f}')
    
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Folds', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Distribution Across 40 LOSO Folds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fold_timing(fold_times, fold_names, path):
    """Training time per fold"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#3498db' if 'H_' in name else '#e74c3c' for name in fold_names]
    ax.bar(range(len(fold_times)), fold_times, color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(np.mean(fold_times), color='black', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(fold_times):.1f}s')
    
    ax.set_xticks(range(len(fold_names)))
    ax.set_xticklabels(fold_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Per Fold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# ═════════════════════════════════════════════════════════
# TRAIN ONE FOLD
# ═════════════════════════════════════════════════════════
def train_one_fold(train_indices, test_indices, spectrograms, labels, subject_ids, fold_num, subject_name):
    """Train and evaluate model on a single LOSO fold with detailed logging."""
    
    log_msg(f"\nFold {fold_num}/40 — Test: {subject_name}")
    
    # ────────────────────────────────────────────────────────
    # 1. DATA PREPARATION
    # ────────────────────────────────────────────────────────
    t_prep = time.time()
    
    y_train_all = labels[train_indices]
    
    # 90/10 stratified split
    h_idx = np.where(y_train_all == 0)[0]
    mdd_idx = np.where(y_train_all == 1)[0]
    
    val_size_h = max(1, int(0.1 * len(h_idx)))
    val_size_mdd = max(1, int(0.1 * len(mdd_idx)))
    
    val_h_rel = np.random.choice(h_idx, val_size_h, replace=False)
    val_mdd_rel = np.random.choice(mdd_idx, val_size_mdd, replace=False)
    val_rel_idx = np.concatenate([val_h_rel, val_mdd_rel])
    tr_rel_idx = np.setdiff1d(np.arange(len(y_train_all)), val_rel_idx)
    
    tr_indices_final = train_indices[tr_rel_idx]
    val_indices_final = train_indices[val_rel_idx]
    
    # Convert to tensors
    def prep_tensors(indices):
        X = torch.from_numpy(spectrograms[indices]).permute(0, 3, 1, 2).float()
        y = torch.from_numpy(labels[indices]).float()
        return X, y
    
    X_tr, y_tr = prep_tensors(tr_indices_final)
    X_val, y_val = prep_tensors(val_indices_final)
    X_te, y_te = prep_tensors(test_indices)
    
    n_h_train = int((y_tr == 0).sum())
    n_mdd_train = int((y_tr == 1).sum())
    
    log_msg(f"Data prep done: tr={len(X_tr)}, val={len(X_val)} ({time.time()-t_prep:.1f}s)")
    
    # ────────────────────────────────────────────────────────
    # 2. DATALOADERS
    # ────────────────────────────────────────────────────────
    train_dataset = EEGDataset(X_tr, y_tr)
    sampler = BalancedBatchSampler(y_tr.numpy(), BATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(EEGDataset(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)
    
    # Log pos_weight for compatibility (even though we don't use it)
    pos_weight_val = n_h_train / n_mdd_train if n_mdd_train > 0 else 1.0
    log_msg(f"pos_weight={pos_weight_val:.3f} (n_H={n_h_train}, n_MDD={n_mdd_train})")
    
    # ────────────────────────────────────────────────────────
    # 3. MODEL & OPTIMIZER
    # ────────────────────────────────────────────────────────
    model = CNN_LSTM().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ────────────────────────────────────────────────────────
    # 4. TRAINING LOOP
    # ────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr = 0
    best_state = None
    
    train_loss_history = []
    val_loss_history = []
    
    log_msg("Starting training loop...")
    
    for epoch in range(MAX_EPOCHS):
        t_epoch = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        train_loss /= train_steps
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, lbls)
                
                val_loss += loss.item()
                val_steps += 1
        
        val_loss /= val_steps
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        epoch_time = time.time() - t_epoch
        log_msg(f"Ep {epoch+1:2d}/{MAX_EPOCHS} | tr_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | {epoch_time:.1f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log_msg(f"Early stop epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break
    
    # ────────────────────────────────────────────────────────
    # 5. EVALUATION
    # ────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(lbls.numpy())
    
    y_true = np.array(all_labels).astype(int)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_score = 0.0
    
    h_mask = (y_true == 0)
    mdd_mask = (y_true == 1)
    
    h_recall = accuracy_score(y_true[h_mask], y_pred[h_mask]) if h_mask.any() else 0.0
    mdd_recall = accuracy_score(y_true[mdd_mask], y_pred[mdd_mask]) if mdd_mask.any() else 0.0
    
    log_msg(f"acc={acc:.3f}  f1={f1:.3f}  H_rec={h_recall:.3f}  MDD_rec={mdd_recall:.3f}\n")
    
    return {
        'fold': fold_num,
        'subject': subject_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc_score,
        'h_recall': h_recall,
        'mdd_recall': mdd_recall,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'train_loss': train_loss_history,
        'val_loss': val_loss_history
    }

# ═════════════════════════════════════════════════════════
# LOSO FOLD GENERATOR
# ═════════════════════════════════════════════════════════
def get_loso_folds(subject_ids_array, allowed_sids):
    folds = []
    for test_sid in allowed_sids:
        test_mask = (subject_ids_array == test_sid)
        train_mask = np.isin(subject_ids_array, [s for s in allowed_sids if s != test_sid])
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds

# ═════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════
def main():
    print("\n" + "="*75)
    print("  FINAL TRAINING: Full Dataset + Balanced Sampling + Enhanced Viz")
    print("="*75 + "\n")
    
    # Setup
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Clear old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    # Load data
    print("  Loading spectrograms...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    
    print(f"  Total samples: {len(spectrograms):,}")
    print(f"  Healthy: {np.sum(labels==0):,}, MDD: {np.sum(labels==1):,}\n")
    
    # LOSO folds
    sids_all = sorted(SUBJECT_ID_TO_KEY.keys())
    folds = get_loso_folds(subject_ids, sids_all)
    
    # Train all folds
    results = []  # For CSV (metrics only)
    full_results = []  # For visualization (includes y_true/y_prob)
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_accuracies = []
    all_subject_names = []
    all_histories = []
    all_fold_times = []
    
    t_total = time.time()
    
    for i, (train_idx, test_idx) in enumerate(folds):
        test_sid = sids_all[i]
        subject_name = SUBJECT_ID_TO_KEY[test_sid]
        
        t_fold = time.time()
        
        result = train_one_fold(
            train_idx, test_idx,
            spectrograms, labels, subject_ids,
            fold_num=i+1,
            subject_name=subject_name
        )
        
        fold_time = time.time() - t_fold
        result['time_sec'] = fold_time
        
        # Store full result with y_true/y_prob for ROC plotting
        full_results.append(result)
        
        # Store filtered result for CSV
        results.append({k: v for k, v in result.items() if k not in ['y_true', 'y_pred', 'y_prob', 'train_loss', 'val_loss']})
        
        all_y_true.extend(result['y_true'])
        all_y_pred.extend(result['y_pred'])
        all_y_prob.extend(result['y_prob'])
        all_accuracies.append(result['accuracy'])
        all_subject_names.append(subject_name)
        all_fold_times.append(fold_time)
        all_histories.append({'train_loss': result['train_loss'], 'val_loss': result['val_loss']})
    
    total_time = time.time() - t_total
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "fold_results.csv"), index=False)
    
    # Generate all visualizations
    print("\n  Generating visualizations...")
    
    log_msg("\n" + "="*70)
    log_msg("GENERATING VISUALIZATIONS")
    log_msg("="*70)
    
    plot_confusion_matrix(all_y_true, all_y_pred,
                          os.path.join(RESULTS_DIR, "1_confusion_matrix.png"),
                          "Confusion Matrix (40 LOSO Folds Aggregated)")
    log_msg("✓ Confusion matrix")
    
    plot_per_fold_accuracy(all_accuracies, all_subject_names,
                           os.path.join(RESULTS_DIR, "2_per_fold_accuracy.png"),
                           "Per-Fold Accuracy")
    log_msg("✓ Per-fold accuracy")
    
    # Prepare fold results for ROC (use full_results which has y_true/y_prob)
    fold_results_for_roc = [
        {'y_true': r['y_true'], 'y_prob': r['y_prob']} 
        for r in full_results
    ]
    
    plot_roc_curves(all_y_true, all_y_prob, fold_results_for_roc,
                    os.path.join(RESULTS_DIR, "3_roc_pr_curves.png"))
    log_msg("✓ ROC & PR curves")
    
    plot_training_curves(all_histories,
                         os.path.join(RESULTS_DIR, "4_training_curves.png"))
    log_msg("✓ Training curves")
    
    plot_class_performance_comparison(df,
                                      os.path.join(RESULTS_DIR, "5_class_performance.png"))
    log_msg("✓ Class performance comparison")
    
    plot_metric_distributions(df,
                              os.path.join(RESULTS_DIR, "6_metric_distributions.png"))
    log_msg("✓ Metric distributions")
    
    plot_per_subject_performance(df,
                                 os.path.join(RESULTS_DIR, "7_subject_heatmap.png"))
    log_msg("✓ Per-subject heatmap")
    
    plot_healthy_vs_mdd_folds(df,
                              os.path.join(RESULTS_DIR, "8_h_vs_mdd_folds.png"))
    log_msg("✓ Healthy vs MDD folds")
    
    plot_correlation_matrix(df,
                            os.path.join(RESULTS_DIR, "9_correlation_matrix.png"))
    log_msg("✓ Correlation matrix")
    
    plot_accuracy_histogram(df,
                            os.path.join(RESULTS_DIR, "10_accuracy_histogram.png"))
    log_msg("✓ Accuracy histogram")
    
    plot_fold_timing(all_fold_times, all_subject_names,
                     os.path.join(RESULTS_DIR, "11_fold_timing.png"))
    log_msg("✓ Fold timing")
    
    # Summary
    print("\n" + "="*75)
    print("  FINAL RESULTS SUMMARY")
    print("="*75)
    
    summary_table = PrettyTable(['Metric', 'Mean', 'Std Dev', 'Min', 'Max'])
    summary_table.add_row(['Accuracy', f"{df['accuracy'].mean():.3f}", f"{df['accuracy'].std():.3f}", f"{df['accuracy'].min():.3f}", f"{df['accuracy'].max():.3f}"])
    summary_table.add_row(['F1 Score', f"{df['f1'].mean():.3f}", f"{df['f1'].std():.3f}", f"{df['f1'].min():.3f}", f"{df['f1'].max():.3f}"])
    summary_table.add_row(['H Recall', f"{df['h_recall'].mean():.3f}", f"{df['h_recall'].std():.3f}", f"{df['h_recall'].min():.3f}", f"{df['h_recall'].max():.3f}"])
    summary_table.add_row(['MDD Recall', f"{df['mdd_recall'].mean():.3f}", f"{df['mdd_recall'].std():.3f}", f"{df['mdd_recall'].min():.3f}", f"{df['mdd_recall'].max():.3f}"])
    
    print(summary_table)
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"  Results saved to: {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()