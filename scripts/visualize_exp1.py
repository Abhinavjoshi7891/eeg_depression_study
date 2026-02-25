"""
visualize_exp1.py
─────────────────
Generate comprehensive visualizations for Experiment 1 results.

Outputs saved to: results/exp1_seq_lstm/figures/

Figures:
  1. Per-subject bar chart (Sensitivity/Specificity per fold)
  2. Confusion matrix (global, aggregated)
  3. Class-wise box plots
  4. Training loss curves (sampled folds)
  5. Threshold distribution histogram
  6. Comparison: Baseline vs Exp 1

HOW TO RUN:
    conda activate eeg_train
    python scripts/visualize_exp1.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "exp1_seq_lstm")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LOG_FILE = os.path.join(RESULTS_DIR, "training_log.txt")
CSV_FILE = os.path.join(RESULTS_DIR, "fold_results.csv")

# ─────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
df['class'] = df['subject'].apply(lambda s: 'Healthy' if s.startswith('H_') else 'MDD')

# For each fold, the "key metric" is specificity (for H) or sensitivity (for MDD)
df['key_metric'] = df.apply(
    lambda r: r['specificity'] if r['class'] == 'Healthy' else r['sensitivity'], axis=1
)

# ─────────────────────────────────────────────────────────────────────────
# Figure 1: Per-Subject Performance Bar Chart
# ─────────────────────────────────────────────────────────────────────────
print("Generating Figure 1: Per-subject bar chart...")
fig, ax = plt.subplots(figsize=(18, 7))

colors = ['#2ecc71' if c == 'Healthy' else '#e74c3c' for c in df['class']]
bars = ax.bar(range(len(df)), df['key_metric'], color=colors, edgecolor='white', linewidth=0.5)

# Highlight outliers
for i, row in df.iterrows():
    if row['key_metric'] < 0.2:
        ax.annotate(row['subject'], (i, row['key_metric'] + 0.02),
                    ha='center', fontsize=7, fontweight='bold', color='red', rotation=45)

ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['subject'], rotation=90, fontsize=7)
ax.set_ylabel('Performance (Specificity for H / Sensitivity for MDD)', fontsize=11)
ax.set_title('Experiment 1: Per-Subject LOSO Performance', fontsize=14, fontweight='bold')
ax.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='Target (75%)')
ax.axhline(y=df['key_metric'].mean(), color='blue', linestyle=':', alpha=0.7,
           label=f'Mean ({df["key_metric"].mean():.1%})')

h_patch = mpatches.Patch(color='#2ecc71', label='Healthy (Specificity)')
m_patch = mpatches.Patch(color='#e74c3c', label='MDD (Sensitivity)')
ax.legend(handles=[h_patch, m_patch, ax.lines[0], ax.lines[1]], loc='lower right')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_per_subject_performance.png"), dpi=150)
plt.close()
print("  Saved: 01_per_subject_performance.png")

# ─────────────────────────────────────────────────────────────────────────
# Figure 2: Global Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────
print("Generating Figure 2: Global confusion matrix...")

# Parse from training log
with open(LOG_FILE, 'r') as f:
    log_text = f.read()

# Find the global confusion matrix
cm_match = re.search(r'Global Confusion Matrix:\n\[\[(\d+)\s+(\d+)\]\n\s*\[(\d+)\s+(\d+)\]\]', log_text)
if cm_match:
    tn, fp, fn, tp = int(cm_match.group(1)), int(cm_match.group(2)), \
                      int(cm_match.group(3)), int(cm_match.group(4))
    cm = np.array([[tn, fp], [fn, tp]])
else:
    # Manually compute from fold results
    # For H folds: all samples are H (label=0), key metric is specificity
    # For MDD folds: all samples are MDD (label=1), key metric is sensitivity
    # This is approximate
    print("  Warning: Could not parse CM from log, computing approximate values...")
    tn = fp = fn = tp = 0
    for _, row in df.iterrows():
        if row['class'] == 'Healthy':
            # Test set is all healthy
            n_test = 150  # approx
            tn += int(row['specificity'] * n_test)
            fp += int((1 - row['specificity']) * n_test)
        else:
            n_test = 160  # approx
            tp += int(row['sensitivity'] * n_test)
            fn += int((1 - row['sensitivity']) * n_test)
    cm = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred Healthy', 'Pred MDD'],
            yticklabels=['True Healthy', 'True MDD'],
            annot_kws={'size': 16})
ax.set_title('Experiment 1: Global Confusion Matrix (All 40 Folds)', fontsize=13, fontweight='bold')

# Add metrics as text
total = cm.sum()
acc = (cm[0, 0] + cm[1, 1]) / total
sens = cm[1, 1] / (cm[1, 0] + cm[1, 1])
spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
ax.text(0.5, -0.12, f'Accuracy: {acc:.1%}  |  Sensitivity: {sens:.1%}  |  Specificity: {spec:.1%}',
        transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_confusion_matrix.png"), dpi=150)
plt.close()
print("  Saved: 02_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────
# Figure 3: Class-wise Box Plots
# ─────────────────────────────────────────────────────────────────────────
print("Generating Figure 3: Class-wise box plots...")

h_df = df[df['class'] == 'Healthy']
m_df = df[df['class'] == 'MDD']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Healthy specificity distribution
ax1.boxplot(h_df['specificity'].values, widths=0.5, patch_artist=True,
            boxprops=dict(facecolor='#2ecc71', alpha=0.6))
ax1.scatter(np.ones(len(h_df)), h_df['specificity'].values, alpha=0.7, color='#27ae60', zorder=3)
for _, row in h_df.iterrows():
    if row['specificity'] < 0.1:
        ax1.annotate(row['subject'], (1.05, row['specificity']), fontsize=8, color='red')
ax1.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='Target (75%)')
ax1.set_title(f'Healthy Subjects: Specificity\n(Mean={h_df["specificity"].mean():.1%}, Median={h_df["specificity"].median():.1%})',
              fontsize=11, fontweight='bold')
ax1.set_ylabel('Specificity')
ax1.set_ylim(-0.05, 1.05)
ax1.legend()

# MDD sensitivity distribution
ax2.boxplot(m_df['sensitivity'].values, widths=0.5, patch_artist=True,
            boxprops=dict(facecolor='#e74c3c', alpha=0.6))
ax2.scatter(np.ones(len(m_df)), m_df['sensitivity'].values, alpha=0.7, color='#c0392b', zorder=3)
for _, row in m_df.iterrows():
    if row['sensitivity'] < 0.2:
        ax2.annotate(row['subject'], (1.05, row['sensitivity']), fontsize=8, color='red')
ax2.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='Target (85%)')
ax2.set_title(f'MDD Subjects: Sensitivity\n(Mean={m_df["sensitivity"].mean():.1%}, Median={m_df["sensitivity"].median():.1%})',
              fontsize=11, fontweight='bold')
ax2.set_ylabel('Sensitivity')
ax2.set_ylim(-0.05, 1.05)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_class_boxplots.png"), dpi=150)
plt.close()
print("  Saved: 03_class_boxplots.png")

# ─────────────────────────────────────────────────────────────────────────
# Figure 4: Training Loss Curves (sampled folds)
# ─────────────────────────────────────────────────────────────────────────
print("Generating Figure 4: Training loss curves...")

# Parse training/validation losses per fold from the log
fold_losses = {}
current_fold = None
current_subject = None

for line in log_text.split('\n'):
    fold_match = re.search(r'FOLD (\d+)/40 — Test Subject: (\S+)', line)
    if fold_match:
        current_fold = int(fold_match.group(1))
        current_subject = fold_match.group(2)
        fold_losses[current_subject] = {'train': [], 'val': []}
        continue

    epoch_match = re.search(r'Epoch\s+\d+/50 \| train=(\d+\.\d+) \| val=(\d+\.\d+)', line)
    if epoch_match and current_subject:
        fold_losses[current_subject]['train'].append(float(epoch_match.group(1)))
        fold_losses[current_subject]['val'].append(float(epoch_match.group(2)))

# Pick representative folds: 1 good H, 1 bad H, 1 good MDD, 1 bad MDD
sample_subjects = ['H_19', 'H_16', 'MDD_1', 'MDD_19']
sample_subjects = [s for s in sample_subjects if s in fold_losses]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, subj in enumerate(sample_subjects):
    ax = axes[i]
    data = fold_losses[subj]
    epochs = range(1, len(data['train']) + 1)
    ax.plot(epochs, data['train'], label='Train Loss', color='#3498db', linewidth=2)
    ax.plot(epochs, data['val'], label='Val Loss', color='#e74c3c', linewidth=2, alpha=0.8)
    
    # Get this subject's performance
    row = df[df['subject'] == subj].iloc[0]
    metric_name = 'Specificity' if subj.startswith('H_') else 'Sensitivity'
    metric_val = row['specificity'] if subj.startswith('H_') else row['sensitivity']
    status = '✓' if metric_val > 0.5 else '✗'
    
    ax.set_title(f'{subj} ({metric_name}: {metric_val:.1%} {status})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Training/Validation Loss Curves (Representative Folds)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_loss_curves.png"), dpi=150)
plt.close()
print("  Saved: 04_loss_curves.png")

# ─────────────────────────────────────────────────────────────────────────
# Figure 5: Threshold Distribution
# ─────────────────────────────────────────────────────────────────────────
print("Generating Figure 5: Threshold distribution...")

fig, ax = plt.subplots(figsize=(8, 5))
thresholds = df['threshold'].values

ax.hist(thresholds, bins=np.arange(0.35, 0.85, 0.05), color='#9b59b6', edgecolor='white',
        alpha=0.8, rwidth=0.85)
ax.axvline(x=np.mean(thresholds), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(thresholds):.2f}')
ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, label='Default (0.50)')
ax.set_xlabel('Optimal Threshold', fontsize=12)
ax.set_ylabel('Number of Folds', fontsize=12)
ax.set_title('Per-Fold Optimal Threshold Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_threshold_distribution.png"), dpi=150)
plt.close()
print("  Saved: 05_threshold_distribution.png")

# ─────────────────────────────────────────────────────────────────────────
# Figure 6: Baseline vs Experiment 1 Comparison
# ─────────────────────────────────────────────────────────────────────────
print("Generating Figure 6: Baseline vs Exp 1 comparison...")

# Baseline numbers (from diagnostic_report.md / previous analysis)
baseline = {
    'Accuracy': 0.68,
    'Sensitivity': 0.756,
    'Specificity': 0.564,
    'F1': 0.72,
}
exp1 = {
    'Accuracy': 0.805,
    'Sensitivity': 0.901,
    'Specificity': 0.636,
    'F1': 0.855,
}
targets = {
    'Accuracy': 0.75,
    'Sensitivity': 0.85,
    'Specificity': 0.75,
    'F1': None,
}

metrics = list(baseline.keys())
x = np.arange(len(metrics))
width = 0.3

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, [baseline[m] for m in metrics], width, label='Baseline (seq=1)',
               color='#95a5a6', edgecolor='white')
bars2 = ax.bar(x, [exp1[m] for m in metrics], width, label='Exp 1 (seq=10)',
               color='#3498db', edgecolor='white')

# Target markers
for i, m in enumerate(metrics):
    if targets[m] is not None:
        ax.plot([i - width*1.5, i + width*1.5], [targets[m], targets[m]],
                color='#e74c3c', linewidth=2, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Baseline vs Experiment 1: LOSO Cross-Validation Results',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.1)

target_line = plt.Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=2)
ax.legend(handles=[bars1, bars2, target_line],
          labels=['Baseline (seq=1)', 'Exp 1 (seq=10)', 'Target'],
          loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_baseline_vs_exp1.png"), dpi=150)
plt.close()
print("  Saved: 06_baseline_vs_exp1.png")

# ─────────────────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL FIGURES SAVED TO:", FIG_DIR)
print("=" * 60)
print(f"\nFiles generated:")
for f in sorted(os.listdir(FIG_DIR)):
    fpath = os.path.join(FIG_DIR, f)
    print(f"  {f} ({os.path.getsize(fpath)//1024} KB)")
