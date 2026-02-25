"""
generate_report.py
──────────────────
Reads the existing 'results/approach1/summary.csv' and generates:
1. A detailed REPORT.md
2. High-quality plots (Confusion Matrix & Accuracy Bar Chart)
3. A console summary table

Usage: python generate_report.py
"""

import os, sys, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix

# CONFIG
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results", "approach1")
CSV_PATH     = os.path.join(RESULTS_DIR, "summary.csv")

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

def generate():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}")
        return

    print("Loading results...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Markdown Report
    report_path = os.path.join(RESULTS_DIR, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("# 📊 Analysis Report: Approach 1 (LOSO)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Total Subjects:** {len(df)} (40)\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write("| Metric | Mean | Std Dev | Description |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Accuracy** | **{df['accuracy'].mean():.3f}** | {df['accuracy'].std():.3f} | Overall correctness |\n")
        f.write(f"| Precision | {df['precision'].mean():.3f} | {df['precision'].std():.3f} | False Positive rate proxy |\n")
        f.write(f"| Recall | {df['recall'].mean():.3f} | {df['recall'].std():.3f} | Ability to find MDD |\n")
        f.write(f"| F1 Score | {df['f1'].mean():.3f} | {df['f1'].std():.3f} | Balance of Prec/Rec |\n")
        f.write("\n")

        f.write("## 2. Per-Class Performance\n")
        # Handle cases where h_recall might be missing or 0 for MDD folds
        h_rec_mean = df[df['test_subject'].str.startswith('H')]['accuracy'].mean()
        m_rec_mean = df[df['test_subject'].str.startswith('MDD')]['accuracy'].mean()

        f.write(f"- **Healthy Recall (Specificity):** `{h_rec_mean:.1%}` (Ability to identify healthy people)\n")
        f.write(f"- **MDD Recall (Sensitivity):** `{m_rec_mean:.1%}` (Ability to detect depression)\n\n")
        
        f.write("## 3. High Variance Subjects\n")
        f.write("Subjects where the model struggled significantly (Accuracy < 50%):\n")
        struggles = df[df['accuracy'] < 0.50]
        if not struggles.empty:
            f.write(struggles[['test_subject', 'accuracy']].to_markdown(index=False))
        else:
            f.write("None.")
        f.write("\n\n")
    
    print(f"    Saved Report: {os.path.basename(report_path)}")

    # 2. Plots
    # For CM, we need individual predictions. We don't have them in summary.csv.
    # We can mostly reconstruct the Acc plot.
    plot_acc(df['accuracy'], df['test_subject'], os.path.join(RESULTS_DIR, "accuracy_plot_HD.png"),
             "Approach 1 — Per-Fold Accuracy (High Def)")

    # 3. Console
    t = PrettyTable(['Metric', 'Mean', 'Std'])
    t.add_row(['Accuracy', f"{df['accuracy'].mean():.3f}", f"{df['accuracy'].std():.3f}"])
    t.add_row(['Specificity (H)', f"{h_rec_mean:.3f}", "-"])
    t.add_row(['Sensitivity (MDD)', f"{m_rec_mean:.3f}", "-"])
    print("\n" + str(t))

if __name__ == "__main__":
    generate()
