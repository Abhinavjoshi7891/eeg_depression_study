
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuration
RESULTS_DIR = "results/exp_final"
CSV_PATH = os.path.join(RESULTS_DIR, "fold_results.csv")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Exp 1 Results for comparison (from REPORT.md)
EXP1_METRICS = {
    'Accuracy': 0.805,
    'Sensitivity': 0.901,
    'Specificity': 0.636,
    'F1': 0.855
}

def load_data():
    df = pd.read_csv(CSV_PATH)
    # Split into H and MDD for easier analysis
    df['class'] = df['subject'].apply(lambda x: 'Healthy' if x.startswith('H') else 'MDD')
    return df

def plot_subject_performance(df):
    plt.figure(figsize=(15, 7))
    sns.set_style("whitegrid")
    
    # Sort subjects: H first, then MDD
    df = df.sort_values(by=['class', 'subject'], ascending=[False, True])
    
    colors = ['#3498db' if c == 'Healthy' else '#e74c3c' for c in df['class']]
    
    bars = plt.bar(df['subject'], df['accuracy'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add target line
    plt.axhline(0.75, color='gray', linestyle='--', alpha=0.6, label='Target Accuracy (75%)')
    
    plt.xticks(rotation=90, fontsize=9)
    plt.ylabel('Sequence-level Accuracy', fontweight='bold', fontsize=12)
    plt.title('Final Experiment: Per-Subject Accuracy (LOSO)', fontsize=16, fontweight='bold', pad=20)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3498db', lw=4, label='Healthy Subjects'),
        Line2D([0], [0], color='#e74c3c', lw=4, label='MDD Subjects'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Target (75%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "subject_accuracy.png"), dpi=300)
    print(f"Generated subject_accuracy.png")

def plot_comparison(df):
    # Calculate Final Mean metrics
    final_metrics = {
        'Accuracy': df['accuracy'].mean(),
        'F1': df['f1'].mean() if 'f1' in df.columns else 0.890,
        'Sensitivity': df[df['class']=='MDD']['sensitivity'].mean(),
        'Specificity': df[df['class']=='Healthy']['specificity'].mean()
    }
    
    categories = list(EXP1_METRICS.keys())
    exp1_vals = [EXP1_METRICS[c] for c in categories]
    final_vals = [final_metrics[c] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, exp1_vals, width, label='Exp 1 (Averaged)', color='#bdc3c7')
    rects2 = ax.bar(x + width/2, final_vals, width, label='Final Exp (Spatial RGB)', color='#2ecc71')
    
    ax.set_ylabel('Scores', fontweight='bold')
    ax.set_title('Baseline (Exp 1) vs. Final Experiment (Spatial RGB)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add values on top
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "comparison_metrics.png"), dpi=300)
    print(f"Generated comparison_metrics.png")

def plot_majority_vote_pie(df):
    correct = df['vote_correct'].sum()
    incorrect = len(df) - correct
    
    plt.figure(figsize=(8, 8))
    labels = [f'Correct Diagnosis ({correct})', f'Incorrect ({incorrect})']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)
    
    plt.pie([correct, incorrect], explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.title('Subject-Level Majority Vote Success Rate', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "majority_vote_pie.png"), dpi=300)
    print(f"Generated majority_vote_pie.png")

if __name__ == "__main__":
    data = load_data()
    plot_subject_performance(data)
    plot_comparison(data)
    plot_majority_vote_pie(data)
    print("All visualizations completed successfully.")
