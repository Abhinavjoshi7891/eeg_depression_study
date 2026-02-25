import pandas as pd
import numpy as np

def get_stats(csv_path, subjects=None):
    df = pd.read_csv(csv_path)
    if subjects is not None:
        df = df[df['subject'].isin(subjects)]
    
    h = df[df['subject'].str.startswith('H_')]['specificity']
    mdd = df[df['subject'].str.startswith('MDD_')]['sensitivity']
    
    return {
        'h_median': h.median(),
        'mdd_median': mdd.median(),
        'h_mean': h.mean(),
        'mdd_mean': mdd.mean()
    }

# Get current subjects
df_final = pd.read_csv('results/exp_final/fold_results.csv')
current_subjects = df_final['subject'].tolist()

final_stats = get_stats('results/exp_final/fold_results.csv')
exp1_stats = get_stats('results/exp1_seq_lstm/fold_results.csv', subjects=current_subjects)

print(f"{'Metric':<25} | {'Exp 1':<10} | {'Final Exp':<10} | {'Diff'}")
print("-" * 60)
print(f"{'Median Specificity (H)':<25} | {exp1_stats['h_median']:<10.4f} | {final_stats['h_median']:<10.4f} | {final_stats['h_median'] - exp1_stats['h_median']:+.4f}")
print(f"{'Median Sensitivity (MDD)':<25} | {exp1_stats['mdd_median']:<10.4f} | {final_stats['mdd_median']:<10.4f} | {final_stats['mdd_median'] - exp1_stats['mdd_median']:+.4f}")
print(f"{'Mean Specificity (H)':<25} | {exp1_stats['h_mean']:<10.4f} | {final_stats['h_mean']:<10.4f} | {final_stats['h_mean'] - exp1_stats['h_mean']:+.4f}")
print(f"{'Mean Sensitivity (MDD)':<25} | {exp1_stats['mdd_mean']:<10.4f} | {final_stats['mdd_mean']:<10.4f} | {final_stats['mdd_mean'] - exp1_stats['mdd_mean']:+.4f}")
