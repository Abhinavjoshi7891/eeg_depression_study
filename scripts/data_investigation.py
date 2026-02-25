"""
data_investigation.py — Deep Analysis of EEG Spectrogram Dataset
═════════════════════════════════════════════════════════════════
Investigates the root cause of poor Healthy classification by analyzing:
1. Label distribution and verification
2. Spectrogram statistics per class
3. Visual comparison of H vs MDD spectrograms
4. Per-subject sample counts
5. Intensity histograms
6. Class-averaged spectrograms
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "spectrograms")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "results", "data_investigation")

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

def main():
    print("\n" + "="*70)
    print("  DATA INVESTIGATION: EEG Spectrogram Analysis")
    print("="*70 + "\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # 1. LOAD DATA
    # ═══════════════════════════════════════════════════════════
    print("Loading data...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    
    print(f"  Spectrograms shape: {spectrograms.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Subject IDs shape: {subject_ids.shape}")
    print(f"  Data type: {spectrograms.dtype}")
    print(f"  Value range: [{spectrograms.min():.4f}, {spectrograms.max():.4f}]")
    
    # ═══════════════════════════════════════════════════════════
    # 2. LABEL VERIFICATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─"*70)
    print("LABEL VERIFICATION")
    print("─"*70)
    
    h_mask = (labels == 0)
    mdd_mask = (labels == 1)
    
    print(f"\n  Label 0 (Healthy): {h_mask.sum():,} samples")
    print(f"  Label 1 (MDD):     {mdd_mask.sum():,} samples")
    print(f"  Ratio H:MDD = 1:{mdd_mask.sum()/h_mask.sum():.2f}")
    
    # Verify labels match subject prefixes
    print("\n  Verifying labels match subject names...")
    mismatches = 0
    for sid in np.unique(subject_ids):
        subj_name = SUBJECT_ID_TO_KEY.get(sid, f"Unknown_{sid}")
        subj_mask = (subject_ids == sid)
        subj_labels = labels[subj_mask]
        
        expected_label = 0 if subj_name.startswith("H_") else 1
        actual_label = int(subj_labels[0])
        
        if expected_label != actual_label:
            print(f"    ⚠️ MISMATCH: {subj_name} has label {actual_label} (expected {expected_label})")
            mismatches += 1
    
    if mismatches == 0:
        print("    ✓ All labels verified correctly!")
    
    # ═══════════════════════════════════════════════════════════
    # 3. PER-SUBJECT SAMPLE COUNTS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─"*70)
    print("PER-SUBJECT SAMPLE COUNTS")
    print("─"*70)
    
    h_counts = []
    mdd_counts = []
    
    print("\n  Healthy subjects:")
    for sid in sorted([s for s in np.unique(subject_ids) if SUBJECT_ID_TO_KEY.get(s, "").startswith("H_")]):
        count = (subject_ids == sid).sum()
        h_counts.append(count)
        print(f"    {SUBJECT_ID_TO_KEY[sid]}: {count:,} samples")
    
    print("\n  MDD subjects:")
    for sid in sorted([s for s in np.unique(subject_ids) if SUBJECT_ID_TO_KEY.get(s, "").startswith("MDD_")]):
        count = (subject_ids == sid).sum()
        mdd_counts.append(count)
        print(f"    {SUBJECT_ID_TO_KEY[sid]}: {count:,} samples")
    
    print(f"\n  Healthy: {len(h_counts)} subjects, avg {np.mean(h_counts):.0f} samples/subject")
    print(f"  MDD:     {len(mdd_counts)} subjects, avg {np.mean(mdd_counts):.0f} samples/subject")
    
    # ═══════════════════════════════════════════════════════════
    # 4. SPECTROGRAM STATISTICS BY CLASS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─"*70)
    print("SPECTROGRAM STATISTICS BY CLASS")
    print("─"*70)
    
    h_specs = spectrograms[h_mask]
    mdd_specs = spectrograms[mdd_mask]
    
    print("\n  Healthy spectrograms:")
    print(f"    Mean: {h_specs.mean():.6f}")
    print(f"    Std:  {h_specs.std():.6f}")
    print(f"    Min:  {h_specs.min():.6f}")
    print(f"    Max:  {h_specs.max():.6f}")
    
    print("\n  MDD spectrograms:")
    print(f"    Mean: {mdd_specs.mean():.6f}")
    print(f"    Std:  {mdd_specs.std():.6f}")
    print(f"    Min:  {mdd_specs.min():.6f}")
    print(f"    Max:  {mdd_specs.max():.6f}")
    
    print("\n  Difference (MDD - H):")
    print(f"    Mean diff: {mdd_specs.mean() - h_specs.mean():.6f}")
    print(f"    Std diff:  {mdd_specs.std() - h_specs.std():.6f}")
    
    # Per-channel statistics
    print("\n  Per-channel analysis:")
    for ch in range(spectrograms.shape[-1]):
        h_ch = h_specs[:, :, :, ch]
        mdd_ch = mdd_specs[:, :, :, ch]
        print(f"    Channel {ch}: H_mean={h_ch.mean():.4f}, MDD_mean={mdd_ch.mean():.4f}, diff={mdd_ch.mean()-h_ch.mean():.4f}")
    
    # ═══════════════════════════════════════════════════════════
    # 5. PER-SUBJECT STATISTICS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─"*70)
    print("PER-SUBJECT MEAN INTENSITY")
    print("─"*70)
    
    subject_means = {}
    for sid in np.unique(subject_ids):
        subj_mask = (subject_ids == sid)
        subj_mean = spectrograms[subj_mask].mean()
        subject_means[SUBJECT_ID_TO_KEY.get(sid, f"Unknown_{sid}")] = subj_mean
    
    h_subject_means = [v for k, v in subject_means.items() if k.startswith("H_")]
    mdd_subject_means = [v for k, v in subject_means.items() if k.startswith("MDD_")]
    
    print(f"\n  Healthy subjects: mean of means = {np.mean(h_subject_means):.6f} (std={np.std(h_subject_means):.6f})")
    print(f"  MDD subjects:     mean of means = {np.mean(mdd_subject_means):.6f} (std={np.std(mdd_subject_means):.6f})")
    
    # ═══════════════════════════════════════════════════════════
    # 6. VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─"*70)
    print("GENERATING VISUALIZATIONS")
    print("─"*70)
    
    # 6a. Class-averaged spectrograms
    print("\n  Creating class-averaged spectrograms...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    h_avg = h_specs.mean(axis=0)
    mdd_avg = mdd_specs.mean(axis=0)
    diff = mdd_avg - h_avg
    
    for ch in range(3):
        axes[0, ch].imshow(h_avg[:, :, ch], aspect='auto', cmap='viridis')
        axes[0, ch].set_title(f'Healthy Avg - Ch {ch}', fontsize=12, fontweight='bold')
        axes[0, ch].set_ylabel('Frequency')
        
        axes[1, ch].imshow(mdd_avg[:, :, ch], aspect='auto', cmap='viridis')
        axes[1, ch].set_title(f'MDD Avg - Ch {ch}', fontsize=12, fontweight='bold')
        axes[1, ch].set_xlabel('Time')
        axes[1, ch].set_ylabel('Frequency')
    
    plt.suptitle('Class-Averaged Spectrograms (H vs MDD)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_class_averaged_spectrograms.png"), dpi=300)
    plt.close()
    print("    ✓ Saved: 1_class_averaged_spectrograms.png")
    
    # 6b. Difference map
    print("  Creating difference map...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ch in range(3):
        im = axes[ch].imshow(diff[:, :, ch], aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[ch].set_title(f'Diff (MDD-H) - Ch {ch}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[ch])
    
    plt.suptitle('Difference Map: MDD minus Healthy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_difference_map.png"), dpi=300)
    plt.close()
    print("    ✓ Saved: 2_difference_map.png")
    
    # 6c. Intensity histograms
    print("  Creating intensity histograms...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ch in range(3):
        axes[ch].hist(h_specs[:, :, :, ch].flatten(), bins=100, alpha=0.5, label='Healthy', color='blue', density=True)
        axes[ch].hist(mdd_specs[:, :, :, ch].flatten(), bins=100, alpha=0.5, label='MDD', color='red', density=True)
        axes[ch].set_title(f'Channel {ch} Intensity Distribution', fontsize=12, fontweight='bold')
        axes[ch].set_xlabel('Intensity')
        axes[ch].set_ylabel('Density')
        axes[ch].legend()
    
    plt.suptitle('Pixel Intensity Histograms: H vs MDD', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_intensity_histograms.png"), dpi=300)
    plt.close()
    print("    ✓ Saved: 3_intensity_histograms.png")
    
    # 6d. Per-subject mean intensity bar plot
    print("  Creating per-subject mean intensity plot...")
    fig, ax = plt.subplots(figsize=(16, 6))
    
    sorted_subjects = sorted(subject_means.items(), key=lambda x: (not x[0].startswith("H_"), x[0]))
    names = [x[0] for x in sorted_subjects]
    values = [x[1] for x in sorted_subjects]
    colors = ['#3498db' if n.startswith("H_") else '#e74c3c' for n in names]
    
    ax.bar(range(len(names)), values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Spectrogram Intensity', fontsize=12, fontweight='bold')
    ax.set_title('Per-Subject Mean Intensity (Blue=Healthy, Red=MDD)', fontsize=14, fontweight='bold')
    ax.axhline(np.mean(h_subject_means), color='blue', linestyle='--', label=f'H mean={np.mean(h_subject_means):.4f}')
    ax.axhline(np.mean(mdd_subject_means), color='red', linestyle='--', label=f'MDD mean={np.mean(mdd_subject_means):.4f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_per_subject_intensity.png"), dpi=300)
    plt.close()
    print("    ✓ Saved: 4_per_subject_intensity.png")
    
    # 6e. Sample spectrograms
    print("  Creating sample spectrograms...")
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    
    # Random H samples
    h_indices = np.where(h_mask)[0]
    for col in range(6):
        idx = np.random.choice(h_indices)
        for ch in range(3):
            if col < 3:
                axes[0, col].imshow(spectrograms[idx, :, :, col], aspect='auto', cmap='viridis')
                axes[0, col].set_title(f'H Sample - Ch {col}', fontsize=10)
    
    # Random MDD samples  
    mdd_indices = np.where(mdd_mask)[0]
    for col in range(6):
        idx = np.random.choice(mdd_indices)
        for ch in range(3):
            if col < 3:
                axes[2, col].imshow(spectrograms[idx, :, :, col], aspect='auto', cmap='viridis')
                axes[2, col].set_title(f'MDD Sample - Ch {col}', fontsize=10)
    
    # Show 3 random H and 3 random MDD side by side
    for i, idx in enumerate(np.random.choice(h_indices, 3)):
        axes[1, i].imshow(spectrograms[idx, :, :, 0], aspect='auto', cmap='viridis')
        axes[1, i].set_title(f'H_{i+1}', fontsize=10)
    
    for i, idx in enumerate(np.random.choice(mdd_indices, 3)):
        axes[1, i+3].imshow(spectrograms[idx, :, :, 0], aspect='auto', cmap='viridis')
        axes[1, i+3].set_title(f'MDD_{i+1}', fontsize=10)
    
    for i, idx in enumerate(np.random.choice(h_indices, 3)):
        axes[3, i].imshow(spectrograms[idx, :, :, 0], aspect='auto', cmap='viridis')
        axes[3, i].set_title(f'H_{i+4}', fontsize=10)
    
    for i, idx in enumerate(np.random.choice(mdd_indices, 3)):
        axes[3, i+3].imshow(spectrograms[idx, :, :, 0], aspect='auto', cmap='viridis')
        axes[3, i+3].set_title(f'MDD_{i+4}', fontsize=10)
    
    plt.suptitle('Random Sample Spectrograms (Channel 0)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_sample_spectrograms.png"), dpi=300)
    plt.close()
    print("    ✓ Saved: 5_sample_spectrograms.png")
    
    # 6f. Subject variance analysis
    print("  Creating subject variance analysis...")
    
    subject_stds = {}
    for sid in np.unique(subject_ids):
        subj_mask = (subject_ids == sid)
        subj_std = spectrograms[subj_mask].std()
        subject_stds[SUBJECT_ID_TO_KEY.get(sid, f"Unknown_{sid}")] = subj_std
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    sorted_subjects = sorted(subject_stds.items(), key=lambda x: (not x[0].startswith("H_"), x[0]))
    names = [x[0] for x in sorted_subjects]
    values = [x[1] for x in sorted_subjects]
    colors = ['#3498db' if n.startswith("H_") else '#e74c3c' for n in names]
    
    ax.bar(range(len(names)), values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Spectrogram Std Dev', fontsize=12, fontweight='bold')
    ax.set_title('Per-Subject Variance (Blue=Healthy, Red=MDD)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    h_stds = [v for k, v in subject_stds.items() if k.startswith("H_")]
    mdd_stds = [v for k, v in subject_stds.items() if k.startswith("MDD_")]
    ax.axhline(np.mean(h_stds), color='blue', linestyle='--', label=f'H mean std={np.mean(h_stds):.4f}')
    ax.axhline(np.mean(mdd_stds), color='red', linestyle='--', label=f'MDD mean std={np.mean(mdd_stds):.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_per_subject_variance.png"), dpi=300)
    plt.close()
    print("    ✓ Saved: 6_per_subject_variance.png")
    
    # ═══════════════════════════════════════════════════════════
    # 7. SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  INVESTIGATION SUMMARY")
    print("="*70)
    
    print(f"""
    Dataset Overview:
    ─────────────────
    Total samples: {len(labels):,}
    Healthy: {h_mask.sum():,} ({100*h_mask.sum()/len(labels):.1f}%)
    MDD: {mdd_mask.sum():,} ({100*mdd_mask.sum()/len(labels):.1f}%)
    
    Subject Counts:
    ───────────────
    Healthy subjects: {len([s for s in SUBJECT_ID_TO_KEY.values() if s.startswith('H_')])}
    MDD subjects: {len([s for s in SUBJECT_ID_TO_KEY.values() if s.startswith('MDD_')])}
    
    Class Statistics:
    ─────────────────
    H mean: {h_specs.mean():.6f}
    MDD mean: {mdd_specs.mean():.6f}
    Difference: {mdd_specs.mean() - h_specs.mean():.6f}
    
    H std: {h_specs.std():.6f}
    MDD std: {mdd_specs.std():.6f}
    
    Per-Subject Variance:
    ─────────────────────
    H subjects std (avg): {np.mean(h_stds):.6f}
    MDD subjects std (avg): {np.mean(mdd_stds):.6f}
    
    Results saved to: {OUTPUT_DIR}
    """)
    
    # Save summary to file
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("DATA INVESTIGATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total samples: {len(labels):,}\n")
        f.write(f"Healthy: {h_mask.sum():,} ({100*h_mask.sum()/len(labels):.1f}%)\n")
        f.write(f"MDD: {mdd_mask.sum():,} ({100*mdd_mask.sum()/len(labels):.1f}%)\n\n")
        f.write(f"H mean: {h_specs.mean():.6f}\n")
        f.write(f"MDD mean: {mdd_specs.mean():.6f}\n")
        f.write(f"H std: {h_specs.std():.6f}\n")
        f.write(f"MDD std: {mdd_specs.std():.6f}\n")
    
    print("    ✓ Saved: summary.txt")

if __name__ == "__main__":
    main()
