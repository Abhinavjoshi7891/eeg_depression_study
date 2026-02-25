"""
inspect_outliers_v2.py
──────────────────────
Deeper spectral analysis of Healthy outliers (H_16, H_24, H_27)
to see if their frequency distribution mimics MDD subjects.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/spectrograms"
OUTPUT_DIR = "results/outlier_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subject mapping
TARGETS = {
    1: "H_16 (Outlier)",
    5: "H_24 (Outlier)",
    7: "H_27 (Outlier)",
    2: "H_19 (Good H)",
    8: "H_28 (Good H)",
    14: "MDD_1 (MDD)"
}

def analyze():
    print("Loading data...")
    spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    subject_ids = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))

    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")

    for sid, name in TARGETS.items():
        mask = (subject_ids == sid)
        sub_data = spectrograms[mask] # (N, 224, 224, 3)
        
        if len(sub_data) == 0:
            print(f"Warning: No data for {name}")
            continue

        # Convert back to single channel (they are replicated anyway)
        # Scalogram row index 0 is high freq, index 223 is low freq (or vice versa depending on origin)
        # In preprocess.py, origin='lower' was used for imshow.
        # Let's just look at the 1D frequency profile (average across time and samples)
        
        # mean across all samples (N), then mean across time (axis 2)
        # result: (224,) representing average energy at each frequency scale
        freq_profile = np.mean(sub_data[:, :, :, 0], axis=(0, 2))
        
        # Smooth for visualization
        window = 5
        freq_profile_smoothed = np.convolve(freq_profile, np.ones(window)/window, mode='same')
        
        label_str = f"MDD (label=1)" if labels[mask][0] == 1 else "Healthy (label=0)"
        color = 'red' if "MDD" in name else ('orange' if "Outlier" in name else 'green')
        linestyle = '-' if "Outlier" in name else '--'
        
        plt.plot(freq_profile_smoothed, label=f"{name} [{label_str}]", color=color, linestyle=linestyle, linewidth=2)

    plt.title("Spectral Energy Profiles: Outliers vs Controls", fontsize=16, fontweight='bold')
    plt.xlabel("Frequency Scale (CWT Row)", fontsize=12)
    plt.ylabel("Normalized Amplitude", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # In CWT/Scalogram, higher rows usually correspond to lower frequencies (larger scales)
    plt.annotate("← High Freq (Beta/Gamma?)", xy=(10, 0), xytext=(10, -0.05), fontsize=10)
    plt.annotate("Low Freq (Delta/Theta?) →", xy=(210, 0), xytext=(150, -0.05), fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "spectral_profiles_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Analysis saved to {plot_path}")

    # Also save individual average spectrograms for visual check
    for sid, name in TARGETS.items():
        mask = (subject_ids == sid)
        avg_spec = np.mean(spectrograms[mask][:, :, :, 0], axis=0)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(avg_spec, aspect='auto', cmap='viridis', origin='lower')
        plt.title(f"Mean Spectrogram: {name}")
        plt.colorbar(label='Intensity')
        plt.savefig(os.path.join(OUTPUT_DIR, f"avg_spec_{name.split(' ')[0]}.png"))
        plt.close()

if __name__ == "__main__":
    analyze()
