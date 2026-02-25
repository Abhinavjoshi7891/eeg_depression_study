
import os
import numpy as np
import collections

DATA_DIR = "../data/spectrograms"

def inspect():
    print("Loading data...")
    try:
        spectrograms = np.load(os.path.join(DATA_DIR, "spectrograms.npy"))
        labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
        subject_ids = np.load(os.path.join(DATA_DIR, "subject_ids.npy"))
    except FileNotFoundError:
        print(f"Error: Data files not found in {DATA_DIR}")
        return

    print(f"Data loaded. Shape: {spectrograms.shape}")

    # Subject IDs
    # 14: MDD_1 (Control - Good)
    # 22: MDD_19 (Outlier - Bad)
    # 38: MDD_5 (Outlier - Bad)
    targets = {
        14: "MDD_1 (Good)",
        22: "MDD_19 (Bad)",
        38: "MDD_5 (Bad)"
    }

    print("-" * 60)
    print(f"{'Subject':<20} | {'Count':<8} | {'Label':<5} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 60)

    for sid, name in targets.items():
        mask = (subject_ids == sid)
        count = np.sum(mask)
        
        if count == 0:
            print(f"{name:<20} | 0        | N/A   | N/A      | N/A      | N/A      | N/A")
            continue

        data = spectrograms[mask]
        sub_labels = labels[mask]
        
        # Check if all labels are correct (MDD=1)
        unique_labels = np.unique(sub_labels)
        label_str = str(unique_labels)

        # Pixel stats
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)

        print(f"{name:<20} | {count:<8} | {label_str:<5} | {mean_val:<8.4f} | {std_val:<8.4f} | {min_val:<8.4f} | {max_val:<8.4f}")

    print("-" * 60)
    
    # Check for NaN or Inf in outliers
    for sid, name in targets.items():
        if sid in [22, 38]: # Outliers
            mask = (subject_ids == sid)
            data = spectrograms[mask]
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
            print(f"Checking {name} for NaN/Inf: NaN={has_nan}, Inf={has_inf}")

if __name__ == "__main__":
    inspect()
